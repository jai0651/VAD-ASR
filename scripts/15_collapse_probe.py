"""
Reproduce CTC blank collapse locally, then find what actually prevents it.

Blank collapse is the failure that cost us a GPU run: the model discovers that
emitting <blank> on every frame is a safe, finite-loss answer, converges into
it, and never emits a label again. It is invisible in the loss (which goes flat
at a plausible value) and invisible in WER (which pins at 1.0, indistinguishable
from "still early").

The point of this script is to stop reasoning about it and MEASURE it. Each
config trains a few hundred steps on a small local subset and reports the
fraction of frames whose CTC argmax is blank. 1.0 means collapsed. The healthy
signature is a value that rises toward ~0.8-0.95 and then comes DOWN as the
model starts emitting labels.

Run:
  uv run python scripts/15_collapse_probe.py
  STEPS=600 CONFIGS=vocab1024_bigbatch,vocab256_smallbatch \\
    uv run python scripts/15_collapse_probe.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.corpus import (  # noqa: E402
    DynamicBatchSampler,
    SpeechCorpus,
    build_manifest,
    build_tokenizer,
    collate,
    warmup_cosine,
)
from src.asr.hybrid import HybridCTCAttention  # noqa: E402
from src.asr.tokenizer import BLANK_ID  # noqa: E402

STEPS = int(os.environ.get("STEPS", "500"))
N_UTTS = int(os.environ.get("N_UTTS", "600"))
SIZE = dict(d_model=144, n_layers=6, n_heads=4, decoder_layers=2)

# One variable changes per row wherever possible, so a difference is attributable.
CONFIGS = {
    #                      vocab  max_frames accum   lr    ctc_w
    "A_gpu_run_that_died": (1024,     24000,    2, 1e-3,   0.3),
    "B_smaller_vocab":     ( 256,     24000,    2, 1e-3,   0.3),
    "C_smaller_batch":     (1024,     12000,    1, 1e-3,   0.3),
    "D_lower_lr":          (1024,     24000,    2, 3e-4,   0.3),
    "E_more_ctc_weight":   (1024,     24000,    2, 1e-3,   0.7),
    "F_known_good_local":  ( 256,     12000,    1, 1e-3,   0.3),
}
ONLY = [c for c in os.environ.get("CONFIGS", "").split(",") if c]


@torch.no_grad()
def blank_fraction(model, batch, device) -> float:
    model.eval()
    f, f_lens = batch[0].to(device), batch[1].to(device)
    enc, enc_lens = model.encode(f, f_lens)
    argmax = model.ctc_head(enc).argmax(-1)
    blanks = total = 0
    for b, n in enumerate(enc_lens.tolist()):
        blanks += int((argmax[b, :n] == BLANK_ID).sum())
        total += n
    model.train()
    return blanks / max(1, total)


def run(name, vocab, max_frames, accum, lr, ctc_w, items, device):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    tok = build_tokenizer(items, vocab, Path("outputs") / f"_probe_bpe_{vocab}.json")
    ds = SpeechCorpus(items, tok, train=True, noise_prob=0.0)
    ds.augment_enabled = False          # isolate the variables under test
    loader = DataLoader(ds, batch_sampler=DynamicBatchSampler(items, max_frames),
                        collate_fn=collate, num_workers=0)
    probe_batch = next(iter(loader))

    model = HybridCTCAttention(vocab_size=tok.vocab_size, ctc_weight=ctc_w,
                               dropout=0.1, **SIZE).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=1e-2)

    it = iter(loader)
    marks, t0 = [], time.time()
    for step in range(1, STEPS + 1):
        opt.zero_grad(set_to_none=True)
        ctc = 0.0
        for _ in range(accum):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader); b = next(it)
            f, fl, y, yl, _ = b
            loss, parts = model(f.to(device), fl.to(device), y.to(device), yl.to(device))
            (loss / accum).backward()
            ctc += parts["ctc"] / accum
        for g in opt.param_groups:
            g["lr"] = warmup_cosine(step, max(50, STEPS // 10), STEPS, lr)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % (STEPS // 5) == 0:
            marks.append((step, ctc, blank_fraction(model, probe_batch, device)))

    print(f"\n{name}  vocab={vocab} max_frames={max_frames} accum={accum} "
          f"lr={lr:g} ctc_w={ctc_w}  ({time.time()-t0:.0f}s)")
    print("     step |    ctc | blank-frac")
    for s, c, bf in marks:
        flag = "  <- COLLAPSED" if bf > 0.995 else ""
        print(f"    {s:5d} | {c:6.3f} | {bf:9.4f}{flag}")
    return marks[-1][2], marks[-1][1]


def main() -> None:
    device = torch.device(os.environ.get("DEVICE",
                          "mps" if torch.backends.mps.is_available() else "cpu"))
    items = build_manifest(split="dev-clean")
    random.Random(0).shuffle(items)
    items = items[:N_UTTS]
    print(f"device={device}  {len(items)} utterances  {STEPS} steps per config")

    results = {}
    for name, cfg in CONFIGS.items():
        if ONLY and name not in ONLY:
            continue
        results[name] = run(name, *cfg, items, device)

    print("\n" + "=" * 64)
    print(f"{'config':24s} {'blank-frac':>11s} {'ctc':>8s}   verdict")
    print("=" * 64)
    for name, (bf, ctc) in results.items():
        verdict = "COLLAPSED" if bf > 0.995 else ("healthy" if bf < 0.99 else "borderline")
        print(f"{name:24s} {bf:11.4f} {ctc:8.3f}   {verdict}")
    print("\nblank-frac 1.0 = emits nothing but blank and will not recover.")


if __name__ == "__main__":
    main()
