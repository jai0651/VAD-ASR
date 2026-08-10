"""
Preflight: everything that must be true BEFORE renting a GPU.

A training bug found at hour four of a rented run costs the run. The same bug
found here costs three minutes. So this checks — on real data, with the real
model — the specific things that fail silently:

  * a target longer than the encoder output. `zero_infinity=True` in CTCLoss
    turns those into a SILENT zero-gradient sample rather than an error, so a
    badly-chosen vocab size just quietly trains on a fraction of your corpus.
  * a parameter that never receives a gradient, i.e. a module you built and
    forgot to wire in. It costs memory and does nothing, and the loss curve
    looks completely normal.
  * NaN/Inf anywhere in the forward, the loss, or the gradients.
  * a data pipeline that produces features the model cannot learn from
    (unnormalised, all-zero after augmentation, mismatched lengths).
  * a decode path that works in training shape but crashes at inference.
  * a resume that is not actually identical (see tests/test_checkpoint.py).

It finishes with a measured throughput and a cost projection, so the decision to
rent is a number rather than a guess.

Run:
  uv run python scripts/14_validate.py
  SIZE=small VOCAB=1024 uv run python scripts/14_validate.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.conformer import ConvSubsampling  # noqa: E402
from src.asr.corpus import (  # noqa: E402
    DynamicBatchSampler,
    SpeechCorpus,
    build_manifest,
    build_tokenizer,
    collate,
    warmup_cosine,
)
from src.asr.hybrid import HybridCTCAttention, count_parameters  # noqa: E402
from src.asr.tokenizer import normalize  # noqa: E402

SIZES = {
    "tiny":  dict(d_model=144, n_layers=6,  n_heads=4, decoder_layers=2),
    "small": dict(d_model=256, n_layers=12, n_heads=4, decoder_layers=4),
    "base":  dict(d_model=384, n_layers=16, n_heads=6, decoder_layers=6),
}
SIZE = os.environ.get("SIZE", "small")
VOCAB = int(os.environ.get("VOCAB", "1024"))
SPLIT = os.environ.get("SPLIT", "dev-clean")
# What the real run will be, used for the cost projection at the end.
PLAN_STEPS = int(os.environ.get("PLAN_STEPS", "40000"))
PLAN_RATE = float(os.environ.get("PLAN_GPU_HR", "0.17"))

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    RESULTS.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


# ---------------------------------------------------------------------------
def validate_data(items, tokenizer):
    section("1. data pipeline")

    ds = SpeechCorpus(items[:24], tokenizer, train=True, noise_prob=0.5)
    feats, tokens, text = ds[0]
    check("features are finite", bool(torch.isfinite(feats).all()))
    check("feature shape is (T, 80)", feats.ndim == 2 and feats.shape[1] == 80,
          f"{tuple(feats.shape)}")
    # CMVN happens before SpecAugment, and SpecAugment zeroes bands, so the
    # post-augmentation stats are only approximately standardised.
    m, s = float(feats.mean()), float(feats.std())
    check("CMVN leaves features standardised", abs(m) < 0.5 and 0.5 < s < 1.6,
          f"mean {m:+.2f} std {s:.2f}")
    check("tokens are non-empty and in range", len(tokens) > 0
          and int(tokens.max()) < tokenizer.vocab_size, f"{len(tokens)} tokens")

    # THE silent killer: CTC needs at least one encoder frame per target token.
    worst, violations = 1e9, []
    for it in items[:400]:
        n_mel = 1 + int((it["duration"] * 16000 - 400) // 160)
        n_enc = int(ConvSubsampling.out_length(torch.tensor([n_mel]))[0])
        n_tok = len(tokenizer.encode(it["text"]))
        ratio = n_enc / max(1, n_tok)
        worst = min(worst, ratio)
        if n_enc < n_tok:
            violations.append((it["duration"], n_enc, n_tok))
    check("every utterance has more encoder frames than tokens",
          not violations,
          f"worst frames/token = {worst:.2f}" if not violations
          else f"{len(violations)} would train on a SILENT zero gradient")

    # Round-trip on real transcripts, not toy strings.
    bad = [it["text"] for it in items[:200]
           if tokenizer.decode(tokenizer.encode(it["text"])) != normalize(it["text"])]
    check("tokenizer round-trips real transcripts", not bad,
          f"{len(bad)} mismatches" if bad else f"{tokenizer.vocab_size} subwords")

    loader_items = items[:24]
    sampler = DynamicBatchSampler(loader_items, max_frames=8000, shuffle=False)
    covered = sorted(i for b in sampler.batches for i in b)
    check("batch sampler covers every utterance exactly once",
          covered == list(range(len(loader_items))),
          f"{len(sampler.batches)} batches")

    batch = collate([ds[i] for i in range(4)])
    f, f_lens, y, y_lens, texts = batch
    check("collate pads consistently",
          f.shape[0] == 4 and int(f_lens.max()) == f.shape[1]
          and int(y_lens.max()) == y.shape[1], f"feats {tuple(f.shape)}")
    check("padded regions are zero",
          all(float(f[i, int(f_lens[i]):].abs().sum()) == 0.0 for i in range(4)))
    return batch


def validate_model(batch, tokenizer, device):
    section(f"2. model ({SIZE})")
    torch.manual_seed(0)
    model = HybridCTCAttention(vocab_size=tokenizer.vocab_size, **SIZES[SIZE]).to(device)
    n = count_parameters(model)
    check("parameter count is plausible", 1e6 < n < 200e6, f"{n/1e6:.1f}M")

    f, f_lens, y, y_lens, _ = batch
    f, f_lens, y, y_lens = (t.to(device) for t in (f, f_lens, y, y_lens))

    enc, enc_lens = model.encode(f, f_lens)
    check("encoder lengths match the subsampling formula",
          enc_lens.tolist() == ConvSubsampling.out_length(f_lens).clamp(min=1).tolist(),
          f"{f.shape[1]} mel -> {enc.shape[1]} enc frames (40 ms each)")
    check("encoder output is finite", bool(torch.isfinite(enc).all()))

    loss, parts = model(f, f_lens, y, y_lens)
    check("loss is finite", bool(torch.isfinite(loss)),
          f"ctc {parts['ctc']:.2f} att {parts['att']:.2f}")

    # What SHOULD an untrained CTC loss be? Not ln(V) — that was this check's
    # first, wrong answer. With reduction="mean" torch divides by TARGET length,
    # while the path probability accumulates over every FRAME. So the expected
    # magnitude is ~(frames/token) x ln(V), which here is ~75, not ~7. An
    # attention decoder normalises per token, so ln(V) IS the right scale there.
    # Getting this wrong makes the validator cry wolf, which is worse than no
    # validator at all.
    ln_v = float(np.log(tokenizer.vocab_size))
    ratio = float(enc_lens.float().mean() / y_lens.float().mean())
    check("CTC loss is at its untrained scale, not diverged",
          0.05 * ratio * ln_v < parts["ctc"] < 1.5 * ratio * ln_v,
          f"{parts['ctc']:.1f} vs expected ~{ratio * ln_v:.0f} "
          f"({ratio:.1f} frames/token x ln V {ln_v:.2f})")
    check("attention loss is at its untrained scale",
          parts["att"] < 1.5 * ln_v, f"{parts['att']:.2f} vs ln(V) {ln_v:.2f}")

    loss.backward()
    missing = [n_ for n_, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    check("every parameter receives a gradient", not missing,
          f"{len(missing)} unwired: {missing[:3]}" if missing else "no dead modules")
    bad = [n_ for n_, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    check("all gradients are finite", not bad, f"{bad[:3]}" if bad else "")
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)
    check("gradient norm is in a sane range", 1e-4 < float(gnorm) < 1e4,
          f"{float(gnorm):.2f}")
    return model


def validate_streaming_and_decode(model, batch, tokenizer, device):
    section("3. inference paths")
    f, f_lens, _, _, _ = batch
    f, f_lens = f.to(device), f_lens.to(device)
    model.eval()

    for mode, kw in [("offline", {}), ("streaming 640ms", {"chunk_size": 16})]:
        try:
            hyps = model.recognize(f[:2], f_lens[:2], beam_size=4, **kw)
            txt = tokenizer.decode(hyps[0])
            check(f"decode runs ({mode})", True, f"{len(hyps)} hyps, text len {len(txt)}")
        except Exception as e:  # noqa: BLE001
            check(f"decode runs ({mode})", False, f"{type(e).__name__}: {e}")

    try:
        greedy = model.recognize(f[:2], f_lens[:2], rescore=False)
        check("CTC greedy path runs", isinstance(greedy[0], list))
    except Exception as e:  # noqa: BLE001
        check("CTC greedy path runs", False, str(e))
    model.train()


def validate_learning(items, tokenizer, device):
    section("4. can it actually learn? (overfit real audio)")
    ds = SpeechCorpus(items[:8], tokenizer, train=False)
    batch = collate([ds[i] for i in range(4)])
    f, f_lens, y, y_lens, texts = batch
    f, f_lens, y, y_lens = (t.to(device) for t in (f, f_lens, y, y_lens))

    torch.manual_seed(0)
    model = HybridCTCAttention(vocab_size=tokenizer.vocab_size,
                               **SIZES["tiny"], dropout=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    t0 = time.time()
    for _ in range(150):
        loss, parts = model(f, f_lens, y, y_lens)
        first = first if first is not None else parts
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    dt = time.time() - t0

    check("CTC loss collapses on 4 memorised utterances",
          parts["ctc"] < 0.5 * first["ctc"],
          f"{first['ctc']:.2f} -> {parts['ctc']:.2f} in 150 steps ({dt:.0f}s)")
    check("attention loss collapses too",
          parts["att"] < 0.5 * first["att"],
          f"{first['att']:.2f} -> {parts['att']:.2f}")

    model.eval()
    hyp = tokenizer.decode(model.recognize(f[:1], f_lens[:1], beam_size=4)[0])
    print(f"       memorised: {hyp[:70]!r}")
    print(f"       reference: {texts[0][:70]!r}")


def validate_schedule():
    section("5. LR schedule")
    steps, warm, peak = PLAN_STEPS, max(200, PLAN_STEPS // 10), 1e-3
    lrs = [warmup_cosine(s, warm, steps, peak) for s in (1, warm, steps // 2, steps)]
    check("warmup starts near zero", lrs[0] < peak * 0.01, f"{lrs[0]:.2e}")
    check("peak is reached at the end of warmup", abs(lrs[1] - peak) < 1e-9,
          f"{lrs[1]:.2e} at step {warm}")
    check("schedule decays monotonically after warmup", lrs[1] > lrs[2] > lrs[3],
          f"{lrs[1]:.1e} -> {lrs[2]:.1e} -> {lrs[3]:.1e}")


def project_cost(model, batch, device):
    section("6. throughput and cost projection")
    f, f_lens, y, y_lens, _ = batch
    f, f_lens, y, y_lens = (t.to(device) for t in (f, f_lens, y, y_lens))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(3):                      # warm up the kernels
        loss, _ = model(f, f_lens, y, y_lens)
        opt.zero_grad(); loss.backward(); opt.step()
    t0 = time.time()
    n_steps = 10
    for _ in range(n_steps):
        loss, _ = model(f, f_lens, y, y_lens)
        opt.zero_grad(); loss.backward(); opt.step()
    local_rate = n_steps / (time.time() - t0)

    print(f"  local ({device}): {local_rate:.2f} steps/s at batch {f.shape[0]}")
    for name, speedup in [("RTX A4000", 3.0), ("RTX A5000", 4.0), ("RTX 4090", 8.0)]:
        hours = PLAN_STEPS / (local_rate * speedup) / 3600
        print(f"  {name:10s} ~{speedup:.0f}x -> {hours:5.1f} h for {PLAN_STEPS} steps"
              f"  ≈ ${hours * PLAN_RATE:.2f} at ${PLAN_RATE}/hr")
    print("  (speedups are rough; the first real run replaces them with a measurement)")


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = torch.device(os.environ.get("DEVICE", "cpu"))
    print(f"validating: SIZE={SIZE} VOCAB={VOCAB} SPLIT={SPLIT} device={device}")

    items = build_manifest(split=SPLIT)
    tokenizer = build_tokenizer(items, VOCAB, Path("outputs") / f"bpe_{VOCAB}.json")

    batch = validate_data(items, tokenizer)
    model = validate_model(batch, tokenizer, device)
    validate_streaming_and_decode(model, batch, tokenizer, device)
    validate_learning(items, tokenizer, device)
    validate_schedule()
    project_cost(model, batch, device)

    section("summary")
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"  {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        for n in failed:
            print(f"    FAILED: {n}")
        print("\n  DO NOT rent a GPU until these pass.")
        sys.exit(1)
    print("\n  All checks passed — safe to run on a rented GPU.")


if __name__ == "__main__":
    main()
