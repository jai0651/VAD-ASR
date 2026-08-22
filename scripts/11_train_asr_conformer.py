"""
Module 6: train the Conformer hybrid CTC/attention recognizer.

This is the same job as scripts/05, with a 2026 architecture and a 2026 recipe.
Everything that changed is listed in docs/10-modern-asr.html; the ones that show
up in this file are:

  dynamic chunk training      one model that runs offline AND streaming
  warmup + cosine LR          attention models diverge without warmup
  gradient accumulation       reach a real effective batch on one laptop
  checkpoint averaging        of the best-K by held-out WER, kept only if it wins
  WER (not CER) on held-out   the number the literature reports

DEFAULTS ARE FOR A LAPTOP. `dev-clean` (5.4 h) exists to prove the pipeline
runs end to end — it is far too little data for a 30M-parameter model and it
WILL overfit. That is the point of this module: the architecture is now
correct, so the remaining gap is data, and you close it by pointing this script
at more of it.

Run:
  uv run python scripts/11_train_asr_conformer.py                     # smoke, dev-clean
  SPLIT=train-clean-100 DOWNLOAD=1 STEPS=40000 SIZE=small \\
      uv run python scripts/11_train_asr_conformer.py                 # the real run
  SIZE=tiny STEPS=2000 uv run python scripts/11_train_asr_conformer.py

Env knobs: SPLIT DOWNLOAD STEPS SIZE LR WARMUP MAX_FRAMES ACCUM VOCAB
           NOISE_PROB DEVICE EVAL_EVERY AVG_LAST CHUNK RESUME NUM_WORKERS
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
    frames_per_hour,
    warmup_cosine,
)
from src.asr.hybrid import HybridCTCAttention, count_parameters  # noqa: E402
from src.asr.search import corpus_wer  # noqa: E402
from src.asr.tokenizer import BLANK_ID  # noqa: E402
from src.checkpoint import clear_resume, load_resume, save_resume  # noqa: E402

OUT = Path("outputs")

# Presets. `tiny` fits comfortably on CPU and is for checking the loop runs;
# `small` is the real one and is roughly the size of a deployable streaming ASR.
SIZES = {
    "tiny":  dict(d_model=144, n_layers=6,  n_heads=4, decoder_layers=2),
    "small": dict(d_model=256, n_layers=12, n_heads=4, decoder_layers=6),
    "base":  dict(d_model=384, n_layers=16, n_heads=6, decoder_layers=6),
}

SPLIT = os.environ.get("SPLIT", "dev-clean")
DOWNLOAD = os.environ.get("DOWNLOAD", "0") == "1"
STEPS = int(os.environ.get("STEPS", "3000"))
SIZE = os.environ.get("SIZE", "tiny")
LR = float(os.environ.get("LR", "1e-3"))
# Warmup. THE setting that caused CTC blank collapse: we used 1,500 where every
# reference recipe uses 15,000-35,000. "Why does CTC result in peaky behavior?"
# (arXiv 2105.14849) names aggressive early learning rates as the cause of
# blank-dominated local convergence, and warmup as the first mitigation. Our LR
# hit peak at step 1,500; CTC went flat at ~1,200 and never recovered.
# ESPnet LibriSpeech-100h uses 15,000 absolute; we cap at STEPS//3 so short
# verification runs still get a sane fraction rather than never leaving warmup.
WARMUP = int(os.environ.get("WARMUP", str(min(15000, max(200, STEPS // 3)))))
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "12000"))
ACCUM = int(os.environ.get("ACCUM", "2"))
VOCAB = int(os.environ.get("VOCAB", "256"))
NOISE_PROB = float(os.environ.get("NOISE_PROB", "0.3"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "500"))
# Augmentation warmup. MEASURED on this repo (60 dev-clean utterances, tiny
# model): CTC sits on a plateau near ln(V) nats/token for a few hundred steps
# while it hunts for ANY alignment, then falls off a cliff — 5.05 -> 0.13 by
# step 400 clean, but only -> 1.06 with SpecAugment on from step 0. Regularising
# a model that hasn't learned the task yet just lengthens the plateau, so we let
# it find an alignment first and only then make its life hard.
AUG_START = int(os.environ.get("AUG_START", str(min(1000, STEPS // 4))))
AVG_LAST = int(os.environ.get("AVG_LAST", "5"))
CHUNK = int(os.environ.get("CHUNK", "16"))   # streaming eval chunk (40 ms units)
# Resume is ON by default and costs nothing when there is nothing to resume.
# It is what makes an INTERRUPTIBLE (spot) GPU safe to use, and spot is 50-80%
# cheaper than on-demand — a far bigger saving than any instance-type choice.
RESUME = os.environ.get("RESUME", "1") == "1"
# Dataloader workers. Default 0 keeps AUG_START exactly live (the loop toggles a
# flag on the shared dataset object) and is right on a laptop, where feature
# extraction is not the bottleneck. On a GPU it IS the bottleneck — measured
# 42% GPU utilisation with 0 workers, i.e. the A4000 idle over half the time
# waiting for log-mels. With workers > 0 each worker gets a COPY of the dataset,
# so the AUG_START toggle only lands when workers respawn at an epoch boundary —
# acceptable, because an epoch here is ~1500 steps.
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "0"))


def get_device() -> torch.device:
    if os.environ.get("DEVICE"):
        return torch.device(os.environ["DEVICE"])
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sample_chunk_config(rng: random.Random) -> tuple[int, int]:
    """Dynamic chunk training (WeNet). Half the batches use full context so the
    model stays strong offline; the other half get a random chunk size so it
    also learns to work with a bounded lookahead. One model, both modes — the
    alternative is training and shipping two."""
    if rng.random() < 0.5:
        return 0, -1                       # offline
    chunk = rng.choice([4, 8, 16, 24, 32])  # 160 ms .. 1.28 s of lookahead
    return chunk, rng.choice([4, 8, 16, -1])


@torch.no_grad()
def blank_fraction(model, loader, device, max_batches: int = 2) -> float:
    """Fraction of encoder frames whose CTC argmax is <blank>.

    THE canary for CTC blank collapse — the degenerate minimum where the model
    discovers that emitting blank everywhere is a safe, finite-loss answer and
    stops trying to align. It is invisible in the loss (which just goes flat at
    a plausible-looking value) and invisible in WER (which pins at 1.0 and could
    equally mean "early"). Measured on this project: 2,000 wasted steps at 100%
    blank before anyone looked at the actual emissions.

    Healthy is ~0.6-0.9 — blank SHOULD dominate, since there are several frames
    per token. 1.0 means the model has stopped emitting labels entirely.
    """
    model.eval()
    blanks = total = 0
    for i, (feats, f_lens, *_rest) in enumerate(loader):
        if i >= max_batches:
            break
        enc, enc_lens = model.encode(feats.to(device), f_lens.to(device))
        argmax = model.ctc_head(enc).argmax(-1)
        for b, n in enumerate(enc_lens.tolist()):
            blanks += int((argmax[b, :n] == BLANK_ID).sum())
            total += n
    model.train()
    return blanks / max(1, total)


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, max_batches: int = 12,
             chunk_size: int = 0, rescore: bool = True) -> tuple[float, list]:
    model.eval()
    pairs, samples = [], []
    for i, (feats, f_lens, _ys, _y_lens, texts) in enumerate(loader):
        if i >= max_batches:
            break
        hyps = model.recognize(
            feats.to(device), f_lens.to(device), beam_size=8,
            chunk_size=chunk_size, rescore=rescore,
        )
        for h, ref in zip(hyps, texts):
            pred = tokenizer.decode(h)
            pairs.append((pred, ref))
            if len(samples) < 3:
                samples.append((pred, ref))
    model.train()
    return corpus_wer(pairs), samples


RESUME_PATH = OUT / "_conformer_resume.pt"


def save_model(state: dict, path: Path, tokenizer_path: str,
               tokenizer_corpus: str = "") -> None:
    """Ship the weights with what isn't recoverable from them.

    The FINGERPRINT matters as much as the path. A checkpoint's token ids are
    only meaningful under the exact tokenizer that produced them, and a later
    run can legitimately regenerate that file in place — which silently
    remaps every id and turns a working model into one that emits real
    structure with corrupted spelling. Recording the fingerprint lets the
    loader refuse instead of guessing.
    """
    torch.save({"model": state, "tokenizer": tokenizer_path,
                "tokenizer_corpus": tokenizer_corpus,
                "causal_conv": True, "n_mels": 80}, path)


def average_checkpoints(paths: list[Path]) -> dict:
    """Uniform weight average of checkpoints.

    The premise: SGD near convergence orbits a basin rather than sitting at its
    bottom, so averaging the orbit approximates the bottom. Usually worth a few
    tenths of WER for free.

    THE PREMISE HAS A PRECONDITION, and we violated it the first time. Averaging
    the LAST 5 checkpoints of a 6000-step run gave WER 0.659 against a best
    single checkpoint of 0.612 — because with EVAL_EVERY=1000 those five
    checkpoints spanned the entire run (WER 0.86 -> 0.61), i.e. different basins,
    not one orbit. Averaging a half-trained model into a trained one is just
    damage. So we average the best-K BY HELD-OUT WER (what ESPnet's n-best
    averaging does), and the caller keeps the result only if it actually wins.
    """
    states = [torch.load(p, map_location="cpu") for p in paths]
    avg = {}
    for k in states[0]:
        if states[0][k].is_floating_point():
            avg[k] = sum(s[k].float() for s in states) / len(states)
        else:
            avg[k] = states[0][k]
    return avg


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    rng = random.Random(1234)
    device = get_device()
    OUT.mkdir(exist_ok=True)

    # ---- data ----------------------------------------------------------
    items = build_manifest(split=SPLIT, download=DOWNLOAD)
    rng.shuffle(items)
    n_dev = max(1, min(300, len(items) // 10))
    dev_items, train_items = items[:n_dev], items[n_dev:]
    print(f"device: {device}   split: {SPLIT}")
    print(f"train: {len(train_items)} utts ({frames_per_hour(train_items):.1f} h)  "
          f"held-out: {len(dev_items)} utts")

    tok_path = str(OUT / f"bpe_{VOCAB}.json")
    tokenizer = build_tokenizer(train_items, VOCAB, tok_path)
    print(f"tokenizer: {tokenizer.vocab_size} subwords "
          f"({len(tokenizer.merges)} merges learned)")

    train_ds = SpeechCorpus(train_items, tokenizer, train=True, noise_prob=NOISE_PROB)
    dev_ds = SpeechCorpus(dev_items, tokenizer, train=False)
    train_loader = DataLoader(
        train_ds, batch_sampler=DynamicBatchSampler(train_items, MAX_FRAMES),
        collate_fn=collate, num_workers=NUM_WORKERS,
        persistent_workers=False,   # respawn per epoch so AUG_START lands
    )
    dev_loader = DataLoader(
        dev_ds, batch_sampler=DynamicBatchSampler(dev_items, MAX_FRAMES, shuffle=False),
        collate_fn=collate, num_workers=0,
    )

    # ---- model ---------------------------------------------------------
    cfg = SIZES[SIZE]
    model = HybridCTCAttention(vocab_size=tokenizer.vocab_size, **cfg).to(device)
    print(f"model: {SIZE} — {count_parameters(model)/1e6:.1f}M params "
          f"(Module 2's was 2.0M)")

    # weight_decay 1e-6, not the 1e-2 we had: ESPnet's LibriSpeech-100h recipe
    # uses 1e-6, and 1e-2 is 10,000x more regularisation pressure on a model
    # that has not yet learned the task.
    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.98),
                            eps=1e-9, weight_decay=float(os.environ.get("WD", "1e-6")))

    # ---- train ---------------------------------------------------------
    step, extra = (load_resume(RESUME_PATH, model, opt, device) if RESUME
                   else (0, {}))
    best_wer = extra.get("best_wer", float("inf"))
    ckpts = [(w, Path(p)) for w, p in extra.get("ckpts", []) if Path(p).exists()]
    if step:
        print(f"resumed from step {step} (best WER {best_wer:.3f}) — "
              f"an interrupted spot instance costs one eval interval, not a run")
    t0 = time.time()
    steps_at_start = step
    model.train()
    data_iter = iter(train_loader)

    while step < STEPS:
        opt.zero_grad(set_to_none=True)
        acc = {"ctc": 0.0, "att": 0.0}
        for _ in range(ACCUM):
            try:
                feats, f_lens, ys, y_lens, _ = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                feats, f_lens, ys, y_lens, _ = next(data_iter)
            chunk, left = sample_chunk_config(rng)
            loss, parts = model(
                feats.to(device), f_lens.to(device), ys.to(device), y_lens.to(device),
                chunk_size=chunk, left_chunks=left,
            )
            (loss / ACCUM).backward()
            acc["ctc"] += parts["ctc"] / ACCUM
            acc["att"] += parts["att"] / ACCUM

        step += 1
        # num_workers=0 keeps the dataset object shared, so this toggle is live.
        train_ds.augment_enabled = step >= AUG_START
        lr = warmup_cosine(step, WARMUP, STEPS, LR)
        for g in opt.param_groups:
            g["lr"] = lr
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % 50 == 0 or step == 1:
            print(f"step {step:6d} | ctc {acc['ctc']:6.3f} | att {acc['att']:6.3f} "
                  f"| lr {lr:.2e} | gnorm {float(grad_norm):5.2f} "
                  f"| {(step-steps_at_start)/(time.time()-t0):.2f} it/s", flush=True)

        if step % EVAL_EVERY == 0 or step == STEPS:
            wer, samples = evaluate(model, dev_loader, tokenizer, device)
            mark = ""
            if wer < best_wer:
                best_wer = wer
                save_model(model.state_dict(), OUT / "asr_conformer.pt", tok_path,
                           tokenizer.corpus)
                mark = "  <- new best"
            blank = blank_fraction(model, dev_loader, device)
            warn = ""
            if blank > 0.995:
                warn = ("   *** CTC BLANK COLLAPSE: every frame decodes to blank. "
                        "The model has stopped emitting labels and will NOT recover. "
                        "Stop, and lower VOCAB and/or MAX_FRAMES. ***")
            print(f"    held-out WER @ {step}: {wer:.3f} (best {best_wer:.3f}) "
                  f"| blank-frames {blank:.3f}{mark}{warn}")
            for pred, ref in samples:
                print(f"      hyp: {pred}\n      ref: {ref}")
            p = OUT / f"_conformer_step{step}.pt"
            torch.save(model.state_dict(), p)
            ckpts.append((wer, p))
            save_resume(RESUME_PATH, model, opt, step,
                        {"best_wer": best_wer,
                         "ckpts": [(w, str(q)) for w, q in ckpts]})
            model.train()

    # ---- checkpoint averaging -----------------------------------------
    if len(ckpts) >= 2:
        best_k = [p for _, p in sorted(ckpts, key=lambda kv: kv[0])[:AVG_LAST]]
        print(f"\naveraging the {len(best_k)} best checkpoints by held-out WER")
        avg = average_checkpoints(best_k)
        model.load_state_dict(avg)
        wer, _ = evaluate(model, dev_loader, tokenizer, device)
        print(f"    averaged WER {wer:.3f} (best single {best_wer:.3f})")
        if wer < best_wer:
            best_wer = wer
            save_model(avg, OUT / "asr_conformer.pt", tok_path, tokenizer.corpus)
            print("    averaged model wins — saved as outputs/asr_conformer.pt")

    # ---- the three numbers that matter --------------------------------
    best = torch.load(OUT / "asr_conformer.pt", map_location=device)
    model.load_state_dict(best["model"])
    print("\n" + "=" * 66)
    greedy, _ = evaluate(model, dev_loader, tokenizer, device, rescore=False)
    offline, _ = evaluate(model, dev_loader, tokenizer, device, rescore=True)
    stream, _ = evaluate(model, dev_loader, tokenizer, device,
                         chunk_size=CHUNK, rescore=True)
    print(f"WER  CTC greedy            {greedy:.3f}")
    print(f"WER  + attention rescoring {offline:.3f}   (what rescoring buys)")
    print(f"WER  streaming, {CHUNK*40:4d} ms chunk {stream:.3f}   "
          f"(what streaming costs)")
    print("=" * 66)
    print(f"\nsaved outputs/asr_conformer.pt and outputs/bpe_{VOCAB}.json")
    print("use it live:  VOICE_ASR_ENGINE=conformer uv run python -m server.app")

    for _, p in ckpts:                   # keep outputs/ tidy
        p.unlink(missing_ok=True)
    clear_resume(RESUME_PATH)            # the run finished; don't resume it


if __name__ == "__main__":
    main()
