"""
Module 1b: retrain the SAME VADNet on real audio — and prove it fixes the bug.

The live-mic failure (docs/02-vad.html §2.5): the synthetic-data VAD read room
ambience as speech, turns only closed at the 30 s cap, and phantom barge-ins
flushed every reply. Architecture unchanged — only the data changes
(src/vad/data_real.py). That's the point: for small speech models the data IS
the model.

The eval at the end scores exactly the scenarios that failed live:
  - pure backgrounds (room noise, babble, hum, digital zeros) -> want P ~ 0
  - real speech over those backgrounds                        -> want P ~ 1

Run:
  uv run python scripts/09_train_vad_real.py
  STEPS=1500 BATCH=16 uv run python scripts/09_train_vad_real.py

Writes outputs/vad_real.pt — the pipeline picks it up automatically
(engines resolve "auto": vad_real.pt if present, else vad.pt).
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.features import log_mel_spectrogram  # noqa: E402
from src.vad.data_real import (  # noqa: E402
    SR,
    _babble,
    _colored_noise,
    _hum,
    list_speech_files,
    load_clip,
    make_real_batch,
    make_real_example,
)
from src.vad.model import VADNet  # noqa: E402

STEPS = int(os.environ.get("STEPS", "1200"))
BATCH = int(os.environ.get("BATCH", "16"))
OUT = Path("outputs")


def get_device() -> torch.device:
    if os.environ.get("DEVICE"):
        return torch.device(os.environ["DEVICE"])
    # Conv-parallel training over batches: MPS genuinely helps here
    # (unlike the TTS decoder loop — see docs/06-decisions.html D12).
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def masked_bce(logits, labels, mask, loss_fn):
    per_frame = loss_fn(logits, labels)
    return (per_frame * mask).sum() / mask.sum()


@torch.no_grad()
def mean_prob(model, device, wav: np.ndarray) -> float:
    feats = log_mel_spectrogram(torch.from_numpy(wav), sr=SR)
    return torch.sigmoid(model(feats.unsqueeze(0).to(device)))[0].mean().item()


def build_eval_cases(files, per_case: int = 4) -> dict[str, list[np.ndarray]]:
    """A FIXED eval set (seeded) of the scenarios that broke the synthetic
    model live — several instances per case so one lucky sample can't hide a
    regression. Built once so every checkpoint is scored on identical audio."""
    rng_state = (random.getstate(), np.random.get_state())
    random.seed(1234)
    np.random.seed(1234)
    n = 4 * SR
    cases: dict[str, list[np.ndarray]] = {}
    for i in range(per_case):
        speech = load_clip(files[i % len(files)], max_s=4.0) * 0.3
        for name, wav in {
            "digital zeros":        np.zeros(n, dtype=np.float32),
            "faint room noise":     _colored_noise(n) * 0.003,
            "loud room noise":      _colored_noise(n) * 0.05,
            "babble (no speaker)":  _babble(n, files) * 0.03,
            "mains hum":            _hum(n) * 0.03,
            "clean speech":         speech,
            "speech + room noise":  speech + _colored_noise(n)[: speech.shape[0]] * 0.01,
            "quiet speech (-26dB)": speech * 0.05,
        }.items():
            cases.setdefault(name, []).append(wav)
    random.setstate(rng_state[0])
    np.random.set_state(rng_state[1])
    return cases


@torch.no_grad()
def hard_case_report(model, device, cases) -> dict[str, float]:
    """Mean P(speech) per scenario over the fixed eval set."""
    return {
        name: float(np.mean([mean_prob(model, device, w) for w in wavs]))
        for name, wavs in cases.items()
    }


def hard_case_score(report: dict[str, float]) -> float:
    """One number to select checkpoints by: mean margin in the right
    direction — (1 - p) on non-speech cases, p on speech cases. The lesson
    from the TTS module applied here: edge-case behavior oscillates between
    checkpoints while training loss stays flat, so 'last' is not 'best' —
    select on the metric you actually care about."""
    return float(np.mean([
        (p if "speech" in name and "no speaker" not in name else 1.0 - p)
        for name, p in report.items()
    ]))


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = get_device()
    print(f"device: {device}")

    files = list_speech_files()
    random.shuffle(files)
    heldout_files = files[:60]
    train_files = files[60:]
    print(f"speech: {len(train_files)} train / {len(heldout_files)} held-out real utterances")

    model = VADNet(n_mels=80).to(device)   # UNCHANGED architecture from Module 1
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    OUT.mkdir(exist_ok=True)
    eval_cases = build_eval_cases(heldout_files)
    best_score = -1.0

    model.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        feats, labels, mask = make_real_batch(train_files, BATCH)
        feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
        logits = model(feats)
        loss = masked_bce(logits, labels, mask, loss_fn)
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            with torch.no_grad():
                preds = (torch.sigmoid(logits) > 0.5).float()
                acc = (((preds == labels) * mask).sum() / mask.sum()).item()
            print(f"step {step:4d} | loss {loss.item():.4f} | frame acc {acc:.3f} "
                  f"| {step / (time.time() - t0):.1f} it/s", flush=True)

        # Checkpoint selection on the hard cases (see hard_case_score): only
        # a model that handles ALL the live failure modes gets shipped.
        if step % 250 == 0 or step == STEPS:
            model.eval()
            score = hard_case_score(hard_case_report(model, device, eval_cases))
            marker = ""
            if score > best_score:
                best_score = score
                torch.save(model.state_dict(), OUT / "vad_real.pt")
                marker = "  <- new best, saved as vad_real.pt"
            print(f"    hard-case score @ {step}: {score:.3f} "
                  f"(best {best_score:.3f}){marker}", flush=True)
            model.train()

    # ---- final report is on the BEST checkpoint, i.e. what ships ----
    model.load_state_dict(torch.load(OUT / "vad_real.pt", map_location=device))
    model.eval()
    accs = []
    with torch.no_grad():
        for _ in range(20):
            feats, labels, mask = make_real_batch(heldout_files, 8)
            feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)
            preds = (torch.sigmoid(model(feats)) > 0.5).float()
            accs.append((((preds == labels) * mask).sum() / mask.sum()).item())
    print(f"\nheld-out frame accuracy (unseen speakers): {np.mean(accs):.3f}")

    # ---- the report that matters: the failure modes that happened live ----
    print("\nmean P(speech) on the hard cases (want ~0 for the first five):")
    report = hard_case_report(model, device, eval_cases)
    for name, p in report.items():
        want_low = "speech" not in name
        ok = (p < 0.25) if want_low else (p > 0.6)
        print(f"  {'PASS' if ok else 'WARN'}  {name:22s} {p:.3f}")

    # ---- visualize one held-out example ----
    feats, labels = make_real_example(heldout_files)
    with torch.no_grad():
        probs = torch.sigmoid(model(feats.unsqueeze(0).to(device)))[0].cpu()
    t = torch.arange(labels.shape[0]) * 0.01
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(t, 0, labels, step="mid", alpha=0.3, label="truth (speech=1)")
    ax.plot(t, probs, color="crimson", label="predicted P(speech)")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    ax.set_xlabel("time (s)")
    ax.set_title("real-audio VAD on a held-out mix (real speech over noise bed)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "09_vad_real.png", dpi=120)
    print(f"\nsaved outputs/vad_real.pt and outputs/09_vad_real.png")
    print("the pipeline now uses vad_real.pt automatically (ckpt resolution = auto)")


if __name__ == "__main__":
    main()
