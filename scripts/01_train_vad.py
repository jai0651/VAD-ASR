"""
Module 1, part 3: train the VAD model and watch it learn.

Run:  uv run python scripts/01_train_vad.py
Outputs:
  - prints training loss + frame accuracy
  - outputs/01_vad_predictions.png : predicted speech probability vs. truth
  - outputs/vad.pt : trained weights (reused by the streaming demo)

Concepts shown:
  - BCEWithLogitsLoss: the standard loss for binary (per-frame) classification.
  - Masking: we ignore padded frames so padding does not corrupt the loss.
  - On-the-fly data: we synthesize new batches every step, so there is effectively
    infinite data and no train/val leakage to worry about for this toy task.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vad.data import make_batch, make_example  # noqa: E402
from src.vad.model import VADNet  # noqa: E402


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def masked_bce(logits, labels, mask, loss_fn) -> torch.Tensor:
    """Per-frame BCE, averaged only over real (non-padded) frames."""
    per_frame = loss_fn(logits, labels)        # (B, T), reduction='none'
    return (per_frame * mask).sum() / mask.sum()


def frame_accuracy(logits, labels, mask) -> float:
    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = ((preds == labels) * mask).sum()
    return (correct / mask.sum()).item()


def main() -> None:
    torch.manual_seed(0)
    device = get_device()
    print(f"device: {device}")

    model = VADNet(n_mels=80).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    steps = 400
    batch_size = 16
    model.train()
    for step in range(1, steps + 1):
        feats, labels, mask = make_batch(batch_size)
        feats, labels, mask = feats.to(device), labels.to(device), mask.to(device)

        logits = model(feats)
        loss = masked_bce(logits, labels, mask, loss_fn)

        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 1:
            acc = frame_accuracy(logits, labels, mask)
            print(f"step {step:4d} | loss {loss.item():.4f} | frame acc {acc:.3f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/vad.pt")

    # Evaluate on a fresh, unseen utterance and visualize.
    model.eval()
    feats, labels = make_example()
    with torch.no_grad():
        probs = torch.sigmoid(model(feats.unsqueeze(0).to(device)))[0].cpu()

    fig, ax = plt.subplots(figsize=(11, 4))
    t = torch.arange(labels.shape[0]) * 0.01  # 10 ms per frame
    ax.fill_between(t, 0, labels, step="mid", alpha=0.3, label="truth (speech=1)")
    ax.plot(t, probs, color="crimson", label="predicted P(speech)")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8, label="0.5 threshold")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speech probability")
    ax.set_title("VAD on a held-out utterance: prediction vs. ground truth")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("outputs/01_vad_predictions.png", dpi=120)
    print("\nSaved outputs/01_vad_predictions.png and outputs/vad.pt")


if __name__ == "__main__":
    main()
