"""
Module 0 demo: see a waveform turn into a log-mel spectrogram.

We synthesize a signal with obvious structure so the spectrogram is easy to read:
  - silence
  - a low 200 Hz tone
  - silence
  - a rising sweep 300 -> 3000 Hz
  - silence
  - two tones stacked (440 Hz + 1500 Hz), like a vowel's harmonics

Run:  uv run python scripts/00_features.py
Output: writes outputs/00_features.png and prints tensor shapes.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.audio.features import log_mel_spectrogram  # noqa: E402

SR = 16_000


def tone(freq: float, dur_s: float, sr: int = SR) -> torch.Tensor:
    t = torch.arange(int(dur_s * sr)) / sr
    return torch.sin(2 * torch.pi * freq * t)


def sweep(f0: float, f1: float, dur_s: float, sr: int = SR) -> torch.Tensor:
    t = torch.arange(int(dur_s * sr)) / sr
    # Linear chirp: frequency rises linearly from f0 to f1.
    inst_freq = f0 + (f1 - f0) * t / dur_s
    phase = 2 * torch.pi * torch.cumsum(inst_freq, dim=0) / sr
    return torch.sin(phase)


def silence(dur_s: float, sr: int = SR) -> torch.Tensor:
    return torch.zeros(int(dur_s * sr))


def build_signal() -> torch.Tensor:
    parts = [
        silence(0.3),
        tone(200, 0.5),
        silence(0.3),
        sweep(300, 3000, 0.6),
        silence(0.3),
        0.6 * tone(440, 0.5) + 0.4 * tone(1500, 0.5),  # stacked harmonics
        silence(0.3),
    ]
    sig = torch.cat(parts)
    # A touch of noise so it is not perfectly clean (real audio never is).
    sig = sig + 0.01 * torch.randn_like(sig)
    return sig


def main() -> None:
    wav = build_signal()
    logmel = log_mel_spectrogram(wav, sr=SR)  # (num_frames, n_mels)

    print(f"waveform samples : {wav.shape[0]}  ({wav.shape[0] / SR:.2f} s)")
    print(f"log-mel shape    : {tuple(logmel.shape)}  (frames, mel-channels)")
    print(f"frames per second: {logmel.shape[0] / (wav.shape[0] / SR):.0f}")

    os.makedirs("outputs", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    t = torch.arange(wav.shape[0]) / SR
    ax1.plot(t, wav, linewidth=0.4)
    ax1.set_title("Waveform (what a microphone records)")
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("amplitude")

    # transpose so frequency (mel) is on the y-axis, time on the x-axis.
    im = ax2.imshow(
        logmel.T.numpy(),
        origin="lower",
        aspect="auto",
        extent=[0, wav.shape[0] / SR, 0, logmel.shape[1]],
        cmap="magma",
    )
    ax2.set_title("Log-mel spectrogram (what the model sees)")
    ax2.set_xlabel("time (s)")
    ax2.set_ylabel("mel channel (low -> high freq)")
    fig.colorbar(im, ax=ax2, label="log energy")

    fig.tight_layout()
    out = "outputs/00_features.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out} — open it to see silence (dark) vs. sound (bright).")


if __name__ == "__main__":
    main()
