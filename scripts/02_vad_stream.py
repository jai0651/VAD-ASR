"""
Module 1 demo: run the trained VAD as if audio were arriving live.

We take a fresh utterance, feed it to the model in small chunks (simulating a mic
delivering ~100 ms at a time), and apply hysteresis smoothing to get clean speech
segments — exactly what you'd send downstream to an ASR worker.

Run:  uv run python scripts/02_vad_stream.py
Outputs:
  - prints detected speech segments (start -> end seconds)
  - outputs/02_vad_stream.png : raw probs, smoothed decision, ground truth
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.vad.data import make_example  # noqa: E402
from src.vad.model import VADNet  # noqa: E402
from src.vad.stream import StreamingVAD, segments_from_decisions  # noqa: E402

HOP_S = 0.01
CHUNK_FRAMES = 10  # 10 frames * 10 ms = 100 ms of audio per streamed chunk


def main() -> None:
    if not os.path.exists("outputs/vad.pt"):
        sys.exit("Train first: uv run python scripts/01_train_vad.py")

    model = VADNet(n_mels=80)
    model.load_state_dict(torch.load("outputs/vad.pt", map_location="cpu"))

    feats, truth = make_example()  # a brand-new utterance
    vad = StreamingVAD(model, context_frames=12)

    raw_probs: list[float] = []
    decisions: list[bool] = []
    # Feed the utterance chunk by chunk, never revealing the future.
    for start in range(0, feats.shape[0], CHUNK_FRAMES):
        chunk = feats[start:start + CHUNK_FRAMES]
        for prob, is_speech in vad.push_frames(chunk):
            raw_probs.append(prob)
            decisions.append(is_speech)

    segments = segments_from_decisions(decisions, HOP_S)
    print("Detected speech segments:")
    for s, e in segments:
        print(f"  {s:5.2f}s -> {e:5.2f}s  ({e - s:.2f}s)")

    # Visualize: ground truth (shaded), raw probability, smoothed decision.
    t = torch.arange(len(raw_probs)) * HOP_S
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.fill_between(t, 0, truth[:len(t)], step="mid", alpha=0.25,
                    label="truth (speech)")
    ax.plot(t, raw_probs, color="crimson", lw=1.0, label="raw P(speech)")
    ax.plot(t, [int(d) for d in decisions], color="navy", lw=1.6,
            label="smoothed decision")
    ax.axhline(0.6, ls=":", color="gray", lw=0.8)
    ax.axhline(0.4, ls=":", color="gray", lw=0.8)
    ax.text(t[-1], 0.61, "on=0.6", ha="right", va="bottom", fontsize=8, color="gray")
    ax.text(t[-1], 0.39, "off=0.4", ha="right", va="top", fontsize=8, color="gray")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("speech")
    ax.set_title("Streaming VAD: raw probability vs. hysteresis-smoothed decision")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig("outputs/02_vad_stream.png", dpi=120)
    print("\nSaved outputs/02_vad_stream.png")


if __name__ == "__main__":
    main()
