"""
Production VAD: Silero VAD v5 (ONNX).

Why not our from-scratch VADNet from Module 1? The *shape* is identical — a
tiny neural net emitting one speech probability per short window — but Silero
was trained on ~100 languages and thousands of hours of hard negatives
(music, keyboard clicks, babble noise), which is exactly the data moat we
cannot rebuild from synthetic data. The model is ~2 MB and runs in well under
1 ms per window on CPU, so it costs nothing to run in front of the ASR.

Interface notes:
  - Silero consumes EXACTLY 512-sample windows at 16 kHz (32 ms). We buffer
    arbitrary-sized input chunks internally and emit one probability per
    complete window; the remainder waits for the next push.
  - The model is stateful (an internal RNN carries context across windows),
    so each concurrent stream needs its own instance / reset. That statefulness
    is what lets a 2 MB model be good: it accumulates evidence over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class VADResult:
    prob: float                 # speech probability for this window
    samples: np.ndarray         # the 512 float32 samples of the window


class SileroVAD:
    """Streaming wrapper: push arbitrary chunks, get per-window probabilities."""

    def __init__(self, sample_rate: int = 16_000, window: int = 512):
        from silero_vad import load_silero_vad  # lazy: heavy import

        assert sample_rate == 16_000, "Silero v5 streaming API expects 16 kHz"
        self.sample_rate = sample_rate
        self.window = window
        self.model = load_silero_vad(onnx=True)
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, samples: np.ndarray) -> list[VADResult]:
        """Feed float32 samples; return one VADResult per complete 32 ms window."""
        self._buf = np.concatenate([self._buf, samples])
        results: list[VADResult] = []
        while self._buf.shape[0] >= self.window:
            win = self._buf[: self.window]
            self._buf = self._buf[self.window:]
            prob = float(
                self.model(torch.from_numpy(win.copy()), self.sample_rate).item()
            )
            results.append(VADResult(prob=prob, samples=win))
        return results

    def reset(self) -> None:
        """Clear the model's internal RNN state + buffer (call between streams)."""
        self.model.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)
