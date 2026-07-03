"""
Module 1, part 4: streaming VAD inference.

Training saw whole utterances at once. Real life is different: audio arrives in
small chunks and you must decide "speech?" *now*, without seeing the future.
Two problems appear that batch evaluation hides:

1. CONTEXT. Our conv model looks at neighboring frames (its receptive field). At
   the live edge there are no future frames yet, and at a chunk's start we would
   lose the previous chunk's frames. We fix this by keeping a small rolling buffer
   of recent feature frames as left-context, so each new frame is scored with the
   same kind of neighborhood it saw in training. (A few frames of *look-ahead*
   would sharpen boundaries further, at the cost of that many ms of latency — the
   fundamental latency/accuracy trade-off of streaming.)

2. JITTER. Raw per-frame probabilities flicker near the threshold. If we toggled
   speech on/off every frame we'd get choppy, useless segments. A hysteresis state
   machine fixes this: require the probability to clearly cross a HIGH threshold to
   *enter* speech and clearly drop below a LOW threshold to *leave*, and require a
   minimum number of consecutive frames before committing to a switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from src.vad.model import VADNet


@dataclass
class HysteresisGate:
    """Turn a stream of per-frame probabilities into stable speech/non-speech.

    on_thresh / off_thresh: the two thresholds (on > off) that create a "dead
        zone" so small wiggles do not cause switching.
    min_on / min_off: how many consecutive qualifying frames are needed before we
        actually flip state (debouncing). At 10 ms/frame, 5 frames = 50 ms.
    """

    on_thresh: float = 0.6
    off_thresh: float = 0.4
    min_on: int = 5
    min_off: int = 8

    is_speech: bool = field(default=False, init=False)
    _on_count: int = field(default=0, init=False)
    _off_count: int = field(default=0, init=False)

    def update(self, prob: float) -> bool:
        """Feed one frame probability; return the (smoothed) speech decision."""
        if self.is_speech:
            # Currently in speech: look for a sustained drop to exit.
            if prob < self.off_thresh:
                self._off_count += 1
                if self._off_count >= self.min_off:
                    self.is_speech = False
                    self._off_count = 0
            else:
                self._off_count = 0
        else:
            # Currently in silence: look for a sustained rise to enter.
            if prob > self.on_thresh:
                self._on_count += 1
                if self._on_count >= self.min_on:
                    self.is_speech = True
                    self._on_count = 0
            else:
                self._on_count = 0
        return self.is_speech


class StreamingVAD:
    """Wraps the trained model + a rolling context buffer for frame-by-frame use."""

    def __init__(self, model: VADNet, context_frames: int = 12,
                 device: torch.device | None = None):
        self.model = model.eval()
        self.device = device or torch.device("cpu")
        self.context = context_frames
        self.n_mels = model.net[0].in_channels
        # Buffer of recent feature frames used as left-context.
        self._buf = torch.zeros(0, self.n_mels)
        self.gate = HysteresisGate()

    @torch.no_grad()
    def push_frames(self, new_feats: torch.Tensor) -> list[tuple[float, bool]]:
        """Process a chunk of new feature frames (T_new, n_mels).

        Returns a list of (raw_prob, smoothed_is_speech) for each new frame.
        """
        # Prepend context, run the model over [context + new], keep only the new
        # frames' outputs — they were computed with proper left-context.
        window = torch.cat([self._buf, new_feats], dim=0)
        logits = self.model(window.unsqueeze(0).to(self.device))[0].cpu()
        probs = torch.sigmoid(logits)[-new_feats.shape[0]:]

        out = []
        for p in probs.tolist():
            out.append((p, self.gate.update(p)))

        # Keep the most recent `context` frames for the next chunk.
        self._buf = window[-self.context:]
        return out


def segments_from_decisions(
    decisions: list[bool], hop_s: float = 0.01
) -> list[tuple[float, float]]:
    """Collapse a per-frame speech/silence list into (start_s, end_s) segments."""
    segs: list[tuple[float, float]] = []
    start = None
    for i, d in enumerate(decisions):
        if d and start is None:
            start = i
        elif not d and start is not None:
            segs.append((start * hop_s, i * hop_s))
            start = None
    if start is not None:
        segs.append((start * hop_s, len(decisions) * hop_s))
    return segs
