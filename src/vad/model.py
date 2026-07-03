"""
Module 1, part 2: the VAD model.

This is a *frame-level binary classifier*: given log-mel features of shape
(T, n_mels), output one speech-probability per frame, shape (T,).

Architecture: a small stack of 1-D convolutions along the time axis.
Why conv-over-time rather than classifying each frame independently?
  - Speech has temporal continuity: a frame is much more likely to be speech if
    its neighbors are. Convolutions let each output frame "see" a window of
    neighboring frames (its receptive field), so the model naturally smooths and
    uses context — the single most important prior for VAD.
  - It is tiny and fast, which matches real VAD (it must be cheaper than the ASR
    it gates).

We treat the n_mels axis as input *channels* to Conv1d, and the time axis as the
length we convolve over.
"""

from __future__ import annotations

import torch
from torch import nn


class VADNet(nn.Module):
    def __init__(self, n_mels: int = 80, hidden: int = 64):
        super().__init__()
        # padding='same'-style (kernel 5, pad 2) keeps the time length unchanged,
        # so output frames line up 1:1 with input frames (and with our labels).
        self.net = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.head = nn.Conv1d(hidden, 1, kernel_size=1)  # per-frame logit

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, n_mels) -> logits: (B, T).

        Conv1d wants (B, channels, length), so we move n_mels into the channel
        slot and time into the length slot, then transpose back.
        """
        x = feats.transpose(1, 2)        # (B, n_mels, T)
        x = self.net(x)                  # (B, hidden, T)
        logits = self.head(x)            # (B, 1, T)
        return logits.squeeze(1)         # (B, T)
