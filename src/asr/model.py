"""
Module 2, part 3: the ASR acoustic model (a CTC encoder).

It maps log-mel features (T, n_mels) to per-frame character distributions
(T, vocab). Structure mirrors classic CTC recognizers (e.g. DeepSpeech):

    conv front-end   -> learns local time-frequency patterns (edges, bands)
    bidirectional GRU -> mixes information across the whole utterance, so the
                         label at frame t can depend on context before AND after
    linear head      -> one score per vocabulary symbol (chars + blank)
    log-softmax      -> log-probabilities, the form CTCLoss and decoders expect

We do a x2 time downsample in the conv stack: adjacent feature frames are highly
redundant, halving T speeds up the GRU and CTC with no accuracy loss, as long as
the remaining frame count stays >= target length (it does for our short words).
"""

from __future__ import annotations

import torch
from torch import nn

from src.asr.text import VOCAB_SIZE


class ASRModel(nn.Module):
    def __init__(self, n_mels: int = 80, hidden: int = 128, vocab: int = VOCAB_SIZE):
        super().__init__()
        # Conv over time; stride-2 in the second layer halves the frame rate.
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, hidden, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, hidden, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
        )
        self.rnn = nn.GRU(
            hidden, hidden, num_layers=2, batch_first=True,
            bidirectional=True, dropout=0.1,
        )
        self.head = nn.Linear(hidden * 2, vocab)  # *2: forward+backward GRU

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: (B, T, n_mels) -> log_probs: (B, T', vocab)."""
        x = feats.transpose(1, 2)     # (B, n_mels, T) for Conv1d
        x = self.conv(x)              # (B, hidden, T')
        x = x.transpose(1, 2)         # (B, T', hidden)
        x, _ = self.rnn(x)            # (B, T', 2*hidden)
        logits = self.head(x)         # (B, T', vocab)
        return torch.log_softmax(logits, dim=-1)

    def downsampled_length(self, input_length: torch.Tensor) -> torch.Tensor:
        """How many output frames remain after the stride-2 conv.

        Needed so CTCLoss knows each example's true (post-downsample) frame count.
        With kernel 5, stride 2, padding 2: T' = floor((T + 1) / 2).
        """
        return torch.div(input_length + 1, 2, rounding_mode="floor")
