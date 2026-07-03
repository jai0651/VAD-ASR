"""
Module 2, part 2: synthetic "spoken text".

We need (audio, text) pairs where the audio is recognizable from the text but the
*timing is not given*. Recipe:
  - Give each character a fixed acoustic "signature": a base pitch + a second
    harmonic band (a fake formant). Different characters sound different.
  - Render each character for a RANDOM duration (40-120 ms) with amplitude
    envelope, slight pitch jitter, and background noise. Random durations are the
    whole point: the model gets, e.g., 60 frames for a 4-char word and must learn
    the alignment itself via CTC.
  - 'space' renders as a short near-silence.

This is deliberately *not* real speech. It exists so the CTC machinery, the
model, and decoding can be built and verified offline. Module 2b swaps in real
LibriSpeech with the SAME model and training code.
"""

from __future__ import annotations

import math
import random

import torch

from src.audio.features import log_mel_spectrogram
from src.asr.text import CHARS, encode

SR = 16_000

# A small vocabulary of real words so predictions are human-readable.
WORDS = [
    "cat", "dog", "hello", "world", "speech", "audio", "model", "data",
    "voice", "sound", "learn", "neural", "text", "open", "blue", "seven",
    "music", "happy", "river", "light", "table", "green", "north", "south",
]


def _char_freqs(ch: str) -> tuple[float, float]:
    """Map a character to a (base, formant) frequency pair deterministically."""
    idx = CHARS.index(ch)
    base = 140.0 + idx * 90.0          # spread bases across ~140-2500 Hz
    formant = 800.0 + (idx % 6) * 350.0
    return base, formant


def _render_char(ch: str) -> torch.Tensor:
    if ch == " ":
        dur = random.uniform(0.04, 0.08)
        return 0.01 * torch.randn(int(dur * SR))  # near-silence gap

    base, formant = _char_freqs(ch)
    dur = random.uniform(0.05, 0.12)
    n = int(dur * SR)
    t = torch.arange(n) / SR

    sig = torch.zeros(n)
    for k in range(1, 16):
        freq = base * k
        if freq >= SR / 2:
            break
        gain = math.exp(-((freq - formant) ** 2) / (2 * 300.0**2)) + 0.3 / k
        jitter = 1.0 + 0.01 * math.sin(2 * math.pi * 5 * random.random())
        sig += gain * torch.sin(2 * math.pi * freq * jitter * t)

    # Raised-cosine envelope so characters fade in/out (no clicks).
    env = torch.hann_window(n) if n > 1 else torch.ones(n)
    sig = sig * env
    sig = sig / (sig.abs().max() + 1e-9)
    return sig + 0.02 * torch.randn(n)


def render_text(text: str) -> torch.Tensor:
    return torch.cat([_render_char(c) for c in text.lower()])


def make_example(max_words: int = 2) -> tuple[torch.Tensor, list[int], str]:
    """Returns (log-mel feats [T, n_mels], target label indices, text)."""
    n_words = random.randint(1, max_words)
    text = " ".join(random.choice(WORDS) for _ in range(n_words))
    wav = render_text(text)
    feats = log_mel_spectrogram(wav, sr=SR, hop_ms=10.0)
    return feats, encode(text), text


def make_batch(batch_size: int, max_words: int = 2):
    """A padded batch ready for torch.nn.CTCLoss.

    Returns:
        feats:          (B, T_max, n_mels)
        input_lengths:  (B,)  real frame count per example (before padding)
        targets:        (sum(target_lengths),)  all targets concatenated
        target_lengths: (B,)
        texts:          list[str] for readable logging
    CTC takes targets concatenated + their lengths (not padded) — that is just
    the API torch chose.
    """
    examples = [make_example(max_words) for _ in range(batch_size)]
    t_max = max(f.shape[0] for f, _, _ in examples)
    n_mels = examples[0][0].shape[1]

    feats = torch.zeros(batch_size, t_max, n_mels)
    input_lengths = torch.zeros(batch_size, dtype=torch.long)
    target_lengths = torch.zeros(batch_size, dtype=torch.long)
    all_targets: list[int] = []
    texts: list[str] = []
    for i, (f, tgt, text) in enumerate(examples):
        feats[i, : f.shape[0]] = f
        input_lengths[i] = f.shape[0]
        target_lengths[i] = len(tgt)
        all_targets.extend(tgt)
        texts.append(text)

    return feats, input_lengths, torch.tensor(all_targets), target_lengths, texts
