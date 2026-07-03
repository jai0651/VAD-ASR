"""
Module 1, part 1: a labeled dataset for Voice Activity Detection.

A VAD model answers, for each short frame of audio: "speech or not?" To train it
we need audio plus a per-frame 0/1 label.

Real pipelines build this from speech corpora (e.g. LibriSpeech) mixed with noise
(e.g. MUSAN); the labels come from knowing where the speech clips were inserted.
We do the same thing, but *synthesize* the speech and noise so it runs offline and
you can see exactly where the labels come from.

What makes a segment sound like speech (a crude but instructive model):
  - VOICED sounds (vowels) = a low fundamental f0 (~80-250 Hz, the pitch of the
    voice) plus its harmonics, shaped by "formants" (resonances of the vocal
    tract) that emphasize certain frequency bands. We also add a slow amplitude
    wobble, because real speech is never a steady tone.
  - UNVOICED sounds (s, f, sh) = filtered noise bursts.
Non-speech is silence (very low noise) or steady colored noise.

A real network never sees our generator, so it must learn the *statistical*
difference (harmonic structure + modulation) rather than memorize exact tones.
"""

from __future__ import annotations

import math

import torch

from src.audio.features import log_mel_spectrogram

SR = 16_000
HOP_MS = 10.0  # must match feature extraction; defines the frame rate (100/s)


def _voiced_segment(dur_s: float) -> torch.Tensor:
    """A vowel-like sound: fundamental + harmonics + formant emphasis + wobble."""
    n = int(dur_s * SR)
    t = torch.arange(n) / SR
    f0 = float(torch.empty(1).uniform_(90, 220))  # speaker pitch
    # Two formants drawn from typical vowel ranges.
    formants = [float(torch.empty(1).uniform_(300, 900)),
                float(torch.empty(1).uniform_(1200, 2600))]

    sig = torch.zeros(n)
    for k in range(1, 25):  # harmonics of the fundamental
        freq = f0 * k
        if freq >= SR / 2:
            break
        # Emphasize harmonics near a formant (resonance), de-emphasize others.
        gain = 0.0
        for fmt in formants:
            gain += math.exp(-((freq - fmt) ** 2) / (2 * 150.0**2))
        sig += gain * torch.sin(2 * torch.pi * freq * t)

    # Slow amplitude modulation (~4-7 Hz), the natural "syllable" rhythm.
    mod_rate = float(torch.empty(1).uniform_(4, 7))
    envelope = 0.6 + 0.4 * torch.sin(2 * torch.pi * mod_rate * t)
    sig = sig * envelope
    return sig / (sig.abs().max() + 1e-9)


def _unvoiced_segment(dur_s: float) -> torch.Tensor:
    """A fricative-like sound: band-emphasized noise (e.g. 's', 'sh')."""
    n = int(dur_s * SR)
    noise = torch.randn(n)
    # Crude high-pass: difference filter pushes energy toward high frequencies.
    noise = noise - torch.cat([torch.zeros(1), noise[:-1]])
    return noise / (noise.abs().max() + 1e-9)


def _noise_segment(dur_s: float, level: float) -> torch.Tensor:
    """Background: low-level white-ish noise (silence is just very low level)."""
    n = int(dur_s * SR)
    return level * torch.randn(n)


def make_utterance(
    total_s: float = 4.0, speech_ratio: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one random utterance and its per-sample speech mask.

    Returns:
        waveform: (num_samples,)
        sample_labels: (num_samples,) of 0/1 — speech where 1.
    """
    chunks: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    elapsed = 0.0
    while elapsed < total_s:
        seg_s = float(torch.empty(1).uniform_(0.2, 0.6))
        is_speech = torch.rand(1).item() < speech_ratio
        if is_speech:
            seg = (_voiced_segment(seg_s) if torch.rand(1).item() < 0.7
                   else _unvoiced_segment(seg_s))
            lab = torch.ones(seg.shape[0])
        else:
            # Either near-silence or audible background noise.
            level = 0.005 if torch.rand(1).item() < 0.5 else 0.05
            seg = _noise_segment(seg_s, level)
            lab = torch.zeros(seg.shape[0])
        chunks.append(seg)
        labels.append(lab)
        elapsed += seg_s

    waveform = torch.cat(chunks)
    sample_labels = torch.cat(labels)

    # Always add a faint background everywhere so "speech" is never the only
    # thing with energy — the model must learn structure, not just loudness.
    waveform = waveform + 0.01 * torch.randn_like(waveform)
    return waveform, sample_labels


def frame_labels_from_samples(
    sample_labels: torch.Tensor, num_frames: int
) -> torch.Tensor:
    """Convert per-sample labels to per-frame labels by majority vote in each hop.

    Our features produce one frame every HOP_MS; a frame is "speech" if most of
    its samples were speech.
    """
    hop = int(SR * HOP_MS / 1000)
    frame_lab = torch.zeros(num_frames)
    for i in range(num_frames):
        start = i * hop
        window = sample_labels[start:start + hop]
        if window.numel() > 0:
            frame_lab[i] = (window.mean() > 0.5).float()
    return frame_lab


def make_example() -> tuple[torch.Tensor, torch.Tensor]:
    """One training example: (log-mel features [T, n_mels], frame labels [T])."""
    wav, sample_labels = make_utterance()
    feats = log_mel_spectrogram(wav, sr=SR, hop_ms=HOP_MS)
    labels = frame_labels_from_samples(sample_labels, feats.shape[0])
    return feats, labels


def make_batch(batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A padded batch.

    Returns:
        feats:   (B, T_max, n_mels)
        labels:  (B, T_max)
        mask:    (B, T_max) of 1 where real (not padding), 0 where padded.
    Padding lets us batch variable-length utterances; the mask tells the loss to
    ignore padded frames.
    """
    examples = [make_example() for _ in range(batch_size)]
    t_max = max(f.shape[0] for f, _ in examples)
    n_mels = examples[0][0].shape[1]

    feats = torch.zeros(batch_size, t_max, n_mels)
    labels = torch.zeros(batch_size, t_max)
    mask = torch.zeros(batch_size, t_max)
    for i, (f, lab) in enumerate(examples):
        t = f.shape[0]
        feats[i, :t] = f
        labels[i, :t] = lab
        mask[i, :t] = 1.0
    return feats, labels, mask
