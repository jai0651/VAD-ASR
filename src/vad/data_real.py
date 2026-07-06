"""
Module 1b: a REAL training set for VAD — the fix for the live-mic failure.

The synthetic VAD (data.py) failed on a real microphone: room ambience read as
P(speech)≈1, turns never ended, and phantom barge-ins flushed every reply
(docs/02-vad.html §2.5). The architecture was never the problem — this file
replaces only the DATA, following the recipe real VAD training sets use
(MUSAN-style): real speech, corrupted by realistic backgrounds, with labels
derived from the clean signal BEFORE mixing.

The three ideas:

  REAL SPEECH, GATED LABELS. Speech clips come from LibriSpeech (real human
  voices, already on disk). But an utterance contains pauses — labeling every
  frame "speech" would teach the model that silence is speech (a miniature of
  the exact bug we're fixing). So we compute per-frame energy on the CLEAN
  clip and gate: frames within 30 dB of the clip's loudest frame are speech.
  Then we mix noise on top — the labels stay true because we made them before
  the corruption. Free supervision, the same trick as Module 1's synthetic
  labels, now on real voices.

  HARD NEGATIVES. What a mic actually sends when nobody is talking: room
  tone (colored noise at many levels), BABBLE (many overlapping distant
  voices — built by summing real LibriSpeech clips, exactly how MUSAN does
  it; the single hardest negative for a VAD), mains hum, near-silence, and
  pure digital zeros (the earlier out-of-distribution bug — now in-domain).

  AUGMENTATION = the serving path's dirt. Random SNR (0-25 dB), random gain
  (quiet mics to hot mics), occasional clipping. If serving can produce it,
  training must contain it.
"""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from src.audio.features import log_mel_spectrogram

SR = 16_000
HOP = 160          # 10 ms — one label per feature frame
FRAME = 400        # 25 ms energy window, matches feature extraction
GATE_DB = 30.0     # frames within this of the clip's peak count as speech
LIBRI_DIR = "data/LibriSpeech/dev-clean"


# ----------------------------------------------------------------------------
# Real speech: files + energy-gated frame labels
# ----------------------------------------------------------------------------
def list_speech_files(root: str = LIBRI_DIR) -> list[str]:
    files = [str(p) for p in Path(root).rglob("*.flac")]
    if not files:
        raise FileNotFoundError(
            f"no .flac under {root} — run scripts/05 once (downloads LibriSpeech)"
        )
    return files


def load_clip(path: str, max_s: float = 4.0) -> np.ndarray:
    """A random crop of one utterance, peak-normalized."""
    audio, sr = sf.read(path, dtype="float32")
    assert sr == SR
    max_len = int(max_s * SR)
    if audio.shape[0] > max_len:
        start = random.randrange(audio.shape[0] - max_len)
        audio = audio[start:start + max_len]
    peak = np.abs(audio).max()
    return audio / peak if peak > 0 else audio


def frame_rms_db(x: np.ndarray) -> np.ndarray:
    """Per-frame RMS in dB (25 ms window / 10 ms hop, same grid as features)."""
    n_frames = max(0, 1 + (x.shape[0] - FRAME) // HOP)
    rms = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        w = x[i * HOP: i * HOP + FRAME]
        rms[i] = np.sqrt((w * w).mean() + 1e-12)
    return 20.0 * np.log10(rms + 1e-12)


def speech_labels(clean: np.ndarray) -> np.ndarray:
    """Per-frame 0/1 labels for a CLEAN speech clip via relative energy gating.

    Computed before any noise is added, so they remain ground truth after
    mixing. Dilated by one frame each side: word onsets/offsets are low-energy
    but perceptually speech, and a slightly generous boundary also teaches the
    pre-roll behavior we want.
    """
    db = frame_rms_db(clean)
    lab = (db > db.max() - GATE_DB).astype(np.float32)
    dilated = lab.copy()
    dilated[1:] = np.maximum(dilated[1:], lab[:-1])
    dilated[:-1] = np.maximum(dilated[:-1], lab[1:])
    return dilated


# ----------------------------------------------------------------------------
# Hard negatives: what a real mic sends when nobody is talking
# ----------------------------------------------------------------------------
def _colored_noise(n: int) -> np.ndarray:
    """White/pink/brown noise (increasingly low-frequency — like room rumble)."""
    white = np.random.randn(n).astype(np.float32)
    kind = random.random()
    if kind < 0.34:
        out = white
    else:
        spec = np.fft.rfft(white)
        f = np.maximum(np.fft.rfftfreq(n, 1 / SR), 1.0)
        spec = spec / (f ** (0.5 if kind < 0.67 else 1.0))  # pink | brown
        out = np.fft.irfft(spec, n=n).astype(np.float32)
    return out / (np.abs(out).max() + 1e-9)


def _babble(n: int, files: list[str], n_voices: int = 6) -> np.ndarray:
    """Overlapping distant voices — the hardest non-speech for a VAD. Summing
    many real speakers destroys the single-voice harmonic/rhythm structure
    the model should key on, while keeping speech-band energy."""
    out = np.zeros(n, dtype=np.float32)
    for _ in range(n_voices):
        clip = load_clip(random.choice(files), max_s=n / SR)
        start = random.randrange(max(1, n - clip.shape[0] + 1))
        out[start:start + clip.shape[0]] += clip[: n - start]
    return out / (np.abs(out).max() + 1e-9)


def _hum(n: int) -> np.ndarray:
    """Mains hum: 50/60 Hz + harmonics, the classic electrical background."""
    t = np.arange(n, dtype=np.float32) / SR
    f0 = random.choice([50.0, 60.0])
    out = sum(
        np.sin(2 * np.pi * f0 * k * t) / k for k in (1, 2, 3)
    ) + 0.2 * np.random.randn(n).astype(np.float32)
    return (out / (np.abs(out).max() + 1e-9)).astype(np.float32)


def noise_bed(n: int, files: list[str]) -> np.ndarray:
    """A random background at a random realistic level (incl. digital zero)."""
    r = random.random()
    if r < 0.10:
        return np.zeros(n, dtype=np.float32)          # exact zeros: in-domain now
    if r < 0.45:
        base = _colored_noise(n)
    elif r < 0.75:
        base = _babble(n, files)
    elif r < 0.85:
        base = _hum(n)
    else:
        base = np.random.randn(n).astype(np.float32) * 0.3
    level_db = random.uniform(-55.0, -18.0)           # faint room tone -> loud
    return base * (10.0 ** (level_db / 20.0))


# ----------------------------------------------------------------------------
# Example builder: speech clips placed over a noise bed at random SNR
# ----------------------------------------------------------------------------
def make_real_example(
    files: list[str], total_s: float = 5.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """One training example: (log-mel feats (T, 80), frame labels (T,))."""
    n = int(total_s * SR)
    mix = noise_bed(n, files)
    labels = np.zeros(1 + (n - FRAME) // HOP, dtype=np.float32)

    # Half the examples contain no speech at all: a VAD must be confident
    # about pure background, not merely prefer speech when both are present.
    for _ in range(random.choice([0, 1, 1, 2])):
        clip = load_clip(random.choice(files), max_s=min(3.0, total_s - 0.5))
        lab = speech_labels(clip)
        start = random.randrange(max(1, n - clip.shape[0] + 1))
        snr_db = random.uniform(0.0, 25.0)
        noise_rms = np.sqrt((mix ** 2).mean() + 1e-12)
        clip_rms = np.sqrt((clip ** 2).mean() + 1e-12)
        gain = (noise_rms / clip_rms) * (10.0 ** (snr_db / 20.0)) if noise_rms > 1e-7 \
            else random.uniform(0.05, 0.8) / (clip_rms + 1e-9)
        mix[start:start + clip.shape[0]] += clip * gain

        f0 = start // HOP
        labels[f0:f0 + lab.shape[0]] = np.maximum(
            labels[f0:f0 + lab.shape[0]], lab[: labels.shape[0] - f0]
        )

    # Serving-path dirt: random overall gain; occasional clipping.
    mix *= 10.0 ** (random.uniform(-20.0, 0.0) / 20.0)
    if random.random() < 0.1:
        mix = np.clip(mix * 3.0, -1.0, 1.0)
    peak = np.abs(mix).max()
    if peak > 1.0:
        mix /= peak

    feats = log_mel_spectrogram(torch.from_numpy(mix), sr=SR)
    t = min(feats.shape[0], labels.shape[0])
    return feats[:t], torch.from_numpy(labels[:t])


def make_real_batch(
    files: list[str], batch_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Padded batch, same format as the synthetic make_batch (feats, labels, mask)."""
    examples = [make_real_example(files) for _ in range(batch_size)]
    t_max = max(f.shape[0] for f, _ in examples)
    feats = torch.zeros(batch_size, t_max, 80)
    labels = torch.zeros(batch_size, t_max)
    mask = torch.zeros(batch_size, t_max)
    for i, (f, lab) in enumerate(examples):
        feats[i, : f.shape[0]] = f
        labels[i, : lab.shape[0]] = lab
        mask[i, : f.shape[0]] = 1.0
    return feats, labels, mask
