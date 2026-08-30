"""
Module 9, part 1: the same corpus, a different label.

Modules 2 and 6 read LibriSpeech and kept the TRANSCRIPT. This file reads the
identical files and keeps the SPEAKER ID instead. That one swap is the whole
conceptual jump:

    ASR encoder      trained to be speaker-invariant   ("throw away who")
    speaker encoder  trained to be content-invariant   ("throw away what")

Same input features, opposite invariance. Everything below exists to force the
"content-invariant" half, because a network handed whole utterances will happily
cheat by memorising which speaker read which book.

THE FOUR DESIGN CHOICES THAT MATTER HERE

  RANDOM FIXED-LENGTH CROPS. We never show the model a whole utterance. Each
  example is a random 2 s window, so across epochs the same speaker appears with
  a different word every time and the only stable signal left is the voice. It
  also makes every tensor the same shape, which means no padding and no masks in
  the training loop.

  WHOLE SPEAKERS ARE HELD OUT, NOT WHOLE UTTERANCES. This is the one that people
  get wrong and then report a beautiful EER. If the evaluation speakers appear in
  training, you are measuring memorisation of 32 known voices, not the ability to
  compare two voices it has never met — which is the actual deployed task. See
  `split_by_speaker`.

  SPEED PERTURBATION MAKES A *NEW SPEAKER*. src/asr/corpus.py resamples to
  0.9x/1.1x and keeps the ASR label, because the words are unchanged. Do that
  here and you actively teach the model to ignore pitch and vocal-tract length —
  the two strongest identity cues it has. The standard fix (Kaldi, WeSpeaker) is
  to treat each speed as a *distinct class*: 32 speakers x 3 speeds = 96 classes.
  Same augmentation, opposite labelling, because the task's invariance flipped.

  MEAN NORMALISATION, NOT MEAN-VARIANCE. Subtracting the per-utterance mean
  removes the channel/microphone offset, which is exactly the nuisance we want
  gone. Dividing by the per-utterance standard deviation also flattens how much
  a voice's energy varies across the band — which is partly identity. Kaldi's
  x-vector recipe normalises the mean only, and that is the default here
  (`norm="mean"`); `norm="meanvar"` is available to try the other way.
"""

from __future__ import annotations

import os
import random

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.asr.corpus import LogMel, build_manifest, mix_noise, speed_perturb

SR = 16_000


# ---------------------------------------------------------------------------
# Manifests: LibriSpeech paths -> (path, speaker, duration)
# ---------------------------------------------------------------------------
def speaker_of(path: str) -> str:
    """LibriSpeech puts the speaker in the filename: `1272-128104-0000.flac`."""
    return os.path.basename(path).split("-")[0]


def build_speaker_manifest(root: str = "data", split: str = "dev-clean",
                           download: bool = False, min_s: float = 2.0,
                           max_s: float = 16.0) -> list[dict]:
    """Reuse the ASR manifest (and its cache), then attach the speaker label.

    `min_s` defaults to 2 s rather than 1 s: a crop shorter than the training
    window has to be looped to fill it, and a looped crop is a weaker example.
    """
    items = build_manifest(root=root, split=split, download=download,
                           min_s=min_s, max_s=max_s)
    return [{**it, "speaker": speaker_of(it["path"])} for it in items]


def split_by_speaker(items: list[dict], n_held_out: int = 8,
                     seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Disjoint speaker sets. The held-out speakers are NEVER trained on.

    Verification is an open-set problem: at test time you compare two voices the
    system has never heard. An EER measured on speakers that were in the training
    softmax is not that number, it is a memorisation score, and it will be
    several times too optimistic.
    """
    speakers = sorted({it["speaker"] for it in items})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    held = set(speakers[:n_held_out])
    train = [it for it in items if it["speaker"] not in held]
    trial = [it for it in items if it["speaker"] in held]
    return train, trial


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def normalize_feats(feats: torch.Tensor, norm: str = "mean") -> torch.Tensor:
    """(T, n_mels) -> normalised. See the module docstring on mean vs mean-var."""
    if norm == "none":
        return feats
    out = feats - feats.mean(0, keepdim=True)
    if norm == "meanvar":
        out = out / (feats.std(0, keepdim=True) + 1e-5)
    return out


def random_crop(wav: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    """A random n-sample window, looping the clip if it is too short."""
    if len(wav) < n:
        reps = n // max(1, len(wav)) + 1
        wav = np.tile(wav, reps)
    if len(wav) == n:
        return wav
    off = rng.randint(0, len(wav) - n)
    return wav[off:off + n]


# ---------------------------------------------------------------------------
class SpeakerCrops(Dataset):
    """Random fixed-length crops labelled with (speaker, speed) class ids.

    Every item comes back as (feats (T, n_mels), label) with T identical across
    the dataset, so the default collate stacks them and the training loop needs
    no lengths and no masking.
    """

    def __init__(self, items: list[dict], crop_seconds: float = 2.0,
                 train: bool = True, speeds: tuple[float, ...] = (0.9, 1.0, 1.1),
                 noise_prob: float = 0.0, snr_range: tuple[float, float] = (5.0, 20.0),
                 n_mels: int = 80, norm: str = "mean", seed: int = 0):
        self.items = items
        self.crop = int(crop_seconds * SR)
        self.train = train
        # At eval time there is no augmentation, so there is exactly one class
        # per speaker and `label()` collapses to the speaker index.
        self.speeds = tuple(speeds) if train else (1.0,)
        self.noise_prob = noise_prob if train else 0.0
        self.snr_range = snr_range
        self.norm = norm
        self.logmel = LogMel(n_mels=n_mels)
        self.speakers = sorted({it["speaker"] for it in items})
        self.spk_to_idx = {s: i for i, s in enumerate(self.speakers)}
        self.seed = seed
        self._noise_files: list[str] | None = None

    def __len__(self) -> int:
        return len(self.items)

    @property
    def n_speakers(self) -> int:
        return len(self.speakers)

    @property
    def n_classes(self) -> int:
        """Speakers x speeds — see "speed perturbation makes a new speaker"."""
        return len(self.speakers) * len(self.speeds)

    def label(self, speaker: str, speed_idx: int) -> int:
        return self.spk_to_idx[speaker] * len(self.speeds) + speed_idx

    def _noise(self, n: int, rng: random.Random) -> np.ndarray:
        """Module 1b's noise generators, reused verbatim."""
        from src.vad.data_real import _babble, _colored_noise, _hum, list_speech_files

        if self._noise_files is None:
            try:
                self._noise_files = list_speech_files()
            except FileNotFoundError:
                self._noise_files = []
        kind = rng.random()
        if kind < 0.5 or not self._noise_files:
            return _colored_noise(n)
        if kind < 0.85:
            return _babble(n, self._noise_files)
        return _hum(n)

    def __getitem__(self, i: int):
        item = self.items[i]
        # Per-item RNG seeded by (epoch-independent) index + a per-worker draw:
        # DataLoader workers each get a fork of `random`, so plain `random` is
        # already independent per worker; we only use a local Random so a test
        # can reproduce one item.
        rng = random
        wav, sr = sf.read(item["path"], dtype="float32")
        assert sr == SR, f"expected {SR} Hz, got {sr}"

        speed_idx = rng.randrange(len(self.speeds)) if self.train else 0
        rate = self.speeds[speed_idx]

        # Crop BEFORE perturbing, and crop `crop * rate` samples so that
        # resampling lands on exactly `crop` samples. Cropping first also means
        # we resample 2 s instead of the whole 15 s utterance.
        need = int(round(self.crop * rate))
        wav = random_crop(wav, need, rng)
        if rate != 1.0:
            wav = speed_perturb(wav, rate)
            wav = random_crop(wav, self.crop, rng)   # fix off-by-one rounding

        if self.noise_prob and rng.random() < self.noise_prob:
            wav = mix_noise(wav, self._noise(len(wav), rng),
                            rng.uniform(*self.snr_range))

        feats = self.logmel(torch.from_numpy(np.ascontiguousarray(wav)))
        feats = normalize_feats(feats, self.norm)
        return feats, self.label(item["speaker"], speed_idx)
