"""
Module 6, part 5: the data pipeline — where most of the WER actually lives.

Module 2 fed the model clean LibriSpeech and nothing else, and the held-out gap
was enormous. That gap is not a mystery: with a few hours of read speech, a
model with any capacity memorises the speakers. The modern recipe attacks this
in three places, and on corpora under ~1000 hours the augmentation is worth
MORE than doubling the data:

  SPEED PERTURBATION (waveform). Resample to 0.9x / 1.1x and keep the label.
  Triples the corpus and, because resampling shifts pitch too, it is really
  speaker augmentation — exactly the axis a small corpus is short on.

  NOISE / SNR MIXING (waveform). Reuses Module 1b's noise generators. A model
  that has only heard studio speech falls apart on a phone call, and this is the
  cheapest possible insurance.

  SPECAUGMENT (features). Mask random frequency bands and time spans. Forces
  the model to infer a phoneme from partial evidence rather than a single cue,
  and it is the single highest-leverage line in any small-data ASR recipe.

Two systems details that are not optional:

  CMVN. Per-utterance mean/variance normalisation of the log-mel. Log-mel values
  live around -20 to +5 with a per-recording offset that depends on microphone
  gain. Feeding that to a LayerNorm-free conv front-end wastes the model's
  capacity learning to subtract a constant, and makes training unstable.

  LENGTH-BUCKETED DYNAMIC BATCHING. Batching by *count* means a batch with one
  20 s utterance pads twelve 2 s ones to 20 s — most of the GPU time computes
  padding. Batching by total FRAMES with length-sorted buckets typically gives
  2-4x more real audio per step at the same memory.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from src.asr.tokenizer import BPETokenizer, normalize
from src.audio.features import frame_signal, mel_filterbank, power_spectrum

SR = 16_000


# ---------------------------------------------------------------------------
class LogMel:
    """The repo's own front-end, with the filterbank built ONCE.

    src/audio/features.py rebuilds the 80 triangular filters on every call
    (a Python loop) — fine for a demo, a real cost when it runs a few hundred
    thousand times during training.
    """

    def __init__(self, sr: int = SR, n_mels: int = 80, frame_ms: float = 25.0,
                 hop_ms: float = 10.0, n_fft: int = 512):
        self.frame_length = int(sr * frame_ms / 1000)
        self.hop_length = int(sr * hop_ms / 1000)
        self.n_fft = n_fft
        self.fb = mel_filterbank(n_mels, n_fft, sr)          # (n_mels, n_fft//2+1)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        frames = frame_signal(wav, self.frame_length, self.hop_length)
        power = power_spectrum(frames, self.n_fft)
        return torch.log(power @ self.fb.T + 1e-10)          # (T, n_mels)


def cmvn(feats: torch.Tensor) -> torch.Tensor:
    return (feats - feats.mean(0, keepdim=True)) / (feats.std(0, keepdim=True) + 1e-5)


def spec_augment(feats: torch.Tensor, n_freq: int = 2, freq_width: int = 27,
                 n_time: int = 2, time_ratio: float = 0.05,
                 max_time_width: int = 40) -> torch.Tensor:
    """Mask bands and spans in place-ish. feats: (T, n_mels)."""
    t, f = feats.shape
    feats = feats.clone()
    for _ in range(n_freq):
        w = random.randint(0, freq_width)
        if w and f > w:
            f0 = random.randint(0, f - w)
            feats[:, f0:f0 + w] = 0.0
    width = min(max_time_width, max(1, int(t * time_ratio)))
    for _ in range(n_time):
        w = random.randint(0, width)
        if w and t > w:
            t0 = random.randint(0, t - w)
            feats[t0:t0 + w, :] = 0.0
    return feats


def speed_perturb(wav: np.ndarray, rate: float) -> np.ndarray:
    """Linear-interpolation resample. Speeds up by `rate` and shifts pitch with
    it — that pitch shift is the point, not a bug (Ko et al. 2015)."""
    if rate == 1.0:
        return wav
    n_out = int(round(len(wav) / rate))
    src = np.linspace(0.0, len(wav) - 1, n_out, dtype=np.float32)
    return np.interp(src, np.arange(len(wav), dtype=np.float32), wav).astype(np.float32)


def mix_noise(wav: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(wav):
        noise = np.tile(noise, len(wav) // max(1, len(noise)) + 1)
    noise = noise[: len(wav)]
    scale = np.sqrt(np.mean(wav ** 2)) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    return (wav + noise * scale * 10.0 ** (-snr_db / 20.0)).astype(np.float32)


# ---------------------------------------------------------------------------
def build_manifest(root: str = "data", split: str = "dev-clean",
                   download: bool = False, min_s: float = 1.0,
                   max_s: float = 16.0, cache_dir: str = "outputs") -> list[dict]:
    """[{path, text, duration}] for a LibriSpeech split, cached to JSON.

    Scanning file headers takes a while on train-clean-100 (28k files); doing it
    once and caching is the difference between a 40 s and a 0.2 s startup.
    """
    cache = Path(cache_dir) / f"manifest_{split}.json"
    items = None
    if cache.exists():
        items = json.loads(cache.read_text())
        # A cached manifest is a list of ABSOLUTE paths. Carry it to a fresh
        # machine (a new GPU pod, say) and it looks valid while pointing at
        # audio that isn't there — and because we'd return early, the download
        # never runs and training dies later on the first file read. Spot-check
        # instead of trusting the cache.
        sample = items[:: max(1, len(items) // 20)] if items else []
        if not items or not all(os.path.exists(it["path"]) for it in sample):
            print(f"manifest cache {cache} points at missing audio — rebuilding")
            items = None
    if items is None:
        import torchaudio

        ds = torchaudio.datasets.LIBRISPEECH(root, url=split, download=download)
        base = os.path.dirname(ds._path)
        items = []
        for i in range(len(ds)):
            rel, _sr, text, *_ = ds.get_metadata(i)
            full = os.path.join(base, rel)
            items.append({"path": full, "text": text,
                          "duration": sf.info(full).duration})
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(items))
    return [it for it in items if min_s <= it["duration"] <= max_s]


class SpeechCorpus(Dataset):
    """(log-mel features, token ids) with waveform + feature augmentation."""

    def __init__(self, items: list[dict], tokenizer: BPETokenizer,
                 train: bool = True, speeds=(0.9, 1.0, 1.1),
                 noise_prob: float = 0.0, snr_range=(5.0, 20.0),
                 n_mels: int = 80):
        self.items = items
        self.tokenizer = tokenizer
        self.train = train
        self.speeds = speeds if train else (1.0,)
        self.noise_prob = noise_prob if train else 0.0
        self.snr_range = snr_range
        self.logmel = LogMel(n_mels=n_mels)
        self._noise_files: list[str] | None = None
        # Toggled by the training loop: see `augment_enabled` below.
        self.augment_enabled = True

    def __len__(self) -> int:
        return len(self.items)

    def _noise(self, n: int) -> np.ndarray:
        from src.vad.data_real import _babble, _colored_noise, _hum, list_speech_files

        if self._noise_files is None:
            try:
                self._noise_files = list_speech_files()
            except FileNotFoundError:
                self._noise_files = []
        kind = random.random()
        if kind < 0.5 or not self._noise_files:
            return _colored_noise(n)
        if kind < 0.85:
            return _babble(n, self._noise_files)
        return _hum(n)

    def __getitem__(self, i: int):
        item = self.items[i]
        wav, sr = sf.read(item["path"], dtype="float32")
        assert sr == SR, f"expected {SR} Hz, got {sr}"

        aug = self.train and self.augment_enabled
        if aug:
            wav = speed_perturb(wav, random.choice(self.speeds))
            if random.random() < self.noise_prob:
                wav = mix_noise(wav, self._noise(len(wav)),
                                random.uniform(*self.snr_range))

        feats = cmvn(self.logmel(torch.from_numpy(np.ascontiguousarray(wav))))
        if aug:
            feats = spec_augment(feats)
        tokens = self.tokenizer.encode(item["text"])
        return feats, torch.tensor(tokens, dtype=torch.long), normalize(item["text"])


def collate(batch):
    """-> feats (B,Tmax,F), feat_lens, ys (B,Lmax), ys_lens, texts."""
    feats, ys, texts = zip(*batch)
    f_lens = torch.tensor([f.size(0) for f in feats], dtype=torch.long)
    y_lens = torch.tensor([y.size(0) for y in ys], dtype=torch.long)
    f_pad = torch.zeros(len(feats), int(f_lens.max()), feats[0].size(1))
    y_pad = torch.zeros(len(ys), int(y_lens.max().clamp(min=1)), dtype=torch.long)
    for i, (f, y) in enumerate(zip(feats, ys)):
        f_pad[i, : f.size(0)] = f
        y_pad[i, : y.size(0)] = y
    return f_pad, f_lens, y_pad, y_lens, list(texts)


class DynamicBatchSampler(torch.utils.data.Sampler):
    """Batches of roughly constant TOTAL frames, grouped by similar length."""

    def __init__(self, items: list[dict], max_frames: int = 12_000,
                 max_batch: int = 32, shuffle: bool = True, seed: int = 0):
        self.batches: list[list[int]] = []
        order = sorted(range(len(items)), key=lambda i: items[i]["duration"])
        cur: list[int] = []
        cur_max = 0
        for i in order:
            frames = int(items[i]["duration"] * 100)     # 10 ms hop
            nxt_max = max(cur_max, frames)
            # Cost of the padded batch is (batch size) x (longest member).
            if cur and (nxt_max * (len(cur) + 1) > max_frames
                        or len(cur) >= max_batch):
                self.batches.append(cur)
                cur, cur_max = [i], frames
            else:
                cur.append(i)
                cur_max = nxt_max
        if cur:
            self.batches.append(cur)
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        batches = list(self.batches)
        if self.shuffle:
            # Shuffle the ORDER of batches, not their contents: keeping each
            # batch length-homogeneous is the whole point.
            random.Random(self.seed + self.epoch).shuffle(batches)
            self.epoch += 1
        return iter(batches)

    def __len__(self) -> int:
        return len(self.batches)


def build_tokenizer(items: list[dict], vocab_size: int, path: str | Path) -> BPETokenizer:
    path = Path(path)
    if path.exists():
        tok = BPETokenizer.load(path)
        if tok.vocab_size == vocab_size:
            return tok
    tok = BPETokenizer.train((it["text"] for it in items), vocab_size=vocab_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    tok.save(path)
    return tok


def frames_per_hour(items: list[dict]) -> float:
    return sum(it["duration"] for it in items) / 3600.0


def warmup_cosine(step: int, warmup: int, total: int, peak: float,
                  floor: float = 0.05) -> float:
    """Linear warmup then cosine decay — the standard transformer schedule.

    Warmup is not optional for attention models: at step 0 the attention is
    uniform, so gradients through the softmax are large and badly scaled, and a
    full-size LR at that point reliably diverges.
    """
    if step < warmup:
        return peak * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    progress = min(1.0, progress)
    return peak * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))
