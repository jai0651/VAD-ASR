"""
Module 2b: real speech via LibriSpeech.

This is the *only* file that changes when moving from synthetic audio to real
human speech. It produces exactly the same batch format as src/asr/data.py
(feats, input_lengths, concatenated targets, target_lengths), so the model in
src/asr/model.py, the CTC loss, and the decoders are all reused unchanged.

LibriSpeech is read English audiobooks at 16 kHz with clean transcripts — the
standard ASR learning corpus. `dev-clean` (~337 MB, ~5.4 h) is small enough to
experiment with. Transcripts are uppercase letters + spaces (plus apostrophes);
our vocabulary is " a-z", so encode() lowercases and drops anything else (e.g.
apostrophes), which is fine for learning.

A note on expectations: training real ASR to low error needs lots of data + GPU
time. On CPU with a subset you will see the loss fall and transcriptions become
*recognizable* (partial words), not perfect. That still proves the whole pipeline
works on real audio.
"""

from __future__ import annotations

import os

import soundfile as sf
import torch
from torch.utils.data import Dataset
import torchaudio

from src.audio.features import log_mel_spectrogram
from src.asr.text import encode

SR = 16_000


class LibriSpeechFeatures(Dataset):
    """Wrap torchaudio's LIBRISPEECH to yield (log-mel feats, target ids, text).

    We filter by clip duration so we only keep short-to-medium utterances. This
    keeps CPU time and memory modest AND avoids ever truncating audio (truncating
    would drop spoken words while keeping the full transcript, which breaks the
    audio<->text correspondence that CTC relies on).

    Set download=True the first time to fetch the data into `root`.
    """

    def __init__(
        self,
        root: str = "data",
        url: str = "dev-clean",
        download: bool = False,
        min_seconds: float = 1.0,
        max_seconds: float = 6.0,
    ):
        ds = torchaudio.datasets.LIBRISPEECH(root, url=url, download=download)
        # get_metadata returns a path relative to the dataset's PARENT dir.
        base = os.path.dirname(ds._path)

        # Read only the file headers (cheap) to get durations, keep the in-range
        # ones, and remember (absolute_path, transcript). We load audio ourselves
        # with soundfile so we do not depend on torchaudio's backend/ffmpeg.
        self.items: list[tuple[str, str]] = []
        for i in range(len(ds)):
            rel_path, _sr, text, *_ = ds.get_metadata(i)
            full = os.path.join(base, rel_path)
            if min_seconds <= sf.info(full).duration <= max_seconds:
                self.items.append((full, text))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, transcript = self.items[i]
        audio, sr = sf.read(path, dtype="float32")  # (samples,) for mono FLAC
        waveform = torch.from_numpy(audio)
        if sr != SR:
            waveform = torchaudio.functional.resample(waveform, sr, SR)
        feats = log_mel_spectrogram(waveform, sr=SR, hop_ms=10.0)
        target = encode(transcript)
        return feats, torch.tensor(target, dtype=torch.long), transcript.lower()


def ctc_collate(batch):
    """Collate a list of (feats, target, text) into the CTC batch format.

    Returns:
        feats          (B, T_max, n_mels)
        input_lengths  (B,)
        targets        (sum target lengths,)
        target_lengths (B,)
        texts          list[str]
    """
    t_max = max(f.shape[0] for f, _, _ in batch)
    n_mels = batch[0][0].shape[1]
    b = len(batch)

    feats = torch.zeros(b, t_max, n_mels)
    input_lengths = torch.zeros(b, dtype=torch.long)
    target_lengths = torch.zeros(b, dtype=torch.long)
    all_targets: list[int] = []
    texts: list[str] = []
    for i, (f, tgt, text) in enumerate(batch):
        feats[i, : f.shape[0]] = f
        input_lengths[i] = f.shape[0]
        target_lengths[i] = tgt.shape[0]
        all_targets.append(tgt)
        texts.append(text)

    targets = torch.cat(all_targets) if all_targets else torch.zeros(0, dtype=torch.long)
    return feats, input_lengths, targets, target_lengths, texts
