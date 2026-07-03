"""
Module 4, part 3: (text, log-mel) pairs from ONE LibriSpeech speaker.

Two dataset decisions that matter for TTS specifically:

  SINGLE SPEAKER. ASR wants many voices (it must be speaker-invariant); a
  small TTS wants ONE voice (it must be speaker-consistent). Mixing speakers
  without a speaker embedding gives the model contradictory targets for the
  same text and it learns an averaged mumble. We use LibriSpeech speaker 84,
  already on disk from Module 2b.

  NORMALIZED, FLOORED MELS. Raw log-mels span ~[-23, +2]: the -23 is just
  log(silence). L1 loss on that range spends most of its gradient learning
  "silence is very negative", so we clamp the floor to log(1e-5) and then
  z-normalize per mel bin with dataset statistics (saved in the checkpoint —
  synthesis must undo exactly this).
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
import torch

from src.asr.text import encode
from src.audio.features import log_mel_spectrogram

SR = 16_000
LOG_MEL_FLOOR = -11.5  # log(1e-5): quiet enough to be "silence" for training
SPEAKER_DIR = "data/LibriSpeech/dev-clean/84"


def load_speaker_utterances(
    root: str = SPEAKER_DIR, min_seconds: float = 0.8, max_seconds: float = 6.0
) -> list[tuple[str, torch.Tensor]]:
    """[(text, log_mel (T, 80))] for every in-range utterance of the speaker."""
    items: list[tuple[str, torch.Tensor]] = []
    for trans in sorted(Path(root).glob("*/*.trans.txt")):
        transcripts = dict(
            line.split(" ", 1) for line in trans.read_text().splitlines()
        )
        for utt_id, text in transcripts.items():
            flac = trans.parent / f"{utt_id}.flac"
            if not (min_seconds <= sf.info(flac).duration <= max_seconds):
                continue
            audio, sr = sf.read(flac, dtype="float32")
            assert sr == SR
            mel = log_mel_spectrogram(torch.from_numpy(audio), sr=SR)
            items.append((text.strip().lower(), mel.clamp(min=LOG_MEL_FLOOR)))
    return items


def mel_statistics(items) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-bin mean/std over the whole dataset (stored in the checkpoint)."""
    all_mel = torch.cat([mel for _, mel in items], dim=0)
    return all_mel.mean(dim=0), all_mel.std(dim=0).clamp(min=1e-3)


def tts_collate(batch, mean, std, r: int = 2):
    """[(text, mel)] -> padded tensors for teacher-forced training.

    Returns:
        text      (B, T_in) label ids, 0-padded
        text_mask (B, T_in) True where real
        mel       (B, T_mel, 80) normalized, zero-padded, T_mel multiple of r
        mel_mask  (B, T_mel) True where real
        stop      (B, T_mel) 1.0 on/after each utterance's final real frame
    """
    encoded = [torch.tensor(encode(t), dtype=torch.long) for t, _ in batch]
    t_in = max(e.shape[0] for e in encoded)
    mel_lens = [mel.shape[0] for _, mel in batch]
    t_mel = max(mel_lens)
    t_mel = ((t_mel + r - 1) // r) * r  # round up to a multiple of r

    b = len(batch)
    text = torch.zeros(b, t_in, dtype=torch.long)
    text_mask = torch.zeros(b, t_in, dtype=torch.bool)
    mel = torch.zeros(b, t_mel, 80)
    mel_mask = torch.zeros(b, t_mel, dtype=torch.bool)
    stop = torch.zeros(b, t_mel)
    for i, (e, (_, m)) in enumerate(zip(encoded, batch)):
        text[i, : e.shape[0]] = e
        text_mask[i, : e.shape[0]] = True
        mel[i, : m.shape[0]] = (m - mean) / std
        mel_mask[i, : m.shape[0]] = True
        stop[i, m.shape[0] - 1 :] = 1.0
    return text, text_mask, mel, mel_mask, stop
