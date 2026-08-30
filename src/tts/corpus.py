"""
Module 8, part 2: your voice on disk.

The corpus format is LJSpeech's (`wavs/*.wav` + a pipe-delimited `metadata.csv`)
for one unglamorous reason: it is what every open TTS trainer already reads. If
you later want to fine-tune XTTS, VITS, StyleTTS2 or Piper on your voice instead
of the model in this repo, the data is already in the shape they expect and you
do not re-record anything.

THE SAMPLE RATE DECISION IS THE ONE THAT CANNOT BE UNDONE.

Everything else in this repo runs at 16 kHz, so 16 kHz is the tempting choice.
Resist it. 16 kHz throws away everything above 8 kHz, which is precisely the
band that carries sibilance and "air" — the difference between a voice and a
voice on a telephone. You can always downsample later; you can never invent the
missing octave. So we CAPTURE at whatever the device gives us (usually 48 kHz)
and STORE at 24 kHz, the standard rate for neural vocoders, then downsample to
16 kHz on the fly for the Module 1/6/7 models that want it.

Cost of that choice: 1.5x the disk of 16 kHz, which for an hour of speech is
~170 MB. Benefit: the corpus outlives the current models.

WHY 16-BIT PCM AND NOT FLOAT32. Microphone input is ~16 effective bits of real
signal buried in an ADC's noise floor; float32 stores the noise at higher
precision and doubles the file size. int16 is lossless with respect to what the
hardware actually resolved.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

VOICE_DIR = "data/voice"
STORE_SR = 24_000     # what we keep on disk
MODEL_SR = 16_000     # what Modules 0-7 consume


@dataclass(frozen=True)
class Utterance:
    utt_id: str
    text: str          # normalized text, exactly what the model is trained on
    raw_text: str      # what was displayed on screen, with punctuation and case
    duration_s: float
    path: Path


class VoiceCorpus:
    """Append-only reader/writer for `data/voice/`.

    APPEND-ONLY IS A FEATURE. A recording session gets interrupted — a phone
    rings, the laptop sleeps, you get bored at prompt 300. Every accepted take
    is flushed to disk immediately and the manifest is only ever appended to, so
    the worst an interruption can cost you is the single take in progress. The
    harness reloads the manifest on start and resumes at the first unrecorded
    prompt.
    """

    def __init__(self, root: str = VOICE_DIR):
        self.root = Path(root)
        self.wav_dir = self.root / "wavs"
        self.manifest = self.root / "metadata.csv"
        self.wav_dir.mkdir(parents=True, exist_ok=True)

    # ---- reading --------------------------------------------------------
    def load(self) -> list[Utterance]:
        if not self.manifest.exists():
            return []
        out: list[Utterance] = []
        with self.manifest.open(newline="") as f:
            for row in csv.reader(f, delimiter="|"):
                if len(row) != 4:
                    continue
                utt_id, text, raw_text, dur = row
                out.append(Utterance(utt_id, text, raw_text,
                                     float(dur), self.wav_dir / f"{utt_id}.wav"))
        return out

    def recorded_texts(self) -> set[str]:
        """Normalized texts already captured — the resume key.

        Keyed on TEXT rather than on prompt index so that editing the prompt
        file (adding sentences, fixing a typo) does not orphan existing takes.
        """
        return {u.text for u in self.load()}

    def total_seconds(self) -> float:
        return sum(u.duration_s for u in self.load())

    # ---- writing --------------------------------------------------------
    def append(self, audio: np.ndarray, sr: int, text: str, raw_text: str) -> Utterance:
        """Resample to STORE_SR, write the wav, then append the manifest row.

        Ordering is deliberate: the wav is fully on disk (and fsync'd by the
        close) before the manifest mentions it. A crash between the two leaves
        an orphan wav — harmless, invisible to the loader — whereas the reverse
        order would leave a manifest row pointing at a file that does not exist,
        which breaks training much later and far away from the cause.
        """
        audio = resample(audio, sr, STORE_SR)
        utt_id = f"utt_{self._next_index():04d}"
        path = self.wav_dir / f"{utt_id}.wav"
        sf.write(path, np.clip(audio, -1.0, 1.0), STORE_SR, subtype="PCM_16")

        dur = len(audio) / STORE_SR
        with self.manifest.open("a", newline="") as f:
            csv.writer(f, delimiter="|").writerow(
                [utt_id, text, raw_text, f"{dur:.3f}"])
        return Utterance(utt_id, text, raw_text, dur, path)

    def _next_index(self) -> int:
        existing = sorted(self.wav_dir.glob("utt_*.wav"))
        return 1 + (int(existing[-1].stem.split("_")[1]) if existing else 0)


def resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """Band-limited resampling. Never decimate by slicing — that aliases."""
    if sr == target_sr:
        return audio.astype(np.float32)
    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    return torchaudio.functional.resample(x, sr, target_sr).squeeze(0).numpy()


def load_utterance(utt: Utterance, sr: int = MODEL_SR) -> torch.Tensor:
    """One utterance as a mono float32 tensor at `sr` — the training entry point."""
    audio, file_sr = sf.read(utt.path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return torch.from_numpy(resample(audio, file_sr, sr))
