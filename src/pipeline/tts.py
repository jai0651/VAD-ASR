"""
Production TTS: Kokoro-82M (ONNX).

Modern TTS lineage in one paragraph: two-stage systems (Tacotron2: text ->
mel spectrogram, then a vocoder: mel -> waveform) gave way to non-autoregressive
acoustic models (FastSpeech2) and then end-to-end models (VITS) that go
text -> waveform in one network. Kokoro is a StyleTTS2-derived model: small
(82M params), CPU-real-time, near-SOTA naturalness — which is why it became
the default open choice for local voice agents.

The production trick in this file is SENTENCE-LEVEL STREAMING. Synthesizing a
whole paragraph then playing it means seconds of dead air. Instead we split
the reply into sentences and synthesize/ship each as soon as it's ready, so
time-to-first-audio ≈ time to synthesize sentence one. (Big providers stream
at finer granularity — chunked vocoding inside a sentence — same idea.)

Model files (~330 MB total) are downloaded once into `models/` on first use.
"""

from __future__ import annotations

import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from src.pipeline.config import PipelineConfig

_RELEASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
_MODEL_FILES = {
    "kokoro-v1.0.onnx": f"{_RELEASE}/kokoro-v1.0.onnx",
    "voices-v1.0.bin": f"{_RELEASE}/voices-v1.0.bin",
}

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+")


@dataclass
class TTSChunk:
    samples: np.ndarray     # float32 waveform
    sample_rate: int
    text: str               # the sentence this chunk voices
    latency_ms: float       # synthesis wall-clock for this chunk


def ensure_models(models_dir: str) -> tuple[Path, Path]:
    """Download Kokoro model + voice files once; return their paths."""
    d = Path(models_dir)
    d.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in _MODEL_FILES.items():
        path = d / name
        if not path.exists():
            tmp = path.with_suffix(path.suffix + ".part")
            print(f"[tts] downloading {name} ...")
            urllib.request.urlretrieve(url, tmp)  # noqa: S310 - fixed https URL
            tmp.rename(path)
        paths.append(path)
    return paths[0], paths[1]


def split_sentences(text: str) -> list[str]:
    """Split reply text into speakable pieces (the streaming unit)."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text)]
    return [p for p in parts if p]


class KokoroTTS:
    def __init__(self, cfg: PipelineConfig):
        from kokoro_onnx import Kokoro  # lazy: heavy import

        self.cfg = cfg
        model_path, voices_path = ensure_models(cfg.models_dir)
        # Loaded once per process, shared across sessions (like the ASR).
        self.model = Kokoro(str(model_path), str(voices_path))

    def synthesize(self, text: str) -> Iterator[TTSChunk]:
        """Yield one audio chunk per sentence, as soon as each is synthesized.

        Blocking + CPU-heavy: callers run this off the event loop. Yielding
        per sentence is what lets the orchestrator (a) start playback early
        and (b) abandon the rest on barge-in without wasted compute.
        """
        for sentence in split_sentences(text):
            t0 = time.perf_counter()
            samples, sr = self.model.create(
                sentence,
                voice=self.cfg.tts_voice,
                speed=self.cfg.tts_speed,
                lang=self.cfg.tts_lang,
            )
            yield TTSChunk(
                samples=samples.astype(np.float32),
                sample_rate=sr,
                text=sentence,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
