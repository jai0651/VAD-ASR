"""
Production ASR: faster-whisper (Whisper weights running on CTranslate2).

Relationship to the learning modules: Whisper is still an "audio features in,
text out" acoustic model like our CTC BiGRU — but it's an encoder-decoder
Transformer trained on 680k hours, and it *generates* text autoregressively
instead of emitting per-frame char distributions. The from-scratch model
needed CTC because it had no decoder; Whisper's decoder does the alignment
implicitly with cross-attention.

Why faster-whisper instead of openai/whisper:
  - CTranslate2 is a purpose-built inference engine: fused kernels, int8
    weight quantization, batching. Same weights, ~4x faster, ~4x less memory.
  - This is the general production lesson: TRAINING frameworks (PyTorch) and
    INFERENCE engines (CTranslate2, ONNX Runtime, TensorRT, vLLM) are
    different tools. You train in one, export to the other.

Streaming note: Whisper is utterance-based, not frame-streaming (unlike
RNN-T). Production voice agents mostly accept this: the endpointer closes a
turn, and the whole utterance is transcribed at once. That's the architecture
we implement. True word-by-word partials need an RNN-T/streaming-Conformer
model — covered in the docs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.pipeline.config import PipelineConfig


@dataclass
class Transcript:
    text: str
    language: str
    audio_s: float          # length of the audio transcribed
    latency_ms: float       # wall-clock transcription time
    segments: list[dict] = field(default_factory=list)

    @property
    def rtf(self) -> float:
        """Real-time factor: processing time / audio time (< 1 = faster than live)."""
        return (self.latency_ms / 1000.0) / max(self.audio_s, 1e-6)


class WhisperASR:
    def __init__(self, cfg: PipelineConfig):
        from faster_whisper import WhisperModel  # lazy: heavy import

        self.cfg = cfg
        # Loaded ONCE per process and shared across sessions — model load is
        # seconds and hundreds of MB; per-request loading is the classic
        # serving mistake.
        self.model = WhisperModel(
            cfg.asr_model, device=cfg.asr_device, compute_type=cfg.asr_compute_type
        )

    def transcribe(self, audio: np.ndarray) -> Transcript:
        """audio: float32 mono 16 kHz in [-1, 1] -> Transcript.

        Blocking + CPU-heavy: callers must run this off the event loop
        (the orchestrator uses asyncio.to_thread).
        """
        t0 = time.perf_counter()
        segments_iter, info = self.model.transcribe(
            audio,
            beam_size=self.cfg.asr_beam_size,
            language="en" if self.cfg.asr_model.endswith(".en") else None,
            # Our own endpointer already trimmed silence; Whisper's built-in
            # VAD filter would just add latency and a second opinion.
            vad_filter=False,
            condition_on_previous_text=False,  # avoids hallucination loops
        )
        segments = [
            {"start": s.start, "end": s.end, "text": s.text} for s in segments_iter
        ]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Transcript(
            text="".join(s["text"] for s in segments).strip(),
            language=info.language,
            audio_s=audio.shape[0] / self.cfg.sample_rate,
            latency_ms=latency_ms,
            segments=segments,
        )
