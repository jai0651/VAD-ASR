"""
Module 5, part 3: putting the denoiser into the live pipeline.

WHERE THE STAGE GOES. Noise suppression belongs at the very front — before
VAD, before endpointing, before ASR — because it is the only stage that can
still see the raw microphone signal:

    mic ─▶ [DENOISE] ─▶ VAD ─▶ endpointer ─▶ ASR ─▶ LLM ─▶ TTS

Mechanically we attach it as a *decorator around the VAD engine* rather than as
a fourth slot in the orchestrator. Not an aesthetic choice: the VAD is the
single point every microphone sample flows through, and the endpointer buffers
exactly the sample arrays the VAD hands back, so wrapping the VAD is the only
place where one object can control BOTH what gets scored and what gets
transcribed — which is precisely the knob we want:

    denoise_target = "both"  clean audio scored by the VAD *and* sent to the ASR
    denoise_target = "vad"   clean audio scored by the VAD; the ASR gets the
                             sample-aligned ORIGINAL audio

THE SECOND ONE IS NOT A GIMMICK. It is the single most surprising result in
speech engineering: an enhancement front-end reliably helps a VAD, and often
*hurts* a strong ASR. Whisper was trained on 680k hours of real-world audio
including noise, so it has learned to hear through it; a denoiser hands it
spectral artifacts it has never seen — an out-of-distribution input, exactly
the failure mode Module 1b hit from the other direction. A small model trained
on clean LibriSpeech (ours) has the opposite prior and loves the clean input.
`scripts/10_denoise_bench.py` measures this on your own machine rather than
asking you to take anyone's word for it.

ALIGNMENT. Every engine here is overlap-add, so its output lags its input by
one hop (16 ms): the second half of the analysis window hasn't arrived yet.
To hand the ASR the *original* audio for the same instants, we keep a raw FIFO
pre-filled with `denoiser.delay` zeros and pop from it in lockstep with the
frames the VAD returns. Get this wrong and the ASR transcribes audio 16 ms out
of step with the endpoints, which clips word onsets — the exact bug pre-roll
exists to prevent.
"""

from __future__ import annotations

import numpy as np

from src.pipeline.config import PipelineConfig
from src.pipeline.vad import VADResult


class PassthroughDenoiser:
    """`denoise_engine=none`. Also the baseline for measuring the stage's cost."""

    name = "none"
    sr = 16_000
    hop = 160
    delay = 0
    reduction_db = 0.0

    def process(self, block: np.ndarray) -> np.ndarray:
        return np.asarray(block, dtype=np.float32)

    def reset(self) -> None:
        pass


def make_denoiser(cfg: PipelineConfig):
    """config string -> denoise engine (the from-scratch one is the default)."""
    name = cfg.denoise_engine
    if name in ("none", "off", ""):
        return PassthroughDenoiser()
    if name == "spectral":
        from src.denoise.spectral import SpectralDenoiser

        return SpectralDenoiser(max_atten_db=cfg.denoise_atten_db)
    if name == "gtcrn":
        from src.denoise.onnx_engines import GTCRNDenoiser

        return GTCRNDenoiser(max_atten_db=cfg.denoise_atten_db)
    if name == "dtln":
        from src.denoise.onnx_engines import DTLNDenoiser

        return DTLNDenoiser(max_atten_db=cfg.denoise_atten_db)
    raise ValueError(
        f"unknown denoise_engine: {name!r} (none|spectral|gtcrn|dtln)"
    )


class DenoisingVAD:
    """Wraps any VAD engine with a front-of-pipeline denoiser.

    Presents the exact VAD engine interface (`window`, `push`, `reset`) so the
    orchestrator and endpointer are unchanged and unaware.
    """

    def __init__(self, vad, denoiser, pass_clean_downstream: bool = True):
        self.vad = vad
        self.denoiser = denoiser
        self._pass_clean = pass_clean_downstream
        self._raw = np.zeros(0, dtype=np.float32)
        self.reduction_db = 0.0
        self.reset()

    @property
    def window(self) -> int:
        return self.vad.window

    def push(self, samples: np.ndarray) -> list[VADResult]:
        clean = self.denoiser.process(samples)
        if not self._pass_clean:
            self._raw = np.concatenate([self._raw, samples])

        results = self.vad.push(clean)
        self.reduction_db = self.denoiser.reduction_db

        if self._pass_clean:
            return results

        # Swap each scored window's audio for the aligned original. The inner
        # VAD consumes `clean` strictly in order, so a FIFO popped by each
        # result's length stays in exact sample-for-sample step.
        aligned: list[VADResult] = []
        for res in results:
            n = res.samples.shape[0]
            if self._raw.shape[0] < n:  # cannot happen after warm-up; be safe
                aligned.append(res)
                continue
            raw, self._raw = self._raw[:n], self._raw[n:]
            aligned.append(VADResult(prob=res.prob, samples=raw))
        return aligned

    def reset(self) -> None:
        self.denoiser.reset()
        self.vad.reset()
        # Pre-fill with the engine's algorithmic delay so raw and clean line up.
        self._raw = np.zeros(getattr(self.denoiser, "delay", 0), dtype=np.float32)
        self.reduction_db = 0.0
