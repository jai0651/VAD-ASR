"""
Engine factory: config string -> constructed engine.

One place decides what runs in each pipeline slot, so the orchestrator,
server, and scripts never import a concrete engine directly. That's the
swap-ability the whole repo is organized around:

    slot       default (what YOU built / free)      alternative
    ----       -------------------------------      -----------
    denoise    Module 5 OM-LSA spectral suppressor  "gtcrn" / "dtln" — trained ONNX
    vad        Module 1 VADNet                      "silero" — Silero VAD v5 (onnx)
    asr        Module 2 CTC BiGRU + your beam search  "conformer" — Module 6, the
                                                    modern arch (streaming-capable)
                                                    "whisper" — faster-whisper
    tts        Module 4 Tacotron-mini + Griffin-Lim   "kokoro" — Kokoro-82M (onnx)
    responder  echo (offline, no keys)              "claude" — a real LLM brain

VAD engines, denoisers and responders are STATEFUL (context buffers / RNN state
/ noise estimates / conversation history) => construct one per session. ASR/TTS
engines hold only immutable weights => load once per process and share across
sessions.
"""

from __future__ import annotations

from src.pipeline.config import PipelineConfig


def make_vad(cfg: PipelineConfig):
    """Build the head of the pipeline: denoiser + VAD.

    The denoiser is a stage in its own right, but it is attached here because
    the VAD is the single point every mic sample passes through — see
    src/pipeline/denoise.py for why that placement is what makes
    `denoise_target` possible at all.
    """
    if cfg.vad_engine == "scratch":
        from src.pipeline.scratch_engines import ScratchVAD

        vad = ScratchVAD(cfg)
    elif cfg.vad_engine == "silero":
        from src.pipeline.vad import SileroVAD

        vad = SileroVAD(cfg.sample_rate, cfg.vad_window)
    else:
        raise ValueError(f"unknown vad_engine: {cfg.vad_engine!r} (scratch|silero)")

    if cfg.denoise_engine in ("none", "off", ""):
        return vad

    from src.pipeline.denoise import DenoisingVAD, make_denoiser

    if cfg.denoise_target not in ("both", "vad"):
        raise ValueError(f"unknown denoise_target: {cfg.denoise_target!r} (both|vad)")
    return DenoisingVAD(
        vad,
        make_denoiser(cfg),
        pass_clean_downstream=(cfg.denoise_target == "both"),
    )


def make_asr(cfg: PipelineConfig):
    if cfg.asr_engine == "scratch":
        from src.pipeline.scratch_engines import ScratchASR

        return ScratchASR(cfg)
    if cfg.asr_engine == "conformer":
        from src.pipeline.scratch_engines import ConformerASR

        return ConformerASR(cfg)
    if cfg.asr_engine == "whisper":
        from src.pipeline.asr import WhisperASR

        return WhisperASR(cfg)
    raise ValueError(
        f"unknown asr_engine: {cfg.asr_engine!r} (scratch|conformer|whisper)"
    )


def make_responder(cfg: PipelineConfig):
    if cfg.responder == "echo":
        from src.pipeline.responder import EchoResponder

        return EchoResponder()
    if cfg.responder == "claude":
        from src.pipeline.responder import ClaudeResponder

        return ClaudeResponder(
            model=cfg.claude_model,
            max_tokens=cfg.llm_max_tokens,
            api_key=cfg.anthropic_api_key,
        )
    if cfg.responder == "openai":
        from src.pipeline.responder import OpenAIResponder

        return OpenAIResponder(
            model=cfg.openai_model,
            max_tokens=cfg.llm_max_tokens,
            api_key=cfg.openai_api_key,
        )
    raise ValueError(f"unknown responder: {cfg.responder!r} (echo|claude|openai)")


def make_tts(cfg: PipelineConfig):
    if cfg.tts_engine == "scratch":
        from src.pipeline.scratch_engines import ScratchTTS

        return ScratchTTS(cfg)
    if cfg.tts_engine == "kokoro":
        from src.pipeline.tts import KokoroTTS

        return KokoroTTS(cfg)
    raise ValueError(f"unknown tts_engine: {cfg.tts_engine!r} (scratch|kokoro)")
