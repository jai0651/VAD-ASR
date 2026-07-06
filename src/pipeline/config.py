"""
All pipeline tunables in one typed object.

Production systems never hard-code thresholds inside model code: endpoint
silence, VAD thresholds, model sizes etc. get tuned per deployment (telephony
vs. web mic vs. far-field) and per language. pydantic-settings gives us typed
defaults that any env var can override, e.g.:

    VOICE_ASR_MODEL=small.en VOICE_END_SILENCE_MS=500 uv run python -m server.app
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseSettings):
    # env_file: secrets and overrides can live in a git-ignored .env at the
    # repo root instead of the shell environment. extra="ignore" lets the
    # .env hold variables we don't model.
    model_config = SettingsConfigDict(
        env_prefix="VOICE_", env_file=".env", extra="ignore"
    )

    # ---- engine selection: "scratch" = the models YOU built (Modules 0-4);
    # the alternatives are what production deployments actually run. Swap per
    # stage and A/B them on the same mic audio:
    #   VOICE_ASR_ENGINE=whisper VOICE_TTS_ENGINE=kokoro uv run python -m server.app
    vad_engine: str = "scratch"      # scratch (your VADNet) | silero
    asr_engine: str = "scratch"      # scratch (your CTC BiGRU) | whisper
    tts_engine: str = "scratch"      # scratch (your Tacotron-mini + Griffin-Lim) | kokoro
    responder: str = "echo"          # echo | claude | openai

    # ---- LLM responder (used when responder != "echo") ----
    # API keys come from the environment or .env — never from code/git.
    llm_max_tokens: int = 300        # spoken replies should be short
    claude_model: str = "claude-opus-4-8"
    anthropic_api_key: str = Field("", validation_alias="ANTHROPIC_API_KEY")
    openai_model: str = Field("gpt-4o-mini", validation_alias="OPENAI_MODEL")
    openai_api_key: str = Field("", validation_alias="OPENAI_API_KEY")

    # ---- from-scratch model checkpoints (produced by the training scripts) ----
    # "auto" = prefer the real-audio VAD (outputs/vad_real.pt, scripts/09) and
    # fall back to the synthetic one (outputs/vad.pt, scripts/01).
    scratch_vad_ckpt: str = "auto"
    scratch_asr_ckpt: str = "outputs/asr_libri.pt"
    scratch_tts_ckpt: str = "outputs/tts.pt"

    # ---- audio format (the contract with the client) ----
    sample_rate: int = 16_000        # Hz; everything upstream of TTS is 16 kHz mono
    # Silero VAD consumes exactly 512-sample windows at 16 kHz = 32 ms each.
    vad_window: int = 512

    # ---- VAD / endpointing ----
    vad_start_threshold: float = 0.5   # prob above which a window counts as speech
    vad_end_threshold: float = 0.35    # hysteresis: lower bar to *stay* in speech
    start_trigger_ms: int = 96         # sustained speech needed to open an utterance
    end_silence_ms: int = 700          # sustained silence that closes an utterance
    pre_roll_ms: int = 320             # audio kept from *before* the trigger point
    min_utterance_ms: int = 250        # discard blips shorter than this
    max_utterance_s: float = 30.0      # hard cap: force-close runaway utterances

    # ---- ASR (faster-whisper / CTranslate2) ----
    asr_model: str = "base.en"         # tiny.en | base.en | small.en | ...
    asr_device: str = "cpu"            # CTranslate2 on Apple Silicon runs CPU int8
    asr_compute_type: str = "int8"     # weight quantization: 4x smaller, ~2-3x faster
    asr_beam_size: int = 1             # greedy; bump to 5 for accuracy over latency

    # ---- TTS (Kokoro-82M ONNX) ----
    tts_voice: str = "af_heart"
    tts_speed: float = 1.0
    tts_lang: str = "en-us"
    models_dir: str = "models"         # where Kokoro onnx/voices files are cached

    # ---- server ----
    host: str = "127.0.0.1"
    port: int = 8000

    @property
    def vad_window_ms(self) -> float:
        return 1000.0 * self.vad_window / self.sample_rate
