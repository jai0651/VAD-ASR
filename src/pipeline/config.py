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
    denoise_engine: str = "spectral"  # none | spectral (yours) | gtcrn | dtln
    vad_engine: str = "scratch"      # scratch (your VADNet) | silero
    asr_engine: str = "scratch"      # scratch (CTC BiGRU) | conformer | whisper
    tts_engine: str = "scratch"      # scratch (your Tacotron-mini + Griffin-Lim) | kokoro
    responder: str = "echo"          # echo | claude | openai

    # ---- noise suppression (Module 5) ----
    # target: "both" = VAD and ASR both consume the cleaned stream;
    #         "vad"  = only the VAD is scored on clean audio, the ASR receives
    #                  the sample-aligned original.
    # Default is "vad" because that's what scripts/10_denoise_bench.py measured:
    #   VAD  — every engine helps at every SNR (room noise P 0.054 -> 0.040,
    #          babble 0.51 -> 0.43, hum 0.010 -> 0.005; speech stays ~0.89).
    #   ASR  — inconsistent, and negative exactly where it matters. Whisper
    #          base.en CER on 24 utterances at 0 dB SNR: 0.068 raw, 0.079 with
    #          the spectral front-end, 0.070 with GTCRN.
    # Take the certain win, skip the uncertain loss. Set "both" if you run the
    # scratch ASR (trained on clean LibriSpeech, so it prefers clean input) or
    # if you need cleaned audio downstream for recording/LLM context.
    denoise_target: str = "vad"      # both | vad
    # Gain floor: how many dB a noise-only band may be attenuated. Unlimited
    # suppression sounds cleaner to a human but strips the noise floor that
    # ASR models expect, and makes silences pump.
    denoise_atten_db: float = 18.0

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

    # ---- Module 6: the modern (Conformer) recognizer ----
    conformer_ckpt: str = "outputs/asr_conformer.pt"
    # "" = read the tokenizer path recorded inside the checkpoint.
    conformer_tokenizer: str = ""
    # Decoder context in 40 ms encoder frames. 0 = offline (best WER);
    # 16 = 640 ms lookahead. This is THE latency/accuracy dial of a streaming
    # recognizer, and it needs no retraining because the model was trained with
    # dynamic chunk masking.
    conformer_chunk: int = 0

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
