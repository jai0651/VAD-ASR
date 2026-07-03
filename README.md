# A Voice Agent From Scratch (for learning)

Build **Voice Activity Detection**, **Automatic Speech Recognition**,
**Text-to-Speech**, and the **real-time voice pipeline** that connects them —
from the ground up in PyTorch, understanding every layer instead of calling a
pretrained black box.

We build in small, runnable steps. Every module has a script you can run and *see* output
(a plot, a number, a transcription, a wav). You never have a non-working system.

## The mental model

```
                     ┌──────────────── the full loop ───────────────┐
mic audio ─▶ VAD ─▶ endpointing ─▶ ASR ─▶ "brain" ─▶ TTS ─▶ speaker
             │          │            │                 │
         Module 1   Module 3     Module 2         Module 4
        "speech?"  "turn done?" "what words?"   "say it back"

  Module 0 (features) feeds everything; Module 3 is also the production
  pipeline: streaming server, orchestrator, barge-in, metrics.
```

VAD is the gentle on-ramp (tiny model, simple labels). ASR is the deep end
(the alignment problem, CTC loss, decoding). TTS mirrors ASR (attention
*decides* durations instead of collapsing them). The pipeline is where it all
becomes a product.

## Curriculum

| Module | What you build | Key concepts |
|--------|----------------|--------------|
| **0. Audio & Features** | Load audio, build a log-mel spectrogram **from scratch** | sampling rate, framing, FFT, mel scale, why log |
| **1. VAD** | Frame-level speech/silence classifier | binary classification, synthetic labels, streaming inference, smoothing |
| **2. ASR** | Char-level speech recognizer | CTC loss, alignment problem, greedy + beam decoding |
| **3. Pipeline** | Real-time voice agent server | endpointing, orchestration, barge-in, WebSocket streaming, latency metrics, engine swapping |
| **4. TTS** | Char-level speech synthesizer | seq2seq attention, guided attention, stop tokens, Griffin-Lim phase recovery |

## Setup

```bash
uv sync          # creates .venv and installs torch, torchaudio, etc.
```

## Run the modules in order

```bash
uv run python scripts/00_features.py      # waveform -> log-mel spectrogram (plot)
uv run python scripts/01_train_vad.py     # train the VAD (-> outputs/vad.pt)
uv run python scripts/02_vad_stream.py    # run VAD in streaming mode + smoothing
uv run python scripts/03_ctc_intuition.py # the CTC collapse rule + a tiny CTC fit
uv run python scripts/04_train_asr.py     # train the CTC ASR (-> outputs/asr.pt)
uv run python scripts/05_train_asr_libri.py # same ASR on real speech (-> asr_libri.pt)
uv run python scripts/07_vocoder_roundtrip.py # hear Griffin-Lim invert your mels
uv run python scripts/08_train_tts.py     # train the TTS (-> outputs/tts.pt)
uv run python scripts/06_pipeline_e2e.py  # full VAD->ASR->TTS pipeline, no mic
```

Each writes a plot or wav to `outputs/`. Open them to *see/hear* what each stage does.

## The live voice agent (Module 3)

```bash
uv run python -m server.app               # then open http://127.0.0.1:8000
```

Talk into your mic: your VAD gates your endpointer, your ASR transcribes the
turn, the reply is spoken by your TTS — and you can interrupt it mid-sentence
(barge-in). `GET /metrics` shows stage latencies (p50/p95).

Every stage is a swappable engine. Compare your models against what production
systems actually run, on the same audio:

```bash
# all three production engines (first run downloads weights):
VOICE_VAD_ENGINE=silero VOICE_ASR_ENGINE=whisper VOICE_TTS_ENGINE=kokoro \
  uv run python -m server.app
# or mix and match, e.g. your VAD + production ASR:
VOICE_ASR_ENGINE=whisper uv run python -m server.app
```

| slot | `scratch` (default: yours) | production alternative |
|------|---------------------------|------------------------|
| vad  | Module 1 VADNet           | `silero` — Silero VAD v5 (2 MB onnx) |
| asr  | Module 2 CTC BiGRU + your beam search | `whisper` — faster-whisper (CTranslate2, int8) |
| tts  | Module 4 Tacotron-mini + Griffin-Lim | `kokoro` — Kokoro-82M (onnx) |

### Add the LLM brain

By default the agent echoes what you said (offline, no keys). Swap in an LLM
and it becomes a real conversational voice agent — the reply slot is the
`Responder` interface in `src/pipeline/responder.py`, one method per vendor.

Put your key in a `.env` file at the repo root (git-ignored; see
`.env.example`), then:

```bash
# OpenAI (reads OPENAI_API_KEY + OPENAI_MODEL from .env):
VOICE_RESPONDER=openai uv run python -m server.app
# Claude (reads ANTHROPIC_API_KEY from .env):
VOICE_RESPONDER=claude uv run python -m server.app
# best-sounding demo: production ASR/TTS + LLM brain:
VOICE_RESPONDER=openai VOICE_ASR_ENGINE=whisper VOICE_TTS_ENGINE=kokoro \
  uv run python -m server.app
```

The responder keeps per-session conversation history (ask "what did I just
say?"), and barge-in still works — interrupt the LLM mid-reply and it stops.

Run the tests: `uv run pytest -q` (endpointing, orchestrator/barge-in, audio I/O).

## Code map

| File | Role |
|------|------|
| `src/audio/features.py` | log-mel spectrogram, every step by hand |
| `src/vad/data.py` | synthetic labeled speech/noise generator |
| `src/vad/model.py` | tiny 1-D conv frame classifier |
| `src/vad/stream.py` | streaming inference + hysteresis smoothing |
| `src/asr/text.py` | char vocabulary + the CTC collapse rule |
| `src/asr/data.py` | synthetic "spoken text" generator |
| `src/asr/model.py` | conv + BiGRU CTC encoder |
| `src/asr/decode.py` | greedy + CTC prefix beam search + CER |
| `src/tts/model.py` | Tacotron-mini: attention seq2seq text→mel |
| `src/tts/vocoder.py` | Griffin-Lim + hand-built STFT/ISTFT (mel→audio) |
| `src/pipeline/endpointing.py` | turn detection state machine |
| `src/pipeline/orchestrator.py` | the voice-agent session (barge-in lives here) |
| `src/pipeline/scratch_engines.py` | YOUR models behind the pipeline interfaces |
| `src/pipeline/engines.py` | engine factory (scratch ↔ production swap) |
| `server/app.py` + `client/` | WebSocket transport + browser mic client |

## Learn the deep internals (docs/)

Open `docs/index.html` — a set of deep-dive pages written alongside this code:
pipeline architecture, VAD, ASR, TTS, production inference, and a decision log
recording *why* each design choice was made (and what production does instead).

## Real speech (LibriSpeech)

The synthetic data proves the algorithms work offline; `src/asr/librispeech.py`
runs the SAME model + CTC loss + decoders on real human speech. Only the data
source changed.

```bash
# one-time download (~337 MB), then train on a subset:
uv run python scripts/05_train_asr_libri.py
# scale up:
SUBSET=400 STEPS=4000 BATCH=8 uv run python scripts/05_train_asr_libri.py
```

We load FLAC with `soundfile` directly (torchaudio's loader needs ffmpeg/torchcodec).
Utterances are filtered to 1-6 s and features are pre-computed once for speed.

What you should see: on **seen** utterances the model reaches ~0 character error
rate (it transcribes real speech!); on **held-out** utterances error stays high
with a small subset. That train/eval gap *is* the data-hunger of ASR — closing it
needs much more data (e.g. `train-clean-100`) + a GPU + longer training.

### To push toward a real recognizer
- Train on `train-clean-100` (or 360) instead of `dev-clean`.
- Use a GPU (CUDA, or this code already targets Apple MPS when available).
- Add SpecAugment (time/freq masking) for regularization.
- Add a language model to the beam search decoder.

## Hardware

Apple Silicon: PyTorch uses the **MPS** (Metal) backend automatically where we enable it.
Everything also runs on CPU (slower, fine for the tiny datasets we use for learning).
