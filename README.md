# A Voice Agent From Scratch (for learning)

Build **Voice Activity Detection**, **Automatic Speech Recognition**,
**Text-to-Speech**, and the **real-time voice pipeline** that connects them —
from the ground up in PyTorch, understanding every layer instead of calling a
pretrained black box.

We build in small, runnable steps. Every module has a script you can run and *see* output
(a plot, a number, a transcription, a wav). You never have a non-working system.

## The mental model

```
                       ┌─────────────── the full loop ───────────────┐
mic ─▶ denoise ─▶ VAD ─▶ endpointing ─▶ ASR ─▶ "brain" ─▶ TTS ─▶ speaker
         │         │          │            │                 │
     Module 5  Module 1   Module 3     Module 2         Module 4
     "cleaner" "speech?"  "turn done?" "what words?"   "say it back"

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
| **5. Noise suppression** | Streaming denoiser **from scratch** | STFT gain estimation, MCRA noise tracking, decision-directed SNR, log-MMSE / OM-LSA, why neural models won |
| **6. Modern ASR** | Conformer + hybrid CTC/attention **from scratch** | BPE subwords, relative-position attention, macaron blocks, dynamic chunk streaming, SpecAugment, warmup/cosine, checkpoint averaging |
| **7. Neural vocoder** | ISTFT vocoder (Vocos-style) **from scratch** | why Griffin-Lim is a ceiling, ConvNeXt blocks, magnitude+phase heads, exact overlap-add via `fold`, multi-resolution STFT loss |

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
uv run python scripts/09_train_vad_real.py  # VAD retrained on REAL audio (-> vad_real.pt)
uv run python scripts/07_vocoder_roundtrip.py # hear Griffin-Lim invert your mels
uv run python scripts/08_train_tts.py     # train the TTS (-> outputs/tts.pt)
uv run python scripts/06_pipeline_e2e.py  # full VAD->ASR->TTS pipeline, no mic
uv run python scripts/10_denoise_bench.py # does noise suppression actually help?
uv run python scripts/11_train_asr_conformer.py # the MODERN ASR (Module 6)
uv run python scripts/12_train_vocoder.py # neural vocoder vs Griffin-Lim (Module 7)
uv run python scripts/13_asr_compare.py   # Module 2 vs Module 6 vs Whisper, same audio
```

### Does the modern architecture actually help? (`scripts/13_asr_compare.py`)

Same 40 held-out utterances, same 4.1 h of training audio for both scratch models:

| engine | WER | CER | RTF | training data |
|---|---|---|---|---|
| scratch (Module 2) — BiGRU + CTC + chars | 1.076 | 0.658 | 0.011 | 4.1 h |
| **conformer (Module 6)** — offline | **0.607** | **0.276** | **0.004** | 4.1 h |
| conformer (Module 6) — streaming, 640 ms | 0.627 | 0.284 | 0.004 | 4.1 h |
| whisper base.en int8 | 0.033 | 0.017 | 0.052 | ~680,000 h |

The architecture change halved the error on identical data, made it 2.7× faster,
and added streaming for a 3% relative cost. The remaining 18× gap to Whisper is
165,000× more data — which is exactly the point of Module 6.

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
| denoise | Module 5 OM-LSA spectral suppressor | `gtcrn` (23.7K params) / `dtln` — trained onnx |
| vad  | Module 1 VADNet           | `silero` — Silero VAD v5 (2 MB onnx) |
| asr  | Module 2 CTC BiGRU + your beam search | `conformer` — Module 6 (streaming-capable) · `whisper` — faster-whisper (CTranslate2, int8) |
| tts  | Module 4 Tacotron-mini + Griffin-Lim | `kokoro` — Kokoro-82M (onnx) |

### Noise suppression (Module 5)

On by default (`VOICE_DENOISE_ENGINE=spectral`), feeding cleaned audio to the
VAD only. Measured on this repo with `scripts/10_denoise_bench.py`:

| engine | ΔSNR | noise removed in pauses | speech damaged | RTF | latency |
|--------|------|--------------------------|----------------|-----|---------|
| `spectral` (yours) | +5.3 dB | 8.1 dB | 0.9 dB | 0.004 | 16 ms |
| `gtcrn` | +4.9 dB | 7.3 dB | 2.0 dB | 0.031 | 16 ms |
| `dtln` | +6.9 dB | 7.9 dB | 1.1 dB | 0.015 | 24 ms |

```bash
bash scripts/fetch_denoise_models.sh                 # only for gtcrn/dtln
uv run python scripts/10_denoise_bench.py            # measure it yourself
WHISPER=1 SNR_DB=0 uv run python scripts/10_denoise_bench.py
VOICE_DENOISE_ENGINE=gtcrn uv run python -m server.app
```

`denoiser/` is the standalone realtime denoiser app (mic → clean → virtual
audio device, for Zoom/Meet/a softphone). It has its own venv and README; the
`src/denoise/` engines above are the same idea integrated into this pipeline.

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
| `src/asr/model.py` | conv + BiGRU CTC encoder (Module 2) |
| `src/asr/decode.py` | greedy + CTC prefix beam search + CER |
| `src/asr/tokenizer.py` | BPE subword tokenizer, trained from the transcripts |
| `src/asr/conformer.py` | Conformer encoder: rel-pos attention, macaron FFN, chunk masks |
| `src/asr/hybrid.py` | hybrid CTC/attention model + attention rescoring |
| `src/asr/corpus.py` | SpecAugment, speed perturb, CMVN, dynamic batching |
| `src/asr/search.py` | log-space CTC prefix beam over subwords + WER |
| `src/tts/model.py` | Tacotron-mini: attention seq2seq text→mel |
| `src/tts/vocoder.py` | Griffin-Lim + hand-built STFT/ISTFT (mel→audio) |
| `src/denoise/spectral.py` | MCRA + decision-directed + log-MMSE/OM-LSA denoiser, by hand |
| `src/denoise/onnx_engines.py` | GTCRN / DTLN streaming denoisers (production alternates) |
| `src/pipeline/denoise.py` | the denoise stage + which stages consume clean audio |
| `src/pipeline/endpointing.py` | turn detection state machine |
| `src/pipeline/orchestrator.py` | the voice-agent session (barge-in lives here) |
| `src/pipeline/scratch_engines.py` | YOUR models behind the pipeline interfaces |
| `src/pipeline/engines.py` | engine factory (scratch ↔ production swap) |
| `server/app.py` + `client/` | WebSocket transport + browser mic client |

## Learn the deep internals (docs/)

**Open `docs/index.html`.** The docs are structured as a book with a reading
order, not a pile of pages:

| | |
|---|---|
| **Part 0 · Synthesis** | [`00-crux.html`](docs/00-crux.html) — **start here.** Mind maps, one idea + one number per module, the full failure catalogue grouped by lesson, ten transferable principles, and interview questions with answers grounded in what we measured |
| **Part 1 · The real-time loop** | pipeline architecture, production inference |
| **Part 2 · The models** | VAD → ASR → TTS → noise suppression, in build order |
| **Part 3 · To the state of the art** | modern Conformer, the full training maths, the neural vocoder |
| **Part 4 · The field** | where SOTA is, and where a small team can still win |
| **Appendix** | 28 numbered decisions, including the ones tried and rejected |

Reading paths are on the contents page: 30 minutes to be able to explain the
project, or an afternoon for the whole build.

Every chapter was written alongside the code it describes, and records the
failures as well as the results — the failure catalogue in Part 0 is the most
useful thing in the repo.

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
