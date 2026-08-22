# Realtime Voice-Call Denoiser

> Vendored into this voice-agent repo so all the voice-AI work lives in one
> place. This is the **standalone app**: real mic → denoise → virtual audio
> device, so Zoom/Meet/a softphone hears only your voice. It keeps its own
> `.venv` and `requirements.txt`.
>
> The same engines are *also* integrated as a pipeline stage under
> `../src/denoise/`, alongside a from-scratch OM-LSA denoiser — see
> `../docs/07-denoise.html`.

Cleans background noise out of your microphone in real time using
[DeepFilterNet3](https://github.com/Rikorose/DeepFilterNet), so calling apps
(Zoom, Meet, Slack, a Plivo softphone, …) hear only your voice.

It works by sitting between your real mic and a **virtual audio device**: it
captures your mic, denoises it, and writes the clean audio to the virtual
device. You then select that virtual device as your "microphone" inside the
calling app.

```
real mic ──▶ run.py live ──▶ BlackHole (virtual mic) ──▶ Zoom / Meet / Plivo
             (denoise)
```

## Engines

The project ships interchangeable denoising engines with a clear trade-off.
Pick with `--backend`.

| Engine | `--backend` | Noise reduction | Latency | CPU (1 core) | How it streams |
| --- | --- | --- | --- | --- | --- |
| **DeepFilterNet3** | `deepfilternet` (default) | **~11 dB** | ~50 ms | ~23% | warm-context recompute |
| **GTCRN** ⭐ | `gtcrn` | ~9 dB* | ~32 ms | ~3% | true per-frame, complex-domain |
| **DTLN** | `dtln` | ~9 dB* | ~32 ms | **~1.5%** | true per-frame, magnitude-domain |

CPU/latency measured on an M-series MacBook Air. Noise-reduction figures are
segmental-SNR on synthetic speech; for a rigorous comparison on **real** noisy
speech (VoiceBank-DEMAND, with PESQ/STOI/SI-SDR), see
[docs/benchmark.md](docs/benchmark.md) — summary below.

**Verified on real 16 kHz noisy speech (60 VoiceBank-DEMAND files):**

| engine | PESQ-WB | SI-SDR | CPU |
| --- | --- | --- | --- |
| noisy (baseline) | 2.19 | 8.6 dB | — |
| **gtcrn** | **2.66** (best) | 14.1 dB | ~3% |
| dtln | 2.55 | **17.3 dB** (best) | ~1.5% |
| deepfilternet | 2.41* | 16.2 dB | ~23% |

**Which to use?** For realtime voice calls (typically 16 kHz), **`gtcrn` is the
recommended engine** — best *perceptual* quality (PESQ) and the Krisp-class
streaming profile (~3% CPU, phase-aware, frame-by-frame). `dtln` squeezes out the
best signal fidelity (SI-SDR) at the lowest CPU. `deepfilternet` is a 48 kHz
full-band model and is handicapped on this 16 kHz test (*it's evaluated on
upsampled audio, outside its training band — see benchmark notes); on native
48 kHz audio it is expected to score higher.

### Why the streaming difference matters

DeepFilterNet3's convolutions and deep-filter carry temporal context that gets
zero-padded at chunk boundaries, so cheap per-frame streaming isn't possible
without a full per-layer state rewrite (which is why production real-time
DeepFilterNet is a Rust library). We work around it by **re-processing 200 ms of
recent audio every hop** — correct and high-quality, but it burns CPU.

GTCRN and DTLN are instead *architected* for streaming: they pass their internal
state (conv caches + RNN state) frame to frame, so each hop runs exactly one
forward — no recompute. That's the same design principle real-time products like
Krisp use, and it's why they run in single-digit % CPU. Our GTCRN port is
verified to reproduce the upstream reference streamed output bit-for-bit.

## How it streams at low latency

DeepFilterNet3 doesn't keep recurrent state between forward passes, so naively
denoising tiny blocks sounds bad. Instead, for each ~40 ms output block the
model re-runs over ~200 ms of *recent past audio* + the new block and keeps only
the new block's output — the model's memory "warms up" over the context every
time. Past audio is already buffered, so this costs CPU, not latency. See
`denoiser/backends.py` for details.

## Setup

Requires macOS with Python 3.13 (torch has no 3.14 wheels yet) — installed here
via Homebrew. From this directory:

```bash
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt
```

The DeepFilterNet3 model downloads automatically on first run. For the low-CPU
`dtln` streaming engine, fetch its models once:

```bash
./scripts/fetch_models.sh
```

### Install a virtual audio device (for live calls)

BlackHole is a free virtual audio driver. Installing it needs your password
(it's a system audio driver), so run it yourself:

```bash
brew install blackhole-2ch
```

Then **log out/in** (or reboot) so the driver registers.

## Usage

List audio devices:

```bash
./.venv/bin/python run.py devices
```

Test on a file — no hardware or virtual device needed (great for A/B listening):

```bash
./.venv/bin/python run.py file noisy.wav clean.wav
```

Run live into a call (default DeepFilterNet engine):

```bash
./.venv/bin/python run.py live --output "BlackHole 2ch"
```

Or the recommended low-CPU streaming engine:

```bash
./.venv/bin/python run.py live --output "BlackHole 2ch" --backend gtcrn
```

You'll see input/output level meters. Then, **in your calling app, choose
"BlackHole 2ch" as the microphone**. To hear yourself for testing, open macOS
*Audio MIDI Setup* and create a Multi-Output device, or add `--output` pointing
at your speakers instead (headphones only — speakers cause feedback).

### Useful options

| Option | Meaning |
| --- | --- |
| `--input NAME/INDEX` | pick the mic (default: system default) |
| `--output NAME/INDEX` | pick the output device (the virtual mic) |
| `--atten-lim-db 25` | keep some residual noise for a more natural sound (default: full suppression) |
| `--hop-ms 40` | output granularity / added latency |
| `--context-ms 200` | warm-context length (more = slightly better, more CPU) |
| `--backend gtcrn` | recommended low-CPU per-frame streaming engine (see table above) |
| `--backend dtln` | alternate, even cheaper streaming engine |
| `--backend passthrough` | bypass denoising (measure baseline latency) |

## Project layout

```
run.py                  CLI: devices / file / live
denoiser/
  compat.py             torchaudio shim required by deepfilternet 0.5.6
  backends.py           DeepFilterNet warm-context streaming + passthrough
  gtcrn.py              GTCRN per-frame streaming backend (ONNX, recommended)
  dtln.py               DTLN true per-frame streaming backend (ONNX)
  pipeline.py           realtime mic->worker->output with ring buffer
  offline.py            file-based denoising
scripts/fetch_models.sh download DTLN ONNX weights
models/                 DTLN ONNX models (after fetch)
tests/test_denoiser.py  ring-buffer, pipeline-flow, and quality tests
```

## Notes & limits

- **CPU-only** here; it's fast enough. A GPU isn't needed.
- DeepFilterNet is trained for **speech** — it suppresses non-speech noise
  (keyboard, fans, traffic, babble). It's not a music/general-audio denoiser.
- If you hear dropouts, the `xruns` counter climbs — raise `--hop-ms` a little.
- Windows/Linux: swap BlackHole for VB-CABLE (Win) or a PulseAudio/PipeWire null
  sink (Linux); the Python code is cross-platform.
