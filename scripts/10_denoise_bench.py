"""
Module 5, part 4: does the denoiser actually help? Measure, don't assume.

A noise suppressor is the one stage that can make the whole pipeline WORSE
while looking like it's working: it visibly removes noise, and quietly removes
speech with it. So this script answers the three questions that decide the
config, on your machine, on your models:

  A. SIGNAL   how much noise is removed, how much speech is damaged, what it
              costs in CPU. ΔSNR alone is a bad summary — an engine that
              attenuates everything by 20 dB scores fine on "noise removed" —
              so we report pause-attenuation and speech-attenuation separately.

  B. VAD      the failure that started Module 1b: room noise read as speech.
              Mean P(speech) per scenario, with and without each engine. This
              is where noise suppression pays off most and least ambiguously.

  C. ASR      character error rate on noisy speech, raw vs denoised, for BOTH
              recognizers. This is the one that decides `denoise_target`:
              a model trained on clean audio (yours) and a model trained on
              680k hours of the internet (Whisper) do not react the same way
              to an enhancement front-end. See docs/07-sota.html.

Run:
  uv run python scripts/10_denoise_bench.py
  N=20 SNR_DB=0 uv run python scripts/10_denoise_bench.py     # harder
  WHISPER=1 uv run python scripts/10_denoise_bench.py         # include Whisper
  ENGINES=none,spectral uv run python scripts/10_denoise_bench.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.decode import char_error_rate  # noqa: E402
from src.audio.features import log_mel_spectrogram  # noqa: E402
from src.pipeline.config import PipelineConfig  # noqa: E402
from src.pipeline.denoise import make_denoiser  # noqa: E402
from src.vad.data_real import (  # noqa: E402
    SR,
    _babble,
    _colored_noise,
    _hum,
    list_speech_files,
    load_clip,
)
from src.vad.model import VADNet  # noqa: E402

N = int(os.environ.get("N", "12"))
SNR_DB = float(os.environ.get("SNR_DB", "5"))
ENGINES = os.environ.get("ENGINES", "none,spectral,gtcrn,dtln").split(",")
WITH_WHISPER = os.environ.get("WHISPER", "0") == "1"


# ---------------------------------------------------------------------------
def engine(name: str, atten_db: float = 18.0):
    cfg = PipelineConfig(denoise_engine=name, denoise_atten_db=atten_db)
    return make_denoiser(cfg)


def run(den, x: np.ndarray, block: int = 320) -> np.ndarray:
    """Stream `x` through the engine and undo its algorithmic delay, so the
    result is sample-aligned with the input and directly comparable."""
    den.reset()
    parts = [den.process(x[i:i + block]) for i in range(0, len(x), block)]
    # Flush: push `delay` extra samples so the tail comes out too.
    parts.append(den.process(np.zeros(max(den.hop, den.delay), np.float32)))
    y = np.concatenate([p for p in parts if len(p)])
    d = den.delay
    out = np.zeros(len(x), np.float32)
    take = min(len(x), max(0, len(y) - d))
    out[:take] = y[d:d + take]
    return out


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    noise = noise[: len(speech)]
    if len(noise) < len(speech):
        noise = np.tile(noise, len(speech) // max(1, len(noise)) + 1)[: len(speech)]
    scale = np.sqrt(np.mean(speech ** 2)) / (np.sqrt(np.mean(noise ** 2)) + 1e-12)
    return (speech + noise * scale * 10.0 ** (-snr_db / 20.0)).astype(np.float32)


def snr_db(clean: np.ndarray, test: np.ndarray) -> float:
    n = min(len(clean), len(test))
    err = test[:n] - clean[:n]
    return float(10 * np.log10((np.sum(clean[:n] ** 2) + 1e-12) / (np.sum(err ** 2) + 1e-12)))


def speech_mask(clean: np.ndarray) -> np.ndarray:
    env = np.convolve(clean ** 2, np.ones(400) / 400, mode="same")
    return env > 0.02 * env.max()


def atten_db(noisy: np.ndarray, out: np.ndarray, mask: np.ndarray) -> float:
    a = np.sqrt(np.mean(noisy[mask] ** 2)) + 1e-12
    b = np.sqrt(np.mean(out[mask] ** 2)) + 1e-12
    return float(20 * np.log10(a / b))


# ---------------------------------------------------------------------------
def load_utterances(n: int) -> list[tuple[np.ndarray, str]]:
    """(audio, reference transcript) pairs from LibriSpeech dev-clean."""
    from src.asr.librispeech import LibriSpeechFeatures

    ds = LibriSpeechFeatures(root="data", url="dev-clean", min_seconds=2.0,
                             max_seconds=6.0)
    rng = random.Random(0)
    items = rng.sample(ds.items, min(n, len(ds.items)))
    import soundfile as sf

    out = []
    for path, text in items:
        audio, sr = sf.read(path, dtype="float32")
        assert sr == SR
        peak = np.abs(audio).max()
        out.append(((audio / peak if peak > 0 else audio), text.lower()))
    return out


def build_noises(files: list[str], n: int) -> dict[str, np.ndarray]:
    random.seed(1234)
    np.random.seed(1234)
    return {
        "colored": _colored_noise(n),
        "babble": _babble(n, files),
        "hum": _hum(n),
    }


# ---------------------------------------------------------------------------
def part_a(utts, files) -> dict[str, dict]:
    print(f"\n{'='*78}\nA. SIGNAL QUALITY — {len(utts)} utterances x 3 noise types "
          f"@ {SNR_DB:.0f} dB SNR\n{'='*78}")
    print(f"{'engine':10s} {'ΔSNR':>7s} {'pause-attn':>11s} {'speech-attn':>12s} "
          f"{'RTF':>8s} {'latency':>8s}")
    results = {}
    for name in ENGINES:
        try:
            den = engine(name)
        except FileNotFoundError as e:
            print(f"{name:10s} skipped: {e}")
            continue
        d_snr, pause, sp, secs, audio_s = [], [], [], 0.0, 0.0
        for clean, _ in utts:
            mask = speech_mask(clean)
            noises = build_noises(files, len(clean))
            for noise in noises.values():
                noisy = mix_at_snr(clean, noise, SNR_DB)
                t0 = time.perf_counter()
                out = run(den, noisy)
                secs += time.perf_counter() - t0
                audio_s += len(noisy) / SR
                d_snr.append(snr_db(clean, out) - snr_db(clean, noisy))
                pause.append(atten_db(noisy, out, ~mask))
                sp.append(atten_db(noisy, out, mask))
        lat = 1000.0 * den.delay / SR
        results[name] = {
            "d_snr": float(np.mean(d_snr)), "pause": float(np.mean(pause)),
            "speech": float(np.mean(sp)), "rtf": secs / audio_s, "latency_ms": lat,
        }
        r = results[name]
        print(f"{name:10s} {r['d_snr']:+6.2f}dB {r['pause']:10.1f}dB "
              f"{r['speech']:11.1f}dB {r['rtf']:8.3f} {lat:6.0f}ms")
    print("\n  ΔSNR         higher is better (0 for 'none' by definition)")
    print("  pause-attn   noise removed where nobody is speaking — want HIGH")
    print("  speech-attn  signal removed where someone IS speaking — want LOW")
    print("  RTF          seconds of CPU per second of audio, 1 core")
    return results


def part_b(files) -> None:
    ckpt = Path("outputs/vad_real.pt")
    if not ckpt.exists():
        print("\nB. skipped — no outputs/vad_real.pt (run scripts/09 first)")
        return
    print(f"\n{'='*78}\nB. VAD — mean P(speech) per scenario "
          f"(the Module 1b hard cases)\n{'='*78}")

    model = VADNet(n_mels=80)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    random.seed(1234)
    np.random.seed(1234)
    n = 4 * SR
    speech = load_clip(sorted(files)[0], max_s=4.0) * 0.3
    cases = {
        "digital zeros": np.zeros(n, dtype=np.float32),
        "room noise": _colored_noise(n) * 0.03,
        "babble": _babble(n, files) * 0.03,
        "mains hum": _hum(n) * 0.03,
        "speech": speech,
        "speech + noise": mix_at_snr(speech, _colored_noise(len(speech)), SNR_DB),
        "quiet speech": speech * 0.1,
    }

    @torch.no_grad()
    def prob(w):
        feats = log_mel_spectrogram(torch.from_numpy(np.ascontiguousarray(w)), sr=SR)
        return float(torch.sigmoid(model(feats.unsqueeze(0)))[0].mean())

    engines = {}
    for name in ENGINES:
        try:
            engines[name] = engine(name)
        except FileNotFoundError:
            pass

    header = "scenario".ljust(16) + "".join(f"{k:>10s}" for k in engines)
    print(header + "     want")
    for case, wav in cases.items():
        want = "HIGH" if "speech" in case else "low"
        row = "".join(f"{prob(run(d, wav)):10.3f}" for d in engines.values())
        print(f"{case:16s}{row}     {want}")
    print("\n  A denoiser in front of the VAD should push the non-speech rows DOWN")
    print("  without pushing the speech rows down with them.")


def part_c(utts, files) -> None:
    print(f"\n{'='*78}\nC. ASR — character error rate on noisy speech "
          f"@ {SNR_DB:.0f} dB SNR\n{'='*78}")

    recognizers = {}
    cfg = PipelineConfig()
    if Path(cfg.scratch_asr_ckpt).exists():
        from src.pipeline.scratch_engines import ScratchASR

        recognizers["scratch"] = ScratchASR(cfg)
    if WITH_WHISPER:
        from src.pipeline.asr import WhisperASR

        recognizers["whisper"] = WhisperASR(cfg)
    if not recognizers:
        print("  no recognizers available")
        return

    engines = {}
    for name in ENGINES:
        try:
            engines[name] = engine(name)
        except FileNotFoundError:
            pass

    print("asr".ljust(10) + "".join(f"{k:>10s}" for k in engines))
    for asr_name, asr in recognizers.items():
        cers = {k: [] for k in engines}
        for clean, ref in utts:
            noisy = mix_at_snr(clean, build_noises(files, len(clean))["colored"], SNR_DB)
            for k, den in engines.items():
                text = asr.transcribe(run(den, noisy)).text.lower().strip(" .,!?")
                cers[k].append(char_error_rate(text, ref.strip(" .,!?")))
        row = "".join(f"{np.mean(v):10.3f}" for v in cers.values())
        print(f"{asr_name:10s}{row}")
    print("\n  Lower is better. If a recognizer's CER goes UP with denoising, that")
    print("  recognizer wants VOICE_DENOISE_TARGET=vad — clean audio for the VAD,")
    print("  original audio for the ASR.")


def main() -> None:
    files = list_speech_files()
    utts = load_utterances(N)
    print(f"engines: {ENGINES}   utterances: {len(utts)}   input SNR: {SNR_DB} dB")
    part_a(utts, files)
    part_b(files)
    part_c(utts, files)


if __name__ == "__main__":
    main()
