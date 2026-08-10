"""Test suite for the realtime denoiser.

    ../.venv/bin/python -m pytest tests/ -s        # or just run this file directly

Covers: ring-buffer FIFO integrity, pipeline data flow (no hardware), and
end-to-end denoising quality on macOS `say`-generated speech + noise.
"""
import warnings; warnings.filterwarnings("ignore")
import math
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from denoiser.pipeline import _Ring, RealtimeDenoiser  # noqa: E402
from denoiser.backends import PassthroughBackend, DeepFilterNetBackend  # noqa: E402

SR = 48000


# ---------------------------------------------------------------- helpers -----
def _snr_db(clean, est):
    n = min(len(clean), len(est))
    clean, est = clean[:n], est[:n]
    best = None
    for lag in range(-1200, 1201, 8):  # fine alignment search (per-engine STFT/hop delay)
        if lag >= 0:
            c, e = clean[lag:], est[:n - lag]
        else:
            c, e = clean[:n + lag], est[-lag:]
        m = min(len(c), len(e))
        if m < SR:
            continue
        c, e = c[:m], e[:m]
        g = np.dot(c, e) / (np.dot(e, e) + 1e-9)  # optimal gain
        err = c - g * e
        s = 10 * math.log10((np.dot(c, c) + 1e-9) / (np.dot(err, err) + 1e-9))
        best = s if best is None or s > best else best
    return best


def _say_speech():
    path = "/tmp/denoiser_test_clean.wav"
    subprocess.run(["say", "-o", path, "--data-format=LEI16@48000",
                    "The quarterly numbers look strong. "
                    "Can everyone hear me clearly on the call today?"], check=True)
    with wave.open(path) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    return a.astype(np.float32) / 32768.0


# ------------------------------------------------------------------ tests -----
def test_ring_fifo_integrity():
    rng = np.random.default_rng(0)
    cap = 1000
    r = _Ring(cap)
    written, read = [], []
    for _ in range(5000):
        if rng.random() < 0.5:
            data = rng.standard_normal(int(rng.integers(1, 300))).astype(np.float32)
            k = r.write(data)
            written.extend(data[:k].tolist())
        else:
            n = int(rng.integers(1, 300))
            out = np.empty(n, np.float32)
            got = r.read(n, out)
            read.extend(out[:got].tolist())
    out = np.empty(cap, np.float32); got = r.read(cap, out); read.extend(out[:got].tolist())
    assert len(read) <= len(written)
    assert np.allclose(read, written[:len(read)], atol=1e-6)
    assert r.count <= cap


def test_pipeline_flow_no_hardware():
    be = PassthroughBackend(sr=SR)
    rd = RealtimeDenoiser(be, None, None, blocksize=be.hop, output_channels=1)
    rd._running.set()
    w = threading.Thread(target=rd._run_worker, daemon=True); w.start()
    sig = np.sin(2 * np.pi * 220 * np.arange(SR) / SR).astype(np.float32)
    bs = be.hop
    for i in range(0, len(sig), bs):
        blk = np.pad(sig[i:i + bs], (0, max(0, bs - len(sig[i:i + bs]))))
        rd._in_cb(blk[:, None], bs, None, None)
    time.sleep(0.5)
    collected = []
    for _ in range(len(sig) // bs + 2):
        out = np.zeros((bs, 1), np.float32)
        rd._out_cb(out, bs, None, None)
        collected.append(out[:, 0].copy())
    rd._running.clear(); rd._q.put(None); w.join(timeout=1)
    assert np.sqrt(np.mean(np.concatenate(collected) ** 2)) > 0.3
    assert rd.xruns == 0


def test_denoise_quality():
    clean = _say_speech()
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= math.sqrt(np.mean(clean ** 2) / (np.mean(noise ** 2) + 1e-9))  # ~0 dB
    noisy = clean + noise

    be = DeepFilterNetBackend()
    be.reset()
    bs = int(0.04 * SR)
    out = np.concatenate([be.process(noisy[i:i + bs]) for i in range(0, len(noisy), bs)])

    in_snr, out_snr = _snr_db(clean, noisy), _snr_db(clean, out)
    print(f"\n  input SNR {in_snr:+.1f} dB -> output SNR {out_snr:+.1f} dB "
          f"({out_snr - in_snr:+.1f} dB)")
    assert out_snr - in_snr > 6.0, "denoiser should improve SNR by well over 6 dB"


def _resample(x, a, b):
    n = int(round(len(x) * b / a))
    return np.interp(np.linspace(0, 1, n, endpoint=False),
                     np.linspace(0, 1, len(x), endpoint=False), x).astype(np.float32)


def test_dtln_streaming_quality_and_cpu():
    """The low-CPU per-frame streaming backend: verify it denoises AND that it is
    genuinely realtime-cheap (RTF well under 1, i.e. true streaming, no recompute)."""
    import os
    import time
    from pathlib import Path
    if not (Path(__file__).resolve().parents[1] / "models" / "dtln_1.onnx").is_file():
        print("\n  [skipped] DTLN models absent — run scripts/fetch_models.sh")
        return
    from denoiser.dtln import DTLNBackend, SR as DSR

    clean48 = _say_speech()
    clean = _resample(clean48, SR, DSR)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= math.sqrt(np.mean(clean ** 2) / (np.mean(noise ** 2) + 1e-9))
    noisy = clean + noise

    be = DTLNBackend()
    be.process(noisy[:DSR // 2])  # warmup
    be.reset()
    t = time.time()
    out = np.concatenate([be.process(noisy[i:i + 160]) for i in range(0, len(noisy), 160)])
    rtf = (time.time() - t) / (len(noisy) / DSR)

    # rebuild global SR for the aligner
    import tests.test_denoiser as T
    old = T.SR; T.SR = DSR
    try:
        gain = _snr_db(clean, out) - _snr_db(clean, noisy)
    finally:
        T.SR = old
    print(f"\n  DTLN gain {gain:+.1f} dB · RTF {rtf:.3f} (~{rtf*100:.0f}% of one core)")
    assert gain > 5.0, "streaming backend should improve SNR by >5 dB"
    assert rtf < 0.5, "streaming backend must be well under realtime"


def test_gtcrn_streaming_and_cpu():
    """GTCRN streaming backend denoises AND is cheap (RTF << 1)."""
    import time
    from pathlib import Path
    if not (Path(__file__).resolve().parents[1] / "models" / "gtcrn.onnx").is_file():
        print("\n  [skipped] GTCRN model absent — run scripts/fetch_models.sh")
        return
    from denoiser.gtcrn import GTCRNBackend, SR as GSR

    clean = _resample(_say_speech(), SR, GSR)
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(len(clean)).astype(np.float32)
    noise *= math.sqrt(np.mean(clean ** 2) / (np.mean(noise ** 2) + 1e-9))
    noisy = clean + noise

    be = GTCRNBackend()
    be.process(noisy[:GSR // 2]); be.reset()
    t = time.time()
    out = np.concatenate([be.process(noisy[i:i + 160]) for i in range(0, len(noisy), 160)])
    rtf = (time.time() - t) / (len(noisy) / GSR)

    import tests.test_denoiser as T
    old = T.SR; T.SR = GSR
    try:
        gain = _snr_db(clean, out) - _snr_db(clean, noisy)
    finally:
        T.SR = old
    print(f"\n  GTCRN gain {gain:+.1f} dB · RTF {rtf:.3f} (~{rtf*100:.0f}% of one core)")
    assert gain > 5.0
    assert rtf < 0.5


def test_gtcrn_matches_reference():
    """Strong correctness check: our GTCRN streaming port must reproduce the
    upstream reference streamed output (up to the one-hop framing latency)."""
    import urllib.request
    from pathlib import Path
    if not (Path(__file__).resolve().parents[1] / "models" / "gtcrn.onnx").is_file():
        print("\n  [skipped] GTCRN model absent"); return
    import soundfile as sf
    from denoiser.gtcrn import GTCRNBackend
    base = "https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/main/stream/test_wavs"
    try:
        for n in ("mix.wav", "enh_stream.wav"):
            p = f"/tmp/gtref_{n}"
            if not Path(p).is_file():
                urllib.request.urlretrieve(f"{base}/{n}", p)
    except Exception as e:
        print(f"\n  [skipped] could not fetch reference wavs: {e}"); return
    mix, _ = sf.read("/tmp/gtref_mix.wav", dtype="float32")
    ref, _ = sf.read("/tmp/gtref_enh_stream.wav", dtype="float32")
    if mix.ndim > 1: mix = mix.mean(1)
    if ref.ndim > 1: ref = ref.mean(1)
    est = GTCRNBackend().process(mix)
    n = min(len(est), len(ref)); e, r = est[:n], ref[:n]
    # align for the one-hop (256-sample) streaming latency, then correlate
    best = max(range(0, 400),
               key=lambda lag: np.corrcoef(e[lag:], r[:n - lag])[0, 1] if n - lag > n // 2 else -1)
    corr = np.corrcoef(e[best:], r[:n - best])[0, 1]
    print(f"\n  GTCRN vs reference: lag={best}, corr={corr:.4f}")
    assert corr > 0.99, "streaming port should match the reference output"


if __name__ == "__main__":
    test_ring_fifo_integrity(); print("test_ring_fifo_integrity: OK")
    test_pipeline_flow_no_hardware(); print("test_pipeline_flow_no_hardware: OK")
    test_denoise_quality(); print("test_denoise_quality: OK")
    test_dtln_streaming_quality_and_cpu(); print("test_dtln_streaming_quality_and_cpu: OK")
    test_gtcrn_streaming_and_cpu(); print("test_gtcrn_streaming_and_cpu: OK")
    test_gtcrn_matches_reference(); print("test_gtcrn_matches_reference: OK")
    print("\nALL TESTS PASSED")
