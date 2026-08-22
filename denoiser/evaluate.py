#!/usr/bin/env python
"""Benchmark the denoising engines on real noisy speech (VoiceBank-DEMAND test
set) with standard speech-enhancement metrics.

    ./.venv/bin/python evaluate.py [N]        # N = number of test files (default 60)

Metrics (higher is better for all):
  PESQ-WB   perceptual speech quality (−0.5..4.5)
  STOI      short-time objective intelligibility (0..1)
  SI-SDR    scale-invariant signal-to-distortion ratio (dB)

Each engine's output is aligned to the clean reference (engines have different
algorithmic latencies) before scoring. The 'noisy' row is the unprocessed
baseline; an engine is only worth it if it beats that row.
"""
import io
import sys
import time
import warnings

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

warnings.filterwarnings("ignore")

PARQUET = "data/vbd_test.parquet"
SR = 16000  # VoiceBank-DEMAND-16k


# ----------------------------------------------------------------- metrics ----
def si_sdr(ref, est):
    ref = ref - ref.mean(); est = est - est.mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-12)
    target = a * ref
    noise = est - target
    return 10 * np.log10((np.dot(target, target) + 1e-12) / (np.dot(noise, noise) + 1e-12))


def align(ref, est, max_lag=800):
    """Shift est to best-correlate with ref, then trim both to equal length."""
    n = min(len(ref), len(est))
    ref, est = ref[:n], est[:n]
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1, 4):
        if lag >= 0:
            a, b = ref[lag:], est[:n - lag]
        else:
            a, b = ref[:n + lag], est[-lag:]
        m = min(len(a), len(b))
        if m < n // 2:
            continue
        c = np.dot(a[:m], b[:m])
        if c > best:
            best, best_lag = c, lag
    lag = best_lag
    if lag >= 0:
        a, b = ref[lag:], est[:n - lag]
    else:
        a, b = ref[:n + lag], est[-lag:]
    m = min(len(a), len(b))
    return a[:m], b[:m]


def score(ref, est):
    from pesq import pesq
    from pystoi import stoi
    ref, est = align(ref, est)
    out = {"si_sdr": si_sdr(ref, est)}
    try:
        out["pesq"] = pesq(SR, ref, est, "wb")
    except Exception:
        out["pesq"] = np.nan
    try:
        out["stoi"] = stoi(ref, est, SR, extended=False)
    except Exception:
        out["stoi"] = np.nan
    return out


# ----------------------------------------------------------------- engines ----
def run_engine(backend, noisy):
    x = noisy if backend.sr == SR else resample_poly(noisy, backend.sr, SR).astype(np.float32)
    backend.reset()
    out = backend.process(x)
    out = np.concatenate([out, backend.process(np.zeros(backend.hop * 4, np.float32))])
    if backend.sr != SR:
        out = resample_poly(out, SR, backend.sr).astype(np.float32)
    return out


def load_pairs(n):
    pf = pq.ParquetFile(PARQUET)
    pairs = []
    for batch in pf.iter_batches(batch_size=64):
        for row in batch.to_pylist():
            clean, _ = sf.read(io.BytesIO(row["clean"]["bytes"]), dtype="float32")
            noisy, _ = sf.read(io.BytesIO(row["noisy"]["bytes"]), dtype="float32")
            if clean.ndim > 1: clean = clean.mean(1)
            if noisy.ndim > 1: noisy = noisy.mean(1)
            pairs.append((row["id"], clean, noisy))
            if len(pairs) >= n:
                return pairs
    return pairs


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    print(f"loading {n} VoiceBank-DEMAND test pairs...", flush=True)
    pairs = load_pairs(n)

    from denoiser.backends import make_backend
    print("loading engines...", flush=True)
    engines = {
        "deepfilternet": make_backend("deepfilternet"),
        "gtcrn": make_backend("gtcrn"),
        "dtln": make_backend("dtln"),
    }

    rows = {name: {"pesq": [], "stoi": [], "si_sdr": []} for name in ["noisy", *engines]}
    t0 = time.time()
    for i, (fid, clean, noisy) in enumerate(pairs):
        for k, v in score(clean, noisy).items():
            rows["noisy"][k].append(v)
        for name, be in engines.items():
            est = run_engine(be, noisy)
            for k, v in score(clean, est).items():
                rows[name][k].append(v)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(pairs)} ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n=== Results on {len(pairs)} VoiceBank-DEMAND test files ===")
    print(f"{'engine':>14} {'PESQ-WB':>9} {'STOI':>7} {'SI-SDR':>8}")
    def m(x): return float(np.nanmean(x))
    for name in ["noisy", "dtln", "gtcrn", "deepfilternet"]:
        r = rows[name]
        tag = "  (baseline)" if name == "noisy" else ""
        print(f"{name:>14} {m(r['pesq']):>9.3f} {m(r['stoi']):>7.3f} {m(r['si_sdr']):>8.2f}{tag}")
    print("\nHigher is better for all. Δ vs noisy shows how much each engine actually helps.")


if __name__ == "__main__":
    main()
