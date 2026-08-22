"""Offline denoising of a WAV file, streamed through the backend exactly as the
realtime pipeline would feed it. Handy for A/B listening and regression checks
without any audio hardware or virtual device."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from .backends import Backend


def denoise_file(backend: Backend, in_path: str, out_path: str,
                 block_ms: float = 40.0) -> dict:
    audio, sr = sf.read(in_path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)  # downmix to mono

    if sr != backend.sr:
        mono = _resample(mono, sr, backend.sr)
        sr = backend.sr

    backend.reset()
    block = max(1, int(round(block_ms / 1000 * sr)))
    out_parts = []
    for i in range(0, len(mono), block):
        out_parts.append(backend.process(mono[i:i + block]))
    # flush any samples still buffered inside the backend
    out_parts.append(backend.process(np.zeros(backend.hop, dtype=np.float32)))
    out = np.concatenate(out_parts) if out_parts else np.zeros(0, np.float32)

    sf.write(out_path, out, sr)
    return {"sr": sr, "in_samples": len(mono), "out_samples": len(out)}


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    xp = np.linspace(0.0, 1.0, len(x), endpoint=False)
    fp = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(fp, xp, x).astype(np.float32)
