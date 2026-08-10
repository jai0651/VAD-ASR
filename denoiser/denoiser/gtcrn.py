"""GTCRN streaming backend — the recommended low-latency, low-CPU engine.

GTCRN (Grouped Temporal Convolutional Recurrent Network; Rong et al., 2024) is
an ultra-light (23.7K parameter) complex-spectrum speech enhancer. Like DTLN it
is architected for true per-frame streaming, but it works in the complex domain
(so it corrects phase, not just magnitude) and is a newer, stronger design.

Streaming ONNX carries three cache tensors (conv / temporal-attention / dual-path
RNN state) frame to frame. Each 16 ms hop runs one forward — no recompute — at
~3% of a CPU core. Runs at 16 kHz. Verified to reproduce the reference GTCRN
streaming output bit-for-bit (up to the one-hop framing latency).
"""
from __future__ import annotations

import os

import numpy as np

from .backends import Backend

SR = 16000
NFFT = 512    # 32 ms analysis window
HOP = 256     # 16 ms hop
# sqrt-Hann on both analysis and synthesis => exact overlap-add at 50% overlap.
WINDOW = np.hanning(NFFT).astype(np.float32) ** 0.5

_CONV_CACHE = (2, 1, 16, 16, 33)
_TRA_CACHE = (2, 3, 1, 1, 16)
_INTER_CACHE = (2, 1, 33, 16)


def _default_model_path() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "models", "gtcrn.onnx")


class GTCRNBackend(Backend):
    def __init__(self, model_path: str | None = None, num_threads: int = 1):
        import onnxruntime as ort

        model_path = model_path or _default_model_path()
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"GTCRN model not found: {model_path}\nRun scripts/fetch_models.sh to download it.")
        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = num_threads
        self._s = ort.InferenceSession(model_path, so, providers=["CPUExecutionProvider"])
        self._out_names = [o.name for o in self._s.get_outputs()]
        self.sr = SR
        self.hop = HOP
        self.reset()

    def reset(self) -> None:
        self._in = np.zeros(NFFT, np.float32)
        self._out = np.zeros(NFFT, np.float32)
        self._cc = np.zeros(_CONV_CACHE, np.float32)
        self._tc = np.zeros(_TRA_CACHE, np.float32)
        self._ic = np.zeros(_INTER_CACHE, np.float32)
        self._buf = np.zeros(0, np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, block])
        out = []
        while len(self._buf) >= HOP:
            out.append(self._hop(self._buf[:HOP]))
            self._buf = self._buf[HOP:]
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def _hop(self, new: np.ndarray) -> np.ndarray:
        self._in = np.roll(self._in, -HOP)
        self._in[-HOP:] = new
        spec = np.fft.rfft(self._in * WINDOW)  # [257] complex
        mix = np.stack([spec.real, spec.imag], -1).astype(np.float32).reshape(1, 257, 1, 2)
        enh, self._cc, self._tc, self._ic = self._s.run(
            self._out_names,
            {"mix": mix, "conv_cache": self._cc, "tra_cache": self._tc, "inter_cache": self._ic})
        spec = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
        frame = np.fft.irfft(spec).astype(np.float32) * WINDOW
        self._out = np.roll(self._out, -HOP)
        self._out[-HOP:] = 0.0
        self._out += frame
        return self._out[:HOP].copy()
