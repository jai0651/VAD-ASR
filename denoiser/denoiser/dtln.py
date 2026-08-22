"""DTLN streaming backend — a true per-frame, low-CPU speech enhancer.

Unlike the DeepFilterNet backend (which re-processes a context window every hop),
DTLN is *architected* for streaming: two ONNX stages, each carrying its own LSTM
state from frame to frame. Every 8 ms hop runs exactly one forward of each stage,
so it costs ~1.5% of a CPU core with ~32 ms latency — Krisp-class engine
characteristics. The trade-off is a few dB less suppression than DeepFilterNet3.

Model: DTLN (Westhausen & Meyer, 2020), pretrained ONNX from the reference repo.
Runs at 16 kHz (wideband voice). See scripts/fetch_models.sh to download weights.
"""
from __future__ import annotations

import os

import numpy as np

from .backends import Backend

SR = 16000
BLOCK = 512   # analysis window -> 32 ms algorithmic latency
SHIFT = 128   # hop -> 8 ms processing granularity
_STATE = (1, 2, 128, 2)  # LSTM state tensor shape per stage


def _default_model_dir() -> str:
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


class DTLNBackend(Backend):
    def __init__(self, model_dir: str | None = None, num_threads: int = 1):
        import onnxruntime as ort

        model_dir = model_dir or _default_model_dir()
        m1 = os.path.join(model_dir, "dtln_1.onnx")
        m2 = os.path.join(model_dir, "dtln_2.onnx")
        for m in (m1, m2):
            if not os.path.isfile(m):
                raise FileNotFoundError(
                    f"DTLN model not found: {m}\nRun scripts/fetch_models.sh to download it.")

        so = ort.SessionOptions()
        so.intra_op_num_threads = num_threads
        so.inter_op_num_threads = num_threads
        self._s1 = ort.InferenceSession(m1, so, providers=["CPUExecutionProvider"])
        self._s2 = ort.InferenceSession(m2, so, providers=["CPUExecutionProvider"])
        self._i1 = [i.name for i in self._s1.get_inputs()]
        self._o1 = [o.name for o in self._s1.get_outputs()]
        self._i2 = [i.name for i in self._s2.get_inputs()]
        self._o2 = [o.name for o in self._s2.get_outputs()]

        self.sr = SR
        self.hop = SHIFT
        self.reset()

    def reset(self) -> None:
        self._in = np.zeros(BLOCK, np.float32)
        self._out = np.zeros(BLOCK, np.float32)
        self._st1 = np.zeros(_STATE, np.float32)
        self._st2 = np.zeros(_STATE, np.float32)
        self._buf = np.zeros(0, np.float32)

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, block])
        out = []
        while len(self._buf) >= SHIFT:
            out.append(self._hop(self._buf[:SHIFT]))
            self._buf = self._buf[SHIFT:]
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def _hop(self, new: np.ndarray) -> np.ndarray:
        # slide input frame
        self._in = np.roll(self._in, -SHIFT)
        self._in[-SHIFT:] = new
        # stage 1: spectral-magnitude mask, phase kept
        spec = np.fft.rfft(self._in)
        mag = np.abs(spec).astype(np.float32)
        mask, self._st1 = self._s1.run(
            self._o1, {self._i1[0]: mag[None, None, :], self._i1[1]: self._st1})
        est = (mag * mask[0, 0]) * np.exp(1j * np.angle(spec))
        frame = np.fft.irfft(est).astype(np.float32)
        # stage 2: learned-domain refinement
        y, self._st2 = self._s2.run(
            self._o2, {self._i2[0]: frame[None, None, :], self._i2[1]: self._st2})
        # overlap-add
        self._out = np.roll(self._out, -SHIFT)
        self._out[-SHIFT:] = 0.0
        self._out += y[0, 0]
        return self._out[:SHIFT].copy()
