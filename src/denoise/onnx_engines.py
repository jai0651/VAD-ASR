"""
Module 5, part 2: the production denoisers — what you'd actually ship.

Both are ported from the sibling `Denoiser/` project (which benchmarked them on
VoiceBank-DEMAND); the streaming mechanics are unchanged, only the interface
was adapted to this repo's stage contract and given an attenuation limit.

    engine   params   latency   CPU     PESQ-WB*  what it fixes vs. spectral.py
    ------   ------   -------   ----    --------  ------------------------------
    spectral  0       32 ms     ~0.5%   —         (the from-scratch baseline)
    gtcrn    23.7K    32 ms     ~3%     2.66      learns noise *types*; complex
                                                  domain, so it fixes phase too
    dtln      1.0 M   32 ms     ~1.5%   2.55      two-stage: spectral mask then
                                                  a learned-domain refinement

    * on 60 VoiceBank-DEMAND test files, 16 kHz; unprocessed noisy = 2.19.

WHY A NEURAL MODEL BEATS THE HAND-WRITTEN ONE. spectral.py assumes noise is
*stationary enough* that its running minimum is a good estimate. That holds for
fans, hiss and hum and fails completely for the noises that actually ruin calls:
a door slam, a keyboard, a barking dog, another person talking. A trained model
has seen thousands of hours of those and recognises them as noise from their
time-frequency *shape*, not their stationarity. It also estimates a complex
mask (GTCRN), so it can repair phase — impossible for any gain-only method.

WHY THEY'RE STILL TINY. Speech enhancement is a per-bin regression with strong
local structure, so it needs capacity in the right places, not in total: grouped
convolutions, sub-band feature extraction, and a dual-path RNN over the
frequency axis. GTCRN gets Krisp-class quality out of 23.7K parameters (39.6 MMACs/s) — roughly
one thousandth of a small ASR encoder.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"

_FETCH_HINT = "bash scripts/fetch_denoise_models.sh"


def _require(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"denoiser weights not found: {path} — run: {_FETCH_HINT}")
    return str(path)


def _session(path: str, num_threads: int = 1):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = num_threads
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


class _OLAEngine:
    """Shared overlap-add streaming shell for the ONNX models.

    Both models are frame-in/frame-out with internal recurrent state, so the
    outer loop is identical: slide a window by `hop`, run one forward, overlap-
    add, emit the completed hop. Subclasses only implement `_forward(frame)`.

    The emitted hop lags its input hop by exactly one hop (the second half of
    the analysis window has not arrived yet) — hence `delay == hop`, which the
    pipeline uses to keep the raw and cleaned streams sample-aligned.
    """

    sr = 16_000
    nfft = 512
    hop = 256
    name = "ola"

    def __init__(self, max_atten_db: float = 18.0):
        # Attenuation limit as a residual mix: y = clean + g·(noisy - clean).
        # g = 0 removes the noise entirely; g = 10^(-18/20) leaves it 18 dB down.
        # Capping suppression matters for ASR (see docs/08-sota.html) and stops
        # the unnatural dead-silence-between-words effect.
        self._resid = float(10.0 ** (-abs(max_atten_db) / 20.0))
        n = np.arange(self.nfft)
        self._win = np.sqrt(0.5 - 0.5 * np.cos(2.0 * np.pi * n / self.nfft)).astype(np.float32)
        self._in_rms = 0.0
        self._out_rms = 0.0

    @property
    def delay(self) -> int:
        return self.hop

    def reset(self) -> None:
        self._in = np.zeros(self.nfft, np.float32)
        self._ola = np.zeros(self.nfft, np.float32)
        # Delay line that keeps the residual mix aligned: the cleaned hop we
        # emit corresponds to input from `delay` samples ago, so the "noisy"
        # term of the mix has to come from the same instants.
        self._raw_delay = np.zeros(self.delay, np.float32)
        self._buf = np.zeros(0, np.float32)
        self._reset_state()

    def _aligned_raw(self, new: np.ndarray) -> np.ndarray:
        self._raw_delay = np.concatenate([self._raw_delay, new])
        out, self._raw_delay = self._raw_delay[: self.hop], self._raw_delay[self.hop:]
        return out

    def _reset_state(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _forward(self, frame: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        self._buf = np.concatenate([self._buf, block])
        out = []
        while self._buf.shape[0] >= self.hop:
            out.append(self._hop(self._buf[: self.hop]))
            self._buf = self._buf[self.hop:]
        if not out:
            return np.zeros(0, np.float32)
        y = np.concatenate(out)
        n = max(1, y.shape[0])
        self._in_rms = float(np.sqrt(np.mean(block.astype(np.float64) ** 2) + 1e-12))
        self._out_rms = float(np.sqrt(np.sum(y.astype(np.float64) ** 2) / n + 1e-12))
        return y

    @property
    def reduction_db(self) -> float:
        return float(20.0 * np.log10((self._in_rms + 1e-12) / (self._out_rms + 1e-12)))

    def _hop(self, new: np.ndarray) -> np.ndarray:
        self._in = np.roll(self._in, -self.hop)
        self._in[-self.hop:] = new
        frame = self._forward(self._in * self._win) * self._win
        self._ola = np.roll(self._ola, -self.hop)
        self._ola[-self.hop:] = 0.0
        self._ola += frame
        clean = self._ola[: self.hop].copy()

        if self._resid > 0.0:
            clean = clean + self._resid * (self._aligned_raw(new) - clean)
        else:
            self._aligned_raw(new)
        return clean


class GTCRNDenoiser(_OLAEngine):
    """GTCRN (Rong et al., 2024) — 23.7K params, complex-domain, per-frame streaming.

    Three cache tensors (grouped-conv / temporal-attention / dual-path RNN
    state) carry context frame to frame, so each 16 ms hop is exactly one
    forward — no recompute, ~3% of a core. Best perceptual quality of the three
    engines here and the recommended production default.
    """

    name = "gtcrn"
    _CONV_CACHE = (2, 1, 16, 16, 33)
    _TRA_CACHE = (2, 3, 1, 1, 16)
    _INTER_CACHE = (2, 1, 33, 16)

    def __init__(self, max_atten_db: float = 18.0, model_path: str | None = None,
                 num_threads: int = 1):
        super().__init__(max_atten_db)
        path = model_path or _require(MODELS_DIR / "gtcrn.onnx")
        self._s = _session(path, num_threads)
        self._out_names = [o.name for o in self._s.get_outputs()]
        self.reset()

    def _reset_state(self) -> None:
        self._cc = np.zeros(self._CONV_CACHE, np.float32)
        self._tc = np.zeros(self._TRA_CACHE, np.float32)
        self._ic = np.zeros(self._INTER_CACHE, np.float32)

    def _forward(self, windowed: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(windowed)
        mix = np.stack([spec.real, spec.imag], -1).astype(np.float32).reshape(1, 257, 1, 2)
        enh, self._cc, self._tc, self._ic = self._s.run(
            self._out_names,
            {"mix": mix, "conv_cache": self._cc, "tra_cache": self._tc,
             "inter_cache": self._ic},
        )
        out = enh[0, :, 0, 0] + 1j * enh[0, :, 0, 1]
        return np.fft.irfft(out).astype(np.float32)


class DTLNDenoiser(_OLAEngine):
    """DTLN (Westhausen & Meyer, 2020) — two LSTM stages, ~1.5% of a core.

    Stage 1 predicts a magnitude mask and keeps the noisy phase (exactly what
    spectral.py does, but learned). Stage 2 then refines the result in a
    *learned* 1-D analysis domain, which is how it recovers some of the phase
    error stage 1 leaves behind. Highest SI-SDR of the three engines.
    """

    name = "dtln"
    _STATE = (1, 2, 128, 2)
    # DTLN is trained for 512-sample frames at a 128-sample shift (75% overlap)
    # with RECTANGULAR analysis/synthesis — the network itself learns to emit
    # frames that overlap-add correctly. Run it at 50% overlap like GTCRN and
    # reconstruction breaks (measured: 19 dB SNR in, 6 dB out). The shift is
    # part of the model, not a tuning knob.
    nfft = 512
    hop = 128

    def __init__(self, max_atten_db: float = 18.0, model_dir: str | None = None,
                 num_threads: int = 1):
        super().__init__(max_atten_db)
        d = Path(model_dir) if model_dir else MODELS_DIR
        self._s1 = _session(_require(d / "dtln_1.onnx"), num_threads)
        self._s2 = _session(_require(d / "dtln_2.onnx"), num_threads)
        self._i1 = [i.name for i in self._s1.get_inputs()]
        self._o1 = [o.name for o in self._s1.get_outputs()]
        self._i2 = [i.name for i in self._s2.get_inputs()]
        self._o2 = [o.name for o in self._s2.get_outputs()]
        self.reset()

    @property
    def delay(self) -> int:
        # With rectangular framing the emitted hop is the OLDEST hop of the
        # current window, so the lag is a full window minus one hop, not one
        # hop (verified by impulse probe: 384 samples = 24 ms).
        return self.nfft - self.hop

    # DTLN was trained with a rectangular analysis window and does its own
    # internal shaping, so we bypass the sqrt-Hann of the shared shell.
    def _hop(self, new: np.ndarray) -> np.ndarray:
        self._in = np.roll(self._in, -self.hop)
        self._in[-self.hop:] = new
        frame = self._forward(self._in)
        self._ola = np.roll(self._ola, -self.hop)
        self._ola[-self.hop:] = 0.0
        self._ola += frame
        clean = self._ola[: self.hop].copy()
        if self._resid > 0.0:
            clean = clean + self._resid * (self._aligned_raw(new) - clean)
        else:
            self._aligned_raw(new)
        return clean

    def _reset_state(self) -> None:
        self._st1 = np.zeros(self._STATE, np.float32)
        self._st2 = np.zeros(self._STATE, np.float32)

    def _forward(self, frame_in: np.ndarray) -> np.ndarray:
        spec = np.fft.rfft(frame_in)
        mag = np.abs(spec).astype(np.float32)
        mask, self._st1 = self._s1.run(
            self._o1, {self._i1[0]: mag[None, None, :], self._i1[1]: self._st1}
        )
        est = (mag * mask[0, 0]) * np.exp(1j * np.angle(spec))
        frame = np.fft.irfft(est).astype(np.float32)
        y, self._st2 = self._s2.run(
            self._o2, {self._i2[0]: frame[None, None, :], self._i2[1]: self._st2}
        )
        return y[0, 0]
