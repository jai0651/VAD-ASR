"""Denoiser backends.

A backend takes mono float32 audio in [-1, 1] at ``self.sr`` and returns
denoised audio. Backends are *stateful*: feed them consecutive, non-overlapping
blocks of audio in stream order. ``process`` buffers internally to the backend's
natural hop size, so it may return fewer or more samples than it was given (it
emits only fully-processed hops); callers should treat it as a stream, not a
1:1 block transform.
"""
from __future__ import annotations

import numpy as np


class Backend:
    sr: int = 48000
    #: Natural processing hop in samples. The pipeline sizes its buffers around
    #: this and it sets the backend's output latency.
    hop: int = 480

    def process(self, block: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class PassthroughBackend(Backend):
    """Does nothing. Baseline for measuring the pipeline's own latency/overhead."""

    def __init__(self, sr: int = 48000):
        self.sr = sr
        self.hop = int(0.02 * sr)

    def process(self, block: np.ndarray) -> np.ndarray:
        return np.asarray(block, dtype=np.float32)


class DeepFilterNetBackend(Backend):
    """DeepFilterNet3 realtime speech enhancer using *warm left-context* streaming.

    DeepFilterNet3's model does not carry recurrent state between separate
    forward passes, so denoising a small block in isolation is poor. Instead,
    for each output hop we re-run the model over a window of recent past audio
    plus the new hop, and keep only the new hop's output. The model's GRU thus
    "warms up" over the context every time, recovering near whole-file quality
    while output latency stays ~= one hop. Cost is extra CPU (the context is
    reprocessed each hop), which is cheap here (RTF well under 1).

    Parameters
    ----------
    hop_ms : output granularity / added latency (default 40 ms).
    context_ms : left-context length (default 200 ms; >200 gives little gain).
    atten_lim_db : cap on attenuation in dB. None = full suppression. A value
        like 25 keeps some residual noise for a more natural sound.
    post_filter : DeepFilterNet's optional extra light suppression.
    """

    def __init__(self, hop_ms: float = 40.0, context_ms: float = 200.0,
                 atten_lim_db: float | None = None, post_filter: bool = False):
        from . import compat  # noqa: F401  (applies torchaudio shim on import)
        compat.apply()

        import torch
        from df.enhance import init_df, enhance

        self._torch = torch
        self._enhance = enhance

        self.model, self.df_state, _ = init_df(
            post_filter=post_filter, log_level="ERROR", log_file=None
        )
        self.model.eval()
        self.sr = self.df_state.sr()
        self.atten_lim_db = atten_lim_db

        base_hop = self.df_state.hop_size()  # 480 @ 48k
        # Round hop/context to whole model hops so the STFT consumes them cleanly.
        self.hop = max(base_hop, int(round(hop_ms / 1000 * self.sr / base_hop)) * base_hop)
        self.context = max(0, int(round(context_ms / 1000 * self.sr / base_hop)) * base_hop)

        self._inbuf = np.zeros(0, dtype=np.float32)      # not-yet-hopped input
        self._history = np.zeros(0, dtype=np.float32)     # recent input for context
        self.reset()

    def reset(self) -> None:
        self._inbuf = np.zeros(0, dtype=np.float32)
        self._history = np.zeros(0, dtype=np.float32)
        self.df_state.reset()

    def process(self, block: np.ndarray) -> np.ndarray:
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        self._inbuf = np.concatenate([self._inbuf, block])
        out_parts = []
        while len(self._inbuf) >= self.hop:
            hop_blk = self._inbuf[:self.hop]
            self._inbuf = self._inbuf[self.hop:]
            out_parts.append(self._process_hop(hop_blk))
        if not out_parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(out_parts)

    def _process_hop(self, hop_blk: np.ndarray) -> np.ndarray:
        torch = self._torch
        window = np.concatenate([self._history, hop_blk])
        t = torch.from_numpy(np.ascontiguousarray(window)).unsqueeze(0)  # [1, T]
        # Fresh STFT state per window (we feed overlapping audio each hop).
        self.df_state.reset()
        with torch.no_grad():
            y = self._enhance(self.model, self.df_state, t, pad=True,
                              atten_lim_db=self.atten_lim_db)
        y = y.squeeze(0).cpu().numpy()
        n = min(len(hop_blk), len(y))
        out = np.zeros(len(hop_blk), dtype=np.float32)
        out[len(hop_blk) - n:] = y[len(y) - n:]
        # Slide the context window forward by this hop.
        self._history = np.concatenate([self._history, hop_blk])
        if len(self._history) > self.context:
            self._history = self._history[-self.context:]
        return out


def make_backend(name: str, **kwargs) -> Backend:
    name = name.lower()
    if name in ("passthrough", "none", "off"):
        return PassthroughBackend(sr=kwargs.get("sr", 48000))
    if name in ("deepfilternet", "dfn", "df"):
        return DeepFilterNetBackend(
            hop_ms=kwargs.get("hop_ms", 40.0),
            context_ms=kwargs.get("context_ms", 200.0),
            atten_lim_db=kwargs.get("atten_lim_db"),
            post_filter=kwargs.get("post_filter", False),
        )
    if name in ("gtcrn", "stream", "streaming"):
        from .gtcrn import GTCRNBackend
        return GTCRNBackend(num_threads=kwargs.get("num_threads", 1))
    if name == "dtln":
        from .dtln import DTLNBackend
        return DTLNBackend(num_threads=kwargs.get("num_threads", 1))
    raise ValueError(f"unknown backend: {name!r}")
