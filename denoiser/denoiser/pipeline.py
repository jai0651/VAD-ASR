"""Realtime streaming pipeline: mic -> denoiser -> output device.

Design: two independent PortAudio streams (input on the mic, output on the
virtual device) decoupled by a worker thread and a sample ring buffer. This
tolerates the two devices having slightly different clocks and keeps torch
inference off the audio callback threads (callbacks must never block).

    InputStream callback --push--> Queue --> worker(denoise) --> RingBuffer
                                                                     |
                                                          OutputStream callback pulls
"""
from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd

from .backends import Backend


class _Ring:
    """Single-producer/single-consumer float32 ring buffer (sample granularity)."""

    def __init__(self, capacity: int):
        self._buf = np.zeros(capacity, dtype=np.float32)
        self._cap = capacity
        self._w = 0
        self._r = 0
        self._count = 0
        self._lock = threading.Lock()

    def write(self, data: np.ndarray) -> int:
        with self._lock:
            n = min(len(data), self._cap - self._count)  # drop overflow
            end = self._w + n
            if end <= self._cap:
                self._buf[self._w:end] = data[:n]
            else:
                first = self._cap - self._w
                self._buf[self._w:] = data[:first]
                self._buf[:end - self._cap] = data[first:n]
            self._w = end % self._cap
            self._count += n
            return n

    def read(self, n: int, out: np.ndarray) -> int:
        with self._lock:
            m = min(n, self._count)
            end = self._r + m
            if end <= self._cap:
                out[:m] = self._buf[self._r:end]
            else:
                first = self._cap - self._r
                out[:first] = self._buf[self._r:]
                out[first:m] = self._buf[:end - self._cap]
            self._r = end % self._cap
            self._count -= m
            if m < n:
                out[m:n] = 0.0  # underrun -> silence
            return m

    @property
    def count(self) -> int:
        with self._lock:
            return self._count


class RealtimeDenoiser:
    def __init__(self, backend: Backend, input_device, output_device,
                 blocksize: int | None = None, output_channels: int = 1,
                 input_gain: float = 1.0, output_gain: float = 1.0,
                 on_level=None):
        self.backend = backend
        self.sr = backend.sr
        self.input_device = input_device
        self.output_device = output_device
        # Audio-callback block; a couple of model hops keeps callbacks light.
        self.blocksize = blocksize or max(256, backend.hop)
        self.output_channels = output_channels
        self.input_gain = input_gain
        self.output_gain = output_gain
        self.on_level = on_level  # optional callback(in_rms, out_rms)

        self._q: "queue.Queue[np.ndarray | None]" = queue.Queue(maxsize=64)
        self._ring = _Ring(self.sr * 4)  # 4 s of slack
        self._worker: threading.Thread | None = None
        self._running = threading.Event()
        self._istream: sd.InputStream | None = None
        self._ostream: sd.OutputStream | None = None
        self._xruns = 0

    # --- audio callbacks (must be fast, no allocation-heavy work) -------------
    def _in_cb(self, indata, frames, time_info, status):
        if status:
            self._xruns += 1
        try:
            self._q.put_nowait(indata[:, 0].copy())
        except queue.Full:
            self._xruns += 1  # drop block if worker fell behind

    def _out_cb(self, outdata, frames, time_info, status):
        if status:
            self._xruns += 1
        tmp = np.empty(frames, dtype=np.float32)
        self._ring.read(frames, tmp)
        tmp *= self.output_gain
        if self.output_channels == 1:
            outdata[:, 0] = tmp
        else:
            outdata[:] = tmp[:, None]  # duplicate mono across channels

    # --- worker ---------------------------------------------------------------
    def _run_worker(self):
        while self._running.is_set():
            try:
                block = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if block is None:
                break
            if self.input_gain != 1.0:
                block = block * self.input_gain
            out = self.backend.process(block)
            if len(out):
                self._ring.write(out)
                if self.on_level is not None:
                    self.on_level(float(np.sqrt(np.mean(block**2)) + 1e-12),
                                  float(np.sqrt(np.mean(out**2)) + 1e-12))

    # --- lifecycle ------------------------------------------------------------
    def start(self):
        self.backend.reset()
        self._running.set()
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()
        self._istream = sd.InputStream(
            samplerate=self.sr, blocksize=self.blocksize, device=self.input_device,
            channels=1, dtype="float32", callback=self._in_cb)
        self._ostream = sd.OutputStream(
            samplerate=self.sr, blocksize=self.blocksize, device=self.output_device,
            channels=self.output_channels, dtype="float32", callback=self._out_cb)
        self._istream.start()
        self._ostream.start()

    def stop(self):
        self._running.clear()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        for s in (self._istream, self._ostream):
            if s is not None:
                s.stop(); s.close()
        if self._worker is not None:
            self._worker.join(timeout=1.0)

    @property
    def latency_ms(self) -> float:
        """Rough added latency: device blocks + model hop + ring backlog."""
        return 1000.0 * (2 * self.blocksize + self.backend.hop) / self.sr

    @property
    def xruns(self) -> int:
        return self._xruns
