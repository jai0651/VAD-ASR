"""
Byte-level audio plumbing.

The wire format between client and server is 16-bit signed little-endian PCM
(what telephony, WebRTC internals, and most speech APIs use). Models want
float32 in [-1, 1]. These two functions are the entire conversion story —
kept separate so the contract is impossible to miss.
"""

from __future__ import annotations

import numpy as np

INT16_MAX = 32768.0


def pcm16_to_float32(data: bytes) -> np.ndarray:
    """Little-endian int16 bytes -> float32 waveform in [-1, 1]."""
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / INT16_MAX


def float32_to_pcm16(samples: np.ndarray) -> bytes:
    """float32 waveform in [-1, 1] -> little-endian int16 bytes (clipped)."""
    clipped = np.clip(samples, -1.0, 1.0)
    return (clipped * (INT16_MAX - 1)).astype("<i2").tobytes()
