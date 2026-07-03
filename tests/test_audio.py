import numpy as np

from src.pipeline.audio import float32_to_pcm16, pcm16_to_float32


def test_roundtrip_preserves_signal():
    x = (np.sin(np.linspace(0, 40 * np.pi, 16_000)) * 0.7).astype(np.float32)
    y = pcm16_to_float32(float32_to_pcm16(x))
    assert y.shape == x.shape
    assert np.abs(y - x).max() < 1e-3  # 16-bit quantization noise only


def test_clipping_is_safe():
    x = np.array([2.0, -2.0, 0.0], dtype=np.float32)
    y = pcm16_to_float32(float32_to_pcm16(x))
    assert np.all(np.abs(y) <= 1.0)
