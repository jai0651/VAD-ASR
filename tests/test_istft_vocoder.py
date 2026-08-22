"""
Module 7 tests.

`test_stft_istft_reconstructs_exactly` is the load-bearing one. The entire
premise of a spectral-domain vocoder is that the inverse STFT is EXACT, so the
only thing the network has to learn is the phase. If the overlap-add is even
slightly lossy, the model spends its capacity compensating for our arithmetic
instead, and every quality number is measuring the wrong thing.

`test_overfits_one_crop` is the vocoder equivalent of the ASR's overfit test:
a model that cannot reproduce a single second of audio it has seen 300 times
has a bug, not a data problem.
"""

from __future__ import annotations

import math

import torch

from src.tts.istft_vocoder import (
    HOP,
    WIN,
    ISTFTVocoder,
    MultiResolutionSTFTLoss,
    count_parameters,
    istft_batch,
    stft_batch,
)


def _tone(n: int, f0: float = 180.0, sr: int = 16_000) -> torch.Tensor:
    t = torch.arange(n, dtype=torch.float32) / sr
    return 0.3 * sum(torch.sin(2 * math.pi * f0 * k * t) / k for k in range(1, 8))


def test_stft_istft_reconstructs_exactly():
    """Analysis window x synthesis window, divided by the summed squared window,
    is mathematically an identity in the interior. Assert machine precision —
    anything looser is hiding a framing bug."""
    x = torch.randn(2, 16_000) * 0.1
    y = istft_batch(stft_batch(x))
    n = min(x.shape[-1], y.shape[-1])
    err = (x[:, WIN:n - WIN] - y[:, WIN:n - WIN]).abs().max()
    assert err < 1e-6, f"reconstruction error {err:.2e}"


def test_istft_matches_the_module4_loop_implementation():
    """The fold-based overlap-add must agree with Module 4's explicit Python
    loop — same maths, one is just usable inside a training step."""
    from src.tts.vocoder import istft as istft_loop

    x = _tone(8000)
    spec = stft_batch(x.unsqueeze(0))
    fast = istft_batch(spec)[0]
    slow = istft_loop(spec[0])
    n = min(len(fast), len(slow))
    torch.testing.assert_close(fast[:n], slow[:n], atol=1e-5, rtol=1e-4)


def test_output_length_formula():
    model = ISTFTVocoder(dim=32, n_blocks=1)
    for t in (50, 100, 233):
        out = model(torch.randn(1, t, 80))
        assert out.shape == (1, WIN + (t - 1) * HOP)


def test_length_argument_trims():
    model = ISTFTVocoder(dim=32, n_blocks=1)
    assert model(torch.randn(1, 60, 80), length=5000).shape == (1, 5000)


def test_forward_backward_is_finite():
    model = ISTFTVocoder(dim=32, n_blocks=2)
    mel = torch.randn(2, 80, 80)
    out = model(mel)
    loss = MultiResolutionSTFTLoss()(out, torch.randn_like(out) * 0.1)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_magnitude_clamp_prevents_inf():
    """A single exp(30) produces inf and poisons the whole batch's gradients.
    Drive the magnitude head hard positive and check the output stays finite."""
    model = ISTFTVocoder(dim=32, n_blocks=1)
    with torch.no_grad():
        model.head.bias[: model.head.bias.numel() // 2] = 50.0
    assert torch.isfinite(model(torch.randn(1, 40, 80))).all()


def test_mr_stft_loss_is_zero_for_identical_signals():
    x = _tone(16_000).unsqueeze(0)
    loss = MultiResolutionSTFTLoss()
    assert float(loss(x, x)) < 1e-5
    assert float(loss(x, torch.randn_like(x) * 0.1)) > 0.1


def test_mr_stft_loss_penalises_pitch_error():
    """Sanity that the metric tracks something perceptual: a wrong-pitch tone
    must score worse than the right one."""
    ref = _tone(16_000, f0=180.0).unsqueeze(0)
    close = _tone(16_000, f0=182.0).unsqueeze(0)
    far = _tone(16_000, f0=300.0).unsqueeze(0)
    loss = MultiResolutionSTFTLoss()
    assert float(loss(close, ref)) < float(loss(far, ref))


def test_overfits_one_crop():
    """One second of audio, 300 steps: the model must learn to reproduce it."""
    torch.manual_seed(0)
    from src.asr.corpus import LogMel

    wav = _tone(16_240).unsqueeze(0)
    mel = LogMel()(wav[0]).unsqueeze(0)

    model = ISTFTVocoder(dim=64, n_blocks=3)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = MultiResolutionSTFTLoss()

    first = None
    for _ in range(300):
        pred = model(mel)
        loss = criterion(pred, wav[:, : pred.shape[-1]])
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    assert float(loss.detach()) < 0.3 * first, f"{first:.3f} -> {float(loss.detach()):.3f}"


def test_parameter_count_is_vocoder_sized():
    n = count_parameters(ISTFTVocoder(dim=256, n_blocks=8))
    # Comparable to Vocos (~13M) / HiFi-GAN v1 (~14M), an order under an ASR encoder.
    assert 1e6 < n < 20e6, f"{n/1e6:.1f}M"
