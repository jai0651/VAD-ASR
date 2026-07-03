"""Module 4 vocoder: the DSP must be exactly invertible where math says it is."""

import torch

from src.tts.vocoder import HOP, SR, griffin_lim, istft, log_mel_to_waveform, stft


def _sine(freq: float, seconds: float = 1.0) -> torch.Tensor:
    t = torch.arange(int(SR * seconds)) / SR
    return 0.5 * torch.sin(2 * torch.pi * freq * t)


def dominant_freq(wav: torch.Tensor) -> float:
    spec = torch.fft.rfft(wav).abs()
    return torch.fft.rfftfreq(wav.shape[0], 1 / SR)[spec.argmax()].item()


def test_istft_inverts_stft_exactly():
    # With a hann window and 60% overlap, windowed overlap-add reconstruction
    # is exact (up to float noise) away from the very edges.
    x = _sine(440.0) + 0.3 * _sine(1300.0)
    y = istft(stft(x), length=x.shape[0])
    core = slice(HOP * 2, x.shape[0] - HOP * 2)  # ignore edge frames
    assert (y[core] - x[core]).abs().max() < 1e-4


def test_griffin_lim_preserves_pitch():
    # Phase is invented (random init — seed for determinism), but the
    # magnitude (hence the pitch) must survive.
    torch.manual_seed(0)
    x = _sine(440.0)
    mag = stft(x).abs()
    y = griffin_lim(mag, n_iters=30, length=x.shape[0])
    assert abs(dominant_freq(y) - 440.0) < 10.0


def test_full_mel_inversion_preserves_pitch():
    # The whole chain: waveform -> log-mel (Module 0) -> waveform (Module 4).
    from src.audio.features import log_mel_spectrogram

    x = _sine(440.0)
    mel = log_mel_spectrogram(x, sr=SR)
    y = log_mel_to_waveform(mel, n_iters=30)
    # Mel pooling blurs frequency: allow a mel-bin-width of slack.
    assert abs(dominant_freq(y) - 440.0) < 40.0
    assert y.abs().max() <= 0.95  # output is normalized
