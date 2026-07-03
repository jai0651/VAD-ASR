"""
Module 0: From a waveform to a log-mel spectrogram, built from scratch.

A neural network cannot easily learn from a raw waveform: it is a 1-D array of
~16000 numbers per second that wiggles up and down. Almost all speech models
instead consume a *spectrogram*: a picture of how much energy lives at each
frequency, over short windows of time. The "mel" variant warps the frequency
axis to match human hearing (we resolve low frequencies much better than high).

The pipeline, step by step:

    waveform ──framing──▶ overlapping windows ──FFT──▶ power spectrum
             ──mel filterbank──▶ mel energies ──log──▶ log-mel spectrogram

We implement each step with only torch primitives (plus torch.fft) so nothing is
hidden. torchaudio has a one-call MelSpectrogram, but the point here is to learn.
"""

from __future__ import annotations

import math

import torch
import torchaudio


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_audio(path: str, target_sr: int = 16_000) -> tuple[torch.Tensor, int]:
    """Load an audio file as a mono waveform at `target_sr` Hz.

    Speech models almost universally use 16 kHz mono. Human speech energy lives
    below ~8 kHz, and by Nyquist a 16 kHz sample rate captures up to 8 kHz, so
    16 kHz is the standard sweet spot of quality vs. compute.
    """
    waveform, sr = torchaudio.load(path)  # shape: (channels, samples)
    if waveform.shape[0] > 1:  # mix stereo down to mono
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        waveform = torchaudio.functional.resample(waveform, sr, target_sr)
    return waveform.squeeze(0), target_sr  # shape: (samples,)


# ----------------------------------------------------------------------------
# Step 1: Framing — chop the signal into short overlapping windows
# ----------------------------------------------------------------------------
def frame_signal(
    waveform: torch.Tensor, frame_length: int, hop_length: int
) -> torch.Tensor:
    """Slice the 1-D waveform into overlapping frames.

    Speech is non-stationary over long spans but roughly stationary over ~25 ms,
    so we analyze it in short frames. We overlap them (hop < frame_length) so we
    do not miss events that straddle a frame boundary.

    Returns shape (num_frames, frame_length).
    """
    num_frames = 1 + (waveform.shape[0] - frame_length) // hop_length
    # unfold creates a sliding-window view: cheap and vectorized.
    return waveform.unfold(0, frame_length, hop_length)  # (num_frames, frame_length)


def hann_window(frame_length: int) -> torch.Tensor:
    """A tapering window applied to each frame before the FFT.

    Cutting a sine wave at arbitrary frame edges creates abrupt discontinuities,
    which smear energy across all frequencies ("spectral leakage"). Multiplying
    by a Hann window fades each frame in and out, dramatically reducing leakage.
    """
    n = torch.arange(frame_length, dtype=torch.float32)
    return 0.5 - 0.5 * torch.cos(2 * math.pi * n / (frame_length - 1))


# ----------------------------------------------------------------------------
# Step 2: FFT — measure energy at each frequency, per frame
# ----------------------------------------------------------------------------
def power_spectrum(frames: torch.Tensor, n_fft: int) -> torch.Tensor:
    """Compute the power spectrum of each (windowed) frame via the real FFT.

    rfft returns n_fft//2 + 1 complex bins (the spectrum of a real signal is
    symmetric, so we keep only the non-redundant half). |X|^2 is the power: how
    much energy sits in each frequency bin.

    Returns shape (num_frames, n_fft // 2 + 1).
    """
    windowed = frames * hann_window(frames.shape[-1])
    spectrum = torch.fft.rfft(windowed, n=n_fft)  # complex, (num_frames, n_fft//2+1)
    return spectrum.real**2 + spectrum.imag**2


# ----------------------------------------------------------------------------
# Step 3: Mel filterbank — warp linear frequency to the perceptual mel scale
# ----------------------------------------------------------------------------
def hz_to_mel(hz: torch.Tensor) -> torch.Tensor:
    # The standard "HTK" mel formula. Mels are roughly linear below 1 kHz and
    # logarithmic above, mirroring how our ears compress high frequencies.
    return 2595.0 * torch.log10(1.0 + hz / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def mel_filterbank(
    n_mels: int, n_fft: int, sr: int, fmin: float = 0.0, fmax: float | None = None
) -> torch.Tensor:
    """Build a matrix of `n_mels` triangular filters over the FFT bins.

    Each row is a triangle that is 1.0 at its center frequency and falls to 0 at
    its neighbors' centers. Multiplying the power spectrum by this matrix sums
    energy into perceptually-spaced bands, reducing hundreds of FFT bins to
    (e.g.) 80 mel channels.

    Returns shape (n_mels, n_fft // 2 + 1).
    """
    if fmax is None:
        fmax = sr / 2  # Nyquist

    # Place n_mels+2 points evenly on the mel scale, then convert back to Hz.
    mel_min, mel_max = hz_to_mel(torch.tensor(fmin)), hz_to_mel(torch.tensor(fmax))
    mel_points = torch.linspace(mel_min.item(), mel_max.item(), n_mels + 2)
    hz_points = mel_to_hz(mel_points)

    # Map each Hz point to the nearest FFT bin index.
    bin_freqs = torch.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = torch.zeros(n_mels, n_fft // 2 + 1)
    for m in range(1, n_mels + 1):
        left, center, right = hz_points[m - 1], hz_points[m], hz_points[m + 1]
        # Rising edge of the triangle (left -> center).
        rising = (bin_freqs - left) / (center - left)
        # Falling edge (center -> right).
        falling = (right - bin_freqs) / (right - center)
        fb[m - 1] = torch.clamp(torch.minimum(rising, falling), min=0.0)
    return fb


# ----------------------------------------------------------------------------
# The full pipeline
# ----------------------------------------------------------------------------
def log_mel_spectrogram(
    waveform: torch.Tensor,
    sr: int = 16_000,
    frame_ms: float = 25.0,
    hop_ms: float = 10.0,
    n_mels: int = 80,
    n_fft: int = 512,
) -> torch.Tensor:
    """Waveform -> log-mel spectrogram, shape (num_frames, n_mels).

    Defaults (25 ms frame, 10 ms hop, 80 mels) are the de-facto standard used by
    Whisper-style ASR. A 10 ms hop means one feature frame every 10 ms = 100
    frames per second, which is the time resolution our models will operate on.
    """
    frame_length = int(sr * frame_ms / 1000)
    hop_length = int(sr * hop_ms / 1000)

    frames = frame_signal(waveform, frame_length, hop_length)
    power = power_spectrum(frames, n_fft)              # (T, n_fft//2+1)
    fb = mel_filterbank(n_mels, n_fft, sr)             # (n_mels, n_fft//2+1)
    mel_energy = power @ fb.T                          # (T, n_mels)

    # log compresses the huge dynamic range of audio energy and matches our
    # roughly-logarithmic loudness perception. +1e-10 avoids log(0).
    return torch.log(mel_energy + 1e-10)
