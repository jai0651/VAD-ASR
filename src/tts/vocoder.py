"""
Module 4, part 1: the vocoder — log-mel spectrogram back to a waveform.

Module 0 threw two things away that we now have to reconstruct:

  1. FREQUENCY DETAIL. The mel filterbank squashed 257 FFT bins into 80
     overlapping triangles. We can't undo a lossy projection exactly, but the
     least-squares best guess is the (pseudo-)inverse of the filterbank matrix.

  2. PHASE. The power spectrum kept |X|^2 and dropped the phase angle of every
     complex FFT value — but the inverse FFT needs both. Griffin-Lim (1984)
     recovers a *plausible* phase by exploiting redundancy: our frames overlap
     (400-sample frames every 160 samples), so neighboring frames share 240
     samples and their phases are heavily constrained by each other. Iterate:

        start with random phase
        repeat: waveform = ISTFT(magnitude · e^{j·phase})   # make it a signal
                phase    = angle(STFT(waveform))            # re-measure phase
                                                            # (keep OUR magnitude)

     Each round trip makes the (magnitude, phase) pair more self-consistent.

This is why Griffin-Lim audio sounds "phasey"/robotic: the phase is merely
consistent, not the true one. Neural vocoders (WaveNet -> HiFi-GAN) replaced
this whole file with a network that *learns* to write plausible waveforms —
that is the single biggest audio-quality jump in the history of TTS.

Everything here reuses Module 0's own pieces (frame_signal, hann_window,
mel_filterbank) so the round trip is exact-by-construction.
"""

from __future__ import annotations

import torch

from src.audio.features import frame_signal, hann_window, mel_filterbank

SR = 16_000
FRAME = 400      # 25 ms — must match Module 0's defaults
HOP = 160        # 10 ms
N_FFT = 512
N_MELS = 80


def stft(waveform: torch.Tensor) -> torch.Tensor:
    """Waveform -> complex spectrogram (T, N_FFT//2+1). Module 0's power_spectrum
    is exactly |stft|^2; here we keep the complex values because Griffin-Lim
    needs the phase angle."""
    frames = frame_signal(waveform, FRAME, HOP)
    return torch.fft.rfft(frames * hann_window(FRAME), n=N_FFT)


def istft(spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
    """Complex spectrogram -> waveform by windowed overlap-add.

    Each frame becomes 512 time samples (irfft); we keep the first 400 (the
    rest is zero-padding from the forward transform), window AGAIN, and add
    each frame into place at its hop offset. Windowing twice then dividing by
    the summed squared window is the standard "synthesis window" trick that
    makes the overlap-add exactly cancel out where windows overlap.
    """
    frames = torch.fft.irfft(spec, n=N_FFT)[:, :FRAME]  # (T, FRAME)
    win = hann_window(FRAME)
    frames = frames * win

    t = frames.shape[0]
    out_len = FRAME + (t - 1) * HOP
    wav = torch.zeros(out_len)
    norm = torch.zeros(out_len)
    win_sq = win * win
    for i in range(t):
        start = i * HOP
        wav[start:start + FRAME] += frames[i]
        norm[start:start + FRAME] += win_sq
    wav = wav / norm.clamp(min=1e-8)
    return wav[:length] if length is not None else wav


_fb_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}


def _filterbank_and_pinv() -> tuple[torch.Tensor, torch.Tensor]:
    if 0 not in _fb_cache:
        fb = mel_filterbank(N_MELS, N_FFT, SR)          # (80, 257)
        _fb_cache[0] = (fb, torch.linalg.pinv(fb))      # pinv: (257, 80)
    return _fb_cache[0]


def log_mel_to_magnitude(log_mel: torch.Tensor) -> torch.Tensor:
    """(T, 80) log-mel -> (T, 257) linear magnitude spectrogram.

    exp undoes Module 0's log; the pseudo-inverse is the least-squares undo of
    the filterbank (negative values are projection artifacts -> clamp); sqrt
    undoes power = magnitude^2.
    """
    _, fb_pinv = _filterbank_and_pinv()
    power = (log_mel.exp() @ fb_pinv.T).clamp(min=0.0)
    return power.sqrt()


def griffin_lim(
    magnitude: torch.Tensor, n_iters: int = 60, length: int | None = None
) -> torch.Tensor:
    """Magnitude spectrogram (T, 257) -> waveform, via iterative phase recovery."""
    # Random initial phase; unit-magnitude complex numbers e^{j·theta}.
    theta = 2 * torch.pi * torch.rand_like(magnitude)
    phase = torch.polar(torch.ones_like(magnitude), theta)
    for _ in range(n_iters):
        wav = istft(magnitude * phase)
        respec = stft(wav)
        # Keep the re-measured phase, discard the re-measured magnitude.
        phase = respec / respec.abs().clamp(min=1e-8)
        # Padded/edge frames can differ in count by one; align defensively.
        phase = phase[: magnitude.shape[0]]
    return istft(magnitude * phase, length=length)


def log_mel_to_waveform(log_mel: torch.Tensor, n_iters: int = 60) -> torch.Tensor:
    """The full inverse of Module 0: (T, 80) log-mel -> (samples,) float32."""
    wav = griffin_lim(log_mel_to_magnitude(log_mel), n_iters=n_iters)
    peak = wav.abs().max()
    return wav / peak * 0.9 if peak > 0 else wav
