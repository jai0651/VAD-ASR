"""
Module 7: a neural vocoder — the single biggest audio-quality jump available.

Module 4 ends with Griffin-Lim, which *guesses* the phase that Module 0 threw
away by iterating toward self-consistency. It is a beautiful algorithm and it is
a HARD CEILING: the phase it finds is merely consistent, not correct, which is
exactly why the output sounds phasey and robotic. No amount of extra training
data for the acoustic model moves that ceiling, because the ceiling is not in
the acoustic model.

The fix is to LEARN the mapping. Two families:

  TIME-DOMAIN (HiFi-GAN, 2020). Upsample the mel 160x with transposed convs
  until it is a waveform. Excellent quality, but the network has to synthesise
  every individual sample — expensive, and it needs adversarial training to
  avoid sounding muffled.

  SPECTRAL-DOMAIN (Vocos / ISTFT-Net, 2023). Keep the STFT. Predict the
  magnitude AND the phase per frame, then run one exact inverse STFT. This is
  what we build. The insight is that the hard part was never "make a waveform"
  — the ISTFT does that exactly — it was "know the phase". So predict the phase
  and stop pretending the rest is hard.

Why spectral-domain is the right choice here specifically:

  IT IS CHEAP. One frame of output per 160 samples instead of 160 upsampling
  steps: roughly two orders of magnitude less compute per second of audio. That
  matters when the target is a laptop CPU inside a latency budget.

  IT SHARES OUR GRID. n_fft/hop/window are Module 0's exact values, so the mels
  our Tacotron already produces drop straight in. No retraining of Module 4.

  IT DEGRADES GRACEFULLY. Trained on multi-resolution STFT loss alone (no GAN),
  it is already far better than Griffin-Lim. Adding discriminators later is a
  quality upgrade, not a prerequisite — so there is a working artifact at every
  stage instead of only at the end.

The backbone is a 1-D ConvNeXt: depthwise conv for context, LayerNorm, and an
inverted-bottleneck MLP per position. No attention — a vocoder needs a receptive
field of a few tens of milliseconds, not a sentence, and convolutions give that
far more cheaply.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from src.audio.features import hann_window

SR = 16_000
N_FFT = 512     # Module 0's values, so mel grids line up exactly
WIN = 400       # 25 ms
HOP = 160       # 10 ms
N_MELS = 80
N_BINS = N_FFT // 2 + 1


# ---------------------------------------------------------------------------
# batched, differentiable STFT / ISTFT
# ---------------------------------------------------------------------------
def stft_batch(wav: torch.Tensor, n_fft: int = N_FFT, hop: int = HOP,
               win: int = WIN) -> torch.Tensor:
    """(B, L) -> complex (B, T, n_fft//2+1). Same framing as Module 0."""
    window = hann_window(win).to(wav.device, wav.dtype)
    frames = wav.unfold(-1, win, hop)                  # (B, T, win)
    return torch.fft.rfft(frames * window, n=n_fft)


def istft_batch(spec: torch.Tensor, n_fft: int = N_FFT, hop: int = HOP,
                win: int = WIN, length: int | None = None) -> torch.Tensor:
    """Complex (B, T, F) -> (B, L), by windowed overlap-add.

    Module 4's `istft` loops over frames in Python — fine for synthesising one
    utterance, hopeless inside a training loop. `F.fold` performs exactly the
    same overlap-add as one fused, differentiable op: it is the transpose of
    the unfold that created the frames, which is precisely what overlap-add is.
    """
    b, t, _ = spec.shape
    window = hann_window(win).to(spec.device, torch.float32)
    frames = torch.fft.irfft(spec, n=n_fft)[..., :win] * window   # (B, T, win)

    out_len = win + (t - 1) * hop
    fold_kw = dict(output_size=(1, out_len), kernel_size=(1, win), stride=(1, hop))
    wav = F.fold(frames.transpose(1, 2), **fold_kw).reshape(b, out_len)

    # Divide out the summed squared window: windowing on analysis AND synthesis
    # means each sample was scaled twice, by a different amount per position.
    win_sq = (window ** 2).view(1, win, 1).expand(1, win, t)
    norm = F.fold(win_sq, **fold_kw).reshape(1, out_len)
    wav = wav / norm.clamp(min=1e-8)
    return wav[:, :length] if length is not None else wav


# ---------------------------------------------------------------------------
class ConvNeXtBlock(nn.Module):
    """Depthwise conv -> LayerNorm -> inverted-bottleneck MLP, with layer scale.

    The 2020s replacement for a ResNet block: one big-kernel depthwise conv for
    context (cheap, since it is per-channel), then all the capacity in a
    pointwise MLP that expands 3x and comes back. `gamma` starts near zero so
    each block is initially close to the identity — which is what lets a deep
    stack train stably without warmup tricks.
    """

    def __init__(self, dim: int, hidden: int, kernel: int = 7,
                 layer_scale: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, hidden)
        self.pw2 = nn.Linear(hidden, dim)
        self.gamma = nn.Parameter(layer_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:      # (B, C, T)
        residual = x
        x = self.dwconv(x).transpose(1, 2)                   # (B, T, C)
        x = self.pw2(F.gelu(self.pw1(self.norm(x)))) * self.gamma
        return residual + x.transpose(1, 2)


class ISTFTVocoder(nn.Module):
    """log-mel (B, T, 80) -> waveform (B, T*hop + win - hop)."""

    def __init__(self, n_mels: int = N_MELS, dim: int = 256, n_blocks: int = 8,
                 hidden_mult: int = 3, kernel: int = 7):
        super().__init__()
        self.embed = nn.Conv1d(n_mels, dim, kernel_size=7, padding=3)
        self.norm_in = nn.LayerNorm(dim)
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(dim, dim * hidden_mult, kernel) for _ in range(n_blocks)
        ])
        self.norm_out = nn.LayerNorm(dim)
        # Two heads sharing a trunk: log-magnitude and phase angle per bin.
        self.head = nn.Linear(dim, 2 * N_BINS)

    def forward(self, log_mel: torch.Tensor, length: int | None = None
                ) -> torch.Tensor:
        x = self.embed(log_mel.transpose(1, 2))              # (B, dim, T)
        x = self.norm_in(x.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        x = self.norm_out(x.transpose(1, 2))                 # (B, T, dim)

        mag_log, phase = self.head(x).chunk(2, dim=-1)
        # Predict LOG magnitude: spectra span ~120 dB, and a linear head cannot
        # represent that range without either saturating or exploding. The clamp
        # is a hard safety rail — a single exp(30) produces inf and poisons the
        # whole batch's gradients.
        mag = torch.exp(mag_log.clamp(max=math.log(1e3)))
        spec = torch.polar(mag, phase)                       # mag · e^{i·phase}
        return istft_batch(spec, length=length)


# ---------------------------------------------------------------------------
class MultiResolutionSTFTLoss(nn.Module):
    """Compare predicted and true waveforms in several STFT resolutions at once.

    A single resolution is a trap: a short window has good time resolution and
    poor frequency resolution, so a model trained on it gets pitch subtly wrong;
    a long window is the reverse and smears transients. Using three at once
    means an error has nowhere to hide.

    Per resolution, two terms:
      SPECTRAL CONVERGENCE  ‖|S|-|Ŝ|‖_F / ‖|S|‖_F — relative, so it is dominated
                            by the loud, high-energy parts.
      LOG-MAGNITUDE L1      the reverse: log compresses the range, so quiet
                            detail (breath, fricatives, decay tails) gets a vote.
    Neither alone is sufficient, which is exactly why the standard loss is both.

    Note what is NOT here: any phase term. Phase is unconstrained by this loss —
    the model is free to choose any phase that makes the *magnitudes* right
    after overlap-add. That is the point: consistent phase emerges from the
    ISTFT's own redundancy rather than being supervised directly.
    """

    RESOLUTIONS = ((512, 160, 400), (1024, 256, 600), (256, 80, 200))

    def __init__(self, resolutions=RESOLUTIONS):
        super().__init__()
        self.resolutions = resolutions

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        n = min(pred.shape[-1], target.shape[-1])
        pred, target = pred[..., :n], target[..., :n]
        total = pred.new_zeros(())
        for n_fft, hop, win in self.resolutions:
            if n < win:
                continue
            p = stft_batch(pred, n_fft, hop, win).abs().clamp(min=1e-7)
            t = stft_batch(target, n_fft, hop, win).abs().clamp(min=1e-7)
            sc = torch.norm(t - p, p="fro") / torch.norm(t, p="fro").clamp(min=1e-7)
            mag = F.l1_loss(torch.log(p), torch.log(t))
            total = total + sc + mag
        return total / len(self.resolutions)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
