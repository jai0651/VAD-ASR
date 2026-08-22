"""
Module 6, part 2: the Conformer encoder, from scratch.

Module 2's encoder was Conv1d + BiGRU. That is a 2015 design, and it fails in
two specific ways: a GRU compresses all history into one fixed-size state (so
long-range context is lossy), and bidirectionality means it cannot emit anything
until the utterance ends. Every 2026 recognizer — Parakeet, Whisper, Qwen3-ASR —
is built on self-attention instead.

But pure self-attention is *worse* than a conv net at the thing speech needs
most: local, translation-invariant patterns (formant transitions, plosive
bursts). Conformer's insight is that these are complementary, so use both in
every block:

    x ← x + ½·FFN(x)          "macaron": half a feed-forward before...
    x ← x + MHSA(x)           global context, with RELATIVE positions
    x ← x + Conv(x)           local context, depthwise, kernel 31 (~150 ms)
    x ← x + ½·FFN(x)          ...and half after
    x ← LayerNorm(x)

Three details in here matter more than the block diagram:

  RELATIVE POSITIONAL ATTENTION. Absolute sinusoids tell the model "this is
  frame 412", which is meaningless for speech — what matters is that a frame is
  30 ms after another. We use the Transformer-XL formulation, where the
  attention score decomposes into content-content and content-position terms
  and the position term is *shifted* rather than indexed. This is why a
  Conformer trained on 10 s utterances still works on 30 s ones.

  CONV SUBSAMPLING ×4. Two strided 2-D convs turn 100 frames/s into 25. This
  is not just a speed trick: attention is O(T²), so ×4 subsampling is a 16×
  reduction in attention cost, and it is what makes self-attention affordable
  for audio at all.

  DYNAMIC CHUNK MASKING. The trick (from WeNet) that gives you ONE model that
  runs both offline and streaming. During training, sample a random chunk size
  per batch; each frame may attend only inside its own chunk plus some chunks
  of history. At inference you pick the chunk size, trading latency for
  accuracy, with no retraining. Combined with a CAUSAL depthwise conv (left
  padding only), the model provably never sees the future beyond its chunk.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# masks
# ---------------------------------------------------------------------------
def make_pad_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """(B,) lengths -> (B, T) bool mask, True where the frame is PADDING."""
    max_len = int(max_len or lengths.max().item())
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return idx >= lengths.unsqueeze(1)


def make_chunk_mask(
    size: int, chunk_size: int, left_chunks: int = -1, device=None
) -> torch.Tensor:
    """(T, T) bool mask, True where attention is ALLOWED.

    chunk_size <= 0 means full (offline) context. Otherwise frame t may attend
    to any frame in its own chunk or in the previous `left_chunks` chunks
    (-1 = all history). Note frames inside a chunk see each other, including
    slightly "future" ones — that lookahead is exactly the latency you pay, and
    it is why chunked models beat strictly-causal ones at the same cost.
    """
    if chunk_size <= 0:
        return torch.ones(size, size, dtype=torch.bool, device=device)
    idx = torch.arange(size, device=device)
    c = idx // chunk_size                       # chunk index per frame
    delta = c.unsqueeze(0) - c.unsqueeze(1)     # [query, key] chunk distance
    allowed = delta <= 0                        # never attend to future chunks
    if left_chunks >= 0:
        allowed &= delta >= -left_chunks
    return allowed


# ---------------------------------------------------------------------------
# front end
# ---------------------------------------------------------------------------
class ConvSubsampling(nn.Module):
    """×4 subsampling: two 3×3 stride-2 convs over the (time, mel) image."""

    def __init__(self, n_mels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2),
            nn.ReLU(),
        )
        freq_out = ((n_mels - 1) // 2 - 1) // 2
        self.out = nn.Linear(d_model * freq_out, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def out_length(lengths: torch.Tensor) -> torch.Tensor:
        return ((lengths - 1) // 2 - 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, n_mels) -> (B, T//4, d_model)."""
        x = self.conv(x.unsqueeze(1))                 # (B, d, T', F')
        b, d, t, f = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b, t, d * f)
        return self.dropout(self.out(x))


class RelPositionalEncoding(nn.Module):
    """Sinusoids for relative offsets +(T-1) … 0 … -(T-1), i.e. length 2T-1."""

    # 2000 subsampled frames = 80 s of audio; grown on demand if ever exceeded.
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        self.d_model = d_model
        self.register_buffer("pe", self._build(max_len, d_model), persistent=False)

    @staticmethod
    def _build(length: int, d_model: int) -> torch.Tensor:
        pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / d_model)
        )
        pos_pe = torch.zeros(length, d_model)
        neg_pe = torch.zeros(length, d_model)
        pos_pe[:, 0::2], pos_pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        neg_pe[:, 0::2], neg_pe[:, 1::2] = torch.sin(-pos * div), torch.cos(-pos * div)
        # Positive offsets descending, then negative offsets ascending.
        return torch.cat([torch.flip(pos_pe, [0]), neg_pe[1:]], dim=0).unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        if self.pe.size(1) < 2 * t - 1:
            self.pe = self._build(t, self.d_model).to(x.device, x.dtype)
        mid = self.pe.size(1) // 2
        return self.pe[:, mid - t + 1: mid + t].to(x.device, x.dtype)


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
class RelPositionMultiHeadAttention(nn.Module):
    """Transformer-XL style attention: score = content·content + content·position."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.pos = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model)
        # Two learned global biases replacing the absolute-position terms that
        # drop out of the Transformer-XL decomposition.
        self.u_bias = nn.Parameter(torch.zeros(self.h, self.dk))
        self.v_bias = nn.Parameter(torch.zeros(self.h, self.dk))
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _rel_shift(x: torch.Tensor) -> torch.Tensor:
        """(B, H, T, 2T-1) -> (B, H, T, T): align each row to its own offsets.

        Pure index bookkeeping. Row t of the input holds scores against
        offsets +(T-1)…-(T-1); we want it to hold scores against absolute keys
        0…T-1. Padding by one and reshaping performs that diagonal shift in
        two ops instead of a gather.
        """
        b, h, t, n = x.shape
        x = torch.cat([x.new_zeros(b, h, t, 1), x], dim=-1)
        x = x.view(b, h, n + 1, t)
        return x[:, :, 1:].view(b, h, t, n)[:, :, :, : n // 2 + 1]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.h, self.dk).transpose(1, 2)   # (B,H,T,dk)
        k = self.k(x).view(b, t, self.h, self.dk).transpose(1, 2)
        v = self.v(x).view(b, t, self.h, self.dk).transpose(1, 2)
        p = self.pos(pos_emb).view(1, -1, self.h, self.dk).transpose(1, 2)

        q_u = q + self.u_bias.view(1, self.h, 1, self.dk)
        q_v = q + self.v_bias.view(1, self.h, 1, self.dk)
        ac = torch.matmul(q_u, k.transpose(-2, -1))                 # (B,H,T,T)
        bd = self._rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))
        scores = (ac + bd) / math.sqrt(self.dk)

        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        # A fully-masked row (all-padding key set) softmaxes to uniform garbage;
        # zero it explicitly so it cannot contribute through the residual.
        attn = attn.masked_fill(~mask.any(-1, keepdim=True), 0.0)

        ctx = torch.matmul(attn, v).transpose(1, 2).reshape(b, t, self.h * self.dk)
        return self.out(ctx)


class ConvolutionModule(nn.Module):
    """Pointwise→GLU→depthwise→norm→SiLU→pointwise. The local-context half."""

    def __init__(self, channels: int, kernel_size: int = 31, causal: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "odd kernel needed for symmetric padding"
        self.norm = nn.LayerNorm(channels)
        self.pointwise1 = nn.Conv1d(channels, 2 * channels, 1)
        self.depthwise = nn.Conv1d(channels, channels, kernel_size, groups=channels)
        self.batch_norm = nn.BatchNorm1d(channels)
        self.pointwise2 = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.causal = causal
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """x: (B,T,C); pad_mask: (B,1,T) True where padding."""
        x = self.norm(x).transpose(1, 2)                  # (B,C,T)
        # Zero the padding BEFORE convolving: otherwise garbage in the padded
        # tail smears left into real frames through the depthwise kernel.
        x = x.masked_fill(pad_mask, 0.0)
        x = F.glu(self.pointwise1(x), dim=1)
        pad = (self.kernel_size - 1, 0) if self.causal else \
              ((self.kernel_size - 1) // 2,) * 2
        x = self.depthwise(F.pad(x, pad))
        x = F.silu(self.batch_norm(x))
        x = self.dropout(self.pointwise2(x))
        return x.masked_fill(pad_mask, 0.0).transpose(1, 2)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_expansion: int = 4,
                 kernel_size: int = 31, causal_conv: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.ff1 = FeedForward(d_model, ff_expansion, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = RelPositionMultiHeadAttention(d_model, n_heads, dropout)
        self.attn_drop = nn.Dropout(dropout)
        self.conv = ConvolutionModule(d_model, kernel_size, causal_conv, dropout)
        self.ff2 = FeedForward(d_model, ff_expansion, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, attn_mask, pad_mask):
        x = x + 0.5 * self.ff1(x)
        x = x + self.attn_drop(self.attn(self.attn_norm(x), pos_emb, attn_mask))
        x = x + self.conv(x, pad_mask)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
class ConformerEncoder(nn.Module):
    """log-mel (B,T,80) -> contextual encodings (B,T/4,d_model)."""

    def __init__(self, n_mels: int = 80, d_model: int = 256, n_layers: int = 12,
                 n_heads: int = 4, ff_expansion: int = 4, kernel_size: int = 31,
                 causal_conv: bool = True, dropout: float = 0.1):
        super().__init__()
        self.subsampling = ConvSubsampling(n_mels, d_model, dropout)
        self.pos_enc = RelPositionalEncoding(d_model)
        self.blocks = nn.ModuleList([
            ConformerBlock(d_model, n_heads, ff_expansion, kernel_size,
                           causal_conv, dropout)
            for _ in range(n_layers)
        ])
        self.d_model = d_model

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor,
                chunk_size: int = 0, left_chunks: int = -1):
        """feats: (B,T,n_mels), lengths: (B,). Returns (out, out_lengths).

        chunk_size is in *subsampled* frames (1 frame = 40 ms). 0 = offline.
        """
        x = self.subsampling(feats)
        out_lengths = self.subsampling.out_length(lengths).clamp(min=1)
        t = x.size(1)

        pad = make_pad_mask(out_lengths, t)                     # (B,T) True=pad
        valid = (~pad).unsqueeze(1)                             # (B,1,T)
        chunk = make_chunk_mask(t, chunk_size, left_chunks, x.device)
        # allowed(b,q,k) = key is real AND within the chunk window
        attn_mask = valid.unsqueeze(1) & chunk.unsqueeze(0).unsqueeze(0)

        pos_emb = self.pos_enc(x)
        x = x.masked_fill(pad.unsqueeze(-1), 0.0)
        for block in self.blocks:
            x = block(x, pos_emb, attn_mask, pad.unsqueeze(1))
        return x.masked_fill(pad.unsqueeze(-1), 0.0), out_lengths
