"""
Module 9, part 2: the speaker encoder — variable-length audio to one vector.

THE ONE HARD PROBLEM. A recogniser can emit one output per frame; an identity
cannot. Two seconds and twenty seconds of the same person have to land on the
*same* point. So somewhere in the network, a (T, C) sequence must collapse to a
fixed (C',) vector, and how you collapse it is most of the accuracy.

  Average over time?  Throws away half the signal. The long-term MEAN of a
  spectrum is roughly the vocal-tract shape; the long-term VARIANCE is how much
  the voice moves around — pitch range, dynamics, articulation habits. Both are
  identity. So we pool the mean AND the standard deviation (Snyder's x-vector,
  2018). Script 16 measures the gap: std alone buys a large chunk of the EER.

  Weight the average?  Not every frame is equally speaker-y. A silent frame or a
  fricative burst carries far less identity than a voiced vowel. Attentive
  statistics pooling (Okabe, 2018) learns a per-frame, per-channel weight and
  takes the weighted mean and std. This is the "attention" of Module 4, applied
  along time with a single query.

WHAT ELSE IS IN HERE, AND WHY

  Dilated 1-D convolutions (TDNN). The classic frame-level trunk: dilation lets
  layer 3 see ~1 s of context for the cost of a kernel-3 conv, and identity cues
  like speaking rate only exist over that span.

  Res2 blocks. Split the channels into 4 groups and chain convolutions across
  them, so one block produces several effective receptive-field sizes at once.
  Formants live at one scale, prosody at another.

  Squeeze-Excitation. A global (whole-utterance) descriptor re-weights channels.
  In a speaker net this is where "this recording is bright / this one is muffled"
  gets normalised out.

  Multi-layer feature aggregation. Pool from the concatenation of ALL blocks,
  not just the last one. Lower layers keep fine spectral detail that a deep
  trunk has already abstracted away, and that detail is timbre.

Together the last three are ECAPA-TDNN (Desplanques, 2020), still the backbone
under most production speaker systems. `ECAPALite` is that architecture at a
size a laptop can train; `XVectorTDNN` is the 2018 baseline it replaced, kept so
the comparison is one env var away.

ALL MASKING IS PLUMBED THROUGH. Training uses fixed 2 s crops and never needs a
mask, but diarization will push variable-length windows through this net in
batches, and a pooling layer that averages over padding produces a subtly wrong
embedding — the kind of bug that shows up as "clustering is mediocre" rather
than as a crash.
"""

from __future__ import annotations

import torch
from torch import nn


def lengths_to_mask(lengths: torch.Tensor, t_max: int) -> torch.Tensor:
    """(B,) frame counts -> (B, 1, T) float mask, 1.0 on real frames."""
    idx = torch.arange(t_max, device=lengths.device).unsqueeze(0)
    return (idx < lengths.unsqueeze(1)).unsqueeze(1).to(torch.float32)


def masked_stats(x: torch.Tensor, mask: torch.Tensor
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """x (B,C,T), mask (B,1,T) -> mean, std over valid frames, each (B,C,1)."""
    n = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
    mean = (x * mask).sum(dim=2, keepdim=True) / n
    var = (((x - mean) ** 2) * mask).sum(dim=2, keepdim=True) / n
    return mean, var.clamp(min=1e-8).sqrt()


# ---------------------------------------------------------------------------
# Pooling: the layer that turns a sequence into an identity
# ---------------------------------------------------------------------------
class StatsPool(nn.Module):
    """Unweighted mean + std over time. The 2018 x-vector pooling."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.size(0), 1, x.size(2))
        mean, std = masked_stats(x, mask)
        return torch.cat([mean, std], dim=1).squeeze(-1)      # (B, 2C)


class AttentiveStatsPool(nn.Module):
    """Learned per-frame, per-channel weights, then weighted mean + std.

    `global_context=True` feeds each frame the utterance mean and std alongside
    itself, so "is this frame informative?" can be judged relative to the
    recording rather than in absolute terms — which is what makes it robust
    across microphones.
    """

    def __init__(self, channels: int, attention_channels: int = 128,
                 global_context: bool = True):
        super().__init__()
        self.global_context = global_context
        in_ch = channels * 3 if global_context else channels
        self.tdnn = nn.Conv1d(in_ch, attention_channels, kernel_size=1)
        self.bn = nn.BatchNorm1d(attention_channels)
        self.conv = nn.Conv1d(attention_channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.size(0), 1, x.size(2))
        h = x
        if self.global_context:
            mean, std = masked_stats(x, mask)
            t = x.size(2)
            h = torch.cat([x, mean.expand(-1, -1, t), std.expand(-1, -1, t)], dim=1)
        a = self.conv(torch.tanh(self.bn(self.tdnn(h))))          # (B, C, T)
        a = a.masked_fill(mask == 0, float("-inf"))
        a = torch.softmax(a, dim=2)
        mean = (a * x).sum(dim=2)
        var = (a * x * x).sum(dim=2) - mean ** 2
        return torch.cat([mean, var.clamp(min=1e-8).sqrt()], dim=1)   # (B, 2C)


# ---------------------------------------------------------------------------
# ECAPA building blocks
# ---------------------------------------------------------------------------
class Res2Conv1d(nn.Module):
    """Split into `scale` groups; each group's conv also sees the previous
    group's output. One block, several effective receptive fields."""

    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 1,
                 scale: int = 4):
        super().__init__()
        assert channels % scale == 0, "channels must divide by scale"
        self.scale = scale
        width = channels // scale
        pad = dilation * (kernel_size - 1) // 2
        self.convs = nn.ModuleList(
            nn.Conv1d(width, width, kernel_size, dilation=dilation, padding=pad)
            for _ in range(scale - 1))
        self.bns = nn.ModuleList(nn.BatchNorm1d(width) for _ in range(scale - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = torch.chunk(x, self.scale, dim=1)
        out = [chunks[0]]                       # group 0 passes straight through
        y = None
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            inp = chunks[i + 1] if y is None else chunks[i + 1] + y
            y = bn(torch.relu(conv(inp)))
            out.append(y)
        return torch.cat(out, dim=1)


class SEBlock(nn.Module):
    """Squeeze the whole utterance to one number per channel, excite with it."""

    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.fc1 = nn.Conv1d(channels, bottleneck, kernel_size=1)
        self.fc2 = nn.Conv1d(bottleneck, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = x.new_ones(x.size(0), 1, x.size(2))
        n = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        s = (x * mask).sum(dim=2, keepdim=True) / n              # (B, C, 1)
        s = torch.sigmoid(self.fc2(torch.relu(self.fc1(s))))
        return x * s


class SERes2Block(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, dilation: int = 2,
                 scale: int = 4, se_bottleneck: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.res2 = Res2Conv1d(channels, kernel_size, dilation, scale)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SEBlock(channels, se_bottleneck)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        y = self.bn1(torch.relu(self.conv1(x)))
        y = self.res2(y)
        y = self.bn2(torch.relu(self.conv2(y)))
        return x + self.se(y, mask)


# ---------------------------------------------------------------------------
# The two encoders
# ---------------------------------------------------------------------------
class ECAPALite(nn.Module):
    """ECAPA-TDNN at laptop scale. feats (B, T, n_mels) -> embedding (B, D)."""

    def __init__(self, n_mels: int = 80, channels: int = 256, embed_dim: int = 192,
                 attention_channels: int = 128, scale: int = 4):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(channels)
        self.block1 = SERes2Block(channels, 3, dilation=2, scale=scale)
        self.block2 = SERes2Block(channels, 3, dilation=3, scale=scale)
        self.block3 = SERes2Block(channels, 3, dilation=4, scale=scale)
        mfa = 3 * channels
        self.mfa = nn.Conv1d(3 * channels, mfa, kernel_size=1)
        self.pool = AttentiveStatsPool(mfa, attention_channels)
        self.bn_pool = nn.BatchNorm1d(2 * mfa)
        self.fc = nn.Linear(2 * mfa, embed_dim)
        self.bn_emb = nn.BatchNorm1d(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor | None = None
                ) -> torch.Tensor:
        x = feats.transpose(1, 2)                                # (B, n_mels, T)
        mask = None if lengths is None else lengths_to_mask(lengths, x.size(2))
        x = self.bn1(torch.relu(self.conv1(x)))
        y1 = self.block1(x, mask)
        y2 = self.block2(y1, mask)
        y3 = self.block3(y2, mask)
        y = torch.relu(self.mfa(torch.cat([y1, y2, y3], dim=1)))  # aggregate ALL
        e = self.bn_pool(self.pool(y, mask))
        return self.bn_emb(self.fc(e))


class XVectorTDNN(nn.Module):
    """The 2018 baseline: plain dilated TDNN + unweighted stats pooling."""

    def __init__(self, n_mels: int = 80, channels: int = 384,
                 stats_channels: int = 1152, embed_dim: int = 192):
        super().__init__()

        def tdnn(i: int, o: int, k: int, d: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv1d(i, o, k, dilation=d, padding=d * (k - 1) // 2),
                nn.ReLU(), nn.BatchNorm1d(o))

        self.frames = nn.Sequential(
            tdnn(n_mels, channels, 5, 1),
            tdnn(channels, channels, 3, 2),
            tdnn(channels, channels, 3, 3),
            tdnn(channels, channels, 1, 1),
            tdnn(channels, stats_channels, 1, 1),
        )
        self.pool = StatsPool()
        self.fc = nn.Linear(2 * stats_channels, embed_dim)
        self.bn_emb = nn.BatchNorm1d(embed_dim)
        self.embed_dim = embed_dim

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor | None = None
                ) -> torch.Tensor:
        x = feats.transpose(1, 2)
        mask = None if lengths is None else lengths_to_mask(lengths, x.size(2))
        x = self.frames(x)
        return self.bn_emb(self.fc(self.pool(x, mask)))


ARCHS = {"ecapa": ECAPALite, "xvector": XVectorTDNN}


def make_speaker_net(arch: str = "ecapa", **kwargs) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}, expected one of {sorted(ARCHS)}")
    return ARCHS[arch](**kwargs)


def load_speaker_encoder(path: str = "outputs/speaker.pt", device="cpu"
                         ) -> tuple[nn.Module, dict]:
    """Rebuild a trained encoder from a script-21 checkpoint, ready to embed.

    The checkpoint carries its own architecture config, so nothing downstream
    has to remember which env vars the run was launched with. Returns
    (model in eval mode, the checkpoint metadata) — the metadata includes the
    feature normalisation the model was trained with, which the caller MUST
    reuse: embedding mean-normalised features with a model trained on raw ones
    silently produces garbage rather than an error.
    """
    ck = torch.load(path, map_location=device)
    model = make_speaker_net(ck.get("arch", "ecapa"),
                             n_mels=ck.get("n_mels", 80),
                             channels=ck.get("channels", 256),
                             embed_dim=ck.get("embed_dim", 192)).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    meta = {k: v for k, v in ck.items() if k != "model"}
    return model, meta
