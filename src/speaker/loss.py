"""
Module 9, part 3: why a plain classifier gives you a bad voiceprint.

The obvious training recipe is: softmax over the N training speakers,
cross-entropy, then throw the classifier away and keep the penultimate layer.
That works — x-vectors were trained exactly like this — but it optimises the
wrong thing, and the reason is worth internalising because it generalises far
beyond speech.

  WHAT SOFTMAX ASKS FOR:  "put speaker A on a different side of a hyperplane
                           than speaker B."
  WHAT YOU ACTUALLY NEED: "put any two clips of A closer together than any clip
                           of A is to any clip of B — including for speakers
                           that were never in the training set."

The first is satisfied the instant the classes are linearly separable. Nothing
in it rewards making a class *tight*, or leaving a *gap* between classes. So the
learned space is separable but not metric, and cosine distance in it — which is
what you deploy — is mediocre.

THE FIX, IN THREE STEPS (ArcFace / AAM-softmax, Deng 2019):

  1. L2-normalise both the embedding and each class weight vector. Now the
     logit is exactly cos(theta) between an embedding and its class centre, so
     the training objective is expressed in the same geometry as the test-time
     metric. Length can no longer be used to win, only direction.

  2. Subtract an angular MARGIN from the target class only: use cos(theta + m)
     in place of cos(theta) for the true speaker. To score well the embedding
     must now beat the other classes by m radians of slack, not by an epsilon.
     That slack is precisely the "gap" plain softmax never asks for, and it is
     what makes the space work for speakers the model has never seen.

  3. Multiply everything by a SCALE s. Cosines live in [-1, 1]; a softmax over
     values that small is nearly uniform and produces almost no gradient. s ~ 30
     restores a usable dynamic range. Steps 2 and 3 are a package — the margin
     does nothing without the scale.

WHAT THIS REPO ACTUALLY MEASURED, WHICH IS LESS THAN THE ABOVE CLAIMS. On
dev-clean (32 training speakers, 8 held out) both losses land in the same place
and the difference is smaller than the run-to-run noise — see script 21's
confidence interval, which is several points wide because 8 speakers is not
enough to resolve it. Everything above is the literature's reasoning and it is
why every current speaker system uses an angular margin; it is NOT something
3 hours of read audiobook speech can demonstrate. The A/B becomes meaningful on
train-clean-100 (251 speakers) and obvious on VoxCeleb2 (~6000). Reporting the
theory as though this repo had confirmed it would be exactly the sort of thing
the rest of these docs exist to avoid.

MARGIN WARMUP IS NOT OPTIONAL. At step 0 the embeddings are random, every angle
is near 90 degrees, and demanding an extra 0.2 rad of margin on top makes the
target logit smaller than the impostor logits. The model's cheapest escape is to
collapse all embeddings together, and runs that start at full margin routinely
stall. Ramp m from 0 to its final value over the first ~30% of training
(`margin_schedule`) and the problem disappears. This is the same failure shape
as the CTC blank collapse in Module 6: a regulariser applied before the model
can do the task at all.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class PlainSoftmax(nn.Module):
    """The baseline: an ordinary linear classifier over training speakers."""

    def __init__(self, embed_dim: int, n_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, n_classes)

    def set_margin(self, margin: float) -> None:   # same interface, no-op
        pass

    def forward(self, emb: torch.Tensor, labels: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.fc(emb)
        return F.cross_entropy(logits, labels), logits


class AAMSoftmax(nn.Module):
    """Additive angular margin softmax (ArcFace).

    Returns (loss, cosines). The second value is the *unmargined* cosine matrix
    — the thing to compute training accuracy from, since it is what inference
    will see.
    """

    def __init__(self, embed_dim: int, n_classes: int, margin: float = 0.2,
                 scale: float = 30.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_classes, embed_dim))
        nn.init.xavier_normal_(self.weight)
        self.scale = scale
        self.set_margin(margin)

    def set_margin(self, margin: float) -> None:
        """Call every step to ramp the margin (see `margin_schedule`)."""
        self.margin = float(margin)
        self.cos_m = math.cos(self.margin)
        self.sin_m = math.sin(self.margin)
        # cos(theta + m) stops being monotone in theta once theta + m > pi.
        # Past that point fall back to a linear penalty, or gradients flip sign
        # on exactly the hardest examples.
        self.th = math.cos(math.pi - self.margin)
        self.mm = math.sin(math.pi - self.margin) * self.margin

    def forward(self, emb: torch.Tensor, labels: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = F.linear(F.normalize(emb), F.normalize(self.weight))   # (B, C)
        cos = cos.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        if self.margin <= 0.0:
            return F.cross_entropy(self.scale * cos, labels), cos

        sin = (1.0 - cos ** 2).clamp(min=0.0).sqrt()
        phi = cos * self.cos_m - sin * self.sin_m                    # cos(t + m)
        phi = torch.where(cos > self.th, phi, cos - self.mm)

        one_hot = torch.zeros_like(cos).scatter_(1, labels.view(-1, 1), 1.0)
        logits = self.scale * (one_hot * phi + (1.0 - one_hot) * cos)
        return F.cross_entropy(logits, labels), cos


def margin_schedule(step: int, total: int, max_margin: float = 0.2,
                    warm_frac: float = 0.3) -> float:
    """Linear ramp 0 -> max_margin over the first `warm_frac` of training."""
    warm = max(1, int(total * warm_frac))
    return max_margin * min(1.0, step / warm)


HEADS = {"aam": AAMSoftmax, "plain": PlainSoftmax}


def make_head(loss: str, embed_dim: int, n_classes: int, **kwargs) -> nn.Module:
    if loss == "plain":
        return PlainSoftmax(embed_dim, n_classes)
    if loss == "aam":
        return AAMSoftmax(embed_dim, n_classes, **kwargs)
    raise ValueError(f"unknown loss {loss!r}, expected one of {sorted(HEADS)}")
