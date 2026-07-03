"""
Module 2, part 4: turning per-frame distributions into text.

GREEDY DECODING
Take the most likely symbol at each frame, then apply the CTC collapse rule
(merge repeats, drop blanks). Fast, simple, and usually fine once the model is
confident. Its weakness: it commits to one alignment, ignoring that many frame
paths lead to the same text.

CTC PREFIX BEAM SEARCH
The "right" decoding sums probability over ALL frame paths that produce a given
text, and keeps the top-k most promising text prefixes as it sweeps left to
right. For each prefix we track two probabilities: ending in blank (p_b) vs.
ending in a real symbol (p_nb). The blank/non-blank split is what correctly
handles repeated characters (the same trick the blank plays in training).
No language model here — just the acoustic probabilities.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

from src.asr.text import BLANK, collapse, decode_labels


def greedy_decode(log_probs: torch.Tensor) -> str:
    """log_probs: (T, vocab) for ONE utterance -> text."""
    best = log_probs.argmax(dim=-1).tolist()  # most likely symbol per frame
    return decode_labels(collapse(best))


def beam_search_decode(
    log_probs: torch.Tensor, beam_width: int = 16
) -> str:
    """CTC prefix beam search (no language model). log_probs: (T, vocab)."""
    probs = log_probs.exp().cpu().numpy()  # work in linear prob space
    T, V = probs.shape

    # Each beam is a prefix (tuple of label ids). For each we keep p_b (prob the
    # path so far ends in blank) and p_nb (ends in a non-blank symbol).
    # Start: empty prefix, all probability in "ends in blank".
    beams: dict[tuple, tuple[float, float]] = {(): (1.0, 0.0)}

    for t in range(T):
        next_beams: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
        for prefix, (p_b, p_nb) in beams.items():
            for s in range(V):
                p = probs[t, s]
                if p < 1e-8:
                    continue
                if s == BLANK:
                    # Staying blank: prefix unchanged, accumulate into its p_b.
                    nb, nnb = next_beams[prefix]
                    next_beams[prefix] = [nb + (p_b + p_nb) * p, nnb]
                else:
                    last = prefix[-1] if prefix else None
                    if s == last:
                        # Repeat of last symbol: extending from a blank-ending
                        # path creates a NEW doubled symbol; extending from a
                        # non-blank path just lengthens the same symbol's run.
                        nb, nnb = next_beams[prefix]
                        next_beams[prefix] = [nb, nnb + p_nb * p]
                        new_prefix = prefix + (s,)
                        nb2, nnb2 = next_beams[new_prefix]
                        next_beams[new_prefix] = [nb2, nnb2 + p_b * p]
                    else:
                        new_prefix = prefix + (s,)
                        nb2, nnb2 = next_beams[new_prefix]
                        next_beams[new_prefix] = [nb2, nnb2 + (p_b + p_nb) * p]

        # Keep the top `beam_width` prefixes by total probability.
        scored = sorted(
            next_beams.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True
        )[:beam_width]
        beams = {prefix: (vals[0], vals[1]) for prefix, vals in scored}

    best_prefix = max(beams.items(), key=lambda kv: kv[1][0] + kv[1][1])[0]
    return decode_labels(list(best_prefix))


def char_error_rate(pred: str, target: str) -> float:
    """Levenshtein distance / target length — the standard ASR character metric."""
    m, n = len(pred), len(target)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if pred[i - 1] == target[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n] / max(1, n)
