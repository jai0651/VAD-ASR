"""
Module 6, part 3: decoding over subwords, and the metric that matters.

Module 2's beam search (src/asr/decode.py) works in LINEAR probability space
over 28 characters. Neither survives the move to a real recognizer:

  UNDERFLOW. A 400-frame utterance multiplies 400 probabilities together. In
  float64 that underflows to exactly 0.0 somewhere around frame 300, and every
  hypothesis ties at zero. Everything here is in LOG space, with a numerically
  safe log-add.

  VOCABULARY SIZE. Looping all 512 subwords for every beam at every frame is
  ~2M Python operations per utterance. But CTC output is extremely peaked —
  almost all mass sits on blank plus a handful of tokens — so we prune to the
  top-k tokens per frame first. This is what every production CTC decoder does,
  and it costs essentially no accuracy.

WORD error rate replaces character error rate as the headline number, because
that is what the literature reports and what a user experiences. CER flatters a
model: "recogniton" is 1/10 character errors but 1/1 word errors.
"""

from __future__ import annotations

import math
from collections import defaultdict

import torch

NEG_INF = -float("inf")


def log_add(a: float, b: float) -> float:
    """log(e^a + e^b), overflow-free."""
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


def ctc_prefix_beam_search(
    log_probs: torch.Tensor,
    beam_size: int = 10,
    blank_id: int = 0,
    topk: int = 10,
) -> list[tuple[tuple[int, ...], float]]:
    """log_probs: (T, V) for ONE utterance. Returns n-best (tokens, log-score).

    Same recurrence as Module 2 — per prefix, track the probability of paths
    ending in blank (p_b) and ending in a real symbol (p_nb), because that
    distinction is what makes a repeated token either a run or a genuine
    doubling — but in log space, pruned, and over subword ids.
    """
    t_max, _ = log_probs.shape
    top_lp, top_idx = log_probs.topk(min(topk, log_probs.size(-1)), dim=-1)
    top_lp, top_idx = top_lp.tolist(), top_idx.tolist()

    beams: dict[tuple, tuple[float, float]] = {(): (0.0, NEG_INF)}  # log p_b, p_nb
    for t in range(t_max):
        nxt: dict[tuple, list[float]] = defaultdict(lambda: [NEG_INF, NEG_INF])
        for prefix, (p_b, p_nb) in beams.items():
            p_total = log_add(p_b, p_nb)
            last = prefix[-1] if prefix else None
            for lp, s in zip(top_lp[t], top_idx[t]):
                if s == blank_id:
                    e = nxt[prefix]
                    e[0] = log_add(e[0], p_total + lp)
                elif s == last:
                    # Extending a blank-ended path writes a NEW (doubled) token;
                    # extending a symbol-ended path just lengthens its run.
                    e = nxt[prefix]
                    e[1] = log_add(e[1], p_nb + lp)
                    e2 = nxt[prefix + (s,)]
                    e2[1] = log_add(e2[1], p_b + lp)
                else:
                    e2 = nxt[prefix + (s,)]
                    e2[1] = log_add(e2[1], p_total + lp)
        beams = dict(
            sorted(nxt.items(), key=lambda kv: log_add(*kv[1]), reverse=True)[:beam_size]
        )
        beams = {k: (v[0], v[1]) for k, v in beams.items()}

    scored = [(p, log_add(b, nb)) for p, (b, nb) in beams.items()]
    return sorted(scored, key=lambda kv: kv[1], reverse=True)


def ctc_greedy(log_probs: torch.Tensor, blank_id: int = 0) -> list[int]:
    """Argmax per frame, then the CTC collapse rule (merge repeats, drop blank)."""
    best = log_probs.argmax(dim=-1).tolist()
    out, prev = [], None
    for x in best:
        if x != prev and x != blank_id:
            out.append(x)
        prev = x
    return out


# ---------------------------------------------------------------------------
def _levenshtein(a: list, b: list) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def word_error_rate(pred: str, target: str) -> float:
    """Edit distance over WORDS / reference word count."""
    ref = target.split()
    return _levenshtein(pred.split(), ref) / max(1, len(ref))


def corpus_wer(pairs) -> float:
    """Aggregate WER = total errors / total reference words.

    NOT the mean of per-utterance WERs: that weights a 2-word utterance the
    same as a 30-word one and is the single most common way published WER
    numbers become incomparable.
    """
    errors = words = 0
    for pred, target in pairs:
        ref = target.split()
        errors += _levenshtein(pred.split(), ref)
        words += len(ref)
    return errors / max(1, words)
