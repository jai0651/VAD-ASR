"""
Module 9, part 4: scoring two voices, and the number that says how well.

Once you have an encoder, verification is embarrassingly simple: embed both
clips, L2-normalise, take the dot product. Everything interesting is in how you
*evaluate* it.

WHY NOT ACCURACY. "Same speaker or not" has a threshold in it, and the useful
question is what happens across ALL thresholds, because the threshold you ship
depends on what a mistake costs. Two mistakes exist and they trade off:

    FALSE REJECT (miss)   two clips of the same person, scored below threshold
    FALSE ACCEPT (alarm)  two different people, scored above threshold

Raise the threshold and misses go up while false accepts go down. The EQUAL
ERROR RATE is the operating point where the two rates cross — one number for the
whole curve, threshold-free, and the standard in the speaker literature. 5% EER
means: at the balanced setting, 5% of same-speaker pairs are rejected and 5% of
impostor pairs are accepted.

EER IS NOT THE OPERATING POINT YOU DEPLOY. Real systems are wildly imbalanced —
in diarization, the vast majority of candidate pairs are different speakers, so
a false-accept is much more likely and much more damaging than the EER suggests.
minDCF fixes a prior on target trials (p_target = 0.01 here) and reports the best
achievable *cost*, which is why every paper reports both. When they disagree,
believe minDCF for a deployment decision.

THE TRIAL LIST IS PART OF THE MEASUREMENT. An EER is meaningless without saying
which pairs were scored. `make_trials` builds a balanced list from held-out
speakers with a fixed seed, so the number is comparable across runs.
"""

from __future__ import annotations

import random
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

from src.asr.corpus import LogMel
from src.speaker.data import SR, normalize_feats


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------
def naive_stats_embedding(feats: torch.Tensor, use_std: bool = True) -> torch.Tensor:
    """The zero-training baseline: mean (and std) of the log-mel over time.

    This is stats pooling with no network in front of it. It is not a strawman —
    the long-term average spectrum really does encode vocal-tract shape, so it
    scores far better than chance, and the gap between it and the trained
    encoder in script 20/17 is exactly what the network bought you.
    """
    if not use_std:
        return feats.mean(0)
    return torch.cat([feats.mean(0), feats.std(0)])


@torch.no_grad()
def embed_file(model, path: str, device, logmel: LogMel | None = None,
               norm: str = "mean", max_seconds: float | None = None
               ) -> torch.Tensor:
    """One file -> one L2-normalised embedding (D,).

    Deliberately one utterance at a time, with no padding: a padded batch pushes
    zeros through the convolutions and shifts the embedding slightly. That is
    tolerable in training and not worth the risk in the number you report.
    """
    logmel = logmel or LogMel()
    wav, sr = sf.read(path, dtype="float32")
    assert sr == SR, f"expected {SR} Hz, got {sr}"
    if max_seconds is not None and len(wav) > int(max_seconds * SR):
        wav = wav[: int(max_seconds * SR)]
    feats = normalize_feats(logmel(torch.from_numpy(np.ascontiguousarray(wav))), norm)
    if model is None:                                  # naive baseline path
        emb = naive_stats_embedding(feats)
    else:
        emb = model(feats.unsqueeze(0).to(device)).squeeze(0).cpu()
    return torch.nn.functional.normalize(emb, dim=0)


@torch.no_grad()
def embed_items(model, items: list[dict], device, norm: str = "mean",
                max_seconds: float | None = 12.0, n_mels: int = 80,
                progress: bool = False) -> dict[str, torch.Tensor]:
    """{path: unit-norm embedding} for a manifest. `model=None` -> baseline."""
    logmel = LogMel(n_mels=n_mels)
    was_training = model.training if model is not None else False
    if model is not None:
        model.eval()
    out = {}
    for i, it in enumerate(items):
        out[it["path"]] = embed_file(model, it["path"], device, logmel, norm, max_seconds)
        if progress and i % 100 == 0:
            print(f"  embedded {i}/{len(items)}", flush=True)
    if model is not None and was_training:
        model.train()
    return out


# ---------------------------------------------------------------------------
# Trials
# ---------------------------------------------------------------------------
def make_trials(items: list[dict], n_pairs: int = 4000, seed: int = 0,
                verbose: bool = True) -> list[tuple[str, str, int]]:
    """A balanced, de-duplicated list of (path_a, path_b, is_same_speaker).

    Same-speaker pairs are the scarce resource: a set with S speakers and n
    utterances each contains only S * C(n, 2) of them, and that runs out long
    before impostor pairs do. If the request cannot be met we shrink the list
    and SAY SO, rather than topping it up with impostors — a trial list that is
    quietly 45% target instead of 50% still produces a confident-looking EER,
    and nothing downstream can detect it.
    """
    by_spk: dict[str, list[str]] = defaultdict(list)
    for it in items:
        by_spk[it["speaker"]].append(it["path"])
    multi = sorted(s for s, v in by_spk.items() if len(v) >= 2)
    all_spk = sorted(by_spk)
    if len(all_spk) < 2:
        raise ValueError("need at least 2 speakers to build trials")

    max_targets = sum(len(v) * (len(v) - 1) // 2 for v in by_spk.values())
    n_target = min(n_pairs // 2, max_targets)
    if n_target < n_pairs // 2 and verbose:
        print(f"make_trials: only {max_targets} distinct same-speaker pairs exist "
              f"across {len(multi)} speakers; returning {2 * n_target} balanced "
              f"trials instead of the {n_pairs} requested")

    rng = random.Random(seed)
    seen: set[tuple[str, str]] = set()
    trials: list[tuple[str, str, int]] = []

    def add(a: str, b: str, same: int) -> bool:
        key = (a, b) if a < b else (b, a)
        if key in seen:
            return False
        seen.add(key)
        trials.append((a, b, same))
        return True

    attempts = 0
    while len(trials) < n_target and attempts < max(1000, n_target * 200) and multi:
        attempts += 1
        paths = by_spk[rng.choice(multi)]
        a, b = rng.sample(paths, 2)
        add(a, b, 1)
    n_target = len(trials)          # in case the sampler fell short anyway

    attempts = 0
    while len(trials) < 2 * n_target and attempts < max(1000, n_target * 200):
        attempts += 1
        s1, s2 = rng.sample(all_spk, 2)
        add(rng.choice(by_spk[s1]), rng.choice(by_spk[s2]), 0)
    return trials


def score_trials(embs: dict[str, torch.Tensor], trials: list[tuple[str, str, int]]
                 ) -> tuple[np.ndarray, np.ndarray]:
    """-> (cosine scores, labels). Embeddings are already unit-norm."""
    a = torch.stack([embs[t[0]] for t in trials])
    b = torch.stack([embs[t[1]] for t in trials])
    scores = (a * b).sum(dim=1).numpy()
    labels = np.array([t[2] for t in trials], dtype=np.int64)
    return scores, labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def det_curve(scores: np.ndarray, labels: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (thresholds, false-reject rate, false-accept rate), all same length."""
    scores = np.asarray(scores, dtype=np.float64)
    tgt = np.sort(scores[np.asarray(labels) == 1])
    non = np.sort(scores[np.asarray(labels) == 0])
    if len(tgt) == 0 or len(non) == 0:
        raise ValueError("trials must contain both target and non-target pairs")
    # Sweep every distinct score, PLUS one threshold above the maximum so that
    # "reject everything" is an available operating point. Without it minDCF can
    # come out above 1.0, which is nonsense: the normaliser is defined so that
    # the trivial reject-everything system scores exactly 1.0.
    thr = np.append(np.unique(scores), np.max(scores) + 1.0)
    # accept iff score >= threshold
    frr = np.searchsorted(tgt, thr, side="left") / len(tgt)
    far = (len(non) - np.searchsorted(non, thr, side="left")) / len(non)
    return thr, frr, far


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """-> (EER as a fraction, the threshold that achieves it)."""
    thr, frr, far = det_curve(scores, labels)
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2.0), float(thr[i])


def min_dcf(scores: np.ndarray, labels: np.ndarray, p_target: float = 0.01,
            c_miss: float = 1.0, c_fa: float = 1.0) -> float:
    """Normalised minimum detection cost at a fixed target prior."""
    _, frr, far = det_curve(scores, labels)
    dcf = c_miss * p_target * frr + c_fa * (1.0 - p_target) * far
    return float(dcf.min() / min(c_miss * p_target, c_fa * (1.0 - p_target)))


def summarize(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    """EER, minDCF, and the mean same/different cosines behind them."""
    eer, thr = compute_eer(scores, labels)
    same = scores[np.asarray(labels) == 1]
    diff = scores[np.asarray(labels) == 0]
    return {
        "eer": eer,
        "threshold": thr,
        "min_dcf": min_dcf(scores, labels),
        "mean_same": float(same.mean()),
        "mean_diff": float(diff.mean()),
        "separation": float(same.mean() - diff.mean()),
    }


def bootstrap_eer(scores: np.ndarray, labels: np.ndarray, n_boot: int = 500,
                  seed: int = 0, ci: float = 0.95) -> tuple[float, float, float]:
    """-> (EER, low, high) by resampling the trial list with replacement.

    Read this before believing any single EER in this module. With 8 held-out
    speakers the trial list is small AND its entries are not independent — the
    same utterance appears in many pairs — so a difference of a point or two
    between two runs is usually noise. This interval is if anything an
    UNDER-estimate of the true uncertainty, because resampling trials does not
    resample the speakers, and speaker identity is the real unit of variation.

    The practical rule: if two systems' intervals overlap, you have not shown
    that one is better, and the fix is more held-out speakers, not more seeds.
    """
    rng = np.random.default_rng(seed)
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n = len(scores)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if labels[idx].min() == labels[idx].max():
            continue                        # degenerate resample, skip
        boots.append(compute_eer(scores[idx], labels[idx])[0])
    lo = float(np.quantile(boots, (1 - ci) / 2))
    hi = float(np.quantile(boots, 1 - (1 - ci) / 2))
    return compute_eer(scores, labels)[0], lo, hi
