"""
Module 9, step 1: is a voice even visible in a spectrogram? Measure it first.

Before training anything, we build the crudest possible voiceprint — the mean
and standard deviation of the log-mel over a whole utterance, with no network at
all — and score it on real trials from held-out speakers. This is statistics
pooling with nothing in front of it.

MEASURED ON dev-clean (8 held-out speakers, 4000 trials):

    raw log-mel mean          EER 10.90%   minDCF 0.597
    raw log-mel std           EER 14.85%   minDCF 0.742
    both                      EER 11.15%   minDCF 0.502

Four things fall out of those three lines, and every one of them shapes the
trained model.

  1. A FLOOR. ~11% EER, from an "embedding" that is 160 numbers of arithmetic.
     If the encoder in script 21 cannot beat this, the training is broken. It is
     very easy to write a speaker model that silently learns nothing and still
     shows a perfectly plausible loss curve.

  2. BOTH HALVES OF THE STATISTIC CARRY IDENTITY. The mean spectrum is roughly
     the vocal-tract shape; the std is how much the voice moves around the band —
     pitch range, dynamics, articulation. Each is usable alone. That is the
     argument for pooling STATISTICS rather than just averaging.

  3. THE MEAN IS A TRAP, AND THE STD IS THE HONEST HALF. Per-utterance mean
     normalisation (CMN) subtracts the long-term average spectrum, so the mean
     baseline is exactly chance after CMN — while the std baseline is bit-for-bit
     UNCHANGED by it (subtracting a per-band constant cannot change a standard
     deviation; the script asserts this). Which matters because in LibriSpeech
     each speaker is one person at one microphone in one room, so a big part of
     what the raw mean "recognises" is the recording, not the voice. It will not
     transfer to the same person on a different mic. This is why the trained
     model is fed mean-normalised features even though that visibly costs it the
     strongest zero-training cue.

  4. EER AND minDCF DISAGREE HERE, AND minDCF WINS. Adding the std barely moves
     EER (10.90 -> 11.15) but cuts minDCF hard (0.597 -> 0.502). minDCF is
     measured at a low false-alarm operating point — which is the regime
     diarization actually runs in, where almost every candidate pair is a
     different speaker. Reporting only EER would have hidden a real improvement.

The PCA panel shows same-speaker points already clumping — the signal genuinely
is in the raw features — while the histogram shows the same/different score
distributions still heavily overlapping. Closing that overlap is the job of
AAM-softmax in script 21.

(The histogram uses CENTERED embeddings: raw log-mel values are all large and
negative, so every embedding points into the same corner and cosine saturates
above 0.96 for any pair whatsoever. Subtracting the dataset mean embedding —
the cheap ancestor of LDA/PLDA in i-vector systems — spreads the scores from a
0.03 gap to a 0.73 one. It does not improve the ranking, it makes it legible.
The trained model gets this for free from the BatchNorm on its embedding.)

Run:
  uv run python scripts/20_speaker_intuition.py
  SPLIT=train-clean-100 DOWNLOAD=1 uv run python scripts/20_speaker_intuition.py

Env knobs: SPLIT DOWNLOAD HELD_OUT TRIALS MAX_UTTS SEED
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import soundfile as sf
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.corpus import LogMel  # noqa: E402
from src.speaker.data import (  # noqa: E402
    build_speaker_manifest,
    normalize_feats,
    split_by_speaker,
)
from src.speaker.verify import make_trials, score_trials, summarize  # noqa: E402

OUT = Path("outputs")
SPLIT = os.environ.get("SPLIT", "dev-clean")
DOWNLOAD = os.environ.get("DOWNLOAD", "0") == "1"
HELD_OUT = int(os.environ.get("HELD_OUT", "8"))
TRIALS = int(os.environ.get("TRIALS", "4000"))
MAX_UTTS = int(os.environ.get("MAX_UTTS", "400"))
SEED = int(os.environ.get("SEED", "0"))


def unit(embs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {k: torch.nn.functional.normalize(v, dim=0) for k, v in embs.items()}


def centered(embs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Subtract the dataset mean embedding, then re-normalise."""
    mu = torch.stack(list(embs.values())).mean(0)
    return unit({k: v - mu for k, v in embs.items()})


def main() -> None:
    items = build_speaker_manifest(split=SPLIT, download=DOWNLOAD)
    _, trial_items = split_by_speaker(items, n_held_out=HELD_OUT, seed=SEED)
    if MAX_UTTS and len(trial_items) > MAX_UTTS:
        trial_items = trial_items[:: len(trial_items) // MAX_UTTS + 1]
    speakers = sorted({it["speaker"] for it in trial_items})
    print(f"split {SPLIT}: {len(items)} utterances, "
          f"{len({i['speaker'] for i in items})} speakers")
    print(f"held out for trials: {len(speakers)} speakers, "
          f"{len(trial_items)} utterances (never trained on)\n")

    logmel = LogMel()
    raw: dict[str, torch.Tensor] = {}
    cmn: dict[str, torch.Tensor] = {}
    for it in trial_items:
        wav, sr = sf.read(it["path"], dtype="float32")
        feats = logmel(torch.from_numpy(np.ascontiguousarray(wav)))
        raw[it["path"]] = feats
        cmn[it["path"]] = normalize_feats(feats, norm="mean")

    # Point 3, asserted rather than claimed: CMN cannot change a std.
    k = next(iter(raw))
    drift = (raw[k].std(0) - cmn[k].std(0)).abs().max().item()
    assert drift < 1e-4, drift
    print(f"check: the std over time is invariant to mean normalisation "
          f"(max drift {drift:.2e})")
    print("       ...so the std is the channel-robust half of the statistic.\n")

    variants = {
        "mean only":            unit({k: f.mean(0) for k, f in raw.items()}),
        "std only":             unit({k: f.std(0) for k, f in raw.items()}),
        "mean + std":           unit({k: torch.cat([f.mean(0), f.std(0)])
                                      for k, f in raw.items()}),
        "mean + std, centered": centered({k: torch.cat([f.mean(0), f.std(0)])
                                          for k, f in raw.items()}),
        "mean only, after CMN": unit({k: f.mean(0) + 1e-8 * torch.randn_like(f.mean(0))
                                      for k, f in cmn.items()}),
    }

    trials = make_trials(trial_items, n_pairs=TRIALS, seed=SEED)
    n_same = sum(t[2] for t in trials)
    print(f"{len(trials)} trials ({n_same} same-speaker, {len(trials)-n_same} impostor)\n")

    results = {}
    for name, embs in variants.items():
        scores, labels = score_trials(embs, trials)
        results[name] = (summarize(scores, labels), scores, labels)
        r = results[name][0]
        print(f"log-mel {name:22s}  EER {r['eer']*100:5.2f}%   minDCF {r['min_dcf']:.3f}"
              f"   cos same {r['mean_same']:+.3f} / diff {r['mean_diff']:+.3f}")

    mean_r = results["mean only"][0]
    std_r = results["std only"][0]
    both_r = results["mean + std"][0]
    cmn_r = results["mean only, after CMN"][0]
    ctr_r = results["mean + std, centered"][0]

    print(f"\nboth halves work alone: mean {mean_r['eer']*100:.2f}% EER, "
          f"std {std_r['eer']*100:.2f}% EER  -> pool STATISTICS, not just an average.")
    print(f"the mean is channel: after CMN it collapses to "
          f"{cmn_r['eer']*100:.1f}% EER (chance), while the std is untouched.")
    print(f"EER vs minDCF disagree: adding the std moves EER "
          f"{mean_r['eer']*100:.2f} -> {both_r['eer']*100:.2f}% (worse) but minDCF "
          f"{mean_r['min_dcf']:.3f} -> {both_r['min_dcf']:.3f} (better).")
    print("   minDCF is the low-false-alarm regime, which is where diarization lives.")
    print(f"centering: EER {both_r['eer']*100:.2f} -> {ctr_r['eer']*100:.2f}%, but the "
          f"score gap opens {both_r['separation']:.3f} -> {ctr_r['separation']:.3f} "
          f"— legible, not better.\n")
    print(f"FLOOR TO BEAT in script 21: {both_r['eer']*100:.2f}% EER / "
          f"{both_r['min_dcf']:.3f} minDCF, with zero training.\n")

    # ---------------------------------------------------------------- plots
    _, scores, labels = results["mean + std, centered"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    ax = axes[0]
    ax.hist(scores[labels == 0], bins=60, alpha=0.65, label="different speakers",
            color="#c44e52", density=True)
    ax.hist(scores[labels == 1], bins=60, alpha=0.65, label="same speaker",
            color="#4c72b0", density=True)
    ax.axvline(ctr_r["threshold"], color="k", ls="--", lw=1,
               label=f"EER threshold ({ctr_r['eer']*100:.1f}% EER)")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title("untrained baseline: the distributions still overlap badly")
    ax.legend(fontsize=8)

    ax = axes[1]
    paths = [it["path"] for it in trial_items]
    spk = [it["speaker"] for it in trial_items]
    X = torch.stack([variants["mean + std, centered"][p] for p in paths])
    X = X - X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(X, q=2)
    P = (X @ V[:, :2]).numpy()
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(speakers):
        m = np.array([x == s for x in spk])
        ax.scatter(P[m, 0], P[m, 1], s=12, alpha=0.75, color=cmap(i % 10), label=s)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("same colour = same speaker (no training at all)")
    ax.legend(fontsize=7, ncol=2, title="speaker")

    fig.suptitle("Module 9: a voice is already visible in the raw log-mel — "
                 "just not cleanly enough to threshold")
    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / "20_speaker_naive.png", dpi=130)
    print(f"wrote {OUT / '20_speaker_naive.png'}")


if __name__ == "__main__":
    main()
