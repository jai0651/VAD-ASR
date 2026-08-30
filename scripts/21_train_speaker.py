"""
Module 9, step 2: train the voiceprint.

Reads LibriSpeech, ignores every transcript, and learns an encoder whose cosine
distance answers "same person?" for speakers it has never heard.

WHAT THIS SCRIPT IS ACTUALLY MEASURING. The training objective is a
classification over the training speakers, but the reported number is EER on a
disjoint set of HELD-OUT speakers, scored by cosine on a fixed trial list. Those
are different tasks on purpose: training accuracy going to 100% tells you the
model memorised 32 voices, and says nothing about whether the embedding space
generalises. Watch the two diverge — it is the clearest illustration in this
repo of why the loss you optimise is not the metric you ship.

THE COMPARISONS THIS SCRIPT EXISTS TO MAKE (one env var each):

    LOSS=plain   ordinary softmax        vs   LOSS=aam  angular margin
    ARCH=xvector 2018 TDNN baseline      vs   ARCH=ecapa  (both ~2.1M params)
    NOISE_PROB=0 clean                   vs   NOISE_PROB=0.4  augmented
    SPEED_AUG=0  no speed perturbation   vs   SPEED_AUG=1 (as NEW classes)

The floor to beat is script 20's untrained baseline, which this script
recomputes on the identical trials so the comparison is exact.

READ THE CONFIDENCE INTERVAL BEFORE BELIEVING ANY OF THOSE A/Bs. dev-clean
leaves only 8 held-out speakers, and an EER measured on 8 speakers has a
bootstrap CI several points wide — wider than most of the differences above.
The end-of-run report prints the interval, the median across evaluations, and
the best checkpoint separately, precisely so the three cannot be confused. On
this split the honest answer to "does the angular margin help?" is "this
experiment cannot tell you"; the comparison needs train-clean-100.

DEFAULTS ARE FOR A LAPTOP. dev-clean gives 32 trainable speakers, which is
nowhere near enough — speaker encoders are hungry for SPEAKERS more than for
hours, and production systems train on 6000+ (VoxCeleb2). 32 speakers will
overfit and the held-out EER will plateau early. That is the honest result, and
the fix is the same as Module 6's: point it at more data.

Run:
  uv run python scripts/21_train_speaker.py                             # laptop smoke
  LOSS=plain uv run python scripts/21_train_speaker.py                  # the A/B
  SPLIT=train-clean-100 DOWNLOAD=1 STEPS=30000 CHANNELS=512 \
      uv run python scripts/21_train_speaker.py                         # the real run

Env knobs: SPLIT DOWNLOAD STEPS ARCH LOSS MARGIN SCALE LR BATCH CROP_S
           NOISE_PROB SPEED_AUG CHANNELS EMBED_DIM HELD_OUT TRIALS
           EVAL_EVERY MAX_EVAL_UTTS DEVICE RESUME NUM_WORKERS SEED TAG
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.corpus import warmup_cosine  # noqa: E402
from src.asr.hybrid import count_parameters  # noqa: E402
from src.checkpoint import clear_resume, load_resume, save_resume  # noqa: E402
from src.speaker.data import (  # noqa: E402
    SpeakerCrops,
    build_speaker_manifest,
    split_by_speaker,
)
from src.speaker.loss import make_head, margin_schedule  # noqa: E402
from src.speaker.model import make_speaker_net  # noqa: E402
from src.speaker.verify import (  # noqa: E402
    bootstrap_eer,
    embed_items,
    make_trials,
    score_trials,
    summarize,
)

OUT = Path("outputs")

SPLIT = os.environ.get("SPLIT", "dev-clean")
DOWNLOAD = os.environ.get("DOWNLOAD", "0") == "1"
STEPS = int(os.environ.get("STEPS", "3000"))
ARCH = os.environ.get("ARCH", "ecapa")
LOSS = os.environ.get("LOSS", "aam")
MARGIN = float(os.environ.get("MARGIN", "0.2"))
SCALE = float(os.environ.get("SCALE", "30.0"))
LR = float(os.environ.get("LR", "1e-3"))
BATCH = int(os.environ.get("BATCH", "64"))
CROP_S = float(os.environ.get("CROP_S", "2.0"))
NOISE_PROB = float(os.environ.get("NOISE_PROB", "0.4"))
SPEED_AUG = os.environ.get("SPEED_AUG", "1") == "1"
CHANNELS = int(os.environ.get("CHANNELS", "256"))
EMBED_DIM = int(os.environ.get("EMBED_DIM", "192"))
HELD_OUT = int(os.environ.get("HELD_OUT", "8"))
TRIALS = int(os.environ.get("TRIALS", "4000"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "500"))
MAX_EVAL_UTTS = int(os.environ.get("MAX_EVAL_UTTS", "300"))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", "2"))
RESUME = os.environ.get("RESUME", "1") == "1"
SEED = int(os.environ.get("SEED", "0"))
TAG = os.environ.get("TAG", "")
# Regenerate the report and figure from a saved checkpoint + history,
# without retraining. Useful after a plot label changes, and the only
# way to re-derive a figure for a run that cost GPU hours.
PLOT_ONLY = os.environ.get("PLOT_ONLY", "0") == "1"

NAME = f"speaker{('_' + TAG) if TAG else ''}"
CKPT_PATH = OUT / f"{NAME}.pt"
RESUME_PATH = OUT / f"{NAME}_resume.pt"


def get_device() -> torch.device:
    if "DEVICE" in os.environ:
        return torch.device(os.environ["DEVICE"])
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate(model, trial_items, trials, device) -> dict[str, float]:
    """Held-out-speaker EER/minDCF. Embeddings are mean-normalised, as trained."""
    embs = embed_items(model, trial_items, device, norm="mean")
    scores, labels = score_trials(embs, trials)
    return summarize(scores, labels)


def main() -> None:
    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(exist_ok=True)
    device = get_device()

    items = build_speaker_manifest(split=SPLIT, download=DOWNLOAD)
    train_items, trial_items = split_by_speaker(items, n_held_out=HELD_OUT, seed=SEED)
    if MAX_EVAL_UTTS and len(trial_items) > MAX_EVAL_UTTS:
        trial_items = trial_items[:: len(trial_items) // MAX_EVAL_UTTS + 1]
    trials = make_trials(trial_items, n_pairs=TRIALS, seed=SEED)

    speeds = (0.9, 1.0, 1.1) if SPEED_AUG else (1.0,)
    ds = SpeakerCrops(train_items, crop_seconds=CROP_S, train=True, speeds=speeds,
                      noise_prob=NOISE_PROB, norm="mean", seed=SEED)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=True,
                        num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)

    model = make_speaker_net(ARCH, n_mels=80, channels=CHANNELS,
                             embed_dim=EMBED_DIM).to(device)
    head = make_head(LOSS, EMBED_DIM, ds.n_classes,
                     **({"margin": MARGIN, "scale": SCALE} if LOSS == "aam" else {})
                     ).to(device)
    # One container so the checkpoint carries the classifier head too: resuming
    # with a re-initialised head throws away everything the margin has learned.
    bundle = torch.nn.ModuleDict({"encoder": model, "head": head})
    opt = torch.optim.Adam(bundle.parameters(), lr=LR, weight_decay=2e-5)

    print(f"device: {device}   split: {SPLIT}   arch: {ARCH}   loss: {LOSS}")
    print(f"train: {len(train_items)} utts / {ds.n_speakers} speakers "
          f"-> {ds.n_classes} classes ({len(speeds)} speeds)")
    print(f"held out: {len({i['speaker'] for i in trial_items})} speakers, "
          f"{len(trial_items)} utts, {len(trials)} trials")
    print(f"encoder: {count_parameters(model)/1e6:.2f}M params, "
          f"{EMBED_DIM}-d embedding, {CROP_S}s crops\n")

    if PLOT_ONLY:
        hist_file = OUT / f"21_{NAME}_history.json"
        history = json.loads(hist_file.read_text())["history"]
        report_and_plot(model, trial_items, trials, history, device, speeds)
        return

    step, extra = (load_resume(RESUME_PATH, bundle, opt, device) if RESUME else (0, {}))
    history: list[tuple[int, float, float]] = [tuple(h) for h in extra.get("history", [])]
    best = float(extra.get("best_eer", 1.0))
    if step:
        print(f"resumed at step {step} (best EER so far {best*100:.2f}%)\n")
    else:
        # The floor from script 20, on the identical trials. norm="none" so the
        # mean half of the statistic survives (see script 20, point 3).
        base = summarize(*score_trials(
            embed_items(None, trial_items, device, norm="none"), trials))
        print(f"untrained floor (log-mel mean+std): EER {base['eer']*100:.2f}%   "
              f"minDCF {base['min_dcf']:.3f}\n")
        history.append((0, base["eer"], base["min_dcf"]))

    running, seen, t0 = 0.0, 0, time.time()
    correct = total = 0
    data_iter = iter(loader)
    while step < STEPS:
        try:
            feats, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            feats, labels = next(data_iter)

        for g in opt.param_groups:
            g["lr"] = warmup_cosine(step, max(100, STEPS // 20), STEPS, LR)
        head.set_margin(margin_schedule(step, STEPS, MARGIN))

        feats, labels = feats.to(device), labels.to(device)
        emb = model(feats)
        loss, cos = head(emb, labels)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(bundle.parameters(), 5.0)
        opt.step()
        step += 1

        running += loss.item()
        seen += 1
        correct += (cos.argmax(dim=1) == labels).sum().item()
        total += labels.numel()

        if step % 100 == 0:
            print(f"step {step:6d}  loss {running/seen:6.3f}  "
                  f"train acc {correct/max(1,total)*100:5.1f}%  "
                  f"margin {head.margin if LOSS=='aam' else 0:.3f}  "
                  f"lr {opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)
            running, seen, correct, total = 0.0, 0, 0, 0

        if step % EVAL_EVERY == 0 or step == STEPS:
            r = evaluate(model, trial_items, trials, device)
            history.append((step, r["eer"], r["min_dcf"]))
            flag = ""
            if r["eer"] < best:
                best = r["eer"]
                torch.save({"model": model.state_dict(), "arch": ARCH,
                            "channels": CHANNELS, "embed_dim": EMBED_DIM,
                            "n_mels": 80, "norm": "mean", "eer": r["eer"],
                            "min_dcf": r["min_dcf"], "step": step, "split": SPLIT},
                           CKPT_PATH)
                flag = "  <- best, saved"
            print(f"  eval @ {step}: EER {r['eer']*100:5.2f}%  "
                  f"minDCF {r['min_dcf']:.3f}  "
                  f"cos same {r['mean_same']:+.3f} / diff {r['mean_diff']:+.3f}{flag}",
                  flush=True)
            save_resume(RESUME_PATH, bundle, opt, step,
                        {"history": history, "best_eer": best})

    clear_resume(RESUME_PATH)
    print(f"\ndone. best held-out EER {best*100:.2f}%  ->  {CKPT_PATH}")
    report_and_plot(model, trial_items, trials, history, device, speeds)


def report_and_plot(model, trial_items, trials, history, device, speeds) -> None:
    """Final numbers + the three-panel figure, from the best checkpoint."""
    best_ck = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(best_ck["model"])
    embs = embed_items(model, trial_items, device, norm="mean")
    scores, labels = score_trials(embs, trials)
    final = summarize(scores, labels)

    # ---- how much of that number is real? -------------------------------
    # Two separate caveats, and with few held-out speakers they are both large.
    #
    #  1. SAMPLING NOISE on a single evaluation, from resampling the trials.
    #  2. SELECTION BIAS from keeping the best of N evaluations. Picking the
    #     minimum of a noisy sequence is biased low by construction — some of
    #     the "best" checkpoint's advantage is just the luckiest draw. The
    #     median across evaluations after the first is the fairer summary of
    #     what this recipe actually delivers.
    eer, lo, hi = bootstrap_eer(scores, labels)
    later = sorted(h[1] for h in history[1:])
    median = later[len(later) // 2] if later else float("nan")
    print(f"\nbest checkpoint : EER {eer*100:5.2f}%   "
          f"95% CI [{lo*100:.2f}, {hi*100:.2f}]  (bootstrap over trials)")
    print(f"median of {len(later)} evals: EER {median*100:5.2f}%   "
          f"range [{min(later)*100:.2f}, {max(later)*100:.2f}]")
    print(f"untrained floor : EER {history[0][1]*100:5.2f}%")
    # Flag on whichever uncertainty is larger: resampling the trials, or the
    # spread across evaluations. Training noise is usually the bigger of the two
    # and the bootstrap alone would understate it.
    spread = (max(later) - min(later)) if later else 0.0
    if max(hi - lo, spread) > 0.02:
        print("\nNOTE: that confidence interval is wide. With "
              f"{len({i['speaker'] for i in trial_items})} held-out speakers this "
              "setup cannot resolve\n      differences of a point or two between "
              "recipes. Compare on more speakers\n      (SPLIT=train-clean-100) "
              "before believing an A/B.")

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    h = np.array(history)
    ax.plot(h[:, 0], h[:, 1] * 100, "o-", color="#4c72b0", label="EER %")
    ax.plot(h[:, 0], h[:, 2] * 100, "s--", color="#dd8452", label="minDCF x100")
    ax.axhline(h[0, 1] * 100, color="k", ls=":", lw=1, label="untrained floor")
    ax.set_xlabel("training step")
    ax.set_ylabel("held-out speakers")
    ax.set_title(f"{ARCH} + {LOSS}: error on speakers never trained on")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.hist(scores[labels == 0], bins=60, alpha=0.65, color="#c44e52",
            density=True, label="different speakers")
    ax.hist(scores[labels == 1], bins=60, alpha=0.65, color="#4c72b0",
            density=True, label="same speaker")
    ax.axvline(final["threshold"], color="k", ls="--", lw=1,
               label=f"EER threshold ({final['eer']*100:.1f}% EER)")
    ax.set_xlabel("cosine similarity")
    ax.set_title("trained: the same plot as script 20")
    ax.legend(fontsize=8)

    ax = axes[2]
    paths = [it["path"] for it in trial_items]
    spk = [it["speaker"] for it in trial_items]
    X = torch.stack([embs[p] for p in paths])
    X = X - X.mean(0, keepdim=True)
    _, _, V = torch.pca_lowrank(X, q=2)
    P = (X @ V[:, :2]).numpy()
    cmap = plt.get_cmap("tab10")
    for i, s in enumerate(sorted(set(spk))):
        m = np.array([x == s for x in spk])
        ax.scatter(P[m, 0], P[m, 1], s=12, alpha=0.8, color=cmap(i % 10), label=s)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("embedding space, held-out speakers")
    ax.legend(fontsize=7, ncol=2, title="speaker")

    fig.suptitle(f"Module 9: {ARCH}/{LOSS} on {SPLIT} — "
                 f"{final['eer']*100:.2f}% EER, {final['min_dcf']:.3f} minDCF "
                 f"on {HELD_OUT} unseen speakers")
    fig.tight_layout()
    fig.savefig(OUT / f"21_{NAME}_eer.png", dpi=130)
    (OUT / f"21_{NAME}_history.json").write_text(json.dumps(
        {"history": history, "final": final, "config": {
            "split": SPLIT, "arch": ARCH, "loss": LOSS, "steps": STEPS,
            "channels": CHANNELS, "embed_dim": EMBED_DIM, "crop_s": CROP_S,
            "noise_prob": NOISE_PROB, "speeds": list(speeds), "held_out": HELD_OUT}},
        indent=2))
    print(f"wrote {OUT / f'21_{NAME}_eer.png'}")


if __name__ == "__main__":
    main()
