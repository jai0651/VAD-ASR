"""
Module 2b demo: train the SAME CTC model on REAL speech (LibriSpeech).

The model (src/asr/model.py), CTC loss, and decoders (src/asr/decode.py) are
unchanged from the synthetic demo. Only the data source changed
(src/asr/librispeech.py). That is the whole point: a correct pipeline is data-
agnostic.

Reality check: training ASR from scratch to low error needs lots of data + a GPU
+ hours. This machine runs on CPU, so by default we train on a SUBSET of short
utterances and pre-compute their features once. You will watch the loss fall and
the transcriptions go from gibberish -> recognizable real English. To push toward
a genuinely good recognizer, raise SUBSET, train longer, and use a GPU.

Run:
  uv run python scripts/05_train_asr_libri.py
Optional env vars:
  SUBSET=200 STEPS=1200 BATCH=8 uv run python scripts/05_train_asr_libri.py
"""

from __future__ import annotations

import os
import random
import sys
import time

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.asr.decode import beam_search_decode, char_error_rate, greedy_decode  # noqa: E402
from src.asr.librispeech import LibriSpeechFeatures, ctc_collate  # noqa: E402
from src.asr.model import ASRModel  # noqa: E402
from src.asr.text import BLANK  # noqa: E402

SUBSET = int(os.environ.get("SUBSET", "150"))   # how many utterances to train on
STEPS = int(os.environ.get("STEPS", "1000"))
BATCH = int(os.environ.get("BATCH", "8"))
HELDOUT = 10  # utterances kept aside to measure generalization


def get_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    device = get_device()
    print(f"device: {device}")

    print("Loading LibriSpeech (filtering by duration)...")
    ds = LibriSpeechFeatures("data", url="dev-clean", download=False,
                             min_seconds=1.0, max_seconds=6.0)
    print(f"  {len(ds)} utterances in 1-6 s after filtering")

    # Deterministic split: first HELDOUT for eval, next SUBSET for training.
    order = list(range(len(ds)))
    random.shuffle(order)
    heldout_idx = order[:HELDOUT]
    train_idx = order[HELDOUT:HELDOUT + SUBSET]

    print(f"Pre-computing features for {len(train_idx)} train + {HELDOUT} heldout...")
    t0 = time.time()
    train = [ds[i] for i in train_idx]       # (feats, target, text)
    heldout = [ds[i] for i in heldout_idx]
    print(f"  done in {time.time() - t0:.1f}s")

    model = ASRModel(n_mels=80, hidden=256).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    def decode_some(examples, n=3, use_beam=False):
        model.eval()
        rows, cers = [], []
        for feats, _tgt, text in examples[:n]:
            with torch.no_grad():
                lp = model(feats.unsqueeze(0).to(device))[0].cpu()
            pred = beam_search_decode(lp, 12) if use_beam else greedy_decode(lp)
            cers.append(char_error_rate(pred, text))
            rows.append((text, pred))
        model.train()
        return rows, (sum(cers) / len(cers) if cers else 0.0)

    print(f"\nTraining {STEPS} steps, batch {BATCH}...")
    model.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        batch = random.sample(train, min(BATCH, len(train)))
        feats, in_lens, targets, tgt_lens, _ = ctc_collate(batch)
        feats = feats.to(device)

        log_probs = model(feats)                       # (B, T', V)
        out_lens = model.downsampled_length(in_lens)
        loss = ctc(log_probs.transpose(0, 1), targets, out_lens, tgt_lens)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % 100 == 0 or step == 1:
            rows, cer = decode_some(train, n=1)
            rate = step / (time.time() - t0)
            print(f"step {step:4d} | loss {loss.item():6.2f} | train CER {cer:.2f} "
                  f"| {rate:.1f} it/s")
            print(f"    truth: {rows[0][0]}")
            print(f"    pred : {rows[0][1]}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/asr_libri.pt")

    print("\n=== TRAIN (seen) examples, beam decoded ===")
    rows, cer = decode_some(train, n=3, use_beam=True)
    for truth, pred in rows:
        print(f"  truth: {truth}\n  pred : {pred}\n")
    print(f"train CER: {cer:.3f}")

    print("\n=== HELD-OUT (unseen) examples, beam decoded ===")
    rows, cer = decode_some(heldout, n=3, use_beam=True)
    for truth, pred in rows:
        print(f"  truth: {truth}\n  pred : {pred}\n")
    print(f"held-out CER: {cer:.3f}  (expected high with a tiny subset — that gap "
          f"is the data-hunger of ASR)")
    print("\nSaved outputs/asr_libri.pt")


if __name__ == "__main__":
    main()
