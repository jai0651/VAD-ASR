"""
Module 2 demo: build CTC intuition (no training yet).

Run:  uv run python scripts/03_ctc_intuition.py

Shows:
  1. The collapse rule on hand-written frame strings.
  2. Why "hello" REQUIRES a blank between the two l's.
  3. That torch.nn.CTCLoss can be driven to ~0 by optimizing raw logits toward a
     target alignment — proving the loss is differentiable and sensible — and that
     greedy decoding then recovers the target text.
"""

from __future__ import annotations

import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.asr.text import BLANK, VOCAB_SIZE, collapse, decode_labels, encode  # noqa: E402
from src.asr.decode import greedy_decode  # noqa: E402

_ = "_"  # display symbol for blank


def show_collapse(frame_str: str) -> None:
    """frame_str uses '_' for blank, letters for chars. Print the collapsed text."""
    ids = [BLANK if c == "_" else encode(c)[0] for c in frame_str]
    out = decode_labels(collapse(ids))
    print(f"  frames '{frame_str}'  ->  '{out}'")


def main() -> None:
    print("1) The CTC collapse rule (merge repeats, then drop blanks '_'):")
    show_collapse("ccaaat")     # -> cat
    show_collapse("c_a_t")      # -> cat
    show_collapse("__cat__")    # -> cat

    print("\n2) Why 'hello' needs a blank between the double letters:")
    show_collapse("hello")      # -> helo   (the two l's merge!)
    show_collapse("hel_lo")     # -> hello  (blank keeps them apart)

    print("\n3) Optimize raw logits with torch.nn.CTCLoss toward target 'cat':")
    target_text = "cat"
    targets = torch.tensor(encode(target_text))
    T = 12  # frames available (>= len(target))
    # Learnable per-frame logits for a single sequence; we optimize them directly
    # (no model) to isolate the loss behavior.
    logits = torch.randn(T, VOCAB_SIZE, requires_grad=True)
    opt = torch.optim.Adam([logits], lr=0.2)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    for step in range(300):
        log_probs = torch.log_softmax(logits, dim=-1).unsqueeze(1)  # (T, N=1, V)
        loss = ctc(
            log_probs,
            targets,
            input_lengths=torch.tensor([T]),
            target_lengths=torch.tensor([len(targets)]),
        )
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 60 == 0 or step == 299:
            decoded = greedy_decode(torch.log_softmax(logits, dim=-1).detach())
            print(f"  step {step:3d} | loss {loss.item():.4f} | greedy='{decoded}'")

    print(f"\nTarget was '{target_text}'. The loss falls and greedy decoding "
          f"recovers it — CTC found a valid alignment on its own.")


if __name__ == "__main__":
    main()
