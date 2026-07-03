"""
Module 2, part 1: the character vocabulary and the CTC "collapse" rule.

ASR outputs text. We model it at the *character* level (simplest possible: no
pronunciation dictionary, no word list). The model emits, for every audio frame,
a distribution over our characters plus one special extra symbol: the BLANK.

THE ALIGNMENT PROBLEM
We have, say, 60 audio frames but the word "cat" has 3 characters. We are never
told "frames 0-20 are 'c', 21-40 are 'a', ...". CTC sidesteps this: the network
emits one label *per frame* (a length-60 string over {chars, blank}), and we
define a deterministic function that *collapses* that frame-string down to the
final text:

    1. merge runs of the SAME character        ("ccaaat"   -> "cat")
    2. then remove all blanks                   ("cc_aa_t"  -> "cat",  _ = blank)

The blank is the clever part: it lets the model (a) emit "nothing" during a frame,
and (b) separate two genuine repeats. To write "hello" (with its double 'l') the
frame-string must put a blank between the l's: "he_l_lo" collapses to "hello",
whereas "hello" without the blank would merge to "helo".

CTC training sums the probability of *every* frame-string that collapses to the
target, so the model is free to discover its own alignment. We implement the
collapse here (used by decoding) and let torch's CTCLoss handle the summation.
"""

from __future__ import annotations

# blank is index 0 (the default torch.nn.CTCLoss expects).
BLANK = 0
CHARS = " abcdefghijklmnopqrstuvwxyz"  # index in vocab = position + 1

_char_to_idx = {c: i + 1 for i, c in enumerate(CHARS)}
_idx_to_char = {i + 1: c for i, c in enumerate(CHARS)}

VOCAB_SIZE = len(CHARS) + 1  # +1 for blank


def encode(text: str) -> list[int]:
    """Text -> list of label indices (no blanks; CTC inserts those itself)."""
    return [_char_to_idx[c] for c in text.lower() if c in _char_to_idx]


def decode_labels(indices: list[int]) -> str:
    """Indices (already collapsed, blanks removed) -> text."""
    return "".join(_idx_to_char.get(i, "") for i in indices)


def collapse(frame_labels: list[int]) -> list[int]:
    """Apply the CTC collapse rule to a per-frame label sequence.

    Step 1: merge adjacent duplicates. Step 2: drop blanks.
    This is exactly what greedy decoding uses after taking the per-frame argmax.
    """
    merged: list[int] = []
    prev = None
    for x in frame_labels:
        if x != prev:          # step 1: collapse repeats
            merged.append(x)
        prev = x
    return [x for x in merged if x != BLANK]  # step 2: remove blanks
