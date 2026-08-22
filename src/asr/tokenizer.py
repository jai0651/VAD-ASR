"""
Module 6, part 1: subword tokenization (BPE), written from scratch.

Module 2 modelled text as 28 characters. Every modern recognizer models it as
250–1024 *subwords*, and the reason is not cosmetic:

  FEWER OUTPUT STEPS. "recognition" is 11 CTC frames' worth of characters but
  ~3 subwords. Shorter targets mean the attention decoder runs 3x fewer steps
  (directly: latency) and CTC has an easier alignment to find.

  SPELLING FOR FREE. A character model must learn English orthography from
  scratch through the decoder. A subword model gets "▁recogn" + "ition" as
  atomic units, so it cannot misspell the inside of a common morpheme. This is
  most of why our char-level CTC produced "he inl san er mianthes" — plausible
  phonetics, impossible English.

  OPEN VOCABULARY. Unlike a word vocabulary, BPE can still spell a name it has
  never seen by falling back to shorter pieces.

THE ALGORITHM (Sennrich et al. 2016). Start with every word as a sequence of
characters. Repeatedly find the most frequent adjacent symbol pair in the
corpus and merge it into one new symbol. Do that `vocab_size` times and you
have learned, purely from statistics, that "th", "ing" and "▁the" deserve to be
single units while "qxz" does not.

We keep a `▁` marker on word starts (the SentencePiece convention) so that
detokenization is exact and lossless: join every piece, turn `▁` back into a
space. Without it "▁a ▁cat" and "▁ac at" would decode identically.

Special tokens, and why these indices:
    0  <blank>   CTC's blank. torch.nn.CTCLoss defaults to blank=0, and we also
                 reuse it as the padding id for attention targets (it can never
                 legitimately appear there, so ignore_index=0 is unambiguous).
    1  <sos/eos> one token for both, as in ESPnet: the decoder starts from it
                 and is trained to emit it when finished.
    2  <unk>     for characters outside the training alphabet.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

BLANK_ID = 0
SOS_ID = 1  # also EOS
UNK_ID = 2
N_SPECIAL = 3

WORD_START = "▁"  # ▁

_KEEP = re.compile(r"[^a-z' ]+")


def normalize(text: str) -> str:
    """Lowercase, drop everything but letters/apostrophe/space, squeeze spaces.

    ASR text normalization is a real design decision, not boilerplate: whatever
    you strip here, the model can never produce, and whatever you keep it must
    learn to place. Keeping the apostrophe matters (don't/dont is a real error);
    keeping punctuation would force the acoustic model to guess commas.
    """
    return " ".join(_KEEP.sub(" ", text.lower()).split())


class BPETokenizer:
    """Byte-pair encoding over words. `merges` is an ORDERED list — the order
    is the algorithm, since encoding replays the merges in the same sequence."""

    def __init__(self, merges: list[tuple[str, str]], vocab: list[str],
                 corpus: str = ""):
        self.merges = [tuple(m) for m in merges]
        self.vocab = vocab
        # Fingerprint of the corpus this was TRAINED on. Carried in the file so
        # a tokenizer can never be silently reused across datasets — which is
        # exactly what happened here: a tokenizer fitted on dev-clean (2,557
        # utterances) was picked up for a train-clean-100 run (28,500) because
        # the only check was `vocab_size`. Merges learned from 11x less text.
        self.corpus = corpus
        self.token_to_id = {t: i for i, t in enumerate(vocab)}
        self._ranks = {pair: i for i, pair in enumerate(self.merges)}
        self._cache: dict[str, list[str]] = {}

    # ---------------------------------------------------------------- training
    @classmethod
    def train(cls, texts, vocab_size: int = 512, min_freq: int = 2) -> "BPETokenizer":
        """Learn `vocab_size` tokens from raw transcripts."""
        word_freq: Counter[str] = Counter()
        for t in texts:
            for w in normalize(t).split():
                word_freq[WORD_START + w] += 1

        # Each word is a mutable tuple of symbols; alphabet seeds the vocab.
        splits: dict[str, list[str]] = {w: list(w) for w in word_freq}
        alphabet = sorted({c for w in word_freq for c in w})

        # Incremental pair statistics. Recounting every pair after every merge
        # is the naive version and is O(merges x corpus); instead we keep, for
        # each pair, its count AND the set of words containing it, so a merge
        # only touches the words it actually affects.
        pair_freq: Counter[tuple[str, str]] = Counter()
        pair_words: dict[tuple[str, str], set[str]] = defaultdict(set)
        for w, f in word_freq.items():
            syms = splits[w]
            for a, b in zip(syms, syms[1:]):
                pair_freq[(a, b)] += f
                pair_words[(a, b)].add(w)

        merges: list[tuple[str, str]] = []
        target = vocab_size - N_SPECIAL - len(alphabet)
        for _ in range(max(0, target)):
            if not pair_freq:
                break
            best, count = pair_freq.most_common(1)[0]
            if count < min_freq:
                break
            merges.append(best)
            new_sym = best[0] + best[1]

            for w in list(pair_words[best]):
                f = word_freq[w]
                syms = splits[w]
                # Remove this word's contribution before rewriting it.
                for a, b in zip(syms, syms[1:]):
                    pair_freq[(a, b)] -= f
                    if pair_freq[(a, b)] <= 0:
                        del pair_freq[(a, b)]
                    pair_words[(a, b)].discard(w)

                out, i = [], 0
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                        out.append(new_sym)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                splits[w] = out

                for a, b in zip(out, out[1:]):
                    pair_freq[(a, b)] += f
                    pair_words[(a, b)].add(w)

        vocab = ["<blank>", "<sos/eos>", "<unk>"] + alphabet
        seen = set(vocab)
        for a, b in merges:
            tok = a + b
            if tok not in seen:
                seen.add(tok)
                vocab.append(tok)
        return cls(merges, vocab)

    # ---------------------------------------------------------------- encoding
    def _split_word(self, word: str) -> list[str]:
        if word in self._cache:
            return self._cache[word]
        syms = list(word)
        while len(syms) > 1:
            # Apply the EARLIEST-learned applicable merge — replaying training
            # order is what makes encoding deterministic and consistent with
            # the statistics the merges were learned from.
            best_rank, best_i = None, -1
            for i, pair in enumerate(zip(syms, syms[1:])):
                r = self._ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_i = r, i
            if best_i < 0:
                break
            syms[best_i:best_i + 2] = [syms[best_i] + syms[best_i + 1]]
        self._cache[word] = syms
        return syms

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in normalize(text).split():
            for tok in self._split_word(WORD_START + w):
                ids.append(self.token_to_id.get(tok, UNK_ID))
        return ids

    def decode(self, ids) -> str:
        pieces = [
            self.vocab[i] for i in ids
            if N_SPECIAL <= i < len(self.vocab)
        ]
        return "".join(pieces).replace(WORD_START, " ").strip()

    # ------------------------------------------------------------ persistence
    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"merges": [list(m) for m in self.merges], "vocab": self.vocab,
             "corpus": self.corpus}
        ))

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        d = json.loads(Path(path).read_text())
        return cls(d["merges"], d["vocab"], d.get("corpus", ""))
