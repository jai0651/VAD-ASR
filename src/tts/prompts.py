"""
Module 8, part 1: choosing WHAT to read into the microphone.

You will record this corpus exactly once. Every choice here is frozen into an
hour of your life, so it is worth more thought than it usually gets.

Two things decide whether a one-hour single-speaker corpus trains a good voice,
and neither of them is "more sentences":

  PHONETIC COVERAGE. The acoustic model can only synthesise sounds it has heard
  you make, in the contexts it has heard them. A corpus that never contains
  "th" before a rounded vowel will produce a bad "th" there forever. Real TTS
  corpora (CMU Arctic, 1132 sentences) were built by greedily selecting
  sentences from a large text pool to maximise DIPHONE coverage — pairs of
  adjacent sounds — because the hard part of speech is the transitions, not the
  steady states. We do the same thing with `select_by_coverage` below.

  DOMAIN MATCH. Prosody is learned, not transferred. If you read only
  19th-century audiobook prose, you get a voice that can *only* narrate: it
  will read "Sure, that's done — want me to push it?" in a stately falling
  cadence, because it has never heard you speak a short conversational turn.
  Since the target here is an agent replying to you, roughly a third of the
  corpus is conversational: `AGENT_SEED`.

WHAT WE APPROXIMATE, AND WHY IT IS FINE FOR NOW.
Greedy coverage below runs over CHARACTER trigrams, not phoneme diphones,
because grapheme-to-phoneme conversion is Module 8's *next* part and this
harness has to be usable tonight. English spelling is an irregular but far from
random encoding of pronunciation, so character trigrams are a decent proxy: the
selected set is visibly more diverse than a random sample of the same size.
When `src/tts/g2p.py` exists, swap `_ngrams` for the phoneme diphones of the
sentence and re-run — the selection code does not change.

The source text pool is the LibriSpeech transcripts already on disk from
Module 2b. No download, and the sentences are known-readable English.
"""

from __future__ import annotations

import random
from pathlib import Path

from src.asr.tokenizer import normalize

LIBRISPEECH_DIR = "data/LibriSpeech/dev-clean"
PROMPTS_PATH = "data/voice/prompts.txt"
WORDS_PER_MINUTE = 150.0  # normal read-aloud pace; the harness measures yours

# Short turns are where a voice agent actually lives, and they are exactly what
# an audiobook corpus lacks. Hand-written rather than sampled so the set covers
# the shapes that carry distinctive prosody: yes/no answers, questions,
# apologies, enumerations, numbers, and mid-sentence self-correction.
AGENT_SEED = [
    "Sure, I can do that.",
    "Done. Anything else you want me to change?",
    "That's already handled.",
    "Hmm, that's not quite what I expected.",
    "Let me check that for you.",
    "I don't think that's going to work.",
    "Give me a second.",
    "Okay, so here's what I found.",
    "Yes, but only for the first case.",
    "No, that file hasn't changed.",
    "Do you want me to keep going?",
    "Should I push this, or wait?",
    "Which one did you mean?",
    "Sorry, I misread that.",
    "You're right, let me fix it.",
    "That took about four seconds.",
    "There are twelve files, and three of them are new.",
    "It failed on line ninety-seven.",
    "The test suite passed, all sixty-two of them.",
    "I found two problems and fixed one.",
    "First, I'll read the config. Then I'll run the migration.",
    "Good morning. What are we working on today?",
    "Good night. I'll leave the server running.",
    "Actually, wait, that's the wrong branch.",
    "I'm not sure. Let me look it up.",
    "That's a good question.",
    "It's already installed.",
    "Try it now.",
    "Nothing changed.",
    "Everything's up to date.",
    "The connection dropped, so I retried.",
    "I've saved it to your downloads folder.",
    "Would you like the short version or the long one?",
    "This might take a few minutes.",
    "Almost done.",
    "Hold on, something's wrong here.",
    "That's the third time it's happened today.",
    "I can't reach that address.",
    "Permission denied. Do you want me to use sudo?",
    "It's running on port eight thousand.",
    "Your meeting starts in ten minutes.",
    "It's about twenty degrees and clear outside.",
    "You have three unread messages.",
    "I'll remind you at half past four.",
    "The flight leaves at six forty in the morning.",
    "That'll be around two hundred and fifty dollars.",
    "Let me summarize what we decided.",
    "In short: it works, but it's slow.",
    "The bug was in the loader, not the model.",
    "I rewrote it, and now it's about twice as fast.",
    "Honestly, I'd start over with a simpler version.",
    "Why don't we try the other approach first?",
    "What if the input is empty?",
    "Are you sure you want to delete all of them?",
    "I'd rather not guess at that.",
    "It depends on how much data you have.",
    "Roughly speaking, yes.",
    "Not exactly, but close enough.",
    "That's a fair point.",
    "I hadn't thought of that.",
    "Thanks, that helps.",
    "No problem at all.",
    "You're welcome.",
    "Talk to you later.",
    "One moment please.",
    "Could you say that again?",
    "I didn't catch the last part.",
    "The word you want is 'idempotent'.",
    "Spell it for me, letter by letter.",
    "Zero, one, two, three, four, five, six, seven, eight, nine.",
    "A, B, C, X, Y, Z.",
    "It's spelled with two Ls and a silent E.",
    "The answer is forty-two point five.",
    "About ninety percent of them succeeded.",
    "On Tuesday the twenty-third, at noon.",
    "January, February, March, April.",
    "It's version three point eleven.",
    "Press control C to stop it.",
    "Open the file called README, all capitals.",
    "The path is slash user slash local slash bin.",
    "Send it to jai at example dot com.",
    "I'll queue it up and let you know when it finishes.",
    "Careful — that's not reversible.",
    "Are you still there?",
    "Okay, I'm listening.",
    "Go ahead.",
    "Wait, say that once more?",
    "Interesting. I didn't expect that either.",
    "That's genuinely strange.",
    "Well, that explains it.",
    "Ah, of course.",
    "Oh no.",
    "Great, that worked.",
    "Perfect.",
    "Hmm.",
    "Right.",
    "Exactly.",
    "Not yet.",
    "Never mind.",
    "I'm on it.",
]


def _ngrams(text: str, order: int = 3) -> set[str]:
    """The coverage units of one sentence. Swap this for phoneme diphones later."""
    t = f" {normalize(text)} "
    return {t[i:i + order] for i in range(len(t) - order + 1)}


def load_source_sentences(
    root: str = LIBRISPEECH_DIR, min_words: int = 4, max_words: int = 16
) -> list[str]:
    """Candidate prompts from LibriSpeech transcripts already on disk.

    The length window is the whole trick of a readable corpus. Under ~4 words
    there is not enough context for natural prosody (you read them as isolated
    tokens); over ~16 you start to stumble, run out of breath, and — worse —
    drift in pace across the take, which teaches the model inconsistency.

    dev-clean yields ~1250 sentences (~85 min of read speech) in this window,
    so a one-hour corpus is SELECTED from the pool, not merely all of it. That
    margin is what makes the selection below worth running at all.
    """
    seen: set[str] = set()
    out: list[str] = []
    for trans in sorted(Path(root).glob("*/*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            if " " not in line:
                continue
            text = normalize(line.split(" ", 1)[1])
            n = len(text.split())
            if not (min_words <= n <= max_words) or text in seen:
                continue
            seen.add(text)
            out.append(text)
    return out


def select_by_coverage(
    candidates: list[str], n: int, order: int = 3, seed_texts: list[str] | None = None
) -> list[str]:
    """Greedily pick the sentence that best balances n-gram FREQUENCY.

    The obvious objective — "cover every n-gram at least once" — is the wrong
    one, and it fails loudly: run it on this pool and it saturates at ~570
    sentences with 100% coverage and nothing left to optimise, half the corpus
    you wanted. The reason it is wrong is that a trigram seen ONCE is not
    learnable. Gradient descent needs repetition; coverage is a set property
    and training is a frequency problem.

    So score a sentence by how much it helps the RAREST units:

        score = sum over its n-grams of 1/(1 + count_so_far)   / sqrt(len)

    An unseen n-gram is worth 1.0, a second sighting 0.5, a tenth 0.09. This
    degrades gracefully into plain set cover at the start (everything is unseen)
    and into frequency balancing afterwards, with no special case and no
    saturation point. What you get is a corpus where the rare sounds appear
    several times each instead of exactly once.

    Two details that matter more than the greedy loop itself:

      NORMALISE BY LENGTH. Dividing by sqrt(len) stops the ranking from
      collapsing into "longest sentence wins" — which would hand you the
      hardest sentences in the pool to read, the opposite of what you want at
      the microphone.

      SEED WITH WHAT YOU ALREADY HAVE. `seed_texts` (the conversational set) is
      counted before selection starts, so the audiobook half spends its budget
      on the gaps the conversational half left rather than re-covering "the".

    Greedy is not optimal — this is NP-hard — but it is within a log factor and,
    at ~1200 candidates, exact optimisation buys nothing you could hear.
    """
    from collections import Counter

    counts: Counter[str] = Counter()
    for t in seed_texts or []:
        counts.update(_ngrams(t, order))

    grams = [(_ngrams(c, order), c) for c in candidates]
    chosen: list[str] = []
    for _ in range(min(n, len(grams))):
        best_i, best_score = -1, -1.0
        for i, (g, c) in enumerate(grams):
            if not g:
                continue
            score = sum(1.0 / (1 + counts[x]) for x in g) / (len(c) ** 0.5)
            if score > best_score:
                best_i, best_score = i, score
        if best_i < 0:
            break
        g, c = grams.pop(best_i)
        counts.update(g)
        chosen.append(c)
    return chosen


def build_prompt_set(target_minutes: float = 60.0, seed: int = 0) -> list[str]:
    """The corpus you will read: conversational seed + frequency-balanced prose.

    Budgeted in MINUTES OF SPEECH, not sentences, because minutes are the thing
    that actually constrains you (an hour at the mic) and the thing that
    constrains the model (an hour of audio). ~150 words per minute is a normal
    read-aloud pace; the harness measures your real pace as you go and tells you
    how far off this estimate you are.

    One hour is the target because it is roughly where fine-tuning a pretrained
    multi-speaker acoustic model stops sounding like a smeared average of you
    and starts sounding like you. Below ~20 minutes it is recognisably you but
    unstable; above ~2 hours the returns flatten hard.

    Shuffled with a FIXED seed, deterministically: the order is part of the
    corpus. You will record this over several sittings, and a stable order is
    what lets the harness resume at prompt 341 tomorrow. Interleaving the
    conversational and prose halves also matters — recording all the audiobook
    prose in one sitting would put a pace-and-mood boundary right through the
    middle of your dataset.
    """
    seed_texts = [normalize(s) for s in AGENT_SEED]
    budget_words = target_minutes * WORDS_PER_MINUTE
    remaining = budget_words - sum(len(t.split()) for t in seed_texts)

    pool = load_source_sentences()
    mean_len = sum(len(c.split()) for c in pool) / max(1, len(pool))
    n_prose = max(0, round(remaining / mean_len))

    prose = select_by_coverage(pool, n_prose, seed_texts=seed_texts)
    prompts = seed_texts + prose
    random.Random(seed).shuffle(prompts)
    return prompts


def load_or_build_prompts(
    path: str = PROMPTS_PATH, target_minutes: float = 60.0
) -> list[str]:
    """Read the frozen prompt list, building it on first run and EXTENDING it
    when you raise the target.

    Frozen on disk on purpose: regenerating the list mid-corpus would reshuffle
    every prompt, and any take whose sentence vanished would be orphaned.

    But "frozen" must not mean "silently ignores you". The natural way to use
    this is to record 20 minutes, decide it is going well, and come back for an
    hour — and a strict freeze would hand you the 20-minute list forever with no
    indication that `--minutes 60` did nothing. So a larger target APPENDS
    newly-selected sentences, credited against the coverage the existing set
    already has, and leaves every existing prompt in place at its existing
    index. A smaller target is ignored: you do not un-record audio.
    """
    p = Path(path)
    if not p.exists():
        prompts = build_prompt_set(target_minutes)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("\n".join(prompts) + "\n")
        return prompts

    existing = [line for line in p.read_text().splitlines() if line.strip()]
    have = sum(len(t.split()) for t in existing) / WORDS_PER_MINUTE
    if target_minutes <= have * 1.05:
        return existing

    pool = [c for c in load_source_sentences() if c not in set(existing)]
    mean_len = sum(len(c.split()) for c in pool) / max(1, len(pool))
    n_more = round((target_minutes - have) * WORDS_PER_MINUTE / mean_len)
    extra = select_by_coverage(pool, n_more, seed_texts=existing)
    if not extra:
        return existing

    print(f"[prompts] extending {len(existing)} -> {len(existing) + len(extra)} "
          f"({have:.0f} -> {target_minutes:.0f} min)")
    p.write_text("\n".join(existing + extra) + "\n")
    return existing + extra


def coverage_report(prompts: list[str], order: int = 3) -> dict:
    """Is this corpus actually balanced? Coverage alone will lie to you.

    `pool_coverage` says how many of the source pool's n-grams appear at all;
    `covered_5x` says how many appear at least five times. The second number is
    the one that predicts whether the model can learn a sound — and it is the
    number that plain set cover leaves at almost zero.
    """
    from collections import Counter

    pool: set[str] = set()
    for s in load_source_sentences():
        pool |= _ngrams(s, order)
    counts: Counter[str] = Counter()
    for s in prompts:
        counts.update(_ngrams(s, order))

    words = sum(len(s.split()) for s in prompts)
    return {
        "sentences": len(prompts),
        "words": words,
        "est_minutes": words / WORDS_PER_MINUTE,
        "distinct_ngrams": len(counts),
        "pool_coverage": len(set(counts) & pool) / max(1, len(pool)),
        "covered_5x": sum(1 for g in pool if counts[g] >= 5) / max(1, len(pool)),
        "median_count": sorted(counts.values())[len(counts) // 2] if counts else 0,
    }
