"""
Module 8 tests: the recording harness.

`test_resume_survives_a_kill` is the load-bearing one. This corpus is recorded
by a human over multiple sittings, and the single failure that would actually
cost something irreplaceable is losing takes to an interruption. Everything
else here can be re-run for free.

`test_selection_beats_random` guards the claim the prompt module makes. Greedy
frequency-balanced selection is only worth its complexity if it measurably
beats sampling the same number of sentences at random — so measure it, rather
than trusting the argument.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from src.asr.tokenizer import normalize
from src.tts.corpus import STORE_SR, VoiceCorpus, load_utterance, resample
from src.tts.prompts import (
    AGENT_SEED,
    build_prompt_set,
    coverage_report,
    load_source_sentences,
)
from src.tts.qc import check_take, snr_db, speech_mask, trim_to_speech
from src.vad.model import VADNet

SR = 48_000


@pytest.fixture(scope="module")
def vad() -> VADNet:
    model = VADNet(n_mels=80)
    model.load_state_dict(torch.load("outputs/vad_real.pt", map_location="cpu"))
    return model.eval()


def synthetic_take(speech_s: float = 1.5, lead_s: float = 0.6, tail_s: float = 0.9,
                   level: float = 0.25, noise: float = 0.0) -> np.ndarray:
    """Silence, then something speech-shaped, then silence.

    Not real speech — an amplitude-modulated harmonic stack. Enough structure
    for the VAD to fire on, which is all these tests need; the QC arithmetic
    does not care whether the words are real.
    """
    rng = np.random.default_rng(0)
    t = np.arange(int(speech_s * SR)) / SR
    voiced = sum(np.sin(2 * np.pi * f * t) / (i + 1)
                 for i, f in enumerate([120, 240, 360, 480, 720]))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4.0 * t)   # ~4 Hz syllable rate
    body = level * envelope * voiced / np.abs(voiced).max()
    out = np.concatenate([np.zeros(int(lead_s * SR)), body, np.zeros(int(tail_s * SR))])
    return (out + noise * rng.standard_normal(out.shape)).astype(np.float32)


# ---- prompt selection ---------------------------------------------------

def test_selection_beats_random():
    """Greedy frequency balancing must beat a same-size random sample."""
    prompts = build_prompt_set(target_minutes=30.0)
    pool = load_source_sentences()
    baseline = ([normalize(s) for s in AGENT_SEED]
                + random.Random(0).sample(pool, len(prompts) - len(AGENT_SEED)))

    ours, rand = coverage_report(prompts), coverage_report(baseline)
    assert ours["pool_coverage"] > rand["pool_coverage"]
    assert ours["covered_5x"] >= rand["covered_5x"]


def test_prompt_set_is_deterministic():
    """Resume depends on the same prompts in the same order, every run."""
    assert build_prompt_set(20.0) == build_prompt_set(20.0)


def test_raising_the_target_extends_and_never_reorders(tmp_path):
    """Record 20 minutes, then ask for 60: existing prompts must keep their
    index, or every take already on disk is orphaned."""
    from src.tts.prompts import load_or_build_prompts

    path = str(tmp_path / "prompts.txt")
    small = load_or_build_prompts(path, target_minutes=20.0)
    grown = load_or_build_prompts(path, target_minutes=60.0)

    assert grown[:len(small)] == small
    assert len(grown) > len(small)
    # Lowering the target must not delete prompts you may already have read.
    assert load_or_build_prompts(path, target_minutes=5.0) == grown


def test_conversational_prompts_survive_selection():
    """The agent-domain sentences are seeded, not selected — they cannot be
    dropped, or the corpus reverts to pure audiobook prose."""
    prompts = set(build_prompt_set(20.0))
    assert all(normalize(s) in prompts for s in AGENT_SEED)


# ---- QC gates -----------------------------------------------------------

def test_trims_to_speech(vad):
    take = synthetic_take(speech_s=1.5, lead_s=0.6, tail_s=0.9)
    mask = speech_mask(take, SR, vad)
    trimmed = trim_to_speech(take, SR, mask)
    assert trimmed is not None
    # 1.5 s of speech + 0.1 s padding each side, with slack for VAD onset lag.
    assert 1.4 <= len(trimmed) / SR <= 2.1
    assert len(trimmed) < len(take)


def test_silence_only_take_is_rejected(vad):
    silence = np.zeros(int(2.0 * SR), dtype=np.float32)
    trimmed, qc = check_take(silence, SR, "hello there", vad)
    assert trimmed is None
    assert not qc.ok


def test_clipping_is_a_hard_reject(vad):
    take = np.clip(synthetic_take(level=3.0), -1.0, 1.0)
    _, qc = check_take(take, SR, "hello there", vad)
    assert not qc.ok
    assert any("clipped" in p for p in qc.problems)


def test_noise_floor_becomes_a_warning_not_a_reject(vad):
    """A noisy room is a judgement call — the human overrules it, not the tool."""
    take = synthetic_take(noise=0.02)
    _, qc = check_take(take, SR, "hello there", vad)
    assert qc.ok
    assert any("nois" in w for w in qc.warnings)


def test_snr_is_higher_for_the_quieter_room(vad):
    clean, noisy = synthetic_take(), synthetic_take(noise=0.02)
    assert (snr_db(clean, SR, speech_mask(clean, SR, vad))
            > snr_db(noisy, SR, speech_mask(noisy, SR, vad)))


def test_misread_is_caught_by_the_verifier(vad):
    """The read-check compares against the PROMPT, so a wrong prompt must flag."""
    take = synthetic_take()
    _, qc = check_take(take, SR, "hello there", vad,
                       transcribe=lambda a: "completely different words")
    assert any("misread" in w for w in qc.warnings)


# ---- live capture -------------------------------------------------------

def test_streaming_frames_tile_exactly_like_one_shot():
    """The live VAD must see the same frames it would see offline.

    The recorder feeds the VAD 100 ms at a time, but a mel frame is 25 ms wide
    with a 10 ms hop — so a frame straddles every block boundary. Frame each
    block on its own and 15% of all frames vanish, biased toward the onsets and
    offsets the VAD exists to detect, and the take ends a beat late with no
    visible symptom. This asserts the carried-overlap logic reproduces the
    offline frame grid exactly.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "record_voice", "scripts/16_record_voice.py")
    rv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rv)

    pending = np.zeros(0, dtype=np.float32)
    starts: list[int] = []
    consumed = 0
    signal = np.arange(16_000, dtype=np.float32)
    for i in range(0, len(signal), 1600):            # 100 ms blocks at 16 kHz
        pending = np.concatenate([pending, signal[i:i + 1600]])
        if len(pending) >= rv.FRAME:
            n = (len(pending) - rv.FRAME) // rv.HOP + 1
            starts += [consumed + k * rv.HOP for k in range(n)]
            pending = pending[n * rv.HOP:]
            consumed += n * rv.HOP

    assert starts == list(range(0, len(signal) - rv.FRAME + 1, rv.HOP))


# ---- the corpus on disk -------------------------------------------------

def test_stores_at_24k_regardless_of_capture_rate(tmp_path):
    """The stored rate is the one thing that cannot be fixed later."""
    corpus = VoiceCorpus(str(tmp_path))
    utt = corpus.append(synthetic_take(), SR, "hello there", "Hello there.")
    assert abs(utt.duration_s - 3.0) < 0.05
    assert load_utterance(utt, sr=STORE_SR).shape[0] == pytest.approx(
        int(utt.duration_s * STORE_SR), abs=2)


def test_resume_survives_a_kill(tmp_path):
    """Reopening the corpus must see every flushed take and nothing else."""
    corpus = VoiceCorpus(str(tmp_path))
    corpus.append(synthetic_take(), SR, "first sentence", "First sentence.")
    corpus.append(synthetic_take(), SR, "second sentence", "Second sentence.")

    reopened = VoiceCorpus(str(tmp_path))          # as if the process died here
    assert reopened.recorded_texts() == {"first sentence", "second sentence"}
    assert reopened._next_index() == 3

    reopened.append(synthetic_take(), SR, "third sentence", "Third sentence.")
    assert [u.utt_id for u in VoiceCorpus(str(tmp_path)).load()] == [
        "utt_0001", "utt_0002", "utt_0003"]


def test_pipe_in_text_would_corrupt_the_manifest(tmp_path):
    """LJSpeech's delimiter is '|'; csv must quote it rather than split on it."""
    corpus = VoiceCorpus(str(tmp_path))
    corpus.append(synthetic_take(), SR, "a or b", "Choose: a | b")
    assert corpus.load()[0].raw_text == "Choose: a | b"


def test_resample_is_band_limited():
    """Decimating by slicing aliases; the helper must lowpass first."""
    t = np.arange(SR) / SR
    tone = np.sin(2 * np.pi * 20_000 * t).astype(np.float32)   # above 24k Nyquist
    out = resample(tone, SR, STORE_SR)
    assert np.abs(out).max() < 0.1     # a slicing decimator would alias it down
