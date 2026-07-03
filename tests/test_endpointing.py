"""The endpointer is pure logic — every production behavior is asserted here."""

import numpy as np

from src.pipeline.endpointing import EndpointDetector, EndpointEvent

SR, WIN = 16_000, 512
WINDOW_MS = 1000 * WIN / SR  # 32 ms


def make_detector(**kw):
    defaults = dict(
        sample_rate=SR, window=WIN,
        start_trigger_ms=96, end_silence_ms=320,
        pre_roll_ms=96, min_utterance_ms=200, max_utterance_s=2.0,
    )
    defaults.update(kw)
    return EndpointDetector(**defaults)


def feed(det, probs, marker=None):
    """Feed one window per prob; window samples are constant `marker` values."""
    events = []
    for i, p in enumerate(probs):
        val = marker if marker is not None else float(i)
        ep = det.update(np.full(WIN, val, dtype=np.float32), p)
        if ep:
            events.append(ep)
    return events


def test_start_requires_sustained_speech():
    det = make_detector()  # 96 ms trigger = 3 windows
    assert feed(det, [0.9, 0.9]) == []           # 2 windows: not yet
    assert not det.in_speech
    evs = feed(det, [0.9])                        # 3rd consecutive: trigger
    assert [e.event for e in evs] == [EndpointEvent.UTTERANCE_START]
    assert det.in_speech


def test_blip_does_not_trigger():
    det = make_detector()
    assert feed(det, [0.9, 0.1, 0.9, 0.1, 0.9, 0.1]) == []  # never 3 in a row
    assert not det.in_speech


def test_end_after_sustained_silence_returns_audio_with_preroll():
    det = make_detector()
    # 300 ms of silence *before* speech: should survive via pre-roll.
    feed(det, [0.0] * 10, marker=-1.0)
    feed(det, [0.9] * 3, marker=1.0)              # trigger (these 3 count too)
    feed(det, [0.9] * 10, marker=1.0)             # body of the utterance
    evs = feed(det, [0.0] * 10, marker=0.0)       # 320 ms silence = 10 windows
    assert [e.event for e in evs] == [EndpointEvent.UTTERANCE_END]
    audio = evs[0].audio
    # Pre-roll: audio must include windows from BEFORE the trigger (-1 marker).
    assert (audio == -1.0).any()
    # And the speech itself.
    assert (audio == 1.0).any()
    # Trailing silence was trimmed: far fewer than all 10 silence windows kept.
    assert (audio == 0.0).sum() <= 3 * WIN
    assert not det.in_speech


def test_short_utterance_is_discarded():
    det = make_detector(min_utterance_ms=2000)
    feed(det, [0.9] * 3)
    evs = feed(det, [0.0] * 10)
    assert [e.event for e in evs] == [EndpointEvent.UTTERANCE_DISCARD]
    assert evs[0].audio is None


def test_max_utterance_force_closes():
    det = make_detector(max_utterance_s=0.5)      # 16 windows
    feed(det, [0.9] * 3)                          # trigger (3 windows buffered)
    evs = feed(det, [0.9] * 13)                   # never goes silent -> cap hits
    assert [e.event for e in evs] == [EndpointEvent.UTTERANCE_END]
    assert not det.in_speech


def test_hysteresis_mid_word_dip_does_not_end_turn():
    det = make_detector()
    feed(det, [0.9] * 3)
    # Dips to 0.4 stay above end_threshold (0.35): silence never accumulates.
    evs = feed(det, [0.9, 0.4, 0.9, 0.4, 0.9] * 10)
    assert evs == []
    assert det.in_speech


def test_detector_is_reusable_after_end():
    det = make_detector()
    evs = feed(det, [0.9] * 13 + [0.0] * 10)      # full turn: start ... end
    assert [e.event for e in evs] == [
        EndpointEvent.UTTERANCE_START, EndpointEvent.UTTERANCE_END,
    ]
    evs = feed(det, [0.9] * 3)                    # next turn triggers cleanly
    assert [e.event for e in evs] == [EndpointEvent.UTTERANCE_START]
