"""
Orchestrator tests with fake engines — no model downloads, no network.

The point of making VoiceSession depend on *interfaces* (push/transcribe/
synthesize/respond) instead of concrete classes is exactly this: the entire
conversation policy — turn taking, event ordering, barge-in — is verifiable
in milliseconds.
"""

from __future__ import annotations

import asyncio
import time

import numpy as np

from src.pipeline.audio import float32_to_pcm16
from src.pipeline.config import PipelineConfig
from src.pipeline.metrics import Metrics
from src.pipeline.orchestrator import State, VoiceSession
from src.pipeline.vad import VADResult

SR, WIN = 16_000, 512


class EnergyVAD:
    """Stands in for a real VAD engine: prob is just 'is this window loud?'."""

    window = WIN  # engines expose their decision rate; endpointer reads it

    def __init__(self):
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, samples):
        self._buf = np.concatenate([self._buf, samples])
        out = []
        while self._buf.shape[0] >= WIN:
            win, self._buf = self._buf[:WIN], self._buf[WIN:]
            rms = float(np.sqrt((win**2).mean()))
            out.append(VADResult(prob=0.95 if rms > 0.1 else 0.02, samples=win))
        return out

    def reset(self):
        self._buf = np.zeros(0, dtype=np.float32)


class FakeASR:
    def __init__(self, text="hello world"):
        self.text = text

    def transcribe(self, audio):
        from src.pipeline.asr import Transcript

        return Transcript(
            text=self.text, language="en",
            audio_s=audio.shape[0] / SR, latency_ms=1.0,
        )


class FakeTTS:
    def __init__(self, sentence_delay_s=0.0, n_sentences=2):
        self.delay = sentence_delay_s
        self.n = n_sentences

    def synthesize(self, text):
        for i in range(self.n):
            time.sleep(self.delay)
            yield type(
                "Chunk", (), {
                    "samples": np.zeros(2400, dtype=np.float32),
                    "sample_rate": 24_000,
                    "text": f"sentence {i}",
                    "latency_ms": self.delay * 1000,
                },
            )()


class FakeResponder:
    def respond(self, transcript):
        return f"echo: {transcript}"


def make_session(events, audio_out, tts=None, asr=None):
    cfg = PipelineConfig(
        start_trigger_ms=96, end_silence_ms=320, pre_roll_ms=96,
        min_utterance_ms=200, max_utterance_s=5.0,
    )

    async def emit_json(ev):
        events.append(ev)

    async def emit_audio(b):
        audio_out.append(b)

    return VoiceSession(
        vad=EnergyVAD(), asr=asr or FakeASR(), tts=tts or FakeTTS(),
        responder=FakeResponder(), cfg=cfg, metrics=Metrics(),
        emit_json=emit_json, emit_audio=emit_audio,
    )


def loud(n_windows):
    return float32_to_pcm16(np.full(n_windows * WIN, 0.5, dtype=np.float32))


def silence(n_windows):
    return float32_to_pcm16(np.zeros(n_windows * WIN, dtype=np.float32))


def event_types(events):
    return [e["type"] for e in events]


async def test_full_turn_event_order():
    events, audio_out = [], []
    s = make_session(events, audio_out)
    await s.start()

    await s.feed(loud(20))       # 640 ms of speech
    await s.feed(silence(15))    # 480 ms silence -> endpoint fires
    assert s._reply_task is not None
    await s._reply_task

    types = event_types(events)
    # The canonical happy path, in order:
    for expected in ["ready", "speech_start", "speech_end",
                     "transcript", "reply", "tts_start", "reply_done"]:
        assert expected in types, f"missing {expected} in {types}"
    assert types.index("speech_start") < types.index("speech_end")
    assert types.index("transcript") < types.index("reply") < types.index("tts_start")

    assert [e for e in events if e["type"] == "transcript"][0]["text"] == "hello world"
    assert [e for e in events if e["type"] == "reply"][0]["text"] == "echo: hello world"
    assert len(audio_out) == 2          # one binary frame per fake sentence
    assert s.state == State.LISTENING   # back to idle, ready for next turn


async def test_barge_in_cancels_reply_and_flushes_client():
    events, audio_out = [], []
    s = make_session(events, audio_out, tts=FakeTTS(sentence_delay_s=0.2, n_sentences=10))
    await s.start()

    await s.feed(loud(20))
    await s.feed(silence(15))    # endpoint -> reply task starts

    # Wait until the agent is actually speaking...
    for _ in range(100):
        if s.state == State.SPEAKING:
            break
        await asyncio.sleep(0.01)
    assert s.state == State.SPEAKING

    # ...then talk over it.
    await s.feed(loud(4))        # > start_trigger -> barge-in

    assert "interrupted" in event_types(events)
    assert s.state == State.USER_SPEAKING
    # The 10-sentence reply was abandoned early.
    await asyncio.sleep(0.05)
    assert len(audio_out) < 10
    assert "reply_done" not in event_types(events)


async def test_empty_transcript_skips_reply():
    events, audio_out = [], []
    s = make_session(events, audio_out, asr=FakeASR(text=""))
    await s.start()

    await s.feed(loud(20))
    await s.feed(silence(15))
    await s._reply_task

    types = event_types(events)
    assert "transcript" in types
    assert "reply" not in types and "tts_start" not in types
    assert s.state == State.LISTENING


async def test_too_short_speech_is_discarded():
    events, audio_out = [], []
    s = make_session(events, audio_out)
    await s.start()

    await s.feed(loud(3))        # ~96 ms: triggers start but under min_utterance
    await s.feed(silence(15))

    types = event_types(events)
    assert "speech_start" in types
    assert "transcript" not in types
    assert s._reply_task is None
    assert s.state == State.LISTENING
