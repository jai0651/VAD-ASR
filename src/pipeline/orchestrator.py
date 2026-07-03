"""
The session orchestrator: one state machine per connected user.

This file is the actual "voice agent". Everything else is a stage; this is
the policy that wires stages into a conversation:

    LISTENING ──speech confirmed──▶ USER_SPEAKING ──endpoint──▶ THINKING
        ▲                                                          │ ASR+respond
        │                                                          ▼
        └───────────reply finished / interrupted◀──────────── SPEAKING

Two production behaviors live here and nowhere else:

  BARGE-IN. Users interrupt. If confirmed speech arrives while the agent is
  THINKING or SPEAKING, we cancel the in-flight reply task, tell the client to
  flush its playback buffer, and treat the new speech as the next turn. A
  voice agent without barge-in feels like an IVR menu from 2005.

  NEVER BLOCK THE EVENT LOOP. VAD is sub-millisecond so it runs inline on
  each audio frame, but ASR and TTS are hundreds of ms of CPU — they run in
  worker threads (asyncio.to_thread). If they ran on the loop, the WebSocket
  would stop draining mic audio and barge-in would be impossible: the agent
  literally could not hear you while it was talking.

The orchestrator is transport-agnostic: it takes two async callbacks
(emit_json, emit_audio) instead of a WebSocket, so tests drive it directly.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Awaitable, Callable, Iterator

import numpy as np

from src.pipeline.audio import float32_to_pcm16, pcm16_to_float32
from src.pipeline.config import PipelineConfig
from src.pipeline.endpointing import EndpointDetector, EndpointEvent
from src.pipeline.metrics import Metrics


class State(str, Enum):
    LISTENING = "listening"
    USER_SPEAKING = "user_speaking"
    THINKING = "thinking"
    SPEAKING = "speaking"


class VoiceSession:
    def __init__(
        self,
        *,
        vad,                     # SileroVAD-like: push(np.f32) -> [VADResult]
        asr,                     # WhisperASR-like: transcribe(np.f32) -> Transcript
        tts,                     # KokoroTTS-like: synthesize(str) -> Iterator[TTSChunk]
        responder,               # Responder: respond(str) -> str
        cfg: PipelineConfig,
        metrics: Metrics,
        emit_json: Callable[[dict], Awaitable[None]],
        emit_audio: Callable[[bytes], Awaitable[None]],
    ):
        self.vad = vad
        self.asr = asr
        self.tts = tts
        self.responder = responder
        self.cfg = cfg
        self.metrics = metrics
        self.emit_json = emit_json
        self.emit_audio = emit_audio

        self.state = State.LISTENING
        self.endpointer = EndpointDetector(
            sample_rate=cfg.sample_rate,
            # The VAD engine defines the decision rate (your VADNet: one prob
            # per 10 ms hop; Silero: per 32 ms window). The endpointer converts
            # its ms-based thresholds using this.
            window=vad.window,
            start_threshold=cfg.vad_start_threshold,
            end_threshold=cfg.vad_end_threshold,
            start_trigger_ms=cfg.start_trigger_ms,
            end_silence_ms=cfg.end_silence_ms,
            pre_roll_ms=cfg.pre_roll_ms,
            min_utterance_ms=cfg.min_utterance_ms,
            max_utterance_s=cfg.max_utterance_s,
        )
        self._reply_task: asyncio.Task | None = None
        self._utterance_ended_at: float = 0.0
        self._probs: list[float] = []
        self._last_vad_emit: float = 0.0

    async def start(self) -> None:
        await self._set_state(State.LISTENING)
        await self.emit_json({
            "type": "ready",
            "input": {"sample_rate": self.cfg.sample_rate, "format": "pcm16"},
            "engines": {
                "vad": self.cfg.vad_engine, "asr": self.cfg.asr_engine,
                "tts": self.cfg.tts_engine, "responder": self.cfg.responder,
            },
        })

    async def feed(self, pcm: bytes) -> None:
        """Entry point for every mic frame the client sends."""
        for res in self.vad.push(pcm16_to_float32(pcm)):
            # Live VAD telemetry (~2/s): the single most useful debugging
            # signal in the whole pipeline. A VAD pinned high (mic noise reads
            # as speech) or low (too quiet) explains "it never responds"
            # instantly — watch this number in the client while you talk.
            self._probs.append(res.prob)
            now = time.perf_counter()
            if now - self._last_vad_emit > 0.5:
                self._last_vad_emit = now
                await self.emit_json({
                    "type": "vad",
                    "prob": round(sum(self._probs) / len(self._probs), 2),
                })
                self._probs.clear()

            endpoint = self.endpointer.update(res.samples, res.prob)
            if endpoint is None:
                continue

            if endpoint.event == EndpointEvent.UTTERANCE_START:
                await self._on_speech_start()
            elif endpoint.event == EndpointEvent.UTTERANCE_DISCARD:
                self.metrics.count("utterances_discarded")
                await self._set_state(State.LISTENING)
            elif endpoint.event == EndpointEvent.UTTERANCE_END:
                self.metrics.count("utterances")
                self.metrics.observe("utterance_ms", endpoint.duration_ms)
                self._utterance_ended_at = time.perf_counter()
                await self.emit_json({
                    "type": "speech_end", "duration_ms": round(endpoint.duration_ms),
                })
                self._reply_task = asyncio.create_task(
                    self._handle_utterance(endpoint.audio)
                )

    async def _on_speech_start(self) -> None:
        # Barge-in: the user started talking over the agent's turn.
        if self.state in (State.THINKING, State.SPEAKING) and self._reply_task:
            self._reply_task.cancel()
            self._reply_task = None
            self.metrics.count("barge_ins")
            # Client must drop any queued-but-unplayed audio immediately.
            await self.emit_json({"type": "interrupted"})
        await self._set_state(State.USER_SPEAKING)
        await self.emit_json({"type": "speech_start"})

    async def _handle_utterance(self, audio: np.ndarray) -> None:
        try:
            await self._set_state(State.THINKING)

            transcript = await asyncio.to_thread(self.asr.transcribe, audio)
            self.metrics.observe("asr_ms", transcript.latency_ms)
            await self.emit_json({
                "type": "transcript",
                "text": transcript.text,
                "audio_s": round(transcript.audio_s, 2),
                "latency_ms": round(transcript.latency_ms),
                "rtf": round(transcript.rtf, 3),
            })
            if not transcript.text:
                await self._set_state(State.LISTENING)
                return

            t0 = time.perf_counter()
            reply = await asyncio.to_thread(self.responder.respond, transcript.text)
            self.metrics.observe("respond_ms", (time.perf_counter() - t0) * 1000)
            await self.emit_json({"type": "reply", "text": reply})

            await self._set_state(State.SPEAKING)
            await self._speak(reply)
            await self._set_state(State.LISTENING)
        except asyncio.CancelledError:
            raise  # barge-in: state already advanced by _on_speech_start
        except Exception as e:  # a failed turn must not kill the session
            self.metrics.count("turn_errors")
            await self.emit_json({"type": "error", "message": str(e)})
            await self._set_state(State.LISTENING)

    async def _speak(self, reply: str) -> None:
        """Stream TTS sentence by sentence; first chunk closes the latency clock."""
        gen: Iterator = self.tts.synthesize(reply)
        first = True
        while True:
            # next() runs a full sentence synthesis — off the loop it goes.
            chunk = await asyncio.to_thread(next, gen, None)
            if chunk is None:
                break
            if first:
                first = False
                # The metric users feel: silence between them stopping and
                # the agent starting to answer.
                turnaround = (time.perf_counter() - self._utterance_ended_at) * 1000
                self.metrics.observe("turnaround_ms", turnaround)
                await self.emit_json({
                    "type": "tts_start",
                    "sample_rate": chunk.sample_rate,
                    "turnaround_ms": round(turnaround),
                })
            self.metrics.observe("tts_sentence_ms", chunk.latency_ms)
            await self.emit_audio(float32_to_pcm16(chunk.samples))
            await self.emit_json({
                "type": "tts_sentence",
                "text": chunk.text,
                "latency_ms": round(chunk.latency_ms),
            })
        await self.emit_json({"type": "reply_done"})

    async def _set_state(self, state: State) -> None:
        if state != self.state:
            self.state = state
            await self.emit_json({"type": "state", "state": state.value})

    async def close(self) -> None:
        if self._reply_task:
            self._reply_task.cancel()
        self.vad.reset()
