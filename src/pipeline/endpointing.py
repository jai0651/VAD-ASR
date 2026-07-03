"""
Endpointing (turn detection): from per-window VAD probabilities to utterances.

This is the layer beginners miss. VAD answers "is this 32 ms window speech?"
— but the product question is "has the user finished talking, so we can send
their utterance to the ASR?". That is a *policy* decision built on top of VAD:

  - Enter an utterance only after `start_trigger_ms` of sustained speech
    (debounce: a cough shouldn't open a turn).
  - Keep `pre_roll_ms` of audio from BEFORE the trigger. VAD confirmation is
    inherently late; without pre-roll the first phoneme gets clipped and the
    ASR mishears word onsets ("...eventy" instead of "seventy").
  - Close the utterance after `end_silence_ms` of sustained silence. This is
    THE latency/accuracy dial of a voice agent: too short cuts users off
    mid-sentence, too long feels laggy. 500-800 ms is the classic range;
    modern agents replace this fixed timeout with a semantic turn-detection
    model, but the fixed version is the baseline everyone ships first.
  - Hysteresis (same idea as Module 1's gate): a lower threshold to *stay* in
    speech than to *enter* it, so mid-word dips don't end the turn.
  - A hard `max_utterance_s` cap so a runaway stream can't buffer forever.

Pure logic, no models, no I/O — which is why it's fully unit-testable.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np


class EndpointEvent(Enum):
    UTTERANCE_START = auto()   # sustained speech confirmed
    UTTERANCE_END = auto()     # sustained silence confirmed -> audio ready for ASR
    UTTERANCE_DISCARD = auto() # turn ended but was too short to be real speech


@dataclass
class Endpoint:
    event: EndpointEvent
    audio: np.ndarray | None = None   # full utterance (pre-roll included) on END
    duration_ms: float = 0.0


@dataclass
class EndpointDetector:
    """Feed (window_samples, vad_prob) per 32 ms window; get turn events out."""

    sample_rate: int = 16_000
    window: int = 512
    start_threshold: float = 0.5
    end_threshold: float = 0.35
    start_trigger_ms: int = 96
    end_silence_ms: int = 700
    pre_roll_ms: int = 320
    min_utterance_ms: int = 250
    max_utterance_s: float = 30.0

    in_speech: bool = field(default=False, init=False)
    _speech_run: int = field(default=0, init=False)
    _silence_run: int = field(default=0, init=False)
    _pre_roll: deque = field(default=None, init=False)  # type: ignore[assignment]
    _utterance: list = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        self._window_ms = 1000.0 * self.window / self.sample_rate
        self._start_windows = max(1, round(self.start_trigger_ms / self._window_ms))
        self._end_windows = max(1, round(self.end_silence_ms / self._window_ms))
        self._max_windows = round(1000.0 * self.max_utterance_s / self._window_ms)
        pre_roll_windows = max(1, round(self.pre_roll_ms / self._window_ms))
        # Pre-roll must also cover the trigger run itself: by the time we
        # confirm speech, `start_trigger_ms` of it already went by.
        self._pre_roll = deque(maxlen=pre_roll_windows + self._start_windows)

    def update(self, samples: np.ndarray, prob: float) -> Endpoint | None:
        """Process one VAD window. Returns an Endpoint when state changes."""
        if not self.in_speech:
            self._pre_roll.append(samples)
            if prob >= self.start_threshold:
                self._speech_run += 1
                if self._speech_run >= self._start_windows:
                    self.in_speech = True
                    self._speech_run = 0
                    self._silence_run = 0
                    self._utterance = list(self._pre_roll)
                    return Endpoint(EndpointEvent.UTTERANCE_START)
            else:
                self._speech_run = 0
            return None

        # --- inside an utterance ---
        self._utterance.append(samples)
        if prob < self.end_threshold:
            self._silence_run += 1
            if self._silence_run >= self._end_windows:
                return self._close(trim_tail=True)
        else:
            self._silence_run = 0
        if len(self._utterance) >= self._max_windows:
            return self._close(trim_tail=False)
        return None

    def _close(self, trim_tail: bool) -> Endpoint:
        windows = self._utterance
        if trim_tail:
            # Drop most of the trailing silence; keep a little so the ASR sees
            # a natural utterance boundary (Whisper behaves better with it).
            keep = max(0, len(windows) - self._end_windows + 3)
            windows = windows[:keep]
        audio = (
            np.concatenate(windows) if windows else np.zeros(0, dtype=np.float32)
        )
        duration_ms = 1000.0 * audio.shape[0] / self.sample_rate

        self.in_speech = False
        self._silence_run = 0
        self._utterance = []
        self._pre_roll.clear()

        if duration_ms < self.min_utterance_ms:
            return Endpoint(EndpointEvent.UTTERANCE_DISCARD, duration_ms=duration_ms)
        return Endpoint(EndpointEvent.UTTERANCE_END, audio=audio, duration_ms=duration_ms)
