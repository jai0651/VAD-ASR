"""
Module 8, part 3: rejecting bad takes at the microphone, not at the loss curve.

Every defect that reaches the corpus becomes a defect the model learns. A clipped
plosive teaches distortion; a truck outside teaches rumble; a misread word
teaches a wrong grapheme-to-sound mapping that you will hear forever in that
context. And you will not notice any of it during training — the loss goes down
smoothly regardless, because the model is faithfully learning exactly what you
gave it. Data QC is the only place these are cheap to catch.

So each take is run past four gates before it is allowed onto disk:

  CLIPPING. Irreversible. Once a sample hits +/-1.0 the waveform is flattened and
  no denoiser recovers it. Checked as a COUNT of samples at the rail, not just
  the peak: a single sample at 0.999 is fine, two hundred of them is a squashed
  plosive.

  NOISE FLOOR. Measured as the gap between speech-frame and silence-frame RMS,
  using your Module 1 VAD to decide which frames are which. Below ~20 dB you are
  recording the room as much as yourself, and the model will synthesise the room
  back to you — TTS reproduces the noise floor of its training data faithfully,
  because from the model's point of view the hiss is part of your voice.

  SILENCE PADDING. Leading and trailing silence is trimmed to a fixed 100 ms.
  Not cosmetic: the acoustic model's stop token learns "how long after the last
  word does audio end", and if that gap varies from 50 ms to 2 s across the
  corpus, the stop head learns a smear and inference either truncates the last
  syllable or trails off into invented silence.

  THE READ ITSELF. Transcribe the take and compare to the prompt. Catches
  misreads, skipped words, and "the" for "a" — the errors your ear glides over
  precisely because you know what you meant to say.

WHY THE VERIFIER IS WHISPER AND NOT YOUR OWN ASR.
The natural instinct is to close the loop with the Module 6 Conformer. It does
not work, and the reason is worth internalising: that model scores CER 0.276 on
LibriSpeech and will do worse on an unseen speaker through an unseen microphone.
A checker with a 30% error rate cannot detect a 5% error rate — it would flag
most of your good takes, you would re-record them, and the harness would make
the corpus worse while appearing rigorous. A VERIFIER MUST BE MORE ACCURATE THAN
THE THING IT VERIFIES. Whisper base.en (CER 0.017 here, `scripts/13_asr_compare.py`)
clears that bar; your Conformer does not. It stays available behind
`--asr scratch` because watching it produce false alarms teaches the point
better than this paragraph does.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from src.asr.decode import char_error_rate
from src.asr.tokenizer import normalize
from src.audio.features import log_mel_spectrogram
from src.tts.corpus import MODEL_SR, resample
from src.vad.stream import HysteresisGate, segments_from_decisions

CLIP_LEVEL = 0.99
MAX_CLIPPED_SAMPLES = 8      # a few rail hits are the ADC, hundreds are distortion
MIN_PEAK = 0.05              # quieter than this and you are too far from the mic
MIN_SNR_DB = 18.0            # calibrated, not guessed — see snr_db()
MAX_CER = 0.15
PAD_MS = 100.0
HOP_S = 0.01                 # log_mel_spectrogram's default 10 ms hop


@dataclass
class TakeQC:
    duration_s: float
    peak: float
    clipped_samples: int
    snr_db: float
    heard: str = ""
    cer: float = 0.0
    problems: list[str] = field(default_factory=list)   # hard rejects
    warnings: list[str] = field(default_factory=list)   # judgement calls

    @property
    def ok(self) -> bool:
        return not self.problems


def speech_mask(audio: np.ndarray, sr: int, vad) -> np.ndarray:
    """Per-frame speech decisions from the Module 1 VAD, at a 10 ms hop.

    Run offline over the whole take (not streaming) — we have all the audio, so
    there is no reason to hobble the model with left-context only. The
    hysteresis gate is still applied frame by frame because its debouncing is
    what turns a jittery probability into usable segment boundaries.
    """
    x = torch.from_numpy(resample(audio, sr, MODEL_SR))
    mel = log_mel_spectrogram(x, sr=MODEL_SR)
    with torch.no_grad():
        probs = torch.sigmoid(vad(mel.unsqueeze(0))[0])
    gate = HysteresisGate()
    return np.array([gate.update(float(p)) for p in probs], dtype=bool)


def trim_to_speech(audio: np.ndarray, sr: int, mask: np.ndarray,
                   pad_ms: float = PAD_MS) -> np.ndarray | None:
    """Cut to [first speech - pad, last speech + pad]. None if no speech found."""
    segs = segments_from_decisions(mask.tolist(), hop_s=HOP_S)
    if not segs:
        return None
    pad = pad_ms / 1000.0
    start = max(0.0, segs[0][0] - pad)
    end = min(len(audio) / sr, segs[-1][1] + pad)
    return audio[int(start * sr):int(end * sr)]


def snr_db(audio: np.ndarray, sr: int, mask: np.ndarray) -> float:
    """RMS(speech frames) - RMS(silence frames), in dB.

    Not a textbook SNR — the "noise" measurement includes any breath and room
    tone during pauses, which is exactly what we want to police. If a take has
    no silent frames at all we cannot measure it, so we return +inf rather than
    inventing a number: unmeasurable is not the same as clean, and the caller
    turns it into a warning.

    WHERE MIN_SNR_DB COMES FROM. Measured, not chosen: run this over 40
    LibriSpeech dev-clean utterances — studio-recorded read speech, the quality
    ceiling for a corpus like ours — and you get min 17.8 dB, median 26.6,
    p90 32.3. A threshold of 20 would therefore warn on 7% of genuinely clean
    takes. Over 900 prompts that is ~60 spurious prompts, which is how a
    warning gets trained out of a human's attention until it is ignored when it
    matters. 18 dB sits just under the observed floor of clean speech, so it
    fires on rooms that are actually bad.
    """
    n = min(len(mask), len(audio) // int(sr * HOP_S))
    if n == 0:
        return float("inf")
    frames = audio[:n * int(sr * HOP_S)].reshape(n, -1)
    rms = np.sqrt((frames.astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    speech, silence = rms[mask[:n]], rms[~mask[:n]]
    if speech.size == 0 or silence.size == 0:
        return float("inf")
    return float(20.0 * np.log10(speech.mean() / max(silence.mean(), 1e-12)))


def check_take(audio: np.ndarray, sr: int, prompt: str, vad,
               transcribe=None, min_s: float = 0.7, max_s: float = 18.0
               ) -> tuple[np.ndarray | None, TakeQC]:
    """Run every gate. Returns (trimmed audio or None, the report)."""
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    clipped = int((np.abs(audio) >= CLIP_LEVEL).sum())

    mask = speech_mask(audio, sr, vad)
    trimmed = trim_to_speech(audio, sr, mask)
    if trimmed is None:
        return None, TakeQC(0.0, peak, clipped, 0.0,
                            problems=["no speech detected"])

    # Re-derive the mask on the trimmed audio so SNR reflects what we keep.
    mask = speech_mask(trimmed, sr, vad)
    dur = len(trimmed) / sr
    qc = TakeQC(dur, peak, clipped, snr_db(trimmed, sr, mask))

    if clipped > MAX_CLIPPED_SAMPLES:
        qc.problems.append(f"clipped ({clipped} samples at the rail) — lower input gain")
    if peak < MIN_PEAK:
        qc.problems.append(f"too quiet (peak {peak:.3f}) — move closer or raise gain")
    if dur < min_s:
        qc.problems.append(f"too short ({dur:.1f}s)")
    if dur > max_s:
        qc.problems.append(f"too long ({dur:.1f}s) — was the VAD stuck on noise?")
    if qc.snr_db == float("inf"):
        qc.warnings.append("no silent frames — SNR unmeasurable")
    elif qc.snr_db < MIN_SNR_DB:
        qc.warnings.append(f"noisy room (SNR {qc.snr_db:.0f} dB, want >{MIN_SNR_DB:.0f})")

    if transcribe is not None:
        qc.heard = normalize(transcribe(resample(trimmed, sr, MODEL_SR)))
        qc.cer = char_error_rate(qc.heard, normalize(prompt))
        if qc.cer > MAX_CER:
            qc.warnings.append(f"misread? heard {qc.heard!r} (CER {qc.cer:.2f})")

    return trimmed, qc
