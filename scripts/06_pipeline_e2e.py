"""
Module 3 smoke test: the production pipeline, end to end, no microphone.

We stream a real LibriSpeech utterance through the same code path the server
uses — VAD -> endpointer -> ASR -> responder -> TTS — in 32 ms chunks, exactly
as if a client were sending it live. Compares the transcript against
LibriSpeech's reference text and writes the spoken reply to outputs/06_reply.wav.

Engines come from the config, so this doubles as the A/B harness:

    # your from-scratch models (the default):
    uv run python scripts/06_pipeline_e2e.py
    # the production stack (first run downloads whisper ~145 MB, kokoro ~330 MB):
    VOICE_VAD_ENGINE=silero VOICE_ASR_ENGINE=whisper VOICE_TTS_ENGINE=kokoro \
        uv run python scripts/06_pipeline_e2e.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.decode import char_error_rate
from src.pipeline.config import PipelineConfig
from src.pipeline.endpointing import EndpointDetector, EndpointEvent
from src.pipeline.engines import make_asr, make_responder, make_tts, make_vad

ROOT = Path(__file__).resolve().parent.parent
FLAC = ROOT / "data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac"
TRANS = ROOT / "data/LibriSpeech/dev-clean/84/121123/84-121123.trans.txt"


def reference_text(utt_id: str) -> str:
    for line in TRANS.read_text().splitlines():
        key, _, text = line.partition(" ")
        if key == utt_id:
            return text.lower()
    raise KeyError(utt_id)


def main() -> None:
    cfg = PipelineConfig()
    print(f"engines: denoise={cfg.denoise_engine}({cfg.denoise_target}) "
          f"vad={cfg.vad_engine} asr={cfg.asr_engine} tts={cfg.tts_engine}")
    print("[1/4] loading models...")
    vad = make_vad(cfg)
    asr = make_asr(cfg)
    tts = make_tts(cfg)
    responder = make_responder(cfg)
    endpointer = EndpointDetector(
        sample_rate=cfg.sample_rate, window=vad.window,
        start_threshold=cfg.vad_start_threshold, end_threshold=cfg.vad_end_threshold,
        start_trigger_ms=cfg.start_trigger_ms, end_silence_ms=cfg.end_silence_ms,
        pre_roll_ms=cfg.pre_roll_ms, min_utterance_ms=cfg.min_utterance_ms,
        max_utterance_s=cfg.max_utterance_s,
    )

    audio, sr = sf.read(FLAC, dtype="float32")
    assert sr == cfg.sample_rate
    # Pad with a second of QUIET NOISE (not digital zeros) so the endpointer
    # can close the turn. Real mics always have a noise floor; exact zeros are
    # out-of-distribution for the scratch VAD (trained on synthetic audio that
    # always had background noise) and make its probability flicker — a real
    # example of train/serve domain mismatch.
    rng = np.random.default_rng(0)
    audio = np.concatenate([audio, (0.005 * rng.standard_normal(sr)).astype(np.float32)])

    print(f"[2/4] streaming {audio.shape[0]/sr:.1f}s of audio in 32 ms chunks...")
    utterance = None
    t0 = time.perf_counter()
    for i in range(0, audio.shape[0], cfg.vad_window):
        for res in vad.push(audio[i : i + cfg.vad_window]):
            ep = endpointer.update(res.samples, res.prob)
            if ep and ep.event == EndpointEvent.UTTERANCE_START:
                print(f"    speech started at ~{i/sr:.2f}s")
            if ep and ep.event == EndpointEvent.UTTERANCE_END:
                utterance = ep.audio
                print(f"    endpoint: utterance of {ep.duration_ms/1000:.2f}s captured")
    vad_ms = (time.perf_counter() - t0) * 1000
    assert utterance is not None, "endpointer never closed an utterance"
    print(f"    VAD+endpointing walltime for whole file: {vad_ms:.0f} ms")

    print("[3/4] transcribing...")
    transcript = asr.transcribe(utterance)
    ref = reference_text(FLAC.stem)
    cer = char_error_rate(transcript.text.lower().strip(" ."), ref)
    print(f"    {cfg.asr_engine}: {transcript.latency_ms:.0f} ms, RTF {transcript.rtf:.2f}")
    print(f"    hyp: {transcript.text}")
    print(f"    ref: {ref}")
    print(f"    CER vs reference: {cer:.3f}")

    print("[4/4] synthesizing reply...")
    reply = responder.respond(transcript.text or "i heard nothing")
    chunks = []
    first_ms = None
    t0 = time.perf_counter()
    for chunk in tts.synthesize(reply):
        if first_ms is None:
            first_ms = (time.perf_counter() - t0) * 1000
        chunks.append(chunk)
    wav = np.concatenate([c.samples for c in chunks])
    out = ROOT / "outputs/06_reply.wav"
    sf.write(out, wav, chunks[0].sample_rate)
    print(f"    {cfg.tts_engine}: first audio in {first_ms:.0f} ms, "
          f"{len(chunks)} sentence chunk(s), {wav.shape[0]/chunks[0].sample_rate:.1f}s "
          f"-> {out.relative_to(ROOT)}")

    if cfg.asr_engine == "whisper":
        # Whisper should nail dev-clean; your scratch model's (high) CER is
        # reported above — the gap IS the lesson, not a failure.
        assert cer < 0.15, f"CER unexpectedly high: {cer}"
    print("\nPASS — full pipeline works end to end.")


if __name__ == "__main__":
    main()
