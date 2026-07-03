"""
Module 3: the production pipeline.

Everything under `src/pipeline` is the "how it's actually done" counterpart to
the from-scratch learning modules. Same concepts (VAD gating, endpointing,
CTC-style acoustic modeling, decoding) — but with battle-tested pretrained
models, a real streaming transport, and production concerns: latency budgets,
barge-in, metrics, and graceful failure.

Layers (each file is one layer, deliberately decoupled):

    audio.py        raw byte <-> float sample plumbing
    vad.py          Silero VAD — "is this 32 ms window speech?"
    endpointing.py  turn detection — "has the user finished their utterance?"
    asr.py          faster-whisper — utterance audio -> text
    responder.py    text -> reply text (pluggable "brain"; echo by default)
    tts.py          Kokoro-82M — reply text -> audio, streamed per sentence
    orchestrator.py the session state machine wiring all of the above
    metrics.py      per-stage latency tracking
    config.py       every tunable in one place, overridable via env vars
"""
