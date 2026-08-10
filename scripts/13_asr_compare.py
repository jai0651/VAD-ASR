"""
The head-to-head: Module 2 vs Module 6 vs production, on identical audio.

Every claim in docs/10-modern-asr.html reduces to one question — did replacing
the architecture actually help? — and the only honest way to answer it is to run
all three recognizers over the same held-out utterances and print one table.

  scratch     Module 2: 2xConv1d + BiGRU, CTC, 28 characters, no LM
  conformer   Module 6: Conformer + hybrid CTC/attention, BPE subwords
  whisper     faster-whisper base.en int8 — 680k hours of training data

Both WER and CER are reported, because they say different things. CER flatters
a model ("recogniton" is 1/10 characters but 1/1 words) and it is the metric
Module 2 was originally scored on, so it keeps the comparison continuous with
the older docs. WER is what the literature reports and what a user feels.

Run:
  uv run python scripts/13_asr_compare.py
  N=100 WHISPER=1 uv run python scripts/13_asr_compare.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.corpus import build_manifest  # noqa: E402
from src.asr.decode import char_error_rate  # noqa: E402
from src.asr.search import corpus_wer  # noqa: E402
from src.asr.tokenizer import normalize  # noqa: E402
from src.pipeline.config import PipelineConfig  # noqa: E402

N = int(os.environ.get("N", "60"))
WITH_WHISPER = os.environ.get("WHISPER", "1") == "1"
SPLIT = os.environ.get("SPLIT", "dev-clean")


def build_engines(cfg: PipelineConfig) -> dict:
    engines: dict = {}
    if Path(cfg.scratch_asr_ckpt).exists():
        from src.pipeline.scratch_engines import ScratchASR

        engines["scratch (M2)"] = ScratchASR(cfg)
    if Path(cfg.conformer_ckpt).exists():
        from src.pipeline.scratch_engines import ConformerASR

        engines["conformer (M6)"] = ConformerASR(cfg)
        # Same weights, streaming decode — the latency/accuracy dial.
        streaming = PipelineConfig(conformer_chunk=16)
        engines["conformer stream"] = ConformerASR(streaming)
    if WITH_WHISPER:
        from src.pipeline.asr import WhisperASR

        engines["whisper base.en"] = WhisperASR(cfg)
    return engines


def main() -> None:
    cfg = PipelineConfig()
    items = build_manifest(split=SPLIT)
    # SAME held-out split the trainer used, so we never score on training audio.
    rng = random.Random(1234)
    rng.shuffle(items)
    dev = items[: max(1, min(300, len(items) // 10))][:N]
    print(f"{len(dev)} held-out {SPLIT} utterances "
          f"({sum(i['duration'] for i in dev)/60:.1f} min)\n")

    engines = build_engines(cfg)
    if not engines:
        print("no engines available — train something first")
        return

    rows = []
    for name, engine in engines.items():
        pairs, cers, secs, audio_s = [], [], 0.0, 0.0
        for it in dev:
            audio, sr = sf.read(it["path"], dtype="float32")
            ref = normalize(it["text"])
            t0 = time.perf_counter()
            hyp = normalize(engine.transcribe(audio).text)
            secs += time.perf_counter() - t0
            audio_s += len(audio) / sr
            pairs.append((hyp, ref))
            cers.append(char_error_rate(hyp, ref))
        rows.append((name, corpus_wer(pairs), float(np.mean(cers)), secs / audio_s,
                     pairs[0]))

    print(f"\n{'engine':20s} {'WER':>7s} {'CER':>7s} {'RTF':>7s}")
    print("-" * 45)
    for name, wer, cer, rtf, _ in rows:
        print(f"{name:20s} {wer:7.3f} {cer:7.3f} {rtf:7.3f}")

    print("\nsame utterance, every engine:")
    print(f"  ref  {rows[0][4][1]}")
    for name, _, _, _, (hyp, _) in rows:
        print(f"  {name:18s} {hyp}")
    print("\nRTF = seconds of CPU per second of audio (lower is faster).")


if __name__ == "__main__":
    main()
