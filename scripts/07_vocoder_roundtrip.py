"""
Module 4, step 1: HEAR the information Module 0 throws away.

Round trip a real LibriSpeech utterance:

    audio -> log-mel (Module 0) -> Griffin-Lim inversion (Module 4) -> audio

No training anywhere. Writes outputs/07_roundtrip.wav next to the original —
listen to both. The reconstruction is intelligible but audibly "phasey" and
muffled: that gap is exactly (1) 257 FFT bins squashed to 80 mel bands and
(2) true phase replaced by Griffin-Lim's self-consistent guess. Neural
vocoders exist to close this gap.

    uv run python scripts/07_vocoder_roundtrip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.audio.features import log_mel_spectrogram
from src.tts.vocoder import log_mel_to_waveform

ROOT = Path(__file__).resolve().parent.parent
FLAC = ROOT / "data/LibriSpeech/dev-clean/84/121123/84-121123-0001.flac"


def main() -> None:
    audio, sr = sf.read(FLAC, dtype="float32")
    wav = torch.from_numpy(audio)
    print(f"original: {wav.shape[0]/sr:.2f}s of speech")

    log_mel = log_mel_spectrogram(wav, sr=sr)          # Module 0, forward
    print(f"log-mel:  {tuple(log_mel.shape)} (frames x mel bands)")

    recon = log_mel_to_waveform(log_mel, n_iters=60)   # Module 4, inverse
    out = ROOT / "outputs/07_roundtrip.wav"
    sf.write(out, recon.numpy(), sr)

    # Numeric sanity: the reconstruction's OWN mel should closely match the
    # target mel (Griffin-Lim converged), even though its waveform differs.
    # Compare only frames that contain speech energy: in silence the log floor
    # (log 1e-10 ≈ -23) just measures the noise floor, not convergence.
    mel2 = log_mel_spectrogram(recon, sr=sr)
    t = min(mel2.shape[0], log_mel.shape[0])
    voiced = log_mel[:t].mean(dim=1) > -5.0
    diff = mel2[:t][voiced] - log_mel[:t][voiced]
    # Output loudness is normalized (a global gain = a constant log offset),
    # so measure shape agreement, not absolute level: remove the median offset.
    mel_mae = (diff - diff.median()).abs().mean().item()
    print(f"mel-domain MAE on voiced frames (gain-invariant): {mel_mae:.3f} (log units)")
    print(f"wrote {out.relative_to(ROOT)} — listen and compare with:\n      {FLAC.relative_to(ROOT)}")
    assert mel_mae < 0.5, "Griffin-Lim did not converge to a consistent signal"
    print("PASS")


if __name__ == "__main__":
    main()
