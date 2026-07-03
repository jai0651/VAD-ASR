"""
Module 4: Text-to-Speech from scratch.

TTS is ASR's mirror image, split into two learnable problems:

    text ──(acoustic model)──▶ log-mel spectrogram ──(vocoder)──▶ waveform

    model.py    the acoustic model: a Tacotron-style attention seq2seq that
                *writes* a spectrogram frame by frame
    vocoder.py  Griffin-Lim: turns a spectrogram back into audio with zero
                training — pure DSP, the exact inverse of Module 0
    data.py     (text, mel) pairs from a single LibriSpeech speaker
    synthesis.py glue: checkpoint -> "say this sentence" -> waveform

Historically real systems were exactly this pair (Tacotron2 + Griffin-Lim,
then + WaveNet/HiFi-GAN vocoders); modern production models (VITS, StyleTTS2,
Kokoro) fuse the two stages end to end. Build the two-stage version and the
fused ones stop being magic.
"""
