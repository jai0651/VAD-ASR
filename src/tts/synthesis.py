"""
Module 4, part 4: glue — checkpoint in, "say this sentence" out.

Chains the two halves and undoes training-time normalization in exactly the
reverse order it was applied:

    text --encode--> ids --TacotronMini--> normalized mel
         --de-normalize (stats from the checkpoint)--> log-mel
         --Griffin-Lim (vocoder.py)--> waveform @ 16 kHz
"""

from __future__ import annotations

import numpy as np
import torch

from src.asr.text import encode
from src.tts.model import TacotronMini
from src.tts.vocoder import SR, log_mel_to_waveform


class Synthesizer:
    def __init__(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.model = TacotronMini(r=ckpt["r"])
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.mean = ckpt["mel_mean"]
        self.std = ckpt["mel_std"]

    @torch.no_grad()
    def tts(self, text: str, max_frames: int = 1000) -> tuple[np.ndarray, int]:
        """Sentence -> (float32 waveform, sample_rate)."""
        ids = torch.tensor([encode(text)], dtype=torch.long)
        if ids.shape[1] == 0:
            return np.zeros(0, dtype=np.float32), SR
        mel_norm, _aligns = self.model.generate(ids, max_frames=max_frames)
        log_mel = mel_norm * self.std + self.mean       # undo z-normalization
        wav = log_mel_to_waveform(log_mel, n_iters=60)
        return wav.numpy().astype(np.float32), SR
