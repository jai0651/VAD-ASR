"""
YOUR models (Modules 0-4) adapted to the production pipeline's engine
interfaces. Same orchestrator, same endpointer, same server — only the
weights change. Running these against the pretrained engines on the same
mic audio is the most honest lesson in the repo: it shows exactly what a
few hundred training steps on tiny data buys you versus 680k hours.

Interfaces implemented here:
    VAD engine:  .window (samples per decision), .push(f32) -> [VADResult], .reset()
    ASR engine:  .transcribe(f32 mono 16 kHz) -> Transcript
    TTS engine:  .synthesize(text) -> Iterator[TTSChunk]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.asr.decode import beam_search_decode
from src.asr.model import ASRModel
from src.audio.features import log_mel_spectrogram, mel_filterbank, power_spectrum
from src.pipeline.asr import Transcript
from src.pipeline.config import PipelineConfig
from src.pipeline.tts import TTSChunk, split_sentences
from src.pipeline.vad import VADResult
from src.vad.model import VADNet


def _require(path: str, train_hint: str) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"missing checkpoint {path} — train it first: {train_hint} "
            f"(or select a pretrained engine via VOICE_*_ENGINE)"
        )
    return p


class ScratchVAD:
    """Your Module 1 VADNet, run streaming.

    Emits one decision per 10 ms hop (window=160 samples) — your feature frame
    rate — instead of Silero's 32 ms. The endpointer doesn't care: it reads
    the engine's `window` and converts its ms thresholds accordingly.

    Streaming mechanics (same trick as src/vad/stream.py): each 25 ms analysis
    frame needs 400 samples, hops are 160, so we keep a 240-sample overlap tail;
    and we keep `context_frames` of recent mel frames so the conv stack sees the
    left-context it saw in training.
    """

    FRAME = 400   # 25 ms analysis window
    window = 160  # 10 ms hop = samples "consumed" per decision

    def __init__(self, cfg: PipelineConfig, context_frames: int = 12):
        if cfg.scratch_vad_ckpt == "auto":
            # Prefer the real-audio model (Module 1b) — the synthetic one is
            # known to pin high on live microphones (docs/02-vad.html §2.5).
            path = "outputs/vad_real.pt"
            if not Path(path).exists():
                path = "outputs/vad.pt"
        else:
            path = cfg.scratch_vad_ckpt
        ckpt = _require(path, "uv run python scripts/09_train_vad_real.py")
        print(f"[vad] scratch checkpoint: {ckpt}")
        self.model = VADNet(n_mels=80)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()
        self.n_fft = 512
        self._fb = mel_filterbank(80, self.n_fft, cfg.sample_rate)  # (80, 257)
        self._samples = np.zeros(0, dtype=np.float32)
        self.context = context_frames
        self._ctx = torch.zeros(0, 80)  # rolling mel-frame left-context

    @torch.no_grad()
    def push(self, samples: np.ndarray) -> list[VADResult]:
        self._samples = np.concatenate([self._samples, samples])
        hops: list[np.ndarray] = []
        frames: list[torch.Tensor] = []
        while self._samples.shape[0] >= self.FRAME:
            frames.append(torch.from_numpy(self._samples[: self.FRAME].copy()))
            hops.append(self._samples[: self.window].copy())
            self._samples = self._samples[self.window:]
        if not frames:
            return []

        # Module 0 pipeline, one frame at a time: window -> power -> mel -> log.
        power = power_spectrum(torch.stack(frames), self.n_fft)   # (N, 257)
        mel = torch.log(power @ self._fb.T + 1e-10)               # (N, 80)

        # Score with left-context, keep only the new frames' outputs.
        window = torch.cat([self._ctx, mel], dim=0)
        logits = self.model(window.unsqueeze(0))[0]
        probs = torch.sigmoid(logits)[-mel.shape[0]:]
        self._ctx = window[-self.context:]

        return [
            VADResult(prob=float(p), samples=hop)
            for p, hop in zip(probs.tolist(), hops)
        ]

    def reset(self) -> None:
        self._samples = np.zeros(0, dtype=np.float32)
        self._ctx = torch.zeros(0, 80)


class ScratchASR:
    """Your Module 2 CTC BiGRU + your prefix beam search decoder."""

    def __init__(self, cfg: PipelineConfig, beam_width: int = 12):
        ckpt = _require(
            cfg.scratch_asr_ckpt, "uv run python scripts/05_train_asr_libri.py"
        )
        self.cfg = cfg
        self.beam_width = beam_width
        # hidden=256 matches scripts/05; the checkpoint fixes the shape anyway.
        self.model = ASRModel(n_mels=80, hidden=256)
        self.model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.model.eval()

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray) -> Transcript:
        t0 = time.perf_counter()
        # EXACTLY the features used in training (librispeech.py): 25/10 ms, 80 mel.
        feats = log_mel_spectrogram(torch.from_numpy(audio), sr=self.cfg.sample_rate)
        log_probs = self.model(feats.unsqueeze(0))[0]
        text = beam_search_decode(log_probs, self.beam_width)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return Transcript(
            text=text.strip(),
            language="en",
            audio_s=audio.shape[0] / self.cfg.sample_rate,
            latency_ms=latency_ms,
        )


class ConformerASR:
    """Module 6: your Conformer hybrid CTC/attention recognizer.

    Same interface as ScratchASR, but two things are genuinely different:

      IT CAN STREAM. Trained with dynamic chunk masking, so `chunk_size` picks
      a point on the latency/accuracy curve at inference with no retraining.
      0 = offline (full context, best WER); 16 = 640 ms of lookahead.
      Module 2's BiGRU could not do this at any setting — bidirectional means
      the first frame's output depends on the last frame's input.

      IT RESCORES. CTC prefix beam proposes n-best, the attention decoder picks
      among them in one batched forward. That second opinion is the part CTC
      structurally cannot provide itself (docs/10-modern-asr.html).
    """

    def __init__(self, cfg: PipelineConfig, beam_size: int = 8):
        from src.asr.hybrid import HybridCTCAttention
        from src.asr.tokenizer import BPETokenizer

        ckpt = _require(
            cfg.conformer_ckpt, "uv run python scripts/11_train_asr_conformer.py"
        )
        state = torch.load(ckpt, map_location="cpu")
        self.model = HybridCTCAttention.from_state_dict(state)
        tok_path = cfg.conformer_tokenizer or state.get("tokenizer", "")
        self.tokenizer = BPETokenizer.load(_require(
            tok_path, "uv run python scripts/11_train_asr_conformer.py"
        ))
        self.cfg = cfg
        self.beam_size = beam_size
        self.chunk_size = cfg.conformer_chunk
        self._logmel = None
        print(f"[asr] conformer checkpoint: {ckpt} "
              f"({self.tokenizer.vocab_size} subwords, "
              f"{'offline' if not self.chunk_size else f'{self.chunk_size*40} ms chunks'})")

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray) -> Transcript:
        from src.asr.corpus import LogMel, cmvn

        if self._logmel is None:
            self._logmel = LogMel()          # builds the filterbank once
        t0 = time.perf_counter()
        feats = cmvn(self._logmel(torch.from_numpy(np.ascontiguousarray(audio))))
        lens = torch.tensor([feats.shape[0]])
        hyp = self.model.recognize(
            feats.unsqueeze(0), lens, beam_size=self.beam_size,
            chunk_size=self.chunk_size,
        )[0]
        return Transcript(
            text=self.tokenizer.decode(hyp).strip(),
            language="en",
            audio_s=audio.shape[0] / self.cfg.sample_rate,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
        )


class ScratchTTS:
    """Your Module 4 Tacotron-mini (text -> mel) + Griffin-Lim (mel -> audio)."""

    def __init__(self, cfg: PipelineConfig):
        from src.tts.synthesis import Synthesizer  # lazy: pulls in Module 4

        ckpt = _require(cfg.scratch_tts_ckpt, "uv run python scripts/08_train_tts.py")
        self.cfg = cfg
        self.synth = Synthesizer(str(ckpt))

    def synthesize(self, text: str) -> Iterator[TTSChunk]:
        for sentence in split_sentences(text):
            t0 = time.perf_counter()
            wav, sr = self.synth.tts(sentence)
            yield TTSChunk(
                samples=wav.astype(np.float32),
                sample_rate=sr,
                text=sentence,
                latency_ms=(time.perf_counter() - t0) * 1000.0,
            )
