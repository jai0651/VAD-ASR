"""
Module 4, step 2: train the Tacotron-mini on one LibriSpeech speaker.

What to WATCH while it trains (this is the fun part):
  - The mel L1 falls quickly (easy: predict blurry averages).
  - The ALIGNMENT is the real event. Early on the attention map
    (outputs/08_alignment.png) is a smear — the model doesn't know which
    character it's saying. Somewhere in training a clean diagonal "clicks in";
    only after that do samples become speech-like. Watching that diagonal
    appear is watching the model discover that text is read left to right.
  - outputs/08_sample_stepN.wav: the same sentence synthesized as training
    progresses, from noise -> mumble -> intelligible (robotic) speech.

Like Module 2b, this OVERFITS a small set on purpose: it proves the machinery
(attention, stop token, vocoder) on hardware you have. Generalizing to
arbitrary text is a data problem (hours of single-speaker studio speech, e.g.
LJSpeech), not an algorithm change.

Run:
  uv run python scripts/08_train_tts.py
  STEPS=4000 BATCH=8 uv run python scripts/08_train_tts.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.tts.data import load_speaker_utterances, mel_statistics, tts_collate  # noqa: E402
from src.tts.model import TacotronMini, guided_attention_penalty  # noqa: E402
from src.tts.synthesis import Synthesizer  # noqa: E402

STEPS = int(os.environ.get("STEPS", "3000"))
BATCH = int(os.environ.get("BATCH", "8"))
R = 2
SAMPLE_TEXT = None  # set after data loads: first training sentence
OUT = Path("outputs")


def get_device() -> torch.device:
    # Deliberately CPU by default: this decoder is a python loop of tiny
    # matmuls, and on MPS each op pays a GPU dispatch cost that dwarfs the
    # math (~13x slower measured). GPUs win on big parallel batches, not on
    # long sequential chains of small ops.
    return torch.device(os.environ.get("DEVICE", "cpu"))


def save_alignment_plot(align: torch.Tensor, path: Path, step: int) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(align.T.cpu(), origin="lower", aspect="auto", interpolation="none")
    ax.set_xlabel("decoder step (time)")
    ax.set_ylabel("text position (character)")
    ax.set_title(f"attention alignment @ step {step} (want: clean diagonal)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    device = get_device()
    print(f"device: {device}")

    print("loading speaker-84 utterances + computing mels...")
    items = load_speaker_utterances(max_seconds=8.0)
    print(f"  {len(items)} utterances, "
          f"{sum(m.shape[0] for _, m in items) / 100 / 60:.1f} min of speech")
    mean, std = mel_statistics(items)
    sample_text = items[0][0]
    print(f"  sample sentence: {sample_text!r}")

    # ASR-as-judge: we transcribe our own synthesis with whisper and keep the
    # checkpoint with the best (lowest) CER on a few SEEN sentences. Generative
    # losses lie (see docs/04-tts.html); free-running quality oscillates between
    # checkpoints, so "last" is usually not "best". This is real production
    # practice (intelligibility evals + checkpoint selection).
    from src.asr.decode import char_error_rate
    from src.pipeline.asr import WhisperASR
    from src.pipeline.config import PipelineConfig

    judge = WhisperASR(PipelineConfig())
    judge_texts = [items[i][0] for i in (0, 3, 7, 10)]
    best_cer = float("inf")

    model = TacotronMini(r=R).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params/1e6:.1f}M params")
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=2000, gamma=0.5)

    def checkpoint(name: str = "tts_last.pt") -> Path:
        path = OUT / name
        torch.save({
            "model": {k: v.cpu() for k, v in model.state_dict().items()},
            "mel_mean": mean, "mel_std": std, "r": R,
        }, path)
        return path

    def evaluate_and_sample(step: int) -> None:
        """Synthesize judge sentences, score with whisper, keep the best ckpt."""
        nonlocal best_cer
        path = checkpoint("tts_last.pt")
        synth = Synthesizer(str(path))
        cers = []
        for text in judge_texts:
            wav, _sr = synth.tts(text)
            hyp = judge.transcribe(wav).text.lower().strip(" .").replace(",", "")
            cers.append(char_error_rate(hyp, text))
        cer = sum(cers) / len(cers)
        marker = ""
        if cer < best_cer:
            best_cer = cer
            checkpoint("tts.pt")          # the pipeline loads THIS one
            marker = "  <- new best, saved as tts.pt"
        print(f"    judge @ step {step}: mean CER {cer:.2f} "
              f"(best {best_cer:.2f}){marker}", flush=True)
        wav, sr = synth.tts(sample_text)
        sf.write(OUT / f"08_sample_step{step}.wav", wav, sr)

    OUT.mkdir(exist_ok=True)
    model.train()
    t0 = time.time()
    for step in range(1, STEPS + 1):
        batch = random.sample(items, min(BATCH, len(items)))
        text, text_mask, mel, mel_mask, stop = tts_collate(batch, mean, std, r=R)
        text, text_mask = text.to(device), text_mask.to(device)
        mel, mel_mask, stop = mel.to(device), mel_mask.to(device), stop.to(device)

        mel_out, stop_logits, aligns = model(text, text_mask, mel)

        mask = mel_mask.unsqueeze(-1)
        mel_l1 = ((mel_out - mel).abs() * mask).sum() / (mask.sum() * 80)
        stop_bce = F.binary_cross_entropy_with_logits(stop_logits, stop)
        # The diagonal prior that makes attention "click in" fast on tiny data.
        attn_pen = guided_attention_penalty(aligns, text_mask, mel_mask, r=R)
        loss = mel_l1 + stop_bce + 5.0 * attn_pen

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 50 == 0 or step == 1:
            rate = step / (time.time() - t0)
            print(f"step {step:4d} | mel L1 {mel_l1.item():.3f} | "
                  f"stop {stop_bce.item():.3f} | attn {attn_pen.item():.3f} | "
                  f"{rate:.2f} it/s", flush=True)
        if step % 1000 == 0 or step == STEPS:
            model.eval()
            # Plot the FREE-RUNNING alignment on the sample sentence (what
            # inference actually does), not the teacher-forced batch one.
            from src.asr.text import encode as _enc
            ids = torch.tensor([_enc(sample_text)], dtype=torch.long, device=device)
            _, gen_align = model.generate(ids)   # (steps, T_in)
            save_alignment_plot(gen_align, OUT / "08_alignment.png", step)
            evaluate_and_sample(step)
            model.train()

    print(f"\nbest judge CER {best_cer:.2f} — that checkpoint is outputs/tts.pt")
    print("plug it into the pipeline:")
    print("  uv run python scripts/06_pipeline_e2e.py   (tts_engine=scratch is the default)")


if __name__ == "__main__":
    main()
