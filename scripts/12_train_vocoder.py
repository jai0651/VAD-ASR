"""
Module 7: train the ISTFT vocoder, and prove it beats Griffin-Lim.

The experiment is deliberately airtight: BOTH vocoders are given the EXACT SAME
log-mel spectrogram of a held-out utterance and asked to produce a waveform.
Nothing else differs. Whatever gap appears is attributable to phase modelling
and nothing else — which is the claim being tested.

TWO metrics, because one of them is rigged and it took a measurement to notice.

  mr-stft  multi-resolution STFT distance — a MAGNITUDE metric. Griffin-Lim's
           entire algorithm is "iterate until the magnitudes are consistent", so
           this scores it on its own objective while being nearly blind to the
           one thing it gets wrong. Reporting only this would have been a
           self-own dressed up as a fair test.
  SI-SDR   scale-invariant SDR in the TIME domain. A wrong phase cannot hide
           here, which is exactly why it belongs in this comparison.

Otherwise the setup is airtight: BOTH vocoders get the EXACT SAME log-mel of a
held-out utterance. Nothing else differs, so any gap is attributable to phase
modelling. The script also writes the wavs, because for a vocoder listening is
the evaluation that actually counts.

Run:
  uv run python scripts/12_train_vocoder.py
  STEPS=8000 DIM=256 BLOCKS=8 uv run python scripts/12_train_vocoder.py
  SPLIT=train-clean-100 DOWNLOAD=1 STEPS=40000 uv run python scripts/12_train_vocoder.py

Writes outputs/vocoder.pt plus outputs/12_{true,neural,griffinlim}.wav
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.asr.corpus import LogMel, build_manifest, warmup_cosine  # noqa: E402
from src.tts.istft_vocoder import (  # noqa: E402
    HOP,
    SR,
    WIN,
    ISTFTVocoder,
    MultiResolutionSTFTLoss,
    count_parameters,
)
from src.tts.vocoder import griffin_lim, log_mel_to_magnitude  # noqa: E402

OUT = Path("outputs")

SPLIT = os.environ.get("SPLIT", "dev-clean")
DOWNLOAD = os.environ.get("DOWNLOAD", "0") == "1"
STEPS = int(os.environ.get("STEPS", "4000"))
BATCH = int(os.environ.get("BATCH", "16"))
CROP_S = float(os.environ.get("CROP_S", "1.0"))
DIM = int(os.environ.get("DIM", "192"))
BLOCKS = int(os.environ.get("BLOCKS", "6"))
LR = float(os.environ.get("LR", "3e-4"))
EVAL_EVERY = int(os.environ.get("EVAL_EVERY", "500"))


def get_device() -> torch.device:
    if os.environ.get("DEVICE"):
        return torch.device(os.environ["DEVICE"])
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class Crops:
    """Random fixed-length waveform crops. A vocoder is a LOCAL map — it never
    needs to see a whole utterance — so training on crops costs nothing and
    makes every batch the same shape, which keeps the STFT loss cheap."""

    def __init__(self, items, crop_samples: int, logmel: LogMel, seed: int = 0):
        self.items = [it for it in items if it["duration"] * SR > crop_samples + HOP]
        self.n = crop_samples
        self.logmel = logmel
        self.rng = random.Random(seed)

    def batch(self, size: int):
        wavs = []
        while len(wavs) < size:
            it = self.rng.choice(self.items)
            audio, sr = sf.read(it["path"], dtype="float32")
            if sr != SR or len(audio) <= self.n:
                continue
            start = self.rng.randrange(len(audio) - self.n)
            wavs.append(audio[start:start + self.n])
        wav = torch.from_numpy(np.stack(wavs))
        mel = torch.stack([self.logmel(w) for w in wav])
        return mel, wav


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    """Scale-invariant SDR, in the TIME domain — so it is sensitive to phase.

    This exists because the training loss is not a fair referee here. Griffin-Lim
    literally optimises magnitude consistency; multi-resolution STFT distance
    measures magnitude consistency. Scoring the two vocoders on that alone asks
    Griffin-Lim how well it did at its own objective while ignoring the one thing
    it gets wrong. SI-SDR compares waveforms, where a wrong phase cannot hide.
    """
    n = min(len(ref), len(est))
    ref, est = ref[:n] - ref[:n].mean(), est[:n] - est[:n].mean()
    target = ref * (np.dot(est, ref) / (np.dot(ref, ref) + 1e-12))
    noise = est - target
    return float(10 * np.log10((np.sum(target ** 2) + 1e-12) /
                               (np.sum(noise ** 2) + 1e-12)))


@torch.no_grad()
def held_out_loss(model, crops: Crops, criterion, device, n: int = 8) -> float:
    model.eval()
    total = 0.0
    for _ in range(n):
        mel, wav = crops.batch(4)
        pred = model(mel.to(device))
        total += float(criterion(pred.cpu(), wav))
    model.train()
    return total / n


def main() -> None:
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    device = get_device()
    OUT.mkdir(exist_ok=True)

    items = build_manifest(split=SPLIT, download=DOWNLOAD)
    random.Random(1234).shuffle(items)
    dev_items, train_items = items[:80], items[80:]
    logmel = LogMel()

    crop_samples = WIN + (int(CROP_S * SR / HOP) - 1) * HOP
    train = Crops(train_items, crop_samples, logmel, seed=0)
    dev = Crops(dev_items, crop_samples, logmel, seed=99)
    print(f"device: {device}   split: {SPLIT}   "
          f"crop {crop_samples/SR:.2f}s ({crop_samples} samples)")
    print(f"train: {len(train.items)} utts   held-out: {len(dev.items)}")

    model = ISTFTVocoder(dim=DIM, n_blocks=BLOCKS).to(device)
    criterion = MultiResolutionSTFTLoss()
    print(f"vocoder: {count_parameters(model)/1e6:.2f}M params "
          f"(Griffin-Lim has 0 — and a ceiling)")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.8, 0.99),
                            weight_decay=1e-2)
    best = float("inf")
    t0 = time.time()
    model.train()

    for step in range(1, STEPS + 1):
        mel, wav = train.batch(BATCH)
        pred = model(mel.to(device))
        loss = criterion(pred, wav.to(device)[:, : pred.shape[-1]])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        for g in opt.param_groups:
            g["lr"] = warmup_cosine(step, max(100, STEPS // 20), STEPS, LR)
        opt.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:6d} | mr-stft {float(loss.detach()):6.3f} "
                  f"| lr {opt.param_groups[0]['lr']:.2e} "
                  f"| {step/(time.time()-t0):.2f} it/s", flush=True)

        if step % EVAL_EVERY == 0 or step == STEPS:
            val = held_out_loss(model, dev, criterion, device)
            mark = ""
            if val < best:
                best = val
                torch.save({"model": model.state_dict(), "dim": DIM,
                            "n_blocks": BLOCKS}, OUT / "vocoder.pt")
                mark = "  <- new best, saved"
            print(f"    held-out mr-stft @ {step}: {val:.4f} (best {best:.4f}){mark}")

    # ---- the comparison ------------------------------------------------
    state = torch.load(OUT / "vocoder.pt", map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    print("\n" + "=" * 66)
    print("SAME mel, two vocoders — held-out utterances")
    print("=" * 66)
    neural_scores, gl_scores = [], []
    neural_sdr, gl_sdr = [], []
    for i in range(6):
        mel, wav = dev.batch(1)
        with torch.no_grad():
            pred = model(mel.to(device)).cpu()[0]
        gl = griffin_lim(log_mel_to_magnitude(mel[0]), n_iters=60)
        n = min(len(pred), len(gl), wav.shape[1])
        neural_scores.append(float(criterion(pred[None, :n], wav[:, :n])))
        gl_scores.append(float(criterion(gl[None, :n], wav[:, :n])))
        ref = wav[0, :n].numpy()
        neural_sdr.append(si_sdr(ref, pred[:n].numpy()))
        gl_sdr.append(si_sdr(ref, gl[:n].numpy()))
        if i == 0:
            sf.write(OUT / "12_true.wav", wav[0, :n].numpy(), SR)
            sf.write(OUT / "12_neural.wav", pred[:n].numpy(), SR)
            sf.write(OUT / "12_griffinlim.wav", gl[:n].numpy(), SR)

    gl_mean, nn_mean = float(np.mean(gl_scores)), float(np.mean(neural_scores))
    gl_s, nn_s = float(np.mean(gl_sdr)), float(np.mean(neural_sdr))
    print(f"{'vocoder':34s} {'mr-stft':>9s} {'SI-SDR':>9s}")
    print(f"  {'Griffin-Lim (60 iters, 0 params)':32s} {gl_mean:9.4f} {gl_s:8.2f}dB")
    print(f"  {f'ISTFT vocoder ({count_parameters(model)/1e6:.2f}M params)':32s} "
          f"{nn_mean:9.4f} {nn_s:8.2f}dB")
    print()
    print(f"  mr-stft (MAGNITUDE, Griffin-Lim's own objective): "
          f"{'neural' if nn_mean < gl_mean else 'Griffin-Lim'} wins")
    print(f"  SI-SDR  (TIME domain, i.e. PHASE):                "
          f"{'neural' if nn_s > gl_s else 'Griffin-Lim'} wins by "
          f"{abs(nn_s - gl_s):.1f} dB")
    print("=" * 66)
    print("Predicting phase should win on SI-SDR — that is the whole thesis.")
    print("Winning on magnitude too takes a longer run and discriminators.")
    print("\nlisten: outputs/12_true.wav vs 12_neural.wav vs 12_griffinlim.wav")
    print("the phasey, robotic quality in the Griffin-Lim file IS the ceiling")
    print("this module removes — see docs/11-vocoder.html")


if __name__ == "__main__":
    main()
