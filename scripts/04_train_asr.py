"""
Module 2 demo: train the CTC ASR model on synthetic spoken text.

Run:  uv run python scripts/04_train_asr.py
Outputs:
  - prints CTC loss + sample transcriptions (greedy vs beam) + character error rate
  - outputs/asr.pt : trained weights
  - outputs/04_asr_emissions.png : the model's per-frame emissions (the "spikes")

Key wiring shown:
  - torch.nn.CTCLoss wants log_probs as (T, B, V) and the (downsampled) input
    lengths, plus concatenated targets and their lengths.
  - CTC emissions are "peaky": mostly blank, with a confident spike per character.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.asr.data import make_batch, make_example  # noqa: E402
from src.asr.decode import beam_search_decode, char_error_rate, greedy_decode  # noqa: E402
from src.asr.model import ASRModel  # noqa: E402
from src.asr.text import BLANK, CHARS  # noqa: E402


def get_device() -> torch.device:
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def main() -> None:
    torch.manual_seed(0)
    device = get_device()
    print(f"device: {device}")

    model = ASRModel(n_mels=80).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    steps, batch_size = 800, 16
    model.train()
    for step in range(1, steps + 1):
        feats, in_lens, targets, tgt_lens, _ = make_batch(batch_size)
        feats = feats.to(device)

        log_probs = model(feats)                       # (B, T', V)
        out_lens = model.downsampled_length(in_lens)   # true frames after stride-2
        # CTCLoss wants (T', B, V).
        loss = ctc(log_probs.transpose(0, 1), targets, out_lens, tgt_lens)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:4d} | ctc loss {loss.item():.3f}")

    os.makedirs("outputs", exist_ok=True)
    torch.save(model.state_dict(), "outputs/asr.pt")

    # Evaluate on fresh utterances with both decoders.
    model.eval()
    print("\nHeld-out transcriptions (greedy vs beam):")
    cers = []
    sample_logprobs = None
    for _ in range(6):
        feats, _, text = make_example()
        with torch.no_grad():
            lp = model(feats.unsqueeze(0).to(device))[0].cpu()  # (T', V)
        g = greedy_decode(lp)
        b = beam_search_decode(lp, beam_width=16)
        cers.append(char_error_rate(b, text))
        print(f"  truth='{text}'\n    greedy='{g}'\n    beam  ='{b}'")
        if sample_logprobs is None:
            sample_logprobs = (lp, text)
    print(f"\nmean CER (beam): {sum(cers) / len(cers):.3f}")

    # Visualize the peaky CTC emissions for one utterance.
    lp, text = sample_logprobs
    probs = lp.exp()
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in range(probs.shape[1]):
        if c == BLANK:
            ax.plot(probs[:, c], color="lightgray", lw=1.0,
                    label="blank" if c == BLANK else None)
        else:
            if probs[:, c].max() > 0.3:  # only label characters that actually fire
                ax.plot(probs[:, c], lw=1.4, label=CHARS[c - 1].replace(" ", "space"))
    ax.set_title(f"CTC emissions over time for '{text}' (note the peaky spikes)")
    ax.set_xlabel("output frame (after x2 downsample)")
    ax.set_ylabel("probability")
    ax.legend(loc="upper right", ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig("outputs/04_asr_emissions.png", dpi=120)
    print("Saved outputs/asr.pt and outputs/04_asr_emissions.png")


if __name__ == "__main__":
    main()
