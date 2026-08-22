# Engine benchmark — VoiceBank-DEMAND (real noisy speech)

60 test files from the VoiceBank-DEMAND 16 kHz test set (real recorded noise:
cafés, offices, traffic, buses). Higher is better for every metric. The `noisy`
row is the unprocessed input — an engine only earns its place by beating it.

| engine | PESQ-WB | STOI | SI-SDR (dB) | CPU (1 core) |
| --- | --- | --- | --- | --- |
| noisy (baseline) | 2.188 | 0.918 | 8.57 | — |
| dtln | 2.552 | 0.924 | **17.33** | ~1.5% |
| **gtcrn** | **2.663** | 0.924 | 14.05 | ~3% |
| deepfilternet* | 2.410 | 0.921 | 16.21 | ~23% |

- **PESQ (perceptual quality): GTCRN wins.** It's the best-sounding engine on
  real 16 kHz noise *and* the cheap streaming one.
- **SI-SDR (signal fidelity): DTLN wins**, DeepFilterNet second.
- **STOI (intelligibility)** barely moves — this test set is only mildly noisy in
  intelligibility terms, so PESQ/SI-SDR are the discriminating metrics.

\*DeepFilterNet is a 48 kHz full-band model, but this test set is 16 kHz. We
upsample 16→48 kHz (which adds no real content above 8 kHz), denoise, and
downsample back — out of its training distribution, so its score here is a
lower bound. A fair head-to-head needs the 48 kHz VoiceBank-DEMAND set.

**Takeaway:** for the realtime / telephony (16 kHz) case, GTCRN is the right
default — best perceptual quality at a fraction of the CPU. Reproduce with
`./.venv/bin/python evaluate.py 60`.
