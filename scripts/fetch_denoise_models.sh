#!/usr/bin/env bash
# Module 5: fetch the pretrained streaming denoiser weights (the production
# alternates to src/denoise/spectral.py). ~4 MB total — these are genuinely
# tiny models; GTCRN is 48K parameters.
#
#   bash scripts/fetch_denoise_models.sh
#
# The from-scratch engine (VOICE_DENOISE_ENGINE=spectral, the default) needs
# none of this — it has no weights at all.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

echo "fetching denoiser ONNX weights into ./models ..."

# GTCRN — Rong et al. 2024; complex-domain, 48K params, best PESQ of the three.
gtcrn="https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/main/stream/onnx_models"
curl -fL --progress-bar -o models/gtcrn.onnx "$gtcrn/gtcrn.onnx"

# DTLN — Westhausen & Meyer 2020; two LSTM stages, lowest CPU, best SI-SDR.
dtln="https://raw.githubusercontent.com/breizhn/DTLN/master/pretrained_model"
curl -fL --progress-bar -o models/dtln_1.onnx "$dtln/model_1.onnx"
curl -fL --progress-bar -o models/dtln_2.onnx "$dtln/model_2.onnx"

echo "done:"
ls -la models/gtcrn.onnx models/dtln_1.onnx models/dtln_2.onnx
