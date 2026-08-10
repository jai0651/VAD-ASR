#!/usr/bin/env bash
# Download the DTLN pretrained ONNX models used by the low-CPU streaming backend.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models
echo "Fetching streaming-engine ONNX models into ./models ..."

# GTCRN — recommended streaming engine (complex-domain, ~48K params)
gtcrn="https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/main/stream/onnx_models"
curl -fL --progress-bar -o models/gtcrn.onnx "$gtcrn/gtcrn.onnx"

# DTLN — alternate streaming engine
dtln="https://raw.githubusercontent.com/breizhn/DTLN/master/pretrained_model"
curl -fL --progress-bar -o models/dtln_1.onnx "$dtln/model_1.onnx"
curl -fL --progress-bar -o models/dtln_2.onnx "$dtln/model_2.onnx"

echo "Done:"
ls -la models/*.onnx
