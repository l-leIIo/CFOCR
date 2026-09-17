#!/usr/bin/env bash
set -euo pipefail
mkdir -p models
BASE="https://raw.githubusercontent.com/boluoreg/4399Register/4b71346e21a23f211e2756f8ad43e7a0a5157456/4399ocr"
curl -L --fail "$BASE/4399ocr.onnx" -o models/4399ocr.onnx
curl -L --fail "$BASE/4399ocr.json" -o models/4399ocr.json
echo "Downloaded:"
ls -lh models
