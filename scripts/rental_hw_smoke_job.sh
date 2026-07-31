#!/usr/bin/env bash
set -euo pipefail
echo "=== host ==="
nvidia-smi || true
python -V
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git >/dev/null
pip install -q -U pip
# Latest transformers needed for Qwen3.5 multimodal arch
pip install -q "git+https://github.com/huggingface/transformers.git" \
  datasets accelerate peft trl bitsandbytes huggingface_hub pyyaml click rich tqdm pydantic numpy
cd "$(dirname "$0")/.."
pip install -q -e .
python scripts/rental_hw_smoke.py
