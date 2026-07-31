#!/usr/bin/env bash
# HF Jobs bootstrap for ticket 17 (QLoRA SFT + optional Gate B).
set -euo pipefail
echo "=== host ==="
nvidia-smi || true
python -V
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq git >/dev/null
pip install -q -U pip
# Qwen3.5 needs a recent transformers; match ticket 09 rental smoke.
pip install -q "git+https://github.com/huggingface/transformers.git" \
  datasets accelerate peft trl bitsandbytes huggingface_hub pyyaml click rich tqdm pydantic numpy
cd /workspace
pip install -q -e .
# Extra args forwarded from HF job command after this script, e.g.:
#   bash scripts/rental_sft_gate_b_job.sh --max-steps 100 --gate-games 2
python scripts/rental_sft_gate_b.py "$@"
