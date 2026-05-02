#!/usr/bin/env bash
# Train both baselines (VLM-only ACT and prop-only MLP).
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pla.train.train_baseline \
  --variant vlm_only \
  --config configs/train/act_baseline.yaml "$@"

python -m pla.train.train_baseline \
  --variant prop_only \
  --config configs/train/act_baseline.yaml "$@"
