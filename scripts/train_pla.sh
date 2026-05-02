#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m pla.train.train_pla --config configs/train/pla.yaml "$@"
