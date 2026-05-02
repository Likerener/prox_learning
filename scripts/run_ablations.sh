#!/usr/bin/env bash
# Ablation ladder (PROJECT.md §4.3, Day 8-9).
#
#   wrist_only      — keep only link6 sensors (mask 8:32)
#   handcrafted     — replace ProximityEncoder with handcrafted features
#   conv2d          — replace shared MLP with Conv2D encoder
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pla.train.train_pla --config configs/train/pla.yaml \
  --mask-sensors 8:32 --run-dir runs/abl_wrist_only

python -m pla.train.train_pla --config configs/train/pla.yaml \
  --encoder handcrafted --run-dir runs/abl_handcrafted

python -m pla.train.train_pla --config configs/train/pla.yaml \
  --encoder conv2d --run-dir runs/abl_conv2d
