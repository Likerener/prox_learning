#!/usr/bin/env bash
# Smoke a single rollout with temporal ensembling on, houses 11-20.
# Usage: scripts/smoke_tempagg.sh <model> <ckpt> <use_proximity>
#   model: tag for the run name (e.g. "pla" or "vlm")
#   ckpt: path to latest.pt
#   use_proximity: true|false
set -euo pipefail
MODEL="${1:?model tag required}"
CKPT="${2:?ckpt path required}"
USE_PROX="${3:?use_proximity (true|false) required}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
RUN_NAME="${MODEL}_smoke_tempagg_${TS}"
LOG="logs/${RUN_NAME}.log"
echo "Run: $RUN_NAME  Log: $LOG"

export PYTHONPATH="$REPO_ROOT/submodules/molmospaces:$REPO_ROOT"
export MLSPACES_ASSETS_DIR="$REPO_ROOT/assets"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1
export PLA_Z_SCALE=0     # isolate temporal-ensemble effect from z-sampling

nohup /opt/conda/envs/mlspaces/bin/python -m pla.rollout_eval \
    --checkpoint "$CKPT" \
    --run_name "$RUN_NAME" \
    --use_proximity "$USE_PROX" \
    --use_language true \
    --seed 2028 \
    --house_inds "11,12,13,14,15,16,17,18,19,20" \
    --samples_per_house 1 \
    --num_workers 2 \
    > "$LOG" 2>&1 &
echo "PID $!"
