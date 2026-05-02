#!/usr/bin/env bash
# Kick off the 1000-trajectory collection in tmux. PROJECT.md §6 Day 3.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-near_contact}        # near_contact | pnp
N_TRAJ=${2:-1000}
SESSION=collect_${TASK}

tmux new-session -d -s "$SESSION" \
  "python -m pla.data.collect \
      --config configs/data/${TASK}.yaml \
      --out-dir data/raw/${TASK} \
      --n-traj ${N_TRAJ} \
      2>&1 | tee reports/logs/collect_${TASK}.log"
echo "tmux session: $SESSION  (attach: tmux attach -t $SESSION)"
