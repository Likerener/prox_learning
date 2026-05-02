#!/usr/bin/env bash
# Eval PLA + baselines on all 4 tasks (PROJECT.md §4.1).
set -euo pipefail
cd "$(dirname "$0")/.."

TASKS="pnp near_contact pnp_color pnp_next_to"
N=${N_EPISODES:-100}

for ckpt in runs/pla/best.pt runs/baseline_vlm_only/best.pt runs/baseline_prop_only/best.pt; do
  [[ -f "$ckpt" ]] || { echo "skip $ckpt (missing)"; continue; }
  name=$(basename "$(dirname "$ckpt")")
  python -m pla.eval.run_eval \
    --checkpoint "$ckpt" \
    --tasks $TASKS \
    --n-episodes "$N" \
    --out "reports/eval/${name}.json"
done
