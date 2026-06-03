#!/usr/bin/env bash
# After mask_zero finishes, kill the slow epoch sweep and restart with wider
# coverage (more epochs, n=10 each) at higher parallelism.

set -e

REPO=/home/jaydv/code/prox_learning
cd "$REPO"
PY=/opt/conda/envs/mlspaces/bin/python

echo "[$(date +%H:%M:%S)] === waiting for mask_zero summary.json ==="
until [ -f eval_output/exp1_mask_zero_n50/summary.json ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] mask_zero done"
cat eval_output/exp1_mask_zero_n50/summary.json

# Kill existing slow epoch sweep (the parallel=2 one).
echo "[$(date +%H:%M:%S)] === killing existing parallel=2 epoch sweep ==="
pkill -f run_pact_epoch_sweep.py 2>/dev/null || true
# Find and kill the run_pact_mask_experiment that's doing epoch_1500
ps -ef | grep -E "epoch_1500" | grep -v grep | awk '{print $2}' | xargs -r kill 2>/dev/null || true
sleep 5
# Wait for any in-flight eval subprocesses to drain.
while pgrep -f "policy_epoch_1500" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] old sweep dead"

# Preserve partial epoch_1500 data by renaming.
if [ -d eval_output/epoch_sweep/epoch_1500 ]; then
    mv eval_output/epoch_sweep/epoch_1500 eval_output/epoch_sweep/epoch_1500_partial
fi

# Launch new wider sweep at parallel=4, n=10 each, over many epochs.
echo "[$(date +%H:%M:%S)] === launching wider epoch sweep parallel=4, n=10 ==="
"$PY" "$REPO/scripts/run_pact_epoch_sweep.py" \
    --ckpt_dir "$REPO/runs/act_prox_mug_v1" \
    --epochs 800,1000,1200,1400,1600,1700,1800,1900,last \
    --n_runs 10 --parallel 4 \
    --output_dir "$REPO/eval_output/epoch_sweep" \
    --wandb 2>&1 | tee "$REPO/eval_output/_exp_logs/epoch_sweep_wide.log"

echo "[$(date +%H:%M:%S)] === wide epoch sweep done ==="
cat "$REPO/eval_output/epoch_sweep/best_epoch.json"
