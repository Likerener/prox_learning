#!/usr/bin/env bash
# Automation that runs AFTER Exp 1 mask_mean completes:
#   1. Kill the master sweep orchestrator (skip queued Exp 2 phases).
#   2. Launch epoch sweep on P+ACT ckpts {1500, 1700, 1900, best, last} at parallel=3.
#   3. After epoch sweep, identify best ckpt.
#   4. Re-run headline (no mask) at n=50 with best ckpt.
#   5. Push paper-aggregate to wandb.
#
# Use: nohup bash scripts/automation_after_mask_mean.sh > eval_output/_exp_logs/automation.log 2>&1 &

set -e

REPO=/home/jaydv/code/prox_learning
cd "$REPO"
PY=/opt/conda/envs/mlspaces/bin/python

echo "[$(date +%H:%M:%S)] === waiting for mask_mean summary.json ==="
until [ -f eval_output/exp1_mask_mean_n50/summary.json ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] mask_mean done"
cat eval_output/exp1_mask_mean_n50/summary.json

echo "[$(date +%H:%M:%S)] === killing master sweep orchestrator (keep current rollouts) ==="
# Kill the bash orchestrator and any pending run_pact_mask_experiment.py launches
pkill -f "run_pact_exp1_exp2_all.sh" 2>/dev/null || true
pkill -f "run_pact_mask_experiment.py" 2>/dev/null || true
# Don't kill running eval_act_with_prox_encoder.py — let in-flight rollouts finish.
sleep 5

echo "[$(date +%H:%M:%S)] === waiting for in-flight eval subprocesses to finish ==="
while pgrep -f eval_act_with_prox_encoder.py > /dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] all subprocess gone"

# 2. Epoch sweep
echo "[$(date +%H:%M:%S)] === launching epoch sweep (5 epochs × 12 each, parallel=3) ==="
"$PY" "$REPO/scripts/run_pact_epoch_sweep.py" \
    --ckpt_dir "$REPO/runs/act_prox_mug_v1" \
    --epochs 1500,1700,1900,best,last \
    --n_runs 12 --parallel 3 \
    --output_dir "$REPO/eval_output/epoch_sweep" \
    --wandb 2>&1 | tee "$REPO/eval_output/_exp_logs/epoch_sweep.log"

# 3. Pick best
BEST_EPOCH=$("$PY" -c "import json; d=json.load(open('$REPO/eval_output/epoch_sweep/best_epoch.json')); print(d['best_epoch'])" 2>/dev/null || echo "")
BEST_RATE=$("$PY" -c "import json; d=json.load(open('$REPO/eval_output/epoch_sweep/best_epoch.json')); print(d['best_rate'])" 2>/dev/null || echo "")
echo "[$(date +%H:%M:%S)] best epoch: $BEST_EPOCH (rate $BEST_RATE)"

if [ -z "$BEST_EPOCH" ] || [ "$BEST_EPOCH" = "best" ] || [ "$BEST_EPOCH" = "last" ]; then
    CKPT_NAME="policy_best.ckpt"
    [ "$BEST_EPOCH" = "last" ] && CKPT_NAME="policy_last.ckpt"
else
    CKPT_NAME="policy_epoch_${BEST_EPOCH}_seed_0.ckpt"
fi
echo "[$(date +%H:%M:%S)] will re-eval with ckpt: $CKPT_NAME"

# 4. Re-run headline at n=50 with best ckpt
echo "[$(date +%H:%M:%S)] === re-eval best ckpt: n=50 no mask ==="
"$PY" "$REPO/scripts/run_pact_mask_experiment.py" \
    --n_runs 50 --parallel 3 \
    --ckpt_dir "$REPO/runs/act_prox_mug_v1" --ckpt_name "$CKPT_NAME" \
    --mask_proximity none --mask_phase none \
    --output_dir "$REPO/eval_output/best_epoch_headline_n50" \
    --use_wandb --wandb_project pact-best-epoch 2>&1 | \
    tee "$REPO/eval_output/_exp_logs/best_epoch_headline.log"

# 5. Re-run mask_zero with best ckpt
echo "[$(date +%H:%M:%S)] === re-eval best ckpt: n=50 mask_zero ==="
"$PY" "$REPO/scripts/run_pact_mask_experiment.py" \
    --n_runs 50 --parallel 3 \
    --ckpt_dir "$REPO/runs/act_prox_mug_v1" --ckpt_name "$CKPT_NAME" \
    --mask_proximity zero --mask_phase none \
    --output_dir "$REPO/eval_output/best_epoch_mask_zero_n50" \
    --use_wandb --wandb_project pact-best-epoch 2>&1 | \
    tee "$REPO/eval_output/_exp_logs/best_epoch_mask_zero.log"

# 6. Make the paper figure + push aggregate
"$PY" "$REPO/scripts/plot_mask_experiments.py" \
    --baseline_dir "$REPO/eval_output/act_house1_mug_random_v1_aggregate_n50" \
    --pact_none_dir "$REPO/eval_output/best_epoch_headline_n50" \
    --exp1_root "$REPO/eval_output" \
    --exp2_root "$REPO/eval_output" \
    --n 50 \
    --output_dir "$REPO/eval_output/exp_plots_best" 2>&1 | \
    tee "$REPO/eval_output/_exp_logs/plot_mask_experiments.log"

"$PY" "$REPO/scripts/paper_figure.py" \
    --plots_dir "$REPO/eval_output/exp_plots_best" \
    --tax_dir "$REPO/eval_output/exp3_failure_taxonomy" \
    --out "$REPO/eval_output/paper_figure_best_epoch.png" \
    --n 50 2>&1 | tee -a "$REPO/eval_output/_exp_logs/plot_mask_experiments.log"

"$PY" "$REPO/scripts/push_exp_aggregate_to_wandb.py" \
    --plots_dir "$REPO/eval_output/exp_plots_best" \
    --tax_dir "$REPO/eval_output/exp3_failure_taxonomy" \
    --epoch_sweep_dir "$REPO/eval_output/epoch_sweep" \
    --project pact-paper-corl2026 \
    --run_name paper_best_epoch_2026_05_26 2>&1 | \
    tee "$REPO/eval_output/_exp_logs/push_aggregate.log"

echo "[$(date +%H:%M:%S)] === automation done ==="
