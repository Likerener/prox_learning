#!/usr/bin/env bash
# Automation that runs AFTER both Exp 1 mask_zero/mean AND the manually-launched
# epoch sweep complete. Picks the best ckpt, kills any remaining master sweep,
# and runs the best-epoch headline + mask_zero ablation at n=50.
#
# Usage:
#   nohup bash scripts/automation_after_epoch_sweep.sh > eval_output/_exp_logs/auto2.log 2>&1 &

set -e

REPO=/home/jaydv/code/prox_learning
cd "$REPO"
PY=/opt/conda/envs/mlspaces/bin/python

echo "[$(date +%H:%M:%S)] === waiting for epoch_sweep best_epoch.json ==="
until [ -f eval_output/epoch_sweep/best_epoch.json ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] epoch sweep done"
cat eval_output/epoch_sweep/best_epoch.json

BEST_EPOCH=$("$PY" -c "import json; d=json.load(open('$REPO/eval_output/epoch_sweep/best_epoch.json')); print(d['best_epoch'])")
BEST_RATE=$("$PY" -c "import json; d=json.load(open('$REPO/eval_output/epoch_sweep/best_epoch.json')); print(d['best_rate'])")
echo "[$(date +%H:%M:%S)] best epoch: $BEST_EPOCH (rate $BEST_RATE)"

case "$BEST_EPOCH" in
    best)  CKPT_NAME="policy_best.ckpt" ;;
    last)  CKPT_NAME="policy_last.ckpt" ;;
    *)     CKPT_NAME="policy_epoch_${BEST_EPOCH}_seed_0.ckpt" ;;
esac
echo "[$(date +%H:%M:%S)] picked ckpt: $CKPT_NAME"

echo "[$(date +%H:%M:%S)] === also waiting for mask_zero summary.json ==="
until [ -f eval_output/exp1_mask_zero_n50/summary.json ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] mask_zero done"
cat eval_output/exp1_mask_zero_n50/summary.json
echo "[$(date +%H:%M:%S)] (mask_mean skipped — master sweep killed earlier)"

# If best ckpt is NOT policy_best.ckpt, re-eval headline + mask_zero at n=50.
if [ "$CKPT_NAME" != "policy_best.ckpt" ]; then
    echo "[$(date +%H:%M:%S)] === re-eval best ckpt: n=50 no mask ==="
    "$PY" "$REPO/scripts/run_pact_mask_experiment.py" \
        --n_runs 50 --parallel 3 \
        --ckpt_dir "$REPO/runs/act_prox_mug_v1" --ckpt_name "$CKPT_NAME" \
        --mask_proximity none --mask_phase none \
        --output_dir "$REPO/eval_output/best_epoch_headline_n50" \
        --use_wandb --wandb_project pact-best-epoch 2>&1 | \
        tee "$REPO/eval_output/_exp_logs/best_epoch_headline.log"

    echo "[$(date +%H:%M:%S)] === re-eval best ckpt: n=50 mask_zero ==="
    "$PY" "$REPO/scripts/run_pact_mask_experiment.py" \
        --n_runs 50 --parallel 3 \
        --ckpt_dir "$REPO/runs/act_prox_mug_v1" --ckpt_name "$CKPT_NAME" \
        --mask_proximity zero --mask_phase none \
        --output_dir "$REPO/eval_output/best_epoch_mask_zero_n50" \
        --use_wandb --wandb_project pact-best-epoch 2>&1 | \
        tee "$REPO/eval_output/_exp_logs/best_epoch_mask_zero.log"
else
    echo "[$(date +%H:%M:%S)] best epoch is policy_best.ckpt — reusing existing n=50 numbers"
fi

# Final aggregate plot + push to wandb.
echo "[$(date +%H:%M:%S)] === making paper figures ==="
PACT_HEADLINE=$( [ "$CKPT_NAME" = "policy_best.ckpt" ] && echo "$REPO/eval_output/act_prox_mug_v1_aggregate_n50" || echo "$REPO/eval_output/best_epoch_headline_n50" )

"$PY" "$REPO/scripts/plot_mask_experiments.py" \
    --baseline_dir "$REPO/eval_output/act_house1_mug_random_v1_aggregate_n50" \
    --pact_none_dir "$PACT_HEADLINE" \
    --exp1_root "$REPO/eval_output" \
    --exp2_root "$REPO/eval_output" \
    --n 50 \
    --output_dir "$REPO/eval_output/exp_plots_final" 2>&1 | \
    tee -a "$REPO/eval_output/_exp_logs/plot_final.log"

"$PY" "$REPO/scripts/paper_figure.py" \
    --plots_dir "$REPO/eval_output/exp_plots_final" \
    --tax_dir "$REPO/eval_output/exp3_failure_taxonomy" \
    --out "$REPO/eval_output/paper_figure_final.png" \
    --n 50 2>&1 | tee -a "$REPO/eval_output/_exp_logs/plot_final.log"

"$PY" "$REPO/scripts/push_exp_aggregate_to_wandb.py" \
    --plots_dir "$REPO/eval_output/exp_plots_final" \
    --tax_dir "$REPO/eval_output/exp3_failure_taxonomy" \
    --epoch_sweep_dir "$REPO/eval_output/epoch_sweep" \
    --project pact-paper-corl2026 \
    --run_name paper_final_2026_05_26 2>&1 | \
    tee "$REPO/eval_output/_exp_logs/push_final.log"

echo "[$(date +%H:%M:%S)] === automation done ==="
