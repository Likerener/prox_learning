#!/usr/bin/env bash
# Run noise and shuffle proximity ablations on the best-known P+ACT ckpt.
# Used as a fallback if mask=zero doesn't show enough degradation.

set -e
REPO=/home/jaydv/code/prox_learning
cd "$REPO"
PY=/opt/conda/envs/mlspaces/bin/python

CKPT_NAME="${CKPT_NAME:-policy_best.ckpt}"
N="${N:-25}"
PARALLEL="${PARALLEL:-3}"

echo "[$(date +%H:%M:%S)] noise + shuffle ablation on $CKPT_NAME, n=$N, parallel=$PARALLEL"

# Noise mask
"$PY" "$REPO/scripts/run_pact_mask_experiment.py" \
    --n_runs "$N" --parallel "$PARALLEL" \
    --ckpt_dir "$REPO/runs/act_prox_mug_v1" --ckpt_name "$CKPT_NAME" \
    --mask_proximity noise --mask_phase none \
    --output_dir "$REPO/eval_output/exp1_mask_noise_n${N}" \
    --use_wandb --wandb_project pact-mask-exp 2>&1 | \
    tee "$REPO/eval_output/_exp_logs/mask_noise.log"

# Shuffle mask
"$PY" "$REPO/scripts/run_pact_mask_experiment.py" \
    --n_runs "$N" --parallel "$PARALLEL" \
    --ckpt_dir "$REPO/runs/act_prox_mug_v1" --ckpt_name "$CKPT_NAME" \
    --mask_proximity shuffle --mask_phase none \
    --output_dir "$REPO/eval_output/exp1_mask_shuffle_n${N}" \
    --use_wandb --wandb_project pact-mask-exp 2>&1 | \
    tee "$REPO/eval_output/_exp_logs/mask_shuffle.log"

echo "[$(date +%H:%M:%S)] noise + shuffle done"
