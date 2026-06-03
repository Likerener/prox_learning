#!/usr/bin/env bash
# Master orchestrator for P+ACT masking experiments (Exp 1 + Exp 2).
#
# Runs all conditions sequentially with --parallel=$PARALLEL workers each.
#
# Outputs:
#   eval_output/exp1_mask_zero_n${N}/        — Exp 1: full proximity zeroing
#   eval_output/exp1_mask_mean_n${N}/        — Exp 1: mean-replacement sanity
#   eval_output/exp2_mask_<phase>_n${N}/     — Exp 2: phase-localized zeroing
#
# Usage:
#   N=50 PARALLEL=3 bash scripts/run_pact_exp1_exp2_all.sh
#
# Resume support: per-condition `--start_idx` is supported in
# run_pact_mask_experiment.py.  For full restarts, delete the output dir first.

set -e

REPO=/home/jaydv/code/prox_learning
PY=/opt/conda/envs/mlspaces/bin/python
RUNNER="$REPO/scripts/run_pact_mask_experiment.py"
N=${N:-50}
PARALLEL=${PARALLEL:-3}
WANDB=${WANDB:-1}

cd "$REPO"

WB_FLAG=""
if [ "$WANDB" = "1" ]; then
    WB_FLAG="--use_wandb --wandb_project pact-mask-exp"
fi

mkdir -p eval_output/_exp_logs

log_and_run() {
    local label="$1"; shift
    local log="eval_output/_exp_logs/${label}.log"
    echo "[$(date +%H:%M:%S)] ==== launching ${label}  log=${log} ===="
    "$PY" "$RUNNER" "$@" $WB_FLAG 2>&1 | tee "$log"
}

# ---------------------------------------------------------------
# Exp 1: Full proximity mask
# ---------------------------------------------------------------
log_and_run "exp1_mask_zero_n${N}" \
    --n_runs "$N" --parallel "$PARALLEL" \
    --mask_proximity zero --mask_phase none \
    --output_dir "eval_output/exp1_mask_zero_n${N}"

log_and_run "exp1_mask_mean_n${N}" \
    --n_runs "$N" --parallel "$PARALLEL" \
    --mask_proximity mean --mask_phase none \
    --output_dir "eval_output/exp1_mask_mean_n${N}"

# ---------------------------------------------------------------
# Exp 2: Per-phase mask
# ---------------------------------------------------------------
for phase in approach pregrasp grasp_lift transit place; do
    log_and_run "exp2_mask_${phase}_n${N}" \
        --n_runs "$N" --parallel "$PARALLEL" \
        --mask_proximity zero --mask_phase "$phase" \
        --output_dir "eval_output/exp2_mask_${phase}_n${N}"
done

echo "[$(date +%H:%M:%S)] All conditions completed."
