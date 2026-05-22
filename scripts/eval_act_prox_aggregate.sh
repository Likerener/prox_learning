#!/usr/bin/env bash
# Aggregate ≥N rollouts of ACT + frozen prox-encoder, matching the layout
# used by eval_output/act_house1_mug_random_v1_aggregate/. Each rollout is
# one molmospaces process (samples_per_house=1, fresh process for fresh
# task_sampler_config draws), then we aggregate with summary stats.
set -euo pipefail

CKPT_DIR="${CKPT_DIR:-/home/jaydv/code/prox_learning/runs/act_prox_mug_v1}"
PROX_ENC="${PROX_ENC:-/home/jaydv/code/prox_learning/pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt}"
PROX_MAP="${PROX_MAP:-/home/jaydv/code/prox_learning/act_style_data/mug_house1_random_everything/prox_mapping.json}"
OUT_ROOT="${OUT_ROOT:-/home/jaydv/code/prox_learning/eval_output/act_prox_mug_v1_aggregate}"
N_ROLLOUTS="${N_ROLLOUTS:-10}"
START_IDX="${START_IDX:-0}"
TASK_HORIZON="${TASK_HORIZON:-300}"
CHUNK_SIZE="${CHUNK_SIZE:-100}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DIM_FFN="${DIM_FFN:-3200}"
EXISTING_PYTHONPATH="${PYTHONPATH:-}"

mkdir -p "$OUT_ROOT"

END_IDX=$((START_IDX + N_ROLLOUTS - 1))
for i in $(seq "$START_IDX" "$END_IDX"); do
    II=$(printf "%02d" "$i")
    OUT="$OUT_ROOT/run_$II"
    LOG="$OUT_ROOT/eval_log_run_$II.txt"
    echo "[loop] === rollout $II -> $OUT ==="
    cd /home/jaydv/code/prox_learning/submodules/act
    PYTHONPATH="$PWD:/home/jaydv/code/prox_learning:$EXISTING_PYTHONPATH" \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    /opt/conda/envs/mlspaces/bin/python /home/jaydv/code/prox_learning/pact/act_prox/eval_act_with_prox_encoder.py \
        --ckpt_dir "$CKPT_DIR" \
        --ckpt_name policy_best.ckpt \
        --prox_encoder_ckpt "$PROX_ENC" \
        --prox_mapping_json "$PROX_MAP" \
        --output_dir "$OUT" \
        --chunk_size "$CHUNK_SIZE" \
        --hidden_dim "$HIDDEN_DIM" \
        --dim_feedforward "$DIM_FFN" \
        --task_horizon "$TASK_HORIZON" \
        2>&1 | tee "$LOG"
done

echo "[loop] all $N_ROLLOUTS rollouts complete."
