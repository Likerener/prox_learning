#!/usr/bin/env bash

set -u

REPO="/home/qinzhengfangli/molmo_test/prox_learning"
PYTHON="/home/qinzhengfangli/molmo_test/molmospaces/.venv/bin/python"

cd "$REPO"
mkdir -p logs eval_output

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/submodules/act:$REPO/submodules/act/detr:$REPO/submodules/molmospaces:$REPO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

HOUSES=(12 13 14 15 16 17 23 25)

MAPPING="$REPO/act_style_data/openfrontcluttered_52_20260623/prox_mapping.json"
ENCODER="$REPO/pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt"

run_one_baseline() {
    local seed="$1"
    local eval_seed="$2"
    local house="$3"

    local ckpt="$REPO/runs/openfrontcluttered_52_baseline_seed${seed}_20260623"
    local out="$REPO/eval_output/openfrontcluttered_52_baseline_seed${seed}_safe/house_${house}_run"
    local log="$REPO/logs/openfrontcluttered_52_baseline_eval_seed${seed}_house${house}.log"
    local done="$out/.done"

    if [[ -f "$done" ]]; then
        echo "[SKIP] baseline seed=$seed house=$house already complete"
        return 0
    fi

    echo
    echo "============================================================"
    echo "BASELINE seed=$seed eval_seed=$eval_seed house=$house"
    echo "============================================================"

    rm -rf "$out"
    mkdir -p "$out"

    "$PYTHON" pact/act_prox/eval_openfrontcluttered_baseline.py \
        --ckpt_dir "$ckpt" \
        --ckpt_name policy_best.ckpt \
        --output_dir "$out" \
        --num_rollouts 1 \
        --house_inds "$house" \
        --task_horizon 300 \
        --chunk_size 100 \
        --kl_weight 10 \
        --hidden_dim 512 \
        --dim_feedforward 3200 \
        --seed "$eval_seed" \
        2>&1 | tee "$log"

    local status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 ]] && grep -qE '\[act-eval\] success [0-9]+/[0-9]+' "$log"; then
        touch "$done"
        echo "[DONE] baseline seed=$seed house=$house"
        return 0
    fi

    echo "[FAILED] baseline seed=$seed house=$house status=$status"
    return 1
}

run_one_pact() {
    local seed="$1"
    local eval_seed="$2"
    local house="$3"

    local ckpt="$REPO/runs/openfrontcluttered_52_pact_seed${seed}_20260623"
    local out="$REPO/eval_output/openfrontcluttered_52_pact_seed${seed}_safe/house_${house}_run"
    local log="$REPO/logs/openfrontcluttered_52_pact_eval_seed${seed}_house${house}.log"
    local done="$out/.done"

    if [[ -f "$done" ]]; then
        echo "[SKIP] PACT seed=$seed house=$house already complete"
        return 0
    fi

    echo
    echo "============================================================"
    echo "PACT seed=$seed eval_seed=$eval_seed house=$house"
    echo "============================================================"

    rm -rf "$out"
    mkdir -p "$out"

    "$PYTHON" pact/act_prox/eval_openfrontcluttered_pact.py \
        --ckpt_dir "$ckpt" \
        --ckpt_name policy_best.ckpt \
        --prox_encoder_ckpt "$ENCODER" \
        --prox_mapping_json "$MAPPING" \
        --output_dir "$out" \
        --num_rollouts 1 \
        --house_inds "$house" \
        --task_horizon 300 \
        --chunk_size 100 \
        --kl_weight 10 \
        --hidden_dim 512 \
        --dim_feedforward 3200 \
        --prox_tokens_per_sensor 6 \
        --seed "$eval_seed" \
        2>&1 | tee "$log"

    local status=${PIPESTATUS[0]}

    if [[ "$status" -eq 0 ]] && grep -qE '\[act-prox-eval\] success [0-9]+/[0-9]+' "$log"; then
        touch "$done"
        echo "[DONE] PACT seed=$seed house=$house"
        return 0
    fi

    echo "[FAILED] PACT seed=$seed house=$house status=$status"
    return 1
}

# Baseline seed 0 already completed successfully in the original full run.
# Run baseline seeds 1 and 2 safely, one house per process.
for seed in 1 2; do
    eval_seed=$((2026 + seed))
    for house in "${HOUSES[@]}"; do
        run_one_baseline "$seed" "$eval_seed" "$house" || {
            echo "Retrying baseline seed=$seed house=$house once..."
            sleep 5
            run_one_baseline "$seed" "$eval_seed" "$house" || true
        }
        sleep 2
    done
done

# Rerun every PACT seed from scratch, one house per process.
for seed in 0 1 2; do
    eval_seed=$((2026 + seed))
    for house in "${HOUSES[@]}"; do
        run_one_pact "$seed" "$eval_seed" "$house" || {
            echo "Retrying PACT seed=$seed house=$house once..."
            sleep 5
            run_one_pact "$seed" "$eval_seed" "$house" || true
        }
        sleep 2
    done
done

echo
echo "============================================================"
echo "SAFE EVALUATION LOOP FINISHED"
echo "============================================================"
