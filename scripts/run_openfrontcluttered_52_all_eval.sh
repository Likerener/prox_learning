#!/usr/bin/env bash

set -euo pipefail

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

run_baseline_eval () {
    local model_seed="$1"
    local eval_seed="$2"

    local ckpt="$REPO/runs/openfrontcluttered_52_baseline_seed${model_seed}_20260623"
    local out="$REPO/eval_output/openfrontcluttered_52_baseline_seed${model_seed}"
    local log="$REPO/logs/openfrontcluttered_52_baseline_eval_seed${model_seed}.log"

    echo
    echo "============================================================"
    echo "BASELINE EVAL: model seed ${model_seed}, eval seed ${eval_seed}"
    echo "============================================================"

    test -f "$ckpt/policy_best.ckpt"
    rm -rf "$out"
    rm -f "$log"

    "$PYTHON" pact/act_prox/eval_openfrontcluttered_baseline.py \
        --ckpt_dir "$ckpt" \
        --ckpt_name policy_best.ckpt \
        --output_dir "$out" \
        --num_rollouts 1 \
        --house_inds "${HOUSES[@]}" \
        --task_horizon 300 \
        --chunk_size 100 \
        --kl_weight 10 \
        --hidden_dim 512 \
        --dim_feedforward 3200 \
        --seed "$eval_seed" \
        2>&1 | tee "$log"
}

run_pact_eval () {
    local model_seed="$1"
    local eval_seed="$2"

    local ckpt="$REPO/runs/openfrontcluttered_52_pact_seed${model_seed}_20260623"
    local out="$REPO/eval_output/openfrontcluttered_52_pact_seed${model_seed}"
    local log="$REPO/logs/openfrontcluttered_52_pact_eval_seed${model_seed}.log"

    echo
    echo "============================================================"
    echo "PACT EVAL: model seed ${model_seed}, eval seed ${eval_seed}"
    echo "============================================================"

    test -f "$ckpt/policy_best.ckpt"
    rm -rf "$out"
    rm -f "$log"

    "$PYTHON" pact/act_prox/eval_openfrontcluttered_pact.py \
        --ckpt_dir "$ckpt" \
        --ckpt_name policy_best.ckpt \
        --prox_encoder_ckpt "$ENCODER" \
        --prox_mapping_json "$MAPPING" \
        --output_dir "$out" \
        --num_rollouts 1 \
        --house_inds "${HOUSES[@]}" \
        --task_horizon 300 \
        --chunk_size 100 \
        --kl_weight 10 \
        --hidden_dim 512 \
        --dim_feedforward 3200 \
        --prox_tokens_per_sensor 6 \
        --seed "$eval_seed" \
        2>&1 | tee "$log"
}

run_baseline_eval 0 2026
run_pact_eval     0 2026

run_baseline_eval 1 2027
run_pact_eval     1 2027

run_baseline_eval 2 2028
run_pact_eval     2 2028

echo
echo "============================================================"
echo "ALL 52-DEMO EVALUATIONS FINISHED"
echo "============================================================"

"$PYTHON" - <<'PY'
from pathlib import Path
import re
import statistics

root = Path("/home/qinzhengfangli/molmo_test/prox_learning")

results = {"Baseline": [], "PACT": []}

print("\n===== MATCHED ROLLOUT RESULTS =====")

for label, slug, pattern in [
    ("Baseline", "baseline", r"\[act-eval\]\s+success\s+(\d+)/(\d+)"),
    ("PACT", "pact", r"\[act-prox-eval\]\s+success\s+(\d+)/(\d+)"),
]:
    for seed in range(3):
        log = root / f"logs/openfrontcluttered_52_{slug}_eval_seed{seed}.log"
        text = log.read_text(errors="ignore")

        matches = re.findall(pattern, text)
        if not matches:
            print(f"{label:8s} seed {seed}: RESULT NOT FOUND")
            continue

        success, total = map(int, matches[-1])
        rate = success / total if total else 0.0
        results[label].append(rate)

        video_dir = root / f"eval_output/openfrontcluttered_52_{slug}_seed{seed}"
        videos = len(list(video_dir.rglob("*.mp4")))

        print(
            f"{label:8s} seed {seed}: "
            f"{success}/{total} = {rate:.1%}, videos={videos}"
        )

print("\n===== THREE-SEED ROLLOUT SUMMARY =====")

for label, values in results.items():
    if len(values) == 3:
        print(
            f"{label:8s}: mean success rate "
            f"{statistics.mean(values):.1%} ± "
            f"{statistics.stdev(values):.1%}"
        )
    else:
        print(f"{label:8s}: only {len(values)}/3 results found")
PY
