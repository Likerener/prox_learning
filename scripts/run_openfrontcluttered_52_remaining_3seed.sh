#!/usr/bin/env bash

set -euo pipefail

REPO="/home/qinzhengfangli/molmo_test/prox_learning"
PYTHON="/home/qinzhengfangli/molmo_test/molmospaces/.venv/bin/python"

cd "$REPO"
mkdir -p logs runs

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO/submodules/act:$REPO/submodules/act/detr:$REPO"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

TASK="openfrontcluttered_52"
MAPPING="act_style_data/openfrontcluttered_52_20260623/prox_mapping.json"
ENCODER="pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt"

run_baseline () {
    local seed="$1"
    local run_dir="runs/openfrontcluttered_52_baseline_seed${seed}_20260623"
    local log="logs/openfrontcluttered_52_baseline_seed${seed}_20260623.log"

    echo
    echo "============================================================"
    echo "START BASELINE SEED ${seed}"
    echo "Run directory: ${run_dir}"
    echo "Log: ${log}"
    echo "============================================================"

    rm -rf "$run_dir"
    rm -f "$log"

    "$PYTHON" submodules/act/imitate_episodes.py \
        --task_name "$TASK" \
        --ckpt_dir "$run_dir" \
        --policy_class ACT \
        --batch_size 4 \
        --seed "$seed" \
        --num_epochs 200 \
        --lr 1e-5 \
        --kl_weight 10 \
        --chunk_size 100 \
        --hidden_dim 512 \
        --dim_feedforward 3200 \
        2>&1 | tee "$log"

    test -f "$run_dir/policy_best.ckpt"

    echo "FINISHED BASELINE SEED ${seed}"
}

run_pact () {
    local seed="$1"
    local run_dir="runs/openfrontcluttered_52_pact_seed${seed}_20260623"
    local log="logs/openfrontcluttered_52_pact_seed${seed}_20260623.log"

    echo
    echo "============================================================"
    echo "START PACT SEED ${seed}"
    echo "Run directory: ${run_dir}"
    echo "Log: ${log}"
    echo "============================================================"

    rm -rf "$run_dir"
    rm -f "$log"

    "$PYTHON" pact/act_prox/imitate_episodes_with_prox.py \
        --task_name "$TASK" \
        --ckpt_dir "$run_dir" \
        --policy_class ACT \
        --batch_size 4 \
        --seed "$seed" \
        --num_epochs 200 \
        --lr 1e-5 \
        --kl_weight 10 \
        --chunk_size 100 \
        --hidden_dim 512 \
        --dim_feedforward 3200 \
        --use_proximity \
        --prox_encoder_ckpt "$ENCODER" \
        --prox_mapping_json "$MAPPING" \
        --prox_tokens_per_sensor 6 \
        2>&1 | tee "$log"

    test -f "$run_dir/policy_best.ckpt"

    echo "FINISHED PACT SEED ${seed}"
}

echo "Checking completed baseline seed 0..."
test -f runs/openfrontcluttered_52_baseline_seed0_20260623/policy_best.ckpt
echo "Baseline seed 0 checkpoint exists."

run_pact 0
run_baseline 1
run_pact 1
run_baseline 2
run_pact 2

echo
echo "============================================================"
echo "ALL TRAINING RUNS FINISHED"
echo "============================================================"

"$PYTHON" - <<'PY'
from pathlib import Path
import re
import statistics

root = Path("/home/qinzhengfangli/molmo_test/prox_learning")

runs = {
    "Baseline": [
        root / f"logs/openfrontcluttered_52_baseline_seed{s}_20260623.log"
        for s in range(3)
    ],
    "PACT": [
        root / f"logs/openfrontcluttered_52_pact_seed{s}_20260623.log"
        for s in range(3)
    ],
}

print("\n===== CHECKPOINT CHECK =====")
missing = []

for method in ("baseline", "pact"):
    for seed in range(3):
        ckpt = (
            root
            / f"runs/openfrontcluttered_52_{method}_seed{seed}_20260623"
            / "policy_best.ckpt"
        )
        status = "OK" if ckpt.exists() else "MISSING"
        print(f"{method:8s} seed {seed}: {status}  {ckpt}")
        if not ckpt.exists():
            missing.append(str(ckpt))

print("\n===== BEST VALIDATION LOSS =====")
all_values = {}

for method, paths in runs.items():
    values = []

    for seed, path in enumerate(paths):
        if not path.exists():
            print(f"{method:8s} seed {seed}: LOG MISSING")
            continue

        text = path.read_text(errors="ignore")

        done_matches = re.findall(
            r"\[done\]\s*best val_loss=([0-9.eE+-]+)\s*at epoch\s*(\d+)",
            text,
        )

        if done_matches:
            val, epoch = done_matches[-1]
            val = float(val)
        else:
            val_matches = re.findall(
                r"\[epoch\s+(\d+)\]\s+val_loss=([0-9.eE+-]+)",
                text,
            )

            if not val_matches:
                print(f"{method:8s} seed {seed}: no validation result found")
                continue

            epoch, val = min(val_matches, key=lambda x: float(x[1]))
            val = float(val)

        values.append(val)
        print(
            f"{method:8s} seed {seed}: "
            f"best val_loss={val:.6f}, epoch={epoch}"
        )

    all_values[method] = values

print("\n===== THREE-SEED SUMMARY =====")

for method, values in all_values.items():
    if len(values) == 3:
        mean = statistics.mean(values)
        std = statistics.stdev(values)
        print(f"{method:8s}: {mean:.6f} ± {std:.6f}")
    else:
        print(f"{method:8s}: only {len(values)}/3 valid results")

if (
    len(all_values.get("Baseline", [])) == 3
    and len(all_values.get("PACT", [])) == 3
):
    b = statistics.mean(all_values["Baseline"])
    p = statistics.mean(all_values["PACT"])
    reduction = (b - p) / b * 100.0
    print(f"PACT mean reduction relative to baseline: {reduction:.2f}%")

if missing:
    raise SystemExit(f"\nERROR: {len(missing)} checkpoint(s) missing")

print("\nAll six checkpoints are present.")
PY
