#!/usr/bin/env bash
# Run the dup250-env eval for each ACT checkpoint trained on the dup250 dataset.
# Sequential to avoid GPU contention; 10 rollouts each.
set -uo pipefail

cd /home/jaydv/code/prox_learning/submodules/act

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

OUT_ROOT=/home/jaydv/code/prox_learning/eval_output
PY=/opt/conda/envs/mlspaces/bin/python

declare -a CKPTS=(
    "act_house1_mug_v1"
    "act_house1_mug_v2_long"
    "act_house1_mug_v3"
)

for tag in "${CKPTS[@]}"; do
    out_dir="${OUT_ROOT}/${tag}_dup250env_10ep"
    echo "[run] ${tag} -> ${out_dir}"
    "${PY}" eval_act_house1_dup250.py \
        --ckpt_dir "/home/jaydv/code/prox_learning/submodules/act/ckpts/${tag}" \
        --output_dir "${out_dir}" \
        --num_rollouts 10 \
        --task_horizon 300 \
        --seed 2026 \
        2>&1 | tee "${out_dir}.log"
done

echo "[run] all evals done."
for tag in "${CKPTS[@]}"; do
    out_dir="${OUT_ROOT}/${tag}_dup250env_10ep"
    if [[ -f "${out_dir}/summary.txt" ]]; then
        echo "=== ${tag} ==="
        cat "${out_dir}/summary.txt"
    fi
done
