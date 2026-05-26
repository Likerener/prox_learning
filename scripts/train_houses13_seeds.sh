#!/bin/bash
# Train vanilla ACT and PACT on the new house_3 dataset across 3 seeds.
# Sequential execution to share the single RTX 4090.
#
# Config mirrors the proven runs/act_prox_mug_v1/ (Δ = seed and dataset only):
#   batch_size=8 num_epochs=2000 lr=1e-6 kl_weight=10
#   chunk_size=100 hidden_dim=512 dim_feedforward=3200
#
# Outputs:
#   runs/act_house3_mug_seed{42,1337,2026}/
#   runs/act_prox_house3_seed{42,1337,2026}/
#   logs/train_house3_seeds_<ts>/<run>.log

set -euo pipefail
cd /home/jaydv/code/prox_learning

PY=/opt/conda/envs/aloha/bin/python
TASK=pla_houses_1_3_mug_random
PROX_CKPT=pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt
PROX_MAP=act_style_data/mug_houses_1_3_random_everything/prox_mapping.json

TS=$(date +%Y%m%d_%H%M%S)
LOG_DIR=logs/train_house3_seeds_${TS}
mkdir -p "${LOG_DIR}"

SEEDS=(42 1337 2026)

# -------- vanilla ACT runs (3 seeds) --------
for s in "${SEEDS[@]}"; do
  RUN_DIR=runs/act_houses13_mug_seed${s}
  LOG="${LOG_DIR}/act_seed${s}.log"
  echo "[launch $(date +%H:%M:%S)] vanilla ACT seed=${s} -> ${RUN_DIR}" | tee -a "${LOG}"
  ${PY} -m pact.act_prox.imitate_episodes_with_prox \
    --task_name ${TASK} --policy_class ACT \
    --ckpt_dir ${RUN_DIR} \
    --batch_size 8 --num_epochs 2000 --lr 1e-6 --seed ${s} \
    --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --dim_feedforward 3200 \
    --use_wandb --wandb_project pact \
    --wandb_run_name act_houses13_mug_seed${s} \
    >> "${LOG}" 2>&1
  echo "[done   $(date +%H:%M:%S)] vanilla ACT seed=${s}" | tee -a "${LOG}"
done

# -------- PACT runs (3 seeds) --------
for s in "${SEEDS[@]}"; do
  RUN_DIR=runs/act_prox_houses13_seed${s}
  LOG="${LOG_DIR}/pact_seed${s}.log"
  echo "[launch $(date +%H:%M:%S)] PACT seed=${s} -> ${RUN_DIR}" | tee -a "${LOG}"
  ${PY} -m pact.act_prox.imitate_episodes_with_prox \
    --task_name ${TASK} --policy_class ACT \
    --ckpt_dir ${RUN_DIR} \
    --batch_size 8 --num_epochs 2000 --lr 1e-6 --seed ${s} \
    --kl_weight 10 --chunk_size 100 \
    --hidden_dim 512 --dim_feedforward 3200 \
    --use_proximity \
    --prox_encoder_ckpt ${PROX_CKPT} \
    --prox_mapping_json ${PROX_MAP} \
    --use_wandb --wandb_project pact \
    --wandb_run_name act_prox_houses13_seed${s} \
    >> "${LOG}" 2>&1
  echo "[done   $(date +%H:%M:%S)] PACT seed=${s}" | tee -a "${LOG}"
done

echo "[pipeline-done $(date +%H:%M:%S)] all 6 runs complete in ${LOG_DIR}"
