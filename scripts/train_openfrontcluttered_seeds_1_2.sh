#!/usr/bin/env bash
set -euo pipefail

REPO=/home/qinzhengfangli/molmo_test/prox_learning
PY=/home/qinzhengfangli/molmo_test/molmospaces/.venv/bin/python

export PYTHONNOUSERSITE=1
export PYTHONPATH="$REPO:$REPO/submodules/act:${PYTHONPATH:-}"

cd "$REPO"
mkdir -p logs runs

for SEED in 1 2; do
  echo "===== BASELINE seed $SEED ====="

  "$PY" -m pact.act_prox.imitate_episodes_with_prox \
    --task_name openfrontcluttered_small \
    --policy_class ACT \
    --ckpt_dir "runs/openfrontcluttered_baseline_seed${SEED}_20260622" \
    --batch_size 4 \
    --num_epochs 200 \
    --lr 1e-4 \
    --seed "$SEED" \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --num_workers 1 \
    2>&1 | tee "logs/openfrontcluttered_baseline_seed${SEED}_20260622.log"

  echo "===== PACT seed $SEED ====="

  "$PY" -m pact.act_prox.imitate_episodes_with_prox \
    --task_name openfrontcluttered_small \
    --policy_class ACT \
    --ckpt_dir "runs/openfrontcluttered_pact_seed${SEED}_20260622" \
    --batch_size 4 \
    --num_epochs 200 \
    --lr 1e-4 \
    --seed "$SEED" \
    --kl_weight 10 \
    --chunk_size 100 \
    --hidden_dim 512 \
    --dim_feedforward 3200 \
    --num_workers 1 \
    --use_proximity \
    --prox_encoder_ckpt pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt \
    --prox_mapping_json act_style_data/openfrontcluttered_small_20260622/prox_mapping.json \
    --prox_tokens_per_sensor 6 \
    2>&1 | tee "logs/openfrontcluttered_pact_seed${SEED}_20260622.log"
done
