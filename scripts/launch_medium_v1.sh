#!/usr/bin/env bash
# Full medium-v1 training + eval pipeline.
#
# Sequencing on the single-GPU box:
#   1. Train PLA      (use_proximity=true,  use_language=true)  ~6h
#   2. Train baseline (use_proximity=false, use_language=true)  ~6h
#   3. Rollout PLA      on houses 21-30, seed 2028             ~1.5h
#   4. Rollout baseline on houses 21-30, seed 2028             ~1.5h
#   5. Compare buckets + write analysis_output/.../comparison.md
#
# Train runs are sequential because both saturate the 4090. The two rollouts
# could in principle parallelize (each uses ~1 worker) but we keep them
# sequential to avoid scene-XML I/O contention with anything else.
#
# Auto-discovers the latest timestamp dir under
# assets/datagen/pick_and_place_skin_pilot_medium_v1/.../<timestamp>/ — so
# you do NOT need to edit the path when re-running.
#
# Usage:
#   bash scripts/launch_medium_v1.sh                # full run
#   DRY=1 bash scripts/launch_medium_v1.sh          # print commands, don't execute
#   SKIP_TRAIN=1 bash scripts/launch_medium_v1.sh   # skip training (rollout-only)
#   SKIP_ROLLOUT=1 bash scripts/launch_medium_v1.sh # skip rollout (train-only)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# environment — TWO env split:
#   TRAIN_PY  → MolmoBot-Pi0 venv (has CLIP/transformers + ACT deps; missing
#               mujoco_warp, which is OK because training only uses recorded
#               h5 data, never the simulator).
#   EVAL_PY   → /opt/conda/envs/mlspaces  (has mujoco_warp + filament; needs
#               `transformers` package — installed once on 2026-05-12).
# Training-only and rollout-only invocations each use the env that has the
# deps they need. Both envs share /home/jaydv/code/prox_learning/runs/, so
# checkpoints saved by MolmoBot-Pi0 load fine in mlspaces.
# ---------------------------------------------------------------------------
TRAIN_VENV="$REPO_ROOT/submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/activate"
EVAL_PY="/opt/conda/envs/mlspaces/bin/python"
if [ ! -f "$TRAIN_VENV" ]; then
    echo "ERROR: training venv not found at $TRAIN_VENV" >&2
    exit 1
fi
if [ ! -x "$EVAL_PY" ]; then
    echo "ERROR: eval interpreter not found at $EVAL_PY" >&2
    exit 1
fi

export MLSPACES_ASSETS_DIR="$REPO_ROOT/assets"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# auto-discover the latest medium-dataset timestamp dir
# ---------------------------------------------------------------------------
DATA_PARENT="$REPO_ROOT/assets/datagen/pick_and_place_skin_pilot_medium_v1/FrankaSkinPickAndPlacePilotMediumConfig"
if [ ! -d "$DATA_PARENT" ]; then
    echo "ERROR: medium dataset parent dir missing: $DATA_PARENT" >&2
    exit 1
fi
DATA_ROOT="$(find "$DATA_PARENT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
if [ -z "$DATA_ROOT" ]; then
    echo "ERROR: no timestamped run dir under $DATA_PARENT" >&2
    exit 1
fi
N_HOUSES="$(find "$DATA_ROOT" -maxdepth 1 -type d -name 'house_*' | wc -l)"
N_H5="$(find "$DATA_ROOT" -name 'trajectories_batch_*.h5' | wc -l)"
echo "[launch] data root: $DATA_ROOT"
echo "[launch] houses: $N_HOUSES   h5 files: $N_H5"

if [ "$N_HOUSES" -lt 50 ]; then
    echo "WARNING: only $N_HOUSES houses present — medium target is ~100." >&2
    echo "         Pass --force-go (or just rerun later) if collection is still in progress." >&2
    if [ "${1:-}" != "--force-go" ]; then
        exit 2
    fi
fi

# ---------------------------------------------------------------------------
# tags + paths
# ---------------------------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
PLA_RUN="medium_pla_v1_${TS}"
VLM_RUN="medium_vlm_v1_${TS}"
PLA_ROLLOUT="rollout_${PLA_RUN}"
VLM_ROLLOUT="rollout_${VLM_RUN}"
COMPARE_DIR="$REPO_ROOT/analysis_output/compare_medium_v1_${TS}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# Eval houses are 21-30 (disjoint from training houses 1-100 by construction:
# training uses procthor-objaverse-train but seed 2026 + sampled tasks per
# house, so even reusing scene 21 with seed 2028 yields a different object/
# receptacle pair — see molmospaces task_sampler.py:1141 for sampling logic).
EVAL_HOUSE_INDS="21,22,23,24,25,26,27,28,29,30"
EVAL_SAMPLES_PER_HOUSE=2
EVAL_SEED=2028
EVAL_WORKERS=2

# ---------------------------------------------------------------------------
# helper: print + run (or just print if DRY=1)
# ---------------------------------------------------------------------------
DRY="${DRY:-0}"
run() {
    echo
    echo "[launch] $*"
    echo
    if [ "$DRY" != "1" ]; then
        "$@"
    fi
}

# ---------------------------------------------------------------------------
# step 1+2: train both policies sequentially
# ---------------------------------------------------------------------------
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    # Training uses the MolmoBot-Pi0 venv.
    # shellcheck source=/dev/null
    . "$TRAIN_VENV"

    echo
    echo "============ STEP 1: train PLA ($PLA_RUN) ============"
    run python -m pla.train \
        --use_proximity true \
        --use_language true \
        --run_name "$PLA_RUN" \
        --data_root "$DATA_ROOT" \
        --num_steps 50000 \
        --num_workers 2 \
        --use_wandb true \
        --wandb_project pla \
        2>&1 | tee "$LOG_DIR/train_${PLA_RUN}.log"

    echo
    echo "============ STEP 2: train baseline ($VLM_RUN) ============"
    run python -m pla.train \
        --use_proximity false \
        --use_language true \
        --run_name "$VLM_RUN" \
        --data_root "$DATA_ROOT" \
        --num_steps 50000 \
        --num_workers 2 \
        --use_wandb true \
        --wandb_project pla \
        2>&1 | tee "$LOG_DIR/train_${VLM_RUN}.log"

    # Drop the training venv before moving to rollout; mlspaces python is
    # invoked by absolute path so we don't need to `deactivate` formally,
    # but unsetting VIRTUAL_ENV keeps `which python` honest if the user
    # inspects the running script.
    deactivate 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# step 3+4: roll out both checkpoints on held-out houses 21-30
# ---------------------------------------------------------------------------
PLA_CKPT="$REPO_ROOT/runs/$PLA_RUN/latest.pt"
VLM_CKPT="$REPO_ROOT/runs/$VLM_RUN/latest.pt"

if [ "${SKIP_ROLLOUT:-0}" != "1" ]; then
    # Rollout uses the mlspaces conda env (mujoco_warp is here).
    # Add molmospaces to PYTHONPATH so the editable-install fallback works
    # consistently whether or not the conda env's pth file is loaded.
    export PYTHONPATH="$REPO_ROOT/submodules/molmospaces:$REPO_ROOT"

    echo
    echo "============ STEP 3: rollout PLA ($PLA_ROLLOUT) ============"
    run "$EVAL_PY" -m pla.rollout_eval \
        --checkpoint "$PLA_CKPT" \
        --run_name "$PLA_ROLLOUT" \
        --use_proximity true \
        --use_language true \
        --seed "$EVAL_SEED" \
        --house_inds "$EVAL_HOUSE_INDS" \
        --samples_per_house "$EVAL_SAMPLES_PER_HOUSE" \
        --num_workers "$EVAL_WORKERS" \
        2>&1 | tee "$LOG_DIR/rollout_${PLA_ROLLOUT}.log"

    echo
    echo "============ STEP 4: rollout baseline ($VLM_ROLLOUT) ============"
    run "$EVAL_PY" -m pla.rollout_eval \
        --checkpoint "$VLM_CKPT" \
        --run_name "$VLM_ROLLOUT" \
        --use_proximity false \
        --use_language true \
        --seed "$EVAL_SEED" \
        --house_inds "$EVAL_HOUSE_INDS" \
        --samples_per_house "$EVAL_SAMPLES_PER_HOUSE" \
        --num_workers "$EVAL_WORKERS" \
        2>&1 | tee "$LOG_DIR/rollout_${VLM_ROLLOUT}.log"

    echo
    echo "============ STEP 5: compare ============"
    run "$EVAL_PY" -m pla.rollout_compare \
        --pla_run "rollout_output/$PLA_ROLLOUT" \
        --baseline_run "rollout_output/$VLM_ROLLOUT" \
        --out_dir "$COMPARE_DIR" \
        2>&1 | tee "$LOG_DIR/compare_medium_v1_${TS}.log"

    echo
    echo "============ RESULTS ============"
    if [ -f "$COMPARE_DIR/comparison.md" ]; then
        cat "$COMPARE_DIR/comparison.md"
    fi
fi

echo
echo "[launch] DONE."
echo "[launch]   PLA checkpoint:   $PLA_CKPT"
echo "[launch]   VLM checkpoint:   $VLM_CKPT"
echo "[launch]   Comparison:       $COMPARE_DIR/comparison.md"
