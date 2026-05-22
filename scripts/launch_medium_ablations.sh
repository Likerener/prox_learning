#!/usr/bin/env bash
# Full ablation launcher: 4 PLA/VLM variants + multi-seed eval harness.
#
# Models trained (sequential, each saturates the GPU):
#   1. medium_pla_v1            — HEADLINE: PLA with link2 masked (default)
#   2. medium_vlm_v1            — BASELINE: RGB only (no proximity)
#   3. medium_pla_no_mask_v1    — ABLATION: PLA without any link masking
#                                 (defends headline against "the mask did the work")
#   4. medium_pla_ee_only_v1    — ABLATION: PLA with link2+link3 masked
#                                 (tests whether the forearm sensor contributes)
#
# (Oracle privileged-state baseline #5 from the launch plan is intentionally
#  NOT here — it needs a new policy class. See task #48 in TODO.md.)
#
# Eval (after all training):
#   pla.eval_harness drives K=5 seeds × 10 houses × 2 samples × 4 models = 400
#   rollouts, then writes one CSV + summary.json + 8 plots + paper_report.md
#   under analysis_output/eval_medium_v1/.
#   Clutter bins are LOCKED from the planner holdout data
#   (scripts/lock_clutter_bins.py).
#
# Usage:
#   bash scripts/launch_medium_ablations.sh                 # full pipeline
#   DRY=1 bash scripts/launch_medium_ablations.sh           # print, don't execute
#   SKIP_TRAIN=1 bash scripts/launch_medium_ablations.sh    # eval-only
#   SKIP_EVAL=1  bash scripts/launch_medium_ablations.sh    # train-only
#   ONLY=pla,vlm bash scripts/launch_medium_ablations.sh    # subset
#   NUM_STEPS=10000 bash scripts/launch_medium_ablations.sh # short sanity run
#
# Environment-split rule preserved:
#   training  → submodules/MolmoBot/MolmoBot-Pi0/.venv  (has CLIP+ACT, no mujoco)
#   rollout   → /opt/conda/envs/mlspaces                (has mujoco_warp)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TRAIN_VENV="$REPO_ROOT/submodules/MolmoBot/MolmoBot-Pi0/.venv/bin/activate"
EVAL_PY="/opt/conda/envs/mlspaces/bin/python"
[ -f "$TRAIN_VENV" ] || { echo "ERROR: $TRAIN_VENV missing" >&2; exit 1; }
[ -x "$EVAL_PY" ]   || { echo "ERROR: $EVAL_PY missing" >&2; exit 1; }

export MLSPACES_ASSETS_DIR="$REPO_ROOT/assets"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export JAX_PLATFORMS=cpu
export PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# auto-discover the latest medium-dataset timestamp dir
# ---------------------------------------------------------------------------
DATA_PARENT="$REPO_ROOT/assets/datagen/pick_and_place_skin_pilot_medium_v1/FrankaSkinPickAndPlacePilotMediumConfig"
[ -d "$DATA_PARENT" ] || { echo "ERROR: $DATA_PARENT missing" >&2; exit 1; }
DATA_ROOT="$(find "$DATA_PARENT" -mindepth 1 -maxdepth 1 -type d -name '20*' | sort | tail -1)"
[ -n "$DATA_ROOT" ] || { echo "ERROR: no timestamped run dir under $DATA_PARENT" >&2; exit 1; }
N_HOUSES="$(find "$DATA_ROOT" -maxdepth 1 -type d -name 'house_*' | wc -l)"
echo "[launch] data root: $DATA_ROOT   ($N_HOUSES houses)"

# ---------------------------------------------------------------------------
# tags
# ---------------------------------------------------------------------------
TS="$(date +%Y%m%d_%H%M%S)"
PLA_RUN="medium_pla_v1_${TS}"
VLM_RUN="medium_vlm_v1_${TS}"
NOMASK_RUN="medium_pla_no_mask_v1_${TS}"
EE_RUN="medium_pla_ee_only_v1_${TS}"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# `model_id` selectors (used by ONLY=... and the eval phase).
declare -A RUN_NAMES=(
    [pla]="$PLA_RUN"
    [vlm]="$VLM_RUN"
    [nomask]="$NOMASK_RUN"
    [eeonly]="$EE_RUN"
)
declare -A USE_PROX=(
    [pla]=true
    [vlm]=false
    [nomask]=true
    [eeonly]=true
)
declare -A MASK_LINKS=(
    [pla]=link2
    [vlm]=link2                   # no proximity used, value is irrelevant
    [nomask]=none
    [eeonly]=link2,link3
)
ORDER=(pla vlm nomask eeonly)
if [ -n "${ONLY:-}" ]; then
    IFS=',' read -ra ORDER <<< "$ONLY"
fi

NUM_STEPS="${NUM_STEPS:-50000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
BATCH_SIZE="${BATCH_SIZE:-8}"
# 50k steps × 4 models × 1.16 GB / ckpt = a lot. Default to every-5k so we
# get 10 intermediate ckpts per model (~46 GB total) instead of 50 (~232 GB).
CKPT_EVERY="${CKPT_EVERY:-5000}"
DRY="${DRY:-0}"

run() { echo; echo "[launch] $*"; echo; [ "$DRY" = "1" ] || "$@"; }

# ---------------------------------------------------------------------------
# Training phase
# ---------------------------------------------------------------------------
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
    # shellcheck source=/dev/null
    . "$TRAIN_VENV"
    for m in "${ORDER[@]}"; do
        rname="${RUN_NAMES[$m]}"
        echo
        echo "============ TRAIN $m → $rname  (use_proximity=${USE_PROX[$m]}, mask_links=${MASK_LINKS[$m]}) ============"
        run python -m pla.train \
            --use_proximity "${USE_PROX[$m]}" \
            --use_language true \
            --mask_links "${MASK_LINKS[$m]}" \
            --run_name "$rname" \
            --data_root "$DATA_ROOT" \
            --num_steps "$NUM_STEPS" \
            --batch_size "$BATCH_SIZE" \
            --num_workers "$NUM_WORKERS" \
            --ckpt_every "$CKPT_EVERY" \
            --use_wandb true \
            --wandb_project pla \
            2>&1 | tee "$LOG_DIR/train_${rname}.log"
    done
    deactivate 2>/dev/null || true
fi

# ---------------------------------------------------------------------------
# Eval phase — single eval_harness call covering all selected models
# ---------------------------------------------------------------------------
if [ "${SKIP_EVAL:-0}" != "1" ]; then
    EVAL_OUT="$REPO_ROOT/analysis_output/eval_medium_v1_${TS}"
    LOCKED_BINS="$REPO_ROOT/analysis_output/eval_medium_v1/clutter_bins.json"
    if [ ! -f "$LOCKED_BINS" ]; then
        echo "[launch] clutter_bins not locked — running scripts.lock_clutter_bins first"
        run "$EVAL_PY" -m scripts.lock_clutter_bins \
            --out "$LOCKED_BINS"
    fi

    # Build --models arg with name=path[:use_prox=...]
    MODELS_ARG=""
    for m in "${ORDER[@]}"; do
        rname="${RUN_NAMES[$m]}"
        ckpt="$REPO_ROOT/runs/$rname/latest.pt"
        # `pla` prefix opts in to use_proximity by default; explicit override
        # for everyone makes the call self-documenting.
        MODELS_ARG="$MODELS_ARG ${m}=${ckpt}:use_prox=${USE_PROX[$m]}"
    done

    export PYTHONPATH="$REPO_ROOT/submodules/molmospaces:$REPO_ROOT"
    echo
    echo "============ EVAL: 4-model harness ============"
    run "$EVAL_PY" -m pla.eval_harness \
        --models $MODELS_ARG \
        --seeds 2028,2029,2030,2031,2032 \
        --house_inds 11,12,13,14,15,16,17,18,19,20 \
        --samples_per_house 2 \
        --num_workers 2 \
        --out_dir "$EVAL_OUT" \
        --clutter_bins_path "$LOCKED_BINS" \
        2>&1 | tee "$LOG_DIR/eval_medium_ablations_${TS}.log"

    echo
    echo "============ RESULTS ============"
    echo "[launch] CSV:        $EVAL_OUT/eval_metrics.csv"
    echo "[launch] summary:    $EVAL_OUT/summary.json"
    echo "[launch] paper md:   $EVAL_OUT/paper_report.md"
fi

echo
echo "[launch] DONE."
for m in "${ORDER[@]}"; do
    rname="${RUN_NAMES[$m]}"
    echo "[launch]   $m:  runs/$rname/latest.pt"
done
