#!/usr/bin/env bash
# ACT vs PACT closed-loop sweep in the open-front cluttered env.
# One cell = one full deterministic pass over HOUSES x SAMPLES. Resumable.
set -uo pipefail

source "$(dirname "$0")/openfront_env.sh"

OUT=${OUT:-$PL/eval_output/openfront_sweep_$(date +%Y%m%d)}
SAMPLES=${SAMPLES:-6}
HOUSES=${HOUSES:-"12,13,14,15,16,17,23,25"}
CONDS=${CONDS:-"none exo_off blackout dim wrist_off"}
SEEDTAG=${SEEDTAG:-seed0}
POLS=${POLS:-"act pact"}

ACT_CKPT=${ACT_CKPT:?e.g. $PL/runs/openfrontcluttered_52_baseline_seed0_20260623}
PACT_CKPT=${PACT_CKPT:?e.g. $PL/runs/openfrontcluttered_52_pact_seed0_20260623}
PROX_ENC=${PROX_ENC:-$PL/pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt}
PROX_MAP=${PROX_MAP:?e.g. $PL/act_style_data/openfrontcluttered_52_20260623/prox_mapping.json}

CSV=$OUT/results.csv
mkdir -p "$OUT"
[ -f "$CSV" ] || echo "policy,condition,seedtag,success,total" > "$CSV"

run_cell() {
    local pol=$1 cond=$2
    local dir=$OUT/$pol-$SEEDTAG/$cond
    [ -f "$dir/DONE" ] && { echo "[skip] $pol/$cond"; return 0; }
    mkdir -p "$dir"
    local log=$dir/run.log
    echo "[running] $pol / $cond"
    case "$pol" in
      act)
        "$PY" "$PL/submodules/act/eval_act_openfrontcluttered.py" \
            --ckpt_dir "$ACT_CKPT" --output_dir "$dir" \
            --house_inds "$HOUSES" --samples_per_house "$SAMPLES" \
            --degrade_vision "$cond" --no_videos >"$log" 2>&1 ;;
      pact)
        "$PY" "$PL/pact/act_prox/eval_pact_openfrontcluttered.py" \
            --ckpt_dir "$PACT_CKPT" --prox_encoder_ckpt "$PROX_ENC" \
            --prox_mapping_json "$PROX_MAP" --output_dir "$dir" \
            --house_inds "$HOUSES" --samples_per_house "$SAMPLES" \
            --degrade_vision "$cond" --no_videos >"$log" 2>&1 ;;
      pact_proxzero)
        "$PY" "$PL/pact/act_prox/eval_pact_openfrontcluttered.py" \
            --ckpt_dir "$PACT_CKPT" --prox_encoder_ckpt "$PROX_ENC" \
            --prox_mapping_json "$PROX_MAP" --output_dir "$dir" \
            --house_inds "$HOUSES" --samples_per_house "$SAMPLES" \
            --mask_proximity zero --degrade_vision "$cond" --no_videos >"$log" 2>&1 ;;
    esac
    local line
    line=$(grep -oE "success [0-9]+/[0-9]+" "$log" | tail -1)
    if [ -n "${line:-}" ]; then
        local st; st=$(echo "$line" | awk '{print $2}')
        echo "$pol,$cond,$SEEDTAG,${st%/*},${st#*/}" >> "$CSV"
        touch "$dir/DONE"
        echo "[done] $pol / $cond : $st"
    else
        echo "[WARN] no success line for $pol/$cond — check $log"
        tail -3 "$log"
    fi
}

for cond in $CONDS; do
    for pol in $POLS; do run_cell "$pol" "$cond"; done
done
[ "$POLS" = "act pact" ] && run_cell pact_proxzero none

echo; echo "== results =="; column -s, -t "$CSV"
