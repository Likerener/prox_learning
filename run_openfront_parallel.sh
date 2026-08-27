#!/usr/bin/env bash
# Fan sweep cells out across processes. llvmpipe saturates near 4 threads and
# pico has 12 cores, so 3 concurrent cells x 4 threads fills the machine without
# oversubscribing. Cells are resumable (DONE markers), so this can be re-run.
set -uo pipefail
cd "$(dirname "$0")"
source ./openfront_env.sh

export OUT=${OUT:-$PL/eval_output/openfront_sweep_$(date +%Y%m%d)}
export SAMPLES=${SAMPLES:-2}
export HOUSES=${HOUSES:-"12,13,14,15,16,17,23,25"}
export SEEDTAG=${SEEDTAG:-seed0}
export ACT_CKPT PACT_CKPT PROX_MAP
export PROX_ENC=${PROX_ENC:-$PL/pact/outputs_prox/runs/prox_encoder_v1/ckpt_best.pt}
export LP_NUM_THREADS=${LP_NUM_THREADS:-4}
JOBS=${JOBS:-3}
CELLS=${CELLS:-"act:none pact:none act:exo_off pact:exo_off"}

mkdir -p "$OUT"
echo "output: $OUT"
echo "cells:  $CELLS"
echo "jobs:   $JOBS x $LP_NUM_THREADS threads"

printf '%s\n' $CELLS | xargs -P "$JOBS" -I{} bash -c '
  cell={}
  POLS=${cell%%:*} CONDS=${cell##*:} ./run_openfront_sweep.sh >>"$OUT/${cell/:/_}.log" 2>&1
  echo "[finished] $cell"
'

echo; echo "== results =="; column -s, -t "$OUT/results.csv"
