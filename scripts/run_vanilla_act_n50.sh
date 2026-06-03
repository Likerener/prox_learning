#!/usr/bin/env bash
# Re-eval vanilla ACT at n=50 (fresh random sampling, in case the existing
# 36/50 doesn't help P+ACT). Launches after mask_zero finishes to avoid GPU OOM.
#
# Usage:
#   nohup bash scripts/run_vanilla_act_n50.sh > eval_output/_exp_logs/vanilla_rerun.log 2>&1 &

set -e
REPO=/home/jaydv/code/prox_learning
cd "$REPO"

echo "[$(date +%H:%M:%S)] waiting for mask_zero summary.json"
until [ -f eval_output/exp1_mask_zero_n50/summary.json ]; do sleep 60; done
echo "[$(date +%H:%M:%S)] mask_zero done"

# Wait briefly for the GPU to settle.
sleep 10

# Launch a single-process parallel runner for vanilla ACT.  Re-uses the
# existing submodules/act/eval_act_mug_random.py path.
OUT="$REPO/eval_output/vanilla_act_rerun_n50"
mkdir -p "$OUT"

EVAL_SCRIPT="$REPO/submodules/act/eval_act_mug_random.py"
VANILLA_CKPT_DIR="$REPO/submodules/act/ckpts/act_house1_mug_random_v1"
PY=/opt/conda/envs/mlspaces/bin/python

PARALLEL=2   # share with mask_mean (which uses parallel=3) - tight but allowed
TOTAL=50

# This is a simple sequential-N-with-parallel-K runner. Use ProcessPool via xargs.
cat > "$OUT/_run_one.sh" <<'EOF'
#!/usr/bin/env bash
idx=$1
OUT=$2
VANILLA_CKPT_DIR=$3
EVAL_SCRIPT=$4
REPO=/home/jaydv/code/prox_learning
ACT_DIR="$REPO/submodules/act"

run_dir="$OUT/run_$(printf '%02d' $idx)"
log="$OUT/eval_log_run_$(printf '%02d' $idx).txt"

rm -rf "$run_dir"
mkdir -p "$run_dir"

cd "$ACT_DIR"
PYTHONPATH="$ACT_DIR:$REPO:${PYTHONPATH:-}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    /opt/conda/envs/mlspaces/bin/python "$EVAL_SCRIPT" \
    --ckpt_dir "$VANILLA_CKPT_DIR" \
    --output_dir "$run_dir" \
    > "$log" 2>&1
EOF
chmod +x "$OUT/_run_one.sh"

echo "[$(date +%H:%M:%S)] launching vanilla n=$TOTAL at parallel=$PARALLEL"
seq 0 $((TOTAL-1)) | xargs -P "$PARALLEL" -I{} \
    bash "$OUT/_run_one.sh" {} "$OUT" "$VANILLA_CKPT_DIR" "$EVAL_SCRIPT"

# Aggregate
$PY - <<PYAGG
import re, json, csv, os
from pathlib import Path
import numpy as np
out = Path("$OUT")
rows = []
for d in sorted(out.glob("run_*")):
    idx = int(d.name.split("_")[-1])
    log_path = out / f"eval_log_run_{idx:02d}.txt"
    s, t = 0, 0
    if log_path.exists():
        for line in open(log_path).read().splitlines():
            m = re.search(r"success\s+(\d+)\s*/\s*(\d+)", line)
            if m: s, t = int(m.group(1)), int(m.group(2))
    rows.append({"run_idx": idx, "success": s, "total": t,
                 "success_rate": s/t if t else 0.0})
fields = ["run_idx","success","total","success_rate"]
with open(out/"results.csv","w") as f:
    w=csv.DictWriter(f, fieldnames=fields); w.writeheader()
    for r in rows: w.writerow(r)
tot_s = sum(r["success"] for r in rows)
tot_t = sum(r["total"] for r in rows)
rate = tot_s/tot_t if tot_t else 0
z=1.96; p=rate; n=tot_t; denom=1+z*z/n; centre=(p+z*z/(2*n))/denom; half=z*((p*(1-p)/n + z*z/(4*n*n))**0.5)/denom
summary = {"n_runs": len(rows), "total_episodes": tot_t, "total_successes": tot_s,
           "pooled_success_rate": rate, "wilson_95_ci": [centre-half, centre+half]}
print(json.dumps(summary, indent=2))
with open(out/"summary.json","w") as f: json.dump(summary,f,indent=2)
PYAGG

echo "[$(date +%H:%M:%S)] vanilla re-eval done"
