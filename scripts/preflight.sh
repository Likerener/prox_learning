#!/usr/bin/env bash
# Pre-flight checks before a long collection run.
#
# Usage:
#   bash scripts/preflight.sh near_contact 1000
#   bash scripts/preflight.sh near_contact 1000 --full     # also run 1-traj round-trip
#   bash scripts/preflight.sh near_contact 1000 --strict   # exit non-zero on failure
#
# What it does:
#   1. pla.data.preflight  — env / MJCF / sensor-render / disk checks
#   2. (optional) one-episode round-trip: write + schema-validate one shard
#   3. Pilot collection of 50 episodes, watched by the sentinel
#   4. pla.data.verify --strict on the pilot
# Only after all four pass does the script exit 0; you then launch the
# 1000-traj collection separately.
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-near_contact}
N_TRAJ=${2:-1000}
shift 2 || true
EXTRA_ARGS="$*"

CFG="configs/data/${TASK}.yaml"
OUT_DIR="data/raw/${TASK}"
PILOT_DIR="data/raw/${TASK}_pilot"
REPORT_DIR="reports/checks/preflight_${TASK}"
mkdir -p "$REPORT_DIR" "$OUT_DIR"

echo "============================================================"
echo "  Pre-flight: task=$TASK  target=$N_TRAJ"
echo "============================================================"
echo

echo "[1/4] Static checks (MJCF, sensor render, disk, env import)"
python -m pla.data.preflight \
    --config "$CFG" \
    --n-traj "$N_TRAJ" \
    --out-dir "$OUT_DIR" \
    --report "$REPORT_DIR/static.json" \
    $EXTRA_ARGS
echo

echo "[2/4] Pilot collection (50 episodes) into $PILOT_DIR"
rm -f "$PILOT_DIR/SENTINEL_ABORT"
mkdir -p "$PILOT_DIR"
# Sentinel runs in the background; abort marker if 5 consecutive bad shards.
python -m pla.data.sentinel \
    --data-dir "$PILOT_DIR" \
    --target-n 50 \
    --bad-streak 5 \
    --heartbeat-every 10 \
    --report "$REPORT_DIR/sentinel_pilot.json" \
    > "$REPORT_DIR/sentinel_pilot.log" 2>&1 &
SENTINEL_PID=$!
trap 'kill $SENTINEL_PID 2>/dev/null || true' EXIT

python -m pla.data.collect \
    --config "$CFG" \
    --out-dir "$PILOT_DIR" \
    --n-traj 50

# Give the sentinel a moment to drain its watch queue, then stop it.
sleep 2
kill $SENTINEL_PID 2>/dev/null || true
trap - EXIT
echo

echo "[3/4] Sentinel report (last 30 lines):"
tail -n 30 "$REPORT_DIR/sentinel_pilot.log" || true
echo

echo "[4/5] Deep verification on the pilot dataset"
python -m pla.data.verify \
    --data-dir "$PILOT_DIR" \
    --report "$REPORT_DIR/verify_pilot.json" \
    --strict
echo

echo "[5/5] Visual audit (you must eyeball these before launching)"
python -m pla.viz.dataset_audit \
    --data-dir "$PILOT_DIR" \
    --out "$REPORT_DIR/audit"
echo

echo "============================================================"
echo "  PRE-FLIGHT PASS. Pilot data in $PILOT_DIR."
echo "  Reports + audit plots in $REPORT_DIR/"
echo
echo "  >>> EYEBALL THE PLOTS BEFORE LAUNCHING THE FULL RUN <<<"
echo "    cat $REPORT_DIR/audit/INDEX.md"
echo "    Open these PNGs in order:"
echo "      $REPORT_DIR/audit/01_tof_montage.png"
echo "      $REPORT_DIR/audit/02_per_sensor_dist.png"
echo "      $REPORT_DIR/audit/03_sensor_coverage.png"
echo "      $REPORT_DIR/audit/04_episode_traces.png"
echo "      $REPORT_DIR/audit/05_rgb_strip.png"
echo "      $REPORT_DIR/audit/06_length_distribution.png"
echo "      $REPORT_DIR/audit/07_action_distribution.png"
echo
echo "  When satisfied, launch the full run:"
echo "    bash scripts/collect_data.sh $TASK $N_TRAJ"
echo "============================================================"
