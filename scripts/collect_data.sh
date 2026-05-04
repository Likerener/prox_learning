#!/usr/bin/env bash
# Collect N trajectories with the streaming sentinel watching alongside.
# PROJECT.md §6 Day 3.
#
# Usage:
#   bash scripts/collect_data.sh near_contact 1000
#   bash scripts/collect_data.sh near_contact 1000 --bad-streak 20
set -euo pipefail
cd "$(dirname "$0")/.."

TASK=${1:-near_contact}
N_TRAJ=${2:-1000}
shift 2 || true
SENTINEL_ARGS="$*"

OUT_DIR="data/raw/${TASK}"
LOG_DIR="reports/logs"
REPORT_DIR="reports/checks/collect_${TASK}_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR" "$LOG_DIR" "$REPORT_DIR"

# Clear any stale abort marker from a previous run.
rm -f "$OUT_DIR/SENTINEL_ABORT"

SESSION="collect_${TASK}"
SENTINEL_SESSION="sentinel_${TASK}"

# Sentinel in its own tmux session.
tmux new-session -d -s "$SENTINEL_SESSION" \
  "python -m pla.data.sentinel \
      --data-dir $OUT_DIR \
      --target-n $N_TRAJ \
      --bad-streak ${BAD_STREAK:-15} \
      --heartbeat-every 25 \
      --report $REPORT_DIR/sentinel.json \
      $SENTINEL_ARGS \
      2>&1 | tee $LOG_DIR/sentinel_${TASK}.log"

# Collector in its own tmux session.
tmux new-session -d -s "$SESSION" \
  "python -m pla.data.collect \
      --config configs/data/${TASK}.yaml \
      --out-dir $OUT_DIR \
      --n-traj ${N_TRAJ} \
      2>&1 | tee $LOG_DIR/collect_${TASK}.log"

echo "Started:"
echo "  collector: tmux attach -t $SESSION"
echo "  sentinel:  tmux attach -t $SENTINEL_SESSION"
echo
echo "If the sentinel writes $OUT_DIR/SENTINEL_ABORT the collector will stop"
echo "cleanly between episodes. Reports land in $REPORT_DIR."
echo
echo "When the run finishes, verify with:"
echo "  python -m pla.data.verify --data-dir $OUT_DIR --strict"
