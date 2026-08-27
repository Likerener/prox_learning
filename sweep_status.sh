#!/usr/bin/env bash
# Progress report for the running open-front sweep.
PL=${PL:-~/molmo_test/prox_learning}
O=$(ls -dt "$PL"/eval_output/openfront_sweep_* 2>/dev/null | head -1)

echo "== tmux =="; tmux ls 2>&1 | head -3
echo
echo "== live processes =="
ps -eo pid,etimes,pcpu,args --sort=-pcpu | grep -E "eval_(act|pact)_openfront" | grep -v grep |
  awk '{printf "  pid=%-7s elapsed=%-6dm cpu=%-6s %s\n", $1, $2/60, $3"%", $5}' || true
[ -z "$O" ] && { echo "no sweep output dir yet"; exit 0; }

echo
echo "== output: $O =="
for d in "$O"/*/*/; do
    [ -f "$d/run.log" ] || continue
    cell=$(echo "$d" | rev | cut -d/ -f2,3 | rev | tr '/' ' ')
    houses=$(grep -c "House .* complete" "$d/run.log" 2>/dev/null || echo 0)
    state=$([ -f "$d/DONE" ] && echo DONE || echo running)
    mtime=$(date -r "$d/run.log" +%H:%M:%S 2>/dev/null)
    printf "  %-28s %-8s houses %s/8   last write %s\n" "$cell" "$state" "$houses" "$mtime"
done

echo
echo "== results so far =="
column -s, -t "$O/results.csv" 2>/dev/null
