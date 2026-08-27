#!/usr/bin/env bash
# Gather every fume-hood house dir under one directory of symlinks so the
# converter (which takes a single --src run dir) can see them all at once.
# House names are prefixed with their source run to keep them unique; the
# converter globs house_*/trajectories*.h5, so the prefix must keep that shape.
set -euo pipefail
PL=${PL:-~/molmo_test/prox_learning}
OUT=$PL/assets/datagen/fumehood_merged
rm -rf "$OUT"; mkdir -p "$OUT"
n=0
for h in $PL/assets/datagen/fumehood_{probe,repro,s9}*/house_*; do
    [ -d "$h" ] || continue
    [ -n "$(find "$h" -maxdepth 1 -name 'trajectories_batch_*.h5' -print -quit)" ] || continue
    run=$(basename "$(dirname "$h")")
    ln -s "$h" "$OUT/house_${run#fumehood_}_$(basename "$h" | sed 's/house_//')"
    n=$((n+1))
done
echo "linked $n house dirs into $OUT"
ls "$OUT" | head -20
