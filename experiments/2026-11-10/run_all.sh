#!/bin/bash
# Reproduce the 2026-11-10 experiment: 4 arms x 8 seeds x 150 generations,
# graded by the shipped evaluator and by a fresh-key copy. Run from the repo root.
set -e
OUT=${OUT:-/tmp/gexp}; mkdir -p "$OUT"
sed "s/0xE7A15EED/0x5EEDF00D/g" evaluator-cli.js > "$OUT/evaluator-fresh.js"
one() { arm=$1; seed=$2; d="$OUT/${arm//+/_}_$seed"
  node run.js --arm "$arm" --gens 150 --seed "$seed" --out "$d" >/dev/null
  node evaluator-cli.js "$d/descendant.html" > "$d/eval.json"
  node "$OUT/evaluator-fresh.js" "$d/descendant.html" > "$d/fresh.json"
  rm -f "$d/descendant.html" "$d/lineage.json"; echo "$arm $seed done"; }
export -f one; export OUT
for s in 101 202 303 404 505 606 707 808; do for a in train hyper code code+scorer; do echo "$a $s"; done; done | xargs -P2 -n2 bash -c 'one "$@"' _
OUT="$OUT" node experiments/2026-11-10/analyze.js
