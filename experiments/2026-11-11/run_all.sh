#!/bin/bash
# Reproduce the 2026-11-11 experiment from the repo root:
# 3 gate configs x 4 arms x 8 seeds x 150 generations (96 runs, ~35 min on 4 cores).
#   A = GTEE 1.1 defaults (gen-0 scorer, 64 resampled paired dev samples, paired_se k=1)
#   B = the 1.0 gate with only the "beat the last loser" bug removed
#       (genome scorer, 16 fixed dev samples, greater, parent rescored)
#   C = --preset v1.0 (reproduces the 2026-11-10 runs exactly)
# Every run is graded by the shipped evaluator and by a fresh key (0x5EEDF00D).
set -e
E=experiments/2026-11-11; mkdir -p "$E/results"
for c in C B A; do for s in 101 202 303 404 505 606 707 808; do for a in train hyper code code+scorer; do
  echo "$c $a $s"; done; done; done | xargs -P4 -L1 bash -c 'node '"$E"'/exp.js "$PWD" $0 $1 $2 '"$E"'/results'
node $E/analyze.js $E/results > $E/summary.json
