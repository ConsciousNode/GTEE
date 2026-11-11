# GTEE: first experiment (Kehai, 2026-11-10)

> **Erratum (2026-11-11).** The 1.0 gate compared each candidate against the *previous record's* score, which was often a rejected candidate's ("beat the last loser"). About 40% of the acceptances below took a candidate scoring worse than the real active parent. With the bug removed, **finding 2, "blind code mutation made learning worse", becomes a tie (t = −1.55)**. Finding 3 (wireheading) stands: it comes from self-scoring. See `AUDIT-1.1.md` and `experiments/2026-11-11/RESULTS.md`.

## Instrument audit

All 12 instrument tests pass. I also checked the code by hand for each item on the spec's forbidden list:

- **No candidate lists.** Every expression comes from `randomExpression` / `mutateAST` acting on RNG draws.
- **No dependence on the generation number** except in logging.
- **Arms differ only in operator probabilities.** `ARM_ENABLED_OPERATORS` feeds `selectOperator`, and nothing else reads the arm.
- **H noise is symmetric and zero-centered** within the documented bounds.
- **The evaluator key appears only in the evaluator files.**

The training, mutation, gate and task code in `trainer.html` is byte-identical to `run.js`, and the evaluator code in `evaluator.html` is byte-identical to `evaluator-cli.js`.

**This is the first Argon delivery this week where I found no scripted outcome.**

## Setup

- Arms: `train`, `hyper`, `code`, `code+scorer`.
- Reproduce with `experiments/2026-11-10/run_all.sh` from the repo root.
- 8 seeds per arm: 101, 202, …, 808.
- 150 generations per run, default settings.
- Every lineage was graded by the shipped evaluator. It was also graded by a copy whose seed key I changed (`0xE7A15EED` → `0x5EEDF00D`), giving **fresh seeds no file had ever used**.
- Figures are token accuracy at the final generation, mean ± sd over the 8 seeds.

| Arm | Evaluator std | Evaluator long | Fresh std | Fresh long | Accepted / proposed |
|---|---|---|---|---|---|
| train | 0.435 ± 0.050 | 0.366 ± 0.048 | 0.394 ± 0.074 | 0.364 ± 0.040 | W 600/1200 |
| hyper | 0.458 ± 0.057 | 0.385 ± 0.041 | 0.416 ± 0.090 | 0.386 ± 0.045 | H 313/619, W 278/581 |
| code | 0.338 ± 0.082 | 0.297 ± 0.066 | 0.305 ± 0.112 | 0.304 ± 0.074 | H 224/452, C 126/299, W 207/449 |
| code+scorer | **0.083 ± 0.093** | 0.079 ± 0.095 | 0.070 ± 0.079 | 0.086 ± 0.106 | H 114/358, C 77/260, W 105/342, S 71/240 |

Mean evaluator std accuracy along the way:

| Arm | gen 0 | gen 25 | gen 50 | gen 100 | gen 150 |
|---|---|---|---|---|---|
| train | 0.250 | 0.377 | 0.393 | 0.474 | 0.435 |
| hyper | 0.250 | 0.373 | 0.407 | 0.449 | 0.458 |
| code | 0.250 | 0.368 | 0.352 | 0.354 | 0.338 |
| code+scorer | 0.250 | 0.338 | 0.224 | 0.111 | 0.083 |

## What happened

1. **Tuning hyperparameters ≈ just training.** The +0.02 is well inside the noise.
2. **Blind code mutation made learning worse.** The gate accepted 126 of 299 rule mutations. Over time the update rules drifted away from following the gradient. One seed ended on pure weight decay (`lr · clip(w) · tiny`); another ended on `abs(exp(sin g) · sin lr)`, which always pushes weights the same way.
3. **Self-editable scorers produced real wireheading, and it was catastrophic.** Within ~50 generations, scorers evolved that reward the wrong thing:
   - `exp(1.14 · loss)`, which rewards *high* loss;
   - `sin(1.12) / acc − …`, which rewards *low* accuracy;
   - `loss² − …`.

   Self-reported dev scores reached 27,182 while the fixed gen-0 scorer said −3.0. Four of the 8 seeds ended at **0% accuracy, below the 25% you'd get by chance**. Those models were actively trained to be wrong. Nothing about this was scripted.

## Why (and what's mine to fix)

The gate is the bottleneck, and its design came from my spec. It accepts any candidate whose dev score beats the parent's, on **16 fixed dev samples**, comparing scores under **each genome's own scorer**. That has three consequences:

- **It rejects good training.** In the train arm, half of the plain training steps were thrown away.
- **It ratchets noise.** Any mutation that gets lucky on 16 samples is kept, so harmful rules creep in.
- **Changing the scorer is unchecked.** A scorer mutation is judged by the new scorer itself, so inflating the scale always wins.

Ideas for GTEE 1.1: a gate that compares under the gen-0 scorer (keeping the genome scorer only as a logged "belief"), more dev samples that are resampled each generation, and a significance margin. Then rerun the same four arms.
