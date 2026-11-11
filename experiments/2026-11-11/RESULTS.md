# GTEE 1.1: second experiment (Kehai, 2026-11-11)

## Question

Last night's 1.0 run (`experiments/2026-11-10`) found two effects:
- blind code mutation hurt;
- scorer mutation wireheaded.

The 1.1 audit (`AUDIT-1.1.md`) then found a bug in the 1.0 gate. **Which effects survive the bug fix, and does the new 1.1 gate stop the wireheading?**

## Setup

- Configs × arms (`train`, `hyper`, `code`, `code+scorer`) × 8 seeds (101, 202, …, 808) × 150 generations.
- Reproduce with `experiments/2026-11-11/run_all.sh` from the repo root.

| Config | Gate |
|---|---|
| **A** | GTEE 1.1 defaults. Compared under the gen-0 scorer, 64 dev samples freshly drawn each generation and shared by candidate and parent, paired-SE rule with k = 1. |
| **B** | The 1.0 gate with *only* the bug removed. Each genome's own scorer, 16 fixed dev samples, `greater`, but the parent is rescored instead of reusing the previous record's score. |
| **C** | `--preset v1.0`, so the 1.0 behaviour, bug included. **It reproduces all 32 of last night's runs exactly**: every per-arm mean matches to three decimals, and code+scorer seed 101 matches on the self-score (47.6586), the update rule and the evaluator accuracy (0.2037). |

- Figures are token accuracy at gen 150, mean ± sd over 8 seeds.
- "Fresh" columns are graded with evaluator key `0x5EEDF00D`, a key no file has ever held.
- **Noise floor.** One evaluator split is 54 tokens, so a single run's accuracy carries about ±0.067 of grading noise, and an 8-seed mean about ±0.024. Read differences under about 0.05 as ties.

| Config | Arm | Eval std | Eval long | Fresh std | Fresh long | Accepted / 150 | Seeds at 0% |
|---|---|---|---|---|---|---|---|
| A · 1.1 default | train | 0.377 ± 0.028 | 0.345 | 0.370 | 0.318 | 21.2 | 0 |
| A · 1.1 default | hyper | 0.407 ± 0.066 | 0.363 | 0.423 | 0.367 | 19.2 | 0 |
| A · 1.1 default | code | 0.398 ± 0.026 | 0.344 | 0.380 | 0.325 | 22.9 | 0 |
| A · 1.1 default | code+scorer | 0.387 ± 0.055 | 0.375 | 0.404 | 0.349 | 30.2 | 0 |
| B · 1.0 gate, bug fixed | train | 0.421 ± 0.058 | 0.361 | 0.346 | 0.345 | 10.9 | 0 |
| B · 1.0 gate, bug fixed | hyper | 0.387 ± 0.032 | 0.355 | 0.361 | 0.339 | 10.4 | 0 |
| B · 1.0 gate, bug fixed | code | 0.382 ± 0.042 | 0.358 | 0.399 | 0.314 | 24.4 | 0 |
| B · 1.0 gate, bug fixed | code+scorer | 0.150 ± 0.098 | 0.141 | 0.135 | 0.137 | 26.9 | 2 |
| C · v1.0 (= last night) | train | 0.435 ± 0.053 | 0.366 | 0.394 | 0.364 | 75.0 | 0 |
| C · v1.0 (= last night) | hyper | 0.458 ± 0.061 | 0.385 | 0.416 | 0.386 | 73.9 | 0 |
| C · v1.0 (= last night) | code | 0.338 ± 0.087 | 0.297 | 0.305 | 0.304 | 69.6 | 0 |
| C · v1.0 (= last night) | code+scorer | 0.083 ± 0.099 | 0.079 | 0.070 | 0.086 | 45.9 | 4 |

Welch t against the same config's `train` arm, on eval std, 8 vs 8 seeds:

| Config | hyper | code | code+scorer |
|---|---|---|---|
| A | +1.18 | +1.54 | +0.43 |
| B | −1.48 | −1.55 | **−6.71** |
| C | +0.81 | **−2.69** | **−8.82** |

## What survives

1. **The 1.0 gate's bug inflated the acceptance count.**
   - In C, about 30 of each run's ~75 acceptances (40%) took a candidate that scored *below the real active parent*. It only had to beat the previous record, which was often a rejected candidate.
   - With the bug fixed (B), acceptances fall to about 11 per run in the train and hyper arms. **Measured.**
2. **"Blind code mutation hurts" does not survive.**
   - It was significant last night: C, t = −2.69.
   - It is a tie once the bug is fixed (B, t = −1.55) and under the 1.1 gate (A, t = +1.54, code slightly *ahead*).
   - Most of last night's code-arm deficit came from the gate accepting worse candidates, not from the code mutations themselves. **Measured.**
3. **Scorer wireheading survives the bug fix, so the cause is self-scoring.**
   - B code+scorer: 0.150, t = −6.71; 2 seeds at 0%; mean self-score minus gen-0 score +162.
   - Comparing each genome under its own scorer is enough to wirehead, even without the bug. **Measured.**
4. **The 1.1 gate eliminates wireheading.**
   - A code+scorer: 0.387 vs C's 0.083, t = +7.56. No seeds at 0%.
   - The evolved scorers still drift (mean self-score minus gen-0 score −0.47), but the gate no longer listens to them. **Measured.**
5. **The honest gate is a little slower.**
   - A train 0.377 vs C train 0.435 (t = −2.72).
   - paired_se with k = 1 on 64 samples accepts only ~21 of ~150 steps, so it throws away some useful training.
   - `--gate-w always` is the obvious next lever: training steps would always land, while mutations still face the gate. **Hypothesis.**

## Bottom line

Under the 1.1 gate all four arms tie, at about 0.38–0.41. At this model size, power and budget:
- letting the lineage rewrite its own update rule or its own scorer neither helps nor hurts;
- the one effect that was real, wireheading, is now fenced off by the instrument.

That's a null result, and that's the point: **the instrument is now honest enough that a positive self-improvement result, if one shows up, would mean something.**
