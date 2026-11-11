# GTEE 1.1: audit (Kehai, 2026-11-11)

## Verdict

**Real, clean instrument.** All 19 tests pass, and the claims I could check independently hold. I found one bug, a logging inconsistency that made the evaluator reject clean v1.0-preset runs, and fixed it.

The audit also turned up a bug in **1.0's gate**. I missed it last night, and it changes one of that experiment's conclusions; see `experiments/2026-11-11/RESULTS.md`.

## What I checked

### v1.0 reproduction is real

Test 13 compares against pinned "1.0 hashes", so I checked those pins against the actual 1.0 code (commit `77e8183`):
- `code+scorer`, seeds 42 and 101, 6 generations.
- All 14 `weightsHash` values and all 14 `recordHash` values match the pins exactly. The pins are genuine, not written in.

Then I checked the long runs:
- `--preset v1.0` reproduces all 32 runs of the 2026-11-10 experiment (4 arms × 8 seeds × 150 generations) to three decimals on every per-arm mean.
- Code+scorer seed 101 matches on its self-score (47.6586), its evolved update rule, and its evaluator accuracy (0.2037).

### The gate

- `evaluateGate` implements `greater`, `margin` and `paired_se` as specified.
- `computePairedStats` uses the n−1 sample variance and SE = √(var/n).
- Outside the v1.0 preset, the parent is **rescored on the candidate's own batch** with the parent's weights and ternary settings. This is the core fix the 1.1 spec asked for.
- Safety checks run before `gateW: "always"`, so explosive W steps are still rejected.
- **Caveat:** under `gateScorer: genome` with `paired_se`, a nonlinear evolved scorer's per-sample mean is not its batch score. That combination is not a default, so it's only a caution for anyone who picks it.

### Isolation and evaluator

- Arm isolation holds under both presets (tests 9 and 17).
- `--evaluator-key` gives the same grades as last night's hand-patched fresh-key evaluator.
- `gate_config` tampering is rejected (test 18).

### Behaviour

In the full 96-run experiment, every run verified under both evaluator keys, and the 1.1 defaults stop the scorer wireheading.

## Bug found and fixed: v1.0-preset runs failed verification

**Symptom.**
- `node run.js --arm code+scorer --gens 150 --seed 101 --preset v1.0`, then grading with the evaluator, returned `tamperReason: "gate_config"`, "Record 6 accepted=true contradicts gateConfig".
- Nothing was tampered with.
- Test 13 only runs 6 generations, and the tamper tests use 3, so neither got far enough to hit it.

**Cause.**
- To reproduce 1.0's hashes, the trainer (correctly) passes the gate `parentDevScore = parentRec.devScore` under the v1.0 preset. That is the *previous record's* score.
- But `gateStats` logged the rescored active parent's score.
- The evaluator re-runs the gate from `gateStats`, so it saw a different comparison than the one the trainer actually made.

**Fix** (`run.js` and the same lines in `trainer.html`): `gateStats.parentDevScore` now logs the value the gate actually used. Under the v1.0 preset that's the previous record's score; otherwise it's the rescored parent's.
- All 19 tests still pass.
- The fix changes no weights.
- v1.0 runs now verify.

## Bug found in 1.0 itself: "beat the last loser"

1.0's gate compared each candidate against `lineage[parentSeqIndex].devScore`, which is the *previous record*. When that record was a rejection, its `devScore` is the rejected candidate's score, not the active parent's. So after a bad candidate was rejected, the next candidate only had to beat that loser.

- In the 2026-11-10 runs, about **40% of acceptances** (≈30 of ≈75 per run in the train, hyper and code arms) took a candidate that scored below the real active parent.
- The 1.1 defaults don't have this bug: they rescore the parent.
- The v1.0 preset keeps it on purpose, as the spec required. It is now labelled in the code.

**Effect on last night's findings:**
- "Blind code mutation hurts" was mostly this bug, and it disappears when the bug is fixed.
- The wireheading finding stands: it comes from self-scoring, not from the bug.
- An erratum is in `experiments/2026-11-10/RESULTS.md`.

This one is mine: I specified a parent comparison and didn't check which record "parent" resolved to.

## Docs

- `README.md` and `DEFAULTS.md` match the code for every `gateConfig` default and preset value I checked.
- `batch_smoke/summary.csv` is Argon's 3-generation smoke run. Nothing passes the 1.1 gate in 3 generations, so all four arms show identical numbers; that is expected, not a bug.
