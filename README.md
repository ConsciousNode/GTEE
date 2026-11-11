# Generational Trainer & External Evaluator (GTEE 1.1)

**ConsciousNode SoftWorks** — Research instrument consisting of a self-contained single-file HTML generational trainer (`trainer.html`), an isolated single-file HTML external evaluator (`evaluator.html`), headless Node.js CLI runners (`run.js`, `evaluator-cli.js`, and `run-batch.js`), and a 19-point verification test suite (`tests/run.js`).

## Overview of Components

1. **`trainer.html`** — Offline single-file HTML generational trainer:
   - Embeds the deterministic `Float32` RWKV-7 ("Goose" x070) forward pass and exact backpropagation through time (BPTT) lifted from `pips-room.html:1345-1648`, combined with `TMAC` / `BitLinear` ternary weight quantization lifted from `pips-room.html:3915-4041`.
   - Generates procedural training and development sequences from 31-bit trainer seeds across 4 task families (`carry_add`, `brackets`, `kv_recall`, `grammar`).
   - Supports 4 configurable arms (`train`, `hyper`, `code`, `code+scorer`) over 4 change operators (`W`, `H`, `C`, `S`).
   - Exposes `gateConfig` controls in the UI (`preset`, `gateScorer`, `devSeedsPerFamily`, `devResample`, `gateRule`, `gateK`, `gateMargin`, `gateW`), records `gateConfig` in generation 0's record, and binds it into the SHA-256 hash chain.
   - Logs dev score under both the genome scorer and the gen-0 scorer, plus their difference (`genomeMinusGen0DevScore`), at every generation.
   - Supports deterministic **Rewind** and **Fork** under every gate configuration, and exports self-contained descendant HTML files via **Save descendant**.

2. **`evaluator.html`** — Offline single-file HTML external evaluator:
   - Holds its own isolated seed key and seed derivation in the disjoint upper 32-bit seed domain (`[0x80000000, 0xFFFFFFFF]`).
   - Parses exported descendant HTML files without executing any scripts from the descendant file.
   - Reads `gateConfig` from the lineage, verifies the hash chain under it, and rejects tampering with the exact reason (`weights`, `record`, `genome`, `reordering`, `spliced_branch`, `forged_gen0`, `gate_config`).
   - Reports per-generation in-distribution (`standard`) and out-of-distribution (`long`) scores under the gen-0 scorer, alongside the trainer's `genomeMinusGen0DevScore` field.

3. **`run.js`** — Headless Node.js generational trainer CLI:
   - Accepts `--arm`, `--gens`, `--seed`, `--preset <default|v1.0>`, and individual gate flags (`--gate-scorer`, `--dev-seeds-per-family`, `--dev-resample`, `--gate-rule`, `--gate-k`, `--gate-margin`, `--gate-w`), writing `lineage.jsonl`, `lineage.json`, and `descendant.html` to `--out <dir>`.

4. **`evaluator-cli.js`** — Headless Node.js evaluator CLI:
   - Grades any `descendant.html` or `lineage.json` file and prints the structured JSON report to stdout. Supports `--evaluator-key <hex>` for secondary held-out grading and `--out <file.json>`.

5. **`run-batch.js`** — Multi-arm, multi-seed batch runner:
   - Runs every `arm × seed` combination, verifies and grades each run with the primary evaluator (and optionally a second held-out evaluator via `--evaluator-key <hex>`), writes `summary.csv`, and deletes large per-run artifacts unless `--keep-artifacts` is set.

6. **`tests/run.js`** — Instrument verification suite (19 tests):
   - Verifies Tests 1–12 from GTEE 1.0 plus Tests 13–19 for GTEE 1.1 (`--preset v1.0` hash reproduction against pinned 1.0 hashes, `gateRule` and `gateScorer` modes, paired dev resampling + rewind, `gateW` modes, arm isolation under 1.1 defaults, `gate_config` tamper rejection, and `< 5s` headless timing with 64 paired dev samples).

7. **`DEFAULTS.md`** — Reference table of every default hyperparameter, `gateConfig` parameter, model dimension, operator probability, seed mask, and DSL primitive with its one-line reason.

## How to Run

### 1. Run the 19-Point Verification Suite
```bash
node tests/run.js
```

### 2. Run the Headless Trainer (`run.js`)
```bash
# Default GTEE 1.1 gate configuration (gen0 scorer, 64 resampled paired dev samples, paired_se rule)
node run.js --arm code+scorer --gens 3 --seed 42 --out ./out_default

# Reproduce GTEE 1.0 behavior via preset
node run.js --arm code+scorer --gens 3 --seed 42 --preset v1.0 --out ./out_v10

# Custom gate configuration flags
node run.js --arm code --gens 3 --seed 42 \
  --gate-scorer gen0 \
  --dev-seeds-per-family 16 \
  --dev-resample true \
  --gate-rule margin \
  --gate-margin 0.01 \
  --gate-w always \
  --out ./out_custom
```

### 3. Run the Batch Runner (`run-batch.js`)
```bash
node run-batch.js \
  --arms train,hyper,code,code+scorer \
  --seeds 101 \
  --gens 3 \
  --preset default \
  --evaluator-key 0xCAFEBABE \
  --out ./batch_smoke
```
Optional batch flags:
- `--preset <default|v1.0>` and any gate flag (`--gate-scorer`, `--dev-seeds-per-family`, `--dev-resample`, `--gate-rule`, `--gate-k`, `--gate-margin`, `--gate-w`)
- `--evaluator-key <hex>`: grades each run a second time using an independent held-out evaluator seed key (never passed to the trainer)
- `--keep-artifacts`: retains per-run `descendant.html` and `lineage.json` files after grading
- `--jobs <J>`: sets batch concurrency parameter

### 4. Grade a Descendant File Headless (`evaluator-cli.js`)
```bash
node evaluator-cli.js ./out_default/descendant.html
node evaluator-cli.js ./out_default/descendant.html --evaluator-key 0xCAFEBABE --out ./eval2.json
```
