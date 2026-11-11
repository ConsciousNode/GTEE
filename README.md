# Generational Trainer & External Evaluator (GTEE)

**ConsciousNode SoftWorks** — Research instrument consisting of a self-contained single-file HTML generational trainer (`trainer.html`), an isolated single-file HTML external evaluator (`evaluator.html`), headless Node.js CLI counterparts (`run.js` and `evaluator-cli.js`), and a 12-point verification test suite (`tests/run.js`).

## Overview of Components

1. **`trainer.html`** — Offline single-file HTML generational trainer:
   - Embeds the deterministic `Float32` RWKV-7 ("Goose" x070) forward pass and exact backpropagation through time (BPTT) lifted from `pips-room.html:1345-1648`, combined with `TMAC` / `BitLinear` ternary weight quantization lifted from `pips-room.html:3915-4041`.
   - Generates procedural training and development sequences from 31-bit trainer seeds across 4 task families (`carry_add`, `brackets`, `kv_recall`, `grammar`).
   - Supports 4 configurable arms (`train`, `hyper`, `code`, `code+scorer`) over 4 change operators (`W`, `H`, `C`, `S`).
   - Maintains an append-only, SHA-256 hash-chained lineage tree with deterministic **Rewind** (rebuilding any generation $k$ from generation 0 plus the lineage records) and **Fork** (branching without merging).
   - Provides a **Save descendant** button that exports a self-contained descendant HTML file with genome, weights, and lineage baked in (producing byte-identical files when exported twice without training).
   - Enforces a strict Content Security Policy (`default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'`) with zero network or persistent storage APIs.

2. **`evaluator.html`** — Offline single-file HTML external evaluator:
   - Holds its own isolated seed key and seed derivation in the disjoint upper 32-bit seed domain (`[0x80000000, 0xFFFFFFFF]`).
   - Parses exported descendant HTML files as inert text without executing any scripts from the descendant file.
   - Verifies the SHA-256 hash chain and rejects tampered lineages with the exact tamper reason (`weights`, `record`, `genome`, `reordering`, `spliced_branch`, `forged_gen0`).
   - Scores every generation in the lineage on both in-distribution (`standard`) and out-of-distribution (`long`) task suites using the generation-0 scorer only.

3. **`run.js`** — Headless Node.js generational trainer CLI:
   - Runs the exact same trainer core from the command line and writes `lineage.jsonl` (one JSONL line per generation), `lineage.json`, and `descendant.html` into the target output directory.

4. **`evaluator-cli.js`** — Headless Node.js evaluator CLI:
   - Runs the exact same verification and external evaluation core as `evaluator.html` on any exported descendant HTML or lineage file and prints the structured JSON report to stdout.

5. **`tests/run.js`** — Instrument verification suite:
   - Exercises all 12 mandatory instrument verification checks:
     1. CSP and static scan (no network or storage APIs in either HTML file)
     2. Determinism (identical weight hashes over 3 generations across repeated runs)
     3. Quine fixed point (exporting twice produces byte-identical files)
     4. Rewind (every generation $k$ in a 6-generation run rebuilds from Gen 0 to match stored hashes)
     5. Fork (both branches verify cleanly)
     6. Tamper rejection (`weights`, `record`, `genome`, `reordering`, `spliced_branch`, `forged_gen0`)
     7. Evaluator isolation (no evaluator key/seeds in trainer; disjoint seed domains)
     8. Operator coverage (all 5 AST operators and all 16 primitives fire over 1,000 random draws, parse, evaluate finitely, and round-trip)
     9. Arm isolation (`code` and `code+scorer` match when `P(S)=0`; `hyper` and `code` match when `P(C)=0`)
     10. Gate enforcement (`NaN`, zero-step, and explosive update rules are rejected and parent weights are preserved)
     11. Evaluator parity (trainer and evaluator model/DSL cores match bit-for-bit on identical weights and inputs)
     12. Timing (one generation completes in under 5 seconds headless)

6. **`DEFAULTS.md`** — Complete table of default hyperparameters, model dimensions, operator probabilities, seed masks, gate thresholds, and the 16-primitive DSL specification with a one-line reason for each.

## How to Run

### 1. Run the Instrument Verification Suite
```bash
node tests/run.js
```

### 2. Run the Headless Trainer (`run.js`)
```bash
node run.js --arm train --gens 3 --seed 42 --out ./smoke_runs/train
node run.js --arm hyper --gens 3 --seed 42 --out ./smoke_runs/hyper
node run.js --arm code --gens 3 --seed 42 --out ./smoke_runs/code
node run.js --arm code+scorer --gens 3 --seed 42 --out ./smoke_runs/code_scorer
```
Each run writes:
- `<out>/lineage.jsonl` — One JSONL record per generation containing every lineage field.
- `<out>/descendant.html` — The self-contained descendant HTML with genome, weights, and lineage baked in.

### 3. Grade a Descendant File Headless (`evaluator-cli.js`)
```bash
node evaluator-cli.js ./smoke_runs/train/descendant.html
```

### 4. Browser Usage
- Open `trainer.html` directly in any modern browser (Chrome, Firefox, Safari) to step generations interactively, inspect the active update rule and scorer expressions, rewind or fork any node in the lineage table, and click **Save descendant** to download a descendant HTML file.
- Open `evaluator.html` in a browser and drag-and-drop any exported `descendant.html` file into the drop zone to verify its hash chain and inspect per-generation standard and long-length evaluation scores.
