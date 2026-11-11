# GTEE Defaults & Primitive Reference (`DEFAULTS.md`)

Every default constant, dimension, bound, probability, seed mask, and DSL primitive in the **Generational Trainer & External Evaluator (GTEE)** instrument is documented below with a one-line engineering reason.

## 1. Model Architecture Defaults (`DEFAULT_MODEL_CONFIG`)

| Parameter | Default | One-Line Reason |
|---|---|---|
| `vocabSize` (`V`) | `24` | Accommodates all control tokens (`PAD`, `BOS`, `SEP`, `QRY`, `EOS`), base-4 digits (`0..3`), brackets, keys (`K0..K5`), and grammar rules (`R0..R3`) without wasted embedding rows. |
| `dim` (`C`) | `12` |Smallest hidden dimension divisible by `headSize = 6` (`H = 2` heads) that completes a multi-step generation in `< 50 ms` headless in Node. |
| `headSize` (`N`) | `6` | Divides `dim = 12` into `H = 2` matrix-memory heads (`6x6` state per head) so multi-head RWKV-7 recurrence is exercised cleanly. |
| `numLayers` (`L`) | `2` | Minimum layer count required to exercise both layer-0 base value path and layer-1+ value residual (`vFirst`) path in RWKV-7 (`pips-room.html:1493-1495`). |
| `dimFFN` (`FF`) | `24` | Standard `2 * dim` channel-mix expansion width matching `pips-room.html:4276`. |
| `dDecay` (`DW`) | `6` | Low-rank decay projection bottleneck width (`w1`, `w2`) matching half the model dimension. |
| `dAAA` (`DA`) | `6` | Low-rank in-context learning rate projection bottleneck width (`a1`, `a2`) matching half the model dimension. |
| `dMV` (`DV`) | `6` | Low-rank value residual gate bottleneck width (`v1`, `v2`) for layers `l > 0`. |
| `dGate` (`DG`) | `6` | Low-rank output gate projection bottleneck width (`g1`, `g2`) matching half the model dimension. |
| `W_SCALE` | `0.606531` | Official RWKV-7 decay constant `exp(-0.5)` lifted directly from `pips-room.html:1343`. |
| `ln_eps` | `1e-5` | Standard LayerNorm variance stabilization epsilon lifted from `pips-room.html:1463`. |
| `group_ln_eps` | `64e-5` | Official RWKV-7 per-head GroupNorm stabilization epsilon lifted from `pips-room.html:1509`. |
| `kk_norm_eps` | `1e-6` | Minimum L2 norm floor when normalizing per-head key removal vector `kk`. |

## 2. Generation-0 Genome Defaults (`createGen0Genome`)

| Parameter | Default | One-Line Reason |
|---|---|---|
| `hyper.lr` | `0.04` | Stable baseline step size for clipped SGD on a 2-layer `dim=12` recurrent network over 18 steps. |
| `hyper.momentum` | `0.90` | Standard exponential moving average factor for first-moment gradient accumulation buffer `m`. |
| `hyper.steps` | `18` | Provides enough gradient updates per generation to move loss while keeping headless runtime well under 0.1 seconds per generation. |
| `hyper.gradClip` | `1.0` | Symmetric elementwise gradient clipping bound `[-1.0, 1.0]` preventing recurrent BPTT spikes. |
| `hyper.ternaryMix` | `0.20` | Blends 20% TMAC ternary forward activation with 80% dense activation via Straight-Through Estimator at Gen 0. |
| `hyper.ternaryThreshold` | `0.50` | Matches `Math.round` cutoff (`|w / gamma| >= 0.5`) lifted from `TMAC.quantizeWeights` in `pips-room.html:3943-3946`. |
| `hyper.curriculum` | `[0.25, 0.25, 0.25, 0.25]` | Uniform initial task sampling weights across all 4 procedural task families. |
| `updateRule` (Gen 0) | `mul(lr, g)` | Plain clipped SGD weight step (`w_new = w - lr * g`) with zero extra constants or heuristics. |
| `scorer` (Gen 0) | `sub(acc, mul(0.25, loss))` | Fixed linear combination (`acc - 0.25 * loss`) rewarding higher accuracy and lower cross-entropy loss. |
| `beta2` (variance EMA) | `0.99` | Standard second-moment squared gradient exponential moving average factor for variance buffer `v`. |

## 3. Hyperparameter Mutation Bounds & Symmetric Noise Scales (`HYPER_BOUNDS`)

| Hyperparameter | Min | Max | Symmetric Noise Half-Width (`±scale`) | One-Line Reason |
|---|---|---|---|---|
| `lr` | `0.001` | `0.15` | `±0.015` | Keeps learning rate positive and below recurrent instability threshold under zero-centered uniform perturbations. |
| `momentum` | `0.00` | `0.98` | `±0.08` | Restricts momentum buffer coefficient to valid stationary EMA range `[0, 0.98]`. |
| `steps` | `6` | `40` | `±4` | Bounds per-generation step count so every generation finishes well under the 5-second headless budget. |
| `gradClip` | `0.10` | `5.00` | `±0.35` | Prevents vanishing clip ceiling (`< 0.1`) or unclipped recurrent explosion (`> 5.0`). |
| `ternaryMix` | `0.00` | `1.00` | `±0.15` | Constrains dense-to-ternary interpolation factor to the unit interval `[0, 1]`. |
| `ternaryThreshold` | `0.15` | `0.85` | `±0.10` | Avoids degenerate all-nonzero (`< 0.15`) or all-zero (`> 0.85`) ternary quantization deadzones. |
| `curriculumWeight` | `0.05` | `0.85` | `±0.10` | Ensures all 4 task families retain positive sampling probability prior to simplex normalization. |

## 4. Operator Selection Weights & Arms (`DEFAULT_OPERATOR_WEIGHTS`)

| Operator | Default Base Weight | One-Line Reason |
|---|---|---|
| `W` (Train only) | `0.30` | Allocates 30% base weight to pure weight continuation steps when all operators are active. |
| `H` (Hyperparameters) | `0.30` | Allocates 30% base weight to symmetric hyperparameter perturbation. |
| `C` (Code / Update Rule) | `0.20` | Allocates 20% base weight to AST mutation of the weight update expression. |
| `S` (Scorer) | `0.20` | Allocates 20% base weight to AST mutation of the dev scorer expression. |

When an arm (`train`, `hyper`, `code`, `code+scorer`) restricts the active operator subset, active base weights are normalized by their sum so that setting any disabled operator's weight to `0` produces identical normalized probabilities across matching arms.

## 5. AST DSL Primitive Set (`PRIMITIVE_SPECS`)

The random expression generator samples uniformly over all 16 primitives below:

| Primitive | Arity | Definition | One-Line Reason |
|---|---|---|---|
| `neg(a)` | 1 | `-a` | Provides sign inversion for gradients, weights, or loss terms. |
| `abs(a)` | 1 | `\|a\|` | Provides magnitude extraction for L1-like updates and norms. |
| `tanh(a)` | 1 | `detTanh(a)` | Smoothly bounds arbitrary sub-expressions to `[-1, 1]`. |
| `exp(a)` | 1 | `detExp(clamp(a, -6, 6))` | Clamped exponential primitive preventing float32 overflow. |
| `log(a)` | 1 | `detLog(max(\|a\|, 1e-6))` | Protected natural logarithm defined over all real inputs. |
| `sqrt(a)` | 1 | `detSqrt(max(\|a\|, 0))` | Protected square root enabling RMSprop/Adam-like denominator terms (`sqrt(v)`). |
| `sin(a)` | 1 | `detSin(a)` | Periodic non-monotonic primitive bounded in `[-1, 1]`. |
| `clip(a)` | 1 | `clamp(a, -1, 1)` | Hard unit-interval clamping primitive. |
| `add(a, b)` | 2 | `a + b` | Fundamental linear combination addition operator. |
| `sub(a, b)` | 2 | `a - b` | Fundamental difference operator. |
| `mul(a, b)` | 2 | `a * b` | Fundamental scaling and product operator. |
| `div(a, b)` | 2 | `a / safe(b, 1e-4)` | Protected division preventing division-by-zero NaNs. |
| `min(a, b)` | 2 | `min(a, b)` | Pairwise lower-bound selector. |
| `max(a, b)` | 2 | `max(a, b)` | Pairwise upper-bound selector. |
| `pow(a, b)` | 2 | `detExp(clamp(b, -3, 3) * detLog(clamp(\|a\|, 1e-4, 10)))` | Protected power primitive guaranteed finite for all float32 inputs. |
| `ite(a, b, c)` | 3 | `a > 0 ? b : c` | Conditional branch primitive (`if-then-else` on sign of `a`). |

### Terminal Sets & AST Budgets
- **Update rule (`C`) terminals (`UPDATE_RULE_VARS`):** `g` (clipped gradient), `w` (weight), `m` (momentum EMA), `v` (variance EMA), `lr` (learning rate), and numeric constants `const`.
- **Scorer (`S`) terminals (`SCORER_VARS`):** `acc` (dev accuracy), `loss` (dev cross-entropy loss), and numeric constants `const`.
- **AST node budget (`AST_MAX_NODES`):** `256` maximum nodes (`64` soft mutation cap, `32` max depth) to guarantee bounded evaluation time and prevent stack overflow.
- **AST mutation sub-operators (`AST_MUTATION_OPS`):** `const_perturb`, `subtree_replace`, `wrap_primitive`, `arg_swap`, `ancestor_splice` (each chosen uniformly with probability `1/5 = 0.20`).

## 6. Tasks, Seeds, and Gate Thresholds

| Constant | Default | One-Line Reason |
|---|---|---|
| `TRAINER_DOMAIN_MASK` | `0x7FFFFFFF` | Clears bit 31 on all trainer seeds so trainer seeds strictly inhabit `[0, 2^31 - 1]`. |
| `EVALUATOR_DOMAIN_BIT` | `0x80000000` | Sets bit 31 on all evaluator seeds so evaluator seeds strictly inhabit `[2^31, 2^32 - 1]`, guaranteeing zero domain overlap. |
| `EVALUATOR_SECRET_SEED_KEY` | `0xE7A15EED` | Fixed seed derivation key present exclusively in `evaluator.html` and `evaluator-cli.js`. |
| `devSeedsPerFamily` (Trainer) | `4` (16 total) | Evaluates 4 seeds across each of the 4 task families per generation for fast, balanced dev scoring. |
| `evalSeedsPerFamily` (Evaluator) | `6` (24 std + 24 OOD) | Evaluates 6 seeds per family in both standard and long-length settings for external grading. |
| `carry_add` lengths | `2` digits (std), `4` digits (long) | Tests causal carry propagation across double the training digit length in OOD evaluation. |
| `brackets` lengths | `3` steps (std), `6` steps (long) | Tests Dyck-2 LIFO stack completion at double the standard nesting depth in OOD evaluation. |
| `kv_recall` lengths | `3` pairs / `1` query (std), `5` pairs / `2` queries (long) | Tests associative memory capacity over longer context and multiple queries in OOD evaluation. |
| `grammar` lengths | `3` symbols (std), `6` symbols (long) | Tests rule-conditioned sequence transformation at double the standard string length in OOD evaluation. |
| `gate_min_step_sum` | `1e-9` | Rejects degenerate zero-step update rules (`totalStepAbsSum <= 1e-9` when gradients are non-zero). |
| `gate_max_step_abs` | `50.0` | Rejects explosive single-step weight jumps (`maxStepAbs > 50.0`). |
| `gate_max_weight_abs` | `100.0` | Rejects diverging parameter magnitudes (`maxWeightAbs > 100.0`). |
