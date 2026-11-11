#!/usr/bin/env node
'use strict';

// ============================================================================
// GTEE Instrument Verification Suite (tests/run.js)
// Implements all 12 mandatory tests specified in §9.
// ============================================================================

const fs = require('fs');
const path = require('path');
const assert = require('assert');

const ROOT_DIR = path.resolve(__dirname, '..');
const trainerMod = require(path.join(ROOT_DIR, 'run.js'));
const evalMod = require(path.join(ROOT_DIR, 'evaluator-cli.js'));

const trainerHTMLPath = path.join(ROOT_DIR, 'trainer.html');
const evaluatorHTMLPath = path.join(ROOT_DIR, 'evaluator.html');

let passedCount = 0;
let failedCount = 0;

function runTest(number, name, fn) {
  const t0 = Date.now();
  try {
    const info = fn();
    const ms = Date.now() - t0;
    passedCount++;
    console.log(`[PASS] Test ${String(number).padStart(2, ' ')}: ${name} (${ms} ms)${info ? ' — ' + info : ''}`);
  } catch (err) {
    failedCount++;
    console.error(`[FAIL] Test ${String(number).padStart(2, ' ')}: ${name}`);
    console.error('       Reason:', err && err.stack ? err.stack : err);
  }
}

console.log('======================================================================');
console.log('GTEE Instrument Verification Suite (12 Mandatory Tests)');
console.log('======================================================================');

// ----------------------------------------------------------------------------
// Test 1: CSP and static scan — no network or storage APIs in either file
// ----------------------------------------------------------------------------
runTest(1, 'CSP and static scan (no network or persistent storage APIs)', () => {
  const trainerHTML = fs.readFileSync(trainerHTMLPath, 'utf8');
  const evaluatorHTML = fs.readFileSync(evaluatorHTMLPath, 'utf8');

  const requiredCSP = `<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; connect-src 'none'">`;
  assert(trainerHTML.includes(requiredCSP), 'trainer.html missing strict CSP meta tag');
  assert(evaluatorHTML.includes(requiredCSP), 'evaluator.html missing strict CSP meta tag');

  const forbiddenPatterns = [
    { name: 'fetch()', re: /\bfetch\s*\(/ },
    { name: 'XMLHttpRequest', re: /\bXMLHttpRequest\b/ },
    { name: 'WebSocket', re: /\bWebSocket\b/ },
    { name: 'EventSource', re: /\bEventSource\b/ },
    { name: 'sendBeacon', re: /\bsendBeacon\b/ },
    { name: 'localStorage', re: /\blocalStorage\b/ },
    { name: 'sessionStorage', re: /\bsessionStorage\b/ },
    { name: 'indexedDB', re: /\bindexedDB\b/ },
    { name: 'caches', re: /\bcaches\b/ },
    { name: 'document.cookie', re: /\bcookie\b/i },
    { name: 'external http/https URL', re: /https?:\/\//i }
  ];

  for (const [label, html] of [['trainer.html', trainerHTML], ['evaluator.html', evaluatorHTML]]) {
    for (const pat of forbiddenPatterns) {
      assert(!pat.re.test(html), `${label} contains forbidden network/storage pattern: ${pat.name}`);
    }
  }
  return 'Both trainer.html and evaluator.html have strict CSP and 0 network/storage APIs';
});

// ----------------------------------------------------------------------------
// Test 2: Determinism — same seed gives identical weight hashes over 3 gens, run twice
// ----------------------------------------------------------------------------
runTest(2, 'Determinism (identical weight & record hashes over 3 generations, run twice)', () => {
  const runA = new trainerMod.GTEETrainer({ seed: 123456, arm: 'code+scorer' });
  runA.initGen0();
  for (let i = 0; i < 3; i++) runA.stepGeneration();

  const runB = new trainerMod.GTEETrainer({ seed: 123456, arm: 'code+scorer' });
  runB.initGen0();
  for (let i = 0; i < 3; i++) runB.stepGeneration();

  assert.strictEqual(runA.lineage.length, 4);
  assert.strictEqual(runB.lineage.length, 4);

  for (let g = 0; g < 4; g++) {
    assert.strictEqual(runA.lineage[g].weightsHash, runB.lineage[g].weightsHash, `Weight hash mismatch at gen ${g}`);
    assert.strictEqual(runA.lineage[g].recordHash, runB.lineage[g].recordHash, `Record hash mismatch at gen ${g}`);
    assert.strictEqual(runA.lineage[g].weightsB64, runB.lineage[g].weightsB64, `Weight bytes mismatch at gen ${g}`);
  }
  return `Gen 3 weightsHash=${runA.lineage[3].weightsHash.slice(0, 16)}... matched bit-for-bit`;
});

// ----------------------------------------------------------------------------
// Test 3: Quine fixed point — exporting twice gives byte-identical files
// ----------------------------------------------------------------------------
runTest(3, 'Quine fixed point (exporting descendant twice gives byte-identical files)', () => {
  const templateHTML = fs.readFileSync(trainerHTMLPath, 'utf8');
  const t = new trainerMod.GTEETrainer({ seed: 777, arm: 'code' });
  t.initGen0();
  t.stepGeneration();
  t.stepGeneration();

  const bundle1 = t.exportStateBundle();
  const exportedHTML1 = trainerMod.exportDescendantHTML(templateHTML, bundle1);

  // Load state from exportedHTML1 into a fresh trainer instance without training, and export again from exportedHTML1
  const parsedBundle = evalMod.parseDescendantWithoutExecution(exportedHTML1);
  const t2 = new trainerMod.GTEETrainer({ seed: 1, arm: 'train' });
  t2.loadStateBundle(parsedBundle);
  const bundle2 = t2.exportStateBundle();
  const exportedHTML2 = trainerMod.exportDescendantHTML(exportedHTML1, bundle2);

  assert.strictEqual(exportedHTML1.length, exportedHTML2.length, 'Exported HTML byte lengths differ');
  assert.strictEqual(
    trainerMod.sha256Bytes(exportedHTML1),
    trainerMod.sha256Bytes(exportedHTML2),
    'Exported HTML files are not byte-identical'
  );
  assert.strictEqual(exportedHTML1, exportedHTML2);
  return `Byte-identical (${exportedHTML1.length} bytes, sha256=${trainerMod.sha256Bytes(exportedHTML1).slice(0, 16)}...)`;
});

// ----------------------------------------------------------------------------
// Test 4: Rewind — for every k in a 6-generation run, rebuilt gen k matches stored hashes
// ----------------------------------------------------------------------------
runTest(4, 'Rewind (every k in 0..6 rebuilt from Gen 0 matches stored weights & genome hashes)', () => {
  const t = new trainerMod.GTEETrainer({ seed: 2026, arm: 'code+scorer' });
  t.initGen0();
  for (let i = 0; i < 6; i++) t.stepGeneration();
  assert.strictEqual(t.lineage.length, 7);

  for (let k = 6; k >= 0; k--) {
    const rw = t.rewindTo(k);
    assert(rw.matches, `Rewind to k=${k} failed hash verification`);
    assert.strictEqual(rw.weightsHash, t.lineage[k].weightsHash, `Rebuilt weightsHash mismatch at k=${k}`);
    assert.strictEqual(rw.genomeHash, t.lineage[k].genomeHash, `Rebuilt genomeHash mismatch at k=${k}`);
  }
  return 'All 7 generations (k=0..6) rebuilt from Gen 0 and matched stored hashes';
});

// ----------------------------------------------------------------------------
// Test 5: Fork — both branches verify
// ----------------------------------------------------------------------------
runTest(5, 'Fork (branching from intermediate generation produces valid unmerged branches)', () => {
  const t = new trainerMod.GTEETrainer({ seed: 909, arm: 'code' });
  t.initGen0();
  t.stepGeneration(); // seq 1, gen 1 on 'main'
  t.stepGeneration(); // seq 2, gen 2 on 'main'
  t.stepGeneration(); // seq 3, gen 3 on 'main'

  // Fork from seq 1 (gen 1)
  const forkInfo = t.forkFrom(1, 'branch-alpha');
  assert.strictEqual(forkInfo.branchId, 'branch-alpha');
  const f1 = t.stepGeneration(); // seq 4, gen 2 on 'branch-alpha'
  const f2 = t.stepGeneration(); // seq 5, gen 3 on 'branch-alpha'

  assert.strictEqual(f1.branchId, 'branch-alpha');
  assert.strictEqual(f1.parentSeqIndex, 1);
  assert.strictEqual(f1.gen, 2);
  assert.strictEqual(f2.branchId, 'branch-alpha');
  assert.strictEqual(f2.parentSeqIndex, 4);
  assert.strictEqual(f2.gen, 3);

  const bundle = t.exportStateBundle();
  const report = evalMod.verifyAndEvaluateDescendant(bundle);
  assert.strictEqual(report.valid, true, `Evaluator rejected forked lineage: ${report.error}`);
  assert.strictEqual(report.generations.length, 6);

  const mainRecs = report.generations.filter(r => r.branchId === 'main');
  const forkRecs = report.generations.filter(r => r.branchId === 'branch-alpha');
  assert.strictEqual(mainRecs.length, 4);
  assert.strictEqual(forkRecs.length, 2);
  return `Main branch (${mainRecs.length} gens) and forked branch (${forkRecs.length} gens) both verified`;
});

// ----------------------------------------------------------------------------
// Test 6: Tamper — each tamper type in §7 is rejected with the right reason
// ----------------------------------------------------------------------------
runTest(6, 'Tamper detection (weights, record, genome, reordering, spliced_branch, forged_gen0)', () => {
  const t = new trainerMod.GTEETrainer({ seed: 31415, arm: 'code' });
  t.initGen0();
  t.stepGeneration();
  t.stepGeneration();
  t.stepGeneration();
  const cleanBundle = t.exportStateBundle();

  // Verify clean bundle passes first
  const cleanRes = evalMod.verifyAndEvaluateDescendant(cleanBundle);
  assert.strictEqual(cleanRes.valid, true, 'Clean bundle should pass verification');

  // 6a. Tamper: weights (flip a float in generation 2 weightsB64)
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    const rawBytes = trainerMod.base64ToBytes(b.lineage[2].weightsB64);
    const fArr = new Float32Array(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength / 4);
    fArr[0] = fArr[0] + 0.5;
    b.lineage[2].weightsB64 = trainerMod.bytesToBase64(new Uint8Array(fArr.buffer));
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false, 'Tampered weights should be rejected');
    assert.strictEqual(res.tamperReason, 'weights', `Expected tamperReason="weights", got "${res.tamperReason}"`);
  }

  // 6b. Tamper: record (alter devScore in record 2 without updating hash)
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[2].devScore = b.lineage[2].devScore + 0.1234;
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false, 'Tampered record should be rejected');
    assert.strictEqual(res.tamperReason, 'record', `Expected tamperReason="record", got "${res.tamperReason}"`);
  }

  // 6c. Tamper: genome (modify candidateGenome in record 2, even if recordHash is recomputed)
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[2].candidateGenome.hyper.lr = 0.12345;
    b.lineage[2].recordHash = trainerMod.computeRecordHash(b.lineage[2]);
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false, 'Tampered genome should be rejected');
    assert.strictEqual(res.tamperReason, 'genome', `Expected tamperReason="genome", got "${res.tamperReason}"`);
  }

  // 6d. Tamper: reordering (swap record 1 and record 2 in lineage array)
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    const tmp = b.lineage[1];
    b.lineage[1] = b.lineage[2];
    b.lineage[2] = tmp;
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false, 'Reordered lineage should be rejected');
    assert.strictEqual(res.tamperReason, 'reordering', `Expected tamperReason="reordering", got "${res.tamperReason}"`);
  }

  // 6e. Tamper: spliced_branch (record 2 claims parentHash from a foreign branch/record)
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[2].parentHash = 'f'.repeat(64);
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false, 'Spliced branch should be rejected');
    assert.strictEqual(res.tamperReason, 'spliced_branch', `Expected tamperReason="spliced_branch", got "${res.tamperReason}"`);
  }

  // 6f. Tamper: forged_gen0 (modify Gen 0 weightsHash & recompute recordHash)
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[0].weightsHash = 'a'.repeat(64);
    b.lineage[0].candidateWeightsHash = 'a'.repeat(64);
    b.lineage[0].recordHash = trainerMod.computeRecordHash(b.lineage[0]);
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false, 'Forged Gen 0 should be rejected');
    assert.strictEqual(res.tamperReason, 'forged_gen0', `Expected tamperReason="forged_gen0", got "${res.tamperReason}"`);
  }

  return 'All 6 tamper types (weights, record, genome, reordering, spliced_branch, forged_gen0) rejected with exact reason';
});

// ----------------------------------------------------------------------------
// Test 7: Evaluator isolation — no evaluator key/seeds in trainer & disjoint domains
// ----------------------------------------------------------------------------
runTest(7, 'Evaluator isolation (no evaluator key in trainer; disjoint seed domains)', () => {
  const trainerHTML = fs.readFileSync(trainerHTMLPath, 'utf8');
  const runJS = fs.readFileSync(path.join(ROOT_DIR, 'run.js'), 'utf8');

  const evalKeyHex = '0xE7A15EED';
  const evalKeyDec = String(0xE7A15EED);
  for (const [name, src] of [['trainer.html', trainerHTML], ['run.js', runJS]]) {
    assert(!src.includes(evalKeyHex), `${name} must not contain evaluator seed key hex`);
    assert(!src.includes(evalKeyDec), `${name} must not contain evaluator seed key decimal`);
    assert(!src.includes('EVALUATOR_SECRET_SEED_KEY'), `${name} must not reference EVALUATOR_SECRET_SEED_KEY`);
    assert(!src.includes('deriveEvaluatorSeed'), `${name} must not contain deriveEvaluatorSeed`);
  }

  // Verify evaluator seeds all lie in [0x80000000, 0xFFFFFFFF] (MSB = 1)
  const evalSeeds = new Set();
  for (let i = 0; i < 500; i++) {
    const sStd = evalMod.deriveEvaluatorSeed(i, false);
    const sLong = evalMod.deriveEvaluatorSeed(i, true);
    assert((sStd >>> 31) === 1, `Evaluator standard seed ${sStd} missing MSB=1`);
    assert((sLong >>> 31) === 1, `Evaluator long seed ${sLong} missing MSB=1`);
    evalSeeds.add(sStd);
    evalSeeds.add(sLong);
  }

  // Verify 10,000 random trainer seeds all lie in [0, 0x7FFFFFFF] (MSB = 0) and never collide with evaluator domain
  const rng = new trainerMod.DeterministicPRNG(999);
  for (let i = 0; i < 10000; i++) {
    const ts = trainerMod.deriveTrainerSeed(rng.nextUint32());
    assert((ts >>> 31) === 0, `Trainer seed ${ts} crossed into evaluator domain (MSB=1)`);
    assert(!evalSeeds.has(ts), `Trainer seed ${ts} collided with evaluator seed`);
  }
  return '0 evaluator keys in trainer; 10,000 trainer seeds strictly in [0, 2^31-1] vs evaluator in [2^31, 2^32-1]';
});

// ----------------------------------------------------------------------------
// Test 8: Operator coverage — over 1,000 random draws, every C/S operator fires,
//         every primitive appears, and every expression parses, evaluates finitely, and round-trips
// ----------------------------------------------------------------------------
runTest(8, 'Operator & primitive coverage over 1,000 random draws (parse, finite eval, round-trip)', () => {
  const rng = new trainerMod.DeterministicPRNG(20261110);
  const opCountsC = {};
  const opCountsS = {};
  const primCounts = {};
  for (const op of trainerMod.AST_MUTATION_OPS) { opCountsC[op] = 0; opCountsS[op] = 0; }
  for (const p of trainerMod.ALL_PRIMITIVE_NAMES) primCounts[p] = 0;

  function countPrims(node) {
    if (!node || typeof node !== 'object') return;
    if (primCounts[node.op] !== undefined) primCounts[node.op]++;
    if (node.args) for (const a of node.args) countPrims(a);
  }

  let curC = trainerMod.createGen0UpdateRule();
  let curS = trainerMod.createGen0Scorer();
  const historyC = [curC];
  const historyS = [curS];

  for (let i = 0; i < 1000; i++) {
    // Also test randomExpression directly
    const randExpr = trainerMod.randomExpression(rng, trainerMod.UPDATE_RULE_VARS, 3, true);
    countPrims(randExpr);
    const randText = trainerMod.formatAST(randExpr);
    const randParsed = trainerMod.parseAST(randText, trainerMod.UPDATE_RULE_VARS);
    assert.strictEqual(trainerMod.formatAST(randParsed), randText, 'Random expression failed serialization round-trip');

    // Mutate C
    const resC = trainerMod.mutateAST(curC, rng, trainerMod.UPDATE_RULE_VARS, historyC);
    opCountsC[resC.detail]++;
    countPrims(resC.ast);
    const strC = trainerMod.formatAST(resC.ast);
    const parsedC = trainerMod.parseAST(strC, trainerMod.UPDATE_RULE_VARS);
    assert.strictEqual(trainerMod.formatAST(parsedC), strC, 'Mutated C expression failed round-trip');

    // Mutate S
    const resS = trainerMod.mutateAST(curS, rng, trainerMod.SCORER_VARS, historyS);
    opCountsS[resS.detail]++;
    countPrims(resS.ast);
    const strS = trainerMod.formatAST(resS.ast);
    const parsedS = trainerMod.parseAST(strS, trainerMod.SCORER_VARS);
    assert.strictEqual(trainerMod.formatAST(parsedS), strS, 'Mutated S expression failed round-trip');

    // Evaluate both on random finite inputs
    const envC = {
      g: rng.nextSymNoise(2.0),
      w: rng.nextSymNoise(2.0),
      m: rng.nextSymNoise(2.0),
      v: Math.abs(rng.nextSymNoise(2.0)),
      lr: 0.05
    };
    const valC = trainerMod.evalAST(parsedC, envC);
    assert(Number.isFinite(valC), `C expression evaluated to non-finite ${valC}: ${strC}`);

    const envS = {
      acc: rng.nextFloat(),
      loss: Math.abs(rng.nextSymNoise(5.0))
    };
    const valS = trainerMod.evalAST(parsedS, envS);
    assert(Number.isFinite(valS), `S expression evaluated to non-finite ${valS}: ${strS}`);

    curC = resC.ast;
    curS = resS.ast;
    if (historyC.length < 16) historyC.push(curC);
    if (historyS.length < 16) historyS.push(curS);
  }

  for (const op of trainerMod.AST_MUTATION_OPS) {
    assert(opCountsC[op] > 100, `C sub-operator ${op} fired too rarely: ${opCountsC[op]}`);
    assert(opCountsS[op] > 100, `S sub-operator ${op} fired too rarely: ${opCountsS[op]}`);
  }
  for (const p of trainerMod.ALL_PRIMITIVE_NAMES) {
    assert(primCounts[p] > 20, `Primitive ${p} appeared too rarely: ${primCounts[p]}`);
  }
  return `All 5 AST ops fired >100x each; all 16 primitives appeared; 1,000/1,000 expressions parsed, evaluated finitely & round-tripped`;
});

// ----------------------------------------------------------------------------
// Test 9: Arm isolation — code and code+scorer produce identical lineages when P(S)=0;
//         hyper and code produce identical lineages when P(C)=0
// ----------------------------------------------------------------------------
runTest(9, 'Arm isolation (code vs code+scorer with P(S)=0; hyper vs code with P(C)=0)', () => {
  // 1. code vs code+scorer when S weight is forced to 0
  const weightsNoS = { W: 0.30, H: 0.30, C: 0.20, S: 0.0 };
  const tCode1 = new trainerMod.GTEETrainer({ seed: 555, arm: 'code', operatorWeights: weightsNoS });
  const tCodeScorer1 = new trainerMod.GTEETrainer({ seed: 555, arm: 'code+scorer', operatorWeights: weightsNoS });
  tCode1.initGen0();
  tCodeScorer1.initGen0();
  for (let i = 0; i < 4; i++) {
    tCode1.stepGeneration();
    tCodeScorer1.stepGeneration();
  }
  for (let g = 0; g <= 4; g++) {
    assert.strictEqual(
      tCode1.lineage[g].recordHash,
      tCodeScorer1.lineage[g].recordHash,
      `Lineage mismatch between 'code' and 'code+scorer' (with P(S)=0) at gen ${g}`
    );
  }

  // 2. hyper vs code when C weight is forced to 0
  const weightsNoC = { W: 0.30, H: 0.30, C: 0.0, S: 0.0 };
  const tHyper2 = new trainerMod.GTEETrainer({ seed: 888, arm: 'hyper', operatorWeights: weightsNoC });
  const tCode2 = new trainerMod.GTEETrainer({ seed: 888, arm: 'code', operatorWeights: weightsNoC });
  tHyper2.initGen0();
  tCode2.initGen0();
  for (let i = 0; i < 4; i++) {
    tHyper2.stepGeneration();
    tCode2.stepGeneration();
  }
  for (let g = 0; g <= 4; g++) {
    assert.strictEqual(
      tHyper2.lineage[g].recordHash,
      tCode2.lineage[g].recordHash,
      `Lineage mismatch between 'hyper' and 'code' (with P(C)=0) at gen ${g}`
    );
  }

  return 'Identical record & weight hashes across 4 generations for both arm pairs';
});

// ----------------------------------------------------------------------------
// Test 10: Gate — injected update rules that produce NaN, zero, or explosive steps
//          are rejected and parent's weights are preserved
// ----------------------------------------------------------------------------
runTest(10, 'Gate rejection of NaN, zero-step, and explosive update rules (preserving parent weights)', () => {
  const t = new trainerMod.GTEETrainer({ seed: 404, arm: 'code' });
  const gen0 = t.initGen0();
  const parentWeightsHash = gen0.weightsHash;
  const parentWeightsB64 = gen0.weightsB64;

  // 10a. Zero-step update rule: const(0)
  {
    const zeroGenome = trainerMod.createGen0Genome();
    zeroGenome.updateRule = { op: 'const', val: 0.0 };
    const rec = t.stepGeneration({ forcedOperator: 'C', injectedGenome: zeroGenome });
    assert.strictEqual(rec.accepted, false, 'Zero-step update rule must be rejected');
    assert.strictEqual(rec.gateReason, 'reject_zero_step');
    assert.strictEqual(rec.weightsHash, parentWeightsHash, 'Parent weightsHash must be preserved after zero-step rejection');
    assert.strictEqual(t.model.weightsToBase64(), parentWeightsB64, 'Parent weights must be preserved bit-for-bit');
  }

  // 10b. Explosive update rule: const(1000)
  {
    const explosiveGenome = trainerMod.createGen0Genome();
    explosiveGenome.updateRule = { op: 'const', val: 1000.0 };
    const rec = t.stepGeneration({ forcedOperator: 'C', injectedGenome: explosiveGenome });
    assert.strictEqual(rec.accepted, false, 'Explosive update rule must be rejected');
    assert.strictEqual(rec.gateReason, 'reject_explosive_step');
    assert.strictEqual(rec.weightsHash, parentWeightsHash, 'Parent weightsHash must be preserved after explosive rejection');
    assert.strictEqual(t.model.weightsToBase64(), parentWeightsB64, 'Parent weights must be preserved bit-for-bit');
  }

  // 10c. NaN update rule: const(NaN)
  {
    const nanGenome = trainerMod.createGen0Genome();
    nanGenome.updateRule = { op: 'const', val: NaN };
    const rec = t.stepGeneration({ forcedOperator: 'C', injectedGenome: nanGenome });
    assert.strictEqual(rec.accepted, false, 'NaN update rule must be rejected');
    assert.strictEqual(rec.gateReason, 'reject_nan_or_inf');
    assert.strictEqual(rec.weightsHash, parentWeightsHash, 'Parent weightsHash must be preserved after NaN rejection');
    assert.strictEqual(t.model.weightsToBase64(), parentWeightsB64, 'Parent weights must be preserved bit-for-bit');
  }

  return 'Zero-step, explosive, and NaN update rules rejected; parent weights preserved bit-for-bit';
});

// ----------------------------------------------------------------------------
// Test 11: Evaluator parity — evaluator's model & DSL produce identical outputs
//          to trainer's on the same weights and inputs (+ BPTT gradient check & ternary invariant)
// ----------------------------------------------------------------------------
runTest(11, 'Evaluator parity + BPTT finite-difference gradient check + TMAC ternary invariant', () => {
  // 11a. Trainer vs Evaluator Model & DSL Parity
  const modelT = new trainerMod.RWKV7(trainerMod.DEFAULT_MODEL_CONFIG, { seed: 1337 });
  const modelE = new evalMod.RWKV7(evalMod.DEFAULT_MODEL_CONFIG, { seed: 9999 });
  modelE.loadWeightsFromBase64(modelT.weightsToBase64());
  assert.strictEqual(modelT.weightsHash(), modelE.weightsHash(), 'Model weightsHash parity failed');

  const samplesT = [];
  const samplesE = [];
  for (let fam = 0; fam < 4; fam++) {
    samplesT.push(trainerMod.generateTaskSample(fam, 500 + fam, false));
    samplesE.push(evalMod.generateTaskSample(fam, 500 + fam, false));
  }

  const astStr = 'add(tanh(mul(acc, 1.5)), neg(mul(0.25, sqrt(loss))))';
  const astT = trainerMod.parseAST(astStr, trainerMod.SCORER_VARS);
  const astE = evalMod.parseAST(astStr, evalMod.SCORER_VARS);

  const resT = trainerMod.evaluateModelOnSamples(modelT, samplesT, astT);
  const resE = evalMod.evaluateModelOnSamples(modelE, samplesE, astE);

  assert.strictEqual(resT.acc, resE.acc, 'Trainer vs Evaluator accuracy mismatch');
  assert.strictEqual(resT.loss, resE.loss, 'Trainer vs Evaluator loss mismatch');
  assert.strictEqual(resT.score, resE.score, 'Trainer vs Evaluator DSL score mismatch');
  assert.strictEqual(resT.gen0Score, resE.gen0Score, 'Trainer vs Evaluator gen0Score mismatch');

  // 11b. TMAC Ternary Quantization Invariant (all unpacked weights in {-1, 0, +1}, gamma > 0)
  const wArr = modelT.p['head.weight'];
  const packed = new Uint8Array(Math.ceil(wArr.length / 4));
  const gamma = trainerMod.TMAC.quantizeWeights(wArr, packed, 0.5);
  assert(gamma > 0 && Number.isFinite(gamma), `Invalid TMAC gamma: ${gamma}`);
  const unpacked = trainerMod.TMAC.unpackToFloat32(packed, wArr.length);
  for (let i = 0; i < unpacked.length; i++) {
    const v = unpacked[i];
    assert(v === -1 || v === 0 || v === 1, `Non-ternary weight value: ${v}`);
  }

  // 11c. Finite-Difference vs Analytical BPTT Gradient Check (ternaryMix = 0 for smooth exact partials)
  const gradCfg = { vocabSize: 12, dim: 6, headSize: 3, numLayers: 1, dimFFN: 12, dDecay: 4, dAAA: 4, dMV: 4, dGate: 4 };
  const gm = new trainerMod.RWKV7(gradCfg, { seed: 42, ternaryMix: 0.0 });
  const seq = [1, 4, 5, 2, 6];
  function computeLoss() {
    let st = gm.getEmptyState();
    let loss = 0;
    for (let t = 0; t < seq.length - 1; t++) {
      const r = gm.forward(seq[t], st);
      st = r.state;
      // Softmax with Math.exp in Float64 for reference finite diff
      let mx = -Infinity;
      for (let v = 0; v < gm.V; v++) if (r.logits[v] > mx) mx = r.logits[v];
      let s = 0;
      for (let v = 0; v < gm.V; v++) s += Number(trainerMod.detExp(r.logits[v] - mx));
      const p = Number(trainerMod.detExp(r.logits[seq[t + 1]] - mx)) / s;
      loss -= Math.log(p);
    }
    return loss / (seq.length - 1);
  }
  // Analytical backward
  gm.zeroGrad();
  const states = [gm.getEmptyState()], caches = [], logitsArr = [];
  for (let t = 0; t < seq.length - 1; t++) {
    const r = gm.forward(seq[t], states[t]);
    states.push(r.state); caches.push(r.cache); logitsArr.push(r.logits);
  }
  let gState = null;
  const invT = 1.0 / (seq.length - 1);
  for (let t = seq.length - 2; t >= 0; t--) {
    const gL = new Float32Array(gm.V);
    let mx = -Infinity;
    for (let v = 0; v < gm.V; v++) if (logitsArr[t][v] > mx) mx = logitsArr[t][v];
    let s = 0;
    const exps = new Float32Array(gm.V);
    for (let v = 0; v < gm.V; v++) { exps[v] = trainerMod.detExp(logitsArr[t][v] - mx); s += exps[v]; }
    for (let v = 0; v < gm.V; v++) {
      const p = exps[v] / s;
      gL[v] = (p - (v === seq[t + 1] ? 1 : 0)) * invT;
    }
    gState = gm.backward(gL, caches[t], gState);
  }
  const analyticGrad = gm.g['blocks.0.att.receptance.weight'].slice(0, 12);
  const numGrad = new Float64Array(12);
  const eps = 1e-3;
  const W = gm.p['blocks.0.att.receptance.weight'];
  for (let i = 0; i < 12; i++) {
    const orig = W[i];
    W[i] = Math.fround(orig + eps);
    const lp = computeLoss();
    W[i] = Math.fround(orig - eps);
    const lm = computeLoss();
    W[i] = orig;
    numGrad[i] = (lp - lm) / (2 * eps);
  }
  let dot = 0, nA = 0, nN = 0;
  for (let i = 0; i < 12; i++) {
    dot += analyticGrad[i] * numGrad[i];
    nA += analyticGrad[i] * analyticGrad[i];
    nN += numGrad[i] * numGrad[i];
  }
  const cosSim = dot / (Math.sqrt(nA) * Math.sqrt(nN) + 1e-12);
  assert(cosSim > 0.98, `BPTT gradient cosine similarity too low: ${cosSim}`);

  return `Bit-identical Trainer/Evaluator parity; ternary weights in {-1,0,+1}; BPTT grad cosine sim=${cosSim.toFixed(4)}`;
});

// ----------------------------------------------------------------------------
// Test 12: Timing — one generation runs in under 5 seconds headless (print measurement)
// ----------------------------------------------------------------------------
runTest(12, 'Timing (1 generation headless in Node completes in < 5,000 ms)', () => {
  const t = new trainerMod.GTEETrainer({ seed: 2026, arm: 'code+scorer' });
  t.initGen0();
  const t0 = Date.now();
  const rec = t.stepGeneration();
  const elapsedMs = Date.now() - t0;
  assert(elapsedMs < 5000, `One generation took ${elapsedMs} ms, exceeding the 5,000 ms limit`);
  return `1 generation completed in ${elapsedMs} ms (budget: < 5,000 ms, operator=${rec.operator})`;
});

console.log('======================================================================');
console.log(`Verification Summary: ${passedCount}/12 passed, ${failedCount} failed`);
console.log('======================================================================');

if (failedCount > 0) {
  process.exit(1);
}