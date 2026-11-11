#!/usr/bin/env node
'use strict';

// ============================================================================
// GTEE 1.1 Instrument Verification Suite (tests/run.js)
// Implements all 19 mandatory tests (Tests 1-12 from 1.0 + Tests 13-19 for 1.1).
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
console.log('GTEE 1.1 Instrument Verification Suite (19 Mandatory Tests)');
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
  t.stepGeneration();
  t.stepGeneration();
  t.stepGeneration();

  const forkInfo = t.forkFrom(1, 'branch-alpha');
  assert.strictEqual(forkInfo.branchId, 'branch-alpha');
  const f1 = t.stepGeneration();
  const f2 = t.stepGeneration();

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
  const t = new trainerMod.GTEETrainer({ seed: 31415, arm: 'code', preset: 'v1.0' });
  t.initGen0();
  t.stepGeneration();
  t.stepGeneration();
  t.stepGeneration();
  const cleanBundle = t.exportStateBundle();

  const cleanRes = evalMod.verifyAndEvaluateDescendant(cleanBundle);
  assert.strictEqual(cleanRes.valid, true, 'Clean bundle should pass verification');

  // 6a. Tamper: weights
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    const rawBytes = trainerMod.base64ToBytes(b.lineage[2].weightsB64);
    const fArr = new Float32Array(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength / 4);
    fArr[0] = fArr[0] + 0.5;
    b.lineage[2].weightsB64 = trainerMod.bytesToBase64(new Uint8Array(fArr.buffer));
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.tamperReason, 'weights', `Expected tamperReason="weights", got "${res.tamperReason}"`);
  }

  // 6b. Tamper: record
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[2].devScore = b.lineage[2].devScore + 0.1234;
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.tamperReason, 'record', `Expected tamperReason="record", got "${res.tamperReason}"`);
  }

  // 6c. Tamper: genome
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[2].candidateGenome.hyper.lr = 0.12345;
    b.lineage[2].recordHash = trainerMod.computeRecordHash(b.lineage[2]);
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.tamperReason, 'genome', `Expected tamperReason="genome", got "${res.tamperReason}"`);
  }

  // 6d. Tamper: reordering
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    const tmp = b.lineage[1];
    b.lineage[1] = b.lineage[2];
    b.lineage[2] = tmp;
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.tamperReason, 'reordering', `Expected tamperReason="reordering", got "${res.tamperReason}"`);
  }

  // 6e. Tamper: spliced_branch
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[2].parentHash = 'f'.repeat(64);
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.tamperReason, 'spliced_branch', `Expected tamperReason="spliced_branch", got "${res.tamperReason}"`);
  }

  // 6f. Tamper: forged_gen0
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[0].weightsHash = 'a'.repeat(64);
    b.lineage[0].candidateWeightsHash = 'a'.repeat(64);
    b.lineage[0].recordHash = trainerMod.computeRecordHash(b.lineage[0]);
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false);
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

  const evalSeeds = new Set();
  for (let i = 0; i < 500; i++) {
    const sStd = evalMod.deriveEvaluatorSeed(i, false);
    const sLong = evalMod.deriveEvaluatorSeed(i, true);
    assert((sStd >>> 31) === 1, `Evaluator standard seed ${sStd} missing MSB=1`);
    assert((sLong >>> 31) === 1, `Evaluator long seed ${sLong} missing MSB=1`);
    evalSeeds.add(sStd);
    evalSeeds.add(sLong);
  }

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
    const randExpr = trainerMod.randomExpression(rng, trainerMod.UPDATE_RULE_VARS, 3, true);
    countPrims(randExpr);
    const randText = trainerMod.formatAST(randExpr);
    const randParsed = trainerMod.parseAST(randText, trainerMod.UPDATE_RULE_VARS);
    assert.strictEqual(trainerMod.formatAST(randParsed), randText, 'Random expression failed serialization round-trip');

    const resC = trainerMod.mutateAST(curC, rng, trainerMod.UPDATE_RULE_VARS, historyC);
    opCountsC[resC.detail]++;
    countPrims(resC.ast);
    const strC = trainerMod.formatAST(resC.ast);
    const parsedC = trainerMod.parseAST(strC, trainerMod.UPDATE_RULE_VARS);
    assert.strictEqual(trainerMod.formatAST(parsedC), strC, 'Mutated C expression failed round-trip');

    const resS = trainerMod.mutateAST(curS, rng, trainerMod.SCORER_VARS, historyS);
    opCountsS[resS.detail]++;
    countPrims(resS.ast);
    const strS = trainerMod.formatAST(resS.ast);
    const parsedS = trainerMod.parseAST(strS, trainerMod.SCORER_VARS);
    assert.strictEqual(trainerMod.formatAST(parsedS), strS, 'Mutated S expression failed round-trip');

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
// Test 9: Arm isolation (under v1.0 preset)
// ----------------------------------------------------------------------------
runTest(9, 'Arm isolation under v1.0 preset (code vs code+scorer with P(S)=0; hyper vs code with P(C)=0)', () => {
  const weightsNoS = { W: 0.30, H: 0.30, C: 0.20, S: 0.0 };
  const tCode1 = new trainerMod.GTEETrainer({ seed: 555, arm: 'code', preset: 'v1.0', operatorWeights: weightsNoS });
  const tCodeScorer1 = new trainerMod.GTEETrainer({ seed: 555, arm: 'code+scorer', preset: 'v1.0', operatorWeights: weightsNoS });
  tCode1.initGen0();
  tCodeScorer1.initGen0();
  for (let i = 0; i < 4; i++) {
    tCode1.stepGeneration();
    tCodeScorer1.stepGeneration();
  }
  for (let g = 0; g <= 4; g++) {
    assert.strictEqual(tCode1.lineage[g].recordHash, tCodeScorer1.lineage[g].recordHash);
  }

  const weightsNoC = { W: 0.30, H: 0.30, C: 0.0, S: 0.0 };
  const tHyper2 = new trainerMod.GTEETrainer({ seed: 888, arm: 'hyper', preset: 'v1.0', operatorWeights: weightsNoC });
  const tCode2 = new trainerMod.GTEETrainer({ seed: 888, arm: 'code', preset: 'v1.0', operatorWeights: weightsNoC });
  tHyper2.initGen0();
  tCode2.initGen0();
  for (let i = 0; i < 4; i++) {
    tHyper2.stepGeneration();
    tCode2.stepGeneration();
  }
  for (let g = 0; g <= 4; g++) {
    assert.strictEqual(tHyper2.lineage[g].recordHash, tCode2.lineage[g].recordHash);
  }

  return 'Identical record & weight hashes across 4 generations for both arm pairs';
});

// ----------------------------------------------------------------------------
// Test 10: Gate — injected update rules that produce NaN, zero, or explosive steps
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
    assert.strictEqual(rec.weightsHash, parentWeightsHash);
    assert.strictEqual(t.model.weightsToBase64(), parentWeightsB64);
  }

  // 10b. Explosive update rule: const(1000)
  {
    const explosiveGenome = trainerMod.createGen0Genome();
    explosiveGenome.updateRule = { op: 'const', val: 1000.0 };
    const rec = t.stepGeneration({ forcedOperator: 'C', injectedGenome: explosiveGenome });
    assert.strictEqual(rec.accepted, false, 'Explosive update rule must be rejected');
    assert.strictEqual(rec.gateReason, 'reject_explosive_step');
    assert.strictEqual(rec.weightsHash, parentWeightsHash);
    assert.strictEqual(t.model.weightsToBase64(), parentWeightsB64);
  }

  // 10c. NaN update rule: const(NaN)
  {
    const nanGenome = trainerMod.createGen0Genome();
    nanGenome.updateRule = { op: 'const', val: NaN };
    const rec = t.stepGeneration({ forcedOperator: 'C', injectedGenome: nanGenome });
    assert.strictEqual(rec.accepted, false, 'NaN update rule must be rejected');
    assert.strictEqual(rec.gateReason, 'reject_nan_or_inf');
    assert.strictEqual(rec.weightsHash, parentWeightsHash);
    assert.strictEqual(t.model.weightsToBase64(), parentWeightsB64);
  }

  return 'Zero-step, explosive, and NaN update rules rejected; parent weights preserved bit-for-bit';
});

// ----------------------------------------------------------------------------
// Test 11: Evaluator parity + BPTT gradient check + TMAC ternary invariant
// ----------------------------------------------------------------------------
runTest(11, 'Evaluator parity + BPTT finite-difference gradient check + TMAC ternary invariant', () => {
  const modelT = new trainerMod.RWKV7(trainerMod.DEFAULT_MODEL_CONFIG, { seed: 1337 });
  const modelE = new evalMod.RWKV7(evalMod.DEFAULT_MODEL_CONFIG, { seed: 9999 });
  modelE.loadWeightsFromBase64(modelT.weightsToBase64());
  assert.strictEqual(modelT.weightsHash(), modelE.weightsHash());

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

  assert.strictEqual(resT.acc, resE.acc);
  assert.strictEqual(resT.loss, resE.loss);
  assert.strictEqual(resT.score, resE.score);
  assert.strictEqual(resT.gen0Score, resE.gen0Score);

  const wArr = modelT.p['head.weight'];
  const packed = new Uint8Array(Math.ceil(wArr.length / 4));
  const gamma = trainerMod.TMAC.quantizeWeights(wArr, packed, 0.5);
  assert(gamma > 0 && Number.isFinite(gamma));
  const unpacked = trainerMod.TMAC.unpackToFloat32(packed, wArr.length);
  for (let i = 0; i < unpacked.length; i++) {
    const v = unpacked[i];
    assert(v === -1 || v === 0 || v === 1);
  }

  const gradCfg = { vocabSize: 12, dim: 6, headSize: 3, numLayers: 1, dimFFN: 12, dDecay: 4, dAAA: 4, dMV: 4, dGate: 4 };
  const gm = new trainerMod.RWKV7(gradCfg, { seed: 42, ternaryMix: 0.0 });
  const seq = [1, 4, 5, 2, 6];
  function computeLoss() {
    let st = gm.getEmptyState();
    let loss = 0;
    for (let t = 0; t < seq.length - 1; t++) {
      const r = gm.forward(seq[t], st);
      st = r.state;
      let mx = -Infinity;
      for (let v = 0; v < gm.V; v++) if (r.logits[v] > mx) mx = r.logits[v];
      let s = 0;
      for (let v = 0; v < gm.V; v++) s += Number(trainerMod.detExp(r.logits[v] - mx));
      const p = Number(trainerMod.detExp(r.logits[seq[t + 1]] - mx)) / s;
      loss -= Math.log(p);
    }
    return loss / (seq.length - 1);
  }
  gm.zeroGrad();
  const states = [gm.getEmptyState()], fwdTraces = [], logitsArr = [];
  for (let t = 0; t < seq.length - 1; t++) {
    const r = gm.forward(seq[t], states[t]);
    states.push(r.state); fwdTraces.push(r.cache); logitsArr.push(r.logits);
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
    gState = gm.backward(gL, fwdTraces[t], gState);
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
// Test 12: Timing (v1.0 preset)
// ----------------------------------------------------------------------------
runTest(12, 'Timing under v1.0 preset (1 generation headless in Node completes in < 5,000 ms)', () => {
  const t = new trainerMod.GTEETrainer({ seed: 2026, arm: 'code+scorer', preset: 'v1.0' });
  t.initGen0();
  const t0 = Date.now();
  const rec = t.stepGeneration();
  const elapsedMs = Date.now() - t0;
  assert(elapsedMs < 5000, `One generation took ${elapsedMs} ms, exceeding the 5,000 ms limit`);
  return `1 generation completed in ${elapsedMs} ms (budget: < 5,000 ms, operator=${rec.operator})`;
});

// ============================================================================
// GTEE 1.1 NEW MANDATORY TESTS (Tests 13 - 19)
// ============================================================================

// Pinned expected hashes from unmodified GTEE 1.0 code (6 generations on seeds 42 and 101, arm="code+scorer")
const PINNED_V10_HASHES = Object.freeze({
  42: [
    { gen: 0, weightsHash: 'a129ddddc7ced8e2a09fdb591caa92219c11c452116462f44f02c92daf662192', recordHash: '0019336b320474d09c840e7c3bd22db61d63813e6e01184d09d83b8abb701b5b' },
    { gen: 1, weightsHash: '981035683d0e1d477f10a3cb4f623027a73927616e5048749eb132de2728feb3', recordHash: '481efebfc7eb9b4651d076b427222705a1043fc9350aa8d01b01e865acfd0922' },
    { gen: 2, weightsHash: 'b65a78a7a452eba6ad1f084ade82fb35efe136fd6b468fd11adf8199273670f0', recordHash: '850effcc69590c09ac11f402cf2a0512fb7152c57f7c82d40c8e03e2dbe9f05a' },
    { gen: 3, weightsHash: 'b65a78a7a452eba6ad1f084ade82fb35efe136fd6b468fd11adf8199273670f0', recordHash: 'bc6726ba3eb37e9111f10dfda19564290683b6339c029d9a5386e140ff7ae6e2' },
    { gen: 4, weightsHash: '675063c1912ab191c4b738ee04f68671d55b54c5ef32a77d6d268f189a66a59d', recordHash: 'b1656fe3dd7564784ba32fe21c5d8d1b9285b2dd937cde18d07e570eec668571' },
    { gen: 5, weightsHash: '675063c1912ab191c4b738ee04f68671d55b54c5ef32a77d6d268f189a66a59d', recordHash: '9f2ca0b883785dc71d6cdb210270f11508a07b837af8b734b97aabc2033b5175' },
    { gen: 6, weightsHash: 'f815cd0603d0c6743773b88609ef36b482a6ef9a6705eeb58e63209d03cdda16', recordHash: 'b4906d6e41bba6850124625522ad654b3b63149a94b15ef27ea5e476bb938f67' }
  ],
  101: [
    { gen: 0, weightsHash: '42c3a39f1f92b547d87c3f16c53d9e5acde4e1162830f9b8254960e53bd83560', recordHash: 'ef7cdb6283a897fecaaccbda14885c09a6f9d41452ac7a82d8cdbe6bf0346d23' },
    { gen: 1, weightsHash: '59a6beb25da60a6cf768986cca73c817e7e0cb9acd9d8234ed52d6e2645ec01d', recordHash: 'ba640fb7fb395dd5ed44aa1ec2c818da15322b8b64506a4f54984540b8e25265' },
    { gen: 2, weightsHash: '661b3fac28a76611ef6acc940f024873870c91d5dd8c00dc548e7e487b30cb9e', recordHash: '57d276f6e5955657f7ff3ebd938e9dc27eaa53ebb5f7faa68c362d9788d2f400' },
    { gen: 3, weightsHash: '661b3fac28a76611ef6acc940f024873870c91d5dd8c00dc548e7e487b30cb9e', recordHash: '64dd7a6d7ed0faefedb21de96632ffe6167b00563633754c9806e5d2a8b38405' },
    { gen: 4, weightsHash: '661b3fac28a76611ef6acc940f024873870c91d5dd8c00dc548e7e487b30cb9e', recordHash: 'f87d5434897986d6c5e7db3f9e0eb7a711e221cc5ab3f25513fdd7883a914f0f' },
    { gen: 5, weightsHash: '661b3fac28a76611ef6acc940f024873870c91d5dd8c00dc548e7e487b30cb9e', recordHash: 'd86a92f31fd76a5b05655afb9620f55cb9cc09a1f1a763f87ba4e4a923beda9f' },
    { gen: 6, weightsHash: '1f90ab2b983803e81b0a6e6a3ee7ebaf78f98de54af4e36b0f9156a323129e80', recordHash: 'eebb6fed38952c77b06e58e58af88e83dd52b62d0f5ff0a31338a4d7ebbffa58' }
  ]
});

// ----------------------------------------------------------------------------
// Test 13: v1.0 reproduction — with --preset v1.0, a 6-generation run on two seeds
//          produces record hashes identical to unmodified GTEE 1.0, apart from gateConfig
// ----------------------------------------------------------------------------
runTest(13, 'v1.0 reproduction (6-gen runs on seeds 42 & 101 match pinned 1.0 hashes)', () => {
  for (const seed of [42, 101]) {
    const t = new trainerMod.GTEETrainer({ seed, arm: 'code+scorer', preset: 'v1.0' });
    t.initGen0();
    for (let i = 0; i < 6; i++) t.stepGeneration();

    const expected = PINNED_V10_HASHES[seed];
    const v10RecordHashes = trainerMod.computeV10RecordHashChain(t.lineage);

    for (let g = 0; g <= 6; g++) {
      assert.strictEqual(
        t.lineage[g].weightsHash,
        expected[g].weightsHash,
        `Seed ${seed} gen ${g} weightsHash mismatch against pinned 1.0 hash`
      );
      assert.strictEqual(
        v10RecordHashes[g],
        expected[g].recordHash,
        `Seed ${seed} gen ${g} v1.0 recordHash (without gateConfig) mismatch against pinned 1.0 hash`
      );
    }
  }
  return `Seeds 42 & 101 matched all 14 pinned GTEE 1.0 weightsHashes & recordHashes (gen 6: ${PINNED_V10_HASHES[42][6].recordHash.slice(0, 12)}..., ${PINNED_V10_HASHES[101][6].recordHash.slice(0, 12)}...)`;
});

// ----------------------------------------------------------------------------
// Test 14: Gate modes — verify each gateRule (accept & reject on each side of threshold)
//          and each gateScorer (including inflated genome scorer with unchanged gen0 score)
// ----------------------------------------------------------------------------
runTest(14, 'Gate modes (greater, margin, paired_se thresholds + gen0 vs genome scorer inflation)', () => {
  const baseSafety = {
    isGen0: false,
    operator: 'C',
    hasNaNOrInf: false,
    totalStepAbsSum: 1.0,
    totalGradAbsSum: 1.0,
    maxStepAbs: 0.1,
    maxWeightAbs: 1.0
  };

  // 14a. gateRule: "greater" — accept above parent, reject at or below parent
  {
    const cfg = trainerMod.resolveGateConfig({ gateScorer: 'gen0', gateRule: 'greater' });
    const acc = trainerMod.evaluateGate(Object.assign({}, baseSafety, {
      gateConfig: cfg,
      candidateDevScore: -0.20,
      gen0DevScore: -0.20,
      parentDevScore: -0.25,
      parentGen0DevScore: -0.25
    }));
    assert.strictEqual(acc.accepted, true, '"greater" should accept when cand > parent');

    const rej = trainerMod.evaluateGate(Object.assign({}, baseSafety, {
      gateConfig: cfg,
      candidateDevScore: -0.25,
      gen0DevScore: -0.25,
      parentDevScore: -0.25,
      parentGen0DevScore: -0.25
    }));
    assert.strictEqual(rej.accepted, false, '"greater" should reject when cand <= parent');
  }

  // 14b. gateRule: "margin" (gateMargin = 0.05) — accept at +0.06, reject at +0.04
  {
    const cfg = trainerMod.resolveGateConfig({ gateScorer: 'gen0', gateRule: 'margin', gateMargin: 0.05 });
    const acc = trainerMod.evaluateGate(Object.assign({}, baseSafety, {
      gateConfig: cfg,
      candidateDevScore: -0.19,
      gen0DevScore: -0.19,
      parentDevScore: -0.25,
      parentGen0DevScore: -0.25
    }));
    assert.strictEqual(acc.accepted, true, '"margin" should accept when diff (0.06) > gateMargin (0.05)');

    const rej = trainerMod.evaluateGate(Object.assign({}, baseSafety, {
      gateConfig: cfg,
      candidateDevScore: -0.21,
      gen0DevScore: -0.21,
      parentDevScore: -0.25,
      parentGen0DevScore: -0.25
    }));
    assert.strictEqual(rej.accepted, false, '"margin" should reject when diff (0.04) <= gateMargin (0.05)');
  }

  // 14c. gateRule: "paired_se" (gateK = 2.0) — accept when meanDiff > 2*SE, reject when meanDiff <= 2*SE
  {
    const cfg = trainerMod.resolveGateConfig({ gateScorer: 'gen0', gateRule: 'paired_se', gateK: 2.0 });
    // Low variance positive difference: diffs = [0.10, 0.12, 0.08, 0.10] -> mean=0.10, var=0.0002667, SE=0.00816, 2*SE=0.0163 < 0.10
    const parentSamples = new Float32Array([0.1, 0.1, 0.1, 0.1]);
    const candSamplesAcc = new Float32Array([0.20, 0.22, 0.18, 0.20]);
    const acc = trainerMod.evaluateGate(Object.assign({}, baseSafety, {
      gateConfig: cfg,
      candidateDevScore: 0.20,
      gen0DevScore: 0.20,
      parentDevScore: 0.10,
      parentGen0DevScore: 0.10,
      candSampleScores: candSamplesAcc,
      parentSampleScores: parentSamples
    }));
    assert.strictEqual(acc.accepted, true, '"paired_se" should accept when meanDiff > gateK * SE');

    // High variance difference: diffs = [0.50, -0.40, 0.50, -0.40] -> mean=0.05, var=0.27, SE=0.2598, 2*SE=0.5196 > 0.05
    const candSamplesRej = new Float32Array([0.60, -0.30, 0.60, -0.30]);
    const rej = trainerMod.evaluateGate(Object.assign({}, baseSafety, {
      gateConfig: cfg,
      candidateDevScore: 0.15,
      gen0DevScore: 0.15,
      parentDevScore: 0.10,
      parentGen0DevScore: 0.10,
      candSampleScores: candSamplesRej,
      parentSampleScores: parentSamples
    }));
    assert.strictEqual(rej.accepted, false, '"paired_se" should reject when meanDiff <= gateK * SE');
  }

  // 14d. Constructed candidate with inflated genome scorer (add(acc, 10.0)) and unchanged gen-0 score:
  //      Must be REJECTED under gateScorer="gen0" and ACCEPTED under gateScorer="genome" with gateRule="greater"
  {
    // Candidate whose weights do not improve gen-0 score (steps=1 with tiny lr=0.00001 on a zero-effect or neutral step),
    // or directly compare identical injected candidate under both gateScorer modes:
    const inflatedScorerAST = trainerMod.parseAST('add(acc, 10.0)', trainerMod.SCORER_VARS);

    // Run 1: gateScorer = "gen0", gateRule = "greater"
    const tGen0 = new trainerMod.GTEETrainer({
      seed: 42,
      arm: 'code+scorer',
      gateConfig: { preset: 'default', gateScorer: 'gen0', devSeedsPerFamily: 4, devResample: false, gateRule: 'greater', gateK: 1.0, gateMargin: 0.01, gateW: 'gated' }
    });
    tGen0.initGen0();
    // Construct candidate with tiny negative/zero weight change (negated small lr*g so gen0 score does not beat parent, but scorer is add(acc, 10))
    const candGenome1 = trainerMod.createGen0Genome();
    candGenome1.hyper.lr = 0.0001;
    candGenome1.updateRule = trainerMod.parseAST('neg(mul(lr, g))', trainerMod.UPDATE_RULE_VARS);
    candGenome1.scorer = inflatedScorerAST;
    const recGen0 = tGen0.stepGeneration({ forcedOperator: 'S', injectedGenome: candGenome1 });

    // Run 2: gateScorer = "genome", gateRule = "greater" on the exact same seed and candidate
    const tGenome = new trainerMod.GTEETrainer({
      seed: 42,
      arm: 'code+scorer',
      gateConfig: { preset: 'v1.0', gateScorer: 'genome', devSeedsPerFamily: 4, devResample: false, gateRule: 'greater', gateK: 1.0, gateMargin: 0.01, gateW: 'gated' }
    });
    tGenome.initGen0();
    const recGenome = tGenome.stepGeneration({ forcedOperator: 'S', injectedGenome: candGenome1 });

    assert(recGen0.devScore > 9.0, `Inflated genome scorer should produce devScore > 9.0, got ${recGen0.devScore}`);
    assert(recGen0.gen0DevScore <= tGen0.lineage[0].gen0DevScore, 'Candidate gen0DevScore should not beat parent');
    assert.strictEqual(recGen0.accepted, false, 'Inflated scorer candidate must be REJECTED under gateScorer="gen0"');
    assert.strictEqual(recGenome.accepted, true, 'Inflated scorer candidate must be ACCEPTED under gateScorer="genome" with gateRule="greater"');
  }

  return 'Verified greater, margin, and paired_se on both sides of threshold; inflated scorer rejected under gen0 & accepted under genome';
});

// ----------------------------------------------------------------------------
// Test 15: Paired resampling — with devResample: true, candidate and parent are scored
//          on identical seed lists, the seeds are recorded, and rewind reproduces them
// ----------------------------------------------------------------------------
runTest(15, 'Paired resampling (candidate & parent scored on identical fresh 64-seed batch; rewind reproduces seeds)', () => {
  const t = new trainerMod.GTEETrainer({
    seed: 2027,
    arm: 'code',
    gateConfig: {
      preset: 'default',
      gateScorer: 'gen0',
      devSeedsPerFamily: 16,
      devResample: true,
      gateRule: 'paired_se',
      gateK: 1.0,
      gateMargin: 0.01,
      gateW: 'gated'
    }
  });
  t.initGen0();
  for (let i = 0; i < 4; i++) t.stepGeneration();

  for (let k = 0; k <= 4; k++) {
    const rec = t.lineage[k];
    assert.strictEqual(rec.seeds.dev.length, 64, `Expected 64 dev seeds at gen ${k}, got ${rec.seeds.dev.length}`);
    if (k > 0) {
      assert.notStrictEqual(
        trainerMod.canonicalJSON(rec.seeds.dev),
        trainerMod.canonicalJSON(t.lineage[k - 1].seeds.dev),
        `Dev seeds at gen ${k} should be freshly resampled`
      );
      assert.strictEqual(rec.pairedStats.count, 64, `Paired comparison count should be 64 at gen ${k}`);

      // Verify parentBatchGen0DevScore equals evaluating the parent's active weights on rec.seeds.dev
      const parentRec = t.lineage[rec.parentSeqIndex];
      const checkModel = new trainerMod.RWKV7(t.modelConfig, { seed: 1 });
      checkModel.loadWeightsFromBase64(parentRec.weightsB64);
      checkModel.ternaryMix = trainerMod.f32(parentRec.activeGenome.hyper.ternaryMix);
      checkModel.ternaryThreshold = trainerMod.f32(parentRec.activeGenome.hyper.ternaryThreshold);
      const samples = rec.seeds.dev.map(sp => trainerMod.generateTaskSample(sp.familyIndex, sp.seed, false));
      const parentOnBatch = trainerMod.evaluateModelOnSamples(checkModel, samples, parentRec.activeGenome.scorer);
      assert.strictEqual(rec.parentBatchGen0DevScore, parentOnBatch.gen0Score, `Parent batch score mismatch at gen ${k}`);
    }
  }

  // Rewind to every k in 0..4 and verify rebuiltDevSeeds matches rec.seeds.dev
  for (let k = 4; k >= 0; k--) {
    const rw = t.rewindTo(k);
    assert.strictEqual(rw.matches, true, `Rewind failed at k=${k}`);
    assert.strictEqual(rw.devSeedsMatch, true, `Rewind devSeedsMatch failed at k=${k}`);
    assert.strictEqual(
      trainerMod.canonicalJSON(rw.rebuiltDevSeeds),
      trainerMod.canonicalJSON(t.lineage[k].seeds.dev),
      `Rebuilt dev seeds at k=${k} did not match recorded seeds`
    );
  }

  return 'All 5 generations drew fresh 64-seed paired batches, evaluated parent & candidate on identical seeds, and rebuilt via rewind';
});

// ----------------------------------------------------------------------------
// Test 16: gateW — under "always", W steps are accepted whenever safety checks pass
//          and dev score is still logged; under "gated", 1.0 behavior holds
// ----------------------------------------------------------------------------
runTest(16, 'gateW modes ("always" accepts safe W steps even when score drops; "gated" rejects)', () => {
  // Construct a W step with high lr=0.14 that degrades dev score on seed 42
  const degradedHyperGenome = trainerMod.createGen0Genome();
  degradedHyperGenome.hyper.lr = 0.14;

  // Under gateW: "gated"
  const tGated = new trainerMod.GTEETrainer({
    seed: 42,
    arm: 'train',
    gateConfig: {
      preset: 'default',
      gateScorer: 'gen0',
      devSeedsPerFamily: 8,
      devResample: false,
      gateRule: 'margin',
      gateK: 1.0,
      gateMargin: 0.50, // high margin so gated W step is guaranteed rejected by score
      gateW: 'gated'
    }
  });
  tGated.initGen0();
  const recGated = tGated.stepGeneration({ forcedOperator: 'W', injectedGenome: degradedHyperGenome });
  assert.strictEqual(recGated.operator, 'W');
  assert.strictEqual(recGated.accepted, false, 'Under gateW="gated", W step failing margin must be rejected');
  assert.strictEqual(recGated.gateReason, 'reject_score_not_beaten');

  // Under gateW: "always" with the exact same seed, margin, and W step
  const tAlways = new trainerMod.GTEETrainer({
    seed: 42,
    arm: 'train',
    gateConfig: {
      preset: 'default',
      gateScorer: 'gen0',
      devSeedsPerFamily: 8,
      devResample: false,
      gateRule: 'margin',
      gateK: 1.0,
      gateMargin: 0.50,
      gateW: 'always'
    }
  });
  tAlways.initGen0();
  const recAlways = tAlways.stepGeneration({ forcedOperator: 'W', injectedGenome: degradedHyperGenome });
  assert.strictEqual(recAlways.operator, 'W');
  assert.strictEqual(recAlways.accepted, true, 'Under gateW="always", safe W step must be accepted');
  assert.strictEqual(recAlways.gateReason, 'accept_w_always');
  assert(Number.isFinite(recAlways.devScore) && Number.isFinite(recAlways.gen0DevScore), 'Dev scores must still be logged under gateW="always"');
  assert.strictEqual(recAlways.gen0DevScore, recGated.gen0DevScore, 'Logged gen0DevScore must match');

  // Verify safety check still rejects an explosive W step even under gateW="always"
  const explosiveW = trainerMod.createGen0Genome();
  explosiveW.updateRule = { op: 'const', val: 500.0 };
  const recExplosiveW = tAlways.stepGeneration({ forcedOperator: 'W', injectedGenome: explosiveW });
  assert.strictEqual(recExplosiveW.accepted, false, 'Explosive W step must still be rejected under gateW="always"');
  assert.strictEqual(recExplosiveW.gateReason, 'reject_explosive_step');

  return 'gateW="always" accepted safe W step and logged devScore/gen0DevScore while rejecting explosive W step; gateW="gated" rejected non-beating W step';
});

// ----------------------------------------------------------------------------
// Test 17: Arm isolation under the new 1.1 defaults
// ----------------------------------------------------------------------------
runTest(17, 'Arm isolation under new GTEE 1.1 defaults (paired_se, 64 resampled dev seeds, gen0 scorer)', () => {
  const weightsNoS = { W: 0.30, H: 0.30, C: 0.20, S: 0.0 };
  const tCode1 = new trainerMod.GTEETrainer({ seed: 555, arm: 'code', preset: 'default', operatorWeights: weightsNoS });
  const tCodeScorer1 = new trainerMod.GTEETrainer({ seed: 555, arm: 'code+scorer', preset: 'default', operatorWeights: weightsNoS });
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
      `1.1 default lineage mismatch between 'code' and 'code+scorer' (P(S)=0) at gen ${g}`
    );
  }

  const weightsNoC = { W: 0.30, H: 0.30, C: 0.0, S: 0.0 };
  const tHyper2 = new trainerMod.GTEETrainer({ seed: 888, arm: 'hyper', preset: 'default', operatorWeights: weightsNoC });
  const tCode2 = new trainerMod.GTEETrainer({ seed: 888, arm: 'code', preset: 'default', operatorWeights: weightsNoC });
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
      `1.1 default lineage mismatch between 'hyper' and 'code' (P(C)=0) at gen ${g}`
    );
  }

  return 'Identical record & weight hashes across 4 generations under GTEE 1.1 default gateConfig';
});

// ----------------------------------------------------------------------------
// Test 18: gate_config tamper is rejected
// ----------------------------------------------------------------------------
runTest(18, 'gate_config tamper rejection (editing recorded gateConfig is rejected with reason "gate_config")', () => {
  const t = new trainerMod.GTEETrainer({ seed: 1010, arm: 'code', preset: 'default' });
  t.initGen0();
  t.stepGeneration();
  t.stepGeneration();
  const cleanBundle = t.exportStateBundle();

  // 18a. Edit gateScorer in lineage[0].gateConfig
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[0].gateConfig.gateScorer = 'genome';
    b.gateConfig.gateScorer = 'genome';
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.tamperReason, 'gate_config', `Expected tamperReason="gate_config", got "${res.tamperReason}"`);
  }

  // 18b. Edit gateRule and recompute recordHash across the chain
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[0].gateConfig.gateRule = 'greater';
    b.gateConfig.gateRule = 'greater';
    for (let i = 0; i < b.lineage.length; i++) {
      if (i > 0) b.lineage[i].parentHash = b.lineage[b.lineage[i].parentSeqIndex].recordHash;
      b.lineage[i].recordHash = trainerMod.computeRecordHash(b.lineage[i]);
    }
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.tamperReason, 'gate_config', `Expected tamperReason="gate_config", got "${res.tamperReason}"`);
  }

  // 18c. Edit devSeedsPerFamily in gateConfig even if gateConfigSig is recomputed
  {
    const b = JSON.parse(JSON.stringify(cleanBundle));
    b.lineage[0].gateConfig.devSeedsPerFamily = 4;
    b.gateConfig.devSeedsPerFamily = 4;
    b.lineage[0].gateConfigSig = trainerMod.computeGateConfigSig(b.lineage[0].gateConfig, b.rootSeed);
    const res = evalMod.verifyAndEvaluateDescendant(b);
    assert.strictEqual(res.valid, false);
    assert.strictEqual(res.tamperReason, 'gate_config', `Expected tamperReason="gate_config", got "${res.tamperReason}"`);
  }

  return 'All gateConfig tamper attempts rejected with exact reason "gate_config"';
});

// ----------------------------------------------------------------------------
// Test 19: Timing — one generation under the 1.1 defaults (64 paired dev samples)
//          runs in under 5 seconds headless (print measurement)
// ----------------------------------------------------------------------------
runTest(19, 'Timing under GTEE 1.1 defaults (64 paired resampled dev samples in < 5,000 ms)', () => {
  const t = new trainerMod.GTEETrainer({ seed: 2026, arm: 'code+scorer', preset: 'default' });
  t.initGen0();
  const t0 = Date.now();
  const rec = t.stepGeneration();
  const elapsedMs = Date.now() - t0;
  assert.strictEqual(rec.seeds.dev.length, 64);
  assert.strictEqual(rec.pairedStats.count, 64);
  assert(elapsedMs < 5000, `One generation under 1.1 defaults took ${elapsedMs} ms, exceeding 5,000 ms limit`);
  return `1 generation (64 paired dev samples) completed in ${elapsedMs} ms (budget: < 5,000 ms, operator=${rec.operator})`;
});

console.log('======================================================================');
console.log(`Verification Summary: ${passedCount}/19 passed, ${failedCount} failed`);
console.log('======================================================================');

if (failedCount > 0) {
  process.exit(1);
}