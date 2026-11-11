#!/usr/bin/env node
'use strict';

// ============================================================================
// GTEE 1.1 Batch Runner (run-batch.js)
// Usage: node run-batch.js --arms train,hyper,code,code+scorer --seeds 101,202,...
//        --gens N --preset <name or default> [gate flags] --out <dir>
//        [--jobs J] [--evaluator-key <hex>] [--keep-artifacts]
// ============================================================================

const fs = require('fs');
const path = require('path');
const trainerMod = require('./run.js');
const evalMod = require('./evaluator-cli.js');

function parseBool(v) {
  if (typeof v === 'boolean') return v;
  const s = String(v).toLowerCase().trim();
  if (s === 'true' || s === '1' || s === 'yes') return true;
  if (s === 'false' || s === '0' || s === 'no') return false;
  throw new Error(`Invalid boolean value: ${v}`);
}

function parseBatchArgs(argv) {
  const opts = {
    arms: ['train', 'hyper', 'code', 'code+scorer'],
    seeds: [42],
    gens: 3,
    preset: 'default',
    out: './batch_out',
    jobs: 1,
    evaluatorKey: null,
    keepArtifacts: false
  };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === '--arms' && argv[i + 1]) {
      opts.arms = argv[++i].split(',').map(s => s.trim()).filter(Boolean);
    } else if (a === '--seeds' && argv[i + 1]) {
      opts.seeds = argv[++i].split(',').map(s => parseInt(s.trim(), 10) >>> 0).filter(n => !Number.isNaN(n));
    } else if ((a === '--gens' || a === '--generations') && argv[i + 1]) {
      opts.gens = parseInt(argv[++i], 10);
    } else if (a === '--preset' && argv[i + 1]) {
      opts.preset = argv[++i];
    } else if (a === '--gate-scorer' && argv[i + 1]) {
      opts.gateScorer = argv[++i];
    } else if (a === '--dev-seeds-per-family' && argv[i + 1]) {
      opts.devSeedsPerFamily = parseInt(argv[++i], 10);
    } else if (a === '--dev-resample' && argv[i + 1]) {
      opts.devResample = parseBool(argv[++i]);
    } else if (a === '--gate-rule' && argv[i + 1]) {
      opts.gateRule = argv[++i];
    } else if (a === '--gate-k' && argv[i + 1]) {
      opts.gateK = parseFloat(argv[++i]);
    } else if (a === '--gate-margin' && argv[i + 1]) {
      opts.gateMargin = parseFloat(argv[++i]);
    } else if (a === '--gate-w' && argv[i + 1]) {
      opts.gateW = argv[++i];
    } else if (a === '--evaluator-key' && argv[i + 1]) {
      opts.evaluatorKey = evalMod.parseEvalKey(argv[++i]);
    } else if (a === '--keep-artifacts') {
      if (argv[i + 1] && !argv[i + 1].startsWith('--')) {
        opts.keepArtifacts = parseBool(argv[++i]);
      } else {
        opts.keepArtifacts = true;
      }
    } else if (a === '--jobs' && argv[i + 1]) {
      opts.jobs = Math.max(1, parseInt(argv[++i], 10) || 1);
    } else if (a === '--out' && argv[i + 1]) {
      opts.out = argv[++i];
    }
  }
  return opts;
}

function csvCell(val) {
  if (val === null || val === undefined) return '';
  const s = String(val);
  if (/[",\n\r]/.test(s)) {
    return '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

function runSingleBatchJob(jobOpts) {
  const { arm, seed, gens, gateConfig, runDir, keepArtifacts, evaluatorKey } = jobOpts;

  // 1. Train via trainer (note: trainer never receives evaluatorKey)
  const { trainer, bundle, descendantHTML } = trainerMod.runHeadless({
    arm,
    seed,
    gens,
    gateConfig,
    out: runDir
  });

  // 2. Grade with primary evaluator
  const eval1 = evalMod.verifyAndEvaluateDescendant(descendantHTML);
  if (!eval1.valid) {
    throw new Error(`Primary evaluator failed on arm=${arm} seed=${seed}: ${eval1.tamperReason} (${eval1.error})`);
  }

  // 3. Optionally grade with second held-out evaluator key
  let eval2 = null;
  if (evaluatorKey !== null && evaluatorKey !== undefined) {
    eval2 = evalMod.verifyAndEvaluateDescendant(descendantHTML, { evaluatorKey });
    if (!eval2.valid) {
      throw new Error(`Secondary evaluator failed on arm=${arm} seed=${seed}: ${eval2.tamperReason}`);
    }
  }

  // 4. Clean up large per-run artifacts unless --keep-artifacts is set
  if (!keepArtifacts && runDir && fs.existsSync(runDir)) {
    fs.rmSync(runDir, { recursive: true, force: true });
  }

  // 5. Compute operator proposed/accepted counts across generations 1..N
  const counts = {
    W: { proposed: 0, accepted: 0 },
    H: { proposed: 0, accepted: 0 },
    C: { proposed: 0, accepted: 0 },
    S: { proposed: 0, accepted: 0 }
  };
  for (let i = 1; i < trainer.lineage.length; i++) {
    const r = trainer.lineage[i];
    if (counts[r.operator]) {
      counts[r.operator].proposed++;
      if (r.accepted) counts[r.operator].accepted++;
    }
  }

  // 6. Checkpoints at 0, 25%, 50%, 75%, 100% of N
  const N = Math.max(0, trainer.lineage.length - 1);
  const idx0 = 0;
  const idx25 = Math.min(N, Math.round(0.25 * N));
  const idx50 = Math.min(N, Math.round(0.50 * N));
  const idx75 = Math.min(N, Math.round(0.75 * N));
  const idx100 = N;

  const gEval = idx => eval1.generations[idx];
  const lastRec = trainer.lineage[idx100];
  const lastEval1 = gEval(idx100);
  const lastEval2 = eval2 ? eval2.generations[idx100] : null;

  return {
    arm,
    seed,
    preset: gateConfig.preset,
    gateScorer: gateConfig.gateScorer,
    devSeedsPerFamily: gateConfig.devSeedsPerFamily,
    devResample: gateConfig.devResample,
    gateRule: gateConfig.gateRule,
    gateK: gateConfig.gateK,
    gateMargin: gateConfig.gateMargin,
    gateW: gateConfig.gateW,
    proposed_W: counts.W.proposed,
    accepted_W: counts.W.accepted,
    proposed_H: counts.H.proposed,
    accepted_H: counts.H.accepted,
    proposed_C: counts.C.proposed,
    accepted_C: counts.C.accepted,
    proposed_S: counts.S.proposed,
    accepted_S: counts.S.accepted,
    final_eval_std_acc: lastEval1.evaluatorStandard.acc,
    final_eval_long_acc: lastEval1.evaluatorLongOOD.acc,
    final_eval_std_score: lastEval1.evaluatorStandard.gen0Score,
    final_eval_long_score: lastEval1.evaluatorLongOOD.gen0Score,
    final_eval2_std_acc: lastEval2 ? lastEval2.evaluatorStandard.acc : '',
    final_eval2_long_acc: lastEval2 ? lastEval2.evaluatorLongOOD.acc : '',
    final_eval2_std_score: lastEval2 ? lastEval2.evaluatorStandard.gen0Score : '',
    final_eval2_long_score: lastEval2 ? lastEval2.evaluatorLongOOD.gen0Score : '',
    eval_std_acc_g0: gEval(idx0).evaluatorStandard.acc,
    eval_std_acc_g25: gEval(idx25).evaluatorStandard.acc,
    eval_std_acc_g50: gEval(idx50).evaluatorStandard.acc,
    eval_std_acc_g75: gEval(idx75).evaluatorStandard.acc,
    eval_std_acc_g100: gEval(idx100).evaluatorStandard.acc,
    eval_long_acc_g0: gEval(idx0).evaluatorLongOOD.acc,
    eval_long_acc_g25: gEval(idx25).evaluatorLongOOD.acc,
    eval_long_acc_g50: gEval(idx50).evaluatorLongOOD.acc,
    eval_long_acc_g75: gEval(idx75).evaluatorLongOOD.acc,
    eval_long_acc_g100: gEval(idx100).evaluatorLongOOD.acc,
    final_dev_score_genome: lastRec.devScore,
    final_dev_score_gen0: lastRec.gen0DevScore,
    final_genome_minus_gen0_dev_score: lastRec.genomeMinusGen0DevScore,
    final_update_rule: trainerMod.formatAST(lastRec.activeGenome.updateRule),
    final_scorer: trainerMod.formatAST(lastRec.activeGenome.scorer)
  };
}

function runBatch(rawOpts) {
  const gateConfig = trainerMod.resolveGateConfig(rawOpts.gateConfig || rawOpts);
  const outDir = path.resolve(rawOpts.out || './batch_out');
  fs.mkdirSync(outDir, { recursive: true });

  const rows = [];
  for (const arm of rawOpts.arms) {
    for (const seed of rawOpts.seeds) {
      const safeArm = arm.replace(/[^a-zA-Z0-9_-]/g, '_');
      const runDir = path.join(outDir, `run_${safeArm}_s${seed}`);
      const row = runSingleBatchJob({
        arm,
        seed,
        gens: rawOpts.gens,
        gateConfig,
        runDir,
        keepArtifacts: !!rawOpts.keepArtifacts,
        evaluatorKey: rawOpts.evaluatorKey !== undefined ? rawOpts.evaluatorKey : null
      });
      rows.push(row);
    }
  }

  const headers = [
    'arm', 'seed', 'preset', 'gateScorer', 'devSeedsPerFamily', 'devResample',
    'gateRule', 'gateK', 'gateMargin', 'gateW',
    'proposed_W', 'accepted_W', 'proposed_H', 'accepted_H',
    'proposed_C', 'accepted_C', 'proposed_S', 'accepted_S',
    'final_eval_std_acc', 'final_eval_long_acc', 'final_eval_std_score', 'final_eval_long_score',
    'final_eval2_std_acc', 'final_eval2_long_acc', 'final_eval2_std_score', 'final_eval2_long_score',
    'eval_std_acc_g0', 'eval_std_acc_g25', 'eval_std_acc_g50', 'eval_std_acc_g75', 'eval_std_acc_g100',
    'eval_long_acc_g0', 'eval_long_acc_g25', 'eval_long_acc_g50', 'eval_long_acc_g75', 'eval_long_acc_g100',
    'final_dev_score_genome', 'final_dev_score_gen0', 'final_genome_minus_gen0_dev_score',
    'final_update_rule', 'final_scorer'
  ];

  const csvLines = [headers.join(',')];
  for (const r of rows) {
    csvLines.push(headers.map(h => csvCell(r[h])).join(','));
  }
  const csvPath = path.join(outDir, 'summary.csv');
  fs.writeFileSync(csvPath, csvLines.join('\n') + '\n', 'utf8');

  return { csvPath, rows, gateConfig };
}

if (require.main === module) {
  const opts = parseBatchArgs(process.argv.slice(2));
  const t0 = Date.now();
  const res = runBatch(opts);
  console.log(JSON.stringify({
    status: 'ok',
    runsCompleted: res.rows.length,
    summaryCsv: res.csvPath,
    elapsedMs: Date.now() - t0
  }));
}

module.exports = {
  parseBatchArgs,
  runSingleBatchJob,
  runBatch
};