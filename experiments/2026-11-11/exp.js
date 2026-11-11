// One run: node exp.js <gteeDir> <config A|B|C> <arm> <seed> <outDir>
const path=require('path'),fs=require('fs');
const [dir,cfg,arm,seedS,out]=process.argv.slice(2); const seed=+seedS;
const T=require(path.join(dir,'run.js')), E=require(path.join(dir,'evaluator-cli.js'));
const CFG={A:{preset:'default'},
  B:{preset:'default',gateScorer:'genome',devSeedsPerFamily:4,devResample:false,gateRule:'greater',gateW:'gated'},
  C:{preset:'v1.0'}}[cfg];
const t0=Date.now();
const {trainer,descendantHTML}=T.runHeadless({arm,seed,gens:150,gateConfig:CFG});
const e1=E.verifyAndEvaluateDescendant(descendantHTML);
const e2=E.verifyAndEvaluateDescendant(descendantHTML,{evaluatorKey:0x5EEDF00D});
if(!e1.valid||!e2.valid) throw new Error('invalid '+e1.tamperReason+' '+e1.error);
const L=trainer.lineage; const acc={W:[0,0],H:[0,0],C:[0,0],S:[0,0]};
let legacy=0, activeDev=L[0].devScore, activeGen0=L[0].gen0DevScore, gen0Drop=0;
for(let i=1;i<L.length;i++){const r=L[i]; acc[r.operator][1]++; if(r.accepted){acc[r.operator][0]++;
  if(!(r.devScore>activeDev)) legacy++;            // accepted without beating the real active parent (own-scorer, fixed seeds)
  if(r.gen0DevScore<activeGen0) gen0Drop++;
  activeDev=r.devScore; activeGen0=r.gen0DevScore;}}
let last=L[0]; for(const r of L) if(r.accepted) last=r;
const g=e1.generations, n=g.length-1, pick=f=>[0,.25,.5,.75,1].map(q=>+g[Math.round(q*n)].evaluatorStandard.acc.toFixed(3));
const res={cfg,arm,seed,stdAcc:g[n].evaluatorStandard.acc,oodAcc:g[n].evaluatorLongOOD.acc,
 freshStd:e2.generations[n].evaluatorStandard.acc,freshOod:e2.generations[n].evaluatorLongOOD.acc,
 curve:pick(),acc,accepted:Object.values(acc).reduce((a,b)=>a+b[0],0),legacyAccepts:legacy,gen0DropAccepts:gen0Drop,
 selfDev:last.devScore,gen0Dev:last.gen0DevScore,update:T.formatAST(last.activeGenome.updateRule),scorer:T.formatAST(last.activeGenome.scorer),ms:Date.now()-t0};
fs.writeFileSync(path.join(out,`${cfg}_${arm.replace('+','_')}_${seed}.json`),JSON.stringify(res));
console.log(cfg,arm,seed,'done',res.ms);
