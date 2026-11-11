const fs=require('fs'),path=require('path');
const T=require(require('path').resolve(__dirname,'../../run.js'));
const arms=['train','hyper','code','code+scorer'];const seeds=[101,202,303,404,505,606,707,808];
const out={};const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;const sd=a=>{const m=mean(a);return Math.sqrt(mean(a.map(x=>(x-m)**2)))};
for(const arm of arms){const rows=[];
 for(const s of seeds){const d=`${process.env.OUT||'/tmp/gexp'}/${arm.replace('+','_')}_${s}`;if(!fs.existsSync(d+'/fresh.json'))continue;
  const ev=require(d+'/eval.json').generations, fr=require(d+'/fresh.json').generations;
  const L=fs.readFileSync(d+'/lineage.jsonl','utf8').trim().split('\n').map(JSON.parse);
  const last=L.length-1;const accBy={};for(const r of L.slice(1)){accBy[r.operator]=accBy[r.operator]||[0,0];accBy[r.operator][1]++;if(r.accepted)accBy[r.operator][0]++;}
  const cp=[0,25,50,100,150].map(g=>+ev[g].evaluatorStandard.acc.toFixed(3));
  const tip=L[last]; const accRecs=L.filter(r=>r.accepted); const lastAcc=accRecs[accRecs.length-1];
  rows.push({seed:s,stdAcc:ev[last].evaluatorStandard.acc,oodAcc:ev[last].evaluatorLongOOD.acc,freshStd:fr[last].evaluatorStandard.acc,freshOod:fr[last].evaluatorLongOOD.acc,curve:cp,accBy,
   selfDev:lastAcc.devScore,gen0Dev:lastAcc.gen0DevScore,update:T.formatAST(tip.activeGenome.updateRule),scorer:T.formatAST(tip.activeGenome.scorer),lr:tip.activeGenome.hyper.lr,steps:tip.activeGenome.hyper.steps});}
 const f=k=>`${mean(rows.map(r=>r[k])).toFixed(3)}±${sd(rows.map(r=>r[k])).toFixed(3)}`;
 console.log(`\n=== ${arm} (n=${rows.length})  evaluator std ${f('stdAcc')}  long ${f('oodAcc')}  | fresh std ${f('freshStd')}  long ${f('freshOod')}`);
 console.log('  mean curve std acc @gen 0/25/50/100/150:',[0,1,2,3,4].map(i=>mean(rows.map(r=>r.curve[i])).toFixed(3)).join(' '));
 const tot={};for(const r of rows)for(const[k,[a,n]] of Object.entries(r.accBy)){tot[k]=tot[k]||[0,0];tot[k][0]+=a;tot[k][1]+=n;}
 console.log('  accepted/proposed by operator:',Object.entries(tot).map(([k,[a,n]])=>`${k} ${a}/${n}`).join('  '));
 if(arm.startsWith('code'))for(const r of rows)console.log(`  seed ${r.seed}: std ${r.stdAcc.toFixed(3)} fresh ${r.freshStd.toFixed(3)} | selfDev ${r.selfDev.toFixed(3)} gen0Dev ${r.gen0Dev.toFixed(3)} | update ${r.update.slice(0,90)}${arm==='code+scorer'?' | scorer '+r.scorer.slice(0,70):''}`);
 out[arm]=rows;}
fs.writeFileSync((process.env.OUT||'/tmp/gexp')+'/summary.json',JSON.stringify(out,null,1));
