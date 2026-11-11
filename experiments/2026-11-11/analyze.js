const fs=require('fs'),path=require('path');
const dir=process.argv[2]||'results';
const rows=fs.readdirSync(dir).map(f=>JSON.parse(fs.readFileSync(path.join(dir,f))));
const arms=['train','hyper','code','code+scorer'], cfgs=['A','B','C'];
const mean=a=>a.reduce((x,y)=>x+y,0)/a.length;
const sd=a=>{const m=mean(a);return Math.sqrt(a.reduce((s,x)=>s+(x-m)**2,0)/(a.length-1));};
const out={};
for(const c of cfgs){out[c]={};for(const a of arms){const r=rows.filter(x=>x.cfg===c&&x.arm===a).sort((x,y)=>x.seed-y.seed);if(!r.length)continue;
 const f=k=>r.map(x=>x[k]);
 out[c][a]={n:r.length,std:mean(f('stdAcc')),stdSD:sd(f('stdAcc')),ood:mean(f('oodAcc')),fresh:mean(f('freshStd')),freshOod:mean(f('freshOod')),
  accepted:mean(f('accepted')),legacy:mean(f('legacyAccepts')),gen0Drop:mean(f('gen0DropAccepts')),
  zeroSeeds:r.filter(x=>x.stdAcc===0).length,
  curve:[0,1,2,3,4].map(i=>+mean(r.map(x=>x.curve[i])).toFixed(3)),
  selfMinusGen0:mean(r.map(x=>x.selfDev-x.gen0Dev)),
  perSeed:r.map(x=>+x.stdAcc.toFixed(3))};}}
console.log(JSON.stringify(out,null,1));
