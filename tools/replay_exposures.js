#!/usr/bin/env node
'use strict';
// Run today's worker on archived pregame inputs; settled scores are grading only.
const fs = require('fs');
const worker = require('../site/showdown-worker.js');
const job = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const seed = Number(process.argv[3] || 356);
job.payload.settings.random_seed = seed;
worker.setModel(job.payload);
const base = {contest:'tournament',entries:20,salaryCap:job.salary_cap,detail:'standard',
  included:job.payload.players.map((_, i)=>i)};
const options = worker.engineOptions(base);
worker.simulate(options.simulations, seed);
const rosters=worker.enumerate(Int32Array.from(base.included), options);
const scored=worker.score(rosters, worker.screen(rosters, options), options, null);
const order=worker.orderBy(scored,'expected');
let hindsightBest=-Infinity;
for(let r=0;r<rosters.salary.length;r++){
  const ids=Array.from(rosters.ids.subarray(r*5,(r+1)*5));
  hindsightBest=Math.max(hindsightBest,ids.reduce((s,i)=>s+job.actuals[i],0)+.5*Math.max(...ids.map(i=>job.actuals[i])));
}
const anchor=worker.describe(scored,[order[0]])[0];
const anchorActual=anchor.ids.reduce((s,i)=>s+job.actuals[i],0)+.5*job.actuals[anchor.superstar];
const rows=[];
for (const [player, star] of [[.75,.25],[1,.25],[.75,1],[1,1]]) {
  const opts={...options,maxPlayerExposure:player,maxSuperstarExposure:star};
  opts.limits=worker.labLimits(20,opts);
  const started=Date.now();
  const built=worker.portfolio(scored,order,opts);
  const lineups=worker.describe(scored,built.chosen);
  const seen=new Set(), pc=new Map(), sc=new Map();
  for(const l of lineups){
    const key=l.ids.slice().sort((a,b)=>a-b).join(',')+':'+l.superstar;
    if(seen.has(key))throw Error('Duplicate entry');seen.add(key);
    if(l.salary>job.salary_cap+1e-8||l.ids.length!==5||!l.ids.includes(l.superstar))throw Error('Invalid roster');
    for(const id of l.ids)pc.set(id,(pc.get(id)||0)+1);
    sc.set(l.superstar,(sc.get(l.superstar)||0)+1);
    l.actual_fp=l.ids.reduce((s,i)=>s+job.actuals[i],0)+.5*job.actuals[l.superstar];
    l.players=l.ids.map(i=>job.payload.players[i].name);
    l.superstar_name=job.payload.players[l.superstar].name;
  }
  if(Math.max(...pc.values())>opts.limits.player||Math.max(...sc.values())>opts.limits.superstar)throw Error('Exposure violation');
  rows.push({setting:player+'/'+star,seed,filled:lineups.length,requested:20,
    actual_best:Math.max(...lineups.map(x=>x.actual_fp)),
    actual_mean:lineups.reduce((s,l)=>s+l.actual_fp,0)/lineups.length,
    max_player_count:Math.max(...pc.values()), max_superstar_count:Math.max(...sc.values()),
    evaluation:built.evaluation, elapsed_ms:Date.now()-started,lineups});
}
process.stdout.write(JSON.stringify({audit:job.audit,seed,simulations:options.simulations,
  max_candidates:options.maxCandidates,valid_rosters:rosters.salary.length,candidates_scored:scored.total,
  hindsight_best_in_model_pool:hindsightBest,anchor_actual:anchorActual,rows}));
