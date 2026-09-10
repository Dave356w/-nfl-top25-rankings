"use strict";
const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const workerPath = path.join(__dirname, '../site/showdown-worker.js');
const w = require(workerPath);

function scoringFixture() {
  const players = [100,50,30,30,30,30].map((fp,i) => ({
    fp, cv:i === 0 ? 10 : 0.01, salary:1, pos:'WR', team:i === 5 ? 'B' : 'A'
  }));
  const payload = { players, latent:Array(15).fill(0), settings:{} };
  w.setModel(payload);
  w.simulate(20000,356);
  const options = {salaryCap:5,minSalaryPct:0,ceilingWeight:.85,maxCandidates:1,
    meanReserve:1,nearOptimalRatio:.95,objective:'floor'};
  const rosters = w.enumerate([0,1,2,3,4,5],options);
  return {payload,options,rosters};
}

function portfolioFixture() {
  const lineups = [
    [0,1,2,3,4], [0,5,6,7,8], [0,9,10,11,12], [0,13,14,15,16],
    [17,18,19,20,21], [22,23,24,25,26]
  ];
  w.setModel({players:Array.from({length:27},()=>({pos:'WR'})),settings:{}});
  return {total:6,ids:Int32Array.from(lineups.flat()),
    superstars:Int32Array.from([0,0,0,0,17,22]),
    tournament:Float64Array.from([100,99,98,97,96,95])};
}
const portfolioOptions = {entries:5,maxPlayerExposure:.7,maxSuperstarExposure:.7,
  maxShared:3,constructionRules:[]};

test('70% caps mean three appearances across five entries', () => {
  const scored = portfolioFixture();
  const result = w.portfolio(scored,[0,1,2,3,4,5],portfolioOptions);
  assert.equal(result.chosen.length,5);
  assert.deepEqual(result.chosen,[0,1,2,4,5]);
  assert.equal(result.chosen.filter(i=>scored.superstars[i]===0).length,3);
});

test('fractional caps below one allow zero; single-entry bypasses caps', () => {
  assert.equal(w.exposureLimit(5,.05),0);
  assert.equal(w.exposureLimit(100,.29),29);
  assert.equal(w.exposureLimit(1,.05),1);
  for (const invalid of [NaN,Infinity,-.1,1.1]) {
    assert.throws(()=>w.exposureLimit(5,invalid),/Exposure/);
  }
});

test('floor screen preserves the stable lineup that upside screening discarded', () => {
  const {options,rosters} = scoringFixture();
  const floor = w.score(rosters,w.screen(rosters,options),options);
  const full = w.score(rosters,w.screen(rosters,{...options,maxCandidates:1000}),options);
  const upside = w.score(rosters,w.screen(rosters,{...options,objective:'tournament'}),options);
  const best = scored=>scored.p25[w.orderBy(scored,'floor')[0]];
  assert.equal(best(floor),best(full));
  assert.ok(best(floor)>best(upside)+40);
});

test('expected and mean screens retain maximum analytic expectation', () => {
  const {options,rosters} = scoringFixture();
  for (const objective of ['expected','mean']) {
    const screened = w.screen(rosters,{...options,objective,meanReserve:0});
    const best = Math.max(...screened.expected);
    assert.ok(Array.from(screened.selected).some(i=>screened.expected[i]===best));
  }
});

test('disabling construction quotas admits otherwise excluded shapes', () => {
  const scored = portfolioFixture();
  const rules = [{name:'QB required',count:1,positions:{QB:1}}];
  const blocked = w.portfolio(scored,[0,1,2,3,4,5],{
    ...portfolioOptions,constructionRules:rules});
  assert.equal(blocked.chosen.length,0);
  const free = w.portfolio(scored,[0,1,2,3,4,5],portfolioOptions);
  assert.equal(free.chosen.length,5);
});

test('worker refuses partial portfolios even with construction quotas off', () => {
  const {payload,options} = scoringFixture();
  const messages=[];
  const self={postMessage:m=>messages.push(m)};
  vm.runInNewContext(fs.readFileSync(workerPath,'utf8'),{self,performance});
  self.onmessage({data:{type:'load',payload}});
  self.onmessage({data:{type:'solve',included:[0,1,2,3,4,5],options:{...options,
    entries:5,maxPlayerExposure:1,maxSuperstarExposure:1,maxShared:3,
    constructionRules:[],simulations:100,seed:356}}});
  assert.equal(messages.at(-1).type,'error');
  assert.match(messages.at(-1).message,/partial result/);
  assert.ok(!messages.some(m=>m.type==='result'));
});
