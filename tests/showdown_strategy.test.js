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


test('tie splitting shares the occupied payout ranks', () => {
  const payouts = w.normalizePayouts([
    {from:1,to:1,amount:100},
    {from:2,to:2,amount:50},
    {from:3,to:5,amount:10},
  ], 10);
  assert.equal(w.payoutForTie(payouts, 1, 2), 75);
  assert.equal(w.payoutForTie(payouts, 2, 3), (50 + 10 + 10) / 3);
});


test('weighted field-strength summary respects sampled lineup weights', () => {
  const summary=w.weightedDistributionSummary([1,2,10],[1,2,1]);
  assert.equal(summary.mean,3.75);
  assert.equal(summary.median,2);
  assert.equal(summary.p90,10);
  assert.equal(summary.p99,10);
  assert.equal(summary.min,1);
  assert.equal(summary.max,10);
});

test('contest readiness requires opponents and payouts', () => {
  assert.equal(w.contestReady({fieldSize:100,entries:5,entryFee:1,payouts:[{from:1,to:1,amount:20}]}), true);
  assert.equal(w.contestReady({fieldSize:5,entries:5,entryFee:1,payouts:[{from:1,to:1,amount:20}]}), false);
  assert.equal(w.contestReady({fieldSize:100,entries:5,entryFee:1,payouts:[]}), false);
});

test('ownership prior always fills exactly five Yahoo roster slots', () => {
  const players = [
    {fp:22,pos:'QB'},{fp:19,pos:'QB'},{fp:17,pos:'RB'},{fp:14,pos:'WR'},
    {fp:10,pos:'WR'},{fp:8,pos:'TE'},{fp:6,pos:'DEF'}
  ].map((p,i)=>({...p,team:i<4?'A':'B',salary:10,cv:.5}));
  w.setModel({players,latent:Array(21).fill(0),settings:{}});
  const own = w.ownershipRates({fieldSize:2000,entryFee:1});
  assert.ok(Math.abs(Array.from(own).reduce((a,b)=>a+b,0)-5) < 1e-8);
  const star = w.superstarOwnershipRates(own);
  assert.ok(Math.abs(Array.from(star).reduce((a,b)=>a+b,0)-1) < 1e-8);
});

test('field EV prices a fully duplicated one-lineup field with tie splitting', () => {
  const players = [10,9,8,7,6].map((fp,i)=>({
    fp,cv:.01,salary:1,pos:i===0?'QB':'WR',team:i<4?'A':'B'
  }));
  const payload={players,latent:Array(10).fill(0),settings:{}};
  w.setModel(payload);
  w.simulate(200,356);
  const options={salaryCap:5,minSalaryPct:0,ceilingWeight:.85,maxCandidates:10,
    meanReserve:10,nearOptimalRatio:.95,objective:'field_ev',entries:1,
    maxPlayerExposure:1,maxSuperstarExposure:1,maxShared:4,constructionRules:[],
    simulations:200,seed:356,fieldSize:10,entryFee:1,
    payouts:[{from:1,to:1,amount:100}],fieldSampleSize:9,fieldSimulations:100};
  const rosters=w.enumerate([0,1,2,3,4],options);
  const scored=w.score(rosters,w.screen(rosters,options),options);
  w.applyFieldEV(scored,rosters,options);
  assert.equal(scored.total,5);
  const best=w.orderBy(scored,'field_ev')[0];
  assert.ok(Number.isFinite(scored.expectedProfit[best]));
  assert.ok(scored.expectedDuplicates[best] >= 0);
  assert.ok(scored.expectedPayout[best] > 0);
});


test('joint portfolio EV inserts all selected entries into the same contest ranks', () => {
  const players = [12,11,10,9,8,7].map((fp,i)=>({
    fp,cv:.05,salary:1,pos:i===0?'QB':'WR',team:i<3?'A':'B'
  }));
  w.setModel({players,latent:Array(15).fill(0),settings:{}});
  w.simulate(400,356);
  const options={salaryCap:5,minSalaryPct:0,ceilingWeight:.85,maxCandidates:40,
    meanReserve:40,nearOptimalRatio:.95,objective:'field_ev',entries:2,
    maxPlayerExposure:1,maxSuperstarExposure:1,maxShared:4,constructionRules:[],
    simulations:400,seed:356,fieldSize:10,entryFee:1,
    payouts:[{from:1,to:1,amount:20},{from:2,to:3,amount:5}],
    fieldSampleSize:9,fieldSimulations:200};
  const rosters=w.enumerate([0,1,2,3,4,5],options);
  const scored=w.score(rosters,w.screen(rosters,options),options);
  w.applyFieldEV(scored,rosters,options);
  const chosen=w.orderBy(scored,'field_ev').slice(0,2);
  const single=w.evaluateJointPortfolioEV(scored,[chosen[0]],options);
  assert.ok(Math.abs(single.expected_profit-scored.expectedProfit[chosen[0]]) < 1e-9,
    'a one-entry joint contest must reduce exactly to standalone EV');

  const evaluation=w.evaluateJointPortfolioEV(scored,chosen,options);
  const described=w.describe(scored,chosen);

  assert.equal(evaluation.objective,'joint_portfolio_contest_ev');
  assert.equal(evaluation.entries,2);
  assert.equal(evaluation.opponent_entries,8);
  assert.equal(evaluation.evaluation_scenarios,200);
  assert.ok(evaluation.profitable_rate >= 0 && evaluation.profitable_rate <= 1);
  assert.ok(evaluation.any_cash_rate >= 0 && evaluation.any_cash_rate <= 1);
  assert.ok(evaluation.any_top_one_rate >= 0 && evaluation.any_top_one_rate <= 1);
  assert.ok(evaluation.any_first_rate >= 0 && evaluation.any_first_rate <= 1);
  assert.ok(evaluation.expected_cashes >= 0 && evaluation.expected_cashes <= 2);
  assert.ok(evaluation.profit_p10 <= evaluation.profit_median);
  assert.ok(evaluation.profit_median <= evaluation.profit_p90);

  const jointProfit=described.reduce((sum,row)=>sum+row.joint_expected_profit,0);
  assert.ok(Math.abs(jointProfit-evaluation.expected_profit) < 1e-9);
  const standaloneProfit=chosen.reduce((sum,c)=>sum+scored.expectedProfit[c],0);
  assert.ok(Math.abs(standaloneProfit-evaluation.standalone_expected_profit) < 1e-9);
  assert.equal(scored.fieldSummary.joint_portfolio_evaluated,true);
  assert.equal(scored.fieldSummary.joint_opponent_entries,8);
  const opponentFP=scored.fieldSummary.opponent_expected_fp;
  assert.equal(opponentFP.sample_size,9);
  assert.ok(opponentFP.min <= opponentFP.median);
  assert.ok(opponentFP.median <= opponentFP.p90);
  assert.ok(opponentFP.p90 <= opponentFP.p99);
  assert.ok(opponentFP.p99 <= opponentFP.max);
  assert.ok(opponentFP.mean >= opponentFP.min && opponentFP.mean <= opponentFP.max);

  const selectedExpected=chosen.map(c=>scored.expected[c]);
  const portfolioFP=scored.fieldSummary.portfolio_expected_fp;
  assert.ok(Math.abs(portfolioFP.mean-
    selectedExpected.reduce((a,b)=>a+b,0)/selectedExpected.length) < 1e-9);
  assert.equal(portfolioFP.min,Math.min(...selectedExpected));
  assert.equal(portfolioFP.max,Math.max(...selectedExpected));
  assert.equal(scored.fieldSummary.h2h_expected_fp,Math.max(...scored.expected));

  const h2h=w.describe(scored,[chosen[0]],null,false)[0];
  assert.equal(h2h.expected_profit,undefined);
  assert.equal(h2h.joint_expected_profit,undefined);
});

function h2hFixture(h2hEntries) {
  // A self-contained two-team game: published slates are replaced every week.
  const spec = [
    ['QB', 'A', 22, 38], ['QB', 'B', 20, 36], ['RB', 'A', 17, 32], ['RB', 'B', 15, 30],
    ['WR', 'A', 14, 28], ['WR', 'B', 13, 27], ['WR', 'A', 9, 18], ['WR', 'B', 8, 16],
    ['TE', 'A', 8, 17], ['TE', 'B', 6, 13], ['DEF', 'A', 7, 12], ['DEF', 'B', 6, 11],
  ];
  const players = spec.map(([pos, team, fp, salary], i) => ({
    name: `P${i}`, pos, team, fp, salary, cv: pos === 'QB' ? .5 : .75, zero: .02,
  }));
  const n = players.length;
  const payload = {players, latent: Array(n * (n - 1) / 2).fill(0.1), settings: {}};
  const options = {salaryCap:120,minSalaryPct:0,ceilingWeight:.85,maxCandidates:2000,
    meanReserve:750,nearOptimalRatio:.95,entries:20,objective:'tournament',positionLimits:{},
    simulations:2000,seed:356,fieldTemperature:1.25,h2hEntries};
  w.setModel(payload);
  w.simulate(options.simulations, options.seed);
  const rosters = w.enumerate(players.map((_, i) => i), options);
  const scored = w.score(rosters, w.screen(rosters, options), options, () => {});
  return {payload, options, result: w.h2hMultiEntry(scored, options)};
}

test('multi-entry H2H: each mode fills the requested entries the way it says', () => {
  const {result} = h2hFixture(4);
  assert.equal(result.entries, 4);
  const {repeat, rotate, next_best: next} = result.modes;
  const key = (e) => e.ids.slice().sort((a, b) => a - b).join(',') + '*' + e.superstar;
  // Repeat: one pair four times. Next-best starts from that same pair.
  assert.equal(new Set(repeat.entries.map(key)).size, 1);
  assert.equal(key(next.entries[0]), key(repeat.entries[0]));
  assert.equal(new Set(next.entries.map(key)).size, 4);
  // Rotate: the same five players, a different Superstar on each entry.
  const roster = (e) => e.ids.slice().sort((a, b) => a - b).join(',');
  assert.equal(new Set(rotate.entries.map(roster)).size, 1);
  assert.equal(new Set(rotate.entries.map((e) => e.superstar)).size, 4);
  assert.equal(rotate.entries[0].superstar, repeat.entries[0].superstar);
});

test('multi-entry H2H: expected wins is the sum of entry win chances', () => {
  const {result} = h2hFixture(3);
  for (const mode of Object.values(result.modes)) {
    for (const opponent of ['sharp', 'public']) {
      const m = mode[opponent];
      const sum = m.per_entry.reduce((a, b) => a + b, 0);
      assert.ok(Math.abs(m.expected_wins - sum) < 1e-9);
      assert.ok(m.zero_wins >= 0 && m.zero_wins <= 1);
      assert.ok(m.all_wins >= 0 && m.all_wins <= 1);
      assert.ok(Math.abs(m.win_rate - m.expected_wins / 3) < 1e-12);
    }
  }
  // Repeated entries all have the same chance; a rotated Superstar never beats the best one.
  const repeat = result.modes.repeat.sharp.per_entry;
  assert.ok(repeat.every((p) => Math.abs(p - repeat[0]) < 1e-12));
  const rotate = result.modes.rotate.sharp.per_entry;
  assert.ok(rotate.slice(1).every((p) => p <= rotate[0] + 1e-9));
});

test('multi-entry H2H: the entry count is clamped to 1..10', () => {
  assert.equal(h2hFixture(25).result.entries, 10);
  assert.equal(h2hFixture(0).result.entries, 3);    // missing -> default
  const ten = h2hFixture(10).result.modes.rotate.entries;
  // Past five entries the rotation starts over.
  assert.equal(ten[5].superstar, ten[0].superstar);
});
