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

/* ---------- solve(): the page's single entry point ---------- */

// A self-contained two-team game: published slates are replaced every week.
function syntheticGame() {
  const spec = [
    ['QB', 'A', 22, 38], ['QB', 'B', 20, 36], ['RB', 'A', 17, 32], ['RB', 'B', 15, 30],
    ['WR', 'A', 14, 28], ['WR', 'B', 13, 27], ['WR', 'A', 9, 18], ['WR', 'B', 8, 16],
    ['TE', 'A', 8, 17], ['TE', 'B', 6, 13], ['DEF', 'A', 7, 12], ['DEF', 'B', 6, 11],
  ];
  const players = spec.map(([pos, team, fp, salary], i) => ({
    name: `P${i}`, pos, team, fp, salary, cv: pos === 'QB' ? .5 : .75, zero: .02,
  }));
  const n = players.length;
  return {
    players, latent: Array(n * (n - 1) / 2).fill(0.1),
    settings: {
      random_seed: 356, simulations: 2000, max_candidate_lineups: 2000,
      min_salary_used_pct: 0, candidate_ceiling_weight: .85, mean_candidate_reserve: 750,
      near_optimal_ratio: .95, max_shared_players: 3, max_player_exposure: .5,
      max_superstar_exposure: .35, tournament_lineups: 20, position_limits: {},
    },
  };
}
const game = syntheticGame();
const byFp = game.players.map((_, i) => i)
  .sort((a, b) => (game.players[b].fp - game.players[a].fp) || (a - b));
function solve(request) {
  w.setModel(game);
  return w.solve({salaryCap: 120, detail: 'full', ...request});
}
const pairKey = (e) => e.ids.slice().sort((a, b) => a - b).join(',') + '*' + e.superstar;
function counts(entries, pick) {
  const map = new Map();
  for (const e of entries) for (const id of pick(e)) map.set(id, (map.get(id) || 0) + 1);
  return map;
}

test('engine options: both contests share the lab rules, the published settings fill the rest', () => {
  w.setModel(game);
  const tournament = w.engineOptions({contest: 'tournament', entries: 20, salaryCap: 120, detail: 'full'});
  assert.equal(tournament.objective, 'portfolio');
  assert.equal(tournament.maxPlayerExposure, .75);   // not the published run's .5
  assert.equal(tournament.maxSuperstarExposure, .25); // not the published run's .35
  assert.equal(tournament.maxShared, null);           // no overlap limit
  assert.deepEqual(tournament.constructionRules, []);
  assert.equal(tournament.simulations, 2000);
  const h2h = w.engineOptions({contest: 'h2h', entries: 3, salaryCap: 120});
  assert.equal(h2h.objective, 'expected');
  assert.equal(h2h.maxPlayerExposure, .75);
  assert.equal(h2h.maxSuperstarExposure, .25);
  assert.equal(h2h.maxShared, null);
  assert.equal(h2h.simulations, 10000);          // standard is the default detail
  assert.equal(w.engineOptions({contest: 'h2h', detail: 'quick'}).simulations, 5000);
  assert.equal(w.engineOptions({contest: 'h2h', maxSuperstarExposure: .5}).maxSuperstarExposure, .5);
});

test('solve refuses requests it cannot answer, in words a visitor can act on', () => {
  w.setModel(game);
  assert.throws(() => w.solve({contest: 'h2h', entries: 3}), /no salary cap/);
  assert.throws(() => solve({contest: 'h2h', entries: 51}), /between 1 and 50/);
  assert.throws(() => solve({contest: 'tournament', entries: 151}), /between 1 and 150/);
  assert.throws(() => solve({contest: 'h2h', entries: 3, included: [0, 1, 2]}), /five are required/);
  assert.throws(() => solve({contest: 'h2h', entries: 3, maxPlayerExposure: 1.5}), /between 0 and 1/);
});

// The rules every lab entry obeys, whichever contest built it.
function assertLabRules(result, included = game.players.map((_, i) => i), shares = {player: .75, superstar: .25}) {
  const {player, superstar} = result.limits;
  assert.equal(player, Math.max(1, Math.floor(result.requested * shares.player + 1e-9)));
  assert.equal(superstar, Math.max(1, Math.floor(result.requested * shares.superstar + 1e-9)));
  assert.ok(Math.max(...counts(result.entries, (e) => e.ids).values()) <= player);
  assert.ok(Math.max(...counts(result.entries, (e) => [e.superstar]).values()) <= superstar);
  assert.equal(new Set(result.entries.map(pairKey)).size, result.entries.length, 'an exact repeat');
  const pool = new Set(included);
  for (const e of result.entries) {
    assert.equal(new Set(e.ids).size, 5);
    assert.ok(e.ids.includes(e.superstar));
    assert.ok(e.salary <= 120);
    assert.ok(e.ids.every((id) => pool.has(id)), 'an excluded player was used');
    assert.deepEqual(new Set(e.ids.map((id) => game.players[id].team)), new Set(['A', 'B']));
  }
}

test('H2H: entries follow projected points under the lab rules', () => {
  const result = solve({contest: 'h2h', entries: 12});
  assert.equal(result.contest, 'h2h');
  assert.equal(result.filled, 12);
  assert.deepEqual(result.limits, {player: 9, superstar: 3});
  assertLabRules(result);
  const fps = result.entries.map((e) => e.expected_fp);
  assert.deepEqual(fps, fps.slice().sort((a, b) => b - a));
  // The best pair is always the first entry.
  w.setModel(game);
  const rosters = w.enumerate(game.players.map((_, i) => i), {salaryCap: 120, minSalaryPct: 0});
  assert.equal(pairKey(result.entries[0]), pairKey(w.topPairs(rosters, 1)[0]));
});

test('H2H: entries may share players, and the same five may return under a new Superstar', () => {
  const result = solve({contest: 'h2h', entries: 4});
  assertLabRules(result);
  assert.deepEqual(result.limits, {player: 3, superstar: 1});
  const rosterKey = (e) => e.ids.slice().sort((a, b) => a - b).join(',');
  const rosters = counts(result.entries, (e) => [rosterKey(e)]);
  assert.ok(Math.max(...rosters.values()) > 1, 'no roster was reused with another Superstar');
});

test('H2H: small counts still allow each player and Superstar once', () => {
  const three = solve({contest: 'h2h', entries: 3});
  assert.deepEqual(three.limits, {player: 2, superstar: 1});
  assert.equal(three.filled, 3);
  assertLabRules(three);
});

test('a pick leaves room for the rest instead of spending a last appearance', () => {
  // Three entries, each player twice at most, each Superstar once. Taking the
  // two best (the same five under two Superstars) spends all five players'
  // last appearance and strands the third entry; the lookahead takes the
  // third-best instead and fills all three.
  w.setModel({players: Array.from({length: 10}, () => ({fp: 1})), settings: {}});
  const lineups = [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4], [0, 5, 6, 7, 8], [1, 5, 6, 7, 9]];
  const pairs = {total: 4, ids: Int32Array.from(lineups.flat()),
    superstars: Int32Array.from([0, 1, 5, 6]), expected: Float64Array.from([40, 39, 30, 29])};
  const order = [0, 1, 2, 3];
  const build = (lookahead) => {
    const track = w.labTracker(pairs, {player: 2, superstar: 1}), chosen = [];
    while (chosen.length < 3) {
      const pick = lookahead ? w.lookaheadPick(track, order, 3 - chosen.length - 1)
        : order.find((c) => track.fits(c));
      if (pick === undefined || pick < 0) break;
      track.take(pick);
      chosen.push(pick);
    }
    return chosen;
  };
  assert.deepEqual(build(false), [0, 1]);       // plain greedy strands the third entry
  assert.deepEqual(build(true), [0, 2, 3]);
});

test('H2H: a custom player limit holds and expected wins sums the entries', () => {
  const result = solve({contest: 'h2h', entries: 12, maxPlayerExposure: .5});
  assert.equal(result.limits.player, 6);
  assert.ok(Math.max(...counts(result.entries, (e) => e.ids).values()) <= 6);
  const sum = result.entries.reduce((total, e) => total + e.win_chance, 0);
  assert.ok(Math.abs(result.expected_wins - sum) < 1e-9);
  assert.ok(Math.abs(result.win_rate - sum / result.filled) < 1e-12);
  assert.ok(result.zero_wins >= 0 && result.zero_wins <= 1);
  assert.equal(result.opponents, 100);
});

test('exclusions and depth-edited projections carry into every entry', () => {
  const included = game.players.map((_, i) => i).filter((i) => i !== byFp[1]);
  for (const contest of ['h2h', 'tournament']) {
    assertLabRules(solve({contest, entries: 8, included}), included);
  }
  // A depth edit reaches the worker as a changed projection in the payload:
  // the promoted player fills their 75% share and leads the first entry.
  const before = solve({contest: 'h2h', entries: 4});
  assert.notEqual(before.entries[0].superstar, 7);
  const promoted = {...game, players: game.players.map((p, i) => i === 7 ? {...p, fp: 40} : p)};
  w.setModel(promoted);
  const after = w.solve({contest: 'h2h', entries: 4, salaryCap: 120, detail: 'full'});
  assert.equal(after.entries[0].superstar, 7);
  assert.equal(after.entries.filter((e) => e.ids.includes(7)).length, after.limits.player);
});

test('H2H: opponents may play a player you left out', () => {
  const star = byFp[0];
  const without = solve({contest: 'h2h', entries: 5,
    included: game.players.map((_, i) => i).filter((i) => i !== star)});
  assert.ok(without.entries.every((e) => !e.ids.includes(star)));
  const all = solve({contest: 'h2h', entries: 5});
  // Your best player is gone but the opponents' is not: you win less often.
  assert.ok(without.expected_wins < all.expected_wins - .3,
    `${without.expected_wins} vs ${all.expected_wins}`);
});

test('topPairs is the highest-expected pairs of an enumeration', () => {
  w.setModel(game);
  const rosters = w.enumerate(game.players.map((_, i) => i), {salaryCap: 120, minSalaryPct: 0});
  const value = (ids, star) => ids.reduce((s, i) => s + game.players[i].fp, 0) + .5 * game.players[star].fp;
  const all = [];
  for (let r = 0; r < rosters.salary.length; r++) {
    const ids = Array.from(rosters.ids.subarray(r * 5, r * 5 + 5));
    for (const star of ids) all.push(value(ids, star));
  }
  all.sort((a, b) => b - a);
  const top = w.topPairs(rosters, 25).map((p) => value(p.ids, p.superstar));
  top.forEach((v, i) => assert.ok(Math.abs(v - all[i]) < 1e-9));
});

test('tournament: entries follow the lab rules with no overlap limit', () => {
  const result = solve({contest: 'tournament', entries: 20});
  assert.equal(result.contest, 'tournament');
  assert.equal(result.requested, 20);
  assert.equal(result.filled, 20);
  assert.equal(result.entries.length, result.filled);
  assert.deepEqual(result.limits, {player: 15, superstar: 5});
  assertLabRules(result);
  // The published run caps shared players at three; the lab does not.
  assert.ok(result.diversity.max_shared > 3, `max shared ${result.diversity.max_shared}`);
});

test('tournament: a short list comes back when the limits run out', () => {
  const result = solve({contest: 'tournament', entries: 20, maxPlayerExposure: .1});
  assert.ok(result.filled >= 1 && result.filled < 20);
  assertLabRules(result, undefined, {player: .1, superstar: .25});
});

test('tournament: one entry is the highest-expected lineup', () => {
  const one = solve({contest: 'tournament', entries: 1});
  w.setModel(game);
  const rosters = w.enumerate(game.players.map((_, i) => i), {salaryCap: 120, minSalaryPct: 0});
  const best = w.topPairs(rosters, 1)[0];
  assert.equal(pairKey(one.entries[0]), pairKey(best));
});

test('the worker wraps solve in load / solve / result messages', () => {
  const messages = [];
  const self = {postMessage: (m) => messages.push(m)};
  vm.runInNewContext(fs.readFileSync(workerPath, 'utf8'), {self, Date, Math});
  self.onmessage({data: {type: 'load', payload: game}});
  assert.equal(messages[0].type, 'loaded');
  assert.equal(messages[0].protocol, w.WORKER_PROTOCOL);
  self.onmessage({data: {type: 'solve', request: {contest: 'h2h', entries: 3, salaryCap: 120, detail: 'full'}}});
  assert.ok(messages.some((m) => m.type === 'progress' && m.stage === 'Pricing head-to-head'));
  assert.equal(messages.at(-1).type, 'result');
  assert.equal(messages.at(-1).result.filled, 3);
  self.onmessage({data: {type: 'solve', request: {contest: 'h2h', entries: 3, detail: 'full'}}});
  assert.equal(messages.at(-1).type, 'error');
  assert.match(messages.at(-1).message, /no salary cap/);
});
