/* Exercise the shipped page's event handlers with controlled fetches/workers. */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");
const html = fs.readFileSync(path.join(__dirname, "../site/showdown.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const flush = () => new Promise(setImmediate);

function payload(id) {
  return {
    game_id: id, matchup: id, generated_utc: "2026-09-09T21:00:00Z",
    snapshot_id: "same-snapshot", salary_cap: 100, kickoff_utc: "2026-09-10T00:00:00Z",
    players: Array.from({ length: 5 }, (_, i) => ({
      name: `Player ${i}`, pos: "WR", team: i < 3 ? "A" : "B", salary: 10, fp: 5,
    })),
    settings: { tournament_lineups: 20, max_shared_players: 3,
      max_player_exposure: 0.5, max_superstar_exposure: 0.35,
      min_salary_used_pct: 0, simulations: 20000, max_candidate_lineups: 25000,
      position_limits: { QB: [0, 2], RB: [0, 2], WR: [0, 3], TE: [0, 2], DEF: [0, 1] } },
    reference: null,
  };
}

async function page() {
  const nodes = {}, workers = [], routes = new Map();
  function node(id) {
    return nodes[id] ||= {
      value: "", innerHTML: "", textContent: "", hidden: false,
      disabled: id === "run", style: {}, dataset: {}, handlers: {},
      addEventListener(event, fn) { this.handlers[event] = fn; },
    };
  }
  node("detail").value = "balanced";
  node("objective").value = "tournament";
  const index = { snapshot_id: "same-snapshot", games: ["A", "B", "C"].map(id => ({
    matchup: id, game_id: id, file: `${id}.json`,
  })) };
  const routesData = { "index.json": index, "A.json": payload("A"),
    "B.json": payload("B"), "C.json": payload("C") };
  class Worker {
    constructor(url) {
      this.url = url;
      this.messages = [];
      this.terminated = false;
      workers.push(this);
    }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    emit(data) { this.onmessage({ data }); }
  }
  const context = {
    Set, Worker, localStorage: { getItem: () => null },
    document: { getElementById: node, documentElement: { dataset: {} }, querySelectorAll: () => [] },
    fetch: async url => {
      const name = url.split("/").pop();
      const result = routes.has(name) ? await routes.get(name)() : routesData[name];
      return { ok: true, json: async () => result };
    },
  };
  vm.runInNewContext(script, context);
  await flush();
  return { node, workers, routes, routesData,
    choose(id) { node("game").value = `${id}.json`; node("game").handlers.change(); } };
}

test("worker URL is snapshot-versioned and protocol mismatches fail visibly", async () => {
  const p = await page();
  assert.match(
    p.workers[0].url,
    /showdown-worker\.js\?protocol=4&snapshot=same-snapshot$/
  );
  p.workers[0].emit({ type: "loaded", players: 5, protocol: 3 });
  assert.equal(p.workers[0].terminated, true);
  assert.equal(p.node("run").disabled, true);
  assert.match(p.node("status-text").innerHTML, /out of sync/);
});

test("switching game during a solve cancels work and restores Optimize", async () => {
  const p = await page();
  p.node("run").handlers.click();
  const old = p.workers[0];
  assert.equal(p.node("run").disabled, true);
  p.choose("B");
  assert.equal(old.terminated, true);
  await flush();
  assert.equal(p.node("run").disabled, false);
  assert.equal(p.node("progress").hidden, true);
  old.emit({ type: "error", message: "obsolete worker" });
  assert.doesNotMatch(p.node("status-text").innerHTML, /obsolete/);
  p.node("run").handlers.click();
  assert.equal(p.workers.at(-1).messages.at(-1).type, "solve");
});

test("out-of-order game responses keep the latest selection", async () => {
  const p = await page();
  let releaseB, releaseC;
  const b = new Promise(resolve => { releaseB = resolve; });
  const c = new Promise(resolve => { releaseC = resolve; });
  p.routes.set("B.json", () => b); p.routes.set("C.json", () => c);
  p.choose("B"); p.choose("C");
  releaseC(payload("C")); await flush();
  releaseB(payload("B")); await flush();
  assert.equal(p.workers.at(-1).messages[0].payload.game_id, "C");
  assert.equal(p.node("run").disabled, false);
});

test("failed loads disable solving and a later selection recovers", async () => {
  const p = await page();
  p.routes.set("B.json", () => Promise.reject(new Error("unavailable")));
  p.choose("B"); await flush();
  assert.equal(p.node("run").disabled, true);
  assert.match(p.node("status-text").innerHTML, /unavailable/);
  p.node("none").handlers.click(); // No payload while loading/failed.
  p.choose("C"); await flush();
  assert.equal(p.node("run").disabled, false);
});

test("a game from another snapshot cannot be solved", async () => {
  const p = await page();
  p.routes.set("B.json", async () => ({ ...payload("B"), snapshot_id: "older" }));
  p.choose("B"); await flush();
  assert.equal(p.node("run").disabled, true);
  assert.match(p.node("status-text").innerHTML, /different snapshots/);
});

test("changing settings invalidates both running and displayed results", async () => {
  const p = await page();
  p.node("run").handlers.click();
  const old = p.workers[0];
  p.node("portfolio").innerHTML = "Previous calculation";
  p.node("entries").handlers.change();
  assert.equal(old.terminated, true);
  assert.equal(p.node("run").disabled, false);
  assert.equal(p.node("portfolio").innerHTML, "");
  assert.match(p.node("portfolio-empty").innerHTML, /Settings changed/);
});

test("results distinguish expectation, sample mean and candidate rates", async () => {
  const p = await page();
  p.workers[0].emit({ type: "result", valid_rosters: 1, candidates_scored: 1,
    timing: { total: 10 }, diversity: null, portfolio: [{
      ids: [0, 1, 2, 3, 4], superstar: 0, salary: 50, expected_fp: 27.5,
      sim_mean: 27.3, floor_p25: 20, ceiling_p90: 40, ceiling_p95: 45,
      near_optimal_rate: 0.1, win_rate: 0.01, tournament_score: 1,
    }] });
  const rendered = p.node("portfolio").innerHTML;
  assert.match(rendered, /Entry 1/);
  assert.match(rendered, /27.50 expected FP/);
  assert.match(rendered, /sim mean 27.30/);
  assert.match(rendered, /candidate-best 1.00%/);
  assert.doesNotMatch(rendered, /<span>win /);
});


test("quota switch and integer exposure counts reach the worker", async () => {
  const p = await page();
  p.node("entries").value = 5;
  p.node("entries").handlers.change();
  assert.match(p.node("exposure-hint").textContent, /at most 2 of 5/);
  assert.equal(p.node("construction").value, "off");
  p.node("run").handlers.click();
  assert.equal(p.workers.at(-1).messages.at(-1).options.constructionRules.length, 0);
  const rules = [{ name: "QB", count: 1, positions: { QB: 1 } }];
  p.routesData["A.json"].settings.portfolio_construction_rules = rules;
  p.node("construction").value = "on";
  p.node("construction").handlers.change();
  p.node("run").handlers.click();
  assert.equal(p.workers.at(-1).messages.at(-1).options.constructionRules, rules);
  p.node("entries").value = 1;
  p.node("entries").handlers.change();
  assert.match(p.node("exposure-hint").textContent, /do not apply/);
});

test("incomplete published portfolios are not displayed as valid entries", async () => {
  const p = await page();
  p.routesData["B.json"].reference = {
    valid_rosters: 10, portfolio: [{ids:[0,1,2,3,4],superstar:0}], reliability:[]
  };
  p.choose("B");
  await flush();
  assert.equal(p.node("reference").innerHTML, "");
  assert.match(p.node("reference-note").textContent, /partial portfolio is hidden/);
});

/* Yahoo does not price every single-game slate. Those games publish a full model
 * with no cap, and the page has to collect one before it will solve -- a guessed
 * cap silently changes which lineups are legal. */

test("a game with no published cap cannot be solved until one is entered", async () => {
  const p = await page();
  p.routesData["B.json"] = { ...payload("B"), salary_cap: null };
  p.choose("B"); await flush();
  assert.equal(p.node("cap-field").hidden, false);
  assert.equal(p.node("run").disabled, true);
  assert.match(p.node("cap-hint").textContent, /Yahoo published no cap/);
  assert.match(p.node("game-hint").textContent, /not published by Yahoo/);

  p.node("cap").value = "130";
  p.node("cap").handlers.input();
  assert.equal(p.node("run").disabled, false);
  assert.match(p.node("cap-hint").textContent, /a cap you entered/);

  p.node("run").handlers.click();
  assert.equal(p.workers.at(-1).messages.at(-1).options.salaryCap, 130);
});

test("a non-positive cap does not unlock solving", async () => {
  const p = await page();
  p.routesData["B.json"] = { ...payload("B"), salary_cap: null };
  p.choose("B"); await flush();
  for (const bad of ["0", "-5", "", "abc"]) {
    p.node("cap").value = bad;
    p.node("cap").handlers.input();
    assert.equal(p.node("run").disabled, true, `cap ${bad} should not unlock Optimize`);
  }
});

test("an entered cap does not leak across games", async () => {
  const p = await page();
  p.routesData["B.json"] = { ...payload("B"), salary_cap: null };
  p.routesData["C.json"] = { ...payload("C"), salary_cap: null };
  p.choose("B"); await flush();
  p.node("cap").value = "130";
  p.node("cap").handlers.input();
  assert.equal(p.node("run").disabled, false);

  p.choose("C"); await flush();
  assert.equal(p.node("run").disabled, true, "C must ask for its own cap");
  assert.equal(p.node("cap").value, "");

  p.choose("B"); await flush();
  assert.equal(p.node("run").disabled, false, "B keeps the cap already entered for it");
  assert.equal(p.node("cap").value, 130);
});

test("a priced game hides the cap field entirely", async () => {
  const p = await page();
  assert.equal(p.node("cap-field").hidden, true);
  assert.equal(p.node("run").disabled, false);
  p.node("run").handlers.click();
  assert.equal(p.workers.at(-1).messages.at(-1).options.salaryCap, 100);
});


test("Tournament EV requires contest inputs and sends the payout curve", async () => {
  const p = await page();
  p.node("objective").value = "field_ev";
  p.node("objective").handlers.change();
  assert.equal(p.node("run").disabled, true);

  p.node("field-size").value = "4700";
  p.node("field-size").handlers.change();
  p.node("entry-fee").value = "0.25";
  p.node("entry-fee").handlers.change();
  p.node("payouts").value = "1=100\n2=50\n3-10=10";
  p.node("payouts").handlers.input();

  assert.equal(p.node("run").disabled, false);
  p.node("run").handlers.click();
  const options = p.workers.at(-1).messages.at(-1).options;
  assert.equal(options.fieldSize, 4700);
  assert.equal(options.entryFee, 0.25);
  assert.equal(
    JSON.stringify(options.payouts.map(row => [row.from, row.to, row.amount])),
    JSON.stringify([[1,1,100],[2,2,50],[3,10,10]])
  );
});

test("GPP results distinguish joint portfolio EV from standalone entry EV", async () => {
  const p = await page();
  p.workers[0].emit({ type: "result", valid_rosters: 1, candidates_scored: 1,
    timing: { total: 10 }, diversity: null,
    field_summary: {sampled_opponents:500,opponent_entries:4699,
      standalone_opponent_entries:4699,joint_opponent_entries:4680,
      evaluation_scenarios:1000,ownership_observations:25,ownership_contests:5,
      opponent_expected_fp:{mean:61.2,median:62.1,p90:72.5,p99:80.4,min:40,max:85,
        sample_size:500},
      portfolio_expected_fp:{mean:75.4,min:70.1,max:82.2},h2h_expected_fp:83.0},
    scenario_evaluation: {objective:"joint_portfolio_contest_ev",expected_profit:1.10,
      expected_payout:6.10,roi:.22,entries:20,profitable_rate:.61,any_cash_rate:.88,
      any_top_one_rate:.42,any_first_rate:.07,expected_cashes:4.2,
      expected_top_one_finishes:.6,profit_p10:-3.5,profit_median:.75,profit_p90:8.5,
      worst_profit:-5,best_profit:95,opponent_entries:4680,evaluation_scenarios:1000,
      standalone_expected_profit:1.25,standalone_expected_payout:6.25,
      standalone_roi:.25,standalone_price_taking:true,experimental_field_model:true},
    portfolio: [{
      ids:[0,1,2,3,4],superstar:0,salary:50,expected_fp:27.5,sim_mean:27.3,
      floor_p25:20,ceiling_p90:40,ceiling_p95:45,near_optimal_rate:.1,win_rate:.01,
      tournament_score:1,expected_payout:.40,expected_profit:.15,roi:.6,cash_rate:.2,
      top_one_rate:.03,first_rate:.001,solo_first_rate:.001,expected_duplicates:2.5,
      joint_expected_payout:.37,joint_expected_profit:.12,joint_roi:.48,
      joint_cash_rate:.18,joint_top_one_rate:.025,joint_first_rate:.0008,
      joint_solo_first_rate:.0008,joint_expected_duplicates:2.4,
    }],
    h2h_anchor: [{
      ids:[0,1,2,3,4],superstar:1,salary:50,expected_fp:28.0,sim_mean:27.8,
      floor_p25:21,ceiling_p90:39,ceiling_p95:44,near_optimal_rate:.09,win_rate:.009,
      tournament_score:.8,
    }]
  });
  assert.match(p.node("portfolio").innerHTML, /joint EV \+\$0\.12/);
  assert.match(p.node("portfolio").innerHTML, /standalone EV \+\$0\.15/);
  assert.match(p.node("portfolio").innerHTML, /joint opp duplicates 2\.4/);
  assert.match(p.node("result-note").innerHTML, /Joint portfolio EV under experimental field prior/);
  assert.match(p.node("result-note").innerHTML, /Sum of standalone entry EVs/);
  assert.match(p.node("result-note").innerHTML, /Not a calibrated return estimate/);
  assert.match(p.node("result-note").innerHTML, /Field-based EV is experimental/);
  assert.equal(p.node("field-strength-card").hidden, false);
  assert.match(p.node("field-strength").innerHTML, /Opponent expected FP/);
  assert.match(p.node("field-strength").innerHTML, /mean 61\.20/);
  assert.match(p.node("field-strength").innerHTML, /P90 72\.50/);
  assert.match(p.node("field-strength").innerHTML, /Selected portfolio expected FP/);
  assert.match(p.node("field-strength").innerHTML, /H2H anchor: 83\.00/);
  assert.match(p.node("field-strength").innerHTML, /selection-biased sparse prior/);
  assert.equal(p.node("h2h-card").hidden, false);
  assert.match(p.node("h2h").innerHTML, /28\.00 expected FP/);
  assert.doesNotMatch(p.node("h2h").innerHTML, /ROI/);
});


test("Yahoo quarter-dollar preset fills the 4704-entry payout ladder", async () => {
  const p = await page();
  p.node("contest-preset").value = "yahoo_025_1k";
  p.node("contest-preset").handlers.change();
  assert.equal(p.node("field-size").value, 4704);
  assert.equal(p.node("entry-fee").value, 0.25);
  assert.match(p.node("payouts").value, /1=100/);
  assert.match(p.node("payouts").value, /501-915=0\.50/);
  p.node("objective").value = "field_ev";
  p.node("objective").handlers.change();
  assert.equal(p.node("run").disabled, false);
});

test("multi-entry H2H compares strategies and switches entries without re-solving", async () => {
  const p = await page();
  const entry = (star) => ({ ids: [0, 1, 2, 3, 4], superstar: star, expected_fp: 30 + star, salary: 50 });
  const stats = (wins, perEntry) => ({ expected_wins: wins, win_rate: wins / 2, sd_wins: .7,
    zero_wins: .12, all_wins: .4, winning_record: .4, per_entry: perEntry });
  p.workers[0].emit({ type: "result", valid_rosters: 1, candidates_scored: 1,
    timing: { total: 10 }, diversity: null, field_summary: null, portfolio: [],
    h2h_anchor: [{ ids: [0, 1, 2, 3, 4], superstar: 0, salary: 50, expected_fp: 30 }],
    h2h_multi: { entries: 2, opponents: { sharp: 100, public: 400 }, modes: {
      rotate: { entries: [entry(0), entry(3)], sharp: stats(1.1, [.6, .5]), public: stats(1.8, [.9, .9]) },
      repeat: { entries: [entry(0), entry(0)], sharp: stats(1.2, [.6, .6]), public: stats(1.8, [.9, .9]) },
      next_best: { entries: [entry(0), entry(1)], sharp: stats(1.19, [.6, .59]), public: stats(1.8, [.9, .9]) },
    } } });
  assert.equal(p.node("h2h-multi-wrap").hidden, false);
  assert.match(p.node("h2h-compare").innerHTML, /Rotate Superstar/);
  assert.match(p.node("h2h-compare").innerHTML, /1\.10/);
  assert.match(p.node("h2h-compare").innerHTML, /12\.0%/);
  assert.match(p.node("h2h-multi").innerHTML, /★ Player 3/);
  assert.match(p.node("h2h-multi").innerHTML, /win vs sharp 50\.0%/);
  const messages = p.workers[0].messages.length;
  p.node("h2h-mode").value = "repeat";
  p.node("h2h-mode").handlers.change();
  assert.doesNotMatch(p.node("h2h-multi").innerHTML, /★ Player 3/);
  assert.match(p.node("h2h-compare").innerHTML, /class='picked'><td>Repeat top lineup/);
  assert.equal(p.workers[0].messages.length, messages);
});

test("a result without multi-entry H2H hides that section", async () => {
  const p = await page();
  p.workers[0].emit({ type: "result", valid_rosters: 1, candidates_scored: 1,
    timing: { total: 10 }, diversity: null, field_summary: null, portfolio: [],
    h2h_anchor: [{ ids: [0, 1, 2, 3, 4], superstar: 0, salary: 50, expected_fp: 30 }] });
  assert.equal(p.node("h2h-multi-wrap").hidden, true);
});

test("head-to-head still shows when the tournament portfolio cannot be filled", async () => {
  const p = await page();
  const entry = { ids: [0, 1, 2, 3, 4], superstar: 0, expected_fp: 30, salary: 50 };
  const stats = { expected_wins: 1.2, win_rate: .6, sd_wins: .7, zero_wins: .12,
    all_wins: .4, winning_record: .4, per_entry: [.6, .6] };
  const mode = { entries: [entry, entry], sharp: stats, public: stats };
  p.workers[0].emit({ type: "error", message: "Built 14 of 20 entries under the construction rules.",
    h2h_anchor: [{ ...entry }],
    h2h_multi: { entries: 2, opponents: { sharp: 100, public: 400 },
      modes: { rotate: mode, repeat: mode, next_best: mode } } });
  assert.match(p.node("result-note").textContent, /Built 14 of 20/);
  assert.equal(p.node("h2h-card").hidden, false);
  assert.equal(p.node("h2h-multi-wrap").hidden, false);
  assert.match(p.node("h2h-multi").innerHTML, /win vs sharp 60\.0%/);
});
