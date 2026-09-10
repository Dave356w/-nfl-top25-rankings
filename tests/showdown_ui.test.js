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
      max_player_exposure: 0.7, max_superstar_exposure: 0.35,
      min_salary_used_pct: 0.75, simulations: 20000, max_candidate_lineups: 25000 },
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
    constructor() { this.messages = []; this.terminated = false; workers.push(this); }
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
  assert.match(p.node("exposure-hint").textContent, /at most 3 of 5/);
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
