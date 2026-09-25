/* Exercise the shipped Showdown page's handlers with controlled fetches and workers.
 *
 * The page script runs in a VM against a minimal fake DOM: every element is a
 * plain object keyed by id, fetches resolve from an in-memory route table, and
 * workers record the messages the page posts to them. The worker's own maths
 * is covered by showdown_strategy.test.js; this file covers what the page asks
 * for and what it shows.
 */
"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");
const { test } = require("node:test");

const SITE = path.join(__dirname, "..", "site");
const html = fs.readFileSync(path.join(SITE, "showdown.html"), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
const depthModelJs = fs.readFileSync(path.join(SITE, "depth-model.js"), "utf8");
const depthModelJson = JSON.parse(fs.readFileSync(path.join(SITE, "data", "depth_model.json"), "utf8"));
const flush = () => new Promise(setImmediate);
const REGRESSION = "Yahoo salary-position-depth regression (trained through 2025)";

function payload(id, extra = {}) {
  return {
    schema: 3, game_id: id, matchup: `${id} game`, generated_utc: "2026-09-09T21:00:00Z",
    snapshot_id: "snap", salary_cap: 100, kickoff_utc: "2026-09-10T00:00:00Z",
    players: [
      { name: "QB A", pos: "QB", team: "A", salary: 30, fp: 20, depth: 1, source: REGRESSION },
      { name: "WR A1", pos: "WR", team: "A", salary: 20, fp: 12, depth: 1, source: REGRESSION },
      { name: "WR A2", pos: "WR", team: "A", salary: 12, fp: 6, depth: 2, source: REGRESSION },
      { name: "QB B", pos: "QB", team: "B", salary: 28, fp: 18, depth: 1, source: REGRESSION },
      { name: "RB B", pos: "RB", team: "B", salary: 22, fp: 13, depth: 1, source: REGRESSION },
      { name: "DEF B", pos: "DEF", team: "B", salary: 10, fp: 6, depth: 1, source: REGRESSION },
    ],
    settings: { max_player_exposure: 0.5, max_superstar_exposure: 0.35, tournament_lineups: 20 },
    reference: { portfolio: [{ ids: [0, 1, 3, 4, 5], superstar: 0 }] },
    ...extra,
  };
}

function makeNode(id) {
  const classes = new Set();
  return {
    id, value: "", innerHTML: "", textContent: "", hidden: false, disabled: false, max: "",
    checked: false, className: "", style: {}, dataset: {}, handlers: {},
    classList: { contains: (c) => classes.has(c), add: (c) => classes.add(c) },
    addEventListener(event, fn) { this.handlers[event] = fn; },
  };
}

async function page({ routes: extraRoutes = {}, depthModel = false } = {}) {
  const nodes = {}, workers = [];
  const node = (id) => (nodes[id] ||= makeNode(id));
  node("detail").value = "standard";
  const radios = ["h2h", "tournament"].map((value) => {
    const radio = node("contest-" + value);
    radio.value = value;
    radio.checked = value === "h2h";
    return radio;
  });
  const index = { snapshot_id: "snap", games: ["A", "B", "C"].map((id) => ({ matchup: id, file: `${id}.json` })) };
  const routes = new Map(Object.entries({
    "index.json": () => index,
    "A.json": () => payload("A"), "B.json": () => payload("B"), "C.json": () => payload("C"),
    "depth_model.json": () => (depthModel ? depthModelJson : Promise.reject(new Error("absent"))),
    ...extraRoutes,
  }));
  class Worker {
    constructor(url) { this.url = url; this.messages = []; this.terminated = false; workers.push(this); }
    postMessage(message) { this.messages.push(message); }
    terminate() { this.terminated = true; }
    emit(data) { this.onmessage({ data }); }
  }
  const store = {};
  const context = {
    Set, Worker, Object, Array, Number, String, Math, Date, isNaN, isFinite, encodeURIComponent, JSON,
    localStorage: { getItem: (k) => store[k] ?? null, setItem: (k, v) => { store[k] = v; }, removeItem: (k) => { delete store[k]; } },
    window: { matchMedia: () => ({ matches: false }) },
    document: {
      getElementById: node,
      documentElement: { dataset: {} },
      querySelectorAll: (selector) => (selector === 'input[name="contest"]' ? radios : []),
    },
    fetch: async (url) => {
      const name = url.split("/").pop();
      if (!routes.has(name)) return { ok: false, status: 404, json: async () => ({}) };
      const result = await routes.get(name)();
      return { ok: true, json: async () => result };
    },
  };
  vm.createContext(context);
  vm.runInContext(depthModelJs, context);
  vm.runInContext(script, context);
  await flush(); await flush();
  return {
    node, workers, routes, store,
    worker: () => workers.at(-1),
    choose(id) { node("game").value = `${id}.json`; node("game").handlers.change(); },
    contest(value) {
      radios.forEach((r) => { r.checked = r.value === value; });
      node("contest-" + value).handlers.change();
    },
    run() { node("run").handlers.click(); return workers.at(-1).messages.at(-1); },
    toggle(index, checked) {
      node("pool").handlers.change({ target: { checked, dataset: { index: String(index) }, classList: { contains: () => false } } });
    },
  };
}

const h2hResult = (extra = {}) => ({
  contest: "h2h", requested: 3, filled: 3, opponents: 100, valid_rosters: 10, elapsed_ms: 1200,
  expected_wins: 1.6, win_rate: 0.533, zero_wins: 0.09, winning_record: 0.58,
  limits: { player: 2, superstar: 1 },
  entries: [
    { ids: [0, 1, 3, 4, 5], superstar: 0, expected_fp: 79, salary: 100, win_chance: 0.62 },
    { ids: [0, 1, 3, 4, 5], superstar: 3, expected_fp: 78, salary: 100, win_chance: 0.55 },
    { ids: [0, 2, 3, 4, 5], superstar: 4, expected_fp: 70, salary: 92, win_chance: 0.43 },
  ],
  ...extra,
});

const tournamentResult = (extra = {}) => ({
  contest: "tournament", requested: 20, filled: 2, valid_rosters: 10, elapsed_ms: 2500,
  expected_fp: 75.5, best_score: 88.1, gain_over_single: 9.4,
  diversity: { max_shared: 4 }, limits: { player: 15, superstar: 5 },
  entries: [
    { ids: [0, 1, 3, 4, 5], superstar: 0, expected_fp: 79, salary: 100, floor_p25: 60, ceiling_p90: 104 },
    { ids: [0, 2, 3, 4, 5], superstar: 3, expected_fp: 72, salary: 92, floor_p25: 55, ceiling_p90: 97 },
  ],
  ...extra,
});

test("the worker URL is versioned and an out-of-date worker is refused", async () => {
  const p = await page();
  assert.match(p.worker().url, /showdown-worker\.js\?protocol=6&snapshot=snap$/);
  assert.equal(p.worker().messages[0].type, "load");
  p.worker().emit({ type: "loaded", players: 6, protocol: 4 });
  assert.equal(p.workers[0].terminated, true);
  assert.equal(p.node("run").disabled, true);
  assert.match(p.node("status-text").innerHTML, /out of date/);
});

test("head-to-head is the default and sends only what the visitor chose", async () => {
  const p = await page();
  assert.equal(p.node("entries").value, 3);
  assert.equal(p.node("player-exposure").value, 0.75);
  assert.equal(p.node("superstar-exposure").value, 0.25);
  assert.equal(p.node("run").disabled, false);
  const message = p.run();
  assert.equal(message.type, "solve");
  assert.deepEqual(JSON.parse(JSON.stringify(message.request)), {
    contest: "h2h", entries: 3, included: [0, 1, 2, 3, 4, 5], salaryCap: 100,
    detail: "standard", maxPlayerExposure: 0.75, maxSuperstarExposure: 0.25,
  });
  assert.equal(p.node("run").disabled, true);
});

test("each contest keeps its own entry count; both share the exposure defaults", async () => {
  const p = await page();
  p.node("entries").value = "7";
  p.node("entries").handlers.change();
  p.contest("tournament");
  assert.equal(p.node("entries").value, 20);
  // The published run's own .5 / .35 settings do not apply to the lab.
  assert.equal(p.node("player-exposure").value, 0.75);
  assert.equal(p.node("superstar-exposure").value, 0.25);
  assert.match(p.node("results-title").textContent, /Tournament/);
  assert.equal(p.run().request.contest, "tournament");
  p.contest("h2h");
  assert.equal(p.node("entries").value, 7);
  p.node("entries").value = "99";
  p.node("entries").handlers.change();
  assert.equal(p.node("entries").value, 50);        // head-to-head tops out at 50
});

test("switching game during a solve cancels it and ignores the old worker", async () => {
  const p = await page();
  p.run();
  const old = p.worker();
  p.choose("B");
  assert.equal(old.terminated, true);
  await flush();
  assert.equal(p.node("run").disabled, false);
  old.emit({ type: "error", message: "obsolete worker" });
  assert.doesNotMatch(p.node("status-text").innerHTML, /obsolete/);
  assert.equal(p.worker().messages[0].payload.game_id, "B");
});

test("the latest game selection wins when responses arrive out of order", async () => {
  let releaseB, releaseC;
  const p = await page({ routes: {
    "B.json": () => new Promise((resolve) => { releaseB = resolve; }),
    "C.json": () => new Promise((resolve) => { releaseC = resolve; }),
  } });
  p.choose("B"); p.choose("C");
  releaseC(payload("C")); await flush();
  releaseB(payload("B")); await flush();
  assert.equal(p.worker().messages[0].payload.game_id, "C");
});

test("a failed or mismatched game cannot be solved, and a later one recovers", async () => {
  const p = await page({ routes: {
    "B.json": () => Promise.reject(new Error("unavailable")),
    "C.json": () => payload("C", { snapshot_id: "older" }),
  } });
  p.choose("B"); await flush();
  assert.equal(p.node("run").disabled, true);
  assert.match(p.node("status-text").innerHTML, /unavailable/);
  p.choose("C"); await flush();
  assert.equal(p.node("run").disabled, true);
  assert.match(p.node("status-text").innerHTML, /different snapshot/);
  p.choose("A"); await flush();
  assert.equal(p.node("run").disabled, false);
});

test("a game without a published cap is labelled and waits for one, per game", async () => {
  const p = await page({ routes: {
    "B.json": () => payload("B", { salary_cap: null }),
    "index.json": () => ({ snapshot_id: "snap", games: [
      { matchup: "A", file: "A.json" }, { matchup: "B", file: "B.json", needs_salary_cap: true },
    ] }),
  } });
  assert.match(p.node("game").innerHTML, /B — needs a salary cap/);
  assert.doesNotMatch(p.node("game").innerHTML, /A — needs/);
  p.choose("B"); await flush();
  assert.equal(p.node("cap-field").hidden, false);
  assert.equal(p.node("run").disabled, true);
  p.node("cap").value = "0";
  p.node("cap").handlers.input();
  assert.equal(p.node("run").disabled, true);
  p.node("cap").value = "120";
  p.node("cap").handlers.input();
  assert.equal(p.node("run").disabled, false);
  assert.equal(p.run().request.salaryCap, 120);
  p.choose("A"); await flush();
  assert.equal(p.node("cap-field").hidden, true);
  p.choose("B"); await flush();
  assert.equal(p.node("cap").value, "120");          // remembered for B only
});

test("left-out players are not sent, and fewer than five cannot be solved", async () => {
  const p = await page();
  p.toggle(2, false);
  assert.match(p.node("pool-count").textContent, /5 of 6/);
  assert.deepEqual(Array.from(p.run().request.included), [0, 1, 3, 4, 5]);
  p.toggle(5, false);
  assert.equal(p.node("run").disabled, true);
  p.node("all").handlers.click();
  assert.equal(p.node("run").disabled, false);
  p.node("none").handlers.click();
  assert.equal(p.node("run").disabled, true);
});

test("a head-to-head result shows the summary, the limits and each entry", async () => {
  const p = await page();
  p.run();
  p.worker().emit({ type: "result", result: h2hResult() });
  assert.equal(p.node("empty").hidden, true);
  assert.match(p.node("stats").innerHTML, /1\.6 of 3/);
  assert.match(p.node("stats").innerHTML, /53\.3%/);
  assert.match(p.node("stats").innerHTML, /9\.0%/);
  assert.match(p.node("explain").textContent, /No player is in more than 2 of 3 entries and no Superstar in more than 1/);
  assert.match(p.node("explain").textContent, /never repeat the same lineup and Superstar/);
  assert.match(p.node("entries-list").innerHTML, /★ QB A/);
  assert.match(p.node("entries-list").innerHTML, /62\.0% win/);
  assert.doesNotMatch(p.node("entries-list").innerHTML, /round|fillers/);
  assert.equal(p.node("result-notice").hidden, true);
  assert.match(p.node("status-text").innerHTML, /Built 3 lineups in 1\.2s/);
  assert.equal(p.node("run").disabled, false);
});

test("a short list says how many entries fitted", async () => {
  const p = await page();
  p.contest("tournament");
  p.run();
  p.worker().emit({ type: "result", result: tournamentResult() });
  assert.equal(p.node("result-notice").hidden, false);
  assert.match(p.node("result-notice").textContent, /Only 2 of 20 entries fit the exposure limits\./);
  assert.match(p.node("stats").innerHTML, /88\.1/);
  assert.match(p.node("explain").textContent, /\+9\.4 points/);
  assert.match(p.node("explain").textContent, /No player is in more than 15 of 20 entries and no Superstar in more than 5/);
  assert.match(p.node("entries-list").innerHTML, /range 60–104/);
  // The first entry is also in the published reference portfolio; the second is not.
  assert.equal((p.node("entries-list").innerHTML.match(/also published/g) || []).length, 1);
});

test("errors are shown, and changing a setting clears old results", async () => {
  const p = await page();
  p.run();
  p.worker().emit({ type: "error", message: "No valid roster fits the salary cap." });
  assert.equal(p.node("result-notice").hidden, false);
  assert.match(p.node("result-notice").textContent, /No valid roster/);
  assert.equal(p.node("run").disabled, false);
  p.run();
  p.worker().emit({ type: "result", result: h2hResult() });
  p.node("detail").value = "quick";
  p.node("detail").handlers.change();
  assert.equal(p.node("entries-list").innerHTML, "");
  assert.equal(p.node("empty").hidden, false);
});

test("changing a setting mid-solve restarts the worker", async () => {
  const p = await page();
  p.run();
  const running = p.worker();
  p.node("superstar-exposure").value = "0.5";
  p.node("superstar-exposure").handlers.change();
  assert.equal(running.terminated, true);
  assert.notEqual(p.worker(), running);
  assert.equal(p.node("run").disabled, false);
  assert.equal(p.run().request.maxSuperstarExposure, 0.5);
});

test("a depth edit re-prices the player and reloads the worker", async () => {
  const p = await page({ depthModel: true });
  await flush(); await flush();
  assert.match(p.node("pool").innerHTML, /class="depth"/);
  const before = p.worker();
  p.node("pool").handlers.change({ target: {
    value: "1", dataset: { index: "2" }, classList: { contains: (c) => c === "depth" },
  } });
  assert.equal(before.terminated, true);
  const edited = p.worker().messages[0].payload.players;
  assert.equal(edited[2].depth, 1);
  assert.equal(edited[1].depth, 2);                 // the teammate moves down
  assert.ok(edited[2].fp > 6, "a promoted receiver projects higher");
  assert.equal(p.node("depth-notice").hidden, false);
  p.node("depth-reset").handlers.click();
  assert.equal(p.worker().messages[0].payload.players[2].depth, 2);
  assert.equal(p.node("depth-notice").hidden, true);
});
