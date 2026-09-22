#!/usr/bin/env node
"use strict";

/* Rebuild a portfolio from an immutable pregame Showdown payload.
 *
 * The request is read from stdin so Python can pass the archived payload without
 * creating a second, mutable copy on disk:
 *
 *   {"payload": {...}, "salary_cap": 123, "simulations": 2000}
 *
 * `simulations` is optional. Production grading omits it and therefore uses the
 * archived value; tests and smoke runs may lower it explicitly.
 */

const path = require("path");
const worker = require(path.join(__dirname, "..", "site", "showdown-worker.js"));

let raw = "";
process.stdin.setEncoding("utf8");
process.stdin.on("data", (chunk) => { raw += chunk; });
process.stdin.on("end", () => {
  try {
    const request = JSON.parse(raw);
    const payload = request.payload;
    const settings = payload.settings || {};
    const salaryCap = Number(request.salary_cap);
    const simulations = request.simulations == null
      ? Number(settings.simulations)
      : Number(request.simulations);
    if (!Number.isFinite(salaryCap) || salaryCap <= 0) {
      throw new Error("A positive completed-slate salary cap is required.");
    }
    if (!Number.isInteger(simulations) || simulations < 100) {
      throw new Error("Simulations must be an integer of at least 100.");
    }

    worker.setModel(payload);
    const requestedObjective = settings.showdown_objective || "auto";
    const objective = requestedObjective === "auto" ? "portfolio" : requestedObjective;
    const options = {
      salaryCap,
      minSalaryPct: Number(settings.min_salary_used_pct),
      ceilingWeight: Number(settings.candidate_ceiling_weight),
      maxCandidates: Number(settings.max_candidate_lineups),
      meanReserve: Number(settings.mean_candidate_reserve),
      nearOptimalRatio: Number(settings.near_optimal_ratio),
      entries: Number(settings.tournament_lineups),
      maxPlayerExposure: Number(settings.max_player_exposure),
      maxSuperstarExposure: Number(settings.max_superstar_exposure),
      maxShared: Number(settings.max_shared_players),
      simulations,
      seed: Number(settings.random_seed),
      objective,
      positionLimits: {},
      constructionRules: settings.use_construction_quotas
        ? (settings.portfolio_construction_rules || [])
        : [],
    };

    worker.simulate(simulations, options.seed);
    const included = Int32Array.from({length: payload.players.length}, (_, i) => i);
    const rosters = worker.enumerate(included, options);
    const screened = worker.screen(rosters, options);
    const scored = worker.score(rosters, screened, options, null);
    const order = worker.orderBy(scored, objective === "portfolio" ? "expected" : objective);
    const built = worker.portfolio(scored, order, options);
    if (built.chosen.length !== options.entries) {
      throw new Error(
        `Built ${built.chosen.length} of ${options.entries} archived entries under the saved limits.`
      );
    }
    process.stdout.write(JSON.stringify({
      portfolio: worker.describe(scored, built.chosen, built.rules),
      scenario_evaluation: built.evaluation || null,
      valid_rosters: rosters.salary.length,
      candidates_scored: scored.total,
      simulations,
      objective,
    }));
  } catch (error) {
    process.stderr.write(String(error && error.stack || error) + "\n");
    process.exitCode = 1;
  }
});
