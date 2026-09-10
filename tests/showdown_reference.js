/* Drive site/showdown-worker.js from Node and print what it computed.
 *
 * tests/test_showdown.py builds a published model with the Python pipeline,
 * runs this against the same payload, and checks the two agree. The parts that
 * are deterministic -- the valid-roster count, a lineup's analytic expectation
 * and its analytic variance -- must match to floating-point tolerance. The
 * simulated summaries only have to agree within Monte Carlo error, because the
 * browser draws its own scenarios from its own generator.
 *
 *     node tests/showdown_reference.js <payload.json>
 */

"use strict";

const fs = require("fs");
const path = require("path");

if (typeof globalThis.performance === "undefined") {
  globalThis.performance = { now: () => Number(process.hrtime.bigint() / 1000n) / 1000 };
}

const worker = require(path.join(__dirname, "..", "site", "showdown-worker.js"));
const payload = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const simulations = Number(process.argv[3] || payload.settings.simulations);

worker.setModel(payload);

const options = {
  salaryCap: payload.salary_cap,
  minSalaryPct: payload.settings.min_salary_used_pct,
  ceilingWeight: payload.settings.candidate_ceiling_weight,
  maxCandidates: payload.settings.max_candidate_lineups,
  meanReserve: payload.settings.mean_candidate_reserve,
  nearOptimalRatio: payload.settings.near_optimal_ratio,
  entries: payload.settings.tournament_lineups,
  maxPlayerExposure: payload.settings.max_player_exposure,
  maxSuperstarExposure: payload.settings.max_superstar_exposure,
  maxShared: payload.settings.max_shared_players,
  simulations,
  seed: payload.settings.random_seed,
  objective: payload.settings.showdown_objective || "tournament",
  positionLimits: {},
};

worker.simulate(simulations, options.seed);
const included = Array.from(payload.players.keys());
const rosters = worker.enumerate(included, options);
const screened = worker.screen(rosters, options);
const scored = worker.score(rosters, screened, options, null);
const order = worker.orderBy(scored, options.objective);
const built = worker.portfolio(scored, order, options);

// The analytic expectation and variance of one named lineup, so the Python side
// can check the covariance path without depending on either RNG.
const covariance = worker.covarianceMatrix();
const n = payload.players.length;
function analytic(ids, superstar) {
  let mean = 0;
  let total = 0;
  let superstarRow = 0;
  for (const i of ids) {
    mean += payload.players[i].fp;
    for (const j of ids) {
      total += covariance[i * n + j];
      if (i === superstar) superstarRow += covariance[i * n + j];
    }
  }
  mean += 0.5 * payload.players[superstar].fp;
  const variance = total + superstarRow + 0.25 * covariance[superstar * n + superstar];
  return { mean, variance };
}

const probe = payload.reference
  ? payload.reference.portfolio[0]
  : { ids: [0, 1, 2, 3, 4], superstar: 0 };

process.stdout.write(JSON.stringify({
  valid_rosters: rosters.salary.length,
  candidates_scored: scored.total,
  portfolio: worker.describe(scored, built.chosen),
  strongest: worker.describe(scored, order.slice(0, 10)),
  diversity: worker.diversity(built.sets, options.maxShared),
  probe: { ...probe, analytic: analytic(probe.ids, probe.superstar) },
}));

