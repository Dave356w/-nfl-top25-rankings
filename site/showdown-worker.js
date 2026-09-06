/* Showdown optimizer: the half of the Colab pipeline that can run in a browser.
 *
 * The Action publishes the model -- means, fitted coefficients of variation and
 * the PSD-repaired latent correlation matrix. Everything below is the part that
 * depends on the questions the Colab notebook used to ask interactively: who is
 * in the pool, what the salary floor is, which positions are capped. None of
 * that changes the model, so none of it needs a feed.
 *
 * Two properties of the model are what make this cheap:
 *
 *  - Excluding a player is a principal submatrix of a PSD matrix, which is still
 *    PSD, so there is no eigensolver and no PSD repair here.
 *  - Because of that the scenarios never have to be redrawn. `simulate` runs
 *    once per pool; `solve` runs on every control change and only re-decides
 *    which lineups are legal.
 *
 * A lineup is five of at most 64 players, so scoring one against a scenario is
 * five multiply-adds -- not the thirty-six a dense (players x candidates)
 * product would do. The scenario array is therefore stored player-major, so
 * those five reads walk five contiguous rows.
 */

"use strict";

const LINEUP_SIZE = 5;

let model = null;     // the published payload
let scenarios = null; // Float32Array, player-major: player * simCount + scenario
let simCount = 0;
let simSeed = 0;
let covariance = null; // Float64Array, n * n

/* ---------- random numbers ---------------------------------------------- */

/* xoshiro128** -- small, fast, and good enough for Monte Carlo. Seeded, so a
 * reload reproduces the same run. It is not NumPy's PCG64, so these numbers
 * will not match the published reference to the last decimal; they agree to
 * within Monte Carlo error, which is the only sense in which either is right. */
function makeRandom(seed) {
  let a = seed >>> 0, b = 0x9e3779b9, c = 0x243f6a88, d = 0xb7e15162;
  for (let i = 0; i < 20; i++) {
    const t = b << 9;
    c ^= a; d ^= b; b ^= c; a ^= d; c ^= t;
    d = (d << 11) | (d >>> 21);
  }
  return function next() {
    const r = Math.imul((b * 5) >>> 0, 7);
    const result = (((r << 7) | (r >>> 25)) >>> 0) * 9;
    const t = b << 9;
    c ^= a; d ^= b; b ^= c; a ^= d; c ^= t;
    d = (d << 11) | (d >>> 21);
    return (result >>> 0) / 4294967296;
  };
}

/* Box-Muller. Returns two standard normals per pair of uniforms. */
function fillNormals(target, random) {
  for (let i = 0; i < target.length; i += 2) {
    let u = random();
    if (u < 1e-12) u = 1e-12;
    const radius = Math.sqrt(-2 * Math.log(u));
    const angle = 2 * Math.PI * random();
    target[i] = radius * Math.cos(angle);
    if (i + 1 < target.length) target[i + 1] = radius * Math.sin(angle);
  }
}

/* ---------- linear algebra ------------------------------------------------ */

/* Rebuild the symmetric matrix the payload stored as a strict upper triangle. */
function latentMatrix(payload) {
  const n = payload.players.length;
  const matrix = new Float64Array(n * n);
  let k = 0;
  for (let i = 0; i < n; i++) {
    matrix[i * n + i] = 1;
    for (let j = i + 1; j < n; j++, k++) {
      const value = payload.latent[k];
      matrix[i * n + j] = value;
      matrix[j * n + i] = value;
    }
  }
  return matrix;
}

/* Lower-triangular Cholesky factor. The published matrix is PSD by
 * construction, but it can be singular to working precision after the PSD
 * repair, so a pivot that goes non-positive is floored rather than throwing. */
function cholesky(matrix, n) {
  const lower = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = matrix[i * n + j];
      for (let k = 0; k < j; k++) sum -= lower[i * n + k] * lower[j * n + k];
      if (i === j) {
        lower[i * n + j] = Math.sqrt(Math.max(sum, 1e-12));
      } else {
        lower[i * n + j] = sum / lower[j * n + j];
      }
    }
  }
  return lower;
}

/* The score-scale correlation implied by the latent matrix, matching
 * `lognormal_score_correlation` in the pipeline. */
function scoreCorrelation(latent, cv, n) {
  const sigma = new Float64Array(n);
  for (let i = 0; i < n; i++) sigma[i] = Math.sqrt(Math.log1p(cv[i] * cv[i]));
  const score = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    score[i * n + i] = 1;
    for (let j = i + 1; j < n; j++) {
      const denominator = cv[i] * cv[j];
      const value = denominator > 0
        ? Math.expm1(latent[i * n + j] * sigma[i] * sigma[j]) / denominator
        : 0;
      const clipped = Math.max(-0.999, Math.min(0.999, value));
      score[i * n + j] = clipped;
      score[j * n + i] = clipped;
    }
  }
  return score;
}

/* ---------- simulation ---------------------------------------------------- */

function simulate(simulations, seed) {
  const players = model.players;
  const n = players.length;
  const cv = Float64Array.from(players, (p) => p.cv);
  const means = Float64Array.from(players, (p) => p.fp);
  const latent = latentMatrix(model);
  const root = cholesky(latent, n);

  const sigma = new Float64Array(n);
  for (let i = 0; i < n; i++) sigma[i] = Math.sqrt(Math.log1p(cv[i] * cv[i]));

  const random = makeRandom(seed);
  const draws = new Float64Array(n);
  const out = new Float32Array(n * simulations);
  for (let s = 0; s < simulations; s++) {
    fillNormals(draws, random);
    for (let i = 0; i < n; i++) {
      let correlated = 0;
      const row = i * n;
      for (let k = 0; k <= i; k++) correlated += root[row + k] * draws[k];
      // Mean-preserving lognormal: E[exp(sigma*Z - sigma^2/2)] = 1.
      out[i * simulations + s] =
        means[i] * Math.exp(correlated * sigma[i] - 0.5 * sigma[i] * sigma[i]);
    }
  }

  const score = scoreCorrelation(latent, cv, n);
  covariance = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      covariance[i * n + j] = means[i] * cv[i] * means[j] * cv[j] * score[i * n + j];
    }
  }
  scenarios = out;
  simCount = simulations;
  simSeed = seed;
}

/* ---------- enumeration --------------------------------------------------- */

/* Every Yahoo-valid roster of five from the included players.
 *
 * Yahoo's single-game rules: five players inside the cap, and at least one
 * non-defense player from each of the two teams. The salary floor is a strategy
 * filter rather than a rule, which is why it is a control on the page. */
function enumerate(included, options) {
  const players = model.players;
  const cap = options.salaryCap;
  const floor = cap * options.minSalaryPct;
  const limits = options.positionLimits || {};
  // Taken from the pool rather than from the payload's away/home labels: the
  // rule is about the two teams actually priced in this game, and the pool is
  // the only thing the enumeration can be wrong about.
  const teams = [];
  for (const index of included) {
    const team = players[index].team;
    if (teams.indexOf(team) < 0) teams.push(team);
  }
  if (teams.length !== 2) {
    throw new Error(
      `Expected exactly two teams in the pool, found ${teams.length || 0}.`
    );
  }
  const size = included.length;

  const salary = Float64Array.from(included, (i) => players[i].salary);
  const teamA = Uint8Array.from(included, (i) => (players[i].team === teams[0] && players[i].pos !== "DEF") ? 1 : 0);
  const teamB = Uint8Array.from(included, (i) => (players[i].team === teams[1] && players[i].pos !== "DEF") ? 1 : 0);
  const limitKeys = Object.keys(limits);
  const limitFlags = limitKeys.map((position) =>
    Uint8Array.from(included, (i) => (players[i].pos === position ? 1 : 0))
  );

  const rosters = [];
  const salaries = [];
  const pick = new Int32Array(LINEUP_SIZE);

  for (let a = 0; a <= size - 5; a++) {
    const sa = salary[a];
    if (sa > cap) continue;
    for (let b = a + 1; b <= size - 4; b++) {
      const sb = sa + salary[b];
      if (sb > cap) continue;
      for (let c = b + 1; c <= size - 3; c++) {
        const sc = sb + salary[c];
        if (sc > cap) continue;
        for (let d = c + 1; d <= size - 2; d++) {
          const sd = sc + salary[d];
          if (sd > cap) continue;
          for (let e = d + 1; e < size; e++) {
            const total = sd + salary[e];
            if (total > cap || total < floor) continue;
            if (!(teamA[a] | teamA[b] | teamA[c] | teamA[d] | teamA[e])) continue;
            if (!(teamB[a] | teamB[b] | teamB[c] | teamB[d] | teamB[e])) continue;
            let ok = true;
            for (let f = 0; f < limitKeys.length && ok; f++) {
              const flag = limitFlags[f];
              const count = flag[a] + flag[b] + flag[c] + flag[d] + flag[e];
              const [low, high] = limits[limitKeys[f]];
              if (count < low || count > high) ok = false;
            }
            if (!ok) continue;
            pick[0] = included[a]; pick[1] = included[b]; pick[2] = included[c];
            pick[3] = included[d]; pick[4] = included[e];
            rosters.push(pick[0], pick[1], pick[2], pick[3], pick[4]);
            salaries.push(total);
          }
        }
      }
    }
  }
  return { ids: Int32Array.from(rosters), salary: Float64Array.from(salaries) };
}

/* ---------- candidate screen ---------------------------------------------- */

/* Rank every (roster, Superstar) pair by the analytic ceiling and keep the
 * strongest, exactly as `enumerate_candidate_lineups` does. With
 * w = 1 + 0.5 * e_s the variance is sum(C) + rowsum_s(C) + 0.25 * C_ss over the
 * roster's own 5x5 submatrix, which avoids building a quadratic form per pair. */
function screen(rosters, options) {
  const players = model.players;
  const n = players.length;
  const count = rosters.salary.length;
  const expected = new Float64Array(count * LINEUP_SIZE);
  const ceiling = new Float64Array(count * LINEUP_SIZE);
  const rowsum = new Float64Array(LINEUP_SIZE);

  for (let r = 0; r < count; r++) {
    const base = r * LINEUP_SIZE;
    let projection = 0;
    let total = 0;
    for (let i = 0; i < LINEUP_SIZE; i++) {
      const idI = rosters.ids[base + i];
      projection += players[idI].fp;
      let sum = 0;
      for (let j = 0; j < LINEUP_SIZE; j++) sum += covariance[idI * n + rosters.ids[base + j]];
      rowsum[i] = sum;
      total += sum;
    }
    for (let s = 0; s < LINEUP_SIZE; s++) {
      const idS = rosters.ids[base + s];
      const variance = Math.max(total + rowsum[s] + 0.25 * covariance[idS * n + idS], 0);
      const mean = projection + 0.5 * players[idS].fp;
      expected[base + s] = mean;
      ceiling[base + s] = mean + options.ceilingWeight * Math.sqrt(variance);
    }
  }

  const population = count * LINEUP_SIZE;
  const keep = new Set();
  topInto(keep, ceiling, Math.min(options.maxCandidates, population));
  topInto(keep, expected, Math.min(options.meanReserve, population));
  const selected = Int32Array.from(keep).sort();

  return { selected, expected, ceiling };
}

/* Add the indices of the `count` largest values to `into`.
 *
 * Quickselect the cut-off, then take everything strictly above it and top up
 * from the ties. `np.argpartition` resolves ties by position too, so the two
 * implementations keep the same candidate set. */
function topInto(into, values, count) {
  if (count <= 0) return;
  const copy = Float64Array.from(values);
  const threshold = quickselect(copy, copy.length - count);
  let added = 0;
  for (let i = 0; i < values.length && added < count; i++) {
    if (values[i] > threshold) { into.add(i); added++; }
  }
  for (let i = 0; i < values.length && added < count; i++) {
    if (values[i] === threshold && !into.has(i)) { into.add(i); added++; }
  }
}

/* In-place nth-element. Returns the value that would sit at `k` if sorted. */
function quickselect(values, k, left, right) {
  left = left === undefined ? 0 : left;
  right = right === undefined ? values.length - 1 : right;
  while (left < right) {
    const pivot = values[(left + right) >> 1];
    let i = left, j = right;
    while (i <= j) {
      while (values[i] < pivot) i++;
      while (values[j] > pivot) j--;
      if (i <= j) {
        const swap = values[i]; values[i] = values[j]; values[j] = swap;
        i++; j--;
      }
    }
    if (k <= j) right = j;
    else if (k >= i) left = i;
    else break;
  }
  return values[k];
}

/* ---------- scoring ------------------------------------------------------- */

/* Write one candidate's score in every scenario into `target`. */
function scoreInto(target, ids, base, superstar) {
  const sims = simCount;
  const i0 = ids[base] * sims, i1 = ids[base + 1] * sims, i2 = ids[base + 2] * sims;
  const i3 = ids[base + 3] * sims, i4 = ids[base + 4] * sims;
  const bonus = superstar * sims;
  for (let s = 0; s < sims; s++) {
    target[s] = scenarios[i0 + s] + scenarios[i1 + s] + scenarios[i2 + s]
              + scenarios[i3 + s] + scenarios[i4 + s] + 0.5 * scenarios[bonus + s];
  }
}

function quantileIndex(fraction) {
  return Math.min(simCount - 1, Math.max(0, Math.round(fraction * (simCount - 1))));
}

function score(rosters, screened, options, progress) {
  const sims = simCount;
  const selected = screened.selected;
  const total = selected.length;
  const ids = new Int32Array(total * LINEUP_SIZE);
  const superstars = new Int32Array(total);
  const salary = new Float64Array(total);
  const expected = new Float64Array(total);

  for (let c = 0; c < total; c++) {
    const flat = selected[c];
    const roster = (flat / LINEUP_SIZE) | 0;
    const slot = flat % LINEUP_SIZE;
    for (let i = 0; i < LINEUP_SIZE; i++) ids[c * LINEUP_SIZE + i] = rosters.ids[roster * LINEUP_SIZE + i];
    superstars[c] = rosters.ids[roster * LINEUP_SIZE + slot];
    salary[c] = rosters.salary[roster];
    expected[c] = screened.expected[flat];
  }

  const mean = new Float64Array(total);
  const sd = new Float64Array(total);
  const p25 = new Float64Array(total);
  const p90 = new Float64Array(total);
  const p95 = new Float64Array(total);
  const nearRate = new Float64Array(total);
  const winRate = new Float64Array(total);
  const best = new Float32Array(sims).fill(-Infinity);

  const buffer = new Float64Array(sims);
  const sorting = new Float64Array(sims);
  const k25 = quantileIndex(0.25), k90 = quantileIndex(0.90), k95 = quantileIndex(0.95);

  for (let c = 0; c < total; c++) {
    scoreInto(buffer, ids, c * LINEUP_SIZE, superstars[c]);
    let sum = 0, sumSquares = 0;
    for (let s = 0; s < sims; s++) {
      const value = buffer[s];
      sum += value;
      sumSquares += value * value;
      if (value > best[s]) best[s] = value;
    }
    const average = sum / sims;
    mean[c] = average;
    sd[c] = Math.sqrt(Math.max(sumSquares / sims - average * average, 0));

    sorting.set(buffer);
    // Three order statistics from one progressively narrowed partition.
    p25[c] = quickselect(sorting, k25, 0, sims - 1);
    p90[c] = quickselect(sorting, k90, k25, sims - 1);
    p95[c] = quickselect(sorting, k95, k90, sims - 1);

    if ((c & 1023) === 0 && progress) progress(0.5 * (c / total));
  }

  for (let c = 0; c < total; c++) {
    scoreInto(buffer, ids, c * LINEUP_SIZE, superstars[c]);
    let near = 0, won = 0;
    for (let s = 0; s < sims; s++) {
      const value = buffer[s];
      const reference = best[s];
      if (value >= reference * options.nearOptimalRatio) near++;
      // Matches np.isclose(rtol=1e-6, atol=1e-5) on the same comparison.
      if (Math.abs(value - reference) <= 1e-5 + 1e-6 * Math.abs(reference)) won++;
    }
    nearRate[c] = near / sims;
    winRate[c] = won / sims;
    if ((c & 1023) === 0 && progress) progress(0.5 + 0.5 * (c / total));
  }

  const tournament = tournamentScore(p90, nearRate, mean, sd);
  return { total, ids, superstars, salary, expected, mean, sd, p25, p90, p95,
           nearRate, winRate, tournament };
}

function zscore(values) {
  const n = values.length;
  let sum = 0;
  for (let i = 0; i < n; i++) sum += values[i];
  const average = sum / n;
  let variance = 0;
  for (let i = 0; i < n; i++) variance += (values[i] - average) ** 2;
  const scale = Math.sqrt(variance / n);
  const out = new Float64Array(n);
  if (scale > 1e-12) for (let i = 0; i < n; i++) out[i] = (values[i] - average) / scale;
  return out;
}

/* The same 0.40 / 0.25 / 0.25 / 0.10 blend the pipeline ranks on. */
function tournamentScore(p90, near, mean, sd) {
  const a = zscore(p90), b = zscore(near), c = zscore(mean), d = zscore(sd);
  const out = new Float64Array(p90.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = 0.40 * a[i] + 0.25 * b[i] + 0.25 * c[i] + 0.10 * d[i];
  }
  return out;
}

/* ---------- selection ----------------------------------------------------- */

function orderBy(scored, objective) {
  const index = Array.from({ length: scored.total }, (_, i) => i);
  const key = {
    tournament: scored.tournament,
    mean: scored.mean,
    floor: scored.p25,
    ceiling: scored.p90,
    expected: scored.expected,
  }[objective] || scored.tournament;
  index.sort((a, b) => (key[b] - key[a]) || (a - b));
  return index;
}

/* Exposure and overlap caps, mirroring `select_tournament_portfolio`. */
function portfolio(scored, order, options) {
  const target = options.entries;
  const maxPlayer = Math.max(1, Math.ceil(target * options.maxPlayerExposure));
  const maxSuperstar = Math.max(1, Math.ceil(target * options.maxSuperstarExposure));
  const playerCounts = new Map();
  const superstarCounts = new Map();
  const chosen = [];
  const sets = [];

  for (const candidate of order) {
    const base = candidate * LINEUP_SIZE;
    const members = [];
    for (let i = 0; i < LINEUP_SIZE; i++) members.push(scored.ids[base + i]);
    if (members.some((id) => (playerCounts.get(id) || 0) >= maxPlayer)) continue;
    const superstar = scored.superstars[candidate];
    if ((superstarCounts.get(superstar) || 0) >= maxSuperstar) continue;
    const set = new Set(members);
    let overlaps = false;
    for (const prior of sets) {
      let shared = 0;
      for (const id of set) if (prior.has(id)) shared++;
      if (shared > options.maxShared) { overlaps = true; break; }
    }
    if (overlaps) continue;
    chosen.push(candidate);
    sets.push(set);
    for (const id of members) playerCounts.set(id, (playerCounts.get(id) || 0) + 1);
    superstarCounts.set(superstar, (superstarCounts.get(superstar) || 0) + 1);
    if (chosen.length === target) break;
  }
  return { chosen, sets };
}

function describe(scored, indices) {
  return indices.map((c) => {
    const base = c * LINEUP_SIZE;
    const ids = [];
    for (let i = 0; i < LINEUP_SIZE; i++) ids.push(scored.ids[base + i]);
    return {
      ids,
      superstar: scored.superstars[c],
      salary: scored.salary[c],
      expected_fp: scored.expected[c],
      sim_mean: scored.mean[c],
      sim_sd: scored.sd[c],
      floor_p25: scored.p25[c],
      ceiling_p90: scored.p90[c],
      ceiling_p95: scored.p95[c],
      near_optimal_rate: scored.nearRate[c],
      win_rate: scored.winRate[c],
      tournament_score: scored.tournament[c],
    };
  });
}

function diversity(sets, maxShared) {
  if (sets.length < 2) return null;
  const overlaps = [];
  for (let i = 0; i < sets.length; i++) {
    for (let j = i + 1; j < sets.length; j++) {
      let shared = 0;
      for (const id of sets[i]) if (sets[j].has(id)) shared++;
      overlaps.push(shared);
    }
  }
  const distinct = new Set();
  for (const set of sets) for (const id of set) distinct.add(id);
  const max = Math.max(...overlaps);
  return {
    lineups: sets.length,
    distinct_players: distinct.size,
    mean_shared: overlaps.reduce((a, b) => a + b, 0) / overlaps.length,
    max_shared: max,
    cap: maxShared,
    at_cap: overlaps.filter((value) => value === maxShared).length,
    near_duplicates: overlaps.filter((value) => value === LINEUP_SIZE - 1).length,
  };
}

/* ---------- message handling ---------------------------------------------- */

/* Everything above is plain functions over typed arrays so the same code can be
 * driven from Node by tests/showdown_reference.js, which checks it against the
 * Python pipeline's own answer for the same published model. */
function setModel(payload) {
  model = payload;
  scenarios = null;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    LINEUP_SIZE, setModel, simulate, enumerate, screen, score, orderBy,
    portfolio, describe, diversity, latentMatrix, cholesky, scoreCorrelation,
    covarianceMatrix: () => covariance,
  };
}

if (typeof self === "undefined") {
  // Running under Node for the equivalence test; there is no message port.
} else {
self.onmessage = (event) => {
  const message = event.data;
  try {
    if (message.type === "load") {
      model = message.payload;
      scenarios = null;
      self.postMessage({ type: "loaded", players: model.players.length });
      return;
    }
    if (message.type !== "solve") return;

    const options = message.options;
    const started = performance.now();
    if (!scenarios || simCount !== options.simulations || simSeed !== options.seed) {
      self.postMessage({ type: "stage", stage: "Drawing scenarios" });
      simulate(options.simulations, options.seed);
    }
    const simulated = performance.now();

    self.postMessage({ type: "stage", stage: "Enumerating rosters" });
    const included = Int32Array.from(message.included);
    if (included.length < LINEUP_SIZE) {
      throw new Error(`Only ${included.length} players are in the pool; five are required.`);
    }
    const rosters = enumerate(included, options);
    if (rosters.salary.length === 0) {
      throw new Error(
        "No valid roster fits these settings. Lower the salary floor, put a " +
        "player back, or relax a position limit."
      );
    }

    self.postMessage({ type: "stage", stage: "Screening candidates" });
    const screened = screen(rosters, options);

    self.postMessage({ type: "stage", stage: "Scoring against scenarios" });
    const scored = score(rosters, screened, options, (fraction) => {
      self.postMessage({ type: "progress", fraction });
    });

    const order = orderBy(scored, options.objective);
    const built = portfolio(scored, order, options);
    self.postMessage({
      type: "result",
      valid_rosters: rosters.salary.length,
      candidates_scored: scored.total,
      portfolio: describe(scored, built.chosen),
      strongest: describe(scored, order.slice(0, 10)),
      diversity: diversity(built.sets, options.maxShared),
      timing: {
        simulate: Math.round(simulated - started),
        total: Math.round(performance.now() - started),
      },
    });
  } catch (error) {
    self.postMessage({ type: "error", message: String(error && error.message || error) });
  }
};
}
