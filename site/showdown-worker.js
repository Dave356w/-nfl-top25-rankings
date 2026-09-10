/* Browser-side Yahoo Showdown optimizer.
 *
 * The Action publishes player means, fitted CVs, and the repaired latent
 * correlation matrix.  This worker handles roster enumeration, shared-scenario
 * simulation, scoring, and portfolio selection without another feed request.
 */

"use strict";

const LINEUP_SIZE = 5;

let model = null;
let scenarios = null; // Float32Array, player-major
let simCount = 0;
let simSeed = 0;
let covariance = null;

/* ---------- random numbers ---------------------------------------------- */

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

/* ---------- linear algebra ---------------------------------------------- */

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

function cholesky(matrix, n) {
  const lower = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = matrix[i * n + j];
      for (let k = 0; k < j; k++) sum -= lower[i * n + k] * lower[j * n + k];
      if (i === j) lower[i * n + j] = Math.sqrt(Math.max(sum, 1e-12));
      else lower[i * n + j] = sum / lower[j * n + j];
    }
  }
  return lower;
}

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

/* ---------- zero hurdle -------------------------------------------------- */
/* Each marginal is an atom at zero of fitted size `zero` plus a lognormal above
 * it, scaled so the unconditional mean and CV are still the published ones.
 * Mirrors `hurdle_transform` and `hurdle_score_correlation` in
 * pipeline/notebook.py; the coefficients below are the same ones. */

function conditionalCv(cv, zero) {
  return Math.sqrt(Math.max(cv * cv * (1 - zero) - zero, 1e-12));
}

/* Hart's double-precision normal CDF. */
function normalCdf(value) {
  const magnitude = Math.abs(value);
  const density = Math.exp(-0.5 * magnitude * magnitude);
  let tail;
  if (magnitude < 7.07106781186547) {
    const num = (((((3.52624965998911e-02 * magnitude + 0.700383064443688)
      * magnitude + 6.37396220353165) * magnitude + 33.912866078383)
      * magnitude + 112.079291497871) * magnitude + 221.213596169931)
      * magnitude + 220.206867912376;
    const den = ((((((8.83883476483184e-02 * magnitude + 1.75566716318264)
      * magnitude + 16.064177579207) * magnitude + 86.7807322029461)
      * magnitude + 296.564248779674) * magnitude + 637.333633378831)
      * magnitude + 793.826512519948) * magnitude + 440.413735824752;
    tail = density * num / den;
  } else {
    const cont = magnitude + 1 / (magnitude + 2 / (magnitude + 3 / (magnitude + 4 / (magnitude + 0.65))));
    tail = density / cont / 2.506628274631;
  }
  return value > 0 ? 1 - tail : tail;
}

/* Acklam's inverse normal CDF. */
const PPF_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
  1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
const PPF_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
  6.680131188771972e+01, -1.328068155288572e+01];
const PPF_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
  -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
const PPF_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
  3.754408661907416e+00];
const PPF_SPLIT = 0.02425;

function normalPpf(probability) {
  const p = Math.min(Math.max(probability, 1e-300), 1 - 1e-16);
  let q, r;
  if (p < PPF_SPLIT || p > 1 - PPF_SPLIT) {
    const lower = p < PPF_SPLIT;
    q = Math.sqrt(-2 * Math.log(lower ? p : 1 - p));
    const top = ((((PPF_C[0] * q + PPF_C[1]) * q + PPF_C[2]) * q + PPF_C[3]) * q + PPF_C[4]) * q + PPF_C[5];
    const bottom = (((PPF_D[0] * q + PPF_D[1]) * q + PPF_D[2]) * q + PPF_D[3]) * q + 1;
    return (lower ? 1 : -1) * top / bottom;
  }
  q = p - 0.5; r = q * q;
  const top = (((((PPF_A[0] * r + PPF_A[1]) * r + PPF_A[2]) * r + PPF_A[3]) * r + PPF_A[4]) * r + PPF_A[5]) * q;
  const bottom = ((((PPF_B[0] * r + PPF_B[1]) * r + PPF_B[2]) * r + PPF_B[3]) * r + PPF_B[4]) * r + 1;
  return top / bottom;
}

const HURDLE_TERMS = 24;
const HURDLE_NODES = 20001;
const HURDLE_LIMIT = 9.0;
const HURDLE_FACTORIAL = (() => {
  const out = new Float64Array(HURDLE_TERMS + 1);
  out[0] = 1;
  for (let k = 1; k <= HURDLE_TERMS; k++) out[k] = out[k - 1] * k;
  return out;
})();
const hurdleCoefficientCache = new Map();

/* Mehler coefficients of one unit-mean marginal, by quadrature on a uniform
 * grid. Cached per distinct (cv, zero): a 36-player pool has about a dozen. */
function hurdleCoefficients(cv, zero) {
  const key = cv + "|" + zero;
  const cached = hurdleCoefficientCache.get(key);
  if (cached) return cached;
  const step = (2 * HURDLE_LIMIT) / (HURDLE_NODES - 1);
  const sigma = Math.sqrt(Math.log1p(conditionalCv(cv, zero) ** 2));
  const plainSigma = Math.sqrt(Math.log1p(cv * cv));
  const weighted = new Float64Array(HURDLE_NODES);
  const grid = new Float64Array(HURDLE_NODES);
  for (let i = 0; i < HURDLE_NODES; i++) {
    const u = -HURDLE_LIMIT + i * step;
    grid[i] = u;
    let value;
    if (zero > 0) {
      const uniform = normalCdf(u);
      if (uniform < zero) value = 0;
      else value = Math.exp(normalPpf((uniform - zero) / (1 - zero)) * sigma - 0.5 * sigma * sigma) / (1 - zero);
    } else {
      value = Math.exp(u * plainSigma - 0.5 * plainSigma * plainSigma);
    }
    weighted[i] = value * Math.exp(-0.5 * u * u) / Math.sqrt(2 * Math.PI);
  }
  const integrate = (poly) => {
    let total = 0;
    for (let i = 0; i < HURDLE_NODES; i++) total += weighted[i] * poly[i];
    return step * (total - 0.5 * (weighted[0] * poly[0]
      + weighted[HURDLE_NODES - 1] * poly[HURDLE_NODES - 1]));
  };
  const out = new Float64Array(HURDLE_TERMS + 1);
  let previous = new Float64Array(HURDLE_NODES).fill(1);
  let current = Float64Array.from(grid);
  out[0] = integrate(previous);
  out[1] = integrate(current);
  for (let k = 1; k < HURDLE_TERMS; k++) {
    const next = new Float64Array(HURDLE_NODES);
    for (let i = 0; i < HURDLE_NODES; i++) next[i] = grid[i] * current[i] - k * previous[i];
    previous = current; current = next;
    out[k + 1] = integrate(current);
  }
  hurdleCoefficientCache.set(key, out);
  return out;
}

function hurdleScoreCorrelation(latent, cv, zero, n) {
  let any = false;
  for (let i = 0; i < n; i++) if (zero[i] > 0) { any = true; break; }
  if (!any) return scoreCorrelation(latent, cv, n);
  const coefficients = [];
  for (let i = 0; i < n; i++) coefficients.push(hurdleCoefficients(cv[i], zero[i]));
  const score = new Float64Array(n * n);
  for (let i = 0; i < n; i++) {
    score[i * n + i] = 1;
    for (let j = i + 1; j < n; j++) {
      const ai = coefficients[i], aj = coefficients[j];
      const rho = latent[i * n + j];
      let total = 0, power = 1;
      for (let k = 1; k <= HURDLE_TERMS; k++) {
        power *= rho;
        total += power * ai[k] * aj[k] / HURDLE_FACTORIAL[k];
      }
      const denominator = cv[i] * cv[j];
      const value = denominator > 0 ? total / denominator : 0;
      const clipped = Math.max(-0.999, Math.min(0.999, value));
      score[i * n + j] = clipped;
      score[j * n + i] = clipped;
    }
  }
  return score;
}

/* ---------- simulation --------------------------------------------------- */

function simulate(simulations, seed) {
  const players = model.players;
  const n = players.length;
  const cv = Float64Array.from(players, (p) => p.cv);
  const means = Float64Array.from(players, (p) => p.fp);
  // An older payload has no fitted zero rate; treating it as 0 reproduces the
  // pure lognormal that payload was built from rather than inventing a floor.
  const zero = Float64Array.from(players, (p) => Number(p.zero) || 0);
  const latent = latentMatrix(model);
  const root = cholesky(latent, n);
  const sigma = new Float64Array(n);
  const scale = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const effective = zero[i] > 0 ? conditionalCv(cv[i], zero[i]) : cv[i];
    sigma[i] = Math.sqrt(Math.log1p(effective * effective));
    scale[i] = means[i] / (1 - zero[i]);
  }

  const random = makeRandom(seed);
  const draws = new Float64Array(n);
  const out = new Float32Array(n * simulations);
  for (let s = 0; s < simulations; s++) {
    fillNormals(draws, random);
    for (let i = 0; i < n; i++) {
      let correlated = 0;
      const row = i * n;
      for (let k = 0; k <= i; k++) correlated += root[row + k] * draws[k];
      const half = 0.5 * sigma[i] * sigma[i];
      if (zero[i] > 0) {
        const uniform = normalCdf(correlated);
        out[i * simulations + s] = uniform < zero[i] ? 0
          : scale[i] * Math.exp(normalPpf((uniform - zero[i]) / (1 - zero[i])) * sigma[i] - half);
      } else {
        out[i * simulations + s] = means[i] * Math.exp(correlated * sigma[i] - half);
      }
    }
  }

  const score = hurdleScoreCorrelation(latent, cv, zero, n);
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

/* ---------- enumeration -------------------------------------------------- */

function enumerate(included, options) {
  const players = model.players;
  const cap = options.salaryCap;
  const floor = cap * options.minSalaryPct;
  const limits = options.positionLimits || {};
  const teams = [];
  for (const index of included) {
    const team = players[index].team;
    if (teams.indexOf(team) < 0) teams.push(team);
  }
  if (teams.length !== 2) {
    throw new Error(`Expected exactly two teams in the pool, found ${teams.length || 0}.`);
  }
  const size = included.length;
  const salary = Float64Array.from(included, (i) => players[i].salary);
  // Yahoo single-game rule: at least one player from each team. Any position
  // counts, a team defense included.
  const teamA = Uint8Array.from(included, (i) => (players[i].team === teams[0] ? 1 : 0));
  const teamB = Uint8Array.from(included, (i) => (players[i].team === teams[1] ? 1 : 0));
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

/* ---------- candidate screen -------------------------------------------- */

function resolvedObjective(options) {
  return options.objective === "auto" ? (options.entries === 1 ? "expected" : "portfolio") : options.objective;
}

function screen(rosters, options) {
  options = {...options, objective: resolvedObjective(options)};
  const players = model.players;
  const n = players.length;
  const count = rosters.salary.length;
  const expected = new Float64Array(count * LINEUP_SIZE);
  const ceiling = new Float64Array(count * LINEUP_SIZE);
  const priority = new Float64Array(count * LINEUP_SIZE);
  const rowsum = new Float64Array(LINEUP_SIZE);

  for (let r = 0; r < count; r++) {
    const base = r * LINEUP_SIZE;
    let projection = 0;
    let total = 0;
    for (let i = 0; i < LINEUP_SIZE; i++) {
      const idI = rosters.ids[base + i];
      projection += players[idI].fp;
      let sum = 0;
      for (let j = 0; j < LINEUP_SIZE; j++) {
        sum += covariance[idI * n + rosters.ids[base + j]];
      }
      rowsum[i] = sum;
      total += sum;
    }
    for (let s = 0; s < LINEUP_SIZE; s++) {
      const idS = rosters.ids[base + s];
      const variance = Math.max(total + rowsum[s] + 0.25 * covariance[idS * n + idS], 0);
      const mean = projection + 0.5 * players[idS].fp;
      expected[base + s] = mean;
      ceiling[base + s] = mean + options.ceilingWeight * Math.sqrt(variance);
      // A sum of correlated lognormals is not itself lognormal. Moment matching
      // is an objective-aware screen only; retained candidates use simulated
      // quantiles below. More candidates reduce screening approximation error.
      priority[base + s] = ceiling[base + s];
      if (options.objective === "expected" || options.objective === "mean") {
        priority[base + s] = mean;
      } else if (options.objective === "floor" || options.objective === "ceiling") {
        const sigma2 = mean > 0 ? Math.log1p(variance / (mean * mean)) : 0;
        const z = options.objective === "floor" ? -0.6744897501960817 : 1.2815515655446004;
        priority[base + s] = mean * Math.exp(z * Math.sqrt(sigma2) - 0.5 * sigma2);
      }
    }
  }

  const population = count * LINEUP_SIZE;
  const keep = new Set();
  topInto(keep, priority, Math.min(options.maxCandidates, population));
  topInto(keep, expected, Math.min(options.meanReserve, population));
  const selected = Int32Array.from(keep).sort();
  return { selected, expected, ceiling };
}

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

/* ---------- scoring ------------------------------------------------------ */

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
    for (let i = 0; i < LINEUP_SIZE; i++) {
      ids[c * LINEUP_SIZE + i] = rosters.ids[roster * LINEUP_SIZE + i];
    }
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
  if (scale > 1e-12) {
    for (let i = 0; i < n; i++) out[i] = (values[i] - average) / scale;
  }
  return out;
}

function tournamentScore(p90, near, mean, sd) {
  const a = zscore(p90), b = zscore(near), c = zscore(mean), d = zscore(sd);
  const out = new Float64Array(p90.length);
  for (let i = 0; i < out.length; i++) {
    out[i] = 0.40 * a[i] + 0.25 * b[i] + 0.25 * c[i] + 0.10 * d[i];
  }
  return out;
}

/* ---------- selection ---------------------------------------------------- */

function objectiveValues(scored, objective) {
  return {
    tournament: scored.tournament,
    mean: scored.mean,
    floor: scored.p25,
    ceiling: scored.p90,
    expected: scored.expected,
  }[objective] || scored.tournament;
}

function orderBy(scored, objective) {
  const index = Array.from({ length: scored.total }, (_, i) => i);
  const key = objectiveValues(scored, objective);
  index.sort((a, b) => (key[b] - key[a]) || (a - b));
  return index;
}

function constructionRules(options) {
  if (Object.prototype.hasOwnProperty.call(options, "constructionRules")) {
    return options.constructionRules || [];
  }
  const settings = (model && model.settings) || {};
  return settings.use_construction_quotas === false ? [] : settings.portfolio_construction_rules || [];
}

function ruleLimits(spec) {
  if (typeof spec === "number") return [spec, spec];
  return [Number(spec[0]), Number(spec[1])];
}

function apportionedRuleCounts(rules, target) {
  if (!rules.length || target <= 0) return [];
  const weights = rules.map((rule) => Math.max(0, Number(rule.count) || 0));
  const total = weights.reduce((a, b) => a + b, 0);
  if (!(total > 0)) return rules.map(() => 0);
  const raw = weights.map((weight) => target * weight / total);
  const counts = raw.map((value) => Math.floor(value));
  let remaining = target - counts.reduce((a, b) => a + b, 0);
  const order = rules.map((_, index) => ({
    index,
    fraction: raw[index] - counts[index],
    weight: weights[index],
  }));
  order.sort((a, b) =>
    (b.fraction - a.fraction) || (b.weight - a.weight) || (a.index - b.index)
  );
  for (let i = 0; i < remaining; i++) counts[order[i].index]++;
  return counts;
}

function constructionSchedule(rules, target) {
  const counts = apportionedRuleCounts(rules, target);
  const remaining = counts.slice();
  const schedule = [];
  while (schedule.length < target && remaining.some((value) => value > 0)) {
    for (let i = 0; i < rules.length && schedule.length < target; i++) {
      if (remaining[i] <= 0) continue;
      schedule.push(rules[i]);
      remaining[i]--;
    }
  }
  return schedule;
}

function matchesConstructionRule(scored, candidate, rule) {
  const base = candidate * LINEUP_SIZE;
  const counts = new Map();
  for (let i = 0; i < LINEUP_SIZE; i++) {
    const position = model.players[scored.ids[base + i]].pos;
    counts.set(position, (counts.get(position) || 0) + 1);
  }
  const limits = rule.positions || {};
  for (const position of Object.keys(limits)) {
    const [low, high] = ruleLimits(limits[position]);
    const value = counts.get(position) || 0;
    if (value < low || value > high) return false;
  }
  return true;
}

function candidateMembers(scored, candidate) {
  const base = candidate * LINEUP_SIZE;
  const members = [];
  for (let i = 0; i < LINEUP_SIZE; i++) members.push(scored.ids[base + i]);
  return members;
}

function overlapsPrior(set, sets, maxShared) {
  for (const prior of sets) {
    let shared = 0;
    for (const id of set) if (prior.has(id)) shared++;
    if (shared > maxShared) return true;
  }
  return false;
}

function buildRuleData(scored, order, rules) {
  return rules.map((rule) => {
    const matching = order.filter((candidate) => matchesConstructionRule(scored, candidate, rule));
    let mandatory = new Set();
    if (matching.length) {
      mandatory = new Set(candidateMembers(scored, matching[0]));
      for (let i = 1; i < matching.length && mandatory.size; i++) {
        const members = new Set(candidateMembers(scored, matching[i]));
        for (const id of Array.from(mandatory)) {
          if (!members.has(id)) mandatory.delete(id);
        }
      }
    }
    return { matching, mandatory };
  });
}

function futureMandatoryDemand(ruleData, remaining) {
  const demand = new Map();
  for (let i = 0; i < remaining.length; i++) {
    if (remaining[i] <= 0) continue;
    for (const id of ruleData[i].mandatory) {
      demand.set(id, (demand.get(id) || 0) + remaining[i]);
    }
  }
  return demand;
}

function exposureLimit(entries, fraction) {
  if (!Number.isFinite(fraction) || fraction < 0 || fraction > 1) {
    throw new Error("Exposure must be a finite fraction between 0 and 1.");
  }
  if (entries <= 0) return 0;
  if (entries === 1) return 1; // There is no multi-entry exposure to diversify.
  return Math.min(entries, Math.floor(entries * fraction + 1e-9));
}

function unrestrictedPortfolio(scored, order, options, target) {
  const maxPlayer = exposureLimit(target, options.maxPlayerExposure);
  const maxSuperstar = exposureLimit(target, options.maxSuperstarExposure);
  const playerCounts = new Map();
  const superstarCounts = new Map();
  const chosen = [];
  const sets = [];
  for (const candidate of order) {
    const members = candidateMembers(scored, candidate);
    if (members.some((id) => (playerCounts.get(id) || 0) >= maxPlayer)) continue;
    const superstar = scored.superstars[candidate];
    if ((superstarCounts.get(superstar) || 0) >= maxSuperstar) continue;
    const set = new Set(members);
    if (overlapsPrior(set, sets, options.maxShared)) continue;
    chosen.push(candidate);
    sets.push(set);
    for (const id of members) playerCounts.set(id, (playerCounts.get(id) || 0) + 1);
    superstarCounts.set(superstar, (superstarCounts.get(superstar) || 0) + 1);
    if (chosen.length === target) break;
  }
  return { chosen, sets, rules: chosen.map(() => null), unfilled: {} };
}

function selectionAttempt(scored, order, options, rules, ruleCounts, ruleData, values, mode) {
  const target = options.entries;
  const maxPlayer = exposureLimit(target, options.maxPlayerExposure);
  const maxSuperstar = exposureLimit(target, options.maxSuperstarExposure);
  const playerCounts = new Map();
  const superstarCounts = new Map();
  const chosen = [];
  const sets = [];
  const assignedRules = [];
  const used = new Set();
  const unfilled = {};
  const remaining = ruleCounts.slice();
  const legacy = constructionSchedule(rules, target).map((rule) => rules.indexOf(rule));
  let legacyCursor = 0;

  while (remaining.reduce((a, b) => a + b, 0) > 0) {
    const active = [];
    for (let i = 0; i < remaining.length; i++) if (remaining[i] > 0) active.push(i);
    let ruleIndex;
    if (mode === "round_robin") {
      while (legacyCursor < legacy.length && remaining[legacy[legacyCursor]] <= 0) legacyCursor++;
      ruleIndex = legacyCursor < legacy.length ? legacy[legacyCursor++] : active[0];
    } else if (mode === "mandatory") {
      active.sort((a, b) =>
        (ruleData[b].mandatory.size - ruleData[a].mandatory.size) ||
        (ruleData[a].matching.length - ruleData[b].matching.length) || (a - b)
      );
      ruleIndex = active[0];
    } else {
      active.sort((a, b) => {
        const scarcityA = ruleData[a].matching.length / Math.max(remaining[a], 1);
        const scarcityB = ruleData[b].matching.length / Math.max(remaining[b], 1);
        return (scarcityA - scarcityB) ||
          (ruleData[b].mandatory.size - ruleData[a].mandatory.size) || (a - b);
      });
      ruleIndex = active[0];
    }

    const rule = rules[ruleIndex];
    remaining[ruleIndex]--;
    const futureDemand = futureMandatoryDemand(ruleData, remaining);
    let picked = null;
    let pickedSet = null;

    for (const candidate of ruleData[ruleIndex].matching) {
      if (used.has(candidate)) continue;
      const members = candidateMembers(scored, candidate);
      if (members.some((id) => (playerCounts.get(id) || 0) >= maxPlayer)) continue;
      const superstar = scored.superstars[candidate];
      if ((superstarCounts.get(superstar) || 0) >= maxSuperstar) continue;
      const set = new Set(members);
      if (overlapsPrior(set, sets, options.maxShared)) continue;

      let reserveOk = true;
      for (const [id, needed] of futureDemand.entries()) {
        const afterPick = (playerCounts.get(id) || 0) + (set.has(id) ? 1 : 0);
        if (afterPick + needed > maxPlayer) { reserveOk = false; break; }
      }
      if (!reserveOk) continue;
      picked = candidate;
      pickedSet = set;
      break;
    }

    if (picked === null) {
      const name = String(rule.name || "Construction rule");
      unfilled[name] = (unfilled[name] || 0) + 1;
      continue;
    }

    const members = candidateMembers(scored, picked);
    const superstar = scored.superstars[picked];
    chosen.push(picked);
    sets.push(pickedSet);
    used.add(picked);
    assignedRules.push(String(rule.name || "Construction rule"));
    for (const id of members) playerCounts.set(id, (playerCounts.get(id) || 0) + 1);
    superstarCounts.set(superstar, (superstarCounts.get(superstar) || 0) + 1);
  }

  let quality = 0;
  for (const candidate of chosen) quality += values[candidate];
  return { chosen, sets, rules: assignedRules, unfilled, quality };
}

/* Greedy marginal expected-best gain; evaluation draws are never used to select. */
function scenarioPortfolio(scored, options, data = scenarios, count = simCount) {
  const target = options.entries;
  if (target > 1 && constructionRules(options).length) {
    throw new Error("Disable roster-shape quotas for shared-scenario portfolio selection.");
  }
  const split = Math.floor(count / 2);
  if (target > 1 && split < 2) throw new Error("At least four scenarios are required.");
  const maxPlayer = exposureLimit(target, options.maxPlayerExposure);
  const maxStar = exposureLimit(target, options.maxSuperstarExposure);
  const order = orderBy(scored, "expected");
  const chosen = [], sets = [], playerCounts = new Map(), starCounts = new Map();
  let best = new Float64Array(split);
  function scores(i, start, stop) {
    const out = new Float64Array(stop - start), ids = candidateMembers(scored, i);
    for (let s = start; s < stop; s++) {
      let value = 0.5 * data[scored.superstars[i] * count + s];
      for (const id of ids) value += data[id * count + s];
      out[s - start] = value;
    }
    return out;
  }
  function feasible(i) {
    const members = candidateMembers(scored, i);
    return !members.some(id => (playerCounts.get(id) || 0) >= maxPlayer) &&
      (starCounts.get(scored.superstars[i]) || 0) < maxStar &&
      !overlapsPrior(new Set(members), sets, options.maxShared);
  }
  function take(i) {
    const members = candidateMembers(scored, i);
    chosen.push(i); sets.push(new Set(members));
    for (const id of members) playerCounts.set(id, (playerCounts.get(id) || 0) + 1);
    const star = scored.superstars[i]; starCounts.set(star, (starCounts.get(star) || 0) + 1);
  }
  if (target > 0 && order.length && feasible(order[0])) {
    take(order[0]);
    if (target > 1) best = scores(order[0], 0, split);
  }
  const heap = [];
  const higher = (a,b) => a.bound > b.bound || (a.bound === b.bound && a.id < b.id);
  function push(item) {
    let i = heap.length; heap.push(item);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!higher(item, heap[p])) break;
      heap[i] = heap[p]; i = p;
    }
    heap[i] = item;
  }
  function pop() {
    const top = heap[0], last = heap.pop();
    if (heap.length) {
      let i = 0;
      while (i * 2 + 1 < heap.length) {
        let j = i * 2 + 1;
        if (j + 1 < heap.length && higher(heap[j + 1], heap[j])) j++;
        if (!higher(heap[j], last)) break;
        heap[i] = heap[j]; i = j;
      }
      heap[i] = last;
    }
    return top;
  }
  if (target > 1) for (const id of order) {
    if (!chosen.includes(id)) push({id, bound: Infinity, epoch: -1});
  }
  while (heap.length && chosen.length < target) {
    const item = pop();
    if (!feasible(item.id)) continue;
    const values = scores(item.id, 0, split);
    if (item.epoch !== chosen.length) {
      let gain = 0;
      for (let s = 0; s < split; s++) gain += Math.max(values[s] - best[s], 0);
      push({id:item.id, bound:gain / split, epoch:chosen.length});
      continue;
    }
    take(item.id);
    for (let s = 0; s < split; s++) best[s] = Math.max(best[s], values[s]);
  }
  const evaluation = {objective: target === 1 ? "expected_points" : "expected_best",
    selection_scenarios: target === 1 ? 0 : split, evaluation_scenarios: count - split};
  if (chosen.length && count > split) {
    const baseline = scores(order[0], split, count), held = new Float64Array(baseline);
    for (const id of chosen.slice(1)) {
      const values = scores(id, split, count);
      for (let s = 0; s < held.length; s++) held[s] = Math.max(held[s], values[s]);
    }
    evaluation.evaluation_best_mean = held.reduce((a,b)=>a+b,0) / held.length;
    evaluation.evaluation_gain_over_single = held.reduce((a,b,i)=>a+b-baseline[i],0) / held.length;
    const ordered = Array.from(held).sort((a,b)=>a-b), q = .25 * (ordered.length - 1);
    evaluation.evaluation_best_p25 = ordered[Math.floor(q)] +
      (q % 1) * (ordered[Math.ceil(q)] - ordered[Math.floor(q)]);
  }
  return {chosen, sets, rules:chosen.map(()=>null), unfilled:{}, evaluation};
}

/* Reserve mandatory future exposure and try several rule schedules. */
function portfolio(scored, order, options) {
  if (options.objective === "auto" || options.objective === "portfolio") {
    return scenarioPortfolio(scored, options);
  }
  const target = options.entries;
  if (target <= 0) return { chosen: [], sets: [], rules: [], unfilled: {} };

  const rules = constructionRules(options);
  // A one-entry contest is lineup selection, not portfolio diversification.
  if (target === 1 || !rules.length) {
    return unrestrictedPortfolio(scored, order, options, target);
  }

  const ruleCounts = apportionedRuleCounts(rules, target);
  const ruleData = buildRuleData(scored, order, rules);
  const values = objectiveValues(scored, options.objective);
  const attempts = ["scarcity", "mandatory", "round_robin"].map((mode) =>
    selectionAttempt(scored, order, options, rules, ruleCounts, ruleData, values, mode)
  );
  attempts.sort((a, b) =>
    (b.chosen.length - a.chosen.length) || (b.quality - a.quality)
  );
  return attempts[0];
}

function describe(scored, indices, construction) {
  return indices.map((c, position) => {
    const ids = candidateMembers(scored, c);
    const result = {
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
    if (construction && construction[position]) result.construction_rule = construction[position];
    return result;
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

/* ---------- message handling -------------------------------------------- */

function setModel(payload) {
  model = payload;
  scenarios = null;
}

if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    LINEUP_SIZE, exposureLimit, scenarioPortfolio, setModel, simulate, enumerate, screen, score, orderBy,
    portfolio, describe, diversity, latentMatrix, cholesky, scoreCorrelation,
    normalCdf, normalPpf, conditionalCv, hurdleCoefficients, hurdleScoreCorrelation,
    constructionSchedule, apportionedRuleCounts, matchesConstructionRule,
    covarianceMatrix: () => covariance,
  };
}

if (typeof self === "undefined") {
  // Node equivalence tests use the exported functions directly.
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
    if (!Number.isFinite(options.salaryCap) || options.salaryCap <= 0) {
      // Yahoo does not price every single-game slate; those games publish a model
      // with no cap and the page collects one before asking for a solve.
      throw new Error("This game has no salary cap. Enter the cap from your contest.");
    }
    if (!Number.isInteger(options.entries) || options.entries < 1 || options.entries > 150) {
      throw new Error("Entries must be a whole number between 1 and 150.");
    }
    if (exposureLimit(options.entries, options.maxPlayerExposure) === 0 ||
        exposureLimit(options.entries, options.maxSuperstarExposure) === 0) {
      throw new Error("The exposure limits allow zero appearances. Increase a limit or request more entries.");
    }
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
        "No valid roster fits these settings. Lower the salary floor, put a player back, " +
        "or relax a position limit."
      );
    }

    self.postMessage({ type: "stage", stage: "Screening candidates" });
    const screened = screen(rosters, options);
    self.postMessage({ type: "stage", stage: "Scoring against scenarios" });
    const scored = score(rosters, screened, options, (fraction) => {
      self.postMessage({ type: "progress", fraction });
    });

    const order = orderBy(scored, ["auto", "portfolio"].includes(options.objective) ? "expected" : options.objective);
    const built = portfolio(scored, order, options);
    if (built.chosen.length < options.entries) {
      const details = Object.entries(built.unfilled)
        .map(([name, count]) => `${name}: ${count}`).join(", ");
      throw new Error(
        `Built ${built.chosen.length} of ${options.entries} entries under the construction, ` +
        "exposure, and overlap rules. " +
        (details ? `Unfilled construction slots: ${details}. ` : "") +
        "Exposure limits use the requested entry count, so the partial result is not displayed. " +
        "Increase detail, disable construction quotas, relax limits, or restore excluded players."
      );
    }

    self.postMessage({
      type: "result",
      valid_rosters: rosters.salary.length,
      candidates_scored: scored.total,
      portfolio: describe(scored, built.chosen, built.rules),
      scenario_evaluation: built.evaluation || null,
      strongest: describe(scored, order.slice(0, 10)),
      construction: {
        requested: options.entries,
        selected: built.chosen.length,
        unfilled: built.unfilled,
      },
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

