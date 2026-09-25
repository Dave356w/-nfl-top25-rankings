/* Depth edits on the pages: the fitted volatility and scoreless-rate tables.
 *
 * The published mean is Sleeper's projection and does not depend on depth, so
 * a depth edit changes a player's volatility (CV) and scoreless rate and leaves
 * his mean alone. `predict` remains for a mean the frozen Yahoo
 * salary-position-depth regression produced (`isModelled`), which the live
 * pages no longer publish. Nothing in here is a second model:
 * it reads the exact coefficients the Python run used, from
 * `data/depth_model.json`, and `tests/test_depth_model_js.py` drives it from
 * Node against `salary_projection.predict` so the two cannot drift.
 *
 * Depth enters the regression only as an indicator for buckets 2, 3 and 4+, so
 * a depth edit moves a mean by a fixed, position-specific amount. The fitted
 * volatility (CV) and scoreless rate are keyed on the same bucket.
 */

"use strict";

var DepthModel = (function () {

  var REGRESSION_SOURCE = "salary-position-depth regression";

  function bucket(depth) {
    var d = Math.round(Number(depth));
    if (!isFinite(d)) d = 1;
    return Math.min(Math.max(d, 1), 4);
  }

  /* `salary_projection.feature_frame` for one player, as name -> value. */
  function features(spec, position, salary, depth) {
    var values = {};
    var scaled = (Number(salary) || 0) / Number(spec.salary_scale || 10);
    var b = bucket(depth);
    spec.positions.forEach(function (pos) {
      var flag = pos === position ? 1 : 0;
      values["position:" + pos] = flag;
      for (var power = 1; power <= Number(spec.degree); power++) {
        values["position:" + pos + ":salary^" + power] = flag * Math.pow(scaled, power);
      }
      for (var k = 2; k <= 4; k++) {
        var hit = flag * (b === k ? 1 : 0);
        values["position:" + pos + ":depth:" + k] = hit;
        if (spec.depth_salary_interactions) {
          values["position:" + pos + ":depth:" + k + ":salary"] = hit * scaled;
        }
      }
    });
    return values;
  }

  /* The regression mean, or null where the model does not price the player. */
  function predict(model, position, salary, depth) {
    var spec = model && model.regression;
    var pos = String(position || "").toUpperCase();
    if (!spec || spec.positions.indexOf(pos) < 0 || !(Number(salary) > 0)) return null;
    var values = features(spec, pos, salary, depth);
    var estimate = Number(spec.intercept);
    spec.features.forEach(function (name, i) {
      var x = values[name] === undefined ? 0 : values[name];
      estimate += Number(spec.coefficients[i]) * (x - Number(spec.center[i])) /
        Number(spec.scale[i]);
    });
    return Math.max(estimate, Number(spec.minimum_projection || 0.05));
  }

  function table(tables, position, depth) {
    var row = tables && tables[String(position || "").toUpperCase()];
    if (!row) return null;
    var value = row[String(bucket(depth))];
    return typeof value === "number" ? value : null;
  }

  function cv(model, position, depth) { return table(model && model.cv, position, depth); }
  function zero(model, position, depth) { return table(model && model.zero, position, depth); }

  /* `lineup_optimizer`'s lognormal P25 / P90 band around a mean. */
  function band(mean, coefficient) {
    if (!(typeof mean === "number" && isFinite(mean)) || !(coefficient > 0)) {
      return { floor: null, ceiling: null };
    }
    var sigma = Math.sqrt(Math.log1p(coefficient * coefficient));
    return {
      floor: mean * Math.exp(-0.67449 * sigma - 0.5 * sigma * sigma),
      ceiling: mean * Math.exp(1.28155 * sigma - 0.5 * sigma * sigma)
    };
  }

  /* Whether a published mean came from the regression, and so may be re-run.
   * A manual override or a rolling-stats fallback is left exactly as published. */
  function isModelled(source) {
    return String(source || "").toLowerCase().indexOf(REGRESSION_SOURCE) >= 0;
  }

  /* Move one player to `depth` and shift the teammates between his old and new
   * spot by one, the way moving a name on a depth chart does. Teammates the
   * page never saw keep their gaps: only ranks inside the moved range change.
   *
   * `players` is a list of { id, group, depth }; `group` is team + position.
   * Returns { id: newDepth } for every player whose depth changed.
   */
  function move(players, id, depth) {
    var target = null;
    players.forEach(function (p) { if (p.id === id) target = p; });
    var to = Math.max(1, Math.round(Number(depth)));
    if (!target || !isFinite(to)) return {};
    var from = Number(target.depth);
    var hasFrom = isFinite(from) && from >= 1;
    var changes = {};
    if (!hasFrom || from !== to) changes[id] = to;
    players.forEach(function (p) {
      if (p.id === id || p.group !== target.group) return;
      var d = Number(p.depth);
      if (!isFinite(d)) return;
      if (!hasFrom) {
        if (d >= to) changes[p.id] = d + 1;
      } else if (to < from && d >= to && d < from) {
        changes[p.id] = d + 1;
      } else if (to > from && d > from && d <= to) {
        changes[p.id] = d - 1;
      }
    });
    return changes;
  }

  /* Per-browser storage of depth edits, keyed to one published snapshot so a
   * new run's chart is never silently overridden by last week's edits. */
  function store(key, snapshot) {
    function read() {
      try {
        var raw = JSON.parse(localStorage.getItem(key) || "null");
        if (raw && raw.snapshot === snapshot && raw.depths) return raw.depths;
      } catch (e) { /* private mode or corrupt entry: start clean */ }
      return {};
    }
    function write(depths) {
      try {
        if (!depths || !Object.keys(depths).length) localStorage.removeItem(key);
        else localStorage.setItem(key, JSON.stringify({ snapshot: snapshot, depths: depths }));
      } catch (e) { /* the edit lasts for this page view only */ }
    }
    return { read: read, write: write };
  }

  function load(url) {
    return fetch(url || "data/depth_model.json", { cache: "no-store" }).then(function (r) {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  return {
    bucket: bucket, predict: predict, cv: cv, zero: zero, band: band,
    isModelled: isModelled, move: move, store: store, load: load
  };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = DepthModel;
}
