/* The lineup page's half of `lineup_optimizer.optimize`.
 *
 * The workflow publishes one roster's lineup. The page lets you ask the same
 * question of a different roster -- a waiver add, a drop, a game-time benching
 * -- and answering it in the browser means running the optimizer's slot rules
 * here rather than waiting for the next scheduled run.
 *
 * Nothing in here is a second opinion about a player. Every number it reads was
 * priced by the Python run that published the page, so an edited lineup differs
 * from the published one only by who was eligible for which slot.
 *
 * `tests/test_lineup_picker.py` drives this from Node against `lo.optimize` on
 * the same rows, because two implementations of one rule drift silently.
 */

"use strict";

var LineupPicker = (function () {

  // `optimize` scores on the objective column and falls back to the mean where
  // a player has no fitted band, which is what pandas' fillna(work.FP) does.
  function score(row, objective) {
    var key = objective === "Floor_P25" ? "floor"
      : objective === "Ceiling_P90" ? "ceiling" : "mean";
    var value = row[key];
    if (typeof value === "number" && isFinite(value)) return value;
    return typeof row.mean === "number" && isFinite(row.mean) ? row.mean : 0;
  }

  /* Fill every fixed slot with the best available player, then give the flex to
   * the best one left over. Slots are taken in the order the payload lists
   * them, which is the order `STARTING_POSITIONS` declares -- a flex filled
   * before the fixed slots would strand a position at a worse player.
   *
   * `rows` are player rows as published; `options` carries the run's own rules:
   * `slots`, `flexEligible`, `objective`, and the `excluded` keys to bench.
   * Returns fresh row objects, so a caller can re-pick as often as it likes.
   */
  function pick(rows, options) {
    var slots = options.slots || {};
    var eligible = options.flexEligible || [];
    var objective = options.objective;
    var excluded = {};
    (options.excluded || []).forEach(function (key) { excluded[key] = true; });

    var ranked = rows.filter(function (r) { return !excluded[r.key]; })
      // Array sort is stable, so equal scores keep roster order -- the tie
      // break pandas' nlargest applies.
      .sort(function (a, b) { return score(b, objective) - score(a, objective); });
    var taken = {}, starters = [];

    function fill(count, wanted, label) {
      for (var i = 0; i < ranked.length && count > 0; i++) {
        var row = ranked[i];
        if (taken[row.key] || !wanted(row)) continue;
        taken[row.key] = true;
        count--;
        starters.push(Object.assign({}, row, { slot: label }));
      }
    }

    Object.keys(slots).forEach(function (position) {
      if (position === "FLEX") return;
      fill(slots[position], function (r) { return r.pos === position; }, position);
    });
    fill(slots.FLEX || 0, function (r) {
      return eligible.indexOf(r.pos) >= 0;
    }, "FLEX");

    return {
      starters: starters,
      bench: rows.filter(function (r) { return !taken[r.key]; })
        .map(function (r) { return Object.assign({}, r); })
        .sort(function (a, b) { return (b.mean || 0) - (a.mean || 0); })
    };
  }

  /* `run_lineup._total`: a starter with no fitted band contributes his mean to
   * both ends rather than voiding the whole range. */
  function total(rows, key) {
    var sum = 0;
    for (var i = 0; i < rows.length; i++) {
      var value = rows[i][key];
      if (typeof value !== "number" || !isFinite(value)) value = rows[i].mean;
      if (typeof value !== "number" || !isFinite(value)) return null;
      sum += value;
    }
    return Math.round(sum * 100) / 100;
  }

  return { score: score, pick: pick, total: total };
})();

if (typeof module !== "undefined" && module.exports) {
  module.exports = LineupPicker;
}
