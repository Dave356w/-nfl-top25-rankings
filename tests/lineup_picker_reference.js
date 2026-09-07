/* Drive site/lineup-picker.js from Node and print what it picked.
 *
 * tests/test_lineup_picker.py builds a roster with the Python optimizer, hands
 * this the same published rows and the same rules, and checks the two agree on
 * every slot. The page re-picks an edited roster in the browser, so a drift
 * between these two would show up as a lineup that only exists on screen.
 *
 *     node tests/lineup_picker_reference.js <input.json>
 */

"use strict";

const fs = require("fs");
const path = require("path");

const picker = require(path.join(__dirname, "..", "site", "lineup-picker.js"));
const input = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const picked = picker.pick(input.rows, input.options);

process.stdout.write(JSON.stringify({
  starters: picked.starters.map((r) => ({ player: r.player, slot: r.slot })),
  bench: picked.bench.map((r) => r.player),
  totals: {
    mean: picker.total(picked.starters, "mean"),
    floor: picker.total(picked.starters, "floor"),
    ceiling: picker.total(picked.starters, "ceiling"),
  },
}));
