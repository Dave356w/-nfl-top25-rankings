"""Shape tests for the published lineup JSON.

The optimizer's feeds are live, so what is pinned here is the contract between
`run_lineup.build_payload` and `site/lineup.html`: every key the page reads, the
NaN handling that would otherwise emit invalid JSON, the slot labels the page
sorts by, and the archive index the footer links to.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lineup_optimizer as lo
import run_lineup
from test_lineup_optimizer import _context, _yahoo_frame

# Keys `site/lineup.html` reads off a player row. A missing one renders as a
# silent "undefined" in the browser rather than an error, so they are pinned.
PLAYER_KEYS = ("player", "key", "pos", "team", "opponent", "kickoff_utc", "mean",
               "floor", "ceiling", "salary", "fppg", "depth", "depth_source",
               "source", "market_quality", "injury", "review")

CONFIGURED = [
    {"Name": "Dak Prescott", "Position": "QB"},
    {"Name": "Breece Hall", "Position": "RB"},
    {"Name": "Jaylen Waddle", "Position": "WR"},
    {"Name": "Tee Higgins", "Position": "WR"},
    {"Name": "Harold Fannin Jr", "Position": "TE"},
    {"Name": "Brandon Aubrey", "Position": "K"},
    {"Name": "Monday Guy", "Position": "WR"},
    {"Name": "Bye Week Guy", "Position": "RB"},
]


def _results(objective="FP"):
    yahoo = _yahoo_frame()
    yahoo["Market_Quality"] = ["good", None, None, None, None]
    yahoo.loc[0, "Projection_Source"] = "market good: Bovada+Underdog"
    context = _context()
    roster = lo.build_roster(CONFIGURED, yahoo, context)
    excluded = lo.reported_out(roster)
    starters, bench = lo.optimize(roster, excluded, objective)
    return {
        "roster": roster, "starters": starters, "bench": bench, "context": context,
        "excluded": excluded,
        "market_audit": {"feeds": ["Bovada", "Underdog"],
                         "notes": ["bovada: 812 player-market records (live)"],
                         "matched": 3, "accepted": 1, "skill_rows": 4,
                         "logit_vig": 0.166, "calibration_pairs": 41},
    }


class PayloadTests(unittest.TestCase):
    def setUp(self):
        self.payload = run_lineup.build_payload(_results(), "FP", ["line one"])

    def test_it_serializes_without_nan(self):
        # json.dumps emits a bare NaN token, which JSON.parse rejects in the
        # browser. A player with no fitted range must land as null, not NaN.
        text = json.dumps(self.payload)
        self.assertNotIn("NaN", text)
        bye = [r for r in self.payload["starters"] + self.payload["bench"]
               if r["player"] == "Bye Week Guy"][0]
        self.assertIsNone(bye["floor"])

    def test_it_carries_every_key_the_page_reads(self):
        for key in ("status", "generated_utc", "run_date", "season", "week",
                    "objective", "slots", "totals", "counts", "starters", "bench",
                    "benched_by_request", "market", "log"):
            self.assertIn(key, self.payload)
        for key in PLAYER_KEYS:
            self.assertIn(key, self.payload["starters"][0])
            self.assertIn(key, self.payload["bench"][0])
        for key in ("feeds", "notes", "matched", "accepted", "skill_rows"):
            self.assertIn(key, self.payload["market"])

    def test_only_starters_carry_a_slot(self):
        slots = [row["slot"] for row in self.payload["starters"]]
        self.assertEqual(sorted(slots), sorted(["QB", "RB", "RB", "WR", "WR", "TE", "K"]))
        self.assertTrue(all("slot" not in row for row in self.payload["bench"]))

    def test_totals_use_the_mean_where_a_player_has_no_band(self):
        # Bye Week Guy has no CV, so his 0.0 stands in at both ends rather than
        # voiding the whole band.
        totals = self.payload["totals"]
        self.assertAlmostEqual(totals["mean"], 76.0)
        self.assertLess(totals["floor"], totals["mean"])
        self.assertGreater(totals["ceiling"], totals["mean"])

    def test_an_out_player_is_benched_and_recorded(self):
        self.assertEqual(self.payload["benched_by_request"], ["Tee Higgins"])
        self.assertNotIn("Tee Higgins", [r["player"] for r in self.payload["starters"]])

    def test_counts_and_context_come_from_the_run(self):
        self.assertEqual(self.payload["counts"]["roster"], len(CONFIGURED))
        self.assertEqual(self.payload["counts"]["projected"], len(CONFIGURED) - 1)
        self.assertEqual(self.payload["week"], 2)
        self.assertEqual(self.payload["season"], 2026)
        self.assertEqual(self.payload["slots"], dict(lo.STARTING_POSITIONS))

    def test_the_browser_can_reproduce_the_slot_and_bench_rules(self):
        # The page re-optimizes an edited roster itself. Without these it would
        # have to hard-code the league's slots and the auto-bench rule, and
        # would go quietly wrong the day either one changed.
        self.assertEqual(self.payload["slots"], dict(lo.STARTING_POSITIONS))
        self.assertEqual(self.payload["flex_eligible"], list(lo.FLEX_ELIGIBLE))
        self.assertEqual(self.payload["auto_exclude_out"], lo.AUTO_EXCLUDE_REPORTED_OUT)

    def test_every_row_carries_the_key_a_roster_edit_matches_on(self):
        rows = self.payload["starters"] + self.payload["bench"]
        for row in rows:
            self.assertEqual(row["key"], lo.normalize_name(row["player"]))
        self.assertEqual(len({row["key"] for row in rows}), len(rows))

    def test_the_objective_is_recorded_as_run(self):
        payload = run_lineup.build_payload(_results("Ceiling_P90"), "Ceiling_P90", [])
        self.assertEqual(payload["objective"], "Ceiling_P90")

    def test_kickoffs_are_published_in_the_format_the_page_parses(self):
        kickoff = self.payload["starters"][0]["kickoff_utc"]
        self.assertRegex(kickoff, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")


class PoolPayloadTests(unittest.TestCase):
    """`pool.json`: the players the page's roster editor can add."""

    def setUp(self):
        self.lineup = run_lineup.build_payload(_results(), "FP", [])
        self.pool = lo.build_pool(_yahoo_frame(), _context(),
                                  {"QB": 5, "RB": 5, "WR": 5, "TE": 5, "K": 5})
        self.payload = run_lineup.build_pool_payload(self.pool, self.lineup)

    def test_it_carries_every_key_the_editor_reads(self):
        for key in ("schema", "generated_utc", "run_date", "season", "week",
                    "limits", "note", "counts", "players"):
            self.assertIn(key, self.payload)
        for key in PLAYER_KEYS:
            self.assertIn(key, self.payload["players"][0])
        self.assertEqual(self.payload["counts"]["players"], len(self.pool))

    def test_it_is_stamped_with_the_run_that_priced_it(self):
        # A pool from an older run than the lineup would price an added player
        # off different market lines than the ones on screen.
        self.assertEqual(self.payload["run_date"], self.lineup["run_date"])
        self.assertEqual(self.payload["generated_utc"], self.lineup["generated_utc"])
        self.assertEqual(self.payload["week"], self.lineup["week"])

    def test_a_pool_player_carries_no_slot(self):
        self.assertTrue(all("slot" not in row for row in self.payload["players"]))

    def test_a_failed_pool_publishes_the_reason_rather_than_nothing(self):
        payload = run_lineup.build_pool_payload(
            pd.DataFrame(), self.lineup, "Add pool unavailable (feed down).")
        json.dumps(payload)
        self.assertEqual(payload["players"], [])
        self.assertIn("unavailable", payload["note"])


class DegradedFeedTests(unittest.TestCase):
    """What the first live run actually hit: nflverse had no 2026 injuries."""

    def setUp(self):
        context = _context()
        context["injuries"] = pd.DataFrame()
        context["notes"] = ["Injury report unavailable (Season must be between "
                            "2009 and 2025); nobody was auto-benched."]
        roster = lo.build_roster(CONFIGURED, _yahoo_frame(), context)
        starters, bench = lo.optimize(roster, [])
        self.payload = run_lineup.build_payload(
            {"roster": roster, "starters": starters, "bench": bench,
             "context": context, "market_audit": lo._empty_market_audit(),
             "excluded": []}, "FP", ["line one"])

    def test_a_missing_injury_feed_still_serializes(self):
        # With no injury frame to merge, `report_status` is pandas' NA
        # sentinel, which json.dumps rejects outright -- the run published
        # nothing until `_clean` learned to treat it as null.
        json.dumps(self.payload)
        self.assertTrue(all(row["injury"] is None
                            for row in self.payload["starters"]))

    def test_the_degradation_is_published_not_just_warned(self):
        # A warning goes to stderr, which the page never sees. A lineup built
        # with no injury data can start a player who is already ruled out.
        self.assertTrue(self.payload["notes"])
        self.assertIn("Injury report unavailable", self.payload["notes"][0])
        self.assertEqual(self.payload["benched_by_request"], [])


class WriteOutputTests(unittest.TestCase):
    def setUp(self):
        self.payload = run_lineup.build_payload(_results(), "FP", ["line one"])

    def test_it_archives_indexes_and_writes_a_flat_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "lineup"
            entries = run_lineup.write_outputs(self.payload, data)

            latest = json.loads((data / "latest.json").read_text())
            self.assertEqual(latest["run_date"], self.payload["run_date"])
            self.assertTrue((data / "history" / f"{latest['run_date']}.json").exists())

            csv = pd.read_csv(data / "latest.csv")
            self.assertEqual(len(csv), len(self.payload["starters"]) + len(self.payload["bench"]))
            self.assertIn("BENCH", set(csv["slot"]))
            self.assertIn("QB", set(csv["slot"]))
            # `key` is a join artefact for the page, not a column anyone wants
            # in a downloaded sheet.
            self.assertNotIn("key", csv.columns)

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["week"], 2)
            index = json.loads((data / "index.json").read_text())
            self.assertEqual(index["runs"], entries)
            # No pool was handed over, so none is written -- and a stale one
            # from an earlier run is never left to look current.
            self.assertFalse((data / "pool.json").exists())

    def test_the_pool_is_published_beside_the_lineup_but_not_archived(self):
        pool = run_lineup.build_pool_payload(
            lo.build_pool(_yahoo_frame(), _context()), self.payload)
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "lineup"
            run_lineup.write_outputs(self.payload, data, pool)

            published = json.loads((data / "pool.json").read_text())
            self.assertEqual(published["counts"]["players"], len(pool["players"]))
            self.assertTrue(published["players"])
            # One copy per run would grow the archive for a file only the
            # current week's editor can use.
            self.assertFalse((data / "history" / f"{pool['run_date']}-pool.json").exists())


@contextlib.contextmanager
def _quiet():
    """`main` prints its run log; the test output does not need a copy."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        yield buffer


class MainTests(unittest.TestCase):
    def test_a_failed_feed_writes_nothing_and_exits_non_zero(self):
        # The published lineup stays live rather than being replaced by a
        # half-built one, which is the daily build's policy too.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lineup"
            with unittest.mock.patch.object(
                lo, "fetch_yahoo", side_effect=RuntimeError("Yahoo feed failed")
            ), _quiet() as printed:
                code = run_lineup.main(["--out", str(out), "--no-market"])
            self.assertEqual(code, 1)
            self.assertIn("no files were written", printed.getvalue())
            self.assertFalse(out.exists())

    def test_it_runs_the_optimizer_end_to_end_and_publishes(self):
        results = _results()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lineup"
            with unittest.mock.patch.object(lo, "fetch_yahoo", return_value=_yahoo_frame()), \
                 unittest.mock.patch.object(lo, "load_nfl_context", return_value=_context()), \
                 unittest.mock.patch.object(lo, "load_roster", return_value=CONFIGURED), \
                 _quiet():
                code = run_lineup.main(["--out", str(out), "--no-market"])
            self.assertEqual(code, 0)
            published = json.loads((out / "latest.json").read_text())
            self.assertEqual(published["objective"], "FP")
            self.assertEqual(published["counts"]["roster"], len(CONFIGURED))
            self.assertTrue(published["log"])
            # --no-market still records why there are no market means.
            self.assertEqual(published["market"]["feeds"], [])
            self.assertIn("market projections disabled", published["market"]["notes"])
            self.assertEqual(len(published["starters"]), len(results["starters"]))

            pool = json.loads((out / "pool.json").read_text())
            self.assertEqual(pool["run_date"], published["run_date"])
            self.assertIn("Jaylen Waddle", [p["player"] for p in pool["players"]])

    def test_a_broken_add_pool_still_publishes_the_lineup(self):
        # The pool only feeds the page's roster editor. Taking the week's
        # lineup down with it would cost far more than the editor is worth.
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "lineup"
            with unittest.mock.patch.object(lo, "fetch_yahoo", return_value=_yahoo_frame()), \
                 unittest.mock.patch.object(lo, "load_nfl_context", return_value=_context()), \
                 unittest.mock.patch.object(lo, "load_roster", return_value=CONFIGURED), \
                 unittest.mock.patch.object(lo, "build_pool",
                                            side_effect=RuntimeError("depth chart empty")), \
                 _quiet():
                code = run_lineup.main(["--out", str(out), "--no-market"])
            self.assertEqual(code, 0)
            self.assertTrue(json.loads((out / "latest.json").read_text())["starters"])
            pool = json.loads((out / "pool.json").read_text())
            self.assertEqual(pool["players"], [])
            self.assertIn("depth chart empty", pool["note"])


if __name__ == "__main__":
    unittest.main()
