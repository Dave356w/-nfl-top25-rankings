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
PLAYER_KEYS = ("player", "pos", "team", "opponent", "kickoff_utc", "mean", "floor",
               "ceiling", "salary", "fppg", "depth", "depth_source", "source",
               "market_quality", "injury", "review")

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

    def test_the_objective_is_recorded_as_run(self):
        payload = run_lineup.build_payload(_results("Ceiling_P90"), "Ceiling_P90", [])
        self.assertEqual(payload["objective"], "Ceiling_P90")

    def test_kickoffs_are_published_in_the_format_the_page_parses(self):
        kickoff = self.payload["starters"][0]["kickoff_utc"]
        self.assertRegex(kickoff, r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")


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

            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["week"], 2)
            index = json.loads((data / "index.json").read_text())
            self.assertEqual(index["runs"], entries)


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


if __name__ == "__main__":
    unittest.main()
