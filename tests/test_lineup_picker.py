"""The page's in-browser re-pick, checked against the Python optimizer.

`site/lineup.html` lets you add, drop or bench a player and re-picks the lineup
without waiting for the next workflow run, which means `site/lineup-picker.js`
implements the same slot rules `lineup_optimizer.optimize` implements. Two
implementations of one rule drift silently, and the drift would surface as a
lineup that only ever existed on screen -- so the agreement is pinned here,
driving the shipped JavaScript through Node rather than restating it.

The rows handed to Node are the published ones, straight out of
`run_lineup._player_row`, so what is compared is exactly what the browser sees:
projections rounded to two decimals, missing bands as null.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lineup_optimizer as lo
import run_lineup
from test_lineup_optimizer import _context, _yahoo_frame

NODE = shutil.which("node")

# Enough players that the fixed slots and the flex both have to make a choice,
# and enough spread that no comparison rests on a coin-flip tie.
EXTRA = [
    # Name, position, team, opponent, salary, fppg, projected, depth
    ("Puka Nacua", "WR", "PHI", "DAL", 30, 15.0, 16.5, 1.0),
    ("Wan'Dale Robinson", "WR", "NE", "NYJ", 17, 8.0, 8.5, 2.0),
    ("Bucky Irving", "RB", "PHI", "DAL", 25, 12.0, 13.0, 1.0),
    ("Tyjae Spears", "RB", "NE", "NYJ", 14, 6.0, 6.5, 2.0),
    ("Dallas Goedert", "TE", "PHI", "DAL", 19, 9.0, 9.5, 1.0),
    ("Tyrod Taylor", "QB", "NE", "NYJ", 12, 5.0, 5.5, 2.0),
]

CONFIGURED = [
    {"Name": "Dak Prescott", "Position": "QB"},
    {"Name": "Breece Hall", "Position": "RB"},
    {"Name": "Jaylen Waddle", "Position": "WR"},
    {"Name": "Tee Higgins", "Position": "WR"},
    {"Name": "Harold Fannin Jr", "Position": "TE"},
    {"Name": "Brandon Aubrey", "Position": "K"},
    {"Name": "Bye Week Guy", "Position": "RB"},
] + [{"Name": name, "Position": position} for name, position, *_ in EXTRA]


def _wide_yahoo() -> pd.DataFrame:
    frame = _yahoo_frame()
    more = pd.DataFrame(EXTRA, columns=[
        "Feed_Name", "Feed_Position", "Team", "Opponent", "Salary", "FPPG",
        "Projected_FP", "Fallback_Depth",
    ])
    more["Key"] = more["Feed_Name"].map(lo.normalize_name)
    more["Game_Time"] = pd.to_datetime("2026-09-13T17:00:00Z")
    more["Projection_Source"] = "Yahoo FPPG + weekly salary prior"
    return pd.concat([frame, more], ignore_index=True)


def _roster() -> pd.DataFrame:
    roster = lo.build_roster(CONFIGURED, _wide_yahoo(), _context())
    # The browser only ever sees the published numbers, so the Python side has
    # to optimize on those too -- otherwise a comparison could fail on a tie
    # that only rounding created.
    for column in ("FP", "Floor_P25", "Ceiling_P90"):
        roster[column] = roster[column].round(2)
    return roster


def _rows(roster: pd.DataFrame) -> list[dict]:
    return [run_lineup._player_row(player) for _, player in roster.iterrows()]


@unittest.skipIf(NODE is None, "Node is required to run the browser picker")
class PickerAgreementTests(unittest.TestCase):
    def setUp(self):
        self.roster = _roster()
        self.rows = _rows(self.roster)

    def _browser(self, excluded, objective):
        payload = {
            "rows": self.rows,
            "options": {
                "slots": dict(lo.STARTING_POSITIONS),
                "flexEligible": list(lo.FLEX_ELIGIBLE),
                "objective": objective,
                "excluded": [lo.normalize_name(name) for name in excluded],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            result = subprocess.run(
                [NODE, str(ROOT / "tests" / "lineup_picker_reference.js"), str(path)],
                capture_output=True, text=True, timeout=60,
            )
        if result.returncode != 0:
            raise AssertionError(f"node harness failed:\n{result.stderr}")
        return json.loads(result.stdout)

    def _python(self, excluded, objective):
        starters, bench = lo.optimize(self.roster, excluded, objective)
        return (
            sorted((p.Name, p.Slot) for _, p in starters.iterrows()),
            sorted(bench["Name"].tolist()),
        )

    def _assert_same(self, excluded, objective):
        picked = self._browser(excluded, objective)
        starters, bench = self._python(excluded, objective)
        self.assertEqual(
            sorted((row["player"], row["slot"]) for row in picked["starters"]),
            starters, f"{objective} with {excluded or 'nobody'} benched")
        self.assertEqual(sorted(picked["bench"]), bench)

    def test_it_fills_the_same_slots_on_every_objective(self):
        for objective in ("FP", "Floor_P25", "Ceiling_P90"):
            self._assert_same([], objective)

    def test_a_benching_moves_the_same_players(self):
        # What the page does when you bench a starter at kickoff: the slot has
        # to fall to the same replacement the workflow would have chosen.
        for objective in ("FP", "Floor_P25", "Ceiling_P90"):
            self._assert_same(["Dak Prescott", "Puka Nacua"], objective)

    def test_a_dropped_player_leaves_the_rest_of_the_lineup_alone(self):
        dropped = "Jaylen Waddle"
        self.roster = self.roster[self.roster.Name.ne(dropped)].copy()
        self.rows = _rows(self.roster)
        self._assert_same([], "FP")

    def test_a_short_roster_leaves_the_slot_empty_rather_than_inventing_one(self):
        self.roster = self.roster[self.roster.Position.ne("K")].copy()
        self.rows = _rows(self.roster)
        picked = self._browser([], "FP")
        self.assertNotIn("K", [row["slot"] for row in picked["starters"]])
        self._assert_same([], "FP")

    def test_totals_match_the_published_arithmetic(self):
        # `run_lineup._total`: a starter with no band contributes his mean to
        # both ends rather than voiding the range. Bye Week Guy has no band.
        # Benching the other backs forces the one starter with no fitted band
        # -- the bye-week player -- into the lineup.
        picked = self._browser(["Breece Hall", "Bucky Irving", "Tyjae Spears"], "FP")
        by_name = {row["player"]: row for row in self.rows}
        started = [by_name[row["player"]] for row in picked["starters"]]
        self.assertIn(None, [row["floor"] for row in started])
        for key in ("mean", "floor", "ceiling"):
            expected = round(sum(
                row[key] if row[key] is not None else row["mean"] for row in started
            ), 2)
            self.assertAlmostEqual(picked["totals"][key], expected, places=2)
        self.assertLess(picked["totals"]["floor"], picked["totals"]["mean"])
        self.assertGreater(picked["totals"]["ceiling"], picked["totals"]["mean"])


if __name__ == "__main__":
    unittest.main()
