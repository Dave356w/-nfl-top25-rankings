"""Tests for the same-game correlation calibration tool.

The tool is offline work, like `calibrate_rb_correlation.py`, and the daily
build never runs it. What it produces is nonetheless an argument about whether
the showdown model's pair constants are right, so the machinery that produces
it is worth pinning: a planted correlation has to come back out, and the
agreement flag has to read the live table rather than a copy of it.

No network. `build_observations` is driven with a synthetic weekly frame in
nflverse's column shape.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import notebook as nb
from tools import calibrate_game_correlations as cal


def weekly_frame(seasons=40, planted=0.60, seed=0):
    """Two teams, one game a week, with a planted same-team QB-WR correlation."""
    rng = np.random.default_rng(seed)
    roster = [("HOME", "QB", 1), ("HOME", "WR", 1), ("HOME", "RB", 1),
              ("AWAY", "QB", 1), ("AWAY", "WR", 1), ("AWAY", "RB", 1)]
    rows = []
    for season in range(2000, 2000 + seasons):
        for week in range(1, 18):
            shocks = {}
            for team in ("HOME", "AWAY"):
                common = rng.normal()
                for position in ("QB", "WR"):
                    private = rng.normal()
                    shocks[(team, position)] = (
                        np.sqrt(planted) * common + np.sqrt(1 - planted) * private
                    )
                shocks[(team, "RB")] = rng.normal()
            for team, position, _ in roster:
                base = {"QB": 18.0, "WR": 11.0, "RB": 10.0}[position]
                rows.append({
                    "player_id": f"{team}-{position}",
                    "player_display_name": f"{team} {position}",
                    "position_group": position,
                    "team": team,
                    "opponent_team": "AWAY" if team == "HOME" else "HOME",
                    "season": season, "week": week, "season_type": "REG",
                    "carries": 10.0, "targets": 5.0,
                    "receptions": 0.0, "receiving_yards": 0.0, "receiving_tds": 0.0,
                    "rushing_tds": 0.0, "passing_yards": 0.0, "passing_tds": 0.0,
                    "passing_interceptions": 0.0,
                    # fantasy_points() weights rushing_yards at 0.10, so this
                    # carries the whole planted signal and nothing else does.
                    "rushing_yards": 10.0 * (base + 4.0 * shocks[(team, position)]),
                })
    return pd.DataFrame(rows)


class MeasurementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.observations = cal.build_observations(weekly_frame())
        cls.fitted = cal.fit(cls.observations)

    def test_it_pairs_only_players_from_the_same_game(self):
        self.assertEqual(self.observations["game"].nunique(),
                         self.observations.groupby("game").size().count())
        for _, game in self.observations.groupby("game"):
            self.assertEqual(set(game["team"]), {"HOME", "AWAY"})

    def test_it_recovers_a_planted_same_team_correlation(self):
        row = self.fitted[self.fitted["Relationship"].eq("Same team QB-WR")]
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row["Measured"].iloc[0]), 0.60, delta=0.10)

    def test_an_unrelated_pair_measures_near_zero(self):
        row = self.fitted[self.fitted["Relationship"].eq("Opposing QB-RB")]
        self.assertEqual(len(row), 1)
        self.assertLess(abs(float(row["Measured"].iloc[0])), 0.10)

    def test_relationships_are_the_pipelines_own_names(self):
        for name in self.fitted["Relationship"]:
            self.assertIsNotNone(
                cal.model_target(name), f"{name} does not map back to the table"
            )


class ComparisonTests(unittest.TestCase):
    def test_model_target_reads_the_live_table(self):
        self.assertEqual(cal.model_target("Same team QB-WR"),
                         round(nb.QB_WR_CORR[1], 3))
        self.assertEqual(cal.model_target("Opposing QB-QB"),
                         round(nb.OPPOSING_OFFENSE_CORR[("QB", "QB")], 3))

    def test_a_defense_pair_has_no_offensive_target(self):
        self.assertIsNone(cal.model_target("Opponent WR-DEF"))

    def test_agreement_is_decided_by_the_confidence_interval(self):
        fitted = pd.DataFrame([
            {"Relationship": "Same team QB-WR", "Pairs": 1000,
             "Measured": 0.215, "CI low": 0.203, "CI high": 0.227},
            {"Relationship": "Opposing WR-WR", "Pairs": 1000,
             "Measured": 0.500, "CI low": 0.480, "CI high": 0.520},
        ])
        result = cal.compare(fitted)
        self.assertEqual(list(result["Agrees"]), ["yes", "REVIEW"])

    def test_seasons_parse_both_ways(self):
        self.assertEqual(cal.parse_seasons("2016-2018"), [2016, 2017, 2018])
        self.assertEqual(cal.parse_seasons("2019,2021"), [2019, 2021])


if __name__ == "__main__":
    unittest.main()


class EmptyResultTests(unittest.TestCase):
    """A run with too little data reports nothing, rather than raising."""

    def test_a_thin_sample_returns_an_empty_table_not_a_keyerror(self):
        fitted = cal.fit(cal.build_observations(weekly_frame(seasons=2)))
        self.assertTrue(fitted.empty)
        self.assertIn("Relationship", fitted.columns)
        self.assertTrue(cal.compare(fitted).empty)
