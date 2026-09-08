"""Guards for the shared rankings/lineup projection snapshot."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lineup_optimizer as lo
import run_synced


class PreparedSlateAdapterTests(unittest.TestCase):
    def test_final_rankings_mean_is_preserved_for_the_lineup(self):
        players = pd.DataFrame([
            {
                "Name": "Tee Higgins",
                "Position": "WR",
                "Team": "CIN",
                "Opponent": "TB",
                "Game Time": "2026-09-13T17:00:00Z",
                "Home Team": "CIN",
                "Away Team": "TB",
                "Salary": 33,
                "FPPG": 12.14,
                "Projected_FP": 13.03,
                "Projection_Source": "market good: Bovada+Underdog",
                "Depth_Rank": 2,
                "Market_Quality": "good",
            }
        ])

        adapted = lo.yahoo_from_prepared_slate(players)

        self.assertEqual(float(adapted.iloc[0]["Projected_FP"]), 13.03)
        self.assertEqual(adapted.iloc[0]["Feed_Name"], "Tee Higgins")
        self.assertEqual(adapted.iloc[0]["Feed_Position"], "WR")
        self.assertEqual(adapted.iloc[0]["Key"], "tee higgins")
        self.assertEqual(float(adapted.iloc[0]["Fallback_Depth"]), 2.0)
        self.assertTrue(str(adapted.iloc[0]["Game_Time"].tz) == "UTC")


class ConsistencyGateTests(unittest.TestCase):
    def setUp(self):
        self.rankings = {
            "snapshot_id": "snap-1",
            "rankings": {
                "QB": [{"player": "Dak Prescott", "fp": 18.8}],
                "WR": [{"player": "Tee Higgins", "fp": 13.03}],
            },
        }
        self.lineup = {
            "snapshot_id": "snap-1",
            "starters": [
                {"player": "Dak Prescott", "pos": "QB", "mean": 18.8},
                {"player": "Tee Higgins", "pos": "WR", "mean": 13.03},
            ],
            "bench": [],
        }
        self.pool = {
            "snapshot_id": "snap-1",
            "players": [
                {"player": "Dak Prescott", "pos": "QB", "mean": 18.8},
                {"player": "Tee Higgins", "pos": "WR", "mean": 13.03},
            ],
        }

    def test_matching_pages_pass(self):
        compared = run_synced.validate_page_consistency(
            self.rankings, self.lineup, self.pool
        )
        self.assertEqual(compared, 4)

    def test_a_projection_difference_blocks_publication(self):
        self.lineup["starters"][1]["mean"] = 12.38
        with self.assertRaisesRegex(ValueError, "tee higgins"):
            run_synced.validate_page_consistency(
                self.rankings, self.lineup, self.pool
            )

    def test_a_snapshot_difference_blocks_publication(self):
        self.lineup["snapshot_id"] = "older-snapshot"
        with self.assertRaisesRegex(ValueError, "snapshot_id"):
            run_synced.validate_page_consistency(
                self.rankings, self.lineup, self.pool
            )


if __name__ == "__main__":
    unittest.main()
