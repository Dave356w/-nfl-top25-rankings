"""Offline checks for tools/evaluate_sources.py scoring."""
import sys
import unittest
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import evaluate_sources as ev  # noqa: E402


class EvaluateSourcesTest(unittest.TestCase):
    def frame(self):
        rows = []
        for i in range(12):
            rows.append({"player_id": str(i), "Position": "RB", "week": 1, "Sleeper": 10.0 + i,
                         "ESPN": 9.0 + i, "CBS_raw": 20.0 + 2 * i, "Actual": 10.0 + i})
        rows.append({"player_id": "x", "Position": "RB", "week": 1, "Sleeper": 2.0, "ESPN": 1.0,
                     "CBS_raw": 4.0, "Actual": None})    # below MIN_PROJECTION everywhere
        return pd.DataFrame(rows)

    def test_cbs_rescaled_from_projections_and_sample_is_pregame(self):
        d = ev.prepare(self.frame())
        self.assertEqual(len(d), 12)
        self.assertAlmostEqual(d["CBS"].iloc[0], 10.0)      # x0.5 from Sleeper/CBS ratio
        self.assertAlmostEqual(d["Mean of 3"].iloc[0], (10 + 9 + 10) / 3)

    def test_metrics_and_paired_differences(self):
        d = ev.prepare(self.frame())
        by = {r["source"]: r for r in ev.metrics(d)}
        self.assertEqual(by["Sleeper"]["mae"], 0.0)
        self.assertEqual(by["ESPN"]["mae"], 1.0)
        paired = {r["source"]: r for r in ev.paired_vs_sleeper(d)}
        self.assertTrue(paired["ESPN"]["significant"])
        self.assertIn("| Source |", ev.markdown({"weeks": [1], "sample": {"player_weeks": 12, "players": 12},
                                                "overall": ev.metrics(d), "paired_vs_sleeper": list(paired.values()),
                                                "time_split": None, "origin": {1: {"ESPN": "refetched"}}}))

    def test_completed_weeks_waits_for_monday_night(self):
        self.assertEqual(ev.completed_weeks(datetime(2026, 9, 15, 11, tzinfo=timezone.utc)), [])
        self.assertEqual(ev.completed_weeks(datetime(2026, 9, 15, 13, tzinfo=timezone.utc)), [1])


if __name__ == "__main__":
    unittest.main()
