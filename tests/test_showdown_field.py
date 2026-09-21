import unittest

import numpy as np
import pandas as pd

from pipeline import showdown_field


class YahooShowdownFieldModelTests(unittest.TestCase):
    def test_archive_is_auditable_and_nonempty(self):
        archive = showdown_field.load_archive()
        self.assertGreaterEqual(len(archive["observations"]), 20)
        self.assertGreaterEqual(len(archive["sources"]), 4)
        self.assertTrue(all(row.get("url") for row in archive["sources"]))

    def test_fitted_model_is_finite(self):
        model = showdown_field.fit_ownership_model()
        self.assertEqual(model["features"], list(showdown_field.FEATURES))
        self.assertEqual(len(model["coefficients"]), len(showdown_field.FEATURES))
        self.assertTrue(np.isfinite(model["coefficients"]).all())

    def test_live_roster_ownership_is_normalized_to_five_slots(self):
        players = pd.DataFrame({
            "Position": ["QB", "QB", "RB", "RB", "WR", "WR", "TE", "DEF"],
            "Projected_FP": [22, 19, 17, 11, 16, 8, 9, 6],
        })
        ownership = showdown_field.predict_roster_ownership(
            players, field_size=2000, entry_fee=1.0
        )
        self.assertAlmostEqual(float(ownership.sum()), 5.0, places=8)
        self.assertTrue(((ownership > 0) & (ownership < 1)).all())

    def test_superstar_shares_sum_to_one(self):
        players = pd.DataFrame({
            "Position": ["QB", "RB", "WR", "TE", "DEF"],
            "Projected_FP": [20, 15, 14, 8, 6],
        })
        roster = np.array([.9, .8, .75, .6, .4])
        star = showdown_field.predict_superstar_ownership(players, roster)
        self.assertAlmostEqual(float(star.sum()), 1.0, places=10)
        self.assertGreater(star[0], star[-1])

    def test_leave_one_contest_out_report_is_explicitly_historical(self):
        report = showdown_field.leave_one_contest_out()
        self.assertEqual(report["observations"], 25)
        self.assertEqual(report["contests"], 5)
        self.assertLess(report["mae"], 0.40)
        self.assertTrue(report["observed_duplication"])
        self.assertIn("leave-one-contest-out", report["method"])


if __name__ == "__main__":
    unittest.main()
