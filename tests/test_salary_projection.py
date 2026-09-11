import unittest

import pandas as pd

from pipeline import salary_projection


class SalaryProjectionTests(unittest.TestCase):
    def test_shipped_model_is_frozen_and_validated(self):
        model = salary_projection.load()
        self.assertEqual(model["trained_through_season"], 2025)
        self.assertGreaterEqual(model["training_rows"], 10_000)
        self.assertLess(model["holdout_metrics"]["mae"], model["holdout_current_fallback"]["mae"])

    def test_salary_position_and_depth_all_affect_the_estimate(self):
        rows = pd.DataFrame([
            {"Position": "WR", "Salary": 30, "Depth_Rank": 1},
            {"Position": "WR", "Salary": 20, "Depth_Rank": 1},
            {"Position": "WR", "Salary": 30, "Depth_Rank": 4},
            {"Position": "RB", "Salary": 30, "Depth_Rank": 1},
        ])
        values = salary_projection.predict(rows, salary_projection.load())
        self.assertNotEqual(values[0], values[1])
        self.assertNotEqual(values[0], values[2])
        self.assertNotEqual(values[0], values[3])

    def test_manual_override_wins(self):
        frame = pd.DataFrame([{
            "Name": "Example", "Position": "RB", "Salary": 25,
            "Depth_Rank": 1, "Projected_FP": 1,
        }])
        out = salary_projection.apply(frame, overrides={"Example": 17.5})
        self.assertEqual(float(out.iloc[0]["Projected_FP"]), 17.5)
        self.assertEqual(out.iloc[0]["Projection_Source"], "manual override")

    def test_unpriced_player_is_not_given_a_projection(self):
        frame = pd.DataFrame([{
            "Name": "Bye", "Position": "RB", "Salary": None,
            "Depth_Rank": 1, "Projected_FP": None,
        }])
        out = salary_projection.apply(frame)
        self.assertTrue(pd.isna(out.iloc[0]["Projected_FP"]))


if __name__ == "__main__":
    unittest.main()
