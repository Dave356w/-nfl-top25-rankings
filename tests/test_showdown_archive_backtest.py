from __future__ import annotations

from datetime import date
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from pipeline.showdown_archive_backtest import (
    completed_player_key, grade_portfolio, load_snapshot, model_player_keys,
    run, snapshot_date_range, summarize_group, yahoo_defense_points,
)


def snapshot(reference=True):
    portfolio = [{"ids": [0, 1, 2, 3, 4], "superstar": 0,
                  "salary": 100, "expected_fp": 50}]
    model = {
        "game_id": "nfl.g.1", "matchup": "AAA vs BBB",
        "kickoff_utc": "2026-09-24T17:00:00-07:00", "salary_cap": 100,
        "players": [
            {"name": f"Player {i}", "team": "AAA" if i < 3 else "BBB",
             "pos": "QB" if i == 0 else "WR", "salary": 20, "fp": 10}
            for i in range(5)
        ],
        "reference": {"portfolio": portfolio} if reference else None,
    }
    return {
        "snapshot_id": "saved", "captured_utc": "2026-09-22T12:00:00Z",
        "showdown_models": [model],
        "predictions": [
            {"game_id": "nfl.g.1", "player": f"Player {i}",
             "player_key": f"yahoo:{i}", "team": "AAA" if i < 3 else "BBB",
             "position": "QB" if i == 0 else "WR", "salary": 20}
            for i in range(5)
        ],
    }


class SnapshotTests(unittest.TestCase):
    def test_latest_snapshot_and_date_range(self):
        with tempfile.TemporaryDirectory() as directory:
            for captured in ("2026-09-21T12:00:00Z", "2026-09-22T12:00:00Z"):
                document = snapshot()
                document["captured_utc"] = captured
                document["snapshot_id"] = captured
                target = Path(directory) / f"{captured[:10]}-{captured[11:13]}.json.gz"
                target.write_bytes(gzip.compress(json.dumps(document).encode()))
            loaded = load_snapshot(directory)
            self.assertEqual(loaded["snapshot_id"], "2026-09-22T12:00:00Z")
            self.assertEqual(snapshot_date_range(loaded), (
                date(2026, 9, 25), date(2026, 9, 26)
            ))

    def test_exact_archived_id_bridge_and_grading(self):
        saved = snapshot()
        model = saved["showdown_models"][0]
        keys = model_player_keys(saved, model)
        actuals = {f"yahoo:{i}": float(i + 1) for i in range(5)}
        graded = grade_portfolio(model, model["reference"]["portfolio"], keys, actuals)
        self.assertEqual(graded[0]["actual_fp"], 15 + 0.5)
        self.assertEqual(graded[0]["superstar_key"], "yahoo:0")

    def test_completed_identity_uses_yahoo_id_or_defense_key(self):
        self.assertEqual(completed_player_key({
            "code": "nfl.p.42", "primaryPosition": "QB",
        }), "yahoo:42")
        self.assertEqual(completed_player_key({
            "code": "nfl.t.1", "primaryPosition": "DEF",
            "team": {"abbr": "JAC"}, "firstName": "Jacksonville",
            "lastName": "Jaguars",
        }), "JAC:jacksonvillejaguars")


class SummaryTests(unittest.TestCase):
    def test_forward_and_reconstructed_are_not_pooled(self):
        games = [
            {"selection_status": "published_forward", "portfolio": [{"actual_fp": 10}],
             "regret_to_best_submitted": 2},
            {"selection_status": "reconstructed_from_pregame_snapshot",
             "portfolio": [{"actual_fp": 20}], "regret_to_best_submitted": 1},
        ]
        forward = summarize_group(games, "published_forward")
        rebuilt = summarize_group(games, "reconstructed_from_pregame_snapshot")
        self.assertEqual(forward["mean_portfolio_best_actual_fp"], 10)
        self.assertEqual(rebuilt["mean_portfolio_best_actual_fp"], 20)

    def test_run_grades_forward_portfolio_and_projection_actuals(self):
        saved = snapshot()
        saved["captured_utc"] = "2026-09-19T12:00:00Z"
        saved["showdown_models"][0]["kickoff_utc"] = "2026-09-20T17:00:00-07:00"
        for row in saved["predictions"]:
            row["kickoff_utc"] = "2026-09-20T17:00:00-07:00"
            row.update({
                "expected_fp": 10, "baseline_fp": 9, "p25": 5, "p90": 20,
                "source": "saved model",
            })

        class Yahoo:
            def players(self, series_id):
                return {"players": {"result": [{
                    "code": f"nfl.p.{i}", "primaryPosition": "QB" if i == 0 else "WR",
                    "salary": 20, "points": i + 1,
                } for i in range(5)]}}

            def benchmark(self, series_id):
                return {"allUserBestLineup": {"result": {"score": 20}}}

        report = run(saved, Yahoo(), [{
            "game_code": "nfl.g.1", "series_id": 7, "salary_cap": 100,
        }])
        self.assertEqual(report["games_completed"], 1)
        self.assertEqual(report["settled_player_actuals"], 5)
        self.assertEqual(report["projection_evaluation"]["overall"]["n"], 5)
        self.assertEqual(report["published_forward"]["games"], 1)
        self.assertEqual(report["reconstructed"]["games"], 0)

    def test_run_uses_nflverse_without_converting_missing_yahoo_to_zero(self):
        saved = snapshot()
        saved["captured_utc"] = "2026-09-19T12:00:00Z"
        saved["showdown_models"][0]["kickoff_utc"] = "2026-09-20T17:00:00-07:00"
        for row in saved["predictions"]:
            row["kickoff_utc"] = "2026-09-20T17:00:00-07:00"
            row.update({
                "expected_fp": 10, "baseline_fp": 9, "p25": 5, "p90": 20,
                "source": "saved model",
            })

        class Yahoo:
            pass

        class Nflverse:
            def actuals(self, model, player_keys):
                return ({key: float(index + 1) for index, key in enumerate(player_keys)},
                        {"game_id": "2026_03_AAA_BBB", "missing_without_zero_fill": []})

        report = run(saved, Yahoo(), [], fetch_benchmark=False, nflverse=Nflverse())
        self.assertEqual(report["games_completed"], 1)
        self.assertEqual(report["games"][0]["actual_source"], "nflverse_reconstructed")
        self.assertIn("No completed Yahoo", report["games"][0]["yahoo_fallback_reason"])


class DefenseScoringTests(unittest.TestCase):
    def test_yahoo_defense_scoring_and_points_allowed_exclusions(self):
        rows = []

        def play(**values):
            base = {
                "posteam": "BBB", "defteam": "AAA", "td_team": None,
                "special_teams_play": 0, "sack": 0, "interception": 0,
                "fumble_lost": 0, "fumble_recovery_1_team": None,
                "fumble_recovery_2_team": None, "safety": 0,
                "blocked_player_id": None, "return_touchdown": 0,
                "defensive_two_point_conv": 0, "touchdown": 0,
                "field_goal_result": None, "extra_point_result": None,
                "two_point_conv_result": None,
            }
            base.update(values)
            rows.append(base)

        play(sack=1)
        play(sack=1)
        play(interception=1)
        play(fumble_lost=1, fumble_recovery_1_team="AAA")
        play(blocked_player_id="blocker")
        play(safety=1)
        play(touchdown=1, return_touchdown=1, td_team="AAA")
        play(posteam="AAA", defteam="BBB", td_team="AAA", special_teams_play=1,
             touchdown=1, return_touchdown=1)
        play(defensive_two_point_conv=1)
        play(touchdown=1, td_team="BBB")
        play(extra_point_result="good")
        play(field_goal_result="made")
        play(two_point_conv_result="success")
        # A return TD against AAA special teams counts toward points allowed.
        play(posteam="AAA", defteam="BBB", td_team="BBB", special_teams_play=1,
             touchdown=1, return_touchdown=1)
        # A pick-six against AAA's offense does not count toward points allowed.
        play(posteam="AAA", defteam="BBB", td_team="BBB", touchdown=1,
             return_touchdown=1)

        points, audit = yahoo_defense_points(pd.DataFrame(rows), "AAA")
        self.assertEqual(audit["points_allowed"], 18)
        self.assertEqual(audit["points_allowed_score"], 1)
        self.assertEqual(points, 25)

if __name__ == "__main__":
    unittest.main()
