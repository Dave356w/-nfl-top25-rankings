import unittest
from datetime import date

import pandas as pd

from pipeline.showdown_backtest import NflverseHistory, YahooDfsClient, yahoo_offensive_points


class FakeYahoo(YahooDfsClient):
    def __init__(self, payload):
        self.payload = payload

    def _get(self, *args, **kwargs):
        return self.payload


class YahooDiscoveryTests(unittest.TestCase):
    def test_discovery_normalizes_historical_teams_and_cap(self):
        payload = {
            "sports": {"result": [{"sportCode": "nfl", "series": [{
                "gameCodeList": ["nfl.g.1"],
                "series": {"id": 22, "name": "TNF", "startTime": 1700000000000,
                           "salaryCapOverride": 125},
            }]}]},
            "games": {"result": {"nfl.g.1": {"game": {
                "homeTeam": {"abbr": "LA"}, "awayTeam": {"abbr": "JAC"},
            }}}},
        }
        rows = FakeYahoo(payload).discover(date(2023, 11, 1), date(2023, 11, 2))
        self.assertEqual(rows[0]["series_id"], 22)
        self.assertEqual(rows[0]["home"], "LAR")
        self.assertEqual(rows[0]["away"], "JAX")
        self.assertEqual(rows[0]["salary_cap"], 125)


class NflverseHistoryTests(unittest.TestCase):
    def setUp(self):
        schedules = pd.DataFrame([
            {"game_id": "2023_01_DET_KC", "season": 2023, "week": 1,
             "gameday": "2023-09-07", "home_team": "KC", "away_team": "DET"},
            {"game_id": "2023_02_MIN_PHI", "season": 2023, "week": 2,
             "gameday": "2023-09-14", "home_team": "PHI", "away_team": "MIN"},
        ])
        base = dict(
            position_group="QB", carries=1, targets=0, attempts=30,
            passing_tds=2, passing_interceptions=0, rushing_yards=10,
            rushing_tds=0, receiving_yards=0, receiving_tds=0, receptions=0,
            sack_fumbles_lost=0, rushing_fumbles_lost=0, receiving_fumbles_lost=0,
            passing_2pt_conversions=0, rushing_2pt_conversions=0,
            receiving_2pt_conversions=0, special_teams_tds=0,
        )
        stats = pd.DataFrame([
            dict(base, player_id="p1", game_id="2023_01_DET_KC", team="KC",
                 passing_yards=250),
            dict(base, player_id="p1", game_id="2023_02_MIN_PHI", team="PHI",
                 passing_yards=300),
        ])
        rosters = pd.DataFrame([
            {"season": 2023, "week": 2, "yahoo_id": "99", "gsis_id": "p1",
             "team": "PHI", "status": "ACT"},
        ])
        ids = pd.DataFrame([{"yahoo_id": "99", "gsis_id": "p1"}])
        self.history = NflverseHistory.prepare(schedules, stats, rosters, ids)

    def test_pregame_features_exclude_the_current_game(self):
        fppg, touches, games = self.history.pregame("p1", date(2023, 9, 14))
        self.assertAlmostEqual(fppg, 19.0)
        self.assertEqual(touches, 31.0)
        self.assertEqual(games, 1)

    def test_schedule_match_uses_historical_teams(self):
        row = self.history.match_game({
            "home": "PHI", "away": "MIN", "kickoff_ms": 1694736900000,
        })
        self.assertEqual(row["game_id"], "2023_02_MIN_PHI")

    def test_yahoo_scoring_is_half_ppr(self):
        frame = pd.DataFrame([{
            "passing_yards": 100, "passing_tds": 1, "rushing_yards": 20,
            "rushing_tds": 1, "receiving_yards": 30, "receiving_tds": 1,
            "receptions": 4, "passing_interceptions": 1,
            "rushing_fumbles_lost": 1,
        }])
        self.assertAlmostEqual(float(yahoo_offensive_points(frame).iloc[0]), 24.0)


if __name__ == "__main__":
    unittest.main()
