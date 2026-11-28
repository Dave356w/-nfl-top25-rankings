from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import ats_research as ats  # noqa: E402


def schedule(rows):
    """rows: (game_id, home, away, gameday, gametime ET, line, home_score, away_score)."""
    return pd.DataFrame([dict(game_id=gid, season=2026, week=12, home_team=h, away_team=a,
                              gameday=day, gametime=time, spread_line=line, total_line=44.0,
                              home_score=hs, away_score=as_)
                         for gid, h, a, day, time, line, hs, as_ in rows])


def composite(board, season_predictions=(), outer=()):
    return {"schema": 1, "revision": "boxscore-composite-v1.7", "season": 2026, "week": 12,
            "generated_utc": "2026-11-25T10:00:00+00:00",
            "recipe": {"family": "rates_core_avail_cs", "half_life": 8.0, "ridge": 0.1},
            "fit": {"training_games": 2000}, "board": list(board),
            "season_predictions": list(season_predictions), "outer_predictions": list(outer)}


def game(gid, home, away, wp, week=12, season=2026, line=None, result=None, ready=True, with_home=None):
    """wp is the probability without home field (the margin's input); with_home the composite's own."""
    return dict(game_id=gid, season=season, week=week, home=home, away=away,
                model_wp=with_home if with_home is not None else wp, model_wp_no_home=wp,
                ready=ready, spread_line=line, result=result, homefield_wp=0.55)


class MarginTests(unittest.TestCase):
    def test_implied_margin_inverts_the_market_conversion(self):
        for line in (-7.0, -2.5, 0.0, 3.0, 13.5):
            self.assertAlmostEqual(ats.implied_margin(ats.market_wp(line)), line, places=6)
        self.assertGreater(ats.implied_margin(0.7), 0)
        self.assertLess(ats.implied_margin(0.3), 0)

    def test_bands_split_at_3_7_10_and_map_is_fixed(self):
        self.assertEqual([ats.band_of(x) for x in (2.9, 3.0, -6.9, 7.0, 9.99, 10.0)],
                         [0, 1, 1, 2, 2, 3])
        self.assertEqual(ats.fit_map(), [3.0, 5.0, 7.0, 13.0])
        self.assertEqual(ats.mapped_margin(-8.0), -7.0)
        self.assertEqual(ats.mapped_margin(11.0), 13.0)

    def test_zero_edge_takes_the_favourite_and_grades_from_the_side(self):
        self.assertEqual(ats.model_side(3.0, 3.0), 1)
        self.assertEqual(ats.model_side(-3.0, -3.0), -1)
        self.assertEqual(ats.model_side(5.0, 2.5), 1)
        self.assertEqual(ats.model_side(5.0, 6.5), -1)
        self.assertEqual(ats.grade(-1, 2.0, 6.5), "W")     # away +6.5, lost by 2
        self.assertEqual(ats.grade(1, 3.0, 3.0), "P")
        self.assertEqual(ats.record(["W", "L", "P"])["cover_rate"], 0.5)

    def test_kickoff_is_read_as_us_eastern(self):
        self.assertEqual(ats.kickoff_utc("2026-11-29", "13:00").isoformat(), "2026-11-29T18:00:00+00:00")
        self.assertIsNone(ats.kickoff_utc(None, None))


class PayloadTests(unittest.TestCase):
    def sched(self, final=False):
        return schedule([
            ("g1", "DET", "CHI", "2026-11-26", "13:00", -2.5, 31.0 if final else None, 21.0 if final else None),
            ("g2", "BUF", "KC", "2026-11-29", "16:25", 2.5, None, None),
        ])

    def board(self):
        # DET 0.70 -> margin 6.49 (3-7 band, maps to 5); BUF 0.40 -> -3.13 (maps to -5).
        return composite([game("g1", "DET", "CHI", 0.70), game("g2", "BUF", "KC", 0.40)])

    def test_rows_use_the_composite_and_the_nflverse_line(self):
        payload, new = ats.build_payload(self.board(), self.sched(), now="2026-11-25T12:00:00Z",
                                         snapshot_id="s1")
        det = next(g for g in payload["games"] if g["home"] == "DET")
        self.assertAlmostEqual(det["raw_margin"], round(ats.implied_margin(0.70), 2))
        self.assertEqual((det["band"], det["mapped_margin"], det["line"]), ("3 to 7", 5.0, -2.5))
        self.assertEqual(det["model_side"], "DET")          # 5 vs CHI favoured by 2.5
        self.assertAlmostEqual(det["market_wp"], round(ats.market_wp(-2.5), 4))
        buf = next(g for g in payload["games"] if g["home"] == "BUF")
        self.assertEqual((buf["mapped_margin"], buf["model_side"]), (-5.0, "KC"))
        self.assertEqual(det["model"], ats.MODEL_NAME)
        self.assertFalse(det["locked"])
        self.assertEqual(new, [])
        self.assertIn("No demonstrated edge", payload["disclaimer"])
        self.assertEqual(payload["method"]["map"]["rule"], [3.0, 5.0, 7.0, 13.0])

    def test_started_games_freeze_and_are_logged_once(self):
        first, _ = ats.build_payload(self.board(), self.sched(), now="2026-11-25T12:00:00Z", snapshot_id="s1")
        # A later composite run moved DET's probability; the frozen row must not change.
        later = composite([game("g1", "DET", "CHI", 0.20), game("g2", "BUF", "KC", 0.40)])
        second, new = ats.build_payload(later, self.sched(), previous=first,
                                        now="2026-11-27T12:00:00Z", snapshot_id="s2")
        det = next(g for g in second["games"] if g["home"] == "DET")
        self.assertTrue(det["locked"])
        self.assertEqual((det["snapshot_id"], det["model_side"]), ("s1", "DET"))
        self.assertEqual([r["game_id"] for r in new], ["g1"])
        third, again = ats.build_payload(later, self.sched(final=True), previous=second, ledger=new,
                                         now="2026-11-28T12:00:00Z", snapshot_id="s3")
        self.assertEqual(again, [])
        det = next(g for g in third["games"] if g["home"] == "DET")
        self.assertEqual((det["result"], third["forward_record"]["wins"]), ("W", 1))

    def test_the_margin_comes_from_the_probability_without_home_field(self):
        board = composite([game("g1", "DET", "CHI", 0.52, with_home=0.70)])
        payload, _ = ats.build_payload(board, self.sched(), now="2026-11-25T12:00:00Z")
        det = payload["games"][0]
        self.assertEqual((det["model_wp"], det["model_wp_no_home"]), (0.7, 0.52))
        self.assertAlmostEqual(det["raw_margin"], round(ats.implied_margin(0.52), 2))   # 0.62, not 6.49
        self.assertEqual((det["mapped_margin"], det["model_side"]), (3.0, "DET"))   # 3 vs CHI -2.5

    def test_a_game_first_seen_after_kickoff_gets_no_row(self):
        payload, new = ats.build_payload(self.board(), self.sched(), now="2026-11-27T12:00:00Z")
        self.assertEqual([g["home"] for g in payload["games"]], ["BUF"])
        self.assertEqual(new, [])

    def test_rows_from_the_retired_points_model_are_not_carried_forward(self):
        old = {"games": [{"game_id": "nfl.g.1", "kickoff_utc": "2026-11-26T18:00:00+00:00",
                          "season": 2026, "week": 12, "home": "DET", "away": "CHI"}]}
        payload, new = ats.build_payload(self.board(), self.sched(), previous=old,
                                         now="2026-11-27T12:00:00Z")
        self.assertNotIn("nfl.g.1", {g["game_id"] for g in payload["games"]})
        self.assertEqual(new, [])

    def test_a_team_without_history_gets_no_side(self):
        board = composite([game("g2", "BUF", "KC", 0.40, ready=False)])
        payload, _ = ats.build_payload(board, self.sched(), now="2026-11-25T12:00:00Z")
        self.assertIsNone(payload["games"][0]["model_side"])
        self.assertIsNone(payload["games"][0]["raw_margin"])

    def test_outputs_append_the_ledger(self):
        first, _ = ats.build_payload(self.board(), self.sched(), now="2026-11-25T12:00:00Z")
        second, new = ats.build_payload(self.board(), self.sched(), previous=first,
                                        now="2026-11-27T12:00:00Z")
        with tempfile.TemporaryDirectory() as directory:
            ats.write_outputs(second, new, directory)
            ats.write_outputs(second, [], directory)
            self.assertEqual(len(ats.read_ledger(Path(directory) / "ledger.jsonl")), 1)
            self.assertTrue((Path(directory) / "latest.csv").exists())


class SeasonTests(unittest.TestCase):
    def test_live_rows_replace_their_replay_game_and_the_rest_stay_replay(self):
        predictions = [game("a", "DET", "CHI", 0.70, week=11, line=-2.5, result=10.0),   # W: DET -> 5 > -2.5
                       game("b", "BUF", "KC", 0.40, week=11, line=2.5, result=7.0),      # L: KC side
                       game("c", "NYJ", "MIA", 0.55, week=12, line=1.0, result=-3.0),
                       game("d", "LV", "DEN", 0.55, week=12, line=1.0, result=None)]     # not final
        live = [dict(season=2026, week=12, nflverse_game_id="c", home="NYJ", away="MIA", line=1.0,
                     result="W"), dict(season=2025, week=12, nflverse_game_id="z", result="W")]
        weeks = ats.season_results(2026, predictions, live)
        self.assertEqual([(w["week"], w["source"]) for w in weeks], [(12, "forward"), (11, "replay")])
        self.assertEqual([g["result"] for g in weeks[1]["games"]], ["W", "L"])   # sorted by away team
        self.assertEqual(weeks[0]["games"][0]["result"], "W")
        pooled = ats.season_record(weeks)
        self.assertEqual((pooled["wins"], pooled["losses"]), (2, 1))

    def test_backtest_grades_model_raw_margin_and_spread_control(self):
        outer = [game("o1", "DET", "CHI", 0.70, season=2024, line=-2.5, result=10.0),
                 game("o2", "BUF", "KC", 0.40, season=2024, line=2.5, result=7.0),
                 game("o3", "NYJ", "MIA", 0.55, season=2024, line=1.0, result=3.0, ready=False)]
        bt = ats.backtest(composite([], outer=outer), long_run={"cover_rate": 0.5})
        self.assertEqual(bt["projection_map"]["seasons"]["2024"]["wins"], 1)
        self.assertEqual(bt["projection_map"]["all"]["losses"], 1)
        self.assertEqual(sum(bt["spread_map_control"]["all"][k] for k in ("wins", "losses", "pushes")), 3)
        self.assertEqual(bt["shown_from"], 2024)
        self.assertEqual(bt["spread_map_control"]["long_run"], {"cover_rate": 0.5})

    def test_headline_pools_held_out_replay_and_live_once_per_game(self):
        outer = [game("o1", "DET", "CHI", 0.70, season=2024, line=-2.5, result=10.0),   # W
                 game("o2", "BUF", "KC", 0.40, season=2025, line=2.5, result=7.0)]      # L
        current = [game("c1", "NYJ", "MIA", 0.70, week=11, line=-2.5, result=10.0),     # replay W
                   game("c2", "LV", "DEN", 0.70, week=12, line=-2.5, result=-7.0),      # replay L, live W
                   game("c3", "SF", "SEA", 0.70, week=12, line=-2.5, result=None)]      # not final
        live = [dict(season=2026, week=12, nflverse_game_id="c2", result="W")]
        hl = ats.pooled_results(composite([], season_predictions=current, outer=outer), live)
        self.assertEqual((hl["wins"], hl["losses"]), (3, 1))
        self.assertEqual((hl["live_games"], hl["replay_games"]), (1, 3))
        self.assertEqual((hl["first_season"], hl["last_season"]), (2024, 2026))

    def test_committed_composite_export_loads(self):
        export = ats.load_composite()
        self.assertTrue(export["board"])
        self.assertGreater(len(export["outer_predictions"]), 500)
        self.assertTrue({"game_id", "home", "away", "model_wp", "model_wp_no_home", "spread_line", "ready"}
                        <= set(export["board"][0]))
        self.assertTrue(all(g.get("model_wp_no_home") is not None for g in export["outer_predictions"]))
        self.assertEqual(export["recipe"]["family"], "rates_core_avail_cs")


if __name__ == "__main__":
    unittest.main()
