from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import ats_research as ats  # noqa: E402

MODEL = {"schema": 1, "coefficients": {"const": 24.0, "off": 0.17, "opp_def": -2.0, "home": 0.86},
         "backtest": {"note": "test"}}


def history(seasons=(2024, 2025), per_week=20, value=6.0):
    """Synthetic map history: every band well populated, signed margin `value`."""
    rows = []
    raws = [1.5, 5.0, 8.5, 12.0]
    for season in seasons:
        for week in range(1, 19):
            for i in range(per_week):
                raw = raws[i % 4] * (1 if i % 2 else -1)
                rows.append(dict(season=season, week=week, raw=raw, line=raw,
                                 actual=np.sign(raw) * value))
    return pd.DataFrame(rows)


def players(rows):
    out = []
    for game_id, kickoff, home, away, strengths in rows:
        for team, (off, dfn) in strengths.items():
            out += [dict(Name=f"{team} QB", Position="QB", Team=team, Projected_FP=off,
                         **{"Game ID": game_id, "Game Time": kickoff, "Home Team": home, "Away Team": away}),
                    dict(Name=f"{team} DEF", Position="DEF", Team=team, Projected_FP=dfn,
                         **{"Game ID": game_id, "Game Time": kickoff, "Home Team": home, "Away Team": away})]
    return pd.DataFrame(out)


def schedule(rows):
    return pd.DataFrame([dict(game_id=gid, season=2026, week=12, home_team=h, away_team=a,
                              game_date=d, spread_line=line, total_line=44.0, location="Home",
                              home_score=hs, away_score=as_)
                         for gid, h, a, d, line, hs, as_ in rows])


class MapTests(unittest.TestCase):
    def test_bands_split_at_3_7_10(self):
        self.assertEqual([ats.band_of(x) for x in (2.9, 3.0, -6.9, 7.0, 9.99, 10.0)],
                         [0, 1, 1, 2, 2, 3])

    def test_rule_needs_two_seasons_and_thirty_per_band(self):
        self.assertIsNone(ats.fit_map(history(seasons=(2025,)), 2026, 1))
        self.assertIsNone(ats.fit_map(history(per_week=1), 2026, 1))
        self.assertEqual(ats.fit_map(history(), 2026, 1), [6.0, 6.0, 6.0, 6.0])

    def test_rule_uses_only_earlier_weeks(self):
        h = history()
        late = h.assign(season=2026, actual=np.sign(h.raw) * 40)
        rule = ats.fit_map(pd.concat([h, late[late.week >= 5]]), 2026, 5)
        self.assertEqual(rule, [6.0, 6.0, 6.0, 6.0])

    def test_band_values_never_decrease(self):
        # Week-8-2026 shape: the 10+ band's median (6) sat below 7-10's (11).
        signed = np.array([3.0] * 5 + [5.0] * 5 + [11.0] * 5 + [6.0] * 5)
        bands = np.repeat(np.arange(4), 5)
        rule = ats.monotone_band_medians(signed, np.ones(20), bands, 4)
        self.assertEqual(rule[:2], [3.0, 5.0])
        self.assertEqual(rule[2], rule[3])
        self.assertTrue(all(a <= b for a, b in zip(rule, rule[1:])))
        self.assertGreaterEqual(ats.mapped_margin(13.4, rule), ats.mapped_margin(8.0, rule))

    def test_committed_history_maps_monotonically_every_week(self):
        h = ats.load_history()
        for week in range(1, 12):
            rule = ats.fit_map(h, 2026, week)
            self.assertTrue(all(a <= b for a, b in zip(rule, rule[1:])), (week, rule))

    def test_weighted_median_and_mapped_margin(self):
        self.assertEqual(ats.weighted_median([1, 2, 10], [1, 1, 5]), 10.0)
        self.assertEqual(ats.mapped_margin(-8.0, [3, 5, 8, 10]), -8.0)

    def test_zero_edge_takes_the_favourite_and_grades_from_the_side(self):
        self.assertEqual(ats.model_side(3.0, 3.0), 1)
        self.assertEqual(ats.model_side(-3.0, -3.0), -1)
        self.assertEqual(ats.model_side(5.0, 2.5), 1)
        self.assertEqual(ats.model_side(5.0, 6.5), -1)
        self.assertEqual(ats.grade(-1, 2.0, 6.5), "W")     # away +6.5, lost by 2
        self.assertEqual(ats.grade(1, 3.0, 3.0), "P")
        self.assertEqual(ats.record(["W", "L", "P"])["cover_rate"], 0.5)


class PayloadTests(unittest.TestCase):
    KICK_A = "2026-11-26T18:00:00Z"   # Thursday
    KICK_B = "2026-11-29T18:00:00Z"   # Sunday

    def slate(self):
        return players([
            ("y1", self.KICK_A, "DET", "CHI", {"DET": (60, 8), "CHI": (55, 6)}),
            ("y2", self.KICK_B, "BUF", "KC", {"BUF": (50, 7), "KC": (58, 9)}),
        ])

    def sched(self, final=False):
        return schedule([
            ("g1", "DET", "CHI", date(2026, 11, 26), -2.5, 31.0 if final else None, 21.0 if final else None),
            ("g2", "BUF", "KC", date(2026, 11, 29), 2.5, None, None),
        ])

    def test_rows_use_the_points_model_and_the_nflverse_line(self):
        payload, new = ats.build_payload(self.slate(), self.sched(), MODEL, history(),
                                         now="2026-11-25T12:00:00Z", snapshot_id="s1")
        det = next(g for g in payload["games"] if g["home"] == "DET")
        self.assertAlmostEqual(det["home_points"], round(24 + .17 * 60 - 2 * 6 + .86, 1))
        self.assertEqual(det["line"], -2.5)
        self.assertEqual(det["mapped_margin"], 6.0)
        self.assertEqual(det["model_side"], "DET")
        self.assertFalse(det["locked"])
        self.assertEqual(new, [])
        self.assertIn("No demonstrated edge", payload["disclaimer"])

    def test_started_games_freeze_and_are_logged_once(self):
        first, _ = ats.build_payload(self.slate(), self.sched(), MODEL, history(),
                                     now="2026-11-25T12:00:00Z", snapshot_id="s1")
        # Thursday game kicked off and Yahoo dropped it from the slate.
        later_slate = self.slate()[lambda f: f["Game ID"].eq("y2")]
        second, new = ats.build_payload(later_slate, self.sched(), MODEL, history(),
                                        previous=first, now="2026-11-27T12:00:00Z", snapshot_id="s2")
        det = next(g for g in second["games"] if g["home"] == "DET")
        self.assertTrue(det["locked"])
        self.assertEqual(det["snapshot_id"], "s1")
        self.assertEqual([r["game_id"] for r in new], ["y1"])
        third, again = ats.build_payload(later_slate, self.sched(final=True), MODEL, history(),
                                         previous=second, ledger=new,
                                         now="2026-11-28T12:00:00Z", snapshot_id="s3")
        self.assertEqual(again, [])
        det = next(g for g in third["games"] if g["home"] == "DET")
        self.assertEqual((det["result"], third["forward_record"]["wins"]), ("W", 1))

    def test_a_game_first_seen_after_kickoff_gets_no_row(self):
        payload, new = ats.build_payload(self.slate(), self.sched(), MODEL, history(),
                                         now="2026-11-27T12:00:00Z", snapshot_id="s1")
        self.assertEqual([g["home"] for g in payload["games"]], ["BUF"])
        self.assertEqual(new, [])

    def test_no_rule_means_no_side(self):
        payload, _ = ats.build_payload(self.slate(), self.sched(), MODEL, history(seasons=(2025,)),
                                       now="2026-11-25T12:00:00Z")
        self.assertTrue(all(g["model_side"] is None for g in payload["games"]))

    def test_outputs_append_the_ledger(self):
        first, _ = ats.build_payload(self.slate(), self.sched(), MODEL, history(),
                                     now="2026-11-25T12:00:00Z", snapshot_id="s1")
        second, new = ats.build_payload(self.slate(), self.sched(), MODEL, history(),
                                        previous=first, now="2026-11-27T12:00:00Z")
        with tempfile.TemporaryDirectory() as directory:
            ats.write_outputs(second, new, directory)
            ats.write_outputs(second, [], directory)
            self.assertEqual(len(ats.read_ledger(Path(directory) / "ledger.jsonl")), 1)
            self.assertTrue((Path(directory) / "latest.csv").exists())

    def test_a_ledger_game_replaces_its_history_row_in_the_map(self):
        h = history().assign(game_id=lambda f: [f"h{i}" for i in range(len(f))])
        # The refit history and the ledger both hold game h0; only one may count.
        dup = {"game_id": "x", "nflverse_game_id": "h0", "season": 2025, "week": 1,
               "raw_margin": float(h.raw.iloc[0]), "line": float(h.line.iloc[0]),
               "model_side_sign": 1}
        sched = self.sched()
        sched = pd.concat([sched, pd.DataFrame([dict(game_id="h0", season=2025, week=1,
                          home_team="X", away_team="Y", game_date=date(2025, 9, 7),
                          spread_line=0, total_line=0, location="Home",
                          home_score=99.0, away_score=0.0)])], ignore_index=True)
        seen = {}
        original = ats.fit_map
        def spy(train, season, week):
            seen["rows"] = len(train)
            return original(train, season, week)
        ats.fit_map = spy
        try:
            ats.build_payload(self.slate(), sched, MODEL, h, ledger=[dup],
                              now="2026-11-25T12:00:00Z")
        finally:
            ats.fit_map = original
        self.assertEqual(seen["rows"], len(h))

    def test_season_by_week_prefers_live_rows_and_pools_the_record(self):
        replay = {"season": 2026, "games": [
            dict(week=1, home="DET", away="CHI", line=-2.5, result="W"),
            dict(week=1, home="BUF", away="KC", line=2.5, result="L"),
            dict(week=2, home="DET", away="CHI", line=1.0, result="L")]}
        live = [dict(season=2026, week=2, home="DET", away="CHI", line=1.0, result="W"),
                dict(season=2025, week=2, home="X", away="Y", line=1.0, result="W")]
        weeks = ats.season_results(2026, replay, live)
        self.assertEqual([(w["week"], w["source"]) for w in weeks], [(2, "forward"), (1, "replay")])
        self.assertEqual(weeks[0]["games"][0]["result"], "W")
        pooled = ats.season_record(weeks)
        self.assertEqual((pooled["wins"], pooled["losses"]), (2, 1))

    def test_payload_carries_the_season(self):
        replay = {"season": 2026, "games": [dict(week=11, home="DET", away="CHI", line=-2.5,
                                                 home_score=31, away_score=21, result="W")]}
        payload, _ = ats.build_payload(self.slate(), self.sched(), MODEL, history(),
                                       now="2026-11-25T12:00:00Z", replay=replay)
        self.assertEqual(payload["season_results"][0]["week"], 11)
        self.assertEqual(payload["season_record"]["wins"], 1)

    def test_committed_season_replay_loads(self):
        replay = ats.load_replay()
        self.assertTrue(replay["games"])
        self.assertTrue({"week", "home", "away", "line", "home_score", "result"} <= set(replay["games"][0]))

    def test_blend_fit_is_only_for_the_season_evaluate_sources_covers(self):
        from tools import build_ats_research_model as tool
        from tools import evaluate_sources as ev
        self.assertTrue(tool.blend_team_games(pd.DataFrame(), ev.SEASON - 1).empty)

    def test_committed_model_records_what_it_was_fitted_on(self):
        fit = ats.load_model()["fit"]
        self.assertIn("source", fit)
        self.assertGreaterEqual(fit["team_games"], 2 * 100 if "blend" in fit["source"] else 1)

    def test_committed_artifacts_load(self):
        model = ats.load_model()
        self.assertEqual(set(model["coefficients"]), {"const", "off", "opp_def", "home"})
        self.assertLess(model["coefficients"]["opp_def"], 0)
        self.assertGreater(len(ats.load_history()), 300)


if __name__ == "__main__":
    unittest.main()
