"""Offline tests for the weekly lineup optimizer.

`lineup_optimizer.run` needs the live Yahoo DFS feed, the sportsbooks and the
nflverse releases, so the fetch path cannot be exercised here. What can be
pinned without a network is everything downstream of the feeds: the name and
team keys the providers are joined on, the market means applied on top of the
Yahoo blend, the kicker and half-PPR arithmetic, the roster assembly
`build_roster` performs on already-fetched frames, and the slot rules
`optimize` applies.

The market tests drive the real pipeline matcher with hand-built `Projection`
objects, so the acceptance and game-time rules are the shipped ones rather than
a restatement of them.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
import unittest.mock
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lineup_optimizer as lo
from pipeline import notebook as nb

KICKOFF = "2026-09-13T17:00:00Z"


def _projection(player, team, points, quality="good", start=KICKOFF, feed="bovada"):
    return nb.Projection(
        event_id="e-" + player, matchup=team, start_time_utc=start, team=team,
        player=player, position="WR", fantasy_points=points, quality=quality,
        stat_means={}, sources={"receiving_yards": feed + " total"},
    )


def _roster_frame():
    rows = [
        ("QB1", "QB", 20.0), ("QB2", "QB", 15.0),
        ("RB1", "RB", 16.0), ("RB2", "RB", 14.0), ("RB3", "RB", 10.0),
        ("WR1", "WR", 15.0), ("WR2", "WR", 13.0), ("WR3", "WR", 12.0),
        ("TE1", "TE", 9.0), ("TE2", "TE", 7.0), ("K1", "K", 8.0),
    ]
    frame = pd.DataFrame(rows, columns=["Name", "Position", "FP"])
    frame["Key"] = frame["Name"].map(lo.normalize_name)
    # A deliberately inverted ceiling: the objective switch has to change the
    # lineup, not just the numbers printed beside it.
    frame["Floor_P25"] = frame["FP"] * 0.7
    frame["Ceiling_P90"] = np.where(frame["Name"].eq("WR3"), 60.0, frame["FP"] * 1.5)
    return frame


def _yahoo_frame():
    rows = [
        # Name, position, team, opponent, salary, fppg, projected, depth
        ("Jaylen Waddle", "WR", "MIA", "BUF", 28, 12.0, 13.5, 1.0),
        ("Tee Higgins", "WR", "CIN", "CLE", 26, 11.0, 12.0, 2.0),
        ("Dak Prescott", "QB", "DAL", "PHI", 30, 18.0, 19.0, 1.0),
        ("Breece Hall", "RB", "NYJ", "NE", 27, 13.0, 14.0, 1.0),
        ("Harold Fannin Jr.", "TE", "CLE", "CIN", 18, 7.0, 7.5, 2.0),
    ]
    frame = pd.DataFrame(rows, columns=[
        "Feed_Name", "Feed_Position", "Team", "Opponent", "Salary", "FPPG",
        "Projected_FP", "Fallback_Depth",
    ])
    frame["Key"] = frame["Feed_Name"].map(lo.normalize_name)
    frame["Game_Time"] = pd.to_datetime("2026-09-13T17:00:00Z")
    frame["Projection_Source"] = "Yahoo FPPG + weekly salary prior"
    return frame


def _stats_frame():
    kicks = [
        # season, week, 0-19, 20-29, 30-39, 40-49, 50-59, 60+, PAT
        (2025, 1, 0, 1, 0, 1, 0, 0, 3),
        (2025, 2, 0, 0, 1, 0, 1, 0, 2),
    ]
    kicker = pd.DataFrame(kicks, columns=[
        "season", "week", "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
        "fg_made_40_49", "fg_made_50_59", "fg_made_60_", "pat_made",
    ])
    kicker["player_display_name"] = "Brandon Aubrey"
    kicker["position"] = "K"
    kicker["team"] = "DAL"
    kicker["season_type"] = "REG"
    kicker["fg_missed"] = 0
    kicker["fantasy_points"] = 0.0
    kicker["fantasy_points_ppr"] = 0.0

    offense = pd.DataFrame({
        "player_display_name": ["Monday Guy"] * 2,
        "position": ["WR"] * 2,
        "team": ["BUF"] * 2,
        "season": [2025, 2025],
        "week": [1, 2],
        "season_type": ["REG", "REG"],
        "fantasy_points": [8.0, 12.0],
        "fantasy_points_ppr": [12.0, 16.0],
    })
    return pd.concat([kicker, offense], ignore_index=True)


def _context():
    depth = pd.DataFrame({
        "Key": [lo.normalize_name(n) for n in
                ["Jaylen Waddle", "Tee Higgins", "Dak Prescott", "Breece Hall",
                 "Harold Fannin Jr.", "Monday Guy"]],
        "Team": ["MIA", "CIN", "DAL", "NYJ", "CLE", "BUF"],
        "Official_Depth": [1, 1, 1, 1, 3, 2],
        "pos_abb": ["WR", "WR", "QB", "RB", "TE", "WR"],
    })
    teams = ["MIA", "CIN", "DAL", "NYJ", "CLE", "BUF", "PHI", "NE"]
    schedule = pd.DataFrame({
        "Team": teams,
        "Opponent": ["BUF", "CLE", "PHI", "NE", "CIN", "MIA", "DAL", "NYJ"],
        "NFL_Week": [2] * len(teams),
        "Home_Away": ["Home"] * len(teams),
        "Vegas_Total": [45.0] * len(teams),
        "Implied_Team_Total": [22.5] * len(teams),
    })
    injuries = pd.DataFrame({
        "Key": [lo.normalize_name("Tee Higgins")],
        "Team": ["CIN"],
        "report_primary_injury": ["Hamstring"],
        "report_status": ["Out"],
        "practice_status": ["Did Not Participate In Practice"],
    })
    return {"season": 2026, "week": 2, "schedule": schedule, "depth": depth,
            "depth_stamp": "2026-09-10 12:00:00", "injuries": injuries,
            "stats": _stats_frame()}


class NameAndTeamKeyTests(unittest.TestCase):
    def test_suffixes_and_punctuation_join_across_providers(self):
        # Yahoo prints "Harold Fannin Jr.", the roster holds "Harold Fannin Jr",
        # nflverse holds "Harold Fannin". All three have to land on one key.
        self.assertEqual(lo.normalize_name("Harold Fannin Jr."),
                         lo.normalize_name("Harold Fannin"))
        self.assertEqual(lo.normalize_name("James Cook III"),
                         lo.normalize_name("James Cook"))
        self.assertEqual(lo.normalize_name("Ja'Marr Chase"),
                         lo.normalize_name("JaMarr Chase"))
        self.assertEqual(lo.normalize_name("Ja’Marr Chase"),
                         lo.normalize_name("Ja'Marr Chase"))
        # Hyphens and periods are dropped in place, not replaced by a space,
        # so a name only joins when both feeds space it the same way.
        self.assertEqual(lo.normalize_name("Amon-Ra St. Brown"), "amonra st brown")

    def test_accents_are_folded_and_non_strings_are_empty(self):
        self.assertEqual(lo.normalize_name("Zamir Whíte"), "zamir white")
        self.assertEqual(lo.normalize_name(None), "")
        self.assertEqual(lo.normalize_name(float("nan")), "")

    def test_team_aliases_collapse_to_the_nflverse_code(self):
        self.assertEqual(lo.normalize_team("JAX"), "JAC")
        self.assertEqual(lo.normalize_team(" wsh "), "WAS")
        self.assertEqual(lo.normalize_team("OAK"), "LV")
        self.assertEqual(lo.normalize_team("KC"), "KC")
        self.assertEqual(lo.normalize_team(np.nan), "")


class KickerEstimateTests(unittest.TestCase):
    def test_it_scores_by_distance_band(self):
        # Week 1: 20-29 (3) + 40-49 (4) + 3 PAT = 10. Week 2: 30-39 (3) +
        # 50-59 (5) + 2 PAT = 10.
        mean, cv, games = lo.kicker_estimate("Brandon Aubrey", _stats_frame())
        self.assertEqual(games, 2)
        self.assertAlmostEqual(mean, 10.0)
        self.assertGreaterEqual(cv, 0.20)
        self.assertLessEqual(cv, 1.50)

    def test_an_unknown_kicker_falls_back_to_the_league_mean(self):
        mean, cv, games = lo.kicker_estimate("Nobody At All", _stats_frame())
        self.assertEqual(games, 0)
        self.assertAlmostEqual(mean, 10.0)   # the only kicker in the fixture
        self.assertGreater(cv, 0)


class OffenseFallbackTests(unittest.TestCase):
    def test_it_averages_half_ppr_over_recent_games(self):
        mean, games = lo.offense_fallback("Monday Guy", "WR", _stats_frame())
        self.assertEqual(games, 2)
        self.assertAlmostEqual(mean, 12.0)   # (10 + 14) / 2

    def test_an_unmatched_player_scores_nothing(self):
        self.assertEqual(lo.offense_fallback("Nobody At All", "WR", _stats_frame()),
                         (0.0, 0))


class BuildRosterTests(unittest.TestCase):
    def setUp(self):
        self.configured = [
            {"Name": "Dak Prescott", "Position": "QB"},
            {"Name": "Breece Hall", "Position": "RB"},
            {"Name": "Jaylen Waddle", "Position": "WR"},
            {"Name": "Tee Higgins", "Position": "WR"},
            {"Name": "Harold Fannin Jr", "Position": "TE"},
            {"Name": "Brandon Aubrey", "Position": "K"},
            {"Name": "Monday Guy", "Position": "WR"},
            {"Name": "Bye Week Guy", "Position": "RB"},
        ]
        self.roster = lo.build_roster(self.configured, _yahoo_frame(), _context())
        self.roster.index = self.roster["Name"]

    def test_every_configured_player_survives_the_joins(self):
        self.assertEqual(len(self.roster), len(self.configured))
        self.assertEqual(self.roster.loc["Jaylen Waddle", "Team"], "MIA")
        self.assertAlmostEqual(self.roster.loc["Jaylen Waddle", "FP"], 13.5)

    def test_kickers_come_from_the_nflverse_logs(self):
        row = self.roster.loc["Brandon Aubrey"]
        self.assertAlmostEqual(row["FP"], 10.0)
        self.assertEqual(row["Depth_Rank"], 1)
        self.assertIn("kicker", row["Projection_Source"])

    def test_a_yahoo_gap_falls_back_to_recent_half_ppr(self):
        # Yahoo omits some games (a Monday-only slate, say). A rostered player
        # whose team is on the schedule must not read as a zero.
        row = self.roster.loc["Monday Guy"]
        self.assertAlmostEqual(row["FP"], 12.0)
        self.assertIn("fallback", row["Projection_Source"])
        self.assertTrue(row["Projection_Available"])

    def test_a_player_on_no_schedule_stays_at_zero_and_is_flagged(self):
        row = self.roster.loc["Bye Week Guy"]
        self.assertEqual(row["FP"], 0)
        self.assertFalse(row["Projection_Available"])
        self.assertIn("no weekly projection/bye", lo.review(row))

    def test_depth_drives_the_calibrated_range(self):
        waddle = self.roster.loc["Jaylen Waddle"]
        self.assertEqual(waddle["Depth_Source"], "nflverse latest depth")
        self.assertAlmostEqual(waddle["Projection_CV"], lo.CALIBRATED_CV["WR"][1])
        self.assertLess(waddle["Floor_P25"], waddle["FP"])
        self.assertGreater(waddle["Ceiling_P90"], waddle["FP"])
        # A WR3's band is wider than a WR1's at the same mean.
        wide = lo.CALIBRATED_CV["WR"][3]
        self.assertGreater(wide, waddle["Projection_CV"])

    def test_the_injury_report_reaches_the_review_column(self):
        row = self.roster.loc["Tee Higgins"]
        self.assertEqual(row["report_status"], "Out")
        self.assertIn("injury Out", lo.review(row))

    def test_manual_depth_overrides_win(self):
        original = dict(lo.MANUAL_DEPTH_OVERRIDES)
        lo.MANUAL_DEPTH_OVERRIDES.update({"Jaylen Waddle": 4})
        try:
            roster = lo.build_roster(self.configured, _yahoo_frame(), _context())
            row = roster[roster.Name.eq("Jaylen Waddle")].iloc[0]
            self.assertEqual(row["Depth_Rank"], 4)
            self.assertEqual(row["Depth_Source"], "manual override")
        finally:
            lo.MANUAL_DEPTH_OVERRIDES.clear()
            lo.MANUAL_DEPTH_OVERRIDES.update(original)

    def test_a_roster_with_no_fitted_range_still_builds(self):
        # Regression: the P25/P90 pass reads Projection_CV unconditionally, so
        # a roster where nothing reaches the depth chart or the kicker branch
        # must still leave the column defined.
        ctx = _context()
        ctx["depth"] = ctx["depth"].iloc[0:0]
        ctx["injuries"] = pd.DataFrame()
        roster = lo.build_roster([{"Name": "Bye Week Guy", "Position": "RB"}],
                                 _yahoo_frame(), ctx)
        self.assertIn("Projection_CV", roster)
        self.assertTrue(roster["Floor_P25"].isna().all())

    def test_output_table_renders_both_halves(self):
        starters, bench = lo.optimize(self.roster.reset_index(drop=True), ["Tee Higgins"])
        table = lo.output_table(starters, True)
        self.assertEqual(list(table.columns)[0], "Slot")
        self.assertNotIn("Tee Higgins", lo.output_table(starters, True)["Player"].tolist())
        self.assertIn("Tee Higgins", lo.output_table(bench, False)["Player"].tolist())


class OptimizeTests(unittest.TestCase):
    def setUp(self):
        self.roster = _roster_frame()

    def test_it_fills_every_slot_exactly_once(self):
        starters, bench = lo.optimize(self.roster, [])
        self.assertEqual(len(starters), 8)
        self.assertEqual(len(bench), len(self.roster) - 8)
        counts = starters["Slot"].value_counts().to_dict()
        self.assertEqual(counts, {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "FLEX": 1})
        self.assertEqual(len(set(starters["Name"])), 8)

    def test_the_flex_takes_the_best_leftover_rb_wr_or_te(self):
        starters, _ = lo.optimize(self.roster, [])
        flex = starters[starters.Slot.eq("FLEX")].iloc[0]
        self.assertIn(flex["Position"], lo.FLEX_ELIGIBLE)
        self.assertEqual(flex["Name"], "WR3")   # 12.0 beats RB3 and TE2

    def test_a_quarterback_can_never_take_the_flex(self):
        starters, _ = lo.optimize(self.roster, ["RB3", "WR3", "TE2"])
        self.assertEqual(starters[starters.Slot.eq("FLEX")].empty, True)

    def test_exclusions_are_honoured_by_normalized_name(self):
        starters, bench = lo.optimize(self.roster, ["qb1"])
        self.assertEqual(starters[starters.Slot.eq("QB")].iloc[0]["Name"], "QB2")
        self.assertIn("QB1", bench["Name"].tolist())

    def test_the_objective_changes_who_gets_which_slot(self):
        # On means WR3 is the third-best receiver and only reaches the flex; on
        # ceilings it is the best, and the flex opens up for someone else.
        by_mean = lo.optimize(self.roster, [], "FP")[0].set_index("Name")["Slot"]
        by_ceiling = lo.optimize(self.roster, [], "Ceiling_P90")[0].set_index("Name")["Slot"]
        self.assertEqual(by_mean["WR3"], "FLEX")
        self.assertEqual(by_ceiling["WR3"], "WR")
        self.assertEqual(by_ceiling["WR2"], "FLEX")

    def test_a_missing_range_falls_back_to_the_mean(self):
        roster = self.roster.copy()
        roster["Ceiling_P90"] = np.nan
        starters, _ = lo.optimize(roster, [], "Ceiling_P90")
        self.assertEqual(starters[starters.Slot.eq("QB")].iloc[0]["Name"], "QB1")

    def test_an_unknown_objective_is_rejected(self):
        with self.assertRaises(ValueError):
            lo.optimize(self.roster, [], "Median")

    def test_a_short_roster_warns_instead_of_inventing_a_starter(self):
        thin = self.roster[self.roster.Position.ne("K")]
        starters, _ = lo.optimize(thin, [])
        self.assertEqual(len(starters), 7)
        self.assertTrue(starters[starters.Slot.eq("K")].empty)


class MarketProjectionTests(unittest.TestCase):
    """The market step, driven with hand-built projections instead of feeds."""

    def setUp(self):
        self.yahoo = _yahoo_frame()

    def _apply(self, projections):
        return lo.apply_market_projections(self.yahoo, projections=projections)

    def test_an_accepted_mean_replaces_the_yahoo_blend(self):
        out, audit = self._apply([_projection("Jaylen Waddle", "MIA", 17.25)])
        row = out[out.Feed_Name.eq("Jaylen Waddle")].iloc[0]
        self.assertAlmostEqual(row["Projected_FP"], 17.25)
        self.assertIn("market good", row["Projection_Source"])
        self.assertEqual(row["Market_Quality"], "good")
        # The displaced Yahoo number is kept, not overwritten in place.
        self.assertAlmostEqual(row["Fallback_Projected_FP"], 13.5)
        self.assertEqual((audit["matched"], audit["accepted"]), (1, 1))

    def test_a_projection_from_another_kickoff_never_crosses_games(self):
        # Same team and name, four hours later: a namesake in a different game
        # must not hand this player a line.
        out, audit = self._apply([
            _projection("Jaylen Waddle", "MIA", 40.0, start="2026-09-13T21:00:00Z"),
        ])
        row = out[out.Feed_Name.eq("Jaylen Waddle")].iloc[0]
        self.assertAlmostEqual(row["Projected_FP"], 13.5)
        self.assertEqual(audit["accepted"], 0)

    def test_a_weak_projection_is_matched_but_not_accepted(self):
        out, audit = self._apply([
            _projection("Jaylen Waddle", "MIA", 17.25, quality="partial"),
        ])
        row = out[out.Feed_Name.eq("Jaylen Waddle")].iloc[0]
        self.assertAlmostEqual(row["Projected_FP"], 13.5)
        self.assertEqual((audit["matched"], audit["accepted"]), (1, 0))

    def test_unpriced_players_are_kept_on_their_yahoo_numbers(self):
        # Dropping an unmatched player is fine for a ranking pool and wrong for
        # my own roster: he would vanish from the lineup with no explanation.
        out, _ = self._apply([_projection("Jaylen Waddle", "MIA", 17.25)])
        self.assertEqual(len(out), len(self.yahoo))
        row = out[out.Feed_Name.eq("Tee Higgins")].iloc[0]
        self.assertAlmostEqual(row["Projected_FP"], 12.0)
        self.assertEqual(row["Projection_Source"], "Yahoo FPPG + weekly salary prior")

    def test_the_depth_fallback_is_reranked_on_the_new_means(self):
        # Yahoo has Waddle ahead of the second Miami receiver; the market does
        # not. Salary-order depth has to follow the number actually used.
        extra = pd.DataFrame([{
            "Feed_Name": "Second Receiver", "Feed_Position": "WR", "Team": "MIA",
            "Opponent": "BUF", "Salary": 20, "FPPG": 8.0, "Projected_FP": 9.0,
            "Fallback_Depth": 2.0, "Key": lo.normalize_name("Second Receiver"),
            "Game_Time": pd.to_datetime(KICKOFF), "Market_Quality": None,
            "Projection_Source": "Yahoo FPPG + weekly salary prior",
        }])
        self.yahoo = pd.concat([self.yahoo, extra], ignore_index=True)
        out, _ = self._apply([_projection("Second Receiver", "MIA", 21.0)])
        depth = out.set_index("Feed_Name")["Fallback_Depth"]
        self.assertEqual(depth["Second Receiver"], 1)
        self.assertEqual(depth["Jaylen Waddle"], 2)

    def test_no_projections_leaves_the_feed_untouched(self):
        out, audit = self._apply([])
        pd.testing.assert_frame_equal(out, self.yahoo)
        self.assertEqual(audit["accepted"], 0)
        self.assertTrue(any("no usable market" in n for n in audit["notes"]))

    def test_a_broken_market_step_degrades_instead_of_stopping(self):
        # One sportsbook outage must not cost the week's lineup.
        with unittest.mock.patch.object(
            nb, "load_market_projection_reference", side_effect=RuntimeError("feed down")
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out, audit = lo.apply_market_projections(self.yahoo)
        pd.testing.assert_frame_equal(out, self.yahoo)
        self.assertTrue(any("feed down" in str(w.message) for w in caught))
        self.assertTrue(any("feed down" in n for n in audit["notes"]))

    def test_settings_never_drop_a_rostered_player(self):
        self.assertFalse(lo.market_settings(nb).market_drop_unmatched)

    def test_market_columns_reach_the_roster(self):
        out, _ = self._apply([_projection("Jaylen Waddle", "MIA", 17.25)])
        roster = lo.build_roster(
            [{"Name": "Jaylen Waddle", "Position": "WR"}], out, _context()
        )
        row = roster.iloc[0]
        self.assertAlmostEqual(row["FP"], 17.25)
        self.assertEqual(row["Market_Quality"], "good")


class RosterFileTests(unittest.TestCase):
    def test_no_path_uses_the_roster_in_the_module(self):
        self.assertEqual(lo.load_roster(), list(lo.MY_TEAM_ROSTER))

    def test_a_json_file_wins_and_positions_are_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roster.json"
            path.write_text(json.dumps({"roster": [
                {"Name": "Someone Else", "Position": "wr"},
            ]}), encoding="utf-8")
            self.assertEqual(lo.load_roster(str(path)),
                             [{"Name": "Someone Else", "Position": "WR"}])

    def test_a_bare_list_is_accepted_and_an_empty_one_is_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "roster.json"
            path.write_text(json.dumps([{"Name": "A Player", "Position": "TE"}]),
                            encoding="utf-8")
            self.assertEqual(len(lo.load_roster(str(path))), 1)
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                lo.load_roster(str(path))

    def test_the_shipped_roster_file_matches_the_module(self):
        shipped = Path(__file__).resolve().parents[1] / "lineup_roster.json"
        if shipped.exists():
            self.assertEqual(lo.load_roster(str(shipped)), list(lo.MY_TEAM_ROSTER))


class ReportedOutTests(unittest.TestCase):
    def test_it_reads_the_injury_report_case_insensitively(self):
        roster = pd.DataFrame({
            "Name": ["A", "B", "C"],
            "report_status": ["OUT", "Questionable", None],
        })
        self.assertEqual(lo.reported_out(roster), ["A"])

    def test_a_roster_with_no_injury_column_reports_nobody(self):
        self.assertEqual(lo.reported_out(pd.DataFrame({"Name": ["A"]})), [])


class SelfTestTests(unittest.TestCase):
    def test_the_shipped_self_test_passes(self):
        lo.self_test()


if __name__ == "__main__":
    unittest.main()
