"""Offline tests for the weekly lineup optimizer.

`lineup_optimizer.run` needs the live Yahoo DFS feed and Sleeper's API, so the
fetch path cannot be exercised here. What can be pinned without a network is
everything downstream of the feeds: the name and team keys the providers are
joined on, the roster assembly `build_roster` performs on a Sleeper reference
built from fixture JSON, and the slot rules `optimize` applies.

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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lineup_optimizer as lo
import sleeper_fixtures as sf
from pipeline import notebook as nb

KICKOFF = "2026-09-13T17:00:00Z"


def _projection(player, team, points, quality="good", start=KICKOFF, feed="bovada"):
    return nb.Projection(
        event_id="e-" + player, matchup=team, start_time_utc=start, team=team,
        player=player, position="WR", fantasy_points=points, quality=quality,
        stat_means={stat: 1.0 for stats in nb.REQUIRED_PROJECTION_COMPONENTS.values()
                    for stat in stats},
        sources={"receiving_yards": feed + " total"},
    )


def _roster_frame():
    rows = [
        ("QB1", "QB", 20.0), ("QB2", "QB", 15.0),
        ("RB1", "RB", 16.0), ("RB2", "RB", 14.0), ("RB3", "RB", 10.0),
        ("WR1", "WR", 15.0), ("WR2", "WR", 13.0), ("WR3", "WR", 12.0),
        ("TE1", "TE", 9.0), ("TE2", "TE", 7.0), ("K1", "K", 8.0),
        ("DEF1", "DEF", 6.0),
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


def _sleeper():
    """Sleeper's view of the fixture week.

    Tee Higgins is ruled Out, Monday Guy plays in a game Yahoo does not price,
    Bye Week Guy has no projection, and Dallas carries a second kicker who must
    not reach the add pool.
    """
    rows = [
        sf.player("1", "Jaylen Waddle", "MIA", "WR", "LWR", 1),
        sf.player("2", "Tee Higgins", "CIN", "WR", "LWR", 1, injury="Out",
                  practice="DNP"),
        sf.player("3", "Dak Prescott", "DAL", "QB", "QB", 1),
        sf.player("4", "Breece Hall", "NYJ", "RB", "RB", 1),
        sf.player("5", "Harold Fannin", "CLE", "TE", "TE", 3),
        sf.player("6", "Monday Guy", "BUF", "WR", "SWR", 2),
        sf.player("7", "Brandon Aubrey", "DAL", "K", "K", 1),
        sf.player("8", "Backup Kicker", "DAL", "K", "K", 2),
        sf.player("9", "Bye Kicker", "SEA", "K", "K", 1),
        sf.player("10", "Bye Week Guy", "SEA", "RB", "RB", 1),
    ]
    points = {"1": 13.2, "2": 11.0, "3": 19.4, "4": 14.1, "5": 7.3, "6": 12.0,
              "7": 9.5, "8": 3.0}
    return sf.dump(rows), sf.projections(points)


def _context():
    reference, _ = sf.reference(*_sleeper())
    context = {"season": 2026, "week": 2, "season_type": "regular",
               "players_fetched_utc": "2026-09-10T12:00:00+00:00"}
    return lo.sleeper_context(reference, context, _yahoo_frame())


class NameAndTeamKeyTests(unittest.TestCase):
    def test_suffixes_and_punctuation_join_across_providers(self):
        # Yahoo prints "Harold Fannin Jr.", the roster holds "Harold Fannin Jr",
        # Sleeper holds "Harold Fannin". All three have to land on one key.
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

    def test_team_aliases_collapse_to_the_yahoo_code(self):
        self.assertEqual(lo.normalize_team("JAX"), "JAC")
        self.assertEqual(lo.normalize_team(" wsh "), "WAS")
        self.assertEqual(lo.normalize_team("OAK"), "LV")
        self.assertEqual(lo.normalize_team("KC"), "KC")
        self.assertEqual(lo.normalize_team(np.nan), "")


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

    def test_yahoo_position_corrects_a_roster_typo(self):
        roster = lo.build_roster(
            [{"Name": "Breece Hall", "Position": "WR"}],
            _yahoo_frame(), _context(),
        )
        row = roster.iloc[0]
        self.assertEqual(row["Position"], "RB")
        self.assertEqual(row["Configured_Position"], "WR")
        self.assertIn("position corrected WR->RB", lo.review(row))
        self.assertEqual(self.roster.loc["Jaylen Waddle", "Team"], "MIA")
        self.assertGreater(self.roster.loc["Jaylen Waddle", "FP"], 0)
        self.assertIn("Sleeper half-PPR projection", self.roster.loc["Jaylen Waddle", "Projection_Source"])

    def test_the_mean_is_sleepers_projection_not_yahoos(self):
        # The Yahoo fixture carries 13.5 for Waddle; Sleeper says 13.2.
        self.assertAlmostEqual(self.roster.loc["Jaylen Waddle", "FP"], 13.2)
        self.assertAlmostEqual(self.roster.loc["Dak Prescott", "FP"], 19.4)

    def test_a_frozen_mean_from_the_rankings_run_is_kept(self):
        yahoo = _yahoo_frame()
        yahoo["Projection_Frozen"] = True
        roster = lo.build_roster(self.configured, yahoo, _context()).set_index("Name")
        self.assertAlmostEqual(roster.loc["Jaylen Waddle", "FP"], 13.5)

    def test_kickers_come_from_sleeper(self):
        row = self.roster.loc["Brandon Aubrey"]
        self.assertAlmostEqual(row["FP"], 9.5)
        self.assertEqual(row["Depth_Rank"], 1)
        self.assertAlmostEqual(row["Projection_CV"], lo.KICKER_CV)
        self.assertIn("Sleeper", row["Projection_Source"])

    def test_a_game_yahoo_omits_is_still_projected(self):
        # Yahoo omits some games (a Monday-only slate, say). Sleeper projects
        # every game, so a rostered player there is not a zero.
        row = self.roster.loc["Monday Guy"]
        self.assertAlmostEqual(row["FP"], 12.0)
        self.assertIn("Sleeper", row["Projection_Source"])
        self.assertTrue(row["Projection_Available"])

    def test_a_player_on_no_schedule_stays_at_zero_and_is_flagged(self):
        row = self.roster.loc["Bye Week Guy"]
        self.assertEqual(row["FP"], 0)
        self.assertFalse(row["Projection_Available"])
        self.assertIn("no weekly projection/bye", lo.review(row))

    def test_depth_drives_the_calibrated_range(self):
        waddle = self.roster.loc["Jaylen Waddle"]
        self.assertEqual(waddle["Depth_Source"], "Sleeper depth chart")
        self.assertAlmostEqual(waddle["Projection_CV"], lo.CALIBRATED_CV["WR"][1])
        self.assertLess(waddle["Floor_P25"], waddle["FP"])
        self.assertGreater(waddle["Ceiling_P90"], waddle["FP"])
        # A WR3's band is wider than a WR1's at the same mean.
        wide = lo.CALIBRATED_CV["WR"][3]
        self.assertGreater(wide, waddle["Projection_CV"])

    def test_the_injury_status_reaches_the_review_column(self):
        row = self.roster.loc["Tee Higgins"]
        self.assertEqual(row["report_status"], "Out")
        self.assertEqual(row["practice_status"], "DNP")
        self.assertIn("unavailable: injury status Out", lo.review(row))

    def test_an_unavailable_player_is_reported_out(self):
        self.assertEqual(lo.reported_out(self.roster.reset_index(drop=True)), ["Tee Higgins"])

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
        # a roster where nothing matches Sleeper must still leave it defined.
        ctx = _context()
        ctx["reference"] = ctx["reference"].iloc[0:0]
        roster = lo.build_roster([{"Name": "Nobody Known", "Position": "RB"}],
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
        self.assertEqual(len(starters), 9)
        self.assertEqual(len(bench), len(self.roster) - 9)
        counts = starters["Slot"].value_counts().to_dict()
        self.assertEqual(counts, {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1, "FLEX": 1})
        self.assertEqual(len(set(starters["Name"])), 9)

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
        self.assertEqual(len(starters), 8)
        self.assertTrue(starters[starters.Slot.eq("K")].empty)


class AddPoolTests(unittest.TestCase):
    """The pool `site/lineup.html` adds a player from.

    A player swapped in on the page has to be priced by the run that published
    the lineup, not by a snapshot the browser kept, so the pool is built through
    `build_roster` rather than lifted out of the Yahoo feed.
    """

    def setUp(self):
        self.yahoo, self.context = _yahoo_frame(), _context()

    def test_entries_are_the_best_at_each_position_within_the_limit(self):
        entries = lo.pool_entries(self.yahoo, self.context, {"QB": 1, "WR": 1, "K": 0})
        self.assertEqual(entries, [{"Name": "Dak Prescott", "Position": "QB"},
                                   {"Name": "Jaylen Waddle", "Position": "WR"}])

    def test_kickers_come_from_sleeper_one_per_team(self):
        # Yahoo prices no kickers at all, so a K the page can add exists only if
        # Sleeper supplies him. The backup and the bye-week kicker stay out.
        entries = lo.pool_entries(self.yahoo, self.context, {"K": 32})
        self.assertEqual(entries, [{"Name": "Brandon Aubrey", "Position": "K"}])

    def test_no_sleeper_reference_costs_the_kickers_not_the_pool(self):
        context = dict(self.context, reference=self.context["reference"].iloc[:0])
        entries = lo.pool_entries(self.yahoo, context, {"QB": 1, "K": 32})
        self.assertEqual(entries, [{"Name": "Dak Prescott", "Position": "QB"}])

    def test_the_pool_is_priced_and_banded_like_the_roster(self):
        pool = lo.build_pool(self.yahoo, self.context,
                             {"QB": 5, "RB": 5, "WR": 5, "TE": 5, "K": 5})
        aubrey = pool[pool.Name.eq("Brandon Aubrey")].iloc[0]
        self.assertAlmostEqual(aubrey["FP"], 9.5)
        self.assertIn("Sleeper", aubrey["Projection_Source"])
        waddle = pool[pool.Name.eq("Jaylen Waddle")].iloc[0]
        self.assertAlmostEqual(waddle["FP"], 13.2)
        self.assertIn("Sleeper half-PPR projection", waddle["Projection_Source"])
        self.assertLess(waddle["Floor_P25"], waddle["FP"])
        self.assertGreater(waddle["Ceiling_P90"], waddle["FP"])
        self.assertEqual(waddle["Team"], "MIA")
        self.assertEqual(waddle["Opponent"], "BUF")

    def test_limits_are_applied_per_position_after_pricing(self):
        pool = lo.build_pool(self.yahoo, self.context,
                             {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "K": 1})
        counts = pool["Position"].value_counts().to_dict()
        self.assertEqual(counts, {"QB": 1, "RB": 1, "WR": 1, "TE": 1, "K": 1})
        # Ranked on the mean, so the WR that survives is the better one.
        self.assertEqual(pool[pool.Position.eq("WR")].iloc[0]["Name"], "Jaylen Waddle")
        self.assertTrue(pool["FP"].is_monotonic_decreasing)

    def test_an_empty_feed_gives_an_empty_pool_rather_than_raising(self):
        empty = self.yahoo.iloc[:0]
        context = dict(self.context, reference=self.context["reference"].iloc[:0])
        self.assertTrue(lo.build_pool(empty, context).empty)


@unittest.skip("sportsbook projection pipeline removed")
class MarketProjectionTests(unittest.TestCase):
    """The market step, driven with hand-built projections instead of feeds."""

    def setUp(self):
        self.yahoo = _yahoo_frame()

    def _apply(self, projections):
        return lo.apply_market_projections(self.yahoo, projections=projections)

    def test_an_accepted_outlier_is_audited_before_it_reaches_the_optimizer(self):
        out, audit = self._apply([_projection("Jaylen Waddle", "MIA", 17.25)])
        row = out[out.Feed_Name.eq("Jaylen Waddle")].iloc[0]
        self.assertAlmostEqual(row["Projected_FP"], 0.8 * 17.25 + 0.2 * 13.5)
        self.assertIn("market good 80% audit", row["Projection_Source"])
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
        self.assertAlmostEqual(row["FP"], 0.8 * 17.25 + 0.2 * 13.5)
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

    def test_the_shipped_roster_file_is_loaded_as_user_configuration(self):
        shipped = Path(__file__).resolve().parents[1] / "lineup_roster.json"
        if shipped.exists():
            document = json.loads(shipped.read_text(encoding="utf-8"))
            expected = [
                {"Name": row["Name"].strip(), "Position": row["Position"].upper()}
                for row in document["roster"]
            ]
            self.assertEqual(lo.load_roster(str(shipped)), expected)


class ReportedOutTests(unittest.TestCase):
    def test_it_benches_whoever_sleeper_says_cannot_play(self):
        # Questionable is a game-time decision, so only the unavailable reason
        # benches; the designation alone does not.
        roster = pd.DataFrame({
            "Name": ["A", "B", "C"],
            "report_status": ["IR", "Questionable", None],
            "Unavailable_Reason": ["injury status IR", None, None],
        })
        self.assertEqual(lo.reported_out(roster), ["A"])

    def test_a_roster_with_no_injury_column_reports_nobody(self):
        self.assertEqual(lo.reported_out(pd.DataFrame({"Name": ["A"]})), [])


class SelfTestTests(unittest.TestCase):
    def test_the_shipped_self_test_passes(self):
        lo.self_test()


if __name__ == "__main__":
    unittest.main()
