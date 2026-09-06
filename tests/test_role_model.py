"""Tests for role tiers, availability, RB-RB correlation and market blending.

These cover the four v3.5 changes that alter what the optimizer is handed:
parallel starting slots instead of one flattened depth ladder, an unknown roster
status meaning unavailable, a role-conditional same-team RB-RB correlation, and
market means blended by quality rather than substituted wholesale.
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import notebook as nb


def depth_chart_rows():
    """A three-receiver set, a single-slot backfield, and a fullback.

    Mirrors the real nflverse layout: the three starting receivers hold slots 1,
    2 and 8 and carry flat ranks 1, 2 and 3, and the second man at slot 1 is flat
    rank 4.
    """
    return pd.DataFrame([
        # team, player, pos_abb, pos_slot, pos_rank
        ("NE", "A.J. Brown", "WR", 1, 1),
        ("NE", "Romeo Doubs", "WR", 2, 2),
        ("NE", "DeMario Douglas", "WR", 8, 3),
        ("NE", "Mack Hollins", "WR", 1, 4),
        ("NE", "Kyle Williams", "WR", 2, 5),
        ("NE", "Drake Maye", "QB", 9, 1),
        ("NE", "Joshua Dobbs", "QB", 9, 2),
        ("NE", "Rhamondre Stevenson", "RB", 11, 1),
        ("NE", "Antonio Gibson", "RB", 11, 2),
        ("NE", "Jack Westover", "FB", 12, 1),
    ], columns=["team", "player_name", "pos_abb", "pos_slot", "pos_rank"])


def loaded_depth_chart():
    frame = nb.rerank_aliased_positions(depth_chart_rows())
    frame = nb.add_slot_role_tiers(frame)
    frame["Position"] = frame["pos_abb"].replace(nb.NFLVERSE_POSITION_ALIASES)
    return frame


def player_pool(rows):
    """Build the player frame the role functions consume."""
    frame = pd.DataFrame(
        rows, columns=["Name", "Team", "Position", "Salary", "Projected_FP"]
    )
    frame["Depth_Rank"] = (
        frame.sort_values(["Team", "Position", "Projected_FP"], ascending=[True, True, False])
        .groupby(["Team", "Position"])
        .cumcount()
        + 1
    ).reindex(frame.index)
    frame["Depth_Source"] = "projection heuristic"
    frame["Projection_Source"] = "yahoo prior"
    frame["FPPG"] = frame["Projected_FP"]
    return frame


class SlotRoleTierTests(unittest.TestCase):
    def setUp(self):
        self.chart = loaded_depth_chart()

    def _tier(self, name):
        row = self.chart[self.chart["player_name"].eq(name)].iloc[0]
        return int(row["Role_Tier"])

    def test_parallel_receivers_are_all_tier_one(self):
        # The whole point: flat ranks 1, 2 and 3 are three starting spots, not a
        # starter and two deep reserves.
        for name in ("A.J. Brown", "Romeo Doubs", "DeMario Douglas"):
            self.assertEqual(self._tier(name), 1, name)

    def test_second_man_at_a_slot_is_tier_two(self):
        self.assertEqual(self._tier("Mack Hollins"), 2)
        self.assertEqual(self._tier("Kyle Williams"), 2)

    def test_single_slot_positions_keep_their_ordinal(self):
        self.assertEqual(self._tier("Drake Maye"), 1)
        self.assertEqual(self._tier("Joshua Dobbs"), 2)
        self.assertEqual(self._tier("Rhamondre Stevenson"), 1)
        self.assertEqual(self._tier("Antonio Gibson"), 2)

    def test_slot_label_distinguishes_parallel_spots(self):
        slots = {
            row["player_name"]: row["Role_Slot"]
            for _, row in self.chart[self.chart["pos_abb"].eq("WR")].iterrows()
        }
        self.assertEqual(slots["A.J. Brown"], "WR slot 1")
        self.assertEqual(slots["DeMario Douglas"], "WR slot 8")

    def test_a_fullback_is_a_specialist_not_a_third_back(self):
        row = self.chart[self.chart["player_name"].eq("Jack Westover")].iloc[0]
        # rerank_aliased_positions pushes him behind the real backs...
        self.assertGreater(int(row["pos_rank"]), 2)
        # ...and he is named for what he is rather than by an ordinal.
        self.assertEqual(nb.role_label("RB", row["Role_Tier"], row["pos_abb"]), "specialist")

    def test_labels_read_as_words(self):
        self.assertEqual(nb.role_label("WR", 1), "starter")
        self.assertEqual(nb.role_label("WR", 2), "rotation")
        self.assertEqual(nb.role_label("WR", 3), "backup")
        self.assertEqual(nb.role_label("WR", 7), "reserve")

    def test_a_chart_without_slots_falls_back_to_the_flat_rank(self):
        legacy = depth_chart_rows().drop(columns=["pos_slot"])
        tiers = nb.add_slot_role_tiers(legacy)
        douglas = tiers[tiers["player_name"].eq("DeMario Douglas")].iloc[0]
        self.assertEqual(int(douglas["Role_Tier"]), 3)


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.players = player_pool([
            ("A.J. Brown", "NE", "WR", 30, 14.0),
            ("Romeo Doubs", "NE", "WR", 20, 10.0),
            ("DeMario Douglas", "NE", "WR", 16, 8.0),
            ("Mack Hollins", "NE", "WR", 10, 4.0),
            ("Drake Maye", "NE", "QB", 40, 22.0),
            ("New England", "NE", "DEF", 12, 7.0),
        ])
        self.status = pd.DataFrame([
            ("A.J. Brown", "NE", "ACT"),
            ("Romeo Doubs", "NE", "RES"),
            ("DeMario Douglas", "NE", "ACT"),
            ("Drake Maye", "NE", "ACT"),
        ], columns=["player_name", "team", "status"])
        self.addCleanup(setattr, nb, "AVAILABILITY_OVERRIDES", nb.AVAILABILITY_OVERRIDES)

    def report(self, cfg=None):
        return nb.build_nflverse_role_report(
            self.players, loaded_depth_chart(), self.status, cfg or nb.CFG
        )

    def test_an_unmatched_player_is_not_assumed_active(self):
        report = self.report().set_index("Player")
        self.assertFalse(bool(report.loc["Mack Hollins", "Available"]))
        self.assertIn("no roster row", report.loc["Mack Hollins", "Status meaning"])

    def test_injured_reserve_is_unavailable_and_active_is_not(self):
        report = self.report().set_index("Player")
        self.assertFalse(bool(report.loc["Romeo Doubs", "Available"]))
        self.assertTrue(bool(report.loc["A.J. Brown", "Available"]))

    def test_team_defense_never_needs_a_roster_row(self):
        report = self.report().set_index("Player")
        self.assertTrue(bool(report.loc["New England", "Available"]))

    def test_an_override_reinstates_a_confirmed_elevation(self):
        nb.AVAILABILITY_OVERRIDES = {"Mack Hollins": True}
        report = self.report().set_index("Player")
        self.assertTrue(bool(report.loc["Mack Hollins", "Available"]))
        self.assertIn("override", report.loc["Mack Hollins", "Status meaning"])

    def test_an_override_can_scratch_a_player_the_feed_calls_active(self):
        nb.AVAILABILITY_OVERRIDES = {"A.J. Brown": False}
        report = self.report().set_index("Player")
        self.assertFalse(bool(report.loc["A.J. Brown", "Available"]))

    def test_keeping_unmatched_players_is_still_configurable(self):
        cfg = nb.replace(nb.CFG, nflverse_drop_unmatched=False)
        report = self.report(cfg).set_index("Player")
        self.assertTrue(bool(report.loc["Mack Hollins", "Available"]))

    def test_a_broken_join_does_not_empty_the_pool(self):
        # Nobody matches, so the feed is describing our name normalization rather
        # than who is playing. Dropping the slate on that basis would be worse
        # than running without the filter.
        report = nb.build_nflverse_role_report(
            self.players, loaded_depth_chart(), self.status.head(0), nb.CFG
        )
        self.assertLess(nb.roster_match_rate(report), nb.CFG.nflverse_min_match_rate)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            kept, blocked = nb.apply_nflverse_availability(self.players, report)
        self.assertEqual(len(kept), len(self.players))
        self.assertTrue(blocked.empty)
        self.assertTrue(any("match" in str(w.message) for w in caught))

    def test_a_working_join_still_drops_the_unavailable(self):
        report = self.report()
        self.assertGreaterEqual(nb.roster_match_rate(report), nb.CFG.nflverse_min_match_rate)
        kept, blocked = nb.apply_nflverse_availability(self.players, report)
        self.assertNotIn("Romeo Doubs", set(kept["Name"]))
        self.assertIn("Romeo Doubs", set(blocked["Player"]))


class OpportunityRankTests(unittest.TestCase):
    def setUp(self):
        self.players = player_pool([
            ("A.J. Brown", "NE", "WR", 30, 14.0),
            ("Romeo Doubs", "NE", "WR", 20, 9.0),
            ("DeMario Douglas", "NE", "WR", 16, 10.0),
            ("Mack Hollins", "NE", "WR", 10, 3.0),
            ("Rhamondre Stevenson", "NE", "RB", 28, 13.0),
            ("Antonio Gibson", "NE", "RB", 14, 6.0),
        ])
        self.report = nb.build_nflverse_role_report(
            self.players, loaded_depth_chart(), None, nb.CFG
        )

    def roles(self):
        with_roles, _ = nb.apply_nflverse_roles(self.players, self.report)
        return nb.apply_opportunity_ranks(with_roles)

    def test_the_chart_supplies_the_role_and_the_blend_supplies_the_rank(self):
        ranked = self.roles().set_index("Name")
        # Douglas is third on the flat chart and second by projection...
        self.assertEqual(int(ranked.loc["DeMario Douglas", "Chart_Rank"]), 3)
        # ...so the blend moves him ahead of Doubs, who is second on both counts
        # only by the chart.
        self.assertEqual(int(ranked.loc["DeMario Douglas", "Depth_Rank"]), 2)
        self.assertEqual(int(ranked.loc["Romeo Doubs", "Depth_Rank"]), 3)
        # But he is a starting slot receiver either way.
        self.assertEqual(ranked.loc["DeMario Douglas", "Role_Label"], "starter")
        self.assertEqual(ranked.loc["Mack Hollins", "Role_Label"], "rotation")

    def test_trusting_the_chart_alone_reproduces_the_published_order(self):
        cfg = nb.replace(nb.CFG, role_market_rank_weight=0.0)
        with_roles, _ = nb.apply_nflverse_roles(self.players, self.report)
        ranked = nb.apply_opportunity_ranks(with_roles, cfg).set_index("Name")
        self.assertEqual(int(ranked.loc["Romeo Doubs", "Depth_Rank"]), 2)
        self.assertEqual(int(ranked.loc["DeMario Douglas", "Depth_Rank"]), 3)

    def test_a_manual_override_is_left_alone(self):
        players = self.players.copy()
        mask = players["Name"].eq("Mack Hollins")
        players.loc[mask, "Depth_Rank"] = 1
        players.loc[mask, "Depth_Source"] = "manual override"
        with_roles, _ = nb.apply_nflverse_roles(players, self.report)
        ranked = nb.apply_opportunity_ranks(with_roles).set_index("Name")
        self.assertEqual(int(ranked.loc["Mack Hollins", "Depth_Rank"]), 1)
        self.assertEqual(ranked.loc["Mack Hollins", "Depth_Source"], "manual override")
        # The override is the user asserting a role, so the tier follows it and
        # the mean multiplier it drives is cleared with it.
        self.assertEqual(int(ranked.loc["Mack Hollins", "Role_Tier"]), 1)

    def test_a_split_backfield_keeps_the_chart_role_and_the_priced_order(self):
        # Seattle's published chart lists Holani first while the team says the two
        # will split the work, and the market prices Price higher. The chart keeps
        # the role structure; the market gets the opportunity ordinal.
        players = player_pool([
            ("Rhamondre Stevenson", "NE", "RB", 28, 6.98),
            ("Antonio Gibson", "NE", "RB", 22, 10.04),
        ])
        report = nb.build_nflverse_role_report(players, loaded_depth_chart(), None, nb.CFG)
        with_roles, _ = nb.apply_nflverse_roles(players, report)
        ranked = nb.apply_opportunity_ranks(with_roles).set_index("Name")
        self.assertEqual(int(ranked.loc["Rhamondre Stevenson", "Chart_Rank"]), 1)
        self.assertEqual(ranked.loc["Rhamondre Stevenson", "Role_Label"], "starter")
        self.assertEqual(ranked.loc["Antonio Gibson", "Role_Label"], "rotation")
        self.assertEqual(int(ranked.loc["Antonio Gibson", "Depth_Rank"]), 1)
        self.assertEqual(int(ranked.loc["Rhamondre Stevenson", "Depth_Rank"]), 2)

    def test_two_players_sharing_a_name_keep_their_own_teams_roles(self):
        # Name alone is not an identity on a full slate. Applying one player's
        # chart row to the other would hand a reserve a starter's mean.
        players = player_pool([
            ("A.J. Brown", "NE", "WR", 30, 14.0),
            ("A.J. Brown", "SEA", "WR", 8, 2.0),
        ])
        chart = pd.concat([
            loaded_depth_chart(),
            nb.add_slot_role_tiers(pd.DataFrame([
                ("SEA", "Jaxon Smith-Njigba", "WR", 1, 1),
                ("SEA", "A.J. Brown", "WR", 1, 5),
            ], columns=["team", "player_name", "pos_abb", "pos_slot", "pos_rank"])),
        ], ignore_index=True)
        chart["Position"] = chart["pos_abb"].replace(nb.NFLVERSE_POSITION_ALIASES)
        report = nb.build_nflverse_role_report(players, chart, None, nb.CFG)
        with_roles, _ = nb.apply_nflverse_roles(players, report)
        ranked = nb.apply_opportunity_ranks(with_roles).set_index("Team")
        self.assertEqual(ranked.loc["NE", "Role_Label"], "starter")
        self.assertEqual(ranked.loc["SEA", "Role_Label"], "rotation")
        self.assertEqual(int(ranked.loc["NE", "Chart_Rank"]), 1)
        self.assertEqual(int(ranked.loc["SEA", "Chart_Rank"]), 5)

    def test_an_unmatched_player_is_ranked_by_his_projection(self):
        players = player_pool([
            ("Nobody Known", "SEA", "WR", 12, 5.0),
            ("Also Unknown", "SEA", "WR", 18, 9.0),
        ])
        report = nb.build_nflverse_role_report(players, loaded_depth_chart(), None, nb.CFG)
        with_roles, _ = nb.apply_nflverse_roles(players, report)
        ranked = nb.apply_opportunity_ranks(with_roles).set_index("Name")
        self.assertEqual(int(ranked.loc["Also Unknown", "Depth_Rank"]), 1)
        self.assertEqual(ranked.loc["Also Unknown", "Depth_Source"], "projection heuristic")


class RoleMeanAdjustmentTests(unittest.TestCase):
    def frame(self, position, role_tier, depth_rank, source="yahoo prior", weight=np.nan):
        return pd.DataFrame({
            "Name": ["Someone"],
            "Position": [position],
            "Projected_FP": [10.0],
            "Projection_Source": [source],
            "Depth_Rank": [depth_rank],
            "Role_Tier": pd.array([role_tier], dtype="Int64"),
            "Market_Weight": [weight],
        })

    def test_a_parallel_starter_keeps_his_whole_projection(self):
        # Flat rank 4 in a four-receiver set, but tier 1 in his own slot: the
        # 0.77x participation haircut is not what is going on with this player.
        out = nb.apply_depth_mean_adjustments(self.frame("WR", 1, 4))
        self.assertAlmostEqual(float(out["Depth_Mean_Multiplier"].iloc[0]), 1.00)

    def test_a_genuine_fourth_receiver_still_takes_the_haircut(self):
        out = nb.apply_depth_mean_adjustments(self.frame("WR", 4, 4))
        self.assertAlmostEqual(float(out["Depth_Mean_Multiplier"].iloc[0]), 0.77)
        self.assertEqual(
            out["Projection_Adjustment"].iloc[0], "historical role mean adjustment"
        )

    def test_a_full_weight_market_mean_is_untouched(self):
        out = nb.apply_depth_mean_adjustments(
            self.frame("WR", 4, 4, source="market good: Bovada", weight=1.0)
        )
        self.assertAlmostEqual(float(out["Depth_Mean_Multiplier"].iloc[0]), 1.00)
        self.assertEqual(out["Projection_Adjustment"].iloc[0], "market mean retained")

    def test_a_partial_market_mean_haircuts_only_its_prior_share(self):
        out = nb.apply_depth_mean_adjustments(
            self.frame("WR", 4, 4, source="market td-estimate 35%: Bovada", weight=0.35)
        )
        # 0.35 believed outright + 0.65 of the prior carrying the 0.77 haircut.
        self.assertAlmostEqual(
            float(out["Depth_Mean_Multiplier"].iloc[0]), 0.35 + 0.65 * 0.77, places=6
        )
        self.assertIn("partial market mean", out["Projection_Adjustment"].iloc[0])


class RunningBackCorrelationTests(unittest.TestCase):
    def back(self, team, depth):
        return {"Position": "RB", "Team": team, "Depth_Rank": depth,
                "Player_Style": "standard"}

    def test_the_lead_pair_is_negative_rather_than_pooled_positive(self):
        value = nb.target_score_correlation(self.back("SEA", 1), self.back("SEA", 2))
        self.assertEqual(value, nb.SAME_TEAM_RB_RB_CORR[(1, 2)])
        self.assertLess(value, 0.0)

    def test_it_is_symmetric(self):
        self.assertEqual(
            nb.target_score_correlation(self.back("SEA", 3), self.back("SEA", 1)),
            nb.target_score_correlation(self.back("SEA", 1), self.back("SEA", 3)),
        )

    def test_ranks_below_the_bucket_floor_collapse_to_four(self):
        self.assertEqual(
            nb.same_team_rb_rb_correlation(self.back("SEA", 1), self.back("SEA", 9)),
            nb.SAME_TEAM_RB_RB_CORR[(1, 4)],
        )

    def test_two_backs_at_the_same_rank_fall_back_to_the_lead_pair(self):
        self.assertEqual(
            nb.same_team_rb_rb_correlation(self.back("SEA", 2), self.back("SEA", 2)),
            nb.SAME_TEAM_RB_RB_CORR[(1, 2)],
        )

    def test_backs_on_opposing_teams_are_not_splitting_anything(self):
        self.assertEqual(
            nb.target_score_correlation(self.back("SEA", 1), self.back("NE", 2)),
            nb.OPPOSING_OFFENSE_CORR[("RB", "RB")],
        )

    def test_the_relationship_is_named_in_the_sanity_report(self):
        self.assertIn("Same team RB-RB", nb.CALIBRATION_SANITY_RELATIONSHIPS)
        self.assertEqual(
            nb._pair_relationship(self.back("SEA", 1), self.back("SEA", 2)),
            "Same team RB-RB",
        )

    def test_every_published_pair_is_reachable(self):
        for first, second in nb.SAME_TEAM_RB_RB_CORR:
            self.assertEqual(
                nb.same_team_rb_rb_correlation(
                    self.back("SEA", first), self.back("SEA", second)
                ),
                nb.SAME_TEAM_RB_RB_CORR[(first, second)],
            )


class MarketBlendTests(unittest.TestCase):
    def setUp(self):
        self.players = pd.DataFrame({
            "Name": ["Priced Player", "Touchdown Only"],
            "Team": ["NE", "NE"],
            "Position": ["WR", "TE"],
            "Projected_FP": [6.0, 4.0],
            "Projection_Source": ["yahoo prior", "yahoo prior"],
        })
        self.report = pd.DataFrame({
            "Player": ["Priced Player", "Touchdown Only"],
            "Team": ["NE", "NE"],
            "Position": ["WR", "TE"],
            "Yahoo projection": [6.0, 4.0],
            "Market projection": [12.0, 14.0],
            "Market quality": ["good", "td-estimate"],
            "Market method": ["component sum", "td regression"],
            "Market feeds": ["Bovada", "Underdog"],
            "Market matched": [True, True],
            "Market accepted": [True, True],
            "Market reason": ["accepted", "accepted"],
        })

    def test_a_fully_priced_player_takes_the_market_mean_outright(self):
        out, _ = nb.apply_market_projection_means(self.players, self.report)
        row = out.set_index("Name").loc["Priced Player"]
        self.assertAlmostEqual(float(row["Projected_FP"]), 12.0)
        self.assertEqual(row["Projection_Source"], "market good: Bovada")

    def test_a_touchdown_only_estimate_moves_the_prior_instead(self):
        out, _ = nb.apply_market_projection_means(self.players, self.report)
        row = out.set_index("Name").loc["Touchdown Only"]
        weight = nb.MARKET_QUALITY_WEIGHT["td-estimate"]
        self.assertAlmostEqual(
            float(row["Projected_FP"]), weight * 14.0 + (1 - weight) * 4.0
        )
        self.assertLess(float(row["Projected_FP"]), 14.0)
        self.assertGreater(float(row["Projected_FP"]), 4.0)
        self.assertIn("35%", row["Projection_Source"])

    def test_the_weight_is_recorded_for_the_downstream_haircut(self):
        out, accepted = nb.apply_market_projection_means(self.players, self.report)
        weights = out.set_index("Name")["Market_Weight"]
        self.assertAlmostEqual(float(weights.loc["Priced Player"]), 1.0)
        self.assertAlmostEqual(float(weights.loc["Touchdown Only"]), 0.35)
        self.assertIn("Blended projection", accepted.columns)

    def test_an_unknown_quality_label_is_only_half_believed(self):
        self.assertAlmostEqual(
            nb.market_blend_weight("something new"), nb.DEFAULT_MARKET_QUALITY_WEIGHT
        )

    def test_the_review_shows_what_was_actually_used(self):
        view = nb.market_projection_review(self.report).set_index("Player")
        self.assertAlmostEqual(float(view.loc["Priced Player", "Blended projection"]), 12.0)
        self.assertLess(float(view.loc["Touchdown Only", "Blended projection"]), 14.0)


class BackupQuarterbackTests(unittest.TestCase):
    def pool(self):
        players = player_pool([
            ("Drake Maye", "NE", "QB", 40, 22.0),
            ("Joshua Dobbs", "NE", "QB", 10, 6.0),
        ])
        with_roles, _ = nb.apply_nflverse_roles(
            players,
            nb.build_nflverse_role_report(players, loaded_depth_chart(), None, nb.CFG),
        )
        return nb.apply_opportunity_ranks(with_roles)

    def test_the_chart_backup_is_removed_and_the_starter_is_not(self):
        kept, removed = nb.apply_default_role_filters(self.pool())
        self.assertEqual(removed, ["Joshua Dobbs"])
        self.assertEqual(list(kept["Name"]), ["Drake Maye"])


if __name__ == "__main__":
    unittest.main()


class CorrelationModelTests(unittest.TestCase):
    """The new RB-RB entry has to survive the lognormal and PSD machinery."""

    def pool(self):
        players = pd.DataFrame({
            "Name": ["QB1", "RB1", "RB2", "WR1", "DEF"],
            "Team": ["NE", "NE", "NE", "SEA", "SEA"],
            "Position": ["QB", "RB", "RB", "WR", "DEF"],
            "Depth_Rank": [1, 1, 2, 1, 1],
            "Projected_FP": [20.0, 13.0, 7.0, 11.0, 8.0],
            "Player_Style": ["standard"] * 5,
        })
        return players

    def test_the_requested_matrix_carries_the_fitted_pair(self):
        players = self.pool()
        matrix = nb.requested_score_correlation_matrix(players)
        self.assertAlmostEqual(matrix[1, 2], nb.SAME_TEAM_RB_RB_CORR[(1, 2)])
        self.assertAlmostEqual(matrix[2, 1], nb.SAME_TEAM_RB_RB_CORR[(1, 2)])

    def test_the_repaired_model_stays_close_to_the_target(self):
        model = nb.build_correlation_model(self.pool())
        eigenvalues = np.linalg.eigvalsh(model["latent_corr"])
        self.assertGreater(float(eigenvalues.min()), 0.0)
        self.assertLess(model["psd_max_score_adjustment"], 0.04)
        self.assertLess(model["score_corr"][1, 2], 0.0)

    def test_the_pair_shows_up_in_the_sanity_report(self):
        players = self.pool()
        cfg = nb.replace(nb.CFG, simulations=4_000)
        outcomes, model = nb.simulate_player_outcomes(players, cfg)
        summary, detail = nb.correlation_sanity_report(players, outcomes, model)
        self.assertIn("Same team RB-RB", set(summary["Relationship"]))
        row = summary[summary["Relationship"].eq("Same team RB-RB")].iloc[0]
        self.assertEqual(row["Sanity"], "PASS")


class RankingRunTests(unittest.TestCase):
    """Exercise `run_position_rankings` end to end with the feeds stubbed out.

    The wiring between the role pass, the opportunity rank, the mean adjustment
    and the published view is the part that breaks silently: a missing column
    only shows up on a live slate at 13:00 UTC.
    """

    def yahoo_payload(self):
        rows = [
            ("Drake Maye", "QB", "NE", 40, 21.0),
            ("Joshua Dobbs", "QB", "NE", 11, 5.0),
            ("A.J. Brown", "WR", "NE", 32, 15.0),
            ("Romeo Doubs", "WR", "NE", 22, 9.0),
            ("DeMario Douglas", "WR", "NE", 18, 10.0),
            ("Mack Hollins", "WR", "NE", 10, 3.0),
            ("Rhamondre Stevenson", "RB", "NE", 27, 13.0),
            ("Antonio Gibson", "RB", "NE", 15, 6.0),
            ("Sam Darnold", "QB", "SEA", 30, 17.0),
            ("Rashid Shaheed", "WR", "SEA", 21, 11.0),
            ("New England", "DEF", "NE", 12, 7.0),
            ("Seattle", "DEF", "SEA", 12, 7.0),
        ]
        players = [{
            "name": name,
            "position": position,
            "team": team,
            "salary": salary,
            "fppg": fppg,
            "gameCode": "nfl.g.1",
            "gameStartTime": "2026-09-13T17:00:00Z",
            "homeTeam": "NE",
            "awayTeam": "SEA",
            "playerCode": f"nfl.p.{index}",
        } for index, (name, position, team, salary, fppg) in enumerate(rows, start=1)]
        return {
            "players": {"result": players},
            "salaryCapInfo": {"result": [{"singleGameSalaryCapMap": {"nfl.g.1": 125}}]},
        }

    def setUp(self):
        roster = pd.DataFrame(
            [(name, "NE", "ACT") for name in (
                "Drake Maye", "Joshua Dobbs", "A.J. Brown", "Romeo Doubs",
                "DeMario Douglas", "Mack Hollins", "Rhamondre Stevenson",
                "Antonio Gibson",
            )] + [("Sam Darnold", "SEA", "ACT"), ("Rashid Shaheed", "SEA", "RES")],
            columns=["player_name", "team", "status"],
        )
        chart = loaded_depth_chart()
        chart = pd.concat([chart, nb.add_slot_role_tiers(pd.DataFrame([
            ("SEA", "Sam Darnold", "QB", 9, 1),
            ("SEA", "Rashid Shaheed", "WR", 1, 1),
        ], columns=["team", "player_name", "pos_abb", "pos_slot", "pos_rank"]))],
            ignore_index=True)
        chart["Position"] = chart["pos_abb"].replace(nb.NFLVERSE_POSITION_ALIASES)

        patches = {
            "fetch_yahoo_data": lambda *a, **k: self.yahoo_payload(),
            "load_market_projection_reference": lambda cfg=None: (
                [], {"feeds": [], "notes": ["stubbed"], "logit_vig": float("nan"),
                     "calibration_pairs": 0},
            ),
            "load_nflverse_reference": lambda players, season, cfg=None: (
                chart, roster, None, ["stubbed"],
            ),
            "display": lambda *a, **k: None,
        }
        for name, value in patches.items():
            self.addCleanup(setattr, nb, name, getattr(nb, name))
            setattr(nb, name, value)

    def test_it_publishes_a_role_for_every_ranked_player(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = nb.run_position_rankings(top_n=5, export_csv=False)
        view = results["rankings"]["WR"]
        self.assertIn("Role", view.columns)
        self.assertIn("Role slot", view.columns)
        self.assertTrue(view["Role"].ne("").all())

        roles = dict(zip(view["Player"], view["Role"]))
        # The three parallel starters read as starters, not as WR1 and two reserves.
        for name in ("A.J. Brown", "Romeo Doubs", "DeMario Douglas"):
            self.assertEqual(roles[name], "starter", name)
        self.assertEqual(roles["Mack Hollins"], "rotation")

    def test_it_drops_the_unavailable_and_the_chart_backup(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = nb.run_position_rankings(top_n=10, export_csv=False)
        names = set(results["players"]["Name"])
        self.assertNotIn("Rashid Shaheed", names)   # reserve / injured reserve
        self.assertNotIn("Joshua Dobbs", names)     # unconfirmed backup quarterback
        self.assertIn("Drake Maye", names)

    def test_a_parallel_starter_is_not_haircut(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = nb.run_position_rankings(top_n=10, export_csv=False)
        players = results["players"].set_index("Name")
        self.assertAlmostEqual(
            float(players.loc["DeMario Douglas", "Depth_Mean_Multiplier"]), 1.00
        )
        self.assertAlmostEqual(
            float(players.loc["Mack Hollins", "Depth_Mean_Multiplier"]),
            nb.DEPTH_MEAN_MULTIPLIER["WR"][2],
        )


class CandidateScoringTests(unittest.TestCase):
    """The scoring kernel carries scores as (candidates x scenarios).

    The layout change is a performance one, so what needs pinning is that the
    numbers it produces still agree with an exact float64 computation of the
    same quantities -- and that they no longer depend on the layout, which is
    what lets a float64 implementation elsewhere reproduce them.
    """

    def setUp(self):
        rng = np.random.default_rng(11)
        self.n_players, self.n_scenarios = 9, 4_000
        self.outcomes = rng.lognormal(
            2.0, 0.6, size=(self.n_scenarios, self.n_players)
        ).astype(np.float32)
        combos = [(0, 1, 2, 3, 4), (0, 1, 2, 3, 5), (2, 3, 4, 5, 6),
                  (1, 4, 5, 6, 7), (0, 2, 4, 6, 8)]
        self.candidates = pd.DataFrame({
            "Player_Ids": combos,
            "Superstar_Id": [ids[0] for ids in combos],
            "Salary": [100.0] * len(combos),
            "Expected_FP": [50.0] * len(combos),
            "Analytic_SD": [10.0] * len(combos),
            "Pre_Sim_Score": [60.0] * len(combos),
        })
        self.cfg = nb.replace(nb.CFG, simulations=self.n_scenarios, lineup_size=5)

    def exact_scores(self):
        """Every lineup's score in every scenario, in full float64."""
        weights = np.zeros((len(self.candidates), self.n_players))
        for row, (ids, superstar) in enumerate(
            zip(self.candidates["Player_Ids"], self.candidates["Superstar_Id"])
        ):
            weights[row, list(ids)] = 1.0
            weights[row, superstar] += 0.5
        return weights @ self.outcomes.T.astype(np.float64)

    def test_the_weight_block_is_candidates_by_players(self):
        ids, superstars = nb._lineup_arrays(self.candidates, 5)
        block = nb._weight_matrix(ids, superstars, 0, len(self.candidates), self.n_players)
        self.assertEqual(block.shape, (len(self.candidates), self.n_players))
        # One 1.5 slot and four 1.0 slots per lineup.
        self.assertTrue(np.allclose(block.sum(axis=1), 5.5))
        self.assertAlmostEqual(float(block[0, self.candidates["Superstar_Id"].iloc[0]]), 1.5)

    def test_the_summaries_match_an_exact_float64_computation(self):
        scored = nb.score_candidates_shared_scenarios(
            self.candidates, self.outcomes, self.cfg, batch_size=2
        )
        exact = self.exact_scores()
        keyed = {tuple(ids): row for row, ids in enumerate(scored["Player_Ids"])}
        for row, ids in enumerate(self.candidates["Player_Ids"]):
            got = scored.iloc[keyed[tuple(ids)]]
            self.assertAlmostEqual(float(got["Sim_Mean"]), exact[row].mean(), places=4)
            self.assertAlmostEqual(float(got["Sim_SD"]), exact[row].std(), places=4)
            self.assertAlmostEqual(
                float(got["Ceiling_P90"]), np.quantile(exact[row], 0.90), places=3
            )
            self.assertAlmostEqual(
                float(got["Floor_P25"]), np.quantile(exact[row], 0.25), places=3
            )

    def test_the_batch_size_does_not_change_the_answer(self):
        """Batching is a memory decision, not a modelling one.

        Not bit-for-bit, though, and the tolerances say why. The scores are a
        float32 matrix product, and BLAS reaches a one-row product by a
        different path than a five-row one, so the two batchings can differ in
        the last couple of float32 digits. The summaries inherit that. The rates
        inherit something coarser: they count scenarios against a threshold, so
        a difference far below float32 precision can still move a scenario
        across it and change a rate by one scenario's worth.
        """
        def by_lineup(batch_size):
            scored = nb.score_candidates_shared_scenarios(
                self.candidates, self.outcomes, self.cfg, batch_size=batch_size
            )
            order = np.argsort([str(ids) for ids in scored["Player_Ids"]])
            return scored.iloc[order].reset_index(drop=True)

        small, large = by_lineup(1), by_lineup(512)
        for column in ("Sim_Mean", "Sim_SD", "Ceiling_P90", "Floor_P25"):
            np.testing.assert_allclose(
                small[column].to_numpy(), large[column].to_numpy(),
                rtol=1e-5, atol=1e-7, err_msg=column,
            )
        one_scenario = 1.0 / self.n_scenarios
        for column in ("Near_Optimal_Rate", "Win_Rate"):
            np.testing.assert_allclose(
                small[column].to_numpy(), large[column].to_numpy(),
                rtol=0, atol=3 * one_scenario, err_msg=column,
            )

    def test_win_rate_is_shared_across_the_same_scenarios(self):
        scored = nb.score_candidates_shared_scenarios(
            self.candidates, self.outcomes, self.cfg, batch_size=2
        )
        # One lineup wins each scenario, so the rates sum to one -- except that
        # the winner is decided by np.isclose, so two lineups within its
        # tolerance in the same scenario are both counted. That can only push
        # the total above one, and only by whole scenarios.
        total = float(scored["Win_Rate"].sum())
        self.assertGreaterEqual(total, 1.0 - 1e-9)
        self.assertLessEqual(total, 1.0 + 10.0 / self.n_scenarios)
