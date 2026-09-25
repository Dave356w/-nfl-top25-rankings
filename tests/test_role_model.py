"""Tests for role tiers, backup-QB filtering, RB-RB correlation and the ranking run.

Roles come from Sleeper's depth chart: its alignment slots (LWR/RWR/SWR) keep
three starting receivers as three starters rather than WR1 and two reserves.
The fixtures are Sleeper JSON pushed through the shipped transforms.
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import sleeper_fixtures as sf
from pipeline import notebook as nb

# (id, name, team, position, depth_chart_position, order, status, injury)
NE_CHART = [
    ("1", "A.J. Brown", "NE", "WR", "LWR", 1, "Active", None),
    ("2", "Romeo Doubs", "NE", "WR", "RWR", 1, "Active", None),
    ("3", "DeMario Douglas", "NE", "WR", "SWR", 1, "Active", None),
    ("4", "Mack Hollins", "NE", "WR", "LWR", 2, "Active", None),
    ("5", "Kyle Williams", "NE", "WR", "RWR", 2, "Active", None),
    ("6", "Drake Maye", "NE", "QB", "QB", 1, "Active", None),
    ("7", "Joshua Dobbs", "NE", "QB", "QB", 2, "Active", None),
    ("8", "Rhamondre Stevenson", "NE", "RB", "RB", 1, "Active", None),
    ("9", "Antonio Gibson", "NE", "RB", "RB", 2, "Active", None),
    ("10", "Jack Westover", "NE", "RB", "FB", 1, "Active", None),
]


def chart_dump(rows, teams=("NE",)):
    return sf.dump([sf.player(pid, name, team, pos, slot, order, status=status, injury=injury)
                    for pid, name, team, pos, slot, order, status, injury in rows], teams)


class SleeperRoleTests(unittest.TestCase):
    def setUp(self):
        points = {str(i): 20.0 - i for i in range(1, 11)}
        self.ref, _ = sf.reference(chart_dump(NE_CHART), sf.projections(points))
        self.ref = self.ref.set_index("Sleeper_Name")

    def test_parallel_starting_receivers_are_all_tier_one(self):
        for name in ("A.J. Brown", "Romeo Doubs", "DeMario Douglas"):
            self.assertEqual(self.ref.loc[name, "Role_Tier"], 1, name)
        self.assertEqual(self.ref.loc["Mack Hollins", "Role_Tier"], 2)

    def test_the_flat_depth_rank_orders_by_role_then_projection(self):
        ranks = self.ref[self.ref["Position"].eq("WR")]["Depth_Rank"].sort_values()
        self.assertEqual(list(ranks.index), ["A.J. Brown", "Romeo Doubs", "DeMario Douglas",
                                             "Mack Hollins", "Kyle Williams"])

    def test_a_fullback_is_a_specialist_behind_the_backs(self):
        self.assertEqual(nb.role_label("RB", 1, "FB"), "specialist")
        self.assertEqual(self.ref.loc["Jack Westover", "Depth_Rank"], 3)


class BackupQuarterbackTests(unittest.TestCase):
    def test_the_chart_backup_is_removed_and_the_starter_is_not(self):
        points = {str(i): 20.0 - i for i in range(1, 11)}
        ref, _ = sf.reference(chart_dump(NE_CHART), sf.projections(points))
        players = pd.DataFrame({
            "Name": ["Drake Maye", "Joshua Dobbs"], "Team": ["NE", "NE"],
            "Position": ["QB", "QB"], "Salary": [40, 10], "FPPG": [20.0, 5.0],
        })
        kept, removed = nb.apply_sleeper_reference(players, ref)
        kept, dropped = nb.apply_default_role_filters(kept)
        self.assertEqual(dropped, ["Joshua Dobbs"])
        self.assertEqual(list(kept["Name"]), ["Drake Maye"])
        self.assertTrue(removed.empty)


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

    The wiring between Sleeper's roles, availability and projections and the
    published view is the part that breaks silently: a missing column only shows
    up on a live slate at 13:00 UTC.
    """

    ROWS = [
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

    def yahoo_payload(self):
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
        } for index, (name, position, team, salary, fppg) in enumerate(self.ROWS, start=1)]
        return {
            "players": {"result": players},
            "salaryCapInfo": {"result": [{"singleGameSalaryCapMap": {"nfl.g.1": 125}}]},
        }

    def setUp(self):
        rows = NE_CHART + [
            ("11", "Sam Darnold", "SEA", "QB", "QB", 1, "Active", None),
            ("12", "Rashid Shaheed", "SEA", "WR", "LWR", 1, "Injured Reserve", "IR"),
        ]
        projected = {name: fppg + 0.5 for name, _, _, _, fppg in self.ROWS}
        points = {pid: projected.get(name, 1.0) for pid, name, *_ in rows}
        points.update({"NE": 7.5, "SEA": 7.5})
        patches = {
            "fetch_yahoo_data": lambda *a, **k: self.yahoo_payload(),
            "load_sleeper_reference": sf.stub_loader(
                chart_dump(rows, teams=("NE", "SEA")), sf.projections(points)),
            "display": lambda *a, **k: None,
        }
        for name, value in patches.items():
            self.addCleanup(setattr, nb, name, getattr(nb, name))
            setattr(nb, name, value)

    def run_rankings(self, top_n=10):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nb.run_position_rankings(top_n=top_n, export_csv=False)

    def test_it_publishes_a_role_for_every_ranked_player(self):
        view = self.run_rankings(top_n=5)["rankings"]["WR"]
        self.assertIn("Role", view.columns)
        self.assertIn("Role slot", view.columns)
        self.assertTrue(view["Role"].ne("").all())

        roles = dict(zip(view["Player"], view["Role"]))
        # The three parallel starters read as starters, not as WR1 and two reserves.
        for name in ("A.J. Brown", "Romeo Doubs", "DeMario Douglas"):
            self.assertEqual(roles[name], "starter", name)
        self.assertEqual(roles["Mack Hollins"], "rotation")

    def test_it_drops_the_unavailable_and_the_chart_backup(self):
        results = self.run_rankings()
        names = set(results["players"]["Name"])
        self.assertNotIn("Rashid Shaheed", names)   # injured reserve
        self.assertNotIn("Joshua Dobbs", names)     # unconfirmed backup quarterback
        self.assertIn("Drake Maye", names)
        removed = results["availability_removed"].set_index("Player")
        self.assertEqual(removed.loc["Rashid Shaheed", "Reason"], "injury status IR")

    def test_the_mean_is_sleepers_projection(self):
        players = self.run_rankings()["players"].set_index("Name")
        self.assertAlmostEqual(players.loc["DeMario Douglas", "Projected_FP"], 10.5)
        self.assertAlmostEqual(players.loc["Seattle", "Projected_FP"], 7.5)
        self.assertIn("Sleeper half-PPR projection", players.loc["Mack Hollins", "Projection_Source"])


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
