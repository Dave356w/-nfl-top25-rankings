"""Teammates move up the depth chart when a player ahead of them is out.

The published chart can lag an injury by days. Until it catches up, the backup
used to be projected as a backup and, at quarterback, removed outright by the
backup-QB filter, even on a slate where he is the only quarterback his team has.
"""

from __future__ import annotations

import sys
import unittest
import warnings
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_daily
from pipeline import notebook as nb
import test_role_model
from test_role_model import loaded_depth_chart


def injury_rows(*rows):
    """Injury report rows shaped like `fetch_nflverse_injury_report` output."""
    frame = pd.DataFrame(rows, columns=["team", "full_name", "report_status"])
    frame["_key"] = frame["team"] + "|" + frame["full_name"].map(nb.normalize_person_name)
    frame["report_primary_injury"] = "Knee"
    frame["practice_status"] = "Did Not Participate"
    return frame


class UnavailableKeyTests(unittest.TestCase):
    def test_out_and_injured_reserve_count_and_questionable_does_not(self):
        roster = pd.DataFrame(
            [("A.J. Brown", "NE", "RES"), ("Romeo Doubs", "NE", "ACT")],
            columns=["player_name", "team", "status"],
        )
        injuries = injury_rows(("NE", "Drake Maye", "Out"),
                               ("NE", "Romeo Doubs", "Questionable"))
        keys = nb.unavailable_chart_keys(roster, injuries)
        self.assertEqual(set(keys), {"NE|ajbrown", "NE|drakemaye"})
        self.assertEqual(keys["NE|drakemaye"], "ruled Out")

    def test_an_availability_override_keeps_a_player_on_the_chart(self):
        injuries = injury_rows(("NE", "Drake Maye", "Out"))
        self.addCleanup(setattr, nb, "AVAILABILITY_OVERRIDES", nb.AVAILABILITY_OVERRIDES)
        nb.AVAILABILITY_OVERRIDES = {"Drake Maye": True}
        self.assertEqual(nb.unavailable_chart_keys(None, injuries), {})


class PromotionTests(unittest.TestCase):
    def setUp(self):
        self.chart = loaded_depth_chart()

    def _row(self, chart, name):
        return chart[chart["player_name"].eq(name)].iloc[0]

    def test_the_backup_quarterback_becomes_the_starter(self):
        chart, promoted = nb.promote_past_unavailable(
            self.chart, {"NE|drakemaye": "ruled Out"}
        )
        self.assertNotIn("Drake Maye", set(chart["player_name"]))
        dobbs = self._row(chart, "Joshua Dobbs")
        self.assertEqual(int(dobbs["pos_rank"]), 1)
        self.assertEqual(int(dobbs["Role_Tier"]), 1)
        self.assertEqual(promoted["Player"].tolist(), ["Joshua Dobbs"])
        self.assertIn("Drake Maye (ruled Out)", promoted.iloc[0]["Replacing"])

    def test_only_the_injured_receivers_slot_moves_up(self):
        chart, promoted = nb.promote_past_unavailable(
            self.chart, {"NE|ajbrown": "reserve / injured reserve"}
        )
        # Hollins was the second man at Brown's slot; he now starts there.
        self.assertEqual(int(self._row(chart, "Mack Hollins")["Role_Tier"]), 1)
        # Williams is behind Doubs at a different slot and stays a backup there.
        self.assertEqual(int(self._row(chart, "Kyle Williams")["Role_Tier"]), 2)
        # The parallel starters keep their tier; only the flat rank compacts.
        self.assertEqual(int(self._row(chart, "Romeo Doubs")["Role_Tier"]), 1)
        self.assertEqual(int(self._row(chart, "Romeo Doubs")["pos_rank"]), 1)
        hollins = promoted.set_index("Player").loc["Mack Hollins"]
        self.assertEqual((hollins["Old tier"], hollins["New tier"]), (2, 1))

    def test_a_fullback_stays_behind_the_promoted_back(self):
        chart, _ = nb.promote_past_unavailable(
            self.chart, {"NE|rhamondrestevenson": "ruled Out"}
        )
        self.assertEqual(int(self._row(chart, "Antonio Gibson")["pos_rank"]), 1)
        self.assertEqual(int(self._row(chart, "Jack Westover")["pos_rank"]), 2)

    def test_a_chart_that_already_reflects_the_injury_is_unchanged(self):
        chart, promoted = nb.promote_past_unavailable(
            self.chart, {"NE|someonealreadygone": "ruled Out"}
        )
        self.assertIs(chart, self.chart)
        self.assertTrue(promoted.empty)

    def test_other_teams_are_untouched(self):
        other = pd.concat([self.chart, nb.add_slot_role_tiers(pd.DataFrame(
            [("SEA", "Sam Darnold", "QB", 9, 1), ("SEA", "Drew Lock", "QB", 9, 2)],
            columns=["team", "player_name", "pos_abb", "pos_slot", "pos_rank"],
        ))], ignore_index=True)
        chart, _ = nb.promote_past_unavailable(other, {"NE|drakemaye": "ruled Out"})
        self.assertEqual(int(self._row(chart, "Drew Lock")["Role_Tier"]), 2)


class InjuredStarterRankingTests(test_role_model.RankingRunTests):
    """The full ranking run with the NE starting quarterback ruled Out."""

    def setUp(self):
        super().setUp()
        injuries = injury_rows(("NE", "Drake Maye", "Out"))
        nb.fetch_nflverse_injury_report = lambda season, cfg=None: (injuries, 3)

    def _run(self, cfg=None):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return nb.run_position_rankings(top_n=10, cfg=cfg, export_csv=False)

    def test_the_backup_is_projected_as_the_starter(self):
        results = self._run()
        players = results["players"].set_index("Name")
        self.assertNotIn("Drake Maye", players.index)
        self.assertIn("Joshua Dobbs", players.index)
        dobbs = players.loc["Joshua Dobbs"]
        self.assertEqual(int(dobbs["Depth_Rank"]), 1)
        self.assertEqual(dobbs["Role_Label"], "starter")
        self.assertEqual(results["depth_promotions"]["Player"].tolist(), ["Joshua Dobbs"])

    def test_the_promotion_is_published(self):
        results = self._run()
        payload = run_daily.build_payload(results, 10, [])
        self.assertEqual(payload["depth_promotions"], [{
            "player": "Joshua Dobbs", "team": "NE", "position": "QB",
            "role": "starter", "replacing": "Drake Maye (ruled Out)",
        }])

    def test_promotion_can_be_switched_off(self):
        results = self._run(nb.replace(nb.CFG, promote_past_unavailable=False))
        self.assertNotIn("Joshua Dobbs", set(results["players"]["Name"]))

    # The inherited cases assume Maye plays; they are covered by RankingRunTests.
    test_it_publishes_a_role_for_every_ranked_player = None
    test_it_drops_the_unavailable_and_the_chart_backup = None
    test_a_parallel_starter_is_not_haircut = None


class InjuredReceiverRankingTests(test_role_model.RankingRunTests):
    """Salary order promotes too: the Out receiver no longer holds a rank."""

    def setUp(self):
        super().setUp()
        injuries = injury_rows(("NE", "A.J. Brown", "Out"))
        nb.fetch_nflverse_injury_report = lambda season, cfg=None: (injuries, 3)

    def test_the_second_man_at_his_slot_starts(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = nb.run_position_rankings(top_n=10, export_csv=False)
        players = results["players"].set_index("Name")
        self.assertEqual(players.loc["Mack Hollins", "Role_Label"], "starter")
        self.assertEqual(
            sorted(players.loc[players["Team"].eq("NE") & players["Position"].eq("WR"),
                               "Depth_Rank"].tolist()),
            [1, 2, 3],
        )

    test_it_publishes_a_role_for_every_ranked_player = None
    test_it_drops_the_unavailable_and_the_chart_backup = None
    test_a_parallel_starter_is_not_haircut = None


if __name__ == "__main__":
    unittest.main()
