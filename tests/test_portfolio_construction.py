import json
import shutil
import subprocess
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from pipeline import portfolio_construction as pc

ROOT = Path(__file__).resolve().parents[1]
NODE = shutil.which("node")


def _cfg(entries=4, player_exposure=1.0, superstar_exposure=1.0, shared=4):
    return SimpleNamespace(
        lineup_size=5,
        tournament_lineups=entries,
        max_player_exposure=player_exposure,
        max_superstar_exposure=superstar_exposure,
        max_shared_players=shared,
    )


def _exposure_starvation_fixture():
    players = pd.DataFrame({
        "Position": ["QB", "QB", "WR", "WR", "WR", "WR", "RB", "RB"],
    })
    rules = [
        {
            "name": "2QB / 2WR / 1RB",
            "count": 2,
            "positions": {"QB": 2, "RB": 1, "WR": 2, "TE": 0, "DEF": 0},
        },
        {
            "name": "1QB / 3WR / 1RB",
            "count": 2,
            "positions": {"QB": 1, "RB": 1, "WR": 3, "TE": 0, "DEF": 0},
        },
    ]
    scored = pd.DataFrame([
        # The first two scarce 1-QB candidates both use QB 0.  Taking both would
        # strand the two future 2-QB slots under a 75% (3 of 4) exposure cap.
        {"Player_Ids": (0, 2, 3, 4, 6), "Superstar_Id": 0, "Tournament_Score": 100.0},
        {"Player_Ids": (0, 2, 3, 5, 7), "Superstar_Id": 0, "Tournament_Score": 99.0},
        {"Player_Ids": (1, 2, 4, 5, 6), "Superstar_Id": 1, "Tournament_Score": 98.0},
        # Every 2-QB candidate necessarily consumes both QBs.
        {"Player_Ids": (0, 1, 2, 3, 6), "Superstar_Id": 0, "Tournament_Score": 90.0},
        {"Player_Ids": (0, 1, 2, 4, 7), "Superstar_Id": 1, "Tournament_Score": 89.0},
        {"Player_Ids": (0, 1, 3, 5, 6), "Superstar_Id": 0, "Tournament_Score": 88.0},
        {"Player_Ids": (0, 1, 4, 5, 7), "Superstar_Id": 1, "Tournament_Score": 87.0},
    ])
    return players, rules, scored


class PortfolioConstructionTests(unittest.TestCase):
    def test_default_construction_mix_is_twenty_entries(self):
        rules = pc.serializable_rules()
        self.assertEqual(sum(rule["count"] for rule in rules), 20)
        self.assertEqual(pc.apportioned_rule_counts(rules, 20), [4, 3, 4, 3, 2, 2, 2])
        for rule in rules:
            self.assertEqual(sum(rule["positions"].values()), 5)

    def test_rule_counts_scale_to_requested_portfolio_size(self):
        rules = pc.serializable_rules()
        counts = pc.apportioned_rule_counts(rules, 10)
        self.assertEqual(sum(counts), 10)
        self.assertTrue(all(count >= 0 for count in counts))
        self.assertEqual(len(pc.rule_schedule(rules, 10)), 10)

    def test_selector_assigns_different_rules_per_entry(self):
        players = pd.DataFrame({
            "Position": ["QB", "QB", "WR", "WR", "RB", "TE", "DEF", "WR", "RB", "WR"],
        })
        rules = [
            {
                "name": "2QB / 2WR / 1RB",
                "count": 1,
                "positions": {"QB": 2, "RB": 1, "WR": 2, "TE": 0, "DEF": 0},
            },
            {
                "name": "1QB / 2WR / 1RB / 1TE",
                "count": 1,
                "positions": {"QB": 1, "RB": 1, "WR": 2, "TE": 1, "DEF": 0},
            },
        ]
        scored = pd.DataFrame([
            {"Player_Ids": (0, 1, 2, 3, 4), "Superstar_Id": 0, "Tournament_Score": 10.0},
            {"Player_Ids": (0, 2, 3, 4, 5), "Superstar_Id": 0, "Tournament_Score": 9.0},
            {"Player_Ids": (0, 1, 2, 7, 8), "Superstar_Id": 1, "Tournament_Score": 8.0},
            {"Player_Ids": (1, 2, 7, 8, 5), "Superstar_Id": 1, "Tournament_Score": 7.0},
        ])

        portfolio = pc.select_portfolio(scored, players, _cfg(4), rules)

        self.assertEqual(len(portfolio), 4)
        self.assertEqual(Counter(portfolio["Construction_Rule"]), {
            "2QB / 2WR / 1RB": 2,
            "1QB / 2WR / 1RB / 1TE": 2,
        })
        for _, row in portfolio.iterrows():
            rule = next(rule for rule in rules if rule["name"] == row["Construction_Rule"])
            self.assertTrue(pc.lineup_matches_rule(players, row["Player_Ids"], rule))

    def test_selector_reserves_mandatory_qb_exposure_for_future_slots(self):
        players, rules, scored = _exposure_starvation_fixture()
        portfolio = pc.select_portfolio(
            scored, players, _cfg(4, player_exposure=0.75), rules
        )

        self.assertEqual(len(portfolio), 4)
        self.assertEqual(portfolio.attrs["construction_unfilled"], {})
        counts = Counter(player for ids in portfolio["Player_Ids"] for player in ids)
        self.assertLessEqual(counts[0], 3)
        self.assertLessEqual(counts[1], 3)
        self.assertEqual(Counter(portfolio["Construction_Rule"]), {
            "2QB / 2WR / 1RB": 2,
            "1QB / 3WR / 1RB": 2,
        })

    def test_single_entry_uses_best_lineup_without_forcing_first_archetype(self):
        players = pd.DataFrame({
            "Position": ["QB", "QB", "WR", "WR", "WR", "RB", "TE"]
        })
        rules = [{
            "name": "2QB only",
            "count": 1,
            "positions": {"QB": 2},
        }]
        scored = pd.DataFrame([
            {"Player_Ids": (0, 2, 3, 4, 5), "Superstar_Id": 0, "Tournament_Score": 10.0},
            {"Player_Ids": (0, 1, 2, 3, 5), "Superstar_Id": 0, "Tournament_Score": 9.0},
        ])

        portfolio = pc.select_portfolio(scored, players, _cfg(1), rules)

        self.assertEqual(len(portfolio), 1)
        self.assertEqual(tuple(portfolio.iloc[0]["Player_Ids"]), (0, 2, 3, 4, 5))
        self.assertNotIn("Construction_Rule", portfolio.columns)

    def test_selector_reports_genuinely_unfilled_rule_slots(self):
        players = pd.DataFrame({"Position": ["QB", "WR", "WR", "RB", "TE"]})
        scored = pd.DataFrame([
            {"Player_Ids": (0, 1, 2, 3, 4), "Superstar_Id": 0, "Tournament_Score": 1.0},
        ])
        rules = [{
            "name": "2QB required",
            "count": 1,
            "positions": {"QB": 2},
        }]

        with self.assertWarnsRegex(UserWarning, "Built 0 of 2"):
            portfolio = pc.select_portfolio(scored, players, _cfg(2), rules)

        self.assertTrue(portfolio.empty)
        self.assertEqual(
            portfolio.attrs["construction_unfilled"], {"2QB required": 2}
        )

    @unittest.skipIf(NODE is None, "Node is required to test the browser selector")
    def test_browser_selector_reserves_the_same_mandatory_exposure(self):
        script = r'''
const worker = require(process.argv[1]);
const rules = [
  {name:"2QB / 2WR / 1RB", count:2, positions:{QB:2,RB:1,WR:2,TE:0,DEF:0}},
  {name:"1QB / 3WR / 1RB", count:2, positions:{QB:1,RB:1,WR:3,TE:0,DEF:0}},
];
worker.setModel({
  players:["QB","QB","WR","WR","WR","WR","RB","RB"].map(pos => ({pos})),
  settings:{portfolio_construction_rules:rules}
});
const lineups = [
  [0,2,3,4,6],[0,2,3,5,7],[1,2,4,5,6],
  [0,1,2,3,6],[0,1,2,4,7],[0,1,3,5,6],[0,1,4,5,7]
];
const scored = {
  total: lineups.length,
  ids: Int32Array.from(lineups.flat()),
  superstars: Int32Array.from([0,0,1,0,1,0,1]),
  salary: Float64Array.from(lineups.map(() => 100)),
  expected: Float64Array.from(lineups.map(() => 50)),
  mean: Float64Array.from(lineups.map(() => 50)),
  sd: Float64Array.from(lineups.map(() => 10)),
  p25: Float64Array.from(lineups.map(() => 40)),
  p90: Float64Array.from(lineups.map(() => 70)),
  p95: Float64Array.from(lineups.map(() => 80)),
  nearRate: Float64Array.from(lineups.map(() => 0.01)),
  winRate: Float64Array.from(lineups.map(() => 0.001)),
  tournament: Float64Array.from([100,99,98,90,89,88,87]),
};
const options = {
  entries:4, maxPlayerExposure:0.75, maxSuperstarExposure:1,
  maxShared:4, constructionRules:rules
};
const built = worker.portfolio(scored, [0,1,2,3,4,5,6], options);
process.stdout.write(JSON.stringify({chosen:built.chosen, rules:built.rules, unfilled:built.unfilled}));
'''
        result = subprocess.run(
            [NODE, "-e", script, str(ROOT / "site" / "showdown-worker.js")],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        built = json.loads(result.stdout)
        self.assertEqual(len(built["chosen"]), 4)
        self.assertEqual(built["unfilled"], {})
        self.assertIn(
            2, built["chosen"],
            "QB-1 alternative must replace the second QB-0 lineup",
        )


if __name__ == "__main__":
    unittest.main()
