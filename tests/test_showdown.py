"""Tests for the published showdown model and the browser optimizer.

Two things need pinning. The payload is a contract with `site/showdown.html`,
so every key the page reads has to be there and has to survive `json.dumps`.
And the JavaScript in `site/showdown-worker.js` is a second implementation of
the pipeline's own enumeration and scoring, so it has to agree with the Python
on everything that is not a random draw: the count of Yahoo-valid rosters, a
lineup's analytic expectation, and its analytic variance.

The JavaScript half needs Node. Where Node is missing the test skips rather
than failing, because the Python half of the repo must stay installable without
it -- GitHub's ubuntu runners have it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import notebook as nb
from pipeline import showdown
import run_showdown

NODE = shutil.which("node")

POOL = [
    ("Drake Maye", "QB", "NE", 40, 21.0), ("Josh Dobbs", "QB", "NE", 11, 5.0),
    ("A.J. Brown", "WR", "NE", 32, 15.0), ("Romeo Doubs", "WR", "NE", 22, 9.0),
    ("DeMario Douglas", "WR", "NE", 18, 10.0), ("Mack Hollins", "WR", "NE", 10, 3.0),
    ("Hunter Henry", "TE", "NE", 16, 7.0),
    ("Rhamondre Stevenson", "RB", "NE", 27, 13.0), ("Antonio Gibson", "RB", "NE", 15, 6.0),
    ("Sam Darnold", "QB", "SEA", 30, 17.0), ("Rashid Shaheed", "WR", "SEA", 21, 11.0),
    ("Jaxon Smith-Njigba", "WR", "SEA", 29, 14.0), ("Cooper Kupp", "WR", "SEA", 19, 8.5),
    ("AJ Barner", "TE", "SEA", 12, 5.5), ("Kenneth Walker", "RB", "SEA", 25, 12.0),
    ("Zach Charbonnet", "RB", "SEA", 14, 6.5),
    ("New England", "DEF", "NE", 12, 7.0), ("Seattle", "DEF", "SEA", 12, 7.0),
]

CHART = [
    ("NE", "Drake Maye", "QB", 9, 1), ("NE", "Josh Dobbs", "QB", 9, 2),
    ("NE", "A.J. Brown", "WR", 1, 1), ("NE", "Romeo Doubs", "WR", 2, 2),
    ("NE", "DeMario Douglas", "WR", 8, 3), ("NE", "Mack Hollins", "WR", 1, 4),
    ("NE", "Hunter Henry", "TE", 10, 1),
    ("NE", "Rhamondre Stevenson", "RB", 11, 1), ("NE", "Antonio Gibson", "RB", 11, 2),
    ("SEA", "Sam Darnold", "QB", 9, 1), ("SEA", "Jaxon Smith-Njigba", "WR", 1, 1),
    ("SEA", "Rashid Shaheed", "WR", 2, 2), ("SEA", "Cooper Kupp", "WR", 8, 3),
    ("SEA", "AJ Barner", "TE", 10, 1), ("SEA", "Kenneth Walker", "RB", 11, 1),
    ("SEA", "Zach Charbonnet", "RB", 11, 2),
]


def yahoo_payload():
    players = [{
        "name": name, "position": position, "team": team, "salary": salary,
        "fppg": fppg, "gameCode": "nfl.g.1",
        "gameStartTime": "2026-09-13T17:00:00Z",
        "homeTeam": "NE", "awayTeam": "SEA", "playerCode": f"nfl.p.{index}",
    } for index, (name, position, team, salary, fppg) in enumerate(POOL, start=1)]
    return {
        "players": {"result": players},
        "salaryCapInfo": {"result": [{"singleGameSalaryCapMap": {"nfl.g.1": 125}}]},
    }


def depth_chart():
    frame = nb.add_slot_role_tiers(pd.DataFrame(
        CHART, columns=["team", "player_name", "pos_abb", "pos_slot", "pos_rank"]
    ))
    frame["Position"] = frame["pos_abb"].replace(nb.NFLVERSE_POSITION_ALIASES)
    return frame


def roster_status():
    return pd.DataFrame(
        [(name, team, "ACT") for name, position, team, _, _ in POOL if position != "DEF"],
        columns=["player_name", "team", "status"],
    )


class _StubbedFeeds:
    """Patch the three network entry points for the duration of a test."""

    PATCHES = {
        "fetch_yahoo_data": lambda *a, **k: yahoo_payload(),
        "load_market_projection_reference": lambda cfg=None: (
            [], {"feeds": ["Bovada"], "notes": ["stubbed"], "logit_vig": 0.17,
                 "calibration_pairs": 41},
        ),
        "display": lambda *a, **k: None,
    }

    def install(self, case):
        chart, roster = depth_chart(), roster_status()
        patches = dict(self.PATCHES)
        patches["load_nflverse_reference"] = (
            lambda players, season, cfg=None: (chart, roster, None, ["stubbed"])
        )
        for name, value in patches.items():
            case.addCleanup(setattr, nb, name, getattr(nb, name))
            setattr(nb, name, value)


def build_payload(simulations=4_000, optimize=True):
    cfg = nb.replace(nb.CFG, simulations=simulations)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        payloads, index = showdown.build_slate(cfg, optimize=optimize)
    return payloads, index


class PayloadShapeTests(unittest.TestCase):
    def setUp(self):
        _StubbedFeeds().install(self)
        self.payloads, self.index = build_payload()
        self.payload = self.payloads[0]

    def test_it_serializes_without_nan(self):
        # json.dumps emits a bare NaN token, which JSON.parse rejects.
        for document in (self.payload, self.index):
            self.assertNotIn("NaN", json.dumps(document))

    def test_it_carries_every_key_the_page_reads(self):
        for key in ("schema", "game_id", "matchup", "away", "home", "kickoff_utc",
                    "salary_cap", "settings", "players", "latent", "model",
                    "dropped_from_pool", "reference"):
            self.assertIn(key, self.payload)
        for key in ("schema", "status", "generated_utc", "games", "skipped",
                    "market", "availability_removed"):
            self.assertIn(key, self.index)
        for key in showdown.EXPOSED_SETTINGS:
            self.assertIn(key, self.payload["settings"])

    def test_every_player_carries_the_fields_the_page_renders(self):
        for player in self.payload["players"]:
            self.assertEqual(set(player), set(showdown.PLAYER_FIELDS))
            self.assertGreater(player["salary"], 0)
            self.assertGreater(player["cv"], 0)

    def test_the_latent_triangle_is_the_right_length(self):
        n = len(self.payload["players"])
        self.assertEqual(len(self.payload["latent"]), n * (n - 1) // 2)
        self.assertTrue(all(abs(value) <= 1 for value in self.payload["latent"]))

    def test_the_triangle_rebuilds_the_matrix_the_pipeline_repaired(self):
        # The page walks this order to rebuild the matrix; if the order here and
        # the order there ever disagreed the correlations would be transposed
        # onto the wrong pairs and nothing would visibly break.
        n = len(self.payload["players"])
        rebuilt = np.eye(n)
        values = iter(self.payload["latent"])
        for i in range(n):
            for j in range(i + 1, n):
                rebuilt[i, j] = rebuilt[j, i] = next(values)
        self.assertTrue(np.allclose(rebuilt, rebuilt.T))
        # Still positive semi-definite after the round trip through 5 decimals.
        self.assertGreater(float(np.linalg.eigvalsh(rebuilt).min()), -1e-4)

    def test_the_reference_portfolio_indexes_into_the_published_pool(self):
        n = len(self.payload["players"])
        reference = self.payload["reference"]
        self.assertTrue(reference["portfolio"])
        for lineup in reference["portfolio"]:
            self.assertEqual(len(lineup["ids"]), nb.CFG.lineup_size)
            self.assertEqual(len(set(lineup["ids"])), nb.CFG.lineup_size)
            self.assertTrue(all(0 <= i < n for i in lineup["ids"]))
            self.assertIn(lineup["superstar"], lineup["ids"])
            self.assertLessEqual(lineup["salary"], self.payload["salary_cap"])

    def test_the_index_points_at_files_that_were_built(self):
        files = {f"{payload['game_id']}.json" for payload in self.payloads}
        self.assertEqual({game["file"] for game in self.index["games"]}, files)

    def test_a_game_with_no_cap_is_skipped_rather_than_guessed(self):
        original = yahoo_payload()
        original["salaryCapInfo"]["result"][0]["singleGameSalaryCapMap"] = {}
        nb.fetch_yahoo_data = lambda *a, **k: original
        payloads, index = build_payload(optimize=False)
        self.assertEqual(payloads, [])
        self.assertTrue(any("salary cap" in note for note in index["skipped"]))

    def test_it_can_publish_without_a_reference_run(self):
        payloads, _ = build_payload(optimize=False)
        self.assertIsNone(payloads[0]["reference"])
        self.assertTrue(payloads[0]["players"])


class WriteOutputTests(unittest.TestCase):
    def setUp(self):
        _StubbedFeeds().install(self)
        self.directory = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.directory, True)
        self.addCleanup(setattr, run_showdown, "DATA", run_showdown.DATA)
        run_showdown.DATA = self.directory

    def test_it_writes_the_games_and_an_index(self):
        payloads, index = build_payload(optimize=False)
        written = run_showdown.write_outputs(payloads, index)
        self.assertEqual(written, ["nfl.g.1.json"])
        self.assertTrue((self.directory / "index.json").exists())
        self.assertEqual(
            json.loads((self.directory / "nfl.g.1.json").read_text())["game_id"],
            "nfl.g.1",
        )

    def test_a_game_from_a_previous_slate_is_removed(self):
        stale = self.directory / "nfl.g.stale.json"
        self.directory.mkdir(parents=True, exist_ok=True)
        stale.write_text("{}")
        payloads, index = build_payload(optimize=False)
        run_showdown.write_outputs(payloads, index)
        self.assertFalse(stale.exists())

    def test_nothing_is_written_when_no_game_is_usable(self):
        original = yahoo_payload()
        original["salaryCapInfo"]["result"][0]["singleGameSalaryCapMap"] = {}
        nb.fetch_yahoo_data = lambda *a, **k: original
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            code = run_showdown.main(["--no-optimize"])
        self.assertEqual(code, 1)
        self.assertFalse((self.directory / "index.json").exists())


@unittest.skipIf(NODE is None, "Node is required to run the browser optimizer")
class BrowserAgreementTests(unittest.TestCase):
    """The JavaScript optimizer against the Python pipeline, same published model."""

    @classmethod
    def setUpClass(cls):
        chart, roster = depth_chart(), roster_status()
        originals = {name: getattr(nb, name) for name in
                     ("fetch_yahoo_data", "load_market_projection_reference",
                      "load_nflverse_reference", "display")}
        nb.fetch_yahoo_data = lambda *a, **k: yahoo_payload()
        nb.load_market_projection_reference = lambda cfg=None: (
            [], {"feeds": [], "notes": [], "logit_vig": 0.17, "calibration_pairs": 0})
        nb.load_nflverse_reference = lambda players, season, cfg=None: (
            chart, roster, None, [])
        nb.display = lambda *a, **k: None
        try:
            payloads, _ = build_payload(simulations=6_000)
        finally:
            for name, value in originals.items():
                setattr(nb, name, value)
        cls.payload = payloads[0]

        directory = Path(tempfile.mkdtemp())
        cls.addClassCleanup(shutil.rmtree, directory, True)
        path = directory / "payload.json"
        path.write_text(json.dumps(cls.payload))
        result = subprocess.run(
            [NODE, str(ROOT / "tests" / "showdown_reference.js"), str(path), "6000"],
            capture_output=True, text=True, timeout=600,
        )
        if result.returncode != 0:
            raise AssertionError(f"node harness failed:\n{result.stderr}")
        cls.js = json.loads(result.stdout)

    def test_both_find_the_same_number_of_valid_rosters(self):
        # Enumeration is pure combinatorics over the same rules -- the cap, the
        # salary floor and Yahoo's one-skill-player-per-team requirement -- so
        # any disagreement here is a rule implemented differently on one side.
        self.assertEqual(
            self.js["valid_rosters"], self.payload["reference"]["valid_rosters"]
        )

    def test_both_screen_to_the_same_number_of_candidates(self):
        self.assertEqual(
            self.js["candidates_scored"], self.payload["reference"]["candidates_scored"]
        )

    def test_the_analytic_expectation_agrees(self):
        probe = self.js["probe"]
        published = self.payload["reference"]["portfolio"][0]
        self.assertEqual(probe["ids"], published["ids"])
        self.assertAlmostEqual(probe["analytic"]["mean"], published["expected_fp"], places=3)

    def test_the_analytic_variance_agrees(self):
        probe = self.js["probe"]
        published = self.payload["reference"]["portfolio"][0]
        self.assertAlmostEqual(
            probe["analytic"]["variance"] ** 0.5, published["analytic_sd"], places=3
        )

    def test_the_simulated_summaries_agree_within_monte_carlo_error(self):
        # Different generators, so these are two samples from one distribution,
        # not two computations of one number.
        published = {tuple(l["ids"]) + (l["superstar"],): l
                     for l in self.payload["reference"]["portfolio"]}
        compared = 0
        for lineup in self.js["portfolio"]:
            key = tuple(lineup["ids"]) + (lineup["superstar"],)
            if key not in published:
                continue
            compared += 1
            reference = published[key]
            self.assertLess(
                abs(lineup["sim_mean"] - reference["sim_mean"]) / reference["sim_mean"],
                0.02, f"{key} mean")
            self.assertLess(
                abs(lineup["ceiling_p90"] - reference["ceiling_p90"]) / reference["ceiling_p90"],
                0.05, f"{key} p90")
        self.assertGreater(compared, 0, "no lineup was selected by both runs")

    def test_every_lineup_it_builds_is_legal(self):
        cap = self.payload["salary_cap"]
        players = self.payload["players"]
        teams = {self.payload["away"], self.payload["home"]}
        for lineup in self.js["portfolio"]:
            self.assertEqual(len(set(lineup["ids"])), 5)
            self.assertLessEqual(lineup["salary"], cap + 1e-6)
            self.assertIn(lineup["superstar"], lineup["ids"])
            skill_teams = {players[i]["team"] for i in lineup["ids"]
                           if players[i]["pos"] != "DEF"}
            self.assertEqual(skill_teams, teams, "each team needs a non-DEF player")

    def test_the_exposure_caps_are_honoured(self):
        settings = self.payload["settings"]
        entries = len(self.js["portfolio"])
        cap = 1 if entries == 1 else int(entries * settings["max_player_exposure"] + 1e-9)
        counts = {}
        for lineup in self.js["portfolio"]:
            for player in lineup["ids"]:
                counts[player] = counts.get(player, 0) + 1
        self.assertLessEqual(max(counts.values()), cap)
        self.assertLessEqual(
            self.js["diversity"]["max_shared"], settings["max_shared_players"]
        )


if __name__ == "__main__":
    unittest.main()
