"""The pages' depth editor, checked against the Python model it re-runs.

`site/depth-model.js` lets a visitor change a player's depth rank and see the
projection the frozen salary-position-depth regression gives at the new depth.
That is a second implementation of `salary_projection.predict`, and of the
lineup band in `lineup_optimizer.build_roster`, so the agreement is pinned here
by driving the shipped JavaScript through Node rather than restating it.
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
sys.path.insert(0, str(ROOT / "tests"))

import run_daily
from pipeline import notebook as nb
from pipeline import salary_projection
from tools import export_depth_model

NODE = shutil.which("node")
MODULE = ROOT / "site" / "depth-model.js"
PUBLISHED = ROOT / "site" / "data" / "depth_model.json"


def run_node(script: str):
    """Run `script` with `DepthModel` and the published model in scope."""
    program = (
        f"const DepthModel = require({json.dumps(str(MODULE))});\n"
        f"const model = JSON.parse(require('fs').readFileSync({json.dumps(str(PUBLISHED))}, 'utf8'));\n"
        f"const out = (function () {{ {script} }})();\n"
        "process.stdout.write(JSON.stringify(out));\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False) as handle:
        handle.write(program)
    try:
        done = subprocess.run([NODE, handle.name], capture_output=True, text=True, check=True)
    finally:
        Path(handle.name).unlink(missing_ok=True)
    return json.loads(done.stdout)


class PublishedModelTests(unittest.TestCase):
    def test_the_published_file_is_the_current_model(self):
        # Regenerate with `python tools/export_depth_model.py` after refitting.
        self.assertEqual(PUBLISHED.read_text(encoding="utf-8"), export_depth_model.render())


@unittest.skipUnless(NODE, "node is not installed")
class PredictParityTests(unittest.TestCase):
    def test_every_position_salary_and_depth_matches_python(self):
        grid = [
            (position, salary, depth)
            for position in ("QB", "RB", "WR", "TE", "DEF")
            for salary in (8, 12, 19.5, 27, 35, 46)
            for depth in (1, 2, 3, 4, 7)
        ]
        frame = pd.DataFrame(grid, columns=["Position", "Salary", "Depth_Rank"])
        expected = salary_projection.predict(frame, salary_projection.load())
        got = run_node(
            f"return {json.dumps(grid)}.map(function (g) {{"
            " return DepthModel.predict(model, g[0], g[1], g[2]); });"
        )
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)

    def test_an_unpriced_or_unmodelled_player_is_left_alone(self):
        got = run_node(
            "return [DepthModel.predict(model, 'K', 10, 1),"
            " DepthModel.predict(model, 'WR', 0, 1),"
            " DepthModel.isModelled('manual override'),"
            " DepthModel.isModelled('Yahoo salary-position-depth regression (trained through 2025)')];"
        )
        self.assertEqual(got, [None, None, False, True])

    def test_volatility_tables_match_the_fitted_ones(self):
        got = run_node(
            "return ['QB','RB','WR','TE'].map(function (p) {"
            " return [1,2,3,4,6].map(function (d) {"
            " return [DepthModel.cv(model, p, d), DepthModel.zero(model, p, d)]; }); });"
        )
        for p_index, position in enumerate(("QB", "RB", "WR", "TE")):
            for d_index, depth in enumerate((1, 2, 3, 4, 6)):
                self.assertEqual(got[p_index][d_index], [
                    nb._calibrated_cv(position, depth), nb._zero_rate(position, depth)
                ])

    def test_the_band_is_the_lineup_optimizer_band(self):
        mean, cv = 14.2, 0.814
        sigma = np.sqrt(np.log1p(cv ** 2))
        got = run_node(f"return DepthModel.band({mean}, {cv});")
        self.assertAlmostEqual(got["floor"], mean * np.exp(-.67449 * sigma - .5 * sigma ** 2), 9)
        self.assertAlmostEqual(got["ceiling"], mean * np.exp(1.28155 * sigma - .5 * sigma ** 2), 9)


@unittest.skipUnless(NODE, "node is not installed")
class MoveTests(unittest.TestCase):
    PLAYERS = [
        {"id": "a", "group": "NE|WR", "depth": 1},
        {"id": "b", "group": "NE|WR", "depth": 2},
        {"id": "c", "group": "NE|WR", "depth": 3},
        {"id": "d", "group": "NE|WR", "depth": 5},   # depth 4 is not on this page
        {"id": "x", "group": "SEA|WR", "depth": 1},
    ]

    def move(self, player, depth):
        return run_node(f"return DepthModel.move({json.dumps(self.PLAYERS)}, {json.dumps(player)}, {depth});")

    def test_moving_up_pushes_the_players_he_passes_down_one(self):
        self.assertEqual(self.move("c", 1), {"c": 1, "a": 2, "b": 3})

    def test_moving_down_pulls_the_players_he_drops_behind_up_one(self):
        self.assertEqual(self.move("a", 3), {"a": 3, "b": 1, "c": 2})

    def test_unseen_gaps_and_other_teams_are_kept(self):
        # d sits at 5 with nobody visible at 4; moving b to 4 passes only c.
        self.assertEqual(self.move("b", 4), {"b": 4, "c": 2})

    def test_no_move_is_no_change(self):
        self.assertEqual(self.move("b", 2), {})


class RankingPoolTests(unittest.TestCase):
    """The rankings page re-ranks from the whole priced pool, not the top N."""

    def test_the_payload_carries_the_whole_pool_and_the_archive_does_not(self):
        import test_role_model

        case = test_role_model.RankingRunTests("test_it_publishes_a_role_for_every_ranked_player")
        case.setUp()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results = nb.run_position_rankings(top_n=2, export_csv=False)
        finally:
            case.doCleanups()
        payload = run_daily.build_payload(results, 2, [])
        self.assertEqual(len(payload["rankings"]["WR"]), 2)
        self.assertGreater(len(payload["pool"]["WR"]), 2)
        self.assertEqual(payload["pool"]["WR"][:2], payload["rankings"]["WR"])

        with tempfile.TemporaryDirectory() as tmp:
            run_daily.write_outputs(payload, results["combined"], data_dir=tmp)
            latest = json.loads((Path(tmp) / "latest.json").read_text())
            archived = json.loads((Path(tmp) / "history" / f"{payload['run_date']}.json").read_text())
        self.assertIn("pool", latest)
        self.assertNotIn("pool", archived)


if __name__ == "__main__":
    unittest.main()
