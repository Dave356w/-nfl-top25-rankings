"""Guards for the shared rankings/lineup projection snapshot."""

from __future__ import annotations

import sys
import contextlib
import io
import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lineup_optimizer as lo
import run_synced
import run_showdown
from pipeline import notebook as nb
from pipeline import showdown
from test_showdown import _StubbedFeeds
from test_lineup_optimizer import _context


class PreparedSlateAdapterTests(unittest.TestCase):
    def test_final_rankings_mean_is_preserved_for_the_lineup(self):
        players = pd.DataFrame([
            {
                "Name": "Tee Higgins",
                "Position": "WR",
                "Team": "CIN",
                "Opponent": "TB",
                "Game Time": "2026-09-13T17:00:00Z",
                "Home Team": "CIN",
                "Away Team": "TB",
                "Salary": 33,
                "FPPG": 12.14,
                "Projected_FP": 13.03,
                "Projection_Source": "market good: Bovada+Underdog",
                "Depth_Rank": 2,
                "Market_Quality": "good",
            }
        ])

        adapted = lo.yahoo_from_prepared_slate(players)

        self.assertEqual(float(adapted.iloc[0]["Projected_FP"]), 13.03)
        self.assertEqual(adapted.iloc[0]["Feed_Name"], "Tee Higgins")
        self.assertEqual(adapted.iloc[0]["Feed_Position"], "WR")
        self.assertEqual(adapted.iloc[0]["Key"], "tee higgins")
        self.assertEqual(float(adapted.iloc[0]["Fallback_Depth"]), 2.0)
        self.assertTrue(str(adapted.iloc[0]["Game_Time"].tz) == "UTC")


class ConsistencyGateTests(unittest.TestCase):
    def setUp(self):
        self.rankings = {
            "snapshot_id": "snap-1",
            "rankings": {
                "QB": [{"player": "Dak Prescott", "fp": 18.8}],
                "WR": [{"player": "Tee Higgins", "fp": 13.03}],
            },
        }
        self.lineup = {
            "snapshot_id": "snap-1",
            "starters": [
                {"player": "Dak Prescott", "pos": "QB", "mean": 18.8},
                {"player": "Tee Higgins", "pos": "WR", "mean": 13.03},
            ],
            "bench": [],
        }
        self.pool = {
            "snapshot_id": "snap-1",
            "players": [
                {"player": "Dak Prescott", "pos": "QB", "mean": 18.8},
                {"player": "Tee Higgins", "pos": "WR", "mean": 13.03},
            ],
        }

    def test_matching_pages_pass(self):
        compared = run_synced.validate_page_consistency(
            self.rankings, self.lineup, self.pool
        )
        self.assertEqual(compared, 4)

    def test_a_projection_difference_blocks_publication(self):
        self.lineup["starters"][1]["mean"] = 12.38
        with self.assertRaisesRegex(ValueError, "tee higgins"):
            run_synced.validate_page_consistency(
                self.rankings, self.lineup, self.pool
            )

    def test_a_snapshot_difference_blocks_publication(self):
        self.lineup["snapshot_id"] = "older-snapshot"
        with self.assertRaisesRegex(ValueError, "snapshot_id"):
            run_synced.validate_page_consistency(
                self.rankings, self.lineup, self.pool
            )

    def showdown_game(self):
        return {
            "game_id": "nfl.g.1", "snapshot_id": "snap-1",
            "players": [{"name": "Dak Prescott", "pos": "QB", "fp": 18.8049}],
        }

    def test_showdown_uses_the_same_projection_allowing_display_rounding(self):
        self.assertEqual(run_synced.validate_page_consistency(
            self.rankings, self.lineup, self.pool, [self.showdown_game()]
        ), 5)

    def test_stale_showdown_snapshot_blocks_publication(self):
        game = self.showdown_game()
        game["snapshot_id"] = "yesterday"
        with self.assertRaisesRegex(ValueError, "snapshot_id"):
            run_synced.validate_page_consistency(
                self.rankings, self.lineup, self.pool, [game]
            )

    def test_showdown_mismatch_outside_the_top_rankings_blocks_publication(self):
        self.pool["players"].append({"player": "Reserve Receiver", "pos": "WR", "mean": 3.0})
        game = self.showdown_game()
        game["players"].append({"name": "Reserve Receiver", "pos": "WR", "fp": 5.0})
        with self.assertRaisesRegex(ValueError, "reserve receiver"):
            run_synced.validate_page_consistency(
                self.rankings, self.lineup, self.pool, [game]
            )

    def test_showdown_index_cannot_reference_an_unpublished_game(self):
        index = {"snapshot_id": "snap-1", "games": [{"game_id": "missing"}]}
        with self.assertRaisesRegex(ValueError, "index"):
            run_synced.validate_page_consistency(
                self.rankings, self.lineup, self.pool, [self.showdown_game()], index
            )


class SharedShowdownTests(unittest.TestCase):
    def setUp(self):
        _StubbedFeeds().install(self)

    def test_prepared_slate_reuses_final_means_without_fetching_again(self):
        with contextlib.redirect_stdout(io.StringIO()):
            prepared = nb.prepare_slate_pool(nb.CFG)
        player = prepared["players"].iloc[0]
        prepared["players"].loc[0, "Projected_FP"] = 12.3456
        generated = datetime(2026, 9, 9, 21, 0, tzinfo=timezone.utc)
        with patch.object(nb, "prepare_slate_pool", side_effect=AssertionError("duplicate fetch")):
            with contextlib.redirect_stdout(io.StringIO()):
                payloads, index = showdown.build_slate(
                    prepared_slate=prepared, generated_at=generated,
                    snapshot_id="shared-test", optimize=False,
                )
        published = next(p for p in payloads[0]["players"] if p["name"] == player["Name"])
        self.assertEqual(published["fp"], 12.3456)
        self.assertEqual(payloads[0]["snapshot_id"], "shared-test")
        self.assertEqual(index["snapshot_id"], "shared-test")
        self.assertEqual(payloads[0]["generated_utc"], "2026-09-09T21:00:00Z")
        self.assertEqual(index["generated_utc"], payloads[0]["generated_utc"])

    def test_empty_shared_slate_replaces_stale_models_in_the_requested_directory(self):
        # An empty cap map no longer empties the slate: those games publish a
        # model and the page collects the cap. A slate is empty when no game has
        # a priced pool at all.
        with contextlib.redirect_stdout(io.StringIO()):
            prepared = nb.prepare_slate_pool(nb.CFG)
            prepared["players"] = prepared["players"].iloc[0:0]
            payloads, index = showdown.build_slate(prepared_slate=prepared, optimize=False)
        self.assertEqual(index["status"], "empty")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "nfl.g.old.json").write_text("{}")
            run_showdown.write_outputs(payloads, index, path)
            self.assertFalse((path / "nfl.g.old.json").exists())
            self.assertEqual(json.loads((path / "index.json").read_text())["games"], [])

    def test_publisher_prepares_once_and_writes_all_views_with_one_snapshot(self):
        configured = [{"Name": "Drake Maye", "Position": "QB"}]
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(nb, "prepare_slate_pool", wraps=nb.prepare_slate_pool) as prepare:
                with patch.object(lo, "load_nfl_context", return_value=_context()):
                    with patch.object(lo, "load_roster", return_value=configured):
                        with contextlib.redirect_stdout(io.StringIO()):
                            code = run_synced.main([
                                "--site-dir", directory, "--showdown-no-optimize",
                            ])
            self.assertEqual(code, 0)
            self.assertEqual(prepare.call_count, 1)
            root = Path(directory) / "data"
            paths = [root / "latest.json", root / "lineup/latest.json",
                     root / "lineup/pool.json", root / "showdown/index.json",
                     root / "showdown/nfl.g.1.json"]
            documents = [json.loads(path.read_text()) for path in paths]
            self.assertEqual(len({doc["snapshot_id"] for doc in documents}), 1)
            self.assertEqual(len({doc["generated_utc"] for doc in documents}), 1)


if __name__ == "__main__":
    unittest.main()
