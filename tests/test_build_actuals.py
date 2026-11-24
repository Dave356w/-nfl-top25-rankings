import gzip
import json
from datetime import date
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import build_actuals as ba  # noqa: E402


def write_snapshot(directory, snapshot_id, predictions, models=()):
    path = Path(directory) / "snapshots" / f"{snapshot_id.replace(':', '')}.json.gz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(dict(
        snapshot_id=snapshot_id, predictions=predictions,
        showdown_models=list(models))).encode()))


def row(game, key, team, pos="WR", kickoff="2026-11-22T18:00:00Z"):
    return dict(game_id=game, player_key=key, team=team, position=pos, kickoff_utc=kickoff)


class FakeNflverse:
    schedules = pd.DataFrame([dict(home_team="DET", away_team="TB", game_date=date(2026, 11, 22))])

    def __init__(self):
        self.models = []

    def actuals(self, model, keys):
        self.models.append(model)
        got = {key: 7.5 for key in keys if key != "yahoo:dnp"}
        return got, {"missing_without_zero_fill": [k for k in keys if k not in got]}


class BuildActualsTests(unittest.TestCase):
    def test_excluded_snapshots_contribute_no_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            write_snapshot(directory, "good", [row("g1", "yahoo:1", "DET")],
                           [dict(game_id="g1", home="DET", away="TB")])
            write_snapshot(directory, "bad", [row("g1", "yahoo:2", "DET")])
            games = ba.archived_games(directory, excluded={"bad"})
        self.assertEqual(set(games["g1"]["players"]), {"yahoo:1"})
        self.assertEqual((games["g1"]["home"], games["g1"]["away"]), ("DET", "TB"))

    def test_missing_players_are_omitted_and_unfinished_games_wait(self):
        games = {
            "g1": dict(home="DET", away="TB", kickoff_utc="2026-11-22T18:00:00Z",
                       players={"yahoo:1": dict(pos="WR", team="DET"),
                                "yahoo:dnp": dict(pos="WR", team="TB")}),
            "later": dict(home="DET", away="TB", kickoff_utc="2026-11-29T18:00:00Z",
                          players={"yahoo:3": dict(pos="WR", team="DET")}),
        }
        frame, audit = ba.build(games, FakeNflverse(), now="2026-11-24T00:00:00Z")
        self.assertEqual(frame.player_key.tolist(), ["yahoo:1"])
        self.assertEqual(frame.realized_at_utc.iloc[0], "2026-11-22T23:00:00+00:00")
        self.assertEqual(audit["players_without_stat_row"], 1)
        self.assertEqual(audit["games_pending"], 1)

    def test_teams_are_inferred_when_no_showdown_model_was_saved(self):
        games = {"g1": dict(kickoff_utc="2026-11-22T18:00:00Z",
                            players={"yahoo:1": dict(pos="WR", team="DET"),
                                     "yahoo:4": dict(pos="RB", team="TB")})}
        fake = FakeNflverse()
        frame, _ = ba.build(games, fake, now="2026-11-24T00:00:00Z")
        self.assertEqual((fake.models[0]["home"], fake.models[0]["away"]), ("DET", "TB"))
        self.assertEqual(len(frame), 2)


if __name__ == "__main__":
    unittest.main()
