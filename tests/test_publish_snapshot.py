"""The publish step must survive concurrent and stale publishers."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import publish_snapshot as ps


def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def _snapshot(repo, snapshot_id, games):
    """Write what run_synced publishes, keyed to one snapshot."""
    data = Path(repo) / "site" / "data"
    run_date = snapshot_id[:10]
    stamp = {"snapshot_id": snapshot_id, "generated_utc": snapshot_id, "run_date": run_date}
    _write(data / "latest.json", stamp)
    _write(data / "latest.csv", f"snapshot\n{snapshot_id}")
    _write(data / "history" / f"{run_date}.json", {**stamp, "slate": {"game_count": len(games)}})
    _write(data / "lineup" / "latest.json", {**stamp, "week": 3, "totals": {"mean": 100}})
    _write(data / "lineup" / "history" / f"{run_date}.json", {**stamp, "week": 3})
    _write(data / "lineup" / "pool.json", stamp)
    showdown = data / "showdown"
    for stale in showdown.glob("*.json"):
        stale.unlink()
    for game in games:
        _write(showdown / f"{game}.json", {**stamp, "game_id": game})
    _write(showdown / "index.json", {**stamp, "games": games})
    _write(data / "projection_archive" / "snapshots" / f"{snapshot_id}.json", stamp)
    ps.rebuild_indexes(repo)


def _read(repo, path):
    return json.loads((Path(repo) / path).read_text(encoding="utf-8"))


class PublishSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.remote = root / "remote.git"
        _git(root, "init", "-q", "--bare", "-b", "main", str(self.remote))
        seed = root / "seed"
        _git(root, "clone", "-q", str(self.remote), str(seed))
        self._configure(seed)
        _write(seed / "site" / "data" / "depth_model.json", {"version": 1})
        _snapshot(seed, "2026-09-26T16:00:00Z", ["g1", "g2"])
        _git(seed, "add", "-A")
        _git(seed, "commit", "-q", "-m", "seed")
        _git(seed, "push", "-q", "origin", "HEAD:main")
        # Two publishers check out the same base before either finishes.
        self.first = self._clone("first")
        self.second = self._clone("second")

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def _configure(repo):
        _git(repo, "config", "user.name", "test")
        _git(repo, "config", "user.email", "test@example.com")

    def _clone(self, name):
        path = Path(self.tmp.name) / name
        _git(path.parent, "clone", "-q", str(self.remote), str(path))
        self._configure(path)
        return path

    def _publish(self, repo):
        return ps.publish(repo, message="data: test", delay=lambda _: None)

    def _main(self):
        return self._clone(f"check{len(list(Path(self.tmp.name).iterdir()))}")

    def test_newer_run_replays_over_a_conflicting_older_one(self):
        _snapshot(self.first, "2026-09-26T23:00:00Z", ["g1", "g2"])
        self.assertEqual(self._publish(self.first), 0)
        _snapshot(self.second, "2026-09-27T15:00:00Z", ["g2", "g3"])
        self.assertEqual(self._publish(self.second), 0)

        main = self._main()
        self.assertEqual(_read(main, "site/data/latest.json")["snapshot_id"], "2026-09-27T15:00:00Z")
        self.assertEqual(_read(main, "site/data/lineup/pool.json")["snapshot_id"], "2026-09-27T15:00:00Z")
        showdown = sorted(p.name for p in (main / "site/data/showdown").glob("*.json"))
        self.assertEqual(showdown, ["g2.json", "g3.json", "index.json"])
        dates = [run["date"] for run in _read(main, "site/data/index.json")["runs"]]
        self.assertEqual(dates, ["2026-09-27", "2026-09-26"])
        self.assertEqual(
            _read(main, "site/data/history/2026-09-26.json")["snapshot_id"],
            "2026-09-26T23:00:00Z",
        )
        archived = sorted(p.name for p in (main / "site/data/projection_archive/snapshots").iterdir())
        self.assertEqual(len(archived), 3)

    def test_stale_run_never_overwrites_newer_pages(self):
        _snapshot(self.first, "2026-09-27T15:00:00Z", ["g2", "g3"])
        self.assertEqual(self._publish(self.first), 0)
        _snapshot(self.second, "2026-09-26T23:00:00Z", ["g1", "g2"])
        self.assertEqual(self._publish(self.second), 0)

        main = self._main()
        self.assertEqual(_read(main, "site/data/latest.json")["snapshot_id"], "2026-09-27T15:00:00Z")
        self.assertEqual(_read(main, "site/data/showdown/index.json")["games"], ["g2", "g3"])
        # The older run's date and forecast archive are still recorded.
        self.assertEqual(
            _read(main, "site/data/history/2026-09-26.json")["snapshot_id"],
            "2026-09-26T23:00:00Z",
        )
        dates = [run["date"] for run in _read(main, "site/data/lineup/index.json")["runs"]]
        self.assertEqual(dates, ["2026-09-27", "2026-09-26"])
        archived = list((main / "site/data/projection_archive/snapshots").iterdir())
        self.assertEqual(len(archived), 3)

    def test_replay_keeps_unrelated_changes_on_main(self):
        _write(self.first / "site/data/depth_model.json", {"version": 2})
        _git(self.first, "commit", "-q", "-am", "code change")
        _snapshot(self.first, "2026-09-26T23:00:00Z", ["g1"])
        self.assertEqual(self._publish(self.first), 0)
        _snapshot(self.second, "2026-09-27T15:00:00Z", ["g1"])
        self.assertEqual(self._publish(self.second), 0)

        main = self._main()
        self.assertEqual(_read(main, "site/data/depth_model.json"), {"version": 2})
        self.assertEqual(_read(main, "site/data/latest.json")["snapshot_id"], "2026-09-27T15:00:00Z")

    def test_no_change_publishes_nothing(self):
        head = _git(self.first, "rev-parse", "HEAD")
        self.assertEqual(self._publish(self.first), 0)
        self.assertEqual(_git(self.first, "rev-parse", "HEAD"), head)


if __name__ == "__main__":
    unittest.main()
