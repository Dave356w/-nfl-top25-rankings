#!/usr/bin/env python3
"""Commit this run's site/data and push it to main without losing a race.

Several publishers (the schedules, manual dispatches, re-runs of an old run)
can build at once, and each commits a whole snapshot. A plain rebase conflicts
whenever two of them rewrote the same current files, so on conflict the run's
commit is replayed onto the latest main instead:

* the projection archive is content-addressed and always kept;
* the current pages (latest files, pool, Showdown) come from whichever
  snapshot is newer, so a stale build can never overwrite a fresher one;
* a dated history file is taken when this run's copy is the newer one;
* the ranking and lineup indexes are rebuilt from the merged history.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATA = "site/data"
ARCHIVE = f"{DATA}/projection_archive/"
SHOWDOWN = f"{DATA}/showdown/"
HISTORY_DIRS = (f"{DATA}/history/", f"{DATA}/lineup/history/")
INDEXES = (f"{DATA}/index.json", f"{DATA}/lineup/index.json")


def _git(repo, *args, check=True):
    return subprocess.run(
        ["git", *args], cwd=repo, check=check, text=True, capture_output=True
    )


def _snapshot_id(repo, rev, path=f"{DATA}/latest.json"):
    shown = _git(repo, "show", f"{rev}:{path}", check=False)
    if shown.returncode != 0:
        return ""
    try:
        return str(json.loads(shown.stdout).get("snapshot_id") or "")
    except json.JSONDecodeError:
        return ""


def _exists(repo, rev, path):
    return _git(repo, "cat-file", "-e", f"{rev}:{path}", check=False).returncode == 0


def _take(repo, rev, path):
    """Make path in the work tree match rev, including a deletion."""
    if _exists(repo, rev, path):
        _git(repo, "checkout", rev, "--", path)
    else:
        _git(repo, "rm", "-q", "--ignore-unmatch", "--", path)


def rebuild_indexes(repo):
    import run_daily
    import run_lineup

    run_daily.write_index(Path(repo) / DATA)
    run_lineup.write_index(Path(repo) / DATA / "lineup")


def replay(repo, ours, upstream):
    """Recreate the run's commit ``ours`` on top of ``upstream``."""
    _git(repo, "reset", "-q", "--hard", upstream)
    newer = _snapshot_id(repo, ours) > _snapshot_id(repo, upstream)
    changed = _git(
        repo, "diff", "--no-renames", "--name-only", f"{ours}^", ours, "--", DATA
    ).stdout.split()

    for path in changed:
        if path in INDEXES or path.startswith(SHOWDOWN):
            continue
        if path.startswith(ARCHIVE):
            _take(repo, ours, path)
        elif path.startswith(HISTORY_DIRS):
            # A date's CSV follows its JSON, which carries the snapshot ID.
            day = str(Path(path).with_suffix(".json"))
            if _snapshot_id(repo, ours, day) > _snapshot_id(repo, upstream, day):
                _take(repo, ours, path)
        elif newer:
            _take(repo, ours, path)

    if newer:
        # Showdown is one set: its index names exactly the game files beside it.
        _git(repo, "rm", "-r", "-q", "--ignore-unmatch", "--", SHOWDOWN)
        if _exists(repo, ours, SHOWDOWN.rstrip("/")):
            _git(repo, "checkout", ours, "--", SHOWDOWN)

    rebuild_indexes(repo)
    _git(repo, "add", "-A", DATA)
    if _git(repo, "diff", "--cached", "--quiet", check=False).returncode == 0:
        print("Main already carries this run's data; nothing to replay.")
        return False
    message = _git(repo, "log", "-1", "--format=%B", ours).stdout
    _git(repo, "commit", "-q", "-m", message)
    kept = "this run's pages" if newer else "main's newer pages"
    print(f"Replayed onto {upstream[:7]}, keeping {kept}.")
    return True


def publish(repo=ROOT, remote="origin", branch="main", attempts=4, message=None,
            delay=lambda attempt: time.sleep(2 ** attempt)):
    _git(repo, "add", DATA)
    if _git(repo, "diff", "--cached", "--quiet", check=False).returncode == 0:
        print("No data change to commit.")
        return 0
    _git(repo, "commit", "-q", "-m", message or "data: shared projection snapshot")
    ours = _git(repo, "rev-parse", "HEAD").stdout.strip()

    for attempt in range(1, attempts + 1):
        _git(repo, "reset", "-q", "--hard", ours)
        fetched = _git(repo, "fetch", "-q", remote, branch, check=False)
        if fetched.returncode == 0:
            upstream = _git(repo, "rev-parse", "FETCH_HEAD").stdout.strip()
            if _git(repo, "rebase", "-q", upstream, check=False).returncode != 0:
                _git(repo, "rebase", "--abort", check=False)
                if not replay(repo, ours, upstream):
                    return 0
            pushed = _git(repo, "push", "-q", remote, f"HEAD:{branch}", check=False)
            if pushed.returncode == 0:
                print(f"Pushed {_git(repo, 'rev-parse', '--short', 'HEAD').stdout.strip()}.")
                return 0
            print(pushed.stderr, file=sys.stderr)
        else:
            print(fetched.stderr, file=sys.stderr)
        print(f"push attempt {attempt} failed; retrying")
        if attempt < attempts:
            delay(attempt)
    return 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--message", required=True)
    parser.add_argument("--remote", default="origin")
    parser.add_argument("--branch", default="main")
    args = parser.parse_args(argv)
    return publish(remote=args.remote, branch=args.branch, message=args.message)


if __name__ == "__main__":
    raise SystemExit(main())
