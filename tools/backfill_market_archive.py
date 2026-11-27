#!/usr/bin/env python3
"""Recover per-player market values for runs archived before snapshots kept them.

Until snapshots carried ``market_fp``, the only record of each run's
display-only market column was the page files that run committed:
``site/data/lineup/pool.json`` (with market quality) and
``site/data/showdown/<game>.json``. Each carries the run's ``snapshot_id``,
the same id as that run's archived snapshot, so every recovered value is
joined to that snapshot's own prediction rows (team + normalized name) for
its game id, archive player key and kickoff. Nothing is matched across runs.

Writes a CSV that ``tools/evaluate_projection_archive.py`` reads with
``--market-backfill``:

    python tools/backfill_market_archive.py --since 2026-11-17 \
        --output model/market_backfill_2026.csv

Needs full git history (``git fetch --unshallow`` in a shallow clone).
"""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.sleeper import normalize_name, normalize_team  # noqa: E402

POOL = 'site/data/lineup/pool.json'
SHOWDOWN = 'site/data/showdown'
COLUMNS = ['snapshot_id', 'captured_utc', 'game_id', 'player_key', 'player', 'team',
           'position', 'kickoff_utc', 'market_fp', 'market_quality', 'origin', 'commit']


def git(*args):
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout


def page_rows(commit):
    """(snapshot_id, file origin, name, team, market_fp, quality) from one commit's pages."""
    paths = [p for p in git('ls-tree', '-r', '--name-only', commit, '--', POOL, SHOWDOWN).split()
             if p == POOL or (p.startswith(SHOWDOWN + '/nfl.g.') and p.endswith('.json'))]
    for path in paths:
        try:
            page = json.loads(git('show', f'{commit}:{path}'))
        except (subprocess.CalledProcessError, ValueError):
            continue
        snapshot_id = page.get('snapshot_id')
        for row in page.get('players') or []:
            value = row.get('market_mean')
            if value is None or not snapshot_id:
                continue
            name = row.get('player') or row.get('name')
            yield (snapshot_id, 'pool' if path == POOL else 'showdown', name, row.get('team'),
                   float(value), row.get('market_quality'))


def archived_snapshots(directory):
    by_id = {}
    for path in Path(directory).glob('*.json.gz'):
        snap = json.loads(gzip.decompress(path.read_bytes()))
        by_id[snap['snapshot_id']] = snap
    return by_id


def build(since, archive_dir, until=None):
    snapshots = archived_snapshots(archive_dir)
    log = ['log', '--format=%H', f'--since={since}']
    if until:
        log.append(f'--until={until}')
    commits = git(*log, '--', POOL, SHOWDOWN).split()
    found = {}
    for commit in commits:
        for snapshot_id, origin, name, team, value, quality in page_rows(commit):
            if not name or not team:
                continue
            ident = (snapshot_id, normalize_team(team), normalize_name(name))
            # Pool rows carry the quality label; prefer them over Showdown copies.
            if ident in found and not (origin == 'pool' and found[ident][1] != 'pool'):
                continue
            found[ident] = (value, origin, quality, commit)
    lookups = {}
    rows, unmatched = [], 0
    for (snapshot_id, team, key), (value, origin, quality, commit) in found.items():
        snap = snapshots.get(snapshot_id)
        if snap is None:
            unmatched += 1
            continue
        if snapshot_id not in lookups:
            index = {}
            for p in snap['predictions']:
                index.setdefault((normalize_team(p['team']), normalize_name(p['player'])), []).append(p)
            lookups[snapshot_id] = index
        hits = lookups[snapshot_id].get((team, key), [])
        if len(hits) != 1:  # missing or ambiguous: never guess
            unmatched += 1
            continue
        p = hits[0]
        rows.append(dict(snapshot_id=snapshot_id, captured_utc=snap['captured_utc'],
                         game_id=str(p['game_id']), player_key=p['player_key'],
                         player=p['player'], team=p['team'], position=p['position'],
                         kickoff_utc=p['kickoff_utc'], market_fp=value,
                         market_quality=quality if origin == 'pool' else None,
                         origin=origin, commit=commit[:12]))
    frame = pd.DataFrame(rows, columns=COLUMNS)
    frame = frame.sort_values(['captured_utc', 'game_id', 'player_key']).reset_index(drop=True)
    return frame, dict(commits=len(commits), values=len(found), matched=len(frame),
                       unmatched=unmatched)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--since', required=True, help='first commit date, e.g. 2026-11-17')
    parser.add_argument('--until', help='last commit date (default: now)')
    parser.add_argument('--archive', default=str(ROOT / 'site/data/projection_archive/snapshots'))
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    if git('rev-parse', '--is-shallow-repository').strip() == 'true':
        raise SystemExit('Shallow clone: run `git fetch --unshallow` first')
    frame, stats = build(args.since, args.archive, args.until)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False)
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
