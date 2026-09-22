#!/usr/bin/env python3
"""Append settled Yahoo results to one immutable pregame Showdown snapshot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.showdown_archive_backtest import (
    DEFAULT_ARCHIVE, NflverseActuals, load_snapshot, run, snapshot_date_range,
)
from pipeline.showdown_backtest import YahooDfsClient


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-id", help="Exact archived snapshot id; latest when omitted")
    parser.add_argument("--archive-dir", default=str(DEFAULT_ARCHIVE))
    parser.add_argument("--cache-dir", default="backtest_cache/yahoo")
    parser.add_argument("--output", required=True)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    parser.add_argument(
        "--nflverse-fallback", action="store_true",
        help="Use reconstructed nflverse actuals when Yahoo settlement is unavailable",
    )
    parser.add_argument(
        "--reconstruct-simulations", type=int,
        help="Smoke-test override; omit for the archived production count",
    )
    args = parser.parse_args(argv)
    if args.reconstruct_simulations is not None and args.reconstruct_simulations < 100:
        parser.error("--reconstruct-simulations must be at least 100")

    snapshot = load_snapshot(args.archive_dir, args.snapshot_id)
    start, end = snapshot_date_range(snapshot)
    yahoo = YahooDfsClient(args.cache_dir, refresh=args.refresh)
    completed = yahoo.discover(start, end)
    nflverse = None
    if args.nflverse_fallback:
        seasons = sorted({
            (pd.Timestamp(model["kickoff_utc"]).year - 1
             if pd.Timestamp(model["kickoff_utc"]).month <= 3
             else pd.Timestamp(model["kickoff_utc"]).year)
            for model in snapshot["showdown_models"]
        })
        nflverse = NflverseActuals.load(seasons)
    report = run(
        snapshot, yahoo, completed,
        reconstruct_simulations=args.reconstruct_simulations,
        fetch_benchmark=not args.no_benchmark,
        nflverse=nflverse,
    )
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"games", "skipped", "limitations"}}, indent=2))
    if report["skipped"]:
        print(json.dumps({"skipped": report["skipped"]}, indent=2))
    return 0 if report["games_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
