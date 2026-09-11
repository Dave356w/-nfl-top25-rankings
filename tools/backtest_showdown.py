#!/usr/bin/env python3
"""Backtest Yahoo NFL single-game lineup selection on completed slates."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import notebook as nb
from pipeline.showdown_backtest import (
    NflverseHistory, YahooDfsClient, run_slate, summarize,
)


def parse_date(value: str) -> date:
    return date.fromisoformat(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", required=True, type=parse_date,
                        help="First kickoff date, YYYY-MM-DD")
    parser.add_argument("--end", required=True, type=parse_date,
                        help="Exclusive end date, YYYY-MM-DD")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="backtest_cache/yahoo")
    parser.add_argument("--max-slates", type=int)
    parser.add_argument("--simulations", type=int, default=2_000)
    parser.add_argument("--entries", type=int, default=1)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--no-benchmark", action="store_true")
    args = parser.parse_args(argv)
    if args.end <= args.start:
        parser.error("--end must be later than --start")
    if args.simulations < 100:
        parser.error("--simulations must be at least 100")
    if args.entries < 1:
        parser.error("--entries must be positive")

    yahoo = YahooDfsClient(args.cache_dir, refresh=args.refresh)
    slates = yahoo.discover(args.start, args.end)
    if args.max_slates is not None:
        slates = slates[:max(0, args.max_slates)]
    if not slates:
        raise ValueError("No completed Yahoo NFL single-game slates found")

    # January/February games belong to the previous NFL season; the extra prior
    # season supplies rolling history for early-season projections.
    seasons = set()
    for slate in slates:
        kickoff = date.fromtimestamp(slate["kickoff_ms"] / 1000)
        season = kickoff.year - 1 if kickoff.month <= 3 else kickoff.year
        seasons.update((season - 1, season))
    history = NflverseHistory.load(sorted(seasons))
    cfg = nb.replace(
        nb.CFG, simulations=args.simulations, tournament_lineups=args.entries,
        use_nflverse=False,
    )
    results, skipped = [], []
    for index, slate in enumerate(slates, 1):
        print(f"[{index}/{len(slates)}] {slate['away']} @ {slate['home']} ", end="", flush=True)
        try:
            result = run_slate(
                slate, yahoo, history, cfg, fetch_benchmark=not args.no_benchmark
            )
            results.append(result)
            print(f"{result['portfolio'][0]['actual_fp']:.2f} FP")
        except Exception as exc:
            skipped.append({"series_id": slate["series_id"], "reason": str(exc)})
            print(f"SKIP: {exc}")
    report = summarize(results, skipped)
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in {"results", "skipped", "limitations"}}, indent=2))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
