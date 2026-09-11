#!/usr/bin/env python3
"""Assemble the team-outcome game corpus and audit its as-of boundary.

Phase P0 of docs/team-outcome-model-plan.md. Writes a summary artifact
recording what was assembled, how the corpus is split, which seasons carry a
closing line, and whether the as-of audit came back clean. No model is fitted
here and nothing is promoted; P0 exists so that later phases start from a
corpus whose leakage properties are already checked in.
"""
import argparse
from datetime import datetime, timezone
import gzip
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import team_outcome as to


CACHE = Path("backtest_cache/nflverse")


def seasons_arg(value: str) -> list[int]:
    start, _, end = value.partition("-")
    return list(range(int(start), int(end or start) + 1))


def pull_efficiency(seasons, cache_dir: Path, refresh: bool) -> pd.DataFrame:
    """Cache the aggregated team-game efficiency, not the raw play frame.

    A season of play-by-play is fifty thousand rows and 372 columns; the corpus
    needs nine columns per team-game. Aggregating before caching keeps the
    cache small enough to reread quickly and makes the stored object the same
    shape the corpus consumes.
    """
    target = cache_dir / f"efficiency_{seasons[0]}_{seasons[-1]}.json.gz"
    if target.exists() and not refresh:
        return pd.DataFrame(json.loads(gzip.decompress(target.read_bytes())))
    frame = to.load_pbp_efficiency(seasons)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(frame.to_json(orient="records", date_format="iso"))
    target.write_bytes(gzip.compress(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    return frame


def pull_fantasy(seasons, schedules, cache_dir: Path, refresh: bool) -> pd.DataFrame:
    """Cache the settled unit-week fantasy frame the fixed-core run consumes."""
    from pipeline import team_outcome_core as core
    target = cache_dir / f"fantasy_{seasons[0]}_{seasons[-1]}.json.gz"
    if target.exists() and not refresh:
        return pd.DataFrame(json.loads(gzip.decompress(target.read_bytes())))
    settled = schedules.loc[schedules.home_score.notna() & schedules.away_score.notna()]
    frame = core.build_fantasy(core.load_player_points(seasons),
                               core.load_defence_points(settled, seasons))
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(frame.to_json(orient="records", date_format="iso"))
    target.write_bytes(gzip.compress(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    return frame


def pull(seasons, cache_dir: Path, refresh: bool) -> pd.DataFrame:
    """Cache the raw schedules pull so a rebuild does not depend on the network."""
    target = cache_dir / f"schedules_{seasons[0]}_{seasons[-1]}.json.gz"
    if target.exists() and not refresh:
        return pd.DataFrame(json.loads(gzip.decompress(target.read_bytes())))
    frame = to.load_schedules(seasons)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.loads(frame.to_json(orient="records", date_format="iso"))
    target.write_bytes(gzip.compress(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(), mtime=0))
    return frame


def summarize(corpus: to.Corpus) -> dict:
    """Inventory the corpus without reporting what happened in a sealed season.

    Coverage counts span every season, because later phases have to know the
    sealed games exist. Outcome statistics do not: a home win rate or a tie
    count for a sealed season is a summary of the holdout, which §6.1 of the
    plan puts off limits until P4.
    """
    games, outcomes = corpus.games, corpus.outcomes
    settled = to.unsealed(games.merge(outcomes, on="game_id", how="inner"))
    market = corpus.market.merge(games[["game_id", "season"]], on="game_id", how="left")
    lines = market.groupby("season").home_moneyline.apply(lambda s: float(s.notna().mean()))
    per_season = settled.groupby("season").agg(
        games=("game_id", "size"),
        home_win_rate=("home_win", lambda s: round(float((s == 1.0).mean()), 4)),
        ties=("tie", lambda s: int(s.sum())),
    ).reset_index()
    per_season["closing_moneyline_coverage"] = per_season.season.map(lines).fillna(0.0).round(4)
    coverage = (games.groupby("season").size().rename("scheduled_games").reset_index())
    coverage["closing_moneyline_coverage"] = coverage.season.map(lines).fillna(0.0).round(4)
    coverage["sealed"] = coverage.season.isin(to.SEALED_SEASONS)
    return {
        "seasons": [int(games.season.min()), int(games.season.max())],
        "scheduled_games": int(len(games)),
        "settled_games": int(len(outcomes)),
        "unscheduled_outcomes": int(len(set(outcomes.game_id) - set(games.game_id))),
        "sealed_seasons": list(to.SEALED_SEASONS),
        "outcome_statistics_span": "unsealed seasons only",
        "unsealed_settled_games": int(len(settled)),
        "home_win_rate": round(float((settled.home_win == 1.0).mean()), 4),
        "home_win_rate_ties_as_half": round(float(settled.home_win.mean()), 4),
        "ties": int(settled.tie.sum()),
        "margin_sd": round(float(settled.margin.std()), 3),
        "margin_mean": round(float(settled.margin.mean()), 3),
        "teams": int(pd.concat([games.home_team, games.away_team]).nunique()),
        "context_columns": sorted(games.columns),
        "withheld_from_context": sorted(
            set(to.MARKET_COLUMNS) | set(to.OUTCOME_SOURCE_COLUMNS) | set(to.WITHHELD_COLUMNS)),
        "per_season_outcomes_unsealed": per_season.to_dict("records"),
        "per_season_coverage": coverage.to_dict("records"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    # argparse runs `type` over command-line strings only, so the default has
    # to arrive already parsed.
    parser.add_argument("--seasons", type=seasons_arg, default=seasons_arg("1999-2025"),
                        help="inclusive range, e.g. 1999-2025")
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--refresh", action="store_true", help="re-pull instead of using the cache")
    parser.add_argument("--audit-every", type=int, default=1,
                        help="audit every Nth settled week; 1 audits all of them")
    parser.add_argument("--output", type=Path, default=Path("model/team_outcome_corpus.json"))
    args = parser.parse_args()

    raw = pull(args.seasons, args.cache, args.refresh)
    corpus = to.build_corpus(raw)
    weeks = corpus.weeks(settled_only=True)[:: max(1, args.audit_every)]
    report = to.audit(corpus, checkpoints=weeks)
    design = to.build_design(corpus)

    artifact = {
        "schema": 1,
        "phase": "P0",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "source": "nflverse schedules via nflreadpy",
        "corpus": summarize(corpus),
        "design": {
            "rows": int(len(design)),
            "features": sorted(to.FEATURES),
            "note": "Harness exercise only. The candidate feature set is fixed at P2.",
        },
        "as_of_audit": {
            "checkpoints": report["checkpoints"],
            "audited_every_nth_week": int(args.audit_every),
            "clean": bool(report["clean"]),
            "findings": report["findings"][:20],
        },
        "limitations": [
            "Schedules only. Play-by-play efficiency, rosters and injuries are not in this corpus.",
            "Closing lines are held in a separate frame as an evaluation benchmark and are never features.",
            "Recorded weather, starting quarterback, officials and coaches are withheld: nflverse "
            "records or restates them after the fact.",
            "History is resolved to whole weeks, so a Thursday result is not available to that "
            "week's later games.",
            "No model is fitted and nothing is promoted at P0.",
            "Outcome statistics cover unsealed seasons only; coverage counts span the corpus.",
            "The as-of audit does run over sealed weeks: it reports whether the boundary held, "
            "never what happened in those games, so it reads nothing the seal protects.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: artifact["corpus"][k] for k in (
        "seasons", "scheduled_games", "settled_games", "sealed_seasons",
        "unsealed_settled_games", "home_win_rate", "ties", "margin_sd")}, indent=2))
    print(f"as-of audit: {'clean' if report['clean'] else 'FINDINGS'} "
          f"over {report['checkpoints']} checkpoints; design rows {len(design)}")
    return 0 if report["clean"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
