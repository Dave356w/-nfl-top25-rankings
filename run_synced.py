#!/usr/bin/env python3
"""Publish rankings, the weekly lineup and Showdown from one projection slate.

Fetch and prepare the feeds once, derive all three views from the same frame,
and validate their shared projections and snapshot IDs before writing files.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import lineup_optimizer as lo
from pipeline import notebook as nb
from pipeline import showdown
from pipeline import projection_archive
import run_daily
import run_lineup
import run_showdown

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"


def _projection_map(rows):
    mapped = {}
    for row in rows:
        player = row.get("player") or row.get("name")
        position = row.get("pos") or row.get("position")
        value = row.get("mean") if "mean" in row else row.get("fp")
        if player and position and value is not None:
            mapped[(lo.normalize_name(str(player)), str(position))] = float(value)
    return mapped


def validate_page_consistency(
    rankings_payload: dict,
    lineup_payload: dict,
    pool_payload: dict | None = None,
    showdown_payloads: list[dict] | None = None,
    showdown_index: dict | None = None,
) -> int:
    """Reject a publish if two page payloads price a shared player differently."""
    ranking_rows = [
        dict(row, position=position)
        for position, rows in (rankings_payload.get("rankings") or {}).items()
        for row in rows
    ]
    ranking = _projection_map(ranking_rows)
    lineup = _projection_map(
        (lineup_payload.get("starters") or []) + (lineup_payload.get("bench") or [])
    )
    pool = _projection_map((pool_payload or {}).get("players") or [])

    snapshots = {
        rankings_payload.get("snapshot_id"),
        lineup_payload.get("snapshot_id"),
    }
    if pool_payload is not None:
        snapshots.add(pool_payload.get("snapshot_id"))
    if showdown_index is not None:
        snapshots.add(showdown_index.get("snapshot_id"))
        indexed = {game["game_id"] for game in showdown_index.get("games", [])}
        exported = {game["game_id"] for game in showdown_payloads or []}
        if indexed != exported:
            raise ValueError("Showdown index does not match exported games")
    for game in showdown_payloads or []:
        snapshots.add(game.get("snapshot_id"))
    if None in snapshots or len(snapshots) != 1:
        raise ValueError("Published pages do not share one snapshot_id")

    mismatches = []
    compared = 0
    known = dict(ranking)
    views = [("lineup", lineup), ("editor pool", pool)]
    views.extend((f"showdown {game['game_id']}", _projection_map(game["players"]))
                 for game in showdown_payloads or [])
    for label, other in views:
        for key in sorted(known.keys() & other.keys()):
            compared += 1
            # Rankings/lineup round to two decimals; Showdown retains four.
            if abs(known[key] - other[key]) > 0.00501:
                mismatches.append(
                    f"{key[0]} ({key[1]}): shared projection {known[key]:.2f}, "
                    f"{label} {other[key]:.2f}"
                )
        known.update(other)
    if mismatches:
        raise ValueError(
            "Cross-page projection mismatch: " + "; ".join(mismatches[:12])
        )
    return compared


def build_synced_payloads(
    *,
    top_n: int = 25,
    positions=nb.VALID_POSITIONS,
    roster_path: str | None = None,
    objective: str = lo.LINEUP_OBJECTIVE,
    excluded: list[str] | None = None,
    no_pool: bool = False,
    showdown_simulations: int | None = None,
    showdown_max_games: int | None = None,
    showdown_optimize: bool = True,
):
    """Build all page contracts without writing partial output."""
    generated_at = datetime.now(timezone.utc)
    snapshot_id = generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    cfg = nb.CFG
    if showdown_simulations is not None:
        if showdown_simulations < 1:
            raise ValueError("Showdown simulations must be positive")
        cfg = nb.replace(cfg, simulations=showdown_simulations)
    if showdown_max_games is not None and showdown_max_games < 1:
        raise ValueError("Showdown max games must be positive")

    ranking_tee = run_daily._Tee(sys.stdout)
    with contextlib.redirect_stdout(ranking_tee):
        prepared = nb.prepare_slate_pool(
            cfg, f"publishing rankings, lineup and Showdown (snapshot {snapshot_id})"
        )
        ranking_results = nb.run_position_rankings(
            top_n=top_n,
            cfg=cfg,
            positions=positions,
            export_csv=False,
            prepared_slate=prepared,
        )

    if roster_path is None and run_lineup.DEFAULT_ROSTER.exists():
        roster_path = str(run_lineup.DEFAULT_ROSTER)
    lineup_tee = run_daily._Tee(sys.stdout)
    with contextlib.redirect_stdout(lineup_tee):
        configured = lo.load_roster(roster_path)
        print(
            f"Roster: {len(configured)} player(s) from "
            f"{roster_path or 'MY_TEAM_ROSTER'}."
        )
        print(f"Projection snapshot: {snapshot_id} (shared with position rankings).")
        lineup_results, pool, pool_note = run_lineup.build_from_prepared_slate(
            prepared,
            configured,
            excluded=excluded,
            objective=objective,
            no_pool=no_pool,
        )

    rankings_payload = run_daily.build_payload(
        ranking_results,
        top_n,
        ranking_tee.lines,
        generated_at=generated_at,
        snapshot_id=snapshot_id,
    )
    lineup_payload = run_lineup.build_payload(
        lineup_results,
        objective,
        ranking_tee.lines + lineup_tee.lines,
        generated_at=generated_at,
        snapshot_id=snapshot_id,
    )
    pool_payload = (
        None
        if no_pool
        else run_lineup.build_pool_payload(pool, lineup_payload, pool_note)
    )

    showdown_tee = run_daily._Tee(sys.stdout)
    with contextlib.redirect_stdout(showdown_tee):
        showdown_payloads, showdown_index = showdown.build_slate(
            cfg, optimize=showdown_optimize, max_games=showdown_max_games,
            prepared_slate=prepared, generated_at=generated_at,
            snapshot_id=snapshot_id,
        )
    showdown_index["log"] = ranking_tee.lines + showdown_tee.lines

    if not any(rankings_payload["rankings"].values()):
        raise ValueError("Run produced no ranked players; refusing to publish")
    if not lineup_payload["starters"]:
        raise ValueError("Run produced no lineup starters; refusing to publish")

    # Validate all contracts before touching any page's current files. An empty
    # Showdown slate publishes an empty index, replacing stale games while the
    # rankings remain usable when Yahoo has not supplied single-game caps.
    json.dumps(rankings_payload, allow_nan=False)
    json.dumps(lineup_payload, allow_nan=False)
    if pool_payload is not None:
        json.dumps(pool_payload, allow_nan=False)
    json.dumps(showdown_payloads, allow_nan=False)
    json.dumps(showdown_index, allow_nan=False)
    compared = validate_page_consistency(
        rankings_payload, lineup_payload, pool_payload,
        showdown_payloads, showdown_index,
    )
    return {
        "projection_archive": projection_archive.capture(prepared, cfg, snapshot_id, showdown_payloads),
        "generated_at": generated_at,
        "snapshot_id": snapshot_id,
        "rankings": rankings_payload,
        "ranking_results": ranking_results,
        "lineup": lineup_payload,
        "pool": pool_payload,
        "showdown": showdown_payloads,
        "showdown_index": showdown_index,
        "compared": compared,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--positions", default=",".join(nb.VALID_POSITIONS))
    parser.add_argument("--roster", default=None)
    parser.add_argument(
        "--objective",
        default=lo.LINEUP_OBJECTIVE,
        choices=["FP", "Floor_P25", "Ceiling_P90"],
    )
    parser.add_argument("--exclude", default="")
    parser.add_argument("--no-pool", action="store_true")
    parser.add_argument("--showdown-simulations", type=int, default=None)
    parser.add_argument("--showdown-max-games", type=int, default=None)
    parser.add_argument("--showdown-no-optimize", action="store_true",
                        help="Publish shared models without a reference simulation")
    parser.add_argument("--site-dir", default=str(SITE))
    args = parser.parse_args(argv)

    positions = tuple(p.strip() for p in args.positions.split(",") if p.strip())
    excluded = [name.strip() for name in args.exclude.split(",") if name.strip()]
    try:
        built = build_synced_payloads(
            top_n=args.top_n,
            positions=positions,
            roster_path=args.roster,
            objective=args.objective,
            excluded=excluded,
            no_pool=args.no_pool,
            showdown_simulations=args.showdown_simulations,
            showdown_max_games=args.showdown_max_games,
            showdown_optimize=not args.showdown_no_optimize,
        )
    except Exception:
        traceback.print_exc()
        print(
            "\nSynchronized run failed; no page should be committed.",
            file=sys.stderr,
        )
        return 1

    site = Path(args.site_dir)
    projection_archive.write(built["projection_archive"], site / "data" / "projection_archive")
    ranking_entries = run_daily.write_outputs(
        built["rankings"],
        built["ranking_results"].get("combined"),
        site / "data",
    )
    lineup_entries = run_lineup.write_outputs(
        built["lineup"],
        site / "data" / "lineup",
        built["pool"],
    )
    showdown_files = run_showdown.write_outputs(
        built["showdown"], built["showdown_index"], site / "data" / "showdown"
    )
    print(
        f"\nPublished synchronized snapshot {built['snapshot_id']}: "
        f"{sum(len(rows) for rows in built['rankings']['rankings'].values())} ranked "
        f"players, {len(built['lineup']['starters'])} lineup starters, "
        f"{len(showdown_files)} Showdown games, "
        f"{built['compared']} cross-page projections verified; "
        f"{len(ranking_entries)} ranking and {len(lineup_entries)} lineup run(s) archived."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

