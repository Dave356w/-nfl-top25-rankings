#!/usr/bin/env python3
"""Publish the rankings and weekly lineup from one canonical projection slate.

The two pages used to run hours apart and independently fetch live sportsbook
props.  That made both pages internally valid but allowed the same player to
show two different projections.  This entry point fetches and prepares the
slate once, derives both views from that in-memory frame, validates their common
players, and only then writes either page's files.
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
from pipeline import market_tail_guard
from pipeline import notebook as nb
import run_daily
import run_lineup

market_tail_guard.install(nb)

ROOT = Path(__file__).resolve().parent
SITE = ROOT / "site"


def _projection_map(rows):
    mapped = {}
    for row in rows:
        player = row.get("player")
        position = row.get("pos") or row.get("position")
        value = row.get("mean") if "mean" in row else row.get("fp")
        if player and position and value is not None:
            mapped[(lo.normalize_name(str(player)), str(position))] = float(value)
    return mapped


def validate_page_consistency(
    rankings_payload: dict,
    lineup_payload: dict,
    pool_payload: dict | None = None,
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
    if None in snapshots or len(snapshots) != 1:
        raise ValueError("Rankings, lineup and editor pool do not share one snapshot_id")

    mismatches = []
    compared = 0
    for label, other in (("lineup", lineup), ("editor pool", pool)):
        for key in sorted(ranking.keys() & other.keys()):
            compared += 1
            if abs(ranking[key] - other[key]) > 0.005:
                mismatches.append(
                    f"{key[0]} ({key[1]}): rankings {ranking[key]:.2f}, "
                    f"{label} {other[key]:.2f}"
                )
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
    no_market: bool = False,
    no_pool: bool = False,
):
    """Build both page contracts without writing partial output."""
    generated_at = datetime.now(timezone.utc)
    snapshot_id = generated_at.strftime("%Y-%m-%dT%H:%M:%SZ")
    cfg = nb.replace(nb.CFG, use_market_projections=not no_market)

    ranking_tee = run_daily._Tee(sys.stdout)
    with contextlib.redirect_stdout(ranking_tee):
        prepared = nb.prepare_slate_pool(
            cfg, f"publishing synchronized rankings and lineup (snapshot {snapshot_id})"
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

    if not any(rankings_payload["rankings"].values()):
        raise ValueError("Run produced no ranked players; refusing to publish")
    if not lineup_payload["starters"]:
        raise ValueError("Run produced no lineup starters; refusing to publish")

    # Validate strict JSON before touching either page's current files.
    json.dumps(rankings_payload, allow_nan=False)
    json.dumps(lineup_payload, allow_nan=False)
    if pool_payload is not None:
        json.dumps(pool_payload, allow_nan=False)
    compared = validate_page_consistency(
        rankings_payload, lineup_payload, pool_payload
    )
    return {
        "generated_at": generated_at,
        "snapshot_id": snapshot_id,
        "rankings": rankings_payload,
        "ranking_results": ranking_results,
        "lineup": lineup_payload,
        "pool": pool_payload,
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
    parser.add_argument("--no-market", action="store_true")
    parser.add_argument("--no-pool", action="store_true")
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
            no_market=args.no_market,
            no_pool=args.no_pool,
        )
    except Exception:
        traceback.print_exc()
        print(
            "\nSynchronized run failed; neither page should be committed.",
            file=sys.stderr,
        )
        return 1

    site = Path(args.site_dir)
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
    print(
        f"\nPublished synchronized snapshot {built['snapshot_id']}: "
        f"{sum(len(rows) for rows in built['rankings']['rankings'].values())} ranked "
        f"players, {len(built['lineup']['starters'])} lineup starters, "
        f"{built['compared']} cross-page projections verified; "
        f"{len(ranking_entries)} ranking and {len(lineup_entries)} lineup run(s) archived."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
