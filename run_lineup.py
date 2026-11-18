#!/usr/bin/env python3
"""Weekly entry point: pick this week's lineup and publish it as JSON.

`lineup_optimizer.run` prompts for benchings and prints a table, which is right
at a terminal and wrong in a workflow. This runs the same optimizer with no
prompt and writes into `site/`, which GitHub Pages serves:

    site/data/lineup/latest.json          this week's lineup, read by lineup.html
    site/data/lineup/pool.json            the add pool the page's roster editor draws from
    site/data/lineup/latest.csv           the same rows as a flat download
    site/data/lineup/history/<date>.json  one archived copy per run date
    site/data/lineup/index.json           the archive listing, newest first

Failure policy matches the daily build: on any error nothing is written and the
process exits non-zero, so the previously published lineup stays live rather
than being replaced by a half-built one.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import lineup_optimizer as lo
# The daily entry point already solved log capture and NaN-safe JSON, and the
# page contract is the same one. Sharing them keeps the two payloads honest.
from run_daily import _Tee, _clean

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "site" / "data" / "lineup"
DEFAULT_ROSTER = ROOT / "lineup_roster.json"
# Display-only market column. Off unless the workflow sets MARKET_DISPLAY=1, so
# tests and local runs never touch the prop feeds.
MARKET_ENV = "MARKET_DISPLAY"
MARKET_KICKOFF_TOLERANCE = pd.Timedelta(hours=6)


def _player_row(player: pd.Series, slot: str | None = None) -> dict:
    row = {
        "slot": slot,
        "player": _clean(player.get("Name")),
        # The page's roster editor matches a dropped or added player against
        # this key rather than reimplementing `normalize_name` in JavaScript.
        "key": _clean(player.get("Key")),
        "pos": _clean(player.get("Position")),
        "team": _clean(player.get("Team")),
        "opponent": _clean(player.get("Opponent")),
        "kickoff_utc": None,
        "mean": _clean(round(float(player.get("FP", 0) or 0), 2)),
        "floor": _clean(player.get("Floor_P25")),
        "ceiling": _clean(player.get("Ceiling_P90")),
        "salary": _clean(player.get("Salary")),
        "depth": _clean(player.get("Depth_Rank")),
        "depth_source": _clean(player.get("Depth_Source")),
        "source": _clean(player.get("Projection_Source")),
        # The two inputs to the blended mean, so a reader can see a disagreement.
        "sleeper_mean": _clean(player.get("Sleeper_Raw_FP")),
        "espn_mean": _clean(player.get("ESPN_FP")),
        "cbs_mean": _clean(player.get("CBS_FP")),
        # Market-implied points from player props, shown beside the mean and
        # never used by the optimizer. Null means the books do not price him.
        "market_mean": _clean(player.get("Market_FP")),
        "market_quality": _clean(player.get("Market_Quality")),
        # Compact posted lines behind market_mean, e.g. "Ru 56.5 · Rec 2.5 · TD 34%".
        "market_props": _clean(player.get("Market_Props")),
        "injury": _clean(player.get("report_status")),
        # Set when Sleeper does not project him this week (ruled out, bye), or
        # null. The page benches on this; `injury` is display only.
        "unavailable": _clean(player.get("Unavailable_Reason")),
        "locked": bool(player.get("Locked", False)),
        "review": lo.review(player),
    }
    kickoff = player.get("Game_Time")
    if isinstance(kickoff, pd.Timestamp) and pd.notna(kickoff):
        row["kickoff_utc"] = kickoff.strftime("%Y-%m-%d %H:%M")
    for key in ("floor", "ceiling"):
        if isinstance(row[key], float):
            row[key] = round(row[key], 2)
    if slot is None:
        row.pop("slot")
    return row


def _total(frame: pd.DataFrame, column: str) -> float | None:
    """Sum a range column, standing a player's mean in where he has no band.

    A starter with no fitted CV -- a bye-week player at 0, or one Sleeper has
    no depth for -- would otherwise void the whole total. His point estimate
    is the honest contribution to both ends of the band.
    """
    if column not in frame or not len(frame):
        return None
    values = pd.to_numeric(frame[column], errors="coerce").fillna(
        pd.to_numeric(frame["FP"], errors="coerce")
    )
    if values.isna().any():
        return None
    # Totals must be reproducible from the two-decimal rows the browser receives.
    return round(float(values.round(2).sum()), 2)


def build_payload(
    results: dict,
    objective: str,
    log_lines: list[str],
    generated_at=None,
    snapshot_id: str | None = None,
) -> dict:
    now = generated_at or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    generated_utc = now.strftime("%Y-%m-%dT%H:%M:%SZ")
    starters, bench = results["starters"], results["bench"]
    context = results["context"]
    roster = results["roster"]
    return {
        "schema": 1,
        "status": "ok",
        "generated_utc": generated_utc,
        "snapshot_id": snapshot_id or generated_utc,
        "run_date": now.strftime("%Y-%m-%d"),
        "season": _clean(context.get("season")),
        "week": _clean(context.get("week")),
        "depth_snapshot": _clean(context.get("depth_stamp")),
        "notes": list(context.get("notes") or []),
        "objective": objective,
        # The page re-optimizes an edited roster in the browser, so it needs the
        # slot rules and the auto-bench rule this run used, not a copy of them.
        "auto_exclude_out": bool(lo.AUTO_EXCLUDE_REPORTED_OUT),
        "slots": dict(lo.STARTING_POSITIONS),
        "flex_eligible": list(lo.FLEX_ELIGIBLE),
        # Browser edits may survive projection refreshes, but must yield when
        # the committed roster itself changes.
        "roster_fingerprint": "|".join(
            f"{row.Key}:{row.Position}" for _, row in roster.iterrows()
        ),
        "totals": {
            "mean": _total(starters, "FP"),
            "floor": _total(starters, "Floor_P25"),
            "ceiling": _total(starters, "Ceiling_P90"),
        },
        "counts": {
            "roster": int(len(roster)),
            "projected": int(roster["Projection_Available"].sum()),
            "starters": int(len(starters)),
        },
        "starters": [_player_row(p, str(p["Slot"])) for _, p in starters.iterrows()],
        "bench": [_player_row(p) for _, p in
                  bench.sort_values("FP", ascending=False).iterrows()],
        "benched_by_request": list(results.get("excluded") or []),
        # Sleeper/ESPN disagreements and how CBS settled each one.
        "conflicts": list(context.get("conflicts") or []),
        "projection": {
            "method": context.get("projection_label") or "Sleeper half-PPR projection",
            "season": _clean(context.get("season")),
            "week": _clean(context.get("week")),
        },
        "log": log_lines,
    }


def build_pool_payload(pool: pd.DataFrame, lineup: dict, note: str | None = None) -> dict:
    """Publish the players the page may add, stamped with the run they came from.

    Kept out of `latest.json` on purpose: it is an order of magnitude bigger
    than the lineup, only the roster editor reads it, and archiving one copy per
    run would grow the history for nothing.
    """
    rows = [_player_row(p) for _, p in pool.iterrows()] if len(pool) else []
    return {
        "schema": 1,
        "generated_utc": lineup["generated_utc"],
        "snapshot_id": lineup.get("snapshot_id", lineup["generated_utc"]),
        "run_date": lineup["run_date"],
        "season": lineup["season"],
        "week": lineup["week"],
        "limits": dict(lo.POOL_LIMITS),
        "note": note,
        "counts": {"players": len(rows)},
        "players": rows,
    }


def _csv_frame(payload: dict) -> pd.DataFrame:
    rows = [dict(row, slot=row.get("slot") or "BENCH")
            for row in payload["starters"] + payload["bench"]]
    # `key` exists for the page's benefit; a downloaded sheet does not want a
    # normalization artefact as a column.
    return pd.DataFrame(rows).drop(columns=["key"], errors="ignore")


def write_outputs(payload: dict, data_dir: Path = None,
                  pool: dict | None = None) -> list[dict]:
    data = Path(data_dir or DATA)
    history = data / "history"
    data.mkdir(parents=True, exist_ok=True)
    history.mkdir(parents=True, exist_ok=True)

    text = json.dumps(payload, indent=2, sort_keys=False)
    (data / "latest.json").write_text(text + "\n", encoding="utf-8")
    (history / f"{payload['run_date']}.json").write_text(text + "\n", encoding="utf-8")

    if pool is not None:
        (data / "pool.json").write_text(
            json.dumps(pool, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )

    frame = _csv_frame(payload)
    frame.to_csv(data / "latest.csv", index=False)
    frame.to_csv(history / f"{payload['run_date']}.csv", index=False)

    return write_index(data)


def write_index(data_dir):
    """Rebuild index.json from the archived runs in history/."""
    data = Path(data_dir)
    history = data / "history"
    entries = []
    for path in sorted(history.glob("*.json"), reverse=True):
        try:
            archived = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        entries.append({
            "date": path.stem,
            "generated_utc": archived.get("generated_utc"),
            "week": archived.get("week"),
            "mean": archived.get("totals", {}).get("mean"),
            "json": f"history/{path.name}",
            "csv": f"history/{path.stem}.csv" if (history / f"{path.stem}.csv").exists() else None,
        })
    (data / "index.json").write_text(
        json.dumps({"runs": entries}, indent=2) + "\n", encoding="utf-8"
    )
    return entries


def attach_market(frames: list[pd.DataFrame], fetch=None) -> list[str]:
    """Add Market_FP/Market_Quality/Market_Props to each frame in place; return run notes.

    Display only. Any failure leaves the columns null and says so in the notes;
    the lineup is never held up or changed by the prop feeds.
    """
    try:
        if fetch is None:
            from pipeline import market, sleeper
            ref, notes = market.market_reference(sleeper.normalize_name, sleeper.normalize_team)
        else:
            ref, notes = fetch()
    except Exception as exc:
        ref, notes = None, [f"market unavailable ({type(exc).__name__}: {exc})"]
    for frame in frames:
        if frame is None or not len(frame):
            continue
        frame["Market_FP"], frame["Market_Quality"], frame["Market_Props"] = None, None, None
        if ref is None or not len(ref):
            continue
        from pipeline.sleeper import normalize_team
        keyed = ref.set_index(["Key", "Team"])
        idx = pd.MultiIndex.from_arrays([frame["Key"], frame["Team"].map(normalize_team)])
        hit = idx.isin(keyed.index)
        # Name+team alone would pin next week's props on a row for this week's
        # game (e.g. a kickoff-locked player): require the same kickoff, within
        # a few hours since books and Yahoo can list slightly different times.
        if hit.any() and "Market_Kickoff" in keyed:
            game = pd.to_datetime(frame["Game_Time"], errors="coerce", utc=True) \
                if "Game_Time" in frame else pd.Series(pd.NaT, index=frame.index)
            book = pd.to_datetime(keyed["Market_Kickoff"].reindex(idx).to_numpy(),
                                  errors="coerce", utc=True)
            same = (abs(pd.Series(book, index=frame.index) - game) <= MARKET_KICKOFF_TOLERANCE)
            hit = hit & same.fillna(False).to_numpy()
        if hit.any():
            sub = keyed.reindex(idx[hit])
            frame.loc[hit, "Market_FP"] = sub["Market_FP"].to_numpy()
            frame.loc[hit, "Market_Quality"] = sub["Market_Quality"].to_numpy()
            if "Market_Props" in sub:
                frame.loc[hit, "Market_Props"] = sub["Market_Props"].to_numpy()
    priced = 0 if ref is None else len(ref)
    return [f"Market (display only): {priced} player(s) priced; " + "; ".join(notes)]


def _finish_lineup(
    configured: list[dict],
    yahoo: pd.DataFrame,
    excluded: list[str],
    objective: str,
    no_pool: bool = False,
    previous: dict | None = None,
) -> tuple[dict, pd.DataFrame, str | None]:
    """Resolve, optimize and optionally build the editor pool from one projection frame.

    ``previous`` is the last published latest.json; players whose game has
    kicked off keep the slot it gave them.
    """
    context = lo.load_sleeper_context(yahoo)
    yahoo = lo.with_sleeper_projections(yahoo, context)
    roster = lo.build_roster(configured, yahoo, context)
    print(f"Season {context['season']} week {context['week']}; "
          f"Sleeper players fetched {context['depth_stamp']}.")
    for note in context.get("notes", []):
        print(note)

    if lo.AUTO_EXCLUDE_REPORTED_OUT:
        excluded = list(dict.fromkeys(excluded + lo.reported_out(roster)))
    if excluded:
        print(f"Benched before optimizing: {', '.join(excluded)}.")
    locks = lo.kickoff_locks(previous, context["season"], context["week"])
    if locks:
        print(f"Locked after kickoff: {', '.join(sorted(locks))}.")
    starters, bench = lo.optimize(roster, excluded, objective, locks=locks)
    results = {
        "roster": roster,
        "starters": starters,
        "bench": bench,
        "context": context,
        "excluded": excluded,
    }

    # The add pool only feeds the page's roster editor, so it degrades without
    # taking the published lineup down with it.
    pool, pool_note = pd.DataFrame(), None
    if not no_pool:
        try:
            pool = lo.build_pool(yahoo, context)
            print(f"Add pool: {len(pool)} player(s) resolved for the page's roster editor.")
        except Exception as exc:
            pool_note = f"Add pool unavailable ({exc}); the page can bench but not add."
            print(pool_note)

    if os.environ.get(MARKET_ENV) == "1":
        for note in attach_market([roster, starters, bench, pool]):
            print(note)
            context.setdefault("notes", []).append(note)
    return results, pool, pool_note


def build_from_prepared_slate(
    prepared_slate: dict,
    configured: list[dict],
    excluded: list[str] | None = None,
    objective: str = lo.LINEUP_OBJECTIVE,
    no_pool: bool = False,
    previous: dict | None = None,
) -> tuple[dict, pd.DataFrame, str | None]:
    """Build a lineup from the exact final projections used by the rankings page."""
    yahoo = lo.yahoo_from_prepared_slate(prepared_slate["players"])
    return _finish_lineup(
        configured,
        yahoo,
        list(excluded or []),
        objective,
        no_pool,
        previous,
    )


def read_previous(data_dir: Path | None) -> dict | None:
    """The last published lineup, or None if missing or unreadable."""
    path = Path(data_dir or DATA) / "latest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roster", default=None,
                        help=f"roster JSON (default {DEFAULT_ROSTER.name} if present)")
    parser.add_argument("--objective", default=lo.LINEUP_OBJECTIVE,
                        choices=["FP", "Floor_P25", "Ceiling_P90"],
                        help="what the lineup maximizes (default FP)")
    parser.add_argument("--exclude", default="",
                        help="comma-separated players to force to the bench")
    parser.add_argument("--no-pool", action="store_true",
                        help="skip the add pool the page's roster editor reads")
    parser.add_argument("--out", default=None, help="output directory")
    args = parser.parse_args(argv)

    roster_path = args.roster
    if roster_path is None and DEFAULT_ROSTER.exists():
        roster_path = str(DEFAULT_ROSTER)
    excluded = [name.strip() for name in args.exclude.split(",") if name.strip()]
    tee = _Tee(sys.stdout)

    try:
        with contextlib.redirect_stdout(tee):
            configured = lo.load_roster(roster_path)
            print(f"Roster: {len(configured)} player(s) from "
                  f"{roster_path or 'MY_TEAM_ROSTER'}.")
            yahoo = lo.fetch_yahoo()
            results, pool, pool_note = _finish_lineup(
                configured,
                yahoo,
                excluded,
                args.objective,
                args.no_pool,
                read_previous(Path(args.out) if args.out else None),
            )
    except Exception:
        traceback.print_exc()
        print("\nRun failed; no files were written. The previously published "
              "lineup stays live.", file=sys.stderr)
        return 1

    payload = build_payload(results, args.objective, tee.lines)
    if not payload["starters"]:
        print("Run produced no starters; refusing to publish.", file=sys.stderr)
        return 1

    pool_payload = None if args.no_pool else build_pool_payload(pool, payload, pool_note)
    entries = write_outputs(payload, Path(args.out) if args.out else None, pool_payload)
    print(f"\nPublished a {len(payload['starters'])}-player lineup "
          f"({payload['totals']['mean']} projected) and a "
          f"{0 if pool_payload is None else pool_payload['counts']['players']}-player "
          f"add pool; {len(entries)} run(s) archived.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
