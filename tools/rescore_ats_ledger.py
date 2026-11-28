#!/usr/bin/env python3
"""Rescore the margin research page's frozen rows with the committed points model.

Used when the points model changes and the season's live rows should be
graded as if the new model had been published. Each row keeps its frozen
inputs (team strength, line, snapshot, capture time); only the outputs are
recomputed: points, raw margin, band, mapped margin, edge and model side.
Changed rows are stamped `rescored_utc` and `rescore_note`, so the page's
record stays honest about what was published before kickoff and what was
rescored afterwards.

Rewrites site/data/ats/ledger.jsonl and latest.json / latest.csv: locked
rows (the page carries them forward from latest.json) are rescored and
regraded, unlocked rows are refreshed, and the records are recomputed. Rows
already matching the model are left untouched, so a second run changes nothing.

    python tools/rescore_ats_ledger.py --note "home field removed from the points model"
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import ats_research as ats  # noqa: E402

DIRECTORY = ROOT / "site" / "data" / "ats"


def rescore_row(row, coef, rule, stamp, note):
    """Return (row, changed). Rows without strength inputs are left alone."""
    needed = ("home_offense_fp", "away_offense_fp", "home_def_fp", "away_def_fp")
    if any(row.get(k) is None for k in needed):
        return row, False
    home_pts = ats.predict_points(coef, row["home_offense_fp"], row["away_def_fp"])
    away_pts = ats.predict_points(coef, row["away_offense_fp"], row["home_def_fp"])
    raw = home_pts - away_pts
    out = dict(row, home_points=round(home_pts, 1), away_points=round(away_pts, 1),
               raw_margin=round(raw, 2), projected_total=round(home_pts + away_pts, 1),
               band=ats.BAND_LABELS[ats.band_of(raw)])
    if row.get("line") is not None:
        mapped = ats.mapped_margin(raw, rule)
        side = ats.model_side(mapped, row["line"])
        out.update(mapped_margin=mapped, edge=round(mapped - row["line"], 2),
                   model_side=(row["home"] if side > 0 else row["away"]) if side else None,
                   model_side_sign=side)
    keys = ("home_points", "away_points", "raw_margin", "projected_total", "band",
            "mapped_margin", "edge", "model_side", "model_side_sign")
    changed = any(out.get(k) != row.get(k) for k in keys)
    if changed:
        out.update(rescored_utc=stamp, rescore_note=note)
    return out, changed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--note", required=True, help="why the rows were rescored")
    parser.add_argument("--season", type=int, help="only rows of this season (default: all)")
    parser.add_argument("--dir", type=Path, default=DIRECTORY)
    args = parser.parse_args(argv)
    coef = ats.load_model()["coefficients"]
    rule = ats.fit_map()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    in_scope = lambda r: args.season is None or r.get("season") == args.season

    ledger_path = args.dir / "ledger.jsonl"
    ledger, changed = [], []
    for row in ats.read_ledger(ledger_path):
        new, did = rescore_row(row, coef, rule, stamp, args.note) if in_scope(row) else (row, False)
        ledger.append(new)
        if did:
            changed.append((row, new))
    ledger_path.write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in ledger))

    latest_path = args.dir / "latest.json"
    if latest_path.exists():
        model = ats.load_model()
        payload = json.loads(latest_path.read_text())
        season = payload.get("season")
        schedule = ats.load_schedule([season] if season else [])
        graded = {r["game_id"]: r for r in ats._grade_ledger(ledger, schedule)}
        games = []
        for row in payload.get("games", []):
            if row.get("locked") and row["game_id"] in graded and in_scope(row):
                row = dict(graded[row["game_id"]], locked=True)
                for k in ("home_score", "away_score"):
                    row.pop(k, None)
            elif not row.get("locked") and in_scope(row):
                # Not yet frozen: the next publish rebuilds it anyway; refresh so the
                # page does not mix models until then. No rescore stamp: nothing was graded.
                row, _ = rescore_row(row, coef, rule, stamp, args.note)
                row.pop("rescored_utc", None)
                row.pop("rescore_note", None)
            games.append(row)
        payload["games"] = games
        payload["method"]["points_model"] = coef
        payload["method"]["points_model_fit"] = model.get("fit")
        payload["backtest"] = model.get("backtest")
        graded_all = list(graded.values())
        payload["forward_record"] = ats.record(r["result"] for r in graded_all if r.get("result"))
        weeks = ats.season_results(season, ats.load_replay(), graded_all)
        payload["season_results"], payload["season_record"] = weeks, ats.season_record(weeks)
        ats.write_outputs(payload, [], args.dir)   # latest.json and latest.csv; appends nothing

    for old, new in changed:
        print(f"{old['season']} wk{old['week']} {old['away']} @ {old['home']}: raw {old['raw_margin']} -> "
              f"{new['raw_margin']}, side {old.get('model_side')} -> {new.get('model_side')}")
    print(f"{len(changed)} of {len(ledger)} ledger rows changed")


if __name__ == "__main__":
    main()
