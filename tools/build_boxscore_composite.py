#!/usr/bin/env python3
"""Run the NFL box-score composite (pipeline/boxscore_composite.py) and export
the compact artifact the margin research page reads.

The composite's own outputs (caches, per-run folders with every CSV and the
HTML board) stay under BOXSCORE_OUTPUT_ROOT (default
boxscore_cache/nfl_boxscore_output_v1_7, git-ignored). Two files persist in
site/data/boxscore/ (committed with every snapshot by tools/publish_snapshot.py)
so the experiment survives fresh checkouts:

  frozen_recipe_<season>.json  the season's frozen recipe (never rewritten)
  forward_predictions.jsonl    the composite's first pre-kickoff snapshots

They are copied into the output folder before the run and the ledger is copied
back after it. The export, site/data/boxscore/latest.json, holds the current
week's board, this season's walk-forward predictions, the held-out 2023+
predictions, the scorecards and the spread-through-map control since 2001;
pipeline/ats_research.py builds the margin research page from it.

    python tools/build_boxscore_composite.py
"""
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "site" / "data" / "boxscore"
EXPORT = MODEL_DIR / "latest.json"
LEDGER = "forward_predictions.jsonl"
GAME_COLS = ["game_id", "season", "week", "home", "away", "gameday", "gametime", "site",
             "ready", "model_wp", "market_wp", "homefield_wp", "spread_line", "result",
             "home_score", "away_score"]


def _clean(value):
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else round(float(value), 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return value


def _rows(frame, cols):
    cols = [c for c in cols if c in frame]
    return [{c: _clean(v) for c, v in zip(cols, row)} for row in frame[cols].itertuples(index=False)]


def seed(root: Path):
    """Copy the committed frozen recipes and ledger into the output folder (never overwrite a recipe)."""
    root.mkdir(parents=True, exist_ok=True)
    for path in MODEL_DIR.glob("frozen_recipe_*.json"):
        target = root / path.name
        if not target.exists():
            shutil.copy2(path, target)
    ledger = MODEL_DIR / LEDGER
    if ledger.exists():
        shutil.copy2(ledger, root / LEDGER)


def export(outdir: Path, root: Path) -> dict:
    manifest = json.loads((outdir / "run_manifest.json").read_text())
    season, week = int(manifest["season"]), int(manifest["week"])
    current = pd.read_csv(outdir / "current_season_predictions.csv")
    outer = pd.read_csv(outdir / "outer_chronological_predictions.csv")
    fit = json.loads((outdir / "current_coefficient_fit.json").read_text())
    weights = pd.read_csv(outdir / "current_composite_weights.csv")
    by_season = pd.read_csv(outdir / "outer_scorecard_by_season.csv")
    pooled = pd.read_csv(outdir / "outer_chronological_scorecard.csv")
    payload = {
        "schema": 1,
        "revision": manifest["revision"],
        "exported_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_utc": manifest["generated_utc"],
        "season": season, "week": week,
        "recipe": manifest["recipe"],
        "market_sigma": 12.37,
        "market_blend_verdict": manifest.get("market_blend_verdict"),
        "coefficients": _rows(weights, list(weights.columns)),
        "fit": {k: _clean(v) for k, v in fit.items() if k in (
            "fit_season", "fit_week", "training_games", "effective_training_games",
            "training_max_season", "training_max_week_in_latest_season")},
        "outer_scorecard": _rows(pooled, list(pooled.columns)),
        "outer_scorecard_by_season": _rows(by_season, list(by_season.columns)),
        "board": _rows(current[current.week == week], GAME_COLS),
        "season_predictions": _rows(current, GAME_COLS),
        "outer_predictions": _rows(outer, GAME_COLS),
        "limitations": manifest.get("limitations", []),
    }
    json.dumps(payload, allow_nan=False)
    return payload


def control_long_run(first=2001, last=None):
    """The fixed map applied to the closing spread itself, every final regular-season game."""
    sys.path.insert(0, str(ROOT))
    from pipeline import ats_research as ats
    last = last or datetime.now(timezone.utc).year
    s = ats.load_schedule(range(first, last + 1))
    s = s[s.home_score.notna() & s.spread_line.notna()]
    results = [ats.grade(ats.pick(line, line)[1], hs - as_, line)
               for line, hs, as_ in zip(s.spread_line, s.home_score, s.away_score)]
    return {**ats.record(results), "first_season": int(s.season.min()), "last_season": int(s.season.max())}


def main():
    root = Path(os.environ.setdefault(
        "BOXSCORE_OUTPUT_ROOT", str(ROOT / "boxscore_cache" / "nfl_boxscore_output_v1_7")))
    seed(root)
    sys.path.insert(0, str(ROOT))
    from pipeline import boxscore_composite as bc
    bc.OUTPUT_NAME = str(root)
    bc.USE_GOOGLE_DRIVE = False
    outdir = bc.main()
    if (root / LEDGER).exists():
        shutil.copy2(root / LEDGER, MODEL_DIR / LEDGER)
    for path in root.glob("frozen_recipe_*.json"):
        if not (MODEL_DIR / path.name).exists():
            shutil.copy2(path, MODEL_DIR / path.name)
    payload = export(Path(outdir), root)
    payload["control_long_run"] = control_long_run(last=payload["season"])
    EXPORT.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    print(f"Exported {EXPORT.relative_to(ROOT)}: season {payload['season']} week {payload['week']}, "
          f"{len(payload['board'])} board games, {len(payload['outer_predictions'])} held-out games.")


if __name__ == "__main__":
    main()
