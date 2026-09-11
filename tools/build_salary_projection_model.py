#!/usr/bin/env python3
"""Fit the Yahoo salary-position-depth model with a season holdout."""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import salary_projection
from pipeline.showdown_backtest import YahooDfsClient, NflverseHistory, build_pool


def metrics(actual, forecast):
    error = np.asarray(forecast) - np.asarray(actual)
    return {
        "n": int(len(error)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "bias": float(np.mean(error)),
        "correlation": float(np.corrcoef(actual, forecast)[0, 1]),
    }


def prior_baseline(frame: pd.DataFrame) -> pd.Series:
    """Reproduce the pre-regression FPPG/salary prior for holdout comparison."""
    result = pd.Series(index=frame.index, dtype=float)
    for _, group in frame.groupby(["Series_ID", "Position"]):
        train = group[np.isfinite(group["FPPG"]) & group["FPPG"].gt(0.25)]
        if len(train) >= 5 and train["Salary"].nunique() >= 3:
            x = train["Salary"].to_numpy(float)
            y = train["FPPG"].to_numpy(float)
            centered = x - x.mean()
            slope = float(np.dot(centered, y - y.mean()) / (np.dot(centered, centered) + 25.0))
            slope = float(np.clip(slope, 0.05, 1.25))
            prior = y.mean() + slope * (group["Salary"].to_numpy(float) - x.mean())
        else:
            ratio = np.median(train["FPPG"] / train["Salary"]) if len(train) else 0.45
            prior = group["Salary"].to_numpy(float) * float(np.clip(ratio, 0.15, 0.90))
        prior = np.maximum(prior, 0.25)
        result.loc[group.index] = np.where(
            group["FPPG"].gt(0.25), 0.70 * group["FPPG"] + 0.30 * prior, 0.80 * prior
        )
    return result.clip(lower=0.05)


def training_rows(start: date, end: date, cache_dir: str) -> tuple[pd.DataFrame, list]:
    yahoo = YahooDfsClient(cache_dir=Path(cache_dir) / "yahoo", pause=0.02)
    slates = yahoo.discover(start, end)
    seasons = list(range(start.year - 1, end.year + 1))
    history = NflverseHistory.load(seasons)
    rows, skipped = [], []
    for number, slate in enumerate(slates, 1):
        try:
            game = history.match_game(slate)
            pool, _ = build_pool(
                slate, yahoo.players(slate["series_id"]), history,
                apply_projection=False,
            )
            for row in pool.to_dict("records"):
                rows.append({
                    "Series_ID": slate["series_id"],
                    "Season": int(game["season"]),
                    "Position": row["Position"],
                    "Depth_Rank": min(int(row["Depth_Rank"]), 4),
                    "Salary": float(row["Salary"]),
                    "Actual_FP": float(row["Yahoo_Actual_FP"]),
                    "FPPG": float(row["FPPG"]),
                })
        except Exception as exc:
            skipped.append({"series_id": slate["series_id"], "error": str(exc)})
        if number % 25 == 0:
            print(f"Processed {number}/{len(slates)} slates; {len(rows):,} rows")
    return pd.DataFrame(rows), skipped


def choose(train: pd.DataFrame, validation: pd.DataFrame):
    candidates = []
    positions = sorted(train["Position"].unique())
    for degree in (1, 2, 3):
        for interactions in (False, True):
            spec = {
                "positions": positions,
                "degree": degree,
                "depth_salary_interactions": interactions,
                "salary_scale": 10.0,
            }
            for ridge in (0.1, 1.0, 10.0, 100.0):
                model = salary_projection.fit(train, spec, ridge)
                score = metrics(validation["Actual_FP"], salary_projection.predict(validation, model))
                candidates.append((score["mae"], score["rmse"], model, score))
    return min(candidates, key=lambda item: (item[0], item[1]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="2026-03-01")
    parser.add_argument("--holdout", type=int, default=2025)
    parser.add_argument("--cache-dir", default="backtest_cache")
    parser.add_argument("--output", default=str(salary_projection.MODEL_PATH))
    args = parser.parse_args(argv)
    frame, skipped = training_rows(date.fromisoformat(args.start), date.fromisoformat(args.end), args.cache_dir)
    frame["Current_Fallback_FP"] = prior_baseline(frame)
    train, validation = frame[frame.Season.lt(args.holdout)], frame[frame.Season.eq(args.holdout)]
    _, _, selected, validation_score = choose(train, validation)
    final = salary_projection.fit(frame[frame.Season.le(args.holdout)], selected, selected["ridge"])
    final.update({
        "schema": 1,
        "target": "Yahoo settled half-PPR fantasy points",
        "trained_through_season": args.holdout,
        "training_rows": int(len(frame[frame.Season.le(args.holdout)])),
        "training_slates": int(frame.Series_ID.nunique()),
        "selection": "minimum MAE on season holdout",
        "holdout_season": args.holdout,
        "holdout_metrics": validation_score,
        "holdout_current_fallback": metrics(validation.Actual_FP, validation.Current_Fallback_FP),
        "skipped_slates": skipped,
        "limitations": [
            "Historical depth is inferred from strictly pregame lagged opportunity.",
            "Salary is a Yahoo consensus proxy and may precede late injury news.",
            "The model does not project kickers; weekly kicker means remain nflverse rolling estimates.",
        ],
    })
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(final, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(target), "selected": {"degree": final["degree"], "depth_salary_interactions": final["depth_salary_interactions"], "ridge": final["ridge"]}, "holdout": validation_score}, indent=2))


if __name__ == "__main__":
    main()
