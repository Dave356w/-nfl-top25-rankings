#!/usr/bin/env python3
"""Fit the played-but-scoreless rate behind `ZERO_RATE` in pipeline/notebook.py.

The showdown simulator draws each player as a mean-preserving lognormal, which
gives a strictly positive score every time. Real deep players post zero often,
and that is precisely the part of the distribution a cheap punt play is bought
for, so overstating their floor is the failure mode that matters most.

What is fitted
--------------
`P(Yahoo points <= 0 | the player took the field)`, by position and opportunity
bucket, on the same population and the same lagged eight-appearance expectation
`calibrate_rb_correlation.py` and `calibrate_game_correlations.py` use.

Scope is the whole argument
---------------------------
A game with no carry, no target and no pass attempt is dropped from *both* the
numerator and the denominator. Such a game is usually a player who was inactive,
and the pipeline's means are market-implied: a sportsbook's line on a doubtful
receiver is already shaded for the chance he does not play. Counting inactive
games here would charge that same risk a second time.

What survives the filter is the pure shape effect -- a fourth receiver who
dressed, ran his routes, and was never thrown to. That is a property of usage,
not of availability, and nothing upstream prices it.

The tool reports both rates so the gap is visible: `any zero` includes the
no-opportunity games, `played` is the one that gets published.

Usage
-----
    python tools/calibrate_zero_rate.py --seasons 2016-2025
    python tools/calibrate_zero_rate.py --seasons 2016-2025 --emit-python

Offline work run by hand, like the other two calibration tools. The daily build
never runs it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import notebook as nb
from tools.calibrate_rb_correlation import (
    EXPECTATION_GAMES,
    MAX_RANK,
    MIN_EXPECTATION_GAMES,
    OPPORTUNITY_GAMES,
    _lagged_rolling,
    fantasy_points,
    load_weekly,
)
from tools.calibrate_game_correlations import parse_seasons

POSITIONS = ("QB", "RB", "WR", "TE")
MIN_EXPECTED_POINTS = 2.0
MIN_PLAYED_GAMES = 150


def build_observations(weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per offensive player-game, with opportunity and points."""
    frame = weekly[weekly["position_group"].isin(POSITIONS)].copy()
    frame = frame.sort_values(["player_id", "season", "week"])
    frame["fantasy_points"] = fantasy_points(frame)
    frame["touches"] = (
        frame["carries"].fillna(0).astype(float)
        + frame["targets"].fillna(0).astype(float)
    )
    dropbacks = (
        frame["attempts"].fillna(0).astype(float)
        if "attempts" in frame else pd.Series(0.0, index=frame.index)
    )
    frame["opportunity"] = frame["touches"] + dropbacks
    frame["expected_points"] = _lagged_rolling(
        frame, "fantasy_points", EXPECTATION_GAMES, MIN_EXPECTATION_GAMES
    )
    frame["expected_touches"] = _lagged_rolling(frame, "touches", OPPORTUNITY_GAMES, 1)
    frame = frame.dropna(subset=["expected_points", "expected_touches"])

    # A player the expectation puts near zero tells us nothing about a pool whose
    # smallest published mean is around one point, and his ratio is unstable.
    frame = frame[frame["expected_points"] >= MIN_EXPECTED_POINTS]

    frame = frame.sort_values(
        ["season", "week", "team", "position_group", "expected_touches", "expected_points"],
        ascending=[True, True, True, True, False, False],
    )
    frame["rank"] = (
        frame.groupby(["season", "week", "team", "position_group"]).cumcount() + 1
    )
    frame["bucket"] = frame["rank"].clip(upper=MAX_RANK)
    return frame[["position_group", "bucket", "opportunity", "fantasy_points"]]


def fit(observations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (position, bucket), group in observations.groupby(["position_group", "bucket"]):
        played = group[group["opportunity"] > 0]
        if len(played) < MIN_PLAYED_GAMES:
            continue
        rate = float((played["fantasy_points"] <= 0).mean())
        error = np.sqrt(max(rate * (1 - rate), 0.0) / len(played))
        rows.append({
            "Pos": position, "Depth": int(bucket), "Games": len(group),
            "Played": len(played),
            "No opportunity": round(float((group["opportunity"] <= 0).mean()), 3),
            "Any zero": round(float((group["fantasy_points"] <= 0).mean()), 3),
            "Played zero": round(rate, 3),
            "SE": round(float(error), 4),
            "Live": nb.ZERO_RATE[position][int(bucket)],
        })
    table = pd.DataFrame(rows).sort_values(["Pos", "Depth"]).reset_index(drop=True)
    table["Gap"] = (table["Played zero"] - table["Live"]).round(3)
    return table


def representable(table: pd.DataFrame) -> pd.DataFrame:
    """Check each fitted rate leaves the conditional variance positive."""
    out = table.copy()
    out["CV"] = [nb.CALIBRATED_CV[p][d] for p, d in zip(out["Pos"], out["Depth"])]
    out["Conditional CV"] = [
        round(float(nb.conditional_lognormal_cv(cv, rate)), 3)
        for cv, rate in zip(out["CV"], out["Played zero"])
    ]
    out["Representable"] = [
        "yes" if nb.zero_rate_is_representable(cv, rate) else "NO"
        for cv, rate in zip(out["CV"], out["Played zero"])
    ]
    return out[["Pos", "Depth", "CV", "Played zero", "Conditional CV", "Representable"]]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seasons", default="2016-2025")
    parser.add_argument("--emit-python", action="store_true",
                        help="Print the fitted rates as a ZERO_RATE literal.")
    args = parser.parse_args(argv)

    observations = build_observations(load_weekly(parse_seasons(args.seasons)))
    table = fit(observations)
    print(f"\n{len(observations):,} player-games, seasons {args.seasons}\n")
    print(table.to_string(index=False))
    print("\n'Any zero' counts scoreless games the player may have been inactive for;")
    print("'Played zero' requires a carry, target or pass attempt and is what is published.")
    print("\nConditional-variance check:")
    print(representable(table).to_string(index=False))

    if args.emit_python:
        print("\nZERO_RATE = {")
        for position in POSITIONS + ("DEF",):
            rows = table[table["Pos"].eq(position)]
            if not len(rows):
                print(f'    "{position}": {{1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}},')
                continue
            values = {int(r.Depth): float(r._7) for r in rows.itertuples()}
            deepest = values[max(values)]
            body = ", ".join(f"{d}: {values.get(d, deepest):.3f}" for d in (1, 2, 3, 4))
            counts = " / ".join(f"{int(r.Played):,}" for r in rows.itertuples())
            print(f'    "{position}": {{{body}}},  # {counts} played games')
        print("}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
