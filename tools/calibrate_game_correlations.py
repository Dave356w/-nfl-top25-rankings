#!/usr/bin/env python3
"""Fit every same-game pair correlation in the showdown model from nflverse.

`calibrate_rb_correlation.py` fits one family of pairs, the same-team running
backs. This fits all of them -- every same-team and opposing offensive pair the
simulation's target matrix carries -- so the constants in
`pipeline/notebook.py` can be checked against data instead of taken on trust.

Why this exists
---------------
The showdown model is a correlated lognormal over one game's players, and the
single number that decides whether stacking is worth anything is the pairwise
correlation. `QB_WR_CORR`, `QB_TE_CORR`, `OPPOSING_OFFENSE_CORR` and
`SAME_TEAM_OTHER_CORR` were carried into the repo as asserted fits with no
script behind them, and their values are small enough (mean absolute
off-diagonal 0.023 on a live pool) that a reader is entitled to ask whether the
model is treating one football game as thirty-two independent players by
mistake. It is not: on 2016-2025 the measured values agree with the table.

What is fitted
--------------
The same *forecast error* scale the rest of the model uses -- actual minus a
lagged eight-appearance rolling expectation, per player, with no in-game
information in the expectation. Errors are standardized inside each
(position, opportunity bucket) group before pooling, so a pair's correlation is
not set by whichever players happen to score the most points.

Opportunity rank is assigned from a lagged four-game touch count within the
team-week and position, clipped at 4, mirroring `_depth_bucket`.

Pairs are formed only between players who actually appeared in the same game,
which is what the showdown model simulates.

Read the result as an upper bound
---------------------------------
The expectation here is a lagged rolling average, which does not know the game
total. A shootout is therefore a surprise to it, and part of what it books as
correlated error is really the shared game environment. The pipeline's means are
market-implied and already price that environment, so the correlation left over
for its residuals is if anything *lower* than what this prints. That asymmetry
is the reason not to read a positive gap here as a licence to raise the table.

Team defenses are not covered: weekly player stats carry no DST scoring, so
`OPPONENT_DEF_CORR` and the DEF entries still rest on their original fit.

Usage
-----
    python tools/calibrate_game_correlations.py --seasons 2016-2025
    python tools/calibrate_game_correlations.py --seasons 2016-2025 --emit-python

Like the RB tool this is offline work run by hand. The daily build never runs it.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from collections import defaultdict
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

POSITIONS = ("QB", "RB", "WR", "TE")
MIN_PAIR_SAMPLES = 300


def build_observations(weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per offensive player-game with a standardized forecast error."""
    frame = weekly[weekly["position_group"].isin(POSITIONS)].copy()
    frame = frame.sort_values(["player_id", "season", "week"])
    frame["fantasy_points"] = fantasy_points(frame)
    frame["touches"] = (
        frame["carries"].fillna(0).astype(float)
        + frame["targets"].fillna(0).astype(float)
    )
    frame["expected_points"] = _lagged_rolling(
        frame, "fantasy_points", EXPECTATION_GAMES, MIN_EXPECTATION_GAMES
    )
    frame["expected_touches"] = _lagged_rolling(
        frame, "touches", OPPORTUNITY_GAMES, 1
    )
    frame = frame.dropna(subset=["expected_points", "expected_touches"])
    frame = frame[frame["expected_points"] > 0]

    # Opportunity rank inside the team-week *and position*, highest lagged touch
    # load first, which is the ordinal CALIBRATED_CV and the pair tables are
    # keyed on. Ties break on the lagged points expectation so it is total.
    frame = frame.sort_values(
        ["season", "week", "team", "position_group", "expected_touches", "expected_points"],
        ascending=[True, True, True, True, False, False],
    )
    frame["rank"] = (
        frame.groupby(["season", "week", "team", "position_group"]).cumcount() + 1
    )
    frame["bucket"] = frame["rank"].clip(upper=MAX_RANK)
    frame["error"] = frame["fantasy_points"] - frame["expected_points"]

    # Standardize within (position, bucket). Pooling raw errors would let a
    # bucket of high-scoring players set a correlation meant to describe all of
    # them.
    frame["z"] = frame.groupby(["position_group", "bucket"])["error"].transform(
        lambda values: values / values.std() if values.std() > 0 else values
    )
    frame["game"] = [
        f"{season}-{week}-" + "-".join(sorted([str(team), str(opponent)]))
        for season, week, team, opponent in zip(
            frame.season, frame.week, frame.team, frame.opponent_team
        )
    ]
    return frame[["game", "team", "position_group", "bucket", "z"]]


def fit(observations: pd.DataFrame, by_depth: bool = False) -> pd.DataFrame:
    """Pool every same-game pair by relationship and report its correlation."""
    buckets: dict = defaultdict(lambda: ([], []))
    for _, game in observations.groupby("game", sort=False):
        rows = game.to_dict("records")
        for first, second in itertools.combinations(rows, 2):
            a = {"Position": first["position_group"], "Team": first["team"],
                 "Depth_Rank": first["bucket"]}
            b = {"Position": second["position_group"], "Team": second["team"],
                 "Depth_Rank": second["bucket"]}
            key = nb._pair_relationship(a, b)
            if by_depth:
                key = (key, min(a["Depth_Rank"], b["Depth_Rank"]),
                       max(a["Depth_Rank"], b["Depth_Rank"]))
            buckets[key][0].append(first["z"])
            buckets[key][1].append(second["z"])

    rows = []
    for key, (left, right) in buckets.items():
        left, right = np.array(left), np.array(right)
        count = len(left)
        if count < MIN_PAIR_SAMPLES:
            continue
        correlation = float(np.corrcoef(left, right)[0, 1])
        error = 1 / np.sqrt(count - 3)
        low = np.tanh(np.arctanh(correlation) - 1.96 * error)
        high = np.tanh(np.arctanh(correlation) + 1.96 * error)
        relationship = key[0] if by_depth else key
        row = {"Relationship": relationship, "Pairs": count,
               "Measured": round(correlation, 3),
               "CI low": round(float(low), 3), "CI high": round(float(high), 3)}
        if by_depth:
            row["Depths"] = f"{key[1]}-{key[2]}"
        rows.append(row)
    order = ["Relationship"] + (["Depths"] if by_depth else [])
    if not rows:
        # Every relationship fell under MIN_PAIR_SAMPLES. Hand back the right
        # columns so a caller can print or compare an empty result instead of
        # tripping over a bare DataFrame.
        columns = ["Relationship", "Pairs", "Measured", "CI low", "CI high"]
        return pd.DataFrame(columns=columns + (["Depths"] if by_depth else []))
    return pd.DataFrame(rows).sort_values(order).reset_index(drop=True)


def model_target(relationship: str, depth: int = 1):
    """The value the live table returns for a representative pair, or None."""
    parts = relationship.split(" ")
    if parts[0] == "Same" and parts[1] == "team":
        same, pair = True, parts[2]
    elif parts[0] == "Opposing":
        same, pair = False, parts[1]
    else:
        return None
    first, second = pair.split("-")
    if first not in POSITIONS or second not in POSITIONS:
        return None
    a = {"Position": first, "Team": "AAA", "Depth_Rank": depth}
    b = {"Position": second, "Team": "AAA" if same else "BBB", "Depth_Rank": depth}
    return round(float(nb.target_score_correlation(a, b)), 3)


def compare(fitted: pd.DataFrame) -> pd.DataFrame:
    """Put the live constant beside the measurement and flag real disagreement."""
    out = fitted.copy()
    out["Model"] = [model_target(name) for name in out["Relationship"]]
    out["Gap"] = (out["Measured"] - out["Model"]).round(3)
    out["Agrees"] = [
        "yes" if model is not None and low <= model <= high else "REVIEW"
        for model, low, high in zip(out["Model"], out["CI low"], out["CI high"])
    ]
    return out


def parse_seasons(text: str) -> list[int]:
    if "-" in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in text.split(",")]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--seasons", default="2016-2025",
                        help="Season range, e.g. 2016-2025, or a comma list.")
    parser.add_argument("--by-depth", action="store_true",
                        help="Break each relationship out by depth-bucket pair.")
    parser.add_argument("--emit-python", action="store_true",
                        help="Print the measured values as a Python literal.")
    args = parser.parse_args(argv)

    weekly = load_weekly(parse_seasons(args.seasons))
    observations = build_observations(weekly)
    fitted = fit(observations, by_depth=args.by_depth)
    print(f"\n{len(observations):,} player-games, "
          f"{int(fitted['Pairs'].sum()):,} same-game pairs, seasons {args.seasons}\n")
    if args.by_depth:
        print(fitted.to_string(index=False))
        return 0

    table = compare(fitted)
    print(table.to_string(index=False))
    disagree = table[table["Agrees"].eq("REVIEW")]
    print(f"\n{len(disagree)} of {len(table)} relationships fall outside the "
          "95% interval of the measurement.")
    print("Read every gap against the docstring's upper-bound caveat before "
          "moving a constant: this expectation does not know the game total and "
          "the pipeline's market means do.")

    if args.emit_python:
        print("\n# Measured same-game correlations, "
              f"seasons {args.seasons}. Pairs in the comment.")
        for row in table.itertuples():
            print(f"# {row.Relationship}: {row.Measured:+.3f} "
                  f"[{row._4:+.3f}, {row._5:+.3f}]  ({row.Pairs:,} pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
