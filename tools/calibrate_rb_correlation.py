#!/usr/bin/env python3
"""Fit same-team RB-RB fantasy-score correlation from nflverse weekly stats.

The showdown correlation model carried one pooled ``("RB", "RB"): 0.013`` entry
for every same-team running-back pair. That single number cannot express the
thing that actually matters on a two-back slate: RB1 and RB2 are dividing one
finite pool of carries, goal-line attempts and clock-killing volume, so their
forecast errors should push against each other once the team's offensive
environment is held fixed, and they should push harder the closer together the
two backs are on the depth chart.

What is fitted
--------------
Two quantities per opportunity-rank pair, both on the *forecast error* scale the
rest of ``CALIBRATED_CV`` / ``SAME_TEAM_OTHER_CORR`` uses -- actual minus a
lagged rolling pregame expectation:

``raw``
    The marginal correlation of the two backs' forecast errors. This is the
    number the joint simulation model needs, because every other entry in the
    target matrix is a marginal correlation too. Feeding it a partial
    correlation while its neighbours are marginal would make the matrix
    internally inconsistent.

``partial``
    The same correlation after projecting out the team's non-RB offensive
    production that week. This isolates the cannibalization the raw number
    hides: a shared good day for the offense pushes both backs up at the same
    time as they take carries from each other, and the raw correlation is the
    sum of those two opposing effects.

Opportunity rank, not chart position
------------------------------------
Rank is assigned from a *lagged* rolling touch count (carries + targets) so no
in-game information leaks into the grouping. That mirrors how the pipeline now
treats depth: the published chart supplies role structure, expected opportunity
supplies the ordinal the fitted tables are keyed on.

Usage
-----
    python tools/calibrate_rb_correlation.py --seasons 2016-2025

Prints a fitted table and, with ``--emit-python``, the literal that belongs in
``pipeline/notebook.py``.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

RELEASE = "https://github.com/nflverse/nflverse-data/releases/download"
CACHE = Path(__file__).resolve().parent / ".nflverse_calibration_cache"

# Yahoo default offensive scoring, matching SCORING_PRESETS["yahoo"].
YAHOO = {
    "passing_yards": 0.04,
    "passing_tds": 4.0,
    "passing_interceptions": -1.0,
    "rushing_yards": 0.10,
    "rushing_tds": 6.0,
    "receiving_yards": 0.10,
    "receiving_tds": 6.0,
    "receptions": 0.50,
}
FUMBLE_COLUMNS = ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
FUMBLE_POINTS = -2.0

# Rank 4 is the "4 or deeper" bucket, matching _depth_bucket in the pipeline.
MAX_RANK = 4
EXPECTATION_GAMES = 8      # matches OFFENSE_FALLBACK_GAMES
MIN_EXPECTATION_GAMES = 3
OPPORTUNITY_GAMES = 4
MIN_PAIR_SAMPLES = 150     # below this a pair is reported but not published


# nflverse republished the weekly player stats under a second release tag and
# the two are not complete mirrors of each other -- 2019 and 2025 are missing
# from `player_stats`. Try the newer tag first and fall back rather than losing
# two seasons of sample to an asset-naming accident.
WEEKLY_STATS_ASSETS = (
    ("stats_player", "stats_player_week_{season}.csv.gz"),
    ("player_stats", "stats_player_week_{season}.csv.gz"),
    ("player_stats", "player_stats_{season}.csv.gz"),
)


def _download(dataset: str, filename: str) -> Path:
    target = CACHE / f"{dataset}__{filename}"
    if target.exists():
        return target
    CACHE.mkdir(parents=True, exist_ok=True)
    url = f"{RELEASE}/{dataset}/{filename}"
    request = Request(url, headers={"User-Agent": "Mozilla/5.0 nfl-top25-calibration"})
    with urlopen(request, timeout=60) as response:
        target.write_bytes(response.read())
    return target


def _download_weekly(season: int) -> Path:
    last = None
    for dataset, pattern in WEEKLY_STATS_ASSETS:
        try:
            return _download(dataset, pattern.format(season=season))
        except HTTPError as exc:
            last = exc
    raise RuntimeError(f"No weekly stats asset for {season}") from last


def load_weekly(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = _download_weekly(season)
        frames.append(pd.read_csv(path, low_memory=False))
    weekly = pd.concat(frames, ignore_index=True)
    return weekly[weekly["season_type"].eq("REG")].copy()


def fantasy_points(frame: pd.DataFrame) -> pd.Series:
    points = pd.Series(0.0, index=frame.index)
    for column, weight in YAHOO.items():
        if column in frame:
            points += frame[column].fillna(0).astype(float) * weight
    for column in FUMBLE_COLUMNS:
        if column in frame:
            points += frame[column].fillna(0).astype(float) * FUMBLE_POINTS
    return points


def _lagged_rolling(frame: pd.DataFrame, column: str, games: int, min_games: int) -> pd.Series:
    """Mean of a player's previous ``games`` appearances, excluding the current one."""
    grouped = frame.groupby("player_id", sort=False)[column]
    return grouped.transform(
        lambda values: values.shift(1).rolling(games, min_periods=min_games).mean()
    )


def build_observations(weekly: pd.DataFrame) -> pd.DataFrame:
    """One row per running back game with a forecast error and an opportunity rank."""
    weekly = weekly.sort_values(["player_id", "season", "week"]).copy()
    weekly["fantasy_points"] = fantasy_points(weekly)
    weekly["touches"] = (
        weekly["carries"].fillna(0).astype(float) + weekly["targets"].fillna(0).astype(float)
    )

    # The team's environment is its non-RB offensive production: quarterback,
    # receiver and tight-end scoring on the same team-week. It moves with game
    # script and passing volume without being mechanically tied to how the
    # backfield splits its own carries.
    offense = weekly[weekly["position_group"].isin(["QB", "WR", "TE", "RB"])]
    non_rb = offense[offense["position_group"].ne("RB")]
    environment = (
        non_rb.groupby(["season", "week", "team"])["fantasy_points"].sum().rename("team_env")
    )

    backs = weekly[weekly["position_group"].eq("RB")].copy()
    backs["expected_points"] = _lagged_rolling(
        backs, "fantasy_points", EXPECTATION_GAMES, MIN_EXPECTATION_GAMES
    )
    backs["expected_touches"] = _lagged_rolling(
        backs, "touches", OPPORTUNITY_GAMES, 1
    )
    backs = backs.dropna(subset=["expected_points", "expected_touches"])
    backs = backs[backs["expected_points"] > 0]

    # Opportunity rank inside the team-week, highest lagged touch load first.
    # Ties break on the lagged points expectation so the ordering is total.
    backs = backs.sort_values(
        ["season", "week", "team", "expected_touches", "expected_points"],
        ascending=[True, True, True, False, False],
    )
    backs["rank"] = backs.groupby(["season", "week", "team"]).cumcount() + 1
    backs["bucket"] = backs["rank"].clip(upper=MAX_RANK)

    backs = backs.merge(environment, on=["season", "week", "team"], how="left")
    env_frame = environment.reset_index().sort_values(["team", "season", "week"])
    env_frame["expected_env"] = env_frame.groupby("team", sort=False)["team_env"].transform(
        lambda values: values.shift(1).rolling(EXPECTATION_GAMES, min_periods=MIN_EXPECTATION_GAMES).mean()
    )
    backs = backs.merge(
        env_frame[["season", "week", "team", "expected_env"]],
        on=["season", "week", "team"],
        how="left",
    )
    backs = backs.dropna(subset=["team_env", "expected_env"])

    backs["error"] = backs["fantasy_points"] - backs["expected_points"]
    backs["touch_error"] = backs["touches"] - backs["expected_touches"]
    backs["env_error"] = backs["team_env"] - backs["expected_env"]
    return backs[[
        "season", "week", "team", "player_id", "player_display_name",
        "rank", "bucket", "fantasy_points", "expected_points", "error",
        "touch_error", "env_error", "expected_touches",
    ]]


def _standardize(values: np.ndarray) -> np.ndarray:
    """Scale forecast errors so one enormous outlier week cannot set the fit."""
    spread = np.std(values)
    return values / spread if spread > 0 else values


def _residualize(values: np.ndarray, control: np.ndarray) -> np.ndarray:
    """Remove the least-squares projection of ``values`` onto ``control``."""
    design = np.column_stack([np.ones_like(control), control])
    coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
    return values - design @ coefficients


def fit_pairs(observations: pd.DataFrame) -> pd.DataFrame:
    """Correlate same-team forecast errors for every opportunity-rank pair."""
    # Two backs can share a bucket only at 4+, where several deep backs collapse
    # into one label. Keep the shallowest of them so each cell holds one player.
    deduped = observations.sort_values("rank").drop_duplicates(
        ["season", "week", "team", "bucket"], keep="first"
    )

    def spread(column):
        return deduped.pivot_table(
            index=["season", "week", "team"], columns="bucket", values=column
        )

    wide_error = spread("error")
    wide_touch = spread("touch_error")
    wide_env = spread("env_error")

    rows = []
    for first, second in itertools.combinations(range(1, MAX_RANK + 1), 2):
        if first not in wide_error or second not in wide_error:
            continue
        block = pd.DataFrame({
            "a": wide_error[first],
            "b": wide_error[second],
            "touch_a": wide_touch[first],
            "touch_b": wide_touch[second],
            "env": wide_env[first],
        }).dropna()
        if len(block) < 30:
            continue
        a = _standardize(block["a"].to_numpy(float))
        b = _standardize(block["b"].to_numpy(float))
        env = block["env"].to_numpy(float)
        raw = float(np.corrcoef(a, b)[0, 1])
        # The same correlation measured on touches instead of points. Carries
        # and targets are the resource the two backs are actually splitting, so
        # this is where cannibalization is visible; comparing the two columns
        # shows how much of it survives into fantasy scoring.
        touches = float(
            np.corrcoef(
                _standardize(block["touch_a"].to_numpy(float)),
                _standardize(block["touch_b"].to_numpy(float)),
            )[0, 1]
        )
        partial = float(
            np.corrcoef(_residualize(a, env), _residualize(b, env))[0, 1]
        )
        # Fisher-z standard error; the sample is large enough for the normal
        # approximation and it is the honest way to say how firm each cell is.
        stderr = 1.0 / np.sqrt(max(len(block) - 3, 1))
        rows.append({
            "pair": f"RB{first}-RB{second}" + ("+" if second == MAX_RANK else ""),
            "rank_a": first,
            "rank_b": second,
            "games": len(block),
            "raw": round(raw, 4),
            "partial": round(partial, 4),
            "touches": round(touches, 4),
            "z_stderr": round(stderr, 4),
            "raw_low": round(np.tanh(np.arctanh(raw) - 1.96 * stderr), 4),
            "raw_high": round(np.tanh(np.arctanh(raw) + 1.96 * stderr), 4),
            "publishable": len(block) >= MIN_PAIR_SAMPLES,
        })
    return pd.DataFrame(rows)


def emit_python(fitted: pd.DataFrame, seasons: list[int]) -> str:
    lines = [
        "# Fitted by tools/calibrate_rb_correlation.py over "
        f"{seasons[0]}-{seasons[-1]} nflverse weekly stats.",
        "SAME_TEAM_RB_RB_CORR = {",
    ]
    for row in fitted.itertuples():
        if not row.publishable:
            continue
        lines.append(
            f"    ({row.rank_a}, {row.rank_b}): {row.raw:+.3f},"
            f"  # {row.games:,} team-games, touches {row.touches:+.3f}"
        )
    lines.append("}")
    return "\n".join(lines)


def parse_seasons(text: str) -> list[int]:
    if "-" in text:
        start, end = text.split("-", 1)
        return list(range(int(start), int(end) + 1))
    return [int(part) for part in text.split(",") if part.strip()]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", default="2016-2025",
                        help="Season range (2016-2025) or list (2019,2020).")
    parser.add_argument("--emit-python", action="store_true",
                        help="Print the dict literal for pipeline/notebook.py.")
    args = parser.parse_args(argv)

    seasons = parse_seasons(args.seasons)
    weekly = load_weekly(seasons)
    observations = build_observations(weekly)
    fitted = fit_pairs(observations)

    print(f"Seasons {seasons[0]}-{seasons[-1]}: {len(observations):,} running-back games")
    print(fitted.to_string(index=False))
    print(
        "\nraw     = marginal forecast-error correlation (what the model matrix takes)"
        "\npartial = after projecting out the team's non-RB offensive forecast error"
        "\ntouches = the same correlation measured on carries + targets"
        "\n\nraw_low/raw_high are the 95% Fisher-z interval. A pair whose interval"
        "\nstraddles zero is published as fitted, not as a confident sign."
    )
    if args.emit_python:
        print()
        print(emit_python(fitted, seasons))
    return 0


if __name__ == "__main__":
    sys.exit(main())
