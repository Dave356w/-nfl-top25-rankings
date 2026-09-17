"""The fixed-core team-outcome hypothesis, rebuilt on nflverse.

The theory under test, from docs/team-outcome-model-foundation.md: aggregating
individual fantasy expectations into a fixed positional core — 1 QB, 3 RB,
3 WR, 2 TE, 1 DEF, no salary cap, no flex — produces a team strength whose
home-minus-away difference carries signal for who wins.

The original test had 67 games, all from Yahoo Showdown slates, because the
player expectations came from Yahoo salaries. That corpus cannot be reached
from here and cannot be grown. This module keeps the hypothesis and changes the
instrument: a player's pregame expectation is the lagged mean of their own
recent fantasy production, which nflverse carries for every game back to 1999.

What that preserves is the structure — fixed positional slots, summed, and
differenced. What it gives up is Yahoo's pricing itself, which is a
market-informed consensus and is not the same thing as a rolling average. A
result here is evidence about the *shape* of the hypothesis, not about whether
Yahoo's numbers in particular carry information.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline import team_outcome as to


# The core exactly as the foundation document specifies it.
FIXED_CORE = (("QB", 1), ("RB", 3), ("WR", 3), ("TE", 2), ("DEF", 1))
CORE_POSITIONS = tuple(position for position, _ in FIXED_CORE)
CORE_SIZE = sum(count for _, count in FIXED_CORE)
FORM_WINDOW = 8  # the repository's existing "lagged eight-appearance average"

# Yahoo team-defence scoring. Points allowed is a tiered bonus, everything else
# is a flat event rate.
DEFENSIVE_WEIGHTS = {
    "def_sacks": 1.0, "def_interceptions": 2.0, "def_fumbles": 2.0,
    "def_tds": 6.0, "def_safeties": 2.0,
    "def_fg_blocks": 2.0, "def_pat_blocks": 2.0, "def_punt_blocks": 2.0,
    "special_teams_tds": 6.0,
}
POINTS_ALLOWED_TIERS = ((0, 10.0), (6, 7.0), (13, 4.0), (20, 1.0),
                        (27, 0.0), (34, -1.0), (10_000, -4.0))


def points_allowed_bonus(points) -> np.ndarray:
    allowed = pd.to_numeric(pd.Series(points), errors="coerce").to_numpy(float)
    bonus = np.full(allowed.shape, np.nan)
    previous = -1
    for ceiling, value in POINTS_ALLOWED_TIERS:
        bonus = np.where((allowed > previous) & (allowed <= ceiling), value, bonus)
        previous = ceiling
    return bonus


def defensive_points(team_weeks: pd.DataFrame, allowed: pd.Series) -> pd.Series:
    """Yahoo DST points for one team-week."""
    points = pd.Series(0.0, index=team_weeks.index, dtype=float)
    for column, weight in DEFENSIVE_WEIGHTS.items():
        if column in team_weeks:
            points += pd.to_numeric(team_weeks[column], errors="coerce").fillna(0.0) * weight
    return points + points_allowed_bonus(allowed)


def load_player_points(seasons=None, loader=None) -> pd.DataFrame:
    """Player-weeks scored in Yahoo half-PPR, the repository's own convention."""
    if loader is not None:
        frame = to._pandas(loader(seasons))
    else:
        try:
            import nflreadpy as nfl
        except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("Install nflreadpy to build the fixed-core corpus") from exc
        raw = (nfl.load_player_stats(summary_level="week") if seasons is None
               else nfl.load_player_stats(seasons=list(seasons), summary_level="week"))
        frame = to._pandas(raw)
    from pipeline.showdown_backtest import yahoo_offensive_points
    team = "team" if "team" in frame.columns else "recent_team"
    return pd.DataFrame({
        "season": frame.season, "week": frame.week, "team": frame[team],
        "unit_id": frame.player_id, "position": frame.position,
        "points": yahoo_offensive_points(frame),
    })


def load_defence_points(schedules: pd.DataFrame, seasons=None, loader=None) -> pd.DataFrame:
    """Team-weeks scored in Yahoo DST, with points allowed from the schedule."""
    if loader is not None:
        frame = to._pandas(loader(seasons))
    else:
        try:
            import nflreadpy as nfl
        except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("Install nflreadpy to build the fixed-core corpus") from exc
        raw = (nfl.load_team_stats(summary_level="week") if seasons is None
               else nfl.load_team_stats(seasons=list(seasons), summary_level="week"))
        frame = to._pandas(raw)
    frame = frame.copy()
    frame["team"] = frame.team.map(to._team)

    # Points allowed is the opponent's score, which the schedule already holds.
    home = schedules.rename(columns={"home_team": "team", "away_score": "allowed",
                                     "home_score": "scored"})[["season", "week", "team", "allowed"]]
    away = schedules.rename(columns={"away_team": "team", "home_score": "allowed"})[
        ["season", "week", "team", "allowed"]]
    allowed = pd.concat([home, away], ignore_index=True)
    allowed["team"] = allowed.team.map(to._team)
    allowed["season"] = allowed.season.astype(int)
    allowed["week"] = allowed.week.astype(int)
    frame["season"] = frame.season.astype(int)
    frame["week"] = frame.week.astype(int)
    merged = frame.merge(allowed, on=["season", "week", "team"], how="inner")
    return pd.DataFrame({
        "season": merged.season, "week": merged.week, "team": merged.team,
        "unit_id": "DEF:" + merged.team, "position": "DEF",
        "points": defensive_points(merged, merged.allowed),
    })


def build_fantasy(players: pd.DataFrame, defences: pd.DataFrame) -> pd.DataFrame:
    """One frame of settled unit-weeks: every core player, plus every defence."""
    frames = []
    if players is not None and len(players):
        frame = players.copy()
        frame["team"] = frame.team.map(to._team)
        frame["position"] = frame.position.astype(str).str.upper()
        frame = frame.loc[frame.position.isin(("QB", "RB", "WR", "TE"))]
        frames.append(frame.rename(columns={"player_id": "unit_id"})[list(to.FANTASY_COLUMNS)])
    if defences is not None and len(defences):
        frame = defences.copy()
        frame["team"] = frame.team.map(to._team)
        frame["unit_id"] = "DEF:" + frame.team
        frame["position"] = "DEF"
        frames.append(frame[list(to.FANTASY_COLUMNS)])
    if not frames:
        return to.EMPTY_FANTASY
    out = pd.concat(frames, ignore_index=True)
    out["season"] = out.season.astype(int)
    out["week"] = out.week.astype(int)
    out["points"] = pd.to_numeric(out.points, errors="coerce")
    out = out.loc[out.points.notna() & out.team.ne("") & out.unit_id.notna()]
    if out.duplicated(["season", "week", "team", "unit_id"]).any():
        raise ValueError("Duplicate unit-weeks in the fantasy frame")
    return out.sort_values(["season", "week", "team", "unit_id"]).reset_index(drop=True)


def attach_fantasy(corpus: to.Corpus, fantasy: pd.DataFrame) -> to.Corpus:
    return to.Corpus(games=corpus.games, outcomes=corpus.outcomes, market=corpus.market,
                     efficiency=corpus.efficiency, fantasy=build_fantasy_frame(fantasy))


def build_fantasy_frame(fantasy: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in to.FANTASY_COLUMNS if c not in fantasy.columns]
    if missing:
        raise ValueError(f"Fantasy frame is missing {missing}")
    return fantasy.reindex(columns=list(to.FANTASY_COLUMNS)).reset_index(drop=True)


def expectations(history: pd.DataFrame, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Each unit's pregame expectation and the team it most recently played for.

    The expectation is the lagged mean of that unit's own last `window`
    appearances. Team membership is the team it last appeared for, which is a
    fact about earlier weeks — asking who is on the roster *this* week would
    mean reading a pregame status feed the corpus deliberately does not carry.
    """
    if history is None or history.empty:
        return pd.DataFrame(columns=["unit_id", "team", "position", "expectation", "appearances"])
    ordered = history.sort_values(["season", "week"])
    recent = ordered.groupby("unit_id", sort=False).tail(window)
    grouped = recent.groupby("unit_id", sort=False)
    frame = grouped.agg(expectation=("points", "mean"), appearances=("points", "size")).reset_index()
    latest = ordered.groupby("unit_id", sort=False).tail(1)[["unit_id", "team", "position"]]
    return frame.merge(latest, on="unit_id", how="left")


def team_cores(history: pd.DataFrame, window: int = FORM_WINDOW,
               core=FIXED_CORE) -> pd.DataFrame:
    """Sum each team's fixed core from the expectations available so far."""
    available = expectations(history, window)
    if available.empty:
        return pd.DataFrame(columns=["team", "core", "slots_filled"])
    rows = []
    for position, count in core:
        block = available.loc[available.position.eq(position)]
        if block.empty:
            continue
        picked = (block.sort_values(["team", "expectation"], ascending=[True, False])
                       .groupby("team", sort=False).head(count))
        rows.append(picked)
    if not rows:
        return pd.DataFrame(columns=["team", "core", "slots_filled"])
    chosen = pd.concat(rows, ignore_index=True)
    return chosen.groupby("team").agg(core=("expectation", "sum"),
                                      slots_filled=("expectation", "size")).reset_index()


def core_edge(view: to.AsOfView, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Home-minus-away fixed-core difference, plus how complete each core was."""
    cores = team_cores(view.fantasy, window)
    index = view.context.index
    if cores.empty:
        return pd.DataFrame({"edge": pd.Series(0.0, index=index),
                             "complete": pd.Series(False, index=index)}, index=index)
    core = cores.set_index("team").core
    filled = cores.set_index("team").slots_filled
    home = view.context.home_team.map(core).astype(float)
    away = view.context.away_team.map(core).astype(float)
    home_slots = view.context.home_team.map(filled).astype(float).fillna(0)
    away_slots = view.context.away_team.map(filled).astype(float).fillna(0)
    return pd.DataFrame({
        "edge": (home.fillna(0.0) - away.fillna(0.0)),
        # The original analysis reported the strict subset where both teams
        # filled all ten slots separately, because the partial ones are a
        # different measurement. Carried through so it can be reported again.
        "complete": (home_slots.eq(CORE_SIZE) & away_slots.eq(CORE_SIZE)),
    }, index=index)


@to.feature("fixed_core_diff")
def _fixed_core_diff(view: to.AsOfView) -> pd.Series:
    return core_edge(view).edge


@to.feature("fixed_core_complete")
def _fixed_core_complete(view: to.AsOfView) -> pd.Series:
    return core_edge(view).complete.astype(float)


def tiers(edge: pd.Series, neutral: float) -> pd.Series:
    """The foundation document's ordinal reading: negative, neutral, positive."""
    return pd.Series(np.where(edge > neutral, "positive",
                              np.where(edge < -neutral, "negative", "neutral")),
                     index=edge.index)
