"""Leakage-safe NFL game corpus and as-of feature harness.

Phase P0 of docs/team-outcome-model-plan.md.

The corpus is split into three frames on purpose. ``games`` carries identity
and pregame context and never carries a score or a closing line; ``outcomes``
carries the targets; ``market`` carries closing lines as an evaluation
benchmark only. nflverse ships all three in one schedules frame under one join
key, so a feature can reach a closing line or a final score by accident. Here
it cannot: the frame handed to a feature does not contain them.

A feature never selects its own rows either. It receives an ``AsOfView`` whose
``history`` holds settled games strictly earlier than the target week, so the
as-of rule is a property of the data handed over rather than a convention the
feature is trusted to follow. ``audit`` tests that boundary from both sides: it asserts
the view holds nothing from the checkpoint onward, and it replaces every later
outcome and closing line and re-derives, so a wrong comparison in the filter
shows up as a feature that moves.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


# Franchise continuity. A team's rolling history has to follow the franchise
# through relocation, and nflverse schedules keep the historical code: the Rams
# appear as STL, then LA, then LAR. Mirrors pipeline/showdown_backtest.py.
TEAM_ALIASES = {
    "JAC": "JAX", "LA": "LAR", "STL": "LAR", "SD": "LAC",
    "OAK": "LV", "WSH": "WAS",
}

# Identity and pregame context: fixed when the schedule is published, or (rest
# days) derived from it. Safe to hand a feature for the target week itself.
CONTEXT_COLUMNS = (
    "game_id", "season", "week", "game_type", "gameday", "gametime",
    "home_team", "away_team", "location", "roof", "surface", "div_game",
    "home_rest", "away_rest", "stadium_id",
)

# Settled results. `result` is the home margin and `total` the combined score:
# both are the target wearing an innocuous name next to `spread_line`.
OUTCOME_SOURCE_COLUMNS = ("home_score", "away_score")

# Closing lines. Benchmark only, never a feature. Kept in their own frame so
# that reaching one requires naming the frame.
MARKET_COLUMNS = (
    "home_moneyline", "away_moneyline", "spread_line", "home_spread_odds",
    "away_spread_odds", "total_line", "over_odds", "under_odds",
)

# Dropped outright: settled results under another name, and fields nflverse
# records after the fact or restates (recorded weather, the quarterback who
# actually started, officials). None of them is available pregame as published.
WITHHELD_COLUMNS = (
    "result", "total", "overtime", "temp", "wind", "referee",
    "home_qb_id", "away_qb_id", "home_qb_name", "away_qb_name",
    "home_coach", "away_coach",
)

REQUIRED_COLUMNS = CONTEXT_COLUMNS + OUTCOME_SOURCE_COLUMNS

# The final holdout of docs/team-outcome-model-plan.md §6.1: the two most recent
# complete seasons, read once at P4 and never before. Sealing is a reporting
# rule, not a loading rule — the corpus has to know these games exist to count
# coverage and to walk forward into them later — so nothing that summarizes an
# outcome may include them until the seal is opened.
SEALED_SEASONS = (2024, 2025)

# Expanding-window training needs a floor before the first evaluated season.
MINIMUM_TRAINING_SEASONS = 8

ELO_START = 1500.0
ELO_K = 20.0
ELO_SCALE = 400.0
ELO_SEASON_REGRESSION = 1.0 / 3.0
# A literature constant, not a value fitted here. P1 reports the textbook Elo
# alongside one whose intercept and slope are fitted per fold, so the choice of
# this number is visible in the gap between them rather than buried.
ELO_HOME_ADVANTAGE = 65.0


ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "docs" / "team-outcome-prereg.md"
PREREG_RECORD = ROOT / "model" / "team_outcome_prereg.json"


def prereg_sha256(path: Path = PREREG_PATH) -> str:
    """Hash the preregistration exactly as it sits on disk."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_prereg(record: Path = PREREG_RECORD, path: Path = PREREG_PATH) -> dict:
    """Confirm the frozen hash still matches the document.

    This is the whole enforcement mechanism. A preregistration nobody checks is
    a wish; one whose hash is asserted in the test suite cannot be edited after
    the fact without the edit becoming visible in a failing build.
    """
    frozen = json.loads(Path(record).read_text(encoding="utf-8"))
    actual = prereg_sha256(path)
    if actual != frozen["sha256"]:
        raise ValueError(
            f"{Path(path).name} has changed since it was frozen: recorded "
            f"{frozen['sha256'][:12]}, found {actual[:12]}. Amending a "
            "preregistration requires a new frozen version recording the "
            "previous hash and the reason; see its Amendments section."
        )
    return frozen


def unsealed(frame: pd.DataFrame, sealed=SEALED_SEASONS) -> pd.DataFrame:
    """Drop sealed seasons. Every outcome summary goes through this."""
    return frame.loc[~frame.season.isin(sealed)]


def evaluation_seasons(corpus, sealed=SEALED_SEASONS,
                       minimum_training=MINIMUM_TRAINING_SEASONS) -> list[int]:
    """Seasons a walk-forward run may score: unsealed, and far enough in."""
    seasons = sorted(int(s) for s in corpus.games.season.unique() if s not in sealed)
    return seasons[minimum_training:]


def _pandas(frame) -> pd.DataFrame:
    # nflreadpy returns polars; to_pandas needs pyarrow, which the pipeline does
    # not require. Same fallback as pipeline/showdown_backtest.py.
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    try:
        return frame.to_pandas()
    except (AttributeError, ModuleNotFoundError):
        return pd.DataFrame(frame.to_dicts())


def _team(value) -> str:
    code = str(value or "").upper()
    return TEAM_ALIASES.get(code, code)


def load_schedules(seasons=None, loader=None) -> pd.DataFrame:
    """Pull nflverse schedules. `loader` is the seam the tests and cache use."""
    if loader is not None:
        return _pandas(loader(seasons))
    try:
        import nflreadpy as nfl
    except ModuleNotFoundError as exc:  # pragma: no cover - environment guard
        raise RuntimeError("Install nflreadpy to build the team-outcome corpus") from exc
    return _pandas(nfl.load_schedules() if seasons is None else nfl.load_schedules(seasons=list(seasons)))


@dataclass(frozen=True)
class AsOfView:
    """Everything a feature for one week is allowed to see, and nothing else."""

    season: int
    week: int
    context: pd.DataFrame   # the target week, pregame columns only
    history: pd.DataFrame   # settled games strictly earlier, with outcomes
    schedule: pd.DataFrame  # published schedule through this week, pregame only


@dataclass(frozen=True)
class Corpus:
    games: pd.DataFrame      # one row per scheduled game, pregame context only
    outcomes: pd.DataFrame   # one row per settled game: margin, home_win, tie
    market: pd.DataFrame     # one row per game with a closing line

    def weeks(self, settled_only: bool = False) -> list[tuple[int, int]]:
        frame = self.games
        if settled_only:
            frame = frame[frame.game_id.isin(self.outcomes.game_id)]
        pairs = frame[["season", "week"]].drop_duplicates().sort_values(["season", "week"])
        return [(int(s), int(w)) for s, w in pairs.itertuples(index=False)]

    def context(self, season: int, week: int) -> pd.DataFrame:
        mask = self.games.season.eq(season) & self.games.week.eq(week)
        return self.games.loc[mask].reset_index(drop=True)

    def schedule_through(self, season: int, week: int) -> pd.DataFrame:
        order = self.games.season * 100 + self.games.week
        return self.games.loc[order.le(season * 100 + week)].reset_index(drop=True)

    def history(self, season: int, week: int) -> pd.DataFrame:
        """Settled games strictly earlier than (season, week).

        Whole weeks, not kickoff timestamps. A Thursday result really is known
        before Sunday, but spending that information costs one game's worth of
        history and buys a class of ordering bug; the week boundary cannot leak.
        """
        joined = self.games.merge(self.outcomes, on="game_id", how="inner")
        order = joined.season * 100 + joined.week
        earlier = joined.loc[order.lt(season * 100 + week)]
        return earlier.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    def view(self, season: int, week: int) -> AsOfView:
        return AsOfView(int(season), int(week), self.context(season, week),
                        self.history(season, week), self.schedule_through(season, week))


def build_corpus(raw: pd.DataFrame) -> Corpus:
    """Validate a schedules frame and split it into the three frames above."""
    missing = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"Schedules frame is missing required columns: {missing}")

    frame = raw.reset_index(drop=True)
    frame["season"] = pd.to_numeric(frame["season"], errors="raise").astype(int)
    frame["week"] = pd.to_numeric(frame["week"], errors="raise").astype(int)
    if frame.game_id.duplicated().any():
        duplicates = sorted(frame.loc[frame.game_id.duplicated(), "game_id"].unique())
        raise ValueError(f"Duplicate game_id in schedules: {duplicates[:5]}")

    home = pd.to_numeric(frame["home_score"], errors="coerce")
    away = pd.to_numeric(frame["away_score"], errors="coerce")
    partial = home.isna() ^ away.isna()
    if partial.any():
        raise ValueError(f"{int(partial.sum())} games carry one score and not the other")

    # (season, week) is the sort key for the as-of rule, so postseason weeks
    # must continue the regular-season numbering rather than restart.
    for season, block in frame.groupby("season"):
        post = block.loc[block.game_type.ne("REG"), "week"]
        regular = block.loc[block.game_type.eq("REG"), "week"]
        if len(post) and len(regular) and int(post.min()) <= int(regular.max()):
            raise ValueError(f"Season {season}: postseason week numbering overlaps the regular season")

    games = frame.reindex(columns=list(CONTEXT_COLUMNS)).copy()
    games["home_team"] = games.home_team.map(_team)
    games["away_team"] = games.away_team.map(_team)
    games["neutral"] = games.location.astype(str).str.lower().eq("neutral")
    # roof is missing for 43 older games; an unknown roof is not a dome.
    games["roof"] = games.roof.astype("string").fillna("unknown")
    games = games.sort_values(["season", "week", "game_id"]).reset_index(drop=True)

    settled = frame.loc[home.notna()].copy()
    margin = home.loc[settled.index] - away.loc[settled.index]
    outcomes = pd.DataFrame({
        "game_id": settled.game_id.to_numpy(),
        "home_score": home.loc[settled.index].astype(float).to_numpy(),
        "away_score": away.loc[settled.index].astype(float).to_numpy(),
        "margin": margin.astype(float).to_numpy(),
        # A tie is half a win. The 2025 fixed-core run dropped its one tie,
        # which is defensible at n=67 and not a rule a corpus can carry.
        "home_win": np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, 0.5)),
        "tie": (margin == 0).to_numpy(),
    })

    present = [c for c in MARKET_COLUMNS if c in frame.columns]
    market = frame.reindex(columns=["game_id", *present]).copy()
    market = market.loc[market[present].notna().any(axis=1)].reset_index(drop=True) if present else market

    overlap = set(games.columns) & (set(MARKET_COLUMNS) | set(OUTCOME_SOURCE_COLUMNS) | set(WITHHELD_COLUMNS))
    if overlap:  # the whole point of the split; assert it rather than assume it
        raise AssertionError(f"Context frame leaked withheld columns: {sorted(overlap)}")
    return Corpus(games=games, outcomes=outcomes, market=market)


FEATURES: dict[str, callable] = {}


def feature(name: str):
    """Register a feature. Every one takes an AsOfView and returns a Series."""
    def register(fn):
        FEATURES[name] = fn
        return fn
    return register


def _shrink(values: pd.Series, counts: pd.Series, prior_games: float) -> pd.Series:
    # n/(n+k) toward the league mean of zero: a team with two games played is
    # not evidence of a two-game team.
    return values.fillna(0.0) * (counts / (counts + prior_games))


def _team_margins(history: pd.DataFrame) -> pd.DataFrame:
    """One row per team-game, margin from that team's point of view."""
    if history.empty:
        return pd.DataFrame(columns=["team", "season", "week", "margin"])
    home = history[["home_team", "season", "week", "margin"]].rename(columns={"home_team": "team"})
    away = history[["away_team", "season", "week", "margin"]].rename(columns={"away_team": "team"})
    away = away.assign(margin=-away.margin)
    return pd.concat([home, away], ignore_index=True).sort_values(["season", "week"])


@feature("home_field")
def _home_field(view: AsOfView) -> pd.Series:
    return pd.Series(1.0, index=view.context.index)


@feature("rest_diff")
def _rest_diff(view: AsOfView) -> pd.Series:
    home = pd.to_numeric(view.context.home_rest, errors="coerce")
    away = pd.to_numeric(view.context.away_rest, errors="coerce")
    return (home - away).astype(float).fillna(0.0)


@feature("short_week_diff")
def _short_week_diff(view: AsOfView) -> pd.Series:
    # Directional by construction. A symmetric "either side is on a short week"
    # indicator cannot carry a sign on a home-margin target, which is the
    # defect this replaces; see the plan's feature table.
    home = pd.to_numeric(view.context.home_rest, errors="coerce").le(4).astype(float)
    away = pd.to_numeric(view.context.away_rest, errors="coerce").le(4).astype(float)
    return away - home


@feature("bye_diff")
def _bye_diff(view: AsOfView) -> pd.Series:
    """Off a bye, read from the published schedule rather than a rest threshold.

    Whether a team has a game in the previous week is a fact about the schedule,
    which the league publishes in May. It is not an outcome, so the target
    week's own schedule row is fair to read.
    """
    previous = view.schedule[
        view.schedule.season.eq(view.season) & view.schedule.week.eq(view.week - 1)
    ]
    played = set(previous.home_team) | set(previous.away_team)
    if view.week <= 1 or not played:  # week 1 and the postseason opener: nobody is "off a bye"
        return pd.Series(0.0, index=view.context.index)
    home_off = (~view.context.home_team.isin(played)).astype(float)
    away_off = (~view.context.away_team.isin(played)).astype(float)
    return home_off - away_off


@feature("neutral_site")
def _neutral_site(view: AsOfView) -> pd.Series:
    return view.context.neutral.astype(float)


@feature("dome")
def _dome(view: AsOfView) -> pd.Series:
    return view.context.roof.astype(str).str.lower().isin(("dome", "closed")).astype(float)


@feature("div_game")
def _div_game(view: AsOfView) -> pd.Series:
    return pd.to_numeric(view.context.div_game, errors="coerce").fillna(0.0).astype(float)


@feature("crowd_absent")
def _crowd_absent(view: AsOfView) -> pd.Series:
    """The 2020 season, played largely without spectators.

    A training-time control, not a prediction-time lever: for any other season
    it is zero. It exists because P0 measured home advantage at 49.8% in 2020
    against 54.8% in 2021-2023, and without it that season drags the fitted
    home-field term for every later fold. The justification is an external fact
    about those games rather than a pattern found in the data.
    """
    return view.context.season.eq(2020).astype(float)


@feature("prior_margin_diff")
def _prior_margin_diff(view: AsOfView, window: int = 8, prior_games: float = 4.0) -> pd.Series:
    """Shrunk difference in recent scoring margin, the harness's history probe.

    A crude strength measure that exercises the rolling path end to end and
    feeds the P1 baselines. It is not part of the preregistered candidate set,
    which is fixed at P2.
    """
    margins = _team_margins(view.history)
    if margins.empty:
        return pd.Series(0.0, index=view.context.index)
    recent = margins.groupby("team").tail(window).groupby("team").margin
    form = _shrink(recent.mean(), recent.count().astype(float), prior_games)
    home = view.context.home_team.map(form).astype(float).fillna(0.0)
    away = view.context.away_team.map(form).astype(float).fillna(0.0)
    return home - away


def elo_ratings(history: pd.DataFrame, target_season=None, k: float = ELO_K,
                regression: float = ELO_SEASON_REGRESSION) -> dict[str, float]:
    """Walk Elo forward through settled games, in order.

    Causal by construction: a rating only ever moves on a game already played.
    Ratings regress a third of the way to 1500 between seasons, including into
    `target_season`, so a week 1 game is not priced on December's ratings.
    """
    ratings: dict[str, float] = {}
    season_seen = None
    def regress():
        for team in ratings:
            ratings[team] = ELO_START + (ratings[team] - ELO_START) * (1.0 - regression)
    for row in history.sort_values(["season", "week", "game_id"]).itertuples(index=False):
        if season_seen is not None and row.season != season_seen:
            regress()
        season_seen = row.season
        home = ratings.get(row.home_team, ELO_START)
        away = ratings.get(row.away_team, ELO_START)
        expected = 1.0 / (1.0 + 10.0 ** (-(home - away + ELO_HOME_ADVANTAGE) / ELO_SCALE))
        actual = 1.0 if row.margin > 0 else (0.0 if row.margin < 0 else 0.5)
        move = k * (actual - expected)
        ratings[row.home_team] = home + move
        ratings[row.away_team] = away - move
    if target_season is not None and season_seen is not None and target_season != season_seen:
        regress()
    return ratings


@feature("elo_diff")
def _elo_diff(view: AsOfView) -> pd.Series:
    ratings = elo_ratings(view.history, target_season=view.season)
    home = view.context.home_team.map(ratings).astype(float).fillna(ELO_START)
    away = view.context.away_team.map(ratings).astype(float).fillna(ELO_START)
    return home - away


def feature_frame(corpus: Corpus, season: int, week: int, features=None) -> pd.DataFrame:
    """Build one week's features. The view is the only thing a feature sees."""
    view = corpus.view(season, week)
    columns = {}
    for name, fn in (features or FEATURES).items():
        values = pd.Series(fn(view), index=view.context.index, dtype=float)
        if not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"Feature {name!r} produced a non-finite value at {season} week {week}")
        columns[name] = values
    frame = pd.DataFrame(columns, index=view.context.index)
    frame.insert(0, "week", view.week)
    frame.insert(0, "season", view.season)
    frame.insert(0, "game_id", view.context.game_id.to_numpy())
    return frame


def build_design(corpus: Corpus, weeks=None, features=None) -> pd.DataFrame:
    """Features and targets for every settled week, ready for walk-forward use."""
    weeks = weeks if weeks is not None else corpus.weeks(settled_only=True)
    frames = [feature_frame(corpus, s, w, features) for s, w in weeks]
    design = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if design.empty:
        return design
    return design.merge(corpus.outcomes[["game_id", "margin", "home_win", "tie"]],
                        on="game_id", how="inner")


def _corrupt(corpus: Corpus, season: int, week: int, rng: np.random.Generator) -> Corpus:
    """Replace every outcome and closing line from (season, week) onward."""
    games = corpus.games
    order = (games.season * 100 + games.week).to_numpy()
    at_or_after = set(games.game_id.to_numpy()[order >= season * 100 + week])

    outcomes = corpus.outcomes.copy()
    hit = outcomes.game_id.isin(at_or_after)
    if hit.any():
        home = rng.integers(0, 60, int(hit.sum())).astype(float)
        away = rng.integers(0, 60, int(hit.sum())).astype(float)
        outcomes.loc[hit, "home_score"] = home
        outcomes.loc[hit, "away_score"] = away
        margin = home - away
        outcomes.loc[hit, "margin"] = margin
        outcomes.loc[hit, "home_win"] = np.where(margin > 0, 1.0, np.where(margin < 0, 0.0, 0.5))
        outcomes.loc[hit, "tie"] = margin == 0

    market = corpus.market.copy()
    if not market.empty:
        hit_market = market.game_id.isin(at_or_after)
        for column in [c for c in MARKET_COLUMNS if c in market.columns]:
            values = pd.to_numeric(market[column], errors="coerce")
            noise = pd.Series(rng.normal(0, 50, len(market)), index=market.index)
            market[column] = values.where(~hit_market, values.fillna(0.0) + noise)
    return Corpus(games=corpus.games.copy(), outcomes=outcomes, market=market)


def audit(corpus: Corpus, checkpoints=None, features=None, seed: int = 0) -> dict:
    """Check the as-of boundary at each checkpoint, two ways.

    *Structural*: the view handed to a feature must contain no game at or after
    the checkpoint, and its pregame frames must carry no withheld column.

    *Perturbation*: replace every outcome and closing line from the checkpoint
    onward and re-derive. A feature whose value moves has been served corrupted
    data, which means the boundary that was supposed to exclude it did not.

    What this cannot catch: a feature that closes over a corpus or a file
    instead of reading its view. Nothing downstream can see that, because the
    value never passes through the boundary being tested. That one stays a
    review rule, and it is why ``feature_frame`` passes a view and not a corpus.
    """
    checkpoints = list(checkpoints if checkpoints is not None else corpus.weeks(settled_only=True))
    withheld = set(MARKET_COLUMNS) | set(OUTCOME_SOURCE_COLUMNS) | set(WITHHELD_COLUMNS)
    rng = np.random.default_rng(seed)
    findings = []
    for season, week in checkpoints:
        cutoff = season * 100 + week
        view = corpus.view(season, week)

        if not view.history.empty:
            late = int((view.history.season * 100 + view.history.week).ge(cutoff).sum())
            if late:
                findings.append({"season": int(season), "week": int(week), "check": "history_boundary",
                                 "detail": f"{late} settled games at or after the checkpoint"})
        if not view.schedule.empty:
            late = int((view.schedule.season * 100 + view.schedule.week).gt(cutoff).sum())
            if late:
                findings.append({"season": int(season), "week": int(week), "check": "schedule_boundary",
                                 "detail": f"{late} scheduled games after the checkpoint"})
        for name, frame in (("context", view.context), ("schedule", view.schedule)):
            leaked = sorted(withheld & set(frame.columns))
            if leaked:
                findings.append({"season": int(season), "week": int(week), "check": "withheld_columns",
                                 "detail": f"{name} carries {leaked}"})

        baseline = feature_frame(corpus, season, week, features)
        replayed = feature_frame(_corrupt(corpus, season, week, rng), season, week, features)
        for name in baseline.columns:
            if name in ("game_id", "season", "week"):
                continue
            if not baseline[name].equals(replayed[name]):
                moved = int((baseline[name] != replayed[name]).sum())
                findings.append({"season": int(season), "week": int(week), "check": "perturbation",
                                 "feature": name, "detail": f"{moved} rows changed"})
    return {"checkpoints": len(checkpoints), "features": sorted((features or FEATURES)),
            "findings": findings, "clean": not findings}
