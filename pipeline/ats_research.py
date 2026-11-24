"""Research page: projection-implied game margins against the market line.

This is model research, not betting advice. Over 2023-2026 walk-forward the
projection margin through the median map covered about 51% of decided games
against closing lines, below the 52.4% break-even at -110; the map on the
spread alone covered 50.2% over 2001-2026. The page shows that record as
prominently as this week's numbers.

Steps, all from the shared projection snapshot:

1. Team strength: starters' projected half-PPR points (QB1, RB1-2, WR1-3,
   TE1) and the team's DEF projection.
2. Points model (fitted offline by tools/build_ats_research_model.py on
   2021+ Sleeper weekly projections and nflverse scores):
   points = const + b_off * offense + b_def * opponent DEF + b_home * home.
   The raw margin is home points minus away points (+ = home favoured).
3. Weekly median map: bands of |raw| (<3, 3-7, 7-10, 10+); each band's value
   is the weighted median final margin of the side the raw margin favoured,
   over earlier games only, with an 8-week half-life (19 weeks per season
   boundary). Band values never decrease with the band: a band whose median
   falls below the one beneath is pooled with it. Mapped margin =
   sign(raw) * band value.
4. Model side: mapped margin vs the nflverse market line; zero edge goes to
   the favourite.
5. A game's row freezes at the last publish before its kickoff. Frozen rows
   are appended once to an append-only ledger and graded from nflverse
   final scores.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.showdown_archive_backtest import _pandas, _team

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "model" / "ats_research"
MODEL_PATH = MODEL_DIR / "model.json"
HISTORY_PATH = MODEL_DIR / "margin_history.csv"
REPLAY_PATH = MODEL_DIR / "season_replay.json"

SCHEMA = 1
TAKE = {"QB": 1, "RB": 2, "WR": 3, "TE": 1}
BANDS = (3.0, 7.0, 10.0)
BAND_LABELS = ("under 3", "3 to 7", "7 to 10", "10 or more")
MAP_HALF_LIFE_WEEKS = 8
WEEKS_PER_SEASON = 19
MIN_SEASONS = 2
MIN_GAMES = 300
MIN_PER_BAND = 30
BREAK_EVEN = 110 / 210
DISCLAIMER = (
    "Experimental model research. No demonstrated edge: in backtests this "
    "method covered about as often as a coin flip, below the 52.4% needed to "
    "break even. Not betting advice."
)


# ---------------------------------------------------------------- strength
def team_strength(players: pd.DataFrame, team_col="team", pos_col="pos", fp_col="fp"):
    """{team: {"off": starters' FP, "def": DEF FP, ...}} from one projection set."""
    frame = players[[team_col, pos_col, fp_col]].copy()
    frame.columns = ["team", "pos", "fp"]
    frame["team"] = frame["team"].map(_team)
    frame["fp"] = pd.to_numeric(frame["fp"], errors="coerce")
    frame = frame[frame["fp"].notna() & frame["fp"].gt(0)]
    out = {}
    for team, group in frame.groupby("team"):
        row = {}
        for pos, count in TAKE.items():
            row[pos] = float(group.loc[group.pos.eq(pos), "fp"].nlargest(count).sum())
        row["off"] = float(sum(row[pos] for pos in TAKE))
        row["def"] = float(group.loc[group.pos.eq("DEF"), "fp"].sum())
        out[team] = row
    return out


def fit_points_model(team_games: pd.DataFrame, ridge: float = 1.0) -> dict:
    """Ridge fit of team points on own offense, opponent DEF and home field."""
    x = np.column_stack([np.ones(len(team_games)), team_games[["off", "opp_def", "home"]]])
    penalty = np.eye(4) * ridge
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(x.T @ x + penalty, x.T @ team_games["pts"].to_numpy(float))
    return dict(zip(("const", "off", "opp_def", "home"), map(float, beta)))


def predict_points(coef: dict, off: float, opp_def: float, home: float) -> float:
    return coef["const"] + coef["off"] * off + coef["opp_def"] * opp_def + coef["home"] * home


# ---------------------------------------------------------------- median map
def band_of(raw: float) -> int:
    return int(np.searchsorted(BANDS, abs(raw), side="right"))


def weighted_median(values, weights) -> float:
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = np.cumsum(weights)
    return float(values[np.searchsorted(cumulative, cumulative[-1] / 2)])


def fit_map(history: pd.DataFrame, season: int, week: int):
    """Band values from games strictly before (season, week); None if too thin.

    `history` needs season, week, raw and actual (home margin)."""
    prior = history[(history.season < season)
                    | ((history.season == season) & (history.week < week))]
    prior = prior[prior.raw.ne(0) & prior.actual.notna()]
    if prior.season.nunique() < MIN_SEASONS or len(prior) < MIN_GAMES:
        return None
    age = ((season - prior.season) * WEEKS_PER_SEASON + (week - prior.week)) / MAP_HALF_LIFE_WEEKS
    weight = np.power(2.0, -age.to_numpy(float))
    bands = prior.raw.map(band_of).to_numpy()
    signed = (np.sign(prior.raw) * prior.actual).to_numpy(float)
    if any((bands == band).sum() < MIN_PER_BAND for band in range(len(BANDS) + 1)):
        return None
    return monotone_band_medians(signed, weight, bands, len(BANDS) + 1)


def monotone_band_medians(signed, weight, bands, count):
    """Weighted median per band, never decreasing with the band.

    A bigger projected margin must not map to a smaller one. Thin, noisy top
    bands broke that (2026 week 8: 3/5/11/6 sent a 13-point favourite to 6), so
    any band whose median falls below the band beneath it is pooled with it and
    the pooled games' median is used for both (pool-adjacent-violators)."""
    blocks = [[band] for band in range(count)]

    def median(members):
        mask = np.isin(bands, members)
        return weighted_median(signed[mask], weight[mask])

    values = [median(block) for block in blocks]
    i = 0
    while i < len(blocks) - 1:
        if values[i + 1] < values[i]:
            blocks[i] = blocks[i] + blocks.pop(i + 1)
            values.pop(i + 1)
            values[i] = median(blocks[i])
            i = max(i - 1, 0)
        else:
            i += 1
    rule = [None] * count
    for block, value in zip(blocks, values):
        for band in block:
            rule[band] = value
    return rule


def mapped_margin(raw: float, rule) -> float:
    return float(np.sign(raw) * rule[band_of(raw)])


def model_side(mapped: float, line: float) -> int:
    """+1 home, -1 away, 0 none. Zero edge takes the market favourite."""
    edge = mapped - line
    if edge != 0:
        return int(np.sign(edge))
    return int(np.sign(line))


def grade(side: int, home_margin: float, line: float) -> str:
    cover = side * (home_margin - line)
    return "W" if cover > 0 else "L" if cover < 0 else "P"


def record(results) -> dict:
    results = list(results)
    wins, losses, pushes = (results.count(k) for k in ("W", "L", "P"))
    decided = wins + losses
    rate = wins / decided if decided else None
    return {
        "wins": wins, "losses": losses, "pushes": pushes,
        "cover_rate": rate,
        "cover_rate_ci95": (1.96 * math.sqrt(rate * (1 - rate) / decided)) if decided else None,
        "roi_at_minus_110": ((wins * 100 / 110 - losses) / (decided + pushes)) if decided + pushes else None,
    }


# ---------------------------------------------------------------- lines
def load_schedule(seasons) -> pd.DataFrame:
    """nflverse regular-season schedule with canonical team codes."""
    import nflreadpy as nfl
    schedule = _pandas(nfl.load_schedules(sorted(set(int(s) for s in seasons))))
    schedule = schedule[schedule["game_type"].eq("REG")].copy()
    for column in ("home_team", "away_team"):
        schedule[column] = schedule[column].map(_team)
    schedule["game_date"] = pd.to_datetime(schedule["gameday"]).dt.date
    return schedule


def _match(schedule: pd.DataFrame, home: str, away: str, kickoff: pd.Timestamp):
    rows = schedule[schedule.home_team.eq(home) & schedule.away_team.eq(away)]
    day = kickoff.tz_convert("America/New_York").date()
    rows = rows[rows.game_date.map(lambda d: abs((d - day).days) <= 1)]
    return rows.iloc[0] if len(rows) == 1 else None


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------- artifacts
def load_model(path=MODEL_PATH) -> dict:
    model = json.loads(Path(path).read_text())
    if model.get("schema") != SCHEMA:
        raise ValueError("Unsupported ATS research model artifact")
    return model


def load_history(path=HISTORY_PATH) -> pd.DataFrame:
    history = pd.read_csv(path, dtype={"game_id": str})
    return history[["season", "week", "game_id", "raw", "line", "actual"]]


def read_ledger(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append_ledger(path, rows) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


# ---------------------------------------------------------------- payload
def build_payload(players, schedule, model, history, previous=None, ledger=(),
                  now=None, snapshot_id=None, replay=None):
    """Return (payload, rows to append to the ledger).

    `players` is the shared prepared slate (Name, Position, Team, Projected_FP,
    Game ID, Game Time, Home Team, Away Team). Rows of `previous` whose kickoff
    has passed are kept unchanged: they froze at the last publish before
    kickoff. Every frozen row not yet in `ledger` is returned for appending.
    """
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    coef = model["coefficients"]
    strength = team_strength(players, "Team", "Position", "Projected_FP")
    games = (players[["Game ID", "Game Time", "Home Team", "Away Team"]]
             .drop_duplicates("Game ID"))

    # Graded forward rows extend the map's training history week by week.
    ledger = list(ledger)
    graded = _grade_ledger(ledger, schedule)
    forward_history = pd.DataFrame([
        {"season": r["season"], "week": r["week"], "game_id": str(r.get("nflverse_game_id")),
         "raw": r["raw_margin"], "line": r["line"], "actual": r["home_margin"]}
        for r in graded if r.get("home_margin") is not None
    ], columns=["season", "week", "game_id", "raw", "line", "actual"])
    # The weekly refit's history and the ledger can both hold a game; the
    # ledger's margin is the one this page published, so it wins.
    if "game_id" in history:
        history = history[~history["game_id"].astype(str).isin(set(forward_history["game_id"]))]
    train = pd.concat([history, forward_history], ignore_index=True)

    rows = {}
    previous_rows = {r["game_id"]: r for r in (previous or {}).get("games", [])}
    for game_id, row in previous_rows.items():
        if pd.Timestamp(row["kickoff_utc"]) <= now:
            rows[game_id] = dict(row, locked=True)

    season = week = None
    rule = None
    for game in games.itertuples(index=False):
        game_id = str(game[0])
        kickoff = pd.Timestamp(pd.to_datetime(game[1], utc=True))
        home, away = _team(game[2]), _team(game[3])
        if game_id in rows:
            continue
        if kickoff <= now:
            continue  # started with no pregame row: no model side, never repaired
        match = _match(schedule, home, away, kickoff)
        if match is None or home not in strength or away not in strength:
            continue
        line = _finite(match.get("spread_line"))
        if season is None:
            season, week = int(match["season"]), int(match["week"])
            rule = fit_map(train, season, week)
        h, a = strength[home], strength[away]
        neutral = str(match.get("location") or "").lower() == "neutral"
        home_pts = predict_points(coef, h["off"], a["def"], 0.0 if neutral else 1.0)
        away_pts = predict_points(coef, a["off"], h["def"], 0.0)
        raw = home_pts - away_pts
        entry = {
            "game_id": game_id, "nflverse_game_id": str(match["game_id"]),
            "season": int(match["season"]), "week": int(match["week"]),
            "kickoff_utc": kickoff.isoformat(), "home": home, "away": away,
            "home_offense_fp": round(h["off"], 2), "away_offense_fp": round(a["off"], 2),
            "home_def_fp": round(h["def"], 2), "away_def_fp": round(a["def"], 2),
            "home_points": round(home_pts, 1), "away_points": round(away_pts, 1),
            "raw_margin": round(raw, 2), "projected_total": round(home_pts + away_pts, 1),
            "line": line, "total_line": _finite(match.get("total_line")),
            "line_source": "nflverse schedules (spread_line)",
            "line_captured_utc": now.isoformat(),
            "band": BAND_LABELS[band_of(raw)],
            "mapped_margin": None, "model_side": None, "edge": None,
            "snapshot_id": snapshot_id, "locked": False,
        }
        if rule is not None and line is not None:
            mapped = mapped_margin(raw, rule)
            side = model_side(mapped, line)
            entry.update(mapped_margin=mapped, edge=round(mapped - line, 2),
                         model_side=(home if side > 0 else away) if side else None,
                         model_side_sign=side)
        rows[game_id] = entry

    if season is None and rows:
        any_row = next(iter(rows.values()))
        season, week = any_row["season"], any_row["week"]
    current = sorted((r for r in rows.values() if (r["season"], r["week"]) == (season, week)),
                     key=lambda r: (r["kickoff_utc"], r["home"]))

    logged = {r["game_id"] for r in ledger}
    new_ledger = [
        {k: v for k, v in r.items() if k != "locked"}
        for r in current if r["locked"] and r["game_id"] not in logged
        and r.get("model_side_sign")
    ]
    graded_all = _grade_ledger(ledger + new_ledger, schedule)
    by_id = {r["game_id"]: r for r in graded_all}
    for r in current:
        if r["game_id"] in by_id:
            r["result"] = by_id[r["game_id"]].get("result")
            r["home_margin"] = by_id[r["game_id"]].get("home_margin")

    payload = {
        "schema": SCHEMA, "snapshot_id": snapshot_id,
        "generated_utc": now.isoformat(), "season": season, "week": week,
        "disclaimer": DISCLAIMER,
        "method": {
            "team_strength": "starters' projected half-PPR points: QB1, RB1-2, WR1-3, TE1; DEF projection",
            "points_model": coef, "points_model_fit": model.get("fit"),
            "map": {"bands": list(BANDS), "band_labels": list(BAND_LABELS),
                    "half_life_weeks": MAP_HALF_LIFE_WEEKS,
                    "minimum": {"seasons": MIN_SEASONS, "games": MIN_GAMES, "per_band": MIN_PER_BAND},
                    "rule": rule},
            "line_source": "nflverse schedules spread_line (open data); no capture time from source",
            "lock": "a game's row freezes at the last publish before kickoff",
        },
        "games": current,
        "forward_record": record(r["result"] for r in graded_all if r.get("result")),
        "season_results": (weeks := season_results(season, replay or {}, graded_all)),
        "season_record": season_record(weeks),
        "backtest": model.get("backtest"),
    }
    json.dumps(payload, allow_nan=False)
    return payload, new_ledger


def _grade_ledger(ledger, schedule):
    finals = {}
    for row in schedule.itertuples(index=False):
        home_score, away_score = _finite(row.home_score), _finite(row.away_score)
        if home_score is not None and away_score is not None:
            finals[str(row.game_id)] = (home_score, away_score)
    out = []
    for row in ledger:
        row = dict(row)
        final = finals.get(str(row.get("nflverse_game_id")))
        if final is not None and row.get("model_side_sign") and row.get("line") is not None:
            row["home_score"], row["away_score"] = int(final[0]), int(final[1])
            row["home_margin"] = final[0] - final[1]
            row["result"] = grade(row["model_side_sign"], row["home_margin"], row["line"])
        out.append(row)
    return out


def load_replay(path=REPLAY_PATH) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {"season": None, "games": []}


def season_record(weeks) -> dict:
    """Every graded game of the season pooled, replayed and live weeks together."""
    return record(g["result"] for w in weeks for g in w["games"] if g.get("result"))


def season_results(season, replay, forward_rows) -> list[dict]:
    """The season by week, newest first. A week the page published live uses its
    frozen ledger rows; earlier weeks use the walk-forward replay."""
    keep = ("week", "home", "away", "raw_margin", "line", "mapped_margin", "model_side",
            "home_score", "away_score", "result")
    forward = {}
    for row in forward_rows:
        if row.get("season") == season:
            forward.setdefault(int(row["week"]), []).append({k: row.get(k) for k in keep})
    replayed = {}
    if replay.get("season") == season:
        for row in replay.get("games", []):
            replayed.setdefault(int(row["week"]), []).append({k: row.get(k) for k in keep})
    weeks = []
    for week in sorted(set(forward) | set(replayed), reverse=True):
        live = week in forward
        games = sorted(forward[week] if live else replayed[week],
                       key=lambda g: (g["away"] or "", g["home"] or ""))
        weeks.append({"week": week, "source": "forward" if live else "replay",
                      "record": record(g["result"] for g in games if g.get("result")),
                      "games": games})
    return weeks


def write_outputs(payload, new_ledger, directory) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    append_ledger(directory / "ledger.jsonl", new_ledger)
    (directory / "latest.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with (directory / "latest.csv").open("w", newline="") as handle:
        fields = ["kickoff_utc", "away", "home", "away_points", "home_points", "raw_margin",
                  "line", "band", "mapped_margin", "edge", "model_side", "locked", "result"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["games"])
