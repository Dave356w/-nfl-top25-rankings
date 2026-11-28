"""Research page: the box-score composite's game margin against the market line.

This is model research, not betting advice. The page shows its record as
prominently as this week's numbers.

Model (replaced the Sleeper projection points model on 2026-11-28): the NFL
box-score composite v1.7, pipeline/boxscore_composite.py, run by
tools/build_boxscore_composite.py, which exports site/data/boxscore/latest.json.

1. The composite is a weekly-refit ridge logistic regression of home W/L on
   home-minus-away team profiles built only from earlier games: eight decayed
   box-score rates for offense and the same eight allowed by the defense, plus
   player availability (projected QB vs the recent QB mix, and each unit's
   snaps lost to injury reports or roster moves). Its recipe (feature family,
   half-life, ridge) was frozen for 2026 from 2021-2025 walk-forward results.
2. Home field removed (2026-11-28): the page drops the composite's home-field
   ('site') term from each game's log-odds, using the coefficient of the very
   fit that produced the game, so the composite and its frozen recipe are
   unchanged (tools/build_boxscore_composite.py checks that adding the term
   back reproduces the composite's probability). The probability without home
   field, p, becomes a margin on the spread's scale: margin = 12.37 *
   probit(p), the inverse of the market conversion P(home) =
   Phi(spread_line / 12.37) the composite uses as its benchmark.
3. Fixed map: bands of |margin| (<3, 3-7, 7-10, 10+) map to 3, 5, 7 and 13
   points, keeping the sign. These values were chosen on replays of the
   earlier Sleeper model, before this model was attached.
4. Model side: mapped margin vs the nflverse market line; zero edge goes to
   the favourite.
5. A game's row freezes at the last publish before its kickoff. Frozen rows
   are appended once to an append-only ledger and graded from nflverse
   final scores. Weeks or games the page did not publish live come from the
   composite's own walk-forward predictions and are tagged replay.
"""
from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pipeline.showdown_archive_backtest import _pandas, _team

ROOT = Path(__file__).resolve().parents[1]
COMPOSITE_PATH = ROOT / "site" / "data" / "boxscore" / "latest.json"

SCHEMA = 2
MODEL_NAME = "NFL box-score composite"
MARGIN_WP = "model_wp_no_home"   # the margin's probability: the composite without its home-field term
MARKET_SIGMA = 12.37
BANDS = (3.0, 7.0, 10.0)
BAND_LABELS = ("under 3", "3 to 7", "7 to 10", "10 or more")
FIXED_RULE = (3.0, 5.0, 7.0, 13.0)
BREAK_EVEN = 110 / 210
DISCLAIMER = (
    "Experimental model research. No demonstrated edge: the box-score composite "
    "predicts winners less accurately than the point spread, and its record "
    "against the spread is within coin-flip noise of the 52.4% needed to break "
    "even. Not betting advice."
)
_NORMAL = NormalDist()


# ---------------------------------------------------------------- margin
def implied_margin(home_wp: float, sigma: float = MARKET_SIGMA) -> float:
    """Home margin on the spread's scale: sigma * probit(p)."""
    p = min(max(float(home_wp), 1e-6), 1 - 1e-6)
    return sigma * _NORMAL.inv_cdf(p)


def market_wp(line: float, sigma: float = MARKET_SIGMA) -> float:
    return _NORMAL.cdf(float(line) / sigma)


def band_of(raw: float) -> int:
    return int(np.searchsorted(BANDS, abs(raw), side="right"))


def fit_map(*_args, **_kwargs):
    """The page's map: the fixed 3/5/7/13 rule."""
    return list(FIXED_RULE)


def mapped_margin(raw: float, rule=FIXED_RULE) -> float:
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


def _round(value, digits=4):
    value = _finite(value)
    return round(value, digits) if value is not None else None


def _finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------- inputs
def load_schedule(seasons) -> pd.DataFrame:
    """nflverse regular-season schedule with canonical team codes."""
    import nflreadpy as nfl
    schedule = _pandas(nfl.load_schedules(sorted(set(int(s) for s in seasons))))
    schedule = schedule[schedule["game_type"].eq("REG")].copy()
    for column in ("home_team", "away_team"):
        schedule[column] = schedule[column].map(_team)
    return schedule


def load_composite(path=COMPOSITE_PATH) -> dict:
    composite = json.loads(Path(path).read_text())
    if composite.get("schema") != 1:
        raise ValueError("Unsupported box-score composite export")
    return composite


def kickoff_utc(gameday, gametime):
    """nflverse gameday/gametime are US Eastern."""
    try:
        naive = datetime.fromisoformat(f"{gameday}T{gametime}")
    except (TypeError, ValueError):
        return None
    return pd.Timestamp(naive.replace(tzinfo=ZoneInfo("America/New_York"))).tz_convert("UTC")


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


# ---------------------------------------------------------------- replay
def pick(raw: float, line):
    """(mapped margin, side) for a raw margin against a line; side 0 without a line."""
    mapped = mapped_margin(raw)
    if line is None:
        return mapped, 0
    return mapped, model_side(mapped, line)


def replay_row(game: dict) -> dict:
    """One walk-forward composite prediction as a page row, graded when final."""
    line, wp = _finite(game.get("spread_line")), _finite(game.get(MARGIN_WP))
    row = {"week": int(game["week"]), "game_id": game["game_id"], "home": game["home"],
           "away": game["away"], "model_wp": _finite(game.get("model_wp")),
           "model_wp_no_home": wp, "line": line, "raw_margin": None,
           "mapped_margin": None, "model_side": None, "result": None,
           "home_score": game.get("home_score"), "away_score": game.get("away_score")}
    if wp is None or not game.get("ready", True):
        return row
    raw = implied_margin(wp)
    mapped, side = pick(raw, line)
    row.update(raw_margin=round(raw, 2), mapped_margin=mapped,
               model_side=(game["home"] if side > 0 else game["away"]) if side else None)
    margin = _finite(game.get("result"))
    if side and margin is not None:
        row["result"] = grade(side, margin, line)
    return row


def _season_of(rows):
    out = {}
    for game in rows:
        out.setdefault(int(game["season"]), []).append(game)
    return out


def backtest(composite: dict, long_run: dict | None = None) -> dict:
    """Held-out seasons (recipe chosen on earlier seasons) plus this season's
    earlier weeks: the composite through the map, the composite's raw margin,
    and the spread through the same map (control)."""
    games = list(composite.get("outer_predictions", []))
    current = [g for g in composite.get("season_predictions", [])
               if g.get("result") is not None]
    games += current
    out = {"projection_map": {}, "raw_margin": {}, "spread_map_control": {}}
    per = {k: {} for k in out}
    for season, rows in sorted(_season_of(games).items()):
        for g in rows:
            line, margin, wp = (_finite(g.get("spread_line")), _finite(g.get("result")),
                                _finite(g.get(MARGIN_WP)))
            if line is None or margin is None:
                continue
            per["spread_map_control"].setdefault(season, []).append(
                grade(pick(line, line)[1], margin, line))
            if wp is None or not g.get("ready", True):
                continue
            raw = implied_margin(wp)
            per["projection_map"].setdefault(season, []).append(grade(pick(raw, line)[1], margin, line))
            per["raw_margin"].setdefault(season, []).append(grade(model_side(raw, line), margin, line))
    for key, seasons in per.items():
        out[key] = {"seasons": {str(s): record(r) for s, r in sorted(seasons.items())},
                    "all": record(r for rs in seasons.values() for r in rs),
                    "first_season": min(seasons) if seasons else None}
    out["projection_map"]["label"] = "Composite margin (no home field) through the fixed map, closing lines"
    out["raw_margin"]["label"] = "Composite margin (no home field) vs the line, no map"
    out["spread_map_control"]["label"] = "Closing spread through the same map (no model)"
    if long_run:
        out["spread_map_control"]["long_run"] = long_run
    out["break_even"] = BREAK_EVEN
    out["shown_from"] = out["projection_map"]["first_season"]
    out["note"] = ("Held-out seasons use the recipe chosen on earlier seasons; this season "
                   "uses the frozen recipe with coefficients refit weekly. Graded at closing "
                   "lines, assumed -110. The map was tuned on the earlier Sleeper model's "
                   "replays, not this model's; these are diagnostics, not forward evidence.")
    return out


# ---------------------------------------------------------------- payload
def build_payload(composite, schedule, previous=None, ledger=(), now=None, snapshot_id=None):
    """Return (payload, rows to append to the ledger).

    Rows of `previous` whose kickoff has passed are kept unchanged: they froze
    at the last publish before kickoff. A game first seen after kickoff gets no
    live row. Every frozen row not yet in `ledger` is returned for appending.
    """
    now = pd.Timestamp(now or datetime.now(timezone.utc))
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    by_id = schedule.set_index(schedule["game_id"].astype(str))
    ledger = list(ledger)

    rows = {}
    previous_rows = {r["game_id"]: r for r in (previous or {}).get("games", [])
                     if r.get("model") == MODEL_NAME}
    for game_id, row in previous_rows.items():
        if pd.Timestamp(row["kickoff_utc"]) <= now:
            rows[game_id] = dict(row, locked=True)

    season, week = int(composite["season"]), int(composite["week"])
    for game in composite.get("board", []):
        game_id = str(game["game_id"])
        if game_id in rows or game_id not in by_id.index:
            continue
        match = by_id.loc[game_id]
        kickoff = kickoff_utc(match.get("gameday"), match.get("gametime"))
        if kickoff is None or kickoff <= now:
            continue  # started with no pregame row: no model side, never repaired
        line = _finite(match.get("spread_line"))
        wp = _finite(game.get(MARGIN_WP))
        entry = {
            "game_id": game_id, "nflverse_game_id": game_id, "model": MODEL_NAME,
            "season": int(game["season"]), "week": int(game["week"]),
            "kickoff_utc": kickoff.isoformat(), "home": match["home_team"], "away": match["away_team"],
            "model_wp": _round(game.get("model_wp")),
            "model_wp_no_home": round(wp, 4) if wp is not None else None,
            "homefield_wp": _finite(game.get("homefield_wp")),
            "market_wp": round(market_wp(line), 4) if line is not None else None,
            "ready": bool(game.get("ready", True)),
            "raw_margin": None, "band": None,
            "line": line, "total_line": _finite(match.get("total_line")),
            "line_source": "nflverse schedules (spread_line)",
            "line_captured_utc": now.isoformat(),
            "mapped_margin": None, "model_side": None, "edge": None,
            "composite_generated_utc": composite.get("generated_utc"),
            "snapshot_id": snapshot_id, "locked": False,
        }
        if wp is not None and entry["ready"]:
            raw = implied_margin(wp)
            entry.update(raw_margin=round(raw, 2), band=BAND_LABELS[band_of(raw)])
            if line is not None:
                mapped, side = pick(raw, line)
                entry.update(mapped_margin=mapped, edge=round(mapped - line, 2),
                             model_side=(entry["home"] if side > 0 else entry["away"]) if side else None,
                             model_side_sign=side)
        rows[game_id] = entry

    current = sorted((r for r in rows.values() if (r["season"], r["week"]) == (season, week)),
                     key=lambda r: (r["kickoff_utc"], r["home"]))

    logged = {r["game_id"] for r in ledger}
    new_ledger = [
        {k: v for k, v in r.items() if k != "locked"}
        for r in current if r["locked"] and r["game_id"] not in logged
        and r.get("model_side_sign")
    ]
    graded_all = _grade_ledger(ledger + new_ledger, schedule)
    graded = {r["game_id"]: r for r in graded_all}
    for r in current:
        if r["game_id"] in graded:
            r["result"] = graded[r["game_id"]].get("result")
            r["home_margin"] = graded[r["game_id"]].get("home_margin")

    weeks = season_results(season, composite.get("season_predictions", []), graded_all)
    headline = pooled_results(composite, graded_all)
    payload = {
        "schema": SCHEMA, "snapshot_id": snapshot_id,
        "generated_utc": now.isoformat(), "season": season, "week": week,
        "disclaimer": DISCLAIMER,
        "method": {
            "model": MODEL_NAME, "revision": composite.get("revision"),
            "recipe": composite.get("recipe"), "fit": composite.get("fit"),
            "coefficients": composite.get("coefficients"),
            "composite_generated_utc": composite.get("generated_utc"),
            "market_sigma": MARKET_SIGMA,
            "margin": "12.37 x probit(home win probability without the composite's home-field term)",
            "home_field": "removed from the margin on 2026-11-28; the composite's own probability keeps it",
            "outer_scorecard": composite.get("outer_scorecard"),
            "outer_scorecard_by_season": composite.get("outer_scorecard_by_season"),
            "market_blend_verdict": composite.get("market_blend_verdict"),
            "map": {"bands": list(BANDS), "band_labels": list(BAND_LABELS),
                    "type": "fixed", "rule": list(FIXED_RULE)},
            "line_source": "nflverse schedules spread_line (open data); no capture time from source",
            "lock": "a game's row freezes at the last publish before kickoff",
        },
        "games": current,
        "forward_record": record(r["result"] for r in graded_all if r.get("result")),
        "headline": headline,
        "season_results": weeks,
        "season_record": season_record(weeks),
        "backtest": backtest(composite, composite.get("control_long_run")),
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


def pooled_results(composite, forward_rows) -> dict:
    """Every graded game the model has a pick for, pooled: held-out seasons and
    this season's walk-forward replay, with a game the page published live
    counted from its frozen row instead of its replay."""
    games = {}
    for game in list(composite.get("outer_predictions", [])) + list(composite.get("season_predictions", [])):
        row = replay_row(game)
        if row.get("result"):
            games[str(game["game_id"])] = (int(game["season"]), row["result"], "replay")
    for row in forward_rows:
        if row.get("result"):
            games[str(row.get("nflverse_game_id"))] = (int(row["season"]), row["result"], "live")
    seasons = [s for s, _, _ in games.values()]
    return {**record(r for _, r, _ in games.values()),
            "live_games": sum(1 for *_, src in games.values() if src == "live"),
            "replay_games": sum(1 for *_, src in games.values() if src == "replay"),
            "first_season": min(seasons) if seasons else None,
            "last_season": max(seasons) if seasons else None}


def season_record(weeks) -> dict:
    """Every graded game of the season pooled, replayed and live games together."""
    return record(g["result"] for w in weeks for g in w["games"] if g.get("result"))


def season_results(season, predictions, forward_rows) -> list[dict]:
    """The season by week, newest first. A game the page published live uses its
    frozen ledger row; every other game uses the composite's walk-forward
    prediction (replay). Only games already final or locked are listed."""
    keep = ("week", "home", "away", "raw_margin", "line", "mapped_margin", "model_side",
            "home_score", "away_score", "result", "model_wp", "model_wp_no_home")
    games = {}
    for game in predictions:
        if int(game.get("season", season)) != season or game.get("result") is None:
            continue
        row = replay_row(game)
        games[str(game["game_id"])] = dict({k: row.get(k) for k in keep}, source="replay")
    for row in forward_rows:
        if row.get("season") == season:
            games[str(row.get("nflverse_game_id"))] = dict({k: row.get(k) for k in keep}, source="forward")
    weeks = []
    for week in sorted({g["week"] for g in games.values()}, reverse=True):
        rows = sorted((g for g in games.values() if g["week"] == week),
                      key=lambda g: (g["away"] or "", g["home"] or ""))
        sources = {g["source"] for g in rows}
        weeks.append({"week": week,
                      "source": "forward" if sources == {"forward"} else "replay" if sources == {"replay"} else "mixed",
                      "record": record(g["result"] for g in rows if g.get("result")),
                      "games": rows})
    return weeks


def write_outputs(payload, new_ledger, directory) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    append_ledger(directory / "ledger.jsonl", new_ledger)
    (directory / "latest.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    with (directory / "latest.csv").open("w", newline="") as handle:
        fields = ["kickoff_utc", "away", "home", "model_wp", "model_wp_no_home", "market_wp", "raw_margin",
                  "line", "band", "mapped_margin", "edge", "model_side", "locked", "result"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(payload["games"])
