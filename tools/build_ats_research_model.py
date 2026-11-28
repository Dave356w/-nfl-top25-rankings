#!/usr/bin/env python3
"""Fit the ATS research page's points model, margin history and backtest.

Sources: Sleeper weekly projections with each week's team
(api.sleeper.com/projections, cached under backtest_cache/sleeper_weekly),
this season's ESPN and CBS projections (via tools/evaluate_sources.py and its
source_eval_cache) and nflverse schedules (scores and closing spread_line).

The live page applies the points model to the site's three-source mean, so the
model is fitted on that same blend: this season's completed weeks, each player
the mean of his Sleeper, ESPN and CBS projections (CBS rescaled to Sleeper per
position-week, as the site does). Until MIN_BLEND_GAMES games exist, the fit
falls back to Sleeper history. The median map and the season replay still use
Sleeper-only walk-forward margins from 2021, since only Sleeper has that much
history. Writes:

  model/ats_research/model.json         points-model coefficients and backtest
  model/ats_research/margin_history.csv walk-forward projection margins

Each season's margins come from a points model fitted on earlier seasons only,
so the map trains on out-of-sample margins. Re-run after a week completes to
refresh the coefficients; the live page also extends the map with its own
graded ledger rows.

    python tools/build_ats_research_model.py
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import ats_research as ats  # noqa: E402
from pipeline.showdown_archive_backtest import _team  # noqa: E402

CACHE = ROOT / "backtest_cache" / "sleeper_weekly"
URL = ("https://api.sleeper.com/projections/nfl/{season}/{week}?season_type=regular"
       "&position[]=QB&position[]=RB&position[]=WR&position[]=TE&position[]=DEF")
FIRST_SEASON = 2021
MIN_BLEND_GAMES = 100


def weekly_projections(season, week, refresh=False, cache=True):
    """Slim Sleeper projection rows for one week; cached only once the week is final."""
    path = CACHE / f"{season}_{week:02d}.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text())
        if not cached or "pid" in cached[0]:
            return cached
    request = urllib.request.Request(URL.format(season=season, week=week),
                                     headers={"User-Agent": "nfl-top25-rankings/1.0"})
    with urllib.request.urlopen(request, timeout=40) as response:
        rows = json.loads(response.read())
    slim = [{"pid": r.get("player_id"), "team": r.get("team"),
             "pos": (r.get("player") or {}).get("position"),
             "fp": (r.get("stats") or {}).get("pts_half_ppr")} for r in rows]
    if cache:
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(slim))
    time.sleep(0.4)
    return slim


def team_games(schedule, seasons):
    """One row per final game with both teams' strength."""
    out = []
    for season in seasons:
        for week in sorted(schedule.loc[schedule.season.eq(season), "week"].unique()):
            week_games = schedule[schedule.season.eq(season) & schedule.week.eq(week)]
            games = week_games[week_games.home_score.notna()]
            if games.empty:
                continue
            complete = len(games) == len(week_games)
            rows = pd.DataFrame(weekly_projections(season, int(week), cache=complete))
            if rows.empty:
                continue
            strength = ats.team_strength(rows.dropna(subset=["team"]))
            for g in games.itertuples(index=False):
                h, a = strength.get(g.home_team), strength.get(g.away_team)
                if not h or not a or not h["def"] or not a["def"]:
                    continue
                out.append(dict(season=season, week=int(week), game_id=g.game_id,
                                home=g.home_team, away=g.away_team,
                                neutral=str(g.location).lower() == "neutral",
                                home_off=h["off"], away_off=a["off"],
                                home_def=h["def"], away_def=a["def"],
                                home_score=float(g.home_score), away_score=float(g.away_score),
                                line=float(g.spread_line) if pd.notna(g.spread_line) else np.nan))
    return pd.DataFrame(out)


def blend_team_games(schedule, season):
    """This season's final games with team strength from the three-source mean."""
    from tools import evaluate_sources as ev
    from pipeline import sleeper
    if season != ev.SEASON:
        return pd.DataFrame()
    raw, _ = sleeper.fetch_players(ROOT / "sleeper_cache")
    players = sleeper.players_frame(raw)[["player_id", "Sleeper_Name", "Key", "Team", "Position"]]
    pregame = ev.archived_pregame()
    out = []
    for week in ev.completed_weeks():
        frame, _ = ev.week_frame(week, players, pregame.get(week, {}))
        # That week's team, not today's: traded players count for the team they played for.
        weekly = {r["pid"]: r["team"] for r in weekly_projections(season, week) if r.get("pid")}
        frame["Team"] = frame["player_id"].map(weekly).fillna(frame["Team"])
        ok = frame["Sleeper"].ge(3) & frame["CBS_raw"].ge(3)
        ratio = (frame["Sleeper"] / frame["CBS_raw"]).where(ok)
        stats = frame.assign(r=ratio).groupby("Position")["r"].agg(["median", "count"])
        scale = stats["median"].where(stats["count"] >= 8, 1.0)
        frame["CBS"] = frame["CBS_raw"] * frame["Position"].map(scale).fillna(1.0)
        sources = frame[["Sleeper", "ESPN", "CBS"]]
        frame["blend"] = sources.where(sources.gt(0)).mean(axis=1)
        strength = ats.team_strength(frame[frame["blend"].notna()], "Team", "Position", "blend")
        games = schedule[schedule.season.eq(season) & schedule.week.eq(week) & schedule.home_score.notna()]
        for g in games.itertuples(index=False):
            h, a = strength.get(g.home_team), strength.get(g.away_team)
            if not h or not a or not h["def"] or not a["def"]:
                continue
            out.append(dict(season=season, week=week, neutral=str(g.location).lower() == "neutral",
                            home_off=h["off"], away_off=a["off"], home_def=h["def"], away_def=a["def"],
                            home_score=float(g.home_score), away_score=float(g.away_score)))
    return pd.DataFrame(out)


def stack(games):
    home = pd.DataFrame(dict(off=games.home_off, opp_def=games.away_def, pts=games.home_score))
    away = pd.DataFrame(dict(off=games.away_off, opp_def=games.home_def, pts=games.away_score))
    return pd.concat([home, away], ignore_index=True)


def raw_margins(games, coef):
    home = ats.predict_points(coef, games.home_off, games.away_def)
    away = ats.predict_points(coef, games.away_off, games.home_def)
    return home - away


def backtest(frame, first_season):
    """Weekly map, model side vs closing line, graded; per season and overall."""
    results = []
    for (season, week), group in frame.groupby(["season", "week"], sort=True):
        if season < first_season:
            continue
        rule = ats.fit_map(frame, season, week)
        if rule is None:
            continue
        for g in group.itertuples(index=False):
            if not np.isfinite(g.line) or g.raw == 0:
                continue
            side = ats.model_side(ats.mapped_margin(g.raw, rule), g.line)
            if side:
                results.append((season, ats.grade(side, g.actual, g.line)))
    by_season = {}
    for season, result in results:
        by_season.setdefault(int(season), []).append(result)
    return {"seasons": {str(s): ats.record(r) for s, r in sorted(by_season.items())},
            "all": ats.record(r for _, r in results),
            "first_season": int(min(by_season)) if by_season else None}


def season_replay(history, season):
    """Every replayed game of `season`: the weekly map's side, graded, with scores."""
    games = []
    for week, group in history[history.season.eq(season)].groupby("week", sort=True):
        rule = ats.fit_map(history, season, int(week))
        for g in group.sort_values("game_id").itertuples(index=False):
            row = {"week": int(week), "game_id": g.game_id, "home": g.home, "away": g.away,
                   "raw_margin": round(float(g.raw), 2), "line": float(g.line),
                   "home_score": int(g.home_score), "away_score": int(g.away_score),
                   "mapped_margin": None, "model_side": None, "result": None}
            if rule is not None and g.raw != 0:
                mapped = ats.mapped_margin(g.raw, rule)
                side = ats.model_side(mapped, g.line)
                if side:
                    row.update(mapped_margin=mapped, model_side=g.home if side > 0 else g.away,
                               result=ats.grade(side, g.actual, g.line))
            games.append(row)
    return games


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    now = datetime.now(timezone.utc)
    # NFL seasons are named for the year they start; January-February belong to the previous one.
    parser.add_argument("--season", type=int, default=now.year if now.month >= 3 else now.year - 1)
    parser.add_argument("--refresh", action="store_true", help="refetch cached weeks")
    args = parser.parse_args(argv)
    seasons = list(range(FIRST_SEASON, args.season + 1))
    schedule = ats.load_schedule(list(range(1999, args.season + 1)))

    games = team_games(schedule, seasons)
    games = games[games.line.notna()].reset_index(drop=True)
    games["actual"] = games.home_score - games.away_score

    # Walk-forward margins: each season scored by a model fitted on earlier seasons.
    parts = []
    for season in seasons[1:]:
        train, test = games[games.season < season], games[games.season == season].copy()
        if train.empty or test.empty:
            continue
        test["raw"] = raw_margins(test, ats.fit_points_model(stack(train)))
        parts.append(test)
    history = pd.concat(parts, ignore_index=True)
    history["raw"] = history["raw"].round(3)

    blend = blend_team_games(schedule, args.season)
    if len(blend) >= MIN_BLEND_GAMES:
        coef = ats.fit_points_model(stack(blend))
        fit = {"team_games": int(len(blend) * 2), "seasons": [args.season, args.season],
               "through_week": int(blend.week.max()),
               "source": "Sleeper + ESPN + CBS mean (the site's blend) + nflverse final scores"}
    else:
        coef = ats.fit_points_model(stack(games))
        last = games.sort_values(["season", "week"]).iloc[-1]
        fit = {"team_games": int(len(games) * 2), "seasons": [seasons[0], int(last.season)],
               "through_week": int(last.week),
               "source": (f"Sleeper weekly projections + nflverse final scores (fewer than "
                          f"{MIN_BLEND_GAMES} blend games this season)")}

    # Shown on the page: last season and this one only. The control's long-run
    # record stays as one summary line so the "no demonstrated edge" note holds.
    shown_from = args.season - 1
    projection_bt = backtest(history[["season", "week", "raw", "line", "actual"]],
                             first_season=shown_from)
    control = schedule[schedule.home_score.notna() & schedule.spread_line.notna()].copy()
    control = pd.DataFrame(dict(season=control.season, week=control.week,
                                raw=control.spread_line, line=control.spread_line,
                                actual=control.home_score - control.away_score))
    control_long = backtest(control, first_season=2001)
    control_bt = {**control_long,
                  "seasons": {k: v for k, v in control_long["seasons"].items() if int(k) >= shown_from}}
    control_bt["all"] = ats.record(
        r for k, v in control_bt["seasons"].items()
        for r in ["W"] * v["wins"] + ["L"] * v["losses"] + ["P"] * v["pushes"])
    control_bt["first_season"] = shown_from
    control_bt["long_run"] = {**control_long["all"], "first_season": control_long["first_season"],
                              "last_season": args.season}

    ats.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history[["season", "week", "game_id", "home", "away", "raw", "line", "actual"]].to_csv(
        ats.HISTORY_PATH, index=False)
    replay = season_replay(history, args.season)
    ats.REPLAY_PATH.write_text(json.dumps({"season": args.season, "games": replay},
                                          indent=1, allow_nan=False) + "\n")
    model = {
        "schema": ats.SCHEMA,
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "coefficients": coef,
        "fit": fit,
        "map_history": "Sleeper-only walk-forward margins from 2021 (the blend has no history)",
        "backtest": {
            "projection_map": {**projection_bt,
                               "label": "Projection margin through the weekly median map, closing lines"},
            "spread_map_control": {**control_bt,
                                   "label": "Closing spread through the same map (no projections)"},
            "break_even": ats.BREAK_EVEN,
            "shown_from": shown_from,
            "note": ("Walk-forward replays graded at an assumed -110. Design choices were made "
                     "after looking at recent results, so these are diagnostics, not forward "
                     "evidence."),
        },
    }
    ats.MODEL_PATH.write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"coefficients": coef, "fit": fit, "history_rows": len(history),
                      "projection_map": projection_bt["all"],
                      "spread_map_control": control_bt["all"],
                      "control_long_run": control_bt["long_run"]}, indent=2))


if __name__ == "__main__":
    main()
