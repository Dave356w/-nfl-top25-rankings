#!/usr/bin/env python3
"""Standalone, keyless season-long NFL weekly lineup optimizer.

Yahoo's public DFS feed supplies current-week FPPG, salary, opponents and game
times, and the daily pipeline's market engine replaces those blends with
de-vigged Bovada and Underdog means wherever the books price a player. nflverse
supplies schedules, its newest depth snapshot, injury reports when published,
and historical kicking logs. No merged CSV is required.

The roster stays in ``MY_TEAM_ROSTER`` because Yahoo's private season-long
roster API requires OAuth. Expected points are the default objective. Historical
depth CVs add P25/P90 diagnostics but do not apply a second role haircut to a
Yahoo projection whose weekly salary already reflects current information.

Run it with no arguments to fetch the feeds and print a lineup; run it with
``--self-test`` for the offline checks in ``self_test``. ``run_lineup.py`` is
the non-interactive entry point that publishes the same lineup to the site.
"""

from __future__ import annotations

import importlib
import json
import re
import subprocess
import sys
import time
import unicodedata
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


# --------------------------- USER SETTINGS -------------------------------

MY_TEAM_ROSTER = [
    {"Name": "Dak Prescott", "Position": "QB"},
    {"Name": "Sam Darnold", "Position": "QB"},
    {"Name": "James Cook III", "Position": "RB"},
    {"Name": "Breece Hall", "Position": "RB"},
    {"Name": "Jaylen Waddle", "Position": "WR"},
    {"Name": "Tee Higgins", "Position": "WR"},
    {"Name": "Harold Fannin Jr", "Position": "TE"},
    {"Name": "Brandon Aubrey", "Position": "K"},
    {"Name": "Jameson Williams", "Position": "WR"},
    {"Name": "Brenton Strange", "Position": "TE"},
    {"Name": "Quentin Johnston", "Position": "WR"},
    {"Name": "Jaylen Warren", "Position": "RB"},
    {"Name": "Rashid Shaheed", "Position": "WR"},
    {"Name": "Isaac TeSlaa", "Position": "WR"},
]

STARTING_POSITIONS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "FLEX": 1}
FLEX_ELIGIBLE = ["RB", "WR", "TE"]
LINEUP_OBJECTIVE = "FP"  # FP, Floor_P25, or Ceiling_P90
EXCLUDED_PLAYERS: list[str] | None = None  # None prompts; [] skips prompt.
MANUAL_DEPTH_OVERRIDES: dict[str, int] = {}
AUTO_EXCLUDE_REPORTED_OUT = True
AUTO_INSTALL_NFLREADPY = True
# How deep the page's add pool goes per position. Deep enough to cover a real
# waiver claim, shallow enough that `pool.json` stays a small download.
POOL_LIMITS = {"QB": 40, "RB": 70, "WR": 90, "TE": 45, "K": 32}

# Market-implied means, from the same Bovada + Underdog engine the daily
# rankings use. Any failure degrades to the Yahoo blend rather than stopping.
USE_MARKET_PROJECTIONS = True
MARKET_SOURCE = "hybrid"  # hybrid, bovada, or underdog
MARKET_CACHE_HOURS = 2.0

YAHOO_URL = "https://dfyql-ro.sports.yahoo.com/v2/external/playersFeed/nfl"
KICKER_SCORING = {"FG_0_39": 3.0, "FG_40_49": 4.0, "FG_50_PLUS": 5.0, "PAT": 1.0}
KICKER_ROLLING_GAMES = 8
OFFENSE_FALLBACK_GAMES = 8

CALIBRATED_CV = {
    "QB": {1: .488, 2: .922, 3: .922, 4: .922},
    "RB": {1: .676, 2: .921, 3: 1.154, 4: 1.207},
    "WR": {1: .695, 2: .814, 3: 1.042, 4: 1.323},
    "TE": {1: .885, 2: 1.324, 3: 1.658, 4: 1.658},
    "DEF": {1: .950, 2: .950, 3: .950, 4: .950},
}
TEAM_MAP = {"JAX": "JAC", "LAR": "LA", "STL": "LA", "WSH": "WAS", "OAK": "LV", "SD": "LAC"}


def normalize_name(value: object) -> str:
    """Create a suffix- and punctuation-insensitive cross-provider name key."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"\s*\b(Jr\.?|Sr\.?|II|III|IV|V|VI|VII|VIII|IX|X)\b", "", value, flags=re.I)
    value = re.sub(r"[*+.'’-]", "", value)
    return re.sub(r"\s+", " ", value).strip().lower()


def normalize_team(value: object) -> str:
    team = "" if pd.isna(value) else str(value).strip().upper()
    return TEAM_MAP.get(team, team)


def ensure_nflreadpy():
    """Import nflreadpy, installing its released package in Colab if needed."""
    try:
        return importlib.import_module("nflreadpy")
    except ImportError as exc:
        if not AUTO_INSTALL_NFLREADPY:
            raise RuntimeError("Install nflreadpy with: pip install nflreadpy") from exc
        print("Installing nflreadpy ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "nflreadpy"])
        return importlib.import_module("nflreadpy")


def to_pandas(polars_frame) -> pd.DataFrame:
    """Convert filtered Polars data without adding a pyarrow dependency."""
    return pd.DataFrame(polars_frame.to_dicts())


def fetch_yahoo(attempts: int = 3, timeout: int = 20) -> pd.DataFrame:
    """Fetch and normalize Yahoo's public current-week NFL player feed."""
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            req = Request(YAHOO_URL, headers={"User-Agent": "Mozilla/5.0 SeasonLineup/1.0"})
            with urlopen(req, timeout=timeout) as response:
                payload = json.loads(response.read().decode())
            rows = payload.get("players", {}).get("result", [])
            if not rows:
                raise ValueError("Yahoo returned no players")
            break
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt == attempts:
                raise RuntimeError(f"Yahoo feed failed: {last_error}") from exc
            time.sleep(.75 * attempt)

    out = pd.DataFrame(rows).rename(columns={
        "name": "Feed_Name", "position": "Feed_Position", "team": "Team",
        "salary": "Salary", "fppg": "FPPG", "gameStartTime": "Game_Time",
        "homeTeam": "Home_Team", "awayTeam": "Away_Team",
    })
    needed = {"Feed_Name", "Feed_Position", "Team", "Salary", "FPPG", "Game_Time", "Home_Team", "Away_Team"}
    missing = sorted(needed - set(out))
    if missing:
        raise ValueError(f"Yahoo schema changed; missing {missing}")
    out["Feed_Position"] = out["Feed_Position"].astype(str).str.upper().replace({"D/ST": "DEF", "DST": "DEF"})
    for col in ["Team", "Home_Team", "Away_Team"]:
        out[col] = out[col].map(normalize_team)
    out["Salary"] = pd.to_numeric(out["Salary"], errors="coerce")
    out["FPPG"] = pd.to_numeric(out["FPPG"], errors="coerce").fillna(0)
    out["Game_Time"] = pd.to_datetime(out["Game_Time"], errors="coerce", utc=True)
    out["Game_Date"] = out["Game_Time"].dt.strftime("%Y-%m-%d")
    out["Opponent"] = np.where(out["Team"].eq(out["Away_Team"]), out["Home_Team"], out["Away_Team"])
    out["Key"] = out["Feed_Name"].map(normalize_name)
    out = out[out["Feed_Position"].isin(["QB", "RB", "WR", "TE", "DEF"]) & out["Salary"].gt(0)].copy()

    out["Salary_Prior"] = np.nan
    for position, group in out.groupby("Feed_Position"):
        train = group[group["FPPG"].gt(.25)]
        if len(train) >= 5 and train["Salary"].nunique() >= 3:
            x, y = train["Salary"].to_numpy(float), train["FPPG"].to_numpy(float)
            centered = x - x.mean()
            slope = np.dot(centered, y - y.mean()) / (np.dot(centered, centered) + 25)
            prior = y.mean() + np.clip(slope, .05, 1.25) * (group["Salary"].to_numpy(float) - x.mean())
        else:
            ratio = np.median(train["FPPG"] / train["Salary"]) if len(train) else .45
            prior = group["Salary"].to_numpy(float) * np.clip(ratio, .15, .90)
        out.loc[group.index, "Salary_Prior"] = np.maximum(prior, .25)
    history = out["FPPG"].gt(.25)
    out["Projected_FP"] = np.where(history, .70*out["FPPG"] + .30*out["Salary_Prior"], .80*out["Salary_Prior"])
    out["Projection_Source"] = np.where(history, "Yahoo FPPG + weekly salary prior", "Yahoo weekly salary prior")
    out["Fallback_Depth"] = out.groupby(["Team", "Feed_Position"])["Projected_FP"].rank(method="first", ascending=False)
    return out.sort_values("Projected_FP", ascending=False).drop_duplicates("Key").reset_index(drop=True)


def yahoo_from_prepared_slate(players: pd.DataFrame) -> pd.DataFrame:
    """Adapt the rankings pipeline's final player pool for the lineup builder.

    `pipeline.notebook.prepare_slate_pool` has already fetched Yahoo and the
    sportsbooks, resolved nflverse roles, applied the confidence audit and
    applied any prior-share role adjustment.  Re-fetching those inputs here is
    what allowed two pages from one site publish to disagree.  This adapter
    preserves that final `Projected_FP` verbatim and only adds the legacy column
    aliases the season-long roster resolver expects.
    """
    required = {
        "Name", "Position", "Team", "Opponent", "Game Time", "Home Team",
        "Away Team", "Salary", "FPPG", "Projected_FP", "Projection_Source",
    }
    missing = sorted(required - set(players.columns))
    if missing:
        raise ValueError(f"Prepared slate is missing columns: {missing}")

    out = players.copy()
    out["Feed_Name"] = out["Name"].astype(str)
    out["Feed_Position"] = out["Position"].astype(str).str.upper()
    out["Game_Time"] = pd.to_datetime(out["Game Time"], errors="coerce", utc=True)
    out["Game_Date"] = out["Game_Time"].dt.strftime("%Y-%m-%d")
    out["Home_Team"] = out["Home Team"].map(normalize_team)
    out["Away_Team"] = out["Away Team"].map(normalize_team)
    out["Team"] = out["Team"].map(normalize_team)
    out["Opponent"] = out["Opponent"].map(normalize_team)
    out["Key"] = out["Feed_Name"].map(normalize_name)

    if "Depth_Rank" in out:
        depth = pd.to_numeric(out["Depth_Rank"], errors="coerce")
    else:
        depth = pd.Series(np.nan, index=out.index, dtype=float)
    fallback = out.groupby(["Team", "Feed_Position"])["Projected_FP"].rank(
        method="first", ascending=False
    )
    out["Fallback_Depth"] = depth.fillna(fallback)
    return out.sort_values("Projected_FP", ascending=False).drop_duplicates("Key").reset_index(drop=True)


def pipeline_module():
    """Import the daily pipeline's market engine, or None if it is not there.

    The optimizer is meant to stay runnable as a single file, so a missing
    `pipeline` package is a downgrade to Yahoo priors, not an error.
    """
    try:
        return importlib.import_module("pipeline.notebook")
    except Exception as exc:  # ImportError, but a broken module should not stop a lineup
        warnings.warn(f"Market engine not importable ({exc}); using Yahoo priors.")
        return None


def market_settings(nb, cfg=None):
    """Settings for the market engine, with unmatched players always kept.

    `market_drop_unmatched` is stated rather than inherited: the rankings pool
    can afford to drop a player the books do not price, but this is my own
    roster, and a dropped player would silently vanish from the lineup.
    """
    if cfg is not None:
        return cfg
    return nb.Settings(
        market_source=MARKET_SOURCE,
        market_cache_hours=MARKET_CACHE_HOURS,
        market_drop_unmatched=False,
    )


def _empty_market_audit(note: str | None = None) -> dict:
    return {"feeds": [], "notes": [note] if note else [], "logit_vig": np.nan,
            "calibration_pairs": 0, "matched": 0, "accepted": 0, "skill_rows": 0}


def apply_market_projections(yahoo: pd.DataFrame, projections=None, cfg=None):
    """Replace Yahoo blends with accepted market means across the whole slate.

    Same order as the daily rankings: an accepted market mean wins, and anything
    the books do not price keeps its Yahoo FPPG and salary prior. Matching is
    the pipeline's own -- team, name key and a hard kickoff-time guard -- so a
    player never inherits a namesake's line from another game.

    `projections` is injectable so the matching rules can be exercised without
    either sportsbook.
    """
    nb = pipeline_module()
    if nb is None:
        return yahoo, _empty_market_audit("pipeline.notebook unavailable")

    audit = _empty_market_audit()
    try:
        settings = market_settings(nb, cfg)
        if projections is None:
            projections, loaded = nb.load_market_projection_reference(settings)
            audit.update(loaded)
        if not projections:
            audit["notes"].append("no usable market projections; keeping Yahoo priors")
            return yahoo, audit

        pool = yahoo.rename(columns={"Feed_Name": "Name", "Feed_Position": "Position"})
        reports = [
            nb.build_market_projection_report(group, projections, {"Game Time": kickoff}, settings)
            for kickoff, group in pool.groupby("Game_Time")
            if len(group)
        ]
        report = pd.concat(reports, ignore_index=True) if reports else pd.DataFrame()
        if not len(report):
            audit["notes"].append("no market row matched a Yahoo game")
            return yahoo, audit

        applied, _ = nb.apply_market_projection_means(pool, report, settings)
        out = applied.rename(columns={"Name": "Feed_Name", "Position": "Feed_Position"})
        # Salary order is the depth fallback, and accepted means have just
        # reordered the pool, so the fallback is re-ranked on the new numbers.
        out["Fallback_Depth"] = out.groupby(["Team", "Feed_Position"])["Projected_FP"].rank(
            method="first", ascending=False
        )
        audit["matched"] = int(report["Market matched"].sum())
        audit["accepted"] = int(report["Market accepted"].sum())
        audit["skill_rows"] = int(report["Position"].ne("DEF").sum())
        return out, audit
    except Exception as exc:  # one bad feed must not cost the week's lineup
        warnings.warn(f"Market projections unavailable ({exc}); using Yahoo priors.")
        audit["notes"].append(f"market step failed ({type(exc).__name__}: {exc})")
        return yahoo, audit


def load_nfl_context(yahoo: pd.DataFrame) -> dict:
    """Load the matching nflverse schedule, newest depth, injuries, and kicker logs."""
    import polars as pl
    nfl = ensure_nflreadpy()
    season = int(yahoo["Game_Time"].dt.year.mode().iloc[0])
    schedule = to_pandas(nfl.load_schedules(season))
    for col in ["away_team", "home_team"]:
        schedule[col] = schedule[col].map(normalize_team)
    schedule["gameday"] = schedule["gameday"].astype(str)
    yg = yahoo[["Game_Date", "Away_Team", "Home_Team"]].drop_duplicates()
    matched = schedule.merge(yg, left_on=["gameday", "away_team", "home_team"], right_on=["Game_Date", "Away_Team", "Home_Team"])
    week = int(matched["week"].mode().iloc[0]) if len(matched) else None
    games = schedule[schedule["week"].eq(week)].copy() if week else matched.copy()

    team_rows = []
    for g in games.itertuples():
        total = float(g.total_line) if pd.notna(g.total_line) else np.nan
        spread = float(g.spread_line) if pd.notna(g.spread_line) else np.nan
        for team, opp, side, sign in [(g.away_team, g.home_team, "Away", -1), (g.home_team, g.away_team, "Home", 1)]:
            implied = total/2 + sign*spread/2 if np.isfinite(total) and np.isfinite(spread) else np.nan
            team_rows.append({"Team": team, "Opponent": opp, "NFL_Week": int(g.week), "Home_Away": side,
                              "Vegas_Total": total, "Implied_Team_Total": implied})
    team_schedule = pd.DataFrame(team_rows)

    raw_depth = nfl.load_depth_charts(season)
    stamp = raw_depth.select(pl.col("dt").max()).item()
    depth = to_pandas(raw_depth.filter(pl.col("dt") == stamp).select(
        [c for c in ["team", "player_name", "pos_abb", "pos_rank"] if c in raw_depth.columns]))
    depth["Team"] = depth["team"].map(normalize_team)
    depth["Key"] = depth["player_name"].map(normalize_name)
    depth["Official_Depth"] = pd.to_numeric(depth["pos_rank"], errors="coerce")
    depth = depth.sort_values("Official_Depth").drop_duplicates(["Key", "Team"])

    notes = []
    try:
        injuries = to_pandas(nfl.load_injuries(season))
        if week is not None:
            injuries = injuries[injuries["week"].eq(week)].copy()
        injuries["Team"] = injuries["team"].map(normalize_team)
        injuries["Key"] = injuries["full_name"].map(normalize_name)
        injuries = injuries.drop_duplicates(["Key", "Team"], keep="last")
    except Exception as exc:
        # A warning goes to stderr, which the published run log never sees. A
        # lineup built with no injury data can start a player who is already
        # ruled out, so this has to reach the page, not just the console.
        notes.append(
            f"Injury report unavailable ({exc}); nobody was auto-benched -- "
            "check the injury news yourself."
        )
        warnings.warn(f"Current injury report unavailable; verify manually: {exc}")
        injuries = pd.DataFrame()

    stat_seasons = [season-2, season-1]
    if season <= int(nfl.get_current_season()) and week and week > 1:
        stat_seasons.append(season)
    raw_stats = nfl.load_player_stats(sorted(set(stat_seasons)))
    cols = ["player_display_name", "position", "team", "season", "week", "season_type",
            "fantasy_points", "fantasy_points_ppr",
            "fg_missed", "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
            "fg_made_40_49", "fg_made_50_59", "fg_made_60_", "pat_made"]
    stats = to_pandas(raw_stats.select([c for c in cols if c in raw_stats.columns]))
    return {"season": season, "week": week, "schedule": team_schedule, "depth": depth,
            "depth_stamp": str(stamp), "injuries": injuries, "stats": stats,
            "notes": notes}


def kicker_estimate(name: str, logs: pd.DataFrame) -> tuple[float, float, int]:
    """Fit kicker mean and shrunk CV from recent nflverse game logs."""
    x = logs.copy()
    x = x[(x["position"] == "K") & (x["season_type"] == "REG")]
    numeric = ["fg_missed", "fg_made_0_19", "fg_made_20_29", "fg_made_30_39",
               "fg_made_40_49", "fg_made_50_59", "fg_made_60_", "pat_made"]
    for col in numeric:
        x[col] = pd.to_numeric(x.get(col, 0), errors="coerce").fillna(0)
    x["KFP"] = (KICKER_SCORING["FG_0_39"]*(x.fg_made_0_19+x.fg_made_20_29+x.fg_made_30_39)
                + KICKER_SCORING["FG_40_49"]*x.fg_made_40_49
                + KICKER_SCORING["FG_50_PLUS"]*(x.fg_made_50_59+x.fg_made_60_)
                + KICKER_SCORING["PAT"]*x.pat_made)
    x["Key"] = x["player_display_name"].map(normalize_name)
    recent = x[x["Key"].eq(normalize_name(name))].sort_values(["season", "week"]).tail(KICKER_ROLLING_GAMES)
    global_mean = float(x.KFP.mean()) if len(x) else 7.0
    global_cv = float(x.KFP.std()/global_mean) if len(x) and global_mean > 0 else .60
    mean = float(recent.KFP.mean()) if len(recent) else global_mean
    player_cv = float(recent.KFP.std()/mean) if len(recent) > 1 and mean > 0 else global_cv
    weight = len(recent)/(len(recent)+8)
    return max(mean, .05), float(np.clip(weight*player_cv+(1-weight)*global_cv, .20, 1.50)), len(recent)


def offense_fallback(name: str, position: str, logs: pd.DataFrame) -> tuple[float, int]:
    """Use recent nflverse half-PPR results when Yahoo omits a scheduled game.

    This is intentionally a fallback, not a preferred projection. It prevents
    Monday-only or otherwise omitted Yahoo DFS games from becoming false zeroes.
    """
    x = logs.copy()
    if "fantasy_points" not in x or "fantasy_points_ppr" not in x:
        return 0.0, 0
    x = x[(x["position"].astype(str).str.upper() == position) &
          (x["season_type"].astype(str).str.upper() == "REG")].copy()
    x["Key"] = x["player_display_name"].map(normalize_name)
    x["Half_PPR"] = .5 * (
        pd.to_numeric(x["fantasy_points"], errors="coerce").fillna(0)
        + pd.to_numeric(x["fantasy_points_ppr"], errors="coerce").fillna(0)
    )
    recent = x[x["Key"].eq(normalize_name(name))].sort_values(["season", "week"]).tail(
        OFFENSE_FALLBACK_GAMES
    )
    return (float(recent["Half_PPR"].mean()), len(recent)) if len(recent) else (0.0, 0)


def build_roster(configured: list[dict], yahoo: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """Resolve the configured roster across Yahoo and nflverse, then add ranges."""
    roster = pd.DataFrame(configured)
    roster["Name"] = roster["Name"].astype(str).str.strip()
    roster["Position"] = roster["Position"].astype(str).str.upper()
    roster["Key"] = roster["Name"].map(normalize_name)
    ycols = ["Key", "Feed_Name", "Feed_Position", "Team", "Opponent", "Game_Time", "Salary", "FPPG",
             "Projected_FP", "Projection_Source", "Fallback_Depth"]
    # Present only once the market step has run.
    ycols += [c for c in ("Market_Quality", "Market_Method", "Fallback_Projected_FP")
              if c in yahoo.columns]
    roster = roster.merge(yahoo[ycols], on="Key", how="left")

    d = ctx["depth"][["Key", "Team", "Official_Depth", "pos_abb"]].rename(columns={"Team": "Depth_Team"})
    d = d.drop_duplicates("Key")
    roster = roster.merge(d, on="Key", how="left")
    roster["Team"] = roster["Team"].fillna(roster["Depth_Team"]).map(normalize_team)
    roster["Depth_Rank"] = pd.to_numeric(roster["Official_Depth"], errors="coerce").fillna(roster["Fallback_Depth"])
    roster["Depth_Source"] = np.where(roster["Official_Depth"].notna(), "nflverse latest depth", "Yahoo projection fallback")
    # Created up front: a roster whose every player misses both the depth chart
    # and the kicker branch would otherwise never define the column at all.
    roster["Projection_CV"] = np.nan

    playing_teams = set(ctx["schedule"]["Team"])
    for i, p in roster.iterrows():
        if p.Position == "K":
            mean, cv, n = kicker_estimate(p.Name, ctx["stats"])
            roster.at[i, "Projected_FP"] = mean
            roster.at[i, "Projection_CV"] = cv
            roster.at[i, "Projection_Source"] = f"nflverse rolling kicker games (n={n})"
            roster.at[i, "Depth_Rank"] = 1
            roster.at[i, "Depth_Source"] = "nflverse PK depth"
        elif pd.isna(p.Projected_FP) and p.Team in playing_teams:
            mean, n = offense_fallback(p.Name, p.Position, ctx["stats"])
            if n:
                roster.at[i, "Projected_FP"] = mean
                roster.at[i, "Projection_Source"] = (
                    f"nflverse rolling half-PPR fallback (n={n}); Yahoo game absent"
                )
        if p.Position != "K" and p.Position in CALIBRATED_CV and pd.notna(roster.at[i, "Depth_Rank"]):
            bucket = min(max(int(roster.at[i, "Depth_Rank"]), 1), 4)
            roster.at[i, "Projection_CV"] = CALIBRATED_CV[p.Position][bucket]

    for name, depth in MANUAL_DEPTH_OVERRIDES.items():
        mask = roster.Key.eq(normalize_name(name))
        roster.loc[mask, "Depth_Rank"] = max(1, int(depth))
        roster.loc[mask, "Depth_Source"] = "manual override"

    roster["Projected_FP"] = pd.to_numeric(roster["Projected_FP"], errors="coerce").fillna(0)
    roster["Projection_Source"] = roster["Projection_Source"].fillna(
        "No weekly projection; bye or unmatched player"
    )
    roster["FP"] = roster["Projected_FP"]
    roster = roster.merge(ctx["schedule"].drop_duplicates("Team"), on="Team", how="left", suffixes=("", "_NFL"))
    roster["Opponent"] = roster["Opponent"].fillna(roster.get("Opponent_NFL"))
    if len(ctx["injuries"]):
        keep = [c for c in ["Key", "Team", "report_primary_injury", "report_status", "practice_status"] if c in ctx["injuries"]]
        roster = roster.merge(ctx["injuries"][keep], on=["Key", "Team"], how="left")
    else:
        roster["report_status"] = pd.NA
        roster["practice_status"] = pd.NA

    roster["Floor_P25"], roster["Ceiling_P90"] = np.nan, np.nan
    fitted = roster["Projection_CV"].notna()
    sigma = np.sqrt(np.log1p(roster.loc[fitted, "Projection_CV"].astype(float)**2))
    mean = roster.loc[fitted, "FP"]
    roster.loc[fitted, "Floor_P25"] = mean*np.exp(-.67449*sigma-.5*sigma**2)
    roster.loc[fitted, "Ceiling_P90"] = mean*np.exp(1.28155*sigma-.5*sigma**2)
    roster["Projection_Available"] = roster["FP"].gt(0)
    return roster


def pool_entries(yahoo: pd.DataFrame, ctx: dict, limits: dict[str, int] | None = None) -> list[dict]:
    """Name the adds worth resolving: this week's best players at each slot.

    The lineup page's roster editor needs a player who is *not* on the roster to
    carry the same resolved projection a rostered player carries, and only
    `build_roster` produces that. Ranking on the Yahoo/market mean before
    resolving is safe, because `build_roster` leaves a skill player's mean
    alone -- the top of this list is the top of the built pool. Kickers are the
    exception: the DFS feed does not price them at all, so they come off the
    depth chart and are ranked only once their rolling-log means exist.
    """
    limits = dict(limits or POOL_LIMITS)
    entries: list[dict] = []
    for position, limit in limits.items():
        if position == "K" or not limit:
            continue
        rows = yahoo[yahoo["Feed_Position"].eq(position)].nlargest(int(limit), "Projected_FP")
        entries += [{"Name": str(name), "Position": position} for name in rows["Feed_Name"]]

    depth = ctx.get("depth")
    if not limits.get("K") or depth is None or "player_name" not in depth:
        return entries
    kickers = depth[depth["pos_abb"].astype(str).str.upper().isin(["K", "PK"])]
    playing = set(ctx["schedule"]["Team"]) if len(ctx.get("schedule", [])) else set()
    if playing:
        kickers = kickers[kickers["Team"].isin(playing)]
    # One kicker per team: a depth chart's second placekicker is a camp body,
    # and his rolling logs would price him like the starter he is not.
    kickers = kickers.sort_values("Official_Depth").drop_duplicates("Team").drop_duplicates("Key")
    entries += [{"Name": str(name), "Position": "K"} for name in kickers["player_name"]]
    return entries


def build_pool(yahoo: pd.DataFrame, ctx: dict, limits: dict[str, int] | None = None) -> pd.DataFrame:
    """Resolve the replacement pool the lineup page can add a player from.

    Same columns `build_roster` gives the roster, so a player swapped in on the
    page is scored by the numbers this run produced rather than by a stale
    snapshot the browser kept.
    """
    limits = dict(limits or POOL_LIMITS)
    entries = pool_entries(yahoo, ctx, limits)
    if not entries:
        return pd.DataFrame()
    pool = build_roster(entries, yahoo, ctx).drop_duplicates("Key")
    pool = pool.sort_values("FP", ascending=False)
    kept = [group.head(int(limits.get(position, 0)))
            for position, group in pool.groupby("Position", sort=False)]
    pool = pd.concat(kept) if kept else pool.iloc[:0]
    return pool.sort_values("FP", ascending=False).reset_index(drop=True)


def load_roster(path: str | None = None) -> list[dict]:
    """Read a roster JSON file, falling back to the roster in this file.

    The workflow needs the roster in data rather than in code, so a bench change
    is a JSON edit and not a Python edit. Shape: `[{"Name": ..., "Position": ...}]`.
    """
    if not path:
        return list(MY_TEAM_ROSTER)
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(entries, dict):
        entries = entries.get("roster", [])
    roster = [{"Name": str(e["Name"]), "Position": str(e["Position"]).upper()}
              for e in entries if e.get("Name") and e.get("Position")]
    if not roster:
        raise ValueError(f"No players in roster file {path}")
    return roster


def reported_out(roster: pd.DataFrame) -> list[str]:
    """Names this week's injury report lists as Out."""
    if "report_status" not in roster:
        return []
    flag = roster["report_status"].fillna("").astype(str).str.casefold().eq("out")
    return roster.loc[flag, "Name"].tolist()


def optimize(roster: pd.DataFrame, excluded: list[str], objective: str = LINEUP_OBJECTIVE):
    """Exactly fill fixed slots and one RB/WR/TE flex for an additive metric."""
    if objective not in {"FP", "Floor_P25", "Ceiling_P90"}:
        raise ValueError("LINEUP_OBJECTIVE must be FP, Floor_P25, or Ceiling_P90")
    work = roster.copy()
    work["Selection_Score"] = pd.to_numeric(work[objective], errors="coerce").fillna(work.FP)
    excluded_keys = {normalize_name(n) for n in excluded}
    available = work[~work.Key.isin(excluded_keys)].copy()
    ids = []
    for position, count in STARTING_POSITIONS.items():
        if position == "FLEX": continue
        chosen = available[available.Position.eq(position)].nlargest(count, "Selection_Score")
        if len(chosen) < count: print(f"Warning: only {len(chosen)} of {count} available for {position}.")
        ids += chosen.index.tolist(); available = available.drop(chosen.index)
    flex = available[available.Position.isin(FLEX_ELIGIBLE)].nlargest(STARTING_POSITIONS.get("FLEX", 0), "Selection_Score")
    ids += flex.index.tolist()
    starters = work.loc[ids].copy(); starters["Slot"] = starters.Position
    starters.loc[flex.index, "Slot"] = "FLEX"
    return starters, work.drop(ids)


def review(p: pd.Series) -> str:
    notes = []
    if not p.Projection_Available: notes.append("no weekly projection/bye")
    if pd.notna(p.get("report_status")): notes.append(f"injury {p.report_status}")
    if pd.notna(p.get("practice_status")): notes.append(str(p.practice_status))
    if pd.notna(p.Depth_Rank) and int(p.Depth_Rank) >= 3 and p.Position != "K": notes.append(f"depth {int(p.Depth_Rank)}")
    return "; ".join(notes) or "ok"


def output_table(frame: pd.DataFrame, starters: bool) -> pd.DataFrame:
    rows = []
    for _, p in frame.iterrows():
        row = {"Player": p.Name, "Pos": p.Position, "Team": p.Team, "Opp": p.Opponent,
               "Mean": round(p.FP,2), "P25": round(p.Floor_P25,2) if pd.notna(p.Floor_P25) else np.nan,
               "P90": round(p.Ceiling_P90,2) if pd.notna(p.Ceiling_P90) else np.nan,
               "Depth": int(p.Depth_Rank) if pd.notna(p.Depth_Rank) else pd.NA,
               "Source": p.Projection_Source, "Review": review(p)}
        if starters: row = {"Slot": p.Slot} | row
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Mean", ascending=False)


def run(use_market: bool | None = None, roster_path: str | None = None) -> dict:
    """Fetch every provider, optimize the configured roster, and print results."""
    print("Loading Yahoo weekly projections ...")
    yahoo = fetch_yahoo()
    market_audit = _empty_market_audit("market projections disabled")
    if USE_MARKET_PROJECTIONS if use_market is None else use_market:
        print("Loading market-implied means from the sportsbook props ...")
        yahoo, market_audit = apply_market_projections(yahoo)
        for note in market_audit["notes"]:
            print(f"  Market: {note}")
        if market_audit["accepted"]:
            feeds = ", ".join(market_audit["feeds"]) or "market"
            print(f"  Market: {feeds}; matched {market_audit['matched']}/"
                  f"{market_audit['skill_rows']} skill-player rows; "
                  f"accepted {market_audit['accepted']} means.")
    print("Loading nflverse schedule, depth, injuries, and kicking logs ...")
    ctx = load_nfl_context(yahoo)
    roster = build_roster(load_roster(roster_path), yahoo, ctx)
    print(f"\nSeason {ctx['season']} week {ctx['week']}; depth snapshot {ctx['depth_stamp']}")
    for note in ctx.get("notes", []):
        print(f"  {note}")
    print("Yahoo projections use DFS half-PPR scoring; verify your league scoring and injury news.")

    if EXCLUDED_PLAYERS is None:
        view = roster.sort_values(["Position", "FP"], ascending=[True, False]).reset_index(drop=True)
        print("\n--- Full Roster ---")
        for n, (_, p) in enumerate(view.iterrows(), 1):
            print(f"{n}. {p.Name} ({p.Position}, {p.Team} vs {p.Opponent}) — {p.FP:.2f}; {review(p)}")
        raw = input("\nNumbers to force to bench, comma-separated, or Enter: ").strip()
        try:
            picks = [int(x.strip())-1 for x in raw.split(",")] if raw else []
            excluded = view.iloc[picks].Name.tolist() if all(0 <= i < len(view) for i in picks) else []
        except ValueError:
            excluded = []
    else:
        excluded = list(EXCLUDED_PLAYERS)
    if AUTO_EXCLUDE_REPORTED_OUT:
        excluded = list(dict.fromkeys(excluded + reported_out(roster)))
    starters, bench = optimize(roster, excluded)
    print("\n--- Recommended Starters ---"); print(output_table(starters, True).to_string(index=False))
    print("\n--- Bench ---"); print(output_table(bench, False).to_string(index=False))
    print(f"\nStarter projected mean: {starters.FP.sum():.2f}")
    return {"yahoo": yahoo, "context": ctx, "roster": roster, "starters": starters,
            "bench": bench, "excluded": excluded, "market_audit": market_audit}


def self_test() -> None:
    """Run small offline checks for joins, kicker scoring, and lineup slots."""
    assert normalize_name("James Cook III") == normalize_name("James Cook")
    roster = pd.DataFrame([
        ("QB1","qb1","QB",20),("QB2","qb2","QB",15),("RB1","rb1","RB",16),
        ("RB2","rb2","RB",14),("RB3","rb3","RB",10),("WR1","wr1","WR",15),
        ("WR2","wr2","WR",13),("WR3","wr3","WR",12),("TE1","te1","TE",9),
        ("TE2","te2","TE",7),("K1","k1","K",8)], columns=["Name","Key","Position","FP"])
    roster["Floor_P25"] = roster.FP*.7; roster["Ceiling_P90"] = roster.FP*1.5
    starters, bench = optimize(roster, [])
    assert len(starters) == 8 and len(bench) == 3 and starters.Slot.eq("FLEX").sum() == 1
    print("Self-test passed.")


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        self_test()
    else:
        run()
