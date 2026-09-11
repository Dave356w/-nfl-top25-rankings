"""Historical Yahoo NFL single-game ingestion and leakage-aware backtesting.

Yahoo's DFS endpoints are public but undocumented.  The immutable fields used
here are the series/game ids, salaries, salary caps, player ids, and settled
points.  Mutable player metadata (team, FPPG, projections, status) is rebuilt
from nflverse instead of being treated as historical truth.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from pipeline import notebook as nb
from pipeline import portfolio_construction, scenario_portfolio


YAHOO_DFS = "https://dfyql-ro.sports.yahoo.com/v2"
TEAM_ALIASES = {
    "JAC": "JAX", "LA": "LAR", "STL": "LAR", "SD": "LAC",
    "OAK": "LV", "WSH": "WAS",
}
OFFENSIVE_POSITIONS = ("QB", "RB", "WR", "TE")


def _team(value) -> str:
    code = str(value or "").upper()
    return TEAM_ALIASES.get(code, code)


def _pandas(frame) -> pd.DataFrame:
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    try:
        return frame.to_pandas()
    except (AttributeError, ModuleNotFoundError):
        return pd.DataFrame(frame.to_dicts())


class YahooDfsClient:
    """Small caching client for Yahoo's read-only DFS endpoints."""

    def __init__(self, cache_dir="backtest_cache/yahoo", refresh=False,
                 timeout=30, pause=0.05):
        self.cache_dir = Path(cache_dir)
        self.refresh = bool(refresh)
        self.timeout = int(timeout)
        self.pause = float(pause)

    def _get(self, path: str, params=None, cache_name=None):
        query = urlencode(params or {})
        url = f"{YAHOO_DFS}/{path}" + (f"?{query}" if query else "")
        name = cache_name or hashlib.sha256(url.encode()).hexdigest()
        target = self.cache_dir / f"{name}.json.gz"
        if target.exists() and not self.refresh:
            return json.loads(gzip.decompress(target.read_bytes()))
        last = None
        for attempt in range(3):
            try:
                request = Request(url, headers={
                    "User-Agent": "Mozilla/5.0 Yahoo-Showdown-Backtest/1.0"
                })
                with urlopen(request, timeout=self.timeout) as response:
                    raw = response.read()
                payload = json.loads(raw)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(gzip.compress(
                    json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(),
                    mtime=0,
                ))
                if self.pause:
                    time.sleep(self.pause)
                return payload
            except Exception as exc:  # bounded retry for an undocumented endpoint
                last = exc
                if attempt < 2:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Yahoo request failed: {url}: {last}") from last

    def discover(self, start: date, end: date) -> list[dict]:
        """Return one normalized row per completed NFL single-game series.

        The interval is half-open. Requests are split into 31-day windows so a
        future Yahoo response cap cannot silently truncate a season.
        """
        found = {}
        cursor = start
        while cursor < end:
            stop = min(cursor + timedelta(days=31), end)
            start_ms = int(datetime.combine(cursor, datetime.min.time(), timezone.utc).timestamp() * 1000)
            stop_ms = int(datetime.combine(stop, datetime.min.time(), timezone.utc).timestamp() * 1000)
            payload = self._get("contestSeries", {
                "state": "completed", "startTimeMin": start_ms,
                "startTimeMax": stop_ms, "slateTypes": "SINGLE_GAME",
            }, f"series_{start_ms}_{stop_ms}")
            games = (payload.get("games", {}).get("result") or {})
            for sport in payload.get("sports", {}).get("result") or []:
                if sport.get("sportCode") != "nfl":
                    continue
                for item in sport.get("series") or []:
                    series = item.get("series") or {}
                    codes = item.get("gameCodeList") or []
                    if len(codes) != 1:
                        continue
                    game_item = games.get(codes[0]) or {}
                    game = game_item.get("game", game_item)
                    found[int(series["id"])] = {
                        "series_id": int(series["id"]),
                        "series_name": series.get("name"),
                        "game_code": str(codes[0]),
                        "kickoff_ms": int(series.get("startTime") or game.get("startTime")),
                        "salary_cap": series.get("salaryCapOverride"),
                        "home": _team((game.get("homeTeam") or {}).get("abbr")
                                      if isinstance(game.get("homeTeam"), dict)
                                      else game.get("homeTeamAbbr") or game.get("homeTeam")),
                        "away": _team((game.get("awayTeam") or {}).get("abbr")
                                      if isinstance(game.get("awayTeam"), dict)
                                      else game.get("awayTeamAbbr") or game.get("awayTeam")),
                    }
            cursor = stop
        return sorted(found.values(), key=lambda row: (row["kickoff_ms"], row["series_id"]))

    def players(self, series_id: int) -> dict:
        return self._get("seriesPlayersLive", {"seriesId": int(series_id)},
                         f"players_{int(series_id)}")

    def benchmark(self, series_id: int) -> dict:
        return self._get(f"optimalLineup/{int(series_id)}", cache_name=f"optimal_{int(series_id)}")


def yahoo_offensive_points(frame: pd.DataFrame) -> pd.Series:
    """Yahoo half-PPR points reconstructed from nflverse player-game stats."""
    weights = {
        "passing_yards": 0.04, "passing_tds": 4.0,
        "passing_interceptions": -1.0, "rushing_yards": 0.10,
        "rushing_tds": 6.0, "receiving_yards": 0.10,
        "receiving_tds": 6.0, "receptions": 0.50,
        "passing_2pt_conversions": 2.0, "rushing_2pt_conversions": 2.0,
        "receiving_2pt_conversions": 2.0, "special_teams_tds": 6.0,
    }
    points = pd.Series(0.0, index=frame.index, dtype=float)
    for column, weight in weights.items():
        if column in frame:
            points += pd.to_numeric(frame[column], errors="coerce").fillna(0) * weight
    fumbles = [
        column for column in
        ("sack_fumbles_lost", "rushing_fumbles_lost", "receiving_fumbles_lost")
        if column in frame
    ]
    for column in fumbles:
        points -= 2.0 * pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return points


@dataclass
class NflverseHistory:
    schedules: pd.DataFrame
    stats: pd.DataFrame
    rosters: pd.DataFrame
    player_ids: pd.DataFrame

    @classmethod
    def load(cls, seasons: list[int]):
        try:
            import nflreadpy as nfl
        except ImportError as exc:
            raise RuntimeError("Install nflreadpy to run the historical backtest") from exc
        seasons = sorted(set(int(value) for value in seasons))
        schedules = _pandas(nfl.load_schedules(seasons))
        stats = _pandas(nfl.load_player_stats(seasons, summary_level="week"))
        rosters = _pandas(nfl.load_rosters_weekly(seasons))
        player_ids = _pandas(nfl.load_ff_playerids())
        return cls.prepare(schedules, stats, rosters, player_ids)

    @classmethod
    def prepare(cls, schedules, stats, rosters, player_ids):
        schedules, stats = schedules.copy(), stats.copy()
        rosters, player_ids = rosters.copy(), player_ids.copy()
        schedules["home_team"] = schedules["home_team"].map(_team)
        schedules["away_team"] = schedules["away_team"].map(_team)
        schedules["game_date"] = pd.to_datetime(schedules["gameday"]).dt.date
        stats["Yahoo_Actual_FP"] = yahoo_offensive_points(stats)
        stats["Touches"] = (
            pd.to_numeric(stats.get("carries", 0), errors="coerce").fillna(0)
            + pd.to_numeric(stats.get("targets", 0), errors="coerce").fillna(0)
            + pd.to_numeric(stats.get("attempts", 0), errors="coerce").fillna(0)
        )
        stats = stats.merge(schedules[["game_id", "game_date"]], on="game_id", how="left")
        stats["team"] = stats["team"].map(_team)
        for frame in (rosters, player_ids):
            frame["yahoo_id"] = frame["yahoo_id"].astype(str).str.replace(r"\.0$", "", regex=True)
        rosters["team"] = rosters["team"].map(_team)
        return cls(schedules, stats, rosters, player_ids)

    def match_game(self, slate: dict) -> pd.Series | None:
        kickoff = datetime.fromtimestamp(slate["kickoff_ms"] / 1000, timezone.utc).date()
        games = self.schedules[
            self.schedules["home_team"].eq(_team(slate["home"]))
            & self.schedules["away_team"].eq(_team(slate["away"]))
        ].copy()
        if games.empty:
            return None
        games["distance"] = games["game_date"].map(lambda value: abs((value - kickoff).days))
        row = games.sort_values(["distance", "game_id"]).iloc[0]
        return row if int(row["distance"]) <= 1 else None

    def yahoo_to_gsis(self) -> dict[str, str]:
        usable = self.player_ids.dropna(subset=["yahoo_id", "gsis_id"])
        return dict(zip(usable["yahoo_id"].astype(str), usable["gsis_id"].astype(str)))

    def pregame(self, gsis_id: str | None, game_date: date) -> tuple[float, float, int]:
        if not gsis_id:
            return 0.0, 0.0, 0
        past = self.stats[
            self.stats["player_id"].eq(gsis_id)
            & self.stats["game_date"].notna()
            & self.stats["game_date"].lt(game_date)
        ].sort_values("game_date")
        if past.empty:
            return 0.0, 0.0, 0
        points = past.tail(8)["Yahoo_Actual_FP"]
        touches = past.tail(4)["Touches"]
        return float(points.mean()), float(touches.mean()), int(len(points))

    def roster_team(self, yahoo_id: str, game: pd.Series) -> tuple[str | None, str | None]:
        rows = self.rosters[
            self.rosters["season"].eq(int(game["season"]))
            & self.rosters["week"].eq(int(game["week"]))
            & self.rosters["yahoo_id"].eq(str(yahoo_id))
        ]
        valid = rows[rows["team"].isin({_team(game["home_team"]), _team(game["away_team"])})]
        if valid.empty:
            return None, None
        row = valid.iloc[-1]
        return str(row["team"]), str(row.get("status") or "")


def _player_name(row: dict) -> str:
    return " ".join(filter(None, [row.get("firstName"), row.get("lastName")])).strip()


def build_pool(slate: dict, payload: dict, history: NflverseHistory,
               apply_projection: bool = True) -> tuple[pd.DataFrame, dict]:
    """Build the exact-priced slate with only pregame nflverse features."""
    game = history.match_game(slate)
    if game is None:
        raise ValueError("No nflverse schedule match")
    game_stats = history.stats[history.stats["game_id"].eq(game["game_id"])]
    id_map = history.yahoo_to_gsis()
    teams = {_team(game["home_team"]), _team(game["away_team"])}
    rows, unresolved, unavailable = [], [], []
    source_counts = {"stats": 0, "roster": 0, "yahoo": 0}
    for raw in payload.get("players", {}).get("result") or []:
        position = str(raw.get("primaryPosition") or "").upper().replace("D/ST", "DEF")
        if position not in nb.VALID_POSITIONS:
            continue
        salary = pd.to_numeric(raw.get("salary"), errors="coerce")
        actual = pd.to_numeric(raw.get("points"), errors="coerce")
        if not np.isfinite(salary) or float(salary) <= 0 or not np.isfinite(actual):
            continue
        code = str(raw.get("code") or "")
        yahoo_match = re.search(r"nfl\.p\.(\d+)", code)
        yahoo_id = yahoo_match.group(1) if yahoo_match else None
        gsis_id = id_map.get(yahoo_id) if yahoo_id else None
        stat = game_stats[game_stats["player_id"].eq(gsis_id)] if gsis_id else game_stats.iloc[0:0]
        status = None
        if position == "DEF":
            team = _team((raw.get("team") or {}).get("abbr"))
            source = "yahoo"
        elif not stat.empty:
            team = _team(stat.iloc[0]["team"])
            source = "stats"
        else:
            team, status = history.roster_team(yahoo_id, game)
            source = "roster"
            if team is None:
                candidate = _team((raw.get("team") or {}).get("abbr"))
                team = candidate if candidate in teams else None
                source = "yahoo"
        if team not in teams:
            unresolved.append({"player": _player_name(raw), "yahoo_id": yahoo_id})
            continue
        # A settled zero is not proof that the player was eligible at lock. For
        # players with no game-stat row, mirror the live pipeline and trust only
        # an active weekly-roster status. A player who actually appeared is kept
        # even if a later roster snapshot disagrees.
        if position != "DEF" and stat.empty and status not in (None, "ACT"):
            unavailable.append({
                "player": _player_name(raw), "yahoo_id": yahoo_id, "status": status,
            })
            continue
        source_counts[source] += 1
        fppg, opportunity, games = history.pregame(gsis_id, game["game_date"])
        nfl_actual = float(stat.iloc[0]["Yahoo_Actual_FP"]) if not stat.empty else None
        rows.append({
            "Name": _player_name(raw), "Position": position, "Team": team,
            "Salary": float(salary), "Original_Salary": raw.get("originalSalary"),
            "FPPG": fppg, "Prior_Games": games, "Expected_Touches": opportunity,
            "Yahoo_Actual_FP": float(actual), "NFLverse_Actual_FP": nfl_actual,
            "Yahoo ID": int(yahoo_id) if yahoo_id else pd.NA,
            "GSIS ID": gsis_id, "Roster_Status": status,
            "Game ID": slate["game_code"],
        })
    pool = pd.DataFrame(rows)
    if pool.empty:
        raise ValueError("Yahoo returned no usable players")
    order = pool.sort_values(
        ["Team", "Position", "Expected_Touches", "Salary", "Name"],
        ascending=[True, True, False, False, True],
    )
    ranks = order.groupby(["Team", "Position"]).cumcount().add(1).reindex(pool.index)
    pool["Depth_Rank"] = ranks.astype(int)
    pool["Role_Tier"] = pool["Depth_Rank"].astype("Int64")
    pool["Depth_Source"] = "nflverse lagged opportunity"
    pool["Role_Label"] = [nb.role_label(pos, rank) for pos, rank in zip(
        pool["Position"], pool["Depth_Rank"]
    )]
    if apply_projection:
        pool = nb.add_projection_priors(pool)
    else:
        pool["Projection_Source"] = "historical feature extraction"
    pool, removed_qbs = nb.apply_default_role_filters(pool, cfg=nb.CFG)
    audit = {
        "unresolved_players": unresolved, "unavailable_players": unavailable,
        "historical_team_sources": source_counts,
        "backup_qbs_removed": list(removed_qbs), "nflverse_game_id": game["game_id"],
    }
    return pool.reset_index(drop=True), audit


def _lineup_actual(lineup, actual: np.ndarray) -> float:
    ids = np.asarray(lineup["Player_Ids"], dtype=int)
    star = int(lineup["Superstar_Id"])
    return float(actual[ids].sum() + 0.5 * actual[star])


def run_slate(slate: dict, yahoo: YahooDfsClient, history: NflverseHistory,
              cfg=None, fetch_benchmark=True) -> dict:
    cfg = nb._cfg(cfg)
    cap = slate.get("salary_cap")
    if cap is None:
        raise ValueError("Yahoo did not preserve this slate's salary cap")
    pool, audit = build_pool(slate, yahoo.players(slate["series_id"]), history)
    model_pool, dropped = nb.trim_player_pool(pool, cfg)
    problem = nb.roster_feasibility_error(model_pool, cfg.lineup_size)
    if problem:
        raise ValueError(problem)
    model = nb.build_correlation_model(model_pool)
    covariance = nb.analytic_covariance(model_pool, model)
    outcomes, _ = nb.simulate_player_outcomes(model_pool, cfg)
    candidates, valid = nb.enumerate_candidate_lineups(model_pool, float(cap), covariance, cfg)
    scored = nb.score_candidates_shared_scenarios(candidates, outcomes, cfg)
    if cfg.showdown_objective == "auto":
        portfolio = scenario_portfolio.select(scored, outcomes, cfg)
    else:
        portfolio = portfolio_construction.select_portfolio(scored, model_pool, cfg)
    actual = model_pool["Yahoo_Actual_FP"].to_numpy(float)
    selected = []
    for number, row in enumerate(portfolio.to_dict("records"), 1):
        ids = [int(value) for value in row["Player_Ids"]]
        selected.append({
            "entry": number, "players": [str(model_pool.iloc[i]["Name"]) for i in ids],
            "superstar": str(model_pool.iloc[int(row["Superstar_Id"])]["Name"]),
            "salary": float(row["Salary"]), "expected_fp": float(row["Expected_FP"]),
            "actual_fp": _lineup_actual(row, actual),
        })
    benchmark = None
    if fetch_benchmark:
        raw = yahoo.benchmark(slate["series_id"])
        result = (raw.get("allUserBestLineup", {}).get("result") or {})
        benchmark = {
            "best_submitted_score": result.get("score"),
            "best_submitted_salary": result.get("totalSalary"),
        }
    cross = model_pool.dropna(subset=["NFLverse_Actual_FP"])
    discrepancy = None
    if len(cross):
        discrepancy = float(np.mean(np.abs(
            cross["Yahoo_Actual_FP"].to_numpy(float)
            - cross["NFLverse_Actual_FP"].to_numpy(float)
        )))
    kickoff = datetime.fromtimestamp(slate["kickoff_ms"] / 1000, timezone.utc)
    return {
        "series_id": slate["series_id"], "game_code": slate["game_code"],
        "matchup": f"{slate['away']} @ {slate['home']}",
        "kickoff_utc": kickoff.isoformat(), "salary_cap": float(cap),
        "players_from_yahoo": int(len(pool)), "players_modeled": int(len(model_pool)),
        "valid_rosters": int(valid), "candidates_scored": int(len(scored)),
        "settings": {
            "simulations": int(cfg.simulations),
            "random_seed": int(cfg.random_seed),
            "entries": int(cfg.tournament_lineups),
            "objective": str(cfg.showdown_objective),
            "max_enumeration_players": int(cfg.max_enumeration_players),
            "max_candidate_lineups": int(cfg.max_candidate_lineups),
        },
        "dropped_from_model_pool": list(dropped), "portfolio": selected,
        "benchmark": benchmark, "nflverse_yahoo_actual_mae": discrepancy,
        "audit": audit,
    }


def summarize(results: list[dict], skipped: list[dict]) -> dict:
    entries = [entry for result in results for entry in result["portfolio"]]
    first = [result["portfolio"][0] for result in results if result["portfolio"]]
    portfolio_best = [
        max(entry["actual_fp"] for entry in result["portfolio"])
        for result in results if result["portfolio"]
    ]
    regrets = []
    for result in results:
        best = (result.get("benchmark") or {}).get("best_submitted_score")
        if best is not None and result["portfolio"]:
            realized = max(entry["actual_fp"] for entry in result["portfolio"])
            regrets.append(float(best) - float(realized))
    return {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "method": "walk-forward nflverse rolling means plus Yahoo salary prior; no historical player props",
        "slates_completed": len(results), "slates_skipped": len(skipped),
        "entries_scored": len(entries),
        "mean_first_entry_actual_fp": float(np.mean([x["actual_fp"] for x in first])) if first else None,
        "mean_portfolio_best_actual_fp": float(np.mean(portfolio_best)) if portfolio_best else None,
        "mean_regret_to_best_submitted": float(np.mean(regrets)) if regrets else None,
        "limitations": [
            "Yahoo settled points are used for contest scoring; nflverse offensive points are an audit.",
            "The production mean is the frozen Yahoo salary-position-depth regression.",
            "Yahoo player team, FPPG, projection, status, and image fields are mutable and are not used as historical facts.",
            "The current fitted CV and correlation constants are reused; use season-frozen fits for a strict out-of-sample parameter test.",
            "Best submitted score is not a contest ROI or payout simulation.",
        ],
        "results": results, "skipped": skipped,
    }
