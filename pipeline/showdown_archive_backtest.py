"""Grade immutable pregame Showdown portfolios against settled Yahoo points.

Published portfolios remain separate from portfolios reconstructed after the
game. Reconstruction may use a completed series' immutable salary cap, but the
players, projections, uncertainty model, settings, and code version all come
from the archived pregame snapshot.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import gzip
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

from pipeline import notebook as nb, projection_archive
from pipeline.showdown_backtest import (
    YahooDfsClient, _pandas, _player_name, _team, yahoo_offensive_points,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = ROOT / "site" / "data" / "projection_archive" / "snapshots"
RECONSTRUCTOR = ROOT / "tools" / "reconstruct_archived_showdown.js"


def _number(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(0.0, index=frame.index)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _strings(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series("", index=frame.index, dtype=str)
    return frame[column].fillna("").astype(str)


def yahoo_defense_points(game_pbp: pd.DataFrame, team: str) -> tuple[float, dict]:
    """Reconstruct Yahoo DFS DEF points from one nflverse game.

    Points allowed are derived from scoring plays rather than the final score so
    opponent defensive touchdowns, safeties, and defensive two-point returns are
    excluded while kick/punt return touchdowns and PATs remain included.
    """
    frame = game_pbp.copy()
    team = _team(team)
    if frame.empty:
        raise ValueError(f"No nflverse play-by-play rows for {team}")
    posteam = _strings(frame, "posteam").map(_team)
    defteam = _strings(frame, "defteam").map(_team)
    td_team = _strings(frame, "td_team").map(_team)
    special = _number(frame, "special_teams_play").eq(1)

    sacks = float(_number(frame, "sack")[defteam.eq(team)].sum())
    interceptions = float(_number(frame, "interception")[defteam.eq(team)].sum())
    lost = _number(frame, "fumble_lost").eq(1)
    recovery_one = _strings(frame, "fumble_recovery_1_team").map(_team).eq(team)
    recovery_two = _strings(frame, "fumble_recovery_2_team").map(_team).eq(team)
    fumble_recoveries = int((lost & (recovery_one | recovery_two)).sum())
    safeties = float(_number(frame, "safety")[defteam.eq(team)].sum())
    blocked = _strings(frame, "blocked_player_id").ne("")
    blocked_kicks = int((blocked & defteam.eq(team)).sum())
    return_tds = int((
        _number(frame, "return_touchdown").eq(1)
        & td_team.eq(team)
        & (posteam.ne(team) | special)
    ).sum())
    two_point_returns = float(
        _number(frame, "defensive_two_point_conv")[defteam.eq(team)].sum()
    )

    opponent = posteam.ne(team)
    offensive_tds = (
        _number(frame, "touchdown").eq(1)
        & td_team.ne(team) & opponent & defteam.eq(team)
    )
    special_return_tds = (
        _number(frame, "return_touchdown").eq(1)
        & td_team.ne(team) & posteam.eq(team) & special
    )
    field_goals = (
        _strings(frame, "field_goal_result").str.lower().eq("made")
        & opponent & defteam.eq(team)
    )
    extra_points = (
        _strings(frame, "extra_point_result").str.lower().isin(("good", "made"))
        & opponent & defteam.eq(team)
    )
    two_points = (
        _strings(frame, "two_point_conv_result").str.lower().isin(("success", "successful", "good"))
        & opponent & defteam.eq(team)
        & _number(frame, "defensive_two_point_conv").ne(1)
    )
    points_allowed = int(
        6 * (offensive_tds | special_return_tds).sum()
        + 3 * field_goals.sum() + extra_points.sum() + 2 * two_points.sum()
    )
    if points_allowed == 0:
        allowed_score = 10
    elif points_allowed <= 6:
        allowed_score = 7
    elif points_allowed <= 13:
        allowed_score = 4
    elif points_allowed <= 20:
        allowed_score = 1
    elif points_allowed <= 27:
        allowed_score = 0
    elif points_allowed <= 34:
        allowed_score = -1
    else:
        allowed_score = -4
    total = (
        sacks + 2 * safeties + 2 * interceptions + 2 * fumble_recoveries
        + 2 * blocked_kicks + 6 * return_tds + 2 * two_point_returns + allowed_score
    )
    return float(total), {
        "sacks": sacks, "safeties": safeties, "interceptions": interceptions,
        "fumble_recoveries": fumble_recoveries, "blocked_kicks": blocked_kicks,
        "return_touchdowns": return_tds, "two_point_returns": two_point_returns,
        "points_allowed": points_allowed, "points_allowed_score": allowed_score,
    }


@dataclass
class NflverseActuals:
    schedules: pd.DataFrame
    stats: pd.DataFrame
    player_ids: pd.DataFrame
    pbp: pd.DataFrame

    @classmethod
    def load(cls, seasons: list[int]):
        try:
            import nflreadpy as nfl
        except ImportError as exc:
            raise RuntimeError("Install nflreadpy to use the nflverse actuals fallback") from exc
        seasons = sorted(set(int(value) for value in seasons))
        return cls.prepare(
            _pandas(nfl.load_schedules(seasons)),
            _pandas(nfl.load_player_stats(seasons, summary_level="week")),
            _pandas(nfl.load_ff_playerids()),
            _pandas(nfl.load_pbp(seasons)),
        )

    @classmethod
    def prepare(cls, schedules, stats, player_ids, pbp):
        schedules, stats = schedules.copy(), stats.copy()
        player_ids, pbp = player_ids.copy(), pbp.copy()
        schedules["home_team"] = schedules["home_team"].map(_team)
        schedules["away_team"] = schedules["away_team"].map(_team)
        schedules["game_date"] = pd.to_datetime(schedules["gameday"]).dt.date
        stats["Yahoo_Actual_FP"] = yahoo_offensive_points(stats)
        stats["team"] = stats["team"].map(_team)
        player_ids["yahoo_id"] = player_ids["yahoo_id"].astype(str).str.replace(
            r"\.0$", "", regex=True
        )
        return cls(schedules, stats, player_ids, pbp)

    def _match_game(self, model: dict) -> pd.Series | None:
        kickoff = pd.Timestamp(model["kickoff_utc"]).tz_convert("UTC").date()
        games = self.schedules[
            self.schedules["home_team"].eq(_team(model["home"]))
            & self.schedules["away_team"].eq(_team(model["away"]))
        ].copy()
        if games.empty:
            return None
        games["distance"] = games["game_date"].map(lambda value: abs((value - kickoff).days))
        result = games.sort_values(["distance", "game_id"]).iloc[0]
        return result if int(result["distance"]) <= 1 else None

    def actuals(self, model: dict, player_keys: list[str]) -> tuple[dict[str, float], dict]:
        game = self._match_game(model)
        if game is None:
            return {}, {"reason": "no matching nflverse schedule game"}
        scores = pd.to_numeric(pd.Series([game.get("home_score"), game.get("away_score")]),
                               errors="coerce")
        if scores.isna().any():
            return {}, {"reason": "nflverse schedule game is not final", "game_id": game["game_id"]}
        game_id = str(game["game_id"])
        game_stats = self.stats[self.stats["game_id"].astype(str).eq(game_id)]
        game_pbp = self.pbp[self.pbp["game_id"].astype(str).eq(game_id)]
        id_rows = self.player_ids.dropna(subset=["yahoo_id", "gsis_id"])
        yahoo_to_gsis = dict(zip(
            id_rows["yahoo_id"].astype(str), id_rows["gsis_id"].astype(str)
        ))
        actuals, missing, defense_audit = {}, [], {}
        for key, player in zip(player_keys, model["players"]):
            if player["pos"] == "DEF":
                try:
                    value, detail = yahoo_defense_points(game_pbp, player["team"])
                    actuals[key] = value
                    defense_audit[player["team"]] = detail
                except ValueError:
                    missing.append(key)
                continue
            yahoo_id = key.split(":", 1)[1] if key.startswith("yahoo:") else None
            gsis_id = yahoo_to_gsis.get(yahoo_id) if yahoo_id else None
            rows = game_stats[game_stats["player_id"].astype(str).eq(str(gsis_id))] if gsis_id else game_stats.iloc[0:0]
            if len(rows) == 1:
                actuals[key] = float(rows.iloc[0]["Yahoo_Actual_FP"])
            else:
                # An absent weekly-stat row is not proof of a played zero.
                missing.append(key)
        return actuals, {
            "game_id": game_id, "matched_actuals": len(actuals),
            "missing_without_zero_fill": missing, "defense": defense_audit,
        }


def load_snapshot(directory=DEFAULT_ARCHIVE, snapshot_id=None) -> dict:
    """Load one immutable snapshot, defaulting to the latest Showdown snapshot."""
    matches = []
    for path in Path(directory).glob("*.json.gz"):
        document = json.loads(gzip.decompress(path.read_bytes()))
        if not document.get("showdown_models"):
            continue
        if snapshot_id is not None and document.get("snapshot_id") != snapshot_id:
            continue
        captured = pd.Timestamp(document["captured_utc"])
        if captured.tzinfo is None:
            raise ValueError(f"Snapshot {path.name} has no capture timezone")
        matches.append((captured, path.name, document))
    if not matches:
        suffix = f" with id {snapshot_id}" if snapshot_id else ""
        raise ValueError(f"No archived Showdown snapshot found{suffix}")
    if snapshot_id is not None and len(matches) != 1:
        raise ValueError(f"Snapshot id {snapshot_id} is not unique")
    return max(matches, key=lambda item: (item[0], item[1]))[2]


def snapshot_date_range(snapshot: dict) -> tuple[date, date]:
    """Return the half-open UTC date range covering every archived game."""
    kickoffs = [pd.Timestamp(model["kickoff_utc"]) for model in snapshot["showdown_models"]]
    if not kickoffs or any(value.tzinfo is None for value in kickoffs):
        raise ValueError("Every archived Showdown model needs a timezone-aware kickoff")
    first = min(value.tz_convert("UTC").date() for value in kickoffs)
    last = max(value.tz_convert("UTC").date() for value in kickoffs)
    return first, last + pd.Timedelta(days=1)


def _position(raw: dict) -> str:
    return str(raw.get("primaryPosition") or "").upper().replace("D/ST", "DEF")


def completed_player_key(raw: dict) -> str | None:
    """Return the same stable key used by the immutable projection archive."""
    code = str(raw.get("code") or "")
    match = re.search(r"nfl\.p\.(\d+)", code)
    if match:
        return f"yahoo:{match.group(1)}"
    if _position(raw) == "DEF":
        # The archive deliberately preserves the page's team code (including
        # JAC), so do not apply the historical schedule aliases used elsewhere.
        team = str((raw.get("team") or {}).get("abbr") or "").upper()
        if team:
            return f"{team}:{nb.normalize_person_name(_player_name(raw))}"
    return None


def settled_actuals(payload: dict) -> tuple[dict[str, float], dict[str, float]]:
    """Extract settled points and salaries without treating missing points as zero."""
    points, salaries = {}, {}
    for raw in payload.get("players", {}).get("result") or []:
        key = completed_player_key(raw)
        actual = pd.to_numeric(raw.get("points"), errors="coerce")
        salary = pd.to_numeric(raw.get("salary"), errors="coerce")
        if key is None or not np.isfinite(actual) or not np.isfinite(salary):
            continue
        if key in points:
            raise ValueError(f"Duplicate completed player identity: {key}")
        points[key] = float(actual)
        salaries[key] = float(salary)
    return points, salaries


def model_player_keys(snapshot: dict, model: dict) -> list[str]:
    """Bridge model rows to archived stable identities using exact saved fields."""
    game_id = str(model["game_id"])
    predictions = [
        row for row in snapshot.get("predictions", [])
        if str(row.get("game_id")) == game_id
    ]
    keys = []
    for player in model["players"]:
        matches = [row for row in predictions if
            row.get("player") == player.get("name") and
            row.get("team") == player.get("team") and
            row.get("position") == player.get("pos") and
            math.isclose(float(row.get("salary")), float(player.get("salary")), abs_tol=1e-9)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one archived identity for {game_id} {player.get('name')}; "
                f"found {len(matches)}"
            )
        keys.append(str(matches[0]["player_key"]))
    return keys


def select_completed_series(model: dict, candidates: list[dict], yahoo: YahooDfsClient,
                            required_keys: list[str]) -> tuple[dict, dict, dict[str, float]]:
    """Choose the completed series with the strongest exact identity/salary match."""
    evaluated = []
    for series in candidates:
        if str(series.get("game_code")) != str(model["game_id"]):
            continue
        payload = yahoo.players(series["series_id"])
        points, salaries = settled_actuals(payload)
        matched = [key for key in required_keys if key in points]
        salary_matches = 0
        for key, player in zip(required_keys, model["players"]):
            if key in salaries and math.isclose(
                salaries[key], float(player["salary"]), abs_tol=1e-9
            ):
                salary_matches += 1
        evaluated.append((len(matched), salary_matches, -int(series["series_id"]), series, payload, points))
    if not evaluated:
        raise ValueError("No completed Yahoo single-game series found")
    _, _, _, series, payload, points = max(evaluated, key=lambda row: row[:3])
    missing = [key for key in required_keys if key not in points]
    if missing:
        raise ValueError(f"Completed Yahoo series is missing {len(missing)} archived players")
    if series.get("salary_cap") is None:
        raise ValueError("Yahoo did not preserve this completed series' salary cap")
    _, salaries = settled_actuals(payload)
    salary_mismatches = [
        key for key, player in zip(required_keys, model["players"])
        if key not in salaries or not math.isclose(
            salaries[key], float(player["salary"]), abs_tol=1e-9
        )
    ]
    if salary_mismatches:
        raise ValueError(
            f"Completed Yahoo series has {len(salary_mismatches)} archived salary mismatches"
        )
    return series, payload, points


def reconstruct_portfolio(model: dict, salary_cap: float, simulations=None,
                          node=None) -> dict:
    """Run the saved browser model with only the immutable cap supplied later."""
    executable = node or shutil.which("node")
    if not executable:
        raise RuntimeError("Node is required to reconstruct an archived portfolio")
    request = {"payload": model, "salary_cap": float(salary_cap)}
    if simulations is not None:
        request["simulations"] = int(simulations)
    result = subprocess.run(
        [executable, str(RECONSTRUCTOR)], input=json.dumps(request),
        capture_output=True, text=True, timeout=1200,
    )
    if result.returncode:
        raise RuntimeError(f"Archived Showdown reconstruction failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def grade_portfolio(model: dict, portfolio: list[dict], player_keys: list[str],
                    actuals: dict[str, float]) -> list[dict]:
    """Attach settled Yahoo scores to a saved or reconstructed portfolio."""
    graded = []
    for number, lineup in enumerate(portfolio, 1):
        ids = [int(value) for value in lineup["ids"]]
        superstar = int(lineup["superstar"])
        keys = [player_keys[index] for index in ids]
        missing = [key for key in keys if key not in actuals]
        if player_keys[superstar] not in actuals:
            missing.append(player_keys[superstar])
        if missing:
            raise ValueError(f"Lineup has unmatched settled identities: {sorted(set(missing))}")
        actual = sum(actuals[key] for key in keys) + 0.5 * actuals[player_keys[superstar]]
        graded.append({
            "entry": number,
            "players": [model["players"][index]["name"] for index in ids],
            "player_keys": keys,
            "superstar": model["players"][superstar]["name"],
            "superstar_key": player_keys[superstar],
            "salary": float(lineup["salary"]),
            "expected_fp": float(lineup["expected_fp"]),
            "actual_fp": float(actual),
        })
    return graded


def summarize_group(games: list[dict], status: str) -> dict:
    selected = [game for game in games if game["selection_status"] == status]
    first = [game["portfolio"][0]["actual_fp"] for game in selected if game["portfolio"]]
    best = [max(row["actual_fp"] for row in game["portfolio"])
            for game in selected if game["portfolio"]]
    regrets = [game["regret_to_best_submitted"] for game in selected
               if game.get("regret_to_best_submitted") is not None]
    return {
        "games": len(selected),
        "entries_scored": sum(len(game["portfolio"]) for game in selected),
        "mean_first_entry_actual_fp": float(np.mean(first)) if first else None,
        "mean_portfolio_best_actual_fp": float(np.mean(best)) if best else None,
        "mean_regret_to_best_submitted": float(np.mean(regrets)) if regrets else None,
    }


def run(snapshot: dict, yahoo: YahooDfsClient, completed_series: list[dict],
        reconstruct_simulations=None, fetch_benchmark=True,
        nflverse: NflverseActuals | None = None) -> dict:
    """Grade all completed games in one archived slate snapshot."""
    by_game = {}
    for series in completed_series:
        by_game.setdefault(str(series["game_code"]), []).append(series)
    games, skipped, actual_rows = [], [], []
    observed_utc = datetime.now(timezone.utc).isoformat()
    for model in snapshot["showdown_models"]:
        game_id = str(model["game_id"])
        try:
            player_keys = model_player_keys(snapshot, model)
            series, actual_source, actual_audit = None, None, None
            yahoo_error = None
            try:
                series, _, actuals = select_completed_series(
                    model, by_game.get(game_id, []), yahoo, player_keys
                )
                actual_source = "yahoo_settled"
                if model.get("salary_cap") is not None and not math.isclose(
                    float(model["salary_cap"]), float(series["salary_cap"]), abs_tol=1e-9
                ):
                    raise ValueError("Completed salary cap disagrees with the pregame archive")
            except Exception as exc:
                yahoo_error = str(exc)
                if nflverse is None:
                    raise
                actuals, actual_audit = nflverse.actuals(model, player_keys)
                actual_source = "nflverse_reconstructed"
                if not actuals:
                    reason = actual_audit.get("reason", "no nflverse actuals")
                    raise ValueError(f"{yahoo_error}; nflverse fallback: {reason}") from exc
            actual_rows.extend({
                "game_id": game_id,
                "player_key": key,
                "actual_fp": actuals[key],
                "realized_at_utc": observed_utc,
            } for key in player_keys if key in actuals)
            reference = model.get("reference")
            if reference and reference.get("portfolio"):
                status = "published_forward"
                portfolio = reference["portfolio"]
                reconstruction = None
            else:
                if series is None:
                    raise ValueError(
                        "nflverse actuals are available, but this game had no pregame "
                        "portfolio and nflverse cannot supply the missing Yahoo salary cap"
                    )
                status = "reconstructed_from_pregame_snapshot"
                reconstruction = reconstruct_portfolio(
                    model, float(series["salary_cap"]), reconstruct_simulations
                )
                portfolio = reconstruction["portfolio"]
            graded = grade_portfolio(model, portfolio, player_keys, actuals)
            benchmark_score = None
            if fetch_benchmark and series is not None:
                raw = yahoo.benchmark(series["series_id"])
                benchmark_score = (raw.get("allUserBestLineup", {}).get("result") or {}).get("score")
            realized = max(row["actual_fp"] for row in graded)
            salary_cap = (
                float(series["salary_cap"]) if series is not None
                else float(model["salary_cap"])
            )
            games.append({
                "game_id": game_id,
                "series_id": int(series["series_id"]) if series is not None else None,
                "matchup": model["matchup"],
                "kickoff_utc": model["kickoff_utc"],
                "selection_status": status,
                "snapshot_id": snapshot["snapshot_id"],
                "salary_cap": salary_cap,
                "salary_cap_source": (
                    "pregame_archive" if model.get("salary_cap") is not None
                    else "completed_yahoo_series"
                ),
                "actual_source": actual_source,
                "actual_audit": actual_audit,
                "yahoo_fallback_reason": yahoo_error if series is None else None,
                "best_submitted_score": benchmark_score,
                "regret_to_best_submitted": (
                    float(benchmark_score) - realized if benchmark_score is not None else None
                ),
                "reconstruction": ({
                    "simulations": reconstruction["simulations"],
                    "objective": reconstruction["objective"],
                    "matches_archived_simulation_count": (
                        int(reconstruction["simulations"]) == int(model["settings"]["simulations"])
                    ),
                } if reconstruction else None),
                "portfolio": graded,
            })
        except Exception as exc:
            skipped.append({"game_id": game_id, "matchup": model.get("matchup"), "reason": str(exc)})
    evaluation = projection_archive.evaluate(
        [snapshot], pd.DataFrame(actual_rows, columns=(
            "game_id", "player_key", "actual_fp", "realized_at_utc"
        ))
    ) if actual_rows else {"overall": {"n": 0}}
    return {
        "schema": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": snapshot["snapshot_id"],
        "captured_utc": snapshot["captured_utc"],
        "games_requested": len(snapshot["showdown_models"]),
        "games_completed": len(games),
        "games_skipped": len(skipped),
        "settled_player_actuals": len(actual_rows),
        "projection_evaluation": evaluation,
        "published_forward": summarize_group(games, "published_forward"),
        "reconstructed": summarize_group(games, "reconstructed_from_pregame_snapshot"),
        "limitations": [
            "Published-forward and postgame-reconstructed portfolios are reported separately.",
            "Reconstruction uses the immutable pregame player model and settings; only the completed series salary cap is added later.",
            "Best submitted score is not a complete-field ROI or payout backtest.",
            "Missing settled player identities are never converted to zero.",
            "NFLverse actuals are a reconstructed fallback; Yahoo settled points remain authoritative.",
            "NFLverse weekly-stat absence is not treated as a played zero.",
        ],
        "games": games,
        "skipped": skipped,
    }
