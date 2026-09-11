from __future__ import annotations

"""Yahoo NFL weekly top-25 position rankings — pipeline.

GENERATED FROM the Colab notebook `Yahoo_Top25_Position_Rankings_v1.ipynb`.

The notebook executed every cell in one namespace, and its functions reach
across cell boundaries for both public helpers and underscore-prefixed ones.
This file therefore keeps all of that code in a single module in the original
order rather than splitting it: a package split would need an explicit export
list per cell and would break silently the first time a private helper moved.

Each notebook cell is marked with a banner below. Edit the configuration
section (cell 1) to change settings; `run_daily.py` is the entry point.
"""

# ============================================================================
# NOTEBOOK CELL 1 - Configuration, settings, and manual overrides
# ============================================================================
# Core imports — no PuLP or scikit-learn required.

import itertools
import math
import json
import re
import shutil
import time
import warnings
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.portfolio_construction import exposure_limit
from pipeline import salary_projection
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

try:
    from IPython.display import display
except ImportError:
    display = print

YAHOO_API_ENDPOINT = "https://dfyql-ro.sports.yahoo.com/v2/external/playersFeed/nfl"
VALID_POSITIONS = ("QB", "RB", "WR", "TE", "DEF")


@dataclass(frozen=True)
class Settings:
    lineup_size: int = 5

    # v3.2: raised from 5,000. At 5,000 the tail metrics were badly under-sampled:
    # `Win_Rate` had a cross-seed rank correlation of only 0.21 and the 20-lineup
    # portfolio changed by half its members when the seed changed. The vectorized
    # enumerator pays for the extra scenarios; total runtime is still lower than v3.1.
    simulations: int = 20_000
    random_seed: int = 356
    tournament_lineups: int = 20
    max_candidate_lineups: int = 25_000
    mean_candidate_reserve: int = 750
    max_enumeration_players: int = 36

    # This is a strategy filter, not a Yahoo rule, so v3.6 defaults it off. At the
    # previous 0.75 it removed 161,439 of the 187,697 cap-legal rosters on the
    # 2026-09-10 SF-LA slate -- 86% of the legal space -- before the model scored
    # one of them. Raise it to refuse lineups that leave salary unused.
    min_salary_used_pct: float = 0.0
    candidate_ceiling_weight: float = 0.85
    near_optimal_ratio: float = 0.95

    # Portfolio controls. On a five-player roster `max_shared_players = 4` permits
    # entries that differ by a single player; the v3.1 default produced 19 such pairs
    # out of 190 on the NE-SEA slate. Lowered to 3 so every pair of entries differs by
    # at least two players. The diversity report prints the overlap actually used.
    max_player_exposure: float = 0.70
    max_superstar_exposure: float = 0.35
    max_shared_players: int = 3
    use_construction_quotas: bool = False
    showdown_objective: str = "auto"

    # Historical backup-QB projections were badly biased. Keep them out unless
    # the user explicitly confirms a package or replacement-starter role.
    exclude_backup_qbs: bool = True

    # Optional user strategy limits. Yahoo itself does not require position limits.
    position_limits: dict | None = None

    # Print the split-half Monte Carlo reliability table with the run.
    report_reliability: bool = True

    # --- nflverse role and availability feed (v3.3) ---------------------------------
    # Salary order is a weak proxy for a depth chart and says nothing at all about
    # who is on injured reserve. These pull the published depth chart and weekly
    # roster status from the nflverse data releases. Any failure degrades to the
    # salary heuristic with a warning; the run never dies on a network problem.
    use_nflverse: bool = True
    nflverse_apply_depth: bool = True
    nflverse_availability_filter: bool = True

    # Statuses treated as available. ACT is the active roster; DEV is the practice
    # squad, INA declared inactive for the game, RES injured reserve, CUT released.
    # Add "DEV" if you deliberately want practice-squad elevation candidates in the
    # pool, or name the individual in AVAILABILITY_OVERRIDES once his game-day
    # elevation is confirmed.
    nflverse_available_status: tuple = ("ACT",)

    # v3.5: a player the weekly roster has no row for is now dropped. Keeping him
    # was internally inconsistent - the run would print "allowed status: ACT" and
    # then rate an unknown-status player as available - and it is the failure mode
    # that eventually puts an ineligible player in a submitted lineup. Name anyone
    # you know is playing in AVAILABILITY_OVERRIDES.
    nflverse_drop_unmatched: bool = True

    # Refuse to run the availability filter at all if the roster feed matched less
    # of the pool than this. Below it the far likelier explanation is a join or
    # schema problem, and dropping most of a slate on that basis is worse than
    # keeping it.
    nflverse_min_match_rate: float = 0.75

    # Weight on Yahoo salary order when blending the published chart with a
    # workload proxy. Salary is available for every slate and is the only live
    # expectation input to the fitted projection model.
    role_salary_rank_weight: float = 0.5

    nflverse_season: int | None = None  # None infers the season from the slate
    nflverse_timeout: int = 30
    nflverse_cache_dir: str = "nflverse_cache"


CFG = Settings()


def _cfg(cfg=None):
    """Resolve the live CFG at call time.

    v3.1 bound `cfg=CFG` as a def-time default. Editing this cell and re-running it
    without also re-running every function cell left the pipeline silently using the
    previous settings object. Passing `cfg=None` now reads the current global.
    """
    return CFG if cfg is None else cfg


# Exact fantasy-point overrides. Example: {"Player Name": 17.8}
PROJECTION_OVERRIDES = {}

# Team-position depth overrides. Example: {"Player Name": 1}
# Use this for a confirmed replacement starter. Depth drives BOTH the calibrated CV
# and the historical mean multiplier, so a promoted QB2 left at depth 2 keeps the
# 0.37x haircut even after you add the name to INCLUDE_BACKUP_QBS.
DEPTH_OVERRIDES = {}

# Supported styles: "rushing_qb", "pass_catching_rb", "committee_rb".
PLAYER_STYLE_OVERRIDES = {}

# Players to remove before optimization. Exact Yahoo names.
EXCLUDE_PLAYERS = set()

# Availability the roster feed cannot know about, e.g. {"Player Name": True} for a
# confirmed game-day practice-squad elevation, or False for a late scratch the
# weekly roster still lists as active. These win over the nflverse status.
AVAILABILITY_OVERRIDES = {}

# Exact backup-QB names to retain despite the default role filter.
# Pair each name with DEPTH_OVERRIDES or PROJECTION_OVERRIDES; the run warns if you do not.
INCLUDE_BACKUP_QBS = set()

# Optional strategic limits, e.g. {"QB": (0, 2), "DEF": (0, 2)}.
# Leave empty to follow Yahoo's position-flexible single-game construction.
POSITION_LIMITS = {}
CFG = replace(CFG, position_limits=POSITION_LIMITS)


# ============================================================================
# NOTEBOOK CELL 3 - Yahoo player feed: fetch, normalize, priors, depth assumptions
# ============================================================================
def fetch_yahoo_data(endpoint=YAHOO_API_ENDPOINT, timeout=15, attempts=3):
    """Fetch the Yahoo player feed with bounded retries and clear errors."""
    last_error = None
    headers = {"User-Agent": "Mozilla/5.0 Yahoo-Showdown-Lineup-Lab/3.2"}
    for attempt in range(1, attempts + 1):
        try:
            request = Request(endpoint, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
            if not payload.get("players", {}).get("result"):
                raise ValueError("Yahoo response did not contain a player list.")
            return payload
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(0.75 * attempt)
    raise RuntimeError(f"Yahoo feed failed after {attempts} attempts: {last_error}")


def _repair_game_assignments(df):
    """Move rows filed under a game their team is not playing in, or drop them.

    Returns the frame with Game ID / Game Time / Home Team / Away Team corrected for
    any row whose team appears in exactly one game on the slate. See the caller for
    why this happens and why the team field is the half worth trusting.
    """
    misfiled = ~(df["Team"].eq(df["Home Team"]) | df["Team"].eq(df["Away Team"]))
    if not misfiled.any():
        return df

    columns = ["Game ID", "Game Time", "Home Team", "Away Team"]
    schedule = df.loc[~misfiled, columns].drop_duplicates("Game ID")
    lookup = pd.concat([
        schedule.assign(_team=schedule["Home Team"]),
        schedule.assign(_team=schedule["Away Team"]),
    ], ignore_index=True)
    # Only trust a team that plays exactly one game on this slate.
    counts = lookup["_team"].value_counts()
    lookup = lookup[lookup["_team"].isin(counts[counts.eq(1)].index)].set_index("_team")

    out = df.copy()
    resolvable = misfiled & out["Team"].isin(lookup.index)
    unresolved = misfiled & ~out["Team"].isin(lookup.index)

    if resolvable.any():
        targets = lookup.loc[out.loc[resolvable, "Team"]]
        moved = [
            f"{name} ({team}: {old_away}@{old_home} -> {new_away}@{new_home})"
            for name, team, old_away, old_home, new_away, new_home in zip(
                out.loc[resolvable, "Name"], out.loc[resolvable, "Team"],
                out.loc[resolvable, "Away Team"], out.loc[resolvable, "Home Team"],
                targets["Away Team"], targets["Home Team"],
            )
        ]
        out.loc[resolvable, columns] = targets[columns].to_numpy()
        warnings.warn(
            f"Reassigned {len(moved)} player(s) filed under a game their team is not "
            "playing in, using the team field and the slate schedule: "
            + ", ".join(sorted(moved))
        )

    if unresolved.any():
        names = [
            f"{name} ({team})"
            for name, team in zip(out.loc[unresolved, "Name"], out.loc[unresolved, "Team"])
        ]
        out = out.loc[~unresolved]
        warnings.warn(
            f"Dropped {len(names)} player(s) whose team plays no unambiguous game on "
            "this slate: " + ", ".join(sorted(names))
        )
    return out.reset_index(drop=True)


def normalize_yahoo_data(payload):
    players = payload["players"]["result"]
    df = pd.DataFrame(players).copy()
    rename = {
        "name": "Name",
        "position": "Position",
        "team": "Team",
        "salary": "Salary",
        "fppg": "FPPG",
        "gameCode": "Game ID",
        "gameStartTime": "Game Time",
        "homeTeam": "Home Team",
        "awayTeam": "Away Team",
    }
    df = df.rename(columns=rename)
    required = [
        "Name", "Position", "Team", "Salary", "FPPG", "Game ID",
        "Game Time", "Home Team", "Away Team",
    ]
    missing = [column for column in required if column not in df]
    if missing:
        raise ValueError(f"Yahoo schema changed; missing columns: {missing}")

    df["Position"] = (
        df["Position"].astype(str).str.upper().replace({"D/ST": "DEF", "DST": "DEF"})
    )
    df["Salary"] = pd.to_numeric(df["Salary"], errors="coerce")
    df["FPPG"] = pd.to_numeric(df["FPPG"], errors="coerce").fillna(0.0)
    df["Game ID"] = df["Game ID"].astype(str)
    # v3.2 keeps Yahoo's own player id ("nfl.p.26753" -> 26753). Name matching against
    # any external depth-chart or injury feed is lossy; this is the stable key. Team
    # defenses carry a team code ("nfl.t.25") instead, so parse leniently.
    if "playerCode" in df:
        df["Yahoo ID"] = pd.to_numeric(
            df["playerCode"].astype(str).str.extract(r"nfl\.p\.(\d+)", expand=False),
            errors="coerce",
        ).astype("Int64")

    # v3.2: the live feed carries rows whose `gameCode` points at a game their team is
    # not playing in - for example Corey Kiner, correctly listed as NE, filed under
    # ARI@LAC. v3.1 gave them a wrong Opponent and, worse, they registered as a third
    # team, so `build_correlation_model` raised "Expected exactly two teams" and the
    # whole run died. On the 2026 week-one feed this killed 6 of 15 games.
    #
    # The team field is the reliable half: nflverse's depth chart confirms Kiner as a
    # New England running back. Since each team plays exactly one game on a slate, the
    # row can be moved to its team's real game rather than thrown away. Only a row
    # whose team is absent or ambiguous is dropped.
    df = _repair_game_assignments(df)
    df["Opponent"] = np.where(
        df["Team"].eq(df["Away Team"]), df["Home Team"], df["Away Team"]
    )

    df = df[
        df["Position"].isin(VALID_POSITIONS)
        & df["Salary"].notna()
        & df["Salary"].gt(0)
    ].copy()
    df = df.drop_duplicates(["Game ID", "Team", "Name", "Position"]).reset_index(drop=True)

    raw_caps = payload.get("salaryCapInfo", {}).get("result", [{}])
    cap_map = raw_caps[0].get("singleGameSalaryCapMap", {}) if raw_caps else {}
    cap_map = {str(key): float(value) for key, value in cap_map.items()}
    return df, cap_map


def _warn_unmatched(names, label, available):
    """Report override names that matched nothing, with the closest feed spellings.

    v3.1 warned on a miss but gave no hint, and a silently ignored projection
    override is the single most damaging user error in this notebook.
    """
    import difflib

    for name in names:
        if name in available:
            continue
        near = difflib.get_close_matches(str(name), list(available), n=3, cutoff=0.6)
        hint = f" Closest feed names: {', '.join(near)}." if near else ""
        warnings.warn(f"{label} did not match any Yahoo name: {name}.{hint}")


def add_projection_priors(df, projection_overrides=None):
    """Apply the season-frozen Yahoo salary, position and depth regression."""
    out = df.copy()
    if "Depth_Rank" not in out:
        order = out.sort_values(
            ["Team", "Position", "Salary", "Name"],
            ascending=[True, True, False, True],
        )
        out["Depth_Rank"] = (
            order.groupby(["Team", "Position"]).cumcount().add(1).reindex(out.index).astype(int)
        )
    return salary_projection.apply(out, overrides=projection_overrides)


def list_games(df):
    games = (
        df[["Game ID", "Game Time", "Away Team", "Home Team"]]
        .drop_duplicates("Game ID")
        .copy()
    )
    games["_sort"] = pd.to_datetime(games["Game Time"], errors="coerce", utc=True)
    games = games.sort_values(["_sort", "Game ID"], na_position="last").drop(columns="_sort")
    games["Matchup"] = games["Away Team"] + " vs " + games["Home Team"]
    return games.reset_index(drop=True)


def select_game_interactive(games):
    print("Available games:")
    for number, row in games.iterrows():
        print(f"{number + 1}. {row['Game Time']}: {row['Matchup']}")
    while True:
        try:
            choice = int(input("Select game number: ")) - 1
            if 0 <= choice < len(games):
                return games.iloc[choice]
        except ValueError:
            pass
        print("Enter one of the listed game numbers.")


def salary_cap_for_game(cap_map, game_id):
    game_id = str(game_id)
    if game_id in cap_map:
        return float(cap_map[game_id])
    while True:
        try:
            value = float(input("Yahoo cap was unavailable. Enter the single-game salary cap: $"))
            if value > 0:
                return value
        except ValueError:
            pass
        print("Enter a positive number.")


def assign_depth_assumptions(players, depth_overrides=None, style_overrides=None):
    """Assign relative team-position ranks without claiming active status.

    The Yahoo feed does not provide a dependable live depth chart. Ranking by the
    unadjusted projection gives a reproducible fallback and avoids circularly
    re-ranking players after their depth haircut. A manual override should be used
    for injury replacements, newly promoted starters, and specialty packages.

    v3.2 breaks projection ties on salary then name instead of on feed row order, so
    two equally projected bench players always receive the same depth ranks.
    """
    depth_overrides = depth_overrides or {}
    style_overrides = style_overrides or {}
    out = players.copy()
    order = out.sort_values(
        ["Team", "Position", "Projected_FP", "Salary", "Name"],
        ascending=[True, True, False, False, True],
    )
    ranks = (order.groupby(["Team", "Position"]).cumcount() + 1).reindex(out.index)
    out["Depth_Rank"] = ranks.astype(int)
    out["Depth_Source"] = "projection heuristic"
    out["Player_Style"] = "standard"

    _warn_unmatched(depth_overrides, "Depth override", set(out["Name"]))
    for name, depth in depth_overrides.items():
        mask = out["Name"].eq(name)
        if mask.any():
            out.loc[mask, "Depth_Rank"] = max(1, int(depth))
            out.loc[mask, "Depth_Source"] = "manual override"

    allowed_styles = {"standard", "rushing_qb", "pass_catching_rb", "committee_rb"}
    _warn_unmatched(style_overrides, "Style override", set(out["Name"]))
    for name, style in style_overrides.items():
        if style not in allowed_styles:
            raise ValueError(f"Unsupported style '{style}' for {name}")
        mask = out["Name"].eq(name)
        if mask.any():
            out.loc[mask, "Player_Style"] = style
    return out


# Actual / rolling-pregame expectation by position and lagged-snap depth.
# Rank-one values are held at 1.0 because the small observed differences were not
# practically important. QB3 had only 23 games and is not promoted as a parameter.
DEPTH_MEAN_MULTIPLIER = {
    "QB": {1: 1.00, 2: 0.37, 3: 0.37, 4: 0.37},
    "RB": {1: 1.00, 2: 0.94, 3: 0.75, 4: 0.70},
    "WR": {1: 1.00, 2: 0.98, 3: 0.98, 4: 0.77},
    "TE": {1: 1.00, 2: 0.94, 3: 0.75, 4: 0.61},
    "DEF": {1: 1.00, 2: 1.00, 3: 1.00, 4: 1.00},
}


def _depth_bucket(depth):
    """Map all fourth-or-deeper roles to the calibrated 4+ bucket."""
    return min(max(int(depth), 1), 4)


def apply_depth_mean_adjustments(players):
    """Compatibility shim: depth is already fitted inside the regression."""
    out = players.copy()
    out["Pre_Depth_Projected_FP"] = out["Projected_FP"].astype(float)
    out["Depth_Mean_Multiplier"] = 1.0
    out["Role_Adjusted_Baseline_FP"] = out["Projected_FP"].astype(float)
    out["Projection_Adjustment"] = "depth included in salary regression"
    return out


def effective_role_tier(players):
    """Role tier where the chart supplied one, otherwise the opportunity rank."""
    if "Role_Tier" not in players:
        return players["Depth_Rank"].astype(float)
    return pd.to_numeric(players["Role_Tier"], errors="coerce").fillna(
        players["Depth_Rank"]
    )


def apply_default_role_filters(players, include_backup_qbs=None, cfg=None):
    """Remove unconfirmed backup QBs while allowing explicit named exceptions.

    QB2 averaged 3.93 points against a 10.52-point pregame expectation in the
    historical fit. That is primarily a participation problem, not useful upside.
    Default exclusion prevents a low-salary backup from entering a lineup solely
    because a high fitted CV produces a long simulated tail.

    The test is the role tier, not the opportunity rank: quarterback has a single
    alignment slot, so tier 1 is the starter and nothing else is. A DEPTH_OVERRIDES
    entry sets both, which is how a confirmed replacement starter gets through.
    """
    cfg = _cfg(cfg)
    include_backup_qbs = set(include_backup_qbs or set())
    out = players.copy()
    tier = effective_role_tier(out)

    # v3.2: keeping a backup QB without also promoting its role leaves the 0.37x
    # historical multiplier in place, silently deleting ~63% of its projection. That
    # made INCLUDE_BACKUP_QBS look broken rather than misconfigured.
    demoted = out[
        out["Name"].isin(include_backup_qbs)
        & out["Position"].eq("QB")
        & tier.gt(1)
        & ~out["Projection_Source"].eq("manual override")
    ]["Name"].tolist()
    if demoted:
        warnings.warn(
            "INCLUDE_BACKUP_QBS retained " + ", ".join(demoted)
            + " but the depth chart still lists them behind QB1, so the 0.37x "
            "historical QB2 mean multiplier is still applied. Add a DEPTH_OVERRIDES "
            "entry of 1, or a PROJECTION_OVERRIDES value, if you expect them to start."
        )

    if not cfg.exclude_backup_qbs:
        return out.reset_index(drop=True), []
    remove = (
        out["Position"].eq("QB")
        & tier.gt(1)
        & ~out["Name"].isin(include_backup_qbs)
    )
    removed = out.loc[remove, "Name"].tolist()
    return out.loc[~remove].reset_index(drop=True), removed


def depth_sanity_report(players):
    """Present every role assumption and its direct projection consequence."""
    rows = []
    for _, player in players.sort_values(["Team", "Position", "Depth_Rank"]).iterrows():
        flags = []
        source = str(player["Depth_Source"])
        if source == "projection heuristic":
            flags.append("no chart entry; role inferred from the projection")
        elif source != "manual override":
            flags.append("chart role")
        if (
            player["FPPG"] <= 0.25
            and player["Projection_Source"] != "manual override"
        ):
            flags.append("zero/low FPPG prior")
        if player["Position"] == "QB" and player["Depth_Rank"] > 1:
            flags.append("verify expected snaps")
        rows.append({
            "Team": player["Team"],
            "Position": player["Position"],
            "Role": player.get("Role_Label", "unknown"),
            "Slot": player.get("Role_Slot") or "-",
            "Opportunity rank": int(player["Depth_Rank"]),
            "Player": player["Name"],
            "Salary": float(player["Salary"]),
            "Raw projection": round(float(player["Pre_Depth_Projected_FP"]), 2),
            "Mean factor": round(float(player["Depth_Mean_Multiplier"]), 2),
            "Adjusted projection": round(float(player["Projected_FP"]), 2),
            "Role source": player["Depth_Source"],
            "Review": "; ".join(flags) or "ok",
        })
    return pd.DataFrame(rows)


def apply_exclusions_interactive(players, preexcluded=None):
    """Display stable row numbers and remove explicitly selected players."""
    preexcluded = set(preexcluded or set())
    _warn_unmatched(preexcluded, "Exclusion", set(players["Name"]))
    out = players[~players["Name"].isin(preexcluded)].copy()
    view = out.sort_values(["Team", "Position", "Depth_Rank", "Salary"], ascending=[True, True, True, False])
    view = view.reset_index(drop=True)
    view.insert(0, "Row", np.arange(len(view)))
    display(view[[
        "Row", "Name", "Position", "Team", "Salary", "FPPG",
        "Pre_Depth_Projected_FP", "Depth_Mean_Multiplier", "Projected_FP",
        "Role_Label", "Depth_Rank", "Projection_Source",
    ]].rename(columns={
        "Pre_Depth_Projected_FP": "Raw projection",
        "Depth_Mean_Multiplier": "Mean factor",
        "Projected_FP": "Adjusted projection",
        "Role_Label": "Role",
        "Depth_Rank": "Opportunity rank",
    }))
    answer = input("Exclude more players? Enter comma-separated Row values, or press Enter: ").strip()
    if not answer:
        return out.reset_index(drop=True)
    try:
        numbers = [int(piece.strip()) for piece in answer.split(",")]
        bad = [number for number in numbers if number < 0 or number >= len(view)]
        if bad:
            raise ValueError(f"row numbers out of range: {bad}")
        names = set(view.iloc[numbers]["Name"])
        return out[~out["Name"].isin(names)].reset_index(drop=True)
    except ValueError as exc:
        raise ValueError(f"Invalid exclusion list: {exc}") from exc


def roster_feasibility_error(players, lineup_size):
    """Return why this pool cannot build a Yahoo-valid roster, or None if it can.

    Yahoo requires five players and at least one non-defense player from each
    team. Checking it up front turns a confusing "no valid rosters" failure
    deep inside enumeration into a message naming the empty team.
    """
    if len(players) < lineup_size:
        return f"Only {len(players)} players remain; {lineup_size} are required."
    teams = sorted(players["Team"].unique())
    if len(teams) != 2:
        return f"Expected exactly two teams, found {teams}."
    for team in teams:
        skill = players[players["Team"].eq(team) & players["Position"].ne("DEF")]
        if skill.empty:
            return (
                f"{team} has no non-DEF player left. Yahoo requires at least one "
                "non-DEF player from each team, so no valid lineup exists."
            )
    return None


def trim_player_pool(players, cfg=None):
    """Fill the configured pool using projection, value, and balanced backfill.

    The first implementation concatenated 26 projection leaders and 10 value leaders.
    When those lists overlapped heavily, deduplication could leave only 26 players
    even though `max_enumeration_players` was 36. The missing slots are now filled by
    a 70/30 projection/value percentile score.

    v3.2 also protects pool quality. A pool whose leaders all belonged to one team
    could strand the other team with only its DEF. That is legal under Yahoo's
    one-player-per-team rule, but a slate reduced to somebody's defense is not a
    pool worth optimizing, so each team's best non-DEF player is reserved before
    ranking.
    """
    cfg = _cfg(cfg)
    if len(players) <= cfg.max_enumeration_players:
        return players.reset_index(drop=True), []
    work = players.copy()
    work["Value"] = work["Projected_FP"] / work["Salary"]
    target = max(1, int(cfg.max_enumeration_players))

    reserved = []
    if {"Team", "Position"}.issubset(work.columns):
        for team in work["Team"].drop_duplicates():
            skill = work[work["Team"].eq(team) & ~work["Position"].eq("DEF")]
            if len(skill):
                reserved.append(skill["Projected_FP"].idxmax())
    reserved = list(dict.fromkeys(reserved))[:target]

    value_count = min(10, max(target - 1, 0))
    anchor_count = target - value_count
    keep = list(reserved)
    keep += [i for i in work.nlargest(anchor_count, "Projected_FP").index if i not in keep]
    keep += [i for i in work.nlargest(value_count, "Value").index if i not in keep]
    keep = list(dict.fromkeys(keep))

    if len(keep) < target:
        # Percentile ranks are scale-free, so a point projection and a
        # points-per-dollar value can be combined without arbitrary units.
        work["Projection_Percentile"] = work["Projected_FP"].rank(pct=True)
        work["Value_Percentile"] = work["Value"].rank(pct=True)
        work["Pool_Priority"] = (
            0.70 * work["Projection_Percentile"]
            + 0.30 * work["Value_Percentile"]
        )
        needed = target - len(keep)
        backfill = (
            work.loc[~work.index.isin(keep)]
            .sort_values(["Pool_Priority", "Projected_FP", "Value"], ascending=False)
            .head(needed)
            .index
            .tolist()
        )
        keep.extend(backfill)
    keep = keep[:target]

    dropped = work.loc[~work.index.isin(keep), "Name"].tolist()
    helper_columns = ["Value", "Projection_Percentile", "Value_Percentile", "Pool_Priority"]
    return (
        work.loc[keep]
        .drop(columns=helper_columns, errors="ignore")
        .reset_index(drop=True),
        dropped,
    )


# ============================================================================
# NOTEBOOK CELL 9 - nflverse depth chart and roster-availability cross-check
# ============================================================================
NFLVERSE_RELEASE_BASE = "https://github.com/nflverse/nflverse-data/releases/download"

# Yahoo and nflverse disagree on exactly one abbreviation on a normal slate. Without
# this alias every Jacksonville player fails to match and the availability filter
# silently switches itself off for that team.
YAHOO_TO_NFLVERSE_TEAM = {"JAC": "JAX"}

# nflverse lists fullbacks separately; Yahoo prices them as running backs.
NFLVERSE_POSITION_ALIASES = {"FB": "RB", "HB": "RB"}


def rerank_aliased_positions(offense):
    """Place fullbacks behind their team's running backs before the ranks are used.

    nflverse ranks fullbacks within fullbacks, so a blocking FB1 naively becomes RB1
    and inherits RB1's 1.00 mean multiplier and low 0.676 CV - strictly worse than the
    salary heuristic it replaced. Offsetting by the deepest running back keeps the
    alias useful without promoting a fullback over the actual starter.
    """
    out = offense.copy()
    is_fullback = out["pos_abb"].eq("FB")
    if not is_fullback.any():
        return out
    deepest = out[out["pos_abb"].eq("RB")].groupby("team")["pos_rank"].max()
    out.loc[is_fullback, "pos_rank"] = (
        out.loc[is_fullback, "team"].map(deepest).fillna(0).astype(int)
        + out.loc[is_fullback, "pos_rank"].astype(int)
    )
    return out

# Roster status codes seen in nflverse weekly rosters. These are the codes the
# 2024 and 2025 weekly assets actually carry; anything else is reported by its
# raw code and treated as unavailable, because an unrecognized status is exactly
# the case where guessing "probably fine" is most expensive. A practice-squad
# player elevated for a game keeps his DEV row, so a confirmed elevation is an
# AVAILABILITY_OVERRIDES entry rather than a status.
NFLVERSE_STATUS_MEANING = {
    "ACT": "active",
    "DEV": "practice squad",
    "INA": "inactive for this game",
    "RES": "reserve / injured reserve",
    "CUT": "released",
    "RET": "retired",
    "EXE": "exempt list",
    "E01": "exempt / commissioner permission",
    "TRC": "reserve / did not report",
    "TRD": "traded",
    "W04": "waived",
}

NO_ROSTER_ROW = "no roster row"

_NAME_SUFFIXES = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
_NAME_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_person_name(name):
    """Collapse a player name to a join key that survives feed spelling differences.

    Yahoo writes "Velus Jones Jr." and "A.J. Brown"; nflverse writes "Velus Jones" and
    "AJ Brown". Punctuation is replaced by a space so generational suffixes can be
    matched on word boundaries, and only then is all whitespace removed - otherwise
    "a.j. brown" collapses to "a j brown" and never meets "aj brown".
    """
    text = _NAME_NON_ALNUM.sub(" ", str(name).lower())
    text = _NAME_SUFFIXES.sub(" ", text)
    return text.replace(" ", "")


def infer_season(game_time, fallback=None):
    """Infer the NFL season year from a slate kickoff timestamp.

    A season is named for the calendar year it starts in, so January and February
    games belong to the previous season.
    """
    stamp = pd.to_datetime(game_time, errors="coerce", utc=True)
    if pd.isna(stamp):
        return fallback
    return int(stamp.year) if stamp.month >= 3 else int(stamp.year) - 1


def _read_nflverse_csv(dataset, filename, cfg):
    """Download one nflverse release asset, caching it for the current UTC day.

    The .csv.gz assets are used rather than .parquet so the notebook keeps working
    without pyarrow. Files are refreshed daily because nflverse republishes the
    depth chart every morning.
    """
    cache = Path(cfg.nflverse_cache_dir)
    stamp = time.strftime("%Y%m%d", time.gmtime())
    target = cache / f"{stamp}_{filename}"
    if not target.exists():
        cache.mkdir(parents=True, exist_ok=True)
        url = f"{NFLVERSE_RELEASE_BASE}/{dataset}/{filename}"
        request = Request(url, headers={"User-Agent": "Mozilla/5.0 Yahoo-Showdown-Lineup-Lab/3.3"})
        with urlopen(request, timeout=cfg.nflverse_timeout) as response:
            target.write_bytes(response.read())
        for stale in cache.glob(f"*_{filename}"):
            if stale != target:
                stale.unlink(missing_ok=True)
    return pd.read_csv(target, low_memory=False)


ROLE_TIER_LABELS = {1: "starter", 2: "rotation", 3: "backup"}
DEEP_ROLE_LABEL = "reserve"
SPECIALIST_LABEL = "specialist"


def add_slot_role_tiers(offense):
    """Derive a role tier per alignment slot instead of one flat position ladder.

    An NFL depth chart is not a single ordered list per position. A team that
    lines up in three-receiver personnel publishes three parallel starting
    receiver spots, and nflverse encodes that in `pos_slot`: on a 2025 Washington
    snapshot the WR rows carry slots 1, 2 and 8, and `pos_rank` walks across the
    slots - McLaurin (slot 1) rank 1, Samuel (slot 2) rank 2, Brown (slot 8) rank
    3, then McCaffrey (slot 1) rank 4 as the *second* man at the first slot.

    Flattening that to WR1 > WR2 > WR3 > WR4 turns three starters into a starter
    and two deep reserves, which then collects a mean haircut meant for players
    who barely take the field. The tier below is the player's rank *within his
    own slot*, so all three of those receivers are tier 1 and McCaffrey is the
    tier-2 man behind McLaurin.
    """
    out = offense.copy()
    if "pos_slot" not in out:
        # Older season schemas have no slot column; the flat rank is all there is.
        out["pos_slot"] = pd.NA
        out["Role_Tier"] = pd.to_numeric(out["pos_rank"], errors="coerce")
        out["Role_Slot"] = None
        return out
    out["pos_slot"] = pd.to_numeric(out["pos_slot"], errors="coerce")
    out["pos_rank"] = pd.to_numeric(out["pos_rank"], errors="coerce")
    ordered = out.sort_values(["team", "pos_abb", "pos_slot", "pos_rank"])
    tiers = (ordered.groupby(["team", "pos_abb", "pos_slot"], dropna=False).cumcount() + 1)
    out["Role_Tier"] = tiers.reindex(out.index)
    # Fall back to the flat rank wherever the slot was missing or unparseable.
    out["Role_Tier"] = out["Role_Tier"].fillna(out["pos_rank"])
    out["Role_Slot"] = np.where(
        out["pos_slot"].notna(),
        out["pos_abb"].astype(str) + " slot " + out["pos_slot"].astype("Int64").astype(str),
        None,
    )
    return out


def role_label(position, tier, chart_position=None):
    """Name a role in words rather than as an ordinal in a flattened list."""
    if str(chart_position) == "FB":
        # A fullback is not the third-best running back; he is a package player.
        return SPECIALIST_LABEL
    try:
        tier = int(tier)
    except (TypeError, ValueError):
        return "unknown"
    return ROLE_TIER_LABELS.get(tier, DEEP_ROLE_LABEL)


def fetch_nflverse_depth_chart(season, cfg=None):
    """Return the most recent depth-chart snapshot for one season, plus its timestamp.

    The 2026 asset is a running log of snapshots rather than one row per week, so the
    latest `dt` is taken. Only the offensive personnel group is kept. `pos_rank` is
    the flat rank within the team's position group; `add_slot_role_tiers` adds the
    per-slot tier that says whether the player is actually a starter.
    """
    cfg = _cfg(cfg)
    frame = _read_nflverse_csv("depth_charts", f"depth_charts_{season}.csv.gz", cfg)
    if "dt" in frame:
        frame = frame.copy()
        frame["dt"] = pd.to_datetime(frame["dt"], errors="coerce", utc=True)
        as_of = frame["dt"].max()
        frame = frame[frame["dt"].eq(as_of)]
    else:  # older seasons use a season/week schema
        as_of = None
        if "week" in frame:
            frame = frame[frame["week"].eq(frame["week"].max())]
    offense = frame[frame["pos_abb"].isin(["QB", "RB", "FB", "WR", "TE"])].copy()
    offense = rerank_aliased_positions(offense)
    offense = add_slot_role_tiers(offense)
    offense["Position"] = offense["pos_abb"].replace(NFLVERSE_POSITION_ALIASES)
    return offense, as_of


def fetch_nflverse_roster_status(season, cfg=None):
    """Return the latest weekly roster snapshot: who is active, cut, IR, or practice squad."""
    cfg = _cfg(cfg)
    frame = _read_nflverse_csv("weekly_rosters", f"roster_weekly_{season}.csv.gz", cfg)
    if "week" in frame and frame["week"].notna().any():
        frame = frame[frame["week"].eq(frame["week"].max())]
    name_column = next(
        (c for c in ("full_name", "player_name", "football_name") if c in frame), None
    )
    if name_column is None:
        raise ValueError("nflverse roster asset has no recognizable name column.")
    out = frame[[name_column, "team", "status"]].copy()
    out.columns = ["player_name", "team", "status"]
    return out


def _keyed(frame, team_column, name_column):
    """Index a reference frame by team + normalized name, dropping ambiguous keys.

    Two different players on one team who normalize to the same key cannot be told
    apart, so both are removed rather than guessed at.
    """
    out = frame.copy()
    out["_key"] = (
        out[team_column].astype(str).str.upper()
        + "|"
        + out[name_column].map(normalize_person_name)
    )
    counts = out["_key"].value_counts()
    return out[out["_key"].isin(counts[counts.eq(1)].index)].set_index("_key")


def build_nflverse_role_report(players, depth_chart, roster_status, cfg=None):
    """Join a Yahoo player pool to nflverse depth and roster status.

    Pure function: it performs no network access, so the smoke test can exercise the
    matching and disagreement logic offline. Returns one row per Yahoo player with
    the heuristic depth, the nflverse depth, the roster status, and whether each
    lookup actually matched.
    """
    cfg = _cfg(cfg)
    allowed = set(cfg.nflverse_available_status)
    overrides = {str(name): bool(value) for name, value in (AVAILABILITY_OVERRIDES or {}).items()}
    _warn_unmatched(overrides, "Availability override", set(players["Name"]))
    pool = players.copy()
    pool["_team"] = pool["Team"].astype(str).str.upper().replace(YAHOO_TO_NFLVERSE_TEAM)
    pool["_key"] = pool["_team"] + "|" + pool["Name"].map(normalize_person_name)

    depth_key, status_key = None, None
    if depth_chart is not None and len(depth_chart):
        depth_key = _keyed(depth_chart, "team", "player_name")
    if roster_status is not None and len(roster_status):
        status_key = _keyed(roster_status, "team", "player_name")

    rows = []
    columns = zip(
        pool["Name"], pool["Team"], pool["Position"], pool["Salary"],
        pool["Depth_Rank"], pool["_key"],
    )
    for name, team, position, salary, yahoo_depth, key in columns:
        is_defense = position == "DEF"
        nfl_depth, nfl_position, status = None, None, None
        role_slot, role_tier = None, None
        if depth_key is not None and not is_defense and key in depth_key.index:
            match = depth_key.loc[key]
            # Only accept the depth entry if the position agrees with Yahoo's.
            if str(match["Position"]) == str(position):
                nfl_depth = int(match["pos_rank"])
                nfl_position = str(match["pos_abb"])
                role_slot = match.get("Role_Slot")
                role_slot = None if pd.isna(role_slot) else str(role_slot)
                tier = match.get("Role_Tier")
                role_tier = None if pd.isna(tier) else int(tier)
        if status_key is not None and not is_defense and key in status_key.index:
            status = str(status_key.loc[key]["status"])

        if is_defense:
            available, reason = True, "team defense"
        elif name in overrides:
            available = bool(overrides[name])
            reason = (
                "manual availability override: "
                + ("available" if available else "unavailable")
            )
        elif status is None:
            # An unknown roster status is not evidence of availability. A player
            # the weekly roster has no row for may be a match failure, but he may
            # equally be a cut, a practice-squad body or someone who was never on
            # the 53. Treating that as "active" is how an invalid lineup gets
            # built; AVAILABILITY_OVERRIDES is the way to say otherwise.
            available = not cfg.nflverse_drop_unmatched
            reason = NO_ROSTER_ROW if available else f"{NO_ROSTER_ROW} (dropped)"
        else:
            available = status in allowed
            reason = NFLVERSE_STATUS_MEANING.get(status, f"unrecognized status {status}")

        rows.append({
            "Player": name,
            "Team": team,
            "Position": position,
            "Salary": float(salary),
            "Yahoo depth": int(yahoo_depth),
            "nflverse depth": nfl_depth,
            "nflverse position": nfl_position,
            "Role slot": role_slot,
            "Role tier": role_tier,
            "Role": (
                role_label(position, role_tier, nfl_position)
                if role_tier is not None else None
            ),
            "Depth matched": nfl_depth is not None,
            "Depth agrees": (nfl_depth is not None and int(nfl_depth) == int(yahoo_depth)),
            "Roster status": status,
            "Status meaning": reason,
            "Available": available,
        })
    report = pd.DataFrame(rows)
    report["Role tier"] = report["Role tier"].astype("Int64")
    report["Depth change"] = np.where(
        report["Depth matched"] & ~report["Depth agrees"],
        report["Yahoo depth"].astype(str) + " -> " + report["nflverse depth"].astype("Int64").astype(str),
        "",
    )
    return report


def roster_match_rate(report):
    """Share of non-defense players the weekly roster feed actually matched."""
    if report is None or report.empty:
        return 1.0
    skill = report[report["Position"].ne("DEF")]
    if skill.empty:
        return 1.0
    return float(skill["Roster status"].notna().mean())


def load_nflverse_reference(players, season, cfg=None):
    """Fetch depth chart and roster status, degrading to None on any failure.

    A network problem must never take down a lineup build, so every failure is
    reported and the run continues on the salary-based heuristic alone.
    """
    cfg = _cfg(cfg)
    depth_chart, as_of, roster_status, notes = None, None, None, []
    try:
        depth_chart, as_of = fetch_nflverse_depth_chart(season, cfg)
        notes.append(
            f"depth chart {season}: {len(depth_chart):,} offensive rows"
            + (f", snapshot {as_of:%Y-%m-%d %H:%M} UTC" if as_of is not None else "")
        )
    except Exception as exc:  # network, 404 for an unstarted season, schema drift
        notes.append(f"depth chart unavailable ({type(exc).__name__}: {exc}); keeping heuristic depth")
    try:
        roster_status = fetch_nflverse_roster_status(season, cfg)
        notes.append(f"roster status {season}: {len(roster_status):,} rows")
    except Exception as exc:
        notes.append(f"roster status unavailable ({type(exc).__name__}: {exc}); no availability filter")
    return depth_chart, roster_status, as_of, notes


def fetch_nflverse_injury_report(season, cfg=None):
    """Load the newest published weekly injury report for a season."""
    cfg = _cfg(cfg)
    frame = _read_nflverse_csv("injuries", f"injuries_{season}.csv", cfg)
    if frame.empty:
        return frame, None
    week = int(pd.to_numeric(frame["week"], errors="coerce").max())
    frame = frame[pd.to_numeric(frame["week"], errors="coerce").eq(week)].copy()
    name_column = "full_name" if "full_name" in frame else "player_name"
    frame["_key"] = (
        frame["team"].astype(str).str.upper().replace(YAHOO_TO_NFLVERSE_TEAM)
        + "|" + frame[name_column].map(normalize_person_name)
    )
    return frame.drop_duplicates("_key", keep="last"), week


def apply_nflverse_injuries(players, injuries):
    """Attach report fields and remove players officially listed Out."""
    out = players.copy()
    for column in ("report_primary_injury", "report_status", "practice_status"):
        out[column] = None
    if injuries is None or injuries.empty:
        return out, out.iloc[:0].copy()
    lookup = injuries.set_index("_key")
    keys = (
        out["Team"].astype(str).str.upper().replace(YAHOO_TO_NFLVERSE_TEAM)
        + "|" + out["Name"].map(normalize_person_name)
    )
    for index, key in zip(out.index, keys):
        if key not in lookup.index:
            continue
        row = lookup.loc[key]
        for column in ("report_primary_injury", "report_status", "practice_status"):
            if column in row:
                out.at[index, column] = row[column]
    is_out = out["report_status"].fillna("").astype(str).str.casefold().eq("out")
    return out.loc[~is_out].reset_index(drop=True), out.loc[is_out].reset_index(drop=True)


def apply_nflverse_roles(players, report, cfg=None):
    """Attach the published chart's role structure and the ordinal it implies.

    Three different things used to share one integer, and collapsing them is what
    made a starting slot receiver read as a deep reserve:

    ``Chart_Rank``
        Where the published chart puts the player in his team's position group.
    ``Role_Tier`` / ``Role_Label``
        His rank *within his own alignment slot*, and the word for it. Every
        parallel starter is tier 1 no matter where he falls in the flat ranking.
    ``Depth_Rank``
        Expected opportunity, set later by `apply_opportunity_ranks`. This is the
        quantity `CALIBRATED_CV` was fitted against, so it stays an ordinal.

    A manual DEPTH_OVERRIDES entry still wins outright: the point of that dict is
    to encode information the user has and the feed does not.
    """
    cfg = _cfg(cfg)
    out = players.copy()
    out["Chart_Rank"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["Role_Tier"] = pd.array([pd.NA] * len(out), dtype="Int64")
    out["Role_Slot"] = None
    out["Role_Label"] = "unknown"
    if not cfg.nflverse_apply_depth or report is None or report.empty:
        return out, pd.DataFrame()

    manual = set(out.loc[out["Depth_Source"].eq("manual override"), "Name"])
    matched = report[report["Depth matched"] & ~report["Player"].isin(manual)]
    # Team plus name, not name alone: two players on a slate can share a name,
    # and the report already carries one row per pool entry.
    by_identity = {
        (row["Team"], row["Player"]): row for row in matched.to_dict("records")
    }
    for index, team, name in zip(out.index, out["Team"], out["Name"]):
        row = by_identity.get((team, name))
        if row is None:
            continue
        out.at[index, "Chart_Rank"] = int(row["nflverse depth"])
        if pd.notna(row["Role tier"]):
            out.at[index, "Role_Tier"] = int(row["Role tier"])
            out.at[index, "Role_Label"] = str(row["Role"])
        if row["Role slot"] is not None and pd.notna(row["Role slot"]):
            out.at[index, "Role_Slot"] = str(row["Role slot"])

    skipped = report[
        report["Depth matched"] & ~report["Depth agrees"] & report["Player"].isin(manual)
    ]
    if len(skipped):
        warnings.warn(
            "Kept your manual DEPTH_OVERRIDES over the nflverse depth chart for: "
            + ", ".join(skipped["Player"])
        )
    return out, matched


def apply_opportunity_ranks(players, cfg=None):
    """Rank expected opportunity by blending the published chart with Yahoo salary.

    The ordinal that keys the fitted mean, CV and correlation tables blends the
    published chart ordering with Yahoo salary ordering. The chart retains sole
    ownership of the alignment-slot role tier.

    Manual DEPTH_OVERRIDES are left exactly where the user put them.
    """
    cfg = _cfg(cfg)
    weight = float(np.clip(cfg.role_salary_rank_weight, 0.0, 1.0))
    out = players.copy()
    if "Chart_Rank" not in out:
        out["Chart_Rank"] = pd.array([pd.NA] * len(out), dtype="Int64")
    if "Role_Tier" not in out:
        out["Role_Tier"] = pd.array([pd.NA] * len(out), dtype="Int64")
        out["Role_Slot"] = None
        out["Role_Label"] = "unknown"

    manual = out["Depth_Source"].eq("manual override")
    salary_rank = out.groupby(["Team", "Position"])["Salary"].rank(
        method="first", ascending=False
    )
    chart = pd.to_numeric(out["Chart_Rank"], errors="coerce")
    # An unmatched player has no chart opinion, so his own projection ordering
    # stands in for it and the blend leaves him where salary put him.
    chart_rank = out.assign(_c=chart.fillna(salary_rank)).groupby(
        ["Team", "Position"]
    )["_c"].rank(method="first", ascending=True)
    blended = (1.0 - weight) * chart_rank + weight * salary_rank
    # A tie goes to salary ordering first, then the chart, then
    # name, so the ordering is total and a rerun on identical inputs is identical.
    order = out.assign(
        _blend=blended, _salary=salary_rank, _chart=chart_rank
    ).sort_values(
        ["Team", "Position", "_blend", "_salary", "_chart", "Salary", "Name"],
        ascending=[True, True, True, True, True, False, True],
    )
    opportunity = (
        order.groupby(["Team", "Position"]).cumcount() + 1
    ).reindex(out.index).astype(int)

    out.loc[~manual, "Depth_Rank"] = opportunity[~manual]
    out["Depth_Rank"] = out["Depth_Rank"].astype(int)
    out.loc[~manual & chart.notna(), "Depth_Source"] = "nflverse chart + Yahoo salary"
    out.loc[~manual & chart.isna(), "Depth_Source"] = "Yahoo salary heuristic"

    # A player the chart never matched still needs a role word. His opportunity
    # rank is the only evidence available, so it names the role, and the source
    # column above already says the chart did not confirm it.
    unknown = out["Role_Tier"].isna()
    out.loc[unknown, "Role_Tier"] = out.loc[unknown, "Depth_Rank"].astype("Int64")
    out.loc[unknown, "Role_Label"] = [
        role_label(position, tier)
        for position, tier in zip(
            out.loc[unknown, "Position"], out.loc[unknown, "Depth_Rank"]
        )
    ]
    return out


def apply_nflverse_availability(players, report, cfg=None):
    """Hard-drop players the published roster says are not available.

    This is deliberately a removal rather than a projection haircut. A player on
    injured reserve has no distribution to simulate, and leaving him in the pool with
    a reduced mean would still let a long lognormal tail put him in a lineup.
    """
    cfg = _cfg(cfg)
    if not cfg.nflverse_availability_filter or report is None or report.empty:
        return players.reset_index(drop=True), pd.DataFrame()

    # Treating an unknown status as unavailable is only safe while the join is
    # working. If most of the pool failed to match, the roster feed is telling us
    # about our own name normalization, not about who is playing.
    match_rate = roster_match_rate(report)
    if match_rate < cfg.nflverse_min_match_rate:
        warnings.warn(
            f"nflverse weekly roster matched only {match_rate:.0%} of the skill-player "
            f"pool, below the {cfg.nflverse_min_match_rate:.0%} floor. Skipping the "
            "availability filter rather than dropping players on a broken join."
        )
        return players.reset_index(drop=True), pd.DataFrame()

    blocked = report[~report["Available"]]
    if blocked.empty:
        return players.reset_index(drop=True), blocked
    kept = players[~players["Name"].isin(set(blocked["Player"]))].reset_index(drop=True)
    return kept, blocked[["Player", "Team", "Position", "Salary", "Yahoo depth", "Roster status", "Status meaning"]]


def nflverse_disagreement_view(report):
    """The rows a human should actually look at before trusting the run."""
    interesting = report[
        (~report["Available"])
        | (report["Depth matched"] & ~report["Depth agrees"])
        | (~report["Depth matched"] & report["Position"].ne("DEF"))
    ]
    columns = [
        "Player", "Team", "Position", "Salary", "Yahoo depth", "nflverse depth",
        "Role slot", "Role tier", "Role", "Depth change", "Roster status",
        "Status meaning", "Available",
    ]
    return interesting[columns].sort_values(
        ["Available", "Salary"], ascending=[True, False]
    ).reset_index(drop=True)


# ============================================================================
# NOTEBOOK CELL 11 - Correlated outcome model and calibrated CVs
# ============================================================================
# Direct fitted forecast-error CV. TE4+ is held at the TE3 value because the
# small historical decline was not meaningful and a monotone risk curve is safer.
# QB3 had only 23 games, so QB2 is the fallback if backup QBs are explicitly kept.
CALIBRATED_CV = {
    "QB": {1: 0.488, 2: 0.922, 3: 0.922, 4: 0.922},
    "RB": {1: 0.676, 2: 0.921, 3: 1.154, 4: 1.207},
    "WR": {1: 0.695, 2: 0.814, 3: 1.042, 4: 1.323},
    "TE": {1: 0.885, 2: 1.324, 3: 1.658, 4: 1.658},
    "DEF": {1: 0.950, 2: 0.950, 3: 0.950, 4: 0.950},
}

# Probability that a player who dressed and touched the ball still finished the
# game on zero Yahoo points, by position and opportunity bucket. Fitted by
# tools/calibrate_zero_rate.py over 2016-2025 nflverse weekly stats, on the same
# population CALIBRATED_CV was fitted on.
#
# Scope is deliberate. A game with no carry, target or pass attempt is *excluded*
# from both numerator and denominator, because that is usually a player who was
# inactive, which is handled separately by nflverse roster and injury filters.
# Counting those games here would mix availability into the conditional scoring
# distribution. What is left is the pure
# shape effect: a WR4 who plays, runs his routes and is never thrown to.
#
# Sample sizes are the played-game counts behind each entry. QB3+ falls back to
# QB2 and TE4 to TE3, matching CALIBRATED_CV's own fallbacks. RB4's 186 games are
# the thinnest cell in the table; it is published because leaving RB4 at the RB3
# rate would understate the one bucket most likely to be a punt play.
#
# DEF is held at zero: weekly player stats carry no team-defense scoring, so
# there is nothing fitted here, and a defense can also post a *negative* score,
# which this model does not represent either.
ZERO_RATE = {
    "QB": {1: 0.014, 2: 0.259, 3: 0.259, 4: 0.259},  # 4,954 / 603 played games
    "RB": {1: 0.006, 2: 0.028, 3: 0.063, 4: 0.140},  # 5,153 / 4,315 / 1,544 / 186
    "WR": {1: 0.020, 2: 0.056, 3: 0.109, 4: 0.185},  # 5,166 / 5,060 / 4,258 / 2,510
    "TE": {1: 0.048, 2: 0.141, 3: 0.176, 4: 0.176},  # 4,874 / 2,502 / 499
    "DEF": {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
}

QB_WR_CORR = {1: 0.217, 2: 0.171, 3: 0.129, 4: 0.076}
QB_TE_CORR = {1: 0.152, 2: 0.085, 3: 0.072, 4: 0.060}
RB_OWN_DEF_CORR = {1: 0.057, 2: 0.015, 3: 0.000, 4: 0.000}
OPPONENT_DEF_CORR = {
    "QB": {1: -0.241, 2: 0.000, 3: 0.000, 4: 0.000},
    "RB": {1: -0.172, 2: -0.104, 3: -0.069, 4: 0.000},
    "WR": {1: -0.112, 2: -0.100, 3: -0.029, 4: -0.052},
    "TE": {1: -0.059, 2: -0.013, 3: -0.021, 4: 0.000},
}
# Same-team running backs, by expected-opportunity rank. Fitted by
# tools/calibrate_rb_correlation.py over 2016-2025 nflverse weekly stats: forecast
# error against a lagged eight-game rolling expectation, ranked within the
# team-week by lagged touch load so no in-game information leaks into the
# grouping. Rank 4 is the 4-or-deeper bucket.
#
# These replace one pooled ("RB", "RB"): 0.013 that could not tell a bellcow and
# his change-of-pace back apart from two deep reserves. The comment on each line
# is the same correlation measured on carries + targets instead of points, and it
# is the reason these numbers are small: the two backs really are splitting a
# fixed pool of work (RB1-RB2 touch errors correlate -0.082), but touchdowns and
# long gains are noisy enough that most of the cannibalization does not survive
# into fantasy scoring. Every 95% interval here straddles zero, so read them as
# the fitted point estimates they are, not as confident signs.
SAME_TEAM_RB_RB_CORR = {
    (1, 2): -0.015,  # 4,965 team-games, touches -0.082
    (1, 3): -0.028,  # 3,276 team-games, touches -0.043
    (1, 4): -0.000,  # 958 team-games, touches -0.106
    (2, 3): +0.025,  # 3,276 team-games, touches +0.039
    (2, 4): +0.003,  # 958 team-games, touches +0.040
    (3, 4): +0.056,  # 958 team-games, touches +0.082
}

SAME_TEAM_OTHER_CORR = {
    ("QB", "QB"): -0.200,
    ("QB", "DEF"): -0.061,
    ("RB", "WR"): -0.002,
    ("RB", "TE"): -0.010,
    ("WR", "WR"): 0.011,
    ("WR", "TE"): 0.009,
    ("TE", "TE"): -0.004,
    ("WR", "DEF"): -0.044,
    ("TE", "DEF"): -0.040,
}
OPPOSING_OFFENSE_CORR = {
    ("QB", "QB"): 0.175,
    ("QB", "RB"): 0.013,
    ("QB", "WR"): 0.042,
    ("QB", "TE"): 0.048,
    ("RB", "RB"): 0.019,
    ("RB", "WR"): 0.011,
    ("RB", "TE"): -0.006,
    ("WR", "WR"): 0.021,
    ("WR", "TE"): 0.016,
    ("TE", "TE"): 0.016,
}
POSITION_ORDER = {position: index for index, position in enumerate(VALID_POSITIONS)}


def _position_pair(first, second):
    """Return a stable order for symmetric position-pair lookup tables."""
    return tuple(sorted((first, second), key=lambda value: POSITION_ORDER[value]))


def _calibrated_cv(position, depth):
    """Return fitted total forecast-error CV for one position/depth bucket."""
    return CALIBRATED_CV[position][_depth_bucket(depth)]


def _zero_rate(position, depth):
    """Return the fitted played-but-scoreless probability for one player."""
    return ZERO_RATE[position][_depth_bucket(depth)]


def conditional_lognormal_cv(cv, zero):
    """CV of the scoring part of a hurdle model that preserves the fitted total.

    A score is `0` with probability `p` and otherwise lognormal with mean
    `m / (1 - p)`, which leaves the unconditional mean at `m`. Writing `c` for
    the conditional CV, the unconditional CV of that mixture is

        CV^2 = (c^2 + p) / (1 - p),

    so holding the fitted `CALIBRATED_CV` fixed pins `c^2 = CV^2 (1 - p) - p`.
    Nothing about the marginal moments moves; only the shape does, with mass
    taken out of the lower body and placed on an atom at zero. Every published
    (CV, rate) pair clears the positivity constraint with room to spare -- the
    tightest is TE1 at 0.783 against 0.050 -- and the floor here is a numerical
    guard, not a modelling choice.
    """
    cv = np.asarray(cv, dtype=float)
    zero = np.asarray(zero, dtype=float)
    return np.sqrt(np.maximum(cv * cv * (1.0 - zero) - zero, 1e-12))


def zero_rate_is_representable(cv, zero):
    """Whether a (CV, zero rate) pair leaves a positive conditional variance."""
    return float(cv) * float(cv) * (1.0 - float(zero)) - float(zero) > 0.0


def normal_cdf(value):
    """Vectorized standard normal CDF, Hart's double-precision rational form.

    Accurate to machine epsilon against `math.erf`, and `site/showdown-worker.js`
    carries the same coefficients so the browser applies the same transform.
    """
    value = np.asarray(value, dtype=float)
    magnitude = np.abs(value)
    density = np.exp(-0.5 * np.square(magnitude))
    numerator = (((((3.52624965998911e-02 * magnitude + 0.700383064443688)
                    * magnitude + 6.37396220353165) * magnitude + 33.912866078383)
                  * magnitude + 112.079291497871) * magnitude + 221.213596169931
                 ) * magnitude + 220.206867912376
    denominator = ((((((8.83883476483184e-02 * magnitude + 1.75566716318264)
                       * magnitude + 16.064177579207) * magnitude + 86.7807322029461)
                     * magnitude + 296.564248779674) * magnitude + 637.333633378831)
                   * magnitude + 793.826512519948) * magnitude + 440.413735824752
    near = magnitude < 7.07106781186547
    tail = np.where(near, density * numerator / denominator, 0.0)
    if not np.all(near):
        distant = np.where(near, 1.0, magnitude)
        fraction = distant + 1.0 / (distant + 2.0 / (distant + 3.0 / (distant + 4.0 / (distant + 0.65))))
        tail = np.where(near, tail, density / fraction / 2.506628274631)
    return np.where(value > 0, 1.0 - tail, tail)


_PPF_A = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
_PPF_B = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
_PPF_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
          -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
_PPF_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
          3.754408661907416e+00)
_PPF_SPLIT = 0.02425


def normal_ppf(probability):
    """Vectorized inverse standard normal CDF (Acklam), mirrored in the worker."""
    probability = np.clip(np.asarray(probability, dtype=float), 1e-300, 1.0 - 1e-16)
    out = np.empty_like(probability)
    lower = probability < _PPF_SPLIT
    upper = probability > 1.0 - _PPF_SPLIT
    middle = ~(lower | upper)

    def tail(values, sign):
        q = np.sqrt(-2.0 * np.log(values))
        top = ((((_PPF_C[0] * q + _PPF_C[1]) * q + _PPF_C[2]) * q + _PPF_C[3]) * q + _PPF_C[4]) * q + _PPF_C[5]
        bottom = (((_PPF_D[0] * q + _PPF_D[1]) * q + _PPF_D[2]) * q + _PPF_D[3]) * q + 1.0
        return sign * top / bottom

    out[lower] = tail(probability[lower], 1.0)
    out[upper] = tail(1.0 - probability[upper], -1.0)
    q = probability[middle] - 0.5
    r = q * q
    top = (((((_PPF_A[0] * r + _PPF_A[1]) * r + _PPF_A[2]) * r + _PPF_A[3]) * r + _PPF_A[4]) * r + _PPF_A[5]) * q
    bottom = ((((_PPF_B[0] * r + _PPF_B[1]) * r + _PPF_B[2]) * r + _PPF_B[3]) * r + _PPF_B[4]) * r + 1.0
    out[middle] = top / bottom
    return out


def hurdle_transform(latent, means, cv, zero):
    """Map correlated standard normals to scores carrying an atom at zero.

    The same latent draw decides both halves: a player scores nothing exactly
    when his own uniform falls under his zero rate, and the remaining uniform is
    stretched back over (0, 1) to place the magnitude. The map is monotone in the
    latent, so the joint distribution stays a Gaussian copula and excluding a
    player is still a principal submatrix -- which is what lets the browser skip
    the eigensolver and reuse one set of scenario draws.
    """
    means = np.asarray(means, dtype=float)
    cv = np.asarray(cv, dtype=float)
    zero = np.asarray(zero, dtype=float)
    plain_sigma = np.sqrt(np.log1p(np.square(cv)))
    plain = means * np.exp(latent * plain_sigma - 0.5 * np.square(plain_sigma))
    if not np.any(zero > 0):
        return plain
    sigma = np.sqrt(np.log1p(np.square(conditional_lognormal_cv(cv, zero))))
    uniform = normal_cdf(latent)
    safe_zero = np.where(zero > 0, zero, 0.0)
    rescaled = np.clip((uniform - safe_zero) / np.maximum(1.0 - safe_zero, 1e-12),
                       1e-300, 1.0 - 1e-16)
    magnitude = (means / np.maximum(1.0 - safe_zero, 1e-12)) * np.exp(
        normal_ppf(rescaled) * sigma - 0.5 * np.square(sigma)
    )
    hurdled = np.where(uniform < safe_zero, 0.0, magnitude)
    return np.where(zero > 0, hurdled, plain)


def same_team_rb_rb_correlation(a, b):
    """Fitted forecast-error correlation for two backs sharing a backfield."""
    first = _depth_bucket(a["Depth_Rank"])
    second = _depth_bucket(b["Depth_Rank"])
    if first == second:
        # Two players the model ranks identically are the same role twice over,
        # which the fit has nothing to say about. Fall back to the shallowest
        # published pair rather than inventing a value.
        return SAME_TEAM_RB_RB_CORR[(1, 2)]
    key = (min(first, second), max(first, second))
    return SAME_TEAM_RB_RB_CORR[key]


def target_score_correlation(a, b):
    """Return the fitted fantasy-score correlation for a player pair.

    Depth-specific estimates are used where the historical results showed a clear
    gradient. Near-zero skill-player relationships are retained near zero instead
    of being forced upward by a shared team factor. Style overrides only affect
    QB-RB: the calibration did not separately identify role styles, so the
    pass-catching value is deliberately modest rather than presented as fitted.

    Same-team RB pairs are the one relationship keyed on both players' ranks,
    because that pair is splitting one pool of carries and goal-line work and a
    single pooled number cannot say how much they overlap.
    """
    pa, pb = a["Position"], b["Position"]
    same_team = a["Team"] == b["Team"]
    pair = _position_pair(pa, pb)

    if same_team:
        if pair == ("QB", "WR"):
            receiver = a if pa == "WR" else b
            return QB_WR_CORR[_depth_bucket(receiver["Depth_Rank"])]
        if pair == ("QB", "TE"):
            receiver = a if pa == "TE" else b
            return QB_TE_CORR[_depth_bucket(receiver["Depth_Rank"])]
        if pair == ("QB", "RB"):
            back = a if pa == "RB" else b
            return 0.080 if back.get("Player_Style", "standard") == "pass_catching_rb" else 0.020
        if pair == ("RB", "DEF"):
            back = a if pa == "RB" else b
            return RB_OWN_DEF_CORR[_depth_bucket(back["Depth_Rank"])]
        if pa == pb == "RB":
            return same_team_rb_rb_correlation(a, b)
        return SAME_TEAM_OTHER_CORR.get(pair, 0.0)

    if "DEF" in pair:
        if pa == pb == "DEF":
            return 0.0
        offense = b if pa == "DEF" else a
        return OPPONENT_DEF_CORR[offense["Position"]][
            _depth_bucket(offense["Depth_Rank"])
        ]
    return OPPOSING_OFFENSE_CORR.get(pair, 0.0)


def requested_score_correlation_matrix(players):
    """Build the symmetric matrix of historical score-correlation targets."""
    matrix = np.eye(len(players), dtype=np.float64)
    records = [players.iloc[index] for index in range(len(players))]
    for first, second in itertools.combinations(range(len(players)), 2):
        value = target_score_correlation(records[first], records[second])
        matrix[first, second] = matrix[second, first] = float(value)
    return matrix


def lognormal_correlation_bounds(cv):
    """Return the attainable score-correlation range for each lognormal pair.

    Two mean-preserving lognormals with coefficients of variation cv_i, cv_j cannot
    reach every correlation in [-1, 1]. With s_i = sqrt(log(1 + cv_i^2)), the score
    correlation is (exp(rho * s_i * s_j) - 1) / (cv_i * cv_j), so it is bounded by
    rho = -1 and rho = +1. High-CV deep-role players have a floor well above -1:
    a WR4/TE4 pair cannot be more negatively correlated than about -0.31.
    """
    sigma = np.sqrt(np.log1p(np.square(cv)))
    outer_sigma = np.outer(sigma, sigma)
    denominator = np.outer(cv, cv)
    with np.errstate(divide="ignore", invalid="ignore"):
        low = np.where(denominator > 0, np.expm1(-outer_sigma) / denominator, -1.0)
        high = np.where(denominator > 0, np.expm1(outer_sigma) / denominator, 1.0)
    return low, high


def score_to_lognormal_latent(score_corr, cv, report=None):
    """Convert desired lognormal score correlations to Gaussian correlations.

    For mean-preserving lognormal variables, score correlation is not the same as
    latent-normal correlation. Solving the closed-form covariance relationship
    before PSD repair keeps the simulated score matrix close to the historical
    targets despite large, depth-dependent CVs.

    v3.2: a target outside the attainable range above used to fall through to
    log(1e-8) and then clip to a latent -0.95, quietly delivering a different
    correlation than requested. Targets are now clipped to the attainable bound
    first, and every clipped pair is counted and reported.
    """
    low, high = lognormal_correlation_bounds(cv)
    margin = 1e-6
    feasible = np.clip(score_corr, low + margin, high - margin)
    np.fill_diagonal(feasible, 1.0)
    infeasible = np.abs(feasible - score_corr) > 1e-9
    np.fill_diagonal(infeasible, False)

    sigma = np.sqrt(np.log1p(np.square(cv)))
    latent = np.eye(len(cv), dtype=np.float64)
    for first, second in itertools.combinations(range(len(cv)), 2):
        argument = 1.0 + feasible[first, second] * cv[first] * cv[second]
        denominator = sigma[first] * sigma[second]
        value = math.log(argument) / denominator if denominator > 0 else 0.0
        latent[first, second] = latent[second, first] = float(np.clip(value, -0.999, 0.999))

    if report is not None:
        report["infeasible_pairs"] = int(infeasible.sum() // 2)
        report["max_infeasible_shift"] = (
            float(np.abs(feasible - score_corr).max()) if infeasible.any() else 0.0
        )
    if infeasible.any():
        warnings.warn(
            f"{int(infeasible.sum() // 2)} correlation target(s) were outside the range "
            "two lognormals with these coefficients of variation can attain and were "
            "clipped to the nearest attainable value (largest shift "
            f"{float(np.abs(feasible - score_corr).max()):.3f})."
        )
    return latent


def repair_correlation_matrix(matrix, eigenvalue_floor=1e-8):
    """Project a symmetric matrix to a numerically safe PSD correlation matrix.

    Independently fitted pairwise targets are not guaranteed to form a valid joint
    distribution. Eigenvalue clipping followed by diagonal renormalization makes
    simulation possible while changing the requested structure as little as this
    lightweight, dependency-free repair permits.
    """
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    repaired = eigenvectors @ np.diag(np.maximum(eigenvalues, eigenvalue_floor)) @ eigenvectors.T
    scale = np.sqrt(np.maximum(np.diag(repaired), eigenvalue_floor))
    repaired = repaired / np.outer(scale, scale)
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def lognormal_score_correlation(latent_corr, cv):
    """Return the score correlation implied by a latent lognormal matrix."""
    sigma = np.sqrt(np.log1p(np.square(cv)))
    numerator = np.expm1(latent_corr * np.outer(sigma, sigma))
    denominator = np.outer(cv, cv)
    score = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0,
    )
    score = np.clip(0.5 * (score + score.T), -0.999, 0.999)
    np.fill_diagonal(score, 1.0)
    return score


HURDLE_HERMITE_TERMS = 24
HURDLE_QUADRATURE_NODES = 20001
HURDLE_QUADRATURE_LIMIT = 9.0
_HURDLE_FACTORIALS = np.array(
    [float(math.factorial(term)) for term in range(HURDLE_HERMITE_TERMS + 1)]
)


@lru_cache(maxsize=None)
def _hurdle_hermite_coefficients(cv, zero):
    """Hermite coefficients of one player's unit-mean score transform.

    For a Gaussian copula, the covariance of two transformed marginals is exactly

        Cov = sum_{k>=1} rho^k a_k^i a_k^j / k!,

    with `a_k = E[g(U) He_k(U)]` (Mehler's formula). Computing the coefficients
    once per distinct (CV, zero rate) pair turns the latent-to-score map into a
    polynomial in `rho`, which is what makes the pairwise inversion below cheap
    enough to run on every publish. The closed form the pure lognormal uses does
    not survive the atom at zero, and Gauss-Hermite quadrature overflows at the
    orders needed here, so the coefficients come from a fine uniform grid.

    `a_0` recovers the mean (1.0) and `sum_k a_k^2 / k!` recovers `CV^2`; both are
    asserted in the tests, and both agree to about 1e-5 on this grid.
    """
    grid = np.linspace(-HURDLE_QUADRATURE_LIMIT, HURDLE_QUADRATURE_LIMIT,
                       HURDLE_QUADRATURE_NODES)
    spacing = grid[1] - grid[0]
    density = np.exp(-0.5 * np.square(grid)) / math.sqrt(2.0 * math.pi)
    weighted = hurdle_transform(grid, 1.0, cv, zero) * density

    def integrate(values):
        # Uniform-grid trapezoid, spelled out so this does not depend on whether
        # the installed numpy calls it trapz or trapezoid.
        return spacing * (values.sum() - 0.5 * (values[0] + values[-1]))

    coefficients = np.empty(HURDLE_HERMITE_TERMS + 1)
    previous = np.ones_like(grid)
    coefficients[0] = integrate(weighted * previous)
    current = grid.copy()
    coefficients[1] = integrate(weighted * current)
    for term in range(1, HURDLE_HERMITE_TERMS):
        following = grid * current - term * previous
        previous, current = current, following
        coefficients[term + 1] = integrate(weighted * current)
    return coefficients


def _hurdle_coefficient_matrix(cv, zero):
    return np.array([
        _hurdle_hermite_coefficients(float(value), float(rate))
        for value, rate in zip(np.asarray(cv, float), np.asarray(zero, float))
    ])


def _hurdle_correlation_from_latent(latent, coefficients, cv):
    """Score correlation implied by a latent correlation, elementwise."""
    terms = np.arange(1, HURDLE_HERMITE_TERMS + 1)
    pairwise = np.einsum(
        "ik,jk->ijk", coefficients[:, 1:], coefficients[:, 1:]
    ) / _HURDLE_FACTORIALS[1:]
    powers = np.power(np.asarray(latent, float)[..., None], terms)
    covariance = np.sum(pairwise * powers, axis=-1)
    denominator = np.outer(cv, cv)
    score = np.divide(covariance, denominator, out=np.zeros_like(covariance),
                      where=denominator > 0)
    score = 0.5 * (score + score.T)
    np.fill_diagonal(score, 1.0)
    return score


def hurdle_score_correlation(latent_corr, cv, zero):
    """Return the score correlation a latent matrix delivers under the hurdle."""
    if not np.any(np.asarray(zero, float) > 0):
        return lognormal_score_correlation(latent_corr, cv)
    coefficients = _hurdle_coefficient_matrix(cv, zero)
    score = _hurdle_correlation_from_latent(latent_corr, coefficients, np.asarray(cv, float))
    return np.clip(score, -0.999, 0.999) * (1 - np.eye(len(cv))) + np.eye(len(cv))


def score_to_hurdle_latent(score_corr, cv, zero, report=None):
    """Invert the latent-to-score map when the marginals carry an atom at zero.

    The pure lognormal case has a closed form and keeps it. With a zero rate the
    map is a polynomial in `rho` rather than an exponential, and it is monotone
    over the attainable range, so each pair is solved by bisection against its own
    Mehler series. Targets outside what the pair can reach are clipped to the
    nearest attainable value and counted, exactly as the lognormal path does --
    the atom narrows that range further, because two marginals that are both zero
    a fifth of the time cannot be driven as far apart as two that never are.
    """
    cv = np.asarray(cv, dtype=float)
    zero = np.asarray(zero, dtype=float)
    if not np.any(zero > 0):
        return score_to_lognormal_latent(score_corr, cv, report=report)

    coefficients = _hurdle_coefficient_matrix(cv, zero)
    size = len(cv)
    limit = np.full((size, size), 0.999)
    low = _hurdle_correlation_from_latent(-limit, coefficients, cv)
    high = _hurdle_correlation_from_latent(limit, coefficients, cv)
    margin = 1e-6
    feasible = np.clip(score_corr, low + margin, high - margin)
    np.fill_diagonal(feasible, 1.0)
    infeasible = np.abs(feasible - score_corr) > 1e-9
    np.fill_diagonal(infeasible, False)

    lower = np.full((size, size), -0.999)
    upper = np.full((size, size), 0.999)
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        implied = _hurdle_correlation_from_latent(middle, coefficients, cv)
        too_low = implied < feasible
        lower = np.where(too_low, middle, lower)
        upper = np.where(too_low, upper, middle)
    latent = 0.5 * (lower + upper)
    latent = 0.5 * (latent + latent.T)
    np.fill_diagonal(latent, 1.0)

    if report is not None:
        report["infeasible_pairs"] = int(infeasible.sum() // 2)
        report["max_infeasible_shift"] = (
            float(np.abs(feasible - score_corr).max()) if infeasible.any() else 0.0
        )
    if infeasible.any():
        warnings.warn(
            f"{int(infeasible.sum() // 2)} correlation target(s) were outside the range "
            "two hurdle marginals with these coefficients of variation and zero rates "
            "can attain and were clipped to the nearest attainable value (largest shift "
            f"{float(np.abs(feasible - score_corr).max()):.3f})."
        )
    return latent


def build_correlation_model(players):
    """Build requested, latent, repaired, and effective score correlations."""
    teams = list(players["Team"].drop_duplicates())
    if len(teams) != 2:
        raise ValueError(f"Expected exactly two teams, found {teams}")
    cv = np.array([
        _calibrated_cv(row.Position, row.Depth_Rank) for row in players.itertuples()
    ])
    zero = np.array([
        _zero_rate(row.Position, row.Depth_Rank) for row in players.itertuples()
    ])
    requested_score = requested_score_correlation_matrix(players)
    feasibility = {}
    requested_latent = score_to_hurdle_latent(requested_score, cv, zero, report=feasibility)
    latent_corr = repair_correlation_matrix(requested_latent)
    effective_score = hurdle_score_correlation(latent_corr, cv, zero)
    off_diagonal = ~np.eye(len(players), dtype=bool)
    max_adjustment = float(
        np.max(np.abs(effective_score[off_diagonal] - requested_score[off_diagonal]))
    ) if len(players) > 1 else 0.0
    return {
        "cv": cv,
        "zero_rate": zero,
        "target_score_corr": requested_score,
        "requested_latent_corr": requested_latent,
        "latent_corr": latent_corr,
        "score_corr": effective_score,
        "psd_max_score_adjustment": max_adjustment,
        "infeasible_pairs": feasibility.get("infeasible_pairs", 0),
        "max_infeasible_shift": feasibility.get("max_infeasible_shift", 0.0),
    }


def latent_root(latent_corr, jitter=1e-10, attempts=6):
    """Return the unique lower-triangular factor R with R @ R.T == latent_corr.

    v3.6 replaced an eigendecomposition root. The correlation targets are looked
    up by (position, depth bucket, team), so players sharing all three get
    identical rows and the matrix carries exactly degenerate eigenvalues -- a
    typical showdown pool has a six-fold one. Eigenvectors spanning a degenerate
    eigenspace are arbitrary, so `eigh` returned a basis that depended on the
    LAPACK build, and `random_seed` did not actually pin the scenario set: a
    1e-13 perturbation of the same matrix re-based that eigenspace and dropped
    the per-player correlation between the two scenario sets to a median of 0.08.
    A Cholesky factor is unique for a positive-definite matrix, so the same seed
    now reproduces the same draws -- and it is the factorization
    `site/showdown-worker.js` already uses.

    `repair_correlation_matrix` floors the eigenvalues, so the input is positive
    definite by construction; the jitter loop only covers the case where that
    floor is thin enough for the factorization to fail in floating point.
    """
    matrix = np.asarray(latent_corr, dtype=np.float64)
    for attempt in range(attempts):
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            matrix = np.asarray(latent_corr, dtype=np.float64) + np.eye(
                len(matrix)
            ) * jitter * (10.0 ** attempt)
    raise np.linalg.LinAlgError(
        "Latent correlation matrix is not positive definite even with jitter."
    )


def simulate_player_outcomes(players, cfg=None):
    """Simulate mean-preserving scores under the repaired joint model.

    Each marginal is a zero-hurdle lognormal: an atom at zero of the fitted
    played-but-scoreless size, and a lognormal above it scaled so the
    unconditional mean is still `Projected_FP` and the unconditional CV is still
    `CALIBRATED_CV`. Only the shape moves, which is the point -- a WR4's floor was
    the one part of this model measurement said was badly wrong.
    """
    cfg = _cfg(cfg)
    model = build_correlation_model(players)
    root = latent_root(model["latent_corr"])
    rng = np.random.default_rng(cfg.random_seed)
    latent = rng.normal(size=(cfg.simulations, len(players))) @ root.T
    means = players["Projected_FP"].to_numpy(float)
    outcomes = hurdle_transform(latent, means, model["cv"], model["zero_rate"])
    return outcomes.astype(np.float32), model


CALIBRATION_SANITY_RELATIONSHIPS = [
    "Same team QB-WR",
    "Same team QB-TE",
    "Same team QB-RB",
    "Same team RB-DEF",
    "Same team RB-RB",
    "Same team WR-WR",
    "Same team WR-TE",
    "Opponent QB-DEF",
    "Opponent WR-DEF",
    "Opponent TE-DEF",
    "Opponent RB-DEF",
    "Opposing QB-QB",
]


def _pair_relationship(a, b):
    pa, pb = a["Position"], b["Position"]
    same = a["Team"] == b["Team"]
    pair = {pa, pb}
    if same:
        if pair == {"QB", "WR"}: return "Same team QB-WR"
        if pair == {"QB", "TE"}: return "Same team QB-TE"
        if pair == {"QB", "RB"}: return "Same team QB-RB"
        if pair == {"RB", "DEF"}: return "Same team RB-DEF"
        if pa == pb == "RB": return "Same team RB-RB"
        if pa == pb == "WR": return "Same team WR-WR"
        if pair == {"WR", "TE"}: return "Same team WR-TE"
        ordered = _position_pair(pa, pb)
        return f"Same team {ordered[0]}-{ordered[1]}"
    if "DEF" in pair:
        offense = pb if pa == "DEF" else pa
        return f"Opponent {offense}-DEF"
    ordered = _position_pair(pa, pb)
    return f"Opposing {ordered[0]}-{ordered[1]}"


def correlation_sanity_report(players, outcomes, factor_model):
    """Compare historical targets, PSD-repaired model values, and simulations."""
    empirical = np.corrcoef(outcomes, rowvar=False)
    target = factor_model["target_score_corr"]
    model_score = factor_model["score_corr"]
    latent = factor_model["latent_corr"]
    records = [players.iloc[index] for index in range(len(players))]
    rows = []
    for i, j in itertools.combinations(range(len(players)), 2):
        a, b = records[i], records[j]
        rows.append({
            "Relationship": _pair_relationship(a, b),
            "Player A": a["Name"],
            "Player B": b["Name"],
            "Target score corr": target[i, j],
            "PSD model score corr": model_score[i, j],
            "Latent corr": latent[i, j],
            "Simulated score corr": empirical[i, j],
            "PSD target gap": model_score[i, j] - target[i, j],
            "Simulation target gap": empirical[i, j] - target[i, j],
        })
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("Relationship")
        .agg(
            Pairs=("Simulated score corr", "size"),
            Target=("Target score corr", "median"),
            Model=("PSD model score corr", "median"),
            Simulated=("Simulated score corr", "median"),
            Max_PSD_Gap=("PSD target gap", lambda values: np.abs(values).max()),
            Max_Sim_Gap=("Simulation target gap", lambda values: np.abs(values).max()),
        )
        .reset_index()
    )
    summary["Sanity"] = np.where(
        summary["Max_PSD_Gap"].le(0.04) & summary["Max_Sim_Gap"].le(0.08),
        "PASS",
        "REVIEW",
    )
    for column in ["Target", "Model", "Simulated", "Max_PSD_Gap", "Max_Sim_Gap"]:
        summary[column] = summary[column].round(3)
    return summary, detail


def analytic_covariance(players, factor_model):
    """Convert model means, fitted CVs, and effective score correlations to covariance."""
    means = players["Projected_FP"].to_numpy(float)
    std = means * factor_model["cv"]
    return np.outer(std, std) * factor_model["score_corr"]


def marginal_sanity_report(players, outcomes):
    """Confirm the simulator reproduced each player's intended mean and CV.

    The candidate screen ranks lineups with the analytic covariance while the final
    ranking uses the simulated draws. If those two descriptions of the same model
    ever diverged, the screen would silently discard good lineups. This checks the
    marginals; `analytic_vs_simulated_lineup_check` checks the joint behaviour.
    """
    intended_mean = players["Projected_FP"].to_numpy(float)
    intended_cv = np.array([
        _calibrated_cv(row.Position, row.Depth_Rank) for row in players.itertuples()
    ])
    simulated_mean = outcomes.mean(axis=0).astype(float)
    simulated_cv = outcomes.std(axis=0).astype(float) / np.maximum(simulated_mean, 1e-9)
    report = pd.DataFrame({
        "Player": players["Name"].to_numpy(),
        "Position": players["Position"].to_numpy(),
        "Depth": players["Depth_Rank"].to_numpy(),
        "Intended mean": np.round(intended_mean, 3),
        "Simulated mean": np.round(simulated_mean, 3),
        "Mean error %": np.round(100 * (simulated_mean / np.maximum(intended_mean, 1e-9) - 1), 2),
        "Intended CV": np.round(intended_cv, 3),
        "Simulated CV": np.round(simulated_cv, 3),
        "CV error %": np.round(100 * (simulated_cv / np.maximum(intended_cv, 1e-9) - 1), 2),
    })
    report["Sanity"] = np.where(
        report["Mean error %"].abs().le(3.0) & report["CV error %"].abs().le(8.0),
        "PASS",
        "REVIEW",
    )
    return report


# ============================================================================
# NOTEBOOK CELL 13 - Candidate lineup generation and shared-scenario scoring
# ============================================================================
# Guard against an enumeration the notebook cannot hold in memory. C(36,5) is
# 376,992; the budget below leaves room to raise max_enumeration_players to ~55.
MAX_COMBINATION_BUDGET = 30_000_000


def _as_range_indexed(players):
    """Every lineup id in this notebook is positional, so force a 0..n-1 index."""
    index = players.index
    if isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1:
        return players
    return players.reset_index(drop=True)


def _lineup_variance(cov, combos, superstar_slot, chunk):
    """Variance of a Yahoo lineup, vectorized over many rosters at once.

    With weights w = 1 + 0.5 * e_s (the Superstar slot carries 1.5x),
        w' C w = sum(C) + rowsum_s(C) + 0.25 * C_ss
    over the 5x5 submatrix of the roster, which avoids building one quadratic form
    per candidate in Python.
    """
    total = len(combos)
    variance = np.empty(total, dtype=np.float64)
    for start in range(0, total, chunk):
        stop = min(start + chunk, total)
        block = combos[start:stop]
        sub = cov[block[:, :, None], block[:, None, :]]
        rows = np.arange(stop - start)
        slots = superstar_slot[start:stop]
        variance[start:stop] = (
            sub.sum(axis=(1, 2))
            + sub.sum(axis=2)[rows, slots]
            + 0.25 * np.diagonal(sub, axis1=1, axis2=2)[rows, slots]
        )
    return np.maximum(variance, 0.0)


def enumerate_candidate_lineups(players, salary_cap, covariance, cfg=None, chunk=150_000):
    """Enumerate Yahoo-valid rosters and retain strong mean/ceiling candidates.

    v3.2 replaces the per-combination Python loop and two heaps with array
    operations and `np.argpartition`. On a 36-player pool this runs about 40x
    faster and returns the same candidate set to floating-point tolerance.
    """
    cfg = _cfg(cfg)
    players = _as_range_indexed(players)
    n = len(players)
    size = int(cfg.lineup_size)
    if n < size:
        raise ValueError(f"Only {n} eligible players remain; need {size}.")

    problem = roster_feasibility_error(players, size)
    if problem:
        raise ValueError(problem)

    total_combinations = math.comb(n, size)
    if total_combinations > MAX_COMBINATION_BUDGET:
        raise ValueError(
            f"{total_combinations:,} rosters exceed the {MAX_COMBINATION_BUDGET:,} "
            "enumeration budget. Lower Settings.max_enumeration_players."
        )

    salary = players["Salary"].to_numpy(float)
    projection = players["Projected_FP"].to_numpy(float)
    positions = players["Position"].to_numpy(str)
    team_values = players["Team"].to_numpy(str)
    teams = list(pd.unique(team_values))
    covariance = np.ascontiguousarray(covariance, dtype=np.float64)
    min_salary = salary_cap * cfg.min_salary_used_pct

    combos = np.fromiter(
        itertools.chain.from_iterable(itertools.combinations(range(n), size)),
        dtype=np.int32,
        count=total_combinations * size,
    ).reshape(total_combinations, size)

    combo_salary = salary[combos].sum(axis=1)
    keep_mask = (combo_salary <= salary_cap) & (combo_salary >= min_salary)
    # Yahoo single-game rule: at least one non-defense player from each team.
    for team in teams:
        keep_mask &= (
            (team_values[combos] == team) & (positions[combos] != "DEF")
        ).any(axis=1)
    for position, (low, high) in (cfg.position_limits or {}).items():
        counts = (positions[combos] == position).sum(axis=1)
        keep_mask &= (counts >= low) & (counts <= high)

    combos = combos[keep_mask]
    combo_salary = combo_salary[keep_mask]
    valid_rosters = int(len(combos))
    if valid_rosters == 0:
        raise ValueError(
            "No valid rosters. Review exclusions, salary floor, cap, or optional position limits."
        )

    # Score every (roster, Superstar) pair: shape (rosters, lineup_size).
    base_projection = projection[combos].sum(axis=1)
    expected = base_projection[:, None] + 0.5 * projection[combos]
    ceiling = np.empty_like(expected)
    for slot in range(size):
        slots = np.full(valid_rosters, slot, dtype=np.int64)
        variance = _lineup_variance(covariance, combos, slots, chunk)
        ceiling[:, slot] = expected[:, slot] + cfg.candidate_ceiling_weight * np.sqrt(variance)

    flat_expected = expected.ravel()
    flat_ceiling = ceiling.ravel()
    population = flat_ceiling.size

    def top_indices(values, count):
        count = min(int(count), population)
        if count <= 0:
            return np.empty(0, dtype=np.int64)
        return np.argpartition(values, population - count)[population - count:]

    selected = np.union1d(
        top_indices(flat_ceiling, cfg.max_candidate_lineups),
        top_indices(flat_expected, cfg.mean_candidate_reserve),
    )
    roster_index, slot_index = np.divmod(selected, size)
    kept = combos[roster_index]
    variance = _lineup_variance(covariance, kept, slot_index, chunk)

    candidates = pd.DataFrame({
        "Player_Ids": [tuple(int(value) for value in row) for row in kept],
        "Superstar_Id": kept[np.arange(len(kept)), slot_index].astype(int),
        "Salary": combo_salary[roster_index],
        "Expected_FP": flat_expected[selected],
        "Analytic_SD": np.sqrt(variance),
        "Pre_Sim_Score": flat_ceiling[selected],
    })
    return candidates.reset_index(drop=True), valid_rosters


def _lineup_arrays(candidates, lineup_size):
    """Materialize lineup ids once instead of re-parsing them per batch."""
    ids = np.empty((len(candidates), lineup_size), dtype=np.int32)
    for row, value in enumerate(candidates["Player_Ids"]):
        ids[row] = value
    return ids, candidates["Superstar_Id"].to_numpy(np.int32)


def _weight_matrix(ids, superstars, start, stop, player_count):
    """Build the (candidates x players) 1.0/1.5 weight block for one batch.

    v3.5 returns the transpose of the v3.2 block. Scores are now carried as
    (candidates x scenarios) so that every per-candidate reduction - and in
    particular the quantile partition, which dominated the whole run - walks a
    contiguous row instead of striding down a column of a 41 MB array. The
    arithmetic is unchanged; only the memory layout moved.
    """
    weights = np.zeros((stop - start, player_count), dtype=np.float32)
    rows = np.arange(stop - start)
    weights[rows[:, None], ids[start:stop]] = 1.0
    weights[rows, superstars[start:stop]] += 0.5
    return weights


def _tournament_score(ceiling_p90, near_optimal, sim_mean, sim_sd):
    """Blend the four ranking inputs on a common z-scale."""
    def zscore(values):
        values = np.asarray(values, dtype=float)
        scale = values.std()
        return (values - values.mean()) / scale if scale > 1e-12 else np.zeros_like(values)

    return (
        0.40 * zscore(ceiling_p90)
        + 0.25 * zscore(near_optimal)
        + 0.25 * zscore(sim_mean)
        + 0.10 * zscore(sim_sd)
    )


def score_candidates_shared_scenarios(candidates, outcomes, cfg=None, batch_size=512):
    """Two passes: distribution summaries, then performance vs each scenario's optimum.

    v3.2 also computes every metric separately on each half of the shared scenarios.
    Those half-sample copies cost almost nothing because the batch scores are already
    in memory, and they let `reliability_report` measure how much of the final ranking
    is signal rather than Monte Carlo noise.
    """
    cfg = _cfg(cfg)
    n_candidates = len(candidates)
    n_scenarios, n_players = outcomes.shape
    half = n_scenarios // 2
    halves = {"": slice(None), "_A": slice(0, half), "_B": slice(half, 2 * half)}
    ids, superstars = _lineup_arrays(candidates, int(cfg.lineup_size))
    # Built once. Every batch multiplies against this rather than transposing a
    # fresh score block, which would give back the layout win it is here for.
    scenarios = np.ascontiguousarray(outcomes.T)

    names = ["Sim_Mean", "Sim_SD", "Floor_P25", "Ceiling_P90", "Ceiling_P95"]
    metrics = {f"{name}{tag}": np.zeros(n_candidates) for name in names for tag in halves}
    scenario_best = np.full(n_scenarios, -np.inf, dtype=np.float32)

    for start in range(0, n_candidates, batch_size):
        stop = min(start + batch_size, n_candidates)
        scores = _weight_matrix(ids, superstars, start, stop, n_players) @ scenarios
        for tag, window in halves.items():
            block = scores[:, window]
            # float64 accumulators. The scores themselves are float32, and a
            # float32 sum over 20,000 scenarios lands ~1e-5 relative away from
            # the exact mean in an order that depends on the array layout - which
            # was enough to reshuffle near-tied candidates when the layout above
            # changed. Accumulating in float64 makes the reduction layout-
            # independent, and makes these numbers reproducible by any float64
            # implementation, the browser included.
            metrics[f"Sim_Mean{tag}"][start:stop] = block.mean(axis=1, dtype=np.float64)
            metrics[f"Sim_SD{tag}"][start:stop] = block.std(axis=1, dtype=np.float64)
            quantiles = np.quantile(block, [0.25, 0.90, 0.95], axis=1)
            metrics[f"Floor_P25{tag}"][start:stop] = quantiles[0]
            metrics[f"Ceiling_P90{tag}"][start:stop] = quantiles[1]
            metrics[f"Ceiling_P95{tag}"][start:stop] = quantiles[2]
        np.maximum(scenario_best, scores.max(axis=0), out=scenario_best)

    rates = {f"{name}{tag}": np.zeros(n_candidates)
             for name in ("Near_Optimal_Rate", "Win_Rate") for tag in halves}
    for start in range(0, n_candidates, batch_size):
        stop = min(start + batch_size, n_candidates)
        scores = _weight_matrix(ids, superstars, start, stop, n_players) @ scenarios
        near = scores >= (scenario_best[None, :] * cfg.near_optimal_ratio)
        won = np.isclose(scores, scenario_best[None, :], rtol=1e-6, atol=1e-5)
        for tag, window in halves.items():
            rates[f"Near_Optimal_Rate{tag}"][start:stop] = near[:, window].mean(axis=1)
            rates[f"Win_Rate{tag}"][start:stop] = won[:, window].mean(axis=1)

    scored = candidates.copy()
    for name, values in {**metrics, **rates}.items():
        scored[name] = values

    # Binomial Monte Carlo standard errors. Both rates are tiny (a top lineup is
    # near-optimal in well under 1% of scenarios), so their sampling error is a
    # large fraction of the spread between candidates. Report it beside the value.
    for name in ("Near_Optimal_Rate", "Win_Rate"):
        rate = scored[name].to_numpy()
        scored[f"{name}_SE"] = np.sqrt(np.maximum(rate * (1 - rate), 0.0) / max(n_scenarios, 1))

    for tag in halves:
        scored[f"Tournament_Score{tag}"] = _tournament_score(
            scored[f"Ceiling_P90{tag}"], scored[f"Near_Optimal_Rate{tag}"],
            scored[f"Sim_Mean{tag}"], scored[f"Sim_SD{tag}"],
        )
    scored.attrs["simulations"] = n_scenarios
    return scored.sort_values("Tournament_Score", ascending=False).reset_index(drop=True)


RELIABILITY_METRICS = [
    "Sim_Mean", "Floor_P25", "Ceiling_P90", "Ceiling_P95",
    "Sim_SD", "Near_Optimal_Rate", "Win_Rate", "Tournament_Score",
]


def reliability_report(scored, top_n=20):
    """Estimate how much of each ranking metric is signal rather than sampling noise.

    Each metric was computed twice, on two independent halves of the shared
    scenarios. Their rank correlation is the reliability at half the sample size;
    the Spearman-Brown formula 2r / (1 + r) up-corrects it to the full run. A metric
    near 1.0 would rank the same candidates the same way under a different seed; a
    metric near 0 is re-ranking noise and should not drive lineup selection.
    """
    rows = []
    for metric in RELIABILITY_METRICS:
        left, right = f"{metric}_A", f"{metric}_B"
        if left not in scored or right not in scored:
            continue
        a, b = scored[left], scored[right]
        if a.std() < 1e-12 or b.std() < 1e-12:
            correlation = float("nan")
        else:
            correlation = float(np.corrcoef(a.rank(), b.rank())[0, 1])
        full = 2 * correlation / (1 + correlation) if correlation > -1 else float("nan")
        overlap = len(set(a.nlargest(top_n).index) & set(b.nlargest(top_n).index))
        rows.append({
            "Metric": metric,
            "Split-half rank corr": round(correlation, 3),
            "Full-run reliability": round(full, 3),
            f"Top-{top_n} overlap": f"{overlap}/{top_n}",
            "Verdict": (
                "stable" if full >= 0.90 else
                "usable" if full >= 0.75 else
                "NOISY - do not rank on this alone"
            ),
        })
    return pd.DataFrame(rows)


def analytic_vs_simulated_lineup_check(scored, sample=2_000, seed=0):
    """Verify the analytic screen agrees with the simulation it is screening for.

    Candidates are enumerated and pruned with the analytic covariance, then ranked
    with the simulated draws. If the two disagreed, the pruning step would be
    discarding lineups the final objective would have liked. This compares the two
    descriptions on the retained candidates.
    """
    rng = np.random.default_rng(seed)
    take = min(int(sample), len(scored))
    rows = scored.iloc[rng.choice(len(scored), size=take, replace=False)]
    mean_error = (rows["Sim_Mean"] - rows["Expected_FP"]).abs() / rows["Expected_FP"].abs().clip(lower=1e-9)
    sd_error = (rows["Sim_SD"] - rows["Analytic_SD"]).abs() / rows["Analytic_SD"].abs().clip(lower=1e-9)
    return pd.DataFrame([{
        "Checked candidates": take,
        "Mean: median abs error %": round(100 * float(mean_error.median()), 3),
        "Mean: worst abs error %": round(100 * float(mean_error.max()), 3),
        "SD: median abs error %": round(100 * float(sd_error.median()), 3),
        "SD: worst abs error %": round(100 * float(sd_error.max()), 3),
        "Mean rank corr": round(float(np.corrcoef(rows["Expected_FP"].rank(), rows["Sim_Mean"].rank())[0, 1]), 4),
        "SD rank corr": round(float(np.corrcoef(rows["Analytic_SD"].rank(), rows["Sim_SD"].rank())[0, 1]), 4),
        "Sanity": "PASS" if float(mean_error.max()) < 0.05 and float(sd_error.max()) < 0.12 else "REVIEW",
    }])


# ============================================================================
# NOTEBOOK CELL 15 - Strongest lineup, tournament portfolio, and exports
# ============================================================================
def select_strongest_lineups(scored, count=10):
    """Rank by analytic expected points, which is seed-independent.

    The tie-breakers are simulated, so v3.2 adds Player_Ids as a final deterministic
    key. Without it two rosters with identical expected points could swap places
    purely because of the random seed.
    """
    ordered = scored.copy()
    ordered["_tiebreak"] = [tuple(ids) for ids in ordered["Player_Ids"]]
    ordered = ordered.sort_values(
        ["Expected_FP", "Floor_P25", "Ceiling_P90", "_tiebreak", "Superstar_Id"],
        ascending=[False, False, False, True, True],
    )
    return ordered.drop(columns="_tiebreak").head(count).reset_index(drop=True)


def select_tournament_portfolio(scored, cfg=None):
    cfg = _cfg(cfg)
    target = cfg.tournament_lineups
    max_player_count = exposure_limit(target, cfg.max_player_exposure)
    max_superstar_count = exposure_limit(target, cfg.max_superstar_exposure)
    player_counts = Counter()
    superstar_counts = Counter()
    selected_rows = []
    selected_sets = []

    for idx, candidate in scored.sort_values("Tournament_Score", ascending=False).iterrows():
        ids = tuple(candidate["Player_Ids"])
        superstar = int(candidate["Superstar_Id"])
        candidate_set = set(ids)
        if any(player_counts[player] >= max_player_count for player in ids):
            continue
        if superstar_counts[superstar] >= max_superstar_count:
            continue
        if any(len(candidate_set & prior) > cfg.max_shared_players for prior in selected_sets):
            continue
        selected_rows.append(idx)
        selected_sets.append(candidate_set)
        player_counts.update(ids)
        superstar_counts.update([superstar])
        if len(selected_rows) == target:
            break

    portfolio = scored.loc[selected_rows].copy().reset_index(drop=True)
    if len(portfolio) < target:
        warnings.warn(
            f"Built {len(portfolio)} of {target} tournament lineups under the current "
            "exposure/overlap caps. Relax a cap if you need the full count."
        )
    return portfolio


def portfolio_diversity_report(portfolio, cfg=None):
    """Show how different the entries really are.

    On a five-player roster `max_shared_players = 4` allows two entries to differ by
    a single player, which is a legitimate choice for a small field but rarely what
    is wanted from twenty tournament entries. v3.2 defaults to 3; this report states
    the overlap actually achieved so the setting is never invisible.
    """
    cfg = _cfg(cfg)
    sets = [set(ids) for ids in portfolio["Player_Ids"]]
    if len(sets) < 2:
        return pd.DataFrame([{"Lineups": len(sets), "Note": "Too few lineups to compare."}])
    overlaps = [
        len(a & b) for a, b in itertools.combinations(sets, 2)
    ]
    counts = Counter(overlaps)
    size = int(cfg.lineup_size)
    near_duplicate = counts.get(size - 1, 0)
    return pd.DataFrame([{
        "Lineups": len(sets),
        "Distinct players used": len(set().union(*sets)),
        "Mean shared players": round(float(np.mean(overlaps)), 2),
        "Max shared players": int(max(overlaps)),
        "Cap in use": int(cfg.max_shared_players),
        "Pairs at the cap": counts.get(int(cfg.max_shared_players), 0),
        f"Pairs sharing {size - 1} of {size}": near_duplicate,
        "Note": (
            "Near-duplicate entries present; lower Settings.max_shared_players."
            if near_duplicate
            else f"Every pair of entries differs by at least "
                 f"{size - int(max(overlaps))} player(s)."
        ),
    }])


def _player_label(players, player_id):
    """Return a compact label that remains readable in wide ranking tables."""
    player = players.iloc[int(player_id)]
    return f"{player['Name']} ({player['Position']}-{player['Team']})"


def _lineup_team_split(players, ids):
    """Describe roster concentration without implying that a 3-2 split is required."""
    counts = players.iloc[list(ids)]["Team"].value_counts()
    return " / ".join(f"{team} {int(count)}" for team, count in counts.items())


def _lineup_construction(players, ids):
    """Summarize actual QB stacks and game-stack structure in plain language."""
    selected = players.iloc[list(ids)]
    notes = []
    qbs = selected[selected["Position"].eq("QB")]
    for _, quarterback in qbs.iterrows():
        receivers = selected[
            selected["Team"].eq(quarterback["Team"])
            & selected["Position"].isin(["WR", "TE"])
        ]
        if len(receivers):
            notes.append(f"{quarterback['Team']} QB + {len(receivers)} WR/TE")
    if len(qbs) == 2:
        notes.append("opposing-QB game stack")
    if not notes:
        notes.append("no QB-receiver stack")
    return "; ".join(notes)


def _lineup_risk_notes(players, ids, salary_left, salary_cap):
    """Flag assumptions worth checking; flags are diagnostics, not hard bans."""
    selected = players.iloc[list(ids)]
    notes = []
    deep = selected[
        selected["Depth_Rank"].ge(3) & ~selected["Position"].eq("DEF")
    ]
    if len(deep):
        notes.append("deep role: " + ", ".join(deep["Name"].tolist()))
    zero_history = selected[
        selected["Projection_Source"].astype(str).str.contains("zero/low FPPG")
    ]
    if len(zero_history):
        notes.append("no FPPG history: " + ", ".join(zero_history["Name"].tolist()))
    backups = selected[
        selected["Position"].eq("QB") & selected["Depth_Rank"].gt(1)
    ]
    if len(backups):
        notes.append("verify backup-QB snaps")
    for _, defense in selected[selected["Position"].eq("DEF")].iterrows():
        conflicts = selected[
            selected["Team"].eq(defense["Opponent"])
            & ~selected["Position"].eq("DEF")
        ]
        if len(conflicts):
            notes.append(f"{defense['Team']} DEF faces {len(conflicts)} selected opponent(s)")
    if salary_cap and salary_left >= 0.10 * salary_cap:
        notes.append("10%+ salary unused")
    return "; ".join(dict.fromkeys(notes)) or "no structural flags"


def lineup_summary(lineups, players, label, salary_cap, rank_start=1):
    """Create an entry-oriented ranking table with distinct mean and tail metrics.

    `Scenario-best %` is the fraction of shared simulations in which the lineup tied
    for the top score among retained candidates. It is not contest win probability
    (ownership and the opponent field are not modeled) and, per the reliability
    report, it is the noisiest column here, so its Monte Carlo standard error is
    printed beside it.
    """
    rows = []
    for number, candidate in lineups.reset_index(drop=True).iterrows():
        ids = list(candidate["Player_Ids"])
        superstar = int(candidate["Superstar_Id"])
        flex = sorted(
            [idx for idx in ids if idx != superstar],
            key=lambda idx: float(players.iloc[idx]["Projected_FP"]),
            reverse=True,
        )
        salary_left = float(salary_cap) - float(candidate["Salary"])
        row = {
            "Use": f"{label} #{number + rank_start}",
            "Superstar (1.5x)": _player_label(players, superstar),
        }
        for slot, player_id in enumerate(flex, 1):
            row[f"Flex {slot}"] = _player_label(players, player_id)
        near = 100 * float(candidate["Near_Optimal_Rate"])
        near_se = 100 * float(candidate.get("Near_Optimal_Rate_SE", float("nan")))
        best = 100 * float(candidate["Win_Rate"])
        best_se = 100 * float(candidate.get("Win_Rate_SE", float("nan")))
        row.update({
            "Team split": _lineup_team_split(players, ids),
            "Construction": _lineup_construction(players, ids),
            "Salary used": round(float(candidate["Salary"]), 2),
            "Salary left": round(salary_left, 2),
            "Projected mean": round(float(candidate["Expected_FP"]), 2),
            "Simulated mean": round(float(candidate["Sim_Mean"]), 2),
            "P25 floor": round(float(candidate["Floor_P25"]), 2),
            "P90 ceiling": round(float(candidate["Ceiling_P90"]), 2),
            "P95 ceiling": round(float(candidate["Ceiling_P95"]), 2),
            "Near-optimal %": round(near, 2),
            "Near-optimal +/-": round(near_se, 3),
            "Scenario-best %": round(best, 3),
            "Scenario-best +/-": round(best_se, 3),
            "Tournament score": round(float(candidate["Tournament_Score"]), 3),
            "Review": _lineup_risk_notes(players, ids, salary_left, salary_cap),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def lineup_comparison_view(summary):
    """Keep notebook displays compact while CSV exports retain all diagnostics."""
    columns = [
        "Use", "Superstar (1.5x)", "Flex 1", "Flex 2", "Flex 3", "Flex 4",
        "Salary left", "Projected mean", "P25 floor", "P90 ceiling",
        "Near-optimal %", "Near-optimal +/-", "Review",
    ]
    return summary[[column for column in columns if column in summary.columns]]


def lineup_entry_rows(lineups, players, label):
    """Expand lineups into Yahoo-entry order and expose every modeling input."""
    rows = []
    for number, candidate in lineups.reset_index(drop=True).iterrows():
        ids = list(candidate["Player_Ids"])
        superstar = int(candidate["Superstar_Id"])
        flex = sorted(
            [idx for idx in ids if idx != superstar],
            key=lambda idx: float(players.iloc[idx]["Projected_FP"]),
            reverse=True,
        )
        ordered = [superstar] + flex
        for slot_number, player_id in enumerate(ordered):
            player = players.iloc[player_id]
            multiplier = 1.5 if slot_number == 0 else 1.0
            raw_projection = player.get(
                "Pre_Depth_Projected_FP", player["Projected_FP"]
            )
            rows.append({
                "Use": label,
                "Lineup": number + 1,
                "Yahoo slot": "SUPERSTAR" if slot_number == 0 else f"FLEX {slot_number}",
                "Player": player["Name"],
                "Position": player["Position"],
                "Team": player["Team"],
                "Salary": round(float(player["Salary"]), 2),
                "Raw projection": round(float(raw_projection), 2),
                "Depth mean factor": round(float(player.get("Depth_Mean_Multiplier", 1.0)), 2),
                "Adjusted projection": round(float(player["Projected_FP"]), 2),
                "Lineup multiplier": multiplier,
                "Projected contribution": round(float(player["Projected_FP"]) * multiplier, 2),
                "Calibrated CV": round(_calibrated_cv(player["Position"], player["Depth_Rank"]), 3),
                "Depth rank": int(player["Depth_Rank"]),
                "Depth source": player["Depth_Source"],
            })
    return pd.DataFrame(rows)


def display_best_lineup(strongest, players, salary_cap):
    """Show the recommended single entry before any alternative rankings."""
    if strongest.empty:
        print("No strongest lineup was produced.")
        return
    print("\nRECOMMENDED SINGLE ENTRY - highest adjusted projected mean")
    summary = lineup_summary(strongest.head(1), players, "Recommended", salary_cap)
    entry = lineup_entry_rows(strongest.head(1), players, "recommended")
    display(entry[[
        "Yahoo slot", "Player", "Position", "Team", "Salary",
        "Adjusted projection", "Lineup multiplier", "Projected contribution",
        "Depth rank",
    ]])
    display(summary[[
        "Salary used", "Salary left", "Projected mean", "Simulated mean",
        "P25 floor", "P90 ceiling", "P95 ceiling",
        "Near-optimal %", "Near-optimal +/-", "Scenario-best %", "Scenario-best +/-",
    ]])
    print(f"Construction: {summary.iloc[0]['Construction']}")
    print(f"Review: {summary.iloc[0]['Review']}")


def display_alternatives(strongest, players, salary_cap, count=4):
    """Show a short comparison set without burying the primary recommendation."""
    alternatives = strongest.iloc[1:1 + count]
    if len(alternatives):
        print("\nNEXT-BEST SINGLE-ENTRY ALTERNATIVES")
        summary = lineup_summary(
            alternatives, players, "Single-entry rank", salary_cap, rank_start=2
        )
        display(lineup_comparison_view(summary))


def exposure_report(portfolio, players):
    """Summarize player and Superstar usage across the portfolio.

    Lineup ids are positional, so v3.2 iterates positions rather than index labels.
    The v3.1 version used `players.iterrows()` and silently reported every exposure
    as zero whenever the caller passed a frame whose index was not already 0..n-1.
    """
    total = len(portfolio)
    player_count = Counter()
    superstar_count = Counter()
    for _, candidate in portfolio.iterrows():
        player_count.update(int(value) for value in candidate["Player_Ids"])
        superstar_count.update([int(candidate["Superstar_Id"])])
    rows = []
    for position in range(len(players)):
        player = players.iloc[position]
        rows.append({
            "Player": player["Name"],
            "Position": player["Position"],
            "Team": player["Team"],
            "Lineups": player_count[position],
            "Exposure %": round(100 * player_count[position] / total, 1) if total else 0.0,
            "Superstar lineups": superstar_count[position],
            "Superstar %": round(100 * superstar_count[position] / total, 1) if total else 0.0,
        })
    return pd.DataFrame(rows).sort_values(
        ["Exposure %", "Superstar %", "Player"], ascending=[False, False, True]
    ).reset_index(drop=True)


def export_results(
    output_dir, strongest, portfolio, players, corr_summary, corr_detail, salary_cap,
    extra_frames=None,
):
    """Write both readable rankings and exact Yahoo-entry rows to one ZIP bundle."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    strongest_summary = lineup_summary(strongest, players, "Strongest", salary_cap)
    tournament_summary = lineup_summary(portfolio, players, "Tournament", salary_cap)
    # v3.2 exports every ranked single-entry alternative, not only the top one.
    strongest_entries = lineup_entry_rows(strongest, players, "strongest")
    tournament_entries = lineup_entry_rows(portfolio, players, "tournament")
    exposures = exposure_report(portfolio, players)

    files = {
        "strongest_lineup.csv": strongest_entries,
        "strongest_rankings.csv": strongest_summary,
        "tournament_lineups.csv": tournament_entries,
        "tournament_rankings.csv": tournament_summary,
        "player_exposures.csv": exposures,
        "player_projections.csv": players,
        "correlation_summary.csv": corr_summary,
        "correlation_detail.csv": corr_detail,
    }
    files.update(extra_frames or {})
    for filename, frame in files.items():
        frame.to_csv(output_dir / filename, index=False)
    zip_path = Path(shutil.make_archive(str(output_dir), "zip", root_dir=output_dir))
    return {name: str(output_dir / name) for name in files} | {"zip": str(zip_path)}


# ============================================================================
# NOTEBOOK CELL 17 - End-to-end showdown runner
# ============================================================================
def run_interactive(cfg=None):
    """Fetch one Yahoo slate, audit assumptions, simulate, rank, and export.

    The order is deliberate: depth is assigned from the unadjusted projection,
    then its historical mean bias is corrected, then unconfirmed backup QBs and
    user exclusions are removed. This prevents adjusted means from redefining the
    very depth role used to choose the adjustment.
    """
    cfg = _cfg(cfg)
    started = time.perf_counter()
    payload = fetch_yahoo_data()
    all_players, cap_map = normalize_yahoo_data(payload)
    all_players = add_projection_priors(all_players, PROJECTION_OVERRIDES)
    games = list_games(all_players)
    selected = select_game_interactive(games)
    salary_cap = salary_cap_for_game(cap_map, selected["Game ID"])

    players = all_players[all_players["Game ID"].eq(str(selected["Game ID"]))].copy()

    players = assign_depth_assumptions(players, DEPTH_OVERRIDES, PLAYER_STYLE_OVERRIDES)

    print(f"\n{selected['Matchup']} - cap ${salary_cap:g}")

    # nflverse role and availability, applied before the depth mean adjustment so the
    # corrected rank drives both the mean multiplier and the volatility prior.
    nflverse_report = pd.DataFrame()
    nflverse_applied = pd.DataFrame()
    nflverse_blocked = pd.DataFrame()
    if cfg.use_nflverse:
        season = cfg.nflverse_season or infer_season(selected["Game Time"])
        depth_chart, roster_status, as_of, notes = load_nflverse_reference(players, season, cfg)
        print("\nnflverse reference feed:")
        for note in notes:
            print(f"  - {note}")
        if depth_chart is not None or roster_status is not None:
            nflverse_report = build_nflverse_role_report(
                players, depth_chart, roster_status, cfg
            )
            players, nflverse_applied = apply_nflverse_roles(players, nflverse_report, cfg)
            matched = int(nflverse_report["Depth matched"].sum())
            skill = int(nflverse_report["Position"].ne("DEF").sum())
            statused = int(nflverse_report["Roster status"].notna().sum())
            print(
                f"  - matched {matched}/{skill} skill players to a depth-chart entry, "
                f"{statused}/{skill} to a roster status ({roster_match_rate(nflverse_report):.0%})"
            )
            print(f"  - role tiers taken from the published chart: {len(nflverse_applied)}")
            starters = int(nflverse_report["Role"].eq("starter").sum())
            print(f"  - players the chart lists in a starting slot: {starters}")
            review = nflverse_disagreement_view(nflverse_report)
            if len(review):
                print(
                    "\nnflverse disagreements and unmatched players - review before "
                    "trusting the run:"
                )
                display(review)
    else:
        print("\nnflverse cross-check disabled (Settings.use_nflverse = False).")

    players = apply_opportunity_ranks(players, cfg)
    players = add_projection_priors(players, PROJECTION_OVERRIDES)
    print(
        "\nRole tier drives the fitted mean correction and expected-opportunity "
        "rank drives the volatility prior; verify injuries, actives, and snaps."
    )
    display(depth_sanity_report(players))

    if cfg.use_nflverse and len(nflverse_report):
        players, nflverse_blocked = apply_nflverse_availability(players, nflverse_report, cfg)
        if len(nflverse_blocked):
            print(
                f"\nAVAILABILITY FILTER removed {len(nflverse_blocked)} player(s) "
                f"(allowed status: {', '.join(cfg.nflverse_available_status)}):"
            )
            display(nflverse_blocked)
        else:
            print("\nAvailability filter: every priced player is on an allowed roster status.")
    players, auto_removed_qbs = apply_default_role_filters(
        players, INCLUDE_BACKUP_QBS, cfg
    )
    if auto_removed_qbs:
        print(
            "Default backup-QB filter removed: " + ", ".join(auto_removed_qbs)
            + ". Add a confirmed replacement to INCLUDE_BACKUP_QBS (and give it a "
            "DEPTH_OVERRIDES entry of 1) to retain it."
        )
    players = apply_exclusions_interactive(players, EXCLUDE_PLAYERS)
    eligible_player_count = len(players)
    players, dropped = trim_player_pool(players, cfg)
    if dropped:
        print(
            f"Enumeration cap retained {len(players)} of {eligible_player_count} "
            f"eligible players and removed {len(dropped)} fringe players: "
            + ", ".join(dropped)
        )
    problem = roster_feasibility_error(players, cfg.lineup_size)
    if problem:
        raise ValueError(problem)

    # Keep all matrices aligned to this exact RangeIndex.
    players = players.reset_index(drop=True)
    outcomes, correlation_model = simulate_player_outcomes(players, cfg)
    corr_summary, corr_detail = correlation_sanity_report(
        players, outcomes, correlation_model
    )

    print("\nCross-position correlation sanity check:")
    focus_corr = corr_summary[
        corr_summary["Relationship"].isin(CALIBRATION_SANITY_RELATIONSHIPS)
    ]
    display(focus_corr)
    print(
        "Largest pairwise score-correlation change required for a valid joint "
        f"matrix: {correlation_model['psd_max_score_adjustment']:.3f}."
    )
    if correlation_model["infeasible_pairs"]:
        print(
            f"{correlation_model['infeasible_pairs']} target(s) were unattainable for "
            "lognormals with these CVs and were clipped (largest shift "
            f"{correlation_model['max_infeasible_shift']:.3f})."
        )
    review = focus_corr[focus_corr["Sanity"].eq("REVIEW")]
    if len(review):
        warnings.warn(
            "One or more displayed relationship groups differs materially from "
            "its fitted target after PSD repair or Monte Carlo sampling."
        )

    print("\nPer-player marginal check (simulator reproduced the intended mean and CV):")
    marginals = marginal_sanity_report(players, outcomes)
    flagged = marginals[marginals["Sanity"].eq("REVIEW")]
    display(flagged if len(flagged) else marginals.head(5))
    if len(flagged):
        warnings.warn(f"{len(flagged)} player marginal(s) drifted from the intended mean or CV.")
    else:
        print(f"All {len(marginals)} players within 3% on the mean and 8% on the CV.")

    covariance = analytic_covariance(players, correlation_model)
    candidates, valid_rosters = enumerate_candidate_lineups(players, salary_cap, covariance, cfg)
    print(
        f"\nYahoo-valid base rosters: {valid_rosters:,}; "
        f"retained Superstar candidates: {len(candidates):,}; "
        f"shared simulations: {cfg.simulations:,}."
    )
    scored = score_candidates_shared_scenarios(candidates, outcomes, cfg)

    screen_check = analytic_vs_simulated_lineup_check(scored)
    print("\nAnalytic screen vs simulation (the pruning step is ranking the same thing):")
    display(screen_check)

    reliability = reliability_report(scored, top_n=cfg.tournament_lineups)
    if cfg.report_reliability:
        print(
            "\nMonte Carlo reliability (split-half, Spearman-Brown corrected). "
            "A low value means the column re-ranks differently under another seed:"
        )
        display(reliability)
        weak = reliability[reliability["Verdict"].str.startswith("NOISY")]["Metric"].tolist()
        if weak:
            print(
                "Noisy at this sample size: " + ", ".join(weak)
                + ". Raise Settings.simulations, or read those columns as approximate."
            )

    strongest = select_strongest_lineups(scored, count=10)
    portfolio = select_tournament_portfolio(scored, cfg)

    display_best_lineup(strongest, players, salary_cap)
    display_alternatives(strongest, players, salary_cap)
    print("\nDIVERSIFIED TOURNAMENT PORTFOLIO")
    tournament_view = lineup_summary(portfolio, players, "Tournament", salary_cap)
    display(lineup_comparison_view(tournament_view))
    diversity = portfolio_diversity_report(portfolio, cfg)
    print("\nPORTFOLIO DIVERSITY")
    display(diversity)
    print("\nTOURNAMENT EXPOSURE")
    exposures = exposure_report(portfolio, players)
    display(exposures)

    safe_matchup = selected["Matchup"].replace(" ", "_").replace("/", "-")
    output_dir = f"yahoo_showdown_{safe_matchup}"
    files = export_results(
        output_dir, strongest, portfolio, players, corr_summary, corr_detail,
        salary_cap,
        extra_frames={
            "reliability_report.csv": reliability,
            "analytic_vs_simulated.csv": screen_check,
            "player_marginals.csv": marginals,
            "portfolio_diversity.csv": diversity,
            **({"nflverse_role_report.csv": nflverse_report} if len(nflverse_report) else {}),
            **({"nflverse_removed.csv": nflverse_blocked} if len(nflverse_blocked) else {}),
        },
    )
    print(f"\nSaved results bundle: {files['zip']}")
    print(f"Total run time: {time.perf_counter() - started:.1f}s")
    return {
        "game": selected.to_dict(),
        "salary_cap": salary_cap,
        "players": players,
        "outcomes": outcomes,
        "correlation_model": correlation_model,
        # Backward-compatible alias for code written against notebook v2.
        "factor_model": correlation_model,
        "correlation_summary": corr_summary,
        "correlation_detail": corr_detail,
        "marginals": marginals,
        "nflverse_report": nflverse_report,
        "nflverse_depth_applied": nflverse_applied,
        "nflverse_removed": nflverse_blocked,
        "reliability": reliability,
        "screen_check": screen_check,
        "diversity": diversity,
        "exposures": exposures,
        "scored_candidates": scored,
        "strongest": strongest,
        "portfolio": portfolio,
        "files": files,
    }


# ============================================================================
# NOTEBOOK CELL 19 - Weekly top-N rankings by position
# ============================================================================
def prepare_slate_pool(cfg=None, purpose=""):
    """Build the priced, role-adjusted, availability-filtered pool for a slate.

    Yahoo supplies prices, the frozen historical model supplies means, and
    nflverse supplies depth, roster status and injury reports. Every consumer
    receives this exact final frame.

    The position rankings and the showdown export both need exactly this and had
    started to drift apart as two copies of it.
    """
    cfg = _cfg(cfg)
    payload = fetch_yahoo_data()
    players, cap_map = normalize_yahoo_data(payload)
    # Preliminary salary depth makes the model usable if nflverse is unavailable.
    players = add_projection_priors(players, PROJECTION_OVERRIDES)
    raw_inputs = {"yahoo": payload}
    games = list_games(players)
    if games.empty:
        raise ValueError("Yahoo returned no usable NFL games")

    print(
        f"Yahoo slate: {len(games)} game(s), {len(players)} priced player(s)"
        + (f"; {purpose}" if purpose else "")
    )

    players = assign_depth_assumptions(
        players, DEPTH_OVERRIDES, PLAYER_STYLE_OVERRIDES
    )

    nflverse_report = pd.DataFrame()
    nflverse_applied = pd.DataFrame()
    nflverse_blocked = pd.DataFrame()
    if cfg.use_nflverse:
        first_kickoff = games.iloc[0]["Game Time"]
        season = cfg.nflverse_season or infer_season(first_kickoff)
        depth_chart, roster_status, as_of, notes = load_nflverse_reference(
            players, season, cfg
        )
        for note in notes:
            print(f"  nflverse: {note}")
        if depth_chart is not None or roster_status is not None:
            nflverse_report = build_nflverse_role_report(
                players, depth_chart, roster_status, cfg
            )
            players, nflverse_applied = apply_nflverse_roles(
                players, nflverse_report, cfg
            )
            print(
                f"  nflverse: role tiers for {len(nflverse_applied)} player(s); "
                f"roster status matched {roster_match_rate(nflverse_report):.0%} "
                "of the skill pool."
            )

    players = apply_opportunity_ranks(players, cfg)
    # Recompute after the chart establishes the final role tier. The fitted model
    # already contains the depth effect, so there is no second depth haircut.
    players = add_projection_priors(players, PROJECTION_OVERRIDES)
    players["Baseline_Projected_FP"] = players["Projected_FP"]

    if cfg.use_nflverse and not nflverse_report.empty:
        players, nflverse_blocked = apply_nflverse_availability(
            players, nflverse_report, cfg
        )
        if len(nflverse_blocked):
            print(
                f"  Availability filter removed {len(nflverse_blocked)} player(s)."
            )

    injury_report = pd.DataFrame()
    injury_removed = pd.DataFrame()
    if cfg.use_nflverse:
        try:
            injury_report, injury_week = fetch_nflverse_injury_report(season, cfg)
            players, injury_removed = apply_nflverse_injuries(players, injury_report)
            print(
                f"  nflverse: injury report week {injury_week}, "
                f"{len(injury_report)} rows; removed {len(injury_removed)} listed Out."
            )
        except Exception as exc:
            warnings.warn(f"nflverse injury report unavailable: {exc}")

    players, backup_qbs_removed = apply_default_role_filters(
        players, INCLUDE_BACKUP_QBS, cfg
    )
    if backup_qbs_removed:
        print(f"  Backup-QB filter removed {len(backup_qbs_removed)} player(s).")

    excluded = set(EXCLUDE_PLAYERS)
    _warn_unmatched(excluded, "Exclusion", set(players["Name"]))
    players = players[~players["Name"].isin(excluded)].copy()
    return {
        "players": players.reset_index(drop=True),
        "raw_inputs": raw_inputs,
        "projection_model": salary_projection.load(),
        "inputs_captured_utc": datetime.now(timezone.utc).isoformat(),
        "games": games,
        "cap_map": cap_map,
        "nflverse_report": nflverse_report,
        "nflverse_applied": nflverse_applied,
        "nflverse_removed": nflverse_blocked,
        "injury_report": injury_report,
        "injury_removed": injury_removed,
        "backup_qbs_removed": backup_qbs_removed,
        "excluded": sorted(excluded),
    }


def run_position_rankings(
    top_n=25,
    cfg=None,
    positions=VALID_POSITIONS,
    export_csv=True,
    prepared_slate=None,
):
    """Build full-slate Yahoo rankings from the notebook's final estimated FP.

    Projection priority is manual override, then the season-frozen Yahoo
    salary-position-depth regression shared by every product.
    """
    cfg = _cfg(cfg)
    top_n = int(top_n)
    if top_n < 1:
        raise ValueError("top_n must be at least 1")

    requested_positions = []
    for position in positions:
        normalized = str(position).upper().replace("D/ST", "DEF").replace("DST", "DEF")
        if normalized not in VALID_POSITIONS:
            raise ValueError(
                f"Unsupported position {position!r}; choose from {VALID_POSITIONS}"
            )
        if normalized not in requested_positions:
            requested_positions.append(normalized)

    started = time.perf_counter()
    # A publisher that serves more than one page can prepare the live feeds once
    # and hand the exact same frame to every view.  Keeping the default here
    # preserves the standalone/Colab API, while `run_synced.py` uses the injected
    # slate to prevent the rankings and weekly-lineup pages from capturing
    # different input snapshots.
    slate = (
        prepared_slate
        if prepared_slate is not None
        else prepare_slate_pool(cfg, f"ranking top {top_n} per position")
    )
    players = slate["players"]
    games = slate["games"]
    nflverse_report = slate["nflverse_report"]
    nflverse_applied = slate["nflverse_applied"]
    nflverse_blocked = slate["nflverse_removed"]
    players["FP_per_Salary"] = players["Projected_FP"] / players["Salary"]

    # Stable tie-breaks make repeated runs deterministic when estimates are equal.
    ordered = players.sort_values(
        ["Position", "Projected_FP", "FPPG", "Salary", "Name"],
        ascending=[True, False, False, False, True],
    ).copy()

    tables = {}
    combined = []
    for position in requested_positions:
        group = ordered[ordered["Position"].eq(position)].head(top_n).copy()
        group.insert(0, "Rank", np.arange(1, len(group) + 1))
        group["Estimated FP"] = group["Projected_FP"].round(2)
        group["FP / salary"] = group["FP_per_Salary"].round(3)
        group["Yahoo FPPG"] = group["FPPG"].round(2)
        group["Kickoff UTC"] = pd.to_datetime(
            group["Game Time"], errors="coerce", utc=True
        ).dt.strftime("%Y-%m-%d %H:%M")
        group["Projection method"] = group["Projection_Source"].astype(str)

        group["Role"] = group["Role_Label"].astype(str)
        group["Role slot"] = group["Role_Slot"].astype("string").fillna("")

        columns = [
            "Rank", "Name", "Team", "Opponent", "Estimated FP", "Yahoo FPPG",
            "Salary", "FP / salary", "Role", "Role slot", "Depth_Rank",
            "Kickoff UTC", "Projection method",
        ]
        view = group[columns].rename(
            columns={"Name": "Player", "Depth_Rank": "Depth"}
        ).reset_index(drop=True)
        tables[position] = view
        combined.append(view.assign(Position=position))

        print(f"\nTOP {min(top_n, len(view))} {position}")
        display(view)

    combined_table = (
        pd.concat(combined, ignore_index=True)
        if combined
        else pd.DataFrame()
    )
    if not combined_table.empty:
        combined_table = combined_table[
            ["Position"] + [c for c in combined_table.columns if c != "Position"]
        ]

    csv_path = None
    if export_csv:
        csv_path = f"yahoo_top_{top_n}_by_position.csv"
        combined_table.to_csv(csv_path, index=False)
        print(f"\nSaved combined rankings: {csv_path}")

    print(f"Ranking run time: {time.perf_counter() - started:.1f}s")
    return {
        "rankings": tables,
        "combined": combined_table,
        "players": players.reset_index(drop=True),
        "games": games,
        "projection_model": slate["projection_model"],
        "nflverse_report": nflverse_report,
        "nflverse_depth_applied": nflverse_applied,
        "nflverse_removed": nflverse_blocked,
        "injury_removed": slate["injury_removed"],
        "csv": csv_path,
    }

