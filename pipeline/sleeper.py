"""Sleeper feed: the single source of depth, availability and projections.

Sleeper's public API (https://docs.sleeper.com) is read-only and needs no key.
Three of its endpoints drive every page:

``/state/nfl``
    The current season, week and season type.
``/players/nfl``
    One ~5MB dump of every player with his team, depth-chart slot and order,
    roster status and injury designation. Sleeper asks callers to pull it at
    most once a day, so it is cached on disk for `PLAYERS_MAX_AGE_HOURS`.
``/projections/nfl/{season_type}/{season}/{week}``
    Weekly projected stat lines keyed by Sleeper ``player_id``. The mean each
    page uses is the half-PPR total, which is Yahoo DFS scoring. This endpoint
    is undocumented and may change shape without notice, so both shapes seen
    in the wild are accepted.

Yahoo stays the source of salaries and the slate. Everything else about a
player -- who starts, who is hurt, what he is expected to score -- comes from
here. The functions below are pure transforms on the fetched JSON so the rules
can be tested without the network.
"""
from __future__ import annotations

import json
import re
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BASE_URL = "https://api.sleeper.app/v1"
PLAYERS_MAX_AGE_HOURS = 24.0
# Sleeper blocks an IP above roughly 1,000 calls a minute. A run makes a
# handful, but the floor keeps any loop that grows later well clear of that.
MIN_CALL_INTERVAL_SECONDS = 0.1
PROJECTION_FIELD = "pts_half_ppr"
POSITIONS = ("QB", "RB", "WR", "TE", "K", "DEF")

# Yahoo abbreviations that differ from Sleeper's. Sleeper also keys each team
# defense by this code, so the same map resolves a Yahoo DEF row to its player.
YAHOO_TO_SLEEPER_TEAM = {"JAC": "JAX", "LA": "LAR", "WSH": "WAS"}

# Injury designations that mean the player will not play this week. Sleeper
# uses "Questionable" and "Doubtful" for game-time decisions; those players stay
# in the pool with the designation shown.
UNAVAILABLE_INJURY_STATUSES = frozenset(
    {"OUT", "IR", "PUP", "SUS", "NFI", "DNR", "COV", "NA"}
)

# Sleeper's depth_chart_position names the alignment slot. Receivers split
# across LWR/RWR/SWR, so three starting receivers are each order 1 in their own
# slot rather than WR1 > WR2 > WR3 in one list.
DEPTH_SLOT_POSITION = {
    "QB": "QB", "RB": "RB", "FB": "RB", "WR": "WR", "LWR": "WR", "RWR": "WR",
    "SWR": "WR", "TE": "TE", "K": "K", "PK": "K",
}

# Slots that are a package role rather than a share of the position's workload.
PACKAGE_SLOTS = frozenset({"FB"})

_last_call = [0.0]


class SleeperError(RuntimeError):
    """A Sleeper request failed or returned something unusable."""


def normalize_team(value) -> str:
    team = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    team = team.strip().upper()
    return YAHOO_TO_SLEEPER_TEAM.get(team, team)


def normalize_name(value) -> str:
    """Suffix-, accent- and punctuation-insensitive name key."""
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(c for c in value if not unicodedata.combining(c))
    value = re.sub(r"\s*\b(Jr\.?|Sr\.?|II|III|IV|V)\b", "", value, flags=re.I)
    value = re.sub(r"[^A-Za-z0-9 ]+", "", value)
    return re.sub(r"\s+", " ", value).strip().lower()


# --------------------------------------------------------------------- fetching
def get_json(path: str, timeout: float = 30, attempts: int = 3):
    """GET one Sleeper endpoint with bounded retries and a call-rate floor."""
    url = BASE_URL + path
    last_error = None
    for attempt in range(1, attempts + 1):
        wait = MIN_CALL_INTERVAL_SECONDS - (time.monotonic() - _last_call[0])
        if wait > 0:
            time.sleep(wait)
        _last_call[0] = time.monotonic()
        try:
            request = Request(url, headers={"User-Agent": "nfl-top25-rankings/1.0"})
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt < attempts:
                time.sleep(1.0 * attempt)
    raise SleeperError(f"Sleeper {path} failed after {attempts} attempts: {last_error}")


def fetch_state(timeout: float = 30) -> dict:
    state = get_json("/state/nfl", timeout)
    if not isinstance(state, dict) or "season" not in state or "week" not in state:
        raise SleeperError(f"Sleeper /state/nfl returned an unexpected shape: {state!r:.200}")
    return state


def fetch_players(cache_dir: str | Path | None = "sleeper_cache", timeout: float = 60,
                  max_age_hours: float = PLAYERS_MAX_AGE_HOURS, now=None) -> tuple[dict, str]:
    """Return the full player dump and the UTC time it was fetched.

    A cached copy younger than `max_age_hours` is reused, which is what keeps
    the dump to one pull a day across the workflow's several daily runs.
    """
    now = now or datetime.now(timezone.utc)
    path = Path(cache_dir) / "players_nfl.json" if cache_dir else None
    if path is not None and path.exists():
        try:
            cached = json.loads(path.read_text(encoding="utf-8"))
            fetched = datetime.fromisoformat(cached["fetched_utc"])
            if (now - fetched).total_seconds() < max_age_hours * 3600 and cached.get("players"):
                return cached["players"], cached["fetched_utc"]
        except (OSError, ValueError, KeyError, TypeError):
            pass  # a corrupt cache is refetched, never trusted
    players = get_json("/players/nfl", timeout)
    if not isinstance(players, dict) or not players:
        raise SleeperError("Sleeper /players/nfl returned no players")
    fetched_utc = now.isoformat()
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fetched_utc": fetched_utc, "players": players}),
                        encoding="utf-8")
    return players, fetched_utc


def fetch_projections(season, week, season_type: str = "regular", timeout: float = 30):
    return get_json(f"/projections/nfl/{season_type}/{int(season)}/{int(week)}", timeout)


# ------------------------------------------------------------------ transforms
def projection_frame(raw) -> pd.DataFrame:
    """Normalize either projection shape to `player_id` -> half-PPR points.

    The v1 endpoint has returned both ``{player_id: {stat: value}}`` and
    ``[{"player_id": ..., "stats": {...}}]``. A row without a half-PPR total
    has nothing to project and is skipped.
    """
    if isinstance(raw, dict):
        items = [(pid, stats) for pid, stats in raw.items()]
    elif isinstance(raw, list):
        items = [(row.get("player_id"), row.get("stats", row)) for row in raw
                 if isinstance(row, dict)]
    else:
        raise SleeperError(f"Unexpected Sleeper projection shape: {type(raw).__name__}")
    rows = []
    for pid, stats in items:
        if pid is None or not isinstance(stats, dict):
            continue
        value = stats.get(PROJECTION_FIELD)
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value):
            # Cents, once, here. Pages round with both Python's `round` and
            # pandas', which disagree on a half-cent (15.525), and the publish
            # check rejects a player priced 0.01 apart on two pages.
            rows.append((str(pid), round(value, 2)))
    return pd.DataFrame(rows, columns=["player_id", "Sleeper_FP"]).drop_duplicates(
        "player_id", keep="last")


def players_frame(raw: dict) -> pd.DataFrame:
    """Flatten the player dump to the fields the pipeline uses, fantasy positions only."""
    rows = []
    for pid, p in (raw or {}).items():
        if not isinstance(p, dict):
            continue
        position = str(p.get("position") or "").upper()
        if position not in POSITIONS:
            continue
        if position == "DEF":
            name = f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip() or str(pid)
            team = normalize_team(p.get("team") or pid)
        else:
            name = p.get("full_name") or f"{p.get('first_name') or ''} {p.get('last_name') or ''}".strip()
            team = normalize_team(p.get("team"))
        yahoo = pd.to_numeric(p.get("yahoo_id"), errors="coerce")
        order = pd.to_numeric(p.get("depth_chart_order"), errors="coerce")
        rows.append({
            "player_id": str(p.get("player_id") or pid),
            "Sleeper_Name": name,
            "Key": normalize_name(name),
            "Team": team,
            "Position": position,
            "Yahoo ID": int(yahoo) if pd.notna(yahoo) else pd.NA,
            "Status": p.get("status"),
            "Active": p.get("active"),
            "Injury_Status": p.get("injury_status"),
            "Injury_Body_Part": p.get("injury_body_part"),
            "Practice_Status": p.get("practice_participation"),
            "Depth_Chart_Position": p.get("depth_chart_position"),
            "Depth_Chart_Order": int(order) if pd.notna(order) else pd.NA,
        })
    frame = pd.DataFrame(rows)
    frame["Yahoo ID"] = frame["Yahoo ID"].astype("Int64")
    frame["Depth_Chart_Order"] = frame["Depth_Chart_Order"].astype("Int64")
    return frame


def unavailable_reason(row) -> str | None:
    """Why a Sleeper player cannot play this week, or None if he can."""
    if row["Position"] == "DEF":
        return None
    if not row["Team"]:
        return "not on an NFL roster"
    injury = str(row.get("Injury_Status") or "").strip().upper()
    if injury in UNAVAILABLE_INJURY_STATUSES:
        return f"injury status {row['Injury_Status']}"
    status = row.get("Status")
    if status and str(status).strip().lower() != "active":
        return f"roster status {status}"
    if row.get("Active") is False:
        return "inactive"
    return None


def build_reference(players: pd.DataFrame, projections: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Attach availability, depth and projection to every Sleeper player.

    Depth is ranked among *available* teammates, so the player behind an Out or
    IR starter moves up without waiting for Sleeper's chart to catch up.
    ``Role_Tier`` is the rank within the player's alignment slot and
    ``Depth_Rank`` the flat opportunity rank within team and position, ordered
    by role tier and then by projection, which is the ordinal the fitted CV,
    scoreless-rate and correlation tables are keyed on.

    Returns ``(reference, promotions)``. `promotions` lists each teammate whose
    role tier improved because someone ahead of him in his slot is unavailable.
    """
    ref = players.merge(projections, on="player_id", how="left")
    ref["Unavailable_Reason"] = ref.apply(unavailable_reason, axis=1)
    ref["Available"] = ref["Unavailable_Reason"].isna()
    ref["Depth_Slot"] = ref["Depth_Chart_Position"].where(
        ref["Depth_Chart_Position"].notna(), None)
    slot_position = ref["Depth_Slot"].map(lambda s: DEPTH_SLOT_POSITION.get(str(s).upper()) if s else None)
    # A chart slot at another position (a receiver listed at RB) still orders
    # him inside his own fantasy position; only the slot name is kept.
    ref["Charted"] = ref["Depth_Chart_Order"].notna() & slot_position.notna()
    ref["Chart_Order"] = pd.to_numeric(ref["Depth_Chart_Order"], errors="coerce")
    ref["Sort_FP"] = ref["Sleeper_FP"].fillna(0.0)

    def slot_tiers(frame):
        charted = frame[frame["Charted"]].sort_values(
            ["Team", "Position", "Depth_Slot", "Chart_Order", "Sort_FP", "Key"],
            ascending=[True, True, True, True, False, True])
        return charted.groupby(["Team", "Position", "Depth_Slot"]).cumcount() + 1

    raw_tier = slot_tiers(ref)
    live = ref[ref["Available"]]
    live_tier = slot_tiers(live)
    ref["Chart_Tier"] = raw_tier.reindex(ref.index).astype("Int64")
    ref["Role_Tier"] = live_tier.reindex(ref.index).astype("Int64")

    # A package slot (a fullback) is tier 1 of its own slot but is not a
    # starter's workload, so it ranks behind every regular slot.
    package = live["Depth_Slot"].astype(str).str.upper().isin(PACKAGE_SLOTS)
    order = live.assign(
        _package=package,
        _tier=ref.loc[live.index, "Role_Tier"].astype(float).fillna(np.inf),
    ).sort_values(["Team", "Position", "_package", "_tier", "Sort_FP", "Key"],
                  ascending=[True, True, True, True, False, True])
    depth = order.groupby(["Team", "Position"]).cumcount() + 1
    ref["Depth_Rank"] = depth.reindex(ref.index).astype("Int64")
    # An uncharted player's role is named by his opportunity rank.
    uncharted = ref["Available"] & ref["Role_Tier"].isna()
    ref.loc[uncharted, "Role_Tier"] = ref.loc[uncharted, "Depth_Rank"]
    ref.loc[ref["Position"].eq("DEF") & ref["Available"], ["Role_Tier", "Depth_Rank"]] = 1

    promoted = ref["Available"] & ref["Charted"] & (ref["Role_Tier"] < ref["Chart_Tier"])
    out = ref[~ref["Available"] & ref["Charted"]]
    rows = []
    for _, row in ref[promoted].iterrows():
        ahead = out[(out["Team"] == row["Team"]) & (out["Position"] == row["Position"])
                    & (out["Depth_Slot"] == row["Depth_Slot"])
                    & (out["Chart_Tier"] < row["Chart_Tier"])]
        rows.append({
            "Player": row["Sleeper_Name"], "Team": row["Team"], "Position": row["Position"],
            "Slot": row["Depth_Slot"], "Old tier": int(row["Chart_Tier"]),
            "New tier": int(row["Role_Tier"]),
            "Replacing": ", ".join(ahead.sort_values("Chart_Tier")["Sleeper_Name"]),
        })
    promotions = pd.DataFrame(rows, columns=["Player", "Team", "Position", "Slot",
                                             "Old tier", "New tier", "Replacing"])
    return ref.drop(columns=["Sort_FP"]), promotions


def match_players(pool: pd.DataFrame, reference: pd.DataFrame) -> pd.Series:
    """Map each pool row to a Sleeper ``player_id``, or NA where none is certain.

    Tried in order: Yahoo's own player id (Sleeper carries it as ``yahoo_id``),
    then name + team + position, then name + team. A step only accepts a key
    that is unique on the Sleeper side, so an ambiguous name never matches.
    DEF rows resolve by team code, which is Sleeper's id for a defense.
    """
    ids = pd.Series(pd.NA, index=pool.index, dtype="object")
    team = pool["Team"].map(normalize_team)
    key = pool["Name"].map(normalize_name)
    position = pool["Position"].astype(str).str.upper()

    defenses = reference[reference["Position"].eq("DEF")].set_index("Team")["player_id"]
    is_def = position.eq("DEF")
    ids[is_def] = team[is_def].map(defenses)

    def unique(frame, columns):
        counts = frame.groupby(columns)["player_id"].transform("size")
        return frame[counts.eq(1)].set_index(columns)["player_id"]

    offense = reference[reference["Position"].ne("DEF")]
    if "Yahoo ID" in pool:
        by_yahoo = unique(offense.dropna(subset=["Yahoo ID"]), ["Yahoo ID"])
        want = ids.isna() & pool["Yahoo ID"].notna()
        ids[want] = pool.loc[want, "Yahoo ID"].map(by_yahoo)
    by_full = unique(offense, ["Key", "Team", "Position"])
    want = ids.isna() & ~is_def
    ids[want] = [by_full.get((k, t, p), pd.NA) for k, t, p in
                 zip(key[want], team[want], position[want])]
    by_team = unique(offense, ["Key", "Team"])
    want = ids.isna() & ~is_def
    ids[want] = [by_team.get((k, t), pd.NA) for k, t in zip(key[want], team[want])]
    return ids


def resolve_week(state: dict, slate_teams, fetch=fetch_projections, players=None):
    """Pick the projection week that matches the Yahoo slate.

    Sleeper's ``week`` can still name the finished week for a day or two after
    Yahoo has opened the next slate. Both candidates are fetched and the one
    that projects more of the slate's teams wins.
    """
    season = int(state.get("season") or state.get("league_season"))
    season_type = str(state.get("season_type") or "regular")
    if season_type not in {"regular", "pre", "post"}:
        season_type = "regular"
    week = int(state.get("week") or state.get("display_week") or 1)
    teams = {normalize_team(t) for t in slate_teams}
    best = None
    for candidate in dict.fromkeys([week, week + 1]):
        frame = projection_frame(fetch(season, candidate, season_type))
        covered = set()
        if players is not None and len(frame):
            projected = players.merge(frame[frame["Sleeper_FP"] > 0], on="player_id")
            covered = set(projected["Team"]) & teams
        score = (len(covered), len(frame) > 0)
        if best is None or score > best[0]:
            best = (score, candidate, frame)
    _, chosen, frame = best
    return {"season": season, "week": chosen, "season_type": season_type}, frame


def load(slate_teams, cache_dir="sleeper_cache", timeout: float = 30):
    """Fetch state, players and projections and build the joined reference."""
    state = fetch_state(timeout)
    raw_players, fetched_utc = fetch_players(cache_dir, max(timeout, 60))
    players = players_frame(raw_players)
    context, projections = resolve_week(state, slate_teams, players=players)
    if projections.empty:
        raise SleeperError(
            f"Sleeper has no projections for {context['season']} week {context['week']}")
    reference, promotions = build_reference(players, projections)
    context.update(players_fetched_utc=fetched_utc, state=state)
    return reference, promotions, context
