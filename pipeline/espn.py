"""ESPN feed: a second, independent weekly projection blended with Sleeper's.

ESPN's fantasy API serves its weekly projections without a key. One request
returns every rostered-or-relevant player with:

* the projected stat line for the scoring period, and the standard-scoring
  total ESPN applies to it (``appliedTotal`` at ``leaguedefaults/1``);
* projected receptions (stat ``53``), so half-PPR is standard + 0.5 * receptions;
* ESPN's current injury status (``ACTIVE``, ``QUESTIONABLE``, ``OUT``, ...).

Why a second source: Sleeper's projections can lag news. On 2026-11-14 Kenneth
Walker III was ruled out Saturday morning; Sleeper dropped him but still had
Emmett Johnson at his backup projection (10.32), while ESPN had already moved
Johnson to a starter's line (15.4 half-PPR) and Walker to OUT/0.

The endpoint is undocumented and may change shape. Every failure here is
non-fatal: `blend` falls back to Sleeper alone and says so in the notes.
"""
from __future__ import annotations

import json
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
URL = ("https://lm-api-reads.fantasy.espn.com/apis/v3/games/ffl/seasons/{season}"
       "/segments/0/leaguedefaults/1?view=kona_player_info&scoringPeriodId={week}")
ESPN_WEIGHT = 0.5          # blended mean = (1 - w) * Sleeper + w * ESPN
# ESPN statuses that mean "will not play". ESPN updates these within minutes of
# a team announcement, so they override a Sleeper projection that has not.
OUT_STATUSES = frozenset({"OUT", "INJURY_RESERVE", "SUSPENSION", "DOUBTFUL_OUT"})
# News override: when ESPN rules out a team's charted starter (Sleeper depth
# order 1) at a position, Sleeper's numbers for his teammates at that position
# were built before the news, and averaging would keep half of a stale backup
# line. That position group uses ESPN alone for the week.
NEWS_OVERRIDE_STATUSES = frozenset({"OUT", "DOUBTFUL", "SUSPENSION"})
# ... but only while Sleeper is still stale: more than this far below/above
# ESPN. Once Sleeper has absorbed the news, the ordinary blend applies.
NEWS_STALE_RATIO = 0.20
DISAGREE_RATIO = 0.40      # flag |S - E| / max(S, E) above this ...
DISAGREE_MIN_FP = 5.0      # ... when the larger projection is at least this
REQUEST_LIMIT = 2000
RECEPTIONS_STAT = "53"
PROJECTED_SOURCE_ID = 1    # statSourceId: 0 actual, 1 projected
WEEKLY_SPLIT_ID = 1        # statSplitTypeId: 0 season, 1 single scoring period

POSITION_BY_ID = {1: "QB", 2: "RB", 3: "WR", 4: "TE", 5: "K", 16: "DEF"}
# ESPN proTeamId -> Sleeper team code (Sleeper keys defenses by this code).
TEAM_BY_ID = {
    1: "ATL", 2: "BUF", 3: "CHI", 4: "CIN", 5: "CLE", 6: "DAL", 7: "DEN", 8: "DET",
    9: "GB", 10: "TEN", 11: "IND", 12: "KC", 13: "LV", 14: "LAR", 15: "MIA",
    16: "MIN", 17: "NE", 18: "NO", 19: "NYG", 20: "NYJ", 21: "PHI", 22: "ARI",
    23: "PIT", 24: "LAC", 25: "SF", 26: "SEA", 27: "TB", 28: "WAS", 29: "CAR",
    30: "JAX", 33: "BAL", 34: "HOU",
}
FRAME_COLUMNS = ["ESPN_ID", "ESPN_Name", "Team", "Position", "ESPN_FP", "ESPN_Status"]


class ESPNError(RuntimeError):
    """The ESPN request failed or returned something unusable."""


# ------------------------------------------------------------------ fetching
def fetch(season: int, week: int, timeout: float = 40, attempts: int = 3) -> dict:
    """GET ESPN's player projections for one scoring period."""
    filt = {"players": {
        "filterSlotIds": {"value": [0, 2, 4, 6, 17, 16]},
        "filterStatsForTopScoringPeriodIds": {
            "value": 2, "additionalValue": [f"1{WEEKLY_SPLIT_ID}{int(season)}{int(week)}"]},
        "limit": REQUEST_LIMIT,
        "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
    }}
    url = URL.format(season=int(season), week=int(week))
    headers = {"User-Agent": "Mozilla/5.0 nfl-top25-rankings/1.0",
               "X-Fantasy-Filter": json.dumps(filt, separators=(",", ":"))}
    last = None
    for attempt in range(1, attempts + 1):
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                raw = json.loads(response.read().decode("utf-8"))
            if not isinstance(raw, dict) or not raw.get("players"):
                raise ValueError("no players in response")
            return raw
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last = exc
            if attempt < attempts:
                time.sleep(1.0 * attempt)
    raise ESPNError(f"ESPN projections failed after {attempts} attempts: {last}")


# ---------------------------------------------------------------- transforms
def projection_frame(raw: dict, week: int, normalize_name) -> pd.DataFrame:
    """One row per ESPN player: team, position, half-PPR projection, status.

    A player ESPN lists without a projection for `week` (bye, deep reserve)
    gets ``ESPN_FP`` NaN, which `blend` treats as "ESPN has no opinion".
    """
    rows = []
    for item in (raw or {}).get("players", []) or []:
        p = (item or {}).get("player") or {}
        position = POSITION_BY_ID.get(p.get("defaultPositionId"))
        team = TEAM_BY_ID.get(p.get("proTeamId"))
        if position is None or team is None:
            continue  # free agent (proTeamId 0) or unsupported slot
        fp = np.nan
        for s in p.get("stats") or []:
            if (s.get("scoringPeriodId") == int(week)
                    and s.get("statSourceId") == PROJECTED_SOURCE_ID
                    and s.get("statSplitTypeId") == WEEKLY_SPLIT_ID):
                total = pd.to_numeric(s.get("appliedTotal"), errors="coerce")
                rec = pd.to_numeric((s.get("stats") or {}).get(RECEPTIONS_STAT, 0), errors="coerce")
                if pd.notna(total):
                    fp = round(float(total) + 0.5 * (0.0 if pd.isna(rec) else float(rec)), 2)
                break
        status = str(p.get("injuryStatus") or "ACTIVE").upper()
        rows.append({"ESPN_ID": p.get("id"), "ESPN_Name": p.get("fullName", ""),
                     "Team": team, "Position": position, "ESPN_FP": fp,
                     "ESPN_Status": status})
    frame = pd.DataFrame(rows, columns=FRAME_COLUMNS)
    frame["Key"] = frame["ESPN_Name"].map(normalize_name)
    return frame


def match(players: pd.DataFrame, espn: pd.DataFrame) -> pd.Series:
    """Map Sleeper ``player_id`` -> ESPN row index, or NA when not certain.

    Order: Sleeper's own ``espn_id`` (sparse for recent players), then
    name + team + position, unique on the ESPN side. Defenses match by team.
    """
    out = pd.Series(pd.NA, index=players.index, dtype="object")
    if espn.empty:
        return out
    by_id = espn.dropna(subset=["ESPN_ID"]).drop_duplicates("ESPN_ID", keep=False)
    by_id = pd.Series(by_id.index, index=by_id["ESPN_ID"].astype("int64"))
    if "ESPN ID" in players:
        ids = pd.to_numeric(players["ESPN ID"], errors="coerce")
        want = ids.notna()
        out[want] = ids[want].astype("int64").map(by_id)
    is_def = players["Position"].eq("DEF")
    defenses = espn[espn["Position"].eq("DEF")].drop_duplicates("Team", keep=False)
    defenses = pd.Series(defenses.index, index=defenses["Team"])
    want = out.isna() & is_def
    out[want] = players.loc[want, "Team"].map(defenses)
    offense = espn[espn["Position"].ne("DEF")]
    counts = offense.groupby(["Key", "Team", "Position"])["Key"].transform("size")
    unique = offense[counts.eq(1)]
    lookup = dict(zip(zip(unique["Key"], unique["Team"], unique["Position"]), unique.index))
    want = out.isna() & ~is_def
    out[want] = [lookup.get(k, pd.NA) for k in
                 zip(players.loc[want, "Key"], players.loc[want, "Team"], players.loc[want, "Position"])]
    return out


def blend(players: pd.DataFrame, projections: pd.DataFrame, espn: pd.DataFrame,
          weight: float = ESPN_WEIGHT) -> tuple[pd.DataFrame, list[str]]:
    """Blend ESPN into Sleeper's ``player_id -> Sleeper_FP`` frame.

    Returns the same shape the rest of the pipeline expects (``player_id``,
    ``Sleeper_FP`` = the blended mean) plus audit columns ``Sleeper_Raw_FP``,
    ``ESPN_FP``, ``ESPN_Status`` and ``Projection_Flag``, and log notes.

    Rules, in order:
      1. ESPN says OUT/IR/suspended          -> 0 (dropped from the pool).
      2. both project                        -> weighted mean.
      3. only Sleeper projects (ESPN silent) -> Sleeper.
      4. only ESPN projects, player active   -> ESPN (covers a backup Sleeper
         has not promoted yet, and Sleeper's missing rookies).
      5. news override: ESPN rules out the team's Sleeper depth-1 player at a
         position -> his teammates at that position take ESPN's number alone.
    """
    notes: list[str] = []
    proj = projections.rename(columns={"Sleeper_FP": "Sleeper_Raw_FP"})
    extra = [c for c in ("ESPN ID", "Depth_Chart_Order") if c in players]
    work = players[["player_id", "Sleeper_Name", "Key", "Team", "Position"] + extra].merge(
        proj, on="player_id", how="left")
    idx = match(work, espn)
    matched = idx.notna()
    for col in ("ESPN_FP", "ESPN_Status"):
        work[col] = pd.Series(pd.NA, index=work.index, dtype="object")
        work.loc[matched, col] = espn.loc[idx[matched].astype(int), col].to_numpy()
    s = pd.to_numeric(work["Sleeper_Raw_FP"], errors="coerce")
    e = pd.to_numeric(work["ESPN_FP"], errors="coerce")
    out_flag = work["ESPN_Status"].isin(OUT_STATUSES)
    s_ok, e_ok = s.gt(0), e.gt(0)

    mean = pd.Series(np.nan, index=work.index)
    both = s_ok & e.notna()
    mean[both] = (1 - weight) * s[both] + weight * e[both]
    mean[s_ok & e.isna()] = s[s_ok & e.isna()]
    espn_only = ~s_ok & e_ok & ~out_flag
    mean[espn_only] = e[espn_only]
    news = pd.Series(False, index=work.index)
    if "Depth_Chart_Order" in work:
        starter = pd.to_numeric(work["Depth_Chart_Order"], errors="coerce").eq(1)
        ruled_out = starter & work["ESPN_Status"].isin(NEWS_OVERRIDE_STATUSES) & work["Position"].ne("DEF")
        groups = set(zip(work.loc[ruled_out, "Team"], work.loc[ruled_out, "Position"]))
        if groups:
            in_group = pd.Series([(t, p) in groups for t, p in zip(work["Team"], work["Position"])],
                                 index=work.index)
            # Only where Sleeper has not caught up yet (NEWS_STALE_RATIO).
            hi_ = pd.concat([s, e], axis=1).max(axis=1)
            stale = s.isna() | ((s - e).abs() / hi_.where(hi_ > 0)).gt(NEWS_STALE_RATIO)
            news = in_group & e.notna() & ~out_flag & ~ruled_out & stale
            mean[news] = e[news]
    mean[out_flag] = 0.0

    hi = pd.concat([s, e], axis=1).max(axis=1)
    gap = (s - e).abs() / hi.where(hi > 0)
    flag = pd.Series(pd.NA, index=work.index, dtype="object")
    flag[both & gap.gt(DISAGREE_RATIO) & hi.ge(DISAGREE_MIN_FP)] = "sources disagree"
    flag[espn_only] = "ESPN only"
    flag[news & e.gt(0)] = "ESPN only (starter out)"
    flag[out_flag & s_ok] = "ESPN: out"
    work["Sleeper_FP"] = mean.round(2)
    work["Projection_Flag"] = flag

    projected = s_ok | e_ok
    notes.append(f"ESPN: matched {int((matched & projected).sum())} of {int(projected.sum())} "
                 f"projected players; blend weight {weight:.2f}.")
    for label in ("ESPN only (starter out)", "ESPN: out", "ESPN only", "sources disagree"):
        rows = work[flag.eq(label)].sort_values("Sleeper_FP", ascending=False)
        if len(rows):
            show = ", ".join(f"{r.Sleeper_Name} ({r.Team}) S={r.Sleeper_Raw_FP} E={r.ESPN_FP}"
                             for r in rows.head(12).itertuples())
            notes.append(f"  {label}: {len(rows)} -- {show}")
    keep = ["player_id", "Sleeper_FP", "Sleeper_Raw_FP", "ESPN_FP", "ESPN_Status", "Projection_Flag"]
    result = work.loc[work["Sleeper_FP"].notna(), keep].drop_duplicates("player_id")
    return result.reset_index(drop=True), notes


def compact(espn: pd.DataFrame) -> list[dict]:
    """The archived form: the joined fields only, not the ~9MB raw response."""
    cols = ["ESPN_ID", "ESPN_Name", "Team", "Position", "ESPN_FP", "ESPN_Status"]
    frame = espn[cols].astype(object).where(espn[cols].notna(), None)
    return frame.to_dict("records")
