#!/usr/bin/env python3
"""Standalone, keyless season-long NFL weekly lineup optimizer.

Yahoo's public DFS feed supplies current-week salary, opponents and game times.
Sleeper's public API supplies everything about the player: the weekly half-PPR
projection that is the mean, the depth chart, and roster and injury status.
No merged CSV is required.

The roster stays in ``MY_TEAM_ROSTER`` because Yahoo's private season-long
roster API requires OAuth. Expected points are the default objective. Fitted
depth CVs add P25/P90 diagnostics around Sleeper's mean.

Run it with no arguments to fetch the feeds and print a lineup; run it with
``--self-test`` for the offline checks in ``self_test``. ``run_lineup.py`` is
the non-interactive entry point that publishes the same lineup to the site.
"""

from __future__ import annotations

import json
import re
import sys
import time
import unicodedata
import warnings
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from pipeline import sleeper


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

STARTING_POSITIONS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "K": 1, "DEF": 1, "FLEX": 1}
FLEX_ELIGIBLE = ["RB", "WR", "TE"]
LINEUP_OBJECTIVE = "FP"  # FP, Floor_P25, or Ceiling_P90
EXCLUDED_PLAYERS: list[str] | None = None  # None prompts; [] skips prompt.
MANUAL_DEPTH_OVERRIDES: dict[str, int] = {}
AUTO_EXCLUDE_REPORTED_OUT = True  # bench anyone Sleeper does not project this week
# How deep the page's add pool goes per position. Deep enough to cover a real
# waiver claim, shallow enough that `pool.json` stays a small download.
POOL_LIMITS = {"QB": 40, "RB": 70, "WR": 90, "TE": 45, "K": 32, "DEF": 32}

YAHOO_URL = "https://dfyql-ro.sports.yahoo.com/v2/external/playersFeed/nfl"
# Sleeper projects a kicker's mean but there is no fitted kicker CV. This is the
# spread of Yahoo kicker scores across recent seasons, used for the P25/P90 band.
KICKER_CV = 0.60
SLEEPER_CACHE_DIR = "sleeper_cache"

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


def fetch_yahoo(attempts: int = 3, timeout: int = 20) -> pd.DataFrame:
    """Fetch Yahoo's current-week salaries, opponents and kickoffs; nothing else is used."""
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
        "salary": "Salary", "gameStartTime": "Game_Time",
        "homeTeam": "Home_Team", "awayTeam": "Away_Team",
    })
    needed = {"Feed_Name", "Feed_Position", "Team", "Salary", "Game_Time", "Home_Team", "Away_Team"}
    missing = sorted(needed - set(out))
    if missing:
        raise ValueError(f"Yahoo schema changed; missing {missing}")
    out["Feed_Position"] = out["Feed_Position"].astype(str).str.upper().replace({"D/ST": "DEF", "DST": "DEF"})
    for col in ["Team", "Home_Team", "Away_Team"]:
        out[col] = out[col].map(normalize_team)
    out["Salary"] = pd.to_numeric(out["Salary"], errors="coerce")
    out["Game_Time"] = pd.to_datetime(out["Game_Time"], errors="coerce", utc=True)
    out["Game_Date"] = out["Game_Time"].dt.strftime("%Y-%m-%d")
    out["Opponent"] = np.where(out["Team"].eq(out["Away_Team"]), out["Home_Team"], out["Away_Team"])
    out["Key"] = out["Feed_Name"].map(normalize_name)
    out = out[out["Feed_Position"].isin(["QB", "RB", "WR", "TE", "DEF"]) & out["Salary"].gt(0)].copy()

    out["Fallback_Depth"] = out.groupby(["Team", "Feed_Position"])["Salary"].rank(method="first", ascending=False)
    # The mean comes from Sleeper once its context is loaded (`with_sleeper_projections`).
    out["Projected_FP"] = np.nan
    out["Projection_Source"] = pd.NA
    out["Projection_Frozen"] = False
    return out.sort_values("Salary", ascending=False).drop_duplicates("Key").reset_index(drop=True)


def yahoo_from_prepared_slate(players: pd.DataFrame) -> pd.DataFrame:
    """Adapt the rankings pipeline's final player pool for the lineup builder.

    `pipeline.notebook.prepare_slate_pool` has already fetched Yahoo and applied
    Sleeper's depth, availability and projection. Re-fetching those inputs here is
    what allowed two pages from one site publish to disagree.  This adapter
    preserves that final `Projected_FP` verbatim and only adds the legacy column
    aliases the season-long roster resolver expects.
    """
    required = {
        "Name", "Position", "Team", "Opponent", "Game Time", "Home Team",
        "Away Team", "Salary", "Projected_FP", "Projection_Source",
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
    out["Projection_Frozen"] = True
    return out.sort_values("Projected_FP", ascending=False).drop_duplicates("Key").reset_index(drop=True)


def load_sleeper_context(yahoo: pd.DataFrame, cache_dir: str = SLEEPER_CACHE_DIR) -> dict:
    """Load Sleeper's week, depth, availability and projections for the slate."""
    reference, _, context = sleeper.load(yahoo["Team"].unique(), cache_dir=cache_dir)
    return sleeper_context(reference, context, yahoo)


def sleeper_context(reference: pd.DataFrame, context: dict, yahoo: pd.DataFrame) -> dict:
    """The lineup builder's view of a Sleeper reference: this module's team codes."""
    reference = reference.copy()
    reference["Team"] = reference["Team"].map(normalize_team)
    reference["Key"] = reference["Sleeper_Name"].map(normalize_name)
    schedule = pd.concat([
        yahoo[["Team", "Opponent"]],
        yahoo[["Opponent", "Team"]].set_axis(["Team", "Opponent"], axis=1),
    ]).dropna().drop_duplicates("Team")
    return {"season": context["season"], "week": context["week"],
            "season_type": context["season_type"],
            "depth_stamp": context["players_fetched_utc"],
            "reference": reference, "schedule": schedule, "notes": []}


def _projection_source(ctx: dict) -> str:
    return f"Sleeper half-PPR projection ({ctx.get('season')} week {ctx.get('week')})"


def _sleeper_match(ctx: dict, key: str, team, position: str):
    """The one Sleeper row for a name, narrowed by team then position, or None."""
    ref = ctx["reference"]
    rows = ref[ref["Key"].eq(key)]
    if len(rows) > 1 and isinstance(team, str) and team:
        narrowed = rows[rows["Team"].eq(team)]
        rows = narrowed if len(narrowed) else rows
    if len(rows) > 1:
        rows = rows[rows["Position"].eq(position)]
    if len(rows) > 1:
        # Two same-named players at one position: the one with a projection
        # this week is the one who is playing.
        rows = rows.sort_values("Sleeper_FP", ascending=False, na_position="last").head(1)
    return rows.iloc[0] if len(rows) == 1 else None


def with_sleeper_projections(yahoo: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """Give every unfrozen Yahoo row Sleeper's mean, so the add pool ranks on it."""
    out = yahoo.copy()
    frozen = out.get("Projection_Frozen", pd.Series(False, index=out.index)).fillna(False).astype(bool)
    for i in out.index[~frozen]:
        row = _sleeper_match(ctx, out.at[i, "Key"], out.at[i, "Team"], out.at[i, "Feed_Position"])
        value = None if row is None else row.get("Sleeper_FP")
        if value is not None and pd.notna(value):
            out.at[i, "Projected_FP"] = float(value)
            out.at[i, "Projection_Source"] = _projection_source(ctx)
    return out


def build_roster(configured: list[dict], yahoo: pd.DataFrame, ctx: dict) -> pd.DataFrame:
    """Resolve the configured roster across Yahoo and Sleeper, then add ranges."""
    roster = pd.DataFrame(configured)
    roster["Name"] = roster["Name"].astype(str).str.strip()
    roster["Position"] = roster["Position"].astype(str).str.upper()
    roster["Key"] = roster["Name"].map(normalize_name)
    yahoo = yahoo.copy()
    if "Projection_Frozen" not in yahoo:
        yahoo["Projection_Frozen"] = False
    ycols = ["Key", "Feed_Name", "Feed_Position", "Team", "Opponent", "Game_Time", "Salary",
             "Projected_FP", "Projection_Source", "Fallback_Depth", "Projection_Frozen"]
    roster = roster.merge(yahoo[ycols], on="Key", how="left")
    roster["Configured_Position"] = roster["Position"]
    feed_position = roster["Feed_Position"].astype("string").str.upper()
    roster["Position_Mismatch"] = (
        feed_position.notna() & feed_position.ne(roster["Configured_Position"])
    )
    # Yahoo owns eligibility when the player matched. This prevents a typo in
    # the editable roster from seating an RB in a WR slot.
    roster.loc[feed_position.notna(), "Position"] = feed_position[feed_position.notna()]
    roster["Projection_Frozen"] = roster["Projection_Frozen"].fillna(False).astype(bool)

    for column in ["Depth_Rank", "Projection_CV"]:
        roster[column] = np.nan
    for column in ["Depth_Source", "report_status", "practice_status",
                   "report_primary_injury"]:
        roster[column] = pd.Series(pd.NA, index=roster.index, dtype="object")

    source = _projection_source(ctx)
    # An unfrozen mean is Sleeper's or nothing: a player Sleeper does not project
    # this week must not keep whatever number the Yahoo frame carried.
    roster.loc[~roster["Projection_Frozen"], ["Projected_FP", "Projection_Source"]] = [np.nan, None]
    for i, p in roster.iterrows():
        row = _sleeper_match(ctx, p.Key, p.Team, p.Position)
        if row is None:
            continue
        if not isinstance(p.Team, str) or not p.Team:
            roster.at[i, "Team"] = row["Team"]
        depth = row.get("Depth_Rank")
        if pd.isna(depth):
            depth = row.get("Chart_Tier")
        if pd.notna(depth):
            roster.at[i, "Depth_Rank"] = int(depth)
            roster.at[i, "Depth_Source"] = (
                "Sleeper depth chart" if pd.notna(row.get("Chart_Tier")) else "Sleeper projection order")
        for column, value in (("report_status", row.get("Injury_Status")),
                              ("practice_status", row.get("Practice_Status")),
                              ("report_primary_injury", row.get("Injury_Body_Part"))):
            if value is not None and pd.notna(value):
                roster.at[i, column] = value
        if not p.Projection_Frozen and pd.notna(row.get("Sleeper_FP")):
            roster.at[i, "Projected_FP"] = float(row["Sleeper_FP"])
            roster.at[i, "Projection_Source"] = source

    roster["Team"] = roster["Team"].map(normalize_team)
    roster["Depth_Rank"] = roster["Depth_Rank"].fillna(roster["Fallback_Depth"])
    roster["Depth_Source"] = roster["Depth_Source"].fillna("Yahoo salary-order fallback")
    for name, depth in MANUAL_DEPTH_OVERRIDES.items():
        mask = roster.Key.eq(normalize_name(name))
        roster.loc[mask, "Depth_Rank"] = max(1, int(depth))
        roster.loc[mask, "Depth_Source"] = "manual override"

    kicker = roster["Position"].eq("K")
    roster.loc[kicker, "Projection_CV"] = KICKER_CV
    for i, p in roster[~kicker].iterrows():
        if p.Position in CALIBRATED_CV and pd.notna(p.Depth_Rank):
            bucket = min(max(int(p.Depth_Rank), 1), 4)
            roster.at[i, "Projection_CV"] = CALIBRATED_CV[p.Position][bucket]

    roster["Projected_FP"] = pd.to_numeric(roster["Projected_FP"], errors="coerce").fillna(0)
    # No projection (a bye, or no match) means no distribution to band.
    roster.loc[roster["Projected_FP"].le(0), "Projection_CV"] = np.nan
    roster["Projection_Source"] = roster["Projection_Source"].fillna(
        "No weekly projection; bye or unmatched player"
    )
    roster["FP"] = roster["Projected_FP"]
    # Sleeper decides who plays: an injured or bye-week player projects at zero.
    roster["Unavailable_Reason"] = np.where(
        roster["FP"].gt(0), None, "not projected this week")
    roster = roster.merge(ctx["schedule"].drop_duplicates("Team"), on="Team", how="left", suffixes=("", "_NFL"))
    roster["Opponent"] = roster["Opponent"].fillna(roster.get("Opponent_NFL"))

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
    `build_roster` produces that. Ranking on Sleeper's mean before resolving is
    safe, because `build_roster` assigns the same mean -- the top of this list is
    the top of the built pool. Kickers are the
    exception: the DFS feed does not price them at all, so they come from
    Sleeper, one available kicker per team with a projection this week.
    """
    limits = dict(limits or POOL_LIMITS)
    entries: list[dict] = []
    for position, limit in limits.items():
        if position == "K" or not limit:
            continue
        rows = yahoo[yahoo["Feed_Position"].eq(position)].dropna(subset=["Projected_FP"])
        rows = rows.nlargest(int(limit), "Projected_FP")
        entries += [{"Name": str(name), "Position": position} for name in rows["Feed_Name"]]

    ref = ctx.get("reference")
    if not limits.get("K") or ref is None or ref.empty:
        return entries
    kickers = ref[ref["Position"].eq("K") & ref["Sleeper_FP"].gt(0)]
    # One kicker per team: the second placekicker on a chart is a camp body.
    kickers = kickers.sort_values(["Depth_Rank", "Sleeper_FP"], ascending=[True, False])
    kickers = kickers.drop_duplicates("Team").drop_duplicates("Key")
    entries += [{"Name": str(name), "Position": "K"} for name in kickers["Sleeper_Name"]]
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
    """Names Sleeper does not project this week: ruled out, on a bye, or unknown.

    Sleeper folds injuries into its projection, so a zero is the signal; its
    injury designation alone (Questionable, even Out) does not bench anyone.
    """
    if "Unavailable_Reason" not in roster:
        return []
    return roster.loc[roster["Unavailable_Reason"].notna(), "Name"].tolist()


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
    if bool(p.get("Position_Mismatch", False)):
        notes.append(f"position corrected {p.Configured_Position}->{p.Position}")
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


def run(roster_path: str | None = None) -> dict:
    """Fetch Yahoo and Sleeper, optimize the configured roster, and print results."""
    print("Loading Yahoo weekly slate ...")
    yahoo = fetch_yahoo()
    print("Loading Sleeper projections, depth and injuries ...")
    ctx = load_sleeper_context(yahoo)
    yahoo = with_sleeper_projections(yahoo, ctx)
    roster = build_roster(load_roster(roster_path), yahoo, ctx)
    print(f"\nSeason {ctx['season']} week {ctx['week']}; Sleeper players fetched {ctx['depth_stamp']}")
    for note in ctx.get("notes", []):
        print(f"  {note}")
    print("Sleeper projections use half-PPR scoring; verify your league scoring and injury news.")

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
            "bench": bench, "excluded": excluded}


def self_test() -> None:
    """Run small offline checks for joins and lineup slots."""
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
