#!/usr/bin/env python3
"""Weekly source-quality check: Sleeper vs ESPN vs CBS vs their mean.

For every completed week this season it scores each source's weekly half-PPR
projection against Sleeper's half-PPR actuals, plus the Sleeper+ESPN blend and
the three-way mean the pipeline publishes. Writes
``site/data/source_quality.json`` and, inside GitHub Actions, a job summary.

Guards (read these before trusting a number):

* Pregame only. Where the projection archive holds a capture taken before the
  week's first Sunday kickoff, that capture is used; otherwise the provider's
  current page for the past week is used and the source is marked
  ``refetched`` -- a provider could have revised it after kickoff. Sleeper's
  refetched past weeks matched archived pregame captures within ~0.1 pt
  (weeks 8-9 2026); ESPN and CBS are archived only from 2026-11-14 on.
* Sample is defined pregame: player-weeks where every source projects him and
  at least one projects >= MIN_PROJECTION. A player with no stat line scored 0.
  CBS lists ~100 per position, so this is fantasy-relevant players, not depth.
* CBS is rescaled to Sleeper per position-week from projections only, never
  from actuals.
* Uncertainty: paired MAE differences vs Sleeper with a player-cluster
  bootstrap, and a time split (first half of weeks vs second half).
"""
from __future__ import annotations

import gzip
import json
import os
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import cbs, espn, sleeper  # noqa: E402

# ------------------------------------------------------------------ settings
SEASON = 2026
WEEK1_SUNDAY = date(2026, 9, 13)      # first-Sunday kickoff anchors the pregame cutoff
CUTOFF_UTC_HOUR = 17                  # 17:00 UTC Sunday: before every 1pm ET kickoff
MIN_PROJECTION = 5.0
BOOTSTRAPS = 2000
CACHE = ROOT / "source_eval_cache"    # past weeks never change; refetch is skipped
ARCHIVE = ROOT / "site" / "data" / "projection_archive"
OUTPUT = ROOT / "site" / "data" / "source_quality.json"
SOURCES = ["Sleeper", "ESPN", "CBS", "Sleeper+ESPN", "Mean of 3"]


def _get(url, headers=None, binary=False):
    last = None
    for attempt in range(3):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 nfl-top25-rankings/1.0", **(headers or {})})
            with urlopen(req, timeout=60) as response:
                body = response.read()
                return body.decode("utf-8", errors="ignore") if binary else json.loads(body)
        except Exception as exc:  # retried, then raised
            last = exc
    raise RuntimeError(f"{url}: {last}")


def _cached(name, fetch):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / name
    if path.exists():
        return json.loads(path.read_text())
    value = fetch()
    path.write_text(json.dumps(value))
    return value


def cutoff(week: int) -> datetime:
    sunday = WEEK1_SUNDAY + timedelta(days=7 * (week - 1))
    return datetime(sunday.year, sunday.month, sunday.day, CUTOFF_UTC_HOUR, tzinfo=timezone.utc)


def completed_weeks(now=None) -> list[int]:
    now = now or datetime.now(timezone.utc)
    # A week is complete once its Monday game is over: Tuesday 12:00 UTC.
    return [w for w in range(1, 19) if cutoff(w) + timedelta(days=1, hours=19) <= now]


# ------------------------------------------------------------- pregame archive
def archived_pregame() -> dict[int, dict[str, object]]:
    """Latest pre-cutoff capture per week per source, from the projection archive."""
    found: dict[int, dict[str, tuple]] = {}
    for path in (ARCHIVE / "snapshots").glob("*.json.gz"):
        try:
            snap = json.loads(gzip.decompress(path.read_bytes()))
        except (OSError, ValueError):
            continue
        model = snap.get("projection_model") or {}
        week, refs = model.get("week"), snap.get("raw_input_refs") or {}
        captured = pd.Timestamp(snap.get("captured_utc"))
        if week is None or captured.tzinfo is None or captured >= pd.Timestamp(cutoff(int(week))):
            continue
        for name in ("sleeper_projections", "espn_projections", "cbs_projections"):
            if name in refs and captured > found.get(int(week), {}).get(name, (pd.Timestamp.min.tz_localize("UTC"),))[0]:
                found.setdefault(int(week), {})[name] = (captured, refs[name])
    out: dict[int, dict[str, object]] = {}
    for week, items in found.items():
        for name, (_, ref) in items.items():
            raw = json.loads(gzip.decompress((ARCHIVE / ref).read_bytes()))
            if name == "sleeper_projections":
                raw = raw.get(f"regular/{SEASON}/{week}")
                if raw is None:
                    continue
            out.setdefault(week, {})[name] = raw
    return out


# ------------------------------------------------------------------ one week
def espn_week(week: int) -> pd.DataFrame:
    def fetch():
        filt = {"players": {"filterSlotIds": {"value": [0, 2, 4, 6, 17, 16]},
                "filterStatsForTopScoringPeriodIds": {"value": 2, "additionalValue": [f"11{SEASON}{week}"]},
                "limit": 2000, "sortPercOwned": {"sortAsc": False, "sortPriority": 1}}}
        raw = _get(espn.URL.format(season=SEASON, week=week),
                   {"X-Fantasy-Filter": json.dumps(filt, separators=(",", ":"))})
        frame = espn.projection_frame(raw, week, sleeper.normalize_name)
        return espn.compact(frame)
    return pd.DataFrame(_cached(f"espn_{week}.json", fetch))


def cbs_week(week: int) -> pd.DataFrame:
    def fetch():
        pages = {pos: cbs.fetch_page(pos, SEASON, week) for pos in cbs.PAGES}
        return cbs.projection_frame(pages, sleeper.normalize_name, sleeper.normalize_team).to_dict("records")
    return pd.DataFrame(_cached(f"cbs_{week}.json", fetch))


def week_frame(week: int, players: pd.DataFrame, pregame: dict) -> tuple[pd.DataFrame, dict]:
    origin = {}
    sp = pregame.get("sleeper_projections")
    origin["Sleeper"] = "pregame archive" if sp is not None else "refetched"
    if sp is None:
        sp = _cached(f"sleeper_proj_{week}.json",
                     lambda: _get(f"{sleeper.BASE_URL}/projections/nfl/regular/{SEASON}/{week}"))
    stats = _cached(f"sleeper_stats_{week}.json",
                    lambda: _get(f"{sleeper.BASE_URL}/stats/nfl/regular/{SEASON}/{week}"))
    s = sleeper.projection_frame(sp).rename(columns={"Sleeper_FP": "Sleeper"})
    a = pd.DataFrame([(pid, (v or {}).get("pts_half_ppr")) for pid, v in stats.items()],
                     columns=["player_id", "Actual"])
    d = players.merge(s, on="player_id", how="left").merge(a, on="player_id", how="left")
    d["Key"] = np.where(d["Position"].eq("DEF"), d["player_id"], d["Key"])

    e_raw = pregame.get("espn_projections")
    origin["ESPN"] = "pregame archive" if e_raw is not None else "refetched"
    e = pd.DataFrame(e_raw) if e_raw is not None else espn_week(week)
    e["Key"] = np.where(e["Position"].eq("DEF"), e["Team"], e["ESPN_Name"].map(sleeper.normalize_name))
    e = e.rename(columns={"ESPN_FP": "ESPN"})[["Key", "Position", "ESPN"]]
    e = e.drop_duplicates(["Key", "Position"], keep=False)

    c_raw = pregame.get("cbs_projections")
    origin["CBS"] = "pregame archive" if c_raw is not None else "refetched"
    c = pd.DataFrame(c_raw) if c_raw is not None else cbs_week(week)
    c = c.rename(columns={"CBS_FP": "CBS_raw"})[["Key", "Position", "CBS_raw"]]
    c = c.drop_duplicates(["Key", "Position"], keep=False)

    d = d.merge(e, on=["Key", "Position"], how="left").merge(c, on=["Key", "Position"], how="left")
    d["week"] = week
    return d, origin


# ------------------------------------------------------------------ scoring
def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    ok = d["Sleeper"].ge(3) & d["CBS_raw"].ge(3)
    ratio = (d["Sleeper"] / d["CBS_raw"]).where(ok)
    grp = d.assign(r=ratio).groupby(["week", "Position"])["r"]
    factor = grp.transform("median").where(grp.transform("count") >= 8, 1.0)
    d["CBS"] = d["CBS_raw"] * factor
    d["Actual"] = d["Actual"].fillna(0.0)
    d = d[d[["Sleeper", "ESPN", "CBS"]].gt(0).all(axis=1)
          & d[["Sleeper", "ESPN", "CBS"]].max(axis=1).ge(MIN_PROJECTION)].copy()
    d["Sleeper+ESPN"] = (d["Sleeper"] + d["ESPN"]) / 2
    d["Mean of 3"] = d[["Sleeper", "ESPN", "CBS"]].mean(axis=1)
    return d


def metrics(d: pd.DataFrame) -> list[dict]:
    rows = []
    for src in SOURCES:
        err = d[src] - d["Actual"]
        rho = [g[src].rank().corr(g["Actual"].rank()) for _, g in d.groupby(["week", "Position"]) if len(g) >= 8]
        slope = float(np.polyfit(d[src], d["Actual"], 1)[0]) if len(d) > 2 else float("nan")
        rows.append({"source": src, "n": int(len(d)), "mae": round(float(err.abs().mean()), 3),
                     "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
                     "bias": round(float(err.mean()), 3),
                     "rank_rho": round(float(np.nanmean(rho)), 3) if rho else None,
                     "calibration_slope": round(slope, 3)})
    return rows


def paired_vs_sleeper(d: pd.DataFrame, seed: int = 7) -> list[dict]:
    rng = np.random.default_rng(seed)
    ids = d["player_id"].to_numpy()
    groups = {p: np.flatnonzero(ids == p) for p in np.unique(ids)}
    keys = list(groups)
    base = (d["Sleeper"] - d["Actual"]).abs().to_numpy()
    out = []
    for src in SOURCES[1:]:
        diff = (d[src] - d["Actual"]).abs().to_numpy() - base
        boots = [diff[np.concatenate([groups[k] for k in rng.choice(keys, len(keys))])].mean()
                 for _ in range(BOOTSTRAPS)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        out.append({"source": src, "mae_minus_sleeper": round(float(diff.mean()), 3),
                    "ci95": [round(float(lo), 3), round(float(hi), 3)],
                    "significant": bool(hi < 0 or lo > 0)})
    return out


def markdown(report: dict) -> str:
    lines = [f"### Projection source quality, {SEASON} weeks {report['weeks'][0]}-{report['weeks'][-1]}",
             f"{report['sample']['player_weeks']} player-weeks, {report['sample']['players']} players. "
             "Lower MAE is better; ranking ρ is within position-week.", "",
             "| Source | MAE | Bias | Rank ρ | Calib. |", "|---|---|---|---|---|"]
    for r in report["overall"]:
        lines.append(f"| {r['source']} | {r['mae']:.2f} | {r['bias']:+.2f} | {r['rank_rho']:.3f} | {r['calibration_slope']:.2f} |")
    lines += ["", "**MAE vs Sleeper** (95% CI, player-cluster bootstrap)", ""]
    for r in report["paired_vs_sleeper"]:
        star = " ✅" if r["significant"] and r["mae_minus_sleeper"] < 0 else ""
        lines.append(f"- {r['source']}: {r['mae_minus_sleeper']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}]{star}")
    split = report.get("time_split")
    if split:
        lines += ["", f"**Held-out later weeks {split['test_weeks']}:** " + ", ".join(
            f"{r['source']} {r['mae']:.2f}" for r in split["test"])]
    refetched = sorted({f"{s} wk{w}" for w, o in report["origin"].items() for s, v in o.items() if v == "refetched"})
    if refetched:
        lines += ["", f"_Refetched (not pregame-verified): {len(refetched)} source-weeks._"]
    return "\n".join(lines) + "\n"


def main() -> int:
    weeks = completed_weeks()
    if not weeks:
        print("No completed weeks yet.")
        return 0
    raw_players, _ = sleeper.fetch_players(ROOT / "sleeper_cache")
    players = sleeper.players_frame(raw_players)[["player_id", "Sleeper_Name", "Key", "Team", "Position"]]
    pregame = archived_pregame()
    frames, origin = [], {}
    for week in weeks:
        try:
            frame, origin[week] = week_frame(week, players, pregame.get(week, {}))
            frames.append(frame)
        except Exception as exc:
            print(f"week {week} skipped: {exc}")
    if not frames:
        print("No week could be built.")
        return 1
    d = prepare(pd.concat(frames, ignore_index=True))
    if d.empty:
        print("Empty evaluation sample.")
        return 1
    half = weeks[len(weeks) // 2:]
    report = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "season": SEASON, "weeks": weeks,
        "sample": {"player_weeks": int(len(d)), "players": int(d["player_id"].nunique()),
                   "min_projection": MIN_PROJECTION},
        "overall": metrics(d),
        "by_position": {pos: metrics(g) for pos, g in d.groupby("Position") if len(g) >= 40},
        "paired_vs_sleeper": paired_vs_sleeper(d),
        "time_split": {"test_weeks": half, "test": metrics(d[d["week"].isin(half)])} if len(weeks) >= 4 else None,
        "by_week_mae": {int(w): {s: round(float((g[s] - g["Actual"]).abs().mean()), 3) for s in SOURCES}
                        for w, g in d.groupby("week")},
        "origin": {int(w): o for w, o in origin.items()},
    }
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    text = markdown(report)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
