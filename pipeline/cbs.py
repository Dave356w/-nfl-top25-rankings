"""CBS Sports: a third projection, used only to referee Sleeper/ESPN conflicts.

CBS publishes weekly projections as server-rendered tables, one page per
position, with no key:
``/fantasy/football/stats/{POS}/{season}/{week}/projections/nonppr/``.
Half-PPR is CBS's standard total plus half a point per projected reception.

CBS does not move the mean of a player the other two agree on. It is read
only where the blend flagged a conflict:

* ``sources disagree`` (Sleeper and ESPN > 40% apart): the mean becomes the
  median of the three -- whichever of Sleeper/ESPN CBS sides with wins, and
  a single outlier cannot drag the number.
* ``ESPN only (starter out)`` (a backup promoted on injury news Sleeper has
  not absorbed): the mean becomes the average of ESPN and CBS, both of which
  have had the news; CBS siding with Sleeper is called out for review.
* ``ESPN: out`` while Sleeper still projects him: the player stays at zero
  (ESPN's status is the freshest signal), but a CBS projection above
  `OUT_CHECK_FP` is called out for review.

Like ESPN, CBS is a second opinion: any failure leaves the blend untouched.
"""
from __future__ import annotations

import html
import re
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ settings
URL = "https://www.cbssports.com/fantasy/football/stats/{pos}/{season}/{week}/projections/nonppr/"
PAGES = {"QB": "QB", "RB": "RB", "WR": "WR", "TE": "TE", "K": "K", "DEF": "DST"}
OUT_CHECK_FP = 3.0         # CBS projection that makes an ESPN "out" worth a look
# CBS's standard scoring is not Yahoo half-PPR (passing TDs and defenses score
# differently). Each position's CBS numbers are rescaled to Sleeper's by the
# median ratio over players both project at least CALIBRATION_MIN_FP.
CALIBRATION_MIN_FP = 3.0
CALIBRATION_MIN_PLAYERS = 8
FRAME_COLUMNS = ["CBS_Name", "Key", "Team", "Position", "CBS_FP"]


class CBSError(RuntimeError):
    """A CBS page failed or no longer parses."""


# ------------------------------------------------------------------ fetching
def fetch_page(position: str, season: int, week: int, timeout: float = 30, attempts: int = 3) -> str:
    url = URL.format(pos=PAGES[position], season=int(season), week=int(week))
    last = None
    for attempt in range(1, attempts + 1):
        try:
            req = Request(url, headers={"User-Agent": "Mozilla/5.0 nfl-top25-rankings/1.0"})
            with urlopen(req, timeout=timeout) as response:
                return response.read().decode("utf-8", errors="ignore")
        except (HTTPError, URLError, TimeoutError) as exc:
            last = exc
            if attempt < attempts:
                time.sleep(1.0 * attempt)
    raise CBSError(f"CBS {position} projections failed: {last}")


def fetch(season: int, week: int) -> dict[str, str]:
    """All position pages, raw HTML keyed by our position code."""
    return {pos: fetch_page(pos, season, week) for pos in PAGES}


# ---------------------------------------------------------------- transforms
def _text(fragment: str) -> str:
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def parse_page(page: str, position: str, normalize_name, normalize_team) -> pd.DataFrame:
    """One position page -> rows of name, team, half-PPR projection.

    Columns are located by their header abbreviations (``fpts``, ``rec``), not
    by index, because each position's table has a different layout.
    """
    thead = re.search(r"<thead.*?</thead>", page, re.S)
    if not thead:
        raise CBSError(f"CBS {position}: no table header")
    header_rows = re.findall(r"<tr[^>]*>(.*?)</tr>", thead.group(0), re.S)
    heads = [_text(h).split(" ")[0].lower() for h in
             re.findall(r"<th[^>]*>(.*?)</th>", header_rows[-1], re.S)]
    if "fpts" not in heads:
        raise CBSError(f"CBS {position}: no fpts column in {heads}")
    i_fp = heads.index("fpts")
    i_rec = heads.index("rec") if "rec" in heads else None
    rows = []
    for body in re.findall(r'<tr class="TableBase-bodyTr[^"]*">(.*?)</tr>', page, re.S):
        cells = re.findall(r"<td[^>]*>(.*?)</td>", body, re.S)
        if len(cells) <= i_fp:
            continue
        if position == "DEF":
            code = re.search(r'/nfl/teams/([A-Z]+)/', cells[0])
            if not code:
                continue
            team = normalize_team(code.group(1))
            name = _text(cells[0])
        else:
            long = re.search(r'CellPlayerName--long.*?<a[^>]*>(.*?)</a>.*?'
                             r'CellPlayerName-team">\s*([A-Z]+)', cells[0], re.S)
            if not long:
                continue
            name, team = html.unescape(long.group(1)).strip(), normalize_team(long.group(2))
        fp = pd.to_numeric(_text(cells[i_fp]), errors="coerce")
        rec = pd.to_numeric(_text(cells[i_rec]), errors="coerce") if i_rec is not None else 0.0
        if pd.isna(fp):
            continue
        rows.append((name, normalize_name(name) if position != "DEF" else team, team, position,
                     round(float(fp) + 0.5 * (0.0 if pd.isna(rec) else float(rec)), 2)))
    return pd.DataFrame(rows, columns=FRAME_COLUMNS)


def projection_frame(pages: dict[str, str], normalize_name, normalize_team) -> pd.DataFrame:
    frames = [parse_page(page, pos, normalize_name, normalize_team) for pos, page in pages.items()]
    frame = pd.concat(frames, ignore_index=True)
    if frame.empty:
        raise CBSError("CBS pages parsed to no players")
    return frame


def attach(players: pd.DataFrame, cbs: pd.DataFrame) -> pd.Series:
    """CBS projection per Sleeper player row (name + team + position, unique)."""
    unique = cbs.drop_duplicates(["Key", "Team", "Position"], keep=False)
    lookup = dict(zip(zip(unique["Key"], unique["Team"], unique["Position"]), unique["CBS_FP"]))
    keys = np.where(players["Position"].eq("DEF"), players["Team"], players["Key"])
    return pd.Series([lookup.get((k, t, p), np.nan) for k, t, p in
                      zip(keys, players["Team"], players["Position"])], index=players.index)


def referee(players: pd.DataFrame, blended: pd.DataFrame, cbs: pd.DataFrame
            ) -> tuple[pd.DataFrame, list[str], list[dict]]:
    """Apply the module rules to flagged rows. Returns (blended, notes, conflicts)."""
    work = blended.merge(players[["player_id", "Sleeper_Name", "Key", "Team", "Position"]],
                         on="player_id", how="left")
    raw_c = attach(work, cbs)
    s = pd.to_numeric(work["Sleeper_Raw_FP"], errors="coerce")
    e = pd.to_numeric(work["ESPN_FP"], errors="coerce")
    flag = work["Projection_Flag"].astype("object")
    scale = {}
    for position, rows in work.groupby("Position").groups.items():
        both = (s[rows].ge(CALIBRATION_MIN_FP) & raw_c[rows].ge(CALIBRATION_MIN_FP)
                & flag[rows].isna())
        ratio = (s[rows][both] / raw_c[rows][both])
        scale[position] = float(ratio.median()) if len(ratio) >= CALIBRATION_MIN_PLAYERS else 1.0
    work["CBS_FP"] = (raw_c * work["Position"].map(scale)).round(2)
    c = pd.to_numeric(work["CBS_FP"], errors="coerce")
    has_c = c.notna()
    verdict = pd.Series(pd.NA, index=work.index, dtype="object")

    disagree = flag.eq("sources disagree")
    rows = disagree & has_c
    work.loc[rows, "Sleeper_FP"] = pd.concat([s, e, c], axis=1)[rows].median(axis=1).round(2)
    verdict[rows] = np.where((c - e).abs()[rows] <= (c - s).abs()[rows],
                             "CBS sides with ESPN", "CBS sides with Sleeper")

    news = flag.eq("ESPN only (starter out)")
    rows = news & has_c
    work.loc[rows, "Sleeper_FP"] = ((e[rows] + c[rows]) / 2).round(2)
    s_ref = s.fillna(0.0)
    verdict[rows] = np.where((c - e).abs()[rows] <= (c - s_ref).abs()[rows],
                             "CBS confirms the promotion", "CBS sides with Sleeper -- review")

    out = flag.eq("ESPN: out")
    verdict[out & c.ge(OUT_CHECK_FP)] = "CBS still projects him -- review"
    verdict[out & has_c & c.lt(OUT_CHECK_FP)] = "CBS agrees: out"
    verdict[(disagree | news | out) & ~has_c] = "not on CBS -- unconfirmed"
    work["CBS_Verdict"] = verdict
    # The page's review column shows the flag; carry the verdict with it.
    work["Projection_Flag"] = flag.where(verdict.isna(), flag.astype(str) + "; " + verdict.astype(str))

    reviewed = work[verdict.notna()].sort_values("Sleeper_FP", ascending=False)
    conflicts = [{
        "player": r.Sleeper_Name, "team": r.Team, "pos": r.Position, "flag": r.Projection_Flag,
        "sleeper": None if pd.isna(r.Sleeper_Raw_FP) else float(r.Sleeper_Raw_FP),
        "espn": None if pd.isna(r.ESPN_FP) else float(r.ESPN_FP),
        "cbs": None if pd.isna(r.CBS_FP) else float(r.CBS_FP),
        "final": float(r.Sleeper_FP), "verdict": r.CBS_Verdict,
    } for r in reviewed.itertuples()]
    notes = [f"CBS referee: {len(conflicts)} flagged player(s) checked, "
             f"{int(has_c[disagree | news | out].sum())} found on CBS; scale to Sleeper "
             + ", ".join(f"{k} x{v:.2f}" for k, v in sorted(scale.items())) + "."]
    for r in conflicts[:15]:
        notes.append(f"  {r['player']} ({r['team']} {r['pos']}): S={r['sleeper']} E={r['espn']} "
                     f"C={r['cbs']} -> {r['final']} [{r['verdict']}]")
    keep = list(blended.columns) + ["CBS_FP", "CBS_Verdict"]
    return work[keep], notes, conflicts


def summary_markdown(conflicts: list[dict], week) -> str:
    """The GitHub Actions job summary: one table, scannable on a phone."""
    if not conflicts:
        return f"### Projection conflicts, week {week}\n\nNone flagged.\n"
    lines = [f"### Projection conflicts, week {week}", "",
             "| Player | S | E | C | Final | Verdict |", "|---|---|---|---|---|---|"]
    fmt = lambda v: "–" if v is None else f"{v:.1f}"
    for r in conflicts:
        lines.append(f"| {r['player']} ({r['team']} {r['pos']}) | {fmt(r['sleeper'])} | "
                     f"{fmt(r['espn'])} | {fmt(r['cbs'])} | {r['final']:.1f} | {r['verdict']} |")
    return "\n".join(lines) + "\n"
