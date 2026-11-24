#!/usr/bin/env python3
"""Build grader actuals (Yahoo scoring) from nflverse for every archived game.

Writes the CSV that tools/evaluate_projection_archive.py reads:
game_id, player_key, actual_fp, realized_at_utc. Identities come from the
pregame archive (Yahoo IDs, mapped to nflverse GSIS IDs); defenses are scored
from play-by-play. A player with no weekly stat row is left out, never zero.
Snapshots listed in excluded_snapshots.json do not contribute identities.

    python tools/build_actuals.py --output model/actual_results_2026.csv
"""
import argparse
import gzip
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.projection_archive import load_exclusions  # noqa: E402
from pipeline.showdown_archive_backtest import NflverseActuals, _pandas  # noqa: E402

ARCHIVE = ROOT / "site/data/projection_archive"
# nflreadpy fetches this through a github.com redirect some proxies refuse.
PLAYER_IDS_FALLBACK = (
    "https://raw.githubusercontent.com/dynastyprocess/data/master/files/db_playerids.csv"
)
# nflverse publishes no final-whistle time; a game is treated as realized this
# long after kickoff, and later games are skipped until then.
REALIZED_AFTER = pd.Timedelta(hours=5)


def archived_games(archive=ARCHIVE, excluded=None):
    """{game_id: {"home", "away", "kickoff_utc", "players": {key: {pos, team}}}}."""
    excluded = set(excluded or ())
    games = {}
    for path in sorted((Path(archive) / "snapshots").glob("*.json.gz")):
        snapshot = json.loads(gzip.decompress(path.read_bytes()))
        if snapshot.get("snapshot_id") in excluded:
            continue
        for model in snapshot.get("showdown_models") or []:
            game = games.setdefault(str(model["game_id"]), {"players": {}})
            game.update(home=model["home"], away=model["away"])
        for row in snapshot.get("predictions") or []:
            game = games.setdefault(str(row["game_id"]), {"players": {}})
            game.setdefault("kickoff_utc", row["kickoff_utc"])
            game["players"][row["player_key"]] = {"pos": row["position"], "team": row["team"]}
    return games


def _infer_teams(game, schedules):
    """Home/away for a game archived without a Showdown model, from the schedule."""
    teams = {p["team"] for p in game["players"].values()}
    kickoff = pd.Timestamp(game["kickoff_utc"]).tz_convert("UTC").date()
    found = schedules[schedules.home_team.isin(teams) & schedules.away_team.isin(teams)]
    found = found[found.game_date.map(lambda day: abs((day - kickoff).days) <= 1)]
    if len(found) != 1:
        return None
    return found.iloc[0].home_team, found.iloc[0].away_team


def build(games, nflverse, now=None):
    """Return (actuals frame, audit) for every archived game that is final."""
    now = pd.Timestamp(now) if now is not None else pd.Timestamp.now(tz="UTC")
    rows = []
    audit = {"games_graded": 0, "games_pending": 0, "no_schedule_match": [],
             "not_final": [], "players_without_stat_row": 0}
    for game_id, game in sorted(games.items()):
        kickoff = pd.Timestamp(game["kickoff_utc"])
        if kickoff + REALIZED_AFTER > now:
            audit["games_pending"] += 1
            continue
        if "home" not in game:
            teams = _infer_teams(game, nflverse.schedules)
            if teams is None:
                audit["no_schedule_match"].append(game_id)
                continue
            game = dict(game, home=teams[0], away=teams[1])
        keys = list(game["players"])
        model = {"home": game["home"], "away": game["away"], "kickoff_utc": game["kickoff_utc"],
                 "players": [game["players"][key] for key in keys]}
        actuals, detail = nflverse.actuals(model, keys)
        if detail.get("reason"):
            audit["not_final"].append({"game_id": game_id, "reason": detail["reason"]})
            continue
        audit["games_graded"] += 1
        audit["players_without_stat_row"] += len(detail["missing_without_zero_fill"])
        realized = (kickoff + REALIZED_AFTER).tz_convert("UTC").isoformat()
        rows += [{"game_id": game_id, "player_key": key, "actual_fp": value,
                  "realized_at_utc": realized} for key, value in actuals.items()]
    frame = pd.DataFrame(rows, columns=["game_id", "player_key", "actual_fp", "realized_at_utc"])
    return frame, audit


def load_nflverse(seasons):
    import nflreadpy as nfl
    try:
        player_ids = _pandas(nfl.load_ff_playerids())
    except Exception:  # proxied environments: same file, direct host
        player_ids = pd.read_csv(PLAYER_IDS_FALLBACK, dtype=str)
    return NflverseActuals.prepare(
        _pandas(nfl.load_schedules(seasons)),
        _pandas(nfl.load_player_stats(seasons, summary_level="week")),
        player_ids,
        _pandas(nfl.load_pbp(seasons)),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, default=2026)
    parser.add_argument("--archive", default=str(ARCHIVE))
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    archive = Path(args.archive)
    games = archived_games(archive, load_exclusions(archive / "excluded_snapshots.json"))
    frame, audit = build(games, load_nflverse([args.season]))
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.sort_values(["game_id", "player_key"]).to_csv(target, index=False)
    print(json.dumps(dict(audit, rows=len(frame)), indent=2))


if __name__ == "__main__":
    main()
