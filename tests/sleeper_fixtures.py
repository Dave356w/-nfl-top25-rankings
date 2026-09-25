"""Offline stand-ins for Sleeper's /players/nfl and /projections responses.

The tests build the raw JSON shapes Sleeper returns and push them through the
shipped transforms in `pipeline.sleeper`, so what is exercised downstream is the
real join, depth and availability logic rather than a restatement of it.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import sleeper

TEAMS = {"NE": "New England Patriots", "SEA": "Seattle Seahawks", "KC": "Kansas City Chiefs",
         "JAX": "Jacksonville Jaguars", "LAR": "Los Angeles Rams"}


def player(pid, name, team, position, slot=None, order=None, status="Active",
           injury=None, yahoo_id=None, practice=None):
    first, _, last = name.partition(" ")
    return str(pid), {
        "player_id": str(pid), "full_name": name, "first_name": first,
        "last_name": last, "team": team, "position": position,
        "depth_chart_position": slot, "depth_chart_order": order,
        "status": status, "injury_status": injury, "active": status == "Active",
        "yahoo_id": yahoo_id, "practice_participation": practice,
    }


def defense(team, name=None):
    first, _, last = (name or TEAMS.get(team, team + " Defense")).rpartition(" ")
    return team, {"player_id": team, "first_name": first, "last_name": last,
                  "team": team, "position": "DEF", "active": True}


def dump(rows, teams=()):
    """`rows` are `player(...)` results; `teams` adds a DEF entry per code."""
    players = dict(rows)
    players.update(defense(team) for team in teams)
    return players


def projections(points):
    """`points` maps player_id -> half-PPR points, in the dict-of-stats shape."""
    return {pid: {"pts_half_ppr": value, "pts_ppr": value + 1.0}
            for pid, value in points.items()}


def reference(raw_players, raw_projections):
    frame = sleeper.players_frame(raw_players)
    return sleeper.build_reference(frame, sleeper.projection_frame(raw_projections))


def stub_loader(raw_players, raw_projections, season=2026, week=2):
    """A drop-in for `notebook.load_sleeper_reference` that never touches the network."""
    ref = reference(raw_players, raw_projections)
    context = {"season": season, "week": week, "season_type": "regular",
               "players_fetched_utc": "2026-09-10T12:00:00+00:00"}

    def load(players, cfg=None):
        return ref, dict(context)
    return load
