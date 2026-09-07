"""Export one Yahoo single-game slate as a payload a browser can optimize.

The Colab showdown runner does four things: it fetches feeds, it builds a
correlation model, it enumerates and scores lineups, and it asks the user
questions. Only the first of those cannot happen in a browser -- Yahoo and the
two sportsbooks are not going to answer a cross-origin request from a Pages
origin, and pointing every visitor's IP at a sportsbook is how the Colab VM got
blocked in the first place.

So the split is: this module runs in Actions and publishes the *model*, and
`site/showdown.html` runs the enumeration and the simulation in the visitor's
browser. The model is small -- a 36-player pool plus the 630-term upper triangle
of the repaired latent correlation matrix is about 7 KB, 1 KB gzipped -- and it
is everything the interactive half needs. Nothing the page can change (which
players are in, the cap floor, position limits, the objective) alters the model,
so no feed is ever needed again once the payload is written.

Two properties of the model make that clean:

* Excluding a player is a principal submatrix of a PSD matrix, which is still
  PSD. The browser never needs an eigensolver or the PSD repair.
* Because of that, the page does not even re-simulate on an exclusion. It draws
  the full pool's scenarios once and exclusions only change which lineups are
  legal.

A reference portfolio is computed here as well, so the page has the
authoritative answer on screen before any JavaScript runs.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from pipeline import notebook as nb

SCHEMA = 1

# Fields the page needs per player. Short keys: the payload is mostly numbers and
# this is the difference between 7 KB and 11 KB per game.
PLAYER_FIELDS = ("name", "team", "pos", "salary", "fp", "cv", "role", "slot", "depth", "source")

# Settings the page exposes as controls. Published with the payload so the page
# starts from the same defaults the server-side reference run used, and so a
# change to Settings does not need a matching edit in the JavaScript.
EXPOSED_SETTINGS = (
    "lineup_size", "simulations", "random_seed", "tournament_lineups",
    "max_candidate_lineups", "mean_candidate_reserve", "min_salary_used_pct",
    "candidate_ceiling_weight", "near_optimal_ratio", "max_player_exposure",
    "max_superstar_exposure", "max_shared_players",
)

LINEUP_METRICS = (
    "Salary", "Expected_FP", "Analytic_SD", "Sim_Mean", "Sim_SD", "Floor_P25",
    "Ceiling_P90", "Ceiling_P95", "Near_Optimal_Rate", "Win_Rate", "Tournament_Score",
)


def _json_key(label):
    """Turn a report column heading into a stable JSON key."""
    text = str(label).strip().lower()
    for character in " -/":
        text = text.replace(character, "_")
    return "".join(c for c in text if c.isalnum() or c == "_").strip("_")


def _clean(value):
    """JSON has no NaN or numpy scalars."""
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None:
        return None
    if pd.api.types.is_scalar(value) and pd.isna(value):
        return None
    return value


def upper_triangle(matrix, decimals=5):
    """Flatten a symmetric matrix to its strict upper triangle, row by row.

    The page rebuilds it by walking the same order. Storing 630 numbers instead
    of 1,296 is not about bytes so much as about making it obvious in the payload
    that the matrix is symmetric with a unit diagonal.
    """
    size = len(matrix)
    return [
        round(float(matrix[i][j]), decimals)
        for i in range(size)
        for j in range(i + 1, size)
    ]


def player_rows(players, cv):
    rows = []
    for index, player in enumerate(players.itertuples()):
        rows.append({
            "name": str(player.Name),
            "team": str(player.Team),
            "pos": str(player.Position),
            "salary": round(float(player.Salary), 2),
            "fp": round(float(player.Projected_FP), 4),
            "cv": round(float(cv[index]), 5),
            "role": str(getattr(player, "Role_Label", "unknown")),
            "slot": _clean(getattr(player, "Role_Slot", None)),
            "depth": int(player.Depth_Rank),
            "source": str(player.Projection_Source),
        })
    return rows


def lineup_rows(frame, limit=None):
    """Serialize scored lineups as player indices into the published pool."""
    rows = []
    view = frame if limit is None else frame.head(limit)
    for lineup in view.to_dict("records"):
        row = {
            "ids": [int(value) for value in lineup["Player_Ids"]],
            "superstar": int(lineup["Superstar_Id"]),
        }
        for metric in LINEUP_METRICS:
            if metric in lineup:
                row[metric.lower()] = _clean(round(float(lineup[metric]), 5))
        rows.append(row)
    return rows


def build_game_payload(game_players, game, salary_cap, cfg=None, optimize=True):
    """Model one game and, by default, solve it once so the page opens on an answer.

    Returns `(payload, note)`. A game that cannot produce a Yahoo-valid roster --
    a team whose skill players were all filtered out, most often -- returns
    `(None, reason)` rather than raising, because one unusable game on a fifteen
    game slate must not cost the other fourteen.
    """
    cfg = nb._cfg(cfg)
    pool, dropped = nb.trim_player_pool(game_players, cfg)
    problem = nb.roster_feasibility_error(pool, int(cfg.lineup_size))
    if problem:
        return None, problem

    model = nb.build_correlation_model(pool)
    payload = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "game_id": str(game["Game ID"]),
        "matchup": str(game["Matchup"]),
        "away": str(game["Away Team"]),
        "home": str(game["Home Team"]),
        "kickoff_utc": _clean(game["Game Time"]),
        "salary_cap": round(float(salary_cap), 2),
        "settings": {name: _clean(getattr(cfg, name)) for name in EXPOSED_SETTINGS},
        "players": player_rows(pool, model["cv"]),
        "latent": upper_triangle(model["latent_corr"]),
        "model": {
            "psd_max_score_adjustment": round(float(model["psd_max_score_adjustment"]), 6),
            "infeasible_pairs": int(model["infeasible_pairs"]),
            "max_infeasible_shift": round(float(model["max_infeasible_shift"]), 6),
        },
        "dropped_from_pool": list(dropped),
        "reference": None,
    }
    if not optimize:
        return payload, None

    outcomes, _ = nb.simulate_player_outcomes(pool, cfg)
    covariance = nb.analytic_covariance(pool, model)
    candidates, valid_rosters = nb.enumerate_candidate_lineups(
        pool, salary_cap, covariance, cfg
    )
    scored = nb.score_candidates_shared_scenarios(candidates, outcomes, cfg)
    portfolio = nb.select_tournament_portfolio(scored, cfg)
    strongest = nb.select_strongest_lineups(scored, count=10)
    reliability = nb.reliability_report(scored)

    payload["reference"] = {
        "valid_rosters": int(valid_rosters),
        "candidates_scored": int(len(scored)),
        "portfolio": lineup_rows(portfolio),
        "strongest": lineup_rows(strongest),
        "reliability": [
            {_json_key(key): _clean(value) for key, value in row.items()}
            for row in reliability.to_dict("records")
        ],
    }
    return payload, None


def build_slate(cfg=None, optimize=True, max_games=None):
    """Fetch the slate once and export every game on it.

    Returns `(payloads, index)`. The index is what the page loads first: it names
    the games, so the page can offer a picker without downloading every model.
    """
    cfg = nb._cfg(cfg)
    slate = nb.prepare_slate_pool(cfg, "exporting single-game models")
    players = slate["players"]
    games = slate["games"]

    payloads, entries, notes = [], [], []
    for position, (_, game) in enumerate(games.iterrows()):
        if max_games is not None and position >= max_games:
            break
        game_id = str(game["Game ID"])
        game_players = players[players["Game ID"].eq(game_id)].copy()
        if game_players.empty:
            notes.append(f"{game['Matchup']}: no priced players survived the filters")
            continue
        salary_cap = slate["cap_map"].get(game_id)
        if salary_cap is None or not float(salary_cap) > 0:
            # The interactive runner asks the user for a cap here. A published
            # build has nobody to ask, and a guessed cap would silently change
            # which lineups are legal, so the game is skipped and said so.
            notes.append(f"{game['Matchup']}: Yahoo published no single-game salary cap")
            continue

        payload, problem = build_game_payload(
            game_players, game, float(salary_cap), cfg, optimize=optimize
        )
        if payload is None:
            notes.append(f"{game['Matchup']}: {problem}")
            continue
        payloads.append(payload)
        entries.append({
            "game_id": payload["game_id"],
            "matchup": payload["matchup"],
            "away": payload["away"],
            "home": payload["home"],
            "kickoff_utc": payload["kickoff_utc"],
            "salary_cap": payload["salary_cap"],
            "players": len(payload["players"]),
            "file": f"{payload['game_id']}.json",
            "optimized": payload["reference"] is not None,
        })
        print(f"  {payload['matchup']}: {len(payload['players'])} players exported")

    audit = slate["market_audit"]
    index = {
        "schema": SCHEMA,
        "status": "ok",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "games": entries,
        "skipped": notes,
        "market": {
            "feeds": list(audit.get("feeds") or []),
            "notes": list(audit.get("notes") or []),
            "calibration_pairs": _clean(audit.get("calibration_pairs")),
            "projection_audit": dict(audit.get("projection_audit") or {}),
        },
        "availability_removed": int(len(slate["nflverse_removed"])),
    }
    return payloads, index
