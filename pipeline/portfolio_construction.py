"""Portfolio-level construction rules for Yahoo NFL Showdown.

Candidate generation stays broad.  Construction rules are applied only when the
portfolio is selected, so one archetype cannot remove candidates needed by a
second archetype.  Exposure and lineup-overlap caps remain global across the
whole portfolio.

A rule is a mapping with:

* ``name``: display label.
* ``count``: relative quota.  The default counts sum to 20; for any other entry
  count they are proportionally re-apportioned with the largest-remainder method.
* ``positions``: exact counts (integer) or inclusive ``[min, max]`` ranges.

Example::

    {"name": "2QB / 2WR / 1RB", "count": 4,
     "positions": {"QB": 2, "RB": 1, "WR": 2, "TE": 0, "DEF": 0}}

Yahoo itself does not require these shapes.  They are portfolio strategy rules.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from typing import Iterable, Mapping, Sequence

import pandas as pd

POSITIONS = ("QB", "RB", "WR", "TE", "DEF")

# A 20-entry default mix.  Every archetype is an exact five-player positional
# signature, which makes the portfolio intentionally different rather than only
# putting broad global min/max limits on every lineup.
DEFAULT_PORTFOLIO_CONSTRUCTION_RULES = (
    {
        "name": "2QB / 2WR / 1RB",
        "count": 4,
        "positions": {"QB": 2, "RB": 1, "WR": 2, "TE": 0, "DEF": 0},
    },
    {
        "name": "2QB / 1WR / 1RB / 1TE",
        "count": 3,
        "positions": {"QB": 2, "RB": 1, "WR": 1, "TE": 1, "DEF": 0},
    },
    {
        "name": "1QB / 2WR / 1RB / 1TE",
        "count": 4,
        "positions": {"QB": 1, "RB": 1, "WR": 2, "TE": 1, "DEF": 0},
    },
    {
        "name": "1QB / 2WR / 1RB / 1DEF",
        "count": 3,
        "positions": {"QB": 1, "RB": 1, "WR": 2, "TE": 0, "DEF": 1},
    },
    {
        "name": "1QB / 1WR / 2RB / 1TE",
        "count": 2,
        "positions": {"QB": 1, "RB": 2, "WR": 1, "TE": 1, "DEF": 0},
    },
    {
        "name": "1QB / 1WR / 2RB / 1DEF",
        "count": 2,
        "positions": {"QB": 1, "RB": 2, "WR": 1, "TE": 0, "DEF": 1},
    },
    {
        "name": "1QB / 3WR / 1RB",
        "count": 2,
        "positions": {"QB": 1, "RB": 1, "WR": 3, "TE": 0, "DEF": 0},
    },
)


def serializable_rules(rules=None):
    """Return a JSON-safe copy of the configured rules."""
    rules = DEFAULT_PORTFOLIO_CONSTRUCTION_RULES if rules is None else rules
    return [
        {
            "name": str(rule["name"]),
            "count": int(rule.get("count", 1)),
            "positions": {
                str(position): (
                    int(spec)
                    if isinstance(spec, int)
                    else [int(spec[0]), int(spec[1])]
                )
                for position, spec in dict(rule.get("positions") or {}).items()
            },
        }
        for rule in rules
    ]


def _limits(spec):
    if isinstance(spec, int):
        return int(spec), int(spec)
    if not isinstance(spec, (list, tuple)) or len(spec) != 2:
        raise ValueError(f"Construction limit must be an integer or [min, max], got {spec!r}")
    low, high = int(spec[0]), int(spec[1])
    if low < 0 or high < low:
        raise ValueError(f"Invalid construction range {spec!r}")
    return low, high


def validate_rules(rules: Sequence[Mapping], lineup_size: int = 5):
    """Validate rule names, quotas, positions and exact-signature totals."""
    names = set()
    for rule in rules:
        name = str(rule.get("name") or "").strip()
        if not name:
            raise ValueError("Every portfolio construction rule needs a name.")
        if name in names:
            raise ValueError(f"Duplicate portfolio construction rule name: {name}")
        names.add(name)
        if int(rule.get("count", 0)) <= 0:
            raise ValueError(f"Construction rule {name!r} needs a positive count.")
        positions = dict(rule.get("positions") or {})
        unknown = set(positions) - set(POSITIONS)
        if unknown:
            raise ValueError(f"Construction rule {name!r} has unknown positions: {sorted(unknown)}")
        exact_total = 0
        all_exact = True
        for spec in positions.values():
            low, high = _limits(spec)
            if high > lineup_size:
                raise ValueError(f"Construction rule {name!r} exceeds lineup size {lineup_size}.")
            if low != high:
                all_exact = False
            exact_total += low
        if all_exact and len(positions) == len(POSITIONS) and exact_total != lineup_size:
            raise ValueError(
                f"Construction rule {name!r} selects {exact_total} players, expected {lineup_size}."
            )
    return True


def apportioned_rule_counts(rules: Sequence[Mapping], target: int):
    """Scale relative rule counts to exactly ``target`` entries.

    The default quotas total 20.  If the browser asks for 10 entries, for example,
    the same mix is retained proportionally instead of silently trying to build 20.
    """
    target = int(target)
    if target <= 0:
        return []
    validate_rules(rules)
    weights = [max(0, int(rule.get("count", 1))) for rule in rules]
    total = sum(weights)
    if total <= 0:
        return [0] * len(rules)
    raw = [target * weight / total for weight in weights]
    counts = [math.floor(value) for value in raw]
    remaining = target - sum(counts)
    order = sorted(range(len(rules)), key=lambda i: (raw[i] - counts[i], weights[i], -i), reverse=True)
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def rule_schedule(rules: Sequence[Mapping], target: int):
    """Interleave rule slots so the first archetype cannot consume all exposures."""
    counts = apportioned_rule_counts(rules, target)
    remaining = counts[:]
    schedule = []
    while len(schedule) < int(target) and any(remaining):
        for index, rule in enumerate(rules):
            if remaining[index] <= 0:
                continue
            schedule.append(rule)
            remaining[index] -= 1
            if len(schedule) == int(target):
                break
    return schedule


def lineup_matches_rule(players: pd.DataFrame, ids: Iterable[int], rule: Mapping):
    """Return whether one roster satisfies one construction archetype."""
    counts = Counter(players.iloc[list(ids)]["Position"].astype(str))
    for position, spec in dict(rule.get("positions") or {}).items():
        low, high = _limits(spec)
        value = int(counts.get(position, 0))
        if value < low or value > high:
            return False
    return True


def select_portfolio(scored: pd.DataFrame, players: pd.DataFrame, cfg, rules=None):
    """Select a tournament portfolio with per-entry construction archetypes.

    Candidates are still ranked by ``Tournament_Score``.  For each scheduled
    archetype slot, take the highest-ranked remaining candidate that matches the
    requested positional shape and all global player/Superstar/overlap caps.
    """
    rules = serializable_rules(rules)
    validate_rules(rules, int(cfg.lineup_size))
    target = int(cfg.tournament_lineups)
    schedule = rule_schedule(rules, target)

    max_player_count = max(1, math.ceil(target * float(cfg.max_player_exposure)))
    max_superstar_count = max(1, math.ceil(target * float(cfg.max_superstar_exposure)))
    player_counts = Counter()
    superstar_counts = Counter()
    selected_rows = []
    selected_sets = []
    selected_indices = set()
    assigned_rules = []
    ordered = list(scored.sort_values("Tournament_Score", ascending=False).index)
    unfilled = Counter()

    for rule in schedule:
        picked = None
        picked_set = None
        for idx in ordered:
            if idx in selected_indices:
                continue
            candidate = scored.loc[idx]
            ids = tuple(candidate["Player_Ids"])
            superstar = int(candidate["Superstar_Id"])
            if not lineup_matches_rule(players, ids, rule):
                continue
            if any(player_counts[player] >= max_player_count for player in ids):
                continue
            if superstar_counts[superstar] >= max_superstar_count:
                continue
            candidate_set = set(ids)
            if any(len(candidate_set & prior) > int(cfg.max_shared_players) for prior in selected_sets):
                continue
            picked = idx
            picked_set = candidate_set
            break

        if picked is None:
            unfilled[str(rule["name"])] += 1
            continue

        candidate = scored.loc[picked]
        ids = tuple(candidate["Player_Ids"])
        superstar = int(candidate["Superstar_Id"])
        selected_rows.append(picked)
        selected_indices.add(picked)
        selected_sets.append(picked_set)
        player_counts.update(ids)
        superstar_counts.update([superstar])
        assigned_rules.append(str(rule["name"]))

    portfolio = scored.loc[selected_rows].copy().reset_index(drop=True)
    if len(portfolio):
        portfolio["Construction_Rule"] = assigned_rules
    else:
        portfolio["Construction_Rule"] = pd.Series(dtype=str)
    portfolio.attrs["construction_requested"] = target
    portfolio.attrs["construction_selected"] = len(portfolio)
    portfolio.attrs["construction_unfilled"] = dict(unfilled)

    if len(portfolio) < target:
        details = ", ".join(f"{name}: {count}" for name, count in unfilled.items()) or "unknown"
        warnings.warn(
            f"Built {len(portfolio)} of {target} tournament lineups under the construction, "
            f"exposure, and overlap rules. Unfilled archetype slots: {details}."
        )
    return portfolio
