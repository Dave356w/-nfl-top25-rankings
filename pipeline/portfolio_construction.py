"""Portfolio-level construction rules for Yahoo NFL Showdown.

Construction rules are applied after candidate scoring.  The selector keeps the
portfolio-wide exposure and overlap constraints, but it also reserves exposure
that future archetype slots are forced to use.  That matters for structures such
as 2-QB lineups: when only two starting QBs are in the pool, every remaining
2-QB slot necessarily needs both of them, so a 1-QB lineup must not spend the
last unit of either QB's exposure first.

A rule is a mapping with:

* ``name``: display label.
* ``count``: relative quota.  The default counts sum to 20; other portfolio
  sizes are proportionally re-apportioned with the largest-remainder method.
* ``positions``: exact counts (integer) or inclusive ``[min, max]`` ranges.

Yahoo itself does not require these shapes.  They are portfolio strategy rules.
"""

from __future__ import annotations

import math
import warnings
from collections import Counter
from typing import Iterable, Mapping, Sequence

import pandas as pd

POSITIONS = ("QB", "RB", "WR", "TE", "DEF")
CANDIDATE_RESERVE_PER_RULE = 500

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
    """Scale relative rule counts to exactly ``target`` entries."""
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
    order = sorted(
        range(len(rules)),
        key=lambda i: (raw[i] - counts[i], weights[i], -i),
        reverse=True,
    )
    for index in order[:remaining]:
        counts[index] += 1
    return counts


def rule_schedule(rules: Sequence[Mapping], target: int):
    """Return the legacy interleaved quota schedule used as one repair attempt."""
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


def _candidate_metadata(scored: pd.DataFrame, players: pd.DataFrame, rules):
    """Precompute roster ids, matching candidates, and forced players per rule."""
    ordered = list(scored.sort_values("Tournament_Score", ascending=False).index)
    ids_by_index = {
        idx: tuple(int(value) for value in scored.at[idx, "Player_Ids"])
        for idx in scored.index
    }
    superstar_by_index = {
        idx: int(scored.at[idx, "Superstar_Id"])
        for idx in scored.index
    }
    position_values = players["Position"].astype(str).to_numpy()
    counts_by_index = {
        idx: Counter(position_values[list(ids)])
        for idx, ids in ids_by_index.items()
    }

    def matches(idx, rule):
        counts = counts_by_index[idx]
        for position, spec in dict(rule.get("positions") or {}).items():
            low, high = _limits(spec)
            value = int(counts.get(position, 0))
            if value < low or value > high:
                return False
        return True

    data = []
    for rule in rules:
        matching = [idx for idx in ordered if matches(idx, rule)]
        mandatory = set(ids_by_index[matching[0]]) if matching else set()
        for idx in matching[1:]:
            mandatory.intersection_update(ids_by_index[idx])
            if not mandatory:
                break
        data.append({"matching": matching, "mandatory": mandatory})
    return ordered, ids_by_index, superstar_by_index, data


def _future_mandatory_demand(rule_data, remaining):
    demand = Counter()
    for index, slots in enumerate(remaining):
        if slots <= 0:
            continue
        for player in rule_data[index]["mandatory"]:
            demand[player] += int(slots)
    return demand


def _attempt_selection(
    scored,
    rules,
    rule_counts,
    ordered,
    ids_by_index,
    superstar_by_index,
    rule_data,
    cfg,
    mode,
):
    """Run one deterministic greedy attempt with future-exposure reservation."""
    target = int(cfg.tournament_lineups)
    max_player_count = max(1, math.ceil(target * float(cfg.max_player_exposure)))
    max_superstar_count = max(1, math.ceil(target * float(cfg.max_superstar_exposure)))
    max_shared = int(cfg.max_shared_players)

    player_counts = Counter()
    superstar_counts = Counter()
    selected = []
    selected_sets = []
    selected_indices = set()
    assigned_rules = []
    unfilled = Counter()
    remaining = list(rule_counts)
    legacy_schedule = [rules.index(rule) for rule in rule_schedule(rules, target)]
    legacy_cursor = 0

    while sum(remaining) > 0:
        active = [i for i, slots in enumerate(remaining) if slots > 0]
        if mode == "round_robin":
            while legacy_cursor < len(legacy_schedule) and remaining[legacy_schedule[legacy_cursor]] <= 0:
                legacy_cursor += 1
            rule_index = legacy_schedule[legacy_cursor] if legacy_cursor < len(legacy_schedule) else active[0]
            legacy_cursor += 1
        elif mode == "mandatory":
            rule_index = min(
                active,
                key=lambda i: (-len(rule_data[i]["mandatory"]), len(rule_data[i]["matching"]), i),
            )
        else:  # scarcity first: few candidate options per remaining slot
            rule_index = min(
                active,
                key=lambda i: (
                    len(rule_data[i]["matching"]) / max(remaining[i], 1),
                    -len(rule_data[i]["mandatory"]),
                    i,
                ),
            )

        rule = rules[rule_index]
        remaining[rule_index] -= 1
        future_demand = _future_mandatory_demand(rule_data, remaining)
        picked = None
        picked_set = None

        for idx in rule_data[rule_index]["matching"]:
            if idx in selected_indices:
                continue
            ids = ids_by_index[idx]
            superstar = superstar_by_index[idx]
            if any(player_counts[player] >= max_player_count for player in ids):
                continue
            if superstar_counts[superstar] >= max_superstar_count:
                continue
            candidate_set = set(ids)
            if any(len(candidate_set & prior) > max_shared for prior in selected_sets):
                continue

            # Do not spend exposure that a future archetype is forced to use.
            reserve_ok = True
            for player, needed in future_demand.items():
                after_pick = player_counts[player] + (1 if player in candidate_set else 0)
                if after_pick + needed > max_player_count:
                    reserve_ok = False
                    break
            if not reserve_ok:
                continue

            picked = idx
            picked_set = candidate_set
            break

        if picked is None:
            unfilled[str(rule["name"])] += 1
            continue

        ids = ids_by_index[picked]
        superstar = superstar_by_index[picked]
        selected.append(picked)
        selected_indices.add(picked)
        selected_sets.append(picked_set)
        player_counts.update(ids)
        superstar_counts.update([superstar])
        assigned_rules.append(str(rule["name"]))

    quality = float(scored.loc[selected, "Tournament_Score"].sum()) if selected else float("-inf")
    return {
        "selected": selected,
        "assigned_rules": assigned_rules,
        "unfilled": unfilled,
        "quality": quality,
    }


def _select_unrestricted(scored: pd.DataFrame, cfg, target: int):
    """Original exposure/overlap selector, used when no archetypes apply."""
    max_player_count = max(1, math.ceil(target * float(cfg.max_player_exposure)))
    max_superstar_count = max(1, math.ceil(target * float(cfg.max_superstar_exposure)))
    player_counts = Counter()
    superstar_counts = Counter()
    selected = []
    selected_sets = []

    for idx, candidate in scored.sort_values("Tournament_Score", ascending=False).iterrows():
        ids = tuple(int(value) for value in candidate["Player_Ids"])
        superstar = int(candidate["Superstar_Id"])
        if any(player_counts[player] >= max_player_count for player in ids):
            continue
        if superstar_counts[superstar] >= max_superstar_count:
            continue
        candidate_set = set(ids)
        if any(len(candidate_set & prior) > int(cfg.max_shared_players) for prior in selected_sets):
            continue
        selected.append(idx)
        selected_sets.append(candidate_set)
        player_counts.update(ids)
        superstar_counts.update([superstar])
        if len(selected) == target:
            break
    return selected


def select_portfolio(scored: pd.DataFrame, players: pd.DataFrame, cfg, rules=None):
    """Select a tournament portfolio with scarcity-aware construction archetypes.

    Single-entry intentionally bypasses archetype quotas: there are no multiple
    portfolio entries to diversify, so the strongest legal lineup should win.
    For multi-entry, three deterministic schedules are attempted.  All reserve
    exposure that future slots are forced to consume, and the best completed
    portfolio is retained; if none completes, the largest partial portfolio is
    returned with explicit unfilled-slot metadata.
    """
    rules = serializable_rules(rules)
    validate_rules(rules, int(cfg.lineup_size))
    target = int(cfg.tournament_lineups)

    if target <= 0:
        portfolio = scored.iloc[0:0].copy()
        portfolio.attrs["construction_requested"] = target
        portfolio.attrs["construction_selected"] = 0
        portfolio.attrs["construction_unfilled"] = {}
        return portfolio

    # A one-entry contest is lineup selection, not portfolio diversification.
    if target == 1:
        selected = _select_unrestricted(scored, cfg, 1)
        portfolio = scored.loc[selected].copy().reset_index(drop=True)
        portfolio.attrs["construction_requested"] = 1
        portfolio.attrs["construction_selected"] = len(portfolio)
        portfolio.attrs["construction_unfilled"] = {}
        return portfolio

    if not rules:
        selected = _select_unrestricted(scored, cfg, target)
        portfolio = scored.loc[selected].copy().reset_index(drop=True)
        portfolio.attrs["construction_requested"] = target
        portfolio.attrs["construction_selected"] = len(portfolio)
        portfolio.attrs["construction_unfilled"] = {}
        return portfolio

    rule_counts = apportioned_rule_counts(rules, target)
    ordered, ids_by_index, superstar_by_index, rule_data = _candidate_metadata(scored, players, rules)
    attempts = [
        _attempt_selection(
            scored, rules, rule_counts, ordered, ids_by_index, superstar_by_index,
            rule_data, cfg, mode,
        )
        for mode in ("scarcity", "mandatory", "round_robin")
    ]
    best = max(attempts, key=lambda result: (len(result["selected"]), result["quality"]))

    portfolio = scored.loc[best["selected"]].copy().reset_index(drop=True)
    if len(portfolio):
        portfolio["Construction_Rule"] = best["assigned_rules"]
    else:
        portfolio["Construction_Rule"] = pd.Series(dtype=str)
    portfolio.attrs["construction_requested"] = target
    portfolio.attrs["construction_selected"] = len(portfolio)
    portfolio.attrs["construction_unfilled"] = dict(best["unfilled"])

    if len(portfolio) < target:
        details = ", ".join(
            f"{name}: {count}" for name, count in best["unfilled"].items()
        ) or "unknown"
        warnings.warn(
            f"Built {len(portfolio)} of {target} tournament lineups under the construction, "
            f"exposure, and overlap rules. Unfilled archetype slots: {details}."
        )
    return portfolio
