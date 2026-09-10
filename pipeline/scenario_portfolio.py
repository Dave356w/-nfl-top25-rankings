"""Greedy expected-best portfolio selection with independent evaluation draws.

Candidates are screened analytically upstream. Selection sees only the first
half of the scenarios; reported portfolio performance uses the other half.
Exposure and overlap constraints are enforced, but there is no global-optimality
claim for constrained greedy selection.
"""
from collections import Counter
import heapq

import numpy as np

from pipeline.portfolio_construction import exposure_limit


def select(scored, outcomes, cfg):
    target = int(cfg.tournament_lineups)
    if getattr(cfg, 'use_construction_quotas', False) and target > 1:
        raise ValueError('Disable roster-shape quotas for shared-scenario portfolio selection.')
    split = len(outcomes) // 2
    if target > 1 and split < 2:
        raise ValueError('At least four scenarios are required for portfolio selection and evaluation.')
    ids = [tuple(map(int, row)) for row in scored['Player_Ids']]
    stars = scored['Superstar_Id'].to_numpy(int)
    expected = scored['Expected_FP'].to_numpy(float)
    keys = [(tuple(sorted(row)), int(star)) for row, star in zip(ids, stars)]
    order = sorted(range(len(scored)), key=lambda i: (-expected[i], keys[i]))
    player_cap = exposure_limit(target, cfg.max_player_exposure)
    star_cap = exposure_limit(target, cfg.max_superstar_exposure)
    chosen, sets = [], []
    player_counts, star_counts = Counter(), Counter()
    best = np.zeros(split, dtype=float)

    def scores(i, window):
        return outcomes[window][:, ids[i]].sum(axis=1, dtype=np.float64) + 0.5 * outcomes[window, stars[i]]

    def feasible(i):
        return (all(player_counts[p] < player_cap for p in ids[i])
                and star_counts[stars[i]] < star_cap
                and all(len(set(ids[i]) & prior) <= cfg.max_shared_players for prior in sets))

    def take(i):
        chosen.append(i)
        sets.append(set(ids[i]))
        player_counts.update(ids[i])
        star_counts.update([stars[i]])

    if target > 0 and order and feasible(order[0]):
        take(order[0])  # Deterministic highest-expectation anchor, also the single-entry objective.
        if target > 1:
            best = scores(order[0], slice(0, split))
    heap = [(-float('inf'), keys[i], i, -1) for i in order if i not in chosen]
    heapq.heapify(heap)
    while heap and len(chosen) < target:
        neg_bound, key, i, epoch = heapq.heappop(heap)
        if not feasible(i):
            continue  # Exposure and overlap feasibility can only shrink.
        if epoch != len(chosen):
            gain = float(np.maximum(scores(i, slice(0, split)) - best, 0).mean())
            heapq.heappush(heap, (-gain, key, i, len(chosen)))
            continue
        take(i)
        best = np.maximum(best, scores(i, slice(0, split)))

    result = scored.iloc[chosen].copy().reset_index(drop=True)
    result.attrs.update(construction_requested=target, construction_selected=len(result),
                        construction_unfilled={})
    metrics = {'objective': 'expected_points' if target == 1 else 'expected_best',
               'selection_scenarios': 0 if target == 1 else split,
               'evaluation_scenarios': len(outcomes) - split}
    if chosen and len(outcomes) > split:
        held = np.maximum.reduce([scores(i, slice(split, None)) for i in chosen])
        baseline = scores(order[0], slice(split, None))
        metrics.update(evaluation_best_mean=float(held.mean()),
                       evaluation_best_p25=float(np.quantile(held, .25)),
                       evaluation_gain_over_single=float((held - baseline).mean()))
    result.attrs['scenario_evaluation'] = metrics
    return result

