#!/usr/bin/env python3
"""Test the fixed-core hypothesis at scale.

docs/team-outcome-model-foundation.md tested it on 67 games and concluded it
was a directional feature, not a calibrated one. This runs the same hypothesis
over every game the corpus holds, with the player expectations rebuilt from
nflverse rather than Yahoo salary — see pipeline/team_outcome_core.py for what
that swap preserves and what it gives up.

Reports the foundation document's own tables so the two are directly
comparable, then the fitted walk-forward evaluation the original outline asked
for and never got: does the edge beat picking the home team, does it beat Elo,
and does it add anything to Elo.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import team_outcome as to
from pipeline import team_outcome_core as core
from pipeline import team_outcome_eval as ev
from pipeline import team_outcome_fit as fit
from tools.build_team_outcome_corpus import pull, pull_fantasy, seasons_arg, CACHE
from tools.backtest_team_outcome_roi import summarize as roi_summary, breakdown as roi_breakdown


def descriptive(frame: pd.DataFrame, neutral_share: float = 0.2) -> dict:
    """The foundation document's tables, rebuilt on this corpus.

    Descriptive, and labelled so: the neutral band and the quintile cuts are
    read off the evaluated games, exactly as the original did. They describe
    the sample; they are not an out-of-sample claim.
    """
    edge = frame.fixed_core_diff
    home_win = frame.home_win
    band = float(edge.abs().quantile(neutral_share))
    tier = core.tiers(edge, band)

    picked = np.where(edge > 0, 1.0, np.where(edge < 0, 0.0, 0.5))
    correct = (picked == home_win) | ((picked != home_win) & ((picked == 0.5) | (home_win == 0.5))) * 0.5
    sign_accuracy = float(np.where(picked == home_win, 1.0,
                          np.where((picked == 0.5) | (home_win == 0.5), 0.5, 0.0)).mean())

    tiers = [{"tier": name, "games": int(len(block)),
              "home_wins": int((block.home_win == 1.0).sum()),
              "home_win_rate": round(float((block.home_win == 1.0).mean()), 4),
              "average_edge": round(float(block.fixed_core_diff.mean()), 2)}
             for name, block in frame.groupby(tier, observed=True)]

    quintile = pd.qcut(edge, 5, labels=False, duplicates="drop")
    quintiles = [{"quintile": int(q) + 1, "games": int(len(block)),
                  "average_edge": round(float(block.fixed_core_diff.mean()), 2),
                  "home_win_rate": round(float((block.home_win == 1.0).mean()), 4),
                  "average_margin": round(float(block.margin.mean()), 2)}
                 for q, block in frame.groupby(quintile, observed=True)]

    return {
        "neutral_band": round(band, 3),
        "sign_accuracy": round(sign_accuracy, 4),
        "home_baseline": round(float((home_win == 1.0).mean()), 4),
        "edge_correlation_with_margin": round(float(np.corrcoef(edge, frame.margin)[0, 1]), 4),
        "tiers": tiers,
        "quintiles": quintiles,
    }


def walk_forward(design: pd.DataFrame, seasons, specs: dict) -> pd.DataFrame:
    """Fit each named feature set on earlier seasons, score the next."""
    blocks = []
    for season in seasons:
        train = design.loc[design.season.lt(season) & ~design.season.isin(to.SEALED_SEASONS)]
        test = design.loc[design.season.eq(season)].copy()
        if train.empty or test.empty:
            continue
        test["home_only"] = float(train.home_win.mean())
        for label, columns in specs.items():
            model = fit.fit_logistic(train[list(columns)].to_numpy(float),
                                     train.home_win.to_numpy(float))
            test[label] = fit.predict(model, test[list(columns)].to_numpy(float))
        blocks.append(test)
    return pd.concat(blocks, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=seasons_arg, default=seasons_arg("1999-2025"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit-every", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("model/fixed_core.json"))
    args = parser.parse_args()

    schedules = pull(args.seasons, args.cache, refresh=False)
    corpus = core.attach_fantasy(to.build_corpus(schedules),
                                 pull_fantasy(args.seasons, schedules, args.cache, refresh=False))

    features = {name: to.FEATURES[name] for name in
                ("fixed_core_diff", "fixed_core_complete", "elo_diff")}
    checkpoints = corpus.weeks(settled_only=True)[:: max(1, args.audit_every)]
    audit = to.audit(corpus, checkpoints=checkpoints, features=features, seed=args.seed)
    canary = to.canary_detected(corpus, checkpoints=checkpoints[:40], features=features,
                                seed=args.seed)

    seasons = to.evaluation_seasons(corpus)
    design = to.build_design(corpus, features=features)
    evaluated = design.loc[design.season.isin(seasons)]

    specs = {"core_only": ("fixed_core_diff",),
             "elo_only": ("elo_diff",),
             "elo_plus_core": ("elo_diff", "fixed_core_diff")}
    scored = walk_forward(design, seasons, specs)

    complete = scored.loc[scored.fixed_core_complete.eq(1.0)]
    results = {name: ev.metrics(scored, name) for name in ("home_only", *specs)}
    comparisons = {
        "core_only_vs_home_only": ev.compare(scored, "core_only", "home_only",
                                             draws=args.draws, seed=args.seed),
        "core_only_vs_elo_only": ev.compare(scored, "core_only", "elo_only",
                                            draws=args.draws, seed=args.seed),
        "elo_plus_core_vs_elo_only": ev.compare(scored, "elo_plus_core", "elo_only",
                                                draws=args.draws, seed=args.seed),
    }

    market = ev.market_probabilities(corpus.market)
    priced = scored.merge(market, on="game_id", how="inner")

    # Would backing the core have made money? Same staking rules as
    # tools/backtest_team_outcome_roi.py: one flat unit, priced against the
    # vig-included quote, a tie is a push.
    roi = {}
    for name in ("core_only", "elo_only", "elo_plus_core"):
        bets = ev.moneyline_bets(scored, corpus.market, name)
        roi[name] = {
            **roi_summary(bets, args.draws, args.seed),
            "share_of_games_with_a_claimed_edge": round(float(len(bets) / len(priced)), 4)
            if len(priced) else None,
            "breakdown": roi_breakdown(bets),
        }
    artifact = {
        "schema": 1,
        "phase": "fixed-core hypothesis at scale",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "A fixed 1 QB / 3 RB / 3 WR / 2 TE / 1 DEF core, summed and "
                      "differenced, carries team-outcome signal.",
        "instrument": "nflverse lagged eight-appearance Yahoo half-PPR production, "
                      "not Yahoo salary; see pipeline/team_outcome_core.py",
        "corpus": {"games": int(len(evaluated)), "seasons": [int(seasons[0]), int(seasons[-1])],
                   "unit_weeks": int(len(corpus.fantasy)),
                   "complete_cores": int(evaluated.fixed_core_complete.sum()),
                   "original_study_games": 67},
        "as_of_audit": {"checkpoints": audit["checkpoints"], "clean": bool(audit["clean"]),
                        "findings": audit["findings"][:20], "canary_detected": bool(canary)},
        "descriptive": descriptive(evaluated),
        "descriptive_complete_cores_only": descriptive(
            evaluated.loc[evaluated.fixed_core_complete.eq(1.0)]),
        "walk_forward": {name: block for name, block in results.items()},
        "walk_forward_complete_cores_only": {
            name: ev.metrics(complete, name) for name in ("home_only", *specs)},
        "comparisons": comparisons,
        "market_context": {
            "n": int(len(priced)),
            "core_only_brier": round(ev.brier(priced, "core_only"), 4),
            "elo_only_brier": round(ev.brier(priced, "elo_only"), 4),
            "market_shin_brier": round(ev.brier(priced, "market_shin"), 4),
        },
        "flat_unit_roi": roi,
        "limitations": [
            "The instrument is a lagged rolling average, not Yahoo salary. This tests the "
            "structure of the hypothesis, not whether Yahoo's pricing carries information.",
            "Roster membership is the team a unit last appeared for. A player who is inactive "
            "this week still occupies a slot, because the corpus carries no pregame status feed.",
            "2024-2025 are no longer an untouched holdout: the seal was spent at P4 on a "
            "different hypothesis. A clean final test of this one needs 2026.",
            "Descriptive tables read their cuts off the evaluated games, as the original did.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")

    d = artifact["descriptive"]
    print(f"fixed core over {len(evaluated)} games ({seasons[0]}-{seasons[-1]}), "
          f"against 67 in the original study")
    print(f"  as-of audit {'clean' if audit['clean'] else 'FINDINGS'} over "
          f"{audit['checkpoints']} weeks, canary {canary}")
    print(f"  raw sign accuracy {d['sign_accuracy']:.4f} vs home baseline {d['home_baseline']:.4f}"
          f"   edge/margin correlation {d['edge_correlation_with_margin']:+.4f}")
    print("  tiers:")
    for row in d["tiers"]:
        print(f"    {row['tier']:9s} games {row['games']:5d}  home win rate {row['home_win_rate']:.4f}")
    print("  walk-forward:")
    for name, block in results.items():
        print(f"    {name:16s} brier {block['brier']:.4f}  acc {block['accuracy']:.4f}  "
              f"auc {block['auc']:.4f}")
    for name, block in comparisons.items():
        print(f"    {name:28s} {block['point']:+.4f} [{block['low']:+.4f}, {block['high']:+.4f}]")
    print("  flat-unit ROI against closing moneylines:")
    for name, block in roi.items():
        if not block["bets"]:
            print(f"    {name:16s} no bets"); continue
        low, high = block["roi_interval"]
        print(f"    {name:16s} bets {block['bets']:5d}  ROI {block['roi']:+7.2%}  "
              f"[{low:+.2%}, {high:+.2%}]  units {block['units']:+8.1f}  "
              f"edge claimed on {block['share_of_games_with_a_claimed_edge']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
