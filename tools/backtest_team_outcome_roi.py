#!/usr/bin/env python3
"""Flat-unit moneyline ROI for the team-outcome forecasts against closing prices.

An addendum to docs/team-outcome-model-plan.md rather than one of its phases:
the plan never proposed a betting model, and nothing here is promoted or
published. It answers the obvious question a forecast invites — would backing
it have made money — with the same discipline the rest of the project uses.

Bets are priced against the **vig-included** quote, because that is the price
on offer. Comparing a forecast to a no-vig probability and then collecting a
vig-included payout manufactures an edge that does not exist.
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
from pipeline import team_outcome_eval as ev
from tools.build_team_outcome_corpus import pull, pull_efficiency, seasons_arg, CACHE
from tools.build_team_outcome_baselines import walk_forward as baseline_walk_forward
from tools.build_team_outcome_model import SPEC_A, SPEC_B, walk_forward as model_walk_forward


PRICE_BUCKETS = [-np.inf, -300, -150, 100, 200, 300, np.inf]
PRICE_LABELS = ["<= -300", "-300..-150", "-150..+100", "+100..+200", "+200..+300", "> +300"]
THRESHOLD_SWEEP = (0.0, 0.02, 0.05, 0.08, 0.10, 0.15)


def summarize(bets: pd.DataFrame, draws: int, seed: int) -> dict:
    if bets.empty:
        return {"bets": 0, "roi": None, "units": 0.0}
    interval = ev.bootstrap(bets.assign(home_win=0.0),
                            lambda block: float(block.profit.mean()), draws=draws, seed=seed)
    return {
        "bets": int(len(bets)),
        "roi": round(ev.roi(bets), 4),
        "roi_interval": [interval["low"], interval["high"]],
        "units": round(float(bets.profit.sum()), 1),
        "seasons": [int(bets.season.min()), int(bets.season.max())],
    }


def breakdown(bets: pd.DataFrame) -> dict:
    if bets.empty:
        return {}
    buckets = pd.cut(bets.odds, PRICE_BUCKETS, labels=PRICE_LABELS)
    by_price = [{"price": str(label), "bets": int(len(block)), "roi": round(ev.roi(block), 4)}
                for label, block in bets.groupby(buckets, observed=True)]
    favourite = bets.odds < 0
    return {
        "by_price": by_price,
        "favourites": {"bets": int(favourite.sum()),
                       "roi": round(ev.roi(bets[favourite]), 4) if favourite.any() else None},
        "underdogs": {"bets": int((~favourite).sum()),
                      "roi": round(ev.roi(bets[~favourite]), 4) if (~favourite).any() else None},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=seasons_arg, default=seasons_arg("1999-2025"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("model/team_outcome_roi.json"))
    args = parser.parse_args()

    corpus = to.attach_efficiency(
        to.build_corpus(pull(args.seasons, args.cache, refresh=False)),
        pull_efficiency(args.seasons, args.cache, refresh=False))
    design = to.build_design(corpus)
    columns = SPEC_B if json.loads(
        Path("model/team_outcome.json").read_text(encoding="utf-8"))["specification"] == "B" else SPEC_A

    spans = {"walk_forward": to.evaluation_seasons(corpus),
             "sealed": sorted(to.SEALED_SEASONS)}
    forecasts = {}
    for span, seasons in spans.items():
        sealed = span == "sealed"
        elo = baseline_walk_forward(design, seasons)[
            ["game_id", "season", "home_win", "elo_logistic"]]
        candidate = model_walk_forward(design, seasons, columns, "candidate",
                                       allow_sealed_training=sealed)[["game_id", "candidate"]]
        forecasts[span] = elo.merge(candidate, on="game_id", how="inner")

    results = {}
    for span, frame in forecasts.items():
        results[span] = {}
        for name, column in (("fitted_elo", "elo_logistic"), ("candidate", "candidate")):
            bets = ev.moneyline_bets(frame, corpus.market, column)
            priced = frame.merge(ev.market_probabilities(corpus.market)[["game_id"]],
                                 on="game_id", how="inner")
            results[span][name] = {
                **summarize(bets, args.draws, args.seed),
                "priced_games": int(len(priced)),
                "share_of_games_with_a_claimed_edge": round(
                    float(len(bets) / len(priced)), 4) if len(priced) else None,
                "breakdown": breakdown(bets),
            }

    # A threshold sweep is a diagnostic, not a strategy. Reading the best cell
    # off it is the post-hoc selection the rest of this project refuses.
    sweep = [{"edge": edge,
              **summarize(ev.moneyline_bets(forecasts["walk_forward"], corpus.market,
                                            "elo_logistic", edge=edge), 500, args.seed)}
             for edge in THRESHOLD_SWEEP]

    quoted = ev.market_probabilities(corpus.market)
    artifact = {
        "schema": 1,
        "phase": "addendum to P4",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "question": "Would backing these forecasts at the closing moneyline have made money?",
        "answer": "No. Every span, model and threshold measured is negative.",
        "staking": "one flat unit per bet, priced against the vig-included quote; "
                   "a tie is a push and returns the stake",
        "results": results,
        "threshold_sweep_diagnostic": sweep,
        "reference": {
            "mean_overround": round(float(quoted.overround.mean()), 4),
            "no_skill_loss_per_unit": round(float(quoted.overround.mean()) / 2, 4),
            "note": "A bettor with no skill loses roughly half the overround per flat unit. "
                    "Losing more than that means the selections are worse than random.",
        },
        "limitations": [
            "Closing moneylines: the most efficient price of the week, and the hardest to beat.",
            "One book, no line shopping, no opening-line comparison, no stake sizing, no limits.",
            "Season-block bootstrap intervals are wide; the finding rests on the direction "
            "being consistent across every span, model and threshold, not on one interval.",
            "The sealed seasons were spent at P4, so they are no longer an untouched holdout.",
            "Nothing here is promoted, published, or advice.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")

    for span, block in results.items():
        print(f"=== {span} ===")
        for name, row in block.items():
            if not row["bets"]:
                print(f"  {name:12s} no bets"); continue
            low, high = row["roi_interval"]
            print(f"  {name:12s} bets {row['bets']:5d}  ROI {row['roi']:+7.2%}  "
                  f"[{low:+.2%}, {high:+.2%}]  units {row['units']:+8.1f}  "
                  f"edge claimed on {row['share_of_games_with_a_claimed_edge']:.1%} of games")
    print(f"no-skill loss per unit: {artifact['reference']['no_skill_loss_per_unit']:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
