#!/usr/bin/env python3
"""Walk-forward baselines for the team-outcome model, with interval estimates.

Phase P1 of docs/team-outcome-model-plan.md. Establishes the lines every later
candidate is measured against: home-only, textbook Elo, fitted Elo, and the
no-vig closing market under both de-vig methods. Nothing here is a candidate
model and nothing is promoted.

Every baseline is fitted on earlier seasons only and scored on the next one.
The sealed seasons are excluded, so this tool never reads them.
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
from pipeline import team_outcome_fit as fit
from tools.build_team_outcome_corpus import pull, seasons_arg, CACHE


PROBABILITIES = ("home_only", "elo_textbook", "elo_logistic")


def walk_forward(design: pd.DataFrame, seasons) -> pd.DataFrame:
    """One expanding-window pass: fit on everything earlier, score the season."""
    blocks, fits = [], []
    for season in seasons:
        train = design.loc[design.season.lt(season) & ~design.season.isin(to.SEALED_SEASONS)]
        test = design.loc[design.season.eq(season)].copy()
        if train.empty or test.empty:
            continue

        test["home_only"] = float(train.home_win.mean())
        test["margin_home_only"] = float(train.margin.mean())

        elo = test.elo_diff.to_numpy(float)
        test["elo_textbook"] = 1.0 / (1.0 + 10.0 ** (
            -(elo + to.ELO_HOME_ADVANTAGE) / to.ELO_SCALE))

        x_train = train[["elo_diff"]].to_numpy(float)
        win = fit.fit_logistic(x_train, train.home_win.to_numpy(float))
        margin = fit.fit_linear(x_train, train.margin.to_numpy(float))
        test["elo_logistic"] = fit.predict(win, test[["elo_diff"]].to_numpy(float))
        test["margin_elo_logistic"] = fit.predict(margin, test[["elo_diff"]].to_numpy(float))

        # Points per 100 Elo and the fold's own home-field term, reported so the
        # textbook constants are visible rather than assumed.
        fits.append({"season": int(season), "training_games": int(len(train)),
                     "home_rate_prior": round(float(train.home_win.mean()), 4),
                     "margin_prior": round(float(train.margin.mean()), 3),
                     "points_per_100_elo": round(float(
                         margin["coefficients"][0] / margin["scale"][0] * 100), 3),
                     "logit_per_100_elo": round(float(
                         win["coefficients"][0] / win["scale"][0] * 100), 4),
                     "fitted_home_logit": round(float(win["intercept"]), 4)})
        blocks.append(test)
    frame = pd.concat(blocks, ignore_index=True)
    frame.attrs["fits"] = fits
    return frame


def score(frame: pd.DataFrame, names, draws: int, seed: int) -> dict:
    out = {}
    for name in names:
        margin_hat = f"margin_{name}" if f"margin_{name}" in frame else None
        summary = ev.metrics(frame, name, margin_hat=margin_hat)
        summary["brier_interval"] = ev.bootstrap(
            frame, lambda block, n=name: ev.brier(block, n), draws=draws, seed=seed)
        summary["accuracy_interval"] = ev.bootstrap(
            frame, lambda block, n=name: ev.accuracy(block, n), draws=draws, seed=seed)
        out[name] = summary
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=seasons_arg, default=seasons_arg("1999-2025"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--draws", type=int, default=2000, help="bootstrap resamples")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("model/team_outcome_baselines.json"))
    args = parser.parse_args()

    corpus = to.build_corpus(pull(args.seasons, args.cache, refresh=False))
    seasons = to.evaluation_seasons(corpus)
    design = to.build_design(corpus)
    frame = walk_forward(design, seasons)
    fits = frame.attrs["fits"]

    baselines = score(frame, PROBABILITIES, args.draws, args.seed)
    comparisons = {
        name: {**ev.compare(frame, name, "home_only", draws=args.draws, seed=args.seed),
               **{"mcnemar": ev.mcnemar_exact(frame, name, "home_only")}}
        for name in ("elo_textbook", "elo_logistic")
    }

    # The market benchmark covers only the seasons that carry a closing line.
    market = ev.market_probabilities(corpus.market)
    priced = frame.merge(market, on="game_id", how="inner")
    market_names = ("market_proportional", "market_shin")
    market_block = {
        "n": int(len(priced)),
        "seasons": [int(priced.season.min()), int(priced.season.max())] if len(priced) else [],
        "mean_overround": round(float(priced.overround.mean()), 4) if len(priced) else None,
        "scored": score(priced, (*market_names, *PROBABILITIES), args.draws, args.seed),
        "elo_logistic_vs_market_shin": {
            **ev.compare(priced, "elo_logistic", "market_shin", draws=args.draws, seed=args.seed),
            **{"mcnemar": ev.mcnemar_exact(priced, "elo_logistic", "market_shin")}},
        "devig_sensitivity": {
            "brier_gap_proportional_minus_shin": round(float(
                ev.metrics(priced, "market_proportional")["brier"]
                - ev.metrics(priced, "market_shin")["brier"]), 5),
            "max_abs_probability_gap": round(float(
                (priced.market_proportional - priced.market_shin).abs().max()), 4),
            "ordering_stable": bool(
                (ev.metrics(priced, "elo_logistic")["brier"] > ev.metrics(priced, "market_shin")["brier"])
                == (ev.metrics(priced, "elo_logistic")["brier"] > ev.metrics(priced, "market_proportional")["brier"])),
        },
    }

    per_season = frame.groupby("season").apply(
        lambda block: pd.Series({
            "n": int(len(block)),
            "home_win_rate": round(float((block.home_win == 1.0).mean()), 4),
            **{f"brier_{name}": ev.metrics(block, name)["brier"] for name in PROBABILITIES},
        }), include_groups=False).reset_index()

    # Home-field drift, re-derived without the sealed seasons: the P0 artifact
    # quoted an era rate that included them.
    settled = to.unsealed(corpus.games.merge(corpus.outcomes, on="game_id"))
    eras = [(1999, 2007), (2008, 2015), (2016, 2019), (2020, 2020), (2021, 2023)]
    trend = [{"era": f"{lo}-{hi}",
              "n": int(len(block := settled[settled.season.between(lo, hi)])),
              "home_win_rate": round(float((block.home_win == 1.0).mean()), 4)}
             for lo, hi in eras]

    artifact = {
        "schema": 1,
        "phase": "P1",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "seal": {
            "sealed_seasons": list(to.SEALED_SEASONS),
            "evaluated_seasons": [int(s) for s in seasons],
            "minimum_training_seasons": to.MINIMUM_TRAINING_SEASONS,
            "note": "Sealed seasons are neither fitted on nor scored here.",
        },
        "walk_forward": {"games": int(len(frame)), "folds": fits},
        "baselines": baselines,
        "comparisons_vs_home_only": comparisons,
        "market": market_block,
        "home_field_trend_unsealed": trend,
        "per_season": per_season.to_dict("records"),
        "limitations": [
            "Baselines only. No candidate model is fitted and nothing is promoted.",
            "Elo uses a literature home-field constant; the fitted Elo baseline is reported "
            "beside it so that choice is visible.",
            "The market benchmark covers only games carrying a closing moneyline.",
            "Intervals are season-block bootstrap percentiles, not analytic.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")

    print(f"evaluated {len(seasons)} seasons, {len(frame)} games "
          f"(sealed {list(to.SEALED_SEASONS)} untouched)")
    for name, block in baselines.items():
        interval = block["brier_interval"]
        print(f"  {name:16s} brier {block['brier']:.4f} [{interval['low']:.4f}, {interval['high']:.4f}]  "
              f"acc {block['accuracy']:.4f}  auc {block['auc']:.4f}  ece {block['ece']:.4f}")
    for name in market_names:
        block = market_block["scored"][name]
        print(f"  {name:16s} brier {block['brier']:.4f}  acc {block['accuracy']:.4f}  "
              f"auc {block['auc']:.4f}  (n={block['n']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
