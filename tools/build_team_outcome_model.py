#!/usr/bin/env python3
"""Walk-forward run of the preregistered team-outcome candidate.

Phase P3 of docs/team-outcome-model-plan.md, governed by
docs/team-outcome-prereg.md. Every choice here is fixed by that document; this
tool executes it and records what happened. It evaluates G1-G4 and G8, reports
G6 without a threshold, and leaves G5 for the single sealed read at P4.

The sealed seasons are never loaded into training or scoring.
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
from tools.build_team_outcome_corpus import pull, pull_efficiency, seasons_arg, CACHE


# Preregistration section 4. `home_field` is the intercept, so it is not a column.
SPEC_A = ("elo_diff", "rest_diff", "short_week_diff", "bye_diff",
          "neutral_site", "dome", "div_game", "crowd_absent")
SPEC_B = SPEC_A + ("epa_off_diff", "epa_def_diff", "success_rate_diff")
RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


def _normal_cdf(x):
    from math import erf, sqrt
    return np.array([0.5 * (1.0 + erf(float(v) / sqrt(2.0))) for v in np.ravel(x)])


def select_ridge(train: pd.DataFrame, columns, target: str, kind: str):
    """Leave-last-season-out inside the training window, as preregistered.

    One held-out season is a noisy selector. It is also what the frozen
    document says, and following a preregistered rule that turns out noisy is
    the point of having frozen it.
    """
    seasons = sorted(train.season.unique())
    if len(seasons) < 2:
        return 1.0, []
    inner_train = train.loc[train.season.lt(seasons[-1])]
    inner_test = train.loc[train.season.eq(seasons[-1])]
    scores = []
    for penalty in RIDGE_GRID:
        x = inner_train[list(columns)].to_numpy(float)
        if kind == "win":
            model = fit.fit_logistic(x, inner_train.home_win.to_numpy(float), ridge=penalty)
            p = np.clip(fit.predict(model, inner_test[list(columns)].to_numpy(float)), 1e-9, 1 - 1e-9)
            y = inner_test.home_win.to_numpy(float)
            score = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        else:
            model = fit.fit_linear(x, inner_train.margin.to_numpy(float), ridge=penalty)
            error = fit.predict(model, inner_test[list(columns)].to_numpy(float)) - inner_test.margin.to_numpy(float)
            score = float(np.sqrt(np.mean(error ** 2)))
        scores.append({"ridge": penalty, "score": round(score, 5)})
    best = min(scores, key=lambda row: row["score"])
    return best["ridge"], scores


def walk_forward(design: pd.DataFrame, seasons, columns, label: str,
                 allow_sealed_training: bool = False) -> pd.DataFrame:
    """Fit on every earlier season, score the next.

    `allow_sealed_training` stays False everywhere except the P4 sealed read,
    where the expanding window has to continue through the sealed block: the
    2025 fold is fitted on everything before it, 2024 included. That is the
    same procedure, not a new one, and it happens inside the single read.
    """
    blocks, folds = [], []
    for season in seasons:
        train = design.loc[design.season.lt(season)]
        if not allow_sealed_training:
            train = train.loc[~train.season.isin(to.SEALED_SEASONS)]
        test = design.loc[design.season.eq(season)].copy()
        if train.empty or test.empty:
            continue
        x_train, x_test = train[list(columns)].to_numpy(float), test[list(columns)].to_numpy(float)

        margin_ridge, margin_scores = select_ridge(train, columns, "margin", "margin")
        win_ridge, win_scores = select_ridge(train, columns, "home_win", "win")
        margin_model = fit.fit_linear(x_train, train.margin.to_numpy(float), ridge=margin_ridge)
        win_model = fit.fit_logistic(x_train, train.home_win.to_numpy(float), ridge=win_ridge)

        # The link's residual scale is part of the model, fitted in-fold.
        residual_sd = float(np.std(fit.predict(margin_model, x_train) - train.margin.to_numpy(float), ddof=1))
        predicted_margin = fit.predict(margin_model, x_test)
        test[f"{label}"] = np.clip(_normal_cdf(predicted_margin / residual_sd), 1e-9, 1 - 1e-9)
        test[f"margin_{label}"] = predicted_margin
        test[f"{label}_direct"] = fit.predict(win_model, x_test)

        # The G3/G4 reference, refit here on the same rows so the comparison is paired.
        elo_model = fit.fit_logistic(train[["elo_diff"]].to_numpy(float), train.home_win.to_numpy(float))
        test["elo_reference"] = fit.predict(elo_model, test[["elo_diff"]].to_numpy(float))

        folds.append({
            "season": int(season), "training_games": int(len(train)),
            "margin_ridge": margin_ridge, "win_ridge": win_ridge,
            "residual_sd": round(residual_sd, 3),
            "home_field_logit": round(float(win_model["intercept"]), 4),
            "coefficients": {name: round(float(c / s), 6) for name, c, s in zip(
                columns, win_model["coefficients"], win_model["scale"])},
            "margin_grid": margin_scores, "win_grid": win_scores,
        })
        blocks.append(test)
    frame = pd.concat(blocks, ignore_index=True)
    frame.attrs["folds"] = folds
    return frame


def feature_diagnostics(design: pd.DataFrame, columns) -> dict:
    """Report what each feature actually varies over the evaluated span.

    A feature can be well defined, pass every leakage check, and still be
    constant — carrying no information at all while occupying a slot under the
    twelve-term cap. Nothing else in the pipeline notices that, so it is
    measured here and recorded rather than left to be spotted by eye.
    """
    rows = {}
    for name in columns:
        values = pd.to_numeric(design[name], errors="coerce")
        rows[name] = {
            "non_zero_rate": round(float((values != 0).mean()), 4),
            "sd": round(float(values.std()), 6),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "degenerate": bool(values.std() == 0),
        }
    return {"features": rows,
            "degenerate": sorted(n for n, r in rows.items() if r["degenerate"])}


def gates(frame: pd.DataFrame, label: str, audit_clean: bool, canary: bool,
          draws: int, seed: int) -> dict:
    seasons = sorted(frame.season.unique())
    scored = ev.metrics(frame, label, margin_hat=f"margin_{label}")
    slope = ev.bootstrap(frame, lambda block: ev.calibration_slope(block, label),
                         draws=draws, seed=seed)
    skill = ev.compare(frame, label, "elo_reference", draws=draws, seed=seed)
    per_season = []
    for season in seasons:
        block = frame.loc[frame.season.eq(season)]
        per_season.append({"season": int(season), "n": int(len(block)),
                           "brier": round(ev.brier(block, label), 4),
                           "brier_elo_reference": round(ev.brier(block, "elo_reference"), 4)})
    beaten = sum(1 for row in per_season if row["brier"] < row["brier_elo_reference"])

    return {
        "G1_coverage": {"passed": bool(len(frame) >= 3000 and len(seasons) >= 10),
                        "games": int(len(frame)), "seasons": len(seasons),
                        "requirement": ">= 3000 games across >= 10 seasons"},
        "G2_calibration": {"passed": bool(scored["ece"] <= 0.025 and slope["low"] <= 1.0 <= slope["high"]),
                           "ece": scored["ece"], "slope": scored["slope"],
                           "slope_interval": [slope["low"], slope["high"]],
                           "requirement": "ECE <= 0.025 and the slope interval contains 1"},
        "G3_skill": {"passed": bool(skill["high"] < 0), "brier_gap": skill["point"],
                     "interval": [skill["low"], skill["high"]],
                     "skill_score": skill["skill_vs_reference"],
                     "mcnemar": ev.mcnemar_exact(frame, label, "elo_reference"),
                     "requirement": "Brier below fitted Elo with the bootstrap upper bound below 0"},
        "G4_stability": {"passed": bool(beaten >= 13), "seasons_beaten": beaten,
                         "of": len(seasons), "requirement": "lower Brier than fitted Elo in >= 13 of 17"},
        "G5_seal": {"passed": None, "note": "Not evaluated. The sealed read is P4."},
        "G8_leakage": {"passed": bool(audit_clean and canary), "audit_clean": bool(audit_clean),
                       "canary_detected": bool(canary),
                       "requirement": "audit clean, canary detected, every feature timestamped"},
        "per_season": per_season,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=seasons_arg, default=seasons_arg("1999-2025"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--audit-every", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("model/team_outcome.json"))
    args = parser.parse_args()

    frozen = to.verify_prereg()
    corpus = to.build_corpus(pull(args.seasons, args.cache, refresh=False))
    efficiency = pull_efficiency(args.seasons, args.cache, refresh=False)
    corpus = to.attach_efficiency(corpus, efficiency)

    # Preregistration section 3: the specification is chosen by an audit result,
    # never by a score. Run it before anything is fitted.
    checkpoints = corpus.weeks(settled_only=True)[:: max(1, args.audit_every)]
    features = {name: to.FEATURES[name] for name in SPEC_B}
    report = to.audit(corpus, checkpoints=checkpoints, features=features, seed=args.seed)
    canary = to.canary_detected(corpus, checkpoints=checkpoints[:40], features=features, seed=args.seed)
    columns = SPEC_B if report["clean"] else SPEC_A
    specification = "B" if report["clean"] else "A"

    seasons = to.evaluation_seasons(corpus)
    design = to.build_design(corpus)
    frame = walk_forward(design, seasons, columns, "candidate")
    measured = gates(frame, "candidate", report["clean"], canary, args.draws, args.seed)
    diagnostics = feature_diagnostics(frame, columns)

    challengers = {}
    quadratic = design.assign(elo_diff_squared=design.elo_diff ** 2)
    for name, spec in (("nonlinearity", (*columns, "elo_diff_squared")),
                       ("redundancy", (*columns, "prior_margin_diff")),
                       ("specification_a", SPEC_A)):
        source = quadratic if name == "nonlinearity" else design
        run = walk_forward(source, seasons, spec, "challenger")
        # Join on game_id rather than trusting two runs to emit rows in the same
        # order; a paired comparison that silently mispairs is worse than none.
        paired = run[["game_id", "season", "home_win", "challenger"]].merge(
            frame[["game_id", "candidate"]], on="game_id", how="inner")
        if len(paired) != len(frame):
            raise ValueError(f"Challenger {name} covered {len(paired)} of {len(frame)} games")
        challengers[name] = {
            "features": list(spec),
            "brier": round(ev.brier(run, "challenger"), 4),
            "accuracy": round(ev.accuracy(run, "challenger"), 4),
            "vs_candidate": ev.compare(paired, "challenger", "candidate",
                                       draws=args.draws, seed=args.seed),
        }

    market = ev.market_probabilities(corpus.market)
    priced = frame.merge(market, on="game_id", how="inner")
    g6 = {"n": int(len(priced)),
          "candidate_brier": round(ev.brier(priced, "candidate"), 4),
          "market_shin_brier": round(ev.brier(priced, "market_shin"), 4),
          "market_proportional_brier": round(ev.brier(priced, "market_proportional"), 4),
          "gap_to_market_shin": ev.compare(priced, "candidate", "market_shin",
                                           draws=args.draws, seed=args.seed),
          "note": "Reported without a threshold. G6 gates nothing."}

    scored = ev.metrics(frame, "candidate", margin_hat="margin_candidate")
    artifact = {
        "schema": 1,
        "phase": "P3",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_sha256": frozen["sha256"],
        "prereg_version": frozen["version"],
        "specification": specification,
        "specification_selected_by": "as-of audit result, not performance",
        "features": list(columns),
        "corpus": {"games": int(len(corpus.games)), "team_games": int(len(corpus.efficiency)),
                   "games_without_play_by_play": int(len(
                       set(corpus.outcomes.game_id) - set(corpus.efficiency.game_id)))},
        "walk_forward": {"seasons": [int(s) for s in seasons], "games": int(len(frame)),
                         "folds": frame.attrs["folds"]},
        "feature_diagnostics": diagnostics,
        "metrics": scored,
        "direct_win_model": ev.metrics(frame, "candidate_direct"),
        "gates": measured,
        "challengers": challengers,
        "G6_market_context": g6,
        "as_of_audit": {"checkpoints": report["checkpoints"], "clean": bool(report["clean"]),
                        "findings": report["findings"][:20], "canary_detected": bool(canary)},
        "approved": False,
        "approval_note": "approved_model requires G5, which is the sealed read at P4. "
                         "No model is promoted at P3.",
        "limitations": [
            "Three 1999-2000 games carry no play-by-play; their team-games are absent "
            "from every rolling window.",
            "Ridge selection uses one held-out season, as preregistered. It is noisy.",
            "short_week_diff is identically zero over the evaluated span: an NFL short week "
            "puts both teams on short rest, so the directional form cannot fire. It is left "
            "in place rather than swapped out, because replacing a feature after seeing a "
            "run is the freedom the preregistration exists to remove.",
            "The sealed seasons are neither fitted on nor scored here.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")

    print(f"specification {specification} ({len(columns)} terms + intercept), "
          f"{len(frame)} games over {len(seasons)} seasons")
    print(f"  candidate  brier {scored['brier']:.4f}  acc {scored['accuracy']:.4f}  "
          f"auc {scored['auc']:.4f}  ece {scored['ece']:.4f}  slope {scored['slope']:.3f}  "
          f"margin rmse {scored['margin_rmse']:.3f}")
    print(f"  reference  brier {ev.brier(frame, 'elo_reference'):.4f} (fitted Elo)")
    if diagnostics["degenerate"]:
        print(f"  DEGENERATE features (constant over the span): {diagnostics['degenerate']}")
    for name, block in measured.items():
        if name.startswith("G"):
            print(f"  {name:16s} {block['passed']}")
    for name, block in challengers.items():
        print(f"  challenger {name:16s} brier {block['brier']:.4f}")
    print(f"  G6 market gap {g6['gap_to_market_shin']['point']:+.4f} "
          f"{g6['gap_to_market_shin']['low']:+.4f}..{g6['gap_to_market_shin']['high']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
