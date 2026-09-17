#!/usr/bin/env python3
"""Phase P4: the single sealed read.

Scores the preregistered candidate on 2024 and 2025 — seasons nothing in this
project has read — and decides the approval flag. The preregistration allows
this once. Afterwards the seal is spent: a changed specification needs a season
that has never been read, and this tool refuses to run again.

The acceptance interval for G5 is computed from the unsealed walk-forward
*before* any sealed data is scored, so the target is not chosen after seeing
the answer.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import team_outcome as to
from pipeline import team_outcome_eval as ev
from tools.build_team_outcome_corpus import pull, pull_efficiency, seasons_arg, CACHE
from tools.build_team_outcome_model import SPEC_A, SPEC_B, walk_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=seasons_arg, default=seasons_arg("1999-2025"))
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--walk-forward", type=Path, default=Path("model/team_outcome.json"))
    parser.add_argument("--prereg-record", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("model/team_outcome_sealed.json"))
    args = parser.parse_args()

    record_path = args.prereg_record or to.PREREG_RECORD
    frozen = to.verify_prereg(record=record_path)
    if frozen.get("sealed_read_recorded"):
        raise SystemExit(
            "Refusing to read the seal twice. It was already spent on "
            f"{frozen.get('sealed_read_utc', 'an earlier run')}. A second attempt needs a "
            "season that has never been read, and a new preregistration."
        )
    prior = json.loads(args.walk_forward.read_text(encoding="utf-8"))
    if prior["prereg_sha256"] != frozen["sha256"]:
        raise SystemExit("The walk-forward run used a different preregistration than the one frozen now.")

    corpus = to.attach_efficiency(
        to.build_corpus(pull(args.seasons, args.cache, refresh=False)),
        pull_efficiency(args.seasons, args.cache, refresh=False))
    columns = SPEC_B if prior["specification"] == "B" else SPEC_A
    design = to.build_design(corpus)

    # Step one, before the seal is touched: reproduce the walk-forward and take
    # its bootstrap interval. This is the acceptance region G5 is judged against.
    unsealed = walk_forward(design, to.evaluation_seasons(corpus), columns, "candidate")
    acceptance = ev.bootstrap(unsealed, lambda block: ev.brier(block, "candidate"),
                              draws=args.draws, seed=args.seed)
    if round(ev.brier(unsealed, "candidate"), 4) != prior["metrics"]["brier"]:
        raise SystemExit("The walk-forward did not reproduce; refusing to spend the seal on a "
                         f"changed pipeline ({ev.brier(unsealed, 'candidate'):.4f} against "
                         f"{prior['metrics']['brier']:.4f}).")

    # Step two: the read.
    sealed_seasons = sorted(to.SEALED_SEASONS)
    sealed = walk_forward(design, sealed_seasons, columns, "candidate",
                          allow_sealed_training=True)
    scored = ev.metrics(sealed, "candidate", margin_hat="margin_candidate")
    inside = bool(acceptance["low"] <= scored["brier"] <= acceptance["high"])

    gates = {name: dict(block) for name, block in prior["gates"].items() if name.startswith("G")}
    gates["G5_seal"] = {
        "passed": inside,
        "sealed_brier": scored["brier"],
        "acceptance_interval": [acceptance["low"], acceptance["high"]],
        "walk_forward_brier": acceptance["point"],
        "requirement": "sealed-season Brier inside the walk-forward bootstrap interval; one read",
    }
    approved_model = all(gates[name]["passed"] for name in
                         ("G1_coverage", "G2_calibration", "G3_skill", "G4_stability",
                          "G5_seal", "G8_leakage"))

    per_season = [{"season": int(season), "n": int(len(block)),
                   "brier": round(ev.brier(block, "candidate"), 4),
                   "brier_elo_reference": round(ev.brier(block, "elo_reference"), 4),
                   "accuracy": round(ev.accuracy(block, "candidate"), 4),
                   "home_win_rate": round(float((block.home_win == 1.0).mean()), 4)}
                  for season, block in sealed.groupby("season")]

    market = ev.market_probabilities(corpus.market)
    priced = sealed.merge(market, on="game_id", how="inner")
    artifact = {
        "schema": 1,
        "phase": "P4",
        "as_of_utc": datetime.now(timezone.utc).isoformat(),
        "prereg_sha256": frozen["sha256"],
        "specification": prior["specification"],
        "features": list(columns),
        "sealed_seasons": sealed_seasons,
        "sealed_games": int(len(sealed)),
        "acceptance_interval_computed_before_the_read": True,
        "metrics": scored,
        "elo_reference_brier": round(ev.brier(sealed, "elo_reference"), 4),
        "per_season": per_season,
        "vs_elo_reference": ev.compare(sealed, "candidate", "elo_reference",
                                       draws=args.draws, seed=args.seed),
        "market_context": {
            "n": int(len(priced)),
            "candidate_brier": round(ev.brier(priced, "candidate"), 4),
            "market_shin_brier": round(ev.brier(priced, "market_shin"), 4),
            "note": "Reported without a threshold. G6 gates nothing.",
        },
        "gates": gates,
        "approved": approved_model,
        "approval_note": (
            "approved_model = G1 and G2 and G3 and G4 and G5 and G8, the rule frozen at P2. "
            "approved_edge is a separate flag evaluated at P5."),
        "limitations": [
            "One read. These seasons cannot be scored again under any rule.",
            "The expanding window continues through the sealed block: the 2025 fold is "
            "fitted on everything before it, 2024 included. Same procedure, inside one read.",
            "Approval is a statement about this specification, not a decision to publish it.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, allow_nan=False) + "\n")

    spent = dict(frozen, sealed_read_recorded=True,
                 sealed_read_utc=artifact["as_of_utc"],
                 sealed_read_artifact=str(args.output))
    Path(record_path).write_text(json.dumps(spent, indent=2, allow_nan=False) + "\n")

    print(f"sealed read: {len(sealed)} games in {sealed_seasons}")
    print(f"  candidate brier {scored['brier']:.4f}  acc {scored['accuracy']:.4f}  "
          f"auc {scored['auc']:.4f}  ece {scored['ece']:.4f}")
    print(f"  elo reference   {ev.brier(sealed, 'elo_reference'):.4f}")
    print(f"  acceptance interval [{acceptance['low']:.4f}, {acceptance['high']:.4f}] "
          f"-> G5 {'inside' if inside else 'OUTSIDE'}")
    for season in per_season:
        print(f"  {season['season']}: n={season['n']} brier {season['brier']:.4f} "
              f"(elo {season['brier_elo_reference']:.4f})")
    print(f"  approved_model: {approved_model}")
    print("  the seal is now spent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
