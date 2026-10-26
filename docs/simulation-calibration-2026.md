# Simulation calibration and portfolio replay — 2026

## Outcome

Participation and omission audits, chronological scoring-distribution calibration, correlation diagnostics and paired portfolio comparisons are complete. The calibrated numerical model remains an opt-in research candidate; production projections and exposure defaults are not promoted by this report.

## Participation and omitted-player audit

- 398 offensive player-games; 265 uniquely matched snap records.
- 45 zero scores occurred despite offensive snaps; 14 had only other snaps. Participation remains unknown for 51 zero-score rows; absence of a snap row is not called inactivity.
- The legacy zero-rate fit required a carry, target or pass attempt. That population omits active players with no recorded opportunity. A positive projection is not a guarantee of participation or scoring opportunity.
- The Sleeper parser drops unsupported primary positions even when fantasy eligibility includes WR or RB. Current source records demonstrate this for Travis Hunter (DB, WR eligible) and Hunter Luepke (FB, RB eligible). The parser fix retains one unambiguous supported fantasy position.
- Ten nonzero settled outcomes occurred outside the current-era frozen pools. Archived removal reasons are in omitted_player_audit.csv. Other reasons include absent projections and later role/pool exclusions whose exact rationale was not archived.
- Historical Sleeper player/projection inputs were not retained. No omitted player is inserted retrospectively from current feeds or postgame scores. New archives retain raw Sleeper inputs and explicit role exclusions.

## Calibration design

- Five expanding fits, one before each evaluated weekly decision block. Only outcomes from games with kickoff at least 24 hours before the block cutoff enter training. No evaluated game enters its own fit.
- The six earlier legacy-model games supply zero/negative frequencies as warm-up data. Positive-tail residuals use only earlier Sleeper-projection games. Each fit lists its training games and cutoff.
- Fixed shrinkage strength: 20 observations per role/position cell, chosen before the calibrated portfolio replay. It was not optimized to portfolio scores.
- Each marginal has a zero atom, a position-pooled negative atom, and a positive lognormal tail. Archived means are preserved exactly. Total variance is recalculated from the fitted mixture instead of retaining an incompatible old CV.
- Player pools, salaries, caps, exposure settings, candidate settings and entry counts are unchanged. The positive tail remains continuous, negative magnitude is a coarse pooled atom, and simulated team totals remain unconstrained.

## Forecast distribution validation

| Metric | Baseline | Chronologically calibrated |
|---|---:|---:|
| Expected zero scores (111 observed) | 35.36 | 82.60 |
| Expected negative scores (5 observed) | 0.00 | 8.46 |
| Zero-probability Brier error, lower better | 0.1934 | 0.1448 |
| Distribution CRPS, lower better | 2.4648 | 2.4588 |
| Inclusive P90 coverage | 87.0% | 88.4% |

- zero_brier: calibrated minus baseline -0.04864; paired game-bootstrap 95% [-0.05914, -0.03799].
- crps: calibrated minus baseline -0.00598; paired game-bootstrap 95% [-0.02566, +0.01469].

Means and mean-forecast MAE are unchanged. Better zero calibration is clear in this sample; overall CRPS improvement is small and uncertain.

## Correlation validation

- Historical targets fall inside exploratory game-block intervals for 27 of 28 pooled relationship groups. This is compatibility with noisy data, not proof of correctness.
- Opposing RB–DEF is flagged: observed standardized residual correlation is about -0.367 versus an average historical target of -0.129 (exploratory 95% interval about -0.513 to -0.153). Multiple relationships were inspected and the sample has only 17 games; this is a follow-up target, not an automatic new coefficient.
- Historical score-correlation targets are retained. New marginals get a new Gaussian latent matrix through numerical inversion and PSD repair; retaining the old latent matrix would silently change score correlations.
- Independent 100,000-scenario checks per game are saved in correlation_simulation_check.json. These check numerical implementation, not empirical football accuracy.

## Portfolio replay

17 games × 10 paired seeds × four exposure settings × 20 entries for each model. Standard accuracy: 10,000 scenarios and 10,000 candidate setting. Each cell is the realized best entry in the 20-entry set, averaged over games and seeds.

| Player / Superstar | Baseline actual best FP | Calibrated actual best FP | Change | Game-bootstrap 95% | Positive seed averages | Games W / T / L |
|---|---:|---:|---:|---|---:|---|
| 0.75/0.25 | 93.95 | 93.74 | -0.21 | [-1.18, +0.72] | 5/10 | 7 / 0 / 10 |
| 1/0.25 | 94.23 | 94.15 | -0.08 | [-0.93, +0.73] | 5/10 | 6 / 0 / 11 |
| 0.75/1 | 94.82 | 94.06 | -0.76 | [-1.65, +0.11] | 2/10 | 6 / 0 / 11 |
| 1/1 | 95.18 | 94.42 | -0.76 | [-1.68, +0.09] | 2/10 | 5 / 0 / 12 |

Within the calibrated model, exposure changes versus 0.75 / 0.25:

| Setting | Best-score change | Game-bootstrap 95% | Positive seed averages |
|---|---:|---|---:|
| 1/0.25 | +0.41 | [-0.19, +1.19] | 9/10 |
| 0.75/1 | +0.32 | [-0.25, +0.92] | 8/10 |
| 1/1 | +0.68 | [-0.12, +1.54] | 9/10 |

## Decision and limits

- Keep the demonstrated parser and archive fixes reviewable as ordinary code changes.
- Keep the calibrated numerical model opt-in. Zero calibration improves, but broader distribution scores, portfolio outcomes, correlation uncertainty and missing-pool provenance do not support calling it fully validated.
- The chronological fitting guards against using each game in its own calibration. This research design was developed after viewing this season, so it is not an untouched prospective test.
- Seventeen games are the outcome sample; seeds repeat algorithm draws. Intervals resample games after averaging seeds and are exploratory across settings.
- The final swap search has a 1.5-second wall-clock budget. Saved selected entries are authoritative; exact entry membership can vary across machines.
- Complete opponent fields and payout curves remain absent. Best-score coverage is not tournament ROI.

## Reproduce

Run the original replay preparation first, or restore the supplied exposure_2026 evidence. Then:

```bash
python tools/prepare_calibrated_replay.py
python tools/evaluate_simulation_calibration.py
python tools/run_calibrated_replay.py
python tools/summarize_calibrated_replay.py
```

The participation audit additionally uses cached nflreadpy 2026 snap counts and player IDs. Raw historical Sleeper inputs do not exist and cannot be reconstructed from current feeds.
