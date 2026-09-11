# Team-outcome model foundation: fixed-core fantasy edge

## Status and decision

The unweighted fixed-core home-team edge contains useful directional signal for
actual NFL game outcomes, but the 2025 holdout does **not** support treating its
magnitude as a smoothly increasing win probability.

It also did **not** improve closing-market winner accuracy. The closing market
picked 42 of 67 games correctly (62.7%); the fixed-core sign picked 41 (61.2%).

For now, interpret the feature as three ordinal states—negative, neutral, and
positive—rather than as a calibrated continuous probability. This result is a
foundation for a future team-outcome model, not evidence that the current player
fantasy-point model is itself a production win/loss predictor.

## Research question

Does aggregating individual Yahoo fantasy-point projections into a fixed team
core produce out-of-sample signal for which team wins an NFL game?

The tested fixed core selected the highest projected players at each position:

- 1 QB
- 3 RB
- 3 WR
- 2 TE
- 1 DEF

There was no salary cap, flex position, or kicker. Home-team edge was:

```text
sum(home fixed-core projections) - sum(away fixed-core projections)
```

## Data and leakage controls

- Source pool: 360 historical Yahoo Showdown games and 10,115 player-games;
  one slate without an NFLVerse schedule match was skipped.
- The coefficients used to generate the team edges were fitted only on
  pre-2025 rows.
- The salary model form and ridge penalty had already been selected using 2025
  player-level fantasy-point error. Game outcomes were not used in that
  selection, but this makes 2025 an exploratory downstream evaluation rather
  than a fully untouched final holdout for the complete modeling pipeline.
- There were 68 matched 2025 games; one tied game was excluded, leaving 67.
- Yahoo historical salary represents pregame player valuation; NFLVerse supplies
  settled schedules and outcomes.

The nominal construction requires ten players per team. After the historical
eligibility filters, only 55 of the 67 games contained every required positional
slot for both teams. The original 67-game analysis summed all available players
up to each positional limit in the other 12 games. Results below preserve that
original calculation for reproducibility and report the strict complete-core
subset separately.

The Yahoo Showdown source creates selection bias toward nationally featured or
otherwise published single-game slates. Results should not yet be generalized to
all NFL games without broader validation.

## Holdout results

Across all 67 holdout games, the sign of the fixed-core edge selected the winner
61.2% of the time (41 of 67), compared with a 56.7% home-team baseline.

| Metric | Result |
| --- | ---: |
| Direction accuracy | 61.2% |
| ROC-AUC | 0.588 |
| Brier score | 0.245 |
| Log loss | 0.687 |
| Edge correlation with actual margin | 0.149 |
| Rank correlation with winning | 0.151 |

These discrimination and calibration statistics are modest. They indicate a
candidate feature, not a complete game model.

## Closing-market comparison

All 67 games joined directly by NFLVerse game ID to complete home and away
moneylines, spreads, spread prices, totals, and total prices. American
moneylines were converted to implied probabilities, then the home and away
probabilities were normalized to remove the bookmaker overround.

| Metric | Fixed-core edge | Closing market |
| --- | ---: | ---: |
| Winner accuracy | 61.2% (41/67) | **62.7% (42/67)** |
| ROC-AUC | 0.588 | **0.629** |
| Brier score | 0.245 | **0.237** |
| Log loss | 0.687 | **0.664** |
| Correlation with actual margin | 0.149 | **0.434** |

The fixed-core edge did not improve the closing market on any reported overall
metric. The one-game accuracy difference is not meaningful: among 25 games in
which the picks differed, the market was correct 13 times and the fixed core 12
times. An exact paired test provides no evidence of a difference.

| Closing-market result | Fixed core wrong | Fixed core correct | Total |
| --- | ---: | ---: | ---: |
| Market wrong | 13 | **12** | 25 |
| Market correct | **13** | 29 | 42 |
| Total | 26 | 41 | 67 |

The directional pick matrix was:

| Closing pick | Fixed-core pick | Games | Home wins | Market accuracy | Core accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Away | Away | 18 | 6 | 66.7% | 66.7% |
| Away | Home | 10 | 6 | 40.0% | **60.0%** |
| Home | Away | 15 | 9 | **60.0%** | 40.0% |
| Home | Home | 24 | 17 | 70.8% | 70.8% |

The apparent asymmetry is exploratory. Accepting only the model's home-team
overrides would have produced 44 of 67 correct (65.7%), but that rule was found
after inspecting these same games. It must be preregistered and tested on a new
season before it can be treated as signal.

On the stricter 55-game subset with both complete ten-player cores, the market
and model each selected 35 winners (63.6%). Their 20 disagreements split exactly
10-10. The incomplete-core games therefore do not change the conclusion: this
sample shows no winner-accuracy improvement over the closing market.

Closing odds are a strong end-of-information benchmark, not a timestamp-matched
feature comparison. They may reflect late news that Yahoo salary did not. A fair
same-information-time comparison requires archived odds captured no later than
the Yahoo salary snapshot or slate lock.

## Monotonicity finding

Ordering the holdout by unweighted fixed-core home-team edge produced:

| Edge quintile | Games | Average edge | Home win rate | Average margin |
| --- | ---: | ---: | ---: | ---: |
| Lowest | 14 | -15.49 | 42.9% | -0.5 |
| 2 | 13 | -6.71 | 46.2% | -2.4 |
| 3 | 13 | -0.12 | 53.8% | -0.8 |
| 4 | 13 | +8.45 | **76.9%** | +6.8 |
| Highest | 14 | +20.14 | 64.3% | +4.0 |

The relationship was not strictly monotonic: the highest quintile fell below
the fourth quintile, creating one violation. Rank correlation with winning was
only 0.151.

The coarser grouping was clearer:

| Edge tier | Games | Home wins | Home win rate |
| --- | ---: | ---: | ---: |
| Negative | 27 | 12 | 44.4% |
| Neutral | 13 | 7 | 53.8% |
| Positive | 27 | 19 | 70.4% |

This supports using the edge as an ordinal directional feature. It does not
support converting each additional edge point into a precise probability gain.
The top-quintile decline could be sampling noise, saturation, slate-selection
bias, or omitted game context.

## Alternatives examined

A position-weighted fixed core did not improve the result. It produced 53.7%
accuracy, 0.577 AUC, 0.248 Brier score, and 0.692 log loss. Although historical
standardized importance estimates assigned 30.8% to TE, 30.6% to QB, 22.3% to
WR, 12.1% to RB, and 4.2% to DEF, those weights were unstable: DEF alone had
0.667 AUC on the 2025 holdout. The unweighted core is therefore the safer base
feature.

An any-position top-salary sweep from N=5 through N=15 was exploratory. N=8
peaked at 64.2% accuracy and 0.604 AUC, but performance was non-monotonic across
N and the same holdout was used to inspect the variants. Selecting N=8 from
those results would overfit the holdout, so it should not be promoted without a
new untouched evaluation period.

## Future prediction-model outline

### 1. Define paired targets

Train and evaluate both:

- home-team win probability; and
- home-team scoring margin.

The probability target measures classification and calibration. The margin
target helps determine whether a feature captures game strength even when the
binary outcome is noisy.

### 2. Establish leakage-safe baselines

Use expanding-window, season-based walk-forward validation, with a final season
kept untouched until model selection is complete. Compare every candidate with:

- home-team-only probability;
- a simple rolling team-strength or Elo baseline;
- no-vig closing moneyline as an evaluation benchmark;
- fixed-core edge tiers alone; and
- the same feature set without fixed-core edge.

Do not select feature definitions, Top N, thresholds, or calibration methods on
the final holdout.

### 3. Treat fixed-core edge conservatively

Begin with a categorical negative/neutral/positive feature:

```text
logit(P(home win)) = intercept + home field + edge tier + team/context features
```

Also test the continuous edge only as a challenger. If a smooth relationship is
needed, fit a monotonic spline or isotonic calibrator entirely inside each
training fold. Never impose or tune that mapping using the final holdout.

### 4. Add pregame team and context features

Candidate inputs should be timestamped and available before kickoff:

- rolling team strength, EPA, success rate, and offense/defense efficiency;
- quarterback status and other roster availability;
- rest, travel, home field, and short-week indicators;
- pace and expected play volume;
- weather for outdoor games; and
- the salary-position-depth aggregate and its positional subcomponents.

NFLVerse can anchor schedules, results, play-by-play efficiency, rosters, depth,
and injury feeds where available. Yahoo historical salary remains a useful
pregame consensus proxy. The proposed model should not require sportsbook lines
or other betting-market inputs. Closing odds are retained as a benchmark, not a
production feature.

For a separate research-only test of incremental information, compare a market
probability with and without the edge tier:

```text
market only: logit(P(home win)) = logit(no-vig market probability)
combined:    logit(P(home win)) = logit(no-vig market probability)
                                  + edge tier
```

Fit the tier effect inside each training fold. This residual-signal test must
not silently turn the market-free production candidate into a market-derived
model.

### 5. Evaluate probability quality, not only picks

Report accuracy, balanced accuracy, ROC-AUC, Brier score, log loss, calibration
curves, margin MAE/RMSE, and season-level confidence intervals. Inspect metrics
by edge tier, favorite strength, season, and slate type. A useful model must be
well calibrated and must not depend on one small or selected season.

### 6. Set promotion gates before testing

A production W/L model should:

- beat the agreed simple baseline on Brier score and log loss across multiple
  walk-forward seasons;
- improve or preserve calibration, not merely raw pick accuracy;
- show incremental value from fixed-core edge in an ablation test;
- report performance against the closing market without tuning rules on that
  comparison sample;
- retain performance on a never-tuned final season; and
- document data timestamps so injuries and other late information cannot leak.

## Recommended disposition

Keep the unweighted fixed-core edge as a research feature and expose it, if
needed, as negative/neutral/positive. Do not display a game win probability from
it yet, and do not override closing-market picks based on the observed home-side
split. The next defensible step is a reproducible, checked-in walk-forward
evaluator over multiple seasons and a broader set of games, with complete-core
coverage reported explicitly.
