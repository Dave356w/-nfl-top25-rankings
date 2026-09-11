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

## Resolving power of this sample

The point estimates above are reported without intervals, which makes the
sample look more decisive than it is. Exact (Clopper-Pearson) 95% intervals on
the same counts:

| Rule | Correct | Accuracy | 95% interval |
| --- | ---: | ---: | :---: |
| Home team always | 38/67 | 56.7% | [44.0%, 68.8%] |
| Fixed-core sign | 41/67 | 61.2% | [48.5%, 72.9%] |
| No-vig closing market | 42/67 | 62.7% | [50.0%, 74.2%] |
| Fixed core, strict 55 | 35/55 | 63.6% | [49.6%, 76.2%] |
| Any-position N=8 | 43/67 | 64.2% | [51.5%, 75.5%] |
| Post-hoc home override | 44/67 | 65.7% | [53.1%, 76.8%] |

Every rule lies inside every other rule's interval. A one-sided exact binomial
puts the fixed core at p = 0.043 against a fair coin, but p = 0.27 against the
56.7% home base rate — the baseline that matters.

The coarse tier grouping reported under *Monotonicity finding*, combined with
the overall count of 41 correct picks, determines where the accuracy came from.
The outer tiers' picks are fixed by construction: a negative edge picks the
away team, a positive edge picks the home team.

| Edge tier | Games | Home wins | Core pick | Core correct | Home-always correct |
| --- | ---: | ---: | :---: | ---: | ---: |
| Negative | 27 | 12 | away | 15 | 12 |
| Neutral | 13 | 7 | by sign | 7 | 7 |
| Positive | 27 | 19 | home | 19 | 19 |
| Total | 67 | 38 | | **41** | **38** |

In the positive tier the fixed core and the home-always rule are the same
rule and cannot disagree. The 70.4% positive-tier home win rate is therefore a
statement about home teams, not about the edge; its own interval,
[49.8%, 86.2%], contains the sample's 56.7% home win rate.

All measured improvement over the home baseline is three net flips among the
27 negative-tier games — 15 correct against 12. Those 27 games are the
complete set of discordant pairs, and an exact McNemar test on 15 versus 12
gives p = 0.70.

This does not show the edge is worthless. It shows the sample holds about
three games' worth of evidence about it. Sizing a comparison that could
resolve the question is the first section of
[`team-outcome-model-plan.md`](team-outcome-model-plan.md).

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

This outline has been expanded into an implementable specification in
[`team-outcome-model-plan.md`](team-outcome-model-plan.md), which fixes
feature definitions, the walk-forward protocol, numeric promotion gates, the
artifact and module layout, and the delivery phases. The summary:

1. **Paired targets.** Fit home scoring margin and home win probability, with
   margin as the primary target and the probability derived from it through a
   fold-fitted link.
2. **Leakage-safe baselines.** Expanding-window season walk-forward, a sealed
   final holdout, and comparison against home-only, a rolling team-strength or
   Elo baseline, the no-vig closing market as a benchmark, edge tiers alone,
   and the same feature set without the edge.
3. **Conservative treatment of the edge.** Categorical negative/neutral/positive
   first; the continuous edge only as a challenger, with any smooth mapping
   fitted entirely inside training folds.
4. **Pregame team and context features.** Timestamped rolling efficiency,
   availability, rest and travel, pace, weather, and the salary-position-depth
   aggregate — with closing odds retained as a benchmark, never a feature.
5. **Probability quality, not picks.** Accuracy, AUC, Brier, log loss,
   calibration curves, margin error, and interval estimates for every
   difference, cut by tier, favorite strength, season and slate type.
6. **Gates set before testing.** Beat the agreed baseline on proper scoring
   rules across multiple walk-forward seasons, preserve calibration, show
   incremental value from the edge in an ablation, retain performance on a
   never-tuned final season, and document every data timestamp.

The plan makes three substantive changes to this sketch. The corpus is split
in two — all nflverse games for the model, the Yahoo-priced subset for the
edge ablation — because 360 priced games cannot settle a Brier difference of
the size at stake. The fixed-core edge moves from the centre of the model to
an ablation run last, since it is not yet distinguishable from picking the
home team. And the gates carry numbers, one of which the current corpus
already fails.

## Recommended disposition

Keep the unweighted fixed-core edge as a research feature and expose it, if
needed, as negative/neutral/positive. Do not display a game win probability from
it yet, and do not override closing-market picks based on the observed home-side
split. The next defensible step is a reproducible, checked-in walk-forward
evaluator over multiple seasons and a broader set of games, with complete-core
coverage reported explicitly. That evaluator and everything it needs are
specified in [`team-outcome-model-plan.md`](team-outcome-model-plan.md).
