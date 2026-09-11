# Team-outcome prediction model: implementation plan

This expands the *Future prediction-model outline* in
[`team-outcome-model-foundation.md`](team-outcome-model-foundation.md) from a
six-step sketch into a specification that can be built, reviewed and gated.
Read the foundation document first: it is the experimental record, and this
plan assumes its results rather than restating them.

Nothing here is implemented. There is no model artifact, no trainer and no
published output. The plan's first job is to say what would have to be true
before any of those exist.

## Summary of changes to the outline

Re-reading the 2025 holdout with interval estimates rather than point
estimates changes three things about the original outline.

1. **The corpus is wrong for the question.** The outline proposes
   walk-forward seasons and Brier/log-loss promotion gates, but the games it
   has are 360 Yahoo Showdown slates, 68 of them in the holdout season.
   Settling a Brier difference of the size actually available needs thousands
   of games. Splitting the work across two corpora — all nflverse games for
   the model, the Yahoo-priced subset for the fixed-core ablation — is the
   structural change this plan makes.
2. **The fixed-core edge belongs at the end, not the start.** Outline step 3
   builds the model around the edge and adds context features afterwards. The
   holdout cannot distinguish the edge from "pick the home team", so the
   fundamentals model has to exist first and the edge has to earn its place
   against it.
3. **The gates need numbers, and one of them already fails.** With 360
   priced games, the edge-ablation gate cannot pass at any effect size worth
   shipping. Recording that as `approved: false` is the honest outcome, the
   same way `model/component_priors.json` records four positions that did not
   clear their gate.

## 1. What the 2025 sample can and cannot support

### 1.1 Every headline number sits inside every other one's interval

Exact (Clopper-Pearson) 95% intervals on the foundation document's counts:

| Rule | Correct | Accuracy | 95% interval | Width |
| --- | ---: | ---: | :---: | ---: |
| Home team always | 38/67 | 56.7% | [44.0%, 68.8%] | 24.7 pts |
| Fixed-core sign | 41/67 | 61.2% | [48.5%, 72.9%] | 24.4 pts |
| No-vig closing market | 42/67 | 62.7% | [50.0%, 74.2%] | 24.2 pts |
| Fixed core, strict 55 | 35/55 | 63.6% | [49.6%, 76.2%] | 26.6 pts |
| Any-position N=8 sweep | 43/67 | 64.2% | [51.5%, 75.5%] | 24.0 pts |
| Post-hoc home-override | 44/67 | 65.7% | [53.1%, 76.8%] | 23.8 pts |

Every rule in that table, including the two the foundation document already
labels exploratory, lies inside every other rule's interval. A one-sided
exact binomial puts the fixed core at p = 0.043 against a fair coin and
p = 0.27 against the 56.7% home-team base rate. Against the baseline that
matters, the feature is not distinguishable from picking the home team.

### 1.2 The measured edge is three games, all of them in one tier

The foundation document's tier table and its overall count of 41 correct
picks determine the decomposition, because the outer tiers' picks are fixed
by construction — a negative edge picks the away team, a positive edge picks
the home team:

| Edge tier | Games | Home wins | Core pick | Core correct | Home-always correct |
| --- | ---: | ---: | :---: | ---: | ---: |
| Negative | 27 | 12 | away | 15 | 12 |
| Neutral | 13 | 7 | by sign | 7 | 7 |
| Positive | 27 | 19 | home | 19 | 19 |
| Total | 67 | 38 | | **41** | **38** |

Two consequences follow, and neither is visible in the published tables.

In the positive tier the fixed core and the home-always rule are the *same
rule*. They cannot disagree, so the 70.4% positive-tier home win rate — the
strongest-looking row in the foundation document — carries no information
about the feature. It is a statement about home teams. Its own interval,
[49.8%, 86.2%], contains the sample's overall 56.7% home win rate.

All of the feature's measured improvement is therefore three net flips among
the 27 negative-tier games: 15 correct where home-always got 12. Those 27
games are the complete set of discordant pairs, and an exact McNemar test on
15 versus 12 gives p = 0.70.

This does not show the edge is worthless. It shows the 2025 sample contains
roughly three games' worth of evidence about it.

### 1.3 The market comparison has no resolving power either

Among the 25 games where the picks differed, the market was right 13 times
and the core 12: exact McNemar p = 1.0. On the strict 55-game subset the 20
disagreements split 10-10: p = 1.0. The correct reading is not "the feature
matches the market"; it is "67 games cannot tell the difference between any
two rules that are both roughly 60% accurate".

### 1.4 How many games a real comparison needs

Minimum detectable difference at 5% significance and 80% power, for a paired
design where the two rules disagree on 25-35% of games:

| Accuracy difference | 25% discordant | 35% discordant |
| --- | ---: | ---: |
| 5 points | ~785 games | ~1,100 games |
| 3 points | ~2,180 games | ~3,050 games |
| 2 points | ~4,900 games | ~6,870 games |
| 1 point | ~19,600 games | ~27,500 games |

For the probability scores the outline actually gates on, using a paired test
on per-game score differences:

| Brier difference | sd 0.06 | sd 0.10 |
| --- | ---: | ---: |
| 0.010 | ~283 games | ~785 games |
| 0.005 | ~1,130 games | ~3,140 games |
| 0.002 | ~7,060 games | ~19,600 games |

A 0.005 Brier improvement is not a small one. The closing market's 0.237 and
an uninformative 0.25 are about as far apart as NFL games allow, so the
differences worth shipping live in the 0.005-0.015 range — and need roughly
1,000-3,000 games to establish.

The Yahoo Showdown corpus has 360 slates in total and had 68 in the holdout
season. **The corpus that produced the feature cannot settle the question the
outline asks about it.**

### 1.5 "Market-free" needs a caveat

The outline states that the proposed model should not require sportsbook
lines, and keeps closing odds as a benchmark only. That is a sound
engineering rule — it is about dependencies, licensing and feed fragility —
and this plan keeps it.

It is not, however, evidence that the fixed-core edge is independent of the
market. Yahoo DFS salaries are set by a pricing desk reading the same public
information, and DEF salary in particular is close to a function of expected
points allowed, which is opponent strength by another name. The foundation
document's observation that DEF alone scored 0.667 AUC on the holdout — higher
than the full core — is what a market-derived team-strength proxy looks like,
not what a defensive fantasy insight looks like.

So: keep the no-sportsbook rule, and stop describing results built on Yahoo
salary aggregates as market-independent.

## 2. Scope: two corpora, two questions

| | Corpus A | Corpus B |
| --- | --- | --- |
| Source | nflverse schedules and play-by-play | Yahoo historical Showdown slates |
| Coverage | every game, 1999-present (~7,300) | 360 priced single games |
| Per season | ~267 through 2020, ~285 from 2021 | ~60, selection-biased |
| Selection | complete | nationally featured slates |
| Answers | baselines, context features, calibration | fixed-core edge only |
| Blocked by | nothing; data is public and settled | corpus size, Yahoo retention |

Two research questions follow, in this order:

- **RQ1 (Corpus A).** Can a market-free fundamentals model reach calibration
  comparable to the no-vig closing market on a large sample of NFL games?
- **RQ2 (Corpus B).** On games where it is defined, does the fixed-core edge
  add anything to the RQ1 model?

RQ1 is the production question. RQ2 is the ablation the foundation document
actually wants, and the only one that needs Yahoo data at all. RQ2 is not
answerable until RQ1's model exists, because "adds anything" needs something
to add to.

One open question worth resolving early, because it changes RQ2's ceiling:
the 360-slate limit is a *Showdown* limit. If Yahoo's read-only endpoints
retain completed full-slate contests as well, corpus B could grow by an order
of magnitude. `pipeline/showdown_backtest.py` already knows how to discover
and cache completed series; extending `discover()` past `SINGLE_GAME` is a
bounded check, and it should be run before anyone concludes RQ2 is
unanswerable.

## 3. Targets

Fit three, report three, promote on one.

| Target | Type | Role |
| --- | --- | --- |
| Home scoring margin | regression | primary; carries the most information per game |
| Home win | classification | direct challenger |
| Margin → win probability | derived | the production probability, if any |

The outline pairs win probability and margin without saying which leads. It
should be margin. Sixty-seven binary outcomes carry about sixty-seven bits;
the same games' margins carry considerably more, which is why the foundation
document's margin correlation separates the market from the core (0.434
versus 0.149) while the accuracy columns cannot. A margin model also yields a
falsifiable residual scale — NFL margin residuals sit near 13 points — and
that scale converts directly to probabilities:

| Predicted margin | P(home win) at sd 13.5 |
| ---: | ---: |
| +1.0 | 0.530 |
| +2.5 | 0.573 |
| +3.0 | 0.588 |
| +6.0 | 0.672 |
| +7.0 | 0.698 |
| +10.0 | 0.771 |

Fit the residual scale inside each training fold; never assume 13.5. Report
the parametric link alongside a fold-fitted isotonic map of margin to outcome
as a diagnostic, because NFL margins pile up on 3 and 7 and a normal link is
smooth where the real distribution is not.

**Ties.** The foundation document dropped the one tied game, which is
reasonable at n=67 and unacceptable as a standing rule. Score a tie as 0.5
for Brier and accuracy, or fit an ordered three-outcome model and report
P(tie) separately. Whichever is chosen goes in the preregistration; a tie
must never be silently discarded or scored as a loss.

## 4. Data contract and as-of reconstruction

### 4.1 Per-source leakage risk

| Source | Fields | Restatement risk | Rule |
| --- | --- | :---: | --- |
| schedules | `result`, scores | none (settled) | target only |
| schedules | moneyline, spread, total | none | **benchmark only, never a feature** |
| schedules | rest days, roof, surface, div/neutral, kickoff time | none | safe; fixed at scheduling |
| schedules | `temp`, `wind` | recorded, partial | treat as unavailable unless a pregame *forecast* is archived |
| play-by-play | EPA, success rate, pace | none | safe for strictly earlier weeks only |
| weekly rosters | status | **high** | current-state feed; see 4.3 |
| depth charts | `pos_slot`, ordering | **high** | current-state feed; see 4.3 |
| injury reports | designations | **high** | current-state feed; see 4.3 |
| Yahoo (corpus B) | salary, cap, settled points | none | immutable per `pipeline/showdown_backtest.py` |
| Yahoo (corpus B) | team, FPPG, projection, status | **high** | already rejected by the backtester; keep rejecting |

Closing lines are the sharpest trap here, because they are in the same
`schedules` frame as the safe scheduling fields and join on the same game id.
The feature builder must not receive that frame; it must receive a projection
of it with the market columns dropped, so a market column cannot be reached by
accident.

### 4.2 The as-of rule, enforced by construction

Every feature for a game in season *s*, week *w* may read only rows strictly
earlier than (*s*, *w*), plus fields of the game itself that are fixed when
the schedule is published.

Enforce that with a single entry point rather than a convention. As built in
`pipeline/team_outcome.py`, a feature receives an `AsOfView` — the target
week's pregame context, settled outcomes strictly earlier, and the schedule
published through that week — and never the corpus. Outcome and market columns
are not in the context frame at all, so the target week's own result is not
something a feature declines to read; it is absent.

`audit` then checks that boundary from both sides, at every settled week:
structurally, that nothing at or after the checkpoint appears in the view and
that no withheld column reaches it; and by perturbation, replacing every
outcome and closing line from the checkpoint onward and re-deriving, so a
feature that moves has been served data the filter should have excluded.

One limit, stated because it decides what the canary can be. Perturbation
catches a wrong comparison in the as-of filter — the off-by-one that turns
`<` into `<=` — and that is the canary `tests/test_team_outcome.py` asserts on.
It cannot catch a feature that closes over a corpus or a file instead of
reading its view, because that value never crosses the boundary being tested.
That case stays a review rule, and it is why the builder passes a view rather
than a corpus: the wrong thing has to be reached for deliberately.

### 4.3 Restated sources are the real hazard

nflverse publishes rosters, depth charts and injury designations as current
state. A 2019 depth chart pulled today is not necessarily the depth chart that
existed before kickoff in 2019. Any historical model that uses them is
reporting a number it could not have had.

Two acceptable responses:

- **(a) Exclude them** from the historical model, and accept that quarterback
  availability enters only through weak proxies (who started the previous
  game).
- **(b) Include them with an explicit contamination label**, and require the
  promotion gate in §9 to pass *without* them as well.

Prefer (a). Option (b) is available only if someone first checks whether
archived historical snapshots exist for the specific feeds involved.

The repository's own pregame archive under `site/data/projection_archive/`
is the honest long-run fix: it captures forecasts before kickoff with capture
timestamps, and it rejects any snapshot taken at or after kickoff. It is also
new, and it accumulates at roughly 285 games a season. Treat it as an asset
that matures around 2028-2029, not as an input available now.

## 5. Feature specification

Corpus A candidates, with the definition each one is promised to have:

| Feature | Definition | Window | Shrinkage | Sign |
| --- | --- | :---: | --- | :---: |
| `home_field` | intercept | — | season-varying allowed | + |
| `elo_diff` | Elo rating difference, K=20, 1/3 regression between seasons | forward-only by construction | none | + |
| `epa_off_diff` | opponent-adjusted offensive EPA/play difference | 8 games | n/(n+4) to league mean | + |
| `epa_def_diff` | opponent-adjusted defensive EPA/play difference | 8 games | n/(n+4) to league mean | + |
| `success_rate_diff` | early-down success rate difference | 8 games | n/(n+4) | + |
| `rest_diff` | home rest days − away rest days | — | none | + |
| `short_week_diff` | away on ≤4 days − home on ≤4 days | — | none | + |
| `bye_diff` | home off a bye − away off a bye, read from the published schedule | — | none | + |
| `neutral_site` | international or neutral | — | none | 0 |
| `travel_tz` | time zones crossed by the away team | — | none | + |
| `qb_continuity` | starter differs from previous game | — | none | — |
| `plays_per_game_diff` | pace proxy | 8 games | n/(n+4) | 0 |
| `dome` | roof closed or fixed | — | none | 0 |

Corpus B adds exactly two, and only for the ablation:

| Feature | Encoding |
| --- | --- |
| `edge_tier` | two dummies — negative and positive, neutral is the reference level |
| `salary_depth_aggregate` | continuous, **challenger only**, never in the primary |

Two disciplines apply to this list.

**Cap the count.** No more than twelve features in the production candidate.
Anything beyond that is a challenger with its own preregistered run. With
~7,300 games and a signal ceiling this low, parameter count is the main way
this project fails.

**Say the collinearity out loud now.** Elo and rolling opponent-adjusted EPA
measure substantially the same thing, and the second will probably add little
once the first is present. Writing that down before the run is the difference
between a confirmed prior and a rationalized result.

## 6. Validation protocol

### 6.1 Walk-forward specification

- Expanding window, season granularity, refit at season boundaries only.
  Within a season no refit is needed because the features are causal by
  construction; state that explicitly so the choice isn't mistaken for an
  oversight.
- Minimum training span before the first evaluated season: 8 seasons.
- Evaluated span: every season from the first eligible one through the last
  complete season minus the sealed seasons.
- **Sealed**: the two most recent complete seasons. Not read, not summarized,
  not plotted, until §6.4.

### 6.2 Preregistration

A committed `docs/team-outcome-prereg.md`, its sha256 recorded in the model
artifact, fixed before the sealed seasons are read. It must pin: the feature
list, the model family, the hyperparameter grid, the calibration method, the
primary metric, the gates in §9, and the decision rule that maps gate results
to `approved`.

This exists because of two specific things in the foundation document: the
any-position sweep from N=5 to N=15 that peaked at N=8 on the same holdout
used to inspect it, and the home-side override rule found by looking at the
pick matrix. Both are correctly labelled exploratory there. Preregistration is
what makes the next round not need that label.

### 6.3 Where each choice may be made

| Decision | May be selected on |
| --- | --- |
| Feature definitions, windows, shrinkage constants | training folds |
| Ridge penalty, tree depth, early stopping | inner folds of the training window |
| Calibration map | training folds only |
| Edge-tier boundaries | training folds only — never a holdout |
| Model family (primary vs challenger) | walk-forward seasons, preregistered |
| Anything at all | **never the sealed seasons** |

### 6.4 One read of the seal

The sealed seasons are scored once. If the gates fail, the artifact records
the failure and the model is not promoted. A second attempt requires a new
sealed season — the next complete one — not a re-read of the same games with
an adjusted rule. `model/component_priors.json` is the precedent: it shipped
with four `approved: false` entries and an empty `priors` list rather than
with a lowered bar.

## 7. Model families and calibration

**Primary.** Ridge logistic regression for the win target and ridge linear
regression for the margin target, on standardized features, with monotone sign
constraints where §5 states a sign. The signal-to-noise ratio here is low
enough that a linear model in Elo-like features is the appropriate default,
not a concession.

**Challenger.** Gradient boosting at depth ≤ 3 with monotone constraints,
heavy shrinkage, and early stopping inside training folds. It is expected to
lose. Record that it was tried and by how much, which is worth more than not
trying it.

**Edge tier.** Enters the corpus-B ablation as two dummies. The continuous
edge is a challenger only, per the foundation document's own finding that the
relationship is not reliably monotonic. If a smooth mapping is ever wanted,
fit an isotonic or monotonic-spline calibrator entirely inside training folds.

**Calibration.** Train with a proper scoring rule and prefer no post-hoc
calibration at all; an uncalibrated model trained on log loss is usually
better calibrated than a badly-fitted correction. If a correction is needed,
use Platt scaling fitted inside each training fold. Do not use isotonic
regression on fewer than about 1,500 rows — a fold with 300 games does not
have the data to support a step function.

## 8. Metrics and uncertainty

Report per evaluated season and pooled:

- accuracy, balanced accuracy, ROC-AUC;
- Brier score, and Brier skill score against the Elo-plus-home baseline;
- log loss;
- a calibration curve with bin counts, plus ECE and the calibration slope;
- margin MAE, RMSE, and residual sd;
- the same cut by favorite strength, season, dome/outdoor, and — on corpus B
  — edge tier and core completeness.

**Uncertainty is not optional.** Every difference gets an interval from a
season-block bootstrap: resample seasons with replacement, then games within
the resampled seasons, 2,000 draws, with paired differences computed on the
same draw. Pick-accuracy comparisons additionally get an exact McNemar test,
because that is the test the §1 re-analysis shows was missing. A bare point
difference is not a result.

**Reference lines**, all four reported together: home-only, Elo-only, the
no-vig closing market, and the model. The market line is a ceiling to measure
against, not a target to beat; a model that lands close to it while reading no
market data is the realistic good outcome. It is also the one line that does
not span the corpus: P0 found closing moneylines only from 2010 (4,363 games),
so report it over that span and say so, rather than quietly evaluating the
whole corpus against a benchmark two thirds of it has.

**De-vig sensitivity.** The foundation document normalized the two implied
probabilities proportionally. That is the standard first pass and it is biased
under favorite-longshot bias. Report the benchmark under proportional
normalization *and* one bias-aware method (Shin, or the power method), and
confirm the conclusion does not depend on which. If it does, say so rather
than picking the flattering one.

**Know the playing field.** Measure it rather than assuming it, but for
sizing: a long-run closing-market Brier on NFL games is commonly in the
0.21-0.22 region against 0.25 for an uninformative forecast, and outcome
entropy puts a floor somewhere near 0.20. The entire space is a few
hundredths wide, which is precisely why §1.4's sample sizes are what they are.

## 9. Promotion gates

Each gate is checkable, and each one records a number in the artifact whether
it passes or fails.

| Gate | Requirement |
| --- | --- |
| **G1 Coverage** | ≥ 3,000 evaluated games across ≥ 10 walk-forward seasons |
| **G2 Calibration** | ECE ≤ 0.025 over ≥ 10 equal-count bins, and the calibration slope's 95% interval contains 1 |
| **G3 Skill** | Brier skill score against the Elo-plus-home baseline with a bootstrap lower bound above 0 |
| **G4 Stability** | positive Brier skill in ≥ 8 of 10 walk-forward seasons |
| **G5 Seal** | sealed-season metrics inside the walk-forward bootstrap interval; one read, no re-read |
| **G6 Market context** | gap to the no-vig market over the 2010+ games that have a closing moneyline, under both de-vig methods; **no threshold** |
| **G7 Edge ablation** | removing `edge_tier` degrades Brier with a bootstrap interval excluding 0, on ≥ 2,000 corpus-B games |
| **G8 Leakage** | as-of test passes, canary detected, every feature carries a documented timestamp |

G6 carries no threshold deliberately. Beating the closing market is not the
goal and, on any sample this project will have, is not a claim that could be
supported.

**G7 cannot currently pass.** Corpus B has 360 games against a requirement of
2,000, and §1.4 shows why the requirement is not negotiable. Unless the
full-slate retention check in §2 changes the corpus size, the expected and
correct result is an artifact recording `edge_tier: approved: false` with the
coverage shortfall as the stated reason. That is a finding, not a failure of
the plan: it converts "the edge is a research feature" from a judgement call
into a number.

## 10. Artifacts, modules and tests

Shaped like the rest of the repository — a frozen JSON artifact, an offline
builder in `tools/`, an importable module in `pipeline/`, and shape tests that
run without network access.

```
pipeline/team_outcome.py               corpus split, as-of view, features, audit       [P0]
pipeline/team_outcome_eval.py          de-vig, metrics, bootstrap, paired tests        [P1]
pipeline/team_outcome_fit.py           ridge logistic and linear fitters               [P1]
tools/build_team_outcome_corpus.py     corpus + audit -> model/team_outcome_corpus.json  [P0]
tools/build_team_outcome_baselines.py  walk-forward baselines -> model/team_outcome_baselines.json [P1]
tests/test_team_outcome.py             as-of/leakage, canary, Elo, seal                [P0/P1]
tests/test_team_outcome_eval.py        de-vig, metrics, intervals, fitters             [P1]
tools/build_team_outcome_model.py      walk-forward trainer -> model/team_outcome.json
model/team_outcome.json                frozen artifact, gate results, approval flag
```

Three modules rather than one: the corpus and its as-of boundary, the fitters,
and the scoring. Scoring is the piece with the most rules attached — two de-vig
methods, ties as half a win, an interval on every difference — and it is worth
being able to test it without a corpus in the room.

The artifact follows the conventions already in `model/`:

```json
{
  "schema": 1,
  "as_of_utc": "...",
  "prereg_sha256": "...",
  "corpus": {"source": "nflverse", "seasons": [1999, 2025], "games": 7300},
  "trained_through_season": 2023,
  "sealed_seasons": [2024, 2025],
  "target": "home scoring margin; win probability via fold-fitted normal link",
  "features": ["home_field", "elo_diff", "..."],
  "walk_forward": {"seasons": [], "brier": [], "brier_skill": [], "log_loss": []},
  "gates": {"G1": {"passed": false, "value": 0, "requirement": 3000}},
  "approved": false,
  "limitations": []
}
```

Two properties matter more than the exact shape. `approved` must be read at
publish time rather than assumed, and — as the README already says of the
component priors — flipping the flag by hand must not be sufficient to enable
anything. Integration stays a reviewed change.

Explicit non-goals: no change to the daily publish path, no new workflow, no
network calls in tests, nothing rendered on the site. nflverse pulls are cached
to disk under the existing `backtest_cache/` convention.

## 11. Production contract, if it ever passes

Only if G1-G5 and G8 pass, and stated here so the bar is set before anyone is
invested in a number:

- A displayed probability is rounded to whole percent and shown with its tier
  or interval. A model this close to the noise floor should not print a third
  significant figure.
- The page states the evaluated sample, the sealed-season result, and the gap
  to the market benchmark.
- The kill switch is the `approved` flag, checked at publish time.
- The existing caveat language stays: historical regression estimates, not
  predictions with a guarantee, and not betting advice.

Until then, the foundation document's disposition stands unchanged: expose the
edge, if at all, as negative/neutral/positive, and display no win probability.

## 12. Delivery phases

| Phase | Work | Exit criterion |
| --- | --- | --- |
| P0 | corpus assembly, as-of harness | **done** — see below |
| P1 | baselines | home-only, Elo-only and market benchmark metrics recorded with intervals, under both de-vig methods |
| P2 | preregistration | `docs/team-outcome-prereg.md` committed, hash recorded |
| P3 | candidates, walk-forward | G1-G4 and G8 evaluated and written to the artifact |
| P4 | sealed read | G5 evaluated; artifact written with `approved` either way |
| P5 | corpus-B ablation | full-slate retention checked; G7 evaluated and recorded, expected to fail on coverage |

Every phase writes an artifact. No phase before P4 reads the seal.

### P0 result

`pipeline/team_outcome.py`, `tools/build_team_outcome_corpus.py` and
`tests/test_team_outcome.py`, with the evidence in
`model/team_outcome_corpus.json`. Every exit criterion met: 7,276 games
assembled across 1999-2025, a 56.3% home win rate against the known league
value, and a clean as-of audit over all 572 settled weeks, with the off-by-one
canary confirmed to fail it.

Four things the build settled that the plan had assumed:

- **The market benchmark does not cover the corpus.** Closing moneylines are
  absent before 2006, patchy through 2009, and complete only from 2010:
  4,363 games, not 7,276. Closing spreads go back to 1999, so a spread-derived
  benchmark is the option for the early seasons. §8's market reference line and
  gate G6 are scoped to 2010 onward.
- **Home field is measurably non-stationary**, which §13 raised as a risk and
  the corpus now quantifies: 57.5% (1999-2007), 56.8% (2008-2015), 56.6%
  (2016-2019), **49.8% in the crowdless 2020 season**, and 54.5% (2021-2025).
  A fixed intercept across 27 seasons is not defensible; decide the
  season-varying form at P2, before it can be chosen on results.
- **Two of §5's features could not carry the sign the table gave them.**
  "Either side on a short week" and "either side off a bye" are symmetric
  indicators on a directional target. Both are now differences, above.
- **Relocations need franchise continuity.** nflverse keeps the historical
  code, so the Rams appear as STL, LA and LAR. Aliasing is applied at corpus
  build; the 35 codes in the feed resolve to 32 franchises.

Margin sd across the corpus is 14.59, which is the raw spread the §3 residual
scale of roughly 13 has to beat. Ties are 15 games, scored as half a win.

## 13. What would invalidate this plan

- **nflverse restates history.** Rosters, depth charts and injury feeds are
  current-state; §4.3 assumes that and routes around it, but a restatement in
  the *schedules* or play-by-play frames would undermine the corpus itself.
  Pin versions and record the pull date in the artifact.
- **The de-vig method flips the ordering.** If the market benchmark's rank
  against the model depends on proportional versus Shin normalization, the
  comparison is not stable enough to report as a single line.
- **Home field is non-stationary.** It has drifted, and 2020 was played
  largely without crowds. A fixed intercept across 27 seasons is the wrong
  model; fit a season-varying intercept or restrict the corpus, and decide
  which before P3.
- **Yahoo stops retaining completed contests.** Corpus B disappears and RQ2
  becomes unanswerable rather than under-powered. Cache aggressively now;
  `pipeline/showdown_backtest.py` already writes deterministic gzip caches.
- **A feature turns out to be post-hoc.** The named suspects are recorded
  weather, anything derived from depth charts, and any quarterback feature
  that reads a status field rather than a prior game's starter.
