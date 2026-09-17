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

**Resolved, and this section was wrong.** The 360-game ceiling was an
instrument limit, not a limit on the hypothesis. Rebuilding the player
expectation from nflverse puts the same fixed core on 4,594 games without
Yahoo at all — see *The same hypothesis at scale* in the foundation document.
RQ2 was answerable the whole time, and deferring it to a gate that could only
fail on coverage was a misreading. The Yahoo question below remains open and
still needs Yahoo.

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

This table is the candidate list as of P1. The binding version — eleven terms
plus an intercept, with four named exclusions — is
[`team-outcome-prereg.md`](team-outcome-prereg.md) §4, frozen at P2.

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

**Spent.** P4 read 2024-2025 on 2026-09-11. The result is in §12; these seasons
are not available to any future evaluation.

The sealed seasons are scored once. If the gates fail, the artifact records
the failure and the model is not promoted. A second attempt requires a new
sealed season — the next complete one — not a re-read of the same games with
an adjusted rule. `model/component_priors.json` is the precedent: it shipped
with four `approved: false` entries and an empty `priors` list rather than
with a lowered bar.

## 7. Model families and calibration

The binding version of this section is
[`team-outcome-prereg.md`](team-outcome-prereg.md) §5, frozen at P2, which also
records the ridge grid and the decision not to run gradient boosting.

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

**Know the playing field.** P1 measured it rather than leaving it an estimate.
On 4,505 priced games in 2007-2023 the no-vig closing market scores 0.2100,
against 0.2456 for picking the home team and 0.2239 for fitted Elo. The entire
space between a good baseline and the market is 0.0141, which is precisely why
§1.4's sample sizes are what they are.

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
| **G7 Edge ablation** | removing `edge_tier` degrades Brier with a bootstrap interval excluding 0, on ≥ 2,000 corpus-B games — a separate `approved_edge` flag, not a term in `approved_model` |
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
pipeline/team_outcome_core.py          the fixed-core hypothesis, rebuilt on nflverse
tools/backtest_fixed_core.py           the fixed core at scale -> model/fixed_core.json
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
| P1 | baselines | **done** — see below |
| P2 | preregistration | **done** — see below |
| P3 | candidates, walk-forward | **done** — see below |
| P4 | sealed read | **done, seal spent** — see below |
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

Margin sd across the corpus is 14.62 on unsealed seasons, which is the raw
spread the §3 residual scale has to beat. Ties are 14 unsealed games, scored as
half a win.

One correction this phase forced on itself: the first P0 artifact reported a
home win rate and a tie count for every season, sealed ones included, and the
first write-up quoted a 2021-2025 era rate. A base rate for a holdout season is
exactly the kind of number that can steer a later modelling choice, so outcome
statistics now stop at the seal while coverage counts still span the corpus —
later phases have to know those games exist. The as-of audit still runs over
sealed weeks: it reports whether the boundary held, never what happened.

### P1 result

`pipeline/team_outcome_eval.py`, `pipeline/team_outcome_fit.py`,
`tools/build_team_outcome_baselines.py` and `tests/test_team_outcome_eval.py`,
with the evidence in `model/team_outcome_baselines.json`. Expanding-window
walk-forward over **2007-2023, 4,594 games in 17 seasons** — the first eight
seasons are training-only and 2024-2025 are sealed and untouched.

| Baseline | Brier | 95% interval | Accuracy | AUC | ECE | Cox slope |
| --- | ---: | :---: | ---: | ---: | ---: | ---: |
| Home team only | 0.2456 | [0.2427, 0.2484] | 56.1% | 0.504 | 0.018 | — |
| Elo, textbook constants | 0.2263 | [0.2213, 0.2311] | 62.2% | 0.673 | 0.047 | 1.41 |
| Elo, fitted per fold | 0.2239 | [0.2182, 0.2293] | 63.4% | 0.673 | 0.020 | 0.97 |
| No-vig market (2007-2023, n=4,505) | **0.2100** | — | 66.4% | 0.722 | 0.018 | 1.01 |

Five things this settles.

- **The room a candidate model has is 0.0141 Brier.** The market beats fitted
  Elo by that much, interval [0.0096, 0.0187], and the plan's §8 estimate of a
  market Brier "commonly in the 0.21-0.22 region" measures at 0.2100. So the
  whole playing field between a good baseline and the market is about fourteen
  thousandths, and §1.4's sample sizes were not pessimism.
- **The span can resolve what is at stake.** Bootstrap intervals on this
  walk-forward are roughly ±0.005 Brier wide, which is the scale of the
  differences worth shipping. G1's coverage floor is met: 4,594 games, 17
  seasons.
- **Elo is a real baseline, not a formality.** It beats home-only by 0.0218
  [0.0168, 0.0271] Brier with an exact-McNemar p below 0.001 on 1,440
  discordant picks. A candidate that does not clear this line is not close.
- **The de-vig choice does not change any conclusion here.** Proportional and
  Shin agree on Brier to four decimals and never differ by more than 0.017 on
  a single game, and the model-versus-market ordering is identical under both.
  Shin is slightly better calibrated (slope 1.007 against 1.042), as its theory
  predicts. §13's "the de-vig method flips the ordering" risk is retired for
  this corpus, and both remain reported.
- **The textbook Elo constants cost calibration, not discrimination.** Fixed
  and fitted Elo share an AUC of 0.673, but the fixed version's ECE is 0.047
  against 0.020 and its Cox slope 1.41 against 0.97 — under-confident, exactly
  the shape a borrowed home-field constant produces. Fitting the mapping per
  fold is what fixes it, which is why §7 fits inside folds rather than adopting
  a published constant.

Two measured constants replace assumptions. The margin residual sd from Elo
alone is **13.76** against home-only's 14.68, so §3's "roughly 13" was close
and slightly optimistic. And the fitted home-field term declines across folds —
a 0.333 logit fitted through 2006 against 0.291 fitted through 2022 — which
corroborates the P0 drift finding from inside the training data alone, with no
sealed season involved.

### P2 result

[`team-outcome-prereg.md`](team-outcome-prereg.md), frozen at sha256
`171bc3eb386e…` and recorded in `model/team_outcome_prereg.json` by
`tools/freeze_team_outcome_prereg.py`. `tests/test_team_outcome_prereg.py`
checks the hash on every run, so the document cannot be edited without the edit
failing the build — which is the only thing that makes a preregistration worth
writing. An amendment before the sealed read needs `--reason` and retains the
previous hash; after a sealed read the tool refuses outright.

Six things were still open after P1 and are now fixed.

- **The season-varying intercept.** §11 of P0's result deferred this to P2.
  Settled: the expanding-window refit already tracks the drift with a lag, so
  the only addition is a `crowd_absent` control for 2020, which keeps that
  season from dragging the fitted home-field term for later folds. No recency
  weighting, no decay parameter, no rolling window — each is a knob, and P1
  showed the drift is gradual enough that the refit absorbs it. The control is
  justified by an external fact about those games, not by a pattern found in
  the data.
- **Two nested specifications, selected by an audit rather than a score.**
  A is schedules-only and runs today; B adds opponent-adjusted play-by-play
  efficiency. B is primary if and only if the play-by-play corpus passes the
  same as-of audit, with zero findings. No performance number enters that
  choice, so the fallback is not a degree of freedom. Feasibility was checked
  before committing: play-by-play loads in about two seconds a season and
  98.9% of pass and run plays carry an EPA value back to 1999.
- **Four features excluded, each with its reason recorded.** `qb_continuity`
  is the one that matters: the only honest pregame version needs to know who
  starts *this* week, and every nflverse source for that is restated current
  state. `travel_tz` costs an outside data dependency for a weak motivation,
  `plays_per_game_diff` loses the cut at twelve terms, and `prior_margin_diff`
  is collinear with Elo by construction and drops to a challenger.
- **Gradient boosting is not run**, departing from §7. The repository ships
  numpy, pandas and nflreadpy, and adding a tree library for a challenger this
  plan already expects to lose is not a trade worth making. Recorded as a
  deviation rather than quietly dropped, with a dependency-free nonlinearity
  challenger in its place.
- **G4 rescaled** from the plan's "8 of 10 seasons" to 13 of 17, the actual
  walk-forward span.
- **The decision rule disambiguated.** §9 left it unclear whether the edge
  ablation gates the model. It does not: `approved_model` is G1-G5 and G8, and
  `approved_edge` is a separate flag carrying G7. The fixed-core edge is a
  corpus-B question about one feature, and the fundamentals model does not
  depend on its answer.

The document opens by listing what has already been seen, because a
preregistration claiming to be blind would be false here. Everything in §1 of
it comes from unsealed seasons; from 2024 and 2025 only the game counts and the
moneyline coverage are known.

### P3 result

`tools/build_team_outcome_model.py` and `model/team_outcome.json`, executing
the frozen preregistration `171bc3eb386e…`. Specification **B** — the
play-by-play corpus passed the same as-of audit over all 572 settled weeks
with zero findings, which is what selects it; no performance number entered
that choice. 4,594 games, 17 seasons, eleven terms plus an intercept. The
sealed seasons were neither fitted on nor scored.

| Forecast | Brier | Accuracy | AUC | ECE | Cox slope | Margin RMSE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Home team only (P1) | 0.2456 | 56.1% | 0.504 | 0.018 | — | 14.69 |
| Fitted Elo (P1, the G3 reference) | 0.2239 | 63.4% | 0.673 | 0.020 | 0.97 | 13.77 |
| **Candidate, specification B** | **0.2190** | 63.7% | 0.692 | 0.019 | 1.08 | 13.58 |
| No-vig market (n=4,505) | 0.2100 | 66.4% | 0.722 | 0.018 | 1.01 | — |

**Every evaluable gate passes.** G1: 4,594 games across 17 seasons. G2: ECE
0.0187 with the Cox slope interval [0.956, 1.212] containing 1. G3: Brier
0.0049 below fitted Elo, interval [-0.0072, -0.0026], entirely below zero. G4:
lower Brier than Elo in 16 of 17 seasons. G8: audit clean, canary detected.
G5 is the sealed read and stays unevaluated, so the artifact records
`approved: false` — no model is promoted at P3.

Four things worth more than the headline.

- **The preregistered expectation was wrong, and the record says so.** §4 of the
  preregistration wrote down that the efficiency terms were "expected to add
  little once Elo is present". They carry almost the entire improvement:
  specification A — Elo plus every schedule-context feature — scores 0.2231,
  barely distinguishable from Elo's own 0.2239, while adding the three
  opponent-adjusted efficiency terms moves it to 0.2190. The ablation gap is
  0.0042 [0.0022, 0.0061]. Writing the expectation down beforehand is what
  turns this into a finding rather than a story told afterwards.
- **The picks are not distinguishable; the probabilities plainly are.** On the
  453 games where the candidate and Elo disagree, the candidate is right 234
  times and Elo 219: exact McNemar p = 0.51. Accuracy rises only 0.3 points
  while Brier improves with an interval well clear of zero. This is the
  argument of §5 — probability quality, not picks — showing up as a measurement
  rather than an assertion.
- **A preregistered feature turned out to be structurally dead.**
  `short_week_diff` is identically zero across all 4,594 games: an NFL short
  week puts *both* teams on short rest, so a directional form can never fire.
  That is a defect in the P0 "correction" recorded above — the plan's original
  symmetric indicator could at least distinguish a Thursday game from a Sunday
  one, and making it directional removed the only information it had. It stays
  in place. Replacing a feature after seeing a run is precisely the freedom the
  preregistration exists to remove, and the feature is inert rather than
  harmful. The run now reports degeneracy for every feature, so this class of
  defect is measured rather than noticed by eye.
- **The noisy selector was noisy, as predicted.** The preregistered
  leave-last-season-out rule picked ridge penalties of 0.01, 10 and 100 across
  folds — bouncing four orders of magnitude on a single held-out season. The
  preregistration said this rule was noisy before it ran; following it anyway
  is what makes the prediction worth anything.

The challengers behaved as specified and none is promoted: the nonlinearity
term is worth +0.0001, adding `prior_margin_diff` is worth −0.0005 with an
interval straddling zero, and specification A is +0.0042 worse.

Against the market the candidate closes about a third of P1's gap, from 0.0141
to **0.0090** [0.0053, 0.0128]. G6 has no threshold and gates nothing; the
market remains ahead, which is the expected outcome for a model that reads no
market data.

### P4 result: the rule says approved, and the evidence is weaker than that word

`tools/read_team_outcome_seal.py` and `model/team_outcome_sealed.json`. The
2024 and 2025 seasons — 570 games nothing in this project had read — were
scored once. The seal is now spent: `model/team_outcome_prereg.json` records
the read, and both the freeze tool and the reader refuse to reopen it.

The acceptance interval was computed from the unsealed walk-forward **before**
any sealed data was scored, and the run refuses to proceed unless the
walk-forward reproduces the Brier the P3 artifact recorded.

| | Walk-forward (2007-2023) | Sealed (2024-2025) |
| --- | ---: | ---: |
| Candidate Brier | 0.2190 | **0.2188** |
| Fitted Elo Brier | 0.2239 | **0.2188** |
| Candidate advantage over Elo | −0.0049 [−0.0072, −0.0026] | **+0.0001 [−0.0059, +0.0063]** |
| ECE | 0.0187 | **0.0367** |
| Gap to the no-vig market | 0.0090 | 0.0134 |

**`approved_model` is `true`.** All six gates in the frozen decision rule
passed, G5 included: the sealed Brier of 0.2188 sits inside the acceptance
interval [0.2135, 0.2246]. That is what the preregistration committed to, and
it is recorded as such.

**It should not be promoted, and the same table says why.**

- **The advantage over the baseline did not reproduce.** The candidate's whole
  claim was that it beat a fitted Elo model — 0.0049 with an interval clear of
  zero across 17 seasons. On the sealed seasons the two are indistinguishable:
  0.2188 against 0.2188, a gap of +0.0001 with an interval straddling zero.
  Elo was better in 2024 (0.2110 against 0.2136) and the candidate in 2025
  (0.2241 against 0.2266), and they cancel.
- **Calibration degraded past the project's own threshold.** Sealed ECE is
  0.0367 against 0.0187 on the walk-forward. G2 requires ≤ 0.025 — applied to
  the sealed seasons, the calibration gate would have failed. G2 was
  preregistered against the walk-forward, so it passes as written.

**G5 was the wrong gate, and the seal is what revealed it.** As frozen, G5
asks whether the Brier *level* reproduces. It does, almost exactly. But a
Brier level can reproduce while the model has no advantage over a baseline a
tenth its complexity — which is precisely what happened. The gate should have
required the *advantage* to reproduce: the candidate-minus-Elo gap inside the
walk-forward interval for that gap, not the raw score inside the interval for
the raw score.

That correction is not applied here. Rewriting a gate after seeing the result
it produced is the exact freedom this machinery exists to remove, and the
2024-2025 seasons cannot be scored again under any rule. It is recorded as the
first amendment the next preregistration must carry, and the next sealed season
is 2026.

**Disposition.** Do not display a win probability. Do not treat the
play-by-play specification as established: its one out-of-sample test says it
matches Elo. The defensible next step is a new preregistration with a
gap-based G5, evaluated on 2026 once that season is complete, with the current
model and fitted Elo both carried forward unchanged so the comparison is
genuinely out of sample for both.

Two things did hold up. The Brier *level* transferred almost exactly, 0.2190
to 0.2188, so the walk-forward was not optimistic about absolute accuracy. And
the margin model's sealed RMSE of 12.99 against the walk-forward's 13.58 is
the sealed seasons being less variable, not the model improving.

### Addendum: would backing it have made money?

Not a phase of this plan — no betting model was ever proposed — but the
question a forecast invites, so it is answered with the same discipline and
kept reproducible in `tools/backtest_team_outcome_roi.py` and
`model/team_outcome_roi.json`.

**No.** One flat unit staked wherever a forecast prices an edge at the quoted
closing moneyline, priced against the **vig-included** quote because that is
the price on offer:

| Forecast | Span | Bets | ROI | 95% interval |
| --- | --- | ---: | ---: | :---: |
| Fitted Elo | 2007-2023 | 4,093 | −5.57% | [−11.46%, +0.38%] |
| Candidate | 2007-2023 | 3,988 | −4.12% | [−9.78%, +1.67%] |
| Fitted Elo | 2024-2025 | 489 | −9.47% | [−22.73%, +3.59%] |
| Candidate | 2024-2025 | 451 | −4.29% | [−16.67%, +8.42%] |

Season-block intervals this wide mean no single row is decisive on its own.
The finding rests on the direction being identical in every span, every model
and every threshold from 0% to 15% — twelve cells, all negative.

Three things make the result legible.

- **The models claim an edge on about 90% of games.** A forecast that finds
  positive expected value nine times in ten has not found edges; it is
  calibrated differently from the market. A genuine edge appears on a small
  minority.
- **The loss is worse than no skill.** The mean overround is 2.7%, so a
  bettor picking sides at random loses about 1.4% per flat unit. Losing 4-6%
  means the selections are worse than random: the model disagrees most where
  the market is most confident, which is where its own errors are largest.
- **This is §8's Brier gap in different units.** A forecast that loses to the
  market on a proper scoring rule cannot systematically find value at that
  market's prices. The 0.0141 gap and the negative ROI are one fact told twice.

The threshold sweep is recorded as a diagnostic and nothing is read off its
best cell; that selection is what the rest of this project refuses. Prices are
closing moneylines from one book, with no line shopping, no opening-line
comparison and no stake sizing. Nothing here is promoted, published, or advice.

### Addendum: would backing it have made money?

Not a phase of this plan — no betting model was ever proposed — but the
question a forecast invites, kept reproducible in
`tools/backtest_team_outcome_roi.py` and `model/team_outcome_roi.json`.

**No.** One flat unit staked wherever a forecast prices an edge at the quoted
closing moneyline, priced against the **vig-included** quote because that is
the price on offer:

| Forecast | Span | Bets | ROI | 95% interval |
| --- | --- | ---: | ---: | :---: |
| Fitted Elo | 2007-2023 | 4,093 | −5.57% | [−11.46%, +0.38%] |
| Candidate | 2007-2023 | 3,988 | −4.12% | [−9.78%, +1.67%] |
| Fitted Elo | 2024-2025 | 489 | −9.47% | [−22.73%, +3.59%] |
| Candidate | 2024-2025 | 451 | −4.29% | [−16.67%, +8.42%] |

Season-block intervals this wide mean no single row is decisive alone. The
finding rests on the direction being identical in every span, model and
threshold from 0% to 15% — twelve cells, all negative.

- **The models claim an edge on about 90% of games.** A forecast finding value
  nine times in ten has not found edges; it is calibrated differently from the
  market. A genuine edge appears on a small minority.
- **The loss is worse than no skill.** A bettor picking sides at random loses
  about half the overround, near 1.4% per unit. Losing 4-6% means the
  selections are worse than random: the model disagrees most where the market
  is most confident, which is where its own errors are largest.
- **This is §8's Brier gap in other units.** A forecast that loses to the
  market on a proper scoring rule cannot systematically find value at that
  market's prices.

The threshold sweep is a diagnostic and nothing is read off its best cell.
Prices are closing moneylines from one book, with no line shopping and no
stake sizing. Nothing here is promoted, published, or advice.

## 13. What would invalidate this plan

- **nflverse restates history.** Rosters, depth charts and injury feeds are
  current-state; §4.3 assumes that and routes around it, but a restatement in
  the *schedules* or play-by-play frames would undermine the corpus itself.
  Pin versions and record the pull date in the artifact.
- ~~**The de-vig method flips the ordering.**~~ Retired at P1: proportional and
  Shin agree on Brier to four decimals over 4,505 priced games and give the
  same model-versus-market ordering. Both stay reported, and this is re-checked
  whenever the priced sample changes.
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
