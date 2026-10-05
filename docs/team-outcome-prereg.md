# Team-outcome model: preregistration

Phase P2 of [`team-outcome-model-plan.md`](team-outcome-model-plan.md). This
document fixes every choice that could otherwise be made after seeing a result.
Its sha256 is recorded in `model/team_outcome_prereg.json` and checked by
`tests/test_team_outcome_prereg.py`, so editing it breaks the build until it is
re-frozen — which is the only thing that makes a preregistration worth writing.

It governs the P3 walk-forward run and the single P4 read of the sealed
seasons. Nothing here may be changed once that read has happened; see
**Amendments** at the end.

## 1. What has already been seen

A preregistration that claimed to be blind would be false. Everything below was
chosen knowing these results, all of them from unsealed seasons:

- **P0**, `model/team_outcome_corpus.json`: 7,276 scheduled games 1999-2025;
  6,706 unsealed settled games at a 56.5% home win rate with 14 ties; margin sd
  14.62; closing moneylines absent before 2006 and complete from 2010; 35
  historical team codes resolving to 32 franchises; a clean as-of audit over
  572 settled weeks.
- **P0 home-field drift**, unsealed eras: 57.6% (1999-2007), 56.8%
  (2008-2015), 56.6% (2016-2019), 49.8% (2020), 54.8% (2021-2023).
- **P1**, `model/team_outcome_baselines.json`: walk-forward 2007-2023, 4,594
  games. Brier 0.2456 home-only, 0.2263 textbook Elo, 0.2239 fitted Elo, 0.2100
  no-vig market on 4,505 priced games. Margin residual sd 13.76 from Elo alone.
  Fitted home-field logit declining 0.333 to 0.291 across folds. Proportional
  and Shin de-vigging agreeing to four decimals.

From the sealed seasons, 2024 and 2025, only two things are known: how many
games they contain, and that both carry complete closing moneylines. No
outcome, rate or score from either season has been read.

Two choices below follow directly from that exposure and are declared as such:
the `crowd_absent` control exists because 2020 is visibly anomalous, and the
decision to fit the Elo mapping per fold rather than adopt textbook constants
follows from P1's calibration gap. Both were settled on unsealed evidence.

## 2. Targets

| Target | Role |
| --- | --- |
| Home scoring margin | primary regression |
| Home win | classification |
| Win probability derived from margin | the reported probability |

A tie scores 0.5 on the win target and 0 on the margin target. Ties are never
dropped, and never scored as a loss.

The reported probability comes from the margin model through a normal link
whose residual sd is estimated **inside each training fold**, never assumed. The
direct win-probability model is fitted and reported alongside it as a check;
whichever of the two is used is fixed here in §5, not chosen from results.

## 3. Corpus and the two specifications

Both specifications are fully specified now. Which one runs is decided by an
audit result, never by a score.

**Specification A — schedules only.** The P0 corpus exactly as it stands.

**Specification B — A plus play-by-play efficiency.** Adds four features built
from nflverse play-by-play, which is verified available and complete back to
1999 (98.9% of pass and run plays carry an EPA value).

**Selection rule.** B is the primary specification if and only if the
play-by-play corpus passes the same as-of audit P0 applies to schedules, with
zero findings, over the same 572 settled weeks. Otherwise A is primary and the
artifact records the audit findings that demoted B. No performance number
enters this decision.

Play-by-play carries a reproducibility caveat that leakage controls do not
cover: nflverse recomputes its EPA model from time to time, so historical
values can be restated. The pull date and the `nflreadpy` version are recorded
in the P3 artifact, and the cached pull under `backtest_cache/` is the object
the run is reproducible against.

## 4. Features

Eleven terms plus an intercept, inside §5's cap of twelve. Every one is
computed from an `AsOfView` and is therefore covered by the as-of audit.

### Specification A

| # | Feature | Definition | Expected sign |
| ---: | --- | --- | :---: |
| — | intercept | home-field term, refit per fold | + |
| 1 | `elo_diff` | Elo rating difference, K=20, 1/3 between-season regression, home advantage 65 | + |
| 2 | `rest_diff` | home rest days − away rest days | + |
| 3 | `short_week_diff` | away on ≤4 days − home on ≤4 days | + |
| 4 | `bye_diff` | home off a bye − away off a bye, from the published schedule | + |
| 5 | `neutral_site` | scheduled at a neutral site | 0 |
| 6 | `dome` | roof dome or closed | 0 |
| 7 | `div_game` | divisional matchup | 0 |
| 8 | `crowd_absent` | the 2020 season | 0 |

### Specification B adds

| # | Feature | Definition | Expected sign |
| ---: | --- | --- | :---: |
| 9 | `epa_off_diff` | opponent-adjusted offensive EPA per play, difference | + |
| 10 | `epa_def_diff` | opponent-adjusted defensive EPA per play allowed, difference | + |
| 11 | `success_rate_diff` | opponent-adjusted success rate, difference | + |

All three use pass and run plays only, an eight-game rolling window per team,
and shrinkage toward the league mean by n/(n+4).

**The opponent adjustment** is one ridge least-squares fit per as-of view over
the rolling window's team-games:

```text
offensive EPA per play ~ off_effect[team] + def_effect[opponent] + home
```

with ridge 1.0 on the effects and none on the intercept, solved by
`pipeline.team_outcome_fit.fit_linear`. `epa_off_diff` is the difference in
`off_effect`, `epa_def_diff` the difference in `def_effect` with the sign
oriented so that a better defence raises the home team's edge, and
`success_rate_diff` the same fit run on the success indicator.

### Deliberate exclusions

- **`qb_continuity`** — excluded. The only honest pregame version would need to
  know who starts *this* week, and every nflverse source for that is restated
  current state (§4.3 of the plan). Which quarterback started an earlier game
  is knowable; which one starts the next is not, and a feature that quietly
  uses the latter is the exact failure this project is built to avoid.
- **`travel_tz`** — excluded. It needs stadium coordinates from outside the
  corpus, and the motivation is weaker than the data dependency is expensive.
- **`plays_per_game_diff`** — excluded to stay inside the twelve-term cap. Pace
  is largely a consequence of game state, so it is the weakest of the
  play-by-play candidates.
- **`prior_margin_diff`** — excluded from the primary. It is collinear with Elo
  by construction. It remains available as a challenger under §5.
- **`temp`, `wind`** — excluded. nflverse records observed weather, not a
  pregame forecast.
- **Every market field** — excluded by construction, not by choice. Closing
  lines live in a separate frame and are a benchmark only.

### Recorded expectation

Elo and the play-by-play efficiency terms measure much the same thing, and the
efficiency terms are expected to add little once Elo is present. This is
written down before the run so that a confirmation is not mistaken for a
discovery, and so that a large gain is recognised as the surprise it would be.

Fitted signs are a **diagnostic, not a constraint**. No sign is imposed on the
fit. A fitted sign that contradicts the table above is reported in the P3
artifact and is not corrected.

## 5. Model family, grid and calibration

**Primary.** Ridge linear regression on the margin target, and ridge logistic
regression on the win target, both standardized, both fitted by
`pipeline.team_outcome_fit`. The reported probability is the one derived from
the **margin** model through the fold-fitted normal link. The direct logistic
model is reported beside it.

**Ridge penalty grid** — `{0.01, 0.1, 1.0, 10.0, 100.0}`, selected inside the
training window by leave-last-season-out on log loss for the win model and RMSE
for the margin model. The selected value per fold is recorded.

**Challengers**, each run once and reported whatever the result:

1. *Nonlinearity*: the primary plus a quadratic term in `elo_diff`.
2. *Redundancy*: the primary plus `prior_margin_diff`.
3. *Specification*: A run alongside B, so the play-by-play contribution is
   visible as an ablation rather than assumed.

Gradient boosting is **not run**, departing from §7 of the plan. The repository
ships numpy, pandas and nflreadpy and nothing else, and adding a tree library
for a challenger the plan already expects to lose is not a trade worth making.
This is recorded as a deviation rather than quietly dropped.

**Calibration.** None post-hoc on the primary. Both models are fitted with a
proper scoring rule, and a correction fitted after the fact is a degree of
freedom this document exists to remove. The margin model's residual sd is part
of the model, fitted per fold, and is not a post-hoc calibration.

**Home-field non-stationarity.** Handled by the expanding-window refit, which
already tracks the drift with a lag, plus the `crowd_absent` control that keeps
2020 from dragging the intercept for later seasons. No recency weighting, no
decay parameter, no rolling window: each is a knob, and P1 showed the drift is
gradual enough that the refit absorbs it.

## 6. Validation

Expanding window, season granularity, refit at season boundaries only, exactly
as P1 ran it. Minimum eight training seasons. Evaluated span **2007-2023**,
4,594 games. Sealed seasons **2024 and 2025**, read once at P4.

| Decision | May be selected on |
| --- | --- |
| Ridge penalty | inner leave-last-season-out inside the training window |
| Opponent-adjustment ridge, rolling window, shrinkage | fixed here; not selected at all |
| Elo constants | fixed here; not selected at all |
| Feature list, model family, link | fixed here; not selected at all |
| Anything whatsoever | **never the sealed seasons** |

## 7. Primary metric, gates and the decision rule

**Primary metric**: Brier score on the win target, pooled over the walk-forward
span. Log loss is reported alongside and is the tie-breaker in G3 only if the
Brier comparison's interval straddles zero.

| Gate | Requirement |
| --- | --- |
| G1 Coverage | ≥ 3,000 evaluated games across ≥ 10 walk-forward seasons |
| G2 Calibration | ECE ≤ 0.025 over 10 equal-count bins, and the Cox slope's 95% interval contains 1 |
| G3 Skill | Brier below the fitted-Elo baseline with the season-block bootstrap's 95% upper bound below 0 |
| G4 Stability | lower Brier than fitted Elo in ≥ 13 of the 17 walk-forward seasons |
| G5 Seal | sealed-season Brier inside the walk-forward bootstrap interval; one read |
| G6 Market context | gap to the no-vig market on 2010+ priced games, both de-vig methods; **no threshold** |
| G7 Edge ablation | corpus B only; ≥ 2,000 priced games and a bootstrap interval excluding 0 |
| G8 Leakage | as-of audit clean, canary detected, every feature timestamped |

**The decision rule, fixed now:**

```text
approved_model = G1 and G2 and G3 and G4 and G5 and G8
approved_edge  = G7                     (a separate flag, evaluated at P5)
```

G6 gates nothing. Beating the closing market is not the goal and, at 4,594
games against a measured 0.0141 gap, is not a claim this project could support.

`approved_edge` is deliberately **not** a term in `approved_model`. The plan
left that ambiguous; the fixed-core edge is a corpus-B question about one
feature, and the fundamentals model does not depend on its answer.

G4's threshold of 13 of 17 is the plan's "8 of 10" rescaled to the actual
walk-forward span, rounded up.

## 8. Stopping rule

The sealed seasons are scored **once**. If the gates fail, the P4 artifact
records `approved: false` with every gate's measured value, and the model is
not promoted. A second attempt requires a season that has never been read —
2026, once complete — and a new preregistration hash. Re-reading 2024-2025 with
an adjusted rule is not available at any point, for any reason.

`model/component_priors.json` is the precedent: it shipped with four
`approved: false` entries and an empty priors list rather than a lowered bar.

## 9. What would count as this preregistration failing

Recorded now so that none of it can be reframed later as a finding:

- A preregistered feature turns out not to be computable as-of. The run
  proceeds without it and the artifact records the removal and the reason; it
  does not get replaced with a substitute chosen after the fact.
- The play-by-play as-of audit reports findings. Specification A becomes
  primary, per §3.
- The fitted signs contradict §4 broadly. Reported, not corrected.
- Gates fail. Recorded as `approved: false`, per §8.

## Amendments

An amendment before the P4 sealed read is permitted only as a new version of
this document with a new sha256, with the previous hash and the reason retained
in `model/team_outcome_prereg.json` under `amendments`. After the sealed read,
no amendment is possible: the seal is spent, and a changed specification needs
a new sealed season.

| Version | Frozen | sha256 | Reason |
| ---: | --- | --- | --- |
| 1 | see `model/team_outcome_prereg.json` | recorded there | initial freeze |
