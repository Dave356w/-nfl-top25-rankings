# NFL Top 25 by Position

A daily rebuild of the Yahoo NFL full-slate top-25 rankings for QB, RB, WR, TE
and DEF, published as a static web page. GitHub Actions runs the pipeline once
a day; GitHub Pages serves the result.

Two more pages share the same projections: [My lineup](#weekly-lineup-optimizer)
starts one team's own roster each week and can be
[re-picked in the browser](#changing-the-roster-on-the-page) for a roster you
edit there, and the
[Showdown lineup lab](#showdown-lineup-lab) is an interactive single-game
optimizer that runs its Monte Carlo in your browser.

The pipeline started as the Colab notebook `Yahoo_Top25_Position_Rankings_v1.ipynb`
and has since diverged in the role and market layers — see
[How this maps to the notebook](#how-this-maps-to-the-notebook).

## What it does

Each run:

1. Pulls the current Yahoo NFL player feed (salaries, FPPG, the slate schedule).
2. Builds market-implied projections from Underdog and Bovada player props —
   paired markets are de-vigged, yardage is fitted as Weibull and counts as
   Poisson, and the means are converted to Yahoo half-PPR scoring.
3. Cross-checks roles and availability against the published
   [nflverse](https://github.com/nflverse/nflverse-data) depth charts and weekly
   roster status, so injured-reserve and practice-squad players drop out and
   role comes from a real depth chart rather than from salary order.
4. Applies the historical role adjustment to the prior share of each estimate,
   then ranks every priced player by final projected fantasy points.
5. Writes JSON and CSV into `site/data/` and deploys `site/` to Pages.

Projection priority is: manual override → market-implied mean → Yahoo FPPG and
salary prior. The market mean is blended in by how completely the props covered
the player rather than substituted outright, and only the prior share carries
the role haircut. Every row on the page shows which of the three produced it.

The role calculation is `w × market + (1 − w) × role multiplier × prior`,
using the market weight after confidence auditing. Coverage requires these core
components before a direct market total can be accepted:

| Position | Required components |
| --- | --- |
| QB | Passing yards, passing TDs, interceptions, rushing yards, own TDs |
| RB | Rushing yards, receiving yards, receptions, own TDs |
| WR / TE | Receiving yards, receptions, own TDs |

Missing core components retain the fallback and appear in the projection method.
A measured zero counts as present. Unknown positions are not rated complete.
Complete core coverage is `good` with a total anchor and `fair` without one;
these labels describe coverage, not measured forecast accuracy. Ancillary stats
such as WR rushing and lost fumbles contribute when supplied but are not required.
The separately labelled `td-estimate` regression remains a low-weight estimate.
Confidence weights and audit caps remain judgment calls pending historical testing.

### Role, depth and mean are three different things

A depth chart is not one ordered list per position. A team in three-receiver
personnel publishes three parallel starting spots, and nflverse encodes them in
`pos_slot`. Flattening that into WR1 → WR2 → WR3 → WR4 turns three starters into
a starter and two deep reserves, and the deep reserves then collect a mean
haircut meant for players who barely take the field.

So the pipeline keeps them apart:

| Column | What it is | What it drives |
| --- | --- | --- |
| `Role_Tier` / `Role_Label` | rank *within the player's own alignment slot*: starter, rotation, backup, reserve, specialist | the historical mean multiplier, the backup-QB filter |
| `Depth_Rank` | expected-opportunity rank, blending the chart's ordering with the current market projection | the fitted coefficients of variation and the pair correlations |
| `Projected_FP` | the fantasy mean itself | everything downstream |

The blend matters where a chart and a market disagree: a team can list one back
first and still say publicly that two of them will split the carries. The chart
keeps the role tier; a straight swap of two adjacent players is decided by the
priced expectation. `Settings.role_market_rank_weight` sets the balance — 0.0
trusts the chart alone, 1.0 ignores it.

## Setup (one time)

1. **Enable Pages.** Settings → Pages → *Build and deployment* → Source:
   **GitHub Actions**. This one is unavoidably manual: `configure-pages`
   supports an `enablement: true` flag that would do it, but `GITHUB_TOKEN`
   is refused the create call with *Resource not accessible by integration*,
   so the workflow cannot turn Pages on for itself. Until this is set, the
   build still runs and still commits data — it just fails at the deploy
   step and nothing is served.
2. **Allow Actions to write.** Settings → Actions → General → *Workflow
   permissions* → **Read and write permissions**. The daily job commits each
   run's data back to `main` so the archive accumulates.
3. **Run it once.** Actions → *Daily rankings* → *Run workflow*. Until the
   first run finishes the page shows a "waiting for the first run" state
   rather than stale or fabricated numbers.

The site then lives at <https://dave356w.github.io/-nfl-top25-rankings/>.

No secrets or API keys are needed. Every feed the pipeline reads is public.

## Schedule

`.github/workflows/daily.yml` refreshes all three pages at **13:00 and 21:00
UTC daily**, plus **15:30 UTC Sunday**. The afternoon refresh also covers
Wednesday, Monday and other evening games. GitHub can delay scheduled jobs;
these times are approximate. This workflow also runs on relevant code pushes.

`Weekly lineup` and `Showdown models` remain available for manual runs with
their respective inputs. Every workflow runs `run_synced.py`, preparing the
feeds once and publishing Rankings, My lineup and Showdown with one
`snapshot_id`. Scheduled refreshes are centralized to avoid duplicate runs.

## Layout

```
pipeline/notebook.py        the notebook's code, one module, cell banners intact
pipeline/showdown.py        exports one single-game model per game
run_daily.py                entry point: rankings -> site/data/*
lineup_optimizer.py         the weekly lineup tool (see below)
run_lineup.py               entry point: lineup -> site/data/lineup/*
run_synced.py               workflow entry point: one slate -> all three pages
run_showdown.py             entry point: showdown models -> site/data/showdown/*
lineup_roster.json          the team the lineup is picked from
site/index.html             the rankings page (no build step, no dependencies)
site/lineup.html            the lineup page
site/lineup-picker.js       the slot rules, re-run in the browser when you edit the roster
site/showdown.html          the showdown lab
site/showdown-worker.js     the optimizer that runs in the visitor's browser
site/data/latest.json       what the rankings page reads
site/data/lineup/latest.json  what the lineup page reads
site/data/lineup/pool.json    the players the page's roster editor can add
site/data/showdown/index.json one entry per game, plus one file per game
site/data/history/          one archived JSON + CSV per run date
tools/                      offline calibration scripts (see below)
tests/                      shape tests for every published payload
```

Three pages, three workflow entry points, one Pages site: all workflows publish
`site/data/latest.*`, `site/data/lineup/*` and `site/data/showdown/*` together.
All deploy the whole `site/` directory; deploys share one concurrency group.

## Running it locally

```bash
pip install -r requirements.txt nflreadpy
python run_synced.py --top-n 25
python -m http.server -d site 8000    # then open http://localhost:8000
```

`--positions QB,RB,WR` limits what gets published. The run needs outbound
network access to Yahoo, the two sportsbooks and the nflverse release CDN.
`run_lineup.py` builds the lineup page the same way — see
[Running it](#running-it) — and `run_showdown.py` builds the showdown models:

```bash
python run_showdown.py --max-games 1        # one game, for a quick look
python run_showdown.py --no-optimize        # models only, no reference run
```

Those standalone entry points are local diagnostics. Use `run_synced.py` for
publication; `--showdown-no-optimize` skips only the reference simulation while
still exporting shared models. If Yahoo supplies no usable single-game caps,
the synchronized run publishes an empty Showdown index and removes stale game
files without blocking the other pages.

Tests do not need the network, and cover both optimizers as well as the
published JSON:

```bash
python -m unittest discover -s tests -v
```

Node is optional but worth having: `tests/test_showdown.py` uses it to drive
`site/showdown-worker.js` and check the browser optimizer against the Python
pipeline. Without Node those tests skip.

## Failure behaviour

If any feed fails hard, `run_synced.py` writes neither page and exits non-zero.
The previously published page stays live rather than being replaced by a
half-built one, and the failure shows up as a red run in the Actions tab.

Softer failures degrade instead of stopping: an unreachable sportsbook falls
back to Yahoo FPPG and salary priors, and an unreachable nflverse release falls
back to the salary-order depth heuristic. Both are recorded in the run log,
which the page renders under *Run details*.

Availability is deliberately asymmetric. A player the weekly roster has no row
for is treated as **unavailable**, because an unknown status is exactly the case
where assuming "probably fine" eventually puts an ineligible player in a
submitted lineup; name anyone you know is playing in `AVAILABILITY_OVERRIDES`.
The one exception is a broken join: if the roster feed matches less of the pool
than `Settings.nflverse_min_match_rate`, the filter turns itself off with a
warning rather than dropping a slate on the strength of a name-matching bug.

## Showdown lineup lab

`site/showdown.html` is an interactive Yahoo single-game optimizer that runs on
GitHub Pages. Pages is static, so the interesting question was which half of the
Colab showdown runner can live in a browser. The answer turned out to be: all of
the part that matters.

**What runs in Actions.** `run_synced.py` fetches Yahoo, the two sportsbooks
and the nflverse depth chart once for all three pages. `pipeline/showdown.py`
reuses those final means, builds the fitted
coefficients of variation, and publishes the PSD-repaired latent correlation
matrix for each game. That payload is about 7 KB for a 36-player pool — 36
players plus the 630-term upper triangle — or 1 KB gzipped. A server-side
reference portfolio is published with it, so the page opens on an answer.

**What runs in the browser.** `site/showdown-worker.js` draws the scenarios,
enumerates every Yahoo-valid roster, screens them on the analytic ceiling,
scores the survivors against shared scenarios and applies the exposure caps.
Exclusions, the salary floor, position limits, the objective and the portfolio
size are all controls, and none of them touches a feed.

Changing games cancels a running calculation; late responses from a previous
selection are ignored. Changing a control clears the old result and asks for
Optimize again. The page rejects game files with a different snapshot from the
index. Entry numbers identify construction order, not performance ranks.
Expected FP is analytic; the sampled mean is shown separately. Candidate-best
and near-best rates compare only screened candidates in the sampled scenarios,
depend on the selected detail level, and are not contest-win probabilities.

The feeds are the reason for the split, not the compute. Yahoo and Bovada will
not answer a cross-origin request from a Pages origin, and pointing every
visitor's IP at a sportsbook is what got the pipeline's own VM blocked in the
first place.

Two properties of the model make the browser half cheap:

* Excluding a player is a principal submatrix of a PSD matrix, which is still
  PSD — so there is no eigensolver and no PSD repair in the JavaScript.
* Because of that the scenarios are never redrawn on an exclusion. They are
  drawn once per pool; a control change only re-decides which lineups are legal.

**Speed.** A lineup is five of at most 36 players, so scoring one against a
scenario is five multiply-adds rather than the thirty-six a dense
(players × candidates) product would do. On a full 36-player pool at 20,000
scenarios and 25,000 candidates the browser solves in about 14 seconds, against
25 seconds for the NumPy path — the JavaScript is faster because it exploits
that sparsity instead of handing a dense matrix to BLAS. The page also offers
smaller presets; at 5,000 scenarios it is well under a second.

**Why your numbers differ from the published run.** The page seeds its own
generator, so simulated means and percentiles land within Monte Carlo error of
the published ones rather than on top of them. The page also defaults to a
smaller detail setting than the published run; *Full* matches its scenario and
candidate counts. Everything analytic — salary,
expected points, the analytic standard deviation, the count of valid rosters —
matches exactly, and `tests/test_showdown.py` pins that agreement by driving the
worker through Node against the same payload.

Entries the page marks *also published* were selected by both runs. That overlap
is worth reading: at twenty entries the portfolio is genuinely seed-sensitive —
changing only the seed turns over about half of it — so an entry that survives
two independent scenario sets is one the ranking actually prefers rather than
one that won a near-tie on sampling noise.

## Weekly lineup optimizer

`lineup_optimizer.py` answers a different question from the rankings page: not
*who are the best 25 players at this position*, but *which of my players should
I start this week*. It has its own page at
[`/lineup.html`](https://dave356w.github.io/-nfl-top25-rankings/lineup.html) and
its own workflow. The workflow prepares the daily pipeline once and derives
both pages from that exact final player frame.

Each run:

1. Pulls the Yahoo DFS feed for the week's FPPG, salary, opponent and kickoff.
2. Replaces those blends with **de-vigged Bovada and Underdog means**, using
   `pipeline/notebook.py`'s market engine — the same de-vigging, Weibull and
   Poisson fits, and quality gate the rankings use. Matching is the pipeline's
   own: team, name key, and a hard kickoff-time guard, so a player never
   inherits a namesake's line from another game.
3. Cross-checks role and availability against the newest nflverse depth
   snapshot and the week's injury report.
4. Fills `QB / RB / RB / WR / WR / TE / K` plus one RB/WR/TE flex and publishes
   starters, bench and a review note per player.

Projection priority is: accepted market mean → Yahoo FPPG and salary prior →
rolling nflverse game logs. Kickers always use the game logs, because the DFS
feed does not price them, and so does a rostered player whose team is playing
but whom Yahoo omits — a Monday-only slate, say — instead of reading as a zero.
The page shows which of the three produced every row.

`Floor_P25` and `Ceiling_P90` come from a lognormal band around the mean, using
depth-calibrated coefficients of variation (`CALIBRATED_CV`). Depth widens the
band but never haircuts the mean a second time: the weekly salary and the
market line already reflect the player's current role.

### Setting your team

`lineup_roster.json` is the roster the lineup is picked from — edit that (or
paste in what the page's editor hands you), not the Python:

```json
{ "roster": [ { "Name": "Dak Prescott", "Position": "QB" } ] }
```

Positions are `QB`, `RB`, `WR`, `TE`, `K`. Yahoo's season-long roster API needs
OAuth, so the team is typed in rather than fetched. Everything else lives in the
constants at the top of `lineup_optimizer.py`:

| Setting | What it does |
| --- | --- |
| `STARTING_POSITIONS`, `FLEX_ELIGIBLE` | your league's slots |
| `LINEUP_OBJECTIVE` | `FP` (mean, the default), `Floor_P25`, or `Ceiling_P90` |
| `USE_MARKET_PROJECTIONS`, `MARKET_SOURCE` | market means on/off; `hybrid`, `bovada` or `underdog` |
| `EXCLUDED_PLAYERS` | `None` prompts for benchings; a list (even empty) skips the prompt |
| `MANUAL_DEPTH_OVERRIDES` | override a stale depth chart |
| `AUTO_EXCLUDE_REPORTED_OUT` | drop anyone the injury report lists as *Out* |
| `POOL_LIMITS` | how many players per position the page's roster editor can add from |

### Changing the roster on the page

The published lineup answers the question for one roster. **Edit roster** on the
page asks it again for a different one — a waiver add, a drop, a benching an
hour before kickoff — without waiting for the next scheduled run:

- **Add** anyone from `site/data/lineup/pool.json`, the week's best players at
  each slot, priced by the same run that published the lineup. `POOL_LIMITS` in
  `lineup_optimizer.py` sets how deep it goes.
- **Drop** a player, or **bench** one so the optimizer has to fill his slot from
  somewhere else. The run's own benchings — anyone the injury report lists as
  *Out* — start out applied, and can be undone.
- The lineup, the totals and the CSV button all switch to the edited roster, and
  a badge in the status bar says the lineup on screen is no longer the published
  one.

Re-picking happens in `site/lineup-picker.js`, which runs the same slot rules
`lineup_optimizer.optimize` runs. `tests/test_lineup_picker.py` drives that
JavaScript from Node against the Python optimizer on the same rows, so the two
cannot drift into disagreement.

Edits live in the browser's `localStorage`, not in the repo, so they change what
you see and not what the workflow builds. A roster change survives the next run;
a benching expires with the run it was made against, because it was a call about
one week's kickoff. **Copy roster JSON** hands back a `lineup_roster.json` —
commit that and the scheduled run picks from it too.

### Running it

```bash
pip install -r requirements.txt nflreadpy
python run_synced.py                        # publish rankings + lineup together
python run_lineup.py                        # standalone lineup-only diagnostic
python lineup_optimizer.py                  # or print a lineup in the terminal
python lineup_optimizer.py --self-test      # offline checks, no network
```

`run_synced.py` is what all three workflows run. It stamps
`data/latest.json`, `data/lineup/latest.json`, `data/lineup/pool.json`, the
Showdown index and every Showdown game with
one `snapshot_id`, verifies every overlapping player has the same projected FP,
then writes the files. `run_lineup.py` remains available for lineup-only local
diagnostics. A sportsbook outage is not a hard failure: the market step
degrades to Yahoo priors and says so in the run log the page renders. Nor is a
failed pool: the page loses its add list, says why, and still drops and benches.

Yahoo prices this feed for **DFS half-PPR**. Check a close call against your
league's scoring, and check the injury news yourself — the nflverse report is
only as current as its last publish.

## How this maps to the notebook

The notebook executed every cell into a single namespace, and its functions
reach across cell boundaries for underscore-prefixed helpers as well as public
ones. `pipeline/notebook.py` therefore keeps all of that code in one module in
the original order, with a banner marking each cell, rather than splitting it
into packages — a split would need an explicit export list per cell and would
break silently the first time a private helper moved.

Two mechanical edits were made when the notebook was imported:

- the notebook's trailing `position_results = run_position_rankings(top_n=25)`
  is dropped, so importing the module defines functions without running a slate;
- the `google.colab.files.download` cell is dropped.

The market engine and the showdown lineup code are otherwise carried over
verbatim. The synchronized daily publisher uses the showdown and lineup paths
alongside the rankings, keeping the correlation model in one place.

Four model changes have since been made here and are **not** in the notebook. To
keep a Colab copy in step, port these:

- `add_slot_role_tiers`, `role_label`, `apply_nflverse_roles` and
  `apply_opportunity_ranks` replace `apply_nflverse_depth`, and
  `apply_depth_mean_adjustments` now keys the multiplier on the role tier
  (cells 3 and 9);
- `SAME_TEAM_RB_RB_CORR` replaces the pooled `("RB", "RB")` entry in
  `SAME_TEAM_OTHER_CORR`, with `same_team_rb_rb_correlation` called from
  `target_score_correlation` (cell 11);
- `MARKET_QUALITY_WEIGHT` and `market_blend_weight` make
  `apply_market_projection_means` a blend rather than a substitution (cell 8);
- `Settings.nflverse_drop_unmatched` defaults to `True`, with
  `AVAILABILITY_OVERRIDES` and the `nflverse_min_match_rate` guard (cells 2 and 9);
- `simulate_player_outcomes` factors the latent matrix with `latent_root`
  (Cholesky) rather than an eigendecomposition. The correlation targets are
  looked up by (position, depth bucket, team), so a pool with several players
  sharing all three carries exactly degenerate eigenvalues — a typical showdown
  pool has a six-fold one — and any eigenvector basis of that eigenspace is a
  valid decomposition. The old root therefore depended on the LAPACK build and
  `random_seed` did not pin the scenario set; a Cholesky factor is unique, and
  it is the factorization `site/showdown-worker.js` already used (cell 11);
- `score_candidates_shared_scenarios` carries scores as (candidates × scenarios)
  against a pre-transposed outcomes array and accumulates the mean and standard
  deviation in float64 (cell 13). Same arithmetic, ~1.75x faster, and the
  reduction no longer depends on the array layout — which is what lets the
  browser optimizer reproduce these numbers.

`prepare_slate_pool` and `pipeline/showdown.py` are repo-only glue with no
notebook counterpart.

## Calibration

`tools/calibrate_rb_correlation.py` fits the same-team RB-RB entries from
nflverse weekly stats. It is offline work, run by hand, and the constants it
prints are pasted into `pipeline/notebook.py`; the daily build never runs it.

```bash
python tools/calibrate_rb_correlation.py --seasons 2016-2025 --emit-python
```

It reports each pair three ways: the marginal forecast-error correlation the
simulation model takes, the same correlation after projecting out the team's
non-RB production, and the correlation measured on carries + targets instead of
points. The third column is the interesting one. Two backs sharing a backfield
really are splitting a fixed pool of work — RB1/RB2 touch errors correlate
−0.082 — but touchdowns and long gains are noisy enough that only about −0.015
of that survives into fantasy scoring, and every 95% interval straddles zero.
Representing the cannibalization is the right thing to do; expecting it to
reshuffle a portfolio is not.

`tools/calibrate_game_correlations.py` does the same job for every *other* pair
in the showdown target matrix — every same-team and opposing offensive
relationship — which previously rested on asserted fits with no script behind
them.

```bash
python tools/calibrate_game_correlations.py --seasons 2016-2025
```

On 2016-2025, 51,874 player-games and 500,463 same-game pairs, the table holds
up. Same-team QB-WR measures **0.215** against a published 0.217; opposing QB-QB
measures 0.164 against 0.175; 14 of 20 relationships sit inside the 95% interval
of the measurement and no gap anywhere exceeds 0.033.

That answers a question the live pool invites. Mean absolute off-diagonal score
correlation on a published 32-player game is 0.023, and cross-player covariance
supplies only 10-15% of a lineup's variance, which looks at first like a model
treating one football game as thirty-two independent players. It is not a bug:
on the forecast-error scale the model works in, same-game fantasy errors really
are close to uncorrelated once each player's own expectation is subtracted.

Read the printed gaps as an **upper bound**. The expectation there is a lagged
eight-game rolling average, which does not know the game total, so a shootout is
a surprise to it and part of what it books as correlated error is really the
shared game environment. The pipeline's means are market-implied and already
price that environment, so the correlation left for its residuals is if anything
lower. A positive gap is therefore not a licence to raise a constant. Team
defenses are not covered: weekly player stats carry no DST scoring, so the DEF
entries still rest on their original fit.

## Caveats

These are market-derived estimates, not predictions with a guarantee. The
Yahoo and sportsbook feeds are public but undocumented and can change shape
without notice. DEF has no dependable player-prop market and always uses the
Yahoo fallback, and neither does a kicker, who is estimated from rolling
nflverse game logs on the lineup page.


### Showdown roster rules

Yahoo requires five players, the salary cap, and **at least one player from each
team**. Any position satisfies the team requirement, a team defense included —
the enumerator used to demand a non-DEF player from each side, which discarded
304 legal rosters on the 2026-09-10 SF-LA slate.

`Settings.min_salary_used_pct` is a strategy filter, not a Yahoo rule, and now
defaults to `0.0`. At the previous `0.75` it removed 161,439 of the 187,697
cap-legal rosters on that same slate — 86% of the legal space — before the model
scored one of them. The candidate screen already ranks on the analytic ceiling,
so it discards weak cheap rosters on their merits: with the floor off, only 449
of the 25,000 retained candidates are newly admitted, none of them inside the
top 1,000. The page keeps the slider for anyone who wants the filter back.

### Showdown selection controls

Exposure percentages round down to whole appearances among the requested entries:
70% across five entries allows three appearances. Single-entry selection bypasses
exposure caps and construction quotas. The page displays the permitted counts and
refuses to present an incomplete portfolio as a completed result; incomplete
published reference portfolios are hidden for the same reason.

Roster-shape quotas are optional. New builds default `use_construction_quotas` to
`False`; the page can enable the existing preset mix. These quotas can exclude
higher-ranked lineups and are not required for a legal roster.

Expected-points objectives screen by analytic mean. Floor and ceiling screen by
approximate P25 and P90 from a lognormal matched to lineup mean and variance,
then rank retained candidates using simulated quantiles. This is an approximation
to a sum of correlated lognormals; screening can still miss the global optimum.
The experimental upside score retains its existing screening and 40% P90 / 25%
near-best / 25% mean / 10% volatility weights. All screens reserve high-mean
candidates. Greedy selection enforces exposure and overlap limits; it does not
optimize joint scenario coverage. Compare detail settings to assess stability.


### Default projection and portfolio process

`Settings.showdown_objective = "auto"` is the default. A single entry selects the
highest analytic expectation among eligible candidates (the screen preserves the
highest expected-points candidate). Multiple entries start with that same anchor
and greedily maximize the average best score of the selected set over shared
scenarios. Exposure and overlap caps remain enforced. This mode requires
construction quotas off; the experimental upside objective still supports them.

The first half of the draws selects the portfolio. The other half reports its
average best score, P25 and gain over the single-lineup anchor. Changing evaluation
draws cannot change selection. This checks performance within the simulation;
it does not establish historical accuracy or a probability of winning a contest.
Candidate screening and constrained greedy selection can miss a global optimum.

The simulator is a correlated **zero-hurdle** lognormal. Each marginal is an atom
at zero of fitted size plus a lognormal above it, and the correlation structure is
still a Gaussian copula, so excluding a player is still a principal submatrix and
the browser still needs no eigensolver.

Explicit participation, negative defense scores and discrete touchdowns are still
not represented. Availability in particular is deliberately left out: the market
means already price it, so modelling it here would charge the same risk twice.

### Zero scores

A mean-preserving lognormal is strictly positive, so the old simulator gave every
player a floor. Measurement says that was right for starters and badly wrong for
everyone else. Over 2016-2025, the share of games landing below each player's
modelled P25 — 25% if the model were calibrated — ran:

| | below modelled P25 | zero games |
| --- | --- | --- |
| QB1 / RB1 / WR1 / TE1 | 23% / 25% / 28% / 28% | 1-6% |
| WR3 / RB3 | 32% / 43% | 14% / 19% |
| WR4 / RB4 / QB2 | 40% / 53% / 58% | 29% / 33% / 26% |

Ceilings were fine everywhere: 85-91% of games fell below the modelled P90.

`ZERO_RATE` therefore carries a fitted probability that a player who **took the
field** still finished on zero, and `hurdle_transform` puts an atom there. Writing
`p` for that rate and `c` for the CV of the scoring part,

```
mean = (1 - p) * (m / (1 - p)) = m          CV^2 = (c^2 + p) / (1 - p)
```

so holding `CALIBRATED_CV` fixed pins `c^2 = CV^2 (1 - p) - p`. **The published
mean and CV do not move.** Only the shape does, with mass taken out of the lower
body and placed on zero.

Scope is the argument. A game with no carry, target or pass attempt is dropped
from both numerator and denominator, because that is usually a player who was
inactive and the market mean already prices that. What is left is the pure usage
effect: a fourth receiver who dressed, ran his routes and was never thrown to.
`tools/calibrate_zero_rate.py` reproduces the table and prints both rates so the
gap is visible.

```bash
python tools/calibrate_zero_rate.py --seasons 2016-2025
```

An atom at zero breaks the closed form `score_to_lognormal_latent` used to hit a
requested correlation, so the inversion is numerical: Mehler's formula turns the
latent-to-score map into a polynomial in the latent correlation whose coefficients
depend only on `(CV, zero rate)`, and each pair is bisected against its own series.
A pool has about a dozen distinct pairs, so the coefficients are cached and the
whole inversion costs about 0.1s per game. With every rate set to zero it reduces
to the old closed form exactly, which the tests pin.

DEF is held at zero throughout: weekly player stats carry no team-defense scoring,
so nothing is fitted, and a defense can post a negative score, which this model
does not represent either.

### Historical component-prior gate

Run `python tools/build_component_priors.py --holdout 2025 --start 2021` to reproduce
`model/component_priors.json`. The script tunes recency weighting on earlier
seasons, then compares total Yahoo-scored point error on 2025 against a lagged
eight-appearance average. Approval requires at least 200 holdout observations and
at least 2% lower MAE. Results were QB 0.41%, RB 0.88%, WR 0.60%, TE 0.45% lower
MAE: **no position passed**. No component priors were promoted into live forecasts.
The existing market-coverage/fallback process therefore remains in production.

This is a conditional-on-appearance historical-prior comparison, not a test of
Yahoo salary priors, the market blend or participation probabilities. Archived
pregame data is required for those comparisons. Merely changing the approval flag
in the JSON does not enable a live model; integration requires a reviewed change.

### Pregame archive and evaluation

Every synchronized publish now retains a compressed immutable snapshot under
`site/data/projection_archive/`. Raw Yahoo and available market payloads are
content-addressed and deduplicated. Forecasts include identity, kickoff, capture
time, final mean, role-adjusted fallback baseline, marginal percentiles, scoring
settings, code hash, market component estimates, role reports and Showdown models.
Snapshots are captured conservatively after feed/role preparation; any captured
at or after kickoff is excluded from evaluation. Existing daily summaries cannot
reconstruct missing historical inputs; this archive starts with the new publisher.

Grade forecasts using:

```bash
python tools/evaluate_projection_archive.py \
  --actuals actual_results.csv \
  --output model/projection_evaluation.json
```

Actuals require `game_id`, `player_key`, `actual_fp`, `realized_at_utc`. Use the
archived keys (Yahoo IDs where present); name matching is not guessed. The grader
uses one latest pregame forecast per game/player, rejects duplicate outcomes,
excludes manual overrides, and never converts an unmatched actual into zero.
It reports MAE, RMSE, bias, paired fallback comparison, P25/P90 coverage and the
observed zero/negative-score rate. There are no graded market-blend results yet.

