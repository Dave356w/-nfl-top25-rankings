# NFL Top 25 by Position

A daily rebuild of the Yahoo NFL full-slate top-25 rankings for QB, RB, WR, TE
and DEF, published as a static web page. GitHub Actions runs the pipeline once
a day; GitHub Pages serves the result.

A second page, [My lineup](#weekly-lineup-optimizer), starts one team's own
roster each week from the same market-implied projections.

The pipeline is the Colab notebook `Yahoo_Top25_Position_Rankings_v1.ipynb`,
unchanged in substance — see [How this maps to the notebook](#how-this-maps-to-the-notebook).

## What it does

Each run:

1. Pulls the current Yahoo NFL player feed (salaries, FPPG, the slate schedule).
2. Builds market-implied projections from Underdog and Bovada player props —
   paired markets are de-vigged, yardage is fitted as Weibull and counts as
   Poisson, and the means are converted to Yahoo half-PPR scoring.
3. Cross-checks roles and availability against the published
   [nflverse](https://github.com/nflverse/nflverse-data) depth charts and weekly
   roster status, so injured-reserve and practice-squad players drop out and
   depth is read from a real depth chart rather than from salary order.
4. Applies the historical depth adjustment to fallback estimates only, then
   ranks every priced player by final projected fantasy points.
5. Writes JSON and CSV into `site/data/` and deploys `site/` to Pages.

Projection priority is: manual override → accepted market-implied mean → Yahoo
FPPG and salary prior. Only the fallback estimates carry the depth haircut.
Every row on the page shows which of the three produced it.

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

`.github/workflows/daily.yml` runs at **13:00 UTC** (09:00 ET) daily, and on
demand from the Actions tab. Change the `cron` line to move it. GitHub queues
scheduled runs under load, so the actual start time drifts by minutes.

`.github/workflows/lineup.yml` runs the lineup at **15:30 UTC Sunday** (11:30
ET, about 90 minutes before the early kickoffs) and **21:00 UTC Thursday**
(17:00 ET, three hours before TNF), and on demand with an objective and a bench
list as inputs.

Both workflows also rebuild on a push that touches their own files, so a UI
change reaches Pages without waiting for the next slate.

## Layout

```
pipeline/notebook.py        the notebook's code, one module, cell banners intact
run_daily.py                entry point: rankings -> site/data/*
lineup_optimizer.py         the weekly lineup tool (see below)
run_lineup.py               entry point: lineup -> site/data/lineup/*
lineup_roster.json          the team the lineup is picked from
site/index.html             the rankings page (no build step, no dependencies)
site/lineup.html            the lineup page
site/data/latest.json       what the rankings page reads
site/data/lineup/latest.json  what the lineup page reads
site/data/history/          one archived JSON + CSV per run date
tests/                      shape tests for both published payloads
```

Two pages, two workflows, one Pages site: `Daily rankings` publishes
`site/data/`, `Weekly lineup` publishes `site/data/lineup/`, and both deploy the
whole `site/` directory. They share one `concurrency` group so the two deploys
serialize instead of overwriting each other.

## Running it locally

```bash
pip install -r requirements.txt
python run_daily.py --top-n 25
python -m http.server -d site 8000    # then open http://localhost:8000
```

`--positions QB,RB,WR` limits what gets published. The run needs outbound
network access to Yahoo, the two sportsbooks and the nflverse release CDN.
`run_lineup.py` builds the lineup page the same way — see
[Running it](#running-it).

Tests do not need the network, and cover the optimizer as well as the
published JSON:

```bash
python -m unittest discover -s tests -v
```

## Failure behaviour

If any feed fails hard, `run_daily.py` (and `run_lineup.py`) writes **nothing**
and exits non-zero.
The previously published page stays live rather than being replaced by a
half-built one, and the failure shows up as a red run in the Actions tab.

Softer failures degrade instead of stopping: an unreachable sportsbook falls
back to Yahoo FPPG and salary priors, and an unreachable nflverse release falls
back to the salary-order depth heuristic. Both are recorded in the run log,
which the page renders under *Run details*.

## Weekly lineup optimizer

`lineup_optimizer.py` answers a different question from the rankings page: not
*who are the best 25 players at this position*, but *which of my players should
I start this week*. It has its own page at
[`/lineup.html`](https://dave356w.github.io/-nfl-top25-rankings/lineup.html) and
its own workflow, and it reuses the daily pipeline's market engine for the
numbers.

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

`lineup_roster.json` is the roster the lineup is picked from — edit that, not
the Python:

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

### Running it

```bash
pip install -r requirements.txt nflreadpy   # nflreadpy is optimizer-only
python run_lineup.py                        # publish to site/data/lineup/
python lineup_optimizer.py                  # or print a lineup in the terminal
python lineup_optimizer.py --self-test      # offline checks, no network
```

`run_lineup.py` is what the workflow runs: same optimizer, no prompt, and it
writes `site/data/lineup/latest.json`, a flat CSV, and one archived copy per run
date. Its flags are `--roster`, `--objective`, `--exclude`, `--no-market` and
`--out`. As with the daily build, a hard failure writes **nothing** and exits
non-zero, so the published lineup stays live rather than being replaced by a
half-built one. A sportsbook outage is not a hard failure: the market step
degrades to Yahoo priors and says so in the run log the page renders.

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

Two edits were made, both mechanical:

- the notebook's trailing `position_results = run_position_rankings(top_n=25)`
  is dropped, so importing the module defines functions without running a slate;
- the `google.colab.files.download` cell is dropped.

Everything else — the market engine, the correlation model, the showdown lineup
code — is carried over verbatim. The showdown/lineup half is unused by the daily
run but kept so the module stays a faithful copy of the notebook.

## Caveats

These are market-derived estimates, not predictions with a guarantee. The
Yahoo and sportsbook feeds are public but undocumented and can change shape
without notice. DEF has no dependable player-prop market and always uses the
Yahoo fallback, and neither does a kicker, who is estimated from rolling
nflverse game logs on the lineup page.
