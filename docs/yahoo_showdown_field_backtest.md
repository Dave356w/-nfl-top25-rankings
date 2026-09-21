# Yahoo NFL Showdown field-model backtest

This report covers the **field/ownership layer only**. Yahoo's publicly indexed completed-contest pages expose field-wide roster percentages for the displayed lineup and leaderboard tie counts, but not a downloadable historical full-field lineup export. Because of that, this report does **not** claim a historical ROI backtest.

## Archived sample

The checked-in archive contains **25 player ownership observations from 5 completed Yahoo NFL single-game contests**. The contests span fields from 10 to 22,341 entries and include free, micro-stakes, and higher-entry-fee contests.

The ownership prior is a ridge-logit model using projected/FPPG strength, position, field size, and entry fee. In live use, predicted marginal ownership is shifted so the player probabilities sum to exactly **5.0 roster slots**, matching Yahoo's five-player single-game lineup.

## Leave-one-contest-out results

| Held-out contest | Observations | MAE | RMSE |
|---|---:|---:|---:|
| 11772338 | 5 | 34.6 pp | 37.8 pp |
| 12786466 | 5 | 25.5 pp | 31.2 pp |
| 13545843 | 5 | 17.9 pp | 23.4 pp |
| 15236154 | 5 | 12.8 pp | 17.7 pp |
| 15747948 | 5 | 18.0 pp | 22.0 pp |
| **Overall** | **25** | **21.8 pp** | **27.4 pp** |

This is deliberately treated as a **sparse field prior**, not a precise ownership forecast. The largest error comes from small/older contests where roster construction and player context differ sharply from the rest of the archive.

## Duplication evidence

Completed Yahoo leaderboards in the archive show first-place ties of **18**, **2**, **1**, and **1** entries in contests where the leaderboard tie count was visible and unambiguous. That is enough to justify modeling duplicate lineups and prize splitting; it is not enough to estimate a universal duplication rate.

## What is and is not validated

The archive directly validates the roster-ownership prior out of contest and documents that first-place duplication occurs. Unit tests separately validate the payout engine, including tied-rank prize splitting and expected-duplicate accounting.

A true historical **contest ROI** validation would require the complete historical field (or a Yahoo endpoint that returns every submitted lineup plus the payout curve) for each contest. Until that data is available, the browser labels Tournament EV as a modeled estimate and reports the ownership archive size and opponent-field sampling details.

Regenerate the machine-readable result with:

```bash
python tools/backtest_yahoo_showdown_field.py
```
