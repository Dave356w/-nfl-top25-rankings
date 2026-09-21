# Yahoo NFL Showdown field-model backtest

This report covers the **field/ownership layer only**. Yahoo's publicly indexed completed-contest pages expose field-wide roster percentages for the displayed lineup and leaderboard tie counts, but not a downloadable historical full-field lineup export. Because of that, this report does **not** claim a historical ROI backtest.

## Archived sample

The checked-in archive contains **35 player ownership observations from 7 completed Yahoo NFL single-game contests**. The contests span fields from 10 to 22,341 entries and include free, micro-stakes, and higher-entry-fee contests. These are the marginal ownership rates for players appearing in the displayed entry on each archived contest page, so the training sample is selection-biased and is not a census of all slate players.

The ownership prior is a ridge-logit model using projected/FPPG strength, position, field size, and entry fee. In live use, predicted marginal ownership is shifted so the player probabilities sum to exactly **5.0 roster slots**, matching Yahoo's five-player single-game lineup. Historical pages expose Yahoo FPPG, while live inference uses the model's Projected FP as the nearest pregame strength proxy; that transfer is approximate.

## Leave-one-contest-out results

| Held-out contest | Observations | MAE | RMSE |
|---|---:|---:|---:|
| 11772338 | 5 | 25.5 pp | 29.0 pp |
| 12786466 | 5 | 26.7 pp | 32.9 pp |
| 13545843 | 5 | 13.1 pp | 16.3 pp |
| 13548081 | 5 | 15.8 pp | 18.4 pp |
| 14512174 | 5 | 14.2 pp | 18.2 pp |
| 15236154 | 5 | 11.4 pp | 14.1 pp |
| 15747948 | 5 | 18.8 pp | 21.3 pp |
| **Overall** | **35** | **17.9 pp** | **22.4 pp** |

This is deliberately treated as a **sparse field prior**, not a precise ownership forecast. The largest error comes from small/older contests where roster construction and player context differ sharply from the rest of the archive.

## Duplication evidence

Completed Yahoo leaderboards in the archive show first-place ties of **18**, **4**, **2**, **1**, and **1** entries in contests where the leaderboard tie count was visible and unambiguous. That is enough to justify modeling duplicate lineups and prize splitting; it is not enough to estimate a universal duplication rate.

## What is and is not validated

The archive directly validates the roster-ownership prior out of contest and documents that first-place duplication occurs. Unit tests separately validate the payout engine, including tied-rank prize splitting and expected-duplicate accounting.

A true historical **contest ROI** validation would require the complete historical field (or a Yahoo endpoint that returns every submitted lineup plus the payout curve) for each contest. Until that data is available, the browser labels Tournament EV as a modeled estimate and reports the ownership archive size and opponent-field sampling details.

Regenerate the machine-readable result with:

```bash
python tools/backtest_yahoo_showdown_field.py
```
