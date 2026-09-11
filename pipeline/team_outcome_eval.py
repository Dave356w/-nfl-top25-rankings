"""Scoring, de-vigging and interval estimates for team-outcome forecasts.

Phase P1 of docs/team-outcome-model-plan.md.

Two rules from the plan are enforced here rather than left to the caller. A
bare point difference is not a result, so every comparison goes through the
season-block bootstrap in ``compare``. And a closing line is de-vigged two
ways, because proportional normalization is biased under favourite-longshot
bias and a conclusion that depends on which method was used is not a
conclusion.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


EPSILON = 1e-12


def american_to_probability(odds) -> np.ndarray:
    """Quoted American odds to an implied probability, vig included.

    A real quote is at most -100 or at least +100; the band between them does
    not name a price. Anything inside it is returned as missing rather than
    converted, so a feed change cannot quietly price a 0 as a certainty.
    """
    values = pd.to_numeric(pd.Series(odds), errors="coerce").to_numpy(float)
    out = np.full(values.shape, np.nan)
    negative, positive = values <= -100.0, values >= 100.0
    out[negative] = -values[negative] / (-values[negative] + 100.0)
    out[positive] = 100.0 / (values[positive] + 100.0)
    return out


def devig_proportional(home_q: np.ndarray, away_q: np.ndarray) -> np.ndarray:
    total = home_q + away_q
    return np.where(total > 0, home_q / total, np.nan)


def devig_shin(home_q: np.ndarray, away_q: np.ndarray, iterations: int = 60) -> np.ndarray:
    """Shin's method: attribute the overround to informed money, not pro rata.

    Solves for the insider fraction z that makes the two adjusted probabilities
    sum to one, by bisection on [0, 0.5). Proportional de-vigging is the z = 0
    case, so the two agree on a book with no favourite-longshot skew and differ
    most where the plan expects them to: long prices.
    """
    home_q = np.asarray(home_q, float)
    away_q = np.asarray(away_q, float)
    total = home_q + away_q
    out = np.full(home_q.shape, np.nan)
    usable = np.isfinite(total) & (total > 0)
    if not usable.any():
        return out

    def adjusted(q, z, book):
        inner = np.sqrt(np.maximum(z * z + 4.0 * (1.0 - z) * q * q / book, 0.0))
        return (inner - z) / (2.0 * (1.0 - z))

    book = total[usable]
    home, away = home_q[usable], away_q[usable]
    low = np.zeros(book.shape)
    high = np.full(book.shape, 0.5 - 1e-9)
    for _ in range(iterations):
        mid = 0.5 * (low + high)
        summed = adjusted(home, mid, book) + adjusted(away, mid, book)
        # The adjusted pair is decreasing in z; too much mass means raise z.
        high = np.where(summed > 1.0, high, mid)
        low = np.where(summed > 1.0, mid, low)
    z = 0.5 * (low + high)
    out[usable] = adjusted(home, z, book)
    return out


def market_probabilities(market: pd.DataFrame) -> pd.DataFrame:
    """Home win probability from a closing moneyline, both de-vig methods."""
    home_q = american_to_probability(market.home_moneyline)
    away_q = american_to_probability(market.away_moneyline)
    frame = pd.DataFrame({
        "game_id": market.game_id.to_numpy(),
        "market_proportional": devig_proportional(home_q, away_q),
        "market_shin": devig_shin(home_q, away_q),
        "overround": home_q + away_q - 1.0,
    })
    return frame.loc[np.isfinite(frame.market_proportional)].reset_index(drop=True)


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    # Ties in the outcome have no side to rank, so they sit out the AUC only.
    decided = y != 0.5
    y, p = y[decided], p[decided]
    if len(np.unique(y)) < 2:
        return float("nan")
    order = np.argsort(p, kind="mergesort")
    ranks = np.empty(len(p), float)
    ranks[order] = np.arange(1, len(p) + 1)
    sorted_p = p[order]
    start = 0
    for index in range(1, len(sorted_p) + 1):  # average ranks within ties
        if index == len(sorted_p) or sorted_p[index] != sorted_p[start]:
            ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    positives, negatives = y == 1.0, y == 0.0
    n_pos, n_neg = int(positives.sum()), int(negatives.sum())
    return float((ranks[positives].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    clipped = np.clip(p, EPSILON, 1 - EPSILON)
    edges = np.quantile(clipped, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    index = np.clip(np.searchsorted(edges, clipped, side="right") - 1, 0, bins - 1)
    ece, curve = 0.0, []
    for b in range(bins):
        hit = index == b
        if not hit.any():
            continue
        predicted, observed = float(clipped[hit].mean()), float(y[hit].mean())
        ece += hit.mean() * abs(predicted - observed)
        curve.append({"bin": b, "n": int(hit.sum()),
                      "predicted": round(predicted, 4), "observed": round(observed, 4)})
    # Cox calibration slope: logistic regression of the outcome on the
    # forecast's own logit, where 1.0 means the forecast is neither over- nor
    # under-confident. It has to be a logistic fit — least squares on a 0/1
    # outcome returns the linear-probability slope, which is not 1.0 even for a
    # perfectly calibrated forecast and drifts with the spread of p.
    logit = np.log(clipped / (1 - clipped))
    if np.ptp(logit) > 0:
        from pipeline.team_outcome_fit import fit_logistic
        model = fit_logistic(logit.reshape(-1, 1), y)
        slope = float(model["coefficients"][0] / model["scale"][0])
    else:
        slope = float("nan")
    return {"ece": round(float(ece), 4), "slope": round(slope, 4), "curve": curve}


def metrics(frame: pd.DataFrame, probability: str, margin_hat: str | None = None,
            bins: int = 10) -> dict:
    """Score one forecast column. Ties count as half a win throughout."""
    y = frame.home_win.to_numpy(float)
    p = np.clip(frame[probability].to_numpy(float), EPSILON, 1 - EPSILON)
    out = {
        "n": int(len(frame)),
        "accuracy": round(accuracy(frame, probability), 4),
        "brier": round(brier(frame, probability), 4),
        "log_loss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
        "auc": round(_auc(y, p), 4),
    }
    out.update(_calibration(y, p, bins))
    if margin_hat and margin_hat in frame:
        error = frame[margin_hat].to_numpy(float) - frame.margin.to_numpy(float)
        out["margin_mae"] = round(float(np.mean(np.abs(error))), 3)
        out["margin_rmse"] = round(float(np.sqrt(np.mean(error ** 2))), 3)
        out["margin_residual_sd"] = round(float(np.std(error, ddof=1)), 3)
    return out


def brier(frame: pd.DataFrame, probability: str) -> float:
    """Just the Brier score.

    The bootstrap evaluates its statistic thousands of times, so the statistic
    must not drag the calibration fit and the AUC sort along behind it.
    """
    y = frame.home_win.to_numpy(float)
    p = np.clip(frame[probability].to_numpy(float), EPSILON, 1 - EPSILON)
    return float(np.mean((p - y) ** 2))


def accuracy(frame: pd.DataFrame, probability: str) -> float:
    """Pick accuracy, with a tie or a coin-flip forecast scoring half."""
    y = frame.home_win.to_numpy(float)
    p = frame[probability].to_numpy(float)
    picked = np.where(p > 0.5, 1.0, np.where(p < 0.5, 0.0, 0.5))
    exact = picked == y
    half = ~exact & ((picked == 0.5) | (y == 0.5))
    return float(exact.mean() + 0.5 * half.mean())


def brier_skill(frame: pd.DataFrame, probability: str, reference: str) -> float:
    y = frame.home_win.to_numpy(float)
    model = np.mean((np.clip(frame[probability].to_numpy(float), EPSILON, 1 - EPSILON) - y) ** 2)
    base = np.mean((np.clip(frame[reference].to_numpy(float), EPSILON, 1 - EPSILON) - y) ** 2)
    return float(1.0 - model / base) if base > 0 else float("nan")


def bootstrap(frame: pd.DataFrame, statistic, draws: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> dict:
    """Season-block bootstrap: resample seasons, then games inside them.

    Games in a season share a schedule, a rule set and a league-wide scoring
    environment, so resampling games alone would understate the spread.
    """
    rng = np.random.default_rng(seed)
    seasons = frame.season.unique()
    blocks = {s: frame.loc[frame.season.eq(s)] for s in seasons}
    values = []
    for _ in range(draws):
        picked = rng.choice(seasons, size=len(seasons), replace=True)
        parts = []
        for season in picked:
            block = blocks[season]
            parts.append(block.iloc[rng.integers(0, len(block), len(block))])
        value = statistic(pd.concat(parts, ignore_index=True))
        if np.isfinite(value):
            values.append(float(value))
    if not values:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan"), "draws": 0}
    return {
        "point": round(float(statistic(frame)), 4),
        "low": round(float(np.quantile(values, alpha / 2)), 4),
        "high": round(float(np.quantile(values, 1 - alpha / 2)), 4),
        "draws": len(values),
    }


def compare(frame: pd.DataFrame, probability: str, reference: str,
            draws: int = 2000, seed: int = 0) -> dict:
    """Paired difference with an interval, on one resample for both forecasts."""
    def brier_gap(block):
        y = block.home_win.to_numpy(float)
        a = np.clip(block[probability].to_numpy(float), EPSILON, 1 - EPSILON)
        b = np.clip(block[reference].to_numpy(float), EPSILON, 1 - EPSILON)
        return np.mean((a - y) ** 2) - np.mean((b - y) ** 2)
    result = bootstrap(frame, brier_gap, draws=draws, seed=seed)
    result["skill_vs_reference"] = round(brier_skill(frame, probability, reference), 4)
    result["reference"] = reference
    return result


def mcnemar_exact(frame: pd.DataFrame, probability: str, reference: str) -> dict:
    """Exact paired test on the picks, the test the 2025 re-analysis was missing."""
    from math import comb
    y = frame.home_win.to_numpy(float)
    decided = y != 0.5
    y = y[decided]
    a = (frame[probability].to_numpy(float)[decided] > 0.5).astype(float)
    b = (frame[reference].to_numpy(float)[decided] > 0.5).astype(float)
    a_only = int(((a == y) & (b != y)).sum())
    b_only = int(((b == y) & (a != y)).sum())
    n = a_only + b_only
    if n == 0:
        return {"discordant": 0, "model_only": 0, "reference_only": 0,
                "p_value": 1.0, "method": "none"}
    k = min(a_only, b_only)
    if n <= 1000:  # 2**n overflows a float well before the approximation bites
        p = min(1.0, 2.0 * sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n))
        method = "exact"
    else:
        from statistics import NormalDist
        z = (abs(a_only - b_only) - 1.0) / (n ** 0.5)  # continuity-corrected
        p = min(1.0, 2.0 * (1.0 - NormalDist().cdf(z)))
        method = "normal approximation"
    return {"discordant": n, "model_only": a_only, "reference_only": b_only,
            "p_value": round(float(p), 4), "method": method}
