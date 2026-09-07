"""Runtime confidence audit for weekly market-implied NFL projections.

The sportsbook engine is intentionally left untouched: it still de-vigs the
feeds, fits the stat distributions and produces the raw market mean. This layer
sits *after* that work and asks a different question -- how much of the market
mean should the weekly lineup optimizer believe when it strongly disagrees with
its independent Yahoo/salary prior?

The first calibration is deliberately conservative and transparent. A normal
``good`` projection remains 100% market. Moderate prior disagreement (at least
25% *and* 2.5 fantasy points) caps market weight at 80%; extreme disagreement
(at least 40% or 5 fantasy points) caps it at 65%. A one-provider ``good`` mean
is capped at 90% even without a prior outlier. Existing ``fair`` and
``td-estimate`` quality weights remain the floor of confidence, so this layer
can only preserve or reduce market weight -- never increase it.

These constants are calibration parameters, not permanent truths. The published
audit fields make it possible to backtest realized error by flag/weight and fit
them from archived 2026 slates later.
"""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd


QUALITY_WEIGHT = {
    "good": 1.00,
    "fair": 0.75,
    "td-estimate": 0.35,
}
DEFAULT_QUALITY_WEIGHT = 0.50

# Moderate disagreement must clear both gates. This avoids flagging a 30% move
# on a tiny projection or a 3-point move on an elite QB where the relative gap
# is ordinary.
OUTLIER_MIN_FP = 2.50
OUTLIER_MIN_PCT = 0.25
OUTLIER_WEIGHT_CAP = 0.80

# Extreme disagreement clears either gate. At that point the prior is useful as
# a stabilizer even when the market construction received a ``good`` label.
EXTREME_MIN_FP = 5.00
EXTREME_MIN_PCT = 0.40
EXTREME_WEIGHT_CAP = 0.65

# ``good`` describes component coverage, not independent-provider agreement.
# A complete projection built from one surviving feed therefore keeps a small
# prior share until provider-level calibration says otherwise.
SINGLE_FEED_GOOD_WEIGHT_CAP = 0.90

AUDIT_COLUMNS = [
    "Market_Raw_FP",
    "Market_Prior_FP",
    "Market_Delta_FP",
    "Market_Delta_Pct",
    "Market_Base_Weight",
    "Market_Weight",
    "Market_Audit_Flag",
    "Market_Audit_Reason",
    "Market_Feed_Count",
]


def _finite(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _quality_weight(quality: object) -> float:
    return float(QUALITY_WEIGHT.get(str(quality), DEFAULT_QUALITY_WEIGHT))


def _feed_count(source: object) -> int:
    """Count provider names in the human-readable projection source label."""
    text = "" if source is None else str(source)
    if ":" in text:
        text = text.split(":", 1)[1]
    providers = {
        token.strip().casefold()
        for token in text.replace(",", "+").split("+")
        if token.strip().casefold() in {"bovada", "underdog"}
    }
    return len(providers)


def _raw_market_mean(row: pd.Series, prior: float, base_weight: float) -> float | None:
    """Recover the pre-blend market mean from the shared pipeline output."""
    explicit = _finite(row.get("Market_Raw_FP"))
    if explicit is not None:
        return explicit
    blended = _finite(row.get("Projected_FP"))
    if blended is None or base_weight <= 0:
        return None
    return (blended - (1.0 - base_weight) * prior) / base_weight


def _audit_decision(
    quality: str,
    raw_market: float,
    prior: float,
    base_weight: float,
    feed_count: int,
) -> tuple[float, str, str, float, float]:
    delta = raw_market - prior
    delta_pct = delta / max(abs(prior), 1.0)
    abs_delta, abs_pct = abs(delta), abs(delta_pct)

    final_weight = base_weight
    flags: list[str] = []
    reasons: list[str] = []

    if quality == "good" and feed_count == 1:
        new_weight = min(final_weight, SINGLE_FEED_GOOD_WEIGHT_CAP)
        if new_weight < final_weight:
            final_weight = new_weight
            flags.append("single-feed")
            reasons.append("good projection has one market provider")

    if abs_delta >= EXTREME_MIN_FP or abs_pct >= EXTREME_MIN_PCT:
        new_weight = min(final_weight, EXTREME_WEIGHT_CAP)
        if new_weight < final_weight:
            final_weight = new_weight
        flags.append("extreme-prior-gap")
        reasons.append(
            f"market/prior gap {delta:+.2f} FP ({delta_pct:+.1%}) exceeds extreme gate"
        )
    elif abs_delta >= OUTLIER_MIN_FP and abs_pct >= OUTLIER_MIN_PCT:
        new_weight = min(final_weight, OUTLIER_WEIGHT_CAP)
        if new_weight < final_weight:
            final_weight = new_weight
        flags.append("prior-gap")
        reasons.append(
            f"market/prior gap {delta:+.2f} FP ({delta_pct:+.1%}) exceeds audit gate"
        )

    if not flags:
        return final_weight, "pass", "within calibration gates", delta, delta_pct
    return final_weight, "+".join(flags), "; ".join(reasons), delta, delta_pct


def apply_projection_audit(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply confidence shrinkage to market-projected rows and annotate them.

    The input is the Yahoo frame *after* ``apply_market_projections``. Manual,
    unmatched and kicker rows have no market quality/prior pair and pass through
    unchanged. The raw market mean and displaced prior are both retained so the
    page can explain every shrink decision.
    """
    out = frame.copy()
    for column in AUDIT_COLUMNS:
        if column not in out:
            out[column] = np.nan if column not in {"Market_Audit_Flag", "Market_Audit_Reason"} else None

    for idx, row in out.iterrows():
        quality_obj = row.get("Market_Quality")
        if pd.isna(quality_obj):
            continue
        quality = str(quality_obj)
        prior = _finite(row.get("Fallback_Projected_FP"))
        if prior is None or prior <= 0:
            continue
        base_weight = _finite(row.get("Market_Weight"))
        if base_weight is None:
            base_weight = _quality_weight(quality)
        base_weight = min(1.0, max(0.0, base_weight))
        raw_market = _raw_market_mean(row, prior, base_weight)
        if raw_market is None or raw_market <= 0:
            continue

        feed_count = _feed_count(row.get("Projection_Source"))
        final_weight, flag, reason, delta, delta_pct = _audit_decision(
            quality, raw_market, prior, base_weight, feed_count
        )
        final_mean = final_weight * raw_market + (1.0 - final_weight) * prior

        out.at[idx, "Market_Raw_FP"] = raw_market
        out.at[idx, "Market_Prior_FP"] = prior
        out.at[idx, "Market_Delta_FP"] = delta
        out.at[idx, "Market_Delta_Pct"] = delta_pct
        out.at[idx, "Market_Base_Weight"] = base_weight
        out.at[idx, "Market_Weight"] = final_weight
        out.at[idx, "Market_Audit_Flag"] = flag
        out.at[idx, "Market_Audit_Reason"] = reason
        out.at[idx, "Market_Feed_Count"] = feed_count
        out.at[idx, "Projected_FP"] = final_mean
        if "FP" in out.columns:
            out.at[idx, "FP"] = final_mean

        if final_weight < base_weight - 1e-12:
            source = str(row.get("Projection_Source") or "market")
            providers = source.split(":", 1)[1].strip() if ":" in source else "market"
            out.at[idx, "Projection_Source"] = (
                f"market {quality} {final_weight:.0%} audit: {providers}"
            )

    return out


def attach_audit_columns(frame: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Copy audit metadata onto a resolved roster/pool by normalized player key."""
    if not len(frame) or "Key" not in frame or "Key" not in reference:
        return frame.copy()
    columns = [column for column in AUDIT_COLUMNS if column in reference]
    if not columns:
        return frame.copy()
    lookup = reference[["Key", *columns]].drop_duplicates("Key", keep="first")
    out = frame.drop(columns=[column for column in columns if column in frame], errors="ignore")
    return out.merge(lookup, on="Key", how="left")


def audit_summary(frame: pd.DataFrame) -> dict:
    flags = frame.get("Market_Audit_Flag")
    if flags is None:
        return {
            "audited": 0,
            "flagged": 0,
            "shrunk": 0,
            "extreme": 0,
            "rules": calibration_rules(),
        }
    audited = flags.notna()
    flagged = audited & flags.ne("pass")
    base = pd.to_numeric(frame.get("Market_Base_Weight"), errors="coerce")
    final = pd.to_numeric(frame.get("Market_Weight"), errors="coerce")
    shrunk = audited & final.lt(base - 1e-12)
    extreme = audited & flags.astype(str).str.contains("extreme-prior-gap", na=False)
    return {
        "audited": int(audited.sum()),
        "flagged": int(flagged.sum()),
        "shrunk": int(shrunk.sum()),
        "extreme": int(extreme.sum()),
        "rules": calibration_rules(),
    }


def calibration_rules() -> dict:
    return {
        "moderate": {
            "min_abs_fp": OUTLIER_MIN_FP,
            "min_abs_pct": OUTLIER_MIN_PCT,
            "weight_cap": OUTLIER_WEIGHT_CAP,
            "logic": "both",
        },
        "extreme": {
            "min_abs_fp": EXTREME_MIN_FP,
            "min_abs_pct": EXTREME_MIN_PCT,
            "weight_cap": EXTREME_WEIGHT_CAP,
            "logic": "either",
        },
        "single_feed_good_weight_cap": SINGLE_FEED_GOOD_WEIGHT_CAP,
    }
