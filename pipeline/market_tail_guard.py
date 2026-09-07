"""Guard sportsbook yardage fits against alternate-line tail explosions.

The market engine deliberately uses Bovada alternate ladders to learn a Weibull
shape around a de-vigged two-way main line.  An alternate ladder can occasionally
fit an extremely low shape parameter, which leaves the main-line survival
probability looking reasonable while putting implausibly large mass in the far
right tail.  Fantasy scoring consumes the *mean*, so that failure mode can add
several fantasy points even when the posted median has not moved.

This module keeps the original market parser and de-vigging intact.  It only
changes the yardage distribution fit when a receiving-yards market has at least
one genuine two-way anchor:

* the fitted Weibull shape may not fall below 1.40; and
* the fitted receiving-yard mean may not exceed 120% of the mean implied by the
  same main-line anchor(s) under the engine's default receiving shape.

The second rule is intentionally generous: alternate markets can contain real
information about skew, but they should not turn a roughly 50/50 main line into
an expected value nearly twice its level.  Alternate-only markets keep the
engine's historical 0.70 shape floor because there is no independent main-line
anchor to protect.

`install()` patches the generated notebook module at runtime.  Keeping the guard
separate avoids hand-editing a generated monolithic notebook while giving daily
rankings, showdown and the season-long lineup one shared implementation.
"""

from __future__ import annotations

import math
from typing import Sequence

BASE_SHAPE_FLOOR = 0.70
ANCHORED_RECEIVING_SHAPE_FLOOR = 1.40
ANCHORED_RECEIVING_MEAN_CAP = 1.20


def _valid_observations(nb, observations: Sequence):
    return [
        item
        for item in observations
        if item.threshold > 0
        and nb.EPSILON < item.probability < 1.0 - nb.EPSILON
    ]


def _fit_weibull(nb, observations: Sequence, default_shape: float, min_shape: float):
    """Notebook Weibull fit with an explicit lower shape bound."""
    valid = _valid_observations(nb, observations)
    if not valid:
        return nb.Distribution("weibull", 0.0, default_shape, nb.EPSILON)

    shape = float(default_shape)
    if len(valid) >= 2 and len({item.threshold for item in valid}) >= 2:
        x_values = [math.log(item.threshold) for item in valid]
        y_values = [math.log(-math.log(item.probability)) for item in valid]
        weights = [item.weight for item in valid]
        weight_sum = sum(weights)
        x_mean = sum(x * w for x, w in zip(x_values, weights)) / weight_sum
        y_mean = sum(y * w for y, w in zip(y_values, weights)) / weight_sum
        denominator = sum(
            w * (x - x_mean) ** 2 for x, w in zip(x_values, weights)
        )
        if denominator > 0:
            fitted_shape = sum(
                w * (x - x_mean) * (y - y_mean)
                for x, y, w in zip(x_values, y_values, weights)
            ) / denominator
            if math.isfinite(fitted_shape) and fitted_shape > 0:
                shape = min(8.0, max(float(min_shape), fitted_shape))

    weights = [item.weight for item in valid]
    weight_sum = sum(weights)
    intercept = sum(
        item.weight
        * (math.log(-math.log(item.probability)) - shape * math.log(item.threshold))
        for item in valid
    ) / weight_sum
    scale = math.exp(-intercept / shape)
    mean = scale * math.gamma(1.0 + 1.0 / shape)
    if not math.isfinite(mean) or mean < 0:
        mean = 0.0
    return nb.Distribution("weibull", mean, shape, scale)


def _fixed_shape_anchor_mean(nb, anchors: Sequence, shape: float) -> float | None:
    """Mean implied by main-line anchors while holding the default shape fixed."""
    valid = _valid_observations(nb, anchors)
    if not valid:
        return None
    weight_sum = sum(item.weight for item in valid)
    intercept = sum(
        item.weight
        * (math.log(-math.log(item.probability)) - shape * math.log(item.threshold))
        for item in valid
    ) / weight_sum
    scale = math.exp(-intercept / shape)
    mean = scale * math.gamma(1.0 + 1.0 / shape)
    return mean if math.isfinite(mean) and mean > 0 else None


def guarded_fit_stat_distribution(nb, stat, market, global_logit_vig):
    """Fit one stat, protecting anchored receiving yards from a runaway tail."""
    observations, source = nb.fair_observations(market, global_logit_vig)
    if stat in nb.YARD_STATS:
        default_shape = nb.DEFAULT_WEIBULL_SHAPES[stat]
        anchored_receiving = stat == "receiving_yards" and bool(market.anchors)
        min_shape = (
            ANCHORED_RECEIVING_SHAPE_FLOOR
            if anchored_receiving
            else BASE_SHAPE_FLOOR
        )
        distribution = _fit_weibull(
            nb, observations, default_shape=default_shape, min_shape=min_shape
        )

        if anchored_receiving:
            anchor_mean = _fixed_shape_anchor_mean(nb, market.anchors, default_shape)
            if anchor_mean is not None:
                maximum_mean = anchor_mean * ANCHORED_RECEIVING_MEAN_CAP
                if distribution.mean > maximum_mean:
                    # Preserve the fitted (already bounded) shape but rescale the
                    # curve so its expectation cannot run away from the main line.
                    scale = maximum_mean / math.gamma(
                        1.0 + 1.0 / distribution.parameter_1
                    )
                    distribution = nb.Distribution(
                        "weibull",
                        maximum_mean,
                        distribution.parameter_1,
                        scale,
                    )
                    source += ";tail-guard"
        return distribution, source

    if stat in nb.COUNT_STATS:
        return nb.fit_poisson(observations), source
    raise ValueError(f"Unsupported stat: {stat}")


def install(nb) -> bool:
    """Install the shared guard on a loaded `pipeline.notebook` module once."""
    if getattr(nb, "_market_tail_guard_installed", False):
        return False

    def _guarded(stat, market, global_logit_vig):
        return guarded_fit_stat_distribution(nb, stat, market, global_logit_vig)

    nb.fit_stat_distribution = _guarded
    nb._market_tail_guard_installed = True
    return True
