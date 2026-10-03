"""Small ridge fitters for the team-outcome baselines and candidates.

Phase P1 of docs/team-outcome-model-plan.md. Deliberately two functions and no
dependency beyond numpy: the plan's primary family is a ridge logistic for the
win target and a ridge linear model for margin, and at this signal-to-noise
ratio that is the appropriate model rather than a concession.

Both standardize inside the fit and leave the intercept unpenalized, matching
the artifact shape already used by pipeline/salary_projection.py, so a fitted
model is a JSON-serializable dict rather than a pickled object.
"""
from __future__ import annotations

import numpy as np


def _standardize(x: np.ndarray):
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1.0
    return center, scale


def fit_linear(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6) -> dict:
    x = np.asarray(x, float).reshape(len(y), -1)
    center, scale = _standardize(x)
    z = np.column_stack([np.ones(len(x)), (x - center) / scale])
    penalty = np.eye(z.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(z.T @ z + penalty, z.T @ np.asarray(y, float))
    return {"kind": "linear", "center": center.tolist(), "scale": scale.tolist(),
            "intercept": float(beta[0]), "coefficients": beta[1:].tolist()}


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = 1e-6,
                 iterations: int = 100, tolerance: float = 1e-10) -> dict:
    """IRLS. A fractional y is valid here and is how a tie enters the fit."""
    x = np.asarray(x, float).reshape(len(y), -1)
    y = np.asarray(y, float)
    center, scale = _standardize(x)
    z = np.column_stack([np.ones(len(x)), (x - center) / scale])
    penalty = np.eye(z.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    beta = np.zeros(z.shape[1])
    for _ in range(iterations):
        eta = z @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        weight = np.maximum(mu * (1.0 - mu), 1e-9)
        gradient = z.T @ (y - mu) - penalty @ beta
        hessian = (z.T * weight) @ z + penalty
        step = np.linalg.solve(hessian, gradient)
        beta = beta + step
        if np.max(np.abs(step)) < tolerance:
            break
    return {"kind": "logistic", "center": center.tolist(), "scale": scale.tolist(),
            "intercept": float(beta[0]), "coefficients": beta[1:].tolist()}


def predict(model: dict, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float).reshape(len(x), -1)
    z = (x - np.asarray(model["center"], float)) / np.asarray(model["scale"], float)
    eta = float(model["intercept"]) + z @ np.asarray(model["coefficients"], float)
    if model["kind"] == "logistic":
        return 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
    return eta
