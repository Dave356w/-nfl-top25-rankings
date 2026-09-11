"""Season-frozen Yahoo salary, position and depth fantasy-point model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "salary_projection.json"


def _depth(frame: pd.DataFrame) -> pd.Series:
    # Training and inference must use the same depth definition. Role_Tier is
    # useful display metadata, but the historical fit is based on Depth_Rank.
    source = frame.get("Depth_Rank", frame.get("Role_Tier", 1))
    return pd.to_numeric(source, errors="coerce").fillna(1).clip(1, 4).astype(int)


def feature_frame(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """Build the exact named design matrix recorded in a model artifact."""
    position = frame["Position"].astype(str).str.upper()
    salary = pd.to_numeric(frame["Salary"], errors="coerce").fillna(0) / float(
        spec.get("salary_scale", 10.0)
    )
    depth = _depth(frame)
    values = {}
    positions = tuple(spec["positions"])
    degree = int(spec["degree"])
    for pos in positions:
        flag = position.eq(pos).astype(float)
        values[f"position:{pos}"] = flag
        for power in range(1, degree + 1):
            values[f"position:{pos}:salary^{power}"] = flag * salary.pow(power)
        for bucket in range(2, 5):
            values[f"position:{pos}:depth:{bucket}"] = (
                flag * depth.eq(bucket).astype(float)
            )
            if spec.get("depth_salary_interactions"):
                values[f"position:{pos}:depth:{bucket}:salary"] = (
                    flag * depth.eq(bucket).astype(float) * salary
                )
    return pd.DataFrame(values, index=frame.index, dtype=float)


def fit(frame: pd.DataFrame, spec: dict, ridge: float) -> dict:
    features = feature_frame(frame, spec)
    center = features.mean()
    scale = features.std().replace(0, 1.0)
    x = ((features - center) / scale).to_numpy(float)
    x = np.column_stack([np.ones(len(x)), x])
    y = pd.to_numeric(frame["Actual_FP"], errors="raise").to_numpy(float)
    penalty = np.eye(x.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return {
        **spec,
        "ridge": float(ridge),
        "features": list(features.columns),
        "center": [float(center[name]) for name in features.columns],
        "scale": [float(scale[name]) for name in features.columns],
        "intercept": float(coefficients[0]),
        "coefficients": [float(value) for value in coefficients[1:]],
        "minimum_projection": 0.05,
    }


def predict(frame: pd.DataFrame, model: dict) -> np.ndarray:
    features = feature_frame(frame, model).reindex(columns=model["features"], fill_value=0)
    x = (features.to_numpy(float) - np.asarray(model["center"], float)) / np.asarray(
        model["scale"], float
    )
    estimate = float(model["intercept"]) + x @ np.asarray(model["coefficients"], float)
    return np.maximum(estimate, float(model.get("minimum_projection", 0.05)))


def load(path: str | Path = MODEL_PATH) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def apply(frame: pd.DataFrame, model: dict | None = None, overrides=None) -> pd.DataFrame:
    """Apply the frozen regression, preserving explicit user projections."""
    out = frame.copy()
    model = model or load()
    if "Projected_FP" not in out:
        out["Projected_FP"] = np.nan
    else:
        out["Projected_FP"] = pd.to_numeric(out["Projected_FP"], errors="coerce").astype(float)
    supported = (
        out["Position"].astype(str).str.upper().isin(model["positions"])
        & pd.to_numeric(out["Salary"], errors="coerce").gt(0)
    )
    out.loc[supported, "Projected_FP"] = predict(out.loc[supported], model)
    label = f"Yahoo salary-position-depth regression (trained through {model['trained_through_season']})"
    out.loc[supported, "Projection_Source"] = label
    overrides = overrides or {}
    for name, value in overrides.items():
        mask = out["Name"].eq(name)
        if mask.any():
            out.loc[mask, "Projected_FP"] = float(value)
            out.loc[mask, "Projection_Source"] = "manual override"
    projected = pd.to_numeric(out.loc[supported, "Projected_FP"], errors="coerce")
    out.loc[supported, "Projected_FP"] = projected.fillna(0.05).clip(lower=0.05)
    return out
