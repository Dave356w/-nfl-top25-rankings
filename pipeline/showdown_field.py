"""Yahoo NFL single-game field/ownership model.

The archive is intentionally small and auditable: public completed-contest pages
expose field-wide roster percentages for the displayed lineup, not a downloadable
full field.  We fit a ridge-logit propensity model to those observations and
renormalize live predictions so marginal roster ownership sums to Yahoo's five
roster slots.  This is a field prior, not a claim that archived ownership is a
complete census.
"""
from __future__ import annotations

from functools import lru_cache
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ARCHIVE = Path(__file__).resolve().parents[1] / "model" / "yahoo_showdown_ownership_observations.json"
FEATURES = ("intercept", "log_fp", "QB", "RB", "TE", "DEF", "log_field_size", "log_entry_fee")
RIDGE = 2.0
LINEUP_SIZE = 5
SUPERSTAR_EXPONENT = 1.15
SUPERSTAR_POSITION_MULTIPLIER = {"QB": 1.08, "RB": 1.00, "WR": 1.00, "TE": 0.96, "DEF": 0.72}


@lru_cache(maxsize=1)
def load_archive(path: str | None = None) -> dict:
    target = Path(path) if path else ARCHIVE
    return json.loads(target.read_text())


def _source_map(archive: dict) -> dict[str, dict]:
    return {str(row["contest_id"]): row for row in archive.get("sources", [])}


def observations_frame(archive: dict | None = None) -> pd.DataFrame:
    archive = archive or load_archive()
    columns = archive["observation_columns"]
    frame = pd.DataFrame(archive["observations"], columns=columns)
    sources = _source_map(archive)
    frame["field_size"] = frame["contest_id"].map(lambda key: sources[str(key)]["entries"])
    frame["entry_fee"] = frame["contest_id"].map(lambda key: sources[str(key)]["entry_fee"])
    return frame


def _design(frame: pd.DataFrame, fp_column: str = "fppg") -> np.ndarray:
    fp = pd.to_numeric(frame[fp_column], errors="coerce").fillna(0).to_numpy(float)
    position = frame["position"].astype(str).str.upper()
    field_size = pd.to_numeric(frame.get("field_size", 1000), errors="coerce").fillna(1000).to_numpy(float)
    entry_fee = pd.to_numeric(frame.get("entry_fee", 1), errors="coerce").fillna(1).to_numpy(float)
    return np.column_stack([
        np.ones(len(frame)),
        np.log1p(np.maximum(fp, 0)),
        position.eq("QB").to_numpy(float),
        position.eq("RB").to_numpy(float),
        position.eq("TE").to_numpy(float),
        position.eq("DEF").to_numpy(float),
        np.log1p(np.maximum(field_size, 1)),
        np.log1p(np.maximum(entry_fee, 0)),
    ])


def _logit(probability) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=float), 0.01, 0.99)
    return np.log(p / (1.0 - p))


def _sigmoid(value) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-value))


def fit_ownership_model(frame: pd.DataFrame | None = None, ridge: float = RIDGE) -> dict:
    frame = observations_frame() if frame is None else frame.copy()
    X = _design(frame)
    y = _logit(pd.to_numeric(frame["rostered"], errors="coerce").fillna(0.01))
    penalty = np.eye(X.shape[1]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(X.T @ X + penalty, X.T @ y)
    fitted = _sigmoid(X @ coefficients)
    return {
        "family": "ridge_logit_roster_propensity",
        "features": list(FEATURES),
        "coefficients": [float(value) for value in coefficients],
        "ridge": float(ridge),
        "training_observations": int(len(frame)),
        "training_contests": int(frame["contest_id"].nunique()),
        "in_sample_mae": float(np.mean(np.abs(fitted - frame["rostered"].to_numpy(float)))),
    }


def _calibrate_sum(logits: np.ndarray, target: float) -> np.ndarray:
    if len(logits) == 0:
        return np.asarray([], dtype=float)
    target = min(max(float(target), 0.0), float(len(logits)))
    low, high = -20.0, 20.0
    for _ in range(80):
        mid = (low + high) / 2.0
        total = float(_sigmoid(logits + mid).sum())
        if total < target:
            low = mid
        else:
            high = mid
    return _sigmoid(logits + (low + high) / 2.0)


def predict_roster_ownership(players: pd.DataFrame, field_size: int = 1000,
                             entry_fee: float = 1.0, model: dict | None = None,
                             lineup_size: int = LINEUP_SIZE) -> np.ndarray:
    model = model or fit_ownership_model()
    frame = pd.DataFrame({
        "fppg": pd.to_numeric(
            players["Projected_FP"] if "Projected_FP" in players else players.get("FPPG", 0),
            errors="coerce",
        ).fillna(0),
        "position": players["Position"].astype(str),
        "field_size": int(field_size),
        "entry_fee": float(entry_fee),
    })
    logits = _design(frame) @ np.asarray(model["coefficients"], dtype=float)
    return _calibrate_sum(logits, lineup_size)


def predict_superstar_ownership(players: pd.DataFrame, roster_ownership: np.ndarray) -> np.ndarray:
    fp = pd.to_numeric(
        players["Projected_FP"] if "Projected_FP" in players else players.get("FPPG", 0),
        errors="coerce",
    ).fillna(0).to_numpy(float)
    pos = players["Position"].astype(str).str.upper().to_numpy()
    multipliers = np.asarray([SUPERSTAR_POSITION_MULTIPLIER.get(value, 1.0) for value in pos])
    weights = np.asarray(roster_ownership, dtype=float) * np.power(np.maximum(fp, 0.25), SUPERSTAR_EXPONENT) * multipliers
    total = float(weights.sum())
    return weights / total if total > 0 else np.repeat(1.0 / len(weights), len(weights))


def export_field_model() -> dict:
    archive = load_archive()
    fitted = fit_ownership_model(observations_frame(archive))
    ties = [row.get("first_place_ties") for row in archive.get("sources", []) if row.get("first_place_ties") is not None]
    return {
        "ownership": fitted,
        "superstar": {
            "formula": "roster_ownership * projected_fp^exponent * position_multiplier",
            "exponent": SUPERSTAR_EXPONENT,
            "position_multiplier": SUPERSTAR_POSITION_MULTIPLIER,
        },
        "archive": {
            "observations": len(archive.get("observations", [])),
            "contests": len(archive.get("sources", [])),
            "sources": archive.get("sources", []),
            "observed_first_place_ties": ties,
        },
        "limitations": [
            "Archived Yahoo pages expose roster percentages only for players in the displayed entry; this is a selection-biased sample, not a census of every slate player or full historical field.",
            "The historical strength feature is Yahoo FPPG; live inference uses Projected_FP as the closest pregame strength proxy, so this transfer is approximate.",
            "The ownership model is a sparse prior. Live rates are normalized to five roster slots and should be overridden by better contest-specific ownership when available.",
            "Superstar ownership is derived from roster ownership and projected scoring strength because archived pages do not expose field-wide Superstar percentages separately.",
        ],
    }


def leave_one_contest_out() -> dict:
    frame = observations_frame()
    rows = []
    predictions = []
    actuals = []
    for contest_id in sorted(frame["contest_id"].astype(str).unique()):
        train = frame[frame["contest_id"].astype(str) != contest_id]
        test = frame[frame["contest_id"].astype(str) == contest_id]
        model = fit_ownership_model(train)
        predicted = _sigmoid(_design(test) @ np.asarray(model["coefficients"], dtype=float))
        actual = test["rostered"].to_numpy(float)
        predictions.extend(predicted.tolist())
        actuals.extend(actual.tolist())
        rows.append({
            "contest_id": contest_id,
            "observations": int(len(test)),
            "mae": float(np.mean(np.abs(predicted - actual))),
            "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        })
    predicted = np.asarray(predictions)
    actual = np.asarray(actuals)
    archive = load_archive()
    tie_rows = [
        {"contest_id": str(row["contest_id"]), "entries": int(row["entries"]),
         "first_place_ties": int(row["first_place_ties"])}
        for row in archive.get("sources", []) if row.get("first_place_ties") is not None
    ]
    return {
        "method": "leave-one-contest-out ridge-logit on archived Yahoo roster percentages",
        "observations": int(len(actual)),
        "contests": int(frame["contest_id"].nunique()),
        "mae": float(np.mean(np.abs(predicted - actual))),
        "rmse": float(np.sqrt(np.mean((predicted - actual) ** 2))),
        "by_contest": rows,
        "observed_duplication": tie_rows,
        "limitations": export_field_model()["limitations"],
    }
