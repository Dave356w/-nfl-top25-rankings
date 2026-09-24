"""Publish the depth-keyed model tables the pages re-run in the browser.

`site/depth-model.js` recomputes a player's mean, CV and scoreless rate when a
visitor edits his depth rank. It reads the exact regression artifact and fitted
tables the Python pipeline uses, written here to `site/data/depth_model.json`.
`tests/test_depth_model_js.py` fails if the published file drifts from them.

    python tools/export_depth_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import notebook as nb  # noqa: E402
from pipeline import salary_projection  # noqa: E402

TARGET = ROOT / "site" / "data" / "depth_model.json"


def depth_model() -> dict:
    def keyed(table):
        return {pos: {str(k): v for k, v in row.items()} for pos, row in table.items()}

    regression = salary_projection.load()
    keep = ("positions", "degree", "depth_salary_interactions", "salary_scale",
            "features", "center", "scale", "intercept", "coefficients",
            "minimum_projection", "trained_through_season")
    return {
        "schema": 1,
        "regression": {key: regression[key] for key in keep if key in regression},
        "cv": keyed(nb.CALIBRATED_CV),
        "zero": keyed(nb.ZERO_RATE),
    }


def render() -> str:
    return json.dumps(depth_model(), indent=1, sort_keys=True) + "\n"


def main() -> None:
    TARGET.write_text(render(), encoding="utf-8")
    print(f"Wrote {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
