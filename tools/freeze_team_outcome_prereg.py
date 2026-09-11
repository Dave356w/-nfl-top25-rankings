#!/usr/bin/env python3
"""Freeze the team-outcome preregistration by recording its hash.

Phase P2 of docs/team-outcome-model-plan.md. Writes model/team_outcome_prereg.json,
which tests/test_team_outcome_prereg.py then checks on every run: once frozen,
the document cannot be edited without the edit failing the build.

Re-freezing an already-frozen document is an amendment, so it requires --reason
and is refused outright once a sealed read has been recorded.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import team_outcome as to


def headings(text: str) -> list[str]:
    return [line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("## ")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document", type=Path, default=to.PREREG_PATH)
    parser.add_argument("--output", type=Path, default=to.PREREG_RECORD)
    parser.add_argument("--reason", help="required when amending an existing freeze")
    args = parser.parse_args()

    text = args.document.read_text(encoding="utf-8")
    try:  # a document outside the repository is only ever a test fixture
        document_name = str(args.document.resolve().relative_to(to.ROOT))
    except ValueError:
        document_name = str(args.document)
    digest = to.prereg_sha256(args.document)
    amendments = []
    version = 1

    if args.output.exists():
        previous = json.loads(args.output.read_text(encoding="utf-8"))
        if previous["sha256"] == digest:
            print(f"unchanged: {args.document.name} is already frozen at {digest[:12]}")
            return 0
        if previous.get("sealed_read_recorded"):
            raise SystemExit(
                "Refusing to amend: a sealed read has already been recorded. The seal "
                "is spent, and a changed specification needs a season never read."
            )
        if not args.reason:
            raise SystemExit(
                "Refusing to amend without --reason. The previous hash and the reason "
                "are retained so the chain of versions stays legible."
            )
        amendments = [*previous.get("amendments", []), {
            "version": previous["version"], "sha256": previous["sha256"],
            "frozen_utc": previous["frozen_utc"], "superseded_utc": datetime.now(timezone.utc).isoformat(),
            "reason": args.reason,
        }]
        version = previous["version"] + 1

    record = {
        "schema": 1,
        "phase": "P2",
        "version": version,
        "frozen_utc": datetime.now(timezone.utc).isoformat(),
        "document": document_name,
        "sha256": digest,
        "bytes": len(text.encode("utf-8")),
        "sections": headings(text),
        "sealed_seasons": list(to.SEALED_SEASONS),
        "sealed_read_recorded": False,
        "primary_metric": "brier on the win target, pooled over the walk-forward span",
        "decision_rule": "approved_model = G1 and G2 and G3 and G4 and G5 and G8; "
                         "approved_edge = G7, a separate flag evaluated at P5",
        "specifications": {
            "A": "schedules only",
            "B": "A plus opponent-adjusted play-by-play efficiency",
            "selection": "B is primary if and only if the play-by-play as-of audit is clean; "
                         "no performance number enters this decision",
        },
        "amendments": amendments,
        "limitations": [
            "A frozen hash constrains this document, not the honesty of the run it governs.",
            "Prior exposure is listed in section 1 of the document; this is not a blind protocol.",
            "Gradient boosting is deliberately not run; the deviation is recorded in section 5.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    print(f"frozen v{version}: {args.document.name} at {digest[:12]} "
          f"({record['bytes']} bytes, {len(record['sections'])} sections)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
