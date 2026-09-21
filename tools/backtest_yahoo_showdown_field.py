"""Print the archived Yahoo Showdown field-model backtest as JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline import showdown_field


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    report = showdown_field.leave_one_contest_out()
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
