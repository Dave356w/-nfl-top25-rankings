#!/usr/bin/env python3
"""Entry point: publish one single-game model per game into `site/data/showdown/`.

    site/data/showdown/index.json      the game picker the page loads first
    site/data/showdown/<game>.json     one game's pool, correlation model and
                                       reference portfolio

The page at `site/showdown.html` reads these and runs the enumeration and the
Monte Carlo in the visitor's browser. Nothing here is interactive, so a game
whose salary cap Yahoo did not publish, or whose pool cannot build a valid
roster, is skipped with a note in the index rather than prompting.

Failure policy matches `run_daily.py`: on any error nothing is written and the
process exits non-zero, so the previously published models stay live.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
import traceback
from pathlib import Path

from pipeline import notebook as nb
from pipeline import showdown

SITE = Path(__file__).resolve().parent / "site"
DATA = SITE / "data" / "showdown"


class _Tee(io.TextIOBase):
    """Echo the run's progress while keeping a copy for the published log."""

    def __init__(self, stream):
        self._stream = stream
        self.lines: list[str] = []
        self._buffer = ""

    def write(self, text):
        self._stream.write(text)
        self._buffer += text
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.lines.append(line.rstrip())
        return len(text)

    def flush(self):
        self._stream.flush()


def _atomic_write(path, payload):
    """Write JSON through a temporary file so a reader never sees a half file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    temporary.replace(path)
    return path


def write_outputs(payloads, index):
    """Replace the published set atomically enough for a static host.

    Games are written first and the index last, so a reader that fetches the
    index can always fetch every file it names. Stale game files from a previous
    slate are removed after the index no longer references them.
    """
    DATA.mkdir(parents=True, exist_ok=True)
    written = set()
    for payload in payloads:
        path = _atomic_write(DATA / f"{payload['game_id']}.json", payload)
        written.add(path.name)
    _atomic_write(DATA / "index.json", index)
    for stale in DATA.glob("*.json"):
        if stale.name != "index.json" and stale.name not in written:
            stale.unlink()
    return sorted(written)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-games", type=int, default=None,
                        help="Export only the first N games (useful for a smoke run).")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Publish the models without a server-side reference portfolio.")
    parser.add_argument("--simulations", type=int, default=None,
                        help="Override Settings.simulations for the reference run.")
    args = parser.parse_args(argv)

    cfg = nb.CFG
    if args.simulations:
        cfg = nb.replace(cfg, simulations=int(args.simulations))

    tee = _Tee(sys.stdout)
    try:
        with contextlib.redirect_stdout(tee):
            payloads, index = showdown.build_slate(
                cfg, optimize=not args.no_optimize, max_games=args.max_games
            )
        if not payloads:
            raise ValueError(
                "No game produced a usable single-game model; nothing was written."
            )
        index["log"] = tee.lines
        written = write_outputs(payloads, index)
    except Exception:
        traceback.print_exc()
        print("\nShowdown export failed; published models were left untouched.",
              file=sys.stderr)
        return 1

    print(f"\nWrote {len(written)} game model(s) and index.json into {DATA}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
