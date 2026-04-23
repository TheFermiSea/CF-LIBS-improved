"""Command-line entry point for the evolution-candidate blocklist scanner.

Allows the same AST check used inside the evolution driver to be invoked
from CI, pre-commit hooks, or one-off inspections::

    python -m cflibs.evolution candidate.py
    python -m cflibs.evolution file_a.py file_b.py
    cat candidate.py | python -m cflibs.evolution -

Exit code is 0 on success (no violations in any input) and 1 otherwise.
Violations are reported to stderr in ``path:line: message`` form, one per
line, to compose cleanly with editor/CI surfaces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from cflibs.evolution.evaluator import scan_source


def _read_source(spec: str) -> tuple[str, str]:
    """Return ``(display_path, source_text)`` for one CLI argument."""
    if spec == "-":
        return "<stdin>", sys.stdin.read()
    path = Path(spec)
    return str(path), path.read_text(encoding="utf-8")


def _scan_one(display_path: str, source: str) -> int:
    """Scan one source text; print violations to stderr, return violation count."""
    violations = scan_source(source)
    for v in violations:
        sys.stderr.write(f"{display_path}:{v.lineno}: {v.format()}\n")
    return len(violations)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m cflibs.evolution",
        description=(
            "Scan Python source files for imports or references to ML "
            "libraries forbidden in the CF-LIBS shipped algorithm. "
            "See beads CF-LIBS-improved-3fy3."
        ),
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Files to scan. Pass '-' to read source from stdin.",
    )
    args = parser.parse_args(argv)

    total_violations = 0
    for spec in args.paths:
        try:
            display, source = _read_source(spec)
        except FileNotFoundError:
            sys.stderr.write(f"{spec}: not found\n")
            total_violations += 1
            continue
        except OSError as exc:
            sys.stderr.write(f"{spec}: {exc}\n")
            total_violations += 1
            continue
        try:
            total_violations += _scan_one(display, source)
        except SyntaxError as exc:
            sys.stderr.write(f"{display}:{exc.lineno or 0}: syntax error: {exc.msg}\n")
            total_violations += 1

    return 0 if total_violations == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
