#!/usr/bin/env python3
"""CLI for ``cflibs.observability.element_confusion``.

Glob a set of per-iter ``id_records.csv`` files, compute per-element
precision/recall/F1, and emit a Markdown report identifying systematic
over- and under-prediction by element.

Example
-------
    python scripts/element_confusion.py \\
      --results-glob '/cluster/shared/cf-libs-bench/results/exp001/shard*/iter-*/id_records.csv' \\
      --output /cluster/shared/cf-libs-bench/results/exp001/element_confusion.md
"""

from __future__ import annotations

import sys

from cflibs.observability.element_confusion import aggregate_confusion, render_markdown

try:
    from scripts._observability_cli import import_observability_cli
except ImportError:
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from _observability_cli import import_observability_cli  # type: ignore[no-redef]


def main(argv: list[str] | None = None) -> int:
    run = import_observability_cli()
    return run(
        logger_name="element_confusion",
        extra_arg_name="--top-n-per-element",
        extra_arg_default=50,
        extra_arg_help=(
            "Number of (workflow, dataset, element) rows to surface in table A "
            "(default: 50)."
        ),
        extra_arg_dest="top_n_per_element",
        aggregate=aggregate_confusion,
        render_markdown=render_markdown,
        description=__doc__.splitlines()[0],
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
