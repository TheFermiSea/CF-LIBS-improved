#!/usr/bin/env python3
"""CLI for ``cflibs.observability.failure_attribution``.

Glob a set of per-iter ``id_records.csv`` files, bucket each row into
named failure categories, and emit a Markdown report.

Example
-------
    python scripts/failure_attribution.py \\
      --results-glob '/cluster/shared/cf-libs-bench/results/exp001/shard*/iter-*/id_records.csv' \\
      --output /cluster/shared/cf-libs-bench/results/exp001/failure_attribution.md
"""

from __future__ import annotations

import sys

from cflibs.observability.failure_attribution import attribute_failures, render_markdown

try:
    from scripts._observability_cli import import_observability_cli
except ImportError:
    from pathlib import Path as _Path

    sys.path.insert(0, str(_Path(__file__).resolve().parent))
    from _observability_cli import import_observability_cli  # type: ignore[no-redef]


def main(argv: list[str] | None = None) -> int:
    run = import_observability_cli()
    return run(
        logger_name="failure_attribution",
        extra_arg_name="--top-n-spectra",
        extra_arg_default=20,
        extra_arg_help="Number of worst-offending spectra to surface (default: 20).",
        extra_arg_dest="top_n_spectra",
        aggregate=attribute_failures,
        render_markdown=render_markdown,
        description=__doc__.splitlines()[0],
        argv=argv,
    )


if __name__ == "__main__":
    sys.exit(main())
