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

import argparse
import glob
import logging
import sys
from pathlib import Path

from cflibs.observability.failure_attribution import attribute_failures, render_markdown


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--results-glob",
        required=True,
        help="Glob pattern matching id_records.csv files (quote it!).",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Path to write the Markdown report.",
    )
    parser.add_argument(
        "--top-n-spectra",
        type=int,
        default=20,
        help="Number of worst-offending spectra to surface (default: 20).",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose logging."
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("failure_attribution")

    matched = sorted(glob.glob(args.results_glob, recursive=True))
    if not matched:
        log.error("no files matched --results-glob %r", args.results_glob)
        return 2
    log.info("matched %d id_records.csv file(s)", len(matched))

    result = attribute_failures([Path(p) for p in matched])
    markdown = render_markdown(
        result,
        top_n_spectra=args.top_n_spectra,
        source_count=len(matched),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    log.info("wrote %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
