"""Shared CLI skeleton for the ``cflibs.observability`` report scripts.

``element_confusion.py`` and ``failure_attribution.py`` are thin CLIs that share
the same shape: glob a set of per-iter ``id_records.csv`` files, run a library
aggregate/attribute function over them, render Markdown, and write it out. This
module factors out that skeleton so the two leaf scripts only declare what
differs (their library functions, the extra ``--top-n-*`` argument, and the
render keyword).
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path
from typing import Callable


def run_observability_cli(
    *,
    logger_name: str,
    extra_arg_name: str,
    extra_arg_default: int,
    extra_arg_help: str,
    extra_arg_dest: str,
    aggregate: Callable[[list[Path]], object],
    render_markdown: Callable[..., str],
    description: str,
    argv: list[str] | None = None,
) -> int:
    """Run a glob -> aggregate -> render-Markdown -> write report CLI.

    Parameters
    ----------
    logger_name:
        Name for the module logger.
    extra_arg_name, extra_arg_default, extra_arg_help, extra_arg_dest:
        Definition of the single report-specific count argument (e.g.
        ``--top-n-per-element`` or ``--top-n-spectra``). ``extra_arg_dest`` is
        also the keyword passed to ``render_markdown``.
    aggregate:
        Library function mapping the matched paths to a result object.
    render_markdown:
        Library renderer taking ``(result, **{extra_arg_dest: int},
        source_count=int)`` and returning Markdown.
    description:
        argparse description (first docstring line of the caller).
    argv:
        Optional argument vector for testing.
    """
    parser = argparse.ArgumentParser(description=description)
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
        extra_arg_name,
        type=int,
        default=extra_arg_default,
        help=extra_arg_help,
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(logger_name)

    matched = sorted(glob.glob(args.results_glob, recursive=True))
    if not matched:
        log.error("no files matched --results-glob %r", args.results_glob)
        return 2
    log.info("matched %d id_records.csv file(s)", len(matched))

    result = aggregate([Path(p) for p in matched])
    markdown = render_markdown(
        result,
        **{extra_arg_dest: getattr(args, extra_arg_dest)},
        source_count=len(matched),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown)
    log.info("wrote %s", args.output)
    return 0


def import_observability_cli() -> Callable[..., int]:
    """Import :func:`run_observability_cli` for both invocation styles.

    Works whether the caller is run as ``python scripts/<name>.py`` (so
    ``scripts/`` is on ``sys.path[0]`` and the bare module is importable) or as
    ``python -m scripts.<name>`` (package import).
    """
    try:
        from scripts._observability_cli import run_observability_cli
    except ImportError:
        # Direct ``python scripts/<name>.py`` invocation: sibling import.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from _observability_cli import run_observability_cli  # type: ignore[no-redef]

    return run_observability_cli
