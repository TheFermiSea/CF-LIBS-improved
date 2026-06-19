"""Shared CSV-ingest helpers for the observability report generators.

Both :mod:`cflibs.observability.element_confusion` and
:mod:`cflibs.observability.failure_attribution` parse the same per-iteration
``id_records.csv`` files. ``load_csvs`` is the single source for the tolerant
read-skip-tag-concat ingest those modules share.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import pandas as pd

logger = logging.getLogger(__name__)


def load_csvs(csv_paths: Sequence[Path]) -> pd.DataFrame:
    """Read, skip-on-error, tag, and concatenate ``id_records.csv`` files.

    Unreadable or empty files are skipped with a warning; each surviving frame
    is tagged with its source path in a ``_source_csv`` column. Returns an empty
    DataFrame when no usable input frames exist.
    """
    frames: list[pd.DataFrame] = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
            logger.warning("skipping unreadable id_records.csv %s: %s", path, exc)
            continue
        if df.empty:
            continue
        df = df.copy()
        df["_source_csv"] = str(path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
