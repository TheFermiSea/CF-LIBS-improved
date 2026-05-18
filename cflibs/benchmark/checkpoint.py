"""
Checkpoint primitives for the unified composition benchmark.

Why a dedicated module
----------------------
``evaluate_composition_workflow`` writes incremental part-files as it
processes each spectrum so a SLURM timeout leaves something on disk.
Future composition workflows (joint optimizers, hybrid manifolds,
posterior samplers...) all want the same primitive, so it lives here
rather than buried in ``unified.py``.

Public surface
--------------
``make_worker_slug``
    Build the ``<hostname>_<pid>_<run_id_slug>`` slug used inside
    part-file names. Hostname disambiguates PID collisions across
    SLURM nodes; the trailing 8-char run-id prefix disambiguates
    same-host / same-PID restarts (Linux PID wraparound + SLURM
    ``--requeue`` can both produce that case).
``new_run_id``
    Generate a fresh UUID4 string for the per-call ``run_id`` that
    every part-file in a single ``evaluate_composition_workflow``
    invocation shares.
``emit_checkpoint_part``
    Write one part-file under
    ``<parts_dir>/part_<worker_slug>_<seq>.parquet`` and return the
    (incremented) sequence number. The write is atomic: the parquet
    is staged into a ``.parquet.tmp`` sibling and then ``rename()``ed
    so a SIGKILL between write+rename leaves no truncated shard
    behind for ``read_parquet_dir`` to choke on. Best-effort: any
    failure is logged at WARNING and the function returns the
    unmodified seq so the caller keeps making progress.

Why the part-file design (vs ``write_parquet(append=True)``)
------------------------------------------------------------
Copilot review on PR #186 flagged the original append-rewrite design
as O(n^2) on large benchmarks: each checkpoint re-read and rewrote
the entire on-disk parquet.  Part-files are O(1) per write; consumers
that need a unified view can ``pyarrow.parquet.read_table(<dir>)``
(row order is filesystem-dependent — sort on
``(dataset_id, spectrum_id, composition_workflow_name)`` if global
ordering matters).
"""

from __future__ import annotations

import os
import socket
import sys
import uuid
from pathlib import Path
from typing import Any, Optional, Sequence

from cflibs.core.logging_config import get_logger

logger = get_logger("benchmark.checkpoint")


def new_run_id() -> str:
    """Return a fresh UUID4 string for one ``evaluate_composition_workflow`` call.

    Sharing a ``run_id`` across an entire invocation means
    ``WHERE run_id = ?`` queries against the merged parquet directory
    consistently return the full set of records for that call (rather
    than silently dropping data when the per-record ``run_id`` drifts
    between part-file writes).
    """
    return str(uuid.uuid4())


def make_worker_slug(run_id: str) -> str:
    """Return ``<hostname>_<pid>_<run_id_slug>`` for part-file names.

    Components:

    ``hostname``
        Sanitized (``.`` and ``/`` replaced with ``_``) so the slug
        is safe to embed in a filename. Disambiguates same-PID
        workers across SLURM nodes.
    ``pid``
        Disambiguates concurrent workers on the same host.
    ``run_id_slug``
        First 8 hex chars of the run-id UUID (dashes stripped).
        Disambiguates same-host / same-PID *restarts* — Linux can
        recycle PIDs after wraparound, and SLURM ``--requeue`` jobs
        frequently land on the same node with the exact same PID;
        without this slug, restart-from-scratch in the same
        ``.parts/`` directory would silently overwrite prior parts.
    """
    host_slug = socket.gethostname().replace(".", "_").replace("/", "_")
    run_id_slug = run_id.replace("-", "")[:8]
    return f"{host_slug}_{os.getpid():d}_{run_id_slug}"


def emit_checkpoint_part(
    *,
    parts_dir: Path,
    run_id: Optional[str],
    worker_slug: str,
    seq: int,
    records: Sequence[Any],
    processed: int,
    final_flush: bool = False,
) -> int:
    """Write one checkpoint part-file and return the new sequence number.

    Parameters
    ----------
    parts_dir
        Directory to write into.  Caller is responsible for ``mkdir``;
        this function does not create parents.
    run_id
        Shared ``run_id`` from :func:`new_run_id` for every part-file
        in this invocation. May be ``None`` (older callers); the
        underlying :func:`cflibs.benchmark.results.write_parquet`
        passes it through unchanged.
    worker_slug
        ``<host>_<pid>_<run_id_slug>`` slug from
        :func:`make_worker_slug`.
    seq
        Current sequence number (pre-increment).  The function
        increments this internally, so the returned value is the
        number *actually used* in the on-disk filename.  Pass that
        return value back as ``seq`` on the next call to keep the
        on-disk numbering gap-free.
    records
        The records to serialize.  Typed loosely as ``Sequence[Any]``
        because :func:`cflibs.benchmark.results.write_parquet` accepts
        both ``IDEvaluationRecord`` and ``CompositionEvaluationRecord``
        lists.
    processed
        Cumulative spectrum count for the surrounding loop; used only
        for the progress marker emitted to stderr.
    final_flush
        When ``True``, tags the stderr marker as a final-flush write
        so log scrapers can distinguish trailing-records flushes from
        periodic checkpoints.

    Returns
    -------
    int
        The (possibly-incremented) sequence number.  On I/O failure
        the input ``seq`` is returned unchanged.

    Atomic-write contract
    ---------------------
    The parquet is staged into ``<part>.parquet.tmp`` and then
    ``Path.rename()``-ed to the final ``.parquet`` suffix. A SIGKILL
    (or any crash) between write and rename leaves a truncated shard
    behind under the ``.tmp`` name; the companion reader
    :func:`cflibs.benchmark.results.read_parquet_dir` globs only
    ``*.parquet`` so the orphan is ignored. ``rename`` is atomic
    within a filesystem — caller MUST keep ``parts_dir`` on the same
    filesystem as the workspace.
    """
    try:
        from cflibs.benchmark.results import (  # noqa: PLC0415
            write_parquet as _checkpoint_write_parquet,
        )

        seq += 1
        part_path = parts_dir / f"part_{worker_slug}_{seq:05d}.parquet"
        # Atomic write: stage into ``.tmp`` then rename. A SLURM SIGKILL
        # (or any crash) mid-write would otherwise leave a truncated
        # parquet shard that breaks the whole-directory read via
        # ``cflibs.benchmark.results.read_parquet_dir`` (pyarrow refuses
        # to concat tables when one shard is corrupt). ``rename`` is
        # atomic within a filesystem.
        tmp_path = part_path.with_suffix(part_path.suffix + ".tmp")
        _checkpoint_write_parquet(
            tmp_path,
            composition_records=list(records),
            run_id=run_id,
        )
        tmp_path.rename(part_path)
        tag = "[checkpoint final-flush]" if final_flush else "[checkpoint]"
        print(
            f"{tag} wrote {len(records)} records to {part_path.name} "
            f"(cumulative {processed} spectra this call) under {parts_dir}",
            file=sys.stderr,
            flush=True,
        )
    except Exception as cp_exc:  # noqa: BLE001
        logger.warning("checkpoint write failed: %s", cp_exc)
    return seq
