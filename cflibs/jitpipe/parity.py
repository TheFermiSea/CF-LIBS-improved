"""Stage-by-stage reference adapters for jitpipe parity tests (ADR-0004 §5.4).

Test-only bridge helpers comparing the jittable stages against the reference
inversion pipeline. This is a carve-out module (alongside ``host.py`` /
``snapshot.py``) permitted to import SQLite-touching reference code
(:mod:`cflibs.atomic.database`, the iterative solver). Production ``jitpipe``
code never imports this module.

J0 ships the snapshot-bridge adapters (used by ``tests/jitpipe/test_snapshot.py``
to assert AC4 field parity vs both legacy builders); per-stage adapters for
J1-J7 are stubs until their stages exist.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def reference_atomic_snapshot(
    elements: list[str],
    wavelength_range: tuple[float, float],
    db_path: str = "ASD_da/libs_production.db",
    *,
    include_levels: bool = True,
) -> Any:
    """Build the reference forward-kernel :class:`AtomicSnapshot` via the DB.

    Mirrors the J0 AC4 reference call: ``AtomicDatabase.snapshot(...)`` over a
    candidate element/wavelength set. Used to assert
    :meth:`PipelineSnapshot.to_atomic_snapshot` parity.
    """
    from cflibs.atomic.database import AtomicDatabase

    db = AtomicDatabase(db_path)
    return db.snapshot(
        elements=elements,
        wavelength_range=wavelength_range,
        include_levels=include_levels,
    )


def reference_lax_snapshot(elements: list[str], db_path: str = "ASD_da/libs_production.db") -> Any:
    """Build the reference lax-solver ``_AtomicSnapshot`` via ``IterativeCFLIBSSolver``.

    Used to assert :meth:`PipelineSnapshot.to_lax_snapshot` parity (J0 AC4) for
    the fields consumed inside ``_run_lax_while_loop``.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver, _AtomicSnapshot

    db = AtomicDatabase(db_path)
    solver = IterativeCFLIBSSolver(atomic_db=db)
    return _AtomicSnapshot.from_solver(solver, elements)


def compare_arrays(a: Any, b: Any, *, rtol: float = 0.0, atol: float = 0.0) -> bool:
    """Return True iff two arrays match within tolerance (NaN-aware).

    Tier-K/S/D/B contracts default to exact equality (rtol=atol=0) for
    atomic-data passthrough; callers loosen as the relevant tier allows.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    return bool(np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True))


def preprocess_parity(*_args: Any, **_kwargs: Any) -> Any:
    """Reference adapter for the preprocess stage (J1). Stub until J1 lands."""
    raise NotImplementedError("preprocess parity adapter lands with J1")


def detect_parity(*_args: Any, **_kwargs: Any) -> Any:
    """Reference adapter for the detect stage (J2). Stub until J2 lands."""
    raise NotImplementedError("detect parity adapter lands with J2")


def identify_parity(*_args: Any, **_kwargs: Any) -> Any:
    """Reference adapter for the identify stage (J4). Stub until J4 lands."""
    raise NotImplementedError("identify parity adapter lands with J4")


def fit_parity(*_args: Any, **_kwargs: Any) -> Any:
    """Reference adapter for the Boltzmann-fit stage (J5). Stub until J5 lands."""
    raise NotImplementedError("fit parity adapter lands with J5")


def solve_parity(*_args: Any, **_kwargs: Any) -> Any:
    """Reference adapter for the iterative-solve stage (J7). Stub until J7 lands."""
    raise NotImplementedError("solve parity adapter lands with J7")
