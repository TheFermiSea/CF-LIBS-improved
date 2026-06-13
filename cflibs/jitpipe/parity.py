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


def reference_front_end(
    wavelength: Any,
    intensity: Any,
    elements: list[str],
    *,
    db_path: str = "ASD_da/libs_production.db",
    preset: str = "raw",
    overrides: dict | None = None,
) -> Any:
    """Run the REAL reference front-end (response/detect/identify/select + Stark).

    The parity oracle for the J8 composition: builds the resolved
    :class:`~cflibs.inversion.pipeline.AnalysisPipelineConfig` and runs
    :func:`cflibs.jitpipe.host.run_front_end` (which wraps the reference
    ``detect_and_select_lines``). Returns the
    :class:`~cflibs.jitpipe.host.FrontEndResult` the jit ``run_one`` consumes,
    so the end-to-end parity test feeds identical observations to both paths.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.pipeline import build_pipeline_config
    from cflibs.jitpipe.host import run_front_end

    ov = {
        "closure_mode": "standard",
        "saha_boltzmann_graph": False,
        "stark_ne": False,
    }
    if overrides:
        ov.update(overrides)
    cfg = build_pipeline_config(elements, preset=preset, overrides=ov)
    db = AtomicDatabase(db_path)
    return run_front_end(wavelength, intensity, db, cfg)


def reference_pipeline(
    wavelength: Any,
    intensity: Any,
    elements: list[str],
    *,
    db_path: str = "ASD_da/libs_production.db",
    preset: str = "raw",
    overrides: dict | None = None,
) -> Any:
    """Run the REAL reference :func:`cflibs.inversion.pipeline.run_pipeline`.

    The end-to-end parity oracle. Returns ``(CFLIBSResult, diagnostics)``. The
    ``raw`` preset + ``saha_boltzmann_graph=False`` + ``stark_ne=False`` +
    ``standard`` closure keep the reference solve on the ``_solve_lax`` /
    ``_solve_python`` standard-closure path that the jit ``scan_solve`` kernel
    mirrors bit-for-bit (J7 parity), so the composition comparison is exact.
    Set ``CFLIBS_USE_LAX_WHILE_LOOP=1`` for the bit-exact lax oracle.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.pipeline import build_pipeline_config, run_pipeline

    ov = {
        "closure_mode": "standard",
        "saha_boltzmann_graph": False,
        "stark_ne": False,
    }
    if overrides:
        ov.update(overrides)
    cfg = build_pipeline_config(elements, preset=preset, overrides=ov)
    db = AtomicDatabase(db_path)
    return run_pipeline(wavelength, intensity, db, cfg)


def preprocess_parity(intensity: Any, **kwargs: Any) -> Any:
    """Reference adapter for the preprocess stage (J1): baseline + noise.

    Delegates to the reference baseline/noise estimators
    (:mod:`cflibs.inversion.preprocess.preprocessing`) so the J1 kernel can be
    bisected against them. Returns ``(baseline, noise)``.
    """
    import numpy as np

    from cflibs.inversion.preprocess.preprocessing import estimate_baseline, estimate_noise

    inten = np.asarray(intensity, dtype=float)
    baseline = np.asarray(estimate_baseline(inten, **kwargs))
    noise = float(estimate_noise(inten, baseline))
    return baseline, noise


def detect_parity(wavelength: Any, intensity: Any, **kwargs: Any) -> Any:
    """Reference adapter for the detect stage (J2): scipy ``find_peaks``.

    Returns the reference peak indices for stage-bisection against the J2
    ``find_peaks_fixed`` kernel.
    """
    import numpy as np
    from scipy.signal import find_peaks

    peaks, _props = find_peaks(np.asarray(intensity, dtype=float), **kwargs)
    return peaks


def identify_parity(
    wavelength: Any,
    intensity: Any,
    elements: list[str],
    *,
    db_path: str = "ASD_da/libs_production.db",
    **kwargs: Any,
) -> Any:
    """Reference adapter for the identify stage (J4): ``detect_line_observations``.

    Returns the reference ``LineDetectionResult`` so the J3/J4 comb/gate kernels
    can be bisected against the real reference observation build.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.identify.line_detection import detect_line_observations

    db = AtomicDatabase(db_path)
    return detect_line_observations(
        wavelength=wavelength, intensity=intensity, atomic_db=db, elements=elements, **kwargs
    )


def fit_parity(observations: Any, *, weight_cap: float = 0.0) -> Any:
    """Reference adapter for the Boltzmann-fit stage (J5): the padded ``(E, N_max)`` block.

    Builds the REAL reference ``_build_padded_arrays_from_obs`` block from a
    ``LineObservation`` list so the J5 fit kernels can be bisected against the
    reference y/x/w layout.
    """
    from collections import defaultdict

    from cflibs.inversion.solve.iterative import _build_padded_arrays_from_obs

    by_el: dict = defaultdict(list)
    for o in observations:
        by_el[o.element].append(o)
    return _build_padded_arrays_from_obs(dict(by_el), weight_cap=weight_cap)


def solve_parity(observations: Any, *, closure_mode: str = "standard", **kwargs: Any) -> Any:
    """Reference adapter for the iterative-solve stage (J7): the REAL reference solve.

    Runs the REAL :func:`cflibs.inversion.solve.iterative.IterativeCFLIBSSolver`
    on a ``LineObservation`` list (standard closure, lax/python parity path) so
    the composed jit solve spine can be bisected against the reference
    ``CFLIBSResult`` (AC5).
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

    db_path = kwargs.pop("db_path", "ASD_da/libs_production.db")
    db = AtomicDatabase(db_path)
    solver = IterativeCFLIBSSolver(atomic_db=db, saha_boltzmann_graph=False, **kwargs)
    return solver.solve(observations, closure_mode=closure_mode)
