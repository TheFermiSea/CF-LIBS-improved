"""Strict / no-fallback mode for the full-spectrum solver.

Covers the high-severity fallback sites audited in
``cflibs/inversion/solve/full_spectrum.py``:

* the bare ``except Exception -> warm start`` (site 637-638),
* the ``real_fit`` convergence heuristic that ignores ``res.status`` (619-636),
* the physical-plausibility adoption gate that launders warm into a
  ``converged=True`` result (653-684),
* the ``fit_* = warm`` fabrication when not real_fit (640-645),

plus the cheaper data-completeness / clamp sites (default ``AW=50.0``, dead
forward, degenerate SVD library, degenerate warm composition).

The whole point of strict mode is that the **default path is byte-identical to
production** and only ``strict=True`` (or ``CFLIBS_NO_FALLBACK``) turns the
silent substitutions into honest failures.  These tests are DB-free and
XLA-free: the strict decisions live in pure helpers (``_decide_convergence``,
``_resolve_adoption``, ``_handle_optimizer_exception``) plus the pure NumPy
composition / SVD helpers, and the one end-to-end check uses a fake forward and
fails before any ``jax.jit`` compile.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from cflibs.inversion.common.strict import (
    IllConditioned,
    MissingAtomicData,
    NonPhysicalResult,
    OptimizerFailure,
    SolveDiagnostics,
    SolverFailure,
)
from cflibs.inversion.solve import full_spectrum as fs
from cflibs.inversion.solve.full_spectrum import (
    FullSpectrumResult,
    _decide_convergence,
    _handle_optimizer_exception,
    _mass_to_number_fractions,
    _normalise_spectrum,
    _number_to_mass_fractions,
    _resolve_adoption,
    build_svd_basis,
)

# ---------------------------------------------------------------------------
# (i) strict OFF == current production behaviour (byte-identical)
# ---------------------------------------------------------------------------


def test_strict_off_matches_production_on_healthy_inputs():
    """Default (strict off) and strict on agree on healthy data; values frozen."""
    mass = {"Si": 0.6, "Al": 0.25, "Fe": 0.15}
    num_off = _mass_to_number_fractions(mass, list(mass))
    num_on = _mass_to_number_fractions(mass, list(mass), strict=True)
    assert np.allclose(num_off, num_on)
    assert num_off.sum() == pytest.approx(1.0, abs=1e-12)

    back_off = _number_to_mass_fractions({el: float(num_off[i]) for i, el in enumerate(mass)})
    back_on = _number_to_mass_fractions(
        {el: float(num_off[i]) for i, el in enumerate(mass)}, strict=True
    )
    assert back_off == back_on
    for el in mass:
        assert back_off[el] == pytest.approx(mass[el], abs=1e-9)

    rng = np.random.default_rng(0)
    latent = np.abs(rng.standard_normal((5, 120)))
    library = np.abs(rng.standard_normal((24, 5))) @ latent
    observed = np.abs(rng.standard_normal(5)) @ latent
    b_off, m_off, k_off = build_svd_basis(library, observed, n_components=20)
    b_on, m_on, k_on = build_svd_basis(library, observed, n_components=20, strict=True)
    assert k_off == k_on
    assert np.allclose(b_off, b_on) and np.allclose(m_off, m_on)


def test_strict_off_preserves_silent_fallbacks():
    """Strict off keeps the silent substitutions exactly as production."""
    # dead forward -> returns un-normalised zeros (no raise)
    z = np.zeros(8)
    assert np.array_equal(_normalise_spectrum(z), z)
    # degenerate SVD library -> junk basis (no raise)
    _, _, k = build_svd_basis(np.ones((4, 10)), np.ones(10))
    assert k >= 1
    # missing element -> fabricated AW=50.0 default (no raise)
    assert _number_to_mass_fractions({"Xx": 1.0}) == {"Xx": pytest.approx(1.0)}
    # genuinely-absent element -> 1e-6 mass floor (non-degenerate softmax seed)
    arr = _mass_to_number_fractions({"Fe": 1.0, "Cu": 0.0}, ["Fe", "Cu"])
    assert arr[1] > 0.0


# ---------------------------------------------------------------------------
# (ii) strict ON raises / flags FAILED on the relevant failure modes
# ---------------------------------------------------------------------------


def test_strict_refuses_missing_atomic_data():
    """Unknown element -> MissingAtomicData (no AW=50.0 substitution)."""
    with pytest.raises(MissingAtomicData):
        _number_to_mass_fractions({"Xx": 1.0}, strict=True)
    with pytest.raises(MissingAtomicData):
        _mass_to_number_fractions({"Xx": 1.0}, ["Xx"], strict=True)


def test_strict_keeps_absent_elements_at_zero():
    """No 1e-6 phantom mass for genuinely-absent elements in strict mode."""
    arr = _mass_to_number_fractions({"Fe": 1.0, "Cu": 0.0}, ["Fe", "Cu"], strict=True)
    assert arr[1] == 0.0
    assert arr[0] == pytest.approx(1.0)


def test_strict_refuses_degenerate_warm_composition():
    with pytest.raises(NonPhysicalResult):
        _mass_to_number_fractions({"Fe": 0.0, "Cu": 0.0}, ["Fe", "Cu"], strict=True)
    with pytest.raises(NonPhysicalResult):
        _number_to_mass_fractions({"Fe": 0.0, "Cu": 0.0}, strict=True)


def test_strict_refuses_dead_forward():
    with pytest.raises(NonPhysicalResult):
        _normalise_spectrum(np.zeros(8), strict=True)


def test_strict_refuses_degenerate_svd_library():
    # All rows identical -> zero total singular-value variance.
    with pytest.raises((IllConditioned, NonPhysicalResult)):
        build_svd_basis(np.ones((4, 10)), np.ones(10), strict=True)


def test_decide_convergence_classifies_every_failure_predicate():
    """The optimizer endpoint is classified, not collapsed into one outcome."""
    ok = _decide_convergence(finite=True, moved=True, improved_obj=True, iterations=5)
    assert ok.real_fit and ok.reason is None and ok.fit_source == "optimizer"

    cases = {
        "nonfinite": dict(finite=False, moved=True, improved_obj=True, iterations=5),
        "zero_iters": dict(finite=True, moved=True, improved_obj=True, iterations=0),
        "no_move": dict(finite=True, moved=False, improved_obj=True, iterations=5),
        "no_improvement": dict(finite=True, moved=True, improved_obj=False, iterations=5),
    }
    for reason, kw in cases.items():
        d = _decide_convergence(**kw)
        assert not d.real_fit
        assert d.reason == reason
        # The real endpoint is surfaced (never fabricated to equal warm).
        assert d.fit_source == "optimizer_rejected"


def test_resolve_adoption_strict_surfaces_fit_not_warm():
    """Box-edge degeneracy: strict surfaces the untrusted FIT, never warm."""
    warm = {"Fe": 0.5, "Cu": 0.5}
    fit = {"Fe": 0.9, "Cu": 0.1}
    common = dict(
        fit_T_K=5.0e4,
        fit_ne=1e17,
        fit_mass=fit,
        warm_T_K=9.0e3,
        warm_ne=2e17,
        warm_mass=warm,
        T_ratio=1.7,
        ne_ratio=0.1,
        t_threshold=float(np.log(1.8)),
        ne_threshold=0.7,
    )
    # Production (strict off): rode to edge -> revert to warm, no failure_reason.
    prod = _resolve_adoption(real_fit=True, physically_near=False, strict=False, **common)
    assert not prod.adopted
    assert prod.concentrations == warm
    assert prod.temperature_K == pytest.approx(9.0e3)
    assert prod.failure_reason is None

    # Strict: surface the FIT composition (untrusted) + a failure_reason.
    strict = _resolve_adoption(real_fit=True, physically_near=False, strict=True, **common)
    assert not strict.adopted
    assert strict.concentrations == fit
    assert strict.temperature_K == pytest.approx(5.0e4)
    assert strict.failure_reason and "adoption_degenerate" in strict.failure_reason

    # A physically-near converged fit is adopted in both modes.
    near = dict(common, T_ratio=0.1, fit_T_K=1.0e4)
    for s in (False, True):
        d = _resolve_adoption(real_fit=True, physically_near=True, strict=s, **near)
        assert d.adopted and d.concentrations == fit


def test_handle_optimizer_exception_raises_in_strict_only():
    """Bare-except: strict re-raises a typed, chained OptimizerFailure."""
    diag_off = SolveDiagnostics(solver="full_spectrum", strict=False)
    # Non-strict: records the reason, does NOT raise (production degrades).
    _handle_optimizer_exception(RuntimeError("boom"), strict=False, diagnostics=diag_off)
    assert diag_off.failure_reason and "optimizer_exception" in diag_off.failure_reason

    diag_on = SolveDiagnostics(solver="full_spectrum", strict=True)
    with pytest.raises(OptimizerFailure) as exc_info:
        _handle_optimizer_exception(RuntimeError("boom"), strict=True, diagnostics=diag_on)
    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert exc_info.value.diagnostics is diag_on


# ---------------------------------------------------------------------------
# Result container: new strict fields are additive / default-safe
# ---------------------------------------------------------------------------


def test_result_new_fields_default_safe():
    """Constructing without the strict fields keeps production defaults."""
    res = FullSpectrumResult(
        temperature_K=1e4,
        electron_density_cm3=1e17,
        concentrations={"Si": 0.6, "Fe": 0.4},
        warm_start_temperature_K=9e3,
        warm_start_electron_density_cm3=2e17,
        warm_start_concentrations={"Si": 0.5, "Fe": 0.5},
        fit_temperature_K=1e4,
        fit_electron_density_cm3=1e17,
        fit_concentrations={"Si": 0.6, "Fe": 0.4},
        converged=True,
        adopted_fit=True,
        initial_objective=1.0,
        final_objective=0.1,
        iterations=12,
        gradient_norm=1e-4,
    )
    assert res.failure_reason is None
    assert res.fit_valid is True
    assert res.fit_source == "optimizer"


# ---------------------------------------------------------------------------
# End-to-end threading through solve_full_spectrum (DB-free, XLA-free)
# ---------------------------------------------------------------------------


class _DeadForward:
    """Fake _ChunkedForward whose forward is dead (all-zero) — no DB, no XLA."""

    def __init__(
        self,
        db_path,
        elements,
        wl,
        *,
        resolving_power=None,
        instrument_fwhm_nm=None,
        strict=False,
        **_,
    ):
        self.elements = list(elements)
        self.n_lines = 10
        self.n_wl = int(np.asarray(wl).shape[0])
        self.plan = SimpleNamespace(nstitch=4, overlap=2)
        self._strict = strict

    def spectrum_numpy(self, T_eV, log_ne, number_fractions):
        return np.zeros(self.n_wl, dtype=np.float64)


def _solver_kwargs():
    return dict(
        wavelength=np.linspace(400.0, 401.0, 16),
        intensity=np.ones(16),
        elements=["Fe", "Cu"],
        db_path="unused.db",
        warm_start_T_K=9000.0,
        warm_start_ne_cm3=1e17,
        warm_start_concentrations={"Fe": 0.5, "Cu": 0.5},
        fit_pixels=None,
    )


def test_solve_full_spectrum_strict_raises_on_dead_forward(monkeypatch):
    """strict=True propagates a SolverFailure (the dead forward is refused)."""
    pytest.importorskip("jax")
    monkeypatch.setattr(fs, "_ChunkedForward", _DeadForward)
    with pytest.raises(SolverFailure):
        fs.solve_full_spectrum(strict=True, **_solver_kwargs())


def test_solve_full_spectrum_strict_via_env(monkeypatch):
    """The env flag CFLIBS_NO_FALLBACK enables strict mode (strict=None)."""
    pytest.importorskip("jax")
    monkeypatch.setattr(fs, "_ChunkedForward", _DeadForward)
    monkeypatch.setenv("CFLIBS_NO_FALLBACK", "1")
    with pytest.raises(SolverFailure):
        fs.solve_full_spectrum(strict=None, **_solver_kwargs())
