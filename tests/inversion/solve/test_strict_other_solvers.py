"""Strict / no-fallback mode tests for the "other" solvers.

Covers ``cflibs/inversion/solve/{joint_optimizer,closed_form,spectral_refiner}.py``.

Each test asserts the two-mode contract:

* strict **off** (the unset default) preserves current production behaviour on a
  healthy case — the silent fallbacks still fire and a result is returned; and
* strict **on** raises the matching typed :class:`SolverFailure` (or refuses)
  on the relevant failure (missing atomic data, unobserved n_e stage,
  degenerate / non-converged fit, optimizer divergence, toy-line fabrication).

Fast by construction: a ``FakeDB`` / ``FakeBasisLibrary`` and a tracer-failing
loss inject the failure modes directly, so no real atomic DB or XLA compile of
a full forward model is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.inversion.common.strict import (
    MissingAtomicData,
    NonIdentifiable,
    NonPhysicalResult,
    NotConverged,
    OptimizerFailure,
    SolverFailure,
    UnobservedStage,
)
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.solve.closed_form import ClosedFormConfig, ClosedFormILRSolver
from cflibs.inversion.solve.spectral_refiner import SpectralRefiner


# --------------------------------------------------------------------------- #
# Fakes                                                                        #
# --------------------------------------------------------------------------- #
class _FakeProvider:
    """Partition-function provider stub returning a constant U(T)."""

    def __init__(self, value: float = 12.0):
        self._value = value

    def at(self, _T_K: float) -> float:
        return self._value


class FakeDB:
    """Minimal AtomicDatabase stand-in for the closed-form solver.

    Configurable to drop partition providers or ionization potentials for a
    chosen element so the strict-mode atomic-data gates can be exercised.
    """

    def __init__(self, drop_partition: set | None = None, drop_ip: set | None = None):
        self._drop_partition = drop_partition or set()
        self._drop_ip = drop_ip or set()

    def partition_function_for(self, element: str, _stage: int):
        if element in self._drop_partition:
            return None
        return _FakeProvider(12.0)

    def get_ionization_potential(self, element: str, _stage: int):
        if element in self._drop_ip:
            return None
        return 7.9  # a real-ish first IP (eV), nowhere near the 15.0 default


class FakeBasisLibrary:
    """Two-element Gaussian basis library for the spectral refiner."""

    def __init__(self, elements=("Fe", "Cu"), n_pixels: int = 64):
        self.elements = list(elements)
        self.wavelength = np.linspace(400.0, 410.0, n_pixels)
        centers = np.linspace(402.0, 408.0, len(self.elements))
        self._basis = np.stack(
            [np.exp(-0.5 * ((self.wavelength - c) / 0.5) ** 2) for c in centers]
        )

    def get_basis_matrix_interp(self, _T_K: float, _ne_cm3: float) -> np.ndarray:
        return self._basis


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _make_neutral_obs(element: str, conc_scale: float, T_eV: float = 1.0):
    """Build physically-consistent neutral lines with a negative Boltzmann slope.

    ``y = ln(I*lambda/(g*A)) = -E/T_eV + ln(conc_scale)``  ->  slope -1/T_eV < 0
    (a physical temperature), distinct upper energies, positive uncertainties.
    """
    g, A, lam = 5, 1.0e8, 400.0
    obs = []
    for E in (1.0, 3.0, 5.0):
        intensity = conc_scale * np.exp(-E / T_eV) * g * A / lam
        obs.append(
            LineObservation(
                wavelength_nm=lam,
                intensity=intensity,
                intensity_uncertainty=0.05 * intensity,
                element=element,
                ionization_stage=1,
                E_k_ev=E,
                g_k=g,
                A_ki=A,
            )
        )
    return obs


def _healthy_observations():
    return _make_neutral_obs("Fe", 0.7) + _make_neutral_obs("Cu", 0.3)


# --------------------------------------------------------------------------- #
# closed_form.py                                                              #
# --------------------------------------------------------------------------- #
class TestClosedFormStrict:
    def test_strict_off_healthy_case_unchanged(self):
        """Default (strict off): a healthy solve returns a converged result."""
        solver = ClosedFormILRSolver(FakeDB(), ClosedFormConfig())
        assert solver.strict is False  # unset default
        result = solver.solve(_healthy_observations(), initial_T_K=10000.0)
        assert result.converged is True
        assert set(result.concentrations) == {"Fe", "Cu"}
        assert 1000.0 < result.temperature_K < 100000.0
        # pressure-balance n_e ran (default path) and produced a positive n_e.
        assert result.electron_density_cm3 > 0.0

    def test_strict_on_missing_partition_raises(self):
        """Strict: a missing partition provider refuses (no canonical fallback)."""
        solver = ClosedFormILRSolver(FakeDB(drop_partition={"Fe"}), strict=True)
        with pytest.raises(MissingAtomicData):
            solver.solve(_healthy_observations(), initial_T_K=10000.0)

    def test_strict_off_missing_partition_falls_back(self):
        """Strict off: a missing partition provider still falls back (unchanged)."""
        solver = ClosedFormILRSolver(FakeDB(drop_partition={"Fe"}), strict=False)
        result = solver.solve(_healthy_observations(), initial_T_K=10000.0)
        assert result.converged is True  # canonical fallback used silently

    def test_strict_on_missing_ionization_potential_raises(self):
        """Strict: a missing first IP refuses instead of the 15.0 eV default."""
        solver = ClosedFormILRSolver(FakeDB(drop_ip={"Cu"}), strict=True)
        with pytest.raises(MissingAtomicData):
            solver.solve(_healthy_observations(), initial_T_K=10000.0)

    def test_strict_on_pressure_ne_refused(self):
        """Strict: the default 1-atm pressure-balance n_e is non-identifiable."""
        solver = ClosedFormILRSolver(FakeDB(), ClosedFormConfig(ne_mode="pressure"), strict=True)
        with pytest.raises(UnobservedStage):
            solver.solve(_healthy_observations(), initial_T_K=10000.0)

    def test_strict_on_no_elements_raises(self):
        solver = ClosedFormILRSolver(FakeDB(), strict=True)
        with pytest.raises(NonIdentifiable):
            solver.solve([], initial_T_K=10000.0)

    def test_strict_off_no_elements_empty_result(self):
        solver = ClosedFormILRSolver(FakeDB(), strict=False)
        result = solver.solve([], initial_T_K=10000.0)
        assert result.converged is False
        assert result.concentrations == {}


# --------------------------------------------------------------------------- #
# spectral_refiner.py                                                         #
# --------------------------------------------------------------------------- #
class TestSpectralRefinerStrict:
    def _refiner(self, strict=None):
        return SpectralRefiner(FakeBasisLibrary(), max_iterations=20, strict=strict)

    def _healthy_inputs(self, lib):
        rng = np.random.default_rng(0)
        wl = lib.wavelength
        # An on-grid observed spectrum that the basis can fit.
        observed = 0.6 * lib._basis[0] + 0.4 * lib._basis[1]
        noise = np.full_like(wl, 0.01)
        return wl, observed, noise

    def test_strict_off_healthy_case_unchanged(self):
        refiner = self._refiner(strict=False)
        wl, observed, noise = self._healthy_inputs(refiner.basis_library)
        res = refiner.refine(
            wavelength=wl,
            observed=observed,
            detected_elements=["Fe", "Cu"],
            T_init_K=10000.0,
            ne_init_cm3=1e17,
            concentrations_init={"Fe": 0.5, "Cu": 0.5},
            noise=noise,
        )
        assert set(res.concentrations) == {"Fe", "Cu"}
        assert abs(sum(res.concentrations.values()) - 1.0) < 1e-6

    def test_strict_off_empty_elements_reports_converged_true(self):
        """Strict off preserves the (misleading) converged=True no-op."""
        refiner = self._refiner(strict=False)
        wl = refiner.basis_library.wavelength
        res = refiner.refine(
            wavelength=wl,
            observed=np.ones_like(wl),
            detected_elements=[],
            T_init_K=10000.0,
            ne_init_cm3=1e17,
            concentrations_init={},
        )
        assert res.converged is True
        assert res.concentrations == {}

    def test_strict_on_empty_elements_raises(self):
        refiner = self._refiner(strict=True)
        wl = refiner.basis_library.wavelength
        with pytest.raises(NonIdentifiable):
            refiner.refine(
                wavelength=wl,
                observed=np.ones_like(wl),
                detected_elements=[],
                T_init_K=10000.0,
                ne_init_cm3=1e17,
                concentrations_init={},
            )

    def test_strict_on_element_absent_from_library_raises(self):
        refiner = self._refiner(strict=True)
        wl = refiner.basis_library.wavelength
        with pytest.raises(NonIdentifiable):
            refiner.refine(
                wavelength=wl,
                observed=np.ones_like(wl),
                detected_elements=["Zn"],  # not in the basis library
                T_init_K=10000.0,
                ne_init_cm3=1e17,
                concentrations_init={"Zn": 1.0},
                noise=np.full_like(wl, 0.01),
            )

    def test_strict_on_noise_none_refused(self):
        refiner = self._refiner(strict=True)
        wl = refiner.basis_library.wavelength
        with pytest.raises(SolverFailure):
            refiner.refine(
                wavelength=wl,
                observed=np.ones_like(wl),
                detected_elements=["Fe"],
                T_init_K=10000.0,
                ne_init_cm3=1e17,
                concentrations_init={"Fe": 1.0},
                noise=None,  # MAD fallback refused in strict mode
            )


# --------------------------------------------------------------------------- #
# joint_optimizer.py  (requires JAX)                                          #
# --------------------------------------------------------------------------- #
def _is_tracer(x) -> bool:
    import jax

    return isinstance(x, jax.core.Tracer)


@pytest.mark.requires_jax
class TestJointOptimizerStrict:
    def _optimizer(self, strict=None):
        from cflibs.inversion.solve.joint_optimizer import JointOptimizer

        def fwd(T_eV, n_e, conc, wavelength):
            import jax.numpy as jnp

            return jnp.zeros_like(wavelength)

        return JointOptimizer(
            fwd, ["Fe", "Cu"], np.linspace(400.0, 410.0, 8), strict=strict
        )

    def _trace_failing_loss(self):
        import jax.numpy as jnp

        def loss(x):
            # jax_minimize traces with abstract tracers: raise there so the
            # optimizer crashes, but stay callable on the concrete x0 (the
            # fallback path computes loss_fn(x0)).
            if _is_tracer(x):
                raise RuntimeError("synthetic optimizer failure during trace")
            return jnp.sum(jnp.asarray(x) ** 2)

        return loss

    def test_strict_off_optimizer_failure_returns_x0(self):
        """Strict off: a crashed optimizer falls back to x0 with status FAILED."""
        import jax.numpy as jnp

        from cflibs.inversion.solve.joint_optimizer import ConvergenceStatus

        opt = self._optimizer(strict=False)
        x0 = jnp.array([0.0, 17.0, 0.0, 0.0])
        final_x, final_loss, iterations, grad_norm, status = opt._run_minimization(
            self._trace_failing_loss(), x0, "BFGS", strict=False
        )
        assert status == ConvergenceStatus.FAILED
        assert np.allclose(np.asarray(final_x), np.asarray(x0))
        assert np.isfinite(final_loss)

    def test_strict_on_optimizer_failure_raises(self):
        """Strict on: a crashed optimizer raises instead of returning x0."""
        import jax.numpy as jnp

        opt = self._optimizer(strict=True)
        x0 = jnp.array([0.0, 17.0, 0.0, 0.0])
        with pytest.raises(OptimizerFailure):
            opt._run_minimization(self._trace_failing_loss(), x0, "BFGS", strict=True)

    def test_strict_off_forward_model_fabricates_default_line(self):
        from cflibs.inversion.solve.joint_optimizer import create_simple_forward_model

        # Cu absent -> 500 nm / strength-1.0 toy line is fabricated (unchanged).
        fwd = create_simple_forward_model(["Fe", "Cu"], {"Fe": [400.0]}, {"Fe": [1.0]})
        assert callable(fwd)

    def test_strict_on_forward_model_missing_line_raises(self):
        from cflibs.inversion.solve.joint_optimizer import create_simple_forward_model

        with pytest.raises(MissingAtomicData):
            create_simple_forward_model(
                ["Fe", "Cu"], {"Fe": [400.0]}, {"Fe": [1.0]}, strict=True
            )


def test_resolve_strict_env(monkeypatch):
    """The shared env flag (CFLIBS_NO_FALLBACK) toggles the unset default."""
    from cflibs.inversion.solve.closed_form import ClosedFormILRSolver

    monkeypatch.setenv("CFLIBS_NO_FALLBACK", "1")
    solver = ClosedFormILRSolver(FakeDB())  # strict unset -> reads env
    assert solver.strict is True
    # With a healthy DB the first strict refusal is the pressure-balance n_e.
    with pytest.raises(SolverFailure):
        solver.solve(_healthy_observations(), initial_T_K=10000.0)
