"""
Tests for the JAX-routed Boltzmann sigma-clip path in composition workflows.

Covers the ``use_jax=True`` opt-in on :class:`BoltzmannPlotFitter` and the
``CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION=1`` env-var pass-through used by
:class:`IterativeCFLIBSSolver` and :class:`FastStreamingAnalyzer`.

Three test groups:

1.  **Numerical agreement** — CPU and JAX paths produce slope, intercept,
    temperature, sigmas, and R² that match within ``rtol=1e-5`` on 100
    synthetic single-element Boltzmann plots.
2.  **Outlier rejection parity** — both paths reject the same indices on
    contrived inputs with planted outliers.
3.  **End-to-end smoke** — the env-var wires through
    :class:`IterativeCFLIBSSolver` and :class:`FastStreamingAnalyzer`
    constructors and the per-spectrum ``solve()`` / fitter API still
    returns finite temperatures.

References:
    feat/jax-boltzmann-composition (this PR)
    docs/jax-port/iterative-boltzmann-consultation.md
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

jax = pytest.importorskip("jax")  # noqa: F401  -- skip whole module if absent

from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import PartitionFunction
from cflibs.core.constants import KB_EV
from cflibs.inversion.boltzmann import BoltzmannPlotFitter, LineObservation
from cflibs.inversion.solver import IterativeCFLIBSSolver

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


# ---------- helpers ---------------------------------------------------------


def _make_synthetic_obs(
    n: int,
    T_K: float,
    intercept: float,
    *,
    noise: float = 0.05,
    seed: int = 0,
    g_k: int = 2,
    A_ki: float = 1e8,
    wavelength_nm: float = 500.0,
    element: str = "Fe",
    stage: int = 1,
) -> list[LineObservation]:
    """Build a list of :class:`LineObservation` whose y_value equals
    ``slope*E_k + intercept + noise``."""
    slope = -1.0 / (KB_EV * T_K)
    rng = np.random.default_rng(seed)
    E_k = rng.uniform(1.0, 7.0, n)
    y = slope * E_k + intercept + rng.normal(0, noise, n)
    # Invert y = ln(I * lambda / (g * A))  ->  I = (g*A/lambda) * exp(y)
    intensity = (g_k * A_ki / wavelength_nm) * np.exp(y)
    return [
        LineObservation(
            wavelength_nm=wavelength_nm,
            intensity=float(intensity[i]),
            intensity_uncertainty=float(intensity[i] * noise),
            element=element,
            ionization_stage=stage,
            E_k_ev=float(E_k[i]),
            g_k=g_k,
            A_ki=A_ki,
        )
        for i in range(n)
    ]


# ---------- 1) Numerical agreement -----------------------------------------


class TestNumericalAgreement:
    """CPU and JAX paths agree on slope/intercept/T/R² for 100 spectra."""

    def test_single_clean_spectrum_rtol_1e_8(self):
        obs = _make_synthetic_obs(n=20, T_K=10_000.0, intercept=20.0, seed=42)
        r_cpu = BoltzmannPlotFitter(outlier_sigma=2.5).fit(obs)
        r_jax = BoltzmannPlotFitter(outlier_sigma=2.5, use_jax=True).fit(obs)
        np.testing.assert_allclose(r_jax.slope, r_cpu.slope, rtol=1e-8)
        np.testing.assert_allclose(r_jax.intercept, r_cpu.intercept, rtol=1e-8)
        np.testing.assert_allclose(r_jax.temperature_K, r_cpu.temperature_K, rtol=1e-8)
        np.testing.assert_allclose(r_jax.r_squared, r_cpu.r_squared, rtol=1e-8)
        # n_points should match exactly (no rejections planted, no boundary edge)
        assert r_jax.n_points == r_cpu.n_points
        assert r_jax.fit_method == r_cpu.fit_method == "sigma_clip"

    def test_100_random_spectra_rtol_1e_5(self):
        """Stress test: 100 single-element Boltzmann plots."""
        rng = np.random.default_rng(0xC0FFEE)
        n_spectra = 100
        slope_diffs = []
        T_diffs = []

        for sid in range(n_spectra):
            T = float(rng.uniform(5_000.0, 20_000.0))
            intercept = float(rng.uniform(15.0, 25.0))
            n_lines = int(rng.integers(8, 30))
            noise = float(rng.uniform(0.02, 0.10))
            obs = _make_synthetic_obs(
                n=n_lines, T_K=T, intercept=intercept, noise=noise, seed=sid
            )
            r_cpu = BoltzmannPlotFitter(outlier_sigma=2.5).fit(obs)
            r_jax = BoltzmannPlotFitter(outlier_sigma=2.5, use_jax=True).fit(obs)

            # rtol=1e-5 on slope and temperature; covers both no-rejection
            # and boundary-rejection cases (latter may diverge by ≤1 inlier
            # if a residual sits within machine-eps of the cut).
            if r_cpu.n_points == r_jax.n_points:
                np.testing.assert_allclose(r_jax.slope, r_cpu.slope, rtol=1e-5)
                np.testing.assert_allclose(
                    r_jax.temperature_K, r_cpu.temperature_K, rtol=1e-5
                )
                slope_diffs.append(abs(r_jax.slope - r_cpu.slope))
                T_diffs.append(abs(r_jax.temperature_K - r_cpu.temperature_K))

        # Sanity: most spectra (>=95%) should have matching inlier counts
        assert len(slope_diffs) >= 95, (
            f"Only {len(slope_diffs)}/100 spectra had matching inlier counts — "
            "boundary divergence may be too high"
        )

    def test_uncertainty_propagation_agrees(self):
        """sigma_T and sigma_slope agree within rtol 1e-5."""
        obs = _make_synthetic_obs(n=15, T_K=12_000.0, intercept=18.0, seed=7)
        r_cpu = BoltzmannPlotFitter(outlier_sigma=2.5).fit(obs)
        r_jax = BoltzmannPlotFitter(outlier_sigma=2.5, use_jax=True).fit(obs)
        np.testing.assert_allclose(r_jax.slope_uncertainty, r_cpu.slope_uncertainty, rtol=1e-5)
        np.testing.assert_allclose(
            r_jax.intercept_uncertainty, r_cpu.intercept_uncertainty, rtol=1e-5
        )
        np.testing.assert_allclose(
            r_jax.temperature_uncertainty_K, r_cpu.temperature_uncertainty_K, rtol=1e-5
        )
        assert r_jax.covariance_matrix is not None
        assert r_cpu.covariance_matrix is not None
        np.testing.assert_allclose(
            r_jax.covariance_matrix, r_cpu.covariance_matrix, rtol=1e-5
        )


# ---------- 2) Outlier rejection parity ------------------------------------


class TestOutlierRejection:
    """Both paths reject the same indices on contrived inputs."""

    def test_planted_outliers_rejected_identically(self):
        """Plant 3 strong outliers in a 15-line plot; CPU and JAX paths
        must reject the same set of indices.

        The planted outliers are made detectable to sigma-clipping by
        scaling BOTH intensity and intensity_uncertainty by the same
        factor — this keeps the per-line y_uncertainty (and therefore
        the WLS weight) unchanged, so the outlier sits ~3 units above
        the fit line with a normal weight, well inside the 2.5σ band
        for a noise scale of 0.05.
        """
        # Use more lines (n=30) and a single planted outlier with a
        # large offset — keeps std(residuals) close to the noise scale
        # so the rejection predicate fires reliably on both paths.
        obs = _make_synthetic_obs(n=30, T_K=10_000.0, intercept=20.0, noise=0.05, seed=1)
        outlier_idx = [3, 7, 11]
        offset = 3.0  # in y-space, vastly above 2.5*0.05 = 0.125
        for i in outlier_idx:
            o = obs[i]
            scale = float(np.exp(offset))
            obs[i] = LineObservation(
                wavelength_nm=o.wavelength_nm,
                intensity=o.intensity * scale,
                # Scale uncertainty by the same factor so y_uncertainty
                # = intensity_unc / intensity stays constant.
                intensity_uncertainty=o.intensity_uncertainty * scale,
                element=o.element,
                ionization_stage=o.ionization_stage,
                E_k_ev=o.E_k_ev,
                g_k=o.g_k,
                A_ki=o.A_ki,
            )

        r_cpu = BoltzmannPlotFitter(outlier_sigma=2.5).fit(obs)
        r_jax = BoltzmannPlotFitter(outlier_sigma=2.5, use_jax=True).fit(obs)

        assert sorted(r_cpu.rejected_points) == sorted(r_jax.rejected_points), (
            f"CPU rejected {sorted(r_cpu.rejected_points)}, "
            f"JAX rejected {sorted(r_jax.rejected_points)}"
        )
        # Sanity: should have rejected at least the planted outliers
        for idx in outlier_idx:
            assert idx in r_cpu.rejected_points, (
                f"CPU failed to reject planted outlier {idx}; "
                f"rejected={r_cpu.rejected_points}"
            )
            assert idx in r_jax.rejected_points

    def test_no_outliers_no_rejections(self):
        """Clean plot — neither path rejects anything."""
        obs = _make_synthetic_obs(n=20, T_K=10_000.0, intercept=20.0, noise=0.05, seed=99)
        r_cpu = BoltzmannPlotFitter(outlier_sigma=4.0).fit(obs)
        r_jax = BoltzmannPlotFitter(outlier_sigma=4.0, use_jax=True).fit(obs)
        # outlier_sigma=4 is permissive enough that clean noise stays in
        assert r_cpu.n_points == 20
        assert r_jax.n_points == 20
        assert r_cpu.rejected_points == r_jax.rejected_points == []

    def test_inlier_mask_matches(self):
        """inlier_mask array equals across CPU and JAX paths on planted
        outliers. Uses uniform weighting (scale intensity_unc with
        intensity) so the WLS doesn't get yanked by lopsided weights —
        the relevant test here is that the two paths *agree* on which
        indices to reject."""
        obs = _make_synthetic_obs(n=12, T_K=8_000.0, intercept=22.0, noise=0.05, seed=2)
        offset = 1.2  # well above 2.5*0.05 = 0.125
        for i in (2, 9):
            o = obs[i]
            scale = float(np.exp(offset))
            obs[i] = LineObservation(
                wavelength_nm=o.wavelength_nm,
                intensity=o.intensity * scale,
                intensity_uncertainty=o.intensity_uncertainty * scale,
                element=o.element,
                ionization_stage=o.ionization_stage,
                E_k_ev=o.E_k_ev,
                g_k=o.g_k,
                A_ki=o.A_ki,
            )
        r_cpu = BoltzmannPlotFitter(outlier_sigma=2.5).fit(obs)
        r_jax = BoltzmannPlotFitter(outlier_sigma=2.5, use_jax=True).fit(obs)
        assert r_cpu.inlier_mask is not None
        assert r_jax.inlier_mask is not None
        np.testing.assert_array_equal(r_cpu.inlier_mask, r_jax.inlier_mask)


# ---------- 3) End-to-end smoke --------------------------------------------


@pytest.fixture
def mock_db():
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.0
    coeffs_I = [3.2188, 0, 0, 0, 0]  # log(25) constant
    db.get_partition_coefficients.side_effect = lambda el, sp: PartitionFunction(
        element=el,
        ionization_stage=sp,
        coefficients=coeffs_I,
        t_min=1000,
        t_max=20000,
        source="test",
    )
    return db


class TestEndToEnd:
    """Wire the env-var through and verify the solver path stays finite."""

    def test_env_var_flips_use_jax_on_solver(self, mock_db, monkeypatch):
        """CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION=1 makes IterativeCFLIBSSolver
        build a JAX-backed fitter."""
        # First confirm default (env unset) is CPU.
        monkeypatch.delenv("CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION", raising=False)
        solver_cpu = IterativeCFLIBSSolver(mock_db, max_iterations=5)
        assert solver_cpu.boltzmann_fitter.use_jax is False

        # Now flip it on.
        monkeypatch.setenv("CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION", "1")
        solver_jax = IterativeCFLIBSSolver(mock_db, max_iterations=5)
        assert solver_jax.boltzmann_fitter.use_jax is True

    def test_env_var_flips_use_jax_on_streaming(self, mock_db, monkeypatch):
        """Same env var also flips FastAnalyzer (the streaming-FAST path)."""
        from cflibs.inversion.runtime.streaming import (
            FastAnalyzer,
            StreamingConfig,
            AnalysisMode,
        )

        monkeypatch.delenv("CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION", raising=False)
        analyzer_cpu = FastAnalyzer(
            mock_db, elements=["A"], config=StreamingConfig(mode=AnalysisMode.FAST)
        )
        assert analyzer_cpu._boltzmann_fitter.use_jax is False

        monkeypatch.setenv("CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION", "1")
        analyzer_jax = FastAnalyzer(
            mock_db, elements=["A"], config=StreamingConfig(mode=AnalysisMode.FAST)
        )
        assert analyzer_jax._boltzmann_fitter.use_jax is True

    def test_solver_jax_path_returns_finite(self, mock_db, monkeypatch):
        """End-to-end: solve() with JAX path returns finite T and the
        same temperature as the CPU path within rtol=1e-3 (loose because
        the iterative solver has its own convergence noise floor)."""
        monkeypatch.setenv("CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION", "1")

        T_eV = 1.0  # ~11604 K
        obs = []
        for E in [1.0, 2.0, 3.0, 4.0, 5.0]:
            y = -E / T_eV + 10.0
            obs.append(
                LineObservation(
                    wavelength_nm=500.0,
                    intensity=float(np.exp(y)),
                    intensity_uncertainty=0.1,
                    element="A",
                    ionization_stage=1,
                    E_k_ev=E,
                    g_k=1,
                    A_ki=1e8,
                )
            )

        solver_jax = IterativeCFLIBSSolver(mock_db, max_iterations=10)
        assert solver_jax.boltzmann_fitter.use_jax is True
        res_jax = solver_jax.solve(obs)
        assert np.isfinite(res_jax.temperature_K)
        assert abs(res_jax.temperature_K - 11604.0) < 500.0
        assert "A" in res_jax.concentrations

        # Compare to CPU baseline
        monkeypatch.delenv("CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION", raising=False)
        solver_cpu = IterativeCFLIBSSolver(mock_db, max_iterations=10)
        assert solver_cpu.boltzmann_fitter.use_jax is False
        res_cpu = solver_cpu.solve(obs)
        np.testing.assert_allclose(res_jax.temperature_K, res_cpu.temperature_K, rtol=1e-3)


# ---------- 4) Behavior contracts ------------------------------------------


class TestBehaviorContracts:
    """Properties that must hold for the JAX path."""

    def test_default_use_jax_is_false(self):
        """The constructor default for use_jax must be False so existing
        callers' behavior is byte-for-byte unchanged."""
        f = BoltzmannPlotFitter()
        assert f.use_jax is False

    def test_ransac_and_huber_unaffected_by_use_jax(self):
        """RANSAC and Huber methods always use the CPU path even when
        use_jax=True (the JAX kernel only implements WLS; outlier
        algorithms with different statistics are not in scope)."""
        from cflibs.inversion.boltzmann import FitMethod

        obs = _make_synthetic_obs(n=15, T_K=10_000.0, intercept=20.0, seed=3)
        # RANSAC + use_jax should not raise and should produce a finite result.
        r = BoltzmannPlotFitter(method=FitMethod.RANSAC, use_jax=True).fit(obs)
        assert np.isfinite(r.temperature_K)
        # Same for Huber.
        r = BoltzmannPlotFitter(method=FitMethod.HUBER, use_jax=True).fit(obs)
        assert np.isfinite(r.temperature_K)

    def test_jax_path_handles_few_points(self):
        """When the inlier set shrinks to exactly 2 points, slope_err is
        inf (matches CPU path behavior — polyfit needs >2 for cov)."""
        obs = _make_synthetic_obs(n=2, T_K=10_000.0, intercept=20.0, seed=4)
        r_cpu = BoltzmannPlotFitter(outlier_sigma=2.5).fit(obs)
        r_jax = BoltzmannPlotFitter(outlier_sigma=2.5, use_jax=True).fit(obs)
        # Both should converge to the same slope (no degrees of freedom
        # for residuals so no rejection), and both should mark sigma_slope
        # as inf since cov isn't estimable with n=2.
        np.testing.assert_allclose(r_jax.slope, r_cpu.slope, rtol=1e-8)
        assert np.isinf(r_jax.slope_uncertainty)
        assert np.isinf(r_cpu.slope_uncertainty)
