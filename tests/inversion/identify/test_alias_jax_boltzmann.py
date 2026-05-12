"""
Tests for the JAX-vectorized Boltzmann temperature fit in
``cflibs.inversion.identify.alias.boltzmann_temperature_jax``.

The new function is an opt-in (``use_jax_boltzmann_fit=True``) replacement
for the per-spectrum ``scipy.stats.linregress`` call inside
``ALIASIdentifier._estimate_plasma_temperature``. These tests verify:

1. Numerical agreement with the CPU linregress path for negative-slope
   (physical) Boltzmann plots, to within 1e-5 relative error.
2. Matched ``inf``/``nan`` sentinels for non-negative slope and degenerate
   inputs, so the consuming code path can rely on identical control flow.
3. End-to-end opt-in behavior via the constructor flag on
   :class:`ALIASIdentifier`.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import linregress

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from cflibs.core.constants import KB_EV  # noqa: E402
from cflibs.inversion.identify.alias import (  # noqa: E402
    ALIASIdentifier,
    boltzmann_temperature_jax,
)

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


# ---------- Helpers ----------------------------------------------------------


def _cpu_boltzmann_temperatures(
    log_I_over_gA: np.ndarray, E_upper: np.ndarray
) -> np.ndarray:
    """Reference CPU temperatures via per-row ``scipy.stats.linregress``.

    Mirrors the inf/NaN sentinel logic of
    :func:`boltzmann_temperature_jax` so the two implementations can be
    compared elementwise.
    """
    B = log_I_over_gA.shape[0]
    T = np.empty(B, dtype=np.float64)
    for i in range(B):
        y = log_I_over_gA[i]
        x = E_upper if E_upper.ndim == 1 else E_upper[i]
        finite = np.isfinite(y) & np.isfinite(x)
        if finite.sum() < 2 or np.ptp(x[finite]) == 0:
            T[i] = np.nan
            continue
        try:
            result = linregress(x[finite], y[finite])
        except (ValueError, ZeroDivisionError):
            T[i] = np.nan
            continue
        slope = result.slope
        if not np.isfinite(slope):
            T[i] = np.nan
        elif slope >= 0:
            T[i] = np.inf
        else:
            T[i] = -1.0 / (slope * KB_EV)
    return T


def _make_synthetic_batch(
    n_spectra: int = 100,
    n_lines: int = 12,
    seed: int = 0xC0FFEE,
    positive_slope_fraction: float = 0.3,
    noise_sigma: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a synthetic (log_I/gA, E_upper) batch.

    A controlled fraction of spectra are flipped to have a positive slope
    (i.e. ``T = +inf`` in the Boltzmann convention).

    Returns
    -------
    log_I : np.ndarray, shape (n_spectra, n_lines)
    E_upper : np.ndarray, shape (n_lines,)
    positive_mask : np.ndarray of bool, shape (n_spectra,)
        True for rows that were generated with non-negative slope.
    """
    rng = np.random.default_rng(seed)
    E_upper = np.sort(rng.uniform(1.0, 7.0, n_lines))
    T_true = rng.uniform(5000.0, 15000.0, n_spectra)
    slope_true = -1.0 / (KB_EV * T_true)  # all negative
    intercept_true = rng.uniform(15.0, 22.0, n_spectra)

    log_I = slope_true[:, None] * E_upper[None, :] + intercept_true[:, None]
    log_I += rng.normal(0.0, noise_sigma, log_I.shape)

    n_pos = int(round(positive_slope_fraction * n_spectra))
    positive_mask = np.zeros(n_spectra, dtype=bool)
    if n_pos > 0:
        pos_idx = rng.choice(n_spectra, size=n_pos, replace=False)
        positive_mask[pos_idx] = True
        # Flip the y-values about the per-row mean -> positive slope of
        # equal magnitude.
        row_mean = log_I[pos_idx].mean(axis=1, keepdims=True)
        log_I[pos_idx] = 2 * row_mean - log_I[pos_idx]

    return log_I, E_upper, positive_mask


# ---------- Tests ------------------------------------------------------------


class TestNegativeSlopeAgreement:
    """JAX and CPU paths must agree on physical (negative-slope) inputs."""

    def test_round_trip_known_temperature(self) -> None:
        """Single-spectrum noiseless round-trip recovers T exactly."""
        T_true = 8500.0
        slope = -1.0 / (KB_EV * T_true)
        E = np.linspace(2.0, 6.0, 10)
        y = slope * E + 19.0  # intercept arbitrary

        T_jax = boltzmann_temperature_jax(y[None, :], E)
        assert T_jax.shape == (1,)
        assert np.isfinite(T_jax[0])
        np.testing.assert_allclose(T_jax[0], T_true, rtol=1e-10)

    def test_batch_matches_cpu_within_tolerance(self) -> None:
        """JAX and CPU temperatures agree to 1e-5 relative on a 100-row batch."""
        log_I, E_upper, positive_mask = _make_synthetic_batch(
            n_spectra=100, positive_slope_fraction=0.0, noise_sigma=0.02
        )
        T_jax = np.asarray(boltzmann_temperature_jax(log_I, E_upper))
        T_cpu = _cpu_boltzmann_temperatures(log_I, E_upper)

        # All rows here are negative-slope -> both paths should be finite.
        assert np.all(np.isfinite(T_jax))
        assert np.all(np.isfinite(T_cpu))
        assert positive_mask.sum() == 0  # sanity

        np.testing.assert_allclose(T_jax, T_cpu, rtol=1e-5, atol=1.0)

    def test_high_precision_clean_data(self) -> None:
        """With zero noise, JAX and CPU agree to ~machine precision."""
        log_I, E_upper, _ = _make_synthetic_batch(
            n_spectra=50, positive_slope_fraction=0.0, noise_sigma=0.0
        )
        T_jax = np.asarray(boltzmann_temperature_jax(log_I, E_upper))
        T_cpu = _cpu_boltzmann_temperatures(log_I, E_upper)
        np.testing.assert_allclose(T_jax, T_cpu, rtol=1e-10, atol=1e-6)


class TestPositiveSlopeSentinels:
    """JAX and CPU paths must emit the same ``inf``/``nan`` sentinels."""

    def test_positive_slope_returns_inf_single(self) -> None:
        """A purely-ascending Boltzmann plot maps to ``T = +inf``."""
        E = np.linspace(1.0, 5.0, 8)
        y = 0.5 * E + 3.0  # slope = +0.5 (positive)
        T = boltzmann_temperature_jax(y[None, :], E)
        assert np.isposinf(T[0])

    def test_mixed_batch_sentinel_parity(self) -> None:
        """A batch of mixed-sign slopes produces identical inf/finite masks."""
        log_I, E_upper, positive_mask = _make_synthetic_batch(
            n_spectra=100, positive_slope_fraction=0.4, seed=12345
        )
        T_jax = np.asarray(boltzmann_temperature_jax(log_I, E_upper))
        T_cpu = _cpu_boltzmann_temperatures(log_I, E_upper)

        # Sentinel masks must agree exactly.
        np.testing.assert_array_equal(np.isposinf(T_jax), np.isposinf(T_cpu))
        np.testing.assert_array_equal(np.isnan(T_jax), np.isnan(T_cpu))
        np.testing.assert_array_equal(np.isfinite(T_jax), np.isfinite(T_cpu))

        # The positive-slope rows we generated should all show up as inf in
        # at least one path (allow for borderline cases where noise flipped
        # the regressed sign — but here the magnitude is large so it should
        # be all-or-nothing).
        assert np.all(np.isposinf(T_jax[positive_mask]))
        assert np.all(np.isposinf(T_cpu[positive_mask]))

        # And the negative-slope rows agree numerically.
        finite_mask = ~positive_mask
        np.testing.assert_allclose(
            T_jax[finite_mask], T_cpu[finite_mask], rtol=1e-5, atol=1.0
        )

    def test_degenerate_zero_spread_returns_nan(self) -> None:
        """Zero spread in E_upper produces NaN, not inf."""
        E = np.full(6, 3.0)  # all identical -> det = 0
        y = np.linspace(0.0, 1.0, 6)
        T = boltzmann_temperature_jax(y[None, :], E)
        assert np.isnan(T[0])

    def test_too_few_points_returns_nan(self) -> None:
        """Fewer than 2 valid (non-NaN) points -> NaN."""
        E = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([np.nan, np.nan, np.nan, 5.0])
        T = boltzmann_temperature_jax(y[None, :], E)
        assert np.isnan(T[0])


class TestPrecisionAndAPI:
    """Numerical precision and API surface checks."""

    def test_return_diagnostics_shapes(self) -> None:
        """``return_diagnostics=True`` returns ``(T, slope, r_squared)``."""
        log_I, E_upper, _ = _make_synthetic_batch(
            n_spectra=10, positive_slope_fraction=0.0
        )
        T, slope, r_sq = boltzmann_temperature_jax(
            log_I, E_upper, return_diagnostics=True
        )
        assert T.shape == (10,)
        assert slope.shape == (10,)
        assert r_sq.shape == (10,)
        # All-negative synthetic batch -> all slopes negative, all R^2 high.
        assert np.all(slope < 0)
        assert np.all((r_sq > 0.95) | np.isnan(r_sq))

    def test_uniform_weights_accepted(self) -> None:
        """Uniform weights are silently accepted (equivalent to unweighted)."""
        log_I, E_upper, _ = _make_synthetic_batch(n_spectra=5)
        w = np.ones_like(log_I)
        T_uniform = np.asarray(boltzmann_temperature_jax(log_I, E_upper, weights=w))
        T_none = np.asarray(boltzmann_temperature_jax(log_I, E_upper, weights=None))
        np.testing.assert_allclose(T_uniform, T_none, rtol=1e-12)

    def test_nonuniform_weights_raises(self) -> None:
        """Non-uniform weights raise ``NotImplementedError`` (no silent misuse)."""
        log_I, E_upper, _ = _make_synthetic_batch(n_spectra=5)
        w = np.linspace(0.1, 1.0, log_I.size).reshape(log_I.shape)
        with pytest.raises(NotImplementedError):
            boltzmann_temperature_jax(log_I, E_upper, weights=w)

    def test_shape_mismatch_raises(self) -> None:
        """Mismatched batch and E_upper shapes raise ``ValueError``."""
        log_I = np.zeros((4, 6))
        E_upper = np.zeros((3, 6))  # wrong batch dim
        with pytest.raises(ValueError):
            boltzmann_temperature_jax(log_I, E_upper)


class TestALIASIdentifierFlag:
    """The opt-in flag on :class:`ALIASIdentifier` itself."""

    def test_constructor_accepts_flag(self) -> None:
        """The constructor accepts the flag and stores it."""

        class _StubDB:
            """Minimal atomic-database stub: ``ALIASIdentifier.__init__`` only
            instantiates :class:`SahaBoltzmannSolver` against it and never
            queries transitions here."""

            def get_available_elements(self):  # pragma: no cover - unused
                return ["Fe"]

        identifier = ALIASIdentifier(
            atomic_db=_StubDB(),
            use_jax_boltzmann_fit=True,
        )
        assert identifier.use_jax_boltzmann_fit is True

        identifier_off = ALIASIdentifier(atomic_db=_StubDB())
        assert identifier_off.use_jax_boltzmann_fit is False
