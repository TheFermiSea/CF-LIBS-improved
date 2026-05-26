"""Regression tests for partition function polynomial extrapolation guards.

Tracks bead CF-LIBS-improved-s1qr.1 (detective verified 2026-05-25):

* ``polynomial_partition_function`` previously evaluated U(T) = exp(quartic
  in ln T) with only a ``T_K <= 1.0 → 1.0`` guard.  Outside the polynomial's
  fit domain (typically ``[2000, 25000]`` K in the production DB) the
  exponential explodes — empirically Ca I gave U(100 000 K) = 1.14e5 vs the
  direct-sum truth of ~200 (560× wrong) — or falls below the ground-state
  degeneracy (Nb I gave U(500 K) = 0.31 < g0 = 1).
* The fix in :func:`cflibs.plasma.partition.polynomial_partition_function`
  adds optional ``t_min``/``t_max``/``g0`` kwargs.  When supplied, ``T_K`` is
  clamped to ``[t_min, t_max]`` before evaluation and the result is floored
  at the ground-state degeneracy.

These tests use the production atomic DB at ``ASD_da/libs_production.db``
(loaded via the standard ``atomic_db`` fixture defined in ``conftest.py``)
so the polynomial coefficients exercised are the real ones currently in
use, not synthetic stubs.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from cflibs.plasma.partition import (
    direct_sum_partition_function,
    polynomial_partition_function,
)

DB_PATH = "ASD_da/libs_production.db"
db_skip = pytest.mark.skipif(
    not os.path.exists(DB_PATH),
    reason="Production atomic DB not available",
)


@pytest.fixture(scope="module")
def production_db():
    """Production atomic DB for partition coefficient lookups."""
    from cflibs.atomic.database import AtomicDatabase

    return AtomicDatabase(DB_PATH)


@db_skip
def test_nb_i_low_temperature_not_below_ground_state(production_db):
    """Nb I at 500 K (below t_min = 2000) must not return U < g0.

    Bug: legacy polynomial extrapolation gave U(Nb I, 500 K) = 0.31, which
    is unphysical — the partition function cannot fall below the ground
    state's statistical weight (the lowest-energy level always contributes
    at least g0 × exp(0) = g0).

    The production DB happens to be missing the ``energy_levels`` rows for
    Nb I — which is exactly *why* the polynomial fallback is needed for
    this species — so we use the conservative default g0 = 1.0 (every
    quantum level has g >= 1).  The contract under test is that the
    polynomial fallback does not return a sub-unity value at low T.
    """
    pf = production_db.get_partition_coefficients("Nb", 1)
    assert pf is not None, "Nb I partition coefficients missing from DB"
    # Conservative default — every energy level has g >= 1.
    g0 = 1.0

    # WITHOUT guards: the legacy extrapolation underflows below g0.
    U_legacy = polynomial_partition_function(500.0, pf.coefficients)
    assert U_legacy < 1.0, (
        f"Sanity: expected unguarded legacy extrapolation to be < 1.0; got {U_legacy}. "
        "If this assertion fails, the DB coefficients have been re-fit and the "
        "regression no longer reproduces — remove the test or update the bound."
    )

    # WITH guards: clamping to t_min plus floor at g0 must satisfy U >= g0.
    U_fixed = polynomial_partition_function(
        500.0,
        pf.coefficients,
        t_min=pf.t_min,
        t_max=pf.t_max,
        g0=g0,
    )
    assert (
        U_fixed >= g0
    ), f"Partition function with extrapolation guard returned {U_fixed} < g0={g0}"


@db_skip
def test_ca_i_high_temperature_does_not_explode(production_db):
    """Ca I at 100 000 K (above t_max = 25000) must not blow up exponentially.

    Bug: legacy polynomial extrapolation gave U(Ca I, 100 000 K) = 1.14e5
    vs the direct-sum truth of ~200 at the same T (560× wrong).  The
    polynomial is exp(quartic in ln T) and its leading high-T behaviour is
    unconstrained outside the fit domain.

    The fix clamps T to the polynomial's validity window, so an
    out-of-range temperature evaluates as poly(t_max).  We then require
    that this clamped value is within 10× of the direct-sum truth at
    t_max (the polynomial fit is itself imperfect at the boundary, so a
    tight 2× bound would over-promise; the goal of the fix is to prevent
    the exponential blow-up, which we verify by comparing against the
    direct-sum truth rather than against the unguarded extrapolation).
    """
    pf = production_db.get_partition_coefficients("Ca", 1)
    assert pf is not None, "Ca I partition coefficients missing from DB"

    # Direct-sum truth at t_max — the physical reference value.
    levels = production_db.get_energy_levels("Ca", 1)
    g_arr = np.array([lev.g for lev in levels], dtype=np.float64)
    E_arr = np.array([lev.energy_ev for lev in levels], dtype=np.float64)
    ip = production_db.get_ionization_potential("Ca", 1)
    U_truth_at_tmax = direct_sum_partition_function(pf.t_max, g_arr, E_arr, ip)

    # WITHOUT guards: extrapolation explodes.
    U_legacy = polynomial_partition_function(100_000.0, pf.coefficients)
    assert U_legacy > 1.0e4, (
        f"Sanity: expected unguarded legacy extrapolation > 1e4; got {U_legacy}. "
        "If this assertion fails, the DB coefficients have been re-fit and the "
        "regression no longer reproduces — update the bound or remove the test."
    )

    # WITH guards: T is clamped to t_max so evaluation occurs at the boundary.
    U_fixed = polynomial_partition_function(
        100_000.0,
        pf.coefficients,
        t_min=pf.t_min,
        t_max=pf.t_max,
    )

    # The clamped value must equal poly(t_max) (within float roundoff).
    U_at_tmax = polynomial_partition_function(pf.t_max, pf.coefficients)
    assert U_fixed == pytest.approx(
        U_at_tmax, rel=1e-12
    ), f"Clamped extrapolation {U_fixed} != poly(t_max) {U_at_tmax}"

    # And it must no longer be exponentially divergent.  We bound it by
    # an order of magnitude around the direct-sum truth at t_max — this
    # is loose enough to tolerate residual polynomial-vs-direct-sum
    # disagreement at the boundary while still rejecting any future
    # regression that lets U scream off to >= 1e4.
    assert U_fixed < 10.0 * U_truth_at_tmax, (
        f"Clamped U(Ca I, 100 000 K) = {U_fixed} > 10× direct-sum truth "
        f"({U_truth_at_tmax}) at t_max — extrapolation guard not effective."
    )


@db_skip
def test_fe_i_in_validity_range_unchanged(production_db):
    """Fe I at 10 000 K (inside t_min/t_max) must not change with guards on.

    The extrapolation fix must be a no-op inside the polynomial's validity
    window — otherwise we've quietly changed production behaviour.
    """
    pf = production_db.get_partition_coefficients("Fe", 1)
    assert pf is not None, "Fe I partition coefficients missing from DB"
    assert pf.t_min <= 10_000.0 <= pf.t_max, "Fe I row should bracket 10 000 K"

    U_unguarded = polynomial_partition_function(10_000.0, pf.coefficients)
    g0 = min(
        production_db.get_energy_levels("Fe", 1),
        key=lambda lev: lev.energy_ev,
    ).g
    U_guarded = polynomial_partition_function(
        10_000.0,
        pf.coefficients,
        t_min=pf.t_min,
        t_max=pf.t_max,
        g0=g0,
    )

    assert U_guarded == pytest.approx(U_unguarded, rel=1e-12), (
        f"In-range value drifted with guards: unguarded={U_unguarded}, " f"guarded={U_guarded}"
    )


@db_skip
@pytest.mark.requires_jax
def test_jax_twin_matches_numpy(production_db):
    """The JAX polynomial twin must give the same value as the NumPy version.

    Exercises the new ``t_min``/``t_max``/``g0`` kwargs on the JAX path so
    the static argument plumbing does not silently diverge from the NumPy
    contract.
    """
    pytest.importorskip("jax")
    import jax.numpy as jnp

    from cflibs.plasma.partition import polynomial_partition_function_jax

    pf = production_db.get_partition_coefficients("Ca", 1)
    assert pf is not None
    coeffs = jnp.asarray(pf.coefficients, dtype=jnp.float64)

    for T in (500.0, 2000.0, 10_000.0, 25_000.0, 100_000.0):
        u_np = polynomial_partition_function(
            T,
            pf.coefficients,
            t_min=pf.t_min,
            t_max=pf.t_max,
            g0=1.0,
        )
        u_jax = polynomial_partition_function_jax(
            jnp.asarray(T, dtype=jnp.float64),
            coeffs,
            t_min=pf.t_min,
            t_max=pf.t_max,
            g0=1.0,
        )
        assert float(u_jax) == pytest.approx(
            u_np, rel=1e-10
        ), f"NumPy/JAX divergence at T={T}: numpy={u_np}, jax={float(u_jax)}"


def test_no_kwargs_preserves_legacy_behaviour():
    """Calling without t_min/t_max/g0 must reproduce the legacy unclamped path.

    Backward-compatibility contract: existing callers that don't pass the
    new kwargs see bit-identical output.
    """
    # Synthetic coefficients to avoid DB dependence — just enough to give a
    # finite value at T=10 000.
    coeffs = [1.0, 0.5, -0.05, 0.001, 0.0]
    T = 10_000.0

    # Reference: hand-compute exp(sum a_n (ln T)^n)
    ln_T = np.log(T)
    expected = float(np.exp(sum(a * ln_T**i for i, a in enumerate(coeffs))))

    assert polynomial_partition_function(T, coeffs) == pytest.approx(expected, rel=1e-12)
