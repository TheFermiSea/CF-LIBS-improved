"""Numerical-equivalence tests for ``SahaBoltzmannSolverJax``.

The JAX solver MUST produce identical output (within ``rtol=1e-5,
atol=1e-7``) to the canonical NumPy ``SahaBoltzmannSolver`` for a range of
physically meaningful LIBS plasma conditions. These tests run on the CPU
backend (forced by ``tests/conftest.py``) so they are CI-safe.

Uses the session-scoped ``atomic_db_session`` fixture from
``tests/plasma/conftest.py`` to avoid rebuilding the in-memory database
on every parametrised case.
"""

from __future__ import annotations

import numpy as np
import pytest

# Uses the session-scoped ``atomic_db_session`` fixture from
# ``tests/plasma/conftest.py``; no production database file required.

jax = pytest.importorskip("jax")

from cflibs.plasma.saha_boltzmann import (  # noqa: E402
    SahaBoltzmannSolver,
    SahaBoltzmannSolverJax,
)


# Representative LIBS plasma sweep — temperatures 6000-15000 K and electron
# densities 1e15-1e17 cm^-3 cover the canonical operating regime called out
# in the task brief. Kept compact (3x3x2 = 18 cases) so the parametrised
# loop runs in seconds; the parity property is shape-invariant.
_T_K_GRID = (6000.0, 10000.0, 15000.0)
_NE_GRID = (1.0e15, 1.0e16, 1.0e17)
_ELEMENTS = ("Fe", "H")  # both populated in the conftest fixture DB


@pytest.mark.parametrize("T_K", _T_K_GRID)
@pytest.mark.parametrize("n_e", _NE_GRID)
@pytest.mark.parametrize("element", _ELEMENTS)
def test_partition_function_jax_matches_numpy(atomic_db_session, element, T_K, n_e):
    """Direct-sum partition function — JAX vs NumPy parity."""
    from cflibs.core.constants import KB_EV

    atomic_db = atomic_db_session
    T_eV = T_K * KB_EV
    np_solver = SahaBoltzmannSolver(atomic_db)
    jax_solver = SahaBoltzmannSolverJax(atomic_db)

    for stage in (1, 2):
        ip = atomic_db.get_ionization_potential(element, stage)
        if ip is None:
            continue
        max_e = ip * 0.95
        u_np = np_solver.calculate_partition_function(element, stage, T_eV, max_energy_ev=max_e)
        u_jax = jax_solver.calculate_partition_function(
            element, stage, T_eV, max_energy_ev=max_e
        )
        np.testing.assert_allclose(u_jax, u_np, rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("T_K", _T_K_GRID)
@pytest.mark.parametrize("n_e", _NE_GRID)
@pytest.mark.parametrize("element", _ELEMENTS)
def test_ionization_balance_jax_matches_numpy(atomic_db_session, element, T_K, n_e):
    """Saha ionization balance — JAX vs NumPy parity."""
    from cflibs.core.constants import KB_EV

    atomic_db = atomic_db_session
    T_eV = T_K * KB_EV
    total_density = 1.0e15
    np_solver = SahaBoltzmannSolver(atomic_db)
    jax_solver = SahaBoltzmannSolverJax(atomic_db)

    out_np = np_solver.solve_ionization_balance(element, T_eV, n_e, total_density)
    out_jax = jax_solver.solve_ionization_balance(element, T_eV, n_e, total_density)

    common_stages = set(out_np.keys()) & set(out_jax.keys())
    assert len(common_stages) >= 1
    for stage in common_stages:
        np.testing.assert_allclose(out_jax[stage], out_np[stage], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("T_K", (6000.0, 10000.0, 15000.0))
@pytest.mark.parametrize("n_e", (1.0e15, 1.0e17))
def test_solve_plasma_jax_matches_numpy(atomic_db_session, T_K, n_e):
    """Full ``solve_plasma`` (ionization + Boltzmann level populations)."""
    from cflibs.plasma.state import SingleZoneLTEPlasma

    atomic_db = atomic_db_session
    plasma = SingleZoneLTEPlasma(
        T_e=T_K,
        n_e=n_e,
        species={"Fe": 1.0e15, "H": 5.0e15},
    )

    np_solver = SahaBoltzmannSolver(atomic_db)
    jax_solver = SahaBoltzmannSolverJax(atomic_db)

    pop_np = np_solver.solve_plasma(plasma)
    pop_jax = jax_solver.solve_plasma(plasma)

    common = set(pop_np.keys()) & set(pop_jax.keys())
    assert len(common) >= 0.9 * min(len(pop_np), len(pop_jax)), (
        f"key sets diverged: numpy={len(pop_np)} jax={len(pop_jax)} common={len(common)}"
    )
    for key in common:
        np.testing.assert_allclose(pop_jax[key], pop_np[key], rtol=1e-5, atol=1e-7)


def test_get_ionization_fractions_jax_matches_numpy(atomic_db_session):
    """Fractional ionization populations should be identical."""
    from cflibs.core.constants import KB_EV

    atomic_db = atomic_db_session
    T_eV = 10000.0 * KB_EV
    n_e = 1.0e16
    np_solver = SahaBoltzmannSolver(atomic_db)
    jax_solver = SahaBoltzmannSolverJax(atomic_db)

    f_np = np_solver.get_ionization_fractions("Fe", T_eV, n_e)
    f_jax = jax_solver.get_ionization_fractions("Fe", T_eV, n_e)

    common = set(f_np) & set(f_jax)
    assert common
    for stage in common:
        np.testing.assert_allclose(f_jax[stage], f_np[stage], rtol=1e-5, atol=1e-7)


def test_jax_solver_uses_jnp_kernel(atomic_db_session):
    """Smoke test: verify the JAX kernel is on the call path.

    Catches the regression where the JAX class silently falls back to the
    NumPy code path (which is what the task brief calls out as the bug).
    """
    from cflibs.core.constants import KB_EV
    from cflibs.plasma import saha_boltzmann as sb

    assert sb.HAS_JAX is True
    solver = SahaBoltzmannSolverJax(atomic_db_session)
    solver.calculate_partition_function("Fe", 1, 10000.0 * KB_EV, max_energy_ev=7.5)
    assert ("Fe", 1) in solver._level_cache
    g_jnp, e_jnp, _ = solver._level_cache[("Fe", 1)]
    assert "jax" in type(g_jnp).__module__
    assert "jax" in type(e_jnp).__module__
