"""
Tests for the Anderson-accelerated Saha-Boltzmann solver.

Tests cover convergence, Anderson vs Picard agreement, iteration reduction,
safeguarding, edge cases, and batch (vmap) execution.

Convention: n_e [cm^-3], T_eV [eV], C_i dimensionless sum-to-1.
"""

import pytest
import numpy as np

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from cflibs.plasma.anderson_solver import (  # noqa: E402
    AtomicDataJAX,
    anderson_solve,
    picard_solve,
)

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Synthetic atomic data (no database needed)
# ---------------------------------------------------------------------------


def _make_simple_atomic_data(
    ionization_potentials: list[list[float]],
    partition_log_u: list[list[float]],
) -> AtomicDataJAX:
    """Create AtomicDataJAX from explicit values.

    Parameters
    ----------
    ionization_potentials : list of list of float
        IP in eV for each element and transition.  E.g., [[7.9, 16.2], [7.7, 20.3]]
        for Fe-like and Cu-like 2-transition systems.
    partition_log_u : list of list of float
        log(U) constant term for each element and species.
        E.g., [[2.5, 1.0, 0.5], [1.5, 0.8, 0.3]] for 3 species per element.
    """
    n_elem = len(ionization_potentials)
    max_transitions = max(len(ips) for ips in ionization_potentials)
    max_species = max_transitions + 1

    ip_arr = np.zeros((n_elem, max_transitions), dtype=np.float64)
    pf_arr = np.zeros((n_elem, max_species, 5), dtype=np.float64)
    ns_arr = np.zeros(n_elem, dtype=np.int32)

    for i, (ips, pf_logs) in enumerate(zip(ionization_potentials, partition_log_u)):
        for t, ip in enumerate(ips):
            ip_arr[i, t] = ip
        ns_arr[i] = len(pf_logs)
        for s, log_u in enumerate(pf_logs):
            pf_arr[i, s, 0] = log_u  # Constant partition function (no T dependence)

    return AtomicDataJAX(
        ionization_potentials=jnp.array(ip_arr),
        partition_coefficients=jnp.array(pf_arr),
        n_stages=jnp.array(ns_arr),
    )


# Standard test data: Fe-like + Cu-like system
# Fe: IP_I=7.9 eV, IP_II=16.2 eV; U_I~25, U_II~10, U_III~1
# Cu: IP_I=7.7 eV, IP_II=20.3 eV; U_I~4, U_II~2, U_III~1
_STANDARD_ATOMIC = _make_simple_atomic_data(
    ionization_potentials=[[7.9, 16.2], [7.7, 20.3]],
    partition_log_u=[
        [np.log(25.0), np.log(10.0), np.log(1.0)],
        [np.log(4.0), np.log(2.0), np.log(1.0)],
    ],
)
_STANDARD_COMP = jnp.array([0.7, 0.3])  # 70% Fe, 30% Cu

# Single element (pure Fe)
_SINGLE_ELEM_ATOMIC = _make_simple_atomic_data(
    ionization_potentials=[[7.9]],
    partition_log_u=[[np.log(25.0), np.log(10.0)]],
)
_SINGLE_ELEM_COMP = jnp.array([1.0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPicardConvergence:
    """test_picard_convergence: m=0 Picard converges for standard conditions."""

    def test_standard_conditions(self):
        result = picard_solve(
            T_eV=1.0,
            compositions=_STANDARD_COMP,
            atomic_data=_STANDARD_ATOMIC,
            n_e_init=1e16,
            n_total_ion=1e17,
            tol=1e-10,
            max_iter=100,
        )
        assert result.converged, f"Picard did not converge: residual={float(result.residual)}"
        assert 1e12 <= float(result.n_e) <= 1e20, f"n_e out of range: {float(result.n_e)}"


class TestAndersonConvergence:
    """test_anderson_convergence: m=3 converges for standard conditions."""

    def test_standard_conditions(self):
        result = anderson_solve(
            T_eV=1.0,
            compositions=_STANDARD_COMP,
            atomic_data=_STANDARD_ATOMIC,
            n_e_init=1e16,
            n_total_ion=1e17,
            m=3,
            tol=1e-10,
            max_iter=50,
        )
        assert result.converged, f"Anderson did not converge: residual={float(result.residual)}"
        assert 1e12 <= float(result.n_e) <= 1e20


class TestAndersonVsPicardSameResult:
    """test_anderson_vs_picard_same_result: converged n_e agrees to <1e-8 relative."""

    def test_agreement(self):
        picard = picard_solve(
            T_eV=1.0,
            compositions=_STANDARD_COMP,
            atomic_data=_STANDARD_ATOMIC,
            n_e_init=1e16,
            n_total_ion=1e17,
            tol=1e-12,
            max_iter=200,
        )
        anderson = anderson_solve(
            T_eV=1.0,
            compositions=_STANDARD_COMP,
            atomic_data=_STANDARD_ATOMIC,
            n_e_init=1e16,
            n_total_ion=1e17,
            m=3,
            tol=1e-12,
            max_iter=200,
        )

        assert picard.converged
        assert anderson.converged

        ne_picard = float(picard.n_e)
        ne_anderson = float(anderson.n_e)
        rel_diff = abs(ne_picard - ne_anderson) / max(abs(ne_picard), 1e-30)
        assert (
            rel_diff < 1e-8
        ), f"Results differ: Picard={ne_picard}, Anderson={ne_anderson}, rel={rel_diff}"


class TestAndersonFewerIterations:
    """test_anderson_fewer_iterations: Anderson m=3 uses fewer iterations than Picard."""

    def test_iteration_reduction(self):
        conditions = [
            (0.8, 1e16, 1e17),
            (1.0, 1e16, 1e17),
            (1.2, 5e15, 5e16),
            (1.5, 1e16, 1e17),
            (0.6, 1e15, 5e16),
            (1.0, 1e17, 5e17),
            (0.9, 5e16, 2e17),
            (1.3, 2e16, 1e17),
            (0.7, 1e16, 1e18),
            (1.1, 5e15, 5e16),
        ]

        picard_iters = []
        anderson_iters = []

        for T, ne_init, n_total in conditions:
            p = picard_solve(
                T_eV=T,
                compositions=_STANDARD_COMP,
                atomic_data=_STANDARD_ATOMIC,
                n_e_init=ne_init,
                n_total_ion=n_total,
                tol=1e-10,
                max_iter=200,
            )
            a = anderson_solve(
                T_eV=T,
                compositions=_STANDARD_COMP,
                atomic_data=_STANDARD_ATOMIC,
                n_e_init=ne_init,
                n_total_ion=n_total,
                m=3,
                tol=1e-10,
                max_iter=200,
            )
            if p.converged and a.converged:
                picard_iters.append(int(p.iterations))
                anderson_iters.append(int(a.iterations))

        assert len(picard_iters) >= 5, "Too few converged cases"

        mean_picard = np.mean(picard_iters)
        mean_anderson = np.mean(anderson_iters)
        speedup = mean_picard / max(mean_anderson, 1)

        assert speedup > 1.5, (
            f"Anderson speedup only {speedup:.2f}x "
            f"(Picard avg={mean_picard:.1f}, Anderson avg={mean_anderson:.1f})"
        )


class TestAndersonMSweep:
    """test_anderson_m_sweep: m=0,1,2,3,5 all converge."""

    def test_all_m_converge(self):
        for m_val in [0, 1, 2, 3, 5]:
            result = anderson_solve(
                T_eV=1.0,
                compositions=_STANDARD_COMP,
                atomic_data=_STANDARD_ATOMIC,
                n_e_init=1e16,
                n_total_ion=1e17,
                m=m_val,
                tol=1e-10,
                max_iter=200,
            )
            assert result.converged, f"Failed to converge with m={m_val}"


class TestSafeguardingClamp:
    """test_safeguarding_clamp: start from n_e=1e25, verify clamping works."""

    def test_extreme_initial_guess(self):
        result = anderson_solve(
            T_eV=1.0,
            compositions=_STANDARD_COMP,
            atomic_data=_STANDARD_ATOMIC,
            n_e_init=1e25,  # Way out of range
            n_total_ion=1e17,
            m=3,
            tol=1e-10,
            max_iter=100,
        )
        ne = float(result.n_e)
        assert 1e12 <= ne <= 1e20, f"n_e not clamped: {ne}"


class TestSingleElement:
    """test_single_element: pure Fe, verify against analytical Saha solution."""

    def test_single_element_analytical(self):
        # For a single element with 2 species (neutral + singly ionized):
        # S = SAHA_CONST / n_e * T^1.5 * (U_II/U_I) * exp(-IP/T)
        # n_e = n_total * S / (1 + S)  (charge neutrality with z=0,1)
        # This is a fixed point in n_e.

        T_eV = 1.0
        n_total = 1e17
        IP = 7.9
        U_I = 25.0
        U_II = 10.0

        result = anderson_solve(
            T_eV=T_eV,
            compositions=_SINGLE_ELEM_COMP,
            atomic_data=_SINGLE_ELEM_ATOMIC,
            n_e_init=1e16,
            n_total_ion=n_total,
            m=3,
            tol=1e-12,
            max_iter=100,
        )
        assert result.converged

        ne = float(result.n_e)

        # Verify against analytical: n_e = n_total * S/(1+S) where S depends on n_e
        # At convergence, check consistency
        from cflibs.core.constants import SAHA_CONST_CM3

        S = SAHA_CONST_CM3 / ne * T_eV**1.5 * (U_II / U_I) * np.exp(-IP / T_eV)
        ne_check = n_total * S / (1 + S)

        rel_err = abs(ne - ne_check) / ne
        assert (
            rel_err < 1e-4
        ), f"Analytical check failed: n_e={ne}, check={ne_check}, rel_err={rel_err}"


class TestHighTemperature:
    """test_high_temperature: T=3eV with significant double ionization."""

    def test_high_T(self):
        result = anderson_solve(
            T_eV=3.0,
            compositions=_STANDARD_COMP,
            atomic_data=_STANDARD_ATOMIC,
            n_e_init=1e17,
            n_total_ion=1e17,
            m=3,
            tol=1e-10,
            max_iter=100,
        )
        assert result.converged, f"High-T did not converge: residual={float(result.residual)}"
        ne = float(result.n_e)
        # At high T, mean charge > 1 so n_e > n_total for singly-charged species
        assert ne > 0, f"Non-positive n_e: {ne}"


class TestVmapBatch:
    """test_vmap_batch: vmap over different (T, n_e_init) pairs."""

    def test_batch_matches_sequential(self):
        T_values = jnp.array([0.8, 1.0, 1.2, 1.5, 2.0])
        ne_inits = jnp.array([1e15, 1e16, 5e16, 1e17, 5e17])
        n_totals = jnp.full(5, 1e17)

        # Sequential results
        seq_results = []
        for i in range(5):
            r = anderson_solve(
                T_eV=float(T_values[i]),
                compositions=_STANDARD_COMP,
                atomic_data=_STANDARD_ATOMIC,
                n_e_init=float(ne_inits[i]),
                n_total_ion=float(n_totals[i]),
                m=3,
                tol=1e-10,
                max_iter=50,
            )
            seq_results.append(float(r.n_e))

        # Batch via vmap
        from cflibs.plasma.anderson_solver import _get_solver

        solver = _get_solver(3, 50)

        def _solve_one(T_eV, log_ne_init, log_n_total):
            return solver(
                T_eV,
                _STANDARD_COMP,
                _STANDARD_ATOMIC,
                log_n_total,
                log_ne_init,
                jnp.float64(1e-10),
                jnp.float64(1e-6),
            )

        log_ne_inits = jnp.log(ne_inits)
        log_n_totals = jnp.log(n_totals)

        batch_results = jax.vmap(_solve_one)(T_values, log_ne_inits, log_n_totals)

        for i in range(5):
            ne_seq = seq_results[i]
            ne_batch = float(batch_results.n_e[i])
            if ne_seq > 0 and ne_batch > 0:
                rel = abs(ne_seq - ne_batch) / ne_seq
                assert rel < 1e-6, f"Batch mismatch at i={i}: seq={ne_seq}, batch={ne_batch}"


class TestResidualHistory:
    """test_residual_history: verify residual generally decreases."""

    def test_residual_decreases(self):
        result = anderson_solve(
            T_eV=1.0,
            compositions=_STANDARD_COMP,
            atomic_data=_STANDARD_ATOMIC,
            n_e_init=1e16,
            n_total_ion=1e17,
            m=3,
            tol=1e-10,
            max_iter=50,
        )
        assert result.converged

        # Get nonzero residuals
        n_iters = int(result.iterations)
        residuals = np.array(result.residual_history[:n_iters])
        residuals = residuals[residuals > 0]

        if len(residuals) > 3:
            # Allow initial transient (first 2 steps) but after that should trend down
            later = residuals[2:]
            # Check that the final residual is much smaller than the initial
            assert (
                later[-1] < residuals[0]
            ), f"Residual did not decrease: initial={residuals[0]}, final={later[-1]}"
