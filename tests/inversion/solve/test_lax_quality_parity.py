"""Lax/Python quality-gate parity tests (audit 02-F8, bead CF-LIBS-improved-cxxq).

The ``jax.lax.while_loop`` solve path previously emitted only
``r_squared_last`` + LTE keys and gated convergence only on Boltzmann
degeneracy — it could report ``converged=True`` with
``quality_metrics.get('boltzmann_r_squared') is None`` on a keystone-collapsed
composition. These tests pin the parity contract:

1. The quality_metrics KEY SETS of the two paths are identical on the same
   small solve (both clean and degenerate inputs).
2. A rigged 4-element collapse through the LAX path reports converged=False,
   ``degenerate_composition`` set, and a populated ``boltzmann_r_squared``.
3. The converged verdicts agree between the paths on both fixtures.
"""

from __future__ import annotations

import logging
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.atomic.structures import PartitionFunction  # noqa: E402
from cflibs.inversion.solve.iterative import (  # noqa: E402
    IterativeCFLIBSSolver,
    LineObservation,
)

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


@pytest.fixture
def mock_db():
    """Mirror the ``test_iterative_lax.py`` mock fixture."""
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.0  # eV
    coeffs_I = [3.2188, 0, 0, 0, 0]  # ln U = 3.2188 -> U = 25

    def _pf(el, sp):
        return PartitionFunction(
            element=el,
            ionization_stage=sp,
            coefficients=coeffs_I,
            t_min=1000,
            t_max=20000,
            source="test",
        )

    db.get_partition_coefficients.side_effect = _pf
    return db


def _clean_obs() -> list:
    """Balanced, clean two-element fixture (converges on both paths)."""
    T_eV = 1.0
    obs = []
    for el in ("A", "B"):
        for E in (1.0, 2.0, 3.0, 4.0, 5.0):
            obs.append(LineObservation(500.0, np.exp(-E / T_eV + 10.0), 0.1, el, 1, E, 1, 1e8))
    return obs


def _degenerate_obs() -> list:
    """Rigged 4-element fixture whose closure collapses onto 'A' (~99.9%)."""
    T_eV = 1.0
    obs = []
    for E in (1.0, 2.0, 3.0, 4.0, 5.0):
        obs.append(LineObservation(500.0, np.exp(-E / T_eV + 13.0), 0.1, "A", 1, E, 1, 1e8))
    for el in ("B", "C", "D"):
        for E in (1.0, 2.0, 3.0, 4.0, 5.0):
            obs.append(LineObservation(500.0, np.exp(-E / T_eV + 5.0), 0.1, el, 1, E, 1, 1e8))
    return obs


def _solve_both_paths(mock_db, obs):
    """Run the same observations through the Python and lax paths."""
    solver_py = IterativeCFLIBSSolver(mock_db, max_iterations=10, use_lax_while_loop=False)
    res_py = solver_py.solve(obs)
    solver_lax = IterativeCFLIBSSolver(mock_db, max_iterations=10, use_lax_while_loop=True)
    res_lax = solver_lax.solve(obs)
    return res_py, res_lax


@pytest.mark.parametrize("obs_factory", [_clean_obs, _degenerate_obs], ids=["clean", "degenerate"])
def test_quality_metrics_key_sets_match_between_paths(mock_db, obs_factory):
    """The two solve paths must emit IDENTICAL quality_metrics key sets."""
    res_py, res_lax = _solve_both_paths(mock_db, obs_factory())

    assert set(res_lax.quality_metrics.keys()) == set(res_py.quality_metrics.keys())
    # The keystone keys must be present (not reachable via .get() == None).
    for key in (
        "boltzmann_r_squared",
        "r_squared_last",
        "n_elements_fit",
        "degenerate_composition",
        "closure_degenerate",
        "boltzmann_degenerate",
        "ne_from_stark",
    ):
        assert key in res_lax.quality_metrics, key
        assert res_lax.quality_metrics[key] is not None, key


def test_converged_semantics_agree_between_paths(mock_db):
    """Clean input converges on both paths; the rigged collapse converges on
    NEITHER (previously the lax path green-lit it)."""
    res_py_clean, res_lax_clean = _solve_both_paths(mock_db, _clean_obs())
    assert res_py_clean.converged is True
    assert res_lax_clean.converged is True

    res_py_deg, res_lax_deg = _solve_both_paths(mock_db, _degenerate_obs())
    assert res_py_deg.converged is False
    assert res_lax_deg.converged is False
    for res in (res_py_deg, res_lax_deg):
        assert res.quality_metrics["degenerate_composition"] == pytest.approx(1.0)


def test_lax_degenerate_solve_reports_honest_metrics(mock_db, caplog):
    """LAX path on the rigged 4-element collapse: converged=False, the
    degeneracy flag set, boltzmann_r_squared POPULATED (the collapse has a
    clean slope, so R^2 is high while the composition is garbage), and the
    keystone warning logged."""
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=10, use_lax_while_loop=True)
    with caplog.at_level(logging.WARNING):
        res = solver.solve(_degenerate_obs())

    assert max(res.concentrations.values()) > 0.8  # independent dominance check
    assert res.converged is False
    assert res.quality_metrics["degenerate_composition"] == pytest.approx(1.0)
    assert res.quality_metrics["boltzmann_degenerate"] == pytest.approx(0.0)
    r2 = res.quality_metrics["boltzmann_r_squared"]
    assert r2 is not None and np.isfinite(r2) and r2 > 0.9
    assert res.quality_metrics["n_elements_fit"] == pytest.approx(4.0)
    assert "Degenerate composition" in " ".join(r.getMessage() for r in caplog.records)


def test_lax_degeneracy_threshold_constructor_parameter(mock_db):
    """The constructor threshold drives the LAX-path gate too."""
    relaxed = IterativeCFLIBSSolver(
        mock_db,
        max_iterations=10,
        use_lax_while_loop=True,
        degeneracy_dominance_threshold=0.9999,
    ).solve(_degenerate_obs())
    assert relaxed.quality_metrics["degenerate_composition"] == pytest.approx(0.0)
