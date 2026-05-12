"""JAX-path tests for :func:`bic_prune_elements` and
:func:`_solve_nnls_subset` in ``cflibs.inversion.identify.model_selection``.

The JAX path is opt-in via ``use_jax_nnls=True``; an additional
``jax_batch_trials=True`` flag triggers vmapped pre-computation of every
leave-one-out trial in the backward-elimination sweep. All three paths
(scipy, jax sequential, jax batched) should converge to the same
selected element set and within ~1e-4 rtol on concentrations.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import nnls as scipy_nnls

from cflibs.inversion.identify.model_selection import (
    _solve_nnls_subset,
    bic_prune_elements,
)
from cflibs.inversion.identify.spectral_nnls import _HAS_JAX

pytestmark = pytest.mark.skipif(not _HAS_JAX, reason="JAX not installed")


@pytest.fixture
def synthetic_basis():
    """A synthetic basis with 8 elements (3 truly active) and 4 continuum
    columns. Coefficients chosen so backward elimination must remove
    spurious elements."""
    rng = np.random.default_rng(7)
    n_pixels = 500
    n_elements = 8
    n_continuum = 4
    n_components = n_elements + n_continuum

    basis = np.abs(rng.normal(size=(n_components, n_pixels))) * 0.01 + 1e-4

    true_coeffs = np.zeros(n_components)
    true_coeffs[[0, 3, 5]] = [1.0, 0.5, 0.7]
    true_coeffs[n_elements] = 0.2  # continuum

    observed = basis.T @ true_coeffs + 0.005 * rng.normal(size=n_pixels)
    elements = [f"E{i}" for i in range(n_elements)]
    coeffs_init, _ = scipy_nnls(basis.T, observed)

    return {
        "observed": observed,
        "basis": basis,
        "elements": elements,
        "coeffs_init": coeffs_init,
        "noise_var": 0.005**2,
        "expected_active": {"E0", "E3", "E5"},
    }


def test_solve_nnls_subset_jax_matches_scipy(synthetic_basis):
    """The internal _solve_nnls_subset helper must agree across paths."""
    sb = synthetic_basis
    n_elements = len(sb["elements"])
    mask = np.zeros(n_elements, dtype=bool)
    mask[[0, 3, 5]] = True

    c_sp, p_sp = _solve_nnls_subset(
        sb["observed"], sb["basis"], mask, n_elements, use_jax_nnls=False
    )
    c_jx, p_jx = _solve_nnls_subset(
        sb["observed"],
        sb["basis"],
        mask,
        n_elements,
        use_jax_nnls=True,
        jax_nnls_max_iter=500,
    )
    # Predicted spectrum is the true invariant.
    assert np.allclose(p_jx, p_sp, rtol=1e-4, atol=1e-7)
    # Coefficients should match closely on this well-conditioned problem.
    assert np.allclose(c_jx, c_sp, rtol=1e-3, atol=1e-7)


def test_solve_nnls_subset_empty_mask_handled(synthetic_basis):
    """Empty active mask + zero continuum -> empty solve, zero predicted."""
    sb = synthetic_basis
    # Force the entire basis to be empty by passing a basis with zero rows
    empty_basis = sb["basis"][:0, :]
    empty_mask = np.zeros(0, dtype=bool)
    c, p = _solve_nnls_subset(
        sb["observed"], empty_basis, empty_mask, 0, use_jax_nnls=True
    )
    assert c.shape == (0,)
    assert np.array_equal(p, np.zeros_like(sb["observed"]))


def test_bic_prune_jax_selects_same_elements(synthetic_basis):
    sb = synthetic_basis
    res_sp = bic_prune_elements(
        sb["observed"],
        sb["basis"],
        sb["elements"],
        sb["coeffs_init"],
        sb["noise_var"],
        use_jax_nnls=False,
    )
    res_jx = bic_prune_elements(
        sb["observed"],
        sb["basis"],
        sb["elements"],
        sb["coeffs_init"],
        sb["noise_var"],
        use_jax_nnls=True,
        jax_nnls_max_iter=500,
    )
    assert set(res_jx.selected_elements) == set(res_sp.selected_elements)
    assert set(res_jx.selected_elements) == sb["expected_active"]


def test_bic_prune_jax_batched_same_result(synthetic_basis):
    sb = synthetic_basis
    res_sp = bic_prune_elements(
        sb["observed"],
        sb["basis"],
        sb["elements"],
        sb["coeffs_init"],
        sb["noise_var"],
        use_jax_nnls=False,
    )
    res_jb = bic_prune_elements(
        sb["observed"],
        sb["basis"],
        sb["elements"],
        sb["coeffs_init"],
        sb["noise_var"],
        use_jax_nnls=True,
        jax_batch_trials=True,
        jax_nnls_max_iter=500,
    )
    assert set(res_jb.selected_elements) == set(res_sp.selected_elements)
    # BIC should be very close
    assert res_jb.bic_final == pytest.approx(res_sp.bic_final, rel=1e-6)


def test_bic_prune_jax_concentrations_close(synthetic_basis):
    sb = synthetic_basis
    res_sp = bic_prune_elements(
        sb["observed"],
        sb["basis"],
        sb["elements"],
        sb["coeffs_init"],
        sb["noise_var"],
        use_jax_nnls=False,
    )
    res_jx = bic_prune_elements(
        sb["observed"],
        sb["basis"],
        sb["elements"],
        sb["coeffs_init"],
        sb["noise_var"],
        use_jax_nnls=True,
        jax_nnls_max_iter=500,
    )
    for el, c_sp in res_sp.concentrations.items():
        assert el in res_jx.concentrations
        assert res_jx.concentrations[el] == pytest.approx(c_sp, rel=1e-3, abs=1e-6)


def test_bic_prune_jax_bic_close(synthetic_basis):
    sb = synthetic_basis
    res_sp = bic_prune_elements(
        sb["observed"],
        sb["basis"],
        sb["elements"],
        sb["coeffs_init"],
        sb["noise_var"],
        use_jax_nnls=False,
    )
    res_jx = bic_prune_elements(
        sb["observed"],
        sb["basis"],
        sb["elements"],
        sb["coeffs_init"],
        sb["noise_var"],
        use_jax_nnls=True,
        jax_nnls_max_iter=500,
    )
    assert res_jx.bic_final == pytest.approx(res_sp.bic_final, rel=1e-6)
    assert res_jx.bic_initial == pytest.approx(res_sp.bic_initial, rel=1e-6)


def test_bic_prune_jax_handles_empty_active():
    """If all coefficients are zero, both paths must return empty."""
    n_elements = 5
    n_continuum = 2
    n_pixels = 100
    basis = np.abs(np.random.default_rng(0).normal(size=(n_elements + n_continuum, n_pixels))) * 0.01
    observed = np.zeros(n_pixels)
    coeffs = np.zeros(n_elements + n_continuum)
    elements = [f"X{i}" for i in range(n_elements)]
    res = bic_prune_elements(observed, basis, elements, coeffs, 1e-6, use_jax_nnls=True)
    assert res.selected_elements == []
    assert set(res.removed_elements) == set(elements)
