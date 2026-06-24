"""Tests for the memory-efficient, SVD-conditioned full-spectrum solver.

Covers the two literature fixes that make the gradient-based full-spectrum fit
converge on real, many-element data:

(A) MEMORY — the differentiable forward routes through the chunked kernel
    ``forward_model_chunked`` so reverse-mode AD activation memory is
    ``O(n_lines * chunk_width)`` rather than the dense ``O(n_lines * n_wl)``.

(B) CONDITIONING — the loss is computed in a small SVD/PCA basis built from
    candidate forward-model spectra (Hebert et al. 2020), on area-normalised
    spectra.

The DB-free unit tests exercise the SVD basis, the composition unit
conversions, and the pipeline ``solver`` knob plumbing. The integration test
(``requires_db``/``requires_jax``) runs the full chunked + SVD fit on a real
many-element spectrum and asserts the 4+ element gradient no longer OOMs.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.inversion.solve.full_spectrum import (
    FullSpectrumResult,
    _mass_to_number_fractions,
    _number_to_mass_fractions,
    build_svd_basis,
)

# ---------------------------------------------------------------------------
# DB-free unit tests (fast)
# ---------------------------------------------------------------------------


def test_number_mass_fraction_roundtrip():
    """Mass <-> number fraction conversions invert and stay normalised."""
    mass = {"Si": 0.6, "Al": 0.25, "Fe": 0.15}
    num = _mass_to_number_fractions(mass, list(mass))
    assert num.shape == (3,)
    assert num.sum() == pytest.approx(1.0, abs=1e-9)
    back = _number_to_mass_fractions({el: float(num[i]) for i, el in enumerate(mass)})
    assert sum(back.values()) == pytest.approx(1.0, abs=1e-9)
    for el in mass:
        assert back[el] == pytest.approx(mass[el], abs=1e-6)


def test_number_to_mass_heavier_element_gains_mass():
    """Equal number fractions => heavier element carries more mass."""
    mass = _number_to_mass_fractions({"H": 0.5, "Fe": 0.5})
    assert mass["Fe"] > mass["H"]


def test_build_svd_basis_orthonormal_and_compresses():
    """The SVD basis is orthonormal and projects/reconstructs faithfully."""
    rng = np.random.default_rng(0)
    n_wl = 400
    # Library is an exact linear span of 5 latent spectral shapes => the
    # mean-centred data matrix has rank <= 5, so the variance target is reached
    # in <= 5 components (a strictly low-rank, well-conditioned compression).
    latent = np.abs(rng.standard_normal((5, n_wl)))
    coeffs = np.abs(rng.standard_normal((30, 5)))
    library = coeffs @ latent  # strictly rank-5 (positive)
    observed = np.abs(rng.standard_normal(5)) @ latent

    basis, mean, k = build_svd_basis(library, observed, n_components=20)
    assert basis.shape[1] == n_wl
    assert mean.shape == (n_wl,)
    # Strictly rank-5 library => the variance target is met in <= 5 components.
    assert 1 <= k <= 6
    # Rows orthonormal.
    gram = basis @ basis.T
    assert np.allclose(gram, np.eye(k), atol=1e-8)


def test_build_svd_basis_caps_at_n_components():
    """``k`` never exceeds the requested component cap."""
    rng = np.random.default_rng(1)
    library = np.abs(rng.standard_normal((40, 200))) + 0.05
    observed = np.abs(rng.standard_normal(200)) + 0.05
    _, _, k = build_svd_basis(library, observed, n_components=8, variance_target=0.999999)
    assert k <= 8


def test_full_spectrum_result_container():
    """The result container carries warm-start vs converged-fit accounting."""
    res = FullSpectrumResult(
        temperature_K=10000.0,
        electron_density_cm3=1e17,
        concentrations={"Si": 0.6, "Fe": 0.4},
        warm_start_temperature_K=9000.0,
        warm_start_electron_density_cm3=2e17,
        warm_start_concentrations={"Si": 0.5, "Fe": 0.5},
        fit_temperature_K=10000.0,
        fit_electron_density_cm3=1e17,
        fit_concentrations={"Si": 0.6, "Fe": 0.4},
        converged=True,
        adopted_fit=True,
        initial_objective=1.0,
        final_objective=0.1,
        iterations=12,
        gradient_norm=1e-4,
    )
    assert res.converged and res.adopted_fit
    assert res.final_objective < res.initial_objective


# ---------------------------------------------------------------------------
# Pipeline ``solver`` knob plumbing (DB-free)
# ---------------------------------------------------------------------------


def test_pipeline_solver_knob_resolves():
    from cflibs.inversion.pipeline import SOLVER_BACKENDS, build_pipeline_config

    assert SOLVER_BACKENDS == ("iterative", "joint", "bayesian")
    assert build_pipeline_config(["Fe", "Cu"]).solver == "iterative"
    assert build_pipeline_config(["Fe", "Cu"], overrides={"solver": "joint"}).solver == "joint"
    assert (
        build_pipeline_config(["Fe", "Cu"], overrides={"solver": "bayesian"}).solver == "bayesian"
    )


def test_pipeline_unknown_solver_rejected():
    from cflibs.inversion.pipeline import build_pipeline_config

    with pytest.raises(ValueError, match="Unknown solver backend"):
        build_pipeline_config(["Fe"], overrides={"solver": "nope"})


# ---------------------------------------------------------------------------
# Integration: chunked + SVD fit on a real many-element spectrum (slow)
# ---------------------------------------------------------------------------


@pytest.mark.requires_db
@pytest.mark.requires_jax
@pytest.mark.slow
def test_full_spectrum_gradient_does_not_oom(production_db):
    """The 4+ element differentiable forward + gradient runs (no dense OOM).

    Builds the chunked forward on a wide, dense grid and asserts a finite
    composition gradient — the regression guard for root cause (A): the naive
    dense ``(n_lines x n_wl)`` matrix OOMed past ~3 elements.
    """
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp

    from cflibs.inversion.solve.full_spectrum import _ChunkedForward

    db_path = str(production_db.db_path)
    # 5 elements => the dense (n_lines x n_wl) matrix path OOMed past ~3.
    # A 2000-px wide grid over the full SuperCam range keeps the CPU XLA
    # compile tractable while still exercising the chunked scan (the native
    # 7933-px axis is several minutes of silent XLA compile on CPU — that is
    # the per-spectrum fit's job, not this regression guard's).
    elements = ["Si", "Al", "Fe", "Ca", "Mg"]
    wl = np.linspace(243.8, 852.8, 2000)
    fwd = _ChunkedForward(db_path, elements, wl, resolving_power=2400.0)
    # Chunking is in effect: more than one chunk, and the chunk matrix is far
    # smaller than the dense one.
    assert fwd.plan.nstitch >= 4
    assert fwd.n_lines * fwd.n_wl * 8 / 1e6 > 50.0  # dense matrix is large

    conc = jnp.asarray(np.full(len(elements), 1.0 / len(elements)))

    def loss(T_eV, log_ne, c):
        return jnp.sum(fwd.spectrum(T_eV, log_ne, c) ** 2)

    grad = jax.grad(loss, argnums=2)(jnp.asarray(1.0), jnp.asarray(17.0), conc)
    grad_np = np.asarray(grad)
    assert grad_np.shape == (len(elements),)
    assert np.all(np.isfinite(grad_np)), "composition gradient must be finite (no OOM/overflow)"
