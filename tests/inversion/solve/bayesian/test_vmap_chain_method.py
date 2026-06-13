"""Test for the NUTS multi-chain vmap path.

``_compute_spectrum`` must be safe to wrap in :func:`jax.vmap` along a
leading "chain" axis, producing per-chain spectra whose rows match the
unbatched per-chain calls (within float64 tolerance). This is the path
NumPyro takes when ``MCMC`` is configured with
``chain_method='vectorized'`` -- a silent broadcast here would mix
compositions across chains.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_jax]

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

# The ``bayesian_db`` fixture is supplied by ``tests/inversion/conftest.py``
# (shared with ``tests/inversion/test_bayesian_forward_model_kernel_migration.py``).

# ---------------------------------------------------------------------------
# 1. vmap correctness: per-chain output matches unbatched runs
# ---------------------------------------------------------------------------


def test_compute_spectrum_vmap_safe(bayesian_db):
    """``_compute_spectrum`` under :func:`jax.vmap` must match per-chain calls.

    Builds a tiny ``BayesianForwardModel`` (2 elements, 32 pixels) and
    wraps :meth:`BayesianForwardModel._compute_spectrum` in
    ``jax.vmap(..., in_axes=(None, None, 0))`` over a 2-chain
    concentrations array of shape ``(2, 2)``. Asserts:

    * The vmapped output has the expected leading chain axis
      ``(num_chains, n_wavelengths)``.
    * Each chain's spectrum is bitwise-close (rtol=1e-10) to the
      corresponding unbatched ``_compute_spectrum`` call -- guards
      against silent broadcasting that would mix compositions
      across chains.
    """
    from cflibs.inversion.solve.bayesian.forward import BayesianForwardModel

    pixels = 32
    elements = ["Fe", "Cu"]
    model = BayesianForwardModel(
        db_path=bayesian_db,
        elements=elements,
        wavelength_range=(300.0, 400.0),
        pixels=pixels,
        instrument_fwhm_nm=0.1,
    )

    t_ev = 1.0  # noqa: N806
    n_e = 1.0e17
    concentrations_batched = jnp.array([[0.7, 0.3], [0.4, 0.6]])
    num_chains = concentrations_batched.shape[0]

    batched_compute = jax.vmap(model._compute_spectrum, in_axes=(None, None, 0))
    spectra_vmap = batched_compute(t_ev, n_e, concentrations_batched)

    spectra_vmap_np = np.asarray(spectra_vmap)
    assert spectra_vmap_np.shape == (num_chains, pixels), (
        f"vmap output must be (num_chains, n_wavelengths)={(num_chains, pixels)}, "
        f"got {spectra_vmap_np.shape}"
    )
    assert np.all(np.isfinite(spectra_vmap_np))
    assert np.all(spectra_vmap_np >= 0.0)

    # Per-chain parity: each row of the vmapped output must match the
    # unbatched ``_compute_spectrum`` call for the same composition.
    for i in range(num_chains):
        spectrum_unbatched = np.asarray(
            model._compute_spectrum(t_ev, n_e, concentrations_batched[i])
        )
        np.testing.assert_allclose(
            spectra_vmap_np[i],
            spectrum_unbatched,
            rtol=1e-10,
            atol=1e-12,
            err_msg=f"vmap chain {i} diverges from unbatched _compute_spectrum",
        )

    # Distinct compositions must yield distinct spectra (guards against a
    # silent broadcast of one chain's composition across both rows).
    assert not np.allclose(
        spectra_vmap_np[0], spectra_vmap_np[1]
    ), "Vmap over distinct concentrations must yield distinct spectra"

