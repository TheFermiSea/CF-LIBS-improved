"""Tests for the NUTS multi-chain vmap path and max-tree-depth default.

These tests pin three invariants for ``MCMCSampler`` and the underlying
``BayesianForwardModel._compute_spectrum``:

1. ``_compute_spectrum`` is safe to wrap in :func:`jax.vmap` along a
   leading "chain" axis, producing per-chain spectra whose rows match
   the unbatched per-chain calls (within float64 tolerance). This is the
   path NumPyro takes when ``MCMC`` is configured with
   ``chain_method='vectorized'``.
2. :meth:`MCMCSampler.run` constructs the underlying ``numpyro.infer.MCMC``
   with ``chain_method='vectorized'`` so multi-chain NUTS actually
   parallelises chains in a single JIT kernel on one device. This
   complements the signature-default guard in
   ``test_bayesian_forward_model_kernel_migration.py`` by introspecting
   the *constructed* ``MCMC`` kwargs instead of just the Python signature.
3. The default ``max_tree_depth`` on :meth:`MCMCSampler.run` is 8 (per the
   Wave-1 throughput rescue plan -- caps leapfrog steps per NUTS draw at
   ``2**8 = 256`` instead of ``2**10 = 1024``).

All three tests run without invoking real MCMC sampling -- they either
short-circuit ``MCMC.run`` via a stub or inspect signatures/shapes only.
"""

from __future__ import annotations

import inspect
import os
import sqlite3
import tempfile
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_jax]

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

# ---------------------------------------------------------------------------
# Minimal DB fixture (mirrors tests/inversion/test_bayesian_forward_model_kernel_migration.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def bayesian_db():
    """Tiny SQLite DB with Fe/Cu lines + Irwin partition coefficients."""
    fd, db_path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            wavelength_nm REAL,
            aki REAL,
            ei_ev REAL,
            ek_ev REAL,
            gi INTEGER,
            gk INTEGER,
            rel_int REAL,
            stark_w REAL,
            stark_alpha REAL
        )
        """)
    conn.execute("""
        CREATE TABLE species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
        """)
    conn.execute("""
        CREATE TABLE partition_functions (
            element TEXT,
            sp_num INTEGER,
            a0 REAL,
            a1 REAL,
            a2 REAL,
            a3 REAL,
            a4 REAL,
            t_min REAL,
            t_max REAL,
            source TEXT,
            PRIMARY KEY (element, sp_num)
        )
        """)
    conn.execute("""
        CREATE TABLE energy_levels (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            g_level INTEGER,
            energy_ev REAL
        )
        """)
    lines_data = [
        ("Fe", 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000, 0.02, 0.5),
        ("Fe", 1, 373.49, 5.0e6, 0.0, 3.32, 9, 9, 500, 0.015, 0.5),
        ("Cu", 1, 324.75, 1.4e8, 0.0, 3.82, 2, 4, 2000, 0.01, 0.5),
        ("Cu", 1, 327.40, 1.4e8, 0.0, 3.79, 2, 2, 1000, 0.01, 0.5),
    ]
    conn.executemany(
        "INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, "
        "gi, gk, rel_int, stark_w, stark_alpha) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        lines_data,
    )
    conn.executemany(
        "INSERT INTO species_physics (element, sp_num, ip_ev) VALUES (?, ?, ?)",
        [
            ("Fe", 1, 7.87),
            ("Fe", 2, 16.18),
            ("Cu", 1, 7.73),
            ("Cu", 2, 20.29),
        ],
    )
    conn.executemany(
        "INSERT INTO partition_functions (element, sp_num, a0, a1, a2, a3, a4) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        [
            ("Fe", 1, 1.40, 0.0, 0.0, 0.0, 0.0),
            ("Fe", 2, 1.60, 0.0, 0.0, 0.0, 0.0),
            ("Cu", 1, 0.30, 0.0, 0.0, 0.0, 0.0),
            ("Cu", 2, 0.0, 0.0, 0.0, 0.0, 0.0),
        ],
    )
    conn.commit()
    conn.close()
    try:
        yield db_path
    finally:
        Path(db_path).unlink(missing_ok=True)


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


# ---------------------------------------------------------------------------
# 2. MCMC constructed with chain_method='vectorized'
# ---------------------------------------------------------------------------


class _MCMCStub:
    """Stub that records constructor kwargs and aborts before sampling.

    Replaces ``cflibs.inversion.solve.bayesian.samplers.MCMC`` so we can
    introspect how :meth:`MCMCSampler.run` configures the underlying
    NumPyro ``MCMC`` object without spending the wall-clock cost of an
    actual NUTS run. ``run()`` raises a sentinel that the test catches
    after asserting on the captured kwargs.
    """

    captured: dict = {}

    def __init__(self, kernel, **kwargs):  # noqa: D401, ARG002
        # Stash both positional kernel and all kwargs.
        type(self).captured = {"kernel": kernel, **kwargs}
        self.num_chains = kwargs.get("num_chains", 1)

    def run(self, *args, **kwargs):  # noqa: ARG002, D401
        # Abort before any real MCMC work happens.
        raise RuntimeError("_MCMCStub.run: short-circuit (no real sampling)")


def test_mcmcsampler_uses_vectorized_chain_method(bayesian_db, monkeypatch):
    """``MCMCSampler.run`` must build the NumPyro ``MCMC`` with ``chain_method='vectorized'``.

    Patches the ``MCMC`` symbol in
    :mod:`cflibs.inversion.solve.bayesian.samplers` with a stub that
    records its constructor kwargs and raises before any sampling. With
    ``num_chains=2`` the captured ``chain_method`` kwarg must equal
    ``'vectorized'`` -- otherwise multi-chain NUTS would silently
    downgrade to a sequential single-chain run on a single GPU (bead
    ``CF-LIBS-improved-xsuj``).
    """
    from cflibs.inversion.solve.bayesian import samplers as samplers_mod
    from cflibs.inversion.solve.bayesian.forward import BayesianForwardModel
    from cflibs.inversion.solve.bayesian.priors import NoiseParameters, PriorConfig
    from cflibs.inversion.solve.bayesian.samplers import MCMCSampler

    model = BayesianForwardModel(
        db_path=bayesian_db,
        elements=["Fe", "Cu"],
        wavelength_range=(300.0, 400.0),
        pixels=32,
        instrument_fwhm_nm=0.1,
    )
    sampler = MCMCSampler(
        model,
        prior_config=PriorConfig(),
        noise_params=NoiseParameters(),
    )

    # Reset and install the stub.
    _MCMCStub.captured = {}
    monkeypatch.setattr(samplers_mod, "MCMC", _MCMCStub)

    observed = np.zeros(model.wavelength.shape, dtype=np.float64)

    # The stub raises RuntimeError on .run() *after* the MCMC constructor
    # has captured its kwargs, so we expect this call to bail out.
    with pytest.raises(RuntimeError, match="short-circuit"):
        sampler.run(
            observed,
            num_warmup=1,
            num_samples=1,
            num_chains=2,
            seed=0,
            progress_bar=False,
        )

    captured = _MCMCStub.captured
    assert captured, "MCMC stub did not capture any constructor kwargs"
    assert (
        captured.get("num_chains") == 2
    ), f"MCMCSampler.run must forward num_chains=2 to MCMC; got {captured.get('num_chains')!r}"
    assert captured.get("chain_method") == "vectorized", (
        "MCMCSampler.run must construct the NumPyro MCMC with "
        f"chain_method='vectorized'; got {captured.get('chain_method')!r}"
    )


# ---------------------------------------------------------------------------
# 3. max_tree_depth default is 8
# ---------------------------------------------------------------------------


def test_mcmcsampler_max_tree_depth_default_is_8():
    """:meth:`MCMCSampler.run`'s default ``max_tree_depth`` must be 8.

    The Wave-1 throughput rescue plan dropped the default from 10 to 8
    so NUTS caps leapfrog steps at ``2**8 = 256`` rather than
    ``2**10 = 1024`` per draw -- a ~4x walltime reduction with
    negligible posterior-quality impact for CF-LIBS likelihood
    geometries.
    """
    from cflibs.inversion.solve.bayesian.samplers import MCMCSampler

    sig = inspect.signature(MCMCSampler.run)
    assert (
        "max_tree_depth" in sig.parameters
    ), "MCMCSampler.run must expose ``max_tree_depth`` as a configurable kwarg"
    default = sig.parameters["max_tree_depth"].default
    assert (
        default == 8
    ), f"MCMCSampler.run default max_tree_depth must be 8 (Wave-1 throughput plan); got {default!r}"
