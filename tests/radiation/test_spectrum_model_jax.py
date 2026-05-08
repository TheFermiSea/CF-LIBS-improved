"""Numerical-equivalence tests for ``SpectrumModelJax``.

The JAX forward model must produce a synthetic spectrum that matches
:class:`SpectrumModel` within ``rtol=1e-5, atol=1e-7`` for a representative
multi-element LIBS plasma. This is the end-to-end gate that the benchmark
harness will exercise once GPU acceleration is wired in.
"""

from __future__ import annotations

import numpy as np
import pytest

# Uses the in-memory ``atomic_db`` fixture from ``tests/conftest.py`` (built
# from ``temp_db``); no production database file required.

jax = pytest.importorskip("jax")

from cflibs.instrument.model import InstrumentModel  # noqa: E402
from cflibs.plasma.state import SingleZoneLTEPlasma  # noqa: E402
from cflibs.radiation.profiles import BroadeningMode  # noqa: E402
from cflibs.radiation.spectrum_model import (  # noqa: E402
    SpectrumModel,
    SpectrumModelJax,
    planck_radiance,
    planck_radiance_jax,
)


def _build_plasma() -> SingleZoneLTEPlasma:
    """Realistic plasma — Si/Fe/Mg/Ca/Al composition, 10 kK / 1e16 cm^-3."""
    return SingleZoneLTEPlasma(
        T_e=10000.0,
        n_e=1.0e16,
        species={
            "Fe": 3.0e15,
            "H": 5.0e15,
        },
    )


def test_planck_radiance_jax_matches_numpy():
    wl = np.linspace(200.0, 800.0, 1000)
    T_eV = 10000.0 * 8.617333262e-5
    expected = planck_radiance(wl, T_eV)
    actual = np.asarray(planck_radiance_jax(wl, T_eV))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-7)


def test_spectrum_model_jax_matches_numpy_legacy(atomic_db):  # function-scoped DB is fine here (one call)
    """Full forward synthesis — JAX vs NumPy parity (LEGACY broadening)."""
    plasma = _build_plasma()
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    common = dict(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=200.0,
        lambda_max=800.0,
        delta_lambda=0.05,
        path_length_m=0.01,
        broadening_mode=BroadeningMode.LEGACY,
    )

    model_np = SpectrumModel(**common)
    model_jax = SpectrumModelJax(**common)

    wl_np, i_np = model_np.compute_spectrum()
    wl_jax, i_jax = model_jax.compute_spectrum()

    np.testing.assert_allclose(wl_jax, wl_np, rtol=0, atol=1e-12)
    # The convolution path uses ``signal.convolve`` (NumPy) vs
    # ``jnp.convolve`` (JAX). Both implement the same mathematical
    # operation; we allow the looser ``rtol=1e-4`` tolerance to absorb
    # the float32 default precision of JAX on CPU when x64 is off (the
    # conftest enables x64, so 1e-5 should hold).
    np.testing.assert_allclose(i_jax, i_np, rtol=1e-5, atol=1e-7)


def test_spectrum_model_jax_matches_numpy_nist_parity(atomic_db):
    """Full forward synthesis — JAX vs NumPy parity (NIST_PARITY mode).

    In NIST_PARITY there is no downstream convolution, so the two paths
    differ only in the per-line broadening kernel and radiative-transfer
    arithmetic — that is, exactly the work we want to check on GPU.
    """
    plasma = _build_plasma()
    instrument = InstrumentModel.from_resolving_power(20000.0)
    common = dict(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=200.0,
        lambda_max=800.0,
        delta_lambda=0.05,
        path_length_m=0.01,
        broadening_mode=BroadeningMode.NIST_PARITY,
    )

    model_np = SpectrumModel(**common)
    model_jax = SpectrumModelJax(**common)

    _, i_np = model_np.compute_spectrum()
    _, i_jax = model_jax.compute_spectrum()
    np.testing.assert_allclose(i_jax, i_np, rtol=1e-5, atol=1e-7)


def test_spectrum_model_jax_uses_jax_solver(atomic_db):
    """Smoke check: JAX model swaps in the JAX solver (no silent NumPy fallback)."""
    from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolverJax

    plasma = _build_plasma()
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    model = SpectrumModelJax(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=300.0,
        lambda_max=500.0,
        delta_lambda=0.1,
    )
    assert isinstance(model.solver, SahaBoltzmannSolverJax)
    # Wavelength grid must already be a jnp array on the model
    assert "jax" in type(model._wavelength_jax).__module__
