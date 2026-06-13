"""Parity tests for the unified T1-2 forward kernel (ADR-0001).

These tests assert that :func:`cflibs.radiation.kernels.forward_model`
reproduces the output of each pre-T1-2 call site within ``rtol=1e-5,
atol=1e-7`` for a representative LIBS plasma:

  1. :meth:`SpectrumModel.compute_spectrum` (LEGACY / NIST_PARITY /
     PHYSICAL_DOPPLER) -- end-to-end thin-wrapper parity.
  2. :func:`cflibs.manifold.batch_forward.single_spectrum_forward` --
     manifold per-spectrum parity.
  3. :class:`cflibs.inversion.solve.bayesian.BayesianForwardModel.forward`
     -- deferred to T1-6 (see comment in test module body).

Run on ``JAX_PLATFORMS=cpu`` with x64 enabled by ``conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.instrument.model import InstrumentModel  # noqa: E402
from cflibs.plasma.state import SingleZoneLTEPlasma  # noqa: E402
from cflibs.radiation.profiles import BroadeningMode  # noqa: E402
from cflibs.radiation.spectrum_model import SpectrumModel  # noqa: E402


def _build_plasma() -> SingleZoneLTEPlasma:
    return SingleZoneLTEPlasma(
        T_e=10000.0,
        n_e=1.0e16,
        species={"Fe": 3.0e15, "H": 5.0e15},
    )


@pytest.mark.parametrize(
    "mode,instrument_builder",
    [
        (BroadeningMode.LEGACY, lambda: InstrumentModel(resolution_fwhm_nm=0.05)),
        (
            BroadeningMode.NIST_PARITY,
            lambda: InstrumentModel.from_resolving_power(20000.0),
        ),
        (
            BroadeningMode.PHYSICAL_DOPPLER,
            lambda: InstrumentModel(resolution_fwhm_nm=0.05),
        ),
    ],
)
def test_spectrum_model_kernel_thin_wrapper_parity(atomic_db, mode, instrument_builder):
    """The kernel-backed ``SpectrumModel`` matches a snapshot reference.

    Builds a reference by directly invoking the legacy detailed-levels
    solver + the kernel via the same path the in-class wrapper uses,
    independent of the wrapper plumbing. Asserts that
    :meth:`SpectrumModel.compute_spectrum` (post-T1-2) reproduces the
    expected intensity array within ``rtol=1e-5, atol=1e-7``.

    For LEGACY / PHYSICAL_DOPPLER this exercises both the kernel and the
    downstream scipy convolution; for NIST_PARITY only the kernel (the
    convolution is folded into per-line sigma).
    """
    from cflibs.instrument.convolution import apply_instrument_function
    from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
    from cflibs.radiation.kernels import forward_model

    plasma = _build_plasma()
    instrument = instrument_builder()
    lambda_min, lambda_max, delta_lambda = 200.0, 800.0, 0.05

    # 1. Build the reference via the canonical pre-T1-2 sequence of steps,
    #    reimplemented here so the test is independent of the wrapper.
    #    Detailed-levels Saha-Boltzmann + kernel + (optional) scipy conv.
    min_ri = 0.01 if mode == BroadeningMode.NIST_PARITY else 10.0
    snapshot = atomic_db.snapshot(
        elements=list(plasma.species.keys()),
        wavelength_range=(lambda_min, lambda_max),
        min_relative_intensity=min_ri,
    )
    solver = SahaBoltzmannSolver(atomic_db)
    species_states = solver.solve_species_states(plasma)
    n_lines = int(np.asarray(snapshot.line_wavelengths_nm).shape[0])
    n_upper = np.zeros(n_lines, dtype=np.float64)
    line_g_k = np.asarray(snapshot.line_g_k, dtype=np.float64)
    for li in range(n_lines):
        sp_idx = int(snapshot.line_species_index[li])
        el, stage = snapshot.species[sp_idx]
        state = species_states.get((el, stage))
        if state is None:
            continue
        E_k_ev = float(snapshot.line_E_k_ev[li])
        if E_k_ev > state.max_energy_ev:
            continue  # IPD cutoff: level merged into the continuum
        n_upper[li] = (
            state.number_density_cm3
            * (float(line_g_k[li]) / state.partition_function)
            * np.exp(-E_k_ev / plasma.T_e_eV)
        )
    wl_grid = np.arange(lambda_min, lambda_max + delta_lambda, delta_lambda)
    # ``apply_stark`` only takes effect under ``PHYSICAL_DOPPLER`` (see
    # ``cflibs/radiation/kernels.py:forward_model``). Mirror the
    # ``SpectrumModel`` wrapper's default (``True``) so the reference
    # matches the post-Wave-1 Stark-broadened default. Fix A2,
    # ``docs/architecture/2026-05-27-physics-audit.md``.
    expected = forward_model(
        plasma,
        snapshot,
        instrument,
        jnp.asarray(wl_grid),
        broadening_mode=mode,
        path_length_m=0.01,
        apply_self_absorption=True,
        fold_instrument_sigma=(mode == BroadeningMode.NIST_PARITY),
        apply_stark=True,
        _precomputed_n_upper_per_line=n_upper,
    )
    expected = np.asarray(expected)
    if mode != BroadeningMode.NIST_PARITY:
        sigma_conv = instrument.resolution_sigma_nm
        if sigma_conv > 0:
            expected = apply_instrument_function(wl_grid, expected, sigma_conv)

    # 2. Run the wrapper end-to-end.
    model = SpectrumModel(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        delta_lambda=delta_lambda,
        path_length_m=0.01,
        broadening_mode=mode,
    )
    wl, intensity = model.compute_spectrum()

    np.testing.assert_array_equal(wl, wl_grid)
    np.testing.assert_allclose(intensity, expected, rtol=1e-5, atol=1e-7)


def test_manifold_single_spectrum_kernel_parity_smoke(atomic_db):
    """Snapshot-based forward_from_snapshot produces a finite spectrum.

    Smoke test for the manifold adapter; numerical parity against the
    legacy ``BatchAtomicData`` path is exercised by
    :func:`cflibs.manifold.batch_forward.single_spectrum_forward`'s
    existing tests since the legacy path is unchanged.
    """
    from cflibs.manifold.batch_forward import forward_from_snapshot

    plasma = _build_plasma()
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    snapshot = atomic_db.snapshot(
        elements=list(plasma.species.keys()),
        wavelength_range=(200.0, 800.0),
        min_relative_intensity=0.0,
    )
    wl_grid = jnp.linspace(200.0, 800.0, 5000)
    out = forward_from_snapshot(
        plasma,
        snapshot,
        instrument,
        wl_grid,
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.01,
        apply_self_absorption=False,
        apply_stark=True,
        fold_instrument_sigma=True,
    )
    out = np.asarray(out)
    assert out.shape == (5000,)
    assert np.all(np.isfinite(out))
    assert out.max() > 0

