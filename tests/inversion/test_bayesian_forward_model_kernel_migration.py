"""Drift guard for the T1-6 BayesianForwardModel kernel migration.

ADR-0001 T1-6 (bead ``CF-LIBS-improved-789h``) migrated
:meth:`BayesianForwardModel._compute_spectrum` so it now dispatches to
:func:`cflibs.radiation.kernels.forward_model` directly instead of
re-implementing Saha-Boltzmann + Voigt summation locally and reaching
through :func:`_atomic_data_arrays_from_snapshot`.

These tests pin two invariants:

1. ``_compute_spectrum`` produces the same array as a hand-built
   :func:`forward_model` call with the same plasma state, snapshot,
   instrument, and broadening knobs (drift guard).
2. The legacy ``_atomic_data_arrays_from_snapshot`` adapter is no longer
   referenced from the ``forward.py`` source body -- only from
   re-exports and back-compat shims (regression guard).

Pre-migration bitwise parity is *not* asserted: the snapshot path uses
canonical Irwin (base-10 log) partition coefficients while the pre-T1-6
``_compute_spectrum`` interpreted the same coefficients as natural-log
Irwin. That convention change is the entire point of the migration; the
docstring on :meth:`BayesianForwardModel._compute_spectrum` notes the
absolute-scale change for posterity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytestmark = [
    pytest.mark.requires_jax,
]

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

# The ``bayesian_db`` fixture is supplied by ``tests/inversion/conftest.py``
# (shared with ``tests/inversion/solve/bayesian/test_vmap_chain_method.py``).

# ---------------------------------------------------------------------------
# Drift guard: _compute_spectrum == forward_model(...)
# ---------------------------------------------------------------------------


def test_compute_spectrum_matches_direct_forward_model_call(bayesian_db):
    """``_compute_spectrum`` must equal a hand-built ``forward_model`` call.

    Pins the migration target: the body of ``_compute_spectrum`` is now a
    thin wrapper around :func:`cflibs.radiation.kernels.forward_model`. If
    a future refactor accidentally re-introduces a divergent local
    summation path, this assertion will fail.
    """
    from cflibs.core.constants import EV_TO_K
    from cflibs.inversion.solve.bayesian.forward import BayesianForwardModel
    from cflibs.plasma.state import SingleZoneLTEPlasma
    from cflibs.radiation.kernels import forward_model
    from cflibs.radiation.profiles import BroadeningMode

    elements = ["Fe", "Cu"]
    wl_range = (200.0, 600.0)
    pixels = 256

    model = BayesianForwardModel(
        db_path=bayesian_db,
        elements=elements,
        wavelength_range=wl_range,
        pixels=pixels,
        instrument_fwhm_nm=0.05,
    )

    T_eV = 1.0
    n_e = 1.0e17
    concentrations = jnp.array([0.7, 0.3])

    spectrum_via_method = model._compute_spectrum(T_eV, n_e, concentrations)

    # Mirror exactly what ``_compute_spectrum`` does internally so we are
    # asserting "no extra physics has been spliced in" rather than re-doing
    # the kernel arithmetic by hand.
    T_eV_jnp = jnp.asarray(T_eV, dtype=spectrum_via_method.dtype)
    n_e_jnp = jnp.asarray(n_e, dtype=spectrum_via_method.dtype)
    total_density = n_e_jnp
    plasma = object.__new__(SingleZoneLTEPlasma)
    plasma.T_e = T_eV_jnp * EV_TO_K
    plasma.n_e = n_e_jnp
    plasma.species = {el: concentrations[i] * total_density for i, el in enumerate(elements)}
    plasma.T_g = None
    plasma.pressure = None
    raw = forward_model(
        plasma,
        model.snapshot,
        model.instrument,
        model.wavelength,
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
        path_length_m=0.0,
        apply_self_absorption=False,
        fold_instrument_sigma=True,
        apply_stark=True,
        total_species_density_cm3=total_density,
    )
    spectrum_via_kernel = jnp.clip(raw, 0.0, 1e12)

    np.testing.assert_allclose(
        np.asarray(spectrum_via_method),
        np.asarray(spectrum_via_kernel),
        rtol=1e-6,
        atol=1e-12,
    )


def test_compute_spectrum_is_finite_and_nonnegative(bayesian_db):
    """Spectrum must be finite and non-negative on a representative input."""
    from cflibs.inversion.solve.bayesian.forward import BayesianForwardModel

    model = BayesianForwardModel(
        db_path=bayesian_db,
        elements=["Fe", "Cu"],
        wavelength_range=(200.0, 600.0),
        pixels=256,
        instrument_fwhm_nm=0.05,
    )
    spectrum = model.forward(
        T_eV=1.0,
        log_ne=17.0,
        concentrations=jnp.array([0.7, 0.3]),
    )
    arr = np.asarray(spectrum)
    assert arr.shape == (256,)
    assert np.all(np.isfinite(arr))
    assert np.all(arr >= 0.0)
    # Lines fall inside the wavelength range, so some emission must be > 0.
    assert arr.max() > 0.0


# ---------------------------------------------------------------------------
# CF-LIBS-improved-vjbh — Stark T-power-law factor regression guard
# ---------------------------------------------------------------------------


def test_stark_gamma_applies_temperature_power_law(bayesian_db):
    """``_per_line_stark_gamma`` must apply ``factor_T = (T/T_ref)^(-alpha)``.

    Pre-fix (CF-LIBS-improved-vjbh): the kernel helper returned
    ``stark_w * (n_e / 1e16)`` unconditionally, dropping the temperature
    dependence that the legacy ``BayesianForwardModel._compute_spectrum``
    had applied. Snapshots built via ``AtomicDatabase.snapshot`` now carry
    ``line_stark_alpha`` and the kernel applies the canonical Griem
    ``(T_eV / 0.86173 eV)^(-alpha)`` factor.

    Test fixture ``bayesian_db`` populates ``stark_alpha = 0.5`` for the
    Fe I and Cu I lines and ``0.6`` for Fe II, so we get a per-line check
    of the factor across two different alpha values.
    """
    from cflibs.atomic import AtomicDatabase
    from cflibs.radiation.kernels import _per_line_stark_gamma

    snapshot = AtomicDatabase(bayesian_db).snapshot(
        elements=["Fe", "Cu"], wavelength_range=(200.0, 600.0)
    )
    alpha = np.asarray(snapshot.line_stark_alpha)
    stark_w = np.asarray(snapshot.line_stark_w)
    assert alpha.shape == stark_w.shape
    # Fixture has Fe I (0.5), Fe II (0.6), Cu I (0.5) — all alpha > 0 so
    # the T-factor is non-trivial line-by-line.
    assert np.all(alpha > 0.0), f"Test fixture alphas must be non-zero: {alpha}"

    n_e = 1.0e17
    REF_T_EV = 0.86173

    # Evaluate at two temperatures bracketing the canonical 0.86 eV reference.
    T_cold = 0.5
    T_hot = 2.0
    gamma_cold = np.asarray(_per_line_stark_gamma(snapshot, n_e, T_cold))
    gamma_hot = np.asarray(_per_line_stark_gamma(snapshot, n_e, T_hot))

    # Direct formula reference: gamma_S = stark_w * (n_e / 1e16) * (T/T_ref)^(-alpha)
    base = stark_w * (n_e / 1.0e16)
    expected_cold = base * np.power(T_cold / REF_T_EV, -alpha)
    expected_hot = base * np.power(T_hot / REF_T_EV, -alpha)

    np.testing.assert_allclose(gamma_cold, expected_cold, rtol=1e-6, atol=0.0)
    np.testing.assert_allclose(gamma_hot, expected_hot, rtol=1e-6, atol=0.0)

    # Cross-check: gamma_cold / gamma_hot must equal (T_hot/T_cold)^alpha line-by-line.
    # (Equivalent to the legacy formula but expressed as a ratio so REF_T_EV cancels.)
    ratio = gamma_cold / np.maximum(gamma_hot, 1e-30)
    expected_ratio = np.power(T_hot / T_cold, alpha)
    np.testing.assert_allclose(ratio, expected_ratio, rtol=1e-6, atol=0.0)


def test_stark_gamma_t_clamped_at_low_temperature(bayesian_db):
    """``_per_line_stark_gamma`` must clamp ``T_eV`` to ``>= 0.1`` to match
    legacy behaviour (avoids ``(T/T_ref)^(-alpha)`` blowing up to infinity
    at ``T -> 0`` while the rest of the forward model still emits zero
    population from Boltzmann/Saha).
    """
    from cflibs.atomic import AtomicDatabase
    from cflibs.radiation.kernels import _per_line_stark_gamma

    snapshot = AtomicDatabase(bayesian_db).snapshot(
        elements=["Fe", "Cu"], wavelength_range=(200.0, 600.0)
    )
    n_e = 1.0e17

    gamma_below_clamp = np.asarray(_per_line_stark_gamma(snapshot, n_e, 0.05))
    gamma_at_clamp = np.asarray(_per_line_stark_gamma(snapshot, n_e, 0.1))
    # T=0.05 and T=0.1 both clamp to T=0.1 -> identical gamma.
    np.testing.assert_allclose(gamma_below_clamp, gamma_at_clamp, rtol=1e-12, atol=0.0)
    # And neither blows up.
    assert np.all(np.isfinite(gamma_below_clamp))


def test_stark_gamma_alpha_zero_collapses_to_legacy_formula(bayesian_db):
    """When ``line_stark_alpha == 0`` the kernel must reduce to the legacy
    temperature-independent formula ``gamma = stark_w * (n_e / 1e16)``.

    The :meth:`AtomicDatabase.snapshot` builder defaults ``stark_alpha = 0``
    for any DB row without catalogued temperature dependence, so this is
    the canonical "missing data" branch — and it falls out of the math
    automatically (``T^0 = 1``) without any kernel-side special-casing.
    """
    from cflibs.atomic import AtomicDatabase
    from cflibs.radiation.kernels import _per_line_stark_gamma

    snapshot = AtomicDatabase(bayesian_db).snapshot(
        elements=["Fe", "Cu"], wavelength_range=(200.0, 600.0)
    )
    # Overwrite alpha array to zero to exercise the "no T-dependence" path
    # without needing a separate DB fixture.
    snap = type(snapshot)(
        **{
            **{f: getattr(snapshot, f) for f in snapshot.__dataclass_fields__},
            "line_stark_alpha": jnp.zeros_like(snapshot.line_stark_alpha),
        }
    )
    n_e = 1.0e17
    gamma_T05 = np.asarray(_per_line_stark_gamma(snap, n_e, 0.5))
    gamma_T20 = np.asarray(_per_line_stark_gamma(snap, n_e, 2.0))
    # Equal across temperature because alpha=0.
    np.testing.assert_allclose(gamma_T05, gamma_T20, rtol=1e-12, atol=0.0)
    expected = np.asarray(snap.line_stark_w) * (n_e / 1.0e16)
    np.testing.assert_allclose(gamma_T05, expected, rtol=1e-12, atol=0.0)


def test_forward_py_body_does_not_call_adapter():
    """``forward.py`` body must no longer reference the adapter (regression).

    The adapter remains exported from
    :mod:`cflibs.inversion.solve.bayesian` for callers outside this code
    path, but :mod:`cflibs.inversion.solve.bayesian.forward` must not
    invoke it. We assert on the *body* (everything after the module
    docstring) to allow the docstring to keep the breadcrumb.
    """
    import cflibs.inversion.solve.bayesian.forward as fwd_mod

    source = Path(fwd_mod.__file__).read_text(encoding="utf-8")
    # Drop the module-level docstring (between the first two triple-quotes).
    first = source.find('"""')
    second = source.find('"""', first + 3)
    assert first != -1 and second != -1, "Could not locate module docstring"
    body = source[second + 3 :]
    assert "_atomic_data_arrays_from_snapshot(" not in body, (
        "BayesianForwardModel.forward.py must not call "
        "_atomic_data_arrays_from_snapshot after the T1-6 migration."
    )


# ---------------------------------------------------------------------------
# Bead xsuj — chain_method='vectorized' guard (NUTS multi-chain on one GPU)
# ---------------------------------------------------------------------------


def test_compute_spectrum_supports_vmap_chain_axis(bayesian_db):
    """``_compute_spectrum`` must broadcast cleanly under ``jax.vmap``.

    What this test covers
    ---------------------
    *Only* vmap correctness: namely, that wrapping
    ``_compute_spectrum`` in ``jax.vmap(..., in_axes=(None, None, 0))``
    over a 2-D ``concentrations`` of shape ``(num_chains, n_elements)``
    yields per-chain spectra of shape ``(num_chains, n_wavelengths)``
    whose rows differ when the input compositions differ. This is the
    code path NumPyro uses when ``MCMCSampler`` runs NUTS with
    ``chain_method='vectorized'`` (the project default; see
    :func:`test_mcmc_sampler_default_chain_method_is_vectorized`).

    What this test does NOT cover
    -----------------------------
    Manual (non-vmap) batching is **not** validated here, and is in
    fact unsupported by ``_compute_spectrum`` -- see the "Batching
    contract" section of that method's docstring. PR #186 swapped a
    per-index ``concentrations[i] * total_density`` for a broadcast
    ``(concentrations * total_density)[..., i]``; both forms pass
    *this* test because ``jax.vmap`` re-traces the function under a
    leading axis and lifts scalar reads regardless. The test's
    discriminating power is therefore against gross shape regressions
    (e.g. accidentally returning a single spectrum) and against
    silently broadcasting one chain's composition over both rows.
    """
    from cflibs.inversion.solve.bayesian.forward import BayesianForwardModel

    pixels = 64
    elements = ["Fe", "Cu"]
    model = BayesianForwardModel(
        db_path=bayesian_db,
        elements=elements,
        wavelength_range=(200.0, 600.0),
        pixels=pixels,
        instrument_fwhm_nm=0.05,
    )

    num_chains = 2
    concentrations_batched = jnp.array([[0.7, 0.3], [0.4, 0.6]])

    batched_compute = jax.vmap(model._compute_spectrum, in_axes=(None, None, 0))
    spectra = batched_compute(1.0, 1.0e17, concentrations_batched)

    spectra_np = np.asarray(spectra)
    assert spectra_np.shape == (num_chains, pixels), (
        f"vmap output must be (num_chains, n_wavelengths)={(num_chains, pixels)}, "
        f"got {spectra_np.shape}"
    )
    assert np.all(np.isfinite(spectra_np))
    assert np.all(spectra_np >= 0.0)
    # The two chains have different concentrations, so the resulting
    # spectra must differ -- guards against silently broadcasting one
    # chain's compositions over both rows.
    assert not np.allclose(
        spectra_np[0], spectra_np[1]
    ), "Vmap over distinct concentrations must yield distinct spectra"


def test_mcmc_sampler_default_chain_method_is_vectorized():
    """``MCMCSampler.run`` defaults ``chain_method`` to ``'vectorized'``.

    Required so multi-chain NUTS on a single GPU actually parallelises
    chains within one JIT kernel instead of silently downgrading to
    sequential single-chain (bead ``CF-LIBS-improved-xsuj``).
    """
    import inspect

    from cflibs.inversion.solve.bayesian.samplers import MCMCSampler

    sig = inspect.signature(MCMCSampler.run)
    assert (
        "chain_method" in sig.parameters
    ), "MCMCSampler.run must expose ``chain_method`` so callers can override"
    assert sig.parameters["chain_method"].default == "vectorized", (
        "MCMCSampler.run default ``chain_method`` must be 'vectorized' for "
        "single-GPU multi-chain NUTS"
    )
