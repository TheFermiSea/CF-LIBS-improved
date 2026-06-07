"""Validation gate for the Stark line-center **shift** (audit Family 10).

The Stark shift moves a line center linearly with electron density::

    delta_lambda = line_stark_d * (n_e / REF_NE)    (REF_NE = 1e17 cm^-3)
    lambda_c     = lambda_0 + delta_lambda

This is the convention of :func:`cflibs.radiation.stark.stark_shift`. The
forward kernel applies it to the *profile centers* (not the hc/4pi*lambda
emissivity prefactor) when ``apply_stark`` is on.

Grounding: John et al. 2023 and Stetzler et al. 2020 both drop the Griem
ion-broadening term for LIBS (a <2-5% effect at n_e <= 1e17 cm^-3), so this
family implements ONLY the shift — the corpus flags it as a real
line-identification error source (Noel 2025). Ion broadening / R_D are
deliberately NOT implemented.

The expected values here are computed from an INDEPENDENT oracle
(``lambda_0 + d_ref * n_e / 1e17``) — never derived from the kernel under
test.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_jax, pytest.mark.physics]

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.core.jax_runtime import AtomicSnapshot  # noqa: E402
from cflibs.instrument.model import InstrumentModel  # noqa: E402
from cflibs.plasma.state import SingleZoneLTEPlasma  # noqa: E402
from cflibs.radiation.kernels import (  # noqa: E402
    BroadeningMode,
    _per_line_stark_shift,
    forward_model,
)
from cflibs.radiation.stark import REF_NE, stark_shift  # noqa: E402

# Independent oracle reference density — must equal stark.REF_NE.
_REF_NE_ORACLE = 1.0e17


def _snapshot(stark_d, lam0=None, n_lines=None):
    """Build a minimal snapshot carrying an explicit ``line_stark_d`` array."""
    stark_d = jnp.asarray(stark_d, dtype=jnp.float64)
    n = int(stark_d.shape[0]) if n_lines is None else n_lines
    if lam0 is None:
        lam0 = jnp.asarray([400.0 + 10.0 * i for i in range(n)], dtype=jnp.float64)
    else:
        lam0 = jnp.asarray(lam0, dtype=jnp.float64)
    return AtomicSnapshot(
        species=(("Fe", 1),),
        line_wavelengths_nm=lam0,
        line_A_ki=jnp.full(n, 1.0e8),
        line_E_k_ev=jnp.zeros(n),
        line_g_k=jnp.ones(n),
        line_E_i_ev=jnp.zeros(n),
        line_g_i=jnp.ones(n),
        line_species_index=jnp.zeros(n, dtype=jnp.int32),
        line_stark_w=jnp.zeros(n),
        line_stark_alpha=jnp.zeros(n),
        line_natural_w=jnp.zeros(n),
        partition_coeffs=jnp.zeros((1, 5)),
        ionization_potential_ev=jnp.array([7.87]),
        line_stark_d=stark_d,
    )


# ---------------------------------------------------------------------------
# Convention sanity: REF_NE and the scalar helper.
# ---------------------------------------------------------------------------


def test_ref_ne_is_1e17():
    """Kernel and helper share the 1e17 cm^-3 reference convention."""
    assert REF_NE == _REF_NE_ORACLE


def test_scalar_stark_shift_oracle():
    """``stark.stark_shift`` reproduces d_ref * (n_e / 1e17) exactly."""
    d_ref = 0.012  # nm
    for n_e in (1.0e16, 5.0e16, 1.0e17, 3.0e17, 1.0e18):
        expected = d_ref * (n_e / _REF_NE_ORACLE)  # independent oracle
        assert stark_shift(n_e, d_ref) == pytest.approx(expected, rel=1e-12)
    # None -> no shift.
    assert stark_shift(1.0e17, None) == 0.0


# ---------------------------------------------------------------------------
# Per-line helper: the validation gate from the audit spec.
# ---------------------------------------------------------------------------


def test_shift_equals_d_ref_at_reference_density():
    """At n_e = REF_NE (1e17), center moves by exactly d_ref."""
    d_ref = np.array([0.010, -0.020, 0.005])
    lam0 = np.array([400.0, 500.0, 600.0])
    snap = _snapshot(d_ref, lam0=lam0)

    centers = np.asarray(_per_line_stark_shift(snap, jnp.asarray(lam0), 1.0e17))
    expected = lam0 + d_ref  # at 1e17 the factor is exactly 1
    np.testing.assert_allclose(centers, expected, rtol=0.0, atol=1e-12)


def test_shift_is_half_at_half_reference_density():
    """At n_e = 5e16 (= 0.5 * REF_NE), center moves by exactly 0.5 * d_ref."""
    d_ref = np.array([0.010, -0.020, 0.005])
    lam0 = np.array([400.0, 500.0, 600.0])
    snap = _snapshot(d_ref, lam0=lam0)

    centers = np.asarray(_per_line_stark_shift(snap, jnp.asarray(lam0), 5.0e16))
    expected = lam0 + 0.5 * d_ref  # independent oracle
    np.testing.assert_allclose(centers, expected, rtol=0.0, atol=1e-12)


def test_zero_d_ref_means_no_shift():
    """line_stark_d == 0 leaves the center exactly where it was."""
    lam0 = np.array([400.0, 500.0, 600.0])
    snap = _snapshot(np.zeros(3), lam0=lam0)

    for n_e in (1.0e16, 5.0e16, 1.0e17, 1.0e18):
        centers = np.asarray(_per_line_stark_shift(snap, jnp.asarray(lam0), n_e))
        np.testing.assert_array_equal(centers, lam0)


def test_none_d_ref_means_no_shift():
    """A snapshot built without line_stark_d (None) is treated as no shift."""
    lam0 = np.array([400.0, 500.0])
    snap = _snapshot(np.zeros(2), lam0=lam0)
    snap = AtomicSnapshot(**{**snap.__dict__, "line_stark_d": None})
    centers = np.asarray(_per_line_stark_shift(snap, jnp.asarray(lam0), 1.0e17))
    np.testing.assert_array_equal(centers, lam0)


def test_shift_is_linear_in_n_e():
    """The shift scales strictly linearly with n_e across the ps-LIBS range."""
    d_ref = np.array([0.0150])
    lam0 = np.array([450.0])
    snap = _snapshot(d_ref, lam0=lam0)

    n_e_grid = np.array([1.0e16, 2.0e16, 5.0e16, 1.0e17, 5.0e17, 1.0e18])
    deltas = np.array(
        [
            float(np.asarray(_per_line_stark_shift(snap, jnp.asarray(lam0), ne))[0]) - lam0[0]
            for ne in n_e_grid
        ]
    )
    expected = d_ref[0] * (n_e_grid / _REF_NE_ORACLE)  # independent oracle
    np.testing.assert_allclose(deltas, expected, rtol=1e-10, atol=1e-14)

    # Linearity: delta(n_e) / n_e is constant (the proportionality slope).
    slopes = deltas / n_e_grid
    np.testing.assert_allclose(slopes, slopes[0], rtol=1e-10, atol=0.0)


# ---------------------------------------------------------------------------
# End-to-end: the emitted peak moves by the expected shift.
# ---------------------------------------------------------------------------


def _peak_wavelength(wl, spectrum):
    return float(np.asarray(wl)[int(np.argmax(np.asarray(spectrum)))])


def test_forward_model_peak_moves_by_shift():
    """A single isolated line's emitted peak shifts by d_ref * (n_e / 1e17).

    Uses LEGACY broadening (single narrow Gaussian) with injected upper-level
    populations so the peak position is unambiguous and independent of the
    Saha-Boltzmann balance.
    """
    lam0 = 500.0
    d_ref = 0.20  # nm — large + grid-resolvable so argmax is decisive
    n_e = 1.0e17  # at REF_NE the shift equals d_ref exactly

    snap = _snapshot(np.array([d_ref]), lam0=np.array([lam0]), n_lines=1)
    plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=n_e, species={"Fe": 1.0e16})
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    wl = jnp.asarray(np.arange(499.0, 501.0 + 1e-9, 0.01))

    common = dict(
        plasma_state=plasma,
        atomic_snapshot=snap,
        instrument=instrument,
        wavelength_grid=wl,
        broadening_mode=BroadeningMode.LEGACY,
        path_length_m=0.01,
        _precomputed_n_upper_per_line=jnp.array([1.0e15]),
    )

    spec_off = forward_model(**common, apply_stark=False)
    spec_on = forward_model(**common, apply_stark=True)

    peak_off = _peak_wavelength(wl, spec_off)
    peak_on = _peak_wavelength(wl, spec_on)

    # Independent oracle: at n_e == REF_NE the center moves by exactly d_ref.
    assert peak_off == pytest.approx(lam0, abs=0.011)
    assert peak_on == pytest.approx(lam0 + d_ref, abs=0.011)
    assert (peak_on - peak_off) == pytest.approx(d_ref, abs=0.011)


def test_forward_model_no_shift_when_d_ref_zero():
    """With line_stark_d == 0, apply_stark on/off give the same peak center."""
    lam0 = 500.0
    snap = _snapshot(np.array([0.0]), lam0=np.array([lam0]), n_lines=1)
    plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=1.0e17, species={"Fe": 1.0e16})
    instrument = InstrumentModel(resolution_fwhm_nm=0.05)
    wl = jnp.asarray(np.arange(499.0, 501.0 + 1e-9, 0.01))

    common = dict(
        plasma_state=plasma,
        atomic_snapshot=snap,
        instrument=instrument,
        wavelength_grid=wl,
        broadening_mode=BroadeningMode.LEGACY,
        path_length_m=0.01,
        _precomputed_n_upper_per_line=jnp.array([1.0e15]),
    )
    peak_off = _peak_wavelength(wl, forward_model(**common, apply_stark=False))
    peak_on = _peak_wavelength(wl, forward_model(**common, apply_stark=True))
    assert peak_on == pytest.approx(peak_off, abs=1e-9)
