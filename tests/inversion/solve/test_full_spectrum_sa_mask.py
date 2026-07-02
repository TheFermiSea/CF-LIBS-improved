"""Tests for the observable self-absorption mask on the full-spectrum fit.

Physics-first-principles audit, Issue 3: the raw-spectrum (full-spectrum /
joint) fit compares an optically-THIN model to optically-THICK data, so a
prior-free fit fakes the missing saturation by riding T/n_e to a tanh box edge.
The fix wires the EXISTING observable-anchored corrector (doublet intensity
ratios; no composition-derived optical depth, no fitted tau DOF) into that path
as an EXCLUSION MASK: observable-thick line windows are dropped from the SVD
residual on both the observed and model side.

Two properties are exercised here without a database or JAX:

* THICK doublet -> the observable corrector solves a bounded optical depth and
  the line windows are excluded.
* THIN doublet -> the measured ratio matches the optically-thin ratio, nothing
  is flagged, and the mask is an EXACT no-op (``keep is None``).

The end-to-end thin-vs-thick behaviour through ``solve_full_spectrum`` is an
opt-in ``requires_db``/``requires_jax`` integration test at the bottom.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.inversion.physics.self_absorption_observable import (
    ThickLineMask,
    build_observed_thick_line_mask,
)

# ---------------------------------------------------------------------------
# Synthetic doublet helpers (DB-free, deterministic)
# ---------------------------------------------------------------------------

# Two lines decaying from ONE shared upper level to two different lower levels
# (a branching doublet — the case the doublet-ratio observable handles). The
# shorter-wavelength member is the resonance line (E_i = 0).
_LAM1, _LAM2 = 400.0, 500.0
_HC_EV_NM = 1239.841984  # hc in eV*nm
_EK = _HC_EV_NM / _LAM1  # upper-level energy so line 1 is a resonance line
_GK = 3.0
_AKI = 1.0e8


def _grid() -> np.ndarray:
    return np.linspace(395.0, 505.0, 1101)


def _spectrum(area1: float, area2: float, sigma_nm: float = 0.08) -> np.ndarray:
    """Two Gaussian lines with prescribed integrated areas."""
    wl = _grid()

    def g(center: float, area: float) -> np.ndarray:
        return (
            area
            / (sigma_nm * np.sqrt(2.0 * np.pi))
            * np.exp(-0.5 * ((wl - center) / sigma_nm) ** 2)
        )

    return g(_LAM1, area1) + g(_LAM2, area2)


def _mask(intensity: np.ndarray, **kw) -> ThickLineMask:
    wl = _grid()
    defaults = dict(
        line_wavelengths_nm=[_LAM1, _LAM2],
        line_elements=["Fe", "Fe"],
        line_ion_stages=[1, 1],
        line_E_k_ev=[_EK, _EK],
        line_g_k=[_GK, _GK],
        line_A_ki=[_AKI, _AKI],
        measure_half_width_nm=0.4,
        mask_half_width_nm=0.6,
        mask_tau_min=0.5,
        min_peak_snr=4.0,
    )
    defaults.update(kw)
    return build_observed_thick_line_mask(wl, intensity, **defaults)


# The optically-thin emission ratio for this doublet is r_thin = lambda2/lambda1
# = 1.25 (equal g_k, A_ki). Self-absorption suppresses the more optically-thick
# member (line 2), so a THICK observed ratio I1/I2 > r_thin. Area (2.0, 1.0)
# gives I1/I2 = 2.0 => a doublet-inferred optical depth well above 0.5.
_R_THIN = _LAM2 / _LAM1


# ---------------------------------------------------------------------------
# THIN doublet -> exact no-op
# ---------------------------------------------------------------------------


def test_thin_doublet_is_exact_no_op():
    """A thin doublet (measured ratio == thin ratio) flags nothing: keep is None."""
    m = _mask(_spectrum(_R_THIN, 1.0))
    assert m.n_flagged == 0
    assert m.keep is None  # exact no-op signal for the caller
    assert m.max_tau == 0.0


def test_thin_doublet_with_noise_is_no_op():
    """Small noise on a thin doublet still flags nothing (deviation below floor)."""
    rng = np.random.default_rng(0)
    spec = _spectrum(_R_THIN, 1.0) + rng.normal(0.0, 0.02, _grid().shape)
    m = _mask(spec)
    assert m.n_flagged == 0
    assert m.keep is None


# ---------------------------------------------------------------------------
# THICK doublet -> flagged + windows excluded
# ---------------------------------------------------------------------------


def test_thick_doublet_is_flagged_and_masked():
    """A self-absorbed doublet is flagged with a bounded optical depth."""
    m = _mask(_spectrum(2.0, 1.0))
    assert m.n_flagged >= 1
    assert m.keep is not None
    # Optical depth is bounded to the doublet validity ceiling (<= 5).
    assert 0.5 <= m.max_tau <= 5.0
    # Both line centers fall in excluded (masked-out) windows.
    wl = _grid()
    for center in (_LAM1, _LAM2):
        j = int(np.argmin(np.abs(wl - center)))
        assert not bool(m.keep[j]), f"{center} nm should be masked out"
    # Only the line windows are removed, not the whole spectrum.
    assert m.keep.sum() > 0.8 * m.keep.size


def test_thick_doublet_flag_is_observable_anchored():
    """Correction records are doublet-observable, never composition-derived."""
    m = _mask(_spectrum(2.0, 1.0))
    methods = {c.method for c in m.corrections.values()}
    assert methods <= {"doublet", "doublet-thin", "suspect", "planck", "none"}
    # At least one genuinely-solved doublet correction drives the mask.
    assert any(c.method == "doublet" and c.correction_factor > 1.0 for c in m.corrections.values())


def test_tau_min_gate_controls_masking():
    """Raising mask_tau_min above the measured tau disables the mask (no-op)."""
    strong = _spectrum(2.0, 1.0)
    m_low = _mask(strong, mask_tau_min=0.5)
    assert m_low.n_flagged >= 1
    huge = m_low.max_tau + 10.0
    m_high = _mask(strong, mask_tau_min=huge)
    assert m_high.n_flagged == 0
    assert m_high.keep is None


# ---------------------------------------------------------------------------
# Guards / degenerate inputs
# ---------------------------------------------------------------------------


def test_empty_inputs_return_no_op():
    m = build_observed_thick_line_mask(
        np.array([]),
        np.array([]),
        line_wavelengths_nm=[],
        line_elements=[],
        line_ion_stages=[],
        line_E_k_ev=[],
        line_g_k=[],
        line_A_ki=[],
    )
    assert m.keep is None
    assert m.n_flagged == 0


def test_mismatched_metadata_lengths_return_no_op():
    wl = _grid()
    m = build_observed_thick_line_mask(
        wl,
        _spectrum(2.0, 1.0),
        line_wavelengths_nm=[_LAM1, _LAM2],
        line_elements=["Fe"],  # wrong length
        line_ion_stages=[1, 1],
        line_E_k_ev=[_EK, _EK],
        line_g_k=[_GK, _GK],
        line_A_ki=[_AKI, _AKI],
    )
    assert m.keep is None


def test_cross_band_pair_ratio_guard_rejects():
    """A UV<->IR branching pair (extreme lambda ratio) is not trusted."""
    # Same shared upper level, but partner at ~5x the wavelength: rho and the
    # cross-band thin ratio are atomic-data fragile, so it must not be masked.
    lam_ir = 2000.0
    ek = _HC_EV_NM / _LAM1  # keep line 1 resonance; shared upper level
    wl = np.linspace(395.0, 2005.0, 3000)

    def g(center, area, sig=0.1):
        return area / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((wl - center) / sig) ** 2)

    spec = g(_LAM1, 2.0) + g(lam_ir, 1.0)
    m = build_observed_thick_line_mask(
        wl,
        spec,
        line_wavelengths_nm=[_LAM1, lam_ir],
        line_elements=["Fe", "Fe"],
        line_ion_stages=[1, 1],
        line_E_k_ev=[ek, ek],
        line_g_k=[_GK, _GK],
        line_A_ki=[_AKI, _AKI],
        measure_half_width_nm=0.5,
        mask_half_width_nm=0.6,
        mask_tau_min=0.5,
        max_pair_wavelength_ratio=1.4,
    )
    assert m.n_flagged == 0
    assert m.keep is None


# ---------------------------------------------------------------------------
# End-to-end thin no-op through solve_full_spectrum (opt-in)
# ---------------------------------------------------------------------------


@pytest.mark.requires_db
@pytest.mark.requires_jax
@pytest.mark.slow
def test_solve_full_spectrum_thin_is_no_op(production_db):
    """On a THIN synthetic the SA mask is empty and the fit is bit-identical.

    Generates an optically-thin Ca/Mg spectrum (tiny path length => no
    saturation), then fits with the SA mask OFF and ON. The mask must find no
    thick line (``keep is None``) so the two fits agree exactly.
    """
    pytest.importorskip("jax")
    from cflibs.instrument import InstrumentModel
    from cflibs.inversion.solve.full_spectrum import solve_full_spectrum
    from cflibs.plasma.state import SingleZoneLTEPlasma
    from cflibs.radiation.profiles import BroadeningMode
    from cflibs.radiation.spectrum_model import SpectrumModel

    db_path = str(production_db.db_path)
    T_K, ne, total = 8000.0, 1e16, 1e17
    plasma = SingleZoneLTEPlasma(T_e=T_K, n_e=ne, species={"Ca": 0.6 * total, "Mg": 0.4 * total})
    inst = InstrumentModel.from_resolving_power(5000)
    model = SpectrumModel(
        plasma=plasma,
        atomic_db=production_db,
        instrument=inst,
        lambda_min=279.0,
        lambda_max=430.0,
        delta_lambda=0.02,
        path_length_m=1e-6,  # optically thin
        broadening_mode=BroadeningMode.PHYSICAL_DOPPLER,
    )
    wl, intensity = model.compute_spectrum()
    warm = {"Ca": 0.712, "Mg": 0.288}

    def solve(sa):
        return solve_full_spectrum(
            wl,
            intensity,
            ["Ca", "Mg"],
            db_path,
            warm_start_T_K=T_K,
            warm_start_ne_cm3=ne,
            warm_start_concentrations=dict(warm),
            resolving_power=5000,
            method="joint",
            apply_self_absorption=sa,
            fit_pixels=900,
            max_iterations=40,
        )

    off = solve("off")
    on = solve("observable")
    sa_diag = on.diagnostics["strict_diagnostics"]["extra"]["self_absorption_mask"]
    assert sa_diag["n_flagged"] == 0, "thin spectrum must flag no thick lines"
    assert np.isclose(off.fit_temperature_K, on.fit_temperature_K)
    assert np.isclose(off.fit_electron_density_cm3, on.fit_electron_density_cm3)
    for el in ("Ca", "Mg"):
        assert np.isclose(off.fit_concentrations[el], on.fit_concentrations[el])
