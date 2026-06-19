"""
Unit tests for opt-in one-point calibration (OPC), Cavalcanti et al. SAB 2013.

These tests are fully self-contained (synthetic Boltzmann-plot points, no atomic
DB) and assert the *physical* acceptance property of OPC:

1. After fitting F(lambda) on a synthetic certified standard, the standard's CF
   composition recovered from the OPC-corrected Boltzmann plot matches its
   certified composition (residual ~ 0).
2. Applying that same F(lambda) to a SECOND synthetic sample carrying the SAME
   wavelength-dependent matrix/response bias improves the recovered composition
   versus uncorrected CF-LIBS.

The OPC default path is never exercised by the standardless pipeline; these
tests only touch :class:`OnePointCalibrator` / :class:`OnePointCalibration`.

Reference
---------
Cavalcanti et al., Spectrochim. Acta B 87 (2013) 51-56,
doi:10.1016/j.sab.2013.05.016.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.core.constants import KB_EV
from cflibs.inversion.common import LineObservation
from cflibs.inversion.physics.matrix_effects import (
    OnePointCalibration,
    OnePointCalibrator,
)

# ----------------------------------------------------------------------------
# Synthetic forward model + minimal CF read-out (single neutral stage)
# ----------------------------------------------------------------------------

# Two-element single-stage (neutral) toy system. Partition functions constant
# at the test temperature (we evaluate them once at T below).
TEMPERATURE_K = 10000.0
PARTITION_FUNCTIONS = {"Fe": {1: 30.0}, "Cu": {1: 8.0}}

# A fixed, smooth, strongly wavelength-dependent intensity bias B(lambda):
# this stands in for spectral response * A_ki error * residual self-absorption.
# It is identical for the standard and the unknown (shared matrix bias).
_WL_MIN, _WL_MAX = 250.0, 600.0


def _bias(wavelength_nm: float | np.ndarray) -> np.ndarray:
    """Smooth, strong, monotone multiplicative bias B(lambda), away from unity.

    Made steeply wavelength-dependent (factor ~6x across the band) so the bias
    does NOT average out across the Fe/Cu line sets in the toy CF read-out: a
    weak bias partly cancels in the closure normalization, which would make the
    "bias corrupts uncorrected CF" sanity check trivially fragile.
    """
    x = (np.asarray(wavelength_nm, dtype=float) - _WL_MIN) / (_WL_MAX - _WL_MIN)
    # exp ramp from ~0.5 to ~3.0; plus a ripple to break any accidental symmetry.
    return 0.5 * np.exp(1.8 * x) * (1.0 + 0.25 * np.sin(2.0 * np.pi * x))


def _line_grid() -> list[dict]:
    """A fixed set of synthetic atomic lines for Fe I and Cu I.

    Each line has a distinct wavelength and a spread of upper-level energies so
    the Boltzmann slope (-> temperature) and intercept (-> composition) are
    well determined.
    """
    rng = np.random.default_rng(20240117)
    lines: list[dict] = []
    # Fe lines occupy the blue half, Cu lines the red half. Because B(lambda) is
    # strongly wavelength-dependent, the two species' overall intensities are
    # biased by DIFFERENT amounts -> the relative (Fe vs Cu) composition read by
    # uncorrected CF is corrupted, which is exactly what OPC corrects.
    fe_wls = np.linspace(260.0, 400.0, 9)
    cu_wls = np.linspace(440.0, 590.0, 9)
    specs = [("Fe", wl) for wl in fe_wls] + [("Cu", wl) for wl in cu_wls]
    n = len(specs)
    for k, (element, wl) in enumerate(specs):
        # Spread upper-level energies 1.5 .. 6.0 eV within each species so the
        # common Boltzmann slope (temperature) is well constrained.
        e_k = 1.5 + 4.5 * ((k % 9) / 8.0)
        g_k = int(rng.integers(2, 12))
        a_ki = float(10 ** rng.uniform(7.0, 8.5))
        lines.append(
            dict(
                wavelength_nm=float(wl),
                element=element,
                ionization_stage=1,
                E_k_ev=float(e_k),
                g_k=g_k,
                A_ki=a_ki,
            )
        )
    assert n == len(lines)
    return lines


def _synthesize_observations(
    number_fractions: dict[str, float],
    *,
    apply_bias: bool,
    global_scale: float = 1.0e6,
) -> list[LineObservation]:
    """Forward-model Boltzmann-plot points via Eq. (1) of the module docstring.

    I_i = F * (g_i A_i / lambda_i) * (n_s / U_s) * exp(-E_i / kT) [* B(lambda)]

    The wavelength division (g A / lambda) is undone exactly by the Boltzmann
    ordinate definition ln(I lambda / (g A)); we keep it so that the synthetic
    intensities are realistic line areas rather than already-log quantities.
    """
    kT = KB_EV * TEMPERATURE_K
    obs: list[LineObservation] = []
    for ln in _line_grid():
        n_s = number_fractions[ln["element"]]
        u_s = PARTITION_FUNCTIONS[ln["element"]][ln["ionization_stage"]]
        intensity = (
            global_scale
            * (ln["g_k"] * ln["A_ki"] / ln["wavelength_nm"])
            * (n_s / u_s)
            * np.exp(-ln["E_k_ev"] / kT)
        )
        if apply_bias:
            intensity *= float(_bias(ln["wavelength_nm"]))
        obs.append(
            LineObservation(
                wavelength_nm=ln["wavelength_nm"],
                intensity=float(intensity),
                intensity_uncertainty=0.0,
                element=ln["element"],
                ionization_stage=ln["ionization_stage"],
                E_k_ev=ln["E_k_ev"],
                g_k=ln["g_k"],
                A_ki=ln["A_ki"],
            )
        )
    return obs


def _cf_number_fractions(observations: list[LineObservation]) -> dict[str, float]:
    """Minimal standardless CF read-out: per-species Boltzmann-intercept -> n_s.

    Groups lines by element (single neutral stage here), fits the common-slope
    Boltzmann line, and reads n_s = U_s * exp(intercept) up to a global constant,
    then closes to sum = 1. This is the relative composition CF-LIBS reports.
    """
    # Stack all points; fit a single common slope (shared T), per-species intercept.
    elements = sorted({o.element for o in observations})
    # Design matrix: columns = [E_k, one-hot(element intercepts)]
    x_e = np.array([o.E_k_ev for o in observations])
    y = np.array([o.y_value for o in observations])
    col_slope = -x_e / (KB_EV * TEMPERATURE_K)  # known T: fix slope contribution
    # We assume the temperature is known (as in OPC); solve only for intercepts.
    resid = y - col_slope
    intercepts: dict[str, float] = {}
    for el in elements:
        mask = np.array([o.element == el for o in observations])
        intercepts[el] = float(np.mean(resid[mask]))
    # n_s proportional to U_s * exp(intercept_s)
    raw = {el: PARTITION_FUNCTIONS[el][1] * np.exp(intercepts[el]) for el in elements}
    total = sum(raw.values())
    return {el: raw[el] / total for el in elements}


def _max_abs_error(recovered: dict[str, float], truth: dict[str, float]) -> float:
    return max(abs(recovered[el] - truth[el]) for el in truth)


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------

CERT_STANDARD = {"Fe": 0.70, "Cu": 0.30}
UNKNOWN_TRUTH = {"Fe": 0.40, "Cu": 0.60}


def test_unbiased_readout_is_exact_sanity():
    """Sanity: with NO bias the toy CF read-out recovers the truth exactly."""
    obs = _synthesize_observations(CERT_STANDARD, apply_bias=False)
    rec = _cf_number_fractions(obs)
    assert _max_abs_error(rec, CERT_STANDARD) < 1e-9


def test_bias_corrupts_uncorrected_cf():
    """The injected bias must actually corrupt uncorrected CF (else test is moot)."""
    obs = _synthesize_observations(UNKNOWN_TRUTH, apply_bias=True)
    rec = _cf_number_fractions(obs)
    # Bias is large; uncorrected error should be clearly non-trivial.
    assert _max_abs_error(rec, UNKNOWN_TRUTH) > 0.05


def test_opc_fit_forces_standard_onto_certified():
    """ACCEPTANCE 1: OPC-corrected standard reproduces its certified composition."""
    std_obs = _synthesize_observations(CERT_STANDARD, apply_bias=True)

    # Before OPC: the standard's own CF result is biased away from certified.
    rec_before = _cf_number_fractions(std_obs)
    err_before = _max_abs_error(rec_before, CERT_STANDARD)
    assert err_before > 0.02  # bias present

    calibrator = OnePointCalibrator()
    opc = calibrator.fit(
        std_obs,
        certified_number_fractions=CERT_STANDARD,
        partition_functions=PARTITION_FUNCTIONS,
        temperature_K=TEMPERATURE_K,
        reference_label="synthetic_FeCu_standard",
    )
    assert isinstance(opc, OnePointCalibration)

    corrected_std = opc.apply(std_obs)
    rec_after = _cf_number_fractions(corrected_std)
    err_after = _max_abs_error(rec_after, CERT_STANDARD)

    # Residual ~ 0: standard CF matches certified composition by construction.
    assert err_after < 1e-6
    assert err_after < err_before


def test_opc_improves_second_sample_with_same_bias():
    """ACCEPTANCE 2: applying F(lambda) to a 2nd sample with the same matrix bias
    improves recovered composition vs uncorrected CF."""
    std_obs = _synthesize_observations(CERT_STANDARD, apply_bias=True)
    unk_obs = _synthesize_observations(UNKNOWN_TRUTH, apply_bias=True)

    # Uncorrected CF on the unknown.
    rec_uncorrected = _cf_number_fractions(unk_obs)
    err_uncorrected = _max_abs_error(rec_uncorrected, UNKNOWN_TRUTH)

    # Fit OPC on the standard, apply to the unknown.
    opc = OnePointCalibrator().fit(
        std_obs,
        certified_number_fractions=CERT_STANDARD,
        partition_functions=PARTITION_FUNCTIONS,
        temperature_K=TEMPERATURE_K,
    )
    corrected_unk = opc.apply(unk_obs)
    rec_corrected = _cf_number_fractions(corrected_unk)
    err_corrected = _max_abs_error(rec_corrected, UNKNOWN_TRUTH)

    # The shared bias is exactly the same B(lambda) at the same wavelengths, so
    # the correction is essentially exact for the unknown too.
    assert err_corrected < err_uncorrected
    assert err_corrected < 1e-6


def test_factor_recovers_inverse_bias_up_to_global_constant():
    """F_corr(lambda) recovers 1/B(lambda) up to a single global constant.

    OPC's per-line factor is exp(delta) = (global) / B(lambda). Verifying the
    ratio F_corr(lambda) * B(lambda) is constant across wavelength confirms the
    physics of Eq. (4) (it cancels exactly the injected multiplicative bias).
    """
    std_obs = _synthesize_observations(CERT_STANDARD, apply_bias=True)
    opc = OnePointCalibrator().fit(
        std_obs,
        certified_number_fractions=CERT_STANDARD,
        partition_functions=PARTITION_FUNCTIONS,
        temperature_K=TEMPERATURE_K,
    )
    products = []
    for o in std_obs:
        products.append(opc.factor_at(o.wavelength_nm) * float(_bias(o.wavelength_nm)))
    products = np.asarray(products)
    # F_corr * B == const for all lines: relative spread must be tiny.
    assert np.std(products) / np.mean(products) < 1e-9


def test_out_of_range_no_extrapolation_is_no_correction():
    """Default extrapolate=False -> wavelengths outside the standard get delta=0."""
    std_obs = _synthesize_observations(CERT_STANDARD, apply_bias=True)
    opc = OnePointCalibrator().fit(
        std_obs,
        certified_number_fractions=CERT_STANDARD,
        partition_functions=PARTITION_FUNCTIONS,
        temperature_K=TEMPERATURE_K,
    )
    below = min(o.wavelength_nm for o in std_obs) - 50.0
    above = max(o.wavelength_nm for o in std_obs) + 50.0
    assert opc.correction_at(below) == 0.0
    assert opc.correction_at(above) == 0.0
    assert opc.factor_at(below) == pytest.approx(1.0)


def test_fit_rejects_missing_certified_fraction():
    """Missing/non-positive certified fraction for an observed element -> error."""
    std_obs = _synthesize_observations(CERT_STANDARD, apply_bias=True)
    with pytest.raises(ValueError):
        OnePointCalibrator().fit(
            std_obs,
            certified_number_fractions={"Fe": 0.70},  # Cu missing
            partition_functions=PARTITION_FUNCTIONS,
            temperature_K=TEMPERATURE_K,
        )


def test_fit_rejects_nonpositive_temperature():
    std_obs = _synthesize_observations(CERT_STANDARD, apply_bias=True)
    with pytest.raises(ValueError):
        OnePointCalibrator().fit(
            std_obs,
            certified_number_fractions=CERT_STANDARD,
            partition_functions=PARTITION_FUNCTIONS,
            temperature_K=0.0,
        )
