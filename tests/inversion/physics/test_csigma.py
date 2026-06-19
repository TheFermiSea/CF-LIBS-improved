"""
Unit tests for the C-sigma (Csigma) generalized curve-of-growth module.

The acceptance property (Aragón & Aguilera, JQSRT 149 (2014) 90):
on synthetic multi-element data with a *known* T and *known* relative
concentrations and a deliberate mix of optically thin and optically thick
lines, the C-sigma common-line fit recovers a consistent temperature and the
relative concentrations within tolerance.

The synthetic intensities are generated from the *same* radiative-transfer COG
relation the fit inverts, so this is a closed-loop self-consistency check of the
physics (no atomic DB, no real spectra).  All tests run in well under 90 s.
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.core.constants import C_LIGHT, H_PLANCK, KB
from cflibs.inversion.common.data_structures import LineObservation
from cflibs.inversion.physics.csigma import (
    CsigmaFitResult,
    build_csigma_graph,
    cog_function,
    fit_csigma,
)

_J_TO_EV = 1.0 / 1.602176634e-19


# --------------------------------------------------------------------------- #
# Curve-of-growth function: analytic limiting behaviour.
# --------------------------------------------------------------------------- #
def test_cog_thin_limit_is_linear() -> None:
    """W(tau0) -> sqrt(pi) * tau0 in the optically-thin limit (slope 1)."""
    for tau0 in (1e-4, 1e-3, 1e-2):
        w = cog_function(tau0)
        assert w == pytest.approx(np.sqrt(np.pi) * tau0, rel=2e-2)


def test_cog_thick_limit_is_sqrt_log() -> None:
    """W(tau0) -> 2*sqrt(ln tau0) in the optically-thick (flat) part."""
    for tau0 in (1e3, 1e4, 1e5):
        w = cog_function(tau0)
        assert w == pytest.approx(2.0 * np.sqrt(np.log(tau0)), rel=5e-2)


def test_cog_monotone_increasing() -> None:
    """W is strictly increasing in tau0 (more opacity -> more absorbed flux)."""
    taus = np.logspace(-3, 5, 40)
    w = np.array([cog_function(t) for t in taus])
    assert np.all(np.diff(w) > 0.0)


# --------------------------------------------------------------------------- #
# Synthetic forward model for the C-sigma graph.
# --------------------------------------------------------------------------- #
def _planck(wavelength_nm: float, temperature_K: float) -> float:
    lam = wavelength_nm * 1e-9
    x = (H_PLANCK * C_LIGHT) / (lam * KB * temperature_K)
    return (2.0 * H_PLANCK * C_LIGHT**2) / (lam**5) / np.expm1(x)


def _sigma_rel(wl_nm: float, e_k_ev: float, g_k: int, a_ki: float, T_K: float) -> float:
    """Mirror csigma._log_sigma_rel (linear, not log) for forward generation."""
    lam_m = wl_nm * 1e-9
    photon_eV = (H_PLANCK * C_LIGHT) / lam_m * _J_TO_EV
    e_lower_eV = e_k_ev - photon_eV
    kt_eV = 8.617333262e-5 * T_K
    boltz_lower = np.exp(-e_lower_eV / kt_eV)
    stim = 1.0 - np.exp(-photon_eV / kt_eV)
    return (wl_nm**2) * g_k * a_ki * boltz_lower * stim


def _make_synthetic_dataset(
    T_true: float,
    conc_true: dict[str, float],
    scale_true: float,
    seed: int = 0,
) -> list[LineObservation]:
    """Build same-stage (singly-ionized) lines for two elements spanning a wide
    range of optical depths (thin -> thick) via the C-sigma RT/COG relation.

    Intensity is generated as I = B(lambda, T) * W(tau0) with
    tau0 = 10**scale * C * sigma_rel, which is exactly the model fit_csigma
    inverts (closed-loop self-consistency).
    """
    rng = np.random.default_rng(seed)
    # (element, wl_nm, E_k_eV, g_k, A_ki) — A_ki & wavelengths spread tau0 over
    # several decades so both thin and thick lines are present.
    specs = [
        # Element "A": strong + weak lines.
        ("A", 280.0, 8.0, 4, 5.0e8),  # strong -> thick
        ("A", 393.0, 6.5, 6, 1.5e8),  # strong -> thick
        ("A", 422.0, 5.8, 2, 2.0e6),  # moderate
        ("A", 445.0, 7.2, 4, 8.0e4),  # weak -> thin
        ("A", 460.0, 9.1, 2, 3.0e3),  # very weak -> thin
        # Element "B": strong + weak lines.
        ("B", 250.0, 8.6, 6, 4.0e8),  # strong -> thick
        ("B", 313.0, 7.0, 4, 1.0e8),  # strong -> thick
        ("B", 370.0, 6.2, 2, 1.0e6),  # moderate
        ("B", 405.0, 7.8, 4, 5.0e4),  # weak -> thin
        ("B", 430.0, 9.4, 6, 2.0e3),  # very weak -> thin
    ]
    out: list[LineObservation] = []
    for elem, wl, e_k, g_k, a_ki in specs:
        sigma = _sigma_rel(wl, e_k, g_k, a_ki, T_true)
        tau0 = (10.0**scale_true) * conc_true[elem] * sigma
        w = cog_function(tau0)
        intensity = _planck(wl, T_true) * w
        # Mild multiplicative noise (~3%) so the fit is non-trivial.
        intensity *= 1.0 + 0.03 * rng.standard_normal()
        out.append(
            LineObservation(
                wavelength_nm=wl,
                intensity=float(intensity),
                intensity_uncertainty=float(0.03 * intensity),
                element=elem,
                ionization_stage=1,
                E_k_ev=e_k,
                g_k=g_k,
                A_ki=a_ki,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Acceptance: T and relative-concentration recovery (thin + thick together).
# --------------------------------------------------------------------------- #
def test_csigma_recovers_temperature_and_concentrations() -> None:
    """ACCEPTANCE: common-line fit recovers consistent T and relative
    concentrations from a thin+thick multi-element C-sigma graph."""
    T_true = 11000.0
    conc_true = {"A": 0.75, "B": 0.25}  # 3:1
    # scale chosen so tau0 straddles 1 -> a genuine mix of thin and thick lines.
    scale_true = -7.0
    obs = _make_synthetic_dataset(T_true, conc_true, scale_true, seed=7)

    result = fit_csigma(obs, t_init_K=9000.0)

    assert isinstance(result, CsigmaFitResult)
    assert result.converged
    # Temperature within 10% of truth.
    assert result.temperature_K == pytest.approx(
        T_true, rel=0.10
    ), f"T={result.temperature_K:.0f} K vs truth {T_true:.0f} K"
    # Relative concentrations within tolerance (normalized to sum=1).
    cA = result.relative_concentrations["A"]
    cB = result.relative_concentrations["B"]
    assert cA == pytest.approx(conc_true["A"], abs=0.07), f"C_A={cA:.3f}"
    assert cB == pytest.approx(conc_true["B"], abs=0.07), f"C_B={cB:.3f}"
    # The dataset truly contains a mix of thin AND thick lines.
    assert result.n_thick >= 2, "expected optically-thick lines in the pool"
    assert result.n_thick < result.n_points, "expected optically-thin lines too"
    # Good closed-loop fit.
    assert result.residual_rms < 0.05


def test_csigma_concentration_ordering_2to1() -> None:
    """Reversed-dominance composition is recovered with correct ordering/ratio."""
    T_true = 9000.0
    conc_true = {"A": 0.20, "B": 0.80}  # B dominant, 1:4
    obs = _make_synthetic_dataset(T_true, conc_true, scale_true=-7.0, seed=3)

    result = fit_csigma(obs, t_init_K=12000.0)

    cA = result.relative_concentrations["A"]
    cB = result.relative_concentrations["B"]
    assert cB > cA  # ordering preserved
    ratio = cB / cA
    assert ratio == pytest.approx(4.0, rel=0.30), f"C_B/C_A = {ratio:.2f}"
    assert result.temperature_K == pytest.approx(T_true, rel=0.12)


# --------------------------------------------------------------------------- #
# Builder structure / guards.
# --------------------------------------------------------------------------- #
def test_build_graph_per_stage_and_structure() -> None:
    """build_csigma_graph returns one point per positive line, all same stage."""
    obs = _make_synthetic_dataset(10000.0, {"A": 0.5, "B": 0.5}, -2.0, seed=1)
    pts = build_csigma_graph(obs, temperature_K=10000.0)
    assert len(pts) == len(obs)
    assert {p.ionization_stage for p in pts} == {1}
    # Abscissa should be finite and span a range (thin..thick).
    xs = np.array([p.log_sigma_rel for p in pts])
    assert np.all(np.isfinite(xs))
    assert xs.max() - xs.min() > 1.0


def test_build_graph_rejects_mixed_stages() -> None:
    """A C-sigma graph is per-ionization-stage; mixing stages is rejected."""
    obs = _make_synthetic_dataset(10000.0, {"A": 0.5, "B": 0.5}, -2.0, seed=1)
    obs[0].ionization_stage = 0  # corrupt one stage
    with pytest.raises(ValueError, match="ionization stage"):
        build_csigma_graph(obs, temperature_K=10000.0)


def test_fit_rejects_mixed_stages_and_too_few() -> None:
    # Two independent datasets so mutating one cannot alias into the other.
    bad = _make_synthetic_dataset(10000.0, {"A": 0.5, "B": 0.5}, -2.0, seed=1)
    bad[0].ionization_stage = 0
    with pytest.raises(ValueError, match="single ionization stage"):
        fit_csigma(bad)

    too_few = _make_synthetic_dataset(10000.0, {"A": 0.5, "B": 0.5}, -2.0, seed=2)[:2]
    with pytest.raises(ValueError, match="at least 3"):
        fit_csigma(too_few)
