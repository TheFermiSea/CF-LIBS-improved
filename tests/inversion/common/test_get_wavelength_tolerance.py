"""Wave-1 Fix A1 regression tests for ``get_wavelength_tolerance``.

The helper used to read a non-existent ``stark_width_nm`` attribute via
``getattr`` (the real field on ``cflibs.atomic.structures.Transition`` is
``stark_w``). Because the fallback was 0.0, ``omega_stark`` collapsed to 0
for every line and the protocol-prescribed
``sqrt(fwhm_inst**2 + omega_stark**2)`` tolerance silently degraded to
``fwhm_inst`` alone.

These tests verify (a) the typo is fixed, (b) the Stark FWHM scales
linearly with electron density via :func:`cflibs.radiation.stark.stark_hwhm`,
and (c) the Konjević-reference defaults match an explicit live (n_e, T)
call at those conditions.

References
----------
- Konjević, Lesage, Fuhr & Wiese (2002) *J. Phys. Chem. Ref. Data* 31, 819.
- Aragón, Pellé & Aguilera (2011) *Anal. Bioanal. Chem.* 400, 3331.
- ``validation/protocol.yaml`` §identification.wavelength_tolerance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import pytest

from cflibs.inversion.common.element_id import get_wavelength_tolerance
from cflibs.radiation.stark import stark_hwhm


@dataclass
class _MockTransition:
    """Minimal stand-in matching the real Transition Stark API."""

    stark_w: Optional[float] = None  # HWHM at REF_NE=1e16 cm^-3, T=10000 K (nm)
    stark_alpha: Optional[float] = None


# ---------------------------------------------------------------------------
# Primary regression: the typo is fixed.
# ---------------------------------------------------------------------------


def test_nonzero_stark_w_widens_tolerance_above_fwhm_inst():
    """With ``stark_w = 0.05`` (5 pm HWHM at reference), the returned
    tolerance must exceed the pure instrument FWHM. This is the canary the
    original ``getattr(transition, "stark_width_nm", 0.0)`` typo would fail:
    it returned ``fwhm_inst`` exactly for every line with non-zero stark_w.
    """
    wl_nm = 500.0
    R = 10000.0
    fwhm_inst = wl_nm / R  # 0.05 nm
    transition = _MockTransition(stark_w=0.05)

    tol = get_wavelength_tolerance(wavelength_nm=wl_nm, transition=transition, resolving_power=R)

    assert tol > fwhm_inst, (
        f"Tolerance must include the Stark contribution, "
        f"but got tol={tol:.6f} nm <= fwhm_inst={fwhm_inst:.6f} nm. "
        "The ``stark_w`` attribute is being read as 0 — typo regression."
    )


def test_tolerance_matches_protocol_quadrature_formula():
    """Spot-check the literal sqrt(fwhm_inst**2 + (2*HWHM)**2) formula.

    At the Konjević reference defaults (n_e=1e17, T=10000K) the stark_w
    value 0.05 nm HWHM at REF_NE=1e16 scales linearly with n_e to
    0.5 nm HWHM -> 1.0 nm FWHM (alpha-only T scaling is unity here since
    T == T_ref). fwhm_inst = 500/10000 = 0.05 nm. Expected:
    sqrt(0.05^2 + 1.0^2) ≈ 1.00125.
    """
    transition = _MockTransition(stark_w=0.05)
    tol = get_wavelength_tolerance(
        wavelength_nm=500.0, transition=transition, resolving_power=10000.0
    )
    expected = math.sqrt(0.05**2 + 1.0**2)
    assert math.isclose(tol, expected, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Dynamic n_e/T plumbing.
# ---------------------------------------------------------------------------


def test_default_matches_explicit_konjevic_reference_call():
    """Calling without (n_e, T) must equal calling with explicit Konjević
    reference values (1e17 cm^-3, 10000 K). This pins the default contract
    so callers without live plasma state still get the literature-justified
    tolerance."""
    transition = _MockTransition(stark_w=0.02, stark_alpha=0.5)

    tol_default = get_wavelength_tolerance(
        wavelength_nm=400.0, transition=transition, resolving_power=10000.0
    )
    tol_explicit = get_wavelength_tolerance(
        wavelength_nm=400.0,
        transition=transition,
        resolving_power=10000.0,
        n_e_cm3=1.0e17,
        T_K=10000.0,
    )
    assert math.isclose(tol_default, tol_explicit, rel_tol=1e-12)


def test_stark_term_scales_linearly_with_electron_density():
    """At fixed T, the Stark FWHM scales linearly with n_e
    (analytic power-law in :func:`stark_hwhm`). Doubling n_e from 1e17 to
    2e17 must double the Stark FWHM contribution; 10x from 1e17 to 1e18
    must give 10x. We back out the Stark contribution from the quadrature
    formula and assert that scaling.
    """
    wl_nm = 500.0
    R = 10000.0
    fwhm_inst = wl_nm / R  # 0.05 nm
    transition = _MockTransition(stark_w=0.02, stark_alpha=0.5)

    def _stark_term(n_e_cm3: float) -> float:
        tol = get_wavelength_tolerance(
            wavelength_nm=wl_nm,
            transition=transition,
            resolving_power=R,
            n_e_cm3=n_e_cm3,
            T_K=10000.0,
        )
        # Invert sqrt(fwhm_inst**2 + omega_stark**2) -> omega_stark
        return math.sqrt(max(tol**2 - fwhm_inst**2, 0.0))

    s_1e17 = _stark_term(1.0e17)
    s_2e17 = _stark_term(2.0e17)
    s_1e18 = _stark_term(1.0e18)

    # Sanity: at 1e17 the term is non-zero (otherwise the test below is vacuous).
    assert s_1e17 > 0.0

    assert math.isclose(s_2e17, 2.0 * s_1e17, rel_tol=1e-6), (
        f"Stark FWHM should double when n_e doubles, "
        f"got s(1e17)={s_1e17:.6f}, s(2e17)={s_2e17:.6f}"
    )
    assert math.isclose(s_1e18, 10.0 * s_1e17, rel_tol=1e-6), (
        f"Stark FWHM should be 10x at 10x n_e, " f"got s(1e17)={s_1e17:.6f}, s(1e18)={s_1e18:.6f}"
    )


def test_stark_term_uses_radiation_stark_hwhm_helper():
    """Reverse-direction consistency: the Stark FWHM extracted from
    ``get_wavelength_tolerance`` must equal exactly ``2 * stark_hwhm(...)``
    with the same parameters. This wires the identification layer back to
    the single source of truth in ``cflibs/radiation/stark.py``.
    """
    wl_nm = 281.6  # Al II 281.6 nm — the line Aragón 2011 calls out for
    # Stark-shift-induced overlap risk.
    R = 10000.0
    fwhm_inst = wl_nm / R
    stark_w = 0.01
    stark_alpha = 0.7
    transition = _MockTransition(stark_w=stark_w, stark_alpha=stark_alpha)

    n_e_cm3 = 3.0e17
    T_K = 12000.0

    tol = get_wavelength_tolerance(
        wavelength_nm=wl_nm,
        transition=transition,
        resolving_power=R,
        n_e_cm3=n_e_cm3,
        T_K=T_K,
    )
    omega_stark_helper = 2.0 * stark_hwhm(n_e_cm3, T_K, stark_w, stark_alpha)
    expected = math.sqrt(fwhm_inst**2 + omega_stark_helper**2)
    assert math.isclose(tol, expected, rel_tol=1e-12)


# ---------------------------------------------------------------------------
# Edge cases preserved by the fix.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("stark_w", [None, 0.0, -1e-9])
def test_missing_or_nonpositive_stark_w_falls_back(stark_w):
    """No Stark data -> protocol fallback (0.05 nm by default)."""
    transition = _MockTransition(stark_w=stark_w)
    tol = get_wavelength_tolerance(
        wavelength_nm=500.0, transition=transition, resolving_power=5000.0
    )
    assert tol == pytest.approx(0.05)


def test_no_transition_returns_fallback():
    assert get_wavelength_tolerance(
        wavelength_nm=500.0, transition=None, resolving_power=5000.0
    ) == pytest.approx(0.05)


def test_custom_fallback_honored_when_no_stark_data():
    assert get_wavelength_tolerance(
        wavelength_nm=500.0,
        transition=None,
        resolving_power=5000.0,
        fallback=0.02,
    ) == pytest.approx(0.02)
