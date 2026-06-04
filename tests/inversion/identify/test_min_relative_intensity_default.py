"""
Regression test for Na-Rydberg suppression on the production detection path
(physics-audit / composition-pipeline-diagnosis: IDENT-RYDBERG).

Weak high-lying (Rydberg) transitions with ``relative_intensity ~ 0`` — e.g.
the Na I 413-421 nm lines at E_k ~ 5 eV, unobservable in a ~1 eV ps-LIBS
plasma — used to be matched to bright wrong-element peaks when no
relative-intensity floor was set. Because the Boltzmann ordinate
``ln(I λ / gA)`` divides by their tiny A_ki, those points extrapolated the
closure intercept to a huge spurious abundance (Na ~ 77-98 wt% vs the BHVO-2
certified 1.65 wt%), crushing every other major element. The historical guard
was an absolute ``min_relative_intensity`` floor — but that also deleted whole
real elements whose tabulated rel_int is small or NULL.

The detection-cascade fix replaces the floor with a gA-Boltzmann comb-strength
ranking (which Boltzmann-suppresses these high-E_k lines out of the comb) plus
a shift-coherence veto. Na Rydberg lines are therefore pruned *even with no
floor*. These tests assert both: the no-floor path no longer admits the
Rydberg lines, and the legacy floor still prunes them when explicitly set.

These run on the real BHVO-2 ChemCam spectrum, so they require the production
atomic database.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cflibs.inversion.identify.line_detection import detect_line_observations
from cflibs.io.spectrum import load_spectrum

pytestmark = [pytest.mark.requires_db, pytest.mark.integration]

_BHVO2_SPECTRUM = Path("data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv")
_ELEMENTS = ["Si", "Ti", "Al", "Fe", "Mg", "Ca", "Na", "K"]
# Spurious Na I Rydberg lines (E_k ~ 5.0-5.1 eV, relative_intensity ~ 0).
_NA_RYDBERG_NM = {413.1, 414.4, 417.2, 420.2, 421.6}


def _detect(db, min_relative_intensity):
    wavelength, intensity = load_spectrum(str(_BHVO2_SPECTRUM))
    return detect_line_observations(
        wavelength,
        intensity,
        db,
        elements=_ELEMENTS,
        wavelength_tolerance_nm=0.1,
        min_peak_height=0.01,
        peak_width_nm=0.1,
        min_relative_intensity=min_relative_intensity,
    )


def test_no_floor_excludes_na_rydberg_lines(production_db):
    """The no-floor path no longer admits the spurious Na Rydberg lines.

    With the gA-Boltzmann comb-strength ranking + shift-coherence veto, the
    high-E_k Na 413-421 nm lines are Boltzmann-suppressed out of the comb, so
    ``min_relative_intensity=None`` no longer re-detonates the Na blowup that
    the absolute floor used to guard against — but without the floor's
    collateral deletion of real majors.
    """
    if not _BHVO2_SPECTRUM.exists():
        pytest.skip("BHVO-2 spectrum not available")
    det = _detect(production_db, None)
    na_nm = {round(o.wavelength_nm, 1) for o in det.observations if o.element == "Na"}
    assert not (na_nm & _NA_RYDBERG_NM), (
        "The no-floor detection path must not admit the weak high-E_k Na "
        f"Rydberg lines (gA-comb suppression); survivors: "
        f"{sorted(na_nm & _NA_RYDBERG_NM)}."
    )


def test_floor_prunes_na_rydberg_lines(production_db):
    """A 100.0 relative-intensity floor removes the spurious Na Rydberg lines."""
    if not _BHVO2_SPECTRUM.exists():
        pytest.skip("BHVO-2 spectrum not available")
    det = _detect(production_db, 100.0)
    na_nm = {round(o.wavelength_nm, 1) for o in det.observations if o.element == "Na"}
    assert not (na_nm & _NA_RYDBERG_NM), (
        "The relative-intensity floor must prune the weak high-E_k Na Rydberg "
        f"lines; survivors: {sorted(na_nm & _NA_RYDBERG_NM)}."
    )
