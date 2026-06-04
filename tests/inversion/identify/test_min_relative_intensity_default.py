"""
Regression test for the relative-intensity floor on the production `invert`
path (physics-audit / composition-pipeline-diagnosis: IDENT-RYDBERG).

With no relative-intensity floor (the old ``cflibs invert`` default of
``min_relative_intensity=None``), weak high-lying (Rydberg) transitions with
``relative_intensity ~ 0`` — e.g. the Na I 413-421 nm lines at E_k ~ 5 eV,
unobservable in a ~1 eV ps-LIBS plasma — are matched to bright wrong-element
peaks. Because the Boltzmann ordinate ``ln(I λ / gA)`` divides by their tiny
A_ki, those points extrapolate the closure intercept to a huge spurious
abundance (Na ~ 77-98 wt% vs the BHVO-2 certified 1.65 wt%), crushing every
other major element.

Setting a sane non-None floor (the shipped example config uses 100.0) prunes
the Rydberg lines and removes this catastrophic per-element failure mode.

This test reproduces the pruning on the real BHVO-2 ChemCam spectrum, so it
requires the production atomic database.
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


def test_no_floor_admits_na_rydberg_lines(production_db):
    """Without a floor, the spurious Na Rydberg lines reach the solver."""
    if not _BHVO2_SPECTRUM.exists():
        pytest.skip("BHVO-2 spectrum not available")
    det = _detect(production_db, None)
    na_nm = {round(o.wavelength_nm, 1) for o in det.observations if o.element == "Na"}
    # At least one of the known spurious Rydberg lines is present with no floor.
    assert na_nm & _NA_RYDBERG_NM, (
        "Expected the no-floor path to admit Na Rydberg lines; got Na lines " f"{sorted(na_nm)}."
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
