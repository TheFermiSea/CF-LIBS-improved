"""Tests for the NIST SRM 1261a-1265a steel benchmark adapter."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from cflibs.benchmark.datasets.nist_steel import (
    CERTIFIED_COMPOSITIONS,
    NISTSteelComposition,
    NISTSteelDataset,
)

EXPECTED_SRM_IDS = {"1261a", "1262a", "1263a", "1264a", "1265a"}


# ---------------------------------------------------------------------------
# Composition table
# ---------------------------------------------------------------------------


def test_all_five_srms_present():
    """The five 1200-series steels must all be in the certified table."""
    assert set(CERTIFIED_COMPOSITIONS) == EXPECTED_SRM_IDS


@pytest.mark.parametrize("srm_id", sorted(EXPECTED_SRM_IDS))
def test_composition_record_shape(srm_id: str):
    comp = CERTIFIED_COMPOSITIONS[srm_id]
    assert isinstance(comp, NISTSteelComposition)
    assert comp.srm_id == srm_id
    assert comp.iron_balance is True
    assert comp.material  # non-empty
    assert comp.certificate_url.startswith("https://tsapps.nist.gov/")
    assert comp.certified_wt_pct, "certified table is empty"
    # Every certified element must have a recorded uncertainty.
    missing = set(comp.certified_wt_pct) - set(comp.uncertainty_wt_pct)
    assert not missing, f"missing uncertainties for {missing}"
    # Fe is the matrix balance and must NOT be a certified element.
    assert "Fe" not in comp.certified_wt_pct


@pytest.mark.parametrize("srm_id", sorted(EXPECTED_SRM_IDS))
def test_composition_sums_to_about_100_wt_pct(srm_id: str):
    """
    With Fe added as the matrix balance the composition sums to ~100 wt%.

    The certified-only sum must always be <= 100 (Fe excluded, so the rest
    cannot exceed the matrix). When we include the implied Fe balance, the
    total is exactly 100 by construction.
    """
    comp = CERTIFIED_COMPOSITIONS[srm_id]
    cert_sum = comp.total_certified_wt_pct
    assert 0.0 < cert_sum < 100.0, f"{srm_id}: certified sum {cert_sum} out of range"
    fe = comp.implied_iron_wt_pct
    assert fe == pytest.approx(100.0 - cert_sum, abs=1e-9)
    # Implied Fe should match the "Iron (by difference)" information value
    # from the certificate to within ~0.1 wt% (certificates round to 3 sig figs).
    fe_info = comp.information_values.get("Fe")
    if fe_info is not None:
        assert fe == pytest.approx(
            fe_info, abs=0.15
        ), f"{srm_id}: implied Fe {fe:.3f} vs information-value Fe {fe_info:.3f}"


@pytest.mark.parametrize("srm_id", sorted(EXPECTED_SRM_IDS))
def test_mass_fractions_sum_to_one(srm_id: str):
    comp = CERTIFIED_COMPOSITIONS[srm_id]
    fractions = comp.as_mass_fractions(include_iron=True)
    assert fractions["Fe"] > 0.9, f"{srm_id}: Fe should dominate the matrix"
    total = sum(fractions.values())
    assert total == pytest.approx(1.0, abs=1e-9)


def test_iron_balance_omits_fe_from_certified_when_excluded():
    comp = CERTIFIED_COMPOSITIONS["1265a"]
    fractions = comp.as_mass_fractions(include_iron=False)
    assert "Fe" not in fractions
    assert sum(fractions.values()) == pytest.approx(comp.total_certified_wt_pct / 100.0)


# ---------------------------------------------------------------------------
# Adapter behaviour
# ---------------------------------------------------------------------------


def test_available_samples_lists_all_five():
    ds = NISTSteelDataset()
    assert ds.available_samples() == sorted(EXPECTED_SRM_IDS)


def test_get_composition_round_trip():
    ds = NISTSteelDataset()
    comp = ds.get_composition("1261a")
    assert comp is CERTIFIED_COMPOSITIONS["1261a"]


def test_get_composition_unknown_raises():
    ds = NISTSteelDataset()
    with pytest.raises(KeyError, match="Unknown NIST steel SRM id"):
        ds.get_composition("9999z")


def test_get_spectrum_returns_none_when_missing(tmp_path: Path):
    """
    NIST does not publish LIBS spectra alongside the SRM Certificate of
    Analysis, so the adapter must degrade to ``None`` when the user has not
    cached anything in the data directory.
    """
    ds = NISTSteelDataset(data_dir=tmp_path)  # empty directory
    for srm_id in ds.available_samples():
        assert ds.get_spectrum(srm_id) is None


def test_get_spectrum_unknown_srm_raises(tmp_path: Path):
    ds = NISTSteelDataset(data_dir=tmp_path)
    with pytest.raises(KeyError, match="Unknown NIST steel SRM id"):
        ds.get_spectrum("9999z")


def test_get_spectrum_loads_cached_csv(tmp_path: Path):
    """When a CSV is cached the adapter wraps it in a BenchmarkSpectrum."""
    srm_id = "1261a"
    csv_path = tmp_path / f"{srm_id}.csv"
    wl = np.linspace(200.0, 900.0, 16)
    intensity = np.linspace(1.0, 16.0, 16)
    with csv_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["wavelength_nm", "intensity"])
        for w, i in zip(wl, intensity):
            writer.writerow([f"{w:.6f}", f"{i:.6f}"])

    ds = NISTSteelDataset(data_dir=tmp_path)
    spectrum = ds.get_spectrum(srm_id)
    assert spectrum is not None
    assert spectrum.spectrum_id == f"nist_srm_{srm_id}"
    assert spectrum.metadata.crm_name == f"NIST SRM {srm_id}"
    np.testing.assert_allclose(spectrum.wavelength_nm, wl)
    np.testing.assert_allclose(spectrum.intensity, intensity)
    # true_composition includes Fe and sums to ~1.0
    assert sum(spectrum.true_composition.values()) == pytest.approx(1.0, abs=1e-9)
    assert "Fe" in spectrum.true_composition
    # composition_uncertainty is in mass fraction (wt% / 100)
    expected_C_unc = CERTIFIED_COMPOSITIONS[srm_id].uncertainty_wt_pct["C"] / 100.0
    assert spectrum.composition_uncertainty["C"] == pytest.approx(expected_C_unc)
