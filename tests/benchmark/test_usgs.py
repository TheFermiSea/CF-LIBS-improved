"""Tests for the USGS BHVO-2 / AGV-2 / BCR-2 / G-2 geochemical-standard adapter."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from cflibs.benchmark.datasets.usgs import (
    CERTIFIED_COMPOSITIONS,
    OXIDE_TO_ELEMENT_FACTOR,
    USGSDataset,
    USGSStandardComposition,
    oxide_to_element_wt,
)

EXPECTED_STANDARD_IDS = {"BHVO-2", "AGV-2", "BCR-2", "G-2"}


# ---------------------------------------------------------------------------
# Stoichiometric oxide -> element conversion
# ---------------------------------------------------------------------------


def test_pure_sio2_yields_silicon_mass_fraction_4674():
    """
    Sanity check: 100% SiO2 must convert to ~46.744% Si by mass.

    Si / (Si + 2*O) = 28.085 / (28.085 + 2 * 15.999) ~= 0.46744.
    """
    elements = oxide_to_element_wt({"SiO2": 100.0})
    assert elements == {"Si": pytest.approx(46.744, abs=1e-2)}


def test_pure_fe2o3_yields_iron_mass_fraction_6994():
    """
    100% Fe2O3 must convert to ~69.944% Fe by mass.

    2*Fe / (2*Fe + 3*O) = 2 * 55.845 / (2 * 55.845 + 3 * 15.999) ~= 0.69944.
    """
    elements = oxide_to_element_wt({"Fe2O3": 100.0})
    assert elements == {"Fe": pytest.approx(69.944, abs=1e-2)}


def test_total_iron_oxide_alias_resolves_to_fe():
    """``Fe2O3T`` (total iron expressed as Fe2O3) shares the Fe2O3 factor."""
    fe_total_factor = OXIDE_TO_ELEMENT_FACTOR["Fe2O3T"]
    fe_oxide_factor = OXIDE_TO_ELEMENT_FACTOR["Fe2O3"]
    assert fe_total_factor == fe_oxide_factor

    elements = oxide_to_element_wt({"Fe2O3T": 10.0, "FeO": 5.0})
    # Both should map to Fe and sum.
    assert set(elements) == {"Fe"}
    expected = 10.0 * fe_oxide_factor[1] + 5.0 * OXIDE_TO_ELEMENT_FACTOR["FeO"][1]
    assert elements["Fe"] == pytest.approx(expected, rel=1e-9)


def test_unknown_oxide_is_silently_skipped():
    """LOI / H2O / CO2 etc. are unknown oxides and must not raise."""
    elements = oxide_to_element_wt({"SiO2": 50.0, "LOI": 0.8, "H2O": 0.3})
    assert set(elements) == {"Si"}


def test_oxide_factors_within_physical_bounds():
    """Every cation mass fraction must be in (0, 1)."""
    for oxide, (element, factor) in OXIDE_TO_ELEMENT_FACTOR.items():
        assert 0.0 < factor < 1.0, f"{oxide} -> {element} factor {factor} unphysical"


# ---------------------------------------------------------------------------
# Composition table
# ---------------------------------------------------------------------------


def test_all_four_standards_present():
    """The four headline USGS rocks must all be in the certified table."""
    assert set(CERTIFIED_COMPOSITIONS) == EXPECTED_STANDARD_IDS


@pytest.mark.parametrize("standard_id", sorted(EXPECTED_STANDARD_IDS))
def test_composition_record_shape(standard_id: str):
    comp = CERTIFIED_COMPOSITIONS[standard_id]
    assert isinstance(comp, USGSStandardComposition)
    assert comp.standard_id == standard_id
    assert comp.material  # non-empty
    assert comp.rock_type in {"basalt", "andesite", "granite"}
    assert comp.locality
    assert comp.oxide_wt_percent, "oxide table is empty"
    # Every oxide must have a recorded uncertainty.
    missing = set(comp.oxide_wt_percent) - set(comp.oxide_uncertainty_wt_percent)
    assert not missing, f"{standard_id}: missing oxide uncertainties for {missing}"
    # Iron must be reported as Fe2O3T per Jochum 2016.
    assert "Fe2O3T" in comp.oxide_wt_percent, f"{standard_id}: Fe must be Fe2O3T"


@pytest.mark.parametrize("standard_id", sorted(EXPECTED_STANDARD_IDS))
def test_major_oxides_sum_to_about_100_wt_pct(standard_id: str):
    """
    Major oxides (SiO2, TiO2, Al2O3, Fe2O3T, MnO, MgO, CaO, Na2O, K2O, P2O5)
    must sum to roughly 95-100 wt%; the missing few percent are LOI, H2O, CO2.
    """
    comp = CERTIFIED_COMPOSITIONS[standard_id]
    total = comp.total_oxide_wt_percent
    assert 95.0 < total < 102.0, f"{standard_id}: oxide sum {total:.2f} out of physical range"


@pytest.mark.parametrize("standard_id", sorted(EXPECTED_STANDARD_IDS))
def test_element_wt_percent_consistent_with_oxide_table(standard_id: str):
    """Cation mass percents must match a fresh ``oxide_to_element_wt`` call."""
    comp = CERTIFIED_COMPOSITIONS[standard_id]
    expected = oxide_to_element_wt(comp.oxide_wt_percent)
    actual = comp.element_wt_percent
    assert set(actual) == set(expected)
    for element, value in expected.items():
        assert actual[element] == pytest.approx(value, rel=1e-9)


def test_silicon_ordering_basalt_to_granite():
    """
    Si content must increase basalt < andesite < granite, sanity-checking the
    petrologic spread the four standards were chosen for.
    """
    si_bhvo = CERTIFIED_COMPOSITIONS["BHVO-2"].element_wt_percent["Si"]
    si_bcr = CERTIFIED_COMPOSITIONS["BCR-2"].element_wt_percent["Si"]
    si_agv = CERTIFIED_COMPOSITIONS["AGV-2"].element_wt_percent["Si"]
    si_g2 = CERTIFIED_COMPOSITIONS["G-2"].element_wt_percent["Si"]
    # BHVO-2 < BCR-2 (both basalts but BCR-2 is more evolved) < AGV-2 < G-2.
    assert si_bhvo < si_bcr < si_agv < si_g2


def test_uncertainty_propagates_with_stoichiometric_factor():
    """
    The cation-uncertainty propagation must scale by the stoichiometric factor
    when only one oxide carries that cation; e.g. Si only comes from SiO2.
    """
    comp = CERTIFIED_COMPOSITIONS["BHVO-2"]
    sio2_sigma = comp.oxide_uncertainty_wt_percent["SiO2"]
    si_factor = OXIDE_TO_ELEMENT_FACTOR["SiO2"][1]
    expected_si_sigma = si_factor * sio2_sigma
    assert comp.element_uncertainty_wt_percent["Si"] == pytest.approx(expected_si_sigma, rel=1e-9)


def test_mass_fractions_are_under_unity():
    """
    Cation-only mass fractions must be < 1.0 (oxygen excluded). Basalts have
    the highest cation total because they are oxygen-poor relative to
    granites; granites have the lowest.
    """
    for standard_id, comp in CERTIFIED_COMPOSITIONS.items():
        fractions = comp.as_mass_fractions()
        total = sum(fractions.values())
        assert 0.4 < total < 0.7, f"{standard_id}: cation total {total:.3f} unphysical"


# ---------------------------------------------------------------------------
# Adapter behaviour
# ---------------------------------------------------------------------------


def test_available_samples_lists_all_four():
    ds = USGSDataset()
    assert ds.available_samples() == sorted(EXPECTED_STANDARD_IDS)


def test_get_composition_round_trip():
    ds = USGSDataset()
    comp = ds.get_composition("BHVO-2")
    assert comp is CERTIFIED_COMPOSITIONS["BHVO-2"]


def test_get_composition_unknown_raises():
    ds = USGSDataset()
    with pytest.raises(KeyError, match="Unknown USGS standard id"):
        ds.get_composition("FAKE-1")


def test_get_spectrum_returns_none_when_missing(tmp_path: Path):
    """
    USGS does not publish LIBS spectra alongside the Information Sheet, so the
    adapter must degrade to ``None`` when the user has not cached anything in
    the data directory.
    """
    ds = USGSDataset(data_dir=tmp_path)  # empty directory
    for standard_id in ds.available_samples():
        assert ds.get_spectrum(standard_id) is None


def test_get_spectrum_returns_none_when_data_dir_missing(tmp_path: Path):
    """A nonexistent data_dir must not raise -- it just yields ``None``."""
    ds = USGSDataset(data_dir=tmp_path / "does_not_exist")
    assert ds.get_spectrum("BHVO-2") is None


def test_get_spectrum_unknown_standard_raises(tmp_path: Path):
    ds = USGSDataset(data_dir=tmp_path)
    with pytest.raises(KeyError, match="Unknown USGS standard id"):
        ds.get_spectrum("FAKE-1")


def test_get_spectrum_loads_cached_csv(tmp_path: Path):
    """When a CSV is cached the adapter wraps it in a BenchmarkSpectrum."""
    standard_id = "BHVO-2"
    csv_path = tmp_path / f"{standard_id}.csv"
    wl = np.linspace(200.0, 900.0, 16)
    intensity = np.linspace(1.0, 16.0, 16)
    with csv_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["wavelength_nm", "intensity"])
        for w, i in zip(wl, intensity):
            writer.writerow([f"{w:.6f}", f"{i:.6f}"])

    ds = USGSDataset(data_dir=tmp_path)
    spectrum = ds.get_spectrum(standard_id)
    assert spectrum is not None
    assert spectrum.spectrum_id == "usgs_bhvo-2"
    assert spectrum.dataset_id == USGSDataset.DATASET_NAME
    assert spectrum.metadata.crm_name == f"USGS {standard_id}"
    assert spectrum.metadata.crm_source == "USGS"
    assert spectrum.spectrum_kind == "geostandard"
    assert spectrum.annotations["rock_type"] == "basalt"
    np.testing.assert_allclose(spectrum.wavelength_nm, wl)
    np.testing.assert_allclose(spectrum.intensity, intensity)
    # true_composition is the cation mass-fraction vector (oxygen-free, so
    # sum < 1.0 -- the BenchmarkSpectrum truth_type must be ASSAY-comparable
    # for the warning logic but the closure does not need to equal 1).
    total = sum(spectrum.true_composition.values())
    assert 0.4 < total < 0.7
    # composition_uncertainty values are mass fractions (wt% / 100).
    assert spectrum.composition_uncertainty["Si"] > 0.0
    assert spectrum.composition_uncertainty["Si"] < 0.01


def test_get_spectrum_accepts_underscored_filename(tmp_path: Path):
    """``BHVO_2.csv`` (underscored) must be picked up just like ``BHVO-2.csv``."""
    csv_path = tmp_path / "BHVO_2.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["wavelength", "intensity"])
        writer.writerow(["200.0", "1.0"])
        writer.writerow(["900.0", "2.0"])
    ds = USGSDataset(data_dir=tmp_path)
    spectrum = ds.get_spectrum("BHVO-2")
    assert spectrum is not None
    assert spectrum.spectrum_id == "usgs_bhvo-2"
