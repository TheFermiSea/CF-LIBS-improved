"""Tests for the extended real-data benchmark adapters (bead A2)."""

from __future__ import annotations

import itertools
import logging

import numpy as np
import pytest

from cflibs.atomic.masses import STANDARD_ATOMIC_MASSES
from cflibs.benchmark.adapters_extended import (
    MANIFEST,
    SpectrumTruth,
    enforce_strictly_increasing,
    iter_chemcam_calibration_spectra,
    iter_csa_planetary_spectra,
    iter_emslibs2019_spectra,
    iter_gibbons2024_spectra,
    iter_silva2022_spectra,
)

pytestmark = pytest.mark.integration

VALID_SYMBOLS = frozenset(STANDARD_ATOMIC_MASSES)

# (adapter name, factory, probe size). Probes are intentionally small so a
# test run touches only the head of each dataset (the EMSLIBS h5 sample
# blocks are ~160 MB each).
_PROBES = [
    ("csa_planetary", iter_csa_planetary_spectra, 6),
    ("chemcam_calib", iter_chemcam_calibration_spectra, 6),
    ("emslibs2019", iter_emslibs2019_spectra, 3),
    ("silva2022", iter_silva2022_spectra, 4),
    ("gibbons2024", iter_gibbons2024_spectra, 4),
]


def _probe(factory, k):
    records = list(itertools.islice(factory(), k))
    if not records:
        pytest.skip("dataset files not available in data/ (adapter skip-with-log)")
    return records


def _assert_record_contract(record) -> None:
    assert isinstance(record, tuple) and len(record) == 4
    spectrum_id, wavelength, intensity, truth = record
    assert isinstance(spectrum_id, str) and spectrum_id
    assert isinstance(wavelength, np.ndarray)
    assert isinstance(intensity, np.ndarray)
    assert isinstance(truth, SpectrumTruth)
    assert wavelength.ndim == 1 and wavelength.shape == intensity.shape
    # Strictly increasing wavelengths in a plausible optical range (nm).
    assert np.all(np.diff(wavelength) > 0)
    assert wavelength[0] >= 150.0 and wavelength[-1] <= 1100.0
    # Truth contract.
    assert isinstance(truth.elements_present, frozenset) and truth.elements_present
    assert truth.elements_present <= VALID_SYMBOLS
    if truth.composition_wt is None:
        assert truth.composition_basis == "presence_only"
    else:
        assert truth.composition_basis == "element_wt"
        values = list(truth.composition_wt.values())
        assert all(0.0 <= v <= 100.0 for v in values)
        assert 0.0 < sum(values) <= 105.0  # element basis (O excluded)
        assert set(truth.composition_wt) <= VALID_SYMBOLS
    assert truth.notes  # provenance is mandatory


@pytest.mark.parametrize("name,factory,k", _PROBES, ids=[p[0] for p in _PROBES])
def test_adapter_yields_contract_conforming_records(name, factory, k):
    for record in _probe(factory, k):
        _assert_record_contract(record)


@pytest.mark.parametrize("name,factory,k", _PROBES, ids=[p[0] for p in _PROBES])
def test_adapter_is_deterministic(name, factory, k):
    first = [(r[0], r[1][0], float(np.nansum(r[2]))) for r in _probe(factory, 2)]
    second = [(r[0], r[1][0], float(np.nansum(r[2]))) for r in _probe(factory, 2)]
    assert first == second


@pytest.mark.parametrize("name,factory,k", _PROBES, ids=[p[0] for p in _PROBES])
def test_adapter_skips_with_log_when_data_absent(name, factory, k, tmp_path, caplog):
    """An empty data dir must yield nothing and log a warning, never raise."""
    with caplog.at_level(logging.WARNING):
        records = list(factory(data_dir=tmp_path))
    assert records == []
    assert any("Skipping" in message or "missing" in message for message in caplog.messages)


def test_manifest_shape_and_unique_names():
    assert len(MANIFEST) == 5
    names = [entry[0] for entry in MANIFEST]
    assert len(set(names)) == len(names)
    for name, factory, tags, notes in MANIFEST:
        assert callable(factory)
        assert isinstance(tags, tuple) and tags
        assert isinstance(notes, str) and notes


def test_spectrum_truth_basis_consistency_enforced():
    with pytest.raises(ValueError):
        SpectrumTruth(
            elements_present=frozenset({"Fe"}),
            composition_wt=None,
            composition_basis="element_wt",
        )
    with pytest.raises(ValueError):
        SpectrumTruth(
            elements_present=frozenset({"Fe"}),
            composition_wt={"Fe": 10.0},
            composition_basis="presence_only",
        )


def test_enforce_strictly_increasing_dedupes_and_sorts():
    wl = np.array([200.0, 201.0, 201.0, 200.5, 202.0])
    inten = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    out_wl, out_inten = enforce_strictly_increasing(wl, inten)
    assert np.all(np.diff(out_wl) > 0)
    assert list(out_wl) == [200.0, 200.5, 201.0, 202.0]
    # Stable: first occurrence of the duplicated 201.0 wins.
    assert list(out_inten) == [1.0, 4.0, 2.0, 5.0]
    # No-op on already-strict grids.
    same_wl, same_inten = enforce_strictly_increasing(out_wl, out_inten)
    assert np.array_equal(same_wl, out_wl) and np.array_equal(same_inten, out_inten)


# ---------------------------------------------------------------------------
# Dataset-specific truth spot checks (values transcribed from the datasets'
# own certificate files; these only run when the data is present).
# ---------------------------------------------------------------------------


def test_csa_bhvo2_truth_matches_certificate():
    """BHVO-2: SiO2 0.499 mass fraction -> Si 23.33 wt%; FeO 0.123 -> Fe 9.56."""
    records = {r[0]: r for r in iter_csa_planetary_spectra() if "BasaltBHVO2" in r[0]}
    if not records:
        pytest.skip("CSA dataset not available")
    # Both the 200AVG large-set and 1000AVG subset spectra must be present.
    assert any("large200" in key for key in records)
    assert any("subset1000" in key for key in records)
    truth = next(iter(records.values()))[3]
    assert truth.composition_wt["Si"] == pytest.approx(0.499 * 100 * 0.46744, rel=1e-3)
    assert truth.composition_wt["Fe"] == pytest.approx(0.123 * 100 * 0.77731, rel=1e-3)
    assert {"Si", "Fe", "Mg", "Ca", "Al", "Ti", "Na"} <= truth.elements_present


def test_csa_incomplete_truth_samples_are_excluded(caplog):
    """Galena/stibnite/gypsum etc. must be skipped, not yielded with bad truth."""
    with caplog.at_level(logging.WARNING):
        ids = [r[0] for r in iter_csa_planetary_spectra()]
    if not ids:
        pytest.skip("CSA dataset not available")
    # hand sample10 = galena (PbS), certified panel < 1% of mass.
    assert not any("hand sample10" in i for i in ids)
    # hand sample17 = pyrite (S uncertified), excluded by name rule.
    assert not any("hand sample17" in i for i in ids)
    # Silicate hand samples with good coverage stay (hand sample57 granite).
    assert any("hand sample57" in i for i in ids)


def test_chemcam_agv2_truth_matches_certificate():
    """AGV2 row: SiO2 59.3 wt% -> Si 27.72; FeOT 6.02 -> Fe 4.679 wt%."""
    record = next(
        (r for r in iter_chemcam_calibration_spectra() if r[0] == "chemcam/AGV2/0"),
        None,
    )
    if record is None:
        pytest.skip("ChemCam calibration dataset not available")
    truth = record[3]
    assert truth.composition_wt["Si"] == pytest.approx(59.3 * 0.46744, rel=1e-3)
    assert truth.composition_wt["Fe"] == pytest.approx(6.02 * 0.77731, rel=1e-3)
    assert truth.resolving_power == pytest.approx(2000.0)


def test_emslibs_presence_panel_is_class_intersection():
    pytest.importorskip("h5py")
    records = list(itertools.islice(iter_emslibs2019_spectra(shots_per_sample=1), 1))
    if not records:
        pytest.skip("EMSLIBS dataset not available")
    sid, wl, inten, truth = records[0]
    assert sid.startswith("emslibs2019/train/")
    assert truth.composition_basis == "presence_only"
    # Sample 001 is class 1 (U ore): the only analyte certified >= cutoff in
    # all 15 class-1 members of MIXED_composition is K.
    assert truth.elements_present == frozenset({"K"})
    assert "intersection" in truth.notes


def test_silva_panel_and_units_caveats():
    records = list(itertools.islice(iter_silva2022_spectra(), 1))
    if not records:
        pytest.skip("Silva 2022 dataset not available")
    truth = records[0][3]
    assert truth.elements_present == frozenset({"P", "K", "Ca", "Mg"})
    assert truth.composition_wt is None
    assert "not vanadium" in truth.notes  # the V column trap is documented


def test_gibbons_nitrogen_quantitation_and_endmembers():
    pytest.importorskip("openpyxl")
    records = list(iter_gibbons2024_spectra())
    if not records:
        pytest.skip("Gibbons dataset not available")
    mixtures = [r for r in records if r[3].composition_wt is not None]
    endmembers = [r for r in records if r[3].composition_wt is None]
    assert len(mixtures) == 150  # 6 NO3 levels x 5 cations x 5 replicates
    assert len(endmembers) == 25  # 5 pure salts x 5 replicates
    # 0.5 wt% NO3- -> 0.1130 wt% N (x 14.007/62.004).
    truth = next(r[3] for r in records if r[0].startswith("gibbons2024/0.5CaNO3"))
    assert truth.composition_wt["N"] == pytest.approx(0.5 * 14.007 / 62.004, rel=1e-4)
    assert truth.elements_present == frozenset({"N", "Ca"})
    # MGS matrix blanks must not be yielded (no derivable truth).
    assert not any("MGS_" in r[0] for r in records)
