"""Tests for BHVO-2 (USGS) and NIST SRM 612 CRM dataset loaders.

Verifies that:
- ``_load_bhvo2_usgs`` and ``_load_nist_srm_612`` return a BenchmarkDataset
  with ≥10 spectra when at least 10 CSV files are present.
- Every spectrum's ``true_composition`` matches the certified-composition
  table in ``REFERENCE_COMPOSITIONS[dataset_id]``.
- Wavelength and intensity arrays are non-empty and all-finite.
- Both loaders return ``None`` gracefully when the data directory is absent
  or empty (pre-ingest state).
- ``build_dataset_registry`` aggregates both datasets when both are present
  and returns an empty list when neither directory exists.

Tests use ``tmp_path`` fixtures with synthetic CSV files so they run without
Ivy's real spectra on disk.  The full integration check (real data, ≥10
spectra, real composition values) is performed separately once the
spectra-ingest PR lands on dev.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from cflibs.benchmark.loaders import (
    _load_bhvo2_usgs,
    _load_crm_dataset,
    _load_nist_srm_612,
    build_dataset_registry,
)
from cflibs.benchmark.reference_compositions import REFERENCE_COMPOSITIONS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_N_SPECTRA = 10  # minimum spectrum count the task requires


def _write_synthetic_csvs(directory: Path, n: int = _N_SPECTRA) -> None:
    """Write *n* minimal two-column CSVs into *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    wl = np.linspace(200.0, 900.0, 500)
    for i in range(n):
        csv_path = directory / f"shot_{i:03d}.csv"
        intensity = rng.exponential(scale=1000.0, size=len(wl))
        with csv_path.open("w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["wavelength_nm", "intensity"])
            for w, iv in zip(wl, intensity):
                writer.writerow([f"{w:.6f}", f"{iv:.6f}"])


# ---------------------------------------------------------------------------
# _load_bhvo2_usgs
# ---------------------------------------------------------------------------


class TestLoadBhvo2Usgs:
    def test_returns_none_when_directory_absent(self, tmp_path: Path):
        """Loader must return None, not raise, when the directory does not exist."""
        result = _load_bhvo2_usgs(data_dir=tmp_path)
        assert result is None

    def test_returns_none_when_directory_empty(self, tmp_path: Path):
        """Loader must return None when the directory exists but contains no CSVs."""
        (tmp_path / "bhvo2_usgs").mkdir()
        result = _load_bhvo2_usgs(data_dir=tmp_path)
        assert result is None

    def test_returns_dataset_with_correct_name(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "bhvo2_usgs")
        ds = _load_bhvo2_usgs(data_dir=tmp_path)
        assert ds is not None
        assert ds.name == "bhvo2_usgs"

    def test_spectrum_count_equals_csv_count(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "bhvo2_usgs", n=_N_SPECTRA)
        ds = _load_bhvo2_usgs(data_dir=tmp_path)
        assert ds is not None
        assert ds.n_spectra >= _N_SPECTRA

    def test_true_composition_matches_reference_table(self, tmp_path: Path):
        """Every spectrum's true_composition must equal REFERENCE_COMPOSITIONS["bhvo2_usgs"]."""
        _write_synthetic_csvs(tmp_path / "bhvo2_usgs")
        ds = _load_bhvo2_usgs(data_dir=tmp_path)
        assert ds is not None
        expected = dict(REFERENCE_COMPOSITIONS["bhvo2_usgs"])
        for spec in ds.spectra:
            assert spec.true_composition == expected, (
                f"Spectrum {spec.spectrum_id} true_composition "
                f"{spec.true_composition} != certified {expected}"
            )

    def test_wavelength_and_intensity_non_empty_and_finite(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "bhvo2_usgs")
        ds = _load_bhvo2_usgs(data_dir=tmp_path)
        assert ds is not None
        for spec in ds.spectra:
            assert len(spec.wavelength_nm) > 0, f"{spec.spectrum_id} wavelength is empty"
            assert len(spec.intensity) > 0, f"{spec.spectrum_id} intensity is empty"
            assert np.all(np.isfinite(spec.wavelength_nm)), (
                f"{spec.spectrum_id} wavelength contains non-finite values"
            )
            assert np.all(np.isfinite(spec.intensity)), (
                f"{spec.spectrum_id} intensity contains non-finite values"
            )

    def test_dataset_id_set_on_every_spectrum(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "bhvo2_usgs")
        ds = _load_bhvo2_usgs(data_dir=tmp_path)
        assert ds is not None
        for spec in ds.spectra:
            assert spec.dataset_id == "bhvo2_usgs"

    def test_elements_list_matches_composition_keys(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "bhvo2_usgs")
        ds = _load_bhvo2_usgs(data_dir=tmp_path)
        assert ds is not None
        expected_elements = sorted(REFERENCE_COMPOSITIONS["bhvo2_usgs"].keys())
        assert list(ds.elements) == expected_elements


# ---------------------------------------------------------------------------
# _load_nist_srm_612
# ---------------------------------------------------------------------------


class TestLoadNistSrm612:
    def test_returns_none_when_directory_absent(self, tmp_path: Path):
        result = _load_nist_srm_612(data_dir=tmp_path)
        assert result is None

    def test_returns_none_when_directory_empty(self, tmp_path: Path):
        (tmp_path / "nist_srm_612").mkdir()
        result = _load_nist_srm_612(data_dir=tmp_path)
        assert result is None

    def test_returns_dataset_with_correct_name(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "nist_srm_612")
        ds = _load_nist_srm_612(data_dir=tmp_path)
        assert ds is not None
        assert ds.name == "nist_srm_612"

    def test_spectrum_count_equals_csv_count(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "nist_srm_612", n=_N_SPECTRA)
        ds = _load_nist_srm_612(data_dir=tmp_path)
        assert ds is not None
        assert ds.n_spectra >= _N_SPECTRA

    def test_true_composition_matches_reference_table(self, tmp_path: Path):
        """Every spectrum's true_composition must equal REFERENCE_COMPOSITIONS["nist_srm_612"]."""
        _write_synthetic_csvs(tmp_path / "nist_srm_612")
        ds = _load_nist_srm_612(data_dir=tmp_path)
        assert ds is not None
        expected = dict(REFERENCE_COMPOSITIONS["nist_srm_612"])
        for spec in ds.spectra:
            assert spec.true_composition == expected, (
                f"Spectrum {spec.spectrum_id} true_composition "
                f"{spec.true_composition} != certified {expected}"
            )

    def test_wavelength_and_intensity_non_empty_and_finite(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "nist_srm_612")
        ds = _load_nist_srm_612(data_dir=tmp_path)
        assert ds is not None
        for spec in ds.spectra:
            assert len(spec.wavelength_nm) > 0, f"{spec.spectrum_id} wavelength is empty"
            assert len(spec.intensity) > 0, f"{spec.spectrum_id} intensity is empty"
            assert np.all(np.isfinite(spec.wavelength_nm)), (
                f"{spec.spectrum_id} wavelength contains non-finite values"
            )
            assert np.all(np.isfinite(spec.intensity)), (
                f"{spec.spectrum_id} intensity contains non-finite values"
            )

    def test_dataset_id_set_on_every_spectrum(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "nist_srm_612")
        ds = _load_nist_srm_612(data_dir=tmp_path)
        assert ds is not None
        for spec in ds.spectra:
            assert spec.dataset_id == "nist_srm_612"

    def test_elements_list_matches_composition_keys(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "nist_srm_612")
        ds = _load_nist_srm_612(data_dir=tmp_path)
        assert ds is not None
        expected_elements = sorted(REFERENCE_COMPOSITIONS["nist_srm_612"].keys())
        assert list(ds.elements) == expected_elements

    def test_nist_srm_612_has_only_major_elements(self, tmp_path: Path):
        """SRM 612 composition table has only Si, Al, Ca, Na (matrix glass majors)."""
        _write_synthetic_csvs(tmp_path / "nist_srm_612")
        ds = _load_nist_srm_612(data_dir=tmp_path)
        assert ds is not None
        assert set(ds.elements) == {"Si", "Al", "Ca", "Na"}


# ---------------------------------------------------------------------------
# _load_crm_dataset — shared error paths
# ---------------------------------------------------------------------------


class TestLoadCrmDatasetErrors:
    def test_raises_for_unregistered_dataset_id(self, tmp_path: Path):
        """An unknown dataset_id must raise ValueError, not return None."""
        (tmp_path / "mystery_material").mkdir()
        # Write one CSV so the directory-absent early-return is bypassed.
        _write_synthetic_csvs(tmp_path / "mystery_material", n=1)
        with pytest.raises(ValueError, match="No reference composition registered"):
            _load_crm_dataset("mystery_material", data_dir=tmp_path)


# ---------------------------------------------------------------------------
# build_dataset_registry
# ---------------------------------------------------------------------------


class TestBuildDatasetRegistry:
    def test_empty_when_neither_directory_exists(self, tmp_path: Path):
        result = build_dataset_registry(data_dir=tmp_path)
        assert result == []

    def test_returns_one_dataset_when_only_bhvo2_present(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "bhvo2_usgs")
        result = build_dataset_registry(data_dir=tmp_path)
        assert len(result) == 1
        assert result[0].name == "bhvo2_usgs"

    def test_returns_one_dataset_when_only_nist_present(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "nist_srm_612")
        result = build_dataset_registry(data_dir=tmp_path)
        assert len(result) == 1
        assert result[0].name == "nist_srm_612"

    def test_returns_both_datasets_when_both_present(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "bhvo2_usgs")
        _write_synthetic_csvs(tmp_path / "nist_srm_612")
        result = build_dataset_registry(data_dir=tmp_path)
        assert len(result) == 2
        names = {ds.name for ds in result}
        assert names == {"bhvo2_usgs", "nist_srm_612"}

    def test_both_datasets_have_correct_compositions(self, tmp_path: Path):
        _write_synthetic_csvs(tmp_path / "bhvo2_usgs")
        _write_synthetic_csvs(tmp_path / "nist_srm_612")
        result = build_dataset_registry(data_dir=tmp_path)
        for ds in result:
            expected = dict(REFERENCE_COMPOSITIONS[ds.name])
            for spec in ds.spectra:
                assert spec.true_composition == expected
