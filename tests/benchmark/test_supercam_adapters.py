"""Tests for the SuperCam labcal / SCCT benchmark adapters (bead w7).

Follows the extended-adapter test pattern: lazy data access, skip-with-log
when ``data/supercam_calib/`` is absent, small islice probes for the cheap
checks and ``slow`` marks for anything that walks the 721 MB lab table or
the 547 FITS products.
"""

from __future__ import annotations

import csv
import itertools
import logging
import re

import numpy as np
import pytest

from cflibs.atomic.masses import STANDARD_ATOMIC_MASSES
from cflibs.benchmark.adapters_extended import (
    DATA_ROOT,
    MANIFEST,
    SpectrumTruth,
    iter_supercam_labcal_spectra,
    iter_supercam_scct_spectra,
)
from cflibs.benchmark.datasets import supercam_labcal

pytestmark = pytest.mark.integration

VALID_SYMBOLS = frozenset(STANDARD_ATOMIC_MASSES)

SUPERCAM_ROOT = DATA_ROOT / "supercam_calib"
LABCAL_CSV = SUPERCAM_ROOT / supercam_labcal.LABCAL_CSV_RELPATH

# The two real inter-spectrometer gaps in the shared 7,933-channel axis (nm).
EXPECTED_GAPS = ((341.0, 380.0), (464.0, 538.0))


def _probe(factory, k):
    records = list(itertools.islice(factory(), k))
    if not records:
        pytest.skip("supercam_calib data not available (adapter skip-with-log)")
    return records


def _assert_record_contract(record) -> None:
    assert isinstance(record, tuple) and len(record) == 4
    spectrum_id, wavelength, intensity, truth = record
    assert isinstance(spectrum_id, str) and spectrum_id
    assert isinstance(wavelength, np.ndarray)
    assert isinstance(intensity, np.ndarray)
    assert isinstance(truth, SpectrumTruth)
    assert wavelength.ndim == 1 and wavelength.shape == intensity.shape
    # Strictly increasing within channels; the two physical gaps are fine.
    assert np.all(np.diff(wavelength) > 0)
    assert 243.0 <= wavelength[0] <= 244.5 and 852.0 <= wavelength[-1] <= 853.5
    assert np.all(np.isfinite(intensity))  # NaN axes break the pipeline
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


def _stream_labcal_rows():
    """Independent raw read of the lab CSV (no adapter code in the loop)."""
    with open(LABCAL_CSV, newline="", encoding="latin-1") as fh:
        reader = csv.reader(fh)
        next(reader)  # category row
        names = next(reader)
        for row in reader:
            yield dict(zip(names[:127], row[:127]))


# ---------------------------------------------------------------------------
# Contract conformance + the spectral-gap geometry
# ---------------------------------------------------------------------------


def test_labcal_contract_and_gaps():
    records = _probe(iter_supercam_labcal_spectra, 4)
    for record in records:
        _assert_record_contract(record)
    wavelength = records[0][1]
    diffs = np.diff(wavelength)
    for lo, hi in EXPECTED_GAPS:
        in_gap = (wavelength[:-1] > lo) & (wavelength[1:] < hi)
        assert diffs[in_gap].max() > 30.0, f"expected spectral gap in {lo}-{hi} nm"


def test_scct_contract():
    pytest.importorskip("astropy")
    for record in _probe(iter_supercam_scct_spectra, 3):
        _assert_record_contract(record)
        # spectrum_id is the CL1 filename stem.
        assert record[0].startswith("scam_") and "_cl1_" in record[0]


@pytest.mark.parametrize(
    "factory", [iter_supercam_labcal_spectra, iter_supercam_scct_spectra], ids=["labcal", "scct"]
)
def test_adapter_skips_with_log_when_data_absent(factory, tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        records = list(factory(data_dir=tmp_path))
    assert records == []
    assert any("Skipping" in message for message in caplog.messages)


def test_manifest_registration_and_tiers():
    entries = {name: (tags, tier) for name, _f, tags, tier, _n in MANIFEST}
    assert entries["supercam_labcal"][1] == "optimization"
    assert entries["supercam_scct"][1] == "holdout"  # real-Mars adoption gate
    assert "supercam" in entries["supercam_labcal"][0]
    assert "mars" in entries["supercam_scct"][0]


# ---------------------------------------------------------------------------
# shift==0 filtering + dedup (capped iteration; raw-CSV cross-check)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_labcal_capped_iteration_matches_raw_shift0_rows():
    records = _probe(iter_supercam_labcal_spectra, 50)
    assert len(records) == 50  # the table holds 1,193 shift==0 base spectra
    ids = [r[0] for r in records]
    assert len(set(ids)) == 50
    # Independent raw read: the first 50 shift==0 Keep rows, in file order.
    expected = []
    for meta in _stream_labcal_rows():
        if meta["shift"] != "0" or meta["Remove_from_all"].strip().lower() == "remove":
            continue
        expected.append(meta["file"])
        if len(expected) == 50:
            break
    if len(expected) < 50:
        pytest.skip("lab CSV not available for the raw cross-check")
    assert ids == expected


# ---------------------------------------------------------------------------
# Truth spot checks against the raw CSV (independent read)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_labcal_truth_spot_checks_two_standards():
    """Two lab standards, one oxide each, vs hard-coded stoichiometric factors."""
    records = []
    seen_targets = set()
    for record in itertools.islice(iter_supercam_labcal_spectra(), 10):
        target = re.search(r"Target_Name='([^']*)'", record[3].notes)
        assert target is not None
        if target.group(1) not in seen_targets:
            seen_targets.add(target.group(1))
            records.append(record)
        if len(records) == 2:
            break
    if len(records) < 2:
        pytest.skip("supercam_calib data not available")
    raw = {}
    wanted = {r[0] for r in records}
    for meta in _stream_labcal_rows():
        if meta["shift"] == "0" and meta["file"] in wanted:
            raw[meta["file"]] = meta
        if len(raw) == len(wanted):
            break
    # Standard 1: SiO2 wt% -> Si via 0.46744; standard 2: CaO wt% -> Ca via 0.71470.
    (sid_a, _, _, truth_a), (sid_b, _, _, truth_b) = records
    assert truth_a.composition_wt["Si"] == pytest.approx(
        float(raw[sid_a]["SiO2"]) * 0.46744, rel=1e-4
    )
    assert truth_b.composition_wt["Ca"] == pytest.approx(
        float(raw[sid_b]["CaO"]) * 0.71470, rel=1e-4
    )


# ---------------------------------------------------------------------------
# SCCT <-> labcal truth join
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_scct_truth_join_matches_lab_chip_row():
    """SCCT_LBHVO20406 (Mars) truth == lab chip LBHVO20401's certified row."""
    from cflibs.benchmark.datasets.supercam_scct import truth_for_flight_target

    if not LABCAL_CSV.is_file():
        pytest.skip("lab CSV not available")
    table = supercam_labcal.load_scct_truth_table(SUPERCAM_ROOT)
    joined = truth_for_flight_target("SCCT_LBHVO20406", table)
    assert joined is not None
    truth, provenance = joined
    assert "LBHVO20401" in provenance
    # Independent raw read of the lab chip row.
    raw = next(
        meta
        for meta in _stream_labcal_rows()
        if meta["shift"] == "0" and meta["Target_Name"] == "LBHVO20401"
    )
    assert truth.composition_wt["Si"] == pytest.approx(float(raw["SiO2"]) * 0.46744, rel=1e-4)
    assert truth.composition_wt["Fe"] == pytest.approx(float(raw["FeOT"]) * 0.77731, rel=1e-4)
    assert {"Si", "Fe", "Mg", "Ca", "Al", "Ti", "Na"} <= truth.elements_present


@pytest.mark.slow
def test_scct_every_flight_target_resolves():
    """All 23 LIBS-observed flight targets join to truth (TITANIUM presence-only)."""
    from cflibs.benchmark.datasets.supercam_scct import CL1_RELPATH, truth_for_flight_target

    cl1_dir = SUPERCAM_ROOT / CL1_RELPATH
    if not (cl1_dir.is_dir() and LABCAL_CSV.is_file()):
        pytest.skip("supercam_calib data not available")

    def _target_from_stem(stem: str):
        # ..._cl1_<inst>_<target>_______<dd>p<dd> — linear string parsing
        # (the previous regex backtracked polynomially on the underscore pad).
        head, _, dist = stem.rpartition("_")
        if not (dist and dist[0].isdigit() and "p" in dist):
            return None
        head = head.rstrip("_")
        _, sep, tail = head.partition("_cl1_")
        if not sep or "_" not in tail:
            return None
        return tail.split("_", 1)[1]

    flight_targets = set()
    for path in cl1_dir.glob("sol_*/*.fits"):
        match = _target_from_stem(path.stem)
        if match:
            flight_targets.add(match.upper())
    assert len(flight_targets) == 23
    table = supercam_labcal.load_scct_truth_table(SUPERCAM_ROOT)
    for flight in sorted(flight_targets):
        joined = truth_for_flight_target(flight, table)
        assert joined is not None, flight
        truth = joined[0]
        if flight.endswith("TITANIUM"):
            assert truth.composition_wt is None
            assert truth.elements_present == frozenset({"Ti", "Al", "V"})
        else:
            assert truth.composition_wt, flight


@pytest.mark.integration
def test_spectrum_target_names_groups_shift0_files():
    """The split-grouping helper streams the full table (no spectral parsing).

    Uncovered until PR #287's Sonar blocker proved it: a stale 4-arg _cell
    call lived here undetected because nothing called this function yet.
    """
    if not LABCAL_CSV.is_file():
        pytest.skip("supercam_calib data not available")
    groups = supercam_labcal.spectrum_target_names(SUPERCAM_ROOT)
    assert len(groups) >= 1000  # ~1,193 shift==0 base spectra
    targets = set(groups.values())
    assert len(targets) >= 300  # ~334 unique standards
    assert any("BHVO" in t.upper() for t in targets)
