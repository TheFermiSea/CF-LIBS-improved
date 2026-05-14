"""Regression tests for the detection-coverage logging contract.

These tests pin the four-layer telemetry contract exposed by
``cflibs.inversion.identify._coverage`` and surfaced on
``ElementIdentificationResult.parameters``:

- ``n_peaks_detected``
- ``elements_with_zero_db_lines_in_range``
- ``elements_with_zero_peak_matches``
- ``elements_below_fingerprint_floor``

The headline scenario (per task spec, item 4) constructs a synthetic
spectrum where Si has lines in the wavelength range but the matcher
fails to pair them with any detected peak.  We assert
``elements_with_zero_peak_matches`` includes ``"Si"`` and that the L1
log payload reports the correct peak count + baseline method.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np
import pytest

from cflibs.inversion.identify._coverage import (
    CoverageTracker,
    count_lines_in_range,
    log_peak_detection,
    merge_coverage_into_parameters,
)
from cflibs.inversion.identify.alias import ALIASIdentifier
from cflibs.inversion.identify.comb import CombIdentifier
from cflibs.inversion.preprocess.preprocessing import (
    BaselineMethod,
    detect_peaks_auto,
)


# ---------------------------------------------------------------------------
# Unit tests for the helper module
# ---------------------------------------------------------------------------


def test_coverage_tracker_payload_shape():
    """The build_payload contract matches the task-spec keys."""
    tracker = CoverageTracker(spectrum_id="sp1", identifier_name="alias")
    tracker.set_n_peaks(42)
    tracker.record_db_lines("Si", 5)
    tracker.record_db_lines("Mg", 0)
    tracker.record_peak_matches("Si", 0)
    tracker.record_peak_matches("Mg", 0)
    tracker.record_fingerprint("Si", passed=False, score=0.0, floor=0.05)
    tracker.record_fingerprint("Mg", passed=False, score=0.0, floor=0.05)

    payload = tracker.build_payload()
    assert set(payload.keys()) == {
        "n_peaks_detected",
        "elements_with_zero_db_lines_in_range",
        "elements_with_zero_peak_matches",
        "elements_below_fingerprint_floor",
    }
    assert payload["n_peaks_detected"] == 42
    # Si has DB lines but no matches -> smoking-gun list.
    assert "Si" in payload["elements_with_zero_peak_matches"]
    # Mg has no DB lines -> classified at the L2 layer, not L3.
    assert "Mg" in payload["elements_with_zero_db_lines_in_range"]
    # Mg also has zero matches but L2-zero takes precedence -- it is
    # *not* added to the smoking-gun L3 list when L2 is already zero.
    assert "Mg" not in payload["elements_with_zero_peak_matches"]
    assert {"Si", "Mg"} <= set(payload["elements_below_fingerprint_floor"])


def test_coverage_tracker_l3_only_smoking_gun():
    """L3 smoking-gun list flags only elements with DB lines but no matches."""
    tracker = CoverageTracker(spectrum_id="sp1", identifier_name="alias")
    tracker.record_db_lines("Si", 12)  # has lines in window
    tracker.record_peak_matches("Si", 0)  # but matcher paired none
    payload = tracker.build_payload()
    assert payload["elements_with_zero_peak_matches"] == ["Si"]
    assert payload["elements_with_zero_db_lines_in_range"] == []


def test_count_lines_in_range_filters_correctly():
    class _T:
        def __init__(self, wl):
            self.wavelength_nm = wl

    transitions = [_T(199.5), _T(200.0), _T(250.0), _T(800.0), _T(800.5)]
    assert count_lines_in_range(transitions, 200.0, 800.0) == 3
    assert count_lines_in_range([], 200.0, 800.0) == 0
    assert count_lines_in_range(None, 200.0, 800.0) == 0


def test_merge_coverage_into_parameters_preserves_other_keys():
    base = {"resolving_power": 5000.0, "extra": "value"}
    payload = {"n_peaks_detected": 17, "elements_below_fingerprint_floor": []}
    merged = merge_coverage_into_parameters(base, payload)
    assert merged["resolving_power"] == 5000.0
    assert merged["extra"] == "value"
    assert merged["n_peaks_detected"] == 17
    # Returned dict is a copy; caller's base is not mutated.
    assert "n_peaks_detected" not in base


def test_log_peak_detection_emits_info_record(caplog):
    """L1 log line is emitted at INFO with the expected fields."""
    peaks = [(0, 250.0), (10, 400.0), (20, 600.0)]
    with caplog.at_level(logging.INFO, logger="cflibs.inversion.detection_coverage"):
        payload = log_peak_detection(
            peaks,
            noise=0.5,
            baseline_method=BaselineMethod.MEDIAN,
            spectrum_id="test-spec",
            identifier_name="comb",
        )

    assert payload["n_peaks"] == 3
    assert payload["baseline_method"] == "median"
    assert payload["min_nm"] == pytest.approx(250.0)
    assert payload["max_nm"] == pytest.approx(600.0)
    # The INFO record carries spectrum_id and identifier_name so
    # downstream parsing can correlate L1 with L2/L3/L4.
    matching = [
        r for r in caplog.records if "L1 peak_detection" in r.getMessage()
    ]
    assert matching, "Expected at least one L1 INFO record"
    msg = matching[-1].getMessage()
    assert "spectrum_id=test-spec" in msg
    assert "identifier=comb" in msg
    assert "n_peaks=3" in msg


def test_log_peak_detection_handles_empty_peaks():
    payload = log_peak_detection(
        [],
        noise=0.1,
        baseline_method=BaselineMethod.AUTO,
        spectrum_id=None,
    )
    assert payload["n_peaks"] == 0
    # Wavelength summary is NaN when peaks is empty (schema stable).
    assert np.isnan(payload["min_nm"])
    assert np.isnan(payload["max_nm"])


def test_detect_peaks_auto_emits_l1_record(caplog):
    """``detect_peaks_auto`` triggers the L1 log record on every call."""
    rng = np.random.default_rng(0)
    wavelength = np.linspace(200.0, 800.0, 2000)
    intensity = np.ones_like(wavelength) * 10.0
    # A few isolated Gaussian peaks.
    for center, amp in [(300.0, 500.0), (450.0, 800.0), (650.0, 400.0)]:
        intensity += amp * np.exp(-0.5 * ((wavelength - center) / 0.1) ** 2)
    intensity += rng.normal(0.0, 0.05 * intensity.max(), size=intensity.shape)

    with caplog.at_level(logging.INFO, logger="cflibs.inversion.detection_coverage"):
        peaks, _, _ = detect_peaks_auto(wavelength, intensity)

    assert len(peaks) >= 1
    l1_records = [
        r for r in caplog.records if "L1 peak_detection" in r.getMessage()
    ]
    assert l1_records, "detect_peaks_auto must emit an L1 record"


# ---------------------------------------------------------------------------
# End-to-end regression test (task spec, item 4)
# ---------------------------------------------------------------------------


@pytest.fixture
def db_with_si_lines(tmp_path):
    """Sqlite DB with both Fe (matches the synthetic spectrum) and Si lines.

    The point of this fixture is to simulate the failure mode the
    task is investigating: an element (Si) has theoretical lines in
    the spectrum's wavelength window, but the experimental spectrum
    contains *no* peaks at those positions.  The matcher must
    therefore report zero peak matches for Si even though L2 (DB
    lines in window) is non-zero.
    """
    import sqlite3

    from cflibs.atomic.database import AtomicDatabase

    db_path = tmp_path / "fe_and_si.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE lines (
            id INTEGER PRIMARY KEY,
            element TEXT, sp_num INTEGER,
            wavelength_nm REAL, aki REAL,
            ei_ev REAL, ek_ev REAL,
            gi INTEGER, gk INTEGER,
            rel_int REAL
        )
        """
    )
    conn.execute(
        "CREATE TABLE energy_levels (element TEXT, sp_num INTEGER, "
        "g_level INTEGER, energy_ev REAL)"
    )
    conn.execute(
        "CREATE TABLE species_physics (element TEXT, sp_num INTEGER, "
        "ip_ev REAL, PRIMARY KEY (element, sp_num))"
    )

    # Fe I lines -- present in the synthetic spectrum, so peak
    # detection has real peaks to chew on.
    conn.execute(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES ('Fe', 1, 371.99, 1.0e7, 0.0, 3.33, 9, 11, 1000),
               ('Fe', 1, 373.49, 5.0e6, 0.0, 3.32, 9, 9, 500),
               ('Fe', 1, 374.95, 2.0e6, 0.0, 3.31, 9, 7, 200),
               ('Fe', 1, 382.04, 6.7e7, 0.0, 3.24, 9, 9, 800),
               ('Fe', 1, 404.58, 8.6e6, 0.86, 3.93, 7, 9, 600)
        """
    )
    # Si I lines in the same 250-450 nm window -- but the synthetic
    # spectrum will not contain peaks at any of these wavelengths,
    # so the matcher should return zero matches for Si.
    conn.execute(
        """
        INSERT INTO lines (element, sp_num, wavelength_nm, aki, ei_ev, ek_ev, gi, gk, rel_int)
        VALUES ('Si', 1, 288.16, 1.9e8, 0.78, 5.08, 3, 5, 900),
               ('Si', 1, 251.61, 1.3e8, 0.0, 4.93, 1, 3, 800),
               ('Si', 1, 252.41, 1.6e8, 0.01, 4.92, 3, 5, 700),
               ('Si', 1, 252.85, 8.5e7, 0.03, 4.93, 5, 7, 600),
               ('Si', 1, 263.13, 6.7e7, 0.78, 5.50, 3, 5, 500)
        """
    )
    conn.execute(
        """
        INSERT INTO energy_levels (element, sp_num, g_level, energy_ev)
        VALUES ('Fe', 1, 9, 0.0), ('Fe', 1, 11, 3.33),
               ('Fe', 1, 9, 3.32), ('Fe', 1, 7, 3.31),
               ('Fe', 1, 9, 3.24), ('Fe', 1, 9, 3.93),
               ('Si', 1, 1, 0.0), ('Si', 1, 3, 0.01),
               ('Si', 1, 5, 0.03), ('Si', 1, 3, 0.78),
               ('Si', 1, 5, 5.08), ('Si', 1, 3, 4.93),
               ('Si', 1, 5, 4.92), ('Si', 1, 7, 4.93),
               ('Si', 1, 5, 5.50)
        """
    )
    conn.execute(
        """
        INSERT INTO species_physics (element, sp_num, ip_ev)
        VALUES ('Fe', 1, 7.87), ('Si', 1, 8.15)
        """
    )
    conn.commit()
    conn.close()
    return AtomicDatabase(str(db_path))


@pytest.mark.requires_db
def test_si_with_db_lines_but_no_peak_matches_is_flagged(
    db_with_si_lines, synthetic_libs_spectrum
):
    """Smoking-gun scenario: Si has DB lines in window but no peaks pair.

    We construct a spectrum that contains Fe lines (so peak detection
    finds real peaks the matcher can iterate over) but *no* Si peaks
    at Si's NIST line positions.  The ALIAS identifier's
    peak-matcher should therefore record zero matches for Si.  Per
    the task:

        "Asserts the logging records elements_with_zero_peak_matches
        >= ['Si']"

    ALIAS is chosen for this test because it uses an explicit
    peak-existence match (tolerance window around each theoretical
    line), so absence-of-peak cleanly maps to ``n_peak_matches=0``.
    Comb's template-correlation matcher is more permissive and can
    fire on baseline / noise structure even where no peak exists,
    which would defeat the test's intent.  (The coverage telemetry
    is wired into all four identifiers; only the regression test
    chosen for the L3 smoking gun is identifier-specific.)
    """
    identifier = ALIASIdentifier(
        atomic_db=db_with_si_lines,
        elements=["Fe", "Si"],
        resolving_power=2000.0,
    )

    # Fe-only spectrum: real peaks at Fe wavelengths, nothing at Si
    # wavelengths.  The DB fixture has Si lines in 250-290 nm window,
    # so L2 (DB lines in window) is > 0 for Si, but L3 (peak matches)
    # is 0 -- the precise scenario the task asks for.
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 500.0), (374.95, 200.0)]},
        wavelength_range=(240.0, 450.0),
        noise_level=0.01,
    )

    result = identifier.identify(
        spectrum["wavelength"],
        spectrum["intensity"],
        spectrum_id="si-missing-test",
    )

    params = result.parameters
    # Coverage keys are present and well-typed.
    assert "n_peaks_detected" in params
    assert "elements_with_zero_db_lines_in_range" in params
    assert "elements_with_zero_peak_matches" in params
    assert "elements_below_fingerprint_floor" in params

    # Real peaks must have been detected (otherwise the test is
    # uninformative -- L3 zero would just be "no peaks at all").
    assert params["n_peaks_detected"] > 0

    # Headline assertion: Si is flagged as having zero peak matches.
    # Per task: "asserts the logging records elements_with_zero_peak_matches
    # >= ['Si']".
    assert "Si" in params["elements_with_zero_peak_matches"], (
        "Si should be flagged at L3 -- it has DB lines in window but no peak "
        f"matches. Got: {params['elements_with_zero_peak_matches']}"
    )

    # Si must also have failed the fingerprint floor (L4) since it
    # had no matched lines.
    assert "Si" in params["elements_below_fingerprint_floor"]

    # And Si should NOT be in the L2 list -- it has DB lines in window.
    assert "Si" not in params["elements_with_zero_db_lines_in_range"]


@pytest.mark.requires_db
def test_identify_default_call_signature_unchanged(
    atomic_db, synthetic_libs_spectrum
):
    """``identify(wavelength, intensity)`` still works without ``spectrum_id``.

    Coverage logging is opt-in -- when ``spectrum_id`` is not provided,
    the identifier behaves byte-identically to dev (the F1 regression
    gate depends on this).
    """
    identifier = CombIdentifier(atomic_db, elements=["Fe"])
    spectrum = synthetic_libs_spectrum(
        elements={"Fe": [(371.99, 1000.0), (373.49, 500.0)]},
        noise_level=0.01,
    )
    result = identifier.identify(spectrum["wavelength"], spectrum["intensity"])
    # Coverage payload is still present (it is always-on), but no
    # spectrum_id was passed so the log records use the sentinel.
    assert "n_peaks_detected" in result.parameters
