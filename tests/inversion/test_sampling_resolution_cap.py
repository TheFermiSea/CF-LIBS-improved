"""Regression tests for the sampling-resolution detection cap (bead qitd).

Structural pathology
--------------------
The shared detection entry (:func:`detect_and_select_lines`) used a fixed
matching tolerance (0.1 nm) and integration width (0.2 nm). Those absolute
constants silently assume a moderate-resolution spectrometer (~0.05-0.1
nm/pixel). On a high-resolution, finely-sampled instrument -- the Silva 2022
tropical-soil echelle data sample at ~0.011 nm/pixel -- a 0.1 nm tolerance spans
~9 sampling steps, so the +/-tolerance match window covers a large fraction of
the densely-peaked axis. Comb matching then degenerates to random coincidence:
every dense-catalog confounder accrues matches, no real element clears the
``matched / total_peaks`` precision gate, and the shift-coherence veto removes
the spurious survivors -- so *every* Silva spectrum raised
``ValueError: No usable spectral lines detected for inversion`` (12/12).

The fix ties the tolerance/width to the instrument's actual sampling step with a
*tighten-only* cap (``min``): coarsely-sampled instruments sit below the cap and
are byte-identical; only a finely-sampled instrument is brought down to its
sampling-resolution scale.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from cflibs.inversion.identify.line_detection import LineDetectionResult
from cflibs.inversion.pipeline import (
    SAMPLING_TOLERANCE_PX,
    SAMPLING_WIDTH_PX,
    detect_and_select_lines,
)


def _empty_detection() -> LineDetectionResult:
    return LineDetectionResult(
        observations=[],
        resonance_lines=set(),
        total_peaks=0,
        matched_peaks=0,
        unmatched_peaks=0,
    )


def _capture_tolerances(wavelength):
    """Run detect_and_select_lines on ``wavelength`` and capture the tolerance /
    peak-width actually handed to ``detect_line_observations``.

    ``detect_line_observations`` is mocked (returns an empty detection) so the
    test is fast, deterministic and needs no atomic database -- it isolates the
    sampling-resolution cap, which lives upstream of detection.
    """
    intensity = np.ones_like(wavelength)
    captured: dict = {}

    def _fake_detect(**kwargs):
        captured["wavelength_tolerance_nm"] = kwargs["wavelength_tolerance_nm"]
        captured["peak_width_nm"] = kwargs["peak_width_nm"]
        return _empty_detection()

    with patch(
        "cflibs.inversion.identify.line_detection.detect_line_observations",
        side_effect=_fake_detect,
    ):
        detect_and_select_lines(
            wavelength,
            intensity,
            atomic_db=None,  # never touched: detection is mocked
            elements=["Ca", "Mg"],
            wavelength_calibration=False,  # skip calibration (needs the DB)
            wavelength_tolerance_nm=0.1,  # legacy default
            peak_width_nm=0.2,  # legacy default
        )
    return captured


class TestSamplingResolutionCap:
    def test_fine_sampling_tightens_tolerance_and_width(self):
        """A 0.011 nm/pixel echelle axis (the Silva soil class) is tightened.

        The cap must bring the 0.1/0.2 nm legacy constants down to the
        sampling-resolution scale, which is what stops the dense-forest
        coincidence-matching collapse.
        """
        wl_step = 0.011
        wavelength = np.arange(200.0, 260.0, wl_step)
        captured = _capture_tolerances(wavelength)

        assert captured["wavelength_tolerance_nm"] == pytest.approx(SAMPLING_TOLERANCE_PX * wl_step)
        assert captured["peak_width_nm"] == pytest.approx(SAMPLING_WIDTH_PX * wl_step)
        # Must be strictly tighter than the legacy constants.
        assert captured["wavelength_tolerance_nm"] < 0.1
        assert captured["peak_width_nm"] < 0.2

    def test_coarse_sampling_is_byte_identical(self):
        """A 0.1 nm/pixel ChemCam-class axis keeps the legacy 0.1/0.2 nm.

        For any instrument with ``wl_step >= 0.05 nm`` (tolerance) /
        ``>= 0.05 nm`` (width) the cap exceeds the legacy constant, so ``min``
        leaves it unchanged -- this is the zero-regression guarantee for the
        existing aalto / csa / chemcam datasets.
        """
        wl_step = 0.1
        wavelength = np.arange(240.0, 900.0, wl_step)
        captured = _capture_tolerances(wavelength)

        assert captured["wavelength_tolerance_nm"] == pytest.approx(0.1)
        assert captured["peak_width_nm"] == pytest.approx(0.2)

    def test_cap_threshold_boundary(self):
        """The cap engages exactly when ``factor * wl_step`` drops below legacy.

        Tolerance cap = ``2 * wl_step``; it bites once ``wl_step < 0.05`` nm.
        At ``wl_step = 0.06`` (cap 0.12 > 0.1) the tolerance is unchanged; at
        ``wl_step = 0.04`` (cap 0.08 < 0.1) it is tightened.
        """
        unchanged = _capture_tolerances(np.arange(300.0, 360.0, 0.06))
        assert unchanged["wavelength_tolerance_nm"] == pytest.approx(0.1)

        tightened = _capture_tolerances(np.arange(300.0, 360.0, 0.04))
        assert tightened["wavelength_tolerance_nm"] == pytest.approx(2.0 * 0.04)


@pytest.mark.requires_db
def test_silva_soil_spectrum_no_longer_crashes(production_db):
    """End-to-end: a real Silva tropical-soil spectrum now yields usable lines.

    Before the fix every Silva spectrum raised
    ``ValueError: No usable spectral lines detected for inversion``. The
    structural soil pathology (fine 0.011 nm/pixel sampling + dense matrix peak
    forest) is what this guards: with the sampling-resolution cap the candidate
    panel's lines survive detection on a high-fertility soil.
    """
    silva_root = (
        __import__("pathlib").Path(__file__).resolve().parents[2]
        / "data"
        / "silva2022_tropical_soils"
    )
    if not (silva_root / "LIBS_data.txt").exists():
        pytest.skip("Silva 2022 dataset not deployed")

    from cflibs.benchmark.datasets import silva2022

    # Sample 4 is a high-fertility soil whose phosphorus lines are detectable.
    target = "silva2022/field1/sample4"
    record = next(
        (rec for rec in silva2022.iter_spectra(silva_root) if rec[0] == target),
        None,
    )
    if record is None:
        pytest.skip(f"Silva spectrum {target} not present")
    _, wavelength, intensity, truth = record

    # Confounder panel mirrors the goal-metric scoreboard candidate policy.
    confounders = {"Ag", "Sn", "W", "Bi", "Th"}
    elements = sorted(set(truth.elements_present) | confounders)

    # The fine sampling step is the structural trigger for the cap.
    wl_step = float(np.median(np.diff(np.asarray(wavelength, dtype=float))))
    assert wl_step < 0.05, "fixture must be the finely-sampled soil class"

    selected = detect_and_select_lines(
        np.asarray(wavelength, dtype=float),
        np.asarray(intensity, dtype=float),
        production_db,
        elements,
        wavelength_tolerance_nm=0.1,
        peak_width_nm=0.2,
    )
    # The pre-fix pathology produced zero usable lines on every soil spectrum.
    assert len(selected) > 0, "soil spectrum still yields no usable spectral lines"
    detected = {obs.element for obs in selected}
    # A certified-panel element must be recovered (P is the reliably detectable
    # analyte in these soils), and no detection should be pure confounders.
    assert detected & truth.elements_present, (
        f"no certified-panel element detected (got {sorted(detected)}; "
        f"panel {sorted(truth.elements_present)})"
    )
