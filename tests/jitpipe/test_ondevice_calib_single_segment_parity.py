"""aa9e regression: on-device segmented calibration delegates single-segment
(seam-free) spectra to the reference robust core; multi-segment broadband stays
on the on-device path.

Single-segment (seam-free) is exactly where the on-device deterministic stratified
RANSAC diverged from the reference random-600 RANSAC (102/289 synthetic axes + the
real aalto muscoviteE35/adulariaE11 board failures), flipping marginal lines and the
solve/fail outcome. The fix (cflibs/jitpipe/host.py:_ondevice_calibrate_segmented,
``seams.size == 0`` branch) delegates those spectra to ``_ld_calibrate`` for parity,
while the multi-segment per-segment path (BHVO-2: 3-segment) is untouched.
"""

from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
REAL = REPO / "data" / "bhvo2_usgs" / "chemcam_bhvo2_loc1_spectrum.csv"
DB = REPO / "ASD_da" / "libs_production.db"
ELEMENTS = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K"]

pytestmark = [pytest.mark.requires_jax, pytest.mark.requires_db]


def _load_bhvo2():
    if not REAL.exists():
        pytest.skip("BHVO-2 fixture not present")
    arr = np.loadtxt(REAL, delimiter=",", skiprows=1)
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


@pytest.fixture(scope="module")
def db():
    if not DB.exists():
        pytest.skip("atomic DB not present")
    from cflibs.atomic.database import AtomicDatabase

    return AtomicDatabase(str(DB))


@pytest.fixture(scope="module")
def cfg():
    from cflibs.inversion.pipeline import build_pipeline_config

    return build_pipeline_config(ELEMENTS, preset="raw")


def test_single_segment_delegates_to_reference(db, cfg):
    """A seam-free spectrum: on-device calibration == reference, byte-identical."""
    from cflibs.inversion.preprocess.wavelength_calibration import detect_ccd_seams
    from cflibs.jitpipe.host import _ld_calibrate, _ondevice_calibrate_segmented

    wl, inten = _load_bhvo2()
    # The red CCD channel slice is a real single-segment (seam-free) spectrum.
    m = (wl >= 473.5) & (wl <= 905.0)
    swl, sin = wl[m], inten[m]
    assert (
        detect_ccd_seams(swl, ratio_threshold=3.0, window=51).size == 0
    ), "fixture must be seam-free for this test to exercise the single-segment branch"

    dev_corr, dev_ok, dev_qp = _ondevice_calibrate_segmented(swl, sin, db, cfg.elements, cfg)
    ref = _ld_calibrate(swl, sin, db, cfg.elements, cfg)
    ref_corr = np.asarray(ref.corrected_wavelength, dtype=float)[: swl.size]

    assert np.array_equal(
        np.asarray(dev_corr, dtype=float), ref_corr
    ), "single-segment on-device axis must match the reference byte-for-byte (aa9e fallback)"
    assert dev_ok == bool(ref.success)
    assert dev_qp == bool(ref.quality_passed)


def test_multi_segment_stays_on_device(db, cfg):
    """Full BHVO-2 is 3-segment: takes the on-device path, parity-faithful to reference."""
    from cflibs.inversion.preprocess.wavelength_calibration import detect_ccd_seams
    from cflibs.jitpipe.host import _ld_calibrate, _ondevice_calibrate_segmented

    wl, inten = _load_bhvo2()
    # seams > 0 -> structurally cannot hit the single-segment fallback branch.
    assert (
        detect_ccd_seams(wl, ratio_threshold=3.0, window=51).size > 0
    ), "BHVO-2 must be multi-segment"

    dev_corr, dev_ok, _dev_qp = _ondevice_calibrate_segmented(wl, inten, db, cfg.elements, cfg)
    dev_corr = np.asarray(dev_corr, dtype=float)
    ref = _ld_calibrate(wl, inten, db, cfg.elements, cfg)
    ref_corr = np.asarray(ref.corrected_wavelength, dtype=float)[: wl.size]

    assert dev_ok, "on-device multi-segment calibration should succeed on BHVO-2"
    # On-device per-segment kernels track the reference to the documented J2 floor.
    assert np.max(np.abs(dev_corr - ref_corr)) <= 1e-3
