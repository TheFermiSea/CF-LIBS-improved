"""Diagnose the R8 segmented-calibration model-flip (on-device vs reference).

Runs the REFERENCE ``calibrate_wavelength_axis_segmented`` and the ON-DEVICE
``_ondevice_calibrate_segmented`` on the real ChemCam BHVO-2 spectrum, and
reports exactly which segment's model class flips and the corrected-axis
divergence it causes. This is the concrete picture behind host.py:1156-1167.

Run:
    JAX_ENABLE_X64=1 PYTHONPATH=$PWD python scripts/diag_segmented_calib_flip.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.pipeline import build_pipeline_config
from cflibs.jitpipe.host import _ld_calibrate, _ondevice_calibrate_segmented

REPO = Path(__file__).resolve().parent.parent
REAL = REPO / "data" / "bhvo2_usgs" / "chemcam_bhvo2_loc1_spectrum.csv"
DB = REPO / "ASD_da" / "libs_production.db"
ELEMENTS = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K"]


def load_real():
    arr = np.loadtxt(REAL, delimiter=",", skiprows=1)
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


def main() -> None:
    wl, inten = load_real()
    db = AtomicDatabase(str(DB))
    cfg = build_pipeline_config(ELEMENTS, preset="raw")
    print(f"axis: N={wl.size}  range=[{wl.min():.2f}, {wl.max():.2f}] nm")
    print(f"affine_coverage_gate={cfg.affine_coverage_gate}")

    # ---- reference ----
    t0 = time.perf_counter()
    ref = _ld_calibrate(wl, inten, db, ELEMENTS, cfg)
    t_ref = time.perf_counter() - t0
    ref_axis = np.asarray(ref.corrected_wavelength, dtype=float)
    print("\n=== REFERENCE ===")
    print(f"  time={t_ref:.2f}s  model={ref.model}  success={ref.success} "
          f"quality_passed={ref.quality_passed}")
    print(f"  global rmse={ref.rmse_nm:.4f}nm n_inliers={ref.n_inliers}")
    print(f"  details keys: {list(ref.details.keys())}")
    seg_diag = ref.details.get("segment_diagnostics", [])
    print(f"  n_segment_diag={len(seg_diag)}")
    for i, s in enumerate(seg_diag):
        if isinstance(s, dict):
            print(f"    seg[{i}] model={s.get('model')!r:>10} status={s.get('status')!r} "
                  f"rmse={s.get('rmse_nm')!r} inliers={s.get('n_inliers')} "
                  f"range=[{s.get('wl_min')},{s.get('wl_max')}] "
                  f"reverted={s.get('reverted', s.get('revert'))}")
        else:
            print(f"    seg[{i}] {s}")

    # ---- on-device ----
    t0 = time.perf_counter()
    dev_axis, dev_success, dev_qp = _ondevice_calibrate_segmented(
        wl, inten, db, ELEMENTS, cfg
    )
    t_dev = time.perf_counter() - t0
    dev_axis = np.asarray(dev_axis, dtype=float)
    print("\n=== ON-DEVICE ===")
    print(f"  time(compile+run)={t_dev:.2f}s  success={dev_success} "
          f"quality_passed={dev_qp}")

    from cflibs.jitpipe import calibrate as _C  # noqa
    print(f"  MODEL_SHIFT={_C.MODEL_SHIFT} MODEL_AFFINE={_C.MODEL_AFFINE}")

    # ---- divergence ----
    n = min(ref_axis.size, dev_axis.size, wl.size)
    d = np.abs(ref_axis[:n] - dev_axis[:n])
    imax = int(np.argmax(d))
    print("\n=== DIVERGENCE (ref vs dev corrected axis) ===")
    print(f"  max|Δ|={d.max():.5f}nm at idx={imax} (wl={wl[imax]:.3f}nm)")
    print(f"  mean|Δ|={d.mean():.5f}nm  >0.02nm in {(d > 0.02).sum()} / {n} samples")
    ref_corr = ref_axis[:n] - wl[:n]
    dev_corr = dev_axis[:n] - wl[:n]
    print(f"  ref correction range=[{ref_corr.min():.4f}, {ref_corr.max():.4f}]nm")
    print(f"  dev correction range=[{dev_corr.min():.4f}, {dev_corr.max():.4f}]nm")
    edges = np.linspace(wl[0], wl[n - 1], 11)
    print("  per-band max|Δ| (nm):")
    for b in range(10):
        m = (wl[:n] >= edges[b]) & (wl[:n] < edges[b + 1])
        if m.any():
            print(f"    [{edges[b]:.1f}-{edges[b+1]:.1f}] max|Δ|={d[m].max():.4f} "
                  f"ref_corr={ref_corr[m].mean():+.4f} dev_corr={dev_corr[m].mean():+.4f}")


if __name__ == "__main__":
    main()
