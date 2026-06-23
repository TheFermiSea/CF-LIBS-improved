"""Isolate the R8 flip to seg[2] and dump per-model BIC margins (ref vs kernel).

The full-axis diagnostic showed seg[2] (473-905 nm) flips shift(ref)->affine(dev).
This slices that segment and runs BOTH calibrators in single-axis mode with each
candidate model forced, dumping (bic, n_inliers, rmse) so we can see the exact
BIC margin the kernel's affine wins by and design the parsimony tiebreak.

    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu PYTHONPATH=$PWD python scripts/diag_seg2_bic_margin.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.preprocess.preprocessing import detect_peaks_auto
from cflibs.inversion.preprocess.wavelength_calibration import (
    _build_reference_line_pool,
    calibrate_wavelength_axis,
)

REPO = Path(__file__).resolve().parent.parent
REAL = REPO / "data" / "bhvo2_usgs" / "chemcam_bhvo2_loc1_spectrum.csv"
DB = REPO / "ASD_da" / "libs_production.db"
ELEMENTS = ["Si", "Ti", "Al", "Fe", "Mn", "Mg", "Ca", "Na", "K"]

# seg[2] range from the full-axis diagnostic.
SEG_LO, SEG_HI = 473.1842, 905.57349
CAL_INLIER_TOL_NM = 0.08
CAL_PAIR_WINDOW_NM = 2.0
CAL_H_AFFINE = 256


def load_real():
    arr = np.loadtxt(REAL, delimiter=",", skiprows=1)
    return arr[:, 0].astype(float), arr[:, 1].astype(float)


def main() -> None:
    wl, inten = load_real()
    db = AtomicDatabase(str(DB))
    m = (wl >= SEG_LO - 1e-6) & (wl <= SEG_HI + 1e-6)
    seg_wl, seg_in = wl[m], inten[m]
    print(f"seg[2]: N={seg_wl.size}  range=[{seg_wl.min():.3f}, {seg_wl.max():.3f}]nm")

    # ----- reference: force each model -----
    print("\n=== REFERENCE calibrate_wavelength_axis (per model) ===")
    for model in ("shift", "affine"):
        r = calibrate_wavelength_axis(
            wavelength=seg_wl,
            intensity=seg_in,
            atomic_db=db,
            elements=ELEMENTS,
            mode="auto",
            candidate_models=(model,),
            inlier_tolerance_nm=CAL_INLIER_TOL_NM,
            max_pair_window_nm=CAL_PAIR_WINDOW_NM,
            apply_quality_gate=False,
            random_seed=42 + 2,  # reference uses random_seed + index; seg index 2
        )
        corr = np.asarray(r.corrected_wavelength) - seg_wl
        print(f"  {model:>7}: bic={r.bic:.4f} n_in={r.n_inliers} rmse={r.rmse_nm:.5f} "
              f"coef={tuple(round(c,6) for c in r.coefficients)} "
              f"corr=[{corr.min():+.4f},{corr.max():+.4f}]")
    r_auto = calibrate_wavelength_axis(
        wavelength=seg_wl, intensity=seg_in, atomic_db=db, elements=ELEMENTS,
        mode="auto", candidate_models=("shift", "affine"),
        inlier_tolerance_nm=CAL_INLIER_TOL_NM, max_pair_window_nm=CAL_PAIR_WINDOW_NM,
        apply_quality_gate=False, random_seed=42 + 2,
    )
    print(f"  AUTO winner: model={r_auto.model} bic={r_auto.bic:.4f} n_in={r_auto.n_inliers}")

    # ----- kernel: force each model -----
    import jax.numpy as jnp
    from cflibs.jitpipe import calibrate as _C

    peaks, _b, _n = detect_peaks_auto(seg_wl, seg_in, threshold_factor=4.0)
    peak_idx = np.asarray([p[0] for p in peaks], dtype=int)
    peak_wl = np.asarray([p[1] for p in peaks], dtype=float)
    peak_amp = np.maximum(seg_in[peak_idx], 1e-12)
    line_wl, line_strength = _build_reference_line_pool(
        atomic_db=db, elements=ELEMENTS,
        wavelength_min=float(seg_wl.min()) - CAL_PAIR_WINDOW_NM,
        wavelength_max=float(seg_wl.max()) + CAL_PAIR_WINDOW_NM,
        max_lines_per_element=60, min_aki_gk=3e3, reference_temperature_K=10000.0,
    )
    order = np.argsort(line_wl)
    line_wl, line_strength = line_wl[order], line_strength[order]

    def _p2(n, fl):
        p = 1
        while p < max(n, fl):
            p *= 2
        return p

    p_max, l_max, w_max = _p2(peak_wl.size, 64), _p2(line_wl.size, 64), _p2(seg_wl.size, 64)

    def pad(a, n, fill=0.0):
        out = np.full(n, fill, dtype=float)
        out[: a.size] = a
        return out

    inputs = dict(
        peak_wl=jnp.asarray(pad(peak_wl, p_max)),
        peak_amp=jnp.asarray(pad(peak_amp, p_max)),
        peak_mask=jnp.asarray(np.r_[np.ones(peak_wl.size, bool), np.zeros(p_max - peak_wl.size, bool)]),
        line_wl=jnp.asarray(pad(line_wl, l_max, fill=1e9)),
        line_strength=jnp.asarray(pad(line_strength, l_max)),
        line_mask=jnp.asarray(np.r_[np.ones(line_wl.size, bool), np.zeros(l_max - line_wl.size, bool)]),
        wavelength=jnp.asarray(pad(seg_wl, w_max, fill=float(seg_wl[-1]))),
        wl_mask=jnp.asarray(np.r_[np.ones(seg_wl.size, bool), np.zeros(w_max - seg_wl.size, bool)]),
    )
    print(f"\n=== KERNEL calibrate_axis_kernel (per model) "
          f"[peaks={peak_wl.size} lines={line_wl.size}] ===")
    for mid, name in ((_C.MODEL_SHIFT, "shift"), (_C.MODEL_AFFINE, "affine")):
        res = _C.calibrate_axis_kernel(
            **inputs, inlier_tolerance_nm=CAL_INLIER_TOL_NM,
            max_pair_window_nm=CAL_PAIR_WINDOW_NM, apply_quality_gate=False,
            candidate_models=(mid,), h_affine=CAL_H_AFFINE, seed=42 + 2,
        )
        corr = np.asarray(res.corrected_wavelength)[: seg_wl.size] - seg_wl
        print(f"  {name:>7}: bic={float(res.bic):.4f} n_in={int(res.n_inliers)} "
              f"rmse={float(res.rmse_nm):.5f} coef={np.asarray(res.coefficients).round(6).tolist()} "
              f"corr=[{corr.min():+.4f},{corr.max():+.4f}]")
    res_auto = _C.calibrate_axis_kernel(
        **inputs, inlier_tolerance_nm=CAL_INLIER_TOL_NM,
        max_pair_window_nm=CAL_PAIR_WINDOW_NM, apply_quality_gate=False,
        candidate_models=(_C.MODEL_SHIFT, _C.MODEL_AFFINE), h_affine=CAL_H_AFFINE, seed=42 + 2,
    )
    print(f"  AUTO winner: model_id={int(res_auto.model_id)} bic={float(res_auto.bic):.4f} "
          f"n_in={int(res_auto.n_inliers)}")


if __name__ == "__main__":
    main()
