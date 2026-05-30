#!/usr/bin/env python
"""
Audit physical sanity of synthetic LIBS forward-model calculations.

This script performs lightweight equation-level sanity checks against expected
trends for a pure-element plasma spectrum:
1. Non-negative spectral intensity.
2. High-energy / low-energy line ratio increases with temperature.
3. Stark-dominated linewidth increases with electron density.

Outputs a JSON report for reproducible tracking.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Keep runtime stable on macOS and CI where GPU JAX can be fragile.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from cflibs.inversion.solve.bayesian import BayesianForwardModel


def _line_peak_intensity(
    wavelength: np.ndarray, spectrum: np.ndarray, center_nm: float, window_nm: float = 0.05
) -> float:
    mask = np.abs(wavelength - center_nm) <= window_nm
    if not np.any(mask):
        return float("nan")
    return float(np.max(spectrum[mask]))


def _estimate_fwhm_nm(
    wavelength: np.ndarray, spectrum: np.ndarray, center_nm: float, search_window_nm: float = 0.3
) -> Optional[float]:
    """Estimate FWHM around a target center using local half-height crossings."""
    mask = np.abs(wavelength - center_nm) <= search_window_nm
    if np.sum(mask) < 5:
        return None

    wl = wavelength[mask]
    sp = spectrum[mask]
    peak_idx_local = int(np.argmax(sp))
    peak = float(sp[peak_idx_local])
    if not np.isfinite(peak) or peak <= 0:
        return None

    half = 0.5 * peak

    # Left crossing
    left_idx = None
    for i in range(peak_idx_local, 0, -1):
        if sp[i] >= half and sp[i - 1] < half:
            left_idx = i
            break

    # Right crossing
    right_idx = None
    for i in range(peak_idx_local, len(sp) - 1):
        if sp[i] >= half and sp[i + 1] < half:
            right_idx = i
            break

    if left_idx is None or right_idx is None:
        return None

    # Linear interpolation on each side
    def interp_x(x0: float, y0: float, x1: float, y1: float, y: float) -> float:
        if abs(y1 - y0) < 1e-12:
            return x0
        t = (y - y0) / (y1 - y0)
        return x0 + t * (x1 - x0)

    x_left = interp_x(
        wl[left_idx - 1],
        sp[left_idx - 1],
        wl[left_idx],
        sp[left_idx],
        half,
    )
    x_right = interp_x(
        wl[right_idx],
        sp[right_idx],
        wl[right_idx + 1],
        sp[right_idx + 1],
        half,
    )

    fwhm = float(x_right - x_left)
    return fwhm if fwhm > 0 else None


def _pick_line_pair(model: BayesianForwardModel) -> Optional[Tuple[int, int]]:
    """Pick low-E and high-E strong lines for temperature trend check."""
    data = model.atomic_data
    n = int(len(data.wavelength_nm))
    if n < 2:
        return None

    ek = np.array(data.ek_ev)
    score = np.array(data.aki) * np.array(data.gk)

    top_k = min(40, n)
    top_idx = np.argsort(score)[-top_k:]
    if top_idx.size < 2:
        return None

    low_idx = int(top_idx[np.argmin(ek[top_idx])])
    high_idx = int(top_idx[np.argmax(ek[top_idx])])
    if low_idx == high_idx:
        return None

    if float(ek[high_idx] - ek[low_idx]) < 0.3:
        return None

    return low_idx, high_idx


def _pick_stark_line(model: BayesianForwardModel) -> Optional[Tuple[int, bool]]:
    data = model.atomic_data
    stark = np.array(data.stark_w)
    score = np.array(data.aki) * np.array(data.gk)
    finite = np.where(np.isfinite(stark) & (stark > 0))[0]
    if finite.size > 0:
        return int(finite[np.argmax(score[finite])]), True
    if score.size == 0:
        return None
    # Fallback to strongest line; model will use internal Stark estimate.
    return int(np.argmax(score)), False


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    model = BayesianForwardModel(
        db_path=args.db_path,
        elements=[args.element],
        wavelength_range=(args.lambda_min, args.lambda_max),
        pixels=args.pixels,
        instrument_fwhm_nm=args.instrument_fwhm_nm,
    )

    conc = np.array([1.0], dtype=float)
    spectrum_low_t = np.array(model.forward_numpy(args.low_T_eV, args.log_ne_ref, conc))
    spectrum_high_t = np.array(model.forward_numpy(args.high_T_eV, args.log_ne_ref, conc))
    wavelength = np.array(model.wavelength)

    checks: Dict[str, Any] = {}

    # Check 1: non-negative intensity
    checks["non_negative_low_T"] = bool(np.all(spectrum_low_t >= 0))
    checks["non_negative_high_T"] = bool(np.all(spectrum_high_t >= 0))

    # Check 2: line-ratio vs temperature
    pair = _pick_line_pair(model)
    if pair is None:
        checks["temperature_line_ratio"] = {
            "ok": False,
            "reason": "no_suitable_line_pair",
        }
    else:
        low_i, high_i = pair
        data = model.atomic_data
        wl_low = float(np.array(data.wavelength_nm)[low_i])
        wl_high = float(np.array(data.wavelength_nm)[high_i])
        ek_low = float(np.array(data.ek_ev)[low_i])
        ek_high = float(np.array(data.ek_ev)[high_i])

        i_low_lowT = _line_peak_intensity(wavelength, spectrum_low_t, wl_low)
        i_high_lowT = _line_peak_intensity(wavelength, spectrum_low_t, wl_high)
        i_low_highT = _line_peak_intensity(wavelength, spectrum_high_t, wl_low)
        i_high_highT = _line_peak_intensity(wavelength, spectrum_high_t, wl_high)

        ratio_lowT = float(i_high_lowT / max(i_low_lowT, 1e-30))
        ratio_highT = float(i_high_highT / max(i_low_highT, 1e-30))
        ok = bool(ratio_highT > ratio_lowT)

        checks["temperature_line_ratio"] = {
            "ok": ok,
            "line_low_nm": wl_low,
            "line_high_nm": wl_high,
            "E_low_eV": ek_low,
            "E_high_eV": ek_high,
            "ratio_lowT": ratio_lowT,
            "ratio_highT": ratio_highT,
        }

    # Check 3: Stark width increases with n_e
    stark_pick = _pick_stark_line(model)
    if stark_pick is None:
        checks["stark_width_vs_density"] = {
            "ok": False,
            "reason": "no_line_available",
        }
    else:
        stark_idx, has_ref_stark = stark_pick
        line_wl = float(np.array(model.atomic_data.wavelength_nm)[stark_idx])

        focus_model = BayesianForwardModel(
            db_path=args.db_path,
            elements=[args.element],
            wavelength_range=(line_wl - 0.5, line_wl + 0.5),
            pixels=2401,
            instrument_fwhm_nm=min(args.instrument_fwhm_nm, 0.02),
        )
        wl_focus = np.array(focus_model.wavelength)

        sp_low_ne = np.array(focus_model.forward_numpy(args.high_T_eV, args.log_ne_low, conc))
        sp_high_ne = np.array(focus_model.forward_numpy(args.high_T_eV, args.log_ne_high, conc))

        fwhm_low = _estimate_fwhm_nm(wl_focus, sp_low_ne, line_wl)
        fwhm_high = _estimate_fwhm_nm(wl_focus, sp_high_ne, line_wl)

        if fwhm_low is None or fwhm_high is None:
            checks["stark_width_vs_density"] = {
                "ok": False,
                "reason": "fwhm_estimation_failed",
                "line_nm": line_wl,
                "fwhm_low_nm": fwhm_low,
                "fwhm_high_nm": fwhm_high,
            }
        else:
            checks["stark_width_vs_density"] = {
                "ok": bool(fwhm_high > fwhm_low),
                "line_nm": line_wl,
                "used_reference_stark": bool(has_ref_stark),
                "fwhm_low_nm": float(fwhm_low),
                "fwhm_high_nm": float(fwhm_high),
            }

    overall_ok = bool(
        checks["non_negative_low_T"]
        and checks["non_negative_high_T"]
        and checks["temperature_line_ratio"].get("ok", False)
        and checks["stark_width_vs_density"].get("ok", False)
    )

    return {
        "element": args.element,
        "db_path": str(args.db_path),
        "wavelength_range_nm": [args.lambda_min, args.lambda_max],
        "checks": checks,
        "overall_ok": overall_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit synthetic LIBS forward-model physics sanity."
    )
    parser.add_argument("--db-path", type=Path, default=Path("ASD_da/libs_production.db"))
    parser.add_argument("--element", type=str, default="Fe")
    parser.add_argument("--lambda-min", type=float, default=220.0)
    parser.add_argument("--lambda-max", type=float, default=270.0)
    parser.add_argument("--pixels", type=int, default=4000)
    parser.add_argument("--instrument-fwhm-nm", type=float, default=0.03)
    parser.add_argument("--low-T-eV", type=float, default=0.8)
    parser.add_argument("--high-T-eV", type=float, default=1.6)
    parser.add_argument("--log-ne-ref", type=float, default=17.0)
    parser.add_argument("--log-ne-low", type=float, default=16.0)
    parser.add_argument("--log-ne-high", type=float, default=18.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/validation/synthetic_physics_audit.json"),
        help="Path for JSON report",
    )

    args = parser.parse_args()

    report = run_audit(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))

    status = "PASS" if report["overall_ok"] else "FAIL"
    print(f"Synthetic physics audit [{status}] -> {args.output}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
