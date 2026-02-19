#!/usr/bin/env python
"""
Generate NIST-referenced synthetic spectra for comparison to real LIBS data.

This script queries the NIST LIBS interface, extracts the emitted line list
from the output page JavaScript payload, broadens the stick spectrum onto the
experimental wavelength grid, and writes comparison artifacts.
"""

import argparse
import ast
import csv
import json
import logging
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.validate_real_data import (  # noqa: E402
    DATASETS,
    estimate_resolving_power,
    load_hdf5,
    load_hdf5_multishot,
    load_netcdf,
    load_scipp,
    load_scipp_depth_scan,
    select_representative_spectrum,
)
from cflibs.inversion.preprocessing import detect_peaks_auto  # noqa: E402

logger = logging.getLogger(__name__)


def _loader_map():
    return {
        "netcdf": load_netcdf,
        "hdf5": load_hdf5,
        "hdf5_multishot": load_hdf5_multishot,
        "scipp": load_scipp,
        "scipp_depth_scan": load_scipp_depth_scan,
    }


def _portable_path(path: Path) -> str:
    """Return repo-relative path when possible, otherwise absolute string."""
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y - np.nanmin(y)
    denom = np.nanmax(y)
    return y / denom if denom > 0 else y


def _synth_from_lines(
    wavelength: np.ndarray, line_wl: np.ndarray, line_strength: np.ndarray, resolution: float
) -> np.ndarray:
    if len(line_wl) == 0:
        return np.zeros_like(wavelength, dtype=float)
    fwhm = np.maximum(line_wl / max(float(resolution), 1e-6), 1e-6)
    sigma = fwhm / 2.355
    x = (wavelength[:, None] - line_wl[None, :]) / sigma[None, :]
    return np.sum(line_strength[None, :] * np.exp(-0.5 * x * x), axis=1)


def _fetch_nist_lines(
    element: str,
    low_w: float,
    upp_w: float,
    resolution: float,
    temp_eV: float,
    eden_cm3: float,
) -> Tuple[str, np.ndarray, np.ndarray]:
    params = {
        "libs": "1",
        "composition": f"{element}:100",
        "spectra": f"{element}0-2",
        "low_w": f"{low_w:.6f}",
        "upp_w": f"{upp_w:.6f}",
        "show_av": "2",
        "unit": "1",
        "resolution": f"{resolution:.6f}",
        "temp": f"{temp_eV:.6f}",
        "eden": f"{eden_cm3:.6e}",
        "maxcharge": "2",
        "min_rel_int": "0.01",
        "int_scale": "1",
    }
    url = "https://physics.nist.gov/cgi-bin/ASD/lines1.pl?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "CF-LIBS-nist-audit/1.0 (contact: maintainers@thefermsea.org)",
        },
    )
    with urllib.request.urlopen(req, timeout=45) as resp:
        status = getattr(resp, "status", None)
        html = resp.read().decode("utf-8", errors="ignore")

    match = re.search(r"var lines = (\[\[.*?\]\]);", html, re.S)
    if not match:
        logger.error(
            "Failed to parse NIST lines for %s (status=%s, html_len=%d)",
            element,
            status,
            len(html),
        )
        logger.debug("NIST response prefix for %s: %s", element, html[:500])
        raise RuntimeError(f"Could not parse NIST lines array for {element}")

    lines = ast.literal_eval(match.group(1))
    line_wl = np.array([row[0] for row in lines], dtype=float)
    line_strength = np.array([row[1] for row in lines], dtype=float)
    return url, line_wl, line_strength


def _best_shift_peak_match(
    peak_wl: np.ndarray, line_wl: np.ndarray, shift_range_nm: float = 1.5, tol_nm: float = 0.08
) -> Dict[str, float]:
    shifts = np.linspace(-shift_range_nm, shift_range_nm, 301)
    best = None
    for shift_nm in shifts:
        shifted = peak_wl + shift_nm
        diffs = np.abs(shifted[:, None] - line_wl[None, :])
        nearest = np.min(diffs, axis=1)
        n_match = int(np.sum(nearest <= tol_nm))
        score = n_match - float(np.median(nearest))
        row = {
            "shift_nm": float(shift_nm),
            "matched_peaks": n_match,
            "total_peaks": int(len(peak_wl)),
            "match_rate": float(n_match / max(len(peak_wl), 1)),
            "median_residual_nm": float(np.median(nearest)),
            "score": score,
        }
        if best is None or row["score"] > best["score"]:
            best = row
    return best or {}


def _resolve_cases(dataset_names: Iterable[str]) -> List[Dict]:
    by_name = {d["name"]: d for d in DATASETS}
    out = []
    for name in dataset_names:
        if name not in by_name:
            raise ValueError(f"Unknown dataset: {name}")
        ds = by_name[name]
        expected = ds.get("expected") or []
        if len(expected) != 1:
            raise ValueError(
                f"Dataset {name} is not pure-element (expected={expected}). "
                "Use pure datasets with exactly one expected element."
            )
        out.append(ds)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate NIST-referenced synthetic spectra.")
    parser.add_argument("--data-dir", default="data", type=str)
    parser.add_argument("--output-dir", default="output/nist_ground_truth", type=str)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Fe_245nm", "Ni_245nm"],
        help="Pure dataset names from scripts/validate_real_data.py",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    data_dir = (ROOT / args.data_dir).resolve()
    out_dir = (ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    loaders = _loader_map()
    cases = _resolve_cases(args.datasets)

    all_summary = []
    for ds in cases:
        name = ds["name"]
        element = ds["expected"][0]
        path = data_dir / ds["path"]
        if not path.exists():
            raise FileNotFoundError(f"Missing data file for {name}: {path}")

        loader = loaders[ds["loader"]]
        wavelength, data, _meta = loader(str(path))
        spectrum = select_representative_spectrum(data, name)
        resolving_power = ds.get("resolving_power", estimate_resolving_power(wavelength, spectrum))

        temps = [0.8, 1.0, 1.2, 1.5, 2.0]
        edens = [1e16, 3e16, 1e17, 3e17]
        rows: List[Dict] = []
        best = None

        for temp_eV in temps:
            for eden_cm3 in edens:
                nist_url, line_wl, line_strength = _fetch_nist_lines(
                    element=element,
                    low_w=float(np.min(wavelength)),
                    upp_w=float(np.max(wavelength)),
                    resolution=float(resolving_power),
                    temp_eV=temp_eV,
                    eden_cm3=eden_cm3,
                )
                synth = _synth_from_lines(wavelength, line_wl, line_strength, resolving_power)
                norm_spec = _normalize(spectrum)
                norm_synth = _normalize(synth)
                if np.allclose(norm_spec, 0.0) or np.allclose(norm_synth, 0.0):
                    corr = 0.0
                else:
                    corr = float(np.corrcoef(norm_spec, norm_synth)[0, 1])
                corr = float(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0))
                mae = float(np.mean(np.abs(norm_spec - norm_synth)))
                row = {
                    "dataset": name,
                    "element": element,
                    "temp_eV": temp_eV,
                    "eden_cm3": eden_cm3,
                    "resolution": float(resolving_power),
                    "corr": corr,
                    "mae": mae,
                    "n_lines": int(len(line_wl)),
                    "nist_url": nist_url,
                }
                rows.append(row)
                if best is None or row["corr"] > best["corr"]:
                    best = {
                        **row,
                        "line_wl": line_wl,
                        "line_strength": line_strength,
                        "synth": synth,
                    }
                time.sleep(0.25)

        # Persist sweep
        grid_csv = out_dir / f"{name}_nist_grid.csv"
        with grid_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "dataset",
                    "element",
                    "temp_eV",
                    "eden_cm3",
                    "resolution",
                    "corr",
                    "mae",
                    "n_lines",
                    "nist_url",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        line_csv = out_dir / f"{name}_nist_lines.csv"
        with line_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength_nm", "strength"])
            for lw, ls in zip(best["line_wl"], best["line_strength"]):
                writer.writerow([f"{lw:.8f}", f"{ls:.8e}"])

        comp_csv = out_dir / f"{name}_nist_best_spectrum.csv"
        with comp_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["wavelength_nm", "measured_norm", "nist_synth_norm"])
            for x, ym, ys in zip(
                wavelength, _normalize(spectrum), _normalize(best["synth"]), strict=False
            ):
                writer.writerow([f"{x:.8f}", f"{ym:.8e}", f"{ys:.8e}"])

        # Plot overlay
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(wavelength, _normalize(spectrum), lw=1.0, label="Measured", alpha=0.85)
        ax.plot(wavelength, _normalize(best["synth"]), lw=1.0, label="NIST synthetic", alpha=0.85)
        ax.grid(alpha=0.2)
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Normalized intensity")
        ax.set_title(
            f"{name} ({element}): corr={best['corr']:.3f}, "
            f"T={best['temp_eV']:.2f} eV, ne={best['eden_cm3']:.2e}, R={best['resolution']:.0f}"
        )
        ax.legend(loc="upper right")
        fig.tight_layout()
        overlay_png = out_dir / f"{name}_nist_overlay.png"
        fig.savefig(overlay_png, dpi=160)
        plt.close(fig)

        # Peak-line shift diagnostics
        peaks, _, _ = detect_peaks_auto(wavelength, spectrum, threshold_factor=4.0)
        peak_wl = np.array([p[1] for p in peaks], dtype=float)
        shift_diag = _best_shift_peak_match(peak_wl=peak_wl, line_wl=best["line_wl"])

        summary = {
            "dataset": name,
            "element": element,
            "best_grid_point": {
                "temp_eV": best["temp_eV"],
                "eden_cm3": best["eden_cm3"],
                "resolution": best["resolution"],
                "corr": best["corr"],
                "mae": best["mae"],
                "n_lines": best["n_lines"],
                "nist_url": best["nist_url"],
            },
            "peak_to_line_shift_diagnostics": shift_diag,
            "outputs": {
                "grid_csv": _portable_path(grid_csv),
                "line_csv": _portable_path(line_csv),
                "comparison_csv": _portable_path(comp_csv),
                "overlay_png": _portable_path(overlay_png),
            },
        }
        (out_dir / f"{name}_summary.json").write_text(json.dumps(summary, indent=2))
        all_summary.append(summary)

    (out_dir / "summary_all.json").write_text(json.dumps(all_summary, indent=2))
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main()
