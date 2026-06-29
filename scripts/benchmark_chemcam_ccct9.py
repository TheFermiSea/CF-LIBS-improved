#!/usr/bin/env python3
"""Second-instrument Ti-6Al-4V (DED matrix) cross-validation on real ChemCam data.

Goal: confirm the shipped constrained {Ti,Al,V} DED mode (constrained
force-extraction + matrix isolation + OPC) -- validated on real Mars SuperCam
Ti-6Al-4V (SCCT_TITANIUM) at OPC held-out 0.648 wt% (``benchmark_pds_opc.py``
Part 3) -- on a DIFFERENT instrument: MSL/Curiosity ChemCam. The ChemCam
calibration-target assembly carries a Ti-6Al-4V titanium plate used for LIBS
wavelength calibration (Wiens et al. 2012), so a clean cross-instrument
confirmation would reuse the exact shipped path with ChemCam's instrument profile.

This script does the PDS RDR retrieval, then -- if a Ti-6Al-4V ChemCam spectrum
is available -- runs the shipped constrained DED OPC (the solve path is imported
verbatim from ``benchmark_pds_opc``; only the instrument constants are
ChemCam-specific).

PDS retrieval reality (reproduced by ``--discover --scan``)
-----------------------------------------------------------
The ChemCam RDR archive (MSL-M-CHEMCAM-LIBS-4/5-RDR-V1.0) has no target-name
column in its master index, and every CCS *label* reports TARGET_NAME=MARS. The
authoritative target identity lives in a comment header inside each CCS *data*
CSV: ``# TARGET = Cal Target N`` plus ``# DISTT = <mm>`` (the 1.56 m on-deck
standoff). This script range-fetches the 2 KB header of one CSV per
(sol, CCAM-target) group across the whole mission (sols 1-4612, 4248 groups) to
enumerate every named calibration target, then downloads one full spectrum per
distinct target name and classifies titanium (Ti II/Ti I forest) vs the
silicate/glass geological standards (Ca II 393/396, Mg II 279/280, Fe II 259/263,
Si I 288). O-777 is NOT used as a discriminator: it is atmospheric (CO2 plasma)
in every Mars LIBS shot.

Empirical finding (full-archive scan): the only named calibration targets in the
CCS/RDR LIBS products are ``Cal Target 1,2,3,4,6,7,8,9`` -- ALL silicate/glass
geological standards at ~1585 mm. There is NO ``Cal Target 5/10/11``, no
``Titanium``, no ``Graphite`` LIBS spectrum anywhere in the archive. (CHEMCAM-PSV
is passive/no-laser; CHEMCAM-MOC/TEC are derived composition tables.) ChemCam
does fire LIBS at the Ti target for wavelength calibration, but those engineering
spectra are consumed internally to produce the archived wavelength-cal
coefficients -- they are NOT distributed as per-sol data products. NOTE: this
also means the corpus mapping ``_CCCT_COMPOSITIONS["CCCT9"] = Ti-6Al-4V`` does
NOT correspond to ChemCam's archived ``Cal Target 9`` (which is a silicate).

=> A true Ti-6Al-4V ChemCam DED cross-validation is not currently possible from
the public RDR archive. This script reproduces that determination and is ready
to run the shipped constrained DED OPC the instant any Ti-6Al-4V ChemCam
spectrum is dropped into ``data/chemcam_ccct9/titanium/*.csv``.

Usage::

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/benchmark_chemcam_ccct9.py --discover
    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/benchmark_chemcam_ccct9.py --discover --scan
    PYTHONPATH=$PWD JAX_PLATFORMS=cpu python scripts/benchmark_chemcam_ccct9.py --solve
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = _REPO_ROOT / "scripts"
for _p in (str(_REPO_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

warnings.filterwarnings("ignore")

# Reuse the SHIPPED DED solve path verbatim from the SuperCam Ti-6Al-4V benchmark.
import benchmark_pds_opc as bp  # noqa: E402

BASE = "https://pds-geosciences.wustl.edu/msl/msl-m-chemcam-libs-4_5-rdr-v1/mslccm_1xxx/"
INDEX_URL = f"{BASE}index/libsindex.tab"
DATA = _REPO_ROOT / "data" / "chemcam_ccct9"
INDEX = DATA / "libsindex.tab"
HDR_CACHE = DATA / "hdr_meta.json"
SPECDIR = DATA / "rep_spectra"
TITANIUM_DIR = DATA / "titanium"  # drop confirmed Ti-6Al-4V ChemCam CSVs here

# Ti-6Al-4V truth, constrained {Ti,Al,V} element set (grade-5 nominal).
_TRUTH_TIALV = {"Ti": 89.5, "Al": 6.1, "V": 4.0}

# ChemCam instrument profile (Wiens et al. 2012): band FWHM ~0.15/0.20/0.65 nm in
# UV/VIO/VNIR; the {Ti,Al,V} 250-500 nm lines live in UV/VIO -> ~0.18 nm.
_CHEMCAM_FWHM_NM = 0.18
# ChemCam inter-spectrometer detector gaps (nm): UV|VIO and VIO|VNIR.
_CHEMCAM_DETECTOR_GAPS = ((341.0, 382.0), (469.0, 473.0))

_GROUP_RE = re.compile(r"CCS_(.+)$")
_CCAM_RE = re.compile(r"CCAM(\d{2})(\d+)P(\d)")

# Clean discriminator lines (nm). Ti-6Al-4V = Ti II/Ti I forest; silicate CCCTs =
# Si/Mg/Ca lines. O-777 excluded (atmospheric CO2 plasma in every Mars shot).
_TI2 = [323.45, 334.94, 336.12, 337.28, 338.38, 368.52, 376.13, 350.49]
_ROCK = [288.16, 285.21, 422.67, 589.0, 393.37, 280.27]  # Si Mg CaI Na CaII MgII


# --------------------------------------------------------------------------- #
# Archive discovery: index -> (sol,CCAM) groups -> CSV-header TARGET/DISTT      #
# --------------------------------------------------------------------------- #


def _ensure_index() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    if INDEX.exists() and INDEX.stat().st_size > 1_000_000:
        return
    print(f"  downloading master index (~31 MB) {INDEX_URL}", flush=True)
    urllib.request.urlretrieve(INDEX_URL, INDEX)  # noqa: S310


def _build_groups(sol_min: int, sol_max: int) -> Dict[str, dict]:
    """One entry per (sol, CCAM-target) raster (raster points share a target)."""
    groups: Dict[str, dict] = {}
    with open(INDEX, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 13 or row[5].strip() != "CHEMCAM-CCS":
                continue
            try:
                sol = int(row[12])
            except ValueError:
                continue
            if not (sol_min <= sol <= sol_max):
                continue
            pid = row[3].strip()
            m = _GROUP_RE.search(pid)
            if not m:
                continue
            gkey = f"{sol}:{m.group(1)}"
            cm = _CCAM_RE.search(pid)
            g = groups.setdefault(
                gkey,
                {
                    "sol": sol,
                    "ccam": int(cm.group(1)) if cm else -1,
                    "n_points": 0,
                    "rep": pid,
                    "path": row[1].strip().lower().rstrip("/"),
                },
            )
            g["n_points"] += 1
    return groups


def _fetch_header(args):
    """Range-fetch the 2 KB CSV header -> authoritative TARGET + DISTT."""
    gkey, path, pid = args
    url = f"{BASE}{path}/{pid.lower()}.csv"
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "cflibs-ccct9/1", "Range": "bytes=0-2047"}
        )
        with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310
            text = r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return gkey, {"error": str(e)}
    out: dict = {}
    m = re.search(r"#\s*TARGET\s*=\s*([^,\n]+)", text)
    if m:
        out["target"] = m.group(1).strip()
    m = re.search(r"#\s*DISTT\s*=\s*([0-9.]+)", text)
    if m:
        out["distt"] = float(m.group(1))
    return gkey, out


def _is_caltarget(name: str) -> bool:
    n = (name or "").lower()
    return "cal target" in n or "ccct" in n or "titan" in n or "graphit" in n


def discover(sol_min: int, sol_max: int, workers: int, do_scan: bool) -> dict:
    _ensure_index()
    groups = _build_groups(sol_min, sol_max)
    cache = json.loads(HDR_CACHE.read_text()) if HDR_CACHE.exists() else {}
    if do_scan:
        todo = [(k, g["path"], g["rep"]) for k, g in groups.items() if k not in cache]
        print(
            f"  {len(groups)} target-groups in sols {sol_min}-{sol_max}; "
            f"{len(todo)} new CSV headers to range-fetch",
            flush=True,
        )
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for gkey, meta in ex.map(_fetch_header, todo):
                g = groups[gkey]
                cache[gkey] = {
                    "sol": g["sol"],
                    "ccam": g["ccam"],
                    "n_points": g["n_points"],
                    "path": g["path"],
                    "rep": g["rep"],
                    **meta,
                }
                done += 1
                if done % 300 == 0:
                    print(f"    {done}/{len(todo)}", flush=True)
                    HDR_CACHE.write_text(json.dumps(cache))
        HDR_CACHE.write_text(json.dumps(cache))
    if not cache:
        print("  no header cache; rerun with --scan to fetch CSV headers")
        return {"scanned": 0}

    cal = [v for v in cache.values() if _is_caltarget(v.get("target", ""))]
    by_name: Dict[str, List[dict]] = defaultdict(list)
    for v in cal:
        by_name[v["target"]].append(v)
    print(
        f"  {len(cache)} groups scanned; {len(cal)} calibration-target observations "
        f"across {len(by_name)} named targets"
    )

    # Classify one full spectrum per distinct named target (titanium vs silicate).
    SPECDIR.mkdir(parents=True, exist_ok=True)
    print(f"\n  {'target':>22} {'n_obs':>6} {'DISTT':>8} {'TiII/rock':>10}  class")
    titanium_targets, results = [], {}
    for name in sorted(by_name):
        recs = by_name[name]
        ds = [r["distt"] for r in recs if isinstance(r.get("distt"), float)]
        dmean = float(np.mean(ds)) if ds else float("nan")
        ratio, peaks = _spectral_signature(recs[0])
        cls = "TITANIUM" if ratio > 1.5 else "silicate"
        if cls == "TITANIUM":
            titanium_targets.append(name)
        results[name] = {
            "n_obs": len(recs),
            "distt_mm": dmean,
            "ti_rock": ratio,
            "class": cls,
            "peaks": peaks,
        }
        print(f"  {name:>22} {len(recs):>6} {dmean:>8.1f} {ratio:>10.3f}  {cls}")

    print(f"\n  TITANIUM (Ti-6Al-4V) calibration targets in archive: {len(titanium_targets)}")
    if not titanium_targets:
        ex_name = max(by_name, key=lambda n: len(by_name[n])) if by_name else None
        if ex_name:
            print(f"  e.g. {ex_name!r} dominant peaks: {results[ex_name]['peaks']}")
            print("       (Ca II 393/396, Mg II 279/280, Fe II 259/263, Si I 288: silicate)")
    return {
        "scanned": len(cache),
        "n_cal_obs": len(cal),
        "named_targets": results,
        "titanium_targets": titanium_targets,
    }


def _download(url, dest) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "cflibs-ccct9/1"})
        with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310
            dest.write_bytes(r.read())
        return True
    except Exception as e:  # noqa: BLE001
        print(f"    ERR {url}: {e}")
        return False


def _loc(wl, it, c, half=0.35, ch=4.0):
    m = (wl >= c - half) & (wl <= c + half)
    if not m.any():
        return 0.0
    cm = (wl >= c - ch) & (wl <= c + ch) & ~m
    cont = np.median(it[cm]) if cm.any() else 0.0
    return max(float(np.max(it[m])) - cont, 0.0) / (cont + 1e-9)


def _spectral_signature(rec: dict):
    """Download one full spectrum -> (Ti II / rock ratio, dominant peaks)."""
    from cflibs.pds.chemcam import ChemCamParser

    dest = SPECDIR / f"{rec['rep'].lower()}.csv"
    if not _download(f"{BASE}{rec['path']}/{rec['rep'].lower()}.csv", dest):
        return 0.0, []
    try:
        sp = ChemCamParser().parse(dest)
    except Exception:  # noqa: BLE001
        return 0.0, []
    wl, it = sp.wavelength, np.clip(sp.intensity, 0.0, None)
    ti = sum(_loc(wl, it, c) for c in _TI2)
    rk = sum(_loc(wl, it, c) for c in _ROCK)
    # dominant peaks 250-600 nm
    m = (wl >= 250) & (wl <= 600)
    x, y = wl[m], it[m]
    idx = [
        i
        for i in range(2, len(y) - 2)
        if y[i] > y[i - 1] and y[i] >= y[i + 1] and y[i] > y[i - 2] and y[i] > y[i + 2]
    ]
    idx.sort(key=lambda i: y[i], reverse=True)
    peaks: List[float] = []
    for i in idx:
        if all(abs(x[i] - p) > 0.5 for p in peaks):
            peaks.append(round(float(x[i]), 2))
        if len(peaks) >= 8:
            break
    return ti / (rk + 1e-9), peaks


# --------------------------------------------------------------------------- #
# Constrained DED OPC solve (shipped path) -- runs IF Ti-6Al-4V spectra exist  #
# --------------------------------------------------------------------------- #


def _chemcam_in_gap(wl_nm: float) -> bool:
    return any(lo <= wl_nm <= hi for lo, hi in _CHEMCAM_DETECTOR_GAPS)


def _chemcam_titanium_line_list(db):
    """{Ti,Al,V} known-position list with ChemCam detector gaps (vs SuperCam)."""
    from tests.benchmarks.ded_precision.line_lists import select_lines

    specs = []
    for el in bp._TI6AL4V_ELEMENTS:
        n = bp._TI6AL4V_FORCED_BUDGET.get(el, 8)
        cand = select_lines(
            db, el, (250.0, 500.0), n * 4, T_K=10000.0, exclude_resonance=False, prefer_spread=False
        )
        cand = [s for s in cand if not _chemcam_in_gap(s.wavelength_nm)]
        specs.extend(cand[:n])
    return specs


def solve(db) -> dict:
    """Shipped constrained {Ti,Al,V} DED OPC on any Ti-6Al-4V ChemCam CSVs."""
    from cflibs.pds.chemcam import ChemCamParser

    TITANIUM_DIR.mkdir(parents=True, exist_ok=True)
    paths = sorted(TITANIUM_DIR.glob("*.csv"))
    print(f"  {len(paths)} Ti-6Al-4V ChemCam CSV(s) in {TITANIUM_DIR}")
    if not paths:
        print("  NO titanium spectra available -> cross-instrument DED solve cannot run.")
        print("  (ChemCam Ti-target spectra are not archived as data products; drop")
        print("   confirmed Ti-6Al-4V ChemCam CSVs here to enable this benchmark.)")
        return {"n_observations": 0, "blocked": True}

    specs = _chemcam_titanium_line_list(db)
    scnt = {e: sum(1 for s in specs if s.element == e) for e in bp._TI6AL4V_ELEMENTS}
    print(f"  ChemCam forced line list: {len(specs)} known lines {scnt}")
    samples = []
    for p in paths:
        sp = ChemCamParser().parse(p)
        wl = sp.wavelength
        inten = np.clip(np.asarray(sp.intensity, dtype=float), 0.0, None)
        obs, stark = bp.forced_observations(
            db, wl, inten, specs, instrument_fwhm_nm=_CHEMCAM_FWHM_NM
        )
        cnt = {e: sum(1 for o in obs if o.element == e) for e in bp._TI6AL4V_ELEMENTS}
        samples.append((p.stem, obs, stark, dict(_TRUTH_TIALV)))
        print(f"    {p.stem}: {len(obs)} obs {cnt}")

    base_pairs, opc_pairs, prov = bp.opc_heldout(db, samples, closure_mode="standard")
    bsc, osc = bp.score(base_pairs), bp.score(opc_pairs)
    print("-" * 74)
    print(f"  baseline (calibration-free) : {bp._fmt_score(bsc)}")
    print(
        f"  OPC same-comp LOO           : {bp._fmt_score(osc)}  "
        f"(selected={prov['n_selected']}, robust_T={prov['robust_T_K']:.0f}K)"
    )
    return {
        "n_observations": len(samples),
        "baseline": bsc,
        "opc_loo": osc,
        "provenance": prov,
        "blocked": False,
    }


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--discover",
        action="store_true",
        help="Discover/classify ChemCam on-deck calibration observations.",
    )
    ap.add_argument(
        "--scan",
        action="store_true",
        help="With --discover: range-fetch CSV-header TARGET/DISTT (cached).",
    )
    ap.add_argument(
        "--solve",
        action="store_true",
        help="Run the shipped constrained DED OPC on any Ti-6Al-4V CSVs.",
    )
    ap.add_argument("--sol-min", type=int, default=1)
    ap.add_argument("--sol-max", type=int, default=4612)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if not (args.discover or args.solve):
        ap.error("pass --discover and/or --solve")

    print("=" * 74)
    print("  ChemCam Ti-6Al-4V second-instrument DED cross-validation")
    print("=" * 74)
    result: dict = {}
    if args.discover:
        result["discovery"] = discover(args.sol_min, args.sol_max, args.workers, args.scan)
    if args.solve:
        import cflibs
        from cflibs.atomic.database import AtomicDatabase

        db_path = bp._db_path()
        print(f"  cflibs={Path(cflibs.__file__).resolve().parent}")
        print(f"  db={db_path}")
        result["solve"] = solve(AtomicDatabase(db_path))

    disc = result.get("discovery", {})
    solv = result.get("solve", {})
    print("\n" + "=" * 74)
    if solv and not solv.get("blocked") and solv.get("n_observations"):
        osc = solv["opc_loo"]
        print(
            f"  VERDICT: ChemCam Ti-6Al-4V OPC held-out overall = "
            f"{osc['rmsep_overall']:.3f} wt% (vs SuperCam 0.648 wt%)."
        )
    else:
        nt = len(disc.get("titanium_targets", []))
        print("  VERDICT: cross-instrument Ti-6Al-4V DED confirmation NOT possible from")
        print("  the public ChemCam RDR archive. The titanium calibration target's LIBS")
        print("  spectra are not distributed as data products (consumed internally for")
        print("  wavelength calibration). The only named LIBS calibration targets in the")
        print(f"  archive are silicate geological standards (titanium found: {nt}).")
    print("=" * 74)

    out_path = Path(args.out) if args.out else (DATA / "ccct9_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=float))
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
