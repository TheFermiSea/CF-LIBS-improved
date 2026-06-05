#!/usr/bin/env python3
"""Falsifiable element-presence measurement on real BHVO-2 (bead 29p1).

Runs the PRODUCTION ``analyze`` path (``_detect_and_select_lines`` ->
``IterativeCFLIBSSolver.solve``) on a real BHVO-2 spectrum and scores the
result against the USGS certified composition. Reports the five-criteria
gate from the 29p1 redesign:

  (a) Al recovered          -- cert Al mass-fraction 0.0714, currently ~0
  (c) no Ag/Sn/W/Bi FPs     -- confounders that must stay absent
  (d) RMSE down, attributed -- not a Mn-deflation artifact
  (e) Na not blown up; Mn   -- Na fix preserved, Mn over-attribution

This is a measurement tool, not a fix. Run on dev for the baseline, then
after each change for the delta. Parametrized so the three interacting
gates (rel-int floor, resonance exclusion, comb strength) can be ablated.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from cflibs.atomic.database import AtomicDatabase
from cflibs.benchmark.reference_compositions import BHVO2_BASALT_USGS
from cflibs.cli.main import _detect_and_select_lines
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
from cflibs.io.spectrum import load_spectrum

# Known FP confounders: neutral-resonance lines in the BHVO-2 band, same
# thermal E_k band as the real majors (29p1 Root cause B).
CONFOUNDERS = ["Ag", "Sn", "W", "Bi", "Th"]
EPS_PRESENT = 5e-3  # mass-fraction above which an element counts as "called present"


def _resolve_db() -> Path:
    for cand in (
        Path("ASD_da/libs_production.db"),
        Path("/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"),
    ):
        if cand.exists():
            return cand
    raise FileNotFoundError("libs_production.db not found")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectrum", default="data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv")
    ap.add_argument("--min-relative-intensity", type=lambda s: None if s == "none" else float(s),
                    default=None)
    ap.add_argument("--exclude-resonance", choices=["auto", "true", "false"], default="auto")
    ap.add_argument("--apply-self-absorption", action="store_true")
    ap.add_argument("--label", default="baseline")
    args = ap.parse_args()

    spec_path = Path(args.spectrum)
    if not spec_path.exists():
        spec_path = Path("/home/brian/code/CF-LIBS-improved") / args.spectrum

    cert = dict(BHVO2_BASALT_USGS)
    cert_elems = list(cert.keys())
    requested = cert_elems + [e for e in CONFOUNDERS if e not in cert_elems]

    wl, inten = load_spectrum(str(spec_path))
    db = AtomicDatabase(_resolve_db())

    excl = {"auto": None, "true": True, "false": False}[args.exclude_resonance]
    sa = bool(args.apply_self_absorption)

    obs = _detect_and_select_lines(
        wl, inten, db, requested,
        min_relative_intensity=args.min_relative_intensity,
        apply_self_absorption=sa,
        exclude_resonance=excl,
    )

    # Per-element observation counts (which elements survived detection+selection)
    obs_counts: dict[str, int] = {}
    for o in obs:
        el = getattr(o, "element", None)
        if el is not None:
            obs_counts[el] = obs_counts.get(el, 0) + 1

    solver = IterativeCFLIBSSolver(atomic_db=db, apply_self_absorption=sa)
    result = solver.solve(obs)
    pred = dict(result.concentrations)

    # --- scoring ---
    rmse = math.sqrt(sum((pred.get(e, 0.0) - cert[e]) ** 2 for e in cert_elems) / len(cert_elems))
    dropped = [e for e in cert_elems if pred.get(e, 0.0) < EPS_PRESENT]
    fps = {e: pred.get(e, 0.0) for e in CONFOUNDERS if pred.get(e, 0.0) >= EPS_PRESENT}

    print(f"\n{'='*72}\n  BHVO-2 PRESENCE MEASUREMENT  [{args.label}]")
    print(f"  spectrum={spec_path.name}  min_rel_int={args.min_relative_intensity}  "
          f"exclude_resonance={args.exclude_resonance}  SA={sa}")
    print(f"  T={result.temperature_K:.0f}K  ne={result.electron_density_cm3:.2e}  "
          f"converged={result.converged}  n_obs={len(obs)}")
    print('='*72)
    print(f"  {'el':<4}{'cert(wt%)':>11}{'pred(wt%)':>11}{'err':>9}{'n_obs':>7}")
    for e in cert_elems:
        c, p = cert[e] * 100, pred.get(e, 0.0) * 100
        print(f"  {e:<4}{c:>11.3f}{p:>11.3f}{p-c:>9.3f}{obs_counts.get(e,0):>7}")
    print(f"\n  Al observation lines surviving detection: {obs_counts.get('Al',0)}")
    print(f"  RMSE (cert-10)        : {rmse*100:.3f} wt%")
    print(f"  Dropped majors (<{EPS_PRESENT}): {dropped}")
    print(f"  FP confounders present : { {k: round(v*100,3) for k,v in fps.items()} or 'NONE'}")
    # confounder obs counts even if below presence eps
    conf_obs = {e: obs_counts.get(e, 0) for e in CONFOUNDERS if obs_counts.get(e, 0)}
    print(f"  Confounder obs lines   : {conf_obs or 'NONE'}")

    summary = {
        "label": args.label, "spectrum": spec_path.name, "rmse_wt%": rmse * 100,
        "T_K": result.temperature_K, "ne": result.electron_density_cm3,
        "pred": {e: pred.get(e, 0.0) for e in requested},
        "obs_counts": obs_counts, "dropped": dropped,
        "fps": fps, "al_obs": obs_counts.get("Al", 0),
    }
    # Repo-local, user-owned output dir (not a world-writable shared /tmp path).
    out_dir = Path(__file__).resolve().parent.parent / "output" / "bhvo2_measure"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.label}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
