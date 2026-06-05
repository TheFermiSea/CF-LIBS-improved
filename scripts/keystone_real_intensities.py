#!/usr/bin/env python3
"""Reproduce the REAL keystone: feed 9 Mg + 9 Fe lines with REAL measured
intensities (sampled from the ChemCam BHVO-2 spectrum at each DB line's
wavelength) into IterativeCFLIBSSolver. The intensities are NOT from a
consistent plasma state, so this is the realistic collapse case.

Also runs the multi-element version (the diagnosis-doc keystone: Mg/Fe -> Na/Ca
dominate) by adding Na + Ca lines too.
"""
from __future__ import annotations

import sqlite3
import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
from cflibs.io.spectrum import load_spectrum

DB = "/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"
SPEC = "/home/brian/code/CF-LIBS-improved/data/bhvo2_usgs/chemcam_bhvo2_loc1_spectrum.csv"
OFFSET = 0.10  # ChemCam constant +0.10 nm offset (diagnosis doc)


def sample_intensity(wl_arr, in_arr, wl0, win=0.15):
    m = np.abs(wl_arr - wl0) <= win
    if not m.any():
        return None
    return float(np.max(in_arr[m]))


def get_lines(element: str, n: int, wl_lo=240, wl_hi=850) -> list[tuple]:
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute(
        "SELECT wavelength_nm, aki, ek_ev, gk FROM lines "
        "WHERE element=? AND sp_num=1 AND aki>1e6 AND ek_ev>0 "
        "AND wavelength_nm BETWEEN ? AND ? ORDER BY ek_ev",
        (element, wl_lo, wl_hi),
    )
    rows = [r for r in cur.fetchall() if r[1] and r[3]]
    con.close()
    if len(rows) <= n:
        return rows
    idx = np.linspace(0, len(rows) - 1, n).astype(int)
    return [rows[i] for i in idx]


def build_obs(db, wl_arr, in_arr, elements, n_each=9):
    observations = []
    for el in elements:
        kept = 0
        for (wl, aki, ek, gk) in get_lines(el, n_each * 3):
            I = sample_intensity(wl_arr, in_arr, wl + OFFSET)
            if I is None or I <= 0:
                continue
            observations.append(
                LineObservation(
                    wavelength_nm=wl, intensity=I, intensity_uncertainty=0.1 * I,
                    element=el, ionization_stage=1, E_k_ev=ek, g_k=int(gk), A_ki=aki,
                )
            )
            kept += 1
            if kept >= n_each:
                break
        print(f"# {el}: kept {kept} real-intensity lines")
    return observations


def run(db, observations, label):
    from collections import defaultdict
    by_el = defaultdict(list)
    for o in observations:
        by_el[o.element].append(o)
    print(f"\n=== {label} ===")
    for el, lst in by_el.items():
        ys = [o.y_value for o in lst]
        eks = [o.E_k_ev for o in lst]
        print(f"  {el}: n={len(lst)} Ek[{min(eks):.2f},{max(eks):.2f}] "
              f"y[{min(ys):.2f},{max(ys):.2f}] yspan={max(ys)-min(ys):.2f}")
    solver = IterativeCFLIBSSolver(atomic_db=db, max_iterations=30)
    res = solver.solve(observations, closure_mode="standard")
    print(f"  T={res.temperature_K:.0f}K ne={res.electron_density_cm3:.2e} "
          f"iters={res.iterations} conv={res.converged} R2={res.quality_metrics.get('r_squared_last'):.4f}")
    tot = sum(res.concentrations.values()) or 1
    print("  C: " + "  ".join(f"{k}={v/tot:.4f}" for k, v in
                              sorted(res.concentrations.items(), key=lambda kv: -kv[1])))
    return res


def main():
    db = AtomicDatabase(DB)
    wl_arr, in_arr = load_spectrum(SPEC)
    in_arr = np.asarray(in_arr, float)
    wl_arr = np.asarray(wl_arr, float)

    # 1) Mg+Fe only, real intensities
    obs = build_obs(db, wl_arr, in_arr, ["Mg", "Fe"])
    run(db, obs, "Mg+Fe REAL intensities")

    # 2) diagnosis keystone: Mg/Fe lines but add Na+Ca lines too
    obs2 = build_obs(db, wl_arr, in_arr, ["Mg", "Fe", "Na", "Ca"])
    run(db, obs2, "Mg+Fe+Na+Ca REAL intensities")


if __name__ == "__main__":
    main()
