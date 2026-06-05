#!/usr/bin/env python3
"""Keystone reproduction + step-by-step trace of solver closure degeneracy.

Feeds ~9 Mg I + 9 Fe I lines (REAL wavelengths/A_ki/E_k from the DB) with
intensities synthesized from a KNOWN plasma state (T, n_e) and KNOWN number
fractions, then feeds them to IterativeCFLIBSSolver and prints the recovered
composition. Ground truth is exactly known, so any collapse is a solver bug.

Synthetic intensity for a neutral line of element s:
    I = C_s * (n_tot / U_s(T)) * (g A / lambda) * exp(-E_k / kT)
with C_s the number fraction. Then y = ln(I*lambda/(gA)) = ln(C_s*n_tot/U_s) - E_k/kT,
so the Boltzmann intercept q_s = ln(C_s * n_tot / U_s) and standard closure
should recover C_s exactly (multiplier=1+S; for neutral-dominant majors S~0).
"""
from __future__ import annotations

import sqlite3
import numpy as np

from cflibs.atomic.database import AtomicDatabase
from cflibs.core.constants import KB_EV
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

DB = "/home/brian/code/CF-LIBS-improved/ASD_da/libs_production.db"


def get_lines(element: str, n: int) -> list[tuple]:
    con = sqlite3.connect(DB)
    cur = con.cursor()
    # neutral lines (sp_num=1), spread in E_k, real A_ki, strong-ish
    cur.execute(
        "SELECT wavelength_nm, aki, ek_ev, gk FROM lines "
        "WHERE element=? AND sp_num=1 AND aki>1e6 AND ek_ev>0 "
        "AND wavelength_nm BETWEEN 240 AND 850 ORDER BY ek_ev",
        (element,),
    )
    rows = [r for r in cur.fetchall() if r[1] and r[3]]
    con.close()
    # pick n lines spread across the E_k range to give a real Boltzmann lever-arm
    if len(rows) <= n:
        return rows
    idx = np.linspace(0, len(rows) - 1, n).astype(int)
    return [rows[i] for i in idx]


def main() -> None:
    db = AtomicDatabase(DB)
    T_true = 8000.0  # K
    n_tot = 1.0e17  # cm^-3 total nuclei (arbitrary scale)
    kT = KB_EV * T_true

    # KNOWN ground-truth number fractions
    C_true = {"Mg": 0.5, "Fe": 0.5}

    elements = ["Mg", "Fe"]
    U = {el: db.partition_function_for(el, 1).at(T_true) for el in elements}
    print(f"# T_true={T_true} K  kT={kT:.4f} eV  U_I: {U}")

    observations: list[LineObservation] = []
    for el in elements:
        lines = get_lines(el, 9)
        print(f"\n# {el}: {len(lines)} lines, E_k span "
              f"{min(r[2] for r in lines):.2f}-{max(r[2] for r in lines):.2f} eV")
        for (wl, aki, ek, gk) in lines:
            # synthetic optically-thin intensity from KNOWN state
            I = C_true[el] * (n_tot / U[el]) * (gk * aki / wl) * np.exp(-ek / kT)
            observations.append(
                LineObservation(
                    wavelength_nm=wl, intensity=I, intensity_uncertainty=0.01 * I,
                    element=el, ionization_stage=1, E_k_ev=ek, g_k=int(gk), A_ki=aki,
                )
            )
            print(f"    {wl:8.3f}nm  Ek={ek:6.3f}  gA={gk*aki:.3e}  "
                  f"y={np.log(I*wl/(gk*aki)):8.3f}  I={I:.3e}")

    # expected intercepts q_s = ln(C_s * n_tot / U_s)
    for el in elements:
        q_exp = np.log(C_true[el] * n_tot / U[el])
        print(f"# EXPECTED q_{el} = {q_exp:.4f}  (=> rel_C = U*exp(q) = {U[el]*np.exp(q_exp):.3e})")

    for mode in ("standard", "ilr"):
        solver = IterativeCFLIBSSolver(atomic_db=db, max_iterations=30)
        res = solver.solve(observations, closure_mode=mode)
        print(f"\n=== closure_mode={mode} ===")
        print(f"  T={res.temperature_K:.0f}K  ne={res.electron_density_cm3:.2e}  "
              f"iters={res.iterations} conv={res.converged} R2={res.quality_metrics.get('r_squared_last'):.4f}")
        print(f"  recovered C (number frac): "
              + "  ".join(f"{k}={v:.4f}" for k, v in res.concentrations.items()))
        print(f"  TRUTH:                     "
              + "  ".join(f"{k}={v:.4f}" for k, v in C_true.items()))


if __name__ == "__main__":
    main()
