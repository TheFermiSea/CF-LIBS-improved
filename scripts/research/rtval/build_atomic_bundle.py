"""Build a static atomic-line bundle for the ExoJAX-grade CF-LIBS reference
forward model from the cflibs production SQLite DB.

The bundle is the *pluggable* atomic data contract consumed by reference_forward.py:
a fixed-shape set of arrays describing N lines and N_elem elements. It is element-
agnostic and self-describing (the element order is stored), so the forward model
and inversion need no DB at runtime.

Bundle arrays (saved as a single .npz):
  Lines (length N_LINES):
    line_wl_nm     (N,) float64  line center, nm (vacuum/air as in DB)
    line_gA        (N,) float64  g_k * A_ki  [s^-1]   (emission weight numerator)
    line_Ek_eV     (N,) float64  upper-level energy E_k [eV]
    line_elem      (N,) int32    index into elements[]
    line_stage     (N,) int32    0 = neutral (sp_num 1), 1 = singly ionized (sp_num 2)
    line_stark_w   (N,) float64  Stark HWHM ref [nm] at n_e=1e17 (0 if unknown)
  Elements (length N_ELEM):
    elements       (N_elem,) str        element symbols, defines composition order
    elem_mass      (N_elem,) float64    atomic mass [amu]
    elem_ip_eV     (N_elem,) float64    first ionization potential [eV]  (sp_num 1)
    U_coeffs       (N_elem, 2, 5) float64  partition poly coeffs a0..a4 per stage;
                                           U_stage = exp( sum_n a_n (ln T)^n )

Selection: per (element, stage) keep the top LINES_PER_ELEM lines by g_k*A_ki in a
wavelength window, requiring E_k>0. Padded to a fixed LINES_PER_ELEM per stage so
shapes are static (zero-gA pad lines contribute nothing).
"""

import argparse
import sqlite3

import numpy as np

DEFAULT_ELEMENTS = ["Si", "Ca", "Fe", "Al", "Mg", "Na", "K", "Ti", "Mn"]


def _partition_coeffs(cur, element, sp_num):
    row = cur.execute(
        "SELECT a0,a1,a2,a3,a4 FROM partition_functions WHERE element=? AND sp_num=?",
        (element, sp_num),
    ).fetchone()
    if row is None:
        # ground-state-degeneracy fallback (ln-poly with only a0 = ln g0)
        return np.array([np.log(2.0), 0.0, 0.0, 0.0, 0.0])
    return np.array([c if c is not None else 0.0 for c in row], dtype=np.float64)


def _species_phys(cur, element, sp_num):
    row = cur.execute(
        "SELECT ip_ev, atomic_mass FROM species_physics WHERE element=? AND sp_num=?",
        (element, sp_num),
    ).fetchone()
    return (None, None) if row is None else (row[0], row[1])


def _top_lines(cur, element, sp_num, wl_min, wl_max, n_keep, aki_min):
    rows = cur.execute(
        """SELECT wavelength_nm, aki, ek_ev, gk, stark_w
           FROM lines
           WHERE element=? AND sp_num=? AND wavelength_nm BETWEEN ? AND ?
                 AND aki > ? AND ek_ev > 0 AND gk > 0
           ORDER BY gk*aki DESC LIMIT ?""",
        (element, sp_num, wl_min, wl_max, aki_min, n_keep),
    ).fetchall()
    return rows


def build(db_path, elements, wl_min, wl_max, lines_per_elem, aki_min, out_path):
    con = sqlite3.connect(db_path)
    cur = con.cursor()

    line_wl, line_gA, line_Ek = [], [], []
    line_elem, line_stage, line_stark = [], [], []
    elem_mass, elem_ip, U_coeffs = [], [], []

    for ei, el in enumerate(elements):
        ip1, mass = _species_phys(cur, el, 1)
        if mass is None:
            mass = 30.0
        if ip1 is None:
            ip1 = 7.0
        elem_mass.append(mass)
        elem_ip.append(ip1)
        U_coeffs.append(
            np.stack([_partition_coeffs(cur, el, 1), _partition_coeffs(cur, el, 2)])
        )

        for stage, sp in ((0, 1), (1, 2)):
            rows = _top_lines(cur, el, sp, wl_min, wl_max, lines_per_elem, aki_min)
            for (wl, aki, ek, gk, sw) in rows:
                line_wl.append(wl)
                line_gA.append(gk * aki)
                line_Ek.append(ek)
                line_elem.append(ei)
                line_stage.append(stage)
                line_stark.append(sw if sw not in (None, "") else 0.0)
            # pad to fixed count with zero-gA dummies
            for _ in range(lines_per_elem - len(rows)):
                line_wl.append(0.5 * (wl_min + wl_max))
                line_gA.append(0.0)
                line_Ek.append(1.0)
                line_elem.append(ei)
                line_stage.append(stage)
                line_stark.append(0.0)

    con.close()

    bundle = dict(
        elements=np.array(elements),
        elem_mass=np.array(elem_mass, dtype=np.float64),
        elem_ip_eV=np.array(elem_ip, dtype=np.float64),
        U_coeffs=np.array(U_coeffs, dtype=np.float64),  # (N_elem, 2, 5)
        line_wl_nm=np.array(line_wl, dtype=np.float64),
        line_gA=np.array(line_gA, dtype=np.float64),
        line_Ek_eV=np.array(line_Ek, dtype=np.float64),
        line_elem=np.array(line_elem, dtype=np.int32),
        line_stage=np.array(line_stage, dtype=np.int32),
        line_stark_w=np.array(line_stark, dtype=np.float64),
        wl_min_nm=np.float64(wl_min),
        wl_max_nm=np.float64(wl_max),
        lines_per_elem=np.int32(lines_per_elem),
        source_db=np.array(db_path),
    )
    np.savez(out_path, **bundle)
    n_real = int(np.sum(bundle["line_gA"] > 0))
    print(
        f"WROTE {out_path}: {len(elements)} elements, "
        f"{len(line_wl)} line slots ({n_real} real, "
        f"{len(line_wl) - n_real} pad), wl=[{wl_min},{wl_max}] nm"
    )
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db-path", required=True)
    ap.add_argument("--elements", nargs="+", default=DEFAULT_ELEMENTS)
    ap.add_argument("--wl-min", type=float, default=240.0)
    ap.add_argument("--wl-max", type=float, default=850.0)
    ap.add_argument("--lines-per-elem", type=int, default=24)
    ap.add_argument("--aki-min", type=float, default=1e5)
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    build(
        a.db_path, a.elements, a.wl_min, a.wl_max,
        a.lines_per_elem, a.aki_min, a.output,
    )


if __name__ == "__main__":
    main()
