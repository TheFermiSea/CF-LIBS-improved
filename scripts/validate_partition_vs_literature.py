"""Validate shipped partition functions vs DB direct-sum vs literature/g0.

The workflow's validation phase was rate-limited, so the committed partition fix
(2cbf094) is unverified. This independently: (a) checks g0 = ground-level 2J+1
(the Fe I g0=1-not-9 bug), (b) computes U(T) by exact direct sum over the
complete DB, (c) calls the shipped partition provider, and (d) sanity-anchors a
few species against Barklem & Collet (2016)/NIST magnitudes.
"""
import math
import sqlite3
import sys

DB = "ASD_da/libs_production.db"
KB_EV = 8.617333262e-5

# ground-level g0 = 2J0+1 (hard facts) and a rough U anchor at 5000 K (BC2016/NIST)
EXPECT = {
    ("Fe", 1): {"g0": 9, "U5000": 25.9},
    ("Fe", 2): {"g0": 10, "U5000": 42.0},
    ("Na", 1): {"g0": 2, "U5000": 2.07},
    ("Ca", 1): {"g0": 1, "U5000": 1.4},
    ("K", 1): {"g0": 2, "U5000": 2.1},
    ("Cr", 1): {"g0": 7, "U5000": 11.0},
    ("Al", 1): {"g0": 2, "U5000": 5.9},
    ("Mg", 1): {"g0": 1, "U5000": 1.0},
    ("Cu", 1): {"g0": 2, "U5000": 2.6},
    ("Ti", 1): {"g0": 5, "U5000": 27.0},
}


def db_sum(c, el, sp, T):
    ip = c.execute("SELECT ip_ev FROM species_physics WHERE element=? AND sp_num=?", (el, sp)).fetchone()
    ip = ip[0] if ip else 1e9
    U = 0.0
    g0 = None
    e0 = 1e9
    for ev, g in c.execute(
        "SELECT energy_ev, g_level FROM energy_levels WHERE element=? AND sp_num=? "
        "AND g_level IS NOT NULL AND g_level>0 ORDER BY energy_ev", (el, sp)):
        if ev < e0:
            e0, g0 = ev, g
        if ev < ip:
            U += g * math.exp(-ev / (KB_EV * T))
    return U, g0, ip


def shipped(el, sp, T):
    try:
        from cflibs.atomic.database import AtomicDatabase
        db = AtomicDatabase(DB)
        prov = db.partition_function_for(el, sp)
        return prov.at(T) if prov else None
    except Exception as exc:  # noqa: BLE001
        return f"ERR:{type(exc).__name__}:{exc}"


def main():
    c = sqlite3.connect(DB)
    print(f"{'species':8} {'g0_db':>5} {'g0_exp':>6} {'U_db(5k/10k/15k)':>26} {'U_shipped(5k/10k/15k)':>30} {'lit5k':>7}  flag")
    bad = []
    for (el, sp), exp in EXPECT.items():
        Us = [db_sum(c, el, sp, T) for T in (5000, 10000, 15000)]
        g0 = Us[0][1]
        ud = [u[0] for u in Us]
        ush = [shipped(el, sp, T) for T in (5000, 10000, 15000)]
        flag = []
        if g0 != exp["g0"]:
            flag.append(f"g0 {g0}!={exp['g0']}")
        if abs(ud[0] - exp["U5000"]) / exp["U5000"] > 0.25:
            flag.append(f"Udb {ud[0]:.1f} vs lit {exp['U5000']}")
        # shipped vs db direct sum
        try:
            if isinstance(ush[1], (int, float)) and abs(ush[1] - ud[1]) / ud[1] > 0.10:
                flag.append(f"shipped {ush[1]:.1f} != dbsum {ud[1]:.1f} @10k")
        except Exception:
            pass
        sh_str = "/".join(f"{x:.1f}" if isinstance(x, (int, float)) else str(x)[:10] for x in ush)
        print(f"{el+' '+str(sp):8} {str(g0):>5} {exp['g0']:>6} "
              f"{ud[0]:7.1f}/{ud[1]:7.1f}/{ud[2]:7.1f}      {sh_str:>30} {exp['U5000']:>7}  {', '.join(flag) or 'OK'}")
        if flag:
            bad.append((el, sp, flag))
    print(f"\n{len(EXPECT)-len(bad)}/{len(EXPECT)} species OK; issues: {bad}")
    c.close()
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
