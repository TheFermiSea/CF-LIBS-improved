#!/usr/bin/env python
"""Stochastically validate the atomic DB against live NIST ASD.

The DB was built from the 2022 ASD59 dump (gold standard); this confirms it still
agrees with CURRENT NIST by fetching a random sample of (element, stage) species
fresh and diffing level counts + ionization potentials. The DB holds the union of
the 2022 dump and an earlier current-NIST level ingest, so DB level count should
be >= the live count; a DB-far-below-live result flags a real gap.
"""
from __future__ import annotations

import argparse
import random
import sqlite3
import sys

_ROMAN = {1: "I", 2: "II", 3: "III"}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default="ASD_da/libs_production.db")
    p.add_argument("--n", type=int, default=18)
    p.add_argument("--seed", type=int, default=20260626)
    args = p.parse_args(argv)

    import datagen_v2 as d

    conn = sqlite3.connect(args.db)
    species = [
        (r[0], r[1])
        for r in conn.execute(
            "SELECT DISTINCT element, sp_num FROM energy_levels WHERE sp_num IN (1,2,3) "
            "AND element NOT GLOB '[0-9]*'"
        )
    ]
    rng = random.Random(args.seed)
    # always include the DED-relevant metals + a random spread
    forced = [("Ti", 1), ("Al", 1), ("V", 2), ("Fe", 2), ("Cr", 1), ("Ni", 1)]
    sample = [s for s in forced if s in species]
    pool = [s for s in species if s not in sample]
    sample += rng.sample(pool, max(0, args.n - len(sample)))

    lvl_ok = ip_ok = lvl_tot = ip_tot = 0
    print(f"{'species':10} {'DBlvl':>6} {'NISTlvl':>7} {'DBip':>8} {'NISTip':>8}  verdict")
    for el, st in sample:
        roman = _ROMAN[st]
        db_lvl = conn.execute(
            "SELECT COUNT(*) FROM energy_levels WHERE element=? AND sp_num=?", (el, st)
        ).fetchone()[0]
        db_ip = conn.execute(
            "SELECT ip_ev FROM species_physics WHERE element=? AND sp_num=?", (el, st)
        ).fetchone()
        db_ip = db_ip[0] if db_ip else None
        try:
            nist_lvl = len(d.fetch_energy_levels(el, roman))
        except Exception:
            nist_lvl = None
        try:
            nist_ip = d.fetch_ionization_potential(el, roman)
        except Exception:
            nist_ip = None

        v = []
        if nist_lvl is not None:
            lvl_tot += 1
            # DB (union) should cover live NIST within 5% (allow small drift)
            if db_lvl >= nist_lvl * 0.95:
                lvl_ok += 1
                v.append("lvl OK")
            else:
                v.append(f"lvl LOW ({db_lvl}<{nist_lvl})")
        if nist_ip and db_ip:
            ip_tot += 1
            if abs(db_ip - nist_ip) / nist_ip < 0.02:
                ip_ok += 1
                v.append("ip OK")
            else:
                v.append(f"ip DIFF ({db_ip} vs {nist_ip})")
        print(f"{el+' '+roman:10} {db_lvl:6} {str(nist_lvl):>7} {str(round(db_ip,3) if db_ip else None):>8} "
              f"{str(round(nist_ip,3) if nist_ip else None):>8}  {', '.join(v)}")

    print(f"\nLEVELS: {lvl_ok}/{lvl_tot} species DB covers live NIST (>=95%)")
    print(f"IPs:    {ip_ok}/{ip_tot} species DB matches live NIST (<2%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
