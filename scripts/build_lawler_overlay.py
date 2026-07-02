#!/usr/bin/env python3
"""Build the lifetime-anchored A_ki OVERLAY database from staged Lawler/Wisconsin tables.

Physics-first-principles audit, Issue 1 (atomic-data gA scale bias). This script
parses the staged Fourier-transform-spectroscopy lab measurements (Lawler / Den
Hartog / Ruffoni / Wood / Sobeck, the "Wisconsin group" line lists), cross-matches
them READ-ONLY against the production atomic DB, and writes a SEPARATE overlay DB
carrying anchored A_ki values + provenance.

HARD IMMUTABILITY MANDATE: the production DB and every standard source are opened
``mode=ro`` only. All enrichment lands in a NEW file (default
``ASD_da/overlays/lawler_anchors_v1.db``). Nothing here ever ALTERs/UPDATEs a
standard DB.

Physics-only: numpy + sqlite3 + stdlib (no ML libraries).

Rerun:  PYTHONPATH=. python3 scripts/build_lawler_overlay.py
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sqlite3
from dataclasses import dataclass, field

import numpy as np

from cflibs.atomic.aki_anchor import a_from_log_gf
from cflibs.atomic.database import AtomicDatabase

# hc conversion: 1 cm^-1 = 1.239841984e-4 eV (matches the production DB's own scale).
CM_TO_EV = 1.239841984e-4

STAGING = "data/atomic/lawler"
DEFAULT_DB = "ASD_da/libs_production.db"
DEFAULT_OVERLAY = "ASD_da/overlays/lawler_anchors_v1.db"


# --------------------------------------------------------------------------- #
# Normalized intermediate records
# --------------------------------------------------------------------------- #
@dataclass
class LevelTau:
    element: str
    sp_num: int
    energy_cm: float
    j: float | None
    tau_ns: float
    tau_unc_frac: float
    source: str


@dataclass
class LineGf:
    element: str
    sp_num: int
    wavelength_air_nm: float
    e_low_cm: float
    e_up_cm: float
    j_low: float | None
    j_up: float | None
    log_gf: float
    log_gf_unc: float | None
    lab_a: float | None  # published transition probability (s^-1), if in the table
    source: str


# --------------------------------------------------------------------------- #
# Small fixed-width helpers
# --------------------------------------------------------------------------- #
def _f(line: str, a: int, b: int) -> float | None:
    """Fixed-width float slice (1-indexed inclusive bytes, CDS convention)."""
    tok = line[a - 1 : b].strip()
    if not tok:
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def _s(line: str, a: int, b: int) -> str:
    return line[a - 1 : b].strip()


def _term_j(term: str) -> float | None:
    """Extract J from a packed term like ``e^7^D_4_`` or ``a^6^D_9/2_``."""
    m = re.findall(r"_(\d+(?:/\d+)?)_", term)
    if not m:
        return None
    tok = m[-1]
    if "/" in tok:
        n, d = tok.split("/")
        return float(n) / float(d)
    return float(tok)


# --------------------------------------------------------------------------- #
# Per-source parsers -> (levels_tau, lines_gf)
# --------------------------------------------------------------------------- #
def parse_fe1_denhartog(base: str) -> tuple[list[LevelTau], list[LineGf]]:
    levels: list[LevelTau] = []
    lines: list[LineGf] = []
    src = "DenHartog2014_ApJS_215_23"
    # table1.dat is pipe-delimited: S|Config|Term|J|E1|E0|lam1|lam2|lambda|tau|tauP e_tauP|r_tauP
    with open(os.path.join(base, "table1.dat")) as fh:
        for raw in fh:
            parts = raw.rstrip("\n").split("|")
            if len(parts) < 10:
                continue
            e_up = _try_float(parts[4])
            jj = _try_float(parts[3])
            tau = _try_float(parts[9])
            if e_up is None or tau is None:
                continue
            levels.append(LevelTau("Fe", 1, e_up, jj, tau, 0.05, src))
    # table4.dat fixed width; loggf at 55-59, only rows with loggf are lab measurements
    with open(os.path.join(base, "table4.dat")) as fh:
        for raw in fh:
            if not raw.strip():
                continue
            lam = _f(raw, 1, 10)
            e_up = _f(raw, 12, 20)
            j_up = _f(raw, 22, 22)
            e_low = _f(raw, 24, 32)
            j_low = _f(raw, 34, 34)
            a_pub = _f(raw, 47, 53)  # 10^6 /s
            loggf = _f(raw, 55, 59)
            e_loggf = _f(raw, 61, 64)
            if lam is None or e_up is None or e_low is None or loggf is None:
                continue
            lines.append(
                LineGf(
                    "Fe",
                    1,
                    lam / 10.0,
                    e_low,
                    e_up,
                    j_low,
                    j_up,
                    loggf,
                    e_loggf,
                    a_pub * 1e6 if a_pub is not None else None,
                    src,
                )
            )
    return levels, lines


def parse_fe1_ruffoni(base: str) -> tuple[list[LevelTau], list[LineGf]]:
    levels: list[LevelTau] = []
    lines: list[LineGf] = []
    src = "Ruffoni2014_MNRAS_441_3127"
    # Parse Note (G1) from ReadMe: letter -> (energy_cm, J, tau_ns)
    state_map: dict[str, tuple[float, float | None, float]] = {}
    with open(os.path.join(base, "ReadMe")) as fh:
        text = fh.read()
    g1 = text.split("Note (G1):")[1].split("Note (G2)")[0]
    # Blocks like:  A = Level 43163.323 cm^-1^ ... e^7^D_4_\n   8.5ns +/-5%; ...
    for m in re.finditer(
        r"\b([A-L])\s*=\s*Level\s+([\d.]+)\s*cm\^-1\^\s+.*?([a-z]\^\d\^[A-Z]\^?o?\^?_\d+(?:/\d+)?_)"
        r".*?([\d.]+)\s*ns",
        g1,
        re.S,
    ):
        letter, e_cm, term, tau = m.group(1), float(m.group(2)), m.group(3), float(m.group(4))
        state_map[letter] = (e_cm, _term_j(term), tau)
    for letter, (e_cm, jj, tau) in state_map.items():
        levels.append(LevelTau("Fe", 1, e_cm, jj, tau, 0.05, src))
    # table3.dat: skip '#' lines. State(1), lam.Air(14-23), sigma(25-33), loggf(54-58)
    with open(os.path.join(base, "table3.dat")) as fh:
        for raw in fh:
            if raw.startswith("#") or not raw.strip():
                continue
            state = _s(raw, 1, 1)
            lam = _f(raw, 14, 23)
            sigma = _f(raw, 25, 33)
            loggf = _f(raw, 54, 58)
            e_loggf = _f(raw, 60, 63)
            if state not in state_map or lam is None or sigma is None or loggf is None:
                continue
            e_up, j_up, _ = state_map[state]
            e_low = e_up - sigma
            lines.append(
                LineGf("Fe", 1, lam / 10.0, e_low, e_up, None, j_up, loggf, e_loggf, None, src)
            )
    return levels, lines


def parse_fe2_denhartog(path: str) -> tuple[list[LevelTau], list[LineGf]]:
    lines: list[LineGf] = []
    src = "DenHartog2019_ApJS_243_33_T6"
    # Fixed width; data starts after the byte-by-byte header. loggf 55-59.
    with open(path) as fh:
        for raw in fh:
            lam = _f(raw, 1, 8)
            e_up = _f(raw, 10, 19)
            j_up = _f(raw, 21, 23)
            e_low = _f(raw, 25, 34)
            j_low = _f(raw, 36, 38)
            a_pub = _f(raw, 40, 46)  # 10^6/s
            loggf = _f(raw, 55, 59)
            e_loggf = _f(raw, 61, 64)
            # Accept only well-formed data rows (header lines fail the numeric slices)
            if lam is None or e_up is None or e_low is None or loggf is None:
                continue
            if lam < 1000 or e_up < 1000:  # guards against header text bleeding in
                continue
            lines.append(
                LineGf(
                    "Fe",
                    2,
                    lam / 10.0,
                    e_low,
                    e_up,
                    j_low,
                    j_up,
                    loggf,
                    e_loggf,
                    a_pub * 1e6 if a_pub is not None else None,
                    src,
                )
            )
    return [], lines


def parse_ti1(path: str) -> tuple[list[LevelTau], list[LineGf]]:
    lines: list[LineGf] = []
    src = "Lawler2013_ApJS_205_11"
    # t3 MRT: WaveAir 1-10, UpLev 12-20, UpJ 22, LowLev 24-32, LowJ 34, TranP 36-43, loggf 53-57
    with open(path) as fh:
        for raw in fh:
            lam = _f(raw, 1, 10)
            e_up = _f(raw, 12, 20)
            j_up = _f(raw, 22, 22)
            e_low = _f(raw, 24, 32)
            j_low = _f(raw, 34, 34)
            a_pub = _f(raw, 36, 43)
            loggf = _f(raw, 53, 57)
            if lam is None or e_up is None or e_low is None or loggf is None:
                continue
            if lam < 1000:
                continue
            lines.append(
                LineGf(
                    "Ti",
                    1,
                    lam / 10.0,
                    e_low,
                    e_up,
                    j_low,
                    j_up,
                    loggf,
                    None,
                    a_pub * 1e6 if a_pub is not None else None,
                    src,
                )
            )
    return [], lines


def parse_ti2(path: str) -> tuple[list[LevelTau], list[LineGf]]:
    lines: list[LineGf] = []
    src = "Wood2013_ApJS_208_27_T4"
    # lamAir 1-9, E1 11-19, J1 21-23, E0 25-33, J0 35-37, A 39-46, loggf 56-60
    with open(path) as fh:
        for raw in fh:
            lam = _f(raw, 1, 9)
            e_up = _f(raw, 11, 19)
            j_up = _f(raw, 21, 23)
            e_low = _f(raw, 25, 33)
            j_low = _f(raw, 35, 37)
            a_pub = _f(raw, 39, 46)
            loggf = _f(raw, 56, 60)
            if lam is None or e_up is None or e_low is None or loggf is None:
                continue
            lines.append(
                LineGf(
                    "Ti",
                    2,
                    lam / 10.0,
                    e_low,
                    e_up,
                    j_low,
                    j_up,
                    loggf,
                    None,
                    a_pub * 1e6 if a_pub is not None else None,
                    src,
                )
            )
    return [], lines


def parse_v1(path: str) -> tuple[list[LevelTau], list[LineGf]]:
    lines: list[LineGf] = []
    src = "Lawler2014_ApJS_215_20_T3"
    # lamAir 1-9, E1 11-19, J1 24-26, E0 28-36, J0 41-43, A 45-52, loggf 62-66
    with open(path) as fh:
        for raw in fh:
            lam = _f(raw, 1, 9)
            e_up = _f(raw, 11, 19)
            j_up = _f(raw, 24, 26)
            e_low = _f(raw, 28, 36)
            j_low = _f(raw, 41, 43)
            a_pub = _f(raw, 45, 52)
            loggf = _f(raw, 62, 66)
            if lam is None or e_up is None or e_low is None or loggf is None:
                continue
            lines.append(
                LineGf(
                    "V",
                    1,
                    lam / 10.0,
                    e_low,
                    e_up,
                    j_low,
                    j_up,
                    loggf,
                    None,
                    a_pub * 1e6 if a_pub is not None else None,
                    src,
                )
            )
    return [], lines


def parse_cr1(path: str) -> tuple[list[LevelTau], list[LineGf]]:
    lines: list[LineGf] = []
    src = "Sobeck2007_ApJ_667_1267_T3"
    # lambda 1-7, EU 9-16, JU 27, EL 29-36, JL 47, Aexp 59-65, loggf 74-78
    with open(path) as fh:
        for raw in fh:
            lam = _f(raw, 1, 7)
            e_up = _f(raw, 9, 16)
            j_up = _f(raw, 27, 27)
            e_low = _f(raw, 29, 36)
            j_low = _f(raw, 47, 47)
            a_pub = _f(raw, 59, 65)
            loggf = _f(raw, 74, 78)
            if lam is None or e_up is None or e_low is None or loggf is None:
                continue
            lines.append(
                LineGf(
                    "Cr",
                    1,
                    lam / 10.0,
                    e_low,
                    e_up,
                    j_low,
                    j_up,
                    loggf,
                    None,
                    a_pub * 1e6 if a_pub is not None else None,
                    src,
                )
            )
    return [], lines


def parse_mn(path: str) -> tuple[list[LevelTau], list[LineGf]]:
    lines: list[LineGf] = []
    src = "DenHartog2011_ApJS_194_35_T3"
    # lamAir 1-7, E1 9-16, J1 21-23, E0 25-32, J0 37-39, TranP 41-46, loggf(recommended) 53-58
    # Half-integer J1 -> Mn I (sp=1); integer J1 -> Mn II (sp=2).
    with open(path) as fh:
        for raw in fh:
            lam = _f(raw, 1, 7)
            e_up = _f(raw, 9, 16)
            j_up = _f(raw, 21, 23)
            e_low = _f(raw, 25, 32)
            j_low = _f(raw, 37, 39)
            a_pub = _f(raw, 41, 46)
            loggf = _f(raw, 53, 58)
            if lam is None or e_up is None or e_low is None or loggf is None or j_up is None:
                continue
            sp = 1 if abs(j_up - round(j_up)) > 0.25 else 2  # half-int -> Mn I
            lines.append(
                LineGf(
                    "Mn",
                    sp,
                    lam / 10.0,
                    e_low,
                    e_up,
                    j_low,
                    j_up,
                    loggf,
                    None,
                    a_pub * 1e6 if a_pub is not None else None,
                    src,
                )
            )
    return [], lines


def parse_ni1(path: str) -> tuple[list[LevelTau], list[LineGf]]:
    lines: list[LineGf] = []
    src = "Wood2014_ApJS_211_20_T3"
    # lambda 1-9, E1 11-19, J1 24, E0 26-34, J0 39, A 41-48, loggf 58-62
    with open(path) as fh:
        for raw in fh:
            lam = _f(raw, 1, 9)
            e_up = _f(raw, 11, 19)
            j_up = _f(raw, 24, 24)
            e_low = _f(raw, 26, 34)
            j_low = _f(raw, 39, 39)
            a_pub = _f(raw, 41, 48)
            loggf = _f(raw, 58, 62)
            if lam is None or e_up is None or e_low is None or loggf is None:
                continue
            lines.append(
                LineGf(
                    "Ni",
                    1,
                    lam / 10.0,
                    e_low,
                    e_up,
                    j_low,
                    j_up,
                    loggf,
                    None,
                    a_pub * 1e6 if a_pub is not None else None,
                    src,
                )
            )
    return [], lines


def _try_float(tok: str) -> float | None:
    tok = tok.strip()
    if not tok:
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def parse_all() -> tuple[list[LevelTau], list[LineGf]]:
    levels: list[LevelTau] = []
    lines: list[LineGf] = []
    fe1_dh = f"{STAGING}/FeI/DenHartog2014_J_ApJS_215_23"
    fe1_ru = f"{STAGING}/FeI/Ruffoni2014_J_MNRAS_441_3127"
    tv = f"{STAGING}/TiVCrMnNi"
    for lv, ln in [
        parse_fe1_denhartog(fe1_dh),
        parse_fe1_ruffoni(fe1_ru),
        parse_fe2_denhartog(f"{STAGING}/FeII/denhartog2019_table6_mrt.txt"),
        parse_ti1(f"{tv}/Ti1_Lawler2013_J_ApJS_205_11/apjs461528t3_mrt.txt"),
        parse_ti2(f"{tv}/Ti2_Wood2013_J_ApJS_208_27/table4.dat"),
        parse_v1(f"{tv}/V1_Lawler2014_J_ApJS_215_20/table3.dat"),
        parse_cr1(f"{tv}/Cr1_Sobeck2007_J_ApJ_667_1267/table3.dat"),
        parse_mn(f"{tv}/Mn1_DenHartog2011_J_ApJS_194_35/table3.dat"),
        parse_ni1(f"{tv}/Ni1_Wood2014_J_ApJS_211_20/table3.dat"),
    ]:
        levels.extend(lv)
        lines.extend(ln)
    return levels, lines


# --------------------------------------------------------------------------- #
# Read-only cross-matching against the production DB
# --------------------------------------------------------------------------- #
def _parse_db_j(s: str | None) -> float | None:
    if not s:
        return None
    s = s.strip()
    if s in ("---", ""):
        return None
    if "/" in s:
        n, d = s.split("/")
        try:
            return float(n) / float(d)
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


@dataclass
class LevelMatch:
    level_id: str | None
    dist_cm: float | None
    status: str  # matched | ambiguous | nomatch


def match_level(
    conn: sqlite3.Connection, element: str, sp: int, energy_cm: float, jf: float | None
) -> LevelMatch:
    """Match a lab level to a unique DB energy level by energy (0.01 then 0.1 cm-1) + J."""
    for tol in (0.01, 0.1):
        lo = (energy_cm - tol) * CM_TO_EV
        hi = (energy_cm + tol) * CM_TO_EV
        rows = conn.execute(
            "SELECT level_id, energy_ev, j_val FROM energy_levels "
            "WHERE element=? AND sp_num=? AND energy_ev BETWEEN ? AND ?",
            (element, sp, lo, hi),
        ).fetchall()
        cands = []
        for r in rows:
            if jf is not None:
                dbj = _parse_db_j(r["j_val"])
                if dbj is not None and abs(dbj - jf) > 0.1:
                    continue
            cands.append(r)
        if len(cands) == 1:
            db_cm = cands[0]["energy_ev"] / CM_TO_EV
            return LevelMatch(cands[0]["level_id"], abs(db_cm - energy_cm), "matched")
        if len(cands) >= 2:
            return LevelMatch(None, None, "ambiguous")
    return LevelMatch(None, None, "nomatch")


@dataclass
class LineMatch:
    line_id: int | None
    gk: float | None
    db_wl_nm: float | None
    db_aki: float | None
    quality: str  # e.g. "level:0.003" / "wl:0.0012" / "ambiguous" / "nomatch"


def match_line(conn: sqlite3.Connection, ln: LineGf) -> LineMatch:
    """Match a lab gf line to a unique DB line: level-pair first, wavelength fallback."""
    up = match_level(conn, ln.element, ln.sp_num, ln.e_up_cm, ln.j_up)
    lo = match_level(conn, ln.element, ln.sp_num, ln.e_low_cm, ln.j_low)
    if up.status == "ambiguous" or lo.status == "ambiguous":
        return LineMatch(None, None, None, None, "ambiguous_level")
    if up.status == "matched" and lo.status == "matched":
        rows = conn.execute(
            "SELECT id, gk, wavelength_nm, aki FROM lines "
            "WHERE element=? AND sp_num=? AND low_level_id=? AND upp_level_id=? AND aki>0",
            (ln.element, ln.sp_num, lo.level_id, up.level_id),
        ).fetchall()
        if len(rows) == 1:
            r = rows[0]
            d = max(up.dist_cm or 0.0, lo.dist_cm or 0.0)
            return LineMatch(int(r["id"]), r["gk"], r["wavelength_nm"], r["aki"], f"level:{d:.4f}")
        if len(rows) >= 2:
            return LineMatch(None, None, None, None, "ambiguous_linepair")
    # Wavelength fallback: air wavelength within 0.004 nm + lower-energy sanity
    tol_nm = 0.004
    rows = conn.execute(
        "SELECT id, gk, wavelength_nm, aki, ei_ev FROM lines "
        "WHERE element=? AND sp_num=? AND aki>0 AND wavelength_nm BETWEEN ? AND ?",
        (ln.element, ln.sp_num, ln.wavelength_air_nm - tol_nm, ln.wavelength_air_nm + tol_nm),
    ).fetchall()
    if ln.e_low_cm is not None:
        e_low_ev = ln.e_low_cm * CM_TO_EV
        rows = [r for r in rows if r["ei_ev"] is None or abs(r["ei_ev"] - e_low_ev) < 0.02]
    if len(rows) == 1:
        r = rows[0]
        return LineMatch(
            int(r["id"]),
            r["gk"],
            r["wavelength_nm"],
            r["aki"],
            f"wl:{abs(r['wavelength_nm'] - ln.wavelength_air_nm):.4f}",
        )
    if len(rows) >= 2:
        return LineMatch(None, None, None, None, "ambiguous_wl")
    return LineMatch(None, None, None, None, "nomatch")


# --------------------------------------------------------------------------- #
# Overlay build
# --------------------------------------------------------------------------- #
def _create_overlay(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE anchored_lines (
            line_id INTEGER PRIMARY KEY,
            element TEXT NOT NULL,
            sp_num INTEGER NOT NULL,
            aki_anchored REAL NOT NULL,
            aki_unc REAL,
            method TEXT NOT NULL,          -- lab_gf | tau_renorm
            source TEXT NOT NULL,
            match_quality TEXT
        );
        CREATE TABLE anchored_levels (
            level_id TEXT PRIMARY KEY,
            element TEXT NOT NULL,
            sp_num INTEGER NOT NULL,
            tau_ns REAL NOT NULL,
            tau_unc REAL,
            source TEXT NOT NULL,
            f_obs REAL
        );
        CREATE TABLE provenance (key TEXT, value TEXT);
        """)
    return conn


@dataclass
class SpeciesStat:
    element: str
    sp_num: int
    n_lab_lines: int = 0
    n_matched: int = 0
    n_ambiguous: int = 0
    n_nomatch: int = 0
    match_dist_cm: list[float] = field(default_factory=list)
    ln_a_ratio: list[float] = field(default_factory=list)  # anchored/DB (scale error)
    n_tau_levels: int = 0
    n_tau_matched: int = 0
    tau_gate_ln: list[float] = field(default_factory=list)  # ln(sum_A_DB * tau)


def build(db_path: str, overlay_path: str) -> dict:
    levels, lines = parse_all()
    # Source DB is IMMUTABLE: open through AtomicDatabase's read-only mode
    # (sqlite mode=ro). No write can reach it and no schema migration runs.
    source_db = AtomicDatabase(db_path, read_only=True)
    conn = source_db.conn
    conn.row_factory = sqlite3.Row
    ov = _create_overlay(overlay_path)

    stats: dict[tuple[str, int], SpeciesStat] = {}

    def stat(el: str, sp: int) -> SpeciesStat:
        return stats.setdefault((el, sp), SpeciesStat(el, sp))

    ambiguous_list: list[str] = []

    # --- Levels: match tau to DB level; store anchored_levels (+ f_obs honesty) ---
    level_id_by_key: dict[tuple[str, int, str], float] = {}  # matched level_id -> tau_s
    for lv in levels:
        s = stat(lv.element, lv.sp_num)
        s.n_tau_levels += 1
        m = match_level(conn, lv.element, lv.sp_num, lv.energy_cm, lv.j)
        if m.status != "matched":
            if m.status == "ambiguous":
                ambiguous_list.append(
                    f"LEVEL {lv.element} {lv.sp_num} E={lv.energy_cm:.3f} J={lv.j} ({lv.source})"
                )
            continue
        s.n_tau_matched += 1
        s.match_dist_cm.append(m.dist_cm or 0.0)
        tau_s = lv.tau_ns * 1e-9
        # f_obs: fraction of 1/tau captured by DB branches from this level (<=1, honest).
        rows = conn.execute(
            "SELECT aki FROM lines WHERE element=? AND sp_num=? AND upp_level_id=? AND aki>0",
            (lv.element, lv.sp_num, m.level_id),
        ).fetchall()
        sum_a_db = float(sum(r["aki"] for r in rows)) if rows else 0.0
        f_obs = min(1.0, sum_a_db * tau_s) if sum_a_db > 0 else 0.0
        if sum_a_db > 0:
            s.tau_gate_ln.append(math.log(sum_a_db * tau_s))
        ov.execute(
            "INSERT OR REPLACE INTO anchored_levels VALUES (?,?,?,?,?,?,?)",
            (
                m.level_id,
                lv.element,
                lv.sp_num,
                lv.tau_ns,
                lv.tau_ns * lv.tau_unc_frac,
                lv.source,
                f_obs,
            ),
        )
        level_id_by_key[(lv.element, lv.sp_num, m.level_id)] = tau_s

    # --- Lines: prefer per-line lab A from lab log gf (method=lab_gf) ---
    lab_matched_line_ids: set[int] = set()
    for ln in lines:
        s = stat(ln.element, ln.sp_num)
        s.n_lab_lines += 1
        m = match_line(conn, ln)
        if m.line_id is None:
            if m.quality.startswith("ambiguous"):
                s.n_ambiguous += 1
                ambiguous_list.append(
                    f"LINE {ln.element} {ln.sp_num} lam={ln.wavelength_air_nm:.4f}nm "
                    f"Eup={ln.e_up_cm:.3f} ({m.quality}, {ln.source})"
                )
            else:
                s.n_nomatch += 1
            continue
        # gk: prefer DB gk, else statistical weight from lab J_up.
        gk = m.gk
        if (gk is None or gk <= 0) and ln.j_up is not None:
            gk = 2.0 * ln.j_up + 1.0
        if gk is None or gk <= 0:
            s.n_nomatch += 1
            continue
        wl = m.db_wl_nm or ln.wavelength_air_nm
        a_anchored = a_from_log_gf(ln.log_gf, gk, wl)
        # uncertainty: fractional from e_loggf (ln10*e) if present, else 5%.
        if ln.log_gf_unc is not None:
            aki_unc = math.log(10.0) * ln.log_gf_unc * a_anchored
        else:
            aki_unc = 0.05 * a_anchored
        s.n_matched += 1
        s.match_dist_cm.append(
            0.0 if m.quality.startswith("wl") else float(m.quality.split(":")[1])
        )
        if m.db_aki and m.db_aki > 0:
            s.ln_a_ratio.append(math.log(a_anchored / m.db_aki))
        ov.execute(
            "INSERT OR REPLACE INTO anchored_lines VALUES (?,?,?,?,?,?,?,?)",
            (m.line_id, ln.element, ln.sp_num, a_anchored, aki_unc, "lab_gf", ln.source, m.quality),
        )
        lab_matched_line_ids.add(m.line_id)

    # --- tau_renorm: ONLY for tau levels whose DB lines got no lab_gf anchor ---
    n_tau_renorm = 0
    for (el, sp, lvl_id), tau_s in level_id_by_key.items():
        rows = conn.execute(
            "SELECT id, aki FROM lines WHERE element=? AND sp_num=? AND upp_level_id=? AND aki>0",
            (el, sp, lvl_id),
        ).fetchall()
        if not rows:
            continue
        unanchored = [r for r in rows if int(r["id"]) not in lab_matched_line_ids]
        if not unanchored:
            continue  # fully covered by preferred lab_gf anchors
        # f_obs from anchored_levels (already stored). Honest residual-branch handling:
        # distribute f_obs/tau across the level's observed DB branches by BR.
        frow = ov.execute(
            "SELECT f_obs FROM anchored_levels WHERE level_id=?", (lvl_id,)
        ).fetchone()
        f_obs = frow[0] if frow and frow[0] is not None else 0.0
        if f_obs <= 0:
            continue  # DB branches look incomplete -> never over-boost; skip
        aki = np.asarray([r["aki"] for r in rows], dtype=float)
        tot = float(aki.sum())
        if tot <= 0:
            continue
        for r in unanchored:
            br = float(r["aki"]) / tot
            a_new = br * f_obs / tau_s
            ov.execute(
                "INSERT OR REPLACE INTO anchored_lines VALUES (?,?,?,?,?,?,?,?)",
                (
                    int(r["id"]),
                    el,
                    sp,
                    a_new,
                    0.05 * a_new,
                    "tau_renorm",
                    "lifetime_renorm",
                    f"f_obs={f_obs:.3f}",
                ),
            )
            n_tau_renorm += 1

    # provenance
    ov.execute("INSERT INTO provenance VALUES (?,?)", ("source_db", db_path))
    ov.execute(
        "INSERT INTO provenance VALUES (?,?)", ("n_ambiguous_refused", str(len(ambiguous_list)))
    )
    ov.execute("INSERT INTO provenance VALUES (?,?)", ("n_tau_renorm_lines", str(n_tau_renorm)))
    ov.commit()

    report = _summarize(stats, ambiguous_list, n_tau_renorm)
    ov.execute("INSERT INTO provenance VALUES (?,?)", ("build_summary", repr(report)))
    ov.commit()
    ov.close()
    source_db.close()
    return report


def _summarize(stats, ambiguous_list, n_tau_renorm) -> dict:
    per_species = {}
    for (el, sp), s in sorted(stats.items()):
        dists = np.asarray(s.match_dist_cm, dtype=float) if s.match_dist_cm else np.array([])
        lnr = np.asarray(s.ln_a_ratio, dtype=float) if s.ln_a_ratio else np.array([])
        gate = np.asarray(s.tau_gate_ln, dtype=float) if s.tau_gate_ln else np.array([])
        per_species[f"{el} {sp}"] = {
            "n_lab_lines": s.n_lab_lines,
            "n_matched": s.n_matched,
            "n_ambiguous": s.n_ambiguous,
            "n_nomatch": s.n_nomatch,
            "match_rate": round(s.n_matched / s.n_lab_lines, 4) if s.n_lab_lines else None,
            "match_dist_cm_median": round(float(np.median(dists)), 5) if dists.size else None,
            "match_dist_cm_p95": round(float(np.percentile(dists, 95)), 5) if dists.size else None,
            "scale_err_ln_median": round(float(np.median(lnr)), 4) if lnr.size else None,
            "scale_err_ln_mad": (
                round(float(np.median(np.abs(lnr - np.median(lnr)))), 4) if lnr.size else None
            ),
            "scale_factor_median": round(float(np.exp(np.median(lnr))), 4) if lnr.size else None,
            "n_tau_levels": s.n_tau_levels,
            "n_tau_matched": s.n_tau_matched,
            "tau_gate_ln_median": round(float(np.median(gate)), 4) if gate.size else None,
            "tau_gate_ln_mad": (
                round(float(np.median(np.abs(gate - np.median(gate)))), 4) if gate.size else None
            ),
            "tau_gate_scale_median": (
                round(float(np.exp(np.median(gate))), 4) if gate.size else None
            ),
        }
    return {
        "per_species": per_species,
        "n_ambiguous_refused": len(ambiguous_list),
        "ambiguous_examples": ambiguous_list[:20],
        "n_tau_renorm_lines": n_tau_renorm,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--overlay", default=DEFAULT_OVERLAY)
    args = ap.parse_args()
    report = build(args.db, args.overlay)
    import json

    print(json.dumps(report, indent=2))
    print(f"\nOverlay written: {args.overlay}")


if __name__ == "__main__":
    main()
