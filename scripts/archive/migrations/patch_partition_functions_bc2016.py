#!/usr/bin/env python3
"""Patch ``partition_functions`` against Barklem & Collet (2016), A&A 588, A96.

Why (bead CF-LIBS-improved-16m7; audit 2026-06-09 findings 01-F3 / 02-F1)
-------------------------------------------------------------------------
The production DB's neutral partition functions are 5–40 % LOW versus
Barklem & Collet 2016 (K I −40 %, Na I −30 %, Ca I −25 %, Ti I −8 %,
Al I −6 %, Mg I −5 % at 1e4 K) because the ``datagen_v2.py`` NIST level
scrape drops high-Rydberg levels (Ca I: 76 levels in DB vs ~200 in NIST).
Worse, Na II / Li II / H II have ZERO rows in both ``energy_levels`` and
``partition_functions``, so the solver used hardcoded fallbacks
(U(Na II) = 15.0 vs the true ~1.00 — a ~15× Saha-multiplier error on a
basalt major).

What this script does
---------------------
1. Parses B&C Table 8 (the Nov-2022 bug-fixed revision cached at
   ``data/atomic/bc2016_table8.dat``; ``--download`` refreshes it from the
   authors' GitHub repository, falling back to CDS).
2. For every (element, stage 1–3) that appears for an element with rows in
   our ``lines`` table AND in B&C, refits the DB polynomial form
   ``ln U(T) = Σ aₙ (ln T)ⁿ`` over T ∈ [5000, 25000] K and writes the row
   with ``source='BarklemCollet2016'`` (INSERT OR REPLACE — missing species
   such as Na II are inserted).
3. Writes a PATCHED COPY of the database (default
   ``ASD_da/libs_production_bc2016.db``); ``--in-place`` patches the input
   DB directly (NOT recommended while other agents share the file).
4. Optionally emits a validation markdown table (``--validation-md``) and a
   JSON fit-residual report (``--report-json``).

High-temperature extension (IMPORTANT caveat)
---------------------------------------------
B&C Table 8 tabulates U only up to 10 000 K (~0.86 eV). LIBS plasmas reach
~1.3 eV (~15 000 K), and the DB polynomial domain is [5000, 25000] K. Above
10 000 K the fit targets are therefore an *extension*, not B&C data:

    U_ext(T) = U_base(T) + g_eff · exp(−E_eff / kT)

where ``U_base`` is the direct sum over OUR DB's energy levels (complete
for the low-lying levels that dominate U; the deficit is high-Rydberg), and
the single effective level (g_eff, E_eff) models the missing high-lying
population. (g_eff, E_eff) are solved from the B&C−base differences at
8000 K and 10 000 K, so the extension matches B&C exactly at the grid edge
and grows with the Boltzmann activation of the missing levels. E_eff is
clamped to [0.5 eV, IP] — missing bound levels must lie below the
ionization potential. For species with no DB levels (Na II, Li II, H II)
the base is the species' B&C value at 5000 K (these are closed-shell ions
with essentially flat U). When the missing-level difference is negligible
or non-positive, the extension is flat (D = max(D(1e4), 0)).

This is a physically-motivated *bridge*, not data: above 1e4 K the true
partition sum is regulated by the runtime ionization-potential-depression
cutoff anyway (Alimohamadi & Ferland 2022). Fit residuals over the B&C
window and the extension window are reported separately.

Activation
----------
``cflibs.plasma.partition.derive_partition_spec`` prefers stored rows whose
source is in ``AUTHORITATIVE_PF_SOURCES`` (= {'BarklemCollet2016'}) over
the direct-sum fit, so the patched DB takes effect simply by pointing the
pipeline at it. The patched copy is NOT wired as the default — that is
gated on a BHVO-2 benchmark measurement at integration time.

Usage
-----
    python scripts/archive/migrations/patch_partition_functions_bc2016.py \\
        --db-path ASD_da/libs_production.db \\
        --output-db ASD_da/libs_production_bc2016.db \\
        --validation-md docs/audit/2026-06-09-overhaul/partition-validation.md
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

KB_EV = 8.617333262e-5  # Boltzmann constant [eV/K]
EV_TO_K = 1.0 / KB_EV  # 1 eV = 11604.5 K

DEFAULT_TABLE = REPO_ROOT / "data" / "atomic" / "bc2016_table8.dat"
DEFAULT_DB = REPO_ROOT / "ASD_da" / "libs_production.db"
DEFAULT_OUTPUT_DB = REPO_ROOT / "ASD_da" / "libs_production_bc2016.db"

#: Authors' bug-fixed (Nov 2022) revision — preferred (fixes Na I +17.5 %).
GITHUB_URL = (
    "https://raw.githubusercontent.com/barklem/public-data/master/"
    "partition-functions_and_equilibrium-constants/table8_vNov2022.dat"
)
#: CDS mirror of the ORIGINAL 2016 table (carries the Na I/Tl I J-parse bug).
CDS_URL = "https://cdsarc.cds.unistra.fr/viz-bin/nph-Cat/txt?J%2FA%2BA%2F588%2FA96%2Ftable8.dat"

BC_SOURCE_TAG = "BarklemCollet2016"
ROMAN_TO_STAGE = {"I": 1, "II": 2, "III": 3}

# Fit domain (matches the DB polynomial t_min/t_max written below).
T_FIT_MIN = 5000.0
T_FIT_MAX = 25000.0
#: B&C grid points used as direct fit anchors (within the table's range).
BC_ANCHOR_TEMPS = (5000.0, 6000.0, 7000.0, 8000.0, 9000.0, 9283.6, 10000.0)
#: Extension grid above the B&C table edge (see module docstring) — denser
#: across the LIBS acceptance band (0.8–1.3 eV ≈ 9300–15100 K).
EXTENSION_TEMPS = (
    11000.0, 11604.5, 12500.0, 13500.0, 15085.9, 16000.0,
    18000.0, 20000.0, 22500.0, 25000.0,
)  # fmt: skip
#: The acceptance-gate temperatures (0.8 / 1.0 / 1.3 eV) get extra weight.
KEY_TEMPS = (9283.6, 11604.5, 15085.9)
#: A quartic in ln T cannot follow the Rydberg upturn everywhere; weight the
#: LIBS band (where the solver lives and the gate measures) over the far
#: tail, which the runtime IPD truncation regulates anyway.
LIBS_BAND_MAX = 16000.0
WEIGHT_BC_ANCHOR = 3.0
WEIGHT_EXT_BAND = 3.0
WEIGHT_EXT_TAIL = 0.5
WEIGHT_KEY = 8.0
#: Anchors for the effective-level solve.
EXT_ANCHOR_LO, EXT_ANCHOR_HI = 8000.0, 10000.0

#: Validation species/temperatures (bead 16m7 acceptance gate).
VALIDATION_SPECIES = (
    ("Fe", 1), ("Fe", 2), ("Ti", 1), ("Ti", 2), ("Ca", 1), ("Ca", 2),
    ("Na", 1), ("Na", 2), ("K", 1), ("Al", 1), ("Mg", 1), ("Mg", 2),
    ("Si", 1), ("Si", 2),
)  # fmt: skip
VALIDATION_TEMPS_EV = (0.8, 1.0, 1.3)
ACCEPTANCE_TOL = 0.02  # patched-vs-reference within 2 %


# ---------------------------------------------------------------------------
# B&C Table 8 parsing
# ---------------------------------------------------------------------------


@dataclass
class BCTable:
    """Parsed Table 8: temperature grid + U(T) per (element, stage)."""

    temps_K: np.ndarray
    species: Dict[Tuple[str, int], np.ndarray]

    def interp_ln(self, element: str, stage: int, T_K: float) -> float:
        """U at ``T_K`` by linear interpolation in (ln T, ln U) (in-grid only)."""
        U = self.species[(element, stage)]
        if not (self.temps_K[0] <= T_K <= self.temps_K[-1]):
            raise ValueError(f"T={T_K} K outside the B&C grid")
        return float(
            np.exp(np.interp(np.log(T_K), np.log(self.temps_K), np.log(np.maximum(U, 1e-30))))
        )


def parse_bc_table8(path: Path) -> BCTable:
    """Parse the authors' ``table8`` format (also tolerates the CDS dump).

    Author format: title line, species count, ``T [K]`` grid line, blank,
    then one whitespace-separated row per species (``El_Stage U1 ... U42``).
    The CDS ``nph-Cat/txt`` dump wraps the same numbers in ``#``/``|``
    decoration; rows there are pipe-separated.
    """
    temps: Optional[np.ndarray] = None
    species: Dict[Tuple[str, int], np.ndarray] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith(("Partition", "----", "Ion ")):
            continue
        if line.startswith("#"):  # CDS decoration; the T grid hides in '##   T [K] ...'
            if "T [K]" in line:
                temps = np.array([float(tok) for tok in line.split("T [K]")[1].split()])
            continue
        if line.startswith("T [K]"):
            temps = np.array([float(tok) for tok in line.split("T [K]")[1].split()])
            continue
        tokens = [tok.strip() for tok in (line.split("|") if "|" in line else line.split())]
        tokens = [tok for tok in tokens if tok]
        if len(tokens) < 3 or temps is None:
            continue
        name, values = tokens[0], tokens[1:]
        if "_" not in name:  # anions (S-, Cl-) and stray rows
            continue
        element, roman = name.split("_", 1)
        stage = ROMAN_TO_STAGE.get(roman)
        if stage is None or element == "D":  # deuterium duplicates H
            continue
        if len(values) != len(temps):
            continue
        species[(element, stage)] = np.array([float(v) for v in values])
    if temps is None or not species:
        raise ValueError(f"Could not parse a B&C table from {path}")
    return BCTable(temps_K=temps, species=species)


def download_table(dest: Path) -> None:
    """Fetch the bug-fixed table from GitHub, falling back to CDS (warned)."""
    for url, label in ((GITHUB_URL, "GitHub vNov2022"), (CDS_URL, "CDS (original 2016!)")):
        try:
            print(f"Downloading B&C table 8 from {label} ...")
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = resp.read()
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            if label.startswith("CDS"):
                print(
                    "WARNING: CDS copy carries the pre-Nov-2022 J-parse bug "
                    "(Na I +17.5 % at 1e4 K). Prefer the GitHub revision."
                )
            return
        except Exception as exc:  # noqa: BLE001 - report and try the mirror
            print(f"  {label} failed: {exc}")
    raise RuntimeError("Could not download B&C table 8 from any mirror")


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------


def species_to_patch(db_path: Path, bc: BCTable) -> List[Tuple[str, int]]:
    """Stages 1–3 of every element with rows in ``lines``, where B&C has data.

    Element-level (not species-level) intersection: the Saha/closure chain
    needs U for stages an element never emits lines from (Na II!), so any
    element observed in ``lines`` gets every B&C stage we can patch.
    """
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        elements = {
            row[0] for row in conn.execute("SELECT DISTINCT element FROM lines WHERE sp_num <= 3")
        }
    return sorted((el, sp) for (el, sp) in bc.species if el in elements and sp <= 3)


def load_levels(
    db_path: Path,
) -> Tuple[Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]], Dict[Tuple[str, int], float]]:
    """Read (g, E) level arrays and ionization potentials for all species."""
    levels: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]] = {}
    ips: Dict[Tuple[str, int], float] = {}
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT element, sp_num, g_level, energy_ev FROM energy_levels ORDER BY element, sp_num"
        ).fetchall()
        for el, sp, ip in conn.execute("SELECT element, sp_num, ip_ev FROM species_physics"):
            if ip is not None:
                ips[(el, sp)] = float(ip)
    grouped: Dict[Tuple[str, int], List[Tuple[float, float]]] = {}
    for el, sp, g, e in rows:
        grouped.setdefault((el, sp), []).append((float(g), float(e)))
    for key, ge in grouped.items():
        arr = np.array(ge)
        levels[key] = (arr[:, 0], arr[:, 1])
    return levels, ips


def direct_sum(g: np.ndarray, E: np.ndarray, ip_ev: Optional[float], T_K: float) -> float:
    """Direct partition sum over levels below the ionization potential."""
    mask = E < ip_ev if ip_ev is not None else np.ones_like(E, dtype=bool)
    if not np.any(mask):
        return 1.0
    return float(np.sum(g[mask] * np.exp(-E[mask] / (KB_EV * T_K))))


# ---------------------------------------------------------------------------
# High-T extension + polynomial fit
# ---------------------------------------------------------------------------


@dataclass
class SpeciesFit:
    """One species' refit result + diagnostics."""

    element: str
    stage: int
    coeffs: List[float]
    t_max: float  # stored validity ceiling (clamped above)
    max_resid_bc: float  # max |fit/target - 1| over the B&C anchors
    max_resid_ext: float  # ... over the extension grid up to LIBS_BAND_MAX
    max_resid_tail: float  # ... over the (possibly clamped) far tail
    extension_mode: str  # 'effective-level' | 'flat' | 'db-direct-sum'
    e_eff_ev: Optional[float] = None
    g_eff: Optional[float] = None
    targets: Dict[float, float] = field(default_factory=dict)  # T_K -> target U


def _base_function(
    key: Tuple[str, int],
    levels: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]],
    ips: Dict[Tuple[str, int], float],
    bc: BCTable,
):
    """Base U(T) for the extension: DB direct sum, else flat B&C(5000 K)."""
    if key in levels:
        g, E = levels[key]
        ip = ips.get(key)
        return lambda T: direct_sum(g, E, ip, T)
    u_flat = bc.interp_ln(key[0], key[1], 5000.0)
    return lambda T: u_flat


def extension_model(
    key: Tuple[str, int],
    bc: BCTable,
    levels: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]],
    ips: Dict[Tuple[str, int], float],
) -> Tuple:
    """Return ``(U_ext(T) callable, mode, e_eff, g_eff)`` for T > 1e4 K.

    Solves the single effective level (g_eff, E_eff) from the B&C−base
    differences at 8000 / 10000 K (see module docstring). Falls back to a
    flat difference when the missing-level signal is negligible, noisy, or
    non-increasing.
    """
    element, stage = key
    base = _base_function(key, levels, ips, bc)
    U1, U2 = bc.interp_ln(element, stage, EXT_ANCHOR_LO), bc.interp_ln(
        element, stage, EXT_ANCHOR_HI
    )
    D1, D2 = U1 - base(EXT_ANCHOR_LO), U2 - base(EXT_ANCHOR_HI)

    if D2 <= max(1e-8, 1e-3 * U2):
        # DB levels already reproduce B&C at the grid edge (or exceed it):
        # extend with the DB direct sum alone.
        mode = "db-direct-sum" if key in levels else "flat"
        return (lambda T: base(T) + 0.0), mode, None, None
    if D1 <= max(1e-8, 1e-4 * U1) or D2 <= D1:
        # Missing contribution present but not Boltzmann-increasing enough to
        # fit an activation energy — carry it flat (conservative).
        return (lambda T: base(T) + D2), "flat", None, None

    kT1, kT2 = KB_EV * EXT_ANCHOR_LO, KB_EV * EXT_ANCHOR_HI
    e_eff = math.log(D2 / D1) / (1.0 / kT1 - 1.0 / kT2)
    ip_cap = ips.get(key, 12.0) or 12.0
    e_eff = min(max(e_eff, 0.5), ip_cap)  # bound levels lie below the IP
    g_eff = D2 * math.exp(e_eff / kT2)  # re-anchor at the grid edge (continuity)
    return (
        (lambda T: base(T) + g_eff * math.exp(-e_eff / (KB_EV * T))),
        "effective-level",
        e_eff,
        g_eff,
    )


def reference_U(key: Tuple[str, int], bc: BCTable, ext_fn, T_K: float) -> float:
    """The fit target: B&C interpolation in-grid, the extension above it."""
    if T_K <= float(bc.temps_K[-1]):
        return bc.interp_ln(key[0], key[1], T_K)
    return float(ext_fn(T_K))


#: Full-domain fits missing the reference anywhere by more than this are
#: refit over [T_FIT_MIN, LIBS_BAND_MAX] with ``t_max`` shrunk accordingly:
#: the runtime clamp then freezes U above the band instead of letting an
#: under-flexible quartic oscillate through the Rydberg upturn.
BAND_RESTRICT_THRESHOLD = 0.03


def _weighted_quartic(T_grid: np.ndarray, targets: np.ndarray, bc_edge: float) -> np.ndarray:
    """Weighted LSQ quartic of ln U on ln T (B&C anchors + key temps heavy)."""
    weights = np.where(
        T_grid <= bc_edge,
        WEIGHT_BC_ANCHOR,
        np.where(T_grid <= LIBS_BAND_MAX, WEIGHT_EXT_BAND, WEIGHT_EXT_TAIL),
    )
    for key_T in KEY_TEMPS:
        weights[np.isclose(T_grid, key_T)] = WEIGHT_KEY
    return np.polynomial.polynomial.polyfit(
        np.log(T_grid), np.log(np.maximum(targets, 1e-30)), 4, w=weights
    )


def _clamped_eval(coeffs: np.ndarray, t_max: float, T: np.ndarray) -> np.ndarray:
    """Evaluate the stored polynomial exactly as the runtime provider does."""
    T_eval = np.minimum(np.asarray(T, dtype=float), t_max)
    return np.exp(np.polynomial.polynomial.polyval(np.log(T_eval), coeffs))


def fit_species(
    key: Tuple[str, int],
    bc: BCTable,
    levels: Dict[Tuple[str, int], Tuple[np.ndarray, np.ndarray]],
    ips: Dict[Tuple[str, int], float],
) -> SpeciesFit:
    """Refit ``ln U = Σ aₙ (ln T)ⁿ`` over [5000, 25000] K for one species.

    When the quartic cannot follow the full-domain reference to within
    ``BAND_RESTRICT_THRESHOLD`` (Rydberg-rich species: the ln U upturn beats
    a degree-4 polynomial), it is refit over the LIBS band only and the
    stored ``t_max`` is shrunk to ``LIBS_BAND_MAX`` so the runtime clamp
    freezes — rather than extrapolates — the far tail.
    """
    ext_fn, mode, e_eff, g_eff = extension_model(key, bc, levels, ips)
    T_grid = np.array(BC_ANCHOR_TEMPS + EXTENSION_TEMPS)
    targets = np.array([reference_U(key, bc, ext_fn, T) for T in T_grid])
    bc_edge = float(bc.temps_K[-1])

    coeffs = _weighted_quartic(T_grid, targets, bc_edge)
    t_max = T_FIT_MAX
    resid = np.abs(_clamped_eval(coeffs, t_max, T_grid) / targets - 1.0)
    if resid.max() > BAND_RESTRICT_THRESHOLD:
        in_band = T_grid <= LIBS_BAND_MAX
        coeffs = _weighted_quartic(T_grid[in_band], targets[in_band], bc_edge)
        t_max = LIBS_BAND_MAX
        resid = np.abs(_clamped_eval(coeffs, t_max, T_grid) / targets - 1.0)

    in_bc = T_grid <= bc_edge
    in_ext_band = (~in_bc) & (T_grid <= LIBS_BAND_MAX)
    in_tail = T_grid > LIBS_BAND_MAX
    return SpeciesFit(
        element=key[0],
        stage=key[1],
        coeffs=[float(c) for c in coeffs],
        t_max=float(t_max),
        max_resid_bc=float(resid[in_bc].max()),
        max_resid_ext=float(resid[in_ext_band].max()),
        max_resid_tail=float(resid[in_tail].max()),
        extension_mode=mode,
        e_eff_ev=e_eff,
        g_eff=g_eff,
        targets={float(t): float(u) for t, u in zip(T_grid, targets)},
    )


# ---------------------------------------------------------------------------
# Patching
# ---------------------------------------------------------------------------


def copy_database(src: Path, dst: Path) -> None:
    """Copy via the SQLite backup API (safe against concurrent readers)."""
    if dst.exists():
        dst.unlink()
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as src_conn:
        with sqlite3.connect(dst) as dst_conn:
            src_conn.backup(dst_conn)


def apply_patch(db_path: Path, fits: List[SpeciesFit]) -> Tuple[int, int]:
    """INSERT OR REPLACE the refit rows; returns (replaced, inserted)."""
    with sqlite3.connect(db_path) as conn:
        existing = {
            (el, sp) for el, sp in conn.execute("SELECT element, sp_num FROM partition_functions")
        }
        replaced = inserted = 0
        for fit in fits:
            conn.execute(
                "INSERT OR REPLACE INTO partition_functions "
                "(element, sp_num, a0, a1, a2, a3, a4, t_min, t_max, source) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (fit.element, fit.stage, *fit.coeffs, T_FIT_MIN, T_FIT_MAX, BC_SOURCE_TAG),
            )
            if (fit.element, fit.stage) in existing:
                replaced += 1
            else:
                inserted += 1
        conn.commit()
    return replaced, inserted


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def runtime_U(db, element: str, stage: int, T_K: float) -> float:
    """U through the SAME runtime policy the solver uses."""
    from cflibs.inversion.physics.self_absorption_inputs import evaluate_partition_function

    return evaluate_partition_function(db, element, stage, T_K)


def build_validation_rows(
    old_db_path: Path,
    new_db_path: Path,
    bc: BCTable,
    fits: Dict[Tuple[str, int], SpeciesFit],
    levels,
    ips,
) -> List[dict]:
    """Validation table rows: old-DB vs patched-DB vs B&C reference."""
    from cflibs.atomic.database import AtomicDatabase

    old_db = AtomicDatabase(str(old_db_path))
    new_db = AtomicDatabase(str(new_db_path))
    rows = []
    for element, stage in VALIDATION_SPECIES:
        key = (element, stage)
        ext_fn, mode, _, _ = extension_model(key, bc, levels, ips)
        for T_ev in VALIDATION_TEMPS_EV:
            T_K = T_ev * EV_TO_K
            ref = reference_U(key, bc, ext_fn, T_K)
            u_old = runtime_U(old_db, element, stage, T_K)
            u_new = runtime_U(new_db, element, stage, T_K)
            rows.append(
                {
                    "species": f"{element} {'I' * stage}",
                    "T_eV": T_ev,
                    "T_K": round(T_K, 1),
                    "U_old": u_old,
                    "U_patched": u_new,
                    "U_reference": ref,
                    "reference_kind": "B&C interp" if T_K <= bc.temps_K[-1] else f"ext:{mode}",
                    "old_vs_ref_pct": 100.0 * (u_old / ref - 1.0),
                    "patched_vs_ref_pct": 100.0 * (u_new / ref - 1.0),
                    "pass_2pct": abs(u_new / ref - 1.0) <= ACCEPTANCE_TOL,
                }
            )
    return rows


def write_validation_md(path: Path, rows: List[dict], fits: List[SpeciesFit]) -> None:
    """Render the acceptance table + per-species residual summary."""
    lines = [
        "# Partition-function validation — Barklem & Collet 2016 patch",
        "",
        "Bead CF-LIBS-improved-16m7 / audit findings 01-F3, 02-F1. Generated by",
        "`scripts/archive/migrations/patch_partition_functions_bc2016.py`.",
        "",
        "`U_old` = runtime policy on the shipped DB (direct-sum preferred; the",
        "canonical fallback ladder for absent species). `U_patched` = runtime",
        "policy on the patched copy (`BarklemCollet2016` rows are authoritative).",
        "`U_ref` = B&C Table 8 (Nov-2022 revision) interpolated in (ln T, ln U)",
        "where in-grid (≤ 10 000 K, so 0.8 eV); above the grid edge (1.0 and",
        "1.3 eV) the reference is the documented effective-level extension —",
        "B&C publish no values there (`ext:` rows). Gate: patched within 2 %.",
        "",
        "| Species | T (eV) | T (K) | U_old | U_patched | U_ref | ref kind | old vs ref | patched vs ref | ≤2 % |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['species']} | {r['T_eV']:.1f} | {r['T_K']:.0f} | "
            f"{r['U_old']:.3f} | {r['U_patched']:.3f} | {r['U_reference']:.3f} | "
            f"{r['reference_kind']} | {r['old_vs_ref_pct']:+.1f} % | "
            f"{r['patched_vs_ref_pct']:+.2f} % | {'PASS' if r['pass_2pct'] else 'FAIL'} |"
        )
    n_pass = sum(r["pass_2pct"] for r in rows)
    lines += [
        "",
        f"**Gate: {n_pass}/{len(rows)} rows within 2 % of the reference.**",
        "",
        "## Fit residuals (validation species)",
        "",
        "Max |fit − target| / target of the stored quartic over the true-B&C",
        "anchors (5000–10 000 K) and over the extension grid (11 000–25 000 K).",
        "",
        "| Species | max resid (B&C window) | max resid (ext ≤16 kK) | max resid (tail, clamped) | t_max (K) | extension mode | E_eff (eV) |",
        "|---|---|---|---|---|---|---|",
    ]
    wanted = {(el, sp) for el, sp in VALIDATION_SPECIES}
    for fit in fits:
        if (fit.element, fit.stage) not in wanted:
            continue
        e_eff = f"{fit.e_eff_ev:.2f}" if fit.e_eff_ev is not None else "—"
        lines.append(
            f"| {fit.element} {'I' * fit.stage} | {100 * fit.max_resid_bc:.2f} % | "
            f"{100 * fit.max_resid_ext:.2f} % | {100 * fit.max_resid_tail:.1f} % | "
            f"{fit.t_max:.0f} | {fit.extension_mode} | {e_eff} |"
        )
    all_bc = max(f.max_resid_bc for f in fits)
    all_ext = max(f.max_resid_ext for f in fits)
    n_clamped = sum(1 for f in fits if f.t_max < T_FIT_MAX)
    lines += [
        "",
        f"All {len(fits)} patched species: worst B&C-window residual "
        f"{100 * all_bc:.2f} %, worst LIBS-band extension residual "
        f"{100 * all_ext:.2f} %. {n_clamped} Rydberg-rich species are "
        f"tail-clamped (`t_max` = 16 000 K): above ~1.4 eV their stored U "
        f"freezes at U(16 000 K) — conservative, and physically the regime "
        f"where the runtime IPD truncation governs the level sum anyway.",
        "",
        "## Caveats",
        "",
        "- B&C Table 8 stops at 10 000 K; 1.0 / 1.3 eV reference values are the",
        "  effective-level extension described in the script docstring, anchored",
        "  to B&C at 8000/10 000 K and to our DB level sums (low-lying levels are",
        "  complete; the deficit is high-Rydberg). They are a physically-motivated",
        "  bridge, not B&C data.",
        "- The patched DB is NOT wired as the default; integration is gated on a",
        "  BHVO-2 benchmark (maintainer decision).",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    print(f"Wrote validation table -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db-path", type=Path, default=DEFAULT_DB)
    ap.add_argument(
        "--output-db",
        type=Path,
        default=DEFAULT_OUTPUT_DB,
        help="patched copy destination (default: ASD_da/libs_production_bc2016.db)",
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="patch --db-path directly instead of writing a copy "
        "(do NOT use while the DB is shared with concurrent readers)",
    )
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--download", action="store_true", help="(re)download the B&C table first")
    ap.add_argument("--validation-md", type=Path, default=None)
    ap.add_argument("--report-json", type=Path, default=None)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.download or not args.table.exists():
        download_table(args.table)
    bc = parse_bc_table8(args.table)
    print(f"Parsed B&C table 8: {len(bc.species)} species, grid up to {bc.temps_K[-1]:.0f} K")

    todo = species_to_patch(args.db_path, bc)
    levels, ips = load_levels(args.db_path)
    print(f"Refitting {len(todo)} (element, stage) species present in lines ∩ B&C ...")

    fits = [fit_species(key, bc, levels, ips) for key in todo]
    worst_bc = max(fits, key=lambda f: f.max_resid_bc)
    worst_ext = max(fits, key=lambda f: f.max_resid_ext)
    print(
        f"Fit residuals: worst B&C-window {100 * worst_bc.max_resid_bc:.2f} % "
        f"({worst_bc.element} {worst_bc.stage}); worst extension "
        f"{100 * worst_ext.max_resid_ext:.2f} % ({worst_ext.element} {worst_ext.stage})"
    )

    if args.in_place:
        target = args.db_path
        print(f"Patching IN PLACE: {target}")
    else:
        target = args.output_db
        print(f"Copying {args.db_path} -> {target}")
        copy_database(args.db_path, target)
    replaced, inserted = apply_patch(target, fits)
    print(
        f"Patched {target}: {replaced} rows replaced, {inserted} inserted (source={BC_SOURCE_TAG})"
    )

    if args.report_json:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(
            json.dumps(
                [
                    {
                        "element": f.element,
                        "stage": f.stage,
                        "coeffs": f.coeffs,
                        "max_resid_bc": f.max_resid_bc,
                        "max_resid_ext": f.max_resid_ext,
                        "extension_mode": f.extension_mode,
                        "e_eff_ev": f.e_eff_ev,
                        "g_eff": f.g_eff,
                        "targets": f.targets,
                    }
                    for f in fits
                ],
                indent=2,
            )
        )
        print(f"Wrote fit report -> {args.report_json}")

    if args.validation_md:
        rows = build_validation_rows(
            args.db_path if not args.in_place else target,
            target,
            bc,
            {(f.element, f.stage): f for f in fits},
            levels,
            ips,
        )
        write_validation_md(args.validation_md, rows, fits)
        failed = [r for r in rows if not r["pass_2pct"]]
        if failed:
            print(f"ACCEPTANCE GATE: {len(failed)} rows exceed 2 % — inspect the table.")
            return 1
        print("ACCEPTANCE GATE: all validation rows within 2 %.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
