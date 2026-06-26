# datagen_v2.py
# --- COLAB SETUP ---
# !pip install requests-cache ASDCache

import sqlite3
import pandas as pd
import requests_cache
import re
import sys
import os
import argparse

# --- CONFIGURATION ---
DEFAULT_DB_NAME = "libs_production.db"

# ULTRAFAST FILTER SETTINGS
# Prune physics that don't exist in <10us cooling plasmas
DEFAULT_MAX_IONIZATION_STAGE = 2  # Keep only I and II (Neutrals & Singly Ionized)
DEFAULT_MAX_UPPER_ENERGY_EV = 12.0  # Drop levels > 12 eV (unlikely to be populated)
DEFAULT_MIN_RELATIVE_INTENSITY = 50  # Drop extremely weak lines

ALL_ELEMENTS = [
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Th",
    "U",
]
# Only fetch stages I, II, III initially, then filter later
STAGES = ["I", "II", "III"]

# Cache for Levels (Long expiry as NIST levels rarely change)
levels_session = requests_cache.CachedSession("nist_levels_cache", expire_after=None)
ie_session = requests_cache.CachedSession("nist_ie_cache", expire_after=2592000)

# --- IMPORT ASDCACHE ---
# Ensure ASDCache is available (standard logic from your original file)
possible_paths = [
    "ASDCache/src",
    "src",
    "antoinetue/asdcache/ASDCache-7c5d709e6c655311993616700e8a23d7e4cfb1fb/src",
]
for path in possible_paths:
    if os.path.exists(path):
        sys.path.append(os.path.abspath(path))
        break

try:
    from ASDCache import SpectraCache
except ImportError:
    print("CRITICAL: ASDCache not found. Please upload the folder or pip install.")

    # Fallback for dev environments without the lib
    class SpectraCache:
        def fetch(self, *args, **kwargs):
            return pd.DataFrame()


CM_TO_EV = 1.23984193e-4


def fetch_ionization_potential(element, stage_roman):
    """Scrapes the NIST IE database. Critical for Saha-Eggert calculations."""
    url = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"
    params = {
        "spectra": f"{element} {stage_roman}",
        "units": 1,
        "format": 3,
        "submit": "Retrieve Data",
    }
    try:
        response = ie_session.get(url, params=params)
        # NIST format=3 is tab-delimited with QUOTED fields, e.g.
        #   Prefix\tIonization Energy (eV)\tSuffix
        #   ""\t"7.9024681"\t""
        # The data line starts with an (empty) quoted prefix, NOT the element,
        # so the old ``startswith(element)`` never matched -> always None. Parse
        # every quote-stripped tab field and take the first plausible eV value.
        for line in response.text.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("prefix") or line.startswith(("Notes", "(")):
                continue
            for part in line.split("\t"):
                clean = re.sub(r'[()\[\]"]', "", part).strip()
                try:
                    val = float(clean)
                    if 0 < val < 5000:
                        return val
                except ValueError:
                    continue
        return None
    except Exception:
        return None


_LEADING_FLOAT_RE = re.compile(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?")


def _g_from_j_field(j_field):
    """Sum (2J+1) over a possibly multi-valued NIST J field, e.g. '7/2,9/2'.

    NIST lists unresolved fine-structure groups with several J values for one
    energy. Each J in the group contributes its own (2J+1) — exactly the
    convention NIST's own g column uses for such rows ('7/2,9/2' -> g=18).
    This is the same parse Barklem & Collet fixed in their Nov-2022 revision
    (their original code read '7/2' as 7.0, doubling those weights).
    """
    total = 0.0
    for token in j_field.replace("---", "").replace("-", ",").split(","):
        token = token.strip()
        if not token:
            continue
        if "/" in token:
            num, den = token.split("/", 1)
            j_val = float(num) / float(den)
        else:
            j_val = float(token)
        total += 2.0 * j_val + 1.0
    return int(round(total)) if total > 0 else None


def _parse_levels_tsv(text):
    """Parse NIST ASD energy1.pl tab-delimited output into [(g, energy_eV)].

    Column-aware replacement for the historical token-soup heuristic, which
    silently dropped or mangled most high-Rydberg levels (the root cause of
    the 5–40 % partition-function deficits documented in audit finding 01-F3:
    Ca I had 76 levels in the DB vs ~370 bound levels in NIST). Format 3 wraps
    the J and Level fields in double quotes, which the old parser never
    stripped — ``float('"6.03270"')`` fails, so the 'last parseable float in
    the row' fell back to the (unquoted, integer) g column, and the
    ``isdigit()`` g-hunt then produced garbage or nothing for those rows.

    Rules:
    - locate the J / g / Level columns from the header (Prefix/Suffix columns
      carry the ``[``/``]`` interpolation markers, NOT the Level field);
    - skip separator / ionization-limit rows (``J == '---'`` or no energy);
    - take g from the g column when present; otherwise derive it as the sum
      of (2J+1) over the (possibly multi-valued) J field;
    - parse the energy as the leading float of the Level field (annotations
      such as ``+x``, ``?``, ``[...]`` are ignored).
    """
    import csv
    import io

    reader = csv.reader(io.StringIO(text), delimiter="\t")
    header = next(reader, None)
    if not header or not any("Level" in (col or "") for col in header):
        # HTML error page (e.g. 'Invalid Column Setting') or empty payload.
        print(" [Levels Error: unexpected NIST response format]", end="")
        return []
    col_index = {}
    for i, col in enumerate(header):
        name = (col or "").strip().strip('"')
        if name.startswith("Level"):
            col_index["level"] = i
        elif name == "J":
            col_index["j"] = i
        elif name == "g":
            col_index["g"] = i
    if "level" not in col_index:
        return []

    data = []
    for row in reader:
        fields = [(f or "").strip().strip('"').strip() for f in row]
        if len(fields) <= col_index["level"]:
            continue
        j_field = fields[col_index["j"]] if "j" in col_index else ""
        if j_field == "---":  # separator / ionization-limit row
            continue
        match = _LEADING_FLOAT_RE.match(fields[col_index["level"]].lstrip("[("))
        if not match:
            continue
        energy = float(match.group(0))
        g_field = fields[col_index["g"]] if "g" in col_index else ""
        if g_field.isdigit() and int(g_field) > 0:
            g = int(g_field)
        else:
            try:
                g = _g_from_j_field(j_field)
            except (ValueError, ZeroDivisionError):
                g = None
        if g and energy >= 0:
            data.append((g, energy))
    return data


def fetch_energy_levels(element, stage_roman):
    """Scrapes the NIST Atomic Levels form for Partition Functions Z(T).

    NOTE (bead CF-LIBS-improved-16m7): unchecked-checkbox parameters
    (``conf_out=off`` etc.) must be OMITTED, not sent — NIST ASD treats the
    mere presence of the parameter as a column request and rejects the
    historical combination with an 'Invalid Column Setting' HTML error page.
    """
    url = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"
    params = {
        "spectrum": f"{element} {stage_roman}",
        "units": 1,  # energies in eV
        "format": 3,  # tab-delimited
        "output": 0,  # all levels in one page
        "multiplet_ordered": 0,
        "level_out": "on",
        "j_out": "on",
        "g_out": "on",
        "submit": "Retrieve Data",
    }

    try:
        response = levels_session.get(url, params=params)
        return _parse_levels_tsv(response.text)
    except Exception as e:
        print(f" [Levels Error: {e}]", end="")
        return []


def build_production_db(
    db_path=DEFAULT_DB_NAME,
    elements=None,
    max_ionization_stage=DEFAULT_MAX_IONIZATION_STAGE,
    max_upper_energy_ev=DEFAULT_MAX_UPPER_ENERGY_EV,
    min_relative_intensity=DEFAULT_MIN_RELATIVE_INTENSITY,
):
    if os.path.exists(db_path):
        os.remove(db_path)

    conn = sqlite3.connect(db_path)

    # 1. SPECTRA TABLE
    conn.execute("""
        CREATE TABLE IF NOT EXISTS lines (
            id INTEGER PRIMARY KEY,
            element TEXT,
            sp_num INTEGER,
            wavelength_nm REAL,
            aki REAL,
            ei_ev REAL,
            ek_ev REAL,
            gi REAL,
            gk REAL,
            rel_int REAL,
            aki_uncertainty REAL,
            accuracy_grade TEXT,
            UNIQUE(element, sp_num, wavelength_nm, ek_ev)
        )
    """)

    # 2. PHYSICS TABLE
    conn.execute("""
        CREATE TABLE IF NOT EXISTS species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
    """)

    # 3. ENERGY LEVELS TABLE
    conn.execute("""
        CREATE TABLE IF NOT EXISTS energy_levels (
            element TEXT,
            sp_num INTEGER,
            g_level INTEGER,
            energy_ev REAL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_levels ON energy_levels(element, sp_num)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_main ON lines(element, sp_num, wavelength_nm)")

    nist = SpectraCache()
    total_lines = 0
    elements_to_process = list(elements) if elements else ALL_ELEMENTS

    print("--- STARTING ULTRAFAST PRODUCTION HARVEST ---")
    print(
        f"Filters: Max Stage={max_ionization_stage} | "
        f"Max Upper Energy={max_upper_energy_ev} eV | "
        f"Min Relative Intensity={min_relative_intensity}"
    )

    for el in elements_to_process:
        print(f"\nProcessing {el}...", end=" ")

        for stage in STAGES:
            sp_int = {"I": 1, "II": 2, "III": 3, "IV": 4}.get(stage, 0)
            if sp_int == 0:
                continue

            # --- A. PHYSICS DATA ---
            # Even if we filter lines, we need IP and Levels for stages I, II, III
            # (Stage III needed for Saha balance of Stage II)
            ip = fetch_ionization_potential(el, stage)
            if ip:
                conn.execute(
                    "INSERT OR REPLACE INTO species_physics VALUES (?,?,?)", (el, sp_int, ip)
                )

            levels = fetch_energy_levels(el, stage)
            if levels:
                rows = [(el, sp_int, g, en) for (g, en) in levels]
                conn.executemany("INSERT INTO energy_levels VALUES (?,?,?,?)", rows)

            # --- B. SPECTRAL LINES (Applied Filters) ---
            if sp_int > max_ionization_stage:
                continue  # Skip fetching lines for high stages

            query = f"{el} {stage}"
            try:
                df = nist.fetch(query, wl_range=(190, 950))  # Mechelle Range
                if df.empty:
                    continue

                # Basic Cleaning
                mask = (
                    df["obs_wl_air(nm)"].notna() & df["Aki(s^-1)"].notna() & df["Ek(cm-1)"].notna()
                )
                clean = df[mask].copy()
                if clean.empty:
                    continue

                # Conversion & Formatting
                aki_uncertainty = (
                    pd.to_numeric(clean["unc_Aki"], errors="coerce") / 100.0
                    if "unc_Aki" in clean
                    else pd.Series(pd.NA, index=clean.index, dtype="float64")
                )
                accuracy_grade = (
                    clean["acc"]
                    if "acc" in clean
                    else pd.Series(pd.NA, index=clean.index, dtype="object")
                )
                sql_df = pd.DataFrame(
                    {
                        "element": el,
                        "sp_num": sp_int,
                        "wavelength_nm": clean["obs_wl_air(nm)"],
                        "aki": clean["Aki(s^-1)"],
                        "ei_ev": clean["Ei(cm-1)"] * CM_TO_EV,
                        "ek_ev": clean["Ek(cm-1)"] * CM_TO_EV,
                        "gi": clean["g_i"],
                        "gk": clean["g_k"],
                        "rel_int": pd.to_numeric(clean["intens"], errors="coerce").fillna(0),
                        "aki_uncertainty": aki_uncertainty,
                        "accuracy_grade": accuracy_grade,
                    }
                )

                # --- ULTRAFAST FILTERS ---
                # 1. Drop high energy levels (Cold plasma tail has no population here)
                sql_df = sql_df[sql_df["ek_ev"] <= max_upper_energy_ev]

                # 2. Drop very weak lines (Noise floor)
                sql_df = sql_df[sql_df["rel_int"] >= min_relative_intensity]

                # Deduplicate
                sql_df = sql_df.drop_duplicates(subset=["wavelength_nm", "ek_ev"])

                if not sql_df.empty:
                    sql_df.to_sql("lines", conn, if_exists="append", index=False)
                    total_lines += len(sql_df)
                    print(f"[{stage}: {len(sql_df)}]", end="")

            except Exception:
                pass

        conn.commit()

    conn.close()
    print("\n\n--- HARVEST COMPLETE ---")
    print(f"Database: {db_path}")
    print(f"Total Optimized Lines: {total_lines}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate CF-LIBS production database from NIST.")
    parser.add_argument("--db-path", default=DEFAULT_DB_NAME, help="Output database path")
    parser.add_argument(
        "--elements",
        nargs="+",
        default=None,
        help="Specific element symbols to include (default: all supported elements)",
    )
    parser.add_argument(
        "--max-ionization-stage",
        type=int,
        default=DEFAULT_MAX_IONIZATION_STAGE,
        help="Maximum ionization stage to include in the lines table",
    )
    parser.add_argument(
        "--max-upper-energy-ev",
        type=float,
        default=DEFAULT_MAX_UPPER_ENERGY_EV,
        help="Maximum upper energy level to retain in eV",
    )
    parser.add_argument(
        "--min-relative-intensity",
        type=float,
        default=DEFAULT_MIN_RELATIVE_INTENSITY,
        help="Minimum relative intensity threshold for retained lines",
    )
    args = parser.parse_args()
    build_production_db(
        db_path=args.db_path,
        elements=args.elements,
        max_ionization_stage=args.max_ionization_stage,
        max_upper_energy_ev=args.max_upper_energy_ev,
        min_relative_intensity=args.min_relative_intensity,
    )
