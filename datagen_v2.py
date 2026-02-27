# datagen_v2.py
# --- COLAB SETUP ---
# !pip install requests-cache ASDCache

import sqlite3
import pandas as pd
import requests_cache
import re
import sys
import os
import io

# --- CONFIGURATION ---
DB_NAME = "libs_production.db"

# ULTRAFAST FILTER SETTINGS
# Prune physics that don't exist in <10us cooling plasmas
MAX_IONIZATION_STAGE = 2      # Keep only I and II (Neutrals & Singly Ionized)
MAX_UPPER_ENERGY_EV = 12.0    # Drop levels > 12 eV (unlikely to be populated)
MIN_RELATIVE_INTENSITY = 50   # Drop extremely weak lines

ALL_ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb",
    "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm",
    "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb",
    "Bi", "Th", "U"
]
# Only fetch stages I, II, III initially, then filter later
STAGES = ["I", "II", "III"] 

# Cache for Levels (Long expiry as NIST levels rarely change)
levels_session = requests_cache.CachedSession('nist_levels_cache', expire_after=None)
ie_session = requests_cache.CachedSession('nist_ie_cache', expire_after=2592000)

# --- IMPORT ASDCACHE ---
# Ensure ASDCache is available (standard logic from your original file)
possible_paths = ['ASDCache/src', 'src', 'antoinetue/asdcache/ASDCache-7c5d709e6c655311993616700e8a23d7e4cfb1fb/src']
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
        def fetch(self, *args, **kwargs): return pd.DataFrame()

CM_TO_EV = 1.23984193e-4 

def fetch_ionization_potential(element, stage_roman):
    """Scrapes the NIST IE database. Critical for Saha-Eggert calculations."""
    url = "https://physics.nist.gov/cgi-bin/ASD/ie.pl"
    params = {'spectra': f"{element} {stage_roman}", 'units': 1, 'format': 3, 'submit': 'Retrieve Data'}
    try:
        response = ie_session.get(url, params=params)
        for line in response.text.splitlines():
            if line.strip().startswith(element):
                parts = line.split()
                for part in parts:
                    clean = re.sub(r'[()\[\]]', '', part)
                    try:
                        val = float(clean)
                        if val > 0 and val < 5000: return val
                    except ValueError: continue
        return None
    except: return None

def fetch_energy_levels(element, stage_roman):
    """Scrapes the NIST Atomic Levels form for Partition Functions Z(T)."""
    url = "https://physics.nist.gov/cgi-bin/ASD/energy1.pl"
    query = f"{element} {stage_roman}"
    params = {
        'spectrum': query, 'units': 1, 'format': 3, 'multiplet_ordered': 0,
        'conf_out': 'off', 'term_out': 'off', 'level_out': 'on',
        'unc_out': 0, 'j_out': 'on', 'g_out': 'on', 'land_out': 'off',
        'submit': 'Retrieve Data'
    }
    
    try:
        response = levels_session.get(url, params=params)
        lines = response.text.splitlines()
        data = []
        for line in lines:
            if "Level" in line and "eV" in line: continue
            parts = re.split(r'\s+', line.strip())
            if len(parts) < 3: continue
            try:
                clean_parts = [re.sub(r'[\[\]\(\)\?]', '', p) for p in parts]
                energy = None
                for p in reversed(clean_parts):
                    try: 
                        energy = float(p)
                        break
                    except: continue
                if energy is None: continue
                g = None
                for p in clean_parts:
                    if p.isdigit():
                        val = int(p)
                        if val > 0 and val < 200: g = val
                if g and energy >= 0:
                    data.append((g, energy))
            except: continue
        return data 
    except Exception as e:
        print(f" [Levels Error: {e}]", end="")
        return []

def build_production_db():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)
    
    conn = sqlite3.connect(DB_NAME)
    
    # 1. SPECTRA TABLE
    conn.execute('''
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
            UNIQUE(element, sp_num, wavelength_nm, ek_ev)
        )
    ''')

    # 2. PHYSICS TABLE
    conn.execute('''
        CREATE TABLE IF NOT EXISTS species_physics (
            element TEXT,
            sp_num INTEGER,
            ip_ev REAL,
            PRIMARY KEY (element, sp_num)
        )
    ''')

    # 3. ENERGY LEVELS TABLE
    conn.execute('''
        CREATE TABLE IF NOT EXISTS energy_levels (
            element TEXT,
            sp_num INTEGER,
            g_level INTEGER,
            energy_ev REAL
        )
    ''')
    conn.execute('CREATE INDEX IF NOT EXISTS idx_levels ON energy_levels(element, sp_num)')
    conn.execute("CREATE INDEX IF NOT EXISTS idx_main ON lines(element, sp_num, wavelength_nm)")
    
    nist = SpectraCache()
    total_lines = 0
    
    print(f"--- STARTING ULTRAFAST PRODUCTION HARVEST ---")
    print(f"Filters: Max Stage={MAX_IONIZATION_STAGE} | Max Upper Energy={MAX_UPPER_ENERGY_EV} eV")
    
    for el in ALL_ELEMENTS:
        print(f"\nProcessing {el}...", end=" ")
        
        for stage in STAGES:
            sp_int = {"I":1, "II":2, "III":3, "IV":4}.get(stage, 0)
            if sp_int == 0: continue
            
            # --- A. PHYSICS DATA ---
            # Even if we filter lines, we need IP and Levels for stages I, II, III 
            # (Stage III needed for Saha balance of Stage II)
            ip = fetch_ionization_potential(el, stage)
            if ip:
                conn.execute("INSERT OR REPLACE INTO species_physics VALUES (?,?,?)", (el, sp_int, ip))
            
            levels = fetch_energy_levels(el, stage)
            if levels:
                rows = [(el, sp_int, g, en) for (g, en) in levels]
                conn.executemany("INSERT INTO energy_levels VALUES (?,?,?,?)", rows)
            
            # --- B. SPECTRAL LINES (Applied Filters) ---
            if sp_int > MAX_IONIZATION_STAGE: 
                continue # Skip fetching lines for high stages

            query = f"{el} {stage}"
            try:
                df = nist.fetch(query, wl_range=(190, 950)) # Mechelle Range
                if df.empty: continue

                # Basic Cleaning
                mask = df['obs_wl_air(nm)'].notna() & df['Aki(s^-1)'].notna() & df['Ek(cm-1)'].notna()
                clean = df[mask].copy()
                if clean.empty: continue
                
                # Conversion & Formatting
                sql_df = pd.DataFrame({
                    'element': el,
                    'sp_num': sp_int,
                    'wavelength_nm': clean['obs_wl_air(nm)'],
                    'aki': clean['Aki(s^-1)'],
                    'ei_ev': clean['Ei(cm-1)'] * CM_TO_EV,
                    'ek_ev': clean['Ek(cm-1)'] * CM_TO_EV,
                    'gi': clean['g_i'],
                    'gk': clean['g_k'],
                    'rel_int': pd.to_numeric(clean['intens'], errors='coerce').fillna(0)
                })

                # --- ULTRAFAST FILTERS ---
                # 1. Drop high energy levels (Cold plasma tail has no population here)
                sql_df = sql_df[sql_df['ek_ev'] <= MAX_UPPER_ENERGY_EV]
                
                # 2. Drop very weak lines (Noise floor)
                sql_df = sql_df[sql_df['rel_int'] >= MIN_RELATIVE_INTENSITY]
                
                # Deduplicate
                sql_df = sql_df.drop_duplicates(subset=['wavelength_nm', 'ek_ev'])
                
                if not sql_df.empty:
                    sql_df.to_sql('lines', conn, if_exists='append', index=False)
                    total_lines += len(sql_df)
                    print(f"[{stage}: {len(sql_df)}]", end="")
                
            except Exception as e:
                pass 
                
        conn.commit()

    conn.close()
    print(f"\n\n--- HARVEST COMPLETE ---")
    print(f"Database: {DB_NAME}")
    print(f"Total Optimized Lines: {total_lines}")

if __name__ == "__main__":
    build_production_db()
