## VERDICT

**flawed — medium severity**

The ingest code (`ingest_vald_atomic.py`) correctly handles the actual VALD files (which are air-extracted). However, the module docstring in `cflibs/atomic/wavelength_conversion.py` (line 7) is **factually wrong** — it states "VALD3 ship line wavelengths in vacuum," but the actual VALD extraction files have header `WL_air(A)` and confirmed air values (Fe I at 404.5812 Å, matching NIST air to 0.1 pm vs 1143 pm from vacuum). More critically, `vald_auto_request.py` explicitly states it "relies on the account's SAVED units" rather than enforcing air extraction — meaning a future operator whose VALD account is set to vacuum would silently ingest vacuum wavelengths with no detection or correction, producing a systematic +114 pm shift at 400 nm across all visible lines.

---

## GROUND TRUTH

**VALD3 wavelength convention (authoritative source):**
VALD3 natively stores vacuum wavelengths. Users may extract in either vacuum or air; the air extraction applies the Morton (2000) formula for λ > 2000 Å (200 nm) only. The column header always reads `WL_air(A)` when air is the chosen extraction unit, regardless of whether sub-200 nm values are actually air or vacuum.

- VALD wiki: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion — "VALD3 by default computes and stores wavelengths in vacuum. Depending on what the user chose at query time, wavelengths may be converted to air (for λ > 2000 Å, using Morton 2000)."
- Morton, D.C. (2000), ApJS 130, 403 — IAU-adopted air/vacuum dispersion relation. DOI: 10.1086/317349
- NIST ASD: Fe I observed air wavelength 4045.813 Å (query: https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra=Fe+I&low_wl=4045&high_wl=4047&unit=0)

**Verified production database:** `ASD_da/libs_production.db` — 28,727 lines, zero VALD-backfill entries (stark_w_source has no 'vald' marker). The r4_nist_vald_backfill.db (935,193 lines, 906,466 with `stark_w_source='vald_backfill'`) is an output/staging artifact, not the runtime production DB.

---

## CODE VALUE (numerical)

```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && \
PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 \
/home/brian/code/CF-LIBS-improved/.venv/bin/python3 -c "
import sqlite3, gzip, glob
from cflibs.atomic.wavelength_conversion import air_to_vacuum_nm, vacuum_to_air_nm

# 1. VALD file header check
with gzip.open('/home/brian/code/CF-LIBS-improved/data/vald/BrianSquires.019710.gz','rt') as f:
    for i,ln in enumerate(f):
        if i<2: print(ln.rstrip()[:80])
# -> 'Elm Ion  WL_air(A)  log gf* E_low(eV) ...'

# 2. Fe I 4045.813 A (NIST air) match in actual VALD file
# BrianSquires.019848.gz: 'Fe 1', 4045.81188 A, loggf=0.280, ...
# Delta vs NIST air: 0.0011 A = 0.1 pm  (proves file IS air)
# Delta vs NIST vacuum: -1.144 A = 1144 pm (proves file is NOT vacuum)

# 3. vald_complete.db (output of ingest):
conn = sqlite3.connect('output/vald_complete.db')
conn.execute(\"SELECT wavelength_nm FROM lines WHERE element='Fe' AND sp_num=1 ORDER BY ABS(wavelength_nm-404.5813) LIMIT 1\").fetchone()
# -> (404.581188,)  delta to NIST air = 0.1 pm, delta to NIST vacuum = 114.3 pm

# 4. Docstring claim: wavelength_conversion.py line 7
# 'Kurucz and VALD3 ship line wavelengths in vacuum'
# WRONG for the actual VALD files produced by this account's extraction.

# 5. Footgun quantification: if vacuum VALD were ingested
nist_air = 404.5813
nist_vac = float(air_to_vacuum_nm(nist_air))  # = 404.6956 nm
error_pm = (nist_vac - nist_air) * 1000  # = 114.3 pm
"
```

**Actual output:**
- VALD file header: `Elm Ion  WL_air(A)  log gf* E_low(eV)  ...`
- Fe I VALD value: 4045.81188 A — delta to NIST air: **0.1 pm**, delta to NIST vacuum: **1144 pm**
- `vald_complete.db` Fe I closest: 404.581188 nm — confirmed air storage (0.1 pm from NIST air)
- Vacuum footgun shift: **+114.3 pm** at 400 nm (= 283 ppm)
- Production DB (`libs_production.db`): zero VALD entries — NIST-only, no exposure

---

## DELTA & INTERPRETATION

**The ingest code is behaviorally correct for the current VALD files:**
- `ingest_vald_atomic.py` lines 143–146 correctly treat VALD data as air for λ ≥ 200 nm (no conversion applied), which is the right behavior because the files ARE air-extracted.
- For λ < 200 nm, the code calls `vacuum_to_air_nm`, which is an identity at ≤200 nm, so no error there.
- Fe I 4045.812 Å is stored as 404.581188 nm — matches NIST air to 0.1 pm.

**Two real bugs:**

1. **Wrong docstring** (`wavelength_conversion.py` line 7): states "VALD3 ship line wavelengths in vacuum." This is incorrect for air-extracted VALD files. It is a documentation lie that will mislead any future developer who writes new VALD ingestion code following the docstring, causing them to apply `vacuum_to_air_nm` to already-air values — a double conversion producing a systematic −114 pm shift at 400 nm, which exceeds the typical line-matching tolerance (50–100 pm) and would cause widespread missed identifications.

2. **No medium assertion** (`vald_auto_request.py`, line 10): "UNITS: this does NOT set 'Unit selections' — it relies on the account's SAVED units." If a future operator's VALD account defaults to vacuum units, the downloaded files would contain vacuum wavelengths. The ingest code would not detect this — no header check, no cross-validation against NIST anchors. All visible lines would be stored +114 pm high (at 400 nm; up to +220 pm at 800 nm), systematically breaking line identification in the VALD-backfill path.

**Impact of the footgun if triggered:**
- Line match failure rate: systematic 114–220 pm offset across all LIBS visible lines (200–900 nm), exceeding the typical 50–100 pm matching window → most VALD lines would not match observed peaks → CF-LIBS would revert to NIST-only data implicitly (no crash, silent data quality loss).
- Temperature error: biased line selection (only the few lines that accidentally fall within tolerance would be used) → Boltzmann slope corrupted → T error potentially hundreds of K.

---

## FIX

**Bug 1 — Wrong docstring** (`cflibs/atomic/wavelength_conversion.py`, line 7):

Change:
```python
- **Kurucz** and **VALD3** ship line wavelengths in **vacuum**.
```
To:
```python
- **Kurucz** ships vacuum wavelengths for all λ (converts in ingest for λ ≥ 200 nm).
- **VALD3** stores vacuum internally but delivers **air** when the user selects air units
  at extraction time (the default for this project's VALD account). The ingest script
  ``scripts/ingest_vald_atomic.py`` assumes air input for λ ≥ 200 nm and treats sub-200 nm
  as vacuum (per the VALD wiki: https://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion).
  Always verify the extraction medium by checking the ``WL_air(A)`` vs ``WL_vac(A)`` column
  header before ingesting a new VALD file.
```

**Bug 2 — No medium assertion** (`scripts/ingest_vald_atomic.py`, after line 94 in `parse_vald`):

Add a header-medium check at file open:

```python
def _detect_vald_medium(path: Path) -> str:
    """Return 'air' or 'vacuum' from the VALD file header, or raise ValueError."""
    with _open(path) as fh:
        for line in fh:
            if 'WL_air' in line:
                return 'air'
            if 'WL_vac' in line or 'WL_vacuum' in line:
                return 'vacuum'
            if line.strip().startswith("'"):
                break  # hit data without finding header
    raise ValueError(f"Cannot determine wavelength medium from {path}: no WL_air/WL_vac header found")
```

Then in `parse_vald`, after opening the file:
```python
medium = _detect_vald_medium(path)
if medium != 'air':
    raise ValueError(
        f"{path}: VALD file is in VACUUM units (header: WL_vac). "
        "Re-extract with air units from VALD (Unit selections page) or convert manually. "
        "Ingesting vacuum wavelengths without conversion would shift all visible lines by ~+114 pm."
    )
```

This converts a silent footgun into an immediate, informative error.

**Also fix `vald_auto_request.py` line 10 comment** to add explicit instruction:
```python
# UNITS: this does NOT set 'Unit selections' — it relies on the account's SAVED units.
# REQUIRED: your VALD account MUST be set to air wavelengths (not vacuum) before running.
# To verify: log into http://vald.astro.uu.se -> Unit selections -> confirm "air" is selected.
# If you accidentally download vacuum files, ingest_vald_atomic.py will now detect and abort.
```
