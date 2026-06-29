The paths check out: `ingest_vald_atomic.py` and `complete_atomic_db.py` exist, `data/vald/` exists, and `ingest_exomol_tio.py` is the new script to be created (Part B). Here is the assembled guide.

---

# HYBRID Execution Guide — Complete Atomic + Molecular Line DB (VALD atomic/7-molecules + ExoMol TiO)

**Goal:** Close the line-list gap so the DB has contiguous **atomic** coverage 100–1000 nm plus a complete **molecular** set (TiO from ExoMol + the 7 minor molecules CN/C₂/OH/CH/CO/NH/MgH from VALD). Two data sources, two ingesters, one final DB.

**Why hybrid:** TiO is 95–98% of every VALD red slice and slams into VALD's ~101k-record/request cap, so TiO is pulled **separately** from ExoMol's complete 'Toto' list (`ingest_exomol_tio.py`), while **everything else** (all atoms + the 7 minor molecules) comes from VALD's *Extract Element* mode with TiO explicitly excluded so the per-request count collapses to the non-TiO total.

---

## PART A — VALD3 "Extract Element": atomic + 7 minor molecules over 323.87–1000 nm (TiO EXCLUDED)

Contiguous atomic+molecular coverage **100.002–323.875 nm** is already on disk (`data/vald/`). Only the **323.87–1000 nm** red gap needs new requests.

### A.1 Why "Extract Element" (and NOT Extract All / Extract Stellar)

| Mode | What it returns | Verdict |
|---|---|---|
| **Extract All** | every species in the range — but TiO saturates it at ~101k records/req, so red chunks must be ~2–3 nm wide | ❌ TiO dominates; un-usable for the red gap |
| **Extract Stellar** | every species, **but applies a depth/line-strength threshold vs a model atmosphere → silently drops weak lines** | ❌ breaks completeness — **do not use** |
| **Extract Element** | **every line of the species you explicitly name, with NO line-strength cut** | ✅ species-selectable *and* strength-complete |

*Extract Element* is the only mode that is both species-selectable and strength-complete. Because we **name every species except TiO**, the per-request record count drops to just the non-TiO total (atomic + 7 molecules), letting each request span **20–30 nm** instead of ~2–3 nm.

### A.2 Species-list strategy (and the 100-species cap split)

VALD has a **~100-species hard cap per Extract-Element request**, and there is **no wildcard / "all atoms except TiO" toggle** — you must enumerate. The full LIBS set (H–U neutral + 1st + 2nd ion ≈ 276 species) + 7 molecules ≈ **283 species**, exceeding 100. So split the **species list** into 3 groups of ≤100, each run over the **same** wide wavelength range:

| Group | Contents | Syntax (one species per line) | ~count |
|---|---|---|---:|
| **G1** | H–U **neutral** + the **7 molecules** | `Fe 1`, `Ca 1`, … `U 1`, then `CN 1` `C2 1` `OH 1` `CH 1` `CO 1` `NH 1` `MgH 1` | ~99 |
| **G2** | H–U **singly ionized** | `Fe 2`, `Ca 2`, … `U 2` | ~92 |
| **G3** | H–U **doubly ionized** | `Fe 3`, `Ca 3`, … `U 3` | ~92 |

- VALD convention: `1` = neutral, `2` = singly ionized, `3` = doubly ionized. Molecules are charge `1` (neutral); token must match VALD naming exactly (`C2`, `MgH` — not `C_2`/`Mg H`).
- **OMIT `TiO 1` from every group** (it comes from ExoMol in Part B). You may drop any element/ion known to be irrelevant to trim further, but 99/92/92 already fits under 100.
- If the live form rejects a 100-species list, drop to **3 groups of ~90**.

**Net file dumps = (wavelength ranges) × 3 species-groups.** With the conservative 18-range plan that's **18 × 3 = 54 dumps**, but only **18 distinct wavelength ranges** — well under 20. The ingest dedups by `(species, ion, wl, E_low, E_up)`, so the 3 groups never collide, and overlap with the existing 100–324 nm slices is harmless.

### A.3 Exact form settings (must match the existing `data/vald/` slices)

| Setting | Value |
|---|---|
| Extraction mode | **Extract Element** |
| Format | **Long format** (4-line-per-transition block — ingest depends on it) |
| Wavelength unit | **Angstrom** |
| Energy unit | **eV** |
| Wavelength medium | **air** (VALD gives air ≥2000 Å; ingest air-converts <200 nm) |
| Isotopic / HFS structure | **HFS on** |
| Retrieval | **FTP** (large dumps; emailed FTP link, like `BrianSquires.0197xx.gz`) |
| All "require lines have a known … (rad/Stark/vdW damping, Landé, term designation)" toggles | **ALL OFF** — any "require known X" filter drops otherwise-complete lines |
| Microturbulence / line-strength threshold | **N/A for Extract Element** (no depth cut — this is the whole point) |

### A.4 Record cap, truncation, and the wide-request table

The ~101k-record cap is on **records returned per dump**, not wavelength width. With TiO excluded, the count collapses to the **non-TiO** total. Measured non-TiO density (atomic + 7 mol, parsed from saturated slices on disk): ~2,900 rec/nm near 324 nm, ~3,300 rec/nm at 500 nm, falling to ~960–1,700 rec/nm beyond 600 nm (e.g. 500–502 nm: total 101,519 → TiO 96,665 → **non-TiO only 4,854**).

**Truncation behavior:** if a single dump *would* exceed the cap, VALD does **not** reject — it **silently truncates** at the first ~101k lines (ordered by wavelength). **Detection:** after every dump, check whether the returned `WL_air(Å)` max reaches the requested upper edge; if it stops short, **split that wavelength range and re-request**. G1 (neutrals + molecules) is the binding group; G2/G3 (ionized, far fewer lines) never approach the cap. The table below is sized to **G1**.

**Recommended (conservative) wide-request table — closes 323.87–1000 nm.** Each row = one wavelength range, run for **all 3 species-groups**. `est non-TiO` is the G1-bounding count (max 85.6k, ~15k cap margin):

| req# | nm range | Ångström range | est non-TiO lines |
|---:|---|---|---:|
| 1 | 323.87–348 | 3238.7–3480 | 73,700 |
| 2 | 348–372 | 3480–3720 | 85,600 |
| 3 | 372–396 | 3720–3960 | 76,300 |
| 4 | 396–420 | 3960–4200 | 68,200 |
| 5 | 420–445 | 4200–4450 | 71,400 |
| 6 | 445–470 | 4450–4700 | 72,000 |
| 7 | 470–495 | 4700–4950 | 75,500 |
| 8 | 495–520 | 4950–5200 | 79,100 |
| 9 | 520–548 | 5200–5480 | 72,600 |
| 10 | 548–580 | 5480–5800 | 66,000 |
| 11 | 580–615 | 5800–6150 | 63,400 |
| 12 | 615–655 | 6150–6550 | 65,200 |
| 13 | 655–700 | 6550–7000 | 69,400 |
| 14 | 700–755 | 7000–7550 | 72,400 |
| 15 | 755–820 | 7550–8200 | 69,800 |
| 16 | 820–895 | 8200–8950 | 77,300 |
| 17 | 895–955 | 8950–9550 | 58,200 |
| 18 | 955–1000 | 9550–10000 | 43,300 |

**18 wavelength ranges**, ~1.26M non-TiO lines total, max chunk 85.6k (15.4k under cap) × 3 species-groups = **54 dumps**.

**Aggressive alternative (14 ranges, max 93.5k, ~7.5k margin)** if you want fewer ranges: edges at 323.87 / 354 / 380 / 408 / 438 / 470 / 500 / 530 / 570 / 620 / 675 / 740 / 820 / 905 / 1000 nm. Use only if confident in the density estimate; otherwise prefer the 18-range plan.

> **Estimate caveat:** counts interpolate measured density at ~22 probe points; real per-chunk counts can deviate ±10–15%. That margin is exactly why the conservative 18-chunk plan (max 85.6k) is recommended over the 14-chunk plan (max 93.5k). The truncation-detection check makes the plan robust regardless of whether VALD applies the cap per species-group or per range.

### A.5 Copy-pasteable Claude-for-Chrome prompt

Paste the following into Claude-for-Chrome with the VALD3 *Extract Element* page open (log in first):

```
You are driving the VALD3 web interface (vald.astro.uu.se) to download line lists via
"Extract Element" mode. We are filling the 323.87–1000 nm gap for a LIBS atomic+molecular
database. Follow EXACTLY — do not improvise modes or settings.

CRITICAL MODE: Use "EXTRACT ELEMENT" (the species-selectable mode). DO NOT use "Extract All"
(TiO saturates it) and DO NOT use "Extract Stellar" (it applies a line-strength/depth
threshold that silently drops weak lines and breaks completeness).

FIXED FORM SETTINGS (set these once and keep for every request):
  - Extraction mode:        Extract Element
  - Format:                 Long format
  - Wavelength unit:        Angstrom
  - Energy unit:            eV
  - Wavelength medium:      air
  - Isotopic/HFS structure: HFS ON
  - Retrieval method:       FTP (results emailed as an FTP .gz link)
  - Every "require lines have a known ... (radiative/Stark/vdW damping, Landé factor,
    term designation, etc.)" checkbox: ALL OFF / UNCHECKED. Any "require known X" filter
    drops otherwise-complete lines — we want completeness.
  - There is NO microturbulence / line-strength threshold in Extract Element — leave any
    such field at default; it does not apply.

WAVELENGTH FIELDS: For EVERY request you MUST enter BOTH the lower and upper wavelength
bounds (in Angstroms) into the two wavelength fields. Never leave one blank.

SPECIES LIST: Extract Element has a ~100-species cap and NO wildcard. Paste the explicit
species list (one species per line) into the species box. We run THREE species groups over
each wavelength range. NEVER include "TiO 1" in any group (TiO is sourced separately).

  GROUP G1 (neutral atoms + the 7 minor molecules):
    H 1, He 1, Li 1, Be 1, B 1, C 1, N 1, O 1, F 1, Ne 1, Na 1, Mg 1, Al 1, Si 1, P 1,
    S 1, Cl 1, Ar 1, K 1, Ca 1, Sc 1, Ti 1, V 1, Cr 1, Mn 1, Fe 1, Co 1, Ni 1, Cu 1,
    Zn 1, Ga 1, Ge 1, As 1, Se 1, Br 1, Kr 1, Rb 1, Sr 1, Y 1, Zr 1, Nb 1, Mo 1, Tc 1,
    Ru 1, Rh 1, Pd 1, Ag 1, Cd 1, In 1, Sn 1, Sb 1, Te 1, I 1, Xe 1, Cs 1, Ba 1, La 1,
    Ce 1, Pr 1, Nd 1, Pm 1, Sm 1, Eu 1, Gd 1, Tb 1, Dy 1, Ho 1, Er 1, Tm 1, Yb 1, Lu 1,
    Hf 1, Ta 1, W 1, Re 1, Os 1, Ir 1, Pt 1, Au 1, Hg 1, Tl 1, Pb 1, Bi 1, Po 1, At 1,
    Rn 1, Fr 1, Ra 1, Ac 1, Th 1, Pa 1, U 1,
    CN 1, C2 1, OH 1, CH 1, CO 1, NH 1, MgH 1
    (≈99 species — DO NOT add TiO)

  GROUP G2 (singly ionized atoms): same element list as G1 atoms but with ion "2":
    H 2, He 2, Li 2, ... U 2   (≈92 species)

  GROUP G3 (doubly ionized atoms): same element list with ion "3":
    H 3, He 3, Li 3, ... U 3   (≈92 species)

  If the form rejects a ~100-species list as too long, split each group into two halves of
  ~50 and run both halves over the same wavelength range.

WAVELENGTH RANGES (run ALL THREE species groups for each — enter BOTH Angstrom bounds):
   1) 3238.7 – 3480     2) 3480 – 3720      3) 3720 – 3960      4) 3960 – 4200
   5) 4200 – 4450       6) 4450 – 4700      7) 4700 – 4950      8) 4950 – 5200
   9) 5200 – 5480      10) 5480 – 5800     11) 5800 – 6150     12) 6150 – 6550
  13) 6550 – 7000      14) 7000 – 7550     15) 7550 – 8200     16) 8200 – 8950
  17) 8950 – 9550      18) 9550 – 10000

PROCEDURE per (range × group):
  1. Confirm mode = Extract Element and all FIXED SETTINGS above.
  2. Enter the LOWER and UPPER Angstrom bounds in the two wavelength fields.
  3. Paste the species list for the current group.
  4. Submit. Record the request number, the FTP link / filename, and the species group.
  5. After all 54 submissions (18 ranges × 3 groups), report the full list of FTP filenames.

TRUNCATION CHECK (do this when results arrive): VALD silently truncates a dump at ~101k
records. For each returned file, compare the MAX air wavelength in the file against the
requested upper bound. If the file's max wavelength stops short of the requested upper edge,
the dump was truncated — flag that (range × group) so it can be split into two narrower
wavelength sub-ranges and re-requested.

Proceed one request at a time, confirm each form state before submitting, and keep a running
table of {request#, range, group, ftp_filename}.
```

---

## PART B — ExoMol TiO 'Toto' bulk download + ingest into `molecular_lines`

### B.1 Exact download mechanism

**Tooling decision: use `radis.api.exomolapi.MdbExomol` directly — NOT top-level `radis`, NOT exojax.**

- Top-level `import radis` (and anything reaching `radis.misc.arrays` / `radis.config`) **crashes** in this venv: `AttributeError: module 'coverage' has no attribute 'types'` (numba/coverage version conflict). The ingest must import **only** `radis.api.exomolapi` and pass `local_databases` explicitly (avoids the broken `radis.config` default lookup).
- exojax's MdbExomol path is gone (`exojax.spec.api` deprecated; `exojax.database.exomol` empty in 2.5.0). RADIS is the only path.
- ⚠️ **Pre-flight blocker:** before the real load, fix the numba breakage (pin `coverage<7.6` or upgrade numba) and run a bounded smoke test — `MdbExomol`'s read path may also touch `radis.lbl.base` (`linestrength_from_Einstein`); whether that import dodges numba is unverified.

**Authoritative dataset facts (from the `.def`, not the trans):** `48Ti-16O__Toto`, version 20240509.

| field | value |
|---|---|
| dataset | **Toto** (48Ti-16O recommended isotopologue; ⁴⁸Ti ≈ 73.7% abundance) |
| **transitions** | **58,983,952 (~59M — NOT ~30M)** |
| states | 301,245 |
| trans files | **1** (`numinf=None`) → single `.trans.bz2`, no wavenumber splitting |
| max wavenumber | 30,000 cm⁻¹ → **bluest line ≈ 333.3 nm vacuum** (no TiO lines below ~333 nm) |
| broadeners | **0** — no `.broad` files (H2/He/air.broad all 404) |
| molmass | 63.943 Da |

**Download sizes (HEAD-verified, no body fetched):**

| file | size |
|---|---|
| `48Ti-16O__Toto.trans.bz2` | **304.75 MB** (the only large file) |
| `48Ti-16O__Toto.states.bz2` | 5.90 MB |
| `48Ti-16O__Toto.pf` | 0.20 MB |
| `48Ti-16O__Toto.def` | 7.5 KB |

→ **one bulk download ≈ 311 MB compressed.** On first load RADIS decompresses + reconverts to an `.h5` vaex/feather cache (expect a few GB of derived cache); the `.trans.bz2` can be deleted afterward. Load + `.h5` build time is **minutes to tens of minutes** (estimated, not measured).

**Ingest ONLY 48Ti-16O.** The other four isotopologues (46/47/49/50Ti-16O) are full ~59M Toto lists (~301–308 MB each) near-identical apart from an isotope mass shift. The `molecular_lines` schema has **no isotopologue field** (`species='TiO'`), so mixing them silently overlays near-duplicate lines for marginal LIBS benefit at typical resolving power. Leave them out unless isotopic structure is explicitly required.

**Exact python call** (window-limited; `nurange` in cm⁻¹ wavenumber, ascending):

```python
import radis.api.exomolapi as ea          # ONLY this submodule (top-level radis is numba-broken)
# air 333–1000 nm -> vacuum 1e7/lambda_vac -> nu in cm-1
#   1000 nm -> 10000 cm-1 ; 333.3 nm -> 30000 cm-1 (TiO maxnu)
mdb = ea.MdbExomol(
    path="TiO/48Ti-16O/Toto",            # -> <local_databases>/TiO/48Ti-16O/Toto/
    molecule="TiO",
    database="Toto",
    local_databases="data/exomol",       # cache root; pass EXPLICITLY (dodges radis.config)
    nurange=[10000.0, 30000.0],          # cm-1; covers 333.3–1000 nm air
    engine="vaex",                       # memory-mapped; REQUIRED for 59M rows
    broadf=False, broadf_download=False, # MANDATORY: Toto has 0 broad files -> avoids 404s
    skip_optional_data=False,            # keep J/quantum labels (set True for speed if unneeded)
    cache=True, verbose=True,
)
# resulting columns: mdb.nu_lines (cm-1, VACUUM), mdb.A (s-1), mdb.elower (cm-1),
#   mdb.gpp (g_upper), mdb.jlower, mdb.jupper ; partition: mdb.QT_interp(T)
```

> `nurange` does **NOT** shrink the *download* (single trans file → the whole 305 MB `.trans.bz2` is fetched regardless); it only limits what gets loaded/cached. Window-filtering to a wavelength range is a post-load filter on `nu_lines`.

### B.2 Field mapping: ExoMol/RADIS → `molecular_lines`

Constants (same as `ingest_vald_atomic.py`): `HC = 1.239841984e-4 eV·cm`, `_AKI_CONST = 6.6702e15`. ExoMol `nu_lines` is **vacuum**; VALD slices are **air** → convert every line with `vacuum_to_air_nm` (cflibs air convention; verified shift +0.096 nm @333 nm → +0.274 nm @1000 nm) so the merged table stays single-medium (air).

| `molecular_lines` column | source / formula |
|---|---|
| `species` | `'TiO'` |
| `charge` | `1` (neutral; matches VALD ion convention) |
| `wavelength_nm` | `vacuum_to_air_nm(1e7 / nu_lines)` |
| `aki` | `A` (Einstein A, s⁻¹, directly) |
| `loggf` | `log10(gk * A * lam_air_A**2 / 6.6702e15)` — same constant as VALD ingest (provenance parity); or NULL |
| `ei_ev` | `elower * 1.239841984e-4` |
| `ek_ev` | `(elower + nu_lines) * 1.239841984e-4` |
| `gi` | `2*jlower + 1` (approx; `pickup_gE` returns only `gup`. Ignores Λ-doubling/spin — same shape as VALD's 2J+1) |
| `gk` | `gpp` (exact, from states table) |
| `gamma_rad_log`, `gamma_stark_log`, `gamma_vdw_log` | **NULL** — Toto ships 0 broad files; do not fabricate |
| `accuracy_grade` | `'C'` blanket for v1 (ExoMol theory/MARVEL-mixed). Optional refinement: `'B'` when both levels are MARVEL (`SourceType=Ma`), else `'C'` |
| `provenance` | `'exomol_toto'` |

### B.3 Ingest-script plan: `scripts/ingest_exomol_tio.py` (new)

Mirror `ingest_vald_atomic.py`: same `molecular_lines` schema (`CREATE TABLE IF NOT EXISTS`), same dedup discipline, write to the **same DB** so VALD's 7 minor molecules + ExoMol TiO coexist.

```python
#!/usr/bin/env python
"""Ingest ExoMol TiO 'Toto' (48Ti-16O) into the cflibs molecular_lines table.
Streams ~59M lines via radis.api.exomolapi.MdbExomol (vaex, memory-mapped),
filters to an AIR wavelength window, converts vacuum->air, batches sqlite inserts.
Run: PYTHONPATH=$PWD .venv/bin/python scripts/ingest_exomol_tio.py \
       --db output/vald_atomic.db --wl-min 200 --wl-max 1000 --local-db data/exomol
"""
import argparse, math, sqlite3
from pathlib import Path
import radis.api.exomolapi as ea          # ONLY this submodule (top-level radis is numba-broken)
from cflibs.atomic.wavelength_conversion import vacuum_to_air_nm

HC_EV_CM  = 1.239841984e-4
AKI_CONST = 6.6702e15
SCHEMA = "<copy molecular_lines CREATE TABLE IF NOT EXISTS from ingest_vald_atomic.py>"

def nm_window_to_nurange(wl_min_air, wl_max_air):
    # air->vacuum (approx, widened by margin), vacuum nm -> cm-1
    return [1e7 / wl_max_air * 0.999, 1e7 / wl_min_air * 1.001]

def main():
    # argparse: --db --wl-min --wl-max --local-db --batch(=50_000) --grade(=C)
    nurange = nm_window_to_nurange(args.wl_min, args.wl_max)
    mdb = ea.MdbExomol(path="TiO/48Ti-16O/Toto", molecule="TiO", database="Toto",
                       local_databases=args.local_db, nurange=nurange,
                       engine="vaex", broadf=False, broadf_download=False,
                       skip_optional_data=False, cache=True, verbose=True)
    conn = sqlite3.connect(args.db); conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=OFF")

    nu, A, El, gu = mdb.nu_lines, mdb.A, mdb.elower, mdb.gpp
    jl = getattr(mdb, "jlower", None)
    INSERT = ("INSERT INTO molecular_lines (species,charge,wavelength_nm,aki,loggf,"
              "ei_ev,ek_ev,gi,gk,gamma_rad_log,gamma_stark_log,gamma_vdw_log,"
              "accuracy_grade,provenance) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)")
    seen, batch, B = set(), [], args.batch
    for k in range(0, len(nu), B):                    # STREAM — never materialize 59M python rows
        s = slice(k, k + B)
        for i in range(len(nu[s])):
            nu_i, A_i, El_i = float(nu[s][i]), float(A[s][i]), float(El[s][i])
            gk = int(gu[s][i])
            lam_air = float(vacuum_to_air_nm(1e7 / nu_i))   # vacuum->air BEFORE window + loggf
            if not (args.wl_min <= lam_air <= args.wl_max):
                continue
            jl_i = float(jl[s][i]) if jl is not None else None
            gi  = int(round(2 * jl_i + 1)) if jl_i is not None else None
            ei, ek = El_i * HC_EV_CM, (El_i + nu_i) * HC_EV_CM
            lam_air_A = lam_air * 10.0
            loggf = (math.log10(gk * A_i * lam_air_A * lam_air_A / AKI_CONST)
                     if gk > 0 and A_i > 0 else None)
            key = ("TiO", 1, round(lam_air, 4), round(ei, 4), round(ek, 4))
            if key in seen:
                continue
            seen.add(key)
            batch.append(("TiO", 1, lam_air, A_i, loggf, ei, ek, gi, gk,
                          None, None, None, args.grade, "exomol_toto"))
            if len(batch) >= B:
                conn.executemany(INSERT, batch); conn.commit(); batch.clear()
    if batch:
        conn.executemany(INSERT, batch); conn.commit()
    conn.execute("PRAGMA synchronous=NORMAL"); conn.close()
```

**Memory/time discipline at 59M rows:**
- `engine="vaex"` memory-maps the `.h5` cache instead of loading 59M×N floats into RAM. First load builds the cache (one-time, minutes–tens of minutes); later loads are fast.
- Stream slices of `B≈50k`; `executemany` + `commit()` per batch — never accumulate all rows or one giant transaction. `PRAGMA journal_mode=WAL; synchronous=OFF` during bulk load.
- The window cut (333–1000 nm air; below ~333 nm is empty) keeps the in-RAM `seen` dedup set bounded. **If you ever ingest the full unfiltered 59M**, prefer a UNIQUE index `(species, charge, round(wl), round(ei), round(ek))` + `INSERT OR IGNORE` and let SQLite dedup, rather than a 59M-tuple Python set.

**Gotchas:** top-level `radis` import crashes (numba/coverage) — import only `radis.api.exomolapi`, pass `local_databases` explicitly; `broadf=False, broadf_download=False` mandatory (0 broad files → 404 otherwise); `nurange` does NOT shrink the 305 MB download; `gi=2J_lower+1` is an approximation; vacuum→air on every line (all TiO ≥333 nm > 200 nm boundary), before the window test and before `loggf`. The `.pf` partition function (fetched by `MdbExomol`, via `mdb.QT_interp(T)`) is **not** needed for the line table but **will** be needed by any future TiO band-emissivity forward model — keep it.

---

## PART C — Final assembly: build the COMPLETE DB + sanity check

Once **all VALD slices** (existing 100–324 nm + the 54 new Extract-Element dumps) and the **ExoMol TiO** dump are downloaded, build the DB in three steps. Run everything from the worktree root with `PYTHONPATH=$PWD` and check the printed `cflibs=` provenance line (worktree-import trap).

```bash
# 0) work from the worktree root; verify the right cflibs is imported
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5
PYTHONPATH=$PWD .venv/bin/python -c "import cflibs, sys; print('cflibs=', cflibs.__file__)"

# 1) Ingest ALL VALD slices -> atomic lines + the 7 minor molecules (TiO NOT in these).
#    One shot over the whole data/vald dir; dedup makes group/slice overlap safe.
PYTHONPATH=$PWD .venv/bin/python scripts/ingest_vald_atomic.py \
    --vald data/vald/*.linelist.gz \
    --db output/vald_atomic.db \
    --wl-min 100 --wl-max 1000

# 2) Ingest ExoMol TiO 'Toto' (48Ti-16O) -> molecular_lines, SAME DB.
#    (Ensure the numba/coverage breakage is fixed first; see Part B.1.)
PYTHONPATH=$PWD .venv/bin/python scripts/ingest_exomol_tio.py \
    --db output/vald_atomic.db \
    --wl-min 200 --wl-max 1000 \
    --local-db data/exomol

# 3) Complete the atomic physics (partition functions, species_physics, etc.)
PYTHONPATH=$PWD .venv/bin/python scripts/complete_atomic_db.py \
    --db output/vald_atomic.db
```

### Coverage / sanity check

Confirm (a) atomic coverage is contiguous 100–1000 nm and (b) `molecular_lines` has TiO **plus** all 7 minor molecules.

```bash
PYTHONPATH=$PWD .venv/bin/python - <<'PY'
import sqlite3
db = sqlite3.connect("output/vald_atomic.db")

# (a) ATOMIC: contiguous 100-1000 nm — count lines per 50 nm bin; flag empty bins.
print("=== Atomic coverage (lines per 50 nm bin, 100-1000 nm) ===")
rows = db.execute("""
    SELECT CAST(wavelength_nm/50 AS INT)*50 AS bin, COUNT(*)
    FROM lines WHERE wavelength_nm BETWEEN 100 AND 1000
    GROUP BY bin ORDER BY bin
""").fetchall()
have = {b for b, _ in rows}
for b, n in rows:
    print(f"  {b:4d}-{b+50:<4d} nm : {n:>8d}")
gaps = [b for b in range(100, 1000, 50) if b not in have]
print("  ATOMIC GAPS (empty 50 nm bins):", gaps if gaps else "NONE -> contiguous OK")

# (b) MOLECULAR: TiO + the 7 minor molecules all present, with counts + medium check.
print("\n=== molecular_lines species present ===")
want = {"TiO", "CN", "C2", "OH", "CH", "CO", "NH", "MgH"}
present = {}
for sp, n, lo, hi in db.execute("""
        SELECT species, COUNT(*), MIN(wavelength_nm), MAX(wavelength_nm)
        FROM molecular_lines GROUP BY species ORDER BY species"""):
    present[sp] = n
    print(f"  {sp:5s}: {n:>9d} lines   {lo:.2f}-{hi:.2f} nm")
missing = want - set(present)
print("  MOLECULAR MISSING:", missing if missing else "NONE -> TiO + 7 minor all present")

# TiO provenance + air-window sanity (should start ~333 nm, none below)
tio_lo = db.execute("SELECT MIN(wavelength_nm) FROM molecular_lines WHERE species='TiO'").fetchone()[0]
print(f"\n  TiO min wavelength = {tio_lo:.2f} nm (expect ~333 nm; none bluer)")
print("  TiO provenance:", db.execute(
    "SELECT DISTINCT provenance FROM molecular_lines WHERE species='TiO'").fetchall())
db.close()
PY
```

**PASS criteria:**
- **Atomic:** every 50 nm bin from 100 to 1000 nm is non-empty → "ATOMIC GAPS: NONE". (If a red bin is empty, a VALD Extract-Element range was truncated — re-check Part A.4 truncation detection and re-request that split range.)
- **Molecular:** "MOLECULAR MISSING: NONE" — `TiO`, `CN`, `C2`, `OH`, `CH`, `CO`, `NH`, `MgH` all present. TiO min wavelength ≈ 333 nm with `provenance='exomol_toto'`; the 7 minor molecules carry the VALD provenance.

---

### Key file paths (absolute)
- VALD ingester: `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/scripts/ingest_vald_atomic.py`
- ExoMol TiO ingester (to create): `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/scripts/ingest_exomol_tio.py`
- Atomic-physics completion: `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/scripts/complete_atomic_db.py`
- VALD slice directory: `/home/brian/code/CF-LIBS-improved/data/vald/`
- ExoMol cache root (to create): `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/data/exomol/`

### Open blockers / unverified mechanics
1. **numba/coverage breakage** (`module 'coverage' has no attribute 'types'`) must be fixed (pin `coverage<7.6` or upgrade numba) before the real `MdbExomol` load; smoke-test that `radis.lbl.base` import also dodges numba.
2. **VALD cap scope** (per species-group vs per range) is the one unverified mechanic — the post-dump truncation check (returned `WL_air` max vs requested edge) makes the plan robust either way.
3. **Line-count estimates** (±10–15%) drive the choice of the conservative 18-range plan; truncation detection backstops any underestimate.