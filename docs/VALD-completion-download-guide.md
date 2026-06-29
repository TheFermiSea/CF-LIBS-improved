# Completing the VALD Atomic Database: Remaining Downloads

This guide takes the CF-LIBS VALD line database from its current 6 partial slices to **complete, contiguous 100–1000 nm coverage**. It tells you exactly what to request from VALD3, with what settings, how to name the files, and how to ingest the result.

---

## 1. Status

**Already downloaded** (6 slices in `data/vald/`, keep them all):

| File | Raw transitions | WL range (nm) |
|------|-----------------|---------------|
| `vald3_extract-all_100-154nm` | 116,551 | 100.00–154.42 |
| `vald3_extract-all_154-210nm` | 125,437 | 154.40–209.89 |
| `vald3_extract-all_220-270nm` | 137,609 | 220.00–269.56 |
| `vald3_extract-all_300-308nm` | 15,743 | 300.00–307.67 |
| `vald3_extract-all_400-402nm` | 4,149 | 399.99–402.15 |
| `vald3_extract-all_500-502nm` | 2,741 | 500.00–501.55 |

**Exact gaps to fill** (everything not covered above): `210–220`, `270–300`, `308–400`, `402–500`, and the entire **`502–1000 nm`** region.

> **Density reality check (load-bearing):** VALD line density does **not** collapse above 300 nm. Complete raw-file slices measure **2,051 lines/nm at 300–308**, **1,932/nm at 400–402**, and **1,767/nm at 500–502** — density stays above ~1,750/nm across the whole 100–500 nm range and only tapers slowly into the IR. The earlier "density falls off a cliff above 300 nm" idea was a partial-bin measurement artifact. **Consequence:** the high-wavelength gaps need ~40–50 nm chunking just like the UV — `502–1000 nm` alone is ~575k lines and needs **8 requests**, not 1.

The densest existing request (220–270 nm) holds 137,609 lines in 49.56 nm — that sits right at VALD's practical per-request ceiling. **Treat the cap as ~137k hard / ~110k safe**, which is why every chunk below is sized to stay under ~110k.

---

## 2. The Request Table

Submit these **14 requests** in order. All are estimated **under 110k lines**. Type the **Angstrom** range into the VALD form (= nm × 10). Each chunk overlaps ~1 nm into its neighbor; the ingest dedups by `(element, ion, wl, E_low, E_up)`, so overlaps lose **zero boundary lines** and create **zero duplicates**.

| R# | nm range | **Angstrom (type this)** | Est. lines | Fills gap | Overlaps into |
|----|----------|--------------------------|------------|-----------|---------------|
| R1 | 209–221 | **2090–2210** | ~23,600 | 210–220 | 154–210 & 220–270 (done) |
| R2 | 269–301 | **2690–3010** | ~65,800 | 270–300 | 220–270 & 300–308 (done) |
| R3 | 307–355 | **3070–3550** | ~96,800 | 308–400 (1/2) | 300–308 (done) |
| R4 | 354–401 | **3540–4010** | ~92,100 | 308–400 (2/2) | R3 & 400–402 (done) |
| R5 | 401–451 | **4010–4510** | ~94,500 | 402–500 (1/2) | 400–402 (done) |
| R6 | 450–501 | **4500–5010** | ~92,200 | 402–500 (2/2) | R5 & 500–502 (done) |
| R7 | 501–556 | **5010–5560** | ~95,700 | 502–1000 (1/8) | 500–502 (done) |
| R8 | 555–610 | **5550–6100** | ~82,400 | 502–1000 (2/8) | R7 |
| R9 | 609–665 | **6090–6650** | ~78,800 | 502–1000 (3/8) | R8 |
| R10 | 664–724 | **6640–7240** | ~79,700 | 502–1000 (4/8) | R9 |
| R11 | 723–787 | **7230–7870** | ~78,600 | 502–1000 (5/8) | R10 |
| R12 | 786–855 | **7860–8550** | ~84,400 | 502–1000 (6/8) | R11 |
| R13 | 854–930 | **8540–9300** | ~79,000 | 502–1000 (7/8) | R12 — *900–930 extrapolated* |
| R14 | 929–1000 | **9290–10000** | ~64,300 | 502–1000 (8/8) | R13 — *930–1000 extrapolated* |

**Per-gap summary:** `210–220` = 1 req · `270–300` = 1 req · `308–400` = 2 reqs · `402–500` = 2 reqs · **`502–1000` = 8 reqs**. Total: **14 requests, ~1.11M new transitions.** The union of done + new is a single contiguous 100–1000 nm block with zero holes.

---

## 3. Exact VALD Form Settings

Use VALD3 at **http://vald.astro.uu.se/** with a **registered account** (free, but login required; results return by email/FTP, not in-browser). Apply this **identical** parameter block to **every** request so each new slice is byte-format-identical to the existing files (whose header reads `Elm Ion WL_air(A) log gf* E_low(eV) J lo E_up(eV) J up …`).

**Mode: `Extract All`** — returns *every* transition in the window with no abundance/depth/stellar cut. Do **not** use *Extract Stellar* (filters out weak/trace lines by line-depth model) or *Extract Element* (one species per request). The existing files are named `vald3_extract-all_*`, confirming this is the correct, already-used mode.

| Form field | Value | Why |
|---|---|---|
| **Output format** | **Long format** | Emits the multi-line record (config + term + Landé + damping + reference) the ingest expects. Short format omits fields and breaks parsing. |
| **Wavelength units** | **Angstrom (Å)** | Ingest does `wl_nm = wl_a / 10.0`; existing column is `WL_air(A)`. |
| **Energy units** | **eV** | Ingest reads `E_low(eV)`/`E_up(eV)` directly. |
| **Wavelength medium** | **air** | Existing column is `WL_air(A)`. Ingest consumes air directly above 200 nm; mixing in a vacuum request would corrupt the >200 nm path. |
| **Wavelength range** | **Angstrom per the table above** (the only field that changes per request) | — |
| **Hyperfine structure (HFS)** | **on** | Keeps HFS-split components; matches existing data, maximizes completeness. |
| **Isotopic scaling / isotopes** | **default** | Existing files used the default; changing it diverges the format. |
| **"Require known terms" / Landé / damping (γ) toggles** | **OFF (require nothing)** | Requiring these would *drop* lines lacking that metadata. Completeness first. |
| **Configuration / level-info** | **default** | Matches existing format. |
| **Retrieval / delivery** | **FTP** | VALD emails a gzipped-result FTP link (how the existing `.gz` files were obtained). |
| **Microturbulence / Vmicro** | leave default | Only used by Extract Stellar; irrelevant here. |

> Field labels can vary slightly by VALD interface version. The load-bearing intent: **Extract All · Long · Å · eV · air · HFS on · drop nothing**.

---

## 4. Per-Request Walkthrough

For each row R1…R14, in order:

1. **Log in** at http://vald.astro.uu.se/ (main VALD3 interface). All extractions are tied to your registered email.
2. **Select `Extract All`** and set the full parameter block from §3.
3. **Enter the Angstrom range** for that row (e.g. R2 → `2690` to `3010`). Submit.
4. **Wait for the FTP email** — minutes to a few hours depending on queue and range size (dense slices take longer). Be patient; **do not resubmit** the same request while waiting.
5. **Download** the two gzipped files from the FTP link: a **line list** (VALD-named like `vald_<jobid>.gz`) and a **bibliography** (`.bib.gz`).
6. **Rename into `data/vald/`** using the exact convention so the ingest glob `data/vald/*.linelist.gz` picks them up:

   ```bash
   mv vald_<jobid>.gz      data/vald/vald3_extract-all_<lo>-<hi>nm.linelist.gz
   mv vald_<jobid>.bib.gz  data/vald/vald3_extract-all_<lo>-<hi>nm.bib.gz
   ```

   where `<lo>-<hi>` are the **nm** bounds (Angstrom ÷ 10). Example for R2 (270–300 nm):

   ```bash
   mv vald_12345.gz      data/vald/vald3_extract-all_270-300nm.linelist.gz
   mv vald_12345.bib.gz  data/vald/vald3_extract-all_270-300nm.bib.gz
   ```

   **Keep files gzipped** — the ingest reads `.gz` directly. Do **not** gunzip.

**Etiquette:** VALD allows only a few requests in flight at once (~3–5 per user). Submit one slice, wait for its email, then submit the next — don't fire all 14 simultaneously. VALD is a free academic service; keep volume reasonable and cite VALD in any publication (use the per-slice `.bib.gz`).

---

## 5. If a Request Is Rejected for Too Many Lines

VALD does **not** silently truncate. If a range exceeds the per-request cap (~125–140k), the extraction **fails** and you get an **email saying the request returned too many lines** — no usable line list attached.

**To fix:** **narrow the wavelength range and resubmit.** Halve the chunk and request the two halves separately with a ~1 nm overlap (dedup makes the overlap free). Example — if R3 (`3070–3550`) is rejected, split into `3070–3320` and `3310–3550`. The estimates above all sit under ~110k with 25–45k of headroom, so rejections are unlikely; the extrapolated IR chunks (R13/R14) are the most uncertain — if either errors, split it further (low risk, both estimated well under cap).

---

## 6. Final Ingest + Verify

Once **all 14 new slices** are in `data/vald/` (alongside the original 6), build the complete database:

```bash
# 1. Ingest every VALD slice into the complete DB (dedups across overlaps)
PYTHONPATH=$PWD .venv/bin/python scripts/ingest_vald_atomic.py \
    --vald data/vald/*.linelist.gz \
    --db output/vald_complete.db \
    --wl-min 100 --wl-max 1000

# 2. Backfill partition functions / species physics into the DB
PYTHONPATH=$PWD .venv/bin/python scripts/complete_atomic_db.py \
    --db output/vald_complete.db

# 3. Coverage check — confirm contiguous 100-1000 nm with no holes
PYTHONPATH=$PWD .venv/bin/python -c "
import sqlite3
c = sqlite3.connect('output/vald_complete.db')
n, lo, hi = c.execute('SELECT COUNT(*), MIN(wavelength_nm), MAX(wavelength_nm) FROM lines').fetchone()
print(f'transitions={n:,}  wl_min={lo:.2f} nm  wl_max={hi:.2f} nm')
# histogram of line counts per 50 nm bin — every bin in [100,1000) should be populated
for b in range(100, 1000, 50):
    k = c.execute('SELECT COUNT(*) FROM lines WHERE wavelength_nm>=? AND wavelength_nm<?', (b, b+50)).fetchone()[0]
    print(f'  {b:4d}-{b+50:4d} nm: {k:>8,}')
"
```

Expect **~1.2M+ total transitions** and **every 50 nm bin from 100 to 1000 nm populated** (no zero bins). If a bin is empty, a slice is missing or misnamed — re-check that its file landed in `data/vald/` as `vald3_extract-all_<lo>-<hi>nm.linelist.gz`.

> Adjust `.venv/bin/python` to your interpreter path, and confirm the table/column names (`lines`, `wl`) against `scripts/ingest_vald_atomic.py` if the coverage snippet errors — schema names there are authoritative.

---

## 7. Caveats

- **IR extrapolation (R13 tail + R14, ~900–1000 nm):** line-count estimates here are **extrapolated beyond Kurucz's 200–900 nm reference coverage** (850 nm value decayed 0.90 per 50 nm). Actual VALD counts may differ; if R13 or R14 errors as too-large, split further (low risk — both are estimated well under the cap).
- **Above-500 nm absolute scale** is anchored to the single measured 500–502 nm slice (1,767 lines/nm), using Kurucz only for the *relative* falloff shape. VALD and Kurucz absolute densities differ ~13×, so a different real ratio would shift the high-region estimates proportionally — but every high chunk carries 25–45k of headroom under the cap, so the plan absorbs that.
- **Per-request cap (~125–140k)** is user-reported and corroborated by the existing dense-UV slices succeeding (116k–138k lines each); it was not re-scraped live from VALD during this work. Behavior on overflow (error email vs. truncation) follows documented VALD3 behavior.
- **Air-medium consistency is mandatory:** all existing and new slices are requested in **air**. The ingest only vacuum→air-converts *below* 200 nm and assumes air above; **never** mix a vacuum-wavelength request into `data/vald/`, or the >200 nm path will be wrong.
- **Overlaps are free insurance:** every ~1 nm overlap is zero-cost because ingest dedups by `(element, ion, wl, E_low, E_up)`. Re-requesting any already-downloaded range is harmless — use this freely to guarantee seamless boundaries.