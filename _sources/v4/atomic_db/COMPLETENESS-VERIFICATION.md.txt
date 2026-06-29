# Atomic DB Completeness — Verification + Build (2026-06-26)

## FINAL STATUS — built from the gold standard + validated

The gold-standard **`ASD59_dump.sql`** (full NIST ASD `monograph8`, MySQL, 115 MB,
2022) was obtained from the user's laptop and loaded into local MariaDB. The DB
was then built directly from it (the `CAAAMLIBS/LIBS` netCDF turned out to be a
LOSSY conversion — it had dropped ~half the lines).

| Table | Final | vs gold standard | Fields |
|---|---|---|---|
| `lines` | **203,695** | 99.6% of 204,574 (rest = degenerate Rydberg lines colliding on the coarse UNIQUE key) | 19 → **31** |
| `energy_levels` (I/II/III) | **62,752** | ≥ dump 62,617 (union of dump + current NIST) | 4 → **11** |
| `species_physics` (IP) | **324, 0 gaps** | complete | 4 |

**Stochastic live-NIST validation (18 random species incl. Ti/Al/V/Fe/Cr/Ni):
18/18 levels covered (DB ≥ live NIST), 17/17 IPs match within 2%.** The DB is the
union of the 2022 dump and a current-NIST level/IP ingest, so it slightly exceeds
current live NIST per species. All 28,331 derived Stark fields preserved.
`integrity_check` = ok. The IP table is from live NIST (the dump's `IPs` table is
corrupt — Fe I shows 9277 eV). Scripts: `ingest_asd_dump_lines.py`,
`add_missing_dump_lines.py` (transition-identity dedup), `ingest_dump_levels.py`,
`ingest_nist_ips.py`, `validate_db_vs_nist.py`.

---

## Earlier verification (netCDF + live NIST, pre-dump)

Cross-checked `ASD_da/libs_production.db` against three sources: the
`CAAAMLIBS/LIBS` netCDF reconstruction (full ASD-5.9, pulled via the LFS API:
Levels 124 MB / Lines 206 MB / IPs 6 MB), live NIST ASD (the fetch scripts), and
internal consistency. Decisions (user): **complete the SQLite DB**;
**dump base + live-NIST reconcile + stochastic direct-NIST validation.**

## Verdict per table

| Table | Rows (DB vs netCDF, I/II/III) | Fields | Status |
|---|---|---|---|
| `energy_levels` | 61,507 vs 57,675 | 4 of 34 | **Rows COMPLETE + current** (faithful to live NIST: Au I 61=61, B I exact incl. 201 eV autoionizing levels). Missing FIELDS: J, term, config, Landé-g, uncertainty, reference. |
| `lines` | **28,721 vs 104,995 (DB has ~30% of ALL observed lines)** | 19 of 42 | **INCOMPLETE — missing ~73,850 observed transitions.** Only 30% carry a *measured* A; the other 70% have observed wavelength (100%), observed intensity (99%), and upper/lower level IDs (83%) — real lines missing only a measured transition probability (derivable from theory/log_gf/level link). These ARE needed: full-database synthetic spectra + better-than-CF-LIBS methods, not just classical CF-LIBS. Missing FIELDS: A, log_gf, osc_str, intens, vac/obs/calc wl, level IDs, type, accuracy, references. |
| `species_physics` (IP) | ip_ev for 174 of 273 | 4 of 24 | **99 IP gaps** (81 stage III) — being filled from live NIST. Missing FIELDS: IP uncertainty, term, config, ground shells. |

**Bottom line:** levels complete + current; IPs now complete. Lines are GENUINELY
incomplete — the DB holds ~30% of all observed transitions. The goal is the FULL
ASD: every observed line + every field, so the database can drive synthetic
spectra over the entire spectrum AND serve methods beyond classical CF-LIBS
(which is the floor, not the target). Lines without a measured A are kept with
their observed intensity + level IDs; A is derived/estimated (with documented
justification) at synthetic-generation time, not by dropping the line.

## Fixes/findings
- **`datagen_v2.fetch_ionization_potential` was broken** (parsed for a line
  starting with the element; NIST format-3 data line starts with a quoted `""`
  and the parser never stripped quotes → always None). FIXED + verified against
  known IPs (Fe I 7.9024681, Cr III 30.959, …). Unblocks IP fill + validation.
- `datagen_v2` has **no lines fetcher** (only levels + IPs). The lines source for
  the reconcile is the netCDF A-valued lines (+ a future NIST lines1.pl scraper).
- netCDF units/semantics decoded: levels `energy` is cm⁻¹ (→ eV /8065.544);
  lines `vac_wl` Å / `obs_wl` nm; IPs `sc` is the electron count (Z−charge).

## Build plan (complete the SQLite DB)
1. **IPs (running):** fill 99 gaps from live NIST (`scripts/ingest_nist_ips.py`).
2. **Lines (full):** ingest ALL ~105k observed lines (not just the A-valued
   subset), with every field — A, log_gf, osc_str, observed intensity, vac/obs/calc
   wavelength, low/upp level IDs (FK to energy_levels), type, accuracy, references.
   Add a `has_aki`/`aki_source` flag so synthetic generation can choose: measured
   A, theory/log_gf-derived A, or intensity-anchored estimate (documented). Keep
   the existing columns/values so the codebase still reads them.
3. **Levels fields:** ALTER `energy_levels` to add J/term/config/Landé-g/uncertainty/
   reference; backfill by matching (element, stage, energy) to the netCDF.
4. **Validate (stochastic):** fetch random (element, stage) chunks directly from
   NIST (levels + IPs; lines via lines1.pl) and diff against the rebuilt DB.
5. Keep SQLite canonical (codebase compatibility); optionally emit a Parquet
   mirror for analytics later. Maintain backward-compatible existing columns.
