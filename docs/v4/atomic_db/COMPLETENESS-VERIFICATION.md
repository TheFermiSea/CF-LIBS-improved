# Atomic DB Completeness — Verification + Build Plan (2026-06-26)

Cross-checked `ASD_da/libs_production.db` against three sources: the
`CAAAMLIBS/LIBS` netCDF reconstruction (full ASD-5.9, pulled via the LFS API:
Levels 124 MB / Lines 206 MB / IPs 6 MB), live NIST ASD (the fetch scripts), and
internal consistency. (`ASD59_dump.sql` in that repo is an empty 0-byte
placeholder; the real dump is on the `ceng` Windows box / a laptop — pending
key authorization or a copy.) Decisions (user): **complete the SQLite DB**;
**netCDF base + live-NIST reconcile + stochastic direct-NIST validation.**

## Verdict per table

| Table | Rows (DB vs netCDF, I/II/III) | Fields | Status |
|---|---|---|---|
| `energy_levels` | 61,507 vs 57,675 | 4 of 34 | **Rows COMPLETE + current** (faithful to live NIST: Au I 61=61, B I exact incl. 201 eV autoionizing levels). Missing FIELDS: J, term, config, Landé-g, uncertainty, reference. |
| `lines` | 28,721 vs **31,145 with A-value** (104,995 incl. 70% observation-only, no A) | 19 of 42 | **~92% of the A-valued (CF-LIBS-useful) lines** (Fe I 2,439 ≥ netCDF-with-A 2,379). Real gap: ~8% A-valued + zero-line species (Ne II, Cs II, Co III…). Missing FIELDS: log_gf, osc_str, vac/obs wl, type, accuracy, references. |
| `species_physics` (IP) | ip_ev for 174 of 273 | 4 of 24 | **99 IP gaps** (81 stage III) — being filled from live NIST. Missing FIELDS: IP uncertainty, term, config, ground shells. |

**Bottom line:** levels are complete + current; the database is far more complete
than the raw line count implied (the 73% "gap" was observation-only lines).
Genuine gaps: 99 IPs, ~8% of A-valued lines + zero-line species, and provenance
fields across all tables.

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
2. **Lines:** add the A-valued netCDF lines my DB lacks (+ zero-line species);
   add fields log_gf/osc_str/vac_wl/obs_wl/type/accuracy/references (ALTER + backfill).
3. **Levels fields:** ALTER `energy_levels` to add J/term/config/Landé-g/uncertainty/
   reference; backfill by matching (element, stage, energy) to the netCDF.
4. **Validate (stochastic):** fetch random (element, stage) chunks directly from
   NIST (levels + IPs; lines via lines1.pl) and diff against the rebuilt DB.
5. Keep SQLite canonical (codebase compatibility); optionally emit a Parquet
   mirror for analytics later. Maintain backward-compatible existing columns.
