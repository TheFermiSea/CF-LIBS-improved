# Provenance: Ruffoni et al. 2014 — table4.dat (omitted / previously-measured Fe I lines)

## Citation
Ruffoni M.P., Den Hartog E.A., Lawler J.E., Brewer N.R., Lind K., Nave G., Pickering J.C.,
"Fe I oscillator strengths for the Gaia-ESO survey."
Mon. Not. R. Astron. Soc., 441, 3127-3136 (2014). Bibcode: 2014MNRAS.441.3127R.

## VizieR / CDS catalog
- Catalog ID: **J/MNRAS/441/3127**
- File: `table4.dat` — "Lines measured in previous studies, or predicted to have a BF
  greater than 1%, that were omitted from the BF measurements shown in table 3"
- Download URL: https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/441/3127/table4.dat
- Downloaded (UTC): 2026-07-02
- Records: 32 data rows (+ `#` header lines).  Lrecl=116.
  md5: d49e0d9e738b317c4f3bf7d0dbb23571  (4182 bytes)

## Column meanings / units (byte-by-byte, from ReadMe — verbatim)
| Bytes  | Fmt  | Units | Label    | Meaning |
|--------|------|-------|----------|---------|
| 1      | A1   | ---   | State    | [A-L] State (Note G1) |
| 3-10   | A8   | ---   | Level    | Lower level label |
| 12     | A1   | ---   | Note     | [*] wavenumber from FT spectra (Note G2) |
| 14-23  |F10.4 |0.1nm  | lam.Air  | Transition air wavelength, Angstrom |
| 25-33  |F9.3  |cm-1   | sigma    | Transition wavenumber |
| 35-39  |F5.2  |[-]    | loggf0   | Best previously published log(gf) |
| 41-44  |F4.2  |[-]    | e_loggf0 | Uncertainty in loggf0 (optional) |
| 45     | A1   |---    | q_loggf0 | Uncertainty flag (D/E) |
| 47-49  | A3   |---    | r_loggf0 | Reference source of loggf0 (O91/B91/B82/M74/K07) |
| 51     | A1   |---    | l_BFp    | '<' if predicted BF < 0.001 |
| 52-57  |F6.4  |---    | BFp      | Predicted branching fraction |
| 59-73  | A15  |---    | Reason   | Reason for omission of the line |
| 75-116 | A42  |---    | Blend    | Blended-line list when Reason='Blended' |

## Content note
This table contains **NO this-work log(gf)** — it lists lines that were *omitted* from the
BF measurement, carrying only previously-published `loggf0` (incl. O'Brian 1991 = O91) and a
predicted BF. It is NOT a source of new lab gf values; retained for completeness/traceability.

## Parse-check (no transformation applied)
- 32 data records. `loggf` (this-work) column: absent by design. 0 new lab-gf lines.
