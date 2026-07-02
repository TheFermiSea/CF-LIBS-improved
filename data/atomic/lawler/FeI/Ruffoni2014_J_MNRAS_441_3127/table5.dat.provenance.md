# Provenance: Ruffoni et al. 2014 — table5.dat (solar-analysis subset)

## Citation
Ruffoni M.P., Den Hartog E.A., Lawler J.E., Brewer N.R., Lind K., Nave G., Pickering J.C.,
"Fe I oscillator strengths for the Gaia-ESO survey."
Mon. Not. R. Astron. Soc., 441, 3127-3136 (2014). Bibcode: 2014MNRAS.441.3127R.

## VizieR / CDS catalog
- Catalog ID: **J/MNRAS/441/3127**
- File: `table5.dat` — "Lines from table 3 selected for solar analysis"
- Download URL: https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/441/3127/table5.dat
- Downloaded (UTC): 2026-07-02
- Records: 36 (Lrecl=62).  md5: af97eb7a590cdc778a0a1b6ab14b43a7  (2268 bytes)

## Column meanings / units (byte-by-byte, from ReadMe — verbatim)
| Bytes | Fmt  | Units | Label    | Meaning |
|-------|------|-------|----------|---------|
| 1-10  |F10.4 |0.1nm  | lam.Air  | Transition air wavelength, Angstrom [4423/10864] |
| 12-16 |F5.3  |eV     | chi      | Lower-level excitation energy |
| 18-25 |F8.3  |---    | VdW      | Van der Waals damping parameter (ABO packed notation; see Note 1) |
| 27-31 |F5.2  |[-]    | loggf    | **This-work log(gf)** |
| 33-36 |F4.2  |[-]    | e_loggf  | Uncertainty in this-work log(gf) |
| 38-42 |F5.2  |[-]    | loggf0   | Best previously published log(gf) |
| 44-47 |F4.2  |[-]    | e_loggf0 | Uncertainty in loggf0 (optional) |
| 48    | A1   |---    | q_loggf0 | Uncertainty flag (D/E) |
| 50-52 | A3   |---    | r_loggf0 | Reference source of loggf0 |
| 54-57 |F4.2  |[-]    | eps      | Solar log-eps(Fe) using this-work loggf |
| 59-62 |F4.2  |[-]    | eps0     | Solar log-eps(Fe) using loggf0 |

## Content note
36-line solar-analysis **subset of table3.dat** (same this-work loggf values). To avoid
double-counting, the FeI lab-gf line total is taken from table3.dat, not this file.

## Parse-check (no transformation applied)
- 36 records; all carry a this-work `loggf`. Subset of table3 — not counted separately.
