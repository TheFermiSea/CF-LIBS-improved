# Provenance: Ruffoni et al. 2014 — table3.dat (Fe I lab log gf / BF / A)

## Citation
Ruffoni M.P., Den Hartog E.A., Lawler J.E., Brewer N.R., Lind K., Nave G., Pickering J.C.,
"Fe I oscillator strengths for the Gaia-ESO survey."
Mon. Not. R. Astron. Soc., 441, 3127-3136 (2014). Bibcode: 2014MNRAS.441.3127R.
DOI (CDS): 10.26093/cds/vizier.74413127.

## VizieR / CDS catalog
- Catalog ID: **J/MNRAS/441/3127**  (catid 74413127)
- File: `table3.dat` — "Experimental branching fractions, transition probabilities,
  and log(gf)s for the Fe I levels listed in table 2 of the paper"
- Download URL: https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/441/3127/table3.dat
- ReadMe: https://cdsarc.cds.unistra.fr/ftp/J/MNRAS/441/3127/ReadMe
- Downloaded (UTC): 2026-07-02
- Records: 167 data rows + 35 in-file `#` header lines = 202 physical lines (Lrecl=102).
  md5: ce141adfb31512de60fd9561c83219fb  (18389 bytes)

## Column meanings / units (byte-by-byte, from ReadMe — verbatim)
| Bytes | Fmt  | Units  | Label    | Meaning |
|-------|------|--------|----------|---------|
| 1     | A1   | ---    | State    | [A-L] Upper-level state key; maps to lifetimes in ReadMe Note (G1) |
| 3-10  | A8   | ---    | Level    | Lower level term label |
| 12    | A1   | ---    | Note     | [*] wavenumber taken from these authors' FT spectra (Note G2) |
| 14-23 |F10.4 | 0.1nm  | lam.Air  | Transition air wavelength, Angstrom [2722/13261] |
| 25-33 |F9.3  | cm-1   | sigma    | Transition wavenumber [7539/36727] |
| 35-40 |F6.4  | ---    | BF       | Branching fraction [0/1] (optional) |
| 42-45 |F4.1  | %      | e_BF     | BF uncertainty, percent (optional) |
| 47-52 |F6.3  |10-6s-1 | Aul      | Transition probability A_ul (optional; note ReadMe unit label 10-6 s-1) |
| 54-58 |F5.2  | [-]    | loggf    | **This-work log(gf)** (optional) |
| 60-63 |F4.2  | [-]    | e_loggf  | Uncertainty in this-work log(gf) (optional) |
| 65-69 |F5.2  | [-]    | loggf0   | Best previously published log(gf) (optional) |
| 71-74 |F4.2  | [-]    | e_loggf0 | Uncertainty in loggf0 (optional) |
| 75-76 | A2   | ---    | q_loggf0 | Uncertainty flag on loggf0 (D/E per Fuhr & Wiese notation) |
| 78-80 | A3   | ---    | r_loggf0 | Ref: O91=O'Brian+1991; B91=Bard+1991; B82=Blackwell+1982; M74=May+1974; K07=Kurucz2007 |
| 82-84 | A3   | ---    | GES      | 'Yes' if line is a Gaia-ESO survey target |

## Radiative lifetimes (tau) — location note
`table3.dat` has **no tau column**. The per-upper-level radiative lifetimes are given in
the **ReadMe Note (G1)** for the 12 states A-L, each stated at **+/-5%**, e.g.:
A=43163.323 cm^-1 (8.5 ns), ... L=54683.318 cm^-1 (23.5 ns). The `State` column (byte 1)
is the join key from each line back to its upper-level lifetime.

## Stated uncertainties
- Per-line `e_BF` (%) and `e_loggf` (dex) provided per row where measured.
- Level lifetimes: +/-5% (ReadMe Note G1).

## Parse-check (no transformation applied)
- 167 data records (35 `#` header/section lines excluded).
- **142 rows carry a this-work `loggf` value** (bytes 54-58 non-blank) — matches the
  abstract's "new experimental oscillator strengths for 142 transitions of Fe I".
- 12 tau-bearing upper levels documented in ReadMe Note (G1).
