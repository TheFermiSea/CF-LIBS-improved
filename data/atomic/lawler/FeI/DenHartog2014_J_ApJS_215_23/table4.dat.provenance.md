# Provenance: Den Hartog et al. 2014 — table4.dat (Fe I lab log gf / BF / A)

## Citation
Den Hartog E.A., Ruffoni M.P., Lawler J.E., Pickering J.C., Lind K., Brewer N.R.,
"Fe I oscillator strengths for transitions from high-lying even-parity levels."
Astrophys. J. Suppl. Ser., 215, 23 (2014). Bibcode: 2014ApJS..215...23D.
DOI (paper): 10.1088/0067-0049/215/2/23.

## VizieR / CDS catalog
- Catalog ID: **J/ApJS/215/23**
- File: `table4.dat` — "Experimental branching fractions, transition probabilities,
  and log(gf)s for 227 lines of Fe I, organized by increasing wavelength"
- Download URL: https://cdsarc.cds.unistra.fr/ftp/J/ApJS/215/23/table4.dat
- ReadMe: https://cdsarc.cds.unistra.fr/ftp/J/ApJS/215/23/ReadMe
- Downloaded (UTC): 2026-07-02
- Records: 227 (fixed-width, Lrecl=82).  md5: 4dcb7c3c5a73874a53a12633bfab76d5  (17333 bytes)

## Column meanings / units (byte-by-byte, from ReadMe — verbatim)
| Bytes | Fmt  | Units | Label      | Meaning |
|-------|------|-------|------------|---------|
| 1-10  |F10.4 |0.1nm  | lamAir     | Wavelength in air, Angstrom [3211/22381] (mostly Nave+ 1994) |
| 12-20 |F9.3  |cm-1   | E1         | Upper energy level [45061.3/53169.2] |
| 22    |I1    |---    | J1         | Upper level J [0/7] |
| 24-32 |F9.3  |cm-1   | E0         | Lower energy level [19350.8/43499.6] |
| 34    |I1    |---    | J0         | Lower level J [0/6] |
| 36-40 |F5.3  |---    | BF         | Branching fraction [0.001/1] (optional) |
| 42-45 |F4.1  |%      | e_BF       | BF uncertainty, percent [0/40] (optional) |
| 47-53 |F7.3  |10+6/s | A          | Transition probability A_ki [0.04/156.3] (optional) |
| 55-59 |F5.2  |[-]    | log(gf)    | **This-work log(gf)** [-3.4/0.7] (optional) |
| 61-64 |F4.2  |[-]    | e_log(gf)  | Uncertainty in this-work log(gf) (optional) |
| 66-70 |F5.2  |[-]    | log(gf)P   | Previously published log(gf) (Fuhr & Wiese 2006 compilation) |
| 72-75 |F4.2  |[-]    | e_log(gf)P | Uncertainty in log(gf)P (optional) |
| 77-80 |A4    |---    | r_log(gf)P | Ref for log(gf)P: OB91=O'Brian+1991; Ma74=May+1974; Ba91=Bard+1991; Ba94=Bard&Kock1994; Br74=Bridges&Kornblith1974 |
| 82    |A1    |---    | Cal        | Calibration method for published data: L / A / P (P=from O'Brian+1991) |

## Stated uncertainties
- Per-line `e_BF` (%) and `e_log(gf)` (dex) are provided per row where measured.
- Lifetimes underlying these gf values are +/-5% (see table1.dat).

## Parse-check (no transformation applied)
- 227 fixed-width records total.
- **203 rows carry a this-work `log(gf)` value** (bytes 55-59 non-blank) — matches the
  abstract's "new oscillator strengths for 203 lines of Fe I".
- Remaining 24 rows have only published (log(gf)P) comparison values, no this-work gf.
- The O'Brian 1991 (OB91/P) values appear here ONLY as the `log(gf)P` comparison column
  for a subset of lines — NOT the full O'Brian dataset (see FeI/README_OBrian1991_source_status.md).
