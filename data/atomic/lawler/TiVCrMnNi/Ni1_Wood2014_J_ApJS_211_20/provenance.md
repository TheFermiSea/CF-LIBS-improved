# Provenance — Ni I laboratory transition probabilities (Wood et al. 2014)

## Citation
Wood M.P., Lawler J.E., Sneden C., Cowan J.J. 2014,
"Improved Ni I log(gf) values and abundance determinations in the photospheres of
the Sun and metal-poor star HD 84937",
*Astrophys. J. Suppl. Ser.* **211**, 20.
Bibcode: 2014ApJS..211...20W.

## Source
- VizieR / CDS catalog **J/ApJS/211/20**.
- Download date: 2026-07-02.
- URLs:
  - ReadMe: https://cdsarc.cds.unistra.fr/ftp/J/ApJS/211/20/ReadMe
  - Data:   https://cdsarc.cds.unistra.fr/ftp/J/ApJS/211/20/table{1,2,3,4,5,7}.dat

## Files (downloaded UNMODIFIED)
- `table1.dat` (37 rows) — FTS observing log (spectra metadata).
- `table2.dat` (16 rows) — Echelle observing log (spectra metadata).
- `table3.dat` (371 rows) — **Primary lab data.** Atomic transition probabilities
  (A and log gf) for **371 lines of Ni I**, by increasing air wavelength.
- `table4.dat` (76 rows) — Solar photospheric Ni abundances from individual lines.
- `table5.dat` (77 rows) — Ni abundances in HD 84937 from individual lines.
- `table7.dat` (303 rows) — Shifted isotopic wavelengths (^58^Ni, ^60^Ni) for
  303 Ni I lines.
- `ReadMe` — CDS byte-by-byte description.

## Column meanings / units — table3.dat (lab transition probabilities)
| Bytes | Label | Units | Meaning |
|-------|-------|-------|---------|
| 1-9   | lambda | 0.1 nm | Air wavelength |
| 11-19 | E1     | cm^-1  | Upper level energy |
| 21-22 | P1     | ---    | Upper level parity (ev/od) |
| 24    | J1     | ---    | Upper level J |
| 26-34 | E0     | cm^-1  | Lower level energy |
| 36-37 | P0     | ---    | Lower level parity (ev/od) |
| 39    | J0     | ---    | Lower level J |
| 41-48 | A      | 10^6 s^-1 | Transition probability |
| 50-56 | e_A    | 10^6 s^-1 | Total uncertainty in A |
| 58-62 | log(gf)| dex    | log of degeneracy x oscillator strength |

## Stated uncertainties
Per-line total uncertainty on A in `e_A` (10^6 s^-1). A = measured branching
fraction x published radiative lifetime; the radiative lifetimes are from prior
work and are NOT tabulated in this catalog (no per-level tau column).

## Notes
- 371 lab-gf lines (table3), all carry log(gf) (parse-check: 0 failures).
- No tau-per-level table in this catalog.
