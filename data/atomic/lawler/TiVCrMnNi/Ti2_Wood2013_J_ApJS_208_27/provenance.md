# Provenance — Ti II laboratory transition probabilities (Wood et al. 2013)

## Citation
Wood M.P., Lawler J.E., Sneden C., Cowan J.J. 2013,
"Improved Ti II log(gf) values and abundance determinations in the photospheres
of the Sun and metal-poor star HD 84937",
*Astrophys. J. Suppl. Ser.* **208**, 27.
Bibcode: 2013ApJS..208...27W.

## Source
- VizieR / CDS catalog **J/ApJS/208/27**.
- Download date: 2026-07-02.
- URLs:
  - ReadMe:  https://cdsarc.cds.unistra.fr/ftp/J/ApJS/208/27/ReadMe
  - Data:    https://cdsarc.cds.unistra.fr/ftp/J/ApJS/208/27/table{3,4,5,6}.dat
  - Landing: https://cdsarc.cds.unistra.fr/viz-bin/cat/J/ApJS/208/27

## Files (downloaded UNMODIFIED)
- `table3.dat` (364 rows) — Comparison of experimental **branching fractions**
  from FTS data for 364 Ti II lines from upper odd-parity levels.
- `table4.dat` (364 rows) — **Primary lab data.** Experimental atomic transition
  probabilities (A and log gf) for 364 Ti II lines, by increasing air wavelength.
- `table5.dat` (43 rows) — Solar photospheric Ti abundances from individual Ti II
  lines (abundance analysis, not lab transition data).
- `table6.dat` (147 rows) — Ti abundances in HD 84937 from individual Ti II lines.
- `ReadMe` — CDS byte-by-byte description.

## Column meanings / units — table4.dat (lab transition probabilities)
| Bytes | Label | Units | Meaning |
|-------|-------|-------|---------|
| 1-9   | lamAir | 0.1 nm | Air wavelength (Peck & Reeder 1972) |
| 11-19 | E1     | cm^-1  | Upper level energy (Saloman 2012) |
| 21-23 | J1     | ---    | Upper level J |
| 25-33 | E0     | cm^-1  | Lower level energy (Saloman 2012) |
| 35-37 | J0     | ---    | Lower level J |
| 39-46 | A      | 10^6 s^-1 | Transition probability |
| 48-54 | e_A    | 10^6 s^-1 | Total uncertainty in A |
| 56-60 | log(gf)| dex    | log of degeneracy x oscillator strength |

## Column meanings / units — table3.dat (branching fractions)
lamAir (1-9, 0.1nm), E1 (11-19, cm^-1), J1 (21-23), E0 (25-33, cm^-1), J0 (35-37);
BF-P01 (39-45) + e_BF-P01 % (47-48) = Pickering+ 2001; BF-B93 (50-57) + e_BF-B93 %
(59-60) = Bizzarri+ 1993; **BF (62-68) + e_BF % (70-71) = this experiment**.

## Stated uncertainties
table4: per-line total uncertainty on A in `e_A` (10^6 s^-1). table3: branching
fraction uncertainties are percentages (`e_BF`). A = BF x (published radiative
lifetime); lifetimes were taken from prior work and are NOT tabulated here (no
per-level tau column).

## Notes
- 364 lab-gf lines (table4), all carry log(gf) (parse-check: 0 failures).
- 364 branching-fraction rows (table3, "this experiment" BF column all populated).
- No tau-per-level table in this catalog.
