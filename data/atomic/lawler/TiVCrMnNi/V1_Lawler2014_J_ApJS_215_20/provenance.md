# Provenance — V I laboratory transition probabilities (Lawler et al. 2014)

## Citation
Lawler J.E., Wood M.P., Den Hartog E.A., Feigenson T., Sneden C., Cowan J.J. 2014,
"Improved V I log(gf) values and abundance determinations in the photospheres of
the Sun and metal-poor star HD 84937",
*Astrophys. J. Suppl. Ser.* **215**, 20.
Bibcode: 2014ApJS..215...20L.

## Source
- VizieR / CDS catalog **J/ApJS/215/20**.
- Download date: 2026-07-02.
- URLs:
  - ReadMe: https://cdsarc.cds.unistra.fr/ftp/J/ApJS/215/20/ReadMe
  - Data:   https://cdsarc.cds.unistra.fr/ftp/J/ApJS/215/20/table{1,2,3,4,5}.dat

## Files (downloaded UNMODIFIED)
- `table1.dat` (14 rows) — FTS observing log (spectra metadata).
- `table2.dat` (19 rows) — Echelle observing log (spectra metadata).
- `table3.dat` (836 rows) — **Primary lab data.** Experimental atomic transition
  probabilities (A and log gf) for **836 lines of V I**, by increasing air wavelength.
- `table4.dat` (26 rows) — Hyperfine-structure (HFS) A coefficients for 25 odd- +
  1 even-parity levels of neutral V (HFS constants, NOT radiative lifetimes).
- `table5.dat` (1480 rows) — HFS line-component patterns for 94 ^51^V I transitions.
- `ReadMe` — CDS byte-by-byte description.

## Column meanings / units — table3.dat (lab transition probabilities)
| Bytes | Label | Units | Meaning |
|-------|-------|-------|---------|
| 1-9   | lamAir | 0.1 nm | Air wavelength (Peck & Reeder 1972) |
| 11-19 | E1     | cm^-1  | Upper level energy (Thorne+ 2011) |
| 21    | P1     | ---    | Upper level parity (o/e) |
| 24-26 | J1     | ---    | Upper level J |
| 28-36 | E0     | cm^-1  | Lower level energy (Thorne+ 2011) |
| 38    | P0     | ---    | Lower level parity (o/e) |
| 41-43 | J0     | ---    | Lower level J |
| 45-52 | A      | 10^6 s^-1 | Transition probability |
| 54-60 | e_A    | 10^6 s^-1 | Total uncertainty in A |
| 62-66 | log(gf)| dex    | log of degeneracy x oscillator strength |

## Column meanings / units — table4.dat (HFS A coefficients, not lifetimes)
Config (1-21), Term (23-33), 2J (35-36), E cm^-1 (40-48), A mK (50-54, HFS A
constant; +/-1 mK typical). These are magnetic-dipole hyperfine constants, NOT tau.

## Stated uncertainties
table3: per-line total uncertainty on A in `e_A` (10^6 s^-1). A = branching
fraction x published (LIF) radiative lifetime; the lifetimes themselves are from
prior laser-induced-fluorescence work and are NOT tabulated in this catalog
(no per-level tau column).

## Notes
- 836 lab-gf lines (table3), all carry log(gf) (parse-check: 0 failures).
- No tau-per-level table; table4 is HFS A constants (mK), not lifetimes.
