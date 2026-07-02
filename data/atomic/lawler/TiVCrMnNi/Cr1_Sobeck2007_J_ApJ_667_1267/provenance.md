# Provenance — Cr I laboratory transition probabilities (Sobeck et al. 2007)

## Citation
Sobeck J.S., Lawler J.E., Sneden C. 2007,
"Improved laboratory transition probabilities for neutral chromium and
redetermination of the chromium abundance for the Sun and three stars",
*Astrophys. J.* **667**, 1267-1282.
Bibcode: 2007ApJ...667.1267S.

## Source
- VizieR / CDS catalog **J/ApJ/667/1267**.
- Download date: 2026-07-02.
- URLs:
  - ReadMe: https://cdsarc.cds.unistra.fr/ftp/J/ApJ/667/1267/ReadMe
  - Data:   https://cdsarc.cds.unistra.fr/ftp/J/ApJ/667/1267/table3.dat

## Files (downloaded UNMODIFIED)
- `table3.dat` (263 rows) — **Primary lab data.** Atomic transition probabilities
  for **263 lines of Cr I**, by increasing air wavelength.
- `ReadMe` — CDS byte-by-byte description.

## Column meanings / units — table3.dat
| Bytes | Label | Units | Meaning |
|-------|-------|-------|---------|
| 1-7   | lambda | 0.1 nm | Air wavelength |
| 9-16  | EU     | cm^-1  | Upper level energy |
| 18-25 | TermU  | ---    | Upper level term |
| 27    | JU     | ---    | Upper level J |
| 29-36 | EL     | cm^-1  | Lower level energy |
| 38-45 | TermL  | ---    | Lower level term |
| 47    | JL     | ---    | Lower level J |
| 49    | N      | ---    | "N" = normalization line for LS calculation |
| 51-57 | Acalc  | 10^6 s^-1 | A from LS calculation (may be blank) |
| 59-65 | Aexp   | 10^6 s^-1 | A from experiment |
| 67-72 | e_Aexp | 10^6 s^-1 | Total uncertainty in Aexp |
| 74-78 | log(gf)| dex    | log of degeneracy x oscillator strength |

## Stated uncertainties
Per-line total uncertainty on the experimental A in `e_Aexp` (10^6 s^-1).
A_exp = measured branching fraction x published radiative lifetime; the radiative
lifetimes are from prior work and are NOT tabulated in this catalog (no per-level
tau column).

## Notes
- 263 lab-gf lines, all carry log(gf) (parse-check: 0 failures).
- `Acalc` (LS-calculated A) is blank for some rows; `Aexp` is populated.
- No tau-per-level table in this catalog.
- The FTP directory also contains `readme.bad` (a CDS internal artifact); not a
  data file and intentionally not staged.
