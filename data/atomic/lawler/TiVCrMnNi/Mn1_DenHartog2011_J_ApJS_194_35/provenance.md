# Provenance — Mn I / Mn II laboratory transition probabilities (Den Hartog et al. 2011)

## Citation
Den Hartog E.A., Lawler J.E., Sobeck J.S., Sneden C., Cowan J.J. 2011,
"Improved log(gf) values of selected lines in Mn I and Mn II for abundance
determinations in FGK dwarfs and giants",
*Astrophys. J. Suppl. Ser.* **194**, 35.
Bibcode: 2011ApJS..194...35D.

## Source
- VizieR / CDS catalog **J/ApJS/194/35**.
- Download date: 2026-07-02.
- URLs:
  - ReadMe: https://cdsarc.cds.unistra.fr/ftp/J/ApJS/194/35/ReadMe
  - Data:   https://cdsarc.cds.unistra.fr/ftp/J/ApJS/194/35/table{3,4}.dat

## Files (downloaded UNMODIFIED)
- `table3.dat` (57 rows) — **Primary lab data.** Recommended atomic transition
  probabilities for Mn I (half-integral J) and Mn II (integral J), by increasing
  air wavelength. 42 rows are Mn I; 15 rows are Mn II (split by J parity).
- `table4.dat` (753 rows) — HFS line-component patterns for ^55^Mn I / ^55^Mn II
  (hyperfine structure, not lifetimes).
- `ReadMe` — CDS byte-by-byte description.

## Column meanings / units — table3.dat
| Bytes | Label | Units | Meaning |
|-------|-------|-------|---------|
| 1-7   | lamAir | 0.1 nm | Air wavelength |
| 9-16  | E1     | cm^-1  | Upper level energy |
| 18-19 | P1     | ---    | Upper level parity |
| 21-23 | J1     | ---    | Upper level J (half-integer -> Mn I; integer -> Mn II) |
| 25-32 | E0     | cm^-1  | Lower level energy |
| 34-35 | P0     | ---    | Lower level parity |
| 37-39 | J0     | ---    | Lower level J |
| 41-46 | TranP  | 10^6 s^-1 | Recommended transition probability |
| 48-51 | e_TranP| 10^6 s^-1 | Total uncertainty in TranP |
| 53-58 | log(gf)| dex    | Recommended log(gf) |
| 60-64 | BWlog(gf)| dex  | log(gf) from Blackwell-Whitehead+ 2005 (may be blank) |
| 66-69 | e_BWlog(gf)| dex| uncertainty in BWlog(gf) (may be blank) |
| 71-76 | TElog(gf)| dex  | log(gf), this experiment |
| 78-82 | e_TElog(gf)| dex| uncertainty in TElog(gf) |

## Stated uncertainties
Per-line: `e_TranP` (10^6 s^-1) on A, and `e_TElog(gf)` / `e_BWlog(gf)` (dex) on
the log(gf) variants. Recommended A = new branching fraction x new radiative
lifetime.

## IMPORTANT — radiative lifetimes (tau)
The paper text reports NEW time-resolved LIF radiative lifetimes for 22 Mn I
levels (e8D, z6P, z6D, z4F, e8S, e6S terms) and 6 Mn II levels (z5P, z7P) — but
these per-level lifetimes are in the article's tables (Table 1 / Table 2) that
were NOT published as machine-readable data at CDS. The CDS catalog only ships
table3 (transition probabilities) and table4 (HFS patterns). Therefore this
staging directory contains **0 machine-readable tau-per-level rows**; the 28
lifetimes would have to be transcribed from the article PDF if needed.

## Notes
- 57 lab-gf lines total (42 Mn I + 15 Mn II), all carry log(gf) (parse-check: 0
  failures).
- Group target is Mn I (42 lines); Mn II (15 lines) is a bonus in the same table.
