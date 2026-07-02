# Provenance — Ti I laboratory transition probabilities (Lawler et al. 2013)

## Citation
Lawler J.E., Guzman A., Wood M.P., Sneden C., Cowan J.J. 2013,
"Improved log(gf) Values for Lines of Ti I and Abundance Determinations in the
Photospheres of the Sun and Metal-Poor Star HD 84937",
*Astrophys. J. Suppl. Ser.* **205**, 11.
Bibcode: 2013ApJS..205...11L. DOI: 10.1088/0067-0049/205/2/11.

## Source
- **NOT available on VizieR/CDS.** VizieR has no catalog `J/ApJS/205/11`
  (confirmed: `asu-tsv` returns "Table or Catalog not found"; the CDS J.ApJS
  index lists J/ApJS/205/{1,5,6,9,13,14} but no /11). Only the Ti II companion
  (Wood+ 2013, J/ApJS/208/27) was ingested.
- Obtained instead from the AAS/IOP electronic-edition **machine-readable
  tables (MRT)**. The article page is marked "Free article".
- Download date: 2026-07-02.
- URLs (IOP content delivery, MRT plain text):
  - Table 3: https://content.cld.iop.org/journals/0067-0049/205/2/11/revision1/apjs461528t3_mrt.txt
  - Table 4: https://content.cld.iop.org/journals/0067-0049/205/2/11/revision1/apjs461528t4_mrt.txt
  - Table 5: https://content.cld.iop.org/journals/0067-0049/205/2/11/revision1/apjs461528t5_mrt.txt
  - Article: https://iopscience.iop.org/article/10.1088/0067-0049/205/2/11

## Files (downloaded UNMODIFIED)
- `apjs461528t3_mrt.txt` — **primary lab data.** Experimental atomic transition
  probabilities for **948 lines of Ti I** from upper odd-parity levels, ordered
  by increasing air wavelength. 948 data rows (23-line MRT header block).
- `apjs461528t4_mrt.txt` — Solar photospheric Ti abundances from individual
  Ti I lines (not lab transition data; abundance analysis).
- `apjs461528t5_mrt.txt` — HD 84937 Ti abundances from individual Ti I lines
  (abundance analysis).

## Column meanings / units — Table 3 (`apjs461528t3_mrt.txt`)
Fixed-width; byte ranges from the MRT byte-by-byte header:
| Bytes | Label | Units | Meaning |
|-------|-------|-------|---------|
| 1-10  | WaveAir | 0.1 nm (Angstrom) | Air wavelength (from energy levels; Peck & Reeder 1972 index of air) |
| 12-20 | UpLev  | cm^-1 | Upper level energy (Saloman 2012) |
| 22    | UpJ    | ---   | Upper level J |
| 24-32 | LowLev | cm^-1 | Lower level energy (Saloman 2012) |
| 34    | LowJ   | ---   | Lower level J |
| 36-43 | TranP  | 10^6 s^-1 | Transition probability A_ki |
| 45-51 | e_TranP| 10^6 s^-1 | Total uncertainty in A_ki |
| 53-57 | log(gf)| dex   | log of degeneracy x oscillator strength |

## Stated uncertainties
Per-line total uncertainty on A given in column `e_TranP` (same units as A).
The paper derives A from measured branching fractions x published radiative
lifetimes; log(gf) inherits the A uncertainty. No separate per-level radiative
lifetime (tau) table is included in the MRT (lifetimes are folded into A).

## Notes
- 948 lab-gf lines; all rows carry a log(gf) value (parse-check: 0 failures).
- No tau-per-level column in any machine-readable table.
