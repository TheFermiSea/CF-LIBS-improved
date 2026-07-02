# Provenance: Den Hartog et al. 2014 — table1.dat (Fe I radiative lifetimes)

## Citation
Den Hartog E.A., Ruffoni M.P., Lawler J.E., Pickering J.C., Lind K., Brewer N.R.,
"Fe I oscillator strengths for transitions from high-lying even-parity levels."
Astrophys. J. Suppl. Ser., 215, 23 (2014). Bibcode: 2014ApJS..215...23D.
DOI (paper): 10.1088/0067-0049/215/2/23. DOI (CDS): 10.26093/cds/vizier.22150023.

## VizieR / CDS catalog
- Catalog ID: **J/ApJS/215/23**  (catid 22150023)
- File: `table1.dat` — "Radiative lifetimes for even parity Fe I levels"
- Download URL: https://cdsarc.cds.unistra.fr/ftp/J/ApJS/215/23/table1.dat
- ReadMe: https://cdsarc.cds.unistra.fr/ftp/J/ApJS/215/23/ReadMe
- Downloaded (UTC): 2026-07-02
- Records: 33 (fixed-width, Lrecl=88).  md5: a0857137baf616314d08d2182ce0f3d1  (2907 bytes)

## Column meanings / units (byte-by-byte, from ReadMe — verbatim)
| Bytes | Fmt | Units | Label   | Meaning |
|-------|-----|-------|---------|---------|
| 1     | I1  | ---   | S       | [1/2] Step-excitation number: 1=two-step, 2=single-step TR-LIF |
| 3-24  | A22 | ---   | Config  | Configuration (from Nave+ 1994, J/ApJS/94/221) |
| 26-30 | A5  | ---   | Term    | Term |
| 32    | I1  | ---   | J       | [0/7] J value |
| 34-42 | F9.3| cm-1  | E1      | Upper energy level [45061.3/56842.8] |
| 44-52 | F9.3| cm-1  | E0      | Intermediate energy level (two-step only; may be blank) |
| 54-60 | F7.3| nm    | lam1    | Step-1 laser wavelength |
| 62-68 | F7.3| nm    | lam2    | Step-2 laser wavelength (two-step only; may be blank) |
| 70-72 | I3  | nm    | lambda  | Observation (fluorescence) wavelength |
| 74-77 | F4.1| ns    | tau     | **This-work radiative lifetime** (may be blank on continuation rows) |
| 79-82 | F4.1| ns    | tauP    | Previously published lifetime (optional) |
| 84-86 | F3.1| ns    | e_tauP  | Uncertainty on tauP (optional) |
| 88    | A1  | ---   | r_tauP  | Reference for tauP: d=Marek+1979; e=O'Brian+1991 (TR-LIF, +/-5%) |

## Stated uncertainties
- This-work lifetimes (`tau`): **+/-5%** (single value stated in abstract/text for all levels).

## Parse-check (no transformation applied)
- 33 fixed-width records; 30 **unique** upper levels (E1), all 30 carry a this-work `tau`.
- 3 levels (E1 = 50342.126, 50534.394, 56842.729 cm^-1) appear on two rows; the 2nd
  row is a continuation with blank `tau` (the value is on the 1st row).
- NOTE / discrepancy: the paper **abstract states 31 even-parity levels**; the VizieR
  `table1.dat` contains **30 distinct tau-bearing levels**. Reported here as-is (30);
  the 31st appears absent from the machine-readable table. No data fabricated.
