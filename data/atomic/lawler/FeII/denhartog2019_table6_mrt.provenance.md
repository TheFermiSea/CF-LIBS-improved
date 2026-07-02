# Provenance: denhartog2019_table6_mrt.txt

## Paper / citation
Den Hartog, E. A.; Lawler, J. E.; Sneden, C.; Cowan, J. J.; Brukhovesky, A.
"Atomic Transition Probabilities for UV and Blue Lines of Fe II and Abundance
Determinations in the Photospheres of the Sun and Metal-poor Star HD 84937."
The Astrophysical Journal Supplement Series, 243:33 (2019).
DOI: 10.3847/1538-4365/ab322e
Bibcode: 2019ApJS..243...33D
arXiv: 1907.11760

This is **Table 6** of that paper: "Experimental Atomic Transition Probabilities
for 131 lines of Fe II." log gf values are derived from FTS branching fractions
(BFs) combined with radiative lifetimes measured by laser-induced fluorescence
(LIF) taken from the literature (adopted lifetimes are listed in the paper's
Table 5; see the PDF provenance file).

## Source / catalog id
NOT a VizieR/CDS catalog. This dataset is **not present on VizieR** (verified:
no `J/ApJS/243/33/` directory at CDS, and the VizieR bibcode resolver returns
"No catalogue or table was specified or found" for 2019ApJS..243...33D).
The machine-readable table (MRT) is hosted by AAS/IOP as the article's
supplementary data.

- AAS machine-readable table id: apjsab322et6_mrt.txt (Table 6)
- Download URL:
  https://content.cld.iop.org/journals/0067-0049/243/2/33/revision1/apjsab322et6_mrt.txt
- Download date (UTC): 2026-07-02
- SHA256: ccbfc98da8999b1ea665aea5b11fbbc3d3a691e57b6a42f12f13b87a55b66453
- Bytes: 10143 ; total lines 154 (23-line AAS MRT header + 131 data rows)

## Column meanings, units, uncertainties (from the file's byte-by-byte header)
Fixed-width AAS MRT. Wavelengths and energy levels are from Nave & Johansson
(2013) [2013ApJS..204....1N].

| Bytes  | Fmt   | Units    | Label     | Meaning                                             |
|--------|-------|----------|-----------|-----------------------------------------------------|
| 1-8    | F8.3  | 0.1 nm   | WaveAir   | Wavelength in air (Angstrom)                        |
| 10-19  | F10.4 | cm^-1    | UpLev     | Upper level energy                                  |
| 21-23  | F3.1  | ---      | UpJ       | Upper level J                                       |
| 25-34  | F10.4 | cm^-1    | LowLev    | Lower level energy                                  |
| 36-38  | F3.1  | ---      | LowJ      | Lower level J                                       |
| 40-46  | F7.3  | 10^6 /s  | TranP     | Transition probability (Einstein A_ki)              |
| 48-53  | F6.3  | 10^6 /s  | e_TranP   | Total uncertainty in TranP                          |
| 55-59  | F5.2  | [-]      | log(gf)   | log10 of degeneracy x oscillator strength           |
| 61-64  | F4.2  | [-]      | e_log(gf) | Uncertainty in log(gf)                              |

## Parse-check (no transformation applied)
- 131 data rows parsed cleanly (fixed-width); 0 bad rows.
- log(gf) parsed for all 131 lines.
- Distinct upper levels (E_U, J_U) present in file = 25 (matches the 25 adopted
  radiative lifetimes in Table 5 -> these 25 levels are the tau-anchors).
- Wavelength range: 2249.178 - 4583.829 Angstrom (air).
- Upper-level terms covered: z6Do, z6Fo, z6Po, z4Fo, z4Do, z4Po of 3d6(5D)4p.

## Notes for Build phase
- These 131 lines are the genuinely lab-measured Fe II log gf (LIF tau x FTS BF).
- File is UNMODIFIED as downloaded. Do not edit; normalize in Build.
