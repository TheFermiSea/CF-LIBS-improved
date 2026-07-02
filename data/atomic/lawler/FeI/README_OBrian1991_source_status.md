# O'Brian et al. 1991 (Fe I) — source status: NOT AVAILABLE on VizieR/CDS

## Paper (verified)
O'Brian T.R., Wickliffe M.E., Lawler J.E., Whaling W., Brault J.W.,
"Lifetimes, transition probabilities, and level energies in Fe I."
J. Opt. Soc. Am. B, 8(6), 1185-1201 (1991). Bibcode: **1991JOSAB...8.1185O**.
Landing: https://opg.optica.org/josab/abstract.cfm?uri=josab-8-6-1185 ;
ADS: https://ui.adsabs.harvard.edu/abs/1991JOSAB...8.1185O

Content (per abstract, verified via web search): radiative lifetimes for **186 Fe I
levels** (25900-60758 cm^-1, TR-LIF), branching fractions giving transition
probabilities for **1174 transitions** (225-2666 nm), plus **640** further transition
probabilities from ICP level-population interpolation. Total **1814** Fe I transitions.
(Matches the task lead: "1814 lines, BF+tau".)

## Why no raw file was staged
The machine-readable table is **NOT hosted on VizieR/CDS**. Verified 2026-07-02 by:
1. CDS FTP `J/JOSAB/8/1185/` and `J/JOSAB/8/` -> HTTP 404 (no such catalog tree).
2. VizieR ReadMe endpoint `J/JOSAB/8/1185` -> 500 (catalog does not exist).
3. VizieR VOTable meta query `-bibcode=1991JOSAB...8.1185O` -> empty `<RESOURCE>` (0 tables).
4. VizieR full-text keyword searches (O'Brian / Wickliffe / Whaling / Brault, "Fe I
   lifetimes transition probabilities") -> no matching catalog IDs.
5. CDS bibcode->catalog association -> no catalog.

This is consistent with the paper's 1991 vintage (pre-dates routine CDS deposition of
supplementary tables).

## Where O'Brian 1991 data DOES survive (for the Build phase, if needed)
- As **comparison columns** inside the two staged catalogs:
  - Den Hartog table4.dat: `log(gf)P` with `r_log(gf)P = OB91` / `Cal = P`.
  - Ruffoni table3.dat & table4.dat: `loggf0` with `r_loggf0 = O91`.
  These are only the subset of O'Brian lines overlapping those newer measurements — NOT
  the full 1814-line dataset.
- Incorporated into the NIST ASD via the Fuhr & Wiese (2006, J. Phys. Chem. Ref. Data 35,
  1669) critical compilation, and into the Kurucz Fe I line list. Both are IMMUTABLE
  standard DBs under the project mandate; no separate overlay ingest of O'Brian is possible
  from a CDS raw file because none exists.

## Action / honesty note
No O'Brian 1991 raw data file was downloaded because none is obtainable from VizieR/CDS.
No data was fabricated or substituted. If the full O'Brian table is required, it must be
sourced from the OSA journal supplementary/PDF (Table digitization) or read (read-only)
from the existing NIST ASD — outside the VizieR acquisition scope of this task.
