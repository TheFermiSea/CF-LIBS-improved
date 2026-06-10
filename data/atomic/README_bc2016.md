# Barklem & Collet (2016) atomic partition functions — Table 8

`bc2016_table8.dat` is the **Oct–Nov 2022 bug-fixed** version
(`table8_vNov2022.dat`) of Table 8 from:

> Barklem, P. S. & Collet, R. (2016), *Partition functions and equilibrium
> constants for diatomic molecules and atoms of astrophysical interest*,
> A&A 588, A96, DOI [10.1051/0004-6361/201526961](https://doi.org/10.1051/0004-6361/201526961)

Downloaded 2026-06-09 from the authors' public repository:
<https://github.com/barklem/public-data/tree/master/partition-functions_and_equilibrium-constants>
(file `table8_vNov2022.dat`). The CDS copy (catalog `J/A+A/588/A96`,
`table8.dat`) carries the **original 2016 values**, which contain a
J-parsing bug fixed in the Nov 2022 revision — Na I was **17.5 % too high**
at 10 000 K, Tl I 4.5 %; all other species <1 % (see
`ReadMe_Bug_fix_Oct-Nov2022` in the authors' repository). The bug-fixed
file is the one used here.

Format: 284 species (rows, named `El_Stage` e.g. `Fe_II`, plus D/anions),
partition function U at 42 temperatures from 1e-5 K to **10 000 K**
(columns; grid in the third header line). Note the grid tops out at
10 000 K (~0.86 eV) — consumers needing higher LIBS temperatures must
extrapolate (see `scripts/patch_partition_functions_bc2016.py` for the
documented effective-level extension used by this project).

Consumed by `scripts/patch_partition_functions_bc2016.py` (bead
CF-LIBS-improved-16m7) to refit the `partition_functions` polynomial rows
of the production atomic database.
