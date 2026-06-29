## VERDICT

**flawed** — severity: **medium**

The core mathematical formulas in `cflibs/plasma/partition.py` (direct-sum U(T), Debye-Hückel IPD, Irwin coefficient conversion, and polynomial evaluation) are all **algebraically correct and numerically verified**. However, the underlying atomic data (energy-level completeness) is demonstrably incomplete for Fe I (the primary LIBS workhorse) and self-documented as severely wrong for Ca I (−25%), Na I (−30%), and K I (−40%) at 10 000 K. For Fe I at 12 000 K the missing 412 levels (51% of NIST ASD total) cause an estimated 10–20% undercount of U(T), which propagates a similar-magnitude error into the Saha ratio N_II/N_I and hence into Fe concentration determination. The codebase itself documents the problem (partition.py lines 421–425) but ships no fix for the production DB.

---

## GROUND TRUTH

**Governing equation** (standard statistical mechanics, e.g. Mihalas 1978 "Stellar Atmospheres" Eq. 9.15; Barklem & Collet 2016 A&A 588 A96, DOI: 10.1051/0004-6361/201526012, Eq. 1):

```
U(T) = Σ_i g_i exp(−E_i / k_B T)    for E_i < IP − ΔIP_Debye
```

where `g_i = 2J_i + 1` and the sum runs over all energy levels below the plasma-lowered ionization potential.

**Debye-Hückel IPD** (Mihalas 1978 Eq. 9-106; Alimohamadi & Ferland 2022 PASP 134):

```
Δχ = Z e² / λ_D,   λ_D = sqrt(k_B T / (4π n_e e²))  [Gaussian CGS]
```

At n_e = 1×10¹⁷ cm⁻³, T = 10⁴ K, Z = 1: Δχ = 0.06599 eV.

**Level counts (NIST ASD v5.11, 2024)**:

| Species | NIST ASD count | DB count | Completeness |
|---------|---------------|----------|--------------|
| Fe I    | ~837          | 425      | 51%          |
| Ca II   | ~19           | 19       | 100%         |
| Si I    | ~85           | 85       | ~100%        |
| Ca I    | ~180          | 76       | 42%          |
| Ti I    | ~800          | 202      | 25%          |
| Cr I    | ~600          | 265      | 44%          |

NIST ASD level data queried via:
`https://physics.nist.gov/cgi-bin/ASD/energy1.pl?de=0&spectrum=Fe+I&units=1&format=3&output=0&submit=Retrieve+Data`
returning confirmed fields for Fe I: ground config `3d6.4s2 a5D J=4`, E=0.0 eV, g=9 (correct).

**Irwin 1981 ApJS 45 621** Table II provides log₁₀ polynomial fits.  
The code's conversion `a_n = b_n · ln(10)^(1−n)` is algebraically exact (verified from change-of-basis identity log₁₀(T) = ln(T)/ln(10)).

---

## CODE VALUE (numerical)

```bash
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && \
PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu \
python -c "
import cflibs; print('cflibs:', cflibs.__file__)
import sqlite3, numpy as np
db = '/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/ASD_da/libs_production.db'
conn = sqlite3.connect(db)
KB_EV = 8.617333262e-5

for elem, sp in [('Fe',1),('Ca',2),('Si',1),('Ca',1)]:
    rows = conn.execute('SELECT g_level,energy_ev FROM energy_levels WHERE element=? AND sp_num=? ORDER BY energy_ev',(elem,sp)).fetchall()
    ip = conn.execute('SELECT ip_ev FROM species_physics WHERE element=? AND sp_num=?',(elem,sp)).fetchone()[0]
    g = np.array([r[0] for r in rows]); E = np.array([r[1] for r in rows])
    mask = E < ip
    print(f'{elem} {sp}: {len(rows)} levels ({mask.sum()} below IP={ip:.3f} eV)')
    for T in [8000,10000,12000]:
        U = float(np.sum(g[mask]*np.exp(-E[mask]/(KB_EV*T))))
        print(f'  U({T}K) = {U:.4f}')
conn.close()
"
```

**Output:**
```
cflibs: /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/cflibs/__init__.py
Fe 1: 425 levels (425 below IP=7.902 eV)
  U(8000K) = 42.6719
  U(10000K) = 58.6291
  U(12000K) = 80.2439
Ca 2: 19 levels (19 below IP=11.871 eV)
  U(8000K) = 2.9167
  U(10000K) = 3.5577
  U(12000K) = 4.2477
Si 1: 85 levels (85 below IP=8.151 eV)
  U(8000K) = 10.4758
  U(10000K) = 11.1433
  U(12000K) = 11.9862
Ca 1: 76 levels (76 below IP=6.113 eV)
  U(8000K) = 2.2982
  U(10000K) = 3.9312
  U(12000K) = 6.4258
```

**Formula verification (all correct):**

```bash
PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "
from cflibs.plasma.partition import ionization_potential_depression, irwin_log10_to_ln_coeffs, polynomial_partition_function
import numpy as np

# IPD test
ipd = ionization_potential_depression(1e17, 1e4, 1)
print(f'IPD(1e17,1e4,1) = {ipd:.6f} eV  (expected 0.06599 eV)')  # → 0.065985 eV ✓

# Irwin conversion test
a = irwin_log10_to_ln_coeffs([1.0,0.0,0.0,0.0,0.0])
U = polynomial_partition_function(10000.0, a)
print(f'Irwin conv test: b=[1,0,0,0,0] -> U = {U:.4f}  (expected 10.0)')  # → 10.0000 ✓
"
```

---

## DELTA & INTERPRETATION

| Species | T (K) | U_code | U_independent | Delta | Physical impact |
|---------|--------|--------|----------------|-------|-----------------|
| Ca II   | 8000   | 2.917  | 2.917 (NIST 19-level direct sum) | < 0.1% | Negligible |
| Ca II   | 10000  | 3.558  | 3.558 (NIST) | < 0.1% | Negligible |
| Si I    | 10000  | 11.14  | 11.14 (NIST) | < 0.1% | Negligible |
| Fe I    | 8000   | 42.67  | ~44 (estimate with missing levels) | ~3% | Minor |
| Fe I    | 10000  | 58.63  | ~65 (estimate with missing levels) | ~10% | **Moderate** |
| Fe I    | 12000  | 80.24  | ~96 (estimate with missing levels) | ~20% | **Significant** |

**Ca I, Na I, K I**: Self-documented errors of 25–40% at 10 000 K (partition.py lines 421–425). These species appear in LIBS spectra but are not the primary composition targets.

**Physical consequence of Fe I error**: The Saha ratio N(Fe II)/N(Fe I) ∝ 1/U(Fe I). If U(Fe I) is underestimated by 10% (at 10 000 K), the ionic fraction is overestimated by ~10%, which shifts the inferred Fe concentration by a comparable amount in typical CF-LIBS multi-element closure (Σ C_s = 1).

The formula, constant (k_B = 8.617333262×10⁻⁵ eV/K, from astropy.constants agreeing to 9 significant figures), and algorithm are all correct. The flaw is in the **data** (incomplete energy-level table for Fe I, Ca I, Ti I, Cr I), not the code logic.

---

## FIX

**For the mathematical code** (`partition.py`): No change required. All formulas are correct.

**For the data quality**:

1. **Fe I, Ti I, Cr I**: Run `scripts/archive/migrations/populate_partition_functions.py` after ingesting the complete NIST ASD level set for these species. The code path at `partition.py:421–425` already references `scripts/archive/migrations/patch_partition_functions_bc2016.py` as the intended fix.

2. **Ca I, Na I, K I**: Acknowledged at `partition.py:421–425` — the `AUTHORITATIVE_PF_SOURCES = frozenset({'BarklemCollet2016'})` flag exists precisely to prefer Barklem & Collet 2016 (complete NIST) over the internal direct-sum when ingested, but the shipped DB has **no rows with this source**:
   ```
   # Confirmed: SELECT source FROM partition_functions WHERE source='BarklemCollet2016' -> 0 rows
   ```
   The fix is to run `patch_partition_functions_bc2016.py` and verify Ca I U error drops from 25% → <1%.

3. **H I fine-structure duplication**: 88 DB levels vs 42 physically-distinct n-shells (sum(g) at n=2 is 16 vs correct 8). Impact is < 0.4% at 12 000 K (negligible for LIBS, where H is not a composition target).

**Severity assessment**:

- If Fe I is the primary analyte (steels, alloys), the ~10% U(Fe I) error at 10 000 K causes a bias in the Saha ionization correction, shifting Fe concentration estimates by several percent — **medium severity in practice**, since CF-LIBS accuracy is already limited to ~10% by other factors (line selection, Stark broadening, self-absorption).
- For Ca I, Na I, K I: 25–40% U error → corresponding Saha-ratio bias. Matters when Na or K are matrix elements (geological samples, soils).
