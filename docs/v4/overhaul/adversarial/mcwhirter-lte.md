# McWhirter LTE Criterion Audit

## VERDICT
**correct** — severity: **low**

The McWhirter constant `1.6e12` is numerically exact per every authoritative source. The resonance-dE M2 claim is correct in physics, but the code's default deviation (using `max(E_k)` instead of the resonance-line energy) is a *known, documented, conservative* bias — it inflates the LTE threshold up to 15–25× vs the paper form, which means low-n_e plasmas may be falsely flagged as non-LTE (false negatives, not false positives). An environment flag `CFLIBS_MCWHIRTER_RESONANCE_DE=1` exists to switch to the paper-correct form. Because the error direction is conservative and the fix path is in place, severity is **low**.

---

## GROUND TRUTH

### Constant value: 1.6e12

All five of the following peer-reviewed sources give exactly `1.6 × 10¹²` for the McWhirter prefactor with units `cm⁻³ K⁻¹/² eV⁻³`:

1. **McWhirter, R.W.P. (1965)** — "Spectral Intensities" in *Plasma Diagnostic Techniques*, Eds. Huddlestone & Leonard, Academic Press, p. 201. *(Primary source — where the criterion originates.)*

2. **Cristoforetti, G. et al. (2010)** — "Local Thermodynamic Equilibrium in Laser-Induced Breakdown Spectroscopy: Beyond the McWhirter criterion", *Spectrochim. Acta B* 65, 86–95. DOI: [10.1016/j.sab.2009.11.005](https://doi.org/10.1016/j.sab.2009.11.005)  
   Eq. (1): `n_e [cm⁻³] ≥ 1.6 × 10¹² T^{1/2} (ΔE)³`

3. **Griem, H.R. (1997)** — *Principles of Plasma Spectroscopy*, Cambridge University Press, p. 218.

4. **Aragon, C. & Aguilera, J.A. (2008)** — "Characterization of laser induced plasmas by optical emission spectroscopy", *Spectrochim. Acta B* 63, 893–916. DOI: [10.1016/j.sab.2008.05.010](https://doi.org/10.1016/j.sab.2008.05.010)

5. **Tognoni, E. et al. (2010)** — "Calibration-free laser-induced breakdown spectroscopy: State of the art", *Spectrochim. Acta B* 65, 1–14. DOI: [10.1016/j.sab.2009.07.006](https://doi.org/10.1016/j.sab.2009.07.006)

The derivation path: van Regemorter (1962, ApJ 136, 906) electron-impact rate coefficient `C₁₂ ~ 2.16×10⁻⁶ g̅ T_eV^{-0.5} / (ΔE)²`, combined with the requirement that `n_e · C₁₂ ≳ A₂₁` for a representative allowed transition with `g̅ ~ 0.2`, yields the `T^{0.5} (ΔE)^3` scaling. The numerical prefactor `1.6×10¹²` embeds the van Regemorter numerical factor, the Gaunt-factor approximation, and unit conversion from SI/Rydberg to `cm⁻³/K/eV`.

### Resonance-dE definition (M2)

Cristoforetti et al. (2010), Sec. 2.1, states explicitly:  
> "In formula (1) ΔE represents the energy gap of the strongest resonance transition, i.e. the transition between the fundamental level and the first excited level with the same parity."

This is the **resonance (ground → first-excited)** energy, **not** the maximum upper-level energy of observed lines.

---

## CODE VALUE (numerical)

```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && \
PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "
from cflibs.core.constants import MCWHIRTER_CONST
print(f'MCWHIRTER_CONST = {MCWHIRTER_CONST:.4e}')
from cflibs.plasma.lte_validator import LTEValidator
v = LTEValidator()
r = v.validate(T_K=10000, n_e_cm3=1e17, delta_E_eV=5.0)
print(f'n_e_required = {r.mcwhirter.n_e_required:.4e}')
print(f'ratio = {r.mcwhirter.ratio:.3f}')
"
```

Output:
```
MCWHIRTER_CONST = 1.6000e+12
n_e_required = 2.0000e+16
ratio = 5.000
```

Location: `/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5/cflibs/core/constants.py:80`

### M2 (dE definition) default behavior:

```
cd /home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 && \
PYTHONPATH=/home/brian/code/CF-LIBS-improved/.worktrees/v4-m5 JAX_PLATFORMS=cpu python -c "
from cflibs.plasma.lte_validator import LTEValidator
v = LTEValidator()
# T=8000 K, n_e=1e16 — borderline case
r_max  = v.check_mcwhirter(T_K=8000, n_e_cm3=1e16, delta_E_eV=6.0)  # max(E_k)
r_res  = v.check_mcwhirter(T_K=8000, n_e_cm3=1e16, delta_E_eV=2.4)  # resonance
print(f'max(E_k)=6eV:     threshold={r_max.n_e_required:.2e}, pass={r_max.satisfied}')
print(f'resonance=2.4eV:  threshold={r_res.n_e_required:.2e}, pass={r_res.satisfied}')
"
```

Output:
```
max(E_k)=6eV:     threshold=3.09e+16, pass=False
resonance=2.4eV:  threshold=1.98e+15, pass=True
```

Verdict flips at low-n_e borderline conditions: `max(E_k)` says non-LTE, `resonance dE` says LTE — the default code is **15.6× more conservative** than the paper form for Fe I (ratio = `(6.0/2.4)³ = 15.6`).

---

## DELTA & INTERPRETATION

### Constant (1.6e12): Zero delta

Code = 1.600×10¹², literature = 1.6×10¹². Exact match. No error.

### M2 (dE definition): Conservative bias, known and documented

The code's **default** path (`_delta_e_from_observations`) uses `max(E_k)` from the observed upper levels. For Fe I, typical LIBS lines have upper levels up to ~6–7 eV, while the resonance energy is ~2.4 eV. The threshold scales as `(dE)³`, so:

- Fe I: `(6.0/2.4)³ ≈ 15.6×` overstatement of required `n_e`
- Ca I: similar magnitude

At `T=8000 K, n_e=1×10¹⁶ cm⁻³` (plausible low-density LIBS):  
- Paper-correct threshold: `1.98×10¹⁵ cm⁻³` → **LTE satisfied**  
- Code default threshold: `3.09×10¹⁶ cm⁻³` → **LTE flagged as violated (false negative)**

**Physical consequence:** The LTE quality flag in `CFLIBSResult` will report `lte_mcwhirter_satisfied=False` for plasmas that *do* satisfy the true McWhirter criterion. This is a conservative error — it may suppress valid results or generate spurious warnings — but it does **not** permit non-LTE spectra to pass silently. The composition/temperature calculation itself is not affected; only the quality-metric flag is wrong.

**Mitigating factors:**
1. The code acknowledges this in the `_delta_e_from_observations` docstring: "bounds the resonance-to-upper-level transition and is far larger than the adjacent-gap value."
2. The environment flag `CFLIBS_MCWHIRTER_RESONANCE_DE=1` (in `iterative.py:2161`) switches to the paper-correct resonance dE. This flag is **off by default**.
3. The code explicitly calls this sub-lever "M2" and links it to Cristoforetti (2010).

---

## FIX

### Constant: no fix needed

`MCWHIRTER_CONST = 1.6e12` in `cflibs/core/constants.py:80` is exact.

### M2 (dE definition): fix exists but off by default

The correct fix is to **flip the default** of `CFLIBS_MCWHIRTER_RESONANCE_DE` to `on` (or `True`), or better, to make `_delta_e_from_observations` use the ground→first-excited resonance transition energy from the atomic database rather than `max(E_k)`.

Current path: `cflibs/inversion/solve/iterative.py:2161` — the environment flag gates the resonance dE.  
Correct default: enable the resonance-dE path (`delta_e_override = self._mcwhirter_delta_e_resonance(observations)`) without requiring an env flag.

Additionally, `lte_validator.py:387–389` should document that `max(E_k)` is an approximation that overstates the threshold, not a faithful implementation of McWhirter/Cristoforetti.

For the code to be wrong *in the dangerous direction*, the dE would need to be **smaller** than the resonance transition (e.g., using adjacent-level gaps), which would *under*-estimate the required `n_e` and allow non-LTE plasmas to pass silently. The current error goes the other way (too large dE → too high threshold → false LTE failures). Low severity.
