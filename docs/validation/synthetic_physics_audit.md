# Synthetic LIBS Physics Audit (Initial)

This document captures the initial equation-to-code audit for synthetic spectrum generation.

## Scope Audited

- `cflibs/plasma/saha_boltzmann.py`
- `cflibs/radiation/emissivity.py`
- `cflibs/radiation/profiles.py`
- `cflibs/radiation/stark.py`
- `cflibs/radiation/spectrum_model.py`
- `cflibs/inversion/bayesian.py`
- `cflibs/benchmark/synthetic.py`

## Equation-to-Code Mapping

1. Saha ionization balance:
   - Implemented in `cflibs/plasma/saha_boltzmann.py` (`solve_ionization_balance`)
   - Implemented in vectorized form in `cflibs/inversion/bayesian.py` (`_compute_spectrum`)
2. Boltzmann level populations:
   - `cflibs/plasma/saha_boltzmann.py` (`solve_level_population`)
   - `cflibs/inversion/bayesian.py` (`n_upper` calculation)
3. Line emissivity:
   - `epsilon = (h c / (4 pi lambda)) A_ki n_upper`
   - `cflibs/radiation/emissivity.py` and `cflibs/inversion/bayesian.py`
4. Broadening:
   - Gaussian/Doppler and Voigt profiles in `cflibs/radiation/profiles.py`
   - Stark width model in `cflibs/radiation/stark.py`
   - JAX Voigt + Stark path in `cflibs/inversion/bayesian.py`
5. Instrument response/convolution:
   - `cflibs/instrument/model.py`, `cflibs/instrument/convolution.py`
   - Applied from `cflibs/radiation/spectrum_model.py`

## Key Findings (Initial)

1. Critical: synthetic benchmark forward-model path was effectively broken.
   - `cflibs/benchmark/synthetic.py` used outdated constructor signatures for `SingleZoneLTEPlasma` and `SpectrumModel`.
   - This could silently force fallback to simplified, non-physical synthetic spectra.
2. High: composition semantics mismatch.
   - Benchmark composition is mass-fraction-like, while plasma solver expects number densities.
   - Direct usage without conversion distorts relative element line strengths.
3. High: there are multiple forward-model implementations with different fidelity assumptions.
   - `cflibs/radiation/spectrum_model.py` uses simplified broadening.
   - `cflibs/inversion/bayesian.py` includes Voigt/Stark but different state parametrization.

## Fixes Implemented in This Step

1. Repaired benchmark forward-model integration in `cflibs/benchmark/synthetic.py`:
   - Correct `SingleZoneLTEPlasma` construction (`T_e`, `n_e`, `species`).
   - Correct `SpectrumModel` construction (`plasma`, `atomic_db`, `instrument`, wavelength range, step).
   - Interpolate model output back to benchmark wavelength grid when needed.
2. Added mass-fraction -> number-density conversion helper:
   - `SyntheticBenchmarkGenerator._composition_to_number_densities(...)`
3. Added tests in `tests/test_benchmark.py`:
   - Forward-model path is used when atomic DB is available.
   - Composition conversion preserves normalization and atomic-mass ordering.

## Remaining Risks / Next Audit Targets

1. Reconcile density parametrization between benchmark/radiation model and Bayesian forward model.
2. Validate Stark broadening and temperature scaling against trusted line subsets.
3. Add quantitative physical sanity checks (line-ratio vs Boltzmann prediction, width-vs-density trend checks).
