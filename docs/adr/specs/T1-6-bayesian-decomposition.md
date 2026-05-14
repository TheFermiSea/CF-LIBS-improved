# T1-6 Implementation Spec — Bayesian Decomposition + Forward-Model Registry

**Bead:** `CF-LIBS-improved-0mor` · **ADR:** [ADR-0001](../ADR-0001-radis-jaxrts-pattern-survey.md) §7.2, §8.1 row T1-6 · **Wave:** 3 (parallel with T1-5) · **Hard deps:** T1-1 (`5oar`), T1-2 (`swgm`) · **Convergent:** D-P1 (petitRADTRANS retrieval), D-P2 (named-model registry), C-P14 (exojax NumPyro coupling), LMFIT `Parameter`

## 1. Goals

Retire the 3264-LOC `cflibs/inversion/solve/bayesian.py` monolith by decomposing along natural Bayesian-inference axes: **priors** (parameter declarations + transforms), **forward** (pure JAX kernel wrapper emitting the NumPyro `obs` site), **samplers** (NumPyro NUTS + dynesty adapters behind a common ABC). Add a named forward-model registry so downstream callers select by string instead of importing a class.

- Each file under 800 LOC.
- Public API frozen: every name currently exported from `cflibs.inversion.bayesian` continues to resolve through a shim.
- `bayesian/forward.py` becomes a thin shell around `cflibs.radiation.kernels.forward_model` (T1-2); no duplicated physics.
- A `Sampler` Protocol lets new samplers (e.g., `BlackjaxSampler`, `EnsembleSampler`) plug in without touching the forward model.

## 2. Target file layout

**Before:**
```
cflibs/inversion/solve/bayesian.py   # 3264 LOC monolith
cflibs/inversion/bayesian.py         # shim: from cflibs.inversion.solve.bayesian import *
```

**After:**
```
cflibs/inversion/solve/bayesian/
    __init__.py            # re-exports for back-compat
    priors.py              # ~400 LOC: Parameter, PriorConfig, NoiseParameters,
                           #           prior_cube_transform, numpyro_sample_from
    forward.py             # ~500 LOC: BayesianForwardModel, TwoZoneBayesianForwardModel,
                           #           bayesian_model() NumPyro graph builder,
                           #           AtomicDataArrays adapter from AtomicSnapshot
    samplers.py            # ~700 LOC: Sampler protocol, NumPyroNUTSSampler (= MCMCSampler),
                           #           DynestyNestedSampler (= NestedSampler),
                           #           TwoZoneMCMCSampler, MCMCResult, NestedSamplingResult,
                           #           run_mcmc convenience wrapper

cflibs/inversion/bayesian.py
    # unchanged shim: from cflibs.inversion.solve.bayesian import *

cflibs/inversion/forward_models/
    __init__.py
        # FORWARD_MODELS registry
```

`forward_models/__init__.py`:
```python
from typing import Mapping, Protocol
from cflibs.radiation.kernels import forward_model as _single_zone_lte_kernel

class ForwardModelFn(Protocol):
    def __call__(self, plasma_state, atomic_snapshot, instrument, wavelength_grid, **kwargs): ...

def _hermann_two_region(...): ...
def _lte_with_self_absorption(plasma_state, atomic_snapshot, instrument, wl, **kw):
    return _single_zone_lte_kernel(plasma_state, atomic_snapshot, instrument, wl,
                                   apply_self_absorption=True, **kw)

FORWARD_MODELS: Mapping[str, ForwardModelFn] = {
    "single_zone_lte":          _single_zone_lte_kernel,
    "hermann_two_region":       _hermann_two_region,
    "lte_with_self_absorption": _lte_with_self_absorption,
}

def get_forward_model(name: str) -> ForwardModelFn:
    if name not in FORWARD_MODELS:
        raise KeyError(f"Unknown forward model {name!r}; available: {sorted(FORWARD_MODELS)}")
    return FORWARD_MODELS[name]
```

`bayesian/__init__.py`:
```python
"""Bayesian sub-package — decomposed from legacy bayesian.py monolith."""
from .priors import (
    Parameter, PriorConfig, NoiseParameters,
    create_temperature_prior, create_density_prior, create_concentration_prior,
)
from .forward import (
    BayesianForwardModel, TwoZoneBayesianForwardModel,
    AtomicDataArrays, bayesian_model, two_zone_bayesian_model,
    load_atomic_data, partition_function, mcwhirter_log_penalty,
    log_likelihood,
)
from .samplers import (
    Sampler, SamplerResult,
    MCMCSampler, NestedSampler, TwoZoneMCMCSampler,
    MCMCResult, NestedSamplingResult, TwoZoneMCMCResult,
    ConvergenceStatus, run_mcmc,
)
```

## 3. `Parameter` dataclass contract

`cflibs/inversion/solve/bayesian/priors.py`:

```python
from dataclasses import dataclass
from typing import Literal, Optional
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpy as np

PriorKind = Literal["uniform", "loguniform", "normal", "truncnormal", "dirichlet"]

@dataclass(frozen=True)
class Parameter:
    """Declarative parameter spec consumable by both NumPyro and nested samplers."""
    name: str
    prior_low: float
    prior_high: float
    prior_kind: PriorKind = "uniform"
    vary: bool = True
    expr: Optional[str] = None         # algebraic constraint, e.g. "1 - C_Cr - C_Ni"
    prior_mean: Optional[float] = None # used by normal/truncnormal
    prior_std: Optional[float] = None

    def numpyro_sample(self) -> "jnp.ndarray":
        """Emit numpyro.sample(self.name, dist) and return draw."""
        if not self.vary:
            return jnp.asarray(self.prior_mean if self.prior_mean is not None else self.prior_low)
        if self.prior_kind == "uniform":
            return numpyro.sample(self.name, dist.Uniform(self.prior_low, self.prior_high))
        if self.prior_kind == "loguniform":
            log_x = numpyro.sample(self.name + "_log",
                dist.Uniform(jnp.log10(self.prior_low), jnp.log10(self.prior_high)))
            return numpyro.deterministic(self.name, jnp.power(10.0, log_x))
        if self.prior_kind == "normal":
            return numpyro.sample(self.name, dist.Normal(self.prior_mean, self.prior_std))
        if self.prior_kind == "truncnormal":
            return numpyro.sample(self.name,
                dist.TruncatedNormal(self.prior_mean, self.prior_std,
                                     low=self.prior_low, high=self.prior_high))
        raise ValueError(f"Unsupported prior_kind: {self.prior_kind!r}")

    def cube_transform(self, u: float) -> float:
        """Adapter to nested-sampling unit-cube prior transform (dynesty)."""
        if not self.vary:
            return float(self.prior_mean if self.prior_mean is not None else self.prior_low)
        if self.prior_kind == "uniform":
            return self.prior_low + u * (self.prior_high - self.prior_low)
        if self.prior_kind == "loguniform":
            log_lo, log_hi = np.log10(self.prior_low), np.log10(self.prior_high)
            return float(10.0 ** (log_lo + u * (log_hi - log_lo)))
        if self.prior_kind == "normal":
            from scipy.stats import norm
            return float(norm.ppf(u, loc=self.prior_mean, scale=self.prior_std))
        if self.prior_kind == "truncnormal":
            from scipy.stats import truncnorm
            a = (self.prior_low - self.prior_mean) / self.prior_std
            b = (self.prior_high - self.prior_mean) / self.prior_std
            return float(truncnorm.ppf(u, a, b, loc=self.prior_mean, scale=self.prior_std))
        raise ValueError(f"Unsupported prior_kind: {self.prior_kind!r}")
```

`PriorConfig` (existing at `bayesian.py:589`) becomes a builder emitting a list of `Parameter` plus a Dirichlet special-case for the concentration simplex (Dirichlet doesn't decompose cleanly into per-coord cube transforms — use dynesty stick-breaking already at `bayesian.py:2123-2149`).

## 4. `Sampler` adapter ABC

`cflibs/inversion/solve/bayesian/samplers.py`:

```python
from typing import Protocol, runtime_checkable
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SamplerResult:
    """Common result envelope across NumPyro and dynesty backends."""
    posterior_samples: dict[str, np.ndarray]   # {name: (n_samples,) or (n_samples, k)}
    log_evidence: float | None                 # None for MCMC; finite for nested
    diagnostics: dict[str, object]             # r_hat, ess, neff, n_iter, time_s, ...
    metadata: dict[str, object]                # sampler name, prior_config dict, ...

@runtime_checkable
class Sampler(Protocol):
    """Common interface for posterior-sampling backends."""
    def fit(self, model: "BayesianForwardModel", data: "SpectrumData", **kwargs) -> SamplerResult: ...

class NumPyroNUTSSampler:
    """NUTS via NumPyro. Replaces MCMCSampler (kept as alias)."""
    def __init__(self, num_warmup=500, num_samples=1000, num_chains=1,
                 target_accept_prob=0.8, max_tree_depth=10): ...
    def fit(self, model, data, *, seed=0, progress_bar=True) -> SamplerResult: ...

class DynestyNestedSampler:
    """Static + dynamic nested sampling via dynesty. Replaces NestedSampler."""
    def __init__(self, nlive=100, dlogz=0.5, bound="multi", sample="rwalk"): ...
    def fit(self, model, data, *, seed=0) -> SamplerResult: ...
```

Legacy `MCMCSampler` (`bayesian.py:1490`) and `NestedSampler` (`bayesian.py:2065`) kept as subclasses/aliases. Their `run(...)` methods keep current signatures and return `MCMCResult`/`NestedSamplingResult` which gain `from_sampler_result()` constructors.

`SpectrumData = @dataclass(frozen=True) class SpectrumData: wavelength: np.ndarray; flux: np.ndarray; noise_params: NoiseParameters = field(default_factory=NoiseParameters)`. Replaces bare `observed` array passed to `.run()`.

## 5. Backward-compat plan

1. **Shim preserves all current imports.** `cflibs/inversion/bayesian.py` continues `from cflibs.inversion.solve.bayesian import *`. Because `bayesian/` is now a package (not module), `__init__.py` re-exports cover every symbol.
2. **No-rename guarantee.** `BayesianForwardModel`, `MCMCSampler`, `NestedSampler`, `TwoZoneBayesianForwardModel`, `TwoZoneMCMCSampler`, `MCMCResult`, `NestedSamplingResult`, `TwoZoneMCMCResult`, `PriorConfig`, `NoiseParameters`, `AtomicDataArrays`, `load_atomic_data`, `partition_function`, `mcwhirter_log_penalty`, `log_likelihood`, `bayesian_model`, `two_zone_bayesian_model`, `run_mcmc`, `create_temperature_prior`, `create_density_prior`, `create_concentration_prior`, `ConvergenceStatus` — all keep names and module-relative paths through shim.
3. **CLI/exporters keep working.** `cflibs/cli/main.py:425` (`from cflibs.inversion.bayesian import BayesianForwardModel, MCMCSampler`) and `cflibs/io/exporters.py:42` (`from cflibs.inversion.bayesian import MCMCResult, NestedSamplingResult`) require zero edits.
4. **Deprecation policy.** New canonical path is `from cflibs.inversion.solve.bayesian import ...` (no `*`); new Protocol-based API (`Sampler`, `NumPyroNUTSSampler`, `SamplerResult`) is what new code uses. Legacy `MCMCSampler.run(observed, num_warmup=...)` kept indefinitely.

## 6. Acceptance criteria

- `from cflibs.inversion.bayesian import BayesianForwardModel, MCMCSampler, NestedSampler, TwoZoneBayesianForwardModel, MCMCResult` all resolve.
- Every file under `bayesian/` < 800 LOC (`wc -l cflibs/inversion/solve/bayesian/*.py`).
- `bayesian/forward.py::BayesianForwardModel.forward` body imports and delegates to `cflibs.radiation.kernels.forward_model` from T1-2. No re-implementation of Voigt/Saha-Boltzmann.
- `FORWARD_MODELS` registry resolves `"single_zone_lte"`, `"hermann_two_region"`, `"lte_with_self_absorption"`.
- Physics-only constraint: no `flax`, `equinox`, `jax.nn`, `sklearn`, `torch`. Ruff TID251 passes.
- Public-API smoke test: every symbol from `cflibs.inversion.bayesian` imports without `AttributeError`.

## 7. Test plan

**Existing tests** (enumerated via `pytest --collect-only tests/ -k bayesian`): `tests/test_bayesian.py`, `tests/test_distributed_mcmc.py`, `tests/benchmark/test_jax_workflows.py` — all unchanged.
- `tests/test_bayesian.py`: `BayesianForwardModel(...).forward(T, log_ne, conc)` output matches pre-refactor at rtol=1e-10 (same kernel under hood after T1-2).
- `tests/test_distributed_mcmc.py`: MCMC parallelization unaffected.

**New** `tests/inversion/test_bayesian_priors.py`:
- `test_parameter_uniform_numpyro_adapter` — inside NumPyro trace, records `numpyro.sample` site with `dist.Uniform`.
- `test_parameter_loguniform_cube_transform` — `Parameter("ne", 1e15, 1e19, "loguniform").cube_transform(0.5)` returns `10**17`.
- `test_parameter_truncnormal_cube_transform` — round-trip ppf at u=0.5 returns mean.
- `test_parameter_frozen` — `FrozenInstanceError` on mutation.

**New** `tests/inversion/test_bayesian_samplers.py`:
- `test_sampler_protocol_runtime_check` — `isinstance(NumPyroNUTSSampler(...), Sampler) is True`.
- `test_sampler_swap` — same model + data → both samplers produce `SamplerResult` with comparable posterior medians for T_eV (within 2σ).
- `test_legacy_MCMCSampler_run_signature` — `MCMCSampler(model).run(observed, num_warmup=100, num_samples=100)` returns `MCMCResult` with legacy attributes.

**New** `tests/inversion/test_forward_models_registry.py`:
- `test_registry_keys` — `set(FORWARD_MODELS) == {"single_zone_lte", "hermann_two_region", "lte_with_self_absorption"}`.
- `test_get_forward_model_raises_unknown` — `KeyError` on bogus name.
- `test_single_zone_lte_matches_kernel` — registry entry produces bitwise-identical output to `cflibs.radiation.kernels.forward_model` on fixed input.

**New** `tests/inversion/test_bayesian_back_compat.py`:
- `from cflibs.inversion.bayesian import *` succeeds.
- All names in pre-refactor public API importable from both `cflibs.inversion.bayesian` and `cflibs.inversion.solve.bayesian`.
- `cflibs.cli.main` and `cflibs.io.exporters` import without error.

**LOC gate** in `tests/test_repo_health.py` (new or appended):
```python
for f in glob("cflibs/inversion/solve/bayesian/*.py"):
    assert wc_l(f) < 800
```

## 8. Dependencies

- **Hard dep T1-1** (`5oar`): `AtomicSnapshot`, pytree-registered `SingleZoneLTEPlasma` + `InstrumentModel`, host/kernel split convention.
- **Hard dep T1-2** (`swgm`): `bayesian/forward.py` imports `cflibs.radiation.kernels.forward_model`. `forward.py` cannot land before T1-2 ships.
- **Sequence:** T1-1 → T1-2 → T1-6. If splitting T1-6 into two PRs: PR-A `priors.py` + `samplers.py` (additive under existing monolith); PR-B `forward.py` + `__init__.py` after T1-2 merges.
- **Coupling note to T1-2:** Both reference shared `AtomicSnapshot` definition (in `cflibs/atomic/snapshot.py` from T1-1). T1-6 inherits whatever T1-2 settled on.

## 9. Risks & rollback

- **Public API regression.** Mitigation: `test_bayesian_back_compat.py` is the canary; any missing symbol blocks merge.
- **`AtomicDataArrays` shape divergence from `AtomicSnapshot`.** Mitigation: adapt `AtomicDataArrays` to construct from `AtomicSnapshot` (add a `from_snapshot()` classmethod); document the contract.
- **NumPyro graph subtle changes** if `Parameter.numpyro_sample` reorders sites. Mitigation: posterior-median parity test (`test_sampler_swap`) at 2σ; run on a fixed-seed fixture in CI.
- **Rollback:** keep `bayesian.py` monolith alongside `bayesian/` package for one release cycle; flip the shim's import line if regression surfaces.
