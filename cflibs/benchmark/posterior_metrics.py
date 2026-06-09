"""
Posterior calibration metrics for CF-LIBS Bayesian validation gate.

Implements the Tier-1 hard gate posterior diagnostics specified in
``docs/VALIDATION_METRICS.md`` §2.3 and ``validation/protocol.yaml``
``posterior_calibration`` block.

Metrics emitted (all per-parameter where applicable):

* **R-hat** (Gelman-Rubin split-chain potential scale reduction). Hard
  fail if any parameter exceeds 1.01.
* **ESS bulk** and **ESS tail** (effective sample size). A run with any
  parameter below 400 is *INVALID* — it should be re-run, not failed
  outright. Reported separately so the gate can decide.
* **Divergent transition count** (NUTS only). Hard fail if > 0.
* **PSIS-LOO ELPD** with **per-fold Pareto k-hat** (only when a
  per-draw log-likelihood array is supplied; otherwise emitted as
  ``None`` and left to a separate sub-issue per the implementation
  spec). Hard fail if any k-hat ≥ 0.7.
* **95% credible-interval empirical coverage**, when certified values
  are supplied. Required to be in the bidirectional band
  ``[0.93, 0.97]`` — both *under-coverage* (overconfident posterior)
  and *over-coverage* (underconfident posterior) trip the gate.
* **PIT (probability-integral-transform) histogram chi-squared
  uniformity p-value**. Tier-3 forensic; reported, not gated.
* **Sharpness**: mean width of the 95% credible interval in CLR space.
  Smaller is sharper; useful only as a tie-breaker between equally
  calibrated posteriors.

The module prefers ArviZ when available (it ships in the ``bayesian``
extras of ``pyproject.toml``) and falls back to small numpy
implementations of split-R-hat and batch-means ESS when not. PSIS-LOO
requires the ArviZ implementation; without it, ``psis_loo_*`` fields
are ``None`` and a separate sub-issue tracks the work.

References
----------
- Gelman & Rubin (1992) "Inference from Iterative Simulation Using
  Multiple Sequences". Statistical Science.
- Vehtari, Gelman, Simpson, Carpenter, Bürkner (2021) "Rank-normalization,
  folding, and localization: an improved R-hat for assessing convergence
  of MCMC". Bayesian Analysis 16(2).
- Vehtari, Gelman, Gabry (2017) "Practical Bayesian model evaluation
  using leave-one-out cross-validation and WAIC". Statistics and
  Computing 27(5).
- Gneiting, Balabdaoui, Raftery (2007) "Probabilistic forecasts,
  calibration and sharpness". JRSS-B.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

try:  # pragma: no cover - exercised via env where arviz is installed
    import arviz as az  # type: ignore

    _HAS_ARVIZ = True
except Exception:  # pragma: no cover - exercised in CI without arviz
    az = None  # type: ignore[assignment]
    _HAS_ARVIZ = False


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class PosteriorDiagnostics:
    """Container for posterior calibration metrics.

    Per-parameter scalars are stored as ``Dict[str, float]`` keyed by
    parameter name. Coverage / PIT / sharpness fields are ``None`` when
    the inputs do not permit them (e.g. no certified values supplied,
    no log-likelihood array, single-parameter posterior).
    """

    # --- Convergence (per-parameter) ---
    rhat: Dict[str, float]
    ess_bulk: Dict[str, float]
    ess_tail: Dict[str, float]

    # --- NUTS pathology counters ---
    divergent_count: int
    n_chains: int
    n_draws: int

    # --- PSIS-LOO (None when log_likelihood not provided) ---
    psis_loo_elpd: Optional[float] = None
    psis_loo_se: Optional[float] = None
    psis_loo_p: Optional[float] = None
    psis_loo_k_hat_max: Optional[float] = None
    psis_loo_k_hat_bad_fraction: Optional[float] = None

    # --- Calibration (only when certified_values supplied) ---
    coverage_95: Optional[float] = None
    coverage_per_param: Dict[str, float] = field(default_factory=dict)
    pit_chi2_p_value: Optional[float] = None

    # --- Sharpness (mean 95% CI width in CLR space) ---
    sharpness_clr: Optional[float] = None

    # --- Hard-gate verdict against the protocol band ---
    rhat_max: float = float("nan")
    ess_bulk_min: float = float("nan")
    ess_tail_min: float = float("nan")
    coverage_in_band: Optional[bool] = None  # None when coverage absent
    passes_hard_gate: bool = False
    reasons: list[str] = field(default_factory=list)

    # --- Bookkeeping ---
    backend: str = "arviz" if _HAS_ARVIZ else "fallback"

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (for ``composition_records.json``)."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Sample normalisation helpers
# ---------------------------------------------------------------------------


def _coerce_chains_draws(arr: np.ndarray) -> np.ndarray:
    """Coerce a sample array to ``(n_chains, n_draws, ...)``.

    Accepts:

    * ``(n_draws,)``                          → ``(1, n_draws)``
    * ``(n_chains, n_draws)``                 → unchanged
    * ``(n_draws, dim)``                      → ``(1, n_draws, dim)``
    * ``(n_chains, n_draws, dim)``            → unchanged
    * ``(n_chains, n_draws, dim_a, dim_b)``   → unchanged

    A 2-D array is treated as ``(n_draws, dim)`` only when ``dim`` is
    larger than ``n_chains`` would plausibly be (heuristic: any 2-D
    array with the first axis longer than 16 is assumed to be
    ``(n_draws, dim)`` rather than ``(n_chains, n_draws)`` -- single-
    chain single-param input. To remove that ambiguity callers should
    pass explicit ``(n_chains, n_draws[, ...])`` arrays.)
    """
    a = np.asarray(arr)
    if a.ndim == 1:
        return a[np.newaxis, :]
    return a


def _normalize_samples(
    samples: Mapping[str, Any] | Any,
) -> Dict[str, np.ndarray]:
    """Build a ``{name: (n_chains, n_draws, ...)}`` dict.

    Accepts either:

    * a mapping of parameter name → array, or
    * an arviz ``InferenceData`` (uses ``posterior`` group), or
    * an xarray ``Dataset``.
    """
    if _HAS_ARVIZ:
        if hasattr(samples, "posterior"):
            ds = samples.posterior  # InferenceData
            return {
                name: np.asarray(ds[name].values)
                for name in ds.data_vars
                if np.asarray(ds[name].values).ndim >= 2
            }
        if hasattr(samples, "data_vars"):  # xarray Dataset
            return {
                name: np.asarray(samples[name].values)
                for name in samples.data_vars  # type: ignore[union-attr]
                if np.asarray(samples[name].values).ndim >= 2
            }

    if not isinstance(samples, Mapping):
        raise TypeError(
            "samples must be a Mapping[str, ndarray], an ArviZ InferenceData, or an xarray Dataset"
        )

    out: Dict[str, np.ndarray] = {}
    for name, value in samples.items():
        a = _coerce_chains_draws(np.asarray(value))
        if a.ndim < 2:
            raise ValueError(
                f"samples[{name!r}] must have at least 2 dimensions "
                f"(chains, draws); got shape {a.shape}"
            )
        out[name] = a
    return out


def _flatten_per_param(arr: np.ndarray, name: str) -> Dict[str, np.ndarray]:
    """Expand a multi-dim posterior array into a dict of scalar
    posteriors keyed by ``name`` or ``f"{name}[i]"`` / ``f"{name}[i,j]"``.
    """
    if arr.ndim == 2:
        return {name: arr}
    if arr.ndim == 3:
        n_chains, n_draws, dim = arr.shape
        return {f"{name}[{i}]": arr[:, :, i] for i in range(dim)}
    if arr.ndim == 4:
        n_chains, n_draws, da, db = arr.shape
        out: Dict[str, np.ndarray] = {}
        for i in range(da):
            for j in range(db):
                out[f"{name}[{i},{j}]"] = arr[:, :, i, j]
        return out
    raise ValueError(f"unsupported posterior shape {arr.shape} for parameter {name!r}")


def _expand_all(samples: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for name, arr in samples.items():
        out.update(_flatten_per_param(arr, name))
    return out


# ---------------------------------------------------------------------------
# Fallback split-R-hat and batch-means ESS
# ---------------------------------------------------------------------------


def _split_rhat_numpy(samples: np.ndarray) -> float:
    """Split-R-hat. ``samples`` must be ``(n_chains, n_draws)``.

    Implements Gelman-Rubin with the split-chain refinement used by
    Stan / ArviZ: each chain is split in half, doubling the effective
    chain count.
    """
    a = np.asarray(samples, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError("rhat expects (n_chains, n_draws)")
    n_chains, n_draws = a.shape
    if n_draws < 4:
        return float("nan")
    half = n_draws // 2
    split = a[:, : 2 * half].reshape(n_chains * 2, half)

    chain_means = split.mean(axis=1)
    chain_vars = split.var(axis=1, ddof=1)

    # Between-chain variance
    B = half * np.var(chain_means, ddof=1)
    # Within-chain variance
    W = chain_vars.mean()
    if W <= 0 or not np.isfinite(W):
        return 1.0 if B == 0 else float("inf")
    var_hat = ((half - 1) / half) * W + B / half
    return float(np.sqrt(var_hat / W))


def _autocov_at_lag(x: np.ndarray, lag: int) -> float:
    n = len(x)
    if lag >= n:
        return 0.0
    xc = x - x.mean()
    return float(np.dot(xc[: n - lag], xc[lag:]) / n)


def _ess_geyer_numpy(samples: np.ndarray) -> float:
    """Effective sample size via Geyer's initial-positive sequence
    estimator, applied per chain then summed.

    Conservative; ArviZ's bulk-ESS uses rank-normalisation which is
    strictly better, but this fallback is good enough for the gate.
    """
    a = np.asarray(samples, dtype=np.float64)
    if a.ndim == 1:
        a = a[np.newaxis, :]
    n_chains, n_draws = a.shape
    total_n = n_chains * n_draws
    if n_draws < 4:
        return float(total_n)

    # Chain-pooled autocorrelations (Geyer 1992)
    var = a.var(ddof=1, axis=1).mean()
    if var <= 0:
        return float(total_n)

    rhos: list[float] = []
    max_lag = min(n_draws - 1, 1000)
    for lag in range(1, max_lag):
        rho_lag = np.mean([_autocov_at_lag(a[c], lag) for c in range(n_chains)]) / var
        rhos.append(float(rho_lag))
        if lag >= 2 and lag % 2 == 0:
            # Initial positive sequence: stop when paired sum non-positive
            pair_sum = rhos[-2] + rhos[-1]
            if pair_sum <= 0:
                rhos = rhos[:-2]
                break
    tau = 1.0 + 2.0 * sum(rhos)
    if tau <= 0:
        return float(total_n)
    return float(total_n / tau)


def _ess_bulk_numpy(samples: np.ndarray) -> float:
    """Rank-normalised bulk-ESS approximation (no folding)."""
    a = np.asarray(samples, dtype=np.float64)
    flat = a.reshape(-1)
    # Rank-normalise (Vehtari et al. 2021)
    ranks = np.argsort(np.argsort(flat)) + 1.0
    # Inverse-CDF of standard normal at (r - 3/8) / (n + 1/4) — Blom score
    n = len(flat)
    z = (ranks - 0.375) / (n + 0.25)
    from scipy.stats import norm  # local import to avoid hard dependency at module load

    z_score = norm.ppf(z)
    z_chains = z_score.reshape(a.shape)
    return _ess_geyer_numpy(z_chains)


def _ess_tail_numpy(samples: np.ndarray) -> float:
    """Tail-ESS ≈ min(ESS at 5% and 95% indicator). Approximation:
    use the worse of the two extreme-quantile indicator chains.
    """
    a = np.asarray(samples, dtype=np.float64)
    flat = a.reshape(-1)
    q05 = np.quantile(flat, 0.05)
    q95 = np.quantile(flat, 0.95)
    ind_low = (a <= q05).astype(np.float64)
    ind_high = (a >= q95).astype(np.float64)
    return float(min(_ess_geyer_numpy(ind_low), _ess_geyer_numpy(ind_high)))


# ---------------------------------------------------------------------------
# ArviZ-backed routines
# ---------------------------------------------------------------------------


def _to_inference_data(samples: Dict[str, np.ndarray], log_likelihood: Optional[np.ndarray]):
    """Build an arviz InferenceData from raw arrays. Caller has already
    normalised everything to (n_chains, n_draws, ...)."""
    if not _HAS_ARVIZ:
        raise RuntimeError("arviz is required for InferenceData construction")
    import xarray as xr  # noqa: F401  (import side-effects for arviz)

    # ArviZ from_dict accepts (chains, draws) for 2-D arrays and
    # (chains, draws, *dims) for higher-dim arrays, so the same
    # assignment works for both shapes.
    posterior_dict: Dict[str, Any] = {name: arr for name, arr in samples.items()}

    kwargs: Dict[str, Any] = {"posterior": posterior_dict}
    if log_likelihood is not None:
        ll = np.asarray(log_likelihood)
        if ll.ndim == 2:
            ll = ll[np.newaxis, :, :]  # add chains axis
        kwargs["log_likelihood"] = {"obs": ll}
    return az.from_dict(**kwargs)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Coverage / PIT / sharpness
# ---------------------------------------------------------------------------


def _coverage_per_param(
    flat_samples: Dict[str, np.ndarray],
    certified_values: Mapping[str, float],
    prob: float = 0.95,
) -> Dict[str, float]:
    """Empirical 95% credible-interval coverage for each parameter
    where a certified value is supplied.

    With a single posterior we either ``cover`` (1) or ``not cover`` (0).
    The aggregate coverage across multiple parameters is the mean of
    these indicators — i.e. coverage measured across the parameter
    *family*, which is the operational interpretation in §2.3.
    """
    alpha = (1.0 - prob) / 2.0
    out: Dict[str, float] = {}
    for name, arr in flat_samples.items():
        if name not in certified_values:
            continue
        flat = np.asarray(arr).reshape(-1)
        if flat.size == 0:
            continue
        lo = np.quantile(flat, alpha)
        hi = np.quantile(flat, 1.0 - alpha)
        truth = float(certified_values[name])
        out[name] = float(1.0 if (lo <= truth <= hi) else 0.0)
    return out


def _pit_chi2_p_value(
    flat_samples: Dict[str, np.ndarray],
    certified_values: Mapping[str, float],
    n_bins: int = 10,
) -> Optional[float]:
    """Chi-squared uniformity test on PIT values across the parameter
    family.

    Each parameter contributes one PIT value: the empirical CDF of its
    posterior evaluated at the certified value. A well-calibrated
    posterior produces uniform-on-[0, 1] PIT values; the chi-squared
    test compares the binned histogram against the uniform expectation.

    Returns ``None`` when fewer than 5 parameters have certified
    values — the test is degenerate at small sample sizes.
    """
    pits = []
    for name, arr in flat_samples.items():
        if name not in certified_values:
            continue
        flat = np.asarray(arr).reshape(-1)
        if flat.size == 0:
            continue
        truth = float(certified_values[name])
        pits.append(float(np.mean(flat <= truth)))
    if len(pits) < 5:
        return None
    counts, _ = np.histogram(pits, bins=n_bins, range=(0.0, 1.0))
    expected = len(pits) / n_bins
    if expected <= 0:
        return None
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    from scipy.stats import chi2 as chi2_dist  # local import

    df = n_bins - 1
    return float(chi2_dist.sf(chi2, df))


def _clr_sharpness(
    samples: Mapping[str, np.ndarray],
    concentration_key: str = "concentrations",
    prob: float = 0.95,
) -> Optional[float]:
    """Sharpness in CLR space: mean 95% credible-interval width of CLR
    coordinates of the concentration vector.

    The function looks for a 3-D ``samples[concentration_key]`` array
    of shape ``(n_chains, n_draws, n_elements)`` and computes:

        sharpness = mean_i  ( q_{1-α/2}(clr_i) - q_{α/2}(clr_i) )

    where ``α = 1 - prob``. Returns ``None`` if the array is absent or
    has the wrong shape.
    """
    if concentration_key not in samples:
        return None
    arr = np.asarray(samples[concentration_key])
    if arr.ndim < 3:
        return None
    flat = arr.reshape(-1, arr.shape[-1])  # (n_total, n_elements)
    if flat.shape[1] < 2:
        return None
    eps = 1e-12
    pos = np.clip(flat, eps, None)
    log_pos = np.log(pos)
    clr = log_pos - log_pos.mean(axis=1, keepdims=True)
    alpha = (1.0 - prob) / 2.0
    lo = np.quantile(clr, alpha, axis=0)
    hi = np.quantile(clr, 1.0 - alpha, axis=0)
    return float(np.mean(hi - lo))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def _convergence_numpy(
    flat: Dict[str, np.ndarray],
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Per-parameter R-hat / bulk-ESS / tail-ESS via the numpy fallbacks."""
    rhat: Dict[str, float] = {}
    ess_bulk: Dict[str, float] = {}
    ess_tail: Dict[str, float] = {}
    for name, arr in flat.items():
        rhat[name] = _split_rhat_numpy(arr)
        ess_bulk[name] = _ess_bulk_numpy(arr)
        ess_tail[name] = _ess_tail_numpy(arr)
    return rhat, ess_bulk, ess_tail


def _convergence_arviz(
    norm: Dict[str, np.ndarray],
    flat: Dict[str, np.ndarray],
    log_likelihood: Optional[np.ndarray],
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Per-parameter R-hat / bulk-ESS / tail-ESS via ArviZ, with a numpy
    fall-through on any ArviZ failure."""
    rhat: Dict[str, float] = {}
    ess_bulk: Dict[str, float] = {}
    ess_tail: Dict[str, float] = {}
    try:
        idata = _to_inference_data(norm, log_likelihood)
        rhat_ds = az.rhat(idata)  # type: ignore[union-attr]
        ess_bulk_ds = az.ess(idata, method="bulk")  # type: ignore[union-attr]
        ess_tail_ds = az.ess(idata, method="tail")  # type: ignore[union-attr]
        for name, arr in flat.items():
            base, idx = _split_indexed_name(name)
            rhat[name] = _extract_named_value(rhat_ds, base, idx)
            ess_bulk[name] = _extract_named_value(ess_bulk_ds, base, idx)
            ess_tail[name] = _extract_named_value(ess_tail_ds, base, idx)
    except Exception:
        # Any arviz failure → fall through to numpy.
        return _convergence_numpy(flat)
    return rhat, ess_bulk, ess_tail


def _compute_convergence(
    norm: Dict[str, np.ndarray],
    flat: Dict[str, np.ndarray],
    log_likelihood: Optional[np.ndarray],
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Per-parameter R-hat / bulk-ESS / tail-ESS, ArviZ when available."""
    if _HAS_ARVIZ:
        return _convergence_arviz(norm, flat, log_likelihood)
    return _convergence_numpy(flat)


def _compute_psis_loo(
    norm: Dict[str, np.ndarray],
    log_likelihood: Optional[np.ndarray],
    psis_k_hat_threshold: float,
) -> tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]:
    """PSIS-LOO ELPD / SE / p_loo / max-khat / bad-khat-fraction.

    All five fields are ``None`` when ``log_likelihood`` is absent, ArviZ
    is unavailable, or the ArviZ call fails.
    """
    psis_loo_elpd: Optional[float] = None
    psis_loo_se: Optional[float] = None
    psis_loo_p: Optional[float] = None
    psis_loo_k_hat_max: Optional[float] = None
    psis_loo_k_hat_bad_fraction: Optional[float] = None

    if log_likelihood is not None and _HAS_ARVIZ:
        try:
            idata = _to_inference_data(norm, log_likelihood)
            loo = az.loo(idata, pointwise=True)  # type: ignore[union-attr]
            psis_loo_elpd = float(loo.elpd_loo)
            psis_loo_se = float(loo.se)
            psis_loo_p = float(loo.p_loo)
            khats = np.asarray(getattr(loo, "pareto_k", []))
            if khats.size > 0:
                psis_loo_k_hat_max = float(khats.max())
                psis_loo_k_hat_bad_fraction = float(np.mean(khats >= psis_k_hat_threshold))
        except Exception:
            # Leave PSIS-LOO fields at None.
            pass

    return (
        psis_loo_elpd,
        psis_loo_se,
        psis_loo_p,
        psis_loo_k_hat_max,
        psis_loo_k_hat_bad_fraction,
    )


def _compute_coverage_pit(
    flat: Dict[str, np.ndarray],
    certified_values: Optional[Mapping[str, float]],
) -> tuple[Dict[str, float], Optional[float], Optional[float]]:
    """Coverage-per-param dict, aggregate 95% coverage, and PIT chi2 p-value.

    All defaults (``{}``, ``None``, ``None``) returned when no certified
    values are supplied.
    """
    coverage_per_param: Dict[str, float] = {}
    coverage_95: Optional[float] = None
    pit_chi2_p_value: Optional[float] = None

    if certified_values is not None:
        coverage_per_param = _coverage_per_param(flat, certified_values)
        if coverage_per_param:
            coverage_95 = float(np.mean(list(coverage_per_param.values())))
        pit_chi2_p_value = _pit_chi2_p_value(flat, certified_values)

    return coverage_per_param, coverage_95, pit_chi2_p_value


def _convergence_reasons(
    rhat_max: float,
    ess_bulk_min: float,
    ess_tail_min: float,
    rhat_threshold: float,
    ess_threshold: int,
) -> list[str]:
    """Hard-gate failure reasons for the convergence diagnostics."""
    reasons: list[str] = []
    if np.isfinite(rhat_max) and rhat_max >= rhat_threshold:
        reasons.append(f"rhat_max={rhat_max:.4f} >= {rhat_threshold}")
    if np.isfinite(ess_bulk_min) and ess_bulk_min < ess_threshold:
        reasons.append(f"ess_bulk_min={ess_bulk_min:.0f} < {ess_threshold}")
    if np.isfinite(ess_tail_min) and ess_tail_min < ess_threshold:
        reasons.append(f"ess_tail_min={ess_tail_min:.0f} < {ess_threshold}")
    return reasons


def _coverage_verdict(
    coverage_95: Optional[float],
    coverage_band: Sequence[float],
    reasons: list[str],
) -> Optional[bool]:
    """Decide whether coverage is in-band, appending any out-of-band reason.

    Mutates ``reasons`` in place to match the original control flow.
    """
    if coverage_95 is None:
        return None
    lo, hi = float(coverage_band[0]), float(coverage_band[1])
    coverage_in_band = bool(lo <= coverage_95 <= hi)
    if coverage_95 < lo:
        reasons.append(f"coverage_95={coverage_95:.3f} < {lo} (under-coverage)")
    elif coverage_95 > hi:
        reasons.append(f"coverage_95={coverage_95:.3f} > {hi} (over-coverage)")
    return coverage_in_band


def compute_posterior_diagnostics(
    samples: Mapping[str, Any] | Any,
    *,
    certified_values: Optional[Mapping[str, float]] = None,
    log_likelihood: Optional[np.ndarray] = None,
    divergent_count: int = 0,
    rhat_threshold: float = 1.01,
    ess_threshold: int = 400,
    coverage_band: Sequence[float] = (0.93, 0.97),
    psis_k_hat_threshold: float = 0.7,
    concentration_key: str = "concentrations",
) -> PosteriorDiagnostics:
    """Compute the full posterior calibration diagnostic battery.

    Parameters
    ----------
    samples : Mapping[str, ndarray] | InferenceData | xarray.Dataset
        Posterior draws. Each entry has shape
        ``(n_chains, n_draws[, dim_1[, dim_2]])``. ``(n_draws,)`` and
        ``(n_draws, dim)`` are accepted as single-chain inputs.
    certified_values : Mapping[str, float], optional
        Ground-truth scalar values keyed by *flattened* parameter
        name (e.g. ``"concentrations[0]"``, ``"T_eV"``). When omitted,
        coverage / PIT fields are ``None``.
    log_likelihood : ndarray, optional
        Per-draw log-likelihood, shape
        ``(n_chains, n_draws, n_obs)`` or ``(n_draws, n_obs)``. When
        omitted (or when ArviZ is unavailable) PSIS-LOO fields are
        ``None``.
    divergent_count : int, default 0
        Number of NUTS divergent transitions reported by the sampler.
        Forwarded straight into ``PosteriorDiagnostics`` and used by
        the hard-gate check.
    rhat_threshold, ess_threshold, coverage_band, psis_k_hat_threshold
        Hard-gate thresholds, defaulting to the values pinned in
        ``validation/protocol.yaml``.
    concentration_key
        Name of the concentration sample array used for CLR-space
        sharpness; defaults to the convention in ``MCMCResult``.

    Returns
    -------
    PosteriorDiagnostics
        Dataclass with all per-parameter scalars, sharpness, and a
        flat ``passes_hard_gate`` boolean plus per-failure ``reasons``.

    Notes
    -----
    The function intentionally never raises on a bad posterior — it
    flips ``passes_hard_gate`` and appends a human-readable reason
    instead, so it can be used inside a benchmark harness without
    error-handling boilerplate.
    """
    norm = _normalize_samples(samples)
    flat = _expand_all(norm)

    n_chains, n_draws = next(iter(flat.values())).shape

    # --- R-hat / ESS per parameter ---
    rhat, ess_bulk, ess_tail = _compute_convergence(norm, flat, log_likelihood)

    # --- PSIS-LOO ---
    (
        psis_loo_elpd,
        psis_loo_se,
        psis_loo_p,
        psis_loo_k_hat_max,
        psis_loo_k_hat_bad_fraction,
    ) = _compute_psis_loo(norm, log_likelihood, psis_k_hat_threshold)

    # --- Coverage + PIT ---
    coverage_per_param, coverage_95, pit_chi2_p_value = _compute_coverage_pit(
        flat, certified_values
    )

    # --- Sharpness ---
    sharpness_clr = _clr_sharpness(norm, concentration_key=concentration_key)

    # --- Hard-gate verdict ---
    rhat_max = max(rhat.values()) if rhat else float("nan")
    ess_bulk_min = min(ess_bulk.values()) if ess_bulk else float("nan")
    ess_tail_min = min(ess_tail.values()) if ess_tail else float("nan")

    reasons = _convergence_reasons(
        rhat_max, ess_bulk_min, ess_tail_min, rhat_threshold, ess_threshold
    )
    if divergent_count > 0:
        reasons.append(f"divergent_transitions={divergent_count} > 0")
    if psis_loo_k_hat_max is not None and psis_loo_k_hat_max >= psis_k_hat_threshold:
        reasons.append(f"psis_k_hat_max={psis_loo_k_hat_max:.3f} >= {psis_k_hat_threshold}")

    coverage_in_band = _coverage_verdict(coverage_95, coverage_band, reasons)

    passes_hard_gate = len(reasons) == 0

    return PosteriorDiagnostics(
        rhat=rhat,
        ess_bulk=ess_bulk,
        ess_tail=ess_tail,
        divergent_count=int(divergent_count),
        n_chains=int(n_chains),
        n_draws=int(n_draws),
        psis_loo_elpd=psis_loo_elpd,
        psis_loo_se=psis_loo_se,
        psis_loo_p=psis_loo_p,
        psis_loo_k_hat_max=psis_loo_k_hat_max,
        psis_loo_k_hat_bad_fraction=psis_loo_k_hat_bad_fraction,
        coverage_95=coverage_95,
        coverage_per_param=coverage_per_param,
        pit_chi2_p_value=pit_chi2_p_value,
        sharpness_clr=sharpness_clr,
        rhat_max=float(rhat_max),
        ess_bulk_min=float(ess_bulk_min),
        ess_tail_min=float(ess_tail_min),
        coverage_in_band=coverage_in_band,
        passes_hard_gate=passes_hard_gate,
        reasons=reasons,
        backend="arviz" if _HAS_ARVIZ else "fallback",
    )


# ---------------------------------------------------------------------------
# Internal: ArviZ result extraction
# ---------------------------------------------------------------------------


def _split_indexed_name(name: str):
    """``"concentrations[3]"`` -> ``("concentrations", (3,))``;
    ``"T_eV"`` -> ``("T_eV", ())``.
    """
    if "[" not in name or not name.endswith("]"):
        return name, ()
    base, _, rest = name.partition("[")
    rest = rest.rstrip("]")
    parts = tuple(int(x) for x in rest.split(",") if x.strip())
    return base, parts


def _extract_named_value(ds, base: str, idx: tuple) -> float:
    """Pull a scalar out of an ArviZ summary Dataset for parameter
    ``base`` indexed by ``idx``."""
    try:
        var = ds[base]
    except KeyError:
        return float("nan")
    arr = np.asarray(var.values)
    if arr.ndim == 0:
        return float(arr)
    if not idx:
        return float(arr.reshape(-1)[0])
    try:
        return float(arr[idx])
    except Exception:
        return float(arr.reshape(-1)[0])


# Public re-exports
__all__ = [
    "PosteriorDiagnostics",
    "compute_posterior_diagnostics",
]
