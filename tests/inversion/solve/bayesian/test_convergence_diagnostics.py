"""Convergence-diagnostics tests: no fabricated single-chain CONVERGED.

These tests pin the audit Family-O fix in
:mod:`cflibs.inversion.solve.bayesian.samplers`. Historically the
single-chain fallback hardcoded ``r_hat = 1.0`` and ``ess = len(samples)``
(uncorrected for autocorrelation), so :func:`_assess_convergence` returned
``CONVERGED`` for *any* sufficiently long single chain -- even a deliberately
under-warmed / non-mixing one. That is a fabricated diagnostic.

The fix computes a real *split*-R-hat (single chains are split into two
halves) and an autocorrelation-corrected ESS. We assert:

1. A strongly autocorrelated / non-mixing single chain reports
   ``ess << num_samples`` and is NOT reported CONVERGED.
2. A well-mixed (iid-like) multi-chain sample reports ``r_hat ~ 1.0``
   (computed, not hardcoded) with full ESS and IS reported CONVERGED.
3. The numpy fallback (ArviZ disabled) agrees with an independent oracle.

The diagnostics are exercised through a lightweight stub that mimics the
``numpyro.infer.MCMC.get_samples(group_by_chain=...)`` contract, so no real
NUTS sampling runs here (watchdog-safe).
"""

from __future__ import annotations

import numpy as np
import pytest

from cflibs.inversion.solve.bayesian.priors import ConvergenceStatus
from cflibs.inversion.solve.bayesian.samplers import (
    _assess_convergence,
    _chain_diagnostics,
    _diagnostics_from_mcmc,
    _ess_numpy,
    _split_rhat_numpy,
)


class _StubMCMC:
    """Minimal stand-in for ``numpyro.infer.MCMC`` for diagnostics tests.

    ``samples`` is a ``{name: ndarray}`` map where each array is shaped
    ``(num_chains, num_draws)``. ``get_samples(group_by_chain=True)`` returns
    it as-is; ``group_by_chain=False`` flattens the chain axis -- matching the
    NumPyro contract that the diagnostics path relies on.
    """

    def __init__(self, samples: dict[str, np.ndarray]):
        self._samples = samples

    def get_samples(self, group_by_chain: bool = False):
        if group_by_chain:
            return self._samples
        return {k: v.reshape(-1, *v.shape[2:]) for k, v in self._samples.items()}


def _ar1_chain(n: int, phi: float, seed: int) -> np.ndarray:
    """Generate a length-``n`` AR(1) series with coefficient ``phi``."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + rng.normal()
    return x


# ---------------------------------------------------------------------------
# 1. Non-mixing single chain must NOT be reported CONVERGED
# ---------------------------------------------------------------------------


def test_single_autocorrelated_chain_not_converged():
    """A strongly autocorrelated single chain must report ESS << N and NOT CONVERGED.

    The old fallback returned ``ess = N`` and ``r_hat = 1.0`` -> trivially
    CONVERGED. With a real autocorrelation-corrected ESS, a phi=0.98 AR(1)
    chain has an effective sample size far below the nominal draw count.
    """
    num_samples = 2000
    chain = _ar1_chain(num_samples, phi=0.98, seed=1)
    samples = {
        "T_eV": chain[None, :],  # (1, N) single chain
        "log_ne": _ar1_chain(num_samples, phi=0.98, seed=2)[None, :],
    }
    mcmc = _StubMCMC(samples)

    r_hat, ess = _diagnostics_from_mcmc(mcmc, num_chains=1)

    # ESS must be drastically corrected for autocorrelation -- never == N.
    assert ess["T_eV"] < num_samples, "ESS must not equal the raw sample count"
    assert (
        ess["T_eV"] < num_samples / 5
    ), f"phi=0.98 AR(1) chain should have ESS << N; got {ess['T_eV']:.1f} of {num_samples}"

    # r_hat must be a real computed number, not the fabricated 1.0 sentinel.
    assert "T_eV" in r_hat and np.isfinite(r_hat["T_eV"])

    status = _assess_convergence(r_hat, ess, num_samples)
    assert status is not ConvergenceStatus.CONVERGED, (
        "A non-mixing single chain must NOT be reported CONVERGED "
        f"(got {status}, ess={ess}, r_hat={r_hat})"
    )


def test_single_trending_chain_high_rhat():
    """A trending (non-stationary) single chain must report split-R-hat > 1.01.

    Splitting the chain into halves exposes the drift between early and late
    samples -- the fabricated ``r_hat = 1.0`` would have hidden it.
    """
    num_samples = 2000
    # Near-random-walk: halves have very different means -> high split-R-hat.
    chain = _ar1_chain(num_samples, phi=0.999, seed=7)
    samples = {"T_eV": chain[None, :], "log_ne": chain[None, :]}
    mcmc = _StubMCMC(samples)

    r_hat, ess = _diagnostics_from_mcmc(mcmc, num_chains=1)

    assert (
        r_hat["T_eV"] > 1.01
    ), f"Trending single chain should give split-R-hat > 1.01; got {r_hat['T_eV']:.4f}"
    status = _assess_convergence(r_hat, ess, num_samples)
    assert status is not ConvergenceStatus.CONVERGED


# ---------------------------------------------------------------------------
# 2. Well-mixed multi-chain sample IS reported CONVERGED with r_hat ~ 1
# ---------------------------------------------------------------------------


def test_well_mixed_multichain_converged():
    """Independent (iid) multi-chain draws must give r_hat ~ 1.0 and CONVERGED.

    r_hat is *computed* from the chains, not hardcoded -- so we assert it
    lands in a tight band around 1.0 and that ESS is close to the full count.
    """
    num_chains = 4
    num_draws = 1000
    rng = np.random.default_rng(123)
    arr = rng.normal(size=(num_chains, num_draws))
    samples = {
        "T_eV": arr,
        "log_ne": rng.normal(size=(num_chains, num_draws)),
    }
    mcmc = _StubMCMC(samples)

    r_hat, ess = _diagnostics_from_mcmc(mcmc, num_chains=num_chains)

    assert (
        0.98 < r_hat["T_eV"] < 1.02
    ), f"iid multi-chain r_hat must be ~1.0 (computed); got {r_hat['T_eV']:.4f}"
    total = num_chains * num_draws
    assert (
        ess["T_eV"] > 0.5 * total
    ), f"iid sample ESS should be a large fraction of {total}; got {ess['T_eV']:.1f}"

    status = _assess_convergence(r_hat, ess, num_draws)
    assert status is ConvergenceStatus.CONVERGED, (
        f"Well-mixed multi-chain must be CONVERGED; got {status} " f"(r_hat={r_hat}, ess={ess})"
    )


def test_well_mixed_single_chain_converged():
    """An iid single chain (split into halves) must still report CONVERGED.

    This guards against an over-correction: the fix must not flag genuinely
    well-mixed single chains as non-converged.
    """
    num_samples = 4000
    rng = np.random.default_rng(99)
    chain = rng.normal(size=num_samples)
    samples = {"T_eV": chain[None, :], "log_ne": rng.normal(size=num_samples)[None, :]}
    mcmc = _StubMCMC(samples)

    r_hat, ess = _diagnostics_from_mcmc(mcmc, num_chains=1)

    assert 0.98 < r_hat["T_eV"] < 1.02
    assert ess["T_eV"] > 0.5 * num_samples
    status = _assess_convergence(r_hat, ess, num_samples)
    assert status is ConvergenceStatus.CONVERGED


# ---------------------------------------------------------------------------
# 3. numpy fallback agrees with an independent oracle
# ---------------------------------------------------------------------------


def test_numpy_fallback_ess_matches_oracle():
    """Hand-rolled FFT-autocorrelation ESS must track an independent oracle.

    Oracle: ESS ~= N / (1 + 2 * sum_{k>=1} rho_k) using a direct (non-FFT)
    autocorrelation estimate with the same Geyer initial-positive-sequence
    truncation. We require order-of-magnitude agreement (within 25%) for an
    AR(1) chain -- enough to prove the FFT path is not returning N.
    """
    n = 4000
    chain = _ar1_chain(n, phi=0.9, seed=11)

    ess_fft = _ess_numpy(chain[None, :])

    # Independent oracle: brute-force autocorrelation + Geyer IPS truncation.
    x = chain - chain.mean()
    denom = np.dot(x, x)
    rho = np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, n)])
    tau = 1.0
    t = 0
    while t + 1 < len(rho):
        pair = rho[t] + rho[t + 1]
        if pair <= 0:
            break
        tau += 2.0 * pair
        t += 2
    ess_oracle = n / max(tau, 1.0)

    assert ess_fft < n, "FFT ESS must be autocorrelation-corrected, not N"
    assert (
        abs(ess_fft - ess_oracle) / ess_oracle < 0.25
    ), f"FFT ESS {ess_fft:.1f} disagrees with oracle {ess_oracle:.1f}"


def test_numpy_fallback_split_rhat_matches_arviz():
    """Hand-rolled split-R-hat must match ArviZ on the same array.

    ArviZ's plain (non-rank-normalised) split-R-hat is the canonical oracle.
    """
    az = pytest.importorskip("arviz")

    n = 3000
    # Two chains with a small mean offset -> r_hat modestly above 1.
    rng = np.random.default_rng(5)
    arr = np.stack([rng.normal(0.0, 1.0, n), rng.normal(0.3, 1.0, n)])

    rhat_numpy = _split_rhat_numpy(arr)
    # ArviZ "split" method (no rank normalisation) is the direct oracle.
    rhat_oracle = float(az.rhat(arr, method="split"))

    assert (
        abs(rhat_numpy - rhat_oracle) < 0.02
    ), f"numpy split-R-hat {rhat_numpy:.4f} != ArviZ {rhat_oracle:.4f}"


def test_chain_diagnostics_never_returns_raw_count_for_autocorr():
    """``_chain_diagnostics`` must never echo the nominal draw count for AR data."""
    n = 2000
    chain = _ar1_chain(n, phi=0.95, seed=3)
    rhat, ess = _chain_diagnostics(chain[None, :])
    assert ess < n
    assert np.isfinite(rhat)
