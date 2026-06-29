"""Realistic DED LIBS noise model (DED-PLAN section 4, Codex-validated).

Five sources, matching single-shot DED reality:
  A. shot-to-shot lognormal intensity jitter (dominant single-shot source).
  B. Poisson photon noise (SNR set by expected photons at the peak).
  C. detector readout noise (additive Gaussian).
  D. plasma T / n_e jitter -- applied by RE-RUNNING the forward model at the
     jittered (T, n_e), NOT post-hoc scaling (the Boltzmann factor is nonlinear
     in T; post-hoc scaling is wrong). Sampled here, consumed by the generator.
  E. polynomial baseline jitter.

``apply_ded_noise`` applies A, B, C, E to a clean spectrum; the generator calls
``sample_plasma_jitter`` for D before forward-modelling each shot. All
randomness flows through a caller-supplied ``numpy.random.Generator`` for
reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DEDNoiseParams:
    """Tunable DED noise parameters (defaults ~ a moderate single-shot LIBS rig)."""

    sigma_shot: float = 0.20  # A: lognormal shot-to-shot (~22% RSD)
    peak_photons: float = 1.0e4  # B: expected photons at clean peak -> SNR_peak=sqrt
    readout_frac: float = 0.005  # C: readout noise as fraction of clean peak
    delta_T_K: float = 300.0  # D: plasma temperature jitter (1-sigma, K)
    delta_log_ne: float = 0.05  # D: plasma log10(n_e) jitter (1-sigma, dex)
    baseline_b0_frac: float = 0.05  # E: constant baseline (fraction of peak)
    baseline_b1_frac: float = 0.01  # E: linear baseline slope (fraction of peak)

    @property
    def peak_snr(self) -> float:
        """Approximate peak SNR from the Poisson photon budget."""
        return float(np.sqrt(self.peak_photons))

    def sample_plasma_jitter(self, T_K: float, ne_cm3: float, rng: np.random.Generator):
        """Sample a jittered (T, n_e) for shot D. Re-forward-model at these."""
        T = float(T_K + rng.normal(0.0, self.delta_T_K))
        log_ne = float(np.log10(ne_cm3) + rng.normal(0.0, self.delta_log_ne))
        return max(T, 1.0), float(10.0**log_ne)


def apply_ded_noise(
    clean_intensity: np.ndarray,
    params: DEDNoiseParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply noise sources A (shot), B (Poisson), C (readout), E (baseline).

    Parameters
    ----------
    clean_intensity : array
        Clean forward-modelled spectrum (already at the jittered plasma state
        if D is in use). Arbitrary linear units.
    params : DEDNoiseParams
    rng : numpy.random.Generator

    Returns
    -------
    array
        Noisy spectrum, same shape, clipped at >= 0.
    """
    x = np.asarray(clean_intensity, dtype=float)
    if x.size == 0:
        return x.copy()
    peak = float(np.max(x))
    if not np.isfinite(peak) or peak <= 0:
        return x.copy()

    # A. shot-to-shot lognormal multiplicative jitter (one scalar per shot)
    shot = float(np.exp(rng.normal(0.0, params.sigma_shot)))
    y = x * shot

    # B. Poisson photon noise. Scale to a photon budget set by the clean peak,
    #    draw counts, scale back to intensity units. Normal approx for big lambda.
    lam = np.clip(params.peak_photons * (y / peak), 0.0, None)
    big = lam > 1.0e6
    detected = np.empty_like(lam)
    detected[~big] = rng.poisson(lam[~big]).astype(float)
    detected[big] = lam[big] + rng.normal(0.0, np.sqrt(lam[big]))
    y = detected / params.peak_photons * peak

    # C. additive Gaussian readout noise (fraction of the clean peak)
    y = y + rng.normal(0.0, params.readout_frac * peak, size=y.shape)

    # E. polynomial baseline (constant + linear ramp, fractions of peak)
    n = y.shape[0]
    ramp = np.linspace(0.0, 1.0, n)
    y = y + params.baseline_b0_frac * peak + params.baseline_b1_frac * peak * ramp

    return np.clip(y, 0.0, None)
