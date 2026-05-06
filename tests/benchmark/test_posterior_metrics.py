"""Tests for ``cflibs.benchmark.posterior_metrics``.

Covers the four canonical fixtures listed in the validation spec:

1. A known-calibrated synthetic posterior (4 chains × 2000 draws from
   ``N(true, σ²)``) -> passes every gate.
2. An over-confident posterior (variance shrunk 5×) -> coverage drops
   below 0.93 and the bidirectional gate trips.
3. A posterior with divergent transitions metadata -> the divergent
   counter is detected and the gate trips.
4. An over-coverage posterior (intervals 5× too wide) -> *also* trips
   the bidirectional gate.

All four use deterministic ``np.random.default_rng`` seeds so the
fixtures are repeatable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from cflibs.benchmark.posterior_metrics import (
    PosteriorDiagnostics,
    compute_posterior_diagnostics,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _gaussian_posterior(
    truth: np.ndarray,
    *,
    sigma: float,
    n_chains: int = 4,
    n_draws: int = 2000,
    seed: int = 0,
) -> dict:
    """Generate a multivariate-Gaussian posterior centred on
    ``truth`` with standard deviation ``sigma`` per parameter.

    Returns a dict matching the public ``samples`` mapping shape
    ``{name: (n_chains, n_draws)}``. The full concentration vector is
    also exposed as ``"concentrations"`` (3-D, used for sharpness).
    """
    rng = np.random.default_rng(seed)
    n_params = len(truth)
    arr = rng.normal(
        loc=truth[None, None, :],
        scale=sigma,
        size=(n_chains, n_draws, n_params),
    )
    return {
        "concentrations": arr,
    }


def _certified_for(truth: np.ndarray, *, base: str = "concentrations") -> dict:
    return {f"{base}[{i}]": float(v) for i, v in enumerate(truth)}


def _calibrated_posterior_family(
    n_params: int = 200,
    *,
    sigma: float = 0.05,
    n_chains: int = 4,
    n_draws: int = 2000,
    seed: int = 11,
):
    """Build a *calibrated* family of Gaussian posteriors for the
    bidirectional coverage test.

    Each of the ``n_params`` parameters has:

    - a true value drawn from N(0, 1)  (the certified "ground truth"),
    - posterior draws sampled around (truth + noise) with std ``sigma``,
      where the noise per-parameter is itself drawn from N(0, sigma).

    With this construction the marginal probability that a 95% CI
    contains the truth is ~0.95 — i.e. coverage averaged across
    parameters lands inside the [0.93, 0.97] band with high
    probability.

    Returns ``(samples, certified)`` -- a samples dict with one
    array shape ``(n_chains, n_draws, n_params)`` and a certified-
    values dict.
    """
    rng = np.random.default_rng(seed)
    truth = rng.normal(0.0, 1.0, size=n_params)
    # Posterior centre is offset from truth by N(0, sigma) noise — this is
    # what an unbiased posterior looks like across many experiments.
    noise = rng.normal(0.0, sigma, size=n_params)
    centre = truth + noise
    arr = rng.normal(
        loc=centre[None, None, :],
        scale=sigma,
        size=(n_chains, n_draws, n_params),
    )
    samples = {"concentrations": arr}
    certified = _certified_for(truth)
    return samples, certified


# ---------------------------------------------------------------------------
# Test 1 — calibrated posterior passes
# ---------------------------------------------------------------------------


def test_calibrated_posterior_passes_all_gates():
    samples, certified = _calibrated_posterior_family(n_params=200, sigma=0.05, seed=42)

    diag = compute_posterior_diagnostics(samples, certified_values=certified)

    assert isinstance(diag, PosteriorDiagnostics)
    assert diag.n_chains == 4
    assert diag.n_draws == 2000
    # Convergence: 4 chains drawn iid from the same Gaussian -> R-hat ~1
    assert diag.rhat_max < 1.01, f"unexpectedly high R-hat: {diag.rhat_max}"
    # ESS: iid Gaussian draws yield close-to-N samples
    assert diag.ess_bulk_min >= 400, f"low bulk-ESS: {diag.ess_bulk_min}"
    assert diag.ess_tail_min >= 400, f"low tail-ESS: {diag.ess_tail_min}"
    assert diag.divergent_count == 0
    # Coverage averaged across 200 parameters with N(0, 1) truth and
    # N(truth, sigma) posterior centred-but-jittered should land near
    # 0.95 — the bidirectional band [0.93, 0.97] is wide enough to
    # comfortably contain it.
    assert diag.coverage_95 is not None
    assert diag.coverage_in_band is True, (
        f"coverage {diag.coverage_95} not in band; reasons={diag.reasons}"
    )
    # PIT chi-squared p-value should be high (uniformity not rejected).
    assert diag.pit_chi2_p_value is not None
    assert diag.pit_chi2_p_value > 0.01
    # Sharpness should be a finite, positive float in CLR space.
    assert diag.sharpness_clr is not None
    assert math.isfinite(diag.sharpness_clr)
    assert diag.sharpness_clr > 0.0
    # Hard gate verdict
    assert diag.passes_hard_gate is True, f"reasons={diag.reasons}"
    assert diag.reasons == []


# ---------------------------------------------------------------------------
# Test 2 — over-confident posterior trips under-coverage
# ---------------------------------------------------------------------------


def test_overconfident_posterior_trips_under_coverage():
    """Same calibrated-family construction, but the posterior σ is 5× too
    narrow vs. the noise on the centre -> intervals miss truth far more
    often than the nominal 5%."""
    rng = np.random.default_rng(7)
    n_params = 200
    truth_sigma = 0.05
    truth = rng.normal(0.0, 1.0, size=n_params)
    # Posterior centre is jittered by the *real* uncertainty but the
    # posterior credibly believes its uncertainty is 5x tighter.
    centre_noise = rng.normal(0.0, truth_sigma, size=n_params)
    centre = truth + centre_noise
    posterior_sigma = truth_sigma / 5.0  # over-confident
    arr = rng.normal(
        loc=centre[None, None, :],
        scale=posterior_sigma,
        size=(4, 2000, n_params),
    )
    samples = {"concentrations": arr}
    certified = _certified_for(truth)

    diag = compute_posterior_diagnostics(samples, certified_values=certified)

    # Coverage should drop well below 0.93.
    assert diag.coverage_95 is not None
    assert diag.coverage_95 < 0.93, f"expected under-coverage, got {diag.coverage_95}"
    assert diag.coverage_in_band is False
    assert diag.passes_hard_gate is False
    assert any("under-coverage" in r or "coverage_95" in r for r in diag.reasons)


# ---------------------------------------------------------------------------
# Test 3 — divergent transitions metadata is detected
# ---------------------------------------------------------------------------


def test_divergent_transitions_trip_gate():
    samples, certified = _calibrated_posterior_family(seed=11)

    diag = compute_posterior_diagnostics(samples, certified_values=certified, divergent_count=7)

    assert diag.divergent_count == 7
    assert diag.passes_hard_gate is False
    assert any("divergent_transitions=7" in r for r in diag.reasons)


# ---------------------------------------------------------------------------
# Test 4 — over-coverage (intervals 5× too wide) also trips the gate
# ---------------------------------------------------------------------------


def test_over_coverage_trips_bidirectional_gate():
    """Construct a posterior whose 95% credible interval is 5× wider
    than the centre noise -> coverage saturates well above 0.97 and
    the bidirectional band trips."""
    rng = np.random.default_rng(99)
    n_params = 200
    truth_sigma = 0.05
    truth = rng.normal(0.0, 1.0, size=n_params)
    centre_noise = rng.normal(0.0, truth_sigma, size=n_params)
    centre = truth + centre_noise
    # Posterior is 5x WIDER than the calibrated noise -> every CI
    # blanket-covers truth; coverage averages ~1.0 across 200 params.
    posterior_sigma = truth_sigma * 5.0
    arr = rng.normal(
        loc=centre[None, None, :],
        scale=posterior_sigma,
        size=(4, 2000, n_params),
    )
    samples = {"concentrations": arr}
    certified = _certified_for(truth)

    diag = compute_posterior_diagnostics(samples, certified_values=certified)

    assert diag.coverage_95 is not None
    assert diag.coverage_95 > 0.97, f"expected over-coverage, got {diag.coverage_95}"
    assert diag.coverage_in_band is False
    assert diag.passes_hard_gate is False
    assert any("over-coverage" in r for r in diag.reasons)


# ---------------------------------------------------------------------------
# Bonus: smoke-test single-chain inputs and missing certified values
# ---------------------------------------------------------------------------


def test_single_chain_input_does_not_crash():
    truth = np.array([0.4, 0.3, 0.3])
    rng = np.random.default_rng(5)
    arr = rng.normal(loc=truth[None, None, :], scale=0.05, size=(1, 1000, 3))
    samples = {"concentrations": arr}
    diag = compute_posterior_diagnostics(samples)
    # Coverage / PIT / sharpness behaviour:
    assert diag.coverage_95 is None
    assert diag.coverage_in_band is None
    assert diag.sharpness_clr is not None
    # Single-chain R-hat is by convention 1 (no between-chain variance);
    # ArviZ may also return NaN when only one chain is supplied.
    assert math.isnan(diag.rhat_max) or diag.rhat_max == pytest.approx(1.0, abs=0.05)


def test_dict_with_separate_param_arrays():
    """Smoke-test mixed scalar + vector parameter dicts.

    ``MCMCResult.samples`` typically provides ``T_eV`` (scalar) and
    ``concentrations`` (vector) side by side; the diagnostics module
    should handle both transparently.
    """
    truth_T = 1.2
    truth_c = np.array([0.5, 0.5])
    rng = np.random.default_rng(13)
    samples = {
        "T_eV": rng.normal(truth_T, 0.05, size=(4, 1500)),
        "concentrations": rng.normal(truth_c[None, None, :], 0.02, size=(4, 1500, 2)),
    }
    certified = {"T_eV": truth_T, "concentrations[0]": 0.5, "concentrations[1]": 0.5}
    diag = compute_posterior_diagnostics(samples, certified_values=certified)
    assert "T_eV" in diag.rhat
    assert "concentrations[0]" in diag.rhat
    assert "concentrations[1]" in diag.rhat
    assert diag.coverage_95 is not None


# ---------------------------------------------------------------------------
# Integration: unified benchmark plumbing via _build_composition_success_record
# ---------------------------------------------------------------------------


def test_composition_record_carries_posterior_diagnostics(tmp_path):
    """End-to-end: a composition workflow that returns a ``posterior_samples``
    payload should land posterior diagnostics in
    ``CompositionEvaluationRecord.annotations["posterior_diagnostics"]``,
    and those diagnostics must survive ``json.dumps`` (the
    ``write_outputs`` path uses ``json.dumps(..., default=str)``).
    """
    import json
    from dataclasses import asdict

    from cflibs.benchmark.dataset import (
        BenchmarkSpectrum,
        InstrumentalConditions,
        SampleMetadata,
        TruthType,
    )
    from cflibs.benchmark.unified import _build_composition_success_record

    elements = ["Si", "Fe", "Mg", "Ca"]
    truth = {"Si": 0.55, "Fe": 0.20, "Mg": 0.15, "Ca": 0.10}
    truth_arr = np.array([truth[e] for e in elements])

    rng = np.random.default_rng(31)
    posterior = rng.normal(loc=truth_arr, scale=0.02, size=(4, 1500, 4))

    prediction = {
        "concentrations": dict(truth),
        "posterior_samples": {"concentrations": posterior},
        "divergent_count": 0,
        "temperature_K": 8500.0,
    }

    spectrum = BenchmarkSpectrum(
        spectrum_id="test-fixture",
        wavelength_nm=np.linspace(200.0, 400.0, 64),
        intensity=np.ones(64),
        true_composition=truth,
        conditions=InstrumentalConditions(laser_wavelength_nm=1064.0, laser_energy_mj=50.0),
        metadata=SampleMetadata(sample_id="probe"),
        truth_type=TruthType.SYNTHETIC,
        dataset_id="unit-test",
    )

    record = _build_composition_success_record(
        spectrum=spectrum,
        id_workflow_name="probe",
        id_config_name="probe_config",
        composition_workflow_name="bayesian_probe",
        composition_config_name="bayesian_probe_default",
        outer_split_id="outer_0",
        tuning_split_id=None,
        elapsed_seconds=0.001,
        candidate_elements=elements,
        concentrations=dict(truth),
        prediction=prediction,
    )
    diag = record.annotations.get("posterior_diagnostics")
    assert diag is not None, f"annotations={record.annotations}"
    # Sanity-check a couple of fields.
    assert "rhat" in diag and "ess_bulk" in diag
    assert diag["divergent_count"] == 0
    assert diag["n_chains"] == 4
    assert diag["n_draws"] == 1500
    # The bulky raw-sample arrays should NOT have leaked into annotations.
    assert "posterior_samples" not in record.annotations
    # And the whole record must be JSON-serialisable (matches write_outputs).
    out = tmp_path / "composition_records.json"
    out.write_text(json.dumps([asdict(record)], indent=2, default=str))
    reloaded = json.loads(out.read_text())
    assert reloaded[0]["annotations"]["posterior_diagnostics"]["n_chains"] == 4
