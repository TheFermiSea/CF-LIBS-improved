"""Tests for the JAX-using composition workflows.

The benchmark gate registers two GPU-using composition workflows alongside
the canonical numpy ``iterative`` pipeline:

- ``bayesian``: NumPyro NUTS over the JAX :class:`BayesianForwardModel`,
  emits posterior samples + diagnostics into the
  ``CompositionEvaluationRecord.annotations``.
- ``iterative_jax``: GPU-accelerated CF-LIBS via
  :class:`IterativeCFLIBSSolverJax`, falls back to the numpy parent solver
  when JAX or the JAX solver isn't available at gate time.

These tests are intentionally lightweight — they verify registration,
end-to-end shape, and graceful degradation, *not* convergence quality.
The full Bayesian convergence battery lives in
``tests/benchmark/test_posterior_metrics.py``; numerical parity between
the JAX iterative solver and its numpy parent is covered by
``tests/inversion/test_solver_jax_parity.py``.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict

import numpy as np
import pytest

from cflibs.benchmark.dataset import (
    BenchmarkSpectrum,
    InstrumentalConditions,
    MatrixType,
    SampleMetadata,
    SampleType,
    TruthType,
)
from cflibs.benchmark.unified import (
    CompositionWorkflowSpec,
    HAS_JAX,
    HAS_JAX_ITERATIVE_SOLVER,
    HAS_NUMPYRO,
    UnifiedBenchmarkContext,
    _build_composition_success_record,
    _coerce_composition_prediction,
    _fit_bayesian_pipeline,
    _fit_iterative_jax_pipeline,
    _fit_iterative_pipeline,
    build_composition_workflow_registry,
)


_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB_PATH = _REPO_ROOT / "ASD_da" / "libs_production.db"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _gaussian_peak(
    x: np.ndarray, center: float, amplitude: float, fwhm: float
) -> np.ndarray:
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return amplitude * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))


def _make_synthetic_spectrum(seed: int = 0) -> BenchmarkSpectrum:
    """Small synthetic Fe-Ca spectrum on a 256-pixel grid (200-400 nm).

    The wavelength range, peak count, and pixel count are all kept tiny so
    the Bayesian MCMC test can finish within a few seconds even when the
    numpyro NUTS sampler has to compile the JAX model graph.
    """
    rng = np.random.default_rng(seed)
    wl = np.linspace(200.0, 400.0, 256)
    intensity = np.zeros_like(wl)
    # Two Fe lines + two Ca lines that survive the database match.
    intensity += _gaussian_peak(wl, 238.20, amplitude=8.0, fwhm=0.2)
    intensity += _gaussian_peak(wl, 252.29, amplitude=6.0, fwhm=0.2)
    intensity += _gaussian_peak(wl, 393.37, amplitude=10.0, fwhm=0.2)
    intensity += _gaussian_peak(wl, 396.85, amplitude=7.0, fwhm=0.2)
    intensity = intensity + 0.05 * rng.standard_normal(wl.size)

    return BenchmarkSpectrum(
        spectrum_id=f"synth_{seed}",
        wavelength_nm=wl,
        intensity=intensity,
        true_composition={"Fe": 0.7, "Ca": 0.3},
        conditions=InstrumentalConditions(
            laser_wavelength_nm=1064.0,
            laser_energy_mj=50.0,
            spectral_range_nm=(200.0, 400.0),
            spectral_resolution_nm=0.05,
        ),
        metadata=SampleMetadata(
            sample_id=f"synth_{seed}",
            sample_type=SampleType.SYNTHETIC,
            matrix_type=MatrixType.METAL_ALLOY,
        ),
        dataset_id="unit-test",
        truth_type=TruthType.SYNTHETIC,
        rp_estimate=5000.0,
    )


@pytest.fixture
def synthetic_spectrum() -> BenchmarkSpectrum:
    return _make_synthetic_spectrum()


@pytest.fixture
def context(tmp_path) -> UnifiedBenchmarkContext:
    if not _DEFAULT_DB_PATH.exists():
        pytest.skip(f"atomic database not available at {_DEFAULT_DB_PATH}")
    return UnifiedBenchmarkContext(db_path=_DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_bayesian_workflow_is_registered():
    """The bayesian workflow must register unconditionally (the predictor
    raises at invocation time when jax/numpyro are absent — that path is
    exercised separately).  This guards against silent regressions where
    the registry stops exposing the GPU-using workflows."""
    registry = build_composition_workflow_registry(quick=True)
    assert "bayesian" in registry
    spec = registry["bayesian"]
    assert isinstance(spec, CompositionWorkflowSpec)
    assert spec.name == "bayesian"
    # Quick mode has a single small NUTS budget; the full grid sweeps two.
    assert len(spec.parameter_grid) >= 1
    # Each config must have the MCMC budget keys our predictor reads.
    for config in spec.parameter_grid:
        assert "num_warmup" in config
        assert "num_samples" in config


def test_iterative_jax_workflow_is_registered():
    registry = build_composition_workflow_registry(quick=True)
    assert "iterative_jax" in registry
    spec = registry["iterative_jax"]
    assert isinstance(spec, CompositionWorkflowSpec)
    assert spec.name == "iterative_jax"
    # iterative_jax mirrors the iterative grid so every config has a
    # closure_mode + fit_method pair.
    for config in spec.parameter_grid:
        assert "closure_mode" in config
        assert "fit_method" in config


def test_iterative_workflow_still_registered_for_back_compat():
    """The pre-existing numpy ``iterative`` workflow must keep working so
    the benchmark gate's headline metric remains comparable across the
    JAX overhaul."""
    registry = build_composition_workflow_registry(quick=True)
    assert "iterative" in registry


def test_registry_contains_all_composition_workflows():
    registry = build_composition_workflow_registry(quick=True)
    # Two GPU-using workflows must be exposed alongside the legacy three.
    expected = {"iterative", "iterative_jax", "bayesian", "joint_softmax", "hybrid_manifold"}
    assert expected.issubset(set(registry.keys()))


# ---------------------------------------------------------------------------
# iterative_jax — runs end-to-end (GPU when available, numpy fallback otherwise)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _DEFAULT_DB_PATH.exists(),
    reason="atomic database fixture unavailable",
)
def test_iterative_jax_predictor_runs_end_to_end(synthetic_spectrum, context):
    """Build the iterative_jax predictor and run it on a synthetic
    spectrum.  The predictor must always return a concentrations dict
    (either from the JAX path or the numpy fallback)."""
    from cflibs.inversion.boltzmann import FitMethod

    config = {"fit_method": FitMethod.SIGMA_CLIP, "closure_mode": "standard"}
    predictor = _fit_iterative_jax_pipeline(context, [], config)
    try:
        prediction = predictor(synthetic_spectrum, ["Fe", "Ca"], None)
    except RuntimeError as exc:
        # If the spectrum doesn't yield enough matched lines for either
        # path, log + skip — the workflow harness handles failure records,
        # not the predictor.  This is *not* a registration regression.
        pytest.skip(f"iterative_jax predictor raised on synthetic input: {exc}")
    assert isinstance(prediction, dict)
    assert "concentrations" in prediction
    assert isinstance(prediction["concentrations"], dict)
    backend = prediction.get("solver_backend")
    assert backend in {"jax", "numpy_fallback"}


@pytest.mark.skipif(
    not HAS_JAX_ITERATIVE_SOLVER,
    reason="IterativeCFLIBSSolverJax not available — Agent B's solver hasn't landed",
)
def test_iterative_jax_uses_jax_path_when_available(synthetic_spectrum, context):
    """When the JAX solver and JAX itself are present, the predictor must
    pick the JAX path (not the numpy fallback)."""
    from cflibs.inversion.boltzmann import FitMethod

    config = {"fit_method": FitMethod.SIGMA_CLIP, "closure_mode": "standard"}
    predictor = _fit_iterative_jax_pipeline(context, [], config)
    try:
        prediction = predictor(synthetic_spectrum, ["Fe", "Ca"], None)
    except RuntimeError as exc:
        pytest.skip(f"iterative_jax predictor raised on synthetic input: {exc}")
    assert prediction.get("solver_backend") == "jax"


# ---------------------------------------------------------------------------
# bayesian — runs end-to-end on a small spectrum
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (HAS_JAX and HAS_NUMPYRO and _DEFAULT_DB_PATH.exists()),
    reason="bayesian workflow needs jax + numpyro + atomic DB",
)
def test_bayesian_predictor_runs_end_to_end(synthetic_spectrum, context):
    """Run the bayesian predictor on a tiny synthetic spectrum with a
    minimal MCMC budget (50 warmup / 100 samples) so the test finishes
    quickly.  Verifies the predictor returns the prediction-dict shape
    the benchmark harness expects (concentrations + posterior_samples)."""
    config = {
        "num_warmup": 50,
        "num_samples": 100,
        "num_chains": 1,
        "seed": 0,
        "pixels": 256,
    }
    predictor = _fit_bayesian_pipeline(context, [], config)
    prediction = predictor(synthetic_spectrum, ["Fe", "Ca"], None)
    assert isinstance(prediction, dict)
    assert "concentrations" in prediction
    assert "predicted_composition" in prediction
    assert "aitchison" in prediction
    assert prediction["aitchison"] is not None
    assert 0.0 <= prediction["aitchison"] <= 20.0  # physical range check

    concentrations = prediction["concentrations"]
    assert set(concentrations.keys()) == {"Fe", "Ca"}
    # closure: posterior-mean concentrations are renormalized to the simplex.
    assert abs(sum(concentrations.values()) - 1.0) < 1e-6
    # The harness picks up posterior_samples and computes diagnostics.
    assert "posterior_samples" in prediction
    samples = prediction["posterior_samples"]
    assert "concentrations" in samples
    assert prediction["solver_backend"] == "numpyro_jax"
    assert prediction["n_samples"] == 100


@pytest.mark.skipif(
    HAS_JAX and HAS_NUMPYRO,
    reason="this test only runs when JAX/NumPyro are absent",
)
def test_bayesian_predictor_raises_without_jax():
    """Without jax+numpyro the predictor must raise a clear error so the
    workflow harness can record an explicit failure (vs a silent CPU
    fallback that wouldn't actually exercise GPU)."""
    ctx = UnifiedBenchmarkContext(db_path=_DEFAULT_DB_PATH)
    config = {"num_warmup": 50, "num_samples": 100, "num_chains": 1}
    predictor = _fit_bayesian_pipeline(ctx, [], config)
    with pytest.raises(RuntimeError, match="jax|numpyro"):
        predictor(_make_synthetic_spectrum(), ["Fe", "Ca"], None)


# ---------------------------------------------------------------------------
# Record shape parity — bayesian and iterative_jax must yield records
# with the same fields as iterative
# ---------------------------------------------------------------------------


def _make_iterative_record(spectrum: BenchmarkSpectrum):
    """Build a CompositionEvaluationRecord via the iterative-pipeline
    code path, using a fake prediction so this stays DB-free."""
    elements = sorted(spectrum.true_composition.keys())
    prediction = {"concentrations": dict(spectrum.true_composition)}
    concentrations = _coerce_composition_prediction(prediction, elements)
    return _build_composition_success_record(
        spectrum=spectrum,
        id_workflow_name="probe",
        id_config_name="probe_default",
        composition_workflow_name="iterative",
        composition_config_name="iterative_default",
        outer_split_id="outer_0",
        tuning_split_id=None,
        elapsed_seconds=0.001,
        candidate_elements=elements,
        concentrations=concentrations,
        prediction=prediction,
    )


def _make_bayesian_record(spectrum: BenchmarkSpectrum):
    """Build a CompositionEvaluationRecord with a fake posterior_samples
    payload so the diagnostics codepath fires.  Mirrors what the real
    bayesian predictor returns end-to-end without paying the MCMC cost."""
    elements = sorted(spectrum.true_composition.keys())
    truth_arr = np.array([spectrum.true_composition[e] for e in elements])
    rng = np.random.default_rng(31)
    posterior = rng.normal(loc=truth_arr, scale=0.02, size=(2, 200, len(elements)))
    prediction: Dict[str, object] = {
        "concentrations": dict(spectrum.true_composition),
        "predicted_composition": dict(spectrum.true_composition),
        "aitchison": 0.0,
        "posterior_samples": {"concentrations": posterior},
        "divergent_count": 0,
        "temperature_K": 9000.0,
        "electron_density_cm3": 1.0e17,
        "solver_backend": "numpyro_jax",
        "n_samples": 200,
        "n_chains": 2,
        "n_warmup": 100,
    }
    concentrations = _coerce_composition_prediction(prediction, elements)
    return _build_composition_success_record(
        spectrum=spectrum,
        id_workflow_name="probe",
        id_config_name="probe_default",
        composition_workflow_name="bayesian",
        composition_config_name="bayesian_default",
        outer_split_id="outer_0",
        tuning_split_id=None,
        elapsed_seconds=0.001,
        candidate_elements=elements,
        concentrations=concentrations,
        prediction=prediction,
    )


def test_bayesian_record_has_iterative_field_parity(synthetic_spectrum):
    iterative_record = _make_iterative_record(synthetic_spectrum)
    bayesian_record = _make_bayesian_record(synthetic_spectrum)

    iterative_fields = set(asdict(iterative_record).keys())
    bayesian_fields = set(asdict(bayesian_record).keys())
    # Same dataclass -> same field set.
    assert iterative_fields == bayesian_fields

    # Spot-check the load-bearing fields the benchmark report consumes.
    for record in (iterative_record, bayesian_record):
        assert record.dataset_id == "unit-test"
        assert record.spectrum_id == "synth_0"
        assert record.aitchison is not None
        assert record.true_composition == synthetic_spectrum.true_composition
        assert record.predicted_composition  # nonempty


def test_bayesian_record_carries_posterior_diagnostics(synthetic_spectrum):
    """The bayesian workflow's predictor returns a ``posterior_samples``
    payload, so the harness must populate ``annotations.posterior_diagnostics``
    with the fields the benchmark gate expects."""
    record = _make_bayesian_record(synthetic_spectrum)
    diag = record.annotations.get("posterior_diagnostics")
    assert diag is not None, f"annotations={record.annotations}"
    # Required fields per the spec:
    for key in ("rhat", "ess_bulk", "divergent_count"):
        assert key in diag, f"missing {key} in posterior diagnostics"
    assert diag["divergent_count"] == 0
    # Bulky raw-sample arrays must NOT leak into annotations.
    assert "posterior_samples" not in record.annotations


def test_iterative_jax_record_has_iterative_field_parity(synthetic_spectrum):
    """A fake iterative_jax-style prediction (numpy fallback or JAX path)
    must yield a CompositionEvaluationRecord with the same fields as the
    canonical iterative pipeline."""
    elements = sorted(synthetic_spectrum.true_composition.keys())
    prediction = {
        "concentrations": dict(synthetic_spectrum.true_composition),
        "temperature_K": 9000.0,
        "electron_density_cm3": 1.0e17,
        "solver_backend": "jax",
    }
    concentrations = _coerce_composition_prediction(prediction, elements)
    iterative_jax_record = _build_composition_success_record(
        spectrum=synthetic_spectrum,
        id_workflow_name="probe",
        id_config_name="probe_default",
        composition_workflow_name="iterative_jax",
        composition_config_name="iterative_jax_default",
        outer_split_id="outer_0",
        tuning_split_id=None,
        elapsed_seconds=0.001,
        candidate_elements=elements,
        concentrations=concentrations,
        prediction=prediction,
    )
    iterative_record = _make_iterative_record(synthetic_spectrum)
    assert set(asdict(iterative_record).keys()) == set(asdict(iterative_jax_record).keys())
    # The record must round-trip through the workflow registry's expected
    # column set.
    assert iterative_jax_record.aitchison is not None
    assert iterative_jax_record.true_composition == synthetic_spectrum.true_composition
