"""
Workflow predictor builders for the unified LIBS benchmark.

This module owns the closure factories — ``_build_*_predictor`` and
``_fit_*_pipeline`` — that translate an :class:`IDWorkflowSpec` /
:class:`CompositionWorkflowSpec` parameter grid into a concrete
``Callable[[BenchmarkSpectrum], ...]`` predictor used by the runner.

Why a dedicated module
----------------------
Extracted from ``cflibs/benchmark/unified.py`` (which had grown to ~3.5
kLoC mixing the workflow registry, the runner class, output writers,
the composition evaluator, and these builders) so the workflow surface
can be tested + reasoned about in isolation.

``unified.py`` re-exports every symbol from this module so existing
callers — ``UnifiedBenchmarkRunner.run_composition``,
``tune_id_workflow``, ``tune_composition_workflow``, third-party tests
that imported the leading-underscore helpers — keep working verbatim.

Why **dependency injection** lives here implicitly
--------------------------------------------------
The builders all depend on a small set of module-level helpers:

* ``_jax_identifier_flags_for`` — toggles JAX kwargs on each identifier
* ``_estimate_rp_for_spectrum`` / ``_estimate_effective_rp_lazy`` — RP fallback
* ``_load_repo_script_module`` and its ``_*_lazy`` thin wrappers — pulls
  legacy ``scripts/*.py`` modules in via ``importlib`` so the heavyweight
  Boltzmann / joint-softmax / hybrid-manifold pipelines stay lazily-loaded
* ``HAS_JAX`` / ``HAS_NUMPYRO`` / ``HAS_JAX_ITERATIVE_SOLVER`` feature flags

Rather than threading these through every builder as explicit kwargs
(which would balloon every signature), we keep them as *module-private*
to ``workflows.py`` and re-export the public surface from ``unified.py``.
``unified.py`` callers (e.g. ``load_aalto_id_dataset``) that previously
called the lazy helpers continue to work because we re-export them from
this module via ``unified.py``. There is no import cycle because this
module never imports ``unified.py`` at runtime — only ``TYPE_CHECKING``
for the :class:`UnifiedBenchmarkContext` type hint.
"""

from __future__ import annotations

import importlib.util
import inspect
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cflibs.benchmark.composition_metrics import aitchison_distance
from cflibs.benchmark.dataset import BenchmarkSpectrum
from cflibs.core.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - import-time hints only
    from cflibs.benchmark.unified import (
        IDWorkflowSpec,  # noqa: F401  (re-export type only)
        UnifiedBenchmarkContext,
    )
    from cflibs.inversion.element_id import ElementIdentificationResult


logger = get_logger("benchmark.workflows")


# ---------------------------------------------------------------------------
# JAX / NumPyro feature flags
# ---------------------------------------------------------------------------
# These mirror the conditional imports that used to live at module load
# time in ``unified.py``.  We re-evaluate them here so the workflow gate
# can fall back cleanly on hosts where JAX / NumPyro / the JAX iterative
# solver aren't installed.

try:
    import jax  # noqa: F401

    HAS_JAX = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_JAX = False

try:
    import numpyro  # noqa: F401

    HAS_NUMPYRO = True
except ImportError:  # pragma: no cover - optional dependency
    HAS_NUMPYRO = False

try:  # IterativeCFLIBSSolverJax is being deprecated per ADR-0001; the
    # benchmark gate must still tolerate its absence (test envs without
    # JAX) so we capture the flag at import time.
    from cflibs.inversion.solver import IterativeCFLIBSSolverJax  # type: ignore

    HAS_JAX_ITERATIVE_SOLVER = True
except (ImportError, AttributeError):  # pragma: no cover - exercised when absent
    IterativeCFLIBSSolverJax = None  # type: ignore[assignment]
    HAS_JAX_ITERATIVE_SOLVER = False


try:
    from scipy.signal import find_peaks
except ImportError:  # pragma: no cover - optional dependency
    find_peaks = None


# ---------------------------------------------------------------------------
# Lazy-loader plumbing for legacy ``scripts/*.py`` modules
# ---------------------------------------------------------------------------
# ``run_boltzmann_pipeline``, ``_pipeline_joint_softmax``, the Aalto case
# loader, etc. live as top-level functions inside repo-root ``scripts/``
# files.  Pulling them in eagerly would force every benchmark consumer
# to pay the script-import cost (and risk circular imports against
# ``cflibs.benchmark.dataset``).  Instead we wrap them in tiny
# ``_*_lazy`` shims that ``importlib.util.spec_from_file_location`` the
# script on first use and cache the module.

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_MODULE_CACHE: Dict[str, Any] = {}


def _load_repo_script_module(module_name: str) -> Any:
    if module_name in _SCRIPT_MODULE_CACHE:
        return _SCRIPT_MODULE_CACHE[module_name]

    script_path = _REPO_ROOT / "scripts" / f"{module_name}.py"
    if not script_path.exists():
        raise ImportError(
            f"Unified benchmark helper '{module_name}' requires {script_path} to be present."
        )

    spec = importlib.util.spec_from_file_location(
        f"_cflibs_repo_scripts_{module_name}", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load helper module from {script_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    _SCRIPT_MODULE_CACHE[module_name] = module
    return module


def _aalto_search_elements() -> List[str]:
    return list(_load_repo_script_module("calibrate_alias").AALTO_SEARCH_ELEMENTS)


def _select_aalto_cases_lazy(data_dir: Path):
    return _load_repo_script_module("calibrate_alias")._select_aalto_cases(data_dir)


def _estimate_effective_rp_lazy(wavelength_nm: np.ndarray, intensity: np.ndarray) -> float:
    return float(
        _load_repo_script_module("calibrate_alias")._estimate_effective_rp(wavelength_nm, intensity)
    )


def _load_real_spectra_lazy(data_dir: Path) -> List[Dict[str, Any]]:
    return _load_repo_script_module("run_comprehensive_benchmark").load_real_spectra(data_dir)


SUPPORTED_BOLTZMANN_WEIGHTINGS = {None, "aki_inverse_variance", "intensity_only"}


def _validate_boltzmann_weighting(weighting: Optional[str]) -> Optional[str]:
    if weighting not in SUPPORTED_BOLTZMANN_WEIGHTINGS:
        allowed = ", ".join(sorted(w for w in SUPPORTED_BOLTZMANN_WEIGHTINGS if w is not None))
        raise ValueError(
            f"Unsupported Boltzmann weighting {weighting!r}; expected one of {allowed}"
        )
    return weighting


def _run_boltzmann_pipeline_lazy(
    spectrum: Dict[str, Any],
    db: Any,
    fit_method: Any,
    closure_mode: str,
    elements: Optional[List[str]] = None,
    **kwargs: Any,
) -> Optional[Dict[str, float]]:
    if "weighting" in kwargs:
        kwargs["weighting"] = _validate_boltzmann_weighting(kwargs["weighting"])
    return _load_repo_script_module("run_comprehensive_benchmark").run_boltzmann_pipeline(
        spectrum,
        db=db,
        fit_method=fit_method,
        closure_mode=closure_mode,
        elements=elements,
        **kwargs,
    )


def _pipeline_joint_softmax_lazy(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    elements: List[str],
) -> Dict[str, Any]:
    return _load_repo_script_module("run_experiments_advanced")._pipeline_joint_softmax(
        wavelength_nm, intensity, elements
    )


def _pipeline_hybrid_manifold_lazy(
    wavelength_nm: np.ndarray,
    intensity: np.ndarray,
    elements: List[str],
) -> Dict[str, Any]:
    return _load_repo_script_module("run_experiments_advanced")._pipeline_hybrid_manifold(
        wavelength_nm, intensity, elements
    )


# ---------------------------------------------------------------------------
# JAX identifier opt-in
# ---------------------------------------------------------------------------


def _jax_identifier_flags_for(cls) -> Dict[str, bool]:
    """
    Return ``{flag: True}`` for every ``use_jax_*`` kwarg accepted by *cls*.

    Gated on ``CFLIBS_USE_JAX_IDENTIFIER=1``.  When the env var is unset or
    "0", returns an empty dict (preserves the as-shipped CPU-only behavior
    of every identifier).  When set to "1", auto-detects each identifier's
    ``use_jax_*`` constructor kwargs via ``inspect.signature`` and turns
    them all on.

    This is the toggle the unified benchmark CLI flips via
    ``--jax-identifier``.  See PR #118, #119, #120, #121, #122 for the
    identifier-side opt-in flags this targets.
    """
    if os.environ.get("CFLIBS_USE_JAX_IDENTIFIER", "0") != "1":
        return {}
    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}
    return {name: True for name in sig.parameters if name.startswith("use_jax_")}


def _estimate_rp_for_spectrum(spectrum: BenchmarkSpectrum) -> float:
    if spectrum.rp_estimate is not None and np.isfinite(spectrum.rp_estimate):
        return float(spectrum.rp_estimate)
    return _estimate_effective_rp_lazy(spectrum.wavelength_nm, spectrum.intensity)


# ---------------------------------------------------------------------------
# Identification workflow predictor builders
# ---------------------------------------------------------------------------


def _build_alias_high_recall_predictor(
    context: "UnifiedBenchmarkContext",
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
    """Predictor for ``alias_high_recall``.

    Constructs ``ALIASIdentifier(high_recall=True)`` WITHOUT explicit
    threshold kwargs so PR #159's recall preset resolves them. Otherwise
    byte-identical to ``_build_alias_predictor``. Records the active
    mode in ``result.parameters['alias_mode']`` for downstream
    audit-ability.
    """

    def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.alias_identifier import ALIASIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            identifier = ALIASIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                high_recall=True,
                chance_window_scale=float(config["chance_window_scale"]),
                max_lines_per_element=int(config["max_lines_per_element"]),
                **_jax_identifier_flags_for(ALIASIdentifier),
            )
            result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
            result.parameters["candidate_elements"] = list(candidate_elements)
            result.parameters["alias_mode"] = "high_recall"
            return result

    return predictor


def _build_alias_v2_predictor(
    context: "UnifiedBenchmarkContext",
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
    """Predictor for ``alias_v2`` — Phase D promotion of the ftp1+dj6y
    sweep winner.

    Bakes in the two Phase B fix flags whose combination won the
    Phase C sweep:
      r2_gate_mode='adaptive_t' (PR #175, fix γ ftp1)
      r2_gate_t_quality_threshold=5500.0  (PR #175 default)
      relative_cl_per_ion_stage=True (PR #176, fix ε dj6y)

    NOT baked in:
      temperature_estimator_mode='legacy' (NOT 'robust' — PR #177 fix δ
      762f cancels ftp1's gain because it warms T above the 5500K
      relaxation threshold, eliminating the adaptive_t branch. See the
      empirical 07 v2 sweep report for the mutual-exclusion analysis.)

    Strict thresholds (intensity_threshold_factor=3.0, detection_threshold=0.02)
    are preserved by NOT passing them through the constructor — same
    pattern as alias_high_recall. ``alias_v2`` is additive: the existing
    ``alias`` workflow is unchanged so the precision=1.000 baseline at
    .swarm/identifier-f1-baseline.json continues to hold.
    """

    def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.alias_identifier import ALIASIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            identifier = ALIASIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                r2_gate_mode="adaptive_t",
                relative_cl_per_ion_stage=True,
                chance_window_scale=float(config["chance_window_scale"]),
                max_lines_per_element=int(config["max_lines_per_element"]),
                **_jax_identifier_flags_for(ALIASIdentifier),
            )
            result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
            result.parameters["candidate_elements"] = list(candidate_elements)
            result.parameters["alias_mode"] = "v2_ftp1_plus_dj6y"
            return result

    return predictor


# ---------------------------------------------------------------------------
# Alias-fix sweep harness (Phase C of jaunty-weaving-mist).
#
# Enumerates the 2^3 = 8 combinations of the three opt-in fix flags landed in
# PRs #175 (ftp1 — adaptive_t r²-gate), #177 (762f — robust temperature
# estimator), and #176 (dj6y — per-ion-stage relative_cl_threshold). Each cell
# constructs an ``ALIASIdentifier`` with the cell-specific fix-flag kwargs,
# while pinning the strict threshold kwargs to their precision-king defaults
# (3.0 / 0.02 / 0.4 / 30) so the ONLY difference across cells is the 3 fix
# flags. Designed for the alias-fix sweep at
# ``scripts/sweep_alias_fixes.py``; expected wall-time on the cluster is
# 8 cells × 5 seeds × ~10 min/run ≈ 7 h serial, ≈ 2.5 h with 3-way parallelism.
# ---------------------------------------------------------------------------

# (cell_name, fix-flag kwargs) — order matches the plan's configuration table.
_ALIAS_SWEEP_CELLS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("baseline", {}),
    ("ftp1", {"r2_gate_mode": "adaptive_t"}),
    ("762f", {"temperature_estimator_mode": "robust"}),
    ("dj6y", {"relative_cl_per_ion_stage": True}),
    ("ftp1+762f", {"r2_gate_mode": "adaptive_t", "temperature_estimator_mode": "robust"}),
    ("ftp1+dj6y", {"r2_gate_mode": "adaptive_t", "relative_cl_per_ion_stage": True}),
    ("762f+dj6y", {"temperature_estimator_mode": "robust", "relative_cl_per_ion_stage": True}),
    (
        "all_three",
        {
            "r2_gate_mode": "adaptive_t",
            "temperature_estimator_mode": "robust",
            "relative_cl_per_ion_stage": True,
        },
    ),
)


# Strict threshold defaults shared by every sweep cell. Pinned explicitly here
# (rather than read from ``_class_default_config``) so a future change to
# ``ALIASIdentifier.__init__`` defaults can't silently flip the sweep's
# baseline. These ARE the precision-king defaults guarded by
# ``test_strict_alias_still_registered_unchanged`` for the strict alias
# workflow.
_ALIAS_SWEEP_BASE_KWARGS: Dict[str, Any] = {
    "intensity_threshold_factor": 3.0,
    "detection_threshold": 0.02,
    "chance_window_scale": 0.4,
    "max_lines_per_element": 30,
}


def _alias_sweep_workflow_configs(
    quick: bool,  # noqa: ARG001 - signature parity
) -> List[Dict[str, Any]]:
    """Single-element config grid shared by every ``alias_sweep_*`` workflow.

    All 8 cells use the SAME threshold kwargs (pinned at strict defaults via
    the predictor factory below) — the only thing that varies is the fix-flag
    kwargs baked into the per-cell predictor closure. So every cell has
    exactly one config; cross-cell variation lives in the predictor, not the
    grid. The ``quick`` parameter is accepted for signature parity with the
    other ``_<wf>_workflow_configs`` helpers but is intentionally unused.
    """
    return [{}]


def _build_alias_sweep_predictor_factory(
    cell_kwargs: Dict[str, Any],
    cell_name: str,
) -> Callable[
    ["UnifiedBenchmarkContext", List[str], Dict[str, Any]],
    Callable[[BenchmarkSpectrum], "ElementIdentificationResult"],
]:
    """Return a ``build_predictor`` callable for one sweep cell.

    The returned callable matches the signature expected by
    ``IDWorkflowSpec.build_predictor``. ``cell_kwargs`` is the per-cell
    fix-flag kwarg dict (one of ``_ALIAS_SWEEP_CELLS`` values); ``cell_name``
    is recorded in ``result.parameters['alias_sweep_cell']`` for downstream
    audit-ability.
    """

    def build_predictor(
        context: "UnifiedBenchmarkContext",
        candidate_elements: List[str],
        config: Dict[str, Any],  # noqa: ARG001 - empty grid, see configs helper
    ) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
        def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
            from cflibs.atomic.database import AtomicDatabase
            from cflibs.inversion.alias_identifier import ALIASIdentifier

            with AtomicDatabase(str(context.db_path)) as db:
                identifier = ALIASIdentifier(
                    atomic_db=db,
                    elements=candidate_elements,
                    resolving_power=_estimate_rp_for_spectrum(spectrum),
                    intensity_threshold_factor=float(
                        _ALIAS_SWEEP_BASE_KWARGS["intensity_threshold_factor"]
                    ),
                    detection_threshold=float(_ALIAS_SWEEP_BASE_KWARGS["detection_threshold"]),
                    chance_window_scale=float(_ALIAS_SWEEP_BASE_KWARGS["chance_window_scale"]),
                    max_lines_per_element=int(_ALIAS_SWEEP_BASE_KWARGS["max_lines_per_element"]),
                    **cell_kwargs,
                    **_jax_identifier_flags_for(ALIASIdentifier),
                )
                result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
                result.parameters["candidate_elements"] = list(candidate_elements)
                result.parameters["alias_sweep_cell"] = cell_name
                result.parameters["alias_sweep_fix_kwargs"] = dict(cell_kwargs)
                return result

        return predictor

    return build_predictor


def _build_alias_predictor(
    context: "UnifiedBenchmarkContext",
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
    def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.alias_identifier import ALIASIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            identifier = ALIASIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                intensity_threshold_factor=float(config["intensity_threshold_factor"]),
                detection_threshold=float(config["detection_threshold"]),
                chance_window_scale=float(config["chance_window_scale"]),
                max_lines_per_element=int(config["max_lines_per_element"]),
                boltzmann_r2_min=float(config.get("boltzmann_r2_min", 0.85)),
                **_jax_identifier_flags_for(ALIASIdentifier),
            )
            result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_comb_predictor(
    context: "UnifiedBenchmarkContext",
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
    def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.comb_identifier import CombIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            identifier = CombIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                min_correlation=float(config["min_correlation"]),
                tooth_activation_threshold=float(config["tooth_activation_threshold"]),
                relative_threshold_scale=float(config["relative_threshold_scale"]),
                min_aki_gk=3000.0,
                **_jax_identifier_flags_for(CombIdentifier),
            )
            result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_correlation_predictor(
    context: "UnifiedBenchmarkContext",
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
    def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.correlation_identifier import CorrelationIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            identifier = CorrelationIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                min_confidence=float(config["min_confidence"]),
                relative_threshold_scale=float(config["relative_threshold_scale"]),
                min_line_strength=float(config["min_line_strength"]),
                T_range_K=(5000, 15000),
                T_steps=7,
                n_e_range_cm3=(1e15, 5e17),
                n_e_steps=4,
                **_jax_identifier_flags_for(CorrelationIdentifier),
            )
            result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity, mode="classic")
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_nnls_predictor(
    context: "UnifiedBenchmarkContext",
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
    def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
        from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier

        basis, basis_fwhm, mismatch = context.basis_for_rp(spectrum.rp_estimate)
        identifier = SpectralNNLSIdentifier(
            basis_library=basis,
            detection_snr=float(config["detection_snr"]),
            continuum_degree=int(config["continuum_degree"]),
            fallback_T_K=float(config["fallback_T_K"]),
            fallback_ne_cm3=1e17,
            **_jax_identifier_flags_for(SpectralNNLSIdentifier),
        )
        result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
        result.parameters["basis_fwhm_nm"] = basis_fwhm
        result.parameters["basis_fwhm_mismatch_nm"] = mismatch
        result.parameters["candidate_elements"] = list(candidate_elements)
        return result

    return predictor


def _build_hybrid_predictor(
    context: "UnifiedBenchmarkContext",
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
    def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.hybrid_identifier import HybridIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            basis, basis_fwhm, mismatch = context.basis_for_rp(spectrum.rp_estimate)
            identifier = HybridIdentifier(
                atomic_db=db,
                basis_library=basis,
                elements=candidate_elements,
                resolving_power=_estimate_rp_for_spectrum(spectrum),
                nnls_detection_snr=float(config["nnls_detection_snr"]),
                alias_detection_threshold=float(config["alias_detection_threshold"]),
                require_both=bool(config["require_both"]),
            )
            result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
            result.parameters["basis_fwhm_nm"] = basis_fwhm
            result.parameters["basis_fwhm_mismatch_nm"] = mismatch
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_hybrid_consensus_2of3_predictor(
    context: "UnifiedBenchmarkContext",
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
    """Predictor for ``hybrid_consensus_2of3``.

    Constructs ALIAS, Comb, and Correlation identifiers, runs them on the
    spectrum, then combines their outputs via a 2-of-3 majority vote using
    :class:`cflibs.inversion.identify.hybrid_consensus.HybridConsensusIdentifier`.

    This workflow trades recall for precision: an element must be confirmed
    by at least 2 of the 3 line-matchers to be reported as detected.
    """

    def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.alias_identifier import ALIASIdentifier
        from cflibs.inversion.comb_identifier import CombIdentifier
        from cflibs.inversion.correlation_identifier import CorrelationIdentifier
        from cflibs.inversion.identify.hybrid_consensus import (
            HybridConsensusIdentifier,
        )

        rp = _estimate_rp_for_spectrum(spectrum)

        with AtomicDatabase(str(context.db_path)) as db:
            # Build the three constituent identifiers with their default
            # strict thresholds (matching the alias/comb/correlation workflows).
            alias_id = ALIASIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=rp,
                intensity_threshold_factor=3.0,
                detection_threshold=0.02,
                chance_window_scale=0.4,
                max_lines_per_element=30,
                **_jax_identifier_flags_for(ALIASIdentifier),
            )
            comb_id = CombIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=rp,
                min_correlation=0.08,
                tooth_activation_threshold=0.35,
                relative_threshold_scale=1.4,
                min_aki_gk=3000.0,
                **_jax_identifier_flags_for(CombIdentifier),
            )
            correlation_id = CorrelationIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=rp,
                min_confidence=0.008,
                relative_threshold_scale=1.2,
                min_line_strength=1000.0,
                T_range_K=(5000, 15000),
                T_steps=7,
                n_e_range_cm3=(1e15, 5e17),
                **_jax_identifier_flags_for(CorrelationIdentifier),
            )

            # Run all three identifiers.
            alias_result = alias_id.identify(spectrum.wavelength_nm, spectrum.intensity)
            comb_result = comb_id.identify(spectrum.wavelength_nm, spectrum.intensity)
            correlation_result = correlation_id.identify(
                spectrum.wavelength_nm, spectrum.intensity, mode="classic"
            )

            # Combine via 2-of-3 majority vote.
            consensus = HybridConsensusIdentifier(
                identifiers=[alias_id, comb_id, correlation_id],
                elements=candidate_elements,
                min_agreeing=2,
            )
            result = consensus.combine([alias_result, comb_result, correlation_result])
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_voigt_alias_predictor(
    context: "UnifiedBenchmarkContext",
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
    from cflibs.inversion.deconvolution import deconvolve_peaks
    from cflibs.inversion.preprocessing import estimate_baseline

    def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
        from cflibs.atomic.database import AtomicDatabase
        from cflibs.inversion.alias_identifier import ALIASIdentifier

        with AtomicDatabase(str(context.db_path)) as db:
            baseline = estimate_baseline(spectrum.wavelength_nm, spectrum.intensity)
            corrected = np.maximum(spectrum.intensity - baseline, 0.0)
            if find_peaks is None:
                cleaned = corrected
            else:
                threshold = (
                    np.percentile(corrected[corrected > 0], 70) if np.any(corrected > 0) else 0.0
                )
                peak_indices, _ = find_peaks(corrected, height=threshold, distance=5)
                if len(peak_indices) == 0:
                    cleaned = corrected
                else:
                    peak_wls = spectrum.wavelength_nm[peak_indices]
                    rp = _estimate_rp_for_spectrum(spectrum)
                    fwhm_est = float(np.median(spectrum.wavelength_nm) / max(rp, 1.0))
                    try:
                        deconv = deconvolve_peaks(
                            spectrum.wavelength_nm,
                            corrected,
                            peak_wls,
                            fwhm_est,
                            grouping_factor=2.0,
                            margin_factor=3.0,
                            use_jax=False,
                        )
                        cleaned = np.maximum(deconv.fitted_spectrum, 0.0)
                    except Exception:  # noqa: BLE001
                        cleaned = corrected

            rp = _estimate_rp_for_spectrum(spectrum)
            alias = ALIASIdentifier(
                atomic_db=db,
                elements=candidate_elements,
                resolving_power=rp,
                intensity_threshold_factor=3.0,
                detection_threshold=float(config["detection_threshold"]),
                chance_window_scale=0.4,
                max_lines_per_element=30,
                **_jax_identifier_flags_for(ALIASIdentifier),
            )
            result = alias.identify(
                spectrum.wavelength_nm,
                cleaned + np.median(spectrum.intensity) * 0.01,
            )
            result.parameters["candidate_elements"] = list(candidate_elements)
            return result

    return predictor


def _build_nnls_concentration_predictor(
    context: "UnifiedBenchmarkContext",
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable[[BenchmarkSpectrum], "ElementIdentificationResult"]:
    def predictor(spectrum: BenchmarkSpectrum) -> "ElementIdentificationResult":
        from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier

        basis, basis_fwhm, mismatch = context.basis_for_rp(spectrum.rp_estimate)
        identifier = SpectralNNLSIdentifier(
            basis_library=basis,
            detection_snr=0.0,
            continuum_degree=int(config["continuum_degree"]),
            fallback_T_K=8000.0,
            fallback_ne_cm3=1e17,
            **_jax_identifier_flags_for(SpectralNNLSIdentifier),
        )
        result = identifier.identify(spectrum.wavelength_nm, spectrum.intensity)
        threshold = float(config["concentration_threshold"])
        for element in result.all_elements:
            concentration = float(element.metadata.get("concentration_estimate", 0.0))
            element.detected = concentration >= threshold
        result.detected_elements = [element for element in result.all_elements if element.detected]
        result.rejected_elements = [
            element for element in result.all_elements if not element.detected
        ]
        result.algorithm = "nnls_concentration_threshold"
        result.parameters["concentration_threshold"] = threshold
        result.parameters["basis_fwhm_nm"] = basis_fwhm
        result.parameters["basis_fwhm_mismatch_nm"] = mismatch
        result.parameters["candidate_elements"] = list(candidate_elements)
        return result

    return predictor


# ---------------------------------------------------------------------------
# Composition workflow fit-predictor builders
# ---------------------------------------------------------------------------


def _fit_iterative_pipeline(
    _context: "UnifiedBenchmarkContext",
    _train_spectra: Sequence[BenchmarkSpectrum],
    config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional["ElementIdentificationResult"]], Dict[str, Any]
]:
    weighting = _validate_boltzmann_weighting(config.get("weighting"))

    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional["ElementIdentificationResult"],
    ) -> Dict[str, Any]:
        from cflibs.atomic.database import AtomicDatabase

        elements = list(candidate_elements)
        if not elements:
            raise ValueError("No candidate elements available for iterative composition workflow")
        with AtomicDatabase(str(_context.db_path)) as db:
            # Extract solver-specific parameters from config (e.g., two_region)
            solver_kwargs = {
                k: v for k, v in config.items() if k not in {"fit_method", "closure_mode"}
            }
            solver_kwargs["weighting"] = weighting
            result = _run_boltzmann_pipeline_lazy(
                {
                    "wavelength": spectrum.wavelength_nm,
                    "intensity": spectrum.intensity,
                    "ground_truth": spectrum.true_composition,
                },
                db=db,
                fit_method=config["fit_method"],
                closure_mode=str(config["closure_mode"]),
                elements=elements,
                **solver_kwargs,
            )
        if result is None:
            raise RuntimeError("Iterative composition workflow failed")

        closure_mode = str(config["closure_mode"])
        prediction = {"concentrations": result}
        if closure_mode == "dirichlet_residual":
            total_conc = sum(result.values())
            prediction["gamma_residual"] = float(max(0.0, 1.0 - total_conc))

        return prediction

    return predictor


def _fit_iterative_jax_pipeline(
    _context: "UnifiedBenchmarkContext",
    _train_spectra: Sequence[BenchmarkSpectrum],
    config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional["ElementIdentificationResult"]], Dict[str, Any]
]:
    """JAX-accelerated iterative CF-LIBS pipeline.

    Mirrors :func:`_fit_iterative_pipeline` but routes the inner-loop linear
    algebra through ``IterativeCFLIBSSolverJax`` so the heavy Boltzmann +
    closure passes execute on GPU when ``JAX_PLATFORMS=cuda`` is set. When
    the JAX solver is unavailable (Agent B's solver hasn't landed, or JAX
    isn't installed), the workflow logs a warning and falls back to the
    numpy ``IterativeCFLIBSSolver`` path so the benchmark gate keeps running
    end-to-end on CPU-only hosts.
    """
    use_jax = bool(HAS_JAX and HAS_JAX_ITERATIVE_SOLVER)
    if use_jax:
        import jax

        # Ensure 64-bit precision for Boltzmann fits; matches PR #269
        jax.config.update("jax_enable_x64", True)
        if os.environ.get("JAX_PLATFORMS") == "cuda":
            jax.config.update("jax_platform_name", "gpu")

    if not use_jax:
        logger.warning(
            "iterative_jax: falling back to numpy iterative pipeline "
            "(HAS_JAX=%s, HAS_JAX_ITERATIVE_SOLVER=%s)",
            HAS_JAX,
            HAS_JAX_ITERATIVE_SOLVER,
        )

    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional["ElementIdentificationResult"],
    ) -> Dict[str, Any]:
        elements = list(candidate_elements)
        if not elements:
            raise ValueError(
                "No candidate elements available for iterative_jax composition workflow"
            )

        if use_jax:
            from cflibs.atomic.database import AtomicDatabase
            from cflibs.inversion.identify.line_detection import detect_line_observations

            with AtomicDatabase(str(_context.db_path)) as db:
                detection = detect_line_observations(
                    spectrum.wavelength_nm,
                    spectrum.intensity,
                    db,
                    elements=elements,
                )
                observations = detection.observations if detection is not None else []
                if not observations:
                    raise RuntimeError("iterative_jax: no matched line observations for spectrum")
                # IterativeCFLIBSSolverJax shares the same call surface as the
                # numpy IterativeCFLIBSSolver — we instantiate, then solve.
                solver = IterativeCFLIBSSolverJax(atomic_db=db)
                result = solver.solve(
                    observations,
                    closure_mode=str(config.get("closure_mode", "standard")),
                )
            if result is None or not getattr(result, "concentrations", None):
                raise RuntimeError("iterative_jax composition workflow failed")
            payload: Dict[str, Any] = {"concentrations": dict(result.concentrations)}
            t_K = getattr(result, "temperature_K", None)
            if t_K is not None:
                payload["temperature_K"] = float(t_K)
            ne = getattr(result, "electron_density_cm3", None)
            if ne is not None:
                payload["electron_density_cm3"] = float(ne)
            payload["solver_backend"] = "jax"
            return payload

        # --- numpy fallback (matches _fit_iterative_pipeline) ---
        from cflibs.atomic.database import AtomicDatabase

        with AtomicDatabase(str(_context.db_path)) as db:
            result = _run_boltzmann_pipeline_lazy(
                {
                    "wavelength": spectrum.wavelength_nm,
                    "intensity": spectrum.intensity,
                    "ground_truth": spectrum.true_composition,
                },
                db=db,
                fit_method=config["fit_method"],
                closure_mode=str(config["closure_mode"]),
                elements=elements,
            )
        if result is None:
            raise RuntimeError("iterative_jax (numpy fallback) failed")
        return {"concentrations": result, "solver_backend": "numpy_fallback"}

    return predictor


def _fit_bayesian_pipeline(
    _context: "UnifiedBenchmarkContext",
    _train_spectra: Sequence[BenchmarkSpectrum],
    config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional["ElementIdentificationResult"]], Dict[str, Any]
]:
    """Bayesian CF-LIBS composition workflow (NumPyro NUTS, JAX backend).

    Builds a :class:`cflibs.inversion.solve.bayesian.BayesianForwardModel`
    over the spectrum's wavelength range, runs short MCMC (NUTS) on the
    observed intensity, and returns a prediction dict with point-estimate
    concentrations (posterior mean) plus the raw posterior samples so
    ``_maybe_compute_posterior_diagnostics`` can compute rhat / ess_bulk /
    divergent_transitions / psis_loo_* and stash them in
    ``CompositionEvaluationRecord.annotations``.

    The MCMC budget is intentionally small (default 200 warmup / 400
    samples / 1 chain) so per-spectrum wall time stays in the 10-30s range
    on a V100 — the benchmark gate runs across many spectra and outer
    folds.  Tighten via ``--quick``-mode parameter grid if needed.

    When JAX or NumPyro isn't installed, the workflow raises a clear error
    at predictor build time so the benchmark report shows an explicit
    failure for that workflow rather than silently regressing.
    """
    if HAS_JAX:
        import jax

        # Ensure 64-bit precision for NumPyro stability; matches PR #269
        jax.config.update("jax_enable_x64", True)
        if os.environ.get("JAX_PLATFORMS") == "cuda":
            jax.config.update("jax_platform_name", "gpu")

    if HAS_NUMPYRO:
        import numpyro

        try:
            # Explicitly pin to GPU if requested; avoids CPU-bound NUTS on heavy tier
            if os.environ.get("JAX_PLATFORMS") == "cuda":
                numpyro.set_platform("gpu")
            # NOTE: ``numpyro.set_host_device_count`` previously lived here but
            # took effect only when called before JAX initialization, which is
            # never true by this point in the runner. The downstream
            # ``MCMCSampler`` now defaults to ``chain_method='vectorized'``,
            # which batches every chain into a single JIT'd kernel on the
            # current device -- no multi-device requirement.
        except Exception as e:
            logger.debug("bayesian: failed to set NumPyro platform: %s", e)

    if not HAS_JAX or not HAS_NUMPYRO:
        missing = []
        if not HAS_JAX:
            missing.append("jax")
        if not HAS_NUMPYRO:
            missing.append("numpyro")
        logger.warning(
            "bayesian composition workflow: missing dependencies %s — "
            "predictor will raise on first invocation",
            ", ".join(missing),
        )

    num_warmup = int(config.get("num_warmup", 200))
    num_samples = int(config.get("num_samples", 400))
    num_chains = int(config.get("num_chains", 1))
    seed = int(config.get("seed", 0))
    target_accept_prob = float(config.get("target_accept_prob", 0.8))
    pixels = int(config.get("pixels", 1024))

    # Cache samplers to avoid redundant JIT compilation.
    # Keyed by (tuple(elements), pixels, wl_min, wl_max).
    sampler_cache: Dict[Tuple[Any, ...], Any] = {}

    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional["ElementIdentificationResult"],
    ) -> Dict[str, Any]:
        if not HAS_JAX or not HAS_NUMPYRO:
            raise RuntimeError(
                "bayesian composition workflow requires jax + numpyro "
                "(install with: pip install jax jaxlib numpyro)"
            )
        elements = list(candidate_elements)
        if not elements:
            raise ValueError("No candidate elements available for bayesian composition workflow")

        from cflibs.inversion.solve.bayesian import (
            BayesianForwardModel,
            MCMCSampler,
            NoiseParameters,
            PriorConfig,
        )

        wl = np.asarray(spectrum.wavelength_nm, dtype=float)
        intensity = np.asarray(spectrum.intensity, dtype=float)
        if wl.size == 0 or intensity.size == 0:
            raise ValueError("bayesian: empty spectrum input")
        wl_min = float(wl.min())
        wl_max = float(wl.max())

        # Amortize JIT by using a fixed-size grid (pixels) and caching the sampler.
        # JIT triggers whenever the model structure or input shapes change.
        elements_key = tuple(sorted(elements))
        # Round wl boundaries to avoid cache misses on tiny float drift
        cache_key = (elements_key, pixels, round(wl_min, 3), round(wl_max, 3))

        if cache_key in sampler_cache:
            forward_model, sampler = sampler_cache[cache_key]
        else:
            # Use fixed pixel count to keep JIT signatures stable across spectra.
            # We interpolate the observed intensity onto this grid below.
            forward_model = BayesianForwardModel(
                db_path=str(_context.db_path),
                elements=elements,
                wavelength_range=(wl_min, wl_max),
                wavelength_grid=None,  # Force uniform grid of 'pixels' size
                pixels=pixels,
                resolving_power=(
                    float(spectrum.rp_estimate)
                    if spectrum.rp_estimate is not None and spectrum.rp_estimate > 0
                    else None
                ),
            )
            sampler = MCMCSampler(
                forward_model,
                prior_config=PriorConfig(),
                noise_params=NoiseParameters(),
            )
            sampler_cache[cache_key] = (forward_model, sampler)

        # Interp the observed intensity onto the fixed forward-model grid.
        grid = np.asarray(forward_model.wavelength)
        obs = np.interp(grid, wl, intensity)

        # Ensure GPU context if available. JAX_PLATFORMS=cuda should handle this,
        # but explicit pinning prevents sporadic CPU-only fallback observed on vasp-03.
        try:
            import jax

            gpu_device = (
                jax.devices("gpu")[0] if os.environ.get("JAX_PLATFORMS") == "cuda" else None
            )
        except (ImportError, RuntimeError, IndexError):
            gpu_device = None

        if gpu_device:
            with jax.default_device(gpu_device):
                result = sampler.run(
                    obs,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=num_chains,
                    seed=seed,
                    target_accept_prob=target_accept_prob,
                    progress_bar=False,
                )
        else:
            result = sampler.run(
                obs,
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                seed=seed,
                target_accept_prob=target_accept_prob,
                progress_bar=False,
            )

        # Posterior mean concentrations -> point-estimate composition.
        # We compute this from the raw samples to ensure we have the mean
        # even if the result object's summary is unpopulated.
        samples = result.samples
        if "concentrations" in samples:
            conc_samples = np.asarray(samples["concentrations"])
            # Handle both (samples, elements) and (chains, samples, elements)
            mean_concs = np.mean(conc_samples, axis=tuple(range(conc_samples.ndim - 1)))
            concentrations = {element: float(mean_concs[i]) for i, element in enumerate(elements)}
        else:
            concentrations = {
                element: float(result.concentrations_mean.get(element, 0.0)) for element in elements
            }

        # Renormalize so the closure residual stays small even if MCMC
        # didn't perfectly hit the simplex constraint.
        # Ensure all values are positive for Aitchison distance calculation.
        concentrations = {el: max(1e-9, v) for el, v in concentrations.items()}
        total = sum(concentrations.values())
        if total > 0:
            concentrations = {el: v / total for el, v in concentrations.items()}

        # Compute Aitchison distance if truth is available
        aitchison = None
        if spectrum.true_composition:
            aitchison = aitchison_distance(spectrum.true_composition, concentrations)

        # Pull the posterior sample dict off the MCMCResult so the
        # benchmark's _maybe_compute_posterior_diagnostics path lights up.
        posterior_samples: Dict[str, Any] = {}
        try:
            posterior_samples = {k: np.asarray(v) for k, v in result.samples.items()}
        except Exception:  # noqa: BLE001 - never block the gate on diag prep
            posterior_samples = {}

        divergent_count = 0
        try:
            # NumPyro stores divergent transitions in mcmc.get_extra_fields(),
            # but MCMCResult doesn't surface them directly — best-effort fetch.
            inference_data = getattr(result, "inference_data", None)
            if inference_data is not None and hasattr(inference_data, "sample_stats"):
                stats = inference_data.sample_stats
                if "diverging" in stats:
                    divergent_count = int(np.asarray(stats["diverging"].values).sum())
        except Exception:  # noqa: BLE001
            divergent_count = 0

        payload: Dict[str, Any] = {
            "concentrations": concentrations,
            "predicted_composition": concentrations,
            "aitchison": aitchison,
            "posterior_samples": posterior_samples,
            "divergent_count": divergent_count,
            "temperature_K": float(result.T_K_mean) if result.T_K_mean else None,
            "electron_density_cm3": (
                float(result.n_e_mean) if getattr(result, "n_e_mean", None) else None
            ),
            "convergence_status": (
                result.convergence_status.value
                if hasattr(result.convergence_status, "value")
                else str(result.convergence_status)
            ),
            "n_samples": int(result.n_samples),
            "n_chains": int(result.n_chains),
            "n_warmup": int(result.n_warmup),
            "solver_backend": "numpyro_jax",
        }
        return payload

    return predictor


def _fit_joint_optimizer_pipeline(
    _context: "UnifiedBenchmarkContext",
    _train_spectra: Sequence[BenchmarkSpectrum],
    _config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional["ElementIdentificationResult"]], Dict[str, Any]
]:
    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional["ElementIdentificationResult"],
    ) -> Dict[str, Any]:
        elements = list(candidate_elements)
        if not elements:
            raise ValueError("No candidate elements available for joint optimizer")
        return _pipeline_joint_softmax_lazy(spectrum.wavelength_nm, spectrum.intensity, elements)

    return predictor


def _fit_hybrid_manifold_pipeline(
    _context: "UnifiedBenchmarkContext",
    _train_spectra: Sequence[BenchmarkSpectrum],
    _config: Dict[str, Any],
) -> Callable[
    [BenchmarkSpectrum, Sequence[str], Optional["ElementIdentificationResult"]], Dict[str, Any]
]:
    def predictor(
        spectrum: BenchmarkSpectrum,
        candidate_elements: Sequence[str],
        _id_result: Optional["ElementIdentificationResult"],
    ) -> Dict[str, Any]:
        elements = list(candidate_elements)
        if not elements:
            raise ValueError("No candidate elements available for hybrid manifold workflow")
        return _pipeline_hybrid_manifold_lazy(spectrum.wavelength_nm, spectrum.intensity, elements)

    return predictor
