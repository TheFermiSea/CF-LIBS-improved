"""
Inversion algorithms for CF-LIBS.

This module provides the core inversion algorithms for calibration-free
LIBS analysis, including Boltzmann plotting, closure equations, and
optional Bayesian inference and Monte Carlo uncertainty quantification.
"""

from cflibs.core.logging_config import get_logger

# --- Core (always available) ---
from cflibs.inversion.boltzmann import (
    LineObservation,
    BoltzmannFitResult,
    BoltzmannPlotFitter,
    FitMethod,
)
from cflibs.inversion.closure import (
    ClosureEquation,
    ClosureResult,
    ClosureMode,
    clr_transform,
    ilr_transform,
    ilr_inverse,
)
from cflibs.inversion.solver import IterativeCFLIBSSolver, CFLIBSResult
from cflibs.inversion.closed_form_solver import ClosedFormILRSolver, ClosedFormConfig
from cflibs.inversion.quality import (
    QualityMetrics,
    QualityAssessor,
    compute_reconstruction_chi_squared,
)
from cflibs.inversion.line_selection import (
    LineScore,
    LineSelectionResult,
    LineSelector,
    identify_resonance_lines,
)
from cflibs.inversion.line_detection import LineDetectionResult, detect_line_observations
from cflibs.inversion.element_id import (
    IdentifiedLine,
    ElementIdentification,
    ElementIdentificationResult,
    to_line_observations,
)
from cflibs.inversion.correlation_identifier import CorrelationIdentifier
from cflibs.inversion.alias_identifier import ALIASIdentifier
from cflibs.inversion.comb_identifier import CombIdentifier
from cflibs.inversion.preprocessing import (
    estimate_baseline,
    estimate_baseline_snip,
    estimate_baseline_als,
    estimate_noise,
    detect_peaks,
    detect_peaks_auto,
    robust_normalize,
    BaselineMethod,
)
from cflibs.inversion.deconvolution import (
    VoigtFitResult,
    DeconvolutionResult,
    deconvolve_peaks,
    group_peaks,
)
from cflibs.inversion.self_absorption import (
    AbsorptionCorrectionResult,
    SelfAbsorptionResult,
    SelfAbsorptionCorrector,
    estimate_optical_depth_from_intensity_ratio,
)
from cflibs.inversion.cdsb import (
    CDSBPlotter,
    CDSBResult,
    CDSBLineObservation,
    CDSBConvergenceStatus,
    LineOpticalDepth,
    create_cdsb_observation,
    from_transition,
)
from cflibs.inversion.uncertainty import (
    MonteCarloUQ,
    MonteCarloResult,
    PerturbationType,
    AtomicDataUncertainty,
    run_monte_carlo_uq,
    HAS_JOBLIB,
)
from cflibs.inversion.outliers import (
    OutlierMethod,
    SAMResult,
    SpectralAngleMapper,
    sam_distance,
    detect_outlier_spectra,
    MADResult,
    MADOutlierDetector,
    mad_outliers_1d,
    mad_outliers_spectra,
    mad_clean_channels,
)
from cflibs.inversion.matrix_effects import (
    MatrixType,
    MatrixClassificationResult,
    CorrectionFactor,
    CorrectionFactorDB,
    MatrixCorrectionResult,
    MatrixEffectCorrector,
    InternalStandardResult,
    InternalStandardizer,
    combine_corrections,
)
from cflibs.inversion.pca import (
    PCAResult,
    PCAPipeline,
    fit_pca,
    denoise_spectra,
    explained_variance_curve,
)
from cflibs.inversion.temporal import (
    PlasmaPhase,
    TemporalGateConfig,
    TimeResolvedSpectrum,
    PlasmaEvolutionPoint,
    PlasmaEvolutionProfile,
    GateOptimizationResult,
    TemporalSelfAbsorptionResult,
    TimeResolvedCFLIBSResult,
    PlasmaEvolutionModel,
    GateTimingOptimizer,
    TemporalSelfAbsorptionCorrector,
    TimeResolvedCFLIBSSolver,
    create_default_evolution_model,
    recommend_gate_timing,
)
from cflibs.inversion.streaming import (
    AnalysisMode,
    StreamingConfig,
    SpectrumPacket,
    StreamingResult,
    LatencyStats,
    SpectrumBuffer,
    LatencyMonitor,
    BaseStreamingAnalyzer,
    FastAnalyzer,
    StandardAnalyzer,
    StreamingAnalyzer,
    EdgeOptimizedModel,
    create_streaming_pipeline,
)

logger = get_logger("inversion")


def _log_optional_import_failure(component: str, exc: Exception) -> None:
    logger.debug(f"Optional inversion component '{component}' unavailable: {exc}")


# --- Optional availability flags ---
HAS_HYBRID = False
HAS_JOINT_OPTIMIZER = False
HAS_PCA_JAX = False
HAS_BAYESIAN = False
HAS_NESTED = False
HAS_UNCERTAINTIES = False

# --- Optional: Hybrid inversion (requires JAX) ---
try:
    from cflibs.inversion.hybrid import (
        HybridInverter,  # noqa: F401
        HybridInversionResult,  # noqa: F401
        SpectralFitter,  # noqa: F401
    )

    HAS_HYBRID = True
except Exception as exc:
    _log_optional_import_failure("hybrid", exc)

# --- Optional: Joint optimization (requires JAX) ---
try:
    from cflibs.inversion.joint_optimizer import (
        JointOptimizer,  # noqa: F401
        JointOptimizationResult,  # noqa: F401
        MultiStartJointOptimizer,  # noqa: F401
        LossType as JointLossType,  # noqa: F401
        ConvergenceStatus as JointConvergenceStatus,  # noqa: F401
        create_simple_forward_model,  # noqa: F401
    )

    HAS_JOINT_OPTIMIZER = True
except Exception as exc:
    _log_optional_import_failure("joint_optimizer", exc)

# --- Optional: PCA JAX functions (requires JAX) ---
try:
    from cflibs.inversion.pca import (
        pca_transform_jax,  # noqa: F401
        pca_inverse_transform_jax,  # noqa: F401
        pca_reconstruction_error_jax,  # noqa: F401
    )

    HAS_PCA_JAX = True
except Exception as exc:
    _log_optional_import_failure("pca_jax", exc)

# --- Optional: Bayesian inference (requires JAX + NumPyro) ---
try:
    from cflibs.inversion.bayesian import (
        BayesianForwardModel,  # noqa: F401
        AtomicDataArrays,  # noqa: F401
        NoiseParameters,  # noqa: F401
        PriorConfig,  # noqa: F401
        MCMCResult,  # noqa: F401
        MCMCSampler,  # noqa: F401
        ConvergenceStatus,  # noqa: F401
        log_likelihood,  # noqa: F401
        bayesian_model,  # noqa: F401
        run_mcmc,  # noqa: F401
        create_temperature_prior,  # noqa: F401
        create_density_prior,  # noqa: F401
        create_concentration_prior,  # noqa: F401
        TwoZoneBayesianForwardModel,  # noqa: F401
        TwoZonePriorConfig,  # noqa: F401
        TwoZoneMCMCSampler,  # noqa: F401
        TwoZoneMCMCResult,  # noqa: F401
        two_zone_bayesian_model,  # noqa: F401
    )

    HAS_BAYESIAN = True
except Exception as exc:
    _log_optional_import_failure("bayesian", exc)

# --- Optional: Nested sampling (requires dynesty) ---
try:
    from cflibs.inversion.bayesian import NestedSampler, NestedSamplingResult  # noqa: F401

    HAS_NESTED = True
except Exception as exc:
    _log_optional_import_failure("nested", exc)

# --- Optional: Uncertainty propagation (requires uncertainties package) ---
try:
    from cflibs.inversion.uncertainty import (
        HAS_UNCERTAINTIES,  # noqa: F401
        create_boltzmann_uncertainties,  # noqa: F401
        temperature_from_slope,  # noqa: F401
        saha_factor_with_uncertainty,  # noqa: F401
        propagate_through_closure_standard,  # noqa: F401
        propagate_through_closure_matrix,  # noqa: F401
        extract_values_and_uncertainties,  # noqa: F401
    )
except ImportError:
    pass

# --- Public API ---
__all__ = [
    # Boltzmann plotting
    "LineObservation",
    "BoltzmannFitResult",
    "BoltzmannPlotFitter",
    "FitMethod",
    # Closure
    "ClosureEquation",
    "ClosureResult",
    "ClosureMode",
    "clr_transform",
    "ilr_transform",
    "ilr_inverse",
    # Solver
    "IterativeCFLIBSSolver",
    "CFLIBSResult",
    "ClosedFormILRSolver",
    "ClosedFormConfig",
    # Quality metrics
    "QualityMetrics",
    "QualityAssessor",
    "compute_reconstruction_chi_squared",
    # Line selection
    "LineScore",
    "LineSelectionResult",
    "LineSelector",
    "identify_resonance_lines",
    "LineDetectionResult",
    "detect_line_observations",
    # Element identification
    "IdentifiedLine",
    "ElementIdentification",
    "ElementIdentificationResult",
    "to_line_observations",
    "CorrelationIdentifier",
    "ALIASIdentifier",
    "CombIdentifier",
    # Preprocessing
    "estimate_baseline",
    "estimate_baseline_snip",
    "estimate_baseline_als",
    "estimate_noise",
    "detect_peaks",
    "detect_peaks_auto",
    "robust_normalize",
    "BaselineMethod",
    # Deconvolution
    "VoigtFitResult",
    "DeconvolutionResult",
    "deconvolve_peaks",
    "group_peaks",
    # Self-absorption
    "AbsorptionCorrectionResult",
    "SelfAbsorptionResult",
    "SelfAbsorptionCorrector",
    "estimate_optical_depth_from_intensity_ratio",
    # CD-SB plotting
    "CDSBPlotter",
    "CDSBResult",
    "CDSBLineObservation",
    "CDSBConvergenceStatus",
    "LineOpticalDepth",
    "create_cdsb_observation",
    "from_transition",
    # Monte Carlo UQ
    "MonteCarloUQ",
    "MonteCarloResult",
    "PerturbationType",
    "AtomicDataUncertainty",
    "run_monte_carlo_uq",
    "HAS_JOBLIB",
    # Outlier detection
    "OutlierMethod",
    "SAMResult",
    "SpectralAngleMapper",
    "sam_distance",
    "detect_outlier_spectra",
    "MADResult",
    "MADOutlierDetector",
    "mad_outliers_1d",
    "mad_outliers_spectra",
    "mad_clean_channels",
    # Matrix effects
    "MatrixType",
    "MatrixClassificationResult",
    "CorrectionFactor",
    "CorrectionFactorDB",
    "MatrixCorrectionResult",
    "MatrixEffectCorrector",
    "InternalStandardResult",
    "InternalStandardizer",
    "combine_corrections",
    # PCA
    "PCAResult",
    "PCAPipeline",
    "fit_pca",
    "denoise_spectra",
    "explained_variance_curve",
    # Temporal dynamics
    "PlasmaPhase",
    "TemporalGateConfig",
    "TimeResolvedSpectrum",
    "PlasmaEvolutionPoint",
    "PlasmaEvolutionProfile",
    "GateOptimizationResult",
    "TemporalSelfAbsorptionResult",
    "TimeResolvedCFLIBSResult",
    "PlasmaEvolutionModel",
    "GateTimingOptimizer",
    "TemporalSelfAbsorptionCorrector",
    "TimeResolvedCFLIBSSolver",
    "create_default_evolution_model",
    "recommend_gate_timing",
    # Real-time streaming
    "AnalysisMode",
    "StreamingConfig",
    "SpectrumPacket",
    "StreamingResult",
    "LatencyStats",
    "SpectrumBuffer",
    "LatencyMonitor",
    "BaseStreamingAnalyzer",
    "FastAnalyzer",
    "StandardAnalyzer",
    "StreamingAnalyzer",
    "EdgeOptimizedModel",
    "create_streaming_pipeline",
    # Availability flags
    "HAS_HYBRID",
    "HAS_JOINT_OPTIMIZER",
    "HAS_BAYESIAN",
    "HAS_NESTED",
    "HAS_UNCERTAINTIES",
    "HAS_PCA_JAX",
]

# Extend __all__ with optional exports
if HAS_HYBRID:
    __all__.extend(["HybridInverter", "HybridInversionResult", "SpectralFitter"])

if HAS_JOINT_OPTIMIZER:
    __all__.extend(
        [
            "JointOptimizer",
            "JointOptimizationResult",
            "MultiStartJointOptimizer",
            "JointLossType",
            "JointConvergenceStatus",
            "create_simple_forward_model",
        ]
    )

if HAS_BAYESIAN:
    __all__.extend(
        [
            "BayesianForwardModel",
            "AtomicDataArrays",
            "NoiseParameters",
            "PriorConfig",
            "MCMCResult",
            "MCMCSampler",
            "ConvergenceStatus",
            "log_likelihood",
            "bayesian_model",
            "run_mcmc",
            "create_temperature_prior",
            "create_density_prior",
            "create_concentration_prior",
            "TwoZoneBayesianForwardModel",
            "TwoZonePriorConfig",
            "TwoZoneMCMCSampler",
            "TwoZoneMCMCResult",
            "two_zone_bayesian_model",
        ]
    )

if HAS_NESTED:
    __all__.extend(["NestedSampler", "NestedSamplingResult"])

if HAS_UNCERTAINTIES:
    __all__.extend(
        [
            "create_boltzmann_uncertainties",
            "temperature_from_slope",
            "saha_factor_with_uncertainty",
            "propagate_through_closure_standard",
            "propagate_through_closure_matrix",
            "extract_values_and_uncertainties",
        ]
    )

if HAS_PCA_JAX:
    __all__.extend(
        [
            "pca_transform_jax",
            "pca_inverse_transform_jax",
            "pca_reconstruction_error_jax",
        ]
    )
