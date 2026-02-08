"""
Inversion algorithms for CF-LIBS.

This module provides the core inversion algorithms for calibration-free
LIBS analysis, including Boltzmann plotting, closure equations, and
optional Bayesian inference and Monte Carlo uncertainty quantification.
"""

# --- Core (always available) ---
from cflibs.inversion.boltzmann import (
    LineObservation,
    BoltzmannFitResult,
    BoltzmannPlotFitter,
    FitMethod,
)
from cflibs.inversion.closure import ClosureEquation, ClosureResult
from cflibs.inversion.solver import IterativeCFLIBSSolver, CFLIBSResult
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
from cflibs.inversion.alias_identifier import ALIASIdentifier
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
from cflibs.inversion.pls import (
    PLSAlgorithm,
    PreprocessingMethod,
    PLSComponents,
    PLSResult,
    CrossValidationResult,
    PLSRegression,
    PLSCrossValidator,
    PLSCalibrationModel,
    build_pls_calibration,
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
from cflibs.inversion.interpretable import (
    FeatureType,
    SpectralFeature,
    FeatureExtractionResult,
    SpectralExplanation,
    ValidationResult,
    PhysicsGuidedFeatureExtractor,
    SpectralExplainer,
    ExplanationValidator,
)
from cflibs.inversion.transfer import (
    DomainAdapter,
    DomainAdaptationMethod,
    DomainAdaptationResult,
    compute_mmd,
    adapt_domains,
    CalibrationTransfer,
    CalibrationMethod,
    TransferResult,
    transfer_calibration,
    FineTuner,
    FineTuneResult,
    TransferLearningPipeline,
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

# --- Optional availability flags ---
HAS_HYBRID = False
HAS_JOINT_OPTIMIZER = False
HAS_PCA_JAX = False
HAS_BAYESIAN = False
HAS_NESTED = False
HAS_UNCERTAINTIES = False
HAS_INTERPRETABLE_ML = False
HAS_PINN = False

# --- Optional: Hybrid inversion (requires JAX) ---
try:
    from cflibs.inversion.hybrid import HybridInverter, HybridInversionResult, SpectralFitter  # noqa: F401
    HAS_HYBRID = True
except ImportError:
    pass

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
except ImportError:
    pass

# --- Optional: PCA JAX functions (requires JAX) ---
try:
    from cflibs.inversion.pca import (
        pca_transform_jax,  # noqa: F401
        pca_inverse_transform_jax,  # noqa: F401
        pca_reconstruction_error_jax,  # noqa: F401
    )
    HAS_PCA_JAX = True
except ImportError:
    pass

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
    )
    HAS_BAYESIAN = True
except ImportError:
    pass

# --- Optional: Nested sampling (requires dynesty) ---
try:
    from cflibs.inversion.bayesian import NestedSampler, NestedSamplingResult  # noqa: F401
    HAS_NESTED = True
except ImportError:
    pass

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

# --- Optional: Interpretable ML (requires sklearn) ---
try:
    from cflibs.inversion.interpretable import InterpretableModel  # noqa: F401

    HAS_INTERPRETABLE_ML = True
except ImportError:
    # Interpretable ML components are optional
    pass

# --- Optional: Physics-Informed Neural Networks (requires JAX + Equinox + Optax) ---
try:
    from cflibs.inversion.pinn import (
        ConstraintType,  # noqa: F401
        PhysicsConstraintConfig,  # noqa: F401
        PINNConfig,  # noqa: F401
        PINNResult,  # noqa: F401
        DifferentiableForwardModel,  # noqa: F401
        PINNInverter,  # noqa: F401
        boltzmann_residual,  # noqa: F401
        saha_residual,  # noqa: F401
        closure_residual,  # noqa: F401
        positivity_penalty,  # noqa: F401
        range_penalty,  # noqa: F401
        create_pinn_from_database,  # noqa: F401
        generate_synthetic_training_data,  # noqa: F401
    )
    HAS_PINN = True
except ImportError:
    # PINN components are optional

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
    # Solver
    "IterativeCFLIBSSolver",
    "CFLIBSResult",
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
    "ALIASIdentifier",
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
    # PLS
    "PLSAlgorithm",
    "PreprocessingMethod",
    "PLSComponents",
    "PLSResult",
    "CrossValidationResult",
    "PLSRegression",
    "PLSCrossValidator",
    "PLSCalibrationModel",
    "build_pls_calibration",
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
    # Transfer learning
    "DomainAdapter",
    "DomainAdaptationMethod",
    "DomainAdaptationResult",
    "compute_mmd",
    "adapt_domains",
    "CalibrationTransfer",
    "CalibrationMethod",
    "TransferResult",
    "transfer_calibration",
    "FineTuner",
    "FineTuneResult",
    "TransferLearningPipeline",
    # Interpretable ML
    "FeatureType",
    "SpectralFeature",
    "FeatureExtractionResult",
    "SpectralExplanation",
    "ValidationResult",
    "PhysicsGuidedFeatureExtractor",
    "SpectralExplainer",
    "ExplanationValidator",
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
    "HAS_PINN",
]

# Extend __all__ with optional exports
if HAS_HYBRID:
    __all__.extend(["HybridInverter", "HybridInversionResult", "SpectralFitter"])

if HAS_JOINT_OPTIMIZER:
    __all__.extend([
        "JointOptimizer", "JointOptimizationResult", "MultiStartJointOptimizer",
        "JointLossType", "JointConvergenceStatus", "create_simple_forward_model",
    ])

if HAS_BAYESIAN:
    __all__.extend([
        "BayesianForwardModel", "AtomicDataArrays", "NoiseParameters", "PriorConfig",
        "MCMCResult", "MCMCSampler", "ConvergenceStatus", "log_likelihood",
        "bayesian_model", "run_mcmc", "create_temperature_prior",
        "create_density_prior", "create_concentration_prior",
    ])

if HAS_NESTED:
    __all__.extend(["NestedSampler", "NestedSamplingResult"])

if HAS_UNCERTAINTIES:
    __all__.extend([
        "create_boltzmann_uncertainties", "temperature_from_slope",
        "saha_factor_with_uncertainty", "propagate_through_closure_standard",
        "propagate_through_closure_matrix", "extract_values_and_uncertainties",
    ])

if HAS_PCA_JAX:
    __all__.extend([
        "pca_transform_jax", "pca_inverse_transform_jax", "pca_reconstruction_error_jax",
    ])

if HAS_INTERPRETABLE_ML:
    __all__.extend(["InterpretableModel", "HAS_INTERPRETABLE_ML"])

if HAS_PINN:
    __all__.extend([
        "ConstraintType",
        "PhysicsConstraintConfig",
        "PINNConfig",
        "PINNResult",
        "DifferentiableForwardModel",
        "PINNInverter",
        "boltzmann_residual",
        "saha_residual",
        "closure_residual",
        "positivity_penalty",
        "range_penalty",
        "create_pinn_from_database",
        "generate_synthetic_training_data",
        "HAS_PINN",
    ])
