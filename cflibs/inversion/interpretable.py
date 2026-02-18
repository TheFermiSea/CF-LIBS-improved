"""
Interpretable ML for LIBS Spectral Analysis.

This module provides tools for building transparent, explainable machine learning
models for LIBS spectral analysis. It addresses the key challenge that black-box ML
predictions limit adoption in regulated industries (medical, nuclear, quality control).

Key Components
--------------
1. **Physics-Guided Feature Extraction**:
   - Extract spectroscopically meaningful features from raw spectra
   - Peak detection with physical attribution (element, ionization state)
   - Spectral region aggregation based on known emission wavelengths

2. **Model Explanation Tools**:
   - SHAP (SHapley Additive exPlanations) wrapper for spectral models
   - LIME (Local Interpretable Model-agnostic Explanations) wrapper
   - Feature importance attribution to specific wavelength regions

3. **Attention Visualization**:
   - Identify influential wavelength regions for predictions
   - Spectral saliency maps showing which features drive predictions
   - Support for both traditional ML and neural network models

4. **Explanation Validation**:
   - Compare ML explanations against spectroscopic knowledge
   - Validate that important features correspond to known emission lines
   - Detect when models may be using spurious correlations

Research Questions Addressed
----------------------------
- What features do successful ML models actually use?
- Can physics constraints improve ML interpretability?
- How to validate ML explanations against spectroscopic knowledge?

Example Usage
-------------
>>> from cflibs.inversion.interpretable import (
...     PhysicsGuidedFeatureExtractor,
...     SpectralExplainer,
...     ExplanationValidator,
... )
>>>
>>> # Extract physics-guided features
>>> extractor = PhysicsGuidedFeatureExtractor(db_path, elements=["Fe", "Cu"])
>>> features = extractor.extract(wavelengths, spectrum)
>>>
>>> # Explain model predictions
>>> explainer = SpectralExplainer(model, wavelengths)
>>> explanation = explainer.explain_shap(spectrum)
>>>
>>> # Validate explanations against spectroscopic knowledge
>>> validator = ExplanationValidator(db_path, elements=["Fe", "Cu"])
>>> validation = validator.validate(explanation, wavelengths)

References
----------
- Lundberg & Lee (2017): SHAP - A Unified Approach to Interpreting Model Predictions
- Ribeiro et al. (2016): LIME - Why Should I Trust You?
- Clegg et al. (2017): Recalibration of the Mars Science Laboratory ChemCam LIBS
- Tognoni et al. (2010): Quantitative LIBS - State of the Art
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.interpretable")

# Optional dependency flags
HAS_SHAP = False
HAS_SKLEARN = False
HAS_SCIPY = False

try:
    import shap

    HAS_SHAP = True
except ImportError:
    shap = None

try:
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

    HAS_SKLEARN = True
except ImportError:
    Ridge = None
    Lasso = None
    RandomForestRegressor = None
    GradientBoostingRegressor = None

try:
    from scipy.signal import find_peaks, savgol_filter
    from scipy.ndimage import gaussian_filter1d

    HAS_SCIPY = True
except ImportError:
    find_peaks = None
    savgol_filter = None
    gaussian_filter1d = None


class FeatureType(Enum):
    """Types of physics-guided features."""

    PEAK_INTENSITY = "peak_intensity"
    PEAK_AREA = "peak_area"
    PEAK_WIDTH = "peak_width"
    REGION_SUM = "region_sum"
    REGION_MEAN = "region_mean"
    RATIO = "ratio"
    BASELINE = "baseline"


@dataclass
class SpectralFeature:
    """
    A physics-guided spectral feature.

    Attributes
    ----------
    name : str
        Feature name (e.g., "Fe_I_371.99nm_intensity")
    feature_type : FeatureType
        Type of feature (peak intensity, area, ratio, etc.)
    wavelength_nm : float
        Central wavelength of feature [nm]
    element : str
        Associated element symbol
    ionization_stage : int
        Ionization stage (0=neutral, 1=singly ionized)
    value : float
        Computed feature value
    uncertainty : float
        Feature uncertainty estimate
    wavelength_range : Tuple[float, float]
        Wavelength range used for feature extraction [nm]
    """

    name: str
    feature_type: FeatureType
    wavelength_nm: float
    element: str
    ionization_stage: int
    value: float
    uncertainty: float = 0.0
    wavelength_range: Tuple[float, float] = (0.0, 0.0)


@dataclass
class FeatureExtractionResult:
    """
    Result of physics-guided feature extraction.

    Attributes
    ----------
    features : List[SpectralFeature]
        Extracted features with physical attribution
    feature_matrix : np.ndarray
        Feature matrix (n_samples, n_features) for ML models
    feature_names : List[str]
        Names of features in the matrix
    wavelengths_used : List[float]
        Wavelengths where features were extracted
    elements_detected : List[str]
        Elements with detected features
    """

    features: List[SpectralFeature]
    feature_matrix: np.ndarray
    feature_names: List[str]
    wavelengths_used: List[float]
    elements_detected: List[str] = field(default_factory=list)


@dataclass
class SpectralExplanation:
    """
    Explanation of a spectral model prediction.

    Attributes
    ----------
    method : str
        Explanation method used ("shap", "lime", "gradient", "attention")
    feature_importances : Dict[str, float]
        Importance scores by feature name
    wavelength_importances : np.ndarray
        Importance by wavelength index
    top_features : List[Tuple[str, float]]
        Top features ranked by importance
    base_value : float
        Expected model output (SHAP base value)
    prediction : float
        Model prediction for this sample
    """

    method: str
    feature_importances: Dict[str, float]
    wavelength_importances: np.ndarray
    top_features: List[Tuple[str, float]]
    base_value: float = 0.0
    prediction: float = 0.0


@dataclass
class ValidationResult:
    """
    Result of explanation validation against spectroscopic knowledge.

    Attributes
    ----------
    is_valid : bool
        Whether explanation aligns with spectroscopic knowledge
    score : float
        Validation score (0-1, higher = better alignment)
    matched_lines : List[Tuple[str, float, float]]
        Lines that match important features: (element, wavelength, importance)
    unmatched_features : List[Tuple[str, float]]
        Important features not matching known lines
    spurious_correlations : List[str]
        Potential spurious correlations detected
    recommendations : List[str]
        Recommendations for improving model interpretability
    """

    is_valid: bool
    score: float
    matched_lines: List[Tuple[str, float, float]]
    unmatched_features: List[Tuple[str, float]]
    spurious_correlations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def _load_lines_from_db(
    db_path: str,
    elements: List[str],
    columns: List[str],
    min_rel_int: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """
    Load known emission lines from atomic database.

    Parameters
    ----------
    db_path : str
        Path to SQLite database
    elements : List[str]
        Elements to load
    columns : List[str]
        Columns to select
    min_rel_int : float, optional
        Minimum relative intensity filter

    Returns
    -------
    List[Dict[str, Any]]
        List of line data dictionaries
    """
    import sqlite3

    # Validate column names to prevent SQL injection
    for col in columns:
        if not col.isidentifier():
            raise ValueError(f"Invalid column name: {col}")

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        placeholders = ",".join(["?"] * len(elements))
        col_str = ", ".join(columns)
        query = f"SELECT {col_str} FROM lines WHERE element IN ({placeholders})"

        params = list(elements)
        if min_rel_int is not None:
            query += " AND rel_int > ?"
            params.append(min_rel_int)

        query += " ORDER BY rel_int DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()

        lines = []
        for row in rows:
            line = {}
            for i, col in enumerate(columns):
                val = row[i]
                if col == "rel_int":
                    val = val if val is not None else 0.0
                line[col] = val
            lines.append(line)

    return lines


class PhysicsGuidedFeatureExtractor:
    """
    Extract physics-guided features from LIBS spectra.

    This class extracts features that are meaningful from a spectroscopic
    perspective, rather than generic signal processing features. Features
    are attributed to specific elements and ionization states based on
    known emission wavelengths.

    Parameters
    ----------
    db_path : str
        Path to atomic database
    elements : List[str]
        Elements to extract features for
    wavelength_tolerance_nm : float
        Tolerance for matching peaks to known lines [nm]
    min_peak_height : float
        Minimum peak height for detection (relative to max)
    peak_width_nm : float
        Expected peak width for integration [nm]

    Example
    -------
    >>> extractor = PhysicsGuidedFeatureExtractor(
    ...     db_path="atomic.db",
    ...     elements=["Fe", "Cu", "Ni"],
    ... )
    >>> features = extractor.extract(wavelengths, spectrum)
    >>> print(f"Extracted {len(features.features)} features")
    """

    def __init__(
        self,
        db_path: str,
        elements: List[str],
        wavelength_tolerance_nm: float = 0.1,
        min_peak_height: float = 0.01,
        peak_width_nm: float = 0.2,
    ):
        self.db_path = db_path
        self.elements = elements
        self.wavelength_tolerance_nm = wavelength_tolerance_nm
        self.min_peak_height = min_peak_height
        self.peak_width_nm = peak_width_nm

        # Load known emission lines from database
        self.known_lines = self._load_known_lines()

        logger.info(
            f"PhysicsGuidedFeatureExtractor: {len(elements)} elements, "
            f"{len(self.known_lines)} known lines"
        )

    def _load_known_lines(self) -> List[Dict[str, Any]]:
        """Load known emission lines from atomic database."""
        columns = ["element", "sp_num", "wavelength_nm", "aki", "ek_ev", "gk", "rel_int"]
        return _load_lines_from_db(self.db_path, self.elements, columns)

    def extract(
        self,
        wavelengths: np.ndarray,
        spectrum: np.ndarray,
        uncertainty: Optional[np.ndarray] = None,
    ) -> FeatureExtractionResult:
        """
        Extract physics-guided features from a spectrum.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength axis [nm]
        spectrum : np.ndarray
            Spectral intensity
        uncertainty : np.ndarray, optional
            Uncertainty in spectral intensity

        Returns
        -------
        FeatureExtractionResult
            Extracted features with physical attribution
        """
        if uncertainty is None:
            # Estimate uncertainty as sqrt(counts) for Poisson noise
            uncertainty = np.sqrt(np.maximum(spectrum, 1.0))

        features = []
        wavelengths_used = []
        elements_detected = set()

        # Find peaks in spectrum
        peaks = self._find_peaks(wavelengths, spectrum)

        # Match peaks to known emission lines
        for peak_idx, peak_wl in peaks:
            matched_line = self._match_to_known_line(peak_wl)

            if matched_line is not None:
                # Extract features for this peak
                peak_features = self._extract_peak_features(
                    wavelengths,
                    spectrum,
                    uncertainty,
                    peak_idx,
                    matched_line,
                )
                features.extend(peak_features)
                wavelengths_used.append(peak_wl)
                elements_detected.add(matched_line["element"])

        # Add ratio features between strong lines of different elements
        ratio_features = self._extract_ratio_features(features)
        features.extend(ratio_features)

        # Build feature matrix
        feature_matrix, feature_names = self._build_feature_matrix(features)

        return FeatureExtractionResult(
            features=features,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            wavelengths_used=wavelengths_used,
            elements_detected=list(elements_detected),
        )

    def _find_peaks(
        self,
        wavelengths: np.ndarray,
        spectrum: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """Find peaks in spectrum."""
        if not HAS_SCIPY:
            # Fallback: simple local maxima detection
            return self._find_peaks_simple(wavelengths, spectrum)

        # Normalize spectrum
        max_intensity = np.max(spectrum)
        if max_intensity <= 0:
            return []

        normalized = spectrum / max_intensity

        # Find peaks with scipy
        peak_indices, properties = find_peaks(
            normalized,
            height=self.min_peak_height,
            distance=5,  # Minimum distance between peaks (pixels)
            prominence=self.min_peak_height / 2,
        )

        peaks = []
        for idx in peak_indices:
            peaks.append((int(idx), float(wavelengths[idx])))

        return peaks

    def _find_peaks_simple(
        self,
        wavelengths: np.ndarray,
        spectrum: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """Simple peak detection without scipy."""
        max_intensity = np.max(spectrum)
        if max_intensity <= 0:
            return []

        threshold = self.min_peak_height * max_intensity
        peaks = []

        for i in range(1, len(spectrum) - 1):
            if spectrum[i] > threshold:
                if spectrum[i] > spectrum[i - 1] and spectrum[i] > spectrum[i + 1]:
                    peaks.append((i, float(wavelengths[i])))

        return peaks

    def _match_to_known_line(
        self,
        peak_wavelength: float,
    ) -> Optional[Dict[str, Any]]:
        """Match a peak to a known emission line."""
        best_match = None
        best_distance = float("inf")

        for line in self.known_lines:
            distance = abs(line["wavelength_nm"] - peak_wavelength)

            if distance < self.wavelength_tolerance_nm and distance < best_distance:
                best_match = line
                best_distance = distance

        return best_match

    def _extract_peak_features(
        self,
        wavelengths: np.ndarray,
        spectrum: np.ndarray,
        uncertainty: np.ndarray,
        peak_idx: int,
        matched_line: Dict[str, Any],
    ) -> List[SpectralFeature]:
        """Extract features for a matched peak."""
        features = []
        element = matched_line["element"]
        ion_stage = matched_line["sp_num"] - 1  # Convert to 0-indexed
        wl = matched_line["wavelength_nm"]

        # Feature name prefix
        ion_roman = "I" if ion_stage == 0 else "II" if ion_stage == 1 else f"{ion_stage + 1}"
        prefix = f"{element}_{ion_roman}_{wl:.2f}nm"

        # Peak intensity feature
        peak_intensity = float(spectrum[peak_idx])
        peak_unc = float(uncertainty[peak_idx])

        features.append(
            SpectralFeature(
                name=f"{prefix}_intensity",
                feature_type=FeatureType.PEAK_INTENSITY,
                wavelength_nm=wl,
                element=element,
                ionization_stage=ion_stage,
                value=peak_intensity,
                uncertainty=peak_unc,
                wavelength_range=(wl - self.peak_width_nm / 2, wl + self.peak_width_nm / 2),
            )
        )

        # Peak area feature (integrate over peak region)
        wl_step = wavelengths[1] - wavelengths[0] if len(wavelengths) > 1 else 0.1
        half_width_pixels = int(self.peak_width_nm / wl_step / 2)

        start_idx = max(0, peak_idx - half_width_pixels)
        end_idx = min(len(spectrum), peak_idx + half_width_pixels + 1)

        if hasattr(np, "trapezoid"):
            peak_area = float(
                np.trapezoid(spectrum[start_idx:end_idx], wavelengths[start_idx:end_idx])
            )
        else:
            peak_area = float(np.trapz(spectrum[start_idx:end_idx], wavelengths[start_idx:end_idx]))
        area_unc = float(np.sqrt(np.sum(uncertainty[start_idx:end_idx] ** 2)) * wl_step)

        features.append(
            SpectralFeature(
                name=f"{prefix}_area",
                feature_type=FeatureType.PEAK_AREA,
                wavelength_nm=wl,
                element=element,
                ionization_stage=ion_stage,
                value=peak_area,
                uncertainty=area_unc,
                wavelength_range=(float(wavelengths[start_idx]), float(wavelengths[end_idx - 1])),
            )
        )

        return features

    def _extract_ratio_features(
        self,
        features: List[SpectralFeature],
    ) -> List[SpectralFeature]:
        """Extract ratio features between strong lines of different elements."""
        ratio_features = []

        # Group intensity features by element
        intensity_by_element: Dict[str, List[SpectralFeature]] = {}
        for f in features:
            if f.feature_type == FeatureType.PEAK_INTENSITY:
                if f.element not in intensity_by_element:
                    intensity_by_element[f.element] = []
                intensity_by_element[f.element].append(f)

        # Sort by intensity within each element
        for el in intensity_by_element:
            intensity_by_element[el].sort(key=lambda x: x.value, reverse=True)

        # Create ratios between strongest lines of different elements
        elements = list(intensity_by_element.keys())
        for i, el1 in enumerate(elements):
            for el2 in elements[i + 1 :]:
                if not intensity_by_element[el1] or not intensity_by_element[el2]:
                    continue

                f1 = intensity_by_element[el1][0]  # Strongest line of el1
                f2 = intensity_by_element[el2][0]  # Strongest line of el2

                if f2.value > 0:
                    ratio_value = f1.value / f2.value
                    # Propagate uncertainty
                    ratio_unc = ratio_value * np.sqrt(
                        (f1.uncertainty / f1.value) ** 2 + (f2.uncertainty / f2.value) ** 2
                    )

                    ratio_features.append(
                        SpectralFeature(
                            name=f"ratio_{el1}_{el2}",
                            feature_type=FeatureType.RATIO,
                            wavelength_nm=(f1.wavelength_nm + f2.wavelength_nm) / 2,
                            element=f"{el1}/{el2}",
                            ionization_stage=0,
                            value=ratio_value,
                            uncertainty=ratio_unc,
                            wavelength_range=(
                                min(f1.wavelength_nm, f2.wavelength_nm),
                                max(f1.wavelength_nm, f2.wavelength_nm),
                            ),
                        )
                    )

        return ratio_features

    def _build_feature_matrix(
        self,
        features: List[SpectralFeature],
    ) -> Tuple[np.ndarray, List[str]]:
        """Build feature matrix from extracted features."""
        feature_names = [f.name for f in features]
        feature_values = np.array([f.value for f in features])

        # Reshape to (1, n_features) for single sample
        feature_matrix = feature_values.reshape(1, -1)

        return feature_matrix, feature_names

    def extract_batch(
        self,
        wavelengths: np.ndarray,
        spectra: np.ndarray,
        uncertainties: Optional[np.ndarray] = None,
    ) -> FeatureExtractionResult:
        """
        Extract features from multiple spectra.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength axis [nm]
        spectra : np.ndarray
            Spectral intensities (n_samples, n_wavelengths)
        uncertainties : np.ndarray, optional
            Uncertainties (n_samples, n_wavelengths)

        Returns
        -------
        FeatureExtractionResult
            Features with matrix of shape (n_samples, n_features)
        """
        n_samples = spectra.shape[0]
        all_features: List[SpectralFeature] = []
        feature_matrices = []

        for i in range(n_samples):
            unc = uncertainties[i] if uncertainties is not None else None
            result = self.extract(wavelengths, spectra[i], unc)

            if i == 0:
                all_features = result.features
                feature_names = result.feature_names

            feature_matrices.append(result.feature_matrix)

        # Stack into single matrix
        feature_matrix = np.vstack(feature_matrices)

        return FeatureExtractionResult(
            features=all_features,
            feature_matrix=feature_matrix,
            feature_names=feature_names,
            wavelengths_used=result.wavelengths_used,
            elements_detected=result.elements_detected,
        )


class SpectralExplainer:
    """
    Explain ML model predictions on LIBS spectra.

    Provides SHAP and LIME-style explanations for spectral models,
    attributing predictions to specific wavelength regions.

    Parameters
    ----------
    model : Callable
        ML model with predict() method
    wavelengths : np.ndarray
        Wavelength axis [nm]
    feature_names : List[str], optional
        Names of input features (default: wavelength values)

    Example
    -------
    >>> explainer = SpectralExplainer(model, wavelengths)
    >>> explanation = explainer.explain_shap(spectrum, n_samples=100)
    >>> print(explanation.top_features[:5])
    """

    def __init__(
        self,
        model: Callable,
        wavelengths: np.ndarray,
        feature_names: Optional[List[str]] = None,
    ):
        self.model = model
        self.wavelengths = wavelengths
        self.feature_names = feature_names or [f"{w:.2f}nm" for w in wavelengths]

        logger.info(f"SpectralExplainer initialized with {len(wavelengths)} wavelengths")

    def _call_model(self, X: np.ndarray) -> np.ndarray:
        """Helper to call model's predict method or the model itself."""
        if hasattr(self.model, "predict"):
            return self.model.predict(X)
        return self.model(X)

    def explain_shap(
        self,
        spectrum: np.ndarray,
        background: Optional[np.ndarray] = None,
        n_samples: int = 100,
    ) -> SpectralExplanation:
        """
        Generate SHAP explanation for a spectrum.

        SHAP values indicate how each feature contributes to the model's
        prediction relative to a baseline (expected value).

        Parameters
        ----------
        spectrum : np.ndarray
            Input spectrum to explain
        background : np.ndarray, optional
            Background samples for SHAP (default: zeros)
        n_samples : int
            Number of samples for SHAP estimation

        Returns
        -------
        SpectralExplanation
            SHAP-based explanation
        """
        if not HAS_SHAP:
            logger.warning("SHAP not available, using permutation importance fallback")
            return self._explain_permutation(spectrum, n_samples)

        # Ensure 2D input
        if spectrum.ndim == 1:
            spectrum = spectrum.reshape(1, -1)

        # Create background if not provided
        if background is None:
            background = np.zeros((n_samples, len(self.wavelengths)))

        # Create SHAP explainer
        try:
            explainer = shap.KernelExplainer(self._call_model, background)

            # Calculate SHAP values
            shap_values = explainer.shap_values(spectrum, nsamples=n_samples)

            if isinstance(shap_values, list):
                # Multi-output: take first output
                shap_values = shap_values[0]

            # Flatten if needed
            shap_values = np.array(shap_values).flatten()

        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}, using fallback")
            return self._explain_permutation(spectrum.flatten(), n_samples)

        # Build explanation
        feature_importances = {
            name: float(val) for name, val in zip(self.feature_names, shap_values)
        }

        # Get prediction and base value
        prediction = float(self._call_model(spectrum)[0])
        base_value = (
            float(explainer.expected_value) if hasattr(explainer, "expected_value") else 0.0
        )

        # Sort by absolute importance
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return SpectralExplanation(
            method="shap",
            feature_importances=feature_importances,
            wavelength_importances=shap_values,
            top_features=sorted_features[:20],
            base_value=base_value,
            prediction=prediction,
        )

    def _explain_permutation(
        self,
        spectrum: np.ndarray,
        n_samples: int = 100,
    ) -> SpectralExplanation:
        """
        Fallback permutation importance explanation.

        Measures importance by permuting each feature and measuring
        change in prediction.
        """
        if spectrum.ndim == 1:
            spectrum_2d = spectrum.reshape(1, -1)
        else:
            spectrum_2d = spectrum

        # Get baseline prediction
        prediction = float(self._call_model(spectrum_2d)[0])

        # Compute permutation importances
        importances = np.zeros(len(self.wavelengths))
        rng = np.random.default_rng(42)

        for i in range(len(self.wavelengths)):
            # Create permuted samples
            changes = []
            for _ in range(min(n_samples, 10)):
                permuted = spectrum_2d.copy()
                permuted[0, i] = rng.normal(permuted[0, i], np.std(spectrum_2d[0]) * 0.1)

                perm_pred = float(self._call_model(permuted)[0])
                changes.append(abs(prediction - perm_pred))

            importances[i] = np.mean(changes)

        # Normalize
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)

        feature_importances = {
            name: float(val) for name, val in zip(self.feature_names, importances)
        }

        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return SpectralExplanation(
            method="permutation",
            feature_importances=feature_importances,
            wavelength_importances=importances,
            top_features=sorted_features[:20],
            base_value=0.0,
            prediction=prediction,
        )

    def explain_lime(
        self,
        spectrum: np.ndarray,
        n_samples: int = 100,
        n_features: int = 10,
        kernel_width: float = 0.25,
    ) -> SpectralExplanation:
        """
        Generate LIME explanation for a spectrum.

        LIME fits a local linear model around the prediction to identify
        which features are most important for this specific prediction.

        Parameters
        ----------
        spectrum : np.ndarray
            Input spectrum to explain
        n_samples : int
            Number of perturbed samples
        n_features : int
            Number of features in local model
        kernel_width : float
            Width of kernel for weighting samples

        Returns
        -------
        SpectralExplanation
            LIME-based explanation
        """
        if not HAS_SKLEARN:
            logger.warning("sklearn not available for LIME")
            return self._explain_permutation(spectrum, n_samples)

        if spectrum.ndim == 1:
            spectrum_2d = spectrum.reshape(1, -1)
        else:
            spectrum_2d = spectrum

        # Get baseline prediction
        prediction = float(self._call_model(spectrum_2d)[0])

        # Generate perturbed samples
        rng = np.random.default_rng(42)
        n_features_input = len(self.wavelengths)

        # Create binary masks (which features to perturb)
        masks = rng.random((n_samples, n_features_input)) > 0.5

        # Generate perturbed samples
        perturbed = np.zeros((n_samples, n_features_input))
        for i in range(n_samples):
            perturbed[i] = spectrum_2d[0].copy()
            # Set masked features to zero (or background)
            perturbed[i, ~masks[i]] = 0

        # Get predictions for perturbed samples
        predictions = self._call_model(perturbed)

        # Compute distances from original
        distances = np.sqrt(np.sum((perturbed - spectrum_2d[0]) ** 2, axis=1))

        # Kernel weights
        weights = np.exp(-(distances**2) / (kernel_width**2))

        # Fit weighted linear model
        ridge = Ridge(alpha=1.0)
        ridge.fit(masks, predictions, sample_weight=weights)

        # Get feature importances
        importances = ridge.coef_
        if importances.ndim > 1:
            importances = importances[0]

        feature_importances = {
            name: float(val) for name, val in zip(self.feature_names, importances)
        }

        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )

        return SpectralExplanation(
            method="lime",
            feature_importances=feature_importances,
            wavelength_importances=importances,
            top_features=sorted_features[:20],
            base_value=0.0,
            prediction=prediction,
        )

    def compute_saliency_map(
        self,
        spectrum: np.ndarray,
        target_output: int = 0,
        smooth_sigma: float = 5.0,
    ) -> np.ndarray:
        """
        Compute saliency map showing influential wavelength regions.

        Uses gradient-based attribution if model supports gradients,
        otherwise falls back to finite differences.

        Parameters
        ----------
        spectrum : np.ndarray
            Input spectrum
        target_output : int
            Target output index (for multi-output models)
        smooth_sigma : float
            Gaussian smoothing sigma for saliency map

        Returns
        -------
        np.ndarray
            Saliency map (same shape as spectrum)
        """
        if spectrum.ndim == 1:
            spectrum_2d = spectrum.reshape(1, -1)
        else:
            spectrum_2d = spectrum

        # Compute gradients via finite differences
        eps = 1e-4
        saliency = np.zeros(len(self.wavelengths))

        for i in range(len(self.wavelengths)):
            # Forward difference
            perturbed_plus = spectrum_2d.copy()
            perturbed_plus[0, i] += eps

            perturbed_minus = spectrum_2d.copy()
            perturbed_minus[0, i] -= eps

            pred_plus = self._call_model(perturbed_plus)[0]
            pred_minus = self._call_model(perturbed_minus)[0]

            if isinstance(pred_plus, np.ndarray) and len(pred_plus) > target_output:
                gradient = (pred_plus[target_output] - pred_minus[target_output]) / (2 * eps)
            else:
                gradient = (pred_plus - pred_minus) / (2 * eps)

            saliency[i] = abs(gradient) * abs(spectrum_2d[0, i])

        # Smooth saliency map
        if HAS_SCIPY and smooth_sigma > 0:
            saliency = gaussian_filter1d(saliency, smooth_sigma)

        # Normalize
        if np.max(saliency) > 0:
            saliency = saliency / np.max(saliency)

        return saliency


class ExplanationValidator:
    """
    Validate ML explanations against spectroscopic knowledge.

    Checks whether important features identified by ML models correspond
    to known emission lines, helping detect spurious correlations.

    Parameters
    ----------
    db_path : str
        Path to atomic database
    elements : List[str]
        Elements expected in samples
    wavelength_tolerance_nm : float
        Tolerance for matching features to known lines [nm]
    min_importance_threshold : float
        Minimum importance to consider a feature as "important"

    Example
    -------
    >>> validator = ExplanationValidator(db_path, elements=["Fe", "Cu"])
    >>> validation = validator.validate(explanation, wavelengths)
    >>> if validation.is_valid:
    ...     print("Model uses spectroscopically meaningful features")
    >>> else:
    ...     print(f"Concerns: {validation.spurious_correlations}")
    """

    def __init__(
        self,
        db_path: str,
        elements: List[str],
        wavelength_tolerance_nm: float = 0.2,
        min_importance_threshold: float = 0.01,
    ):
        self.db_path = db_path
        self.elements = elements
        self.wavelength_tolerance_nm = wavelength_tolerance_nm
        self.min_importance_threshold = min_importance_threshold

        # Load known emission lines
        self.known_lines = self._load_known_lines()

        logger.info(
            f"ExplanationValidator: {len(elements)} elements, "
            f"{len(self.known_lines)} known lines"
        )

    def _load_known_lines(self) -> List[Dict[str, Any]]:
        """Load known emission lines from atomic database."""
        columns = ["element", "sp_num", "wavelength_nm", "rel_int"]
        return _load_lines_from_db(self.db_path, self.elements, columns, min_rel_int=100)

    def validate(
        self,
        explanation: SpectralExplanation,
        wavelengths: np.ndarray,
    ) -> ValidationResult:
        """
        Validate an explanation against spectroscopic knowledge.

        Parameters
        ----------
        explanation : SpectralExplanation
            Model explanation to validate
        wavelengths : np.ndarray
            Wavelength axis [nm]

        Returns
        -------
        ValidationResult
            Validation results with matched/unmatched features
        """
        matched_lines: List[Tuple[str, float, float]] = []
        unmatched_features: List[Tuple[str, float]] = []
        spurious_correlations: List[str] = []
        recommendations: List[str] = []

        # Get important features
        important_features = [
            (name, importance)
            for name, importance in explanation.feature_importances.items()
            if abs(importance) >= self.min_importance_threshold
        ]

        # Sort by absolute importance
        important_features.sort(key=lambda x: abs(x[1]), reverse=True)

        # Check each important feature against known lines
        total_importance = sum(abs(imp) for _, imp in important_features)
        matched_importance = 0.0

        for feature_name, importance in important_features:
            # Extract wavelength from feature name
            wavelength = self._extract_wavelength(feature_name, wavelengths)

            if wavelength is None:
                continue

            # Check if wavelength matches a known line
            matched_line = self._find_matching_line(wavelength)

            if matched_line is not None:
                matched_lines.append((matched_line["element"], wavelength, importance))
                matched_importance += abs(importance)
            else:
                unmatched_features.append((feature_name, importance))

                # Check if this might be a spurious correlation
                if abs(importance) > 0.05 * total_importance:
                    spurious_correlations.append(
                        f"High importance feature at {wavelength:.2f}nm does not match "
                        f"any known emission line for {self.elements}"
                    )

        # Compute validation score
        if total_importance > 0:
            score = matched_importance / total_importance
        else:
            score = 0.0

        # Generate recommendations
        if score < 0.5:
            recommendations.append(
                "Less than 50% of model importance aligns with known emission lines. "
                "Consider retraining with physics-guided features."
            )

        if len(spurious_correlations) > 3:
            recommendations.append(
                f"Found {len(spurious_correlations)} potential spurious correlations. "
                "Model may be overfitting to noise or instrumental artifacts."
            )

        if len(matched_lines) < len(self.elements):
            recommendations.append(
                "Some elements have no matched important features. "
                "Consider adding constraints for minimum element representation."
            )

        # Determine validity
        is_valid = score >= 0.5 and len(spurious_correlations) < 5

        return ValidationResult(
            is_valid=is_valid,
            score=score,
            matched_lines=matched_lines,
            unmatched_features=unmatched_features,
            spurious_correlations=spurious_correlations,
            recommendations=recommendations,
        )

    def _extract_wavelength(
        self,
        feature_name: str,
        wavelengths: np.ndarray,
    ) -> Optional[float]:
        """Extract wavelength from feature name."""
        # Try to parse wavelength from name (e.g., "324.75nm")
        import re

        match = re.search(r"(\d+\.?\d*)nm", feature_name)
        if match:
            return float(match.group(1))

        # Try to parse as index
        try:
            idx = int(feature_name)
            if 0 <= idx < len(wavelengths):
                return float(wavelengths[idx])
        except ValueError:
            pass

        return None

    def _find_matching_line(
        self,
        wavelength: float,
    ) -> Optional[Dict[str, Any]]:
        """Find a known line matching the wavelength."""
        for line in self.known_lines:
            if abs(line["wavelength_nm"] - wavelength) < self.wavelength_tolerance_nm:
                return line

        return None

    def generate_report(
        self,
        validation: ValidationResult,
    ) -> str:
        """
        Generate a human-readable validation report.

        Parameters
        ----------
        validation : ValidationResult
            Validation results

        Returns
        -------
        str
            Formatted validation report
        """
        lines = [
            "=" * 70,
            "ML Model Explanation Validation Report",
            "=" * 70,
            f"Valid: {'Yes' if validation.is_valid else 'No'}",
            f"Alignment Score: {validation.score:.1%}",
            "",
            "-" * 70,
            "Matched Features (aligned with known emission lines):",
            "-" * 70,
        ]

        if validation.matched_lines:
            for element, wavelength, importance in validation.matched_lines[:10]:
                lines.append(
                    f"  {element:>4} @ {wavelength:>8.2f} nm  (importance: {importance:+.4f})"
                )
        else:
            lines.append("  None")

        lines.extend(
            [
                "",
                "-" * 70,
                "Unmatched Features (potential concerns):",
                "-" * 70,
            ]
        )

        if validation.unmatched_features:
            for feature_name, importance in validation.unmatched_features[:10]:
                lines.append(f"  {feature_name:>20}  (importance: {importance:+.4f})")
        else:
            lines.append("  None")

        if validation.spurious_correlations:
            lines.extend(
                [
                    "",
                    "-" * 70,
                    "Spurious Correlations Detected:",
                    "-" * 70,
                ]
            )
            for warning in validation.spurious_correlations[:5]:
                lines.append(f"  * {warning}")

        if validation.recommendations:
            lines.extend(
                [
                    "",
                    "-" * 70,
                    "Recommendations:",
                    "-" * 70,
                ]
            )
            for rec in validation.recommendations:
                lines.append(f"  * {rec}")

        lines.append("=" * 70)
        return "\n".join(lines)


class InterpretableModel:
    """
    Wrapper for creating interpretable ML models for LIBS.

    Combines physics-guided feature extraction with transparent ML models
    (linear models, decision trees) that have built-in interpretability.

    Parameters
    ----------
    db_path : str
        Path to atomic database
    elements : List[str]
        Elements to model
    model_type : str
        Model type: "ridge", "lasso", "random_forest", "gradient_boost"
    **model_kwargs
        Additional arguments for the underlying model

    Example
    -------
    >>> model = InterpretableModel(db_path, elements=["Fe", "Cu"], model_type="ridge")
    >>> model.fit(wavelengths, spectra, concentrations)
    >>> predictions, explanations = model.predict_explain(wavelengths, test_spectra)
    """

    SUPPORTED_MODELS = ["ridge", "lasso", "random_forest", "gradient_boost"]

    def __init__(
        self,
        db_path: str,
        elements: List[str],
        model_type: str = "ridge",
        **model_kwargs,
    ):
        if not HAS_SKLEARN:
            raise ImportError("sklearn required. Install with: pip install scikit-learn")

        if model_type not in self.SUPPORTED_MODELS:
            raise ValueError(f"model_type must be one of {self.SUPPORTED_MODELS}")

        self.db_path = db_path
        self.elements = elements
        self.model_type = model_type
        self.model_kwargs = model_kwargs

        # Create feature extractor
        self.feature_extractor = PhysicsGuidedFeatureExtractor(db_path, elements)

        # Create underlying model
        self._model = self._create_model(model_type, model_kwargs)

        # Storage for fitted model info
        self.feature_names_: Optional[List[str]] = None
        self.feature_importances_: Optional[np.ndarray] = None
        self.is_fitted_ = False

        logger.info(f"InterpretableModel created: {model_type} for {len(elements)} elements")

    def _create_model(self, model_type: str, kwargs: Dict) -> Any:
        """Create the underlying sklearn model."""
        if model_type == "ridge":
            return Ridge(**kwargs)
        elif model_type == "lasso":
            return Lasso(**kwargs)
        elif model_type == "random_forest":
            kwargs.setdefault("n_estimators", 100)
            return RandomForestRegressor(**kwargs)
        elif model_type == "gradient_boost":
            kwargs.setdefault("n_estimators", 100)
            return GradientBoostingRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def fit(
        self,
        wavelengths: np.ndarray,
        spectra: np.ndarray,
        targets: np.ndarray,
    ) -> "InterpretableModel":
        """
        Fit the interpretable model.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength axis [nm]
        spectra : np.ndarray
            Training spectra (n_samples, n_wavelengths)
        targets : np.ndarray
            Target values (n_samples,) or (n_samples, n_outputs)

        Returns
        -------
        self
            Fitted model
        """
        # Extract physics-guided features
        features_result = self.feature_extractor.extract_batch(wavelengths, spectra)

        self.feature_names_ = features_result.feature_names
        X = features_result.feature_matrix

        # Fit model
        self._model.fit(X, targets)
        self.is_fitted_ = True

        # Extract feature importances
        if hasattr(self._model, "feature_importances_"):
            self.feature_importances_ = self._model.feature_importances_
        elif hasattr(self._model, "coef_"):
            self.feature_importances_ = np.abs(self._model.coef_)
            assert self.feature_importances_ is not None
            if self.feature_importances_.ndim > 1:
                self.feature_importances_ = np.mean(self.feature_importances_, axis=0)

        logger.info(f"Model fitted with {X.shape[1]} features on {X.shape[0]} samples")

        return self

    def predict(
        self,
        wavelengths: np.ndarray,
        spectra: np.ndarray,
    ) -> np.ndarray:
        """
        Make predictions.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength axis [nm]
        spectra : np.ndarray
            Input spectra (n_samples, n_wavelengths) or (n_wavelengths,)

        Returns
        -------
        np.ndarray
            Predictions
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")

        # Handle single spectrum
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)

        # Extract features
        features_result = self.feature_extractor.extract_batch(wavelengths, spectra)
        X = features_result.feature_matrix

        return self._model.predict(X)

    def predict_explain(
        self,
        wavelengths: np.ndarray,
        spectra: np.ndarray,
    ) -> Tuple[np.ndarray, List[Dict[str, float]]]:
        """
        Make predictions with feature-level explanations.

        Parameters
        ----------
        wavelengths : np.ndarray
            Wavelength axis [nm]
        spectra : np.ndarray
            Input spectra

        Returns
        -------
        predictions : np.ndarray
            Model predictions
        explanations : List[Dict[str, float]]
            Feature contributions for each sample
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before prediction")
        if self.feature_names_ is None:
            raise ValueError("Model has no feature names; fit with store_feature_names=True")

        # Handle single spectrum
        single_input = spectra.ndim == 1
        if single_input:
            spectra = spectra.reshape(1, -1)

        # Extract features
        features_result = self.feature_extractor.extract_batch(wavelengths, spectra)
        X = features_result.feature_matrix

        # Predictions
        predictions = self._model.predict(X)

        # Generate explanations
        explanations = []
        for i in range(X.shape[0]):
            explanation = {}

            if hasattr(self._model, "coef_"):
                # Linear model: contribution = coef * value
                coefs = self._model.coef_
                if coefs.ndim > 1:
                    coefs = coefs[0]

                for j, name in enumerate(self.feature_names_):
                    explanation[name] = float(coefs[j] * X[i, j])

            elif self.feature_importances_ is not None:
                # Tree model: use global importances weighted by feature value
                for j, name in enumerate(self.feature_names_):
                    explanation[name] = float(self.feature_importances_[j] * X[i, j])

            explanations.append(explanation)

        if single_input:
            predictions = predictions[0]
            explanations = explanations[0]

        return predictions, explanations

    def get_feature_importance_table(self) -> str:
        """
        Get formatted feature importance table.

        Returns
        -------
        str
            Formatted table of feature importances
        """
        if not self.is_fitted_ or self.feature_importances_ is None:
            return "Model not fitted or no feature importances available"

        # Sort by importance
        indices = np.argsort(self.feature_importances_)[::-1]

        lines = [
            "=" * 60,
            f"Feature Importances ({self.model_type})",
            "=" * 60,
            f"{'Rank':<6} {'Feature':<35} {'Importance':>12}",
            "-" * 60,
        ]

        if self.feature_names_ is None:
            raise ValueError("feature_names_ not set")
        for rank, idx in enumerate(indices[:20], 1):
            name = self.feature_names_[idx]
            importance = self.feature_importances_[idx]
            lines.append(f"{rank:<6} {name:<35} {importance:>12.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)
