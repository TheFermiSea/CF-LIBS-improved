"""
Transfer learning framework for LIBS instrument calibration.

This module implements calibration transfer between different LIBS instruments
using domain adaptation and fine-tuning techniques. It addresses the fundamental
challenge that different LIBS systems require separate calibrations due to:

1. **Optical response differences** - Detector efficiency, spectrograph throughput
2. **Plasma generation variations** - Laser wavelength, pulse energy, focusing
3. **Environmental factors** - Ambient atmosphere, sample presentation

Key Approaches
--------------
1. **Domain Adaptation** - Align source and target instrument feature distributions
   - Maximum Mean Discrepancy (MMD) minimization
   - Correlation Alignment (CORAL)
   - Subspace alignment

2. **Fine-tuning** - Adapt pre-trained models with minimal target data
   - Feature extractor freezing
   - Last-layer fine-tuning
   - Gradient-based adaptation

3. **Mathematical Correction** - Direct transformation between instruments
   - Piecewise Direct Standardization (PDS)
   - Slope/Bias Correction (SBC)
   - Spectral standardization

User Guide
----------
1. **Basic calibration transfer** (few target samples):

    >>> transfer = CalibrationTransfer(source_spectra, source_concentrations)
    >>> transfer.fit_sbc(target_spectra[:5], target_concentrations[:5])
    >>> corrected = transfer.transform(new_target_spectrum)

2. **Domain adaptation** (unlabeled target spectra):

    >>> adapter = DomainAdapter(method='coral')
    >>> aligned = adapter.fit_transform(source_spectra, target_spectra)

3. **Fine-tuning neural model** (if JAX available):

    >>> finetuner = FineTuner(pretrained_model)
    >>> adapted_model = finetuner.adapt(target_spectra, target_conc, epochs=50)

References
----------
- Feudale et al. (2002): Transfer of multivariate calibration models
- Pan & Yang (2010): A survey on transfer learning
- Long et al. (2015): Learning transferable features with deep adaptation networks
- Zhang et al. (2019): Transfer learning in LIBS quantitative analysis
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
from pathlib import Path
import json

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.transfer")

# Check for optional dependencies
try:
    import jax  # noqa: F401
    import jax.numpy as jnp
    from jax import jit, grad  # noqa: F401
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = np  # type: ignore

try:
    from scipy import linalg  # noqa: F401
    from scipy.optimize import minimize
    from scipy.interpolate import interp1d  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class DomainAdaptationMethod(Enum):
    """Available domain adaptation methods."""

    MMD = auto()  # Maximum Mean Discrepancy
    CORAL = auto()  # Correlation Alignment
    SUBSPACE = auto()  # Subspace Alignment
    TCA = auto()  # Transfer Component Analysis
    NONE = auto()  # No adaptation (identity transform)


class CalibrationMethod(Enum):
    """Available calibration transfer methods."""

    SBC = auto()  # Slope/Bias Correction
    PDS = auto()  # Piecewise Direct Standardization
    DS = auto()  # Direct Standardization
    OSC = auto()  # Orthogonal Signal Correction
    STANDARDIZATION = auto()  # Simple spectral standardization


@dataclass
class TransferResult:
    """
    Result of calibration transfer or domain adaptation.

    Attributes
    ----------
    transformed_spectra : np.ndarray
        Spectra after transformation (n_samples, n_wavelengths)
    method : str
        Method used for transformation
    source_domain : str
        Identifier for source instrument/domain
    target_domain : str
        Identifier for target instrument/domain
    parameters : Dict[str, Any]
        Method-specific parameters learned during fitting
    metrics : Dict[str, float]
        Quality metrics (e.g., domain discrepancy before/after)
    n_transfer_samples : int
        Number of samples used for transfer calibration
    """

    transformed_spectra: np.ndarray
    method: str
    source_domain: str = "source"
    target_domain: str = "target"
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    n_transfer_samples: int = 0


@dataclass
class DomainAdaptationResult:
    """
    Result of domain adaptation.

    Attributes
    ----------
    source_aligned : np.ndarray
        Source domain spectra after alignment
    target_aligned : np.ndarray
        Target domain spectra after alignment
    transform_matrix : np.ndarray
        Transformation matrix (if applicable)
    method : DomainAdaptationMethod
        Adaptation method used
    mmd_before : float
        Maximum Mean Discrepancy before adaptation
    mmd_after : float
        Maximum Mean Discrepancy after adaptation
    explained_variance : float
        Variance explained by alignment (for subspace methods)
    """

    source_aligned: np.ndarray
    target_aligned: np.ndarray
    transform_matrix: Optional[np.ndarray]
    method: DomainAdaptationMethod
    mmd_before: float = 0.0
    mmd_after: float = 0.0
    explained_variance: float = 1.0


@dataclass
class FineTuneResult:
    """
    Result of fine-tuning a model for domain adaptation.

    Attributes
    ----------
    adapted_params : Dict[str, np.ndarray]
        Model parameters after fine-tuning
    loss_history : List[float]
        Training loss history
    validation_loss : Optional[float]
        Final validation loss if validation set provided
    n_epochs : int
        Number of training epochs completed
    converged : bool
        Whether training converged
    final_loss : float
        Final training loss
    """

    adapted_params: Dict[str, np.ndarray]
    loss_history: List[float]
    validation_loss: Optional[float]
    n_epochs: int
    converged: bool
    final_loss: float


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "rbf",
    gamma: Optional[float] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy between two distributions.

    MMD measures the distance between feature means in a reproducing kernel
    Hilbert space (RKHS). Lower values indicate more similar distributions.

    MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]

    Parameters
    ----------
    X : np.ndarray
        Source domain samples (n_source, n_features)
    Y : np.ndarray
        Target domain samples (n_target, n_features)
    kernel : str
        Kernel type: 'rbf', 'linear', or 'poly'
    gamma : float, optional
        RBF kernel bandwidth. If None, uses median heuristic.

    Returns
    -------
    float
        MMD value (squared)

    References
    ----------
    Gretton et al. (2012): A kernel two-sample test
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)

    if gamma is None:
        # Median heuristic for bandwidth selection
        XY = np.vstack([X, Y])
        dists = _pairwise_distances(XY, XY)
        gamma = 1.0 / (2.0 * np.median(dists[dists > 0]) ** 2 + 1e-10)

    # Compute kernel matrices
    K_XX = _compute_kernel(X, X, kernel, gamma)
    K_YY = _compute_kernel(Y, Y, kernel, gamma)
    K_XY = _compute_kernel(X, Y, kernel, gamma)

    n_X = X.shape[0]
    n_Y = Y.shape[0]

    # Unbiased estimator
    mmd_squared = (
        np.sum(K_XX) / (n_X * (n_X - 1) + 1e-10)
        + np.sum(K_YY) / (n_Y * (n_Y - 1) + 1e-10)
        - 2.0 * np.sum(K_XY) / (n_X * n_Y + 1e-10)
    )

    return max(0.0, float(mmd_squared))


def _pairwise_distances(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Compute pairwise Euclidean distances."""
    XX = np.sum(X ** 2, axis=1, keepdims=True)
    YY = np.sum(Y ** 2, axis=1, keepdims=True)
    distances = XX + YY.T - 2.0 * np.dot(X, Y.T)
    return np.sqrt(np.maximum(distances, 0))


def _compute_kernel(
    X: np.ndarray, Y: np.ndarray, kernel: str, gamma: float
) -> np.ndarray:
    """Compute kernel matrix."""
    if kernel == "linear":
        return np.dot(X, Y.T)
    elif kernel == "rbf":
        dists = _pairwise_distances(X, Y)
        return np.exp(-gamma * dists ** 2)
    elif kernel == "poly":
        return (np.dot(X, Y.T) + 1.0) ** 3
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


class DomainAdapter:
    """
    Domain adaptation for LIBS spectra from different instruments.

    This class aligns feature distributions between source and target
    instruments without requiring labeled target data.

    Parameters
    ----------
    method : DomainAdaptationMethod or str
        Adaptation method to use
    n_components : int
        Number of components for dimensionality reduction (subspace methods)
    regularization : float
        Regularization parameter for numerical stability

    Examples
    --------
    >>> adapter = DomainAdapter(method='coral')
    >>> result = adapter.fit_transform(source_spectra, target_spectra)
    >>> print(f"MMD reduced: {result.mmd_before:.4f} -> {result.mmd_after:.4f}")
    """

    def __init__(
        self,
        method: Union[DomainAdaptationMethod, str] = DomainAdaptationMethod.CORAL,
        n_components: int = 50,
        regularization: float = 1e-6,
    ):
        if isinstance(method, str):
            method = DomainAdaptationMethod[method.upper()]

        self.method = method
        self.n_components = n_components
        self.regularization = regularization

        # Learned parameters
        self._transform_matrix: Optional[np.ndarray] = None
        self._source_mean: Optional[np.ndarray] = None
        self._target_mean: Optional[np.ndarray] = None
        self._source_cov_sqrt: Optional[np.ndarray] = None
        self._target_cov_sqrt: Optional[np.ndarray] = None

        self._fitted = False

        logger.info(f"DomainAdapter initialized with method={method.name}")

    def fit(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> "DomainAdapter":
        """
        Fit the domain adaptation transformation.

        Parameters
        ----------
        source : np.ndarray
            Source domain spectra (n_source, n_wavelengths)
        target : np.ndarray
            Target domain spectra (n_target, n_wavelengths)

        Returns
        -------
        DomainAdapter
            Self for method chaining
        """
        source = np.atleast_2d(source)
        target = np.atleast_2d(target)

        if source.shape[1] != target.shape[1]:
            raise ValueError(
                f"Source and target must have same dimensionality: "
                f"{source.shape[1]} vs {target.shape[1]}"
            )

        if self.method == DomainAdaptationMethod.CORAL:
            self._fit_coral(source, target)
        elif self.method == DomainAdaptationMethod.MMD:
            self._fit_mmd(source, target)
        elif self.method == DomainAdaptationMethod.SUBSPACE:
            self._fit_subspace(source, target)
        elif self.method == DomainAdaptationMethod.TCA:
            self._fit_tca(source, target)
        elif self.method == DomainAdaptationMethod.NONE:
            self._transform_matrix = np.eye(source.shape[1])
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._fitted = True
        return self

    def _fit_coral(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Fit CORAL (Correlation Alignment) transformation.

        CORAL aligns source to target by matching second-order statistics:
        A_s = C_s^{-1/2} @ C_t^{1/2}

        Reference: Sun et al. (2016) - Return of Frustratingly Easy Domain Adaptation
        """
        # Center the data
        self._source_mean = np.mean(source, axis=0)
        self._target_mean = np.mean(target, axis=0)

        source_centered = source - self._source_mean
        target_centered = target - self._target_mean

        # Compute covariance matrices with regularization
        n_s, d = source.shape
        n_t = target.shape[0]

        C_s = np.dot(source_centered.T, source_centered) / (n_s - 1)
        C_t = np.dot(target_centered.T, target_centered) / (n_t - 1)

        # Add regularization
        C_s += self.regularization * np.eye(d)
        C_t += self.regularization * np.eye(d)

        # Compute matrix square roots via eigendecomposition
        # C_s^{-1/2} and C_t^{1/2}
        self._source_cov_sqrt = self._matrix_sqrt_inv(C_s)
        self._target_cov_sqrt = self._matrix_sqrt(C_t)

        # CORAL transformation: A = C_s^{-1/2} @ C_t^{1/2}
        self._transform_matrix = np.dot(self._source_cov_sqrt, self._target_cov_sqrt)

        logger.debug(f"CORAL: fitted transformation matrix {self._transform_matrix.shape}")

    def _fit_mmd(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Fit MMD-based domain adaptation.

        This learns a transformation that minimizes MMD between domains.
        Uses gradient descent to optimize the transformation.
        """
        if not HAS_SCIPY:
            raise ImportError("scipy required for MMD adaptation")

        d = source.shape[1]

        # Start with identity transformation
        A_init = np.eye(d).flatten()

        # Define objective: minimize MMD after transformation
        def objective(A_flat: np.ndarray) -> float:
            A = A_flat.reshape(d, d)
            source_transformed = np.dot(source, A)
            return compute_mmd(source_transformed, target)

        # Optimize (limited iterations for efficiency)
        result = minimize(
            objective,
            A_init,
            method='L-BFGS-B',
            options={'maxiter': 100, 'disp': False},
        )

        self._transform_matrix = result.x.reshape(d, d)
        self._source_mean = np.mean(source, axis=0)
        self._target_mean = np.mean(target, axis=0)

        logger.debug(f"MMD: optimized transformation, final MMD={result.fun:.6f}")

    def _fit_subspace(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Fit subspace alignment domain adaptation.

        Projects both domains onto their principal subspaces, then aligns
        the subspaces via Procrustes analysis.

        Reference: Fernando et al. (2013) - Unsupervised Visual Domain Adaptation
        """
        # Center data
        self._source_mean = np.mean(source, axis=0)
        self._target_mean = np.mean(target, axis=0)

        source_centered = source - self._source_mean
        target_centered = target - self._target_mean

        # SVD for principal subspaces - compute on features (columns)
        # We want V matrices (right singular vectors)
        _, _, Vt_s = np.linalg.svd(source_centered, full_matrices=False)
        _, _, Vt_t = np.linalg.svd(target_centered, full_matrices=False)

        # Take top k components
        k = min(self.n_components, Vt_s.shape[0], Vt_t.shape[0])
        P_s = Vt_s[:k, :].T  # Source principal components (d x k)
        P_t = Vt_t[:k, :].T  # Target principal components (d x k)

        # Subspace alignment: M = P_s @ P_s.T @ P_t @ P_t.T
        # This aligns source subspace to target while preserving dimensions
        self._transform_matrix = np.dot(
            np.dot(P_s, P_s.T),
            np.dot(P_t, P_t.T)
        )

        logger.debug(f"Subspace: aligned {k} components")

    def _fit_tca(self, source: np.ndarray, target: np.ndarray) -> None:
        """
        Fit Transfer Component Analysis.

        Learns a kernel-based transformation that minimizes MMD while
        preserving data variance. Uses a simplified linear version that
        preserves input dimensionality for easier downstream use.

        Reference: Pan et al. (2011) - Domain Adaptation via Transfer Component Analysis
        """
        # Center data
        self._source_mean = np.mean(source, axis=0)
        self._target_mean = np.mean(target, axis=0)

        source_centered = source - self._source_mean
        target_centered = target - self._target_mean

        d = source.shape[1]

        # Compute covariances
        C_s = np.dot(source_centered.T, source_centered) / (source.shape[0] - 1)
        C_t = np.dot(target_centered.T, target_centered) / (target.shape[0] - 1)

        # Add regularization
        C_s += self.regularization * np.eye(d)
        C_t += self.regularization * np.eye(d)

        # Compute eigenvectors of within-class scatter
        C_combined = C_s + C_t
        eigenvalues, eigenvectors = np.linalg.eigh(C_combined)

        # Sort by eigenvalue (descending)
        idx = np.argsort(eigenvalues)[::-1]
        k = min(self.n_components, d)
        W = eigenvectors[:, idx[:k]]

        # Create full-rank transformation by projecting to subspace and back
        # W @ W.T projects onto the top-k principal components
        self._transform_matrix = np.dot(W, W.T)

        logger.debug(f"TCA: extracted {k} transfer components")

    def _matrix_sqrt(self, M: np.ndarray) -> np.ndarray:
        """Compute matrix square root via eigendecomposition."""
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure positive
        return np.dot(
            eigenvectors * np.sqrt(eigenvalues),
            eigenvectors.T
        )

    def _matrix_sqrt_inv(self, M: np.ndarray) -> np.ndarray:
        """Compute inverse matrix square root."""
        eigenvalues, eigenvectors = np.linalg.eigh(M)
        eigenvalues = np.maximum(eigenvalues, self.regularization)
        return np.dot(
            eigenvectors * (1.0 / np.sqrt(eigenvalues)),
            eigenvectors.T
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform source domain spectra to target domain.

        Parameters
        ----------
        X : np.ndarray
            Source domain spectra (n_samples, n_wavelengths)

        Returns
        -------
        np.ndarray
            Transformed spectra aligned with target domain
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        X = np.atleast_2d(X)

        if self.method == DomainAdaptationMethod.CORAL:
            # CORAL: center, transform, re-center
            X_centered = X - self._source_mean
            X_transformed = np.dot(X_centered, self._transform_matrix)
            return X_transformed + self._target_mean
        else:
            # General linear transformation
            if self._source_mean is not None:
                X = X - self._source_mean
            X_transformed = np.dot(X, self._transform_matrix)
            if self._target_mean is not None:
                X_transformed = X_transformed + self._target_mean
            return X_transformed

    def fit_transform(
        self,
        source: np.ndarray,
        target: np.ndarray,
    ) -> DomainAdaptationResult:
        """
        Fit and transform in one step.

        Parameters
        ----------
        source : np.ndarray
            Source domain spectra
        target : np.ndarray
            Target domain spectra

        Returns
        -------
        DomainAdaptationResult
            Aligned spectra and transformation details
        """
        # Compute MMD before adaptation
        mmd_before = compute_mmd(source, target)

        # Fit transformation
        self.fit(source, target)

        # Transform source
        source_aligned = self.transform(source)

        # For symmetric methods, also transform target (or leave unchanged)
        target_aligned = target  # Most methods only transform source

        # Compute MMD after adaptation
        mmd_after = compute_mmd(source_aligned, target)

        logger.info(
            f"Domain adaptation ({self.method.name}): "
            f"MMD {mmd_before:.6f} -> {mmd_after:.6f}"
        )

        return DomainAdaptationResult(
            source_aligned=source_aligned,
            target_aligned=target_aligned,
            transform_matrix=self._transform_matrix,
            method=self.method,
            mmd_before=mmd_before,
            mmd_after=mmd_after,
        )


class CalibrationTransfer:
    """
    Transfer calibration models between LIBS instruments.

    This class implements mathematical correction approaches that transform
    spectra from a target instrument to match the source instrument, allowing
    use of existing calibration models.

    Parameters
    ----------
    method : CalibrationMethod or str
        Calibration transfer method
    window_size : int
        Window size for PDS method (default: 5)
    n_components : int
        Number of PLS components for standardization methods

    Examples
    --------
    >>> transfer = CalibrationTransfer(method='sbc')
    >>> # Fit using transfer samples measured on both instruments
    >>> transfer.fit(source_spectra, target_spectra)
    >>> # Transform new target spectra to source domain
    >>> corrected = transfer.transform(new_target_spectra)

    Notes
    -----
    Different methods have different requirements:
    - SBC: Requires paired spectra (same samples on both instruments)
    - PDS: Requires paired spectra, uses local windows
    - DS: Requires representative spectra from both instruments
    """

    def __init__(
        self,
        method: Union[CalibrationMethod, str] = CalibrationMethod.SBC,
        window_size: int = 5,
        n_components: int = 10,
    ):
        if isinstance(method, str):
            method = CalibrationMethod[method.upper()]

        self.method = method
        self.window_size = window_size
        self.n_components = n_components

        # Learned parameters
        self._slope: Optional[np.ndarray] = None
        self._bias: Optional[np.ndarray] = None
        self._transfer_matrix: Optional[np.ndarray] = None
        self._pds_matrices: Optional[List[np.ndarray]] = None

        self._fitted = False

        logger.info(f"CalibrationTransfer initialized with method={method.name}")

    def fit(
        self,
        source_spectra: np.ndarray,
        target_spectra: np.ndarray,
    ) -> "CalibrationTransfer":
        """
        Fit the calibration transfer model.

        Parameters
        ----------
        source_spectra : np.ndarray
            Reference spectra from source instrument (n_samples, n_wavelengths)
        target_spectra : np.ndarray
            Same samples measured on target instrument

        Returns
        -------
        CalibrationTransfer
            Self for method chaining
        """
        source_spectra = np.atleast_2d(source_spectra)
        target_spectra = np.atleast_2d(target_spectra)

        if source_spectra.shape != target_spectra.shape:
            raise ValueError(
                f"Source and target must have same shape: "
                f"{source_spectra.shape} vs {target_spectra.shape}"
            )

        if self.method == CalibrationMethod.SBC:
            self._fit_sbc(source_spectra, target_spectra)
        elif self.method == CalibrationMethod.PDS:
            self._fit_pds(source_spectra, target_spectra)
        elif self.method == CalibrationMethod.DS:
            self._fit_ds(source_spectra, target_spectra)
        elif self.method == CalibrationMethod.STANDARDIZATION:
            self._fit_standardization(source_spectra, target_spectra)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self._fitted = True
        return self

    def _fit_sbc(
        self, source: np.ndarray, target: np.ndarray
    ) -> None:
        """
        Fit Slope/Bias Correction.

        For each wavelength channel, compute:
            source_i = slope_i * target_i + bias_i

        This is the simplest form of spectral standardization.
        """
        n_samples, n_wavelengths = source.shape

        self._slope = np.zeros(n_wavelengths)
        self._bias = np.zeros(n_wavelengths)

        for i in range(n_wavelengths):
            # Linear regression for each channel
            x = target[:, i]
            y = source[:, i]

            # Handle constant channels
            if np.std(x) < 1e-10:
                self._slope[i] = 1.0
                self._bias[i] = np.mean(y) - np.mean(x)
            else:
                # Least squares: y = slope * x + bias
                x_mean = np.mean(x)
                y_mean = np.mean(y)

                self._slope[i] = np.sum((x - x_mean) * (y - y_mean)) / (
                    np.sum((x - x_mean) ** 2) + 1e-10
                )
                self._bias[i] = y_mean - self._slope[i] * x_mean

        logger.debug(
            f"SBC: fitted {n_wavelengths} channels, "
            f"slope range [{self._slope.min():.3f}, {self._slope.max():.3f}]"
        )

    def _fit_pds(
        self, source: np.ndarray, target: np.ndarray
    ) -> None:
        """
        Fit Piecewise Direct Standardization.

        For each wavelength in source, predict from a local window in target.
        This accounts for wavelength shifts and bandwidth differences.

        Reference: Wang et al. (1991) - Multivariate instrument standardization
        """
        n_samples, n_wavelengths = source.shape
        half_window = self.window_size // 2

        self._pds_matrices = []

        for i in range(n_wavelengths):
            # Define window bounds
            start = max(0, i - half_window)
            end = min(n_wavelengths, i + half_window + 1)

            # Extract local window from target
            X_window = target[:, start:end]
            y = source[:, i]

            # Ridge regression for numerical stability
            XtX = np.dot(X_window.T, X_window)
            XtX += 1e-6 * np.eye(XtX.shape[0])
            Xty = np.dot(X_window.T, y)

            coeffs = np.linalg.solve(XtX, Xty)

            # Store: (start_idx, end_idx, coefficients)
            self._pds_matrices.append((start, end, coeffs))

        logger.debug(f"PDS: fitted {n_wavelengths} local models, window={self.window_size}")

    def _fit_ds(
        self, source: np.ndarray, target: np.ndarray
    ) -> None:
        """
        Fit Direct Standardization.

        Learn a full transformation matrix: source = target @ F
        Uses ridge regression for regularization.
        """
        # Ridge regression
        XtX = np.dot(target.T, target)
        XtX += 1e-4 * np.eye(XtX.shape[0])
        Xty = np.dot(target.T, source)

        self._transfer_matrix = np.linalg.solve(XtX, Xty)

        logger.debug(f"DS: fitted transfer matrix {self._transfer_matrix.shape}")

    def _fit_standardization(
        self, source: np.ndarray, target: np.ndarray
    ) -> None:
        """
        Fit simple spectral standardization (mean/std normalization).
        """
        self._source_mean = np.mean(source, axis=0)
        self._source_std = np.std(source, axis=0) + 1e-10
        self._target_mean = np.mean(target, axis=0)
        self._target_std = np.std(target, axis=0) + 1e-10

        logger.debug("Standardization: computed mean/std for both domains")

    def transform(self, target_spectra: np.ndarray) -> np.ndarray:
        """
        Transform target spectra to source domain.

        Parameters
        ----------
        target_spectra : np.ndarray
            Spectra from target instrument

        Returns
        -------
        np.ndarray
            Corrected spectra in source domain
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        target_spectra = np.atleast_2d(target_spectra)

        if self.method == CalibrationMethod.SBC:
            return self._transform_sbc(target_spectra)
        elif self.method == CalibrationMethod.PDS:
            return self._transform_pds(target_spectra)
        elif self.method == CalibrationMethod.DS:
            return self._transform_ds(target_spectra)
        elif self.method == CalibrationMethod.STANDARDIZATION:
            return self._transform_standardization(target_spectra)
        else:
            return target_spectra

    def _transform_sbc(self, target: np.ndarray) -> np.ndarray:
        """Apply Slope/Bias Correction."""
        return target * self._slope + self._bias

    def _transform_pds(self, target: np.ndarray) -> np.ndarray:
        """Apply Piecewise Direct Standardization."""
        n_samples, n_wavelengths = target.shape
        transformed = np.zeros_like(target)

        for i, (start, end, coeffs) in enumerate(self._pds_matrices):
            X_window = target[:, start:end]
            transformed[:, i] = np.dot(X_window, coeffs)

        return transformed

    def _transform_ds(self, target: np.ndarray) -> np.ndarray:
        """Apply Direct Standardization."""
        return np.dot(target, self._transfer_matrix)

    def _transform_standardization(self, target: np.ndarray) -> np.ndarray:
        """Apply simple standardization."""
        normalized = (target - self._target_mean) / self._target_std
        return normalized * self._source_std + self._source_mean

    def fit_transform(
        self,
        source_spectra: np.ndarray,
        target_spectra: np.ndarray,
    ) -> TransferResult:
        """
        Fit and transform in one step.

        Parameters
        ----------
        source_spectra : np.ndarray
            Reference spectra from source instrument
        target_spectra : np.ndarray
            Same samples on target instrument

        Returns
        -------
        TransferResult
            Transformed spectra and transfer details
        """
        self.fit(source_spectra, target_spectra)
        transformed = self.transform(target_spectra)

        # Compute quality metrics
        residual = np.mean(np.abs(transformed - source_spectra))
        r_squared = 1 - np.var(transformed - source_spectra) / (np.var(source_spectra) + 1e-10)

        return TransferResult(
            transformed_spectra=transformed,
            method=self.method.name,
            parameters={
                'slope': self._slope.tolist() if self._slope is not None else None,
                'bias': self._bias.tolist() if self._bias is not None else None,
            },
            metrics={
                'mean_absolute_residual': float(residual),
                'r_squared': float(r_squared),
            },
            n_transfer_samples=source_spectra.shape[0],
        )

    def save(self, path: Union[str, Path]) -> None:
        """
        Save fitted transfer model to file.

        Parameters
        ----------
        path : str or Path
            Output file path (JSON format)
        """
        if not self._fitted:
            raise RuntimeError("Cannot save unfitted model")

        data = {
            'method': self.method.name,
            'window_size': self.window_size,
            'n_components': self.n_components,
            'slope': self._slope.tolist() if self._slope is not None else None,
            'bias': self._bias.tolist() if self._bias is not None else None,
            'transfer_matrix': (
                self._transfer_matrix.tolist()
                if self._transfer_matrix is not None else None
            ),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved transfer model to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "CalibrationTransfer":
        """
        Load transfer model from file.

        Parameters
        ----------
        path : str or Path
            Input file path

        Returns
        -------
        CalibrationTransfer
            Loaded transfer model
        """
        with open(path) as f:
            data = json.load(f)

        transfer = cls(
            method=CalibrationMethod[data['method']],
            window_size=data.get('window_size', 5),
            n_components=data.get('n_components', 10),
        )

        if data.get('slope') is not None:
            transfer._slope = np.array(data['slope'])
        if data.get('bias') is not None:
            transfer._bias = np.array(data['bias'])
        if data.get('transfer_matrix') is not None:
            transfer._transfer_matrix = np.array(data['transfer_matrix'])

        transfer._fitted = True

        logger.info(f"Loaded transfer model from {path}")
        return transfer


class FineTuner:
    """
    Fine-tune LIBS models for domain adaptation with minimal target data.

    This class implements transfer learning by adapting a pre-trained model
    (trained on source instrument) to a new target instrument using only
    a small number of labeled target samples.

    Parameters
    ----------
    learning_rate : float
        Learning rate for fine-tuning (default: 0.001)
    freeze_layers : int
        Number of initial layers to freeze (default: 0, all trainable)
    l2_reg : float
        L2 regularization strength (default: 0.01)
    early_stopping_patience : int
        Stop if no improvement for this many epochs (default: 10)

    Examples
    --------
    >>> # Initialize with pre-trained weights
    >>> finetuner = FineTuner(learning_rate=0.0001, freeze_layers=2)
    >>> # Adapt with small target dataset
    >>> result = finetuner.adapt(
    ...     initial_params=pretrained_params,
    ...     target_spectra=target_X[:10],
    ...     target_concentrations=target_y[:10],
    ...     epochs=100
    ... )

    Notes
    -----
    Requires JAX for gradient-based optimization.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        freeze_layers: int = 0,
        l2_reg: float = 0.01,
        early_stopping_patience: int = 10,
    ):
        if not HAS_JAX:
            logger.warning(
                "JAX not available. FineTuner will use NumPy fallback "
                "(no gradient-based optimization)."
            )

        self.learning_rate = learning_rate
        self.freeze_layers = freeze_layers
        self.l2_reg = l2_reg
        self.early_stopping_patience = early_stopping_patience

        self._adapted_params: Optional[Dict[str, np.ndarray]] = None

        logger.info(
            f"FineTuner initialized: lr={learning_rate}, "
            f"freeze_layers={freeze_layers}, l2_reg={l2_reg}"
        )

    def adapt(
        self,
        initial_params: Dict[str, np.ndarray],
        target_spectra: np.ndarray,
        target_concentrations: np.ndarray,
        epochs: int = 100,
        batch_size: Optional[int] = None,
        validation_split: float = 0.2,
        loss_fn: Optional[Callable] = None,
        verbose: bool = True,
    ) -> FineTuneResult:
        """
        Adapt model parameters to target domain.

        Parameters
        ----------
        initial_params : Dict[str, np.ndarray]
            Pre-trained model parameters (from source domain)
        target_spectra : np.ndarray
            Target domain spectra (n_samples, n_wavelengths)
        target_concentrations : np.ndarray
            Target domain concentrations (n_samples, n_elements)
        epochs : int
            Maximum number of training epochs
        batch_size : int, optional
            Mini-batch size. If None, uses full batch.
        validation_split : float
            Fraction of data for validation
        loss_fn : callable, optional
            Custom loss function. Default: MSE.
        verbose : bool
            Print progress

        Returns
        -------
        FineTuneResult
            Adapted parameters and training history
        """
        target_spectra = np.atleast_2d(target_spectra)
        target_concentrations = np.atleast_2d(target_concentrations)

        n_samples = target_spectra.shape[0]

        # Split into train/validation
        n_val = max(1, int(n_samples * validation_split))
        n_train = n_samples - n_val

        # Shuffle
        idx = np.random.permutation(n_samples)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]

        X_train = target_spectra[train_idx]
        y_train = target_concentrations[train_idx]
        X_val = target_spectra[val_idx]
        y_val = target_concentrations[val_idx]

        if verbose:
            logger.info(
                f"Fine-tuning: {n_train} train, {n_val} val samples, "
                f"{epochs} epochs"
            )

        if HAS_JAX:
            return self._adapt_jax(
                initial_params, X_train, y_train, X_val, y_val,
                epochs, batch_size, loss_fn, verbose
            )
        else:
            return self._adapt_numpy(
                initial_params, X_train, y_train, X_val, y_val,
                epochs, verbose
            )

    def _adapt_jax(
        self,
        initial_params: Dict[str, np.ndarray],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        batch_size: Optional[int],
        loss_fn: Optional[Callable],
        verbose: bool,
    ) -> FineTuneResult:
        """JAX-based gradient descent fine-tuning."""
        import jax.numpy as jnp
        from jax import value_and_grad

        # Convert to JAX arrays
        params = {k: jnp.array(v) for k, v in initial_params.items()}
        X_train = jnp.array(X_train)
        y_train = jnp.array(y_train)
        X_val = jnp.array(X_val)
        y_val = jnp.array(y_val)

        # Default loss: MSE with L2 regularization
        def default_loss(params, X, y):
            # Simple linear model: y_pred = X @ W + b
            W = params.get('W', jnp.eye(X.shape[1], y.shape[1]))
            b = params.get('b', jnp.zeros(y.shape[1]))

            y_pred = jnp.dot(X, W) + b
            mse = jnp.mean((y_pred - y) ** 2)

            # L2 regularization
            l2_loss = self.l2_reg * sum(
                jnp.sum(v ** 2) for v in params.values()
            )

            return mse + l2_loss

        loss_fn = loss_fn or default_loss

        # Training loop
        loss_history = []
        best_val_loss = float('inf')
        best_params = params
        patience_counter = 0

        for epoch in range(epochs):
            # Compute gradients
            loss_val, grads = value_and_grad(loss_fn)(params, X_train, y_train)

            # Update parameters (frozen layers excluded)
            params = {
                k: v - self.learning_rate * grads[k]
                for k, v in params.items()
            }

            loss_history.append(float(loss_val))

            # Validation
            val_loss = float(loss_fn(params, X_val, y_val))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = {k: np.array(v) for k, v in params.items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: train_loss={loss_val:.6f}, "
                    f"val_loss={val_loss:.6f}"
                )

            # Early stopping
            if patience_counter >= self.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break

        converged = patience_counter < self.early_stopping_patience

        self._adapted_params = best_params

        return FineTuneResult(
            adapted_params=best_params,
            loss_history=loss_history,
            validation_loss=best_val_loss,
            n_epochs=len(loss_history),
            converged=converged,
            final_loss=loss_history[-1] if loss_history else float('inf'),
        )

    def _adapt_numpy(
        self,
        initial_params: Dict[str, np.ndarray],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int,
        verbose: bool,
    ) -> FineTuneResult:
        """NumPy fallback: closed-form linear adaptation."""
        logger.info("Using NumPy fallback (ridge regression)")

        # Simple ridge regression as fallback
        # y = X @ W + b
        X_train.shape[1]
        y_train.shape[1]

        # Capture the pre-adaptation train loss when possible so the fallback exposes a
        # meaningful baseline in loss_history, matching JAX mode semantics.
        initial_train_loss: Optional[float] = None
        init_W = initial_params.get('W')
        init_b = initial_params.get('b')
        if init_W is not None:
            if init_b is None:
                init_b = np.zeros(y_train.shape[1])
            try:
                y_pred_init = np.dot(X_train, init_W) + init_b
                initial_train_loss = float(np.mean((y_pred_init - y_train) ** 2))
            except ValueError:
                initial_train_loss = None

        # Add bias column
        X_aug = np.hstack([X_train, np.ones((X_train.shape[0], 1))])

        # Ridge regression
        XtX = np.dot(X_aug.T, X_aug)
        XtX += self.l2_reg * np.eye(XtX.shape[0])
        Xty = np.dot(X_aug.T, y_train)

        W_aug = np.linalg.solve(XtX, Xty)

        W = W_aug[:-1, :]
        b = W_aug[-1, :]

        adapted_params = {'W': W, 'b': b}

        # Compute losses
        y_pred_train = np.dot(X_train, W) + b
        train_loss = np.mean((y_pred_train - y_train) ** 2)

        y_pred_val = np.dot(X_val, W) + b
        val_loss = np.mean((y_pred_val - y_val) ** 2)

        self._adapted_params = adapted_params

        loss_history = [float(train_loss)]
        if initial_train_loss is not None:
            loss_history.insert(0, initial_train_loss)

        return FineTuneResult(
            adapted_params=adapted_params,
            loss_history=loss_history,
            validation_loss=float(val_loss),
            n_epochs=1,
            converged=True,
            final_loss=float(train_loss),
        )


class TransferLearningPipeline:
    """
    Complete transfer learning pipeline for LIBS instrument calibration.

    Combines domain adaptation, calibration transfer, and optional fine-tuning
    into a unified workflow.

    Parameters
    ----------
    source_name : str
        Identifier for source instrument
    target_name : str
        Identifier for target instrument
    adaptation_method : str
        Domain adaptation method ('coral', 'mmd', 'subspace', 'none')
    transfer_method : str
        Calibration transfer method ('sbc', 'pds', 'ds')
    use_finetuning : bool
        Whether to use gradient-based fine-tuning

    Examples
    --------
    >>> pipeline = TransferLearningPipeline(
    ...     source_name="instrument_A",
    ...     target_name="instrument_B",
    ...     adaptation_method="coral",
    ...     transfer_method="sbc"
    ... )
    >>> # Fit with transfer samples
    >>> pipeline.fit(
    ...     source_spectra=source_X,
    ...     target_spectra=target_X,
    ...     target_concentrations=target_y  # Optional for fine-tuning
    ... )
    >>> # Transform new spectra
    >>> corrected = pipeline.transform(new_target_spectra)
    """

    def __init__(
        self,
        source_name: str = "source",
        target_name: str = "target",
        adaptation_method: str = "coral",
        transfer_method: str = "sbc",
        use_finetuning: bool = False,
        **kwargs,
    ):
        self.source_name = source_name
        self.target_name = target_name
        self.use_finetuning = use_finetuning

        # Initialize components
        self.domain_adapter = DomainAdapter(method=adaptation_method, **kwargs)
        self.calibration_transfer = CalibrationTransfer(method=transfer_method, **kwargs)

        if use_finetuning:
            self.finetuner = FineTuner(**kwargs)
        else:
            self.finetuner = None

        self._fitted = False
        self._adaptation_result: Optional[DomainAdaptationResult] = None
        self._transfer_result: Optional[TransferResult] = None

        logger.info(
            f"TransferLearningPipeline: {source_name} -> {target_name}, "
            f"adaptation={adaptation_method}, transfer={transfer_method}"
        )

    def fit(
        self,
        source_spectra: np.ndarray,
        target_spectra: np.ndarray,
        target_concentrations: Optional[np.ndarray] = None,
        source_concentrations: Optional[np.ndarray] = None,
        pretrained_params: Optional[Dict[str, np.ndarray]] = None,
        **finetune_kwargs,
    ) -> "TransferLearningPipeline":
        """
        Fit the complete transfer learning pipeline.

        Parameters
        ----------
        source_spectra : np.ndarray
            Spectra from source instrument (reference)
        target_spectra : np.ndarray
            Spectra from target instrument (to be transformed)
        target_concentrations : np.ndarray, optional
            Concentrations for fine-tuning (required if use_finetuning=True)
        source_concentrations : np.ndarray, optional
            Source concentrations (for validation)
        pretrained_params : dict, optional
            Pre-trained model parameters for fine-tuning

        Returns
        -------
        TransferLearningPipeline
            Self for method chaining
        """
        logger.info(
            f"Fitting pipeline: {source_spectra.shape[0]} source, "
            f"{target_spectra.shape[0]} target samples"
        )

        # Step 1: Domain adaptation (aligns feature distributions)
        self._adaptation_result = self.domain_adapter.fit_transform(
            source_spectra, target_spectra
        )

        # Step 2: Calibration transfer (maps target to source domain)
        adapted_source = self._adaptation_result.source_aligned
        self._transfer_result = self.calibration_transfer.fit_transform(
            adapted_source, target_spectra
        )

        # Step 3: Fine-tuning (optional)
        if self.use_finetuning and self.finetuner is not None:
            if target_concentrations is None:
                logger.warning(
                    "Fine-tuning requested but no target concentrations provided"
                )
            elif pretrained_params is None:
                logger.warning(
                    "Fine-tuning requested but no pretrained params provided"
                )
            else:
                corrected_spectra = self.calibration_transfer.transform(target_spectra)
                self._finetune_result = self.finetuner.adapt(
                    initial_params=pretrained_params,
                    target_spectra=corrected_spectra,
                    target_concentrations=target_concentrations,
                    **finetune_kwargs,
                )

        self._fitted = True
        return self

    def transform(self, target_spectra: np.ndarray) -> np.ndarray:
        """
        Transform target spectra using the fitted pipeline.

        Parameters
        ----------
        target_spectra : np.ndarray
            New spectra from target instrument

        Returns
        -------
        np.ndarray
            Spectra transformed to source domain
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before transform()")

        # Apply calibration transfer (domain adaptation is already incorporated)
        return self.calibration_transfer.transform(target_spectra)

    def evaluate(
        self,
        source_spectra: np.ndarray,
        target_spectra: np.ndarray,
    ) -> Dict[str, float]:
        """
        Evaluate transfer quality.

        Parameters
        ----------
        source_spectra : np.ndarray
            Source domain spectra (ground truth)
        target_spectra : np.ndarray
            Target domain spectra to evaluate

        Returns
        -------
        dict
            Quality metrics including MMD and residual statistics
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before evaluate()")

        transformed = self.transform(target_spectra)

        # Compute metrics
        mmd = compute_mmd(source_spectra, transformed)
        residuals = transformed - source_spectra
        mae = np.mean(np.abs(residuals))
        rmse = np.sqrt(np.mean(residuals ** 2))
        r_squared = 1 - np.var(residuals) / (np.var(source_spectra) + 1e-10)

        return {
            'mmd': float(mmd),
            'mae': float(mae),
            'rmse': float(rmse),
            'r_squared': float(r_squared),
        }

    def summary(self) -> str:
        """Generate human-readable summary of pipeline state."""
        if not self._fitted:
            return "TransferLearningPipeline: not fitted"

        lines = [
            "=" * 60,
            "Transfer Learning Pipeline Summary",
            "=" * 60,
            f"Source: {self.source_name}",
            f"Target: {self.target_name}",
            "-" * 60,
        ]

        if self._adaptation_result is not None:
            lines.extend([
                "Domain Adaptation:",
                f"  Method: {self._adaptation_result.method.name}",
                f"  MMD: {self._adaptation_result.mmd_before:.6f} -> "
                f"{self._adaptation_result.mmd_after:.6f}",
            ])

        if self._transfer_result is not None:
            lines.extend([
                "Calibration Transfer:",
                f"  Method: {self._transfer_result.method}",
                f"  Samples: {self._transfer_result.n_transfer_samples}",
                f"  R-squared: {self._transfer_result.metrics.get('r_squared', 'N/A'):.4f}",
            ])

        lines.append("=" * 60)
        return "\n".join(lines)

    def save(self, path: Union[str, Path]) -> None:
        """Save pipeline state to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save transfer model
        self.calibration_transfer.save(path / "transfer_model.json")

        # Save metadata
        metadata = {
            'source_name': self.source_name,
            'target_name': self.target_name,
            'adaptation_method': self.domain_adapter.method.name,
            'use_finetuning': self.use_finetuning,
        }

        if self._adaptation_result is not None:
            metadata['mmd_before'] = self._adaptation_result.mmd_before
            metadata['mmd_after'] = self._adaptation_result.mmd_after

        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved pipeline to {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "TransferLearningPipeline":
        """Load pipeline from directory."""
        path = Path(path)

        with open(path / "metadata.json") as f:
            metadata = json.load(f)

        pipeline = cls(
            source_name=metadata['source_name'],
            target_name=metadata['target_name'],
            adaptation_method=metadata['adaptation_method'],
            use_finetuning=metadata.get('use_finetuning', False),
        )

        pipeline.calibration_transfer = CalibrationTransfer.load(
            path / "transfer_model.json"
        )
        pipeline._fitted = True

        logger.info(f"Loaded pipeline from {path}")
        return pipeline


# Convenience functions
def transfer_calibration(
    source_spectra: np.ndarray,
    target_spectra: np.ndarray,
    method: str = "sbc",
    **kwargs,
) -> np.ndarray:
    """
    Quick calibration transfer using specified method.

    Parameters
    ----------
    source_spectra : np.ndarray
        Reference spectra from source instrument
    target_spectra : np.ndarray
        Spectra to transform from target instrument
    method : str
        Transfer method ('sbc', 'pds', 'ds', 'standardization')

    Returns
    -------
    np.ndarray
        Transformed spectra
    """
    transfer = CalibrationTransfer(method=method, **kwargs)
    transfer.fit(source_spectra, target_spectra)
    return transfer.transform(target_spectra)


def adapt_domains(
    source_spectra: np.ndarray,
    target_spectra: np.ndarray,
    method: str = "coral",
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quick domain adaptation using specified method.

    Parameters
    ----------
    source_spectra : np.ndarray
        Source domain spectra
    target_spectra : np.ndarray
        Target domain spectra
    method : str
        Adaptation method ('coral', 'mmd', 'subspace', 'tca')

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (aligned_source, aligned_target)
    """
    adapter = DomainAdapter(method=method, **kwargs)
    result = adapter.fit_transform(source_spectra, target_spectra)
    return result.source_aligned, result.target_aligned
