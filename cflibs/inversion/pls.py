"""
Partial Least Squares (PLS) Regression for LIBS multivariate quantitative analysis.

PLS is a widely-used chemometric technique for quantitative LIBS analysis that relates
spectral features (X) to elemental concentrations (Y) by projecting both matrices onto
a common latent space that maximizes covariance.

Key advantages for LIBS:
- Handles high-dimensional spectra with correlated wavelengths
- Robust to multicollinearity (overlapping spectral lines)
- Works well with more variables than samples
- Natural dimensionality reduction for noisy spectra

This module provides:
- **PLSRegression**: Core NIPALS algorithm implementation
- **PLSCrossValidator**: k-fold cross-validation for component selection
- **PLSCalibrationModel**: High-level interface for LIBS calibration workflows

Physics Background
------------------
In LIBS chemometrics, PLS finds linear combinations of wavelengths (latent variables)
that best explain both spectral variance and concentration variance. Unlike PCA
(which maximizes only spectral variance), PLS specifically targets predictive
relationships.

The model decomposes:
    X = T @ P^T + E_x   (spectra = scores x loadings + residuals)
    Y = T @ Q^T + E_y   (concentrations = scores x Y-loadings + residuals)

where T are the shared latent scores that link X and Y.

References
----------
- Wold, S., Sjostrom, M., & Eriksson, L. (2001). PLS-regression: a basic tool of
  chemometrics. Chemometrics and Intelligent Laboratory Systems, 58(2), 109-130.
- Clegg, S. M., et al. (2017). Multivariate analysis of remote LIBS spectra using
  partial least squares. Spectrochimica Acta Part B, 129, 64-85.
- de Jong, S. (1993). SIMPLS: An alternative approach to partial least squares
  regression. Chemometrics and Intelligent Laboratory Systems, 18(3), 251-263.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.pls")


class PLSAlgorithm(Enum):
    """Algorithm variant for PLS computation."""

    NIPALS = "nipals"  # Nonlinear Iterative Partial Least Squares
    SIMPLS = "simpls"  # SIMPLS (faster, equivalent for single Y)


class PreprocessingMethod(Enum):
    """Preprocessing methods for spectral data."""

    NONE = "none"
    CENTER = "center"  # Mean-center only
    AUTOSCALE = "autoscale"  # Mean-center and scale to unit variance (standard)
    SNV = "snv"  # Standard Normal Variate (per-spectrum normalization)
    PARETO = "pareto"  # Mean-center and scale by sqrt(std)


@dataclass
class PLSComponents:
    """
    Decomposition components from PLS regression.

    Attributes
    ----------
    n_components : int
        Number of latent components extracted
    x_scores : np.ndarray
        X scores matrix T (n_samples x n_components)
    x_loadings : np.ndarray
        X loadings matrix P (n_features x n_components)
    y_loadings : np.ndarray
        Y loadings matrix Q (n_targets x n_components)
    x_weights : np.ndarray
        X weights matrix W (n_features x n_components)
    coefficients : np.ndarray
        Regression coefficients B (n_features x n_targets)
    x_explained_variance : np.ndarray
        Variance explained in X by each component
    y_explained_variance : np.ndarray
        Variance explained in Y by each component
    """

    n_components: int
    x_scores: np.ndarray
    x_loadings: np.ndarray
    y_loadings: np.ndarray
    x_weights: np.ndarray
    coefficients: np.ndarray
    x_explained_variance: np.ndarray
    y_explained_variance: np.ndarray


@dataclass
class PLSResult:
    """
    Result of PLS regression prediction.

    Attributes
    ----------
    predictions : np.ndarray
        Predicted Y values (n_samples x n_targets)
    scores : np.ndarray
        Sample scores in latent space (n_samples x n_components)
    residuals : np.ndarray
        Prediction residuals (Y_true - Y_pred), if Y was provided
    r2 : Optional[float]
        Coefficient of determination (if Y was provided)
    rmse : Optional[float]
        Root mean squared error (if Y was provided)
    """

    predictions: np.ndarray
    scores: np.ndarray
    residuals: Optional[np.ndarray] = None
    r2: Optional[float] = None
    rmse: Optional[float] = None

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "PLS Prediction Results:",
            f"  Samples: {self.predictions.shape[0]}",
            f"  Targets: {self.predictions.shape[1]}",
        ]
        if self.r2 is not None:
            lines.append(f"  R2: {self.r2:.4f}")
        if self.rmse is not None:
            lines.append(f"  RMSE: {self.rmse:.4f}")
        return "\n".join(lines)


@dataclass
class CrossValidationResult:
    """
    Result of PLS cross-validation for component selection.

    Attributes
    ----------
    n_components_tested : np.ndarray
        Array of component numbers tested
    cv_rmse : np.ndarray
        Cross-validation RMSE for each number of components
    cv_r2 : np.ndarray
        Cross-validation R2 for each number of components
    optimal_n_components : int
        Optimal number of components (minimum CV-RMSE or one-sigma rule)
    fold_predictions : List[np.ndarray]
        Predictions for each fold (for detailed analysis)
    press : np.ndarray
        Predicted residual error sum of squares for each n_components
    """

    n_components_tested: np.ndarray
    cv_rmse: np.ndarray
    cv_r2: np.ndarray
    optimal_n_components: int
    fold_predictions: List[np.ndarray] = field(default_factory=list)
    press: np.ndarray = field(default_factory=lambda: np.array([]))

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "PLS Cross-Validation Results:",
            f"  Components tested: {self.n_components_tested[0]}-{self.n_components_tested[-1]}",
            f"  Optimal components: {self.optimal_n_components}",
            f"  Best CV-R2: {self.cv_r2[self.optimal_n_components - 1]:.4f}",
            f"  Best CV-RMSE: {self.cv_rmse[self.optimal_n_components - 1]:.4f}",
        ]
        return "\n".join(lines)

    def plot_components(self) -> Tuple:
        """
        Generate data for component selection plot.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]
            (n_components, cv_rmse, cv_r2) arrays for plotting
        """
        return (self.n_components_tested, self.cv_rmse, self.cv_r2)


class PLSRegression:
    """
    Partial Least Squares Regression (PLSR) for multivariate calibration.

    Implements the NIPALS algorithm for PLS1 (single Y) and PLS2 (multiple Y)
    regression, commonly used for LIBS quantitative analysis.

    Parameters
    ----------
    n_components : int
        Number of latent components to extract (default: 10)
    algorithm : PLSAlgorithm
        PLS algorithm variant (default: NIPALS)
    preprocessing : PreprocessingMethod
        Spectral preprocessing (default: AUTOSCALE)
    max_iter : int
        Maximum iterations for NIPALS convergence (default: 500)
    tol : float
        Convergence tolerance (default: 1e-8)

    Examples
    --------
    >>> # Single element calibration
    >>> pls = PLSRegression(n_components=5)
    >>> pls.fit(spectra_train, concentrations_train)
    >>> result = pls.predict(spectra_test)
    >>> print(f"Predicted concentrations: {result.predictions}")

    >>> # Multi-element calibration with cross-validation
    >>> cv = PLSCrossValidator(max_components=15, n_folds=10)
    >>> cv_result = cv.validate(spectra, concentrations)
    >>> print(f"Optimal components: {cv_result.optimal_n_components}")

    Notes
    -----
    For LIBS applications:
    - Typical component counts: 3-15 for geological samples
    - Always use cross-validation to select components
    - Autoscaling is recommended for spectra with varying intensities
    - VIP (Variable Importance in Projection) can identify key wavelengths
    """

    def __init__(
        self,
        n_components: int = 10,
        algorithm: PLSAlgorithm = PLSAlgorithm.NIPALS,
        preprocessing: PreprocessingMethod = PreprocessingMethod.AUTOSCALE,
        max_iter: int = 500,
        tol: float = 1e-8,
    ):
        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")

        self.n_components = n_components
        self.algorithm = algorithm
        self.preprocessing = preprocessing
        self.max_iter = max_iter
        self.tol = tol

        # Fitted parameters (set during fit)
        self._x_mean: Optional[np.ndarray] = None
        self._x_std: Optional[np.ndarray] = None
        self._y_mean: Optional[np.ndarray] = None
        self._components: Optional[PLSComponents] = None
        self._is_fitted: bool = False

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def components(self) -> Optional[PLSComponents]:
        """Fitted PLS components."""
        return self._components

    @property
    def coefficients(self) -> Optional[np.ndarray]:
        """Regression coefficients B (n_features x n_targets)."""
        if self._components is None:
            return None
        return self._components.coefficients

    def _preprocess_x(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Apply preprocessing to X matrix.

        Parameters
        ----------
        X : np.ndarray
            Input spectra (n_samples x n_features)
        fit : bool
            If True, compute and store preprocessing parameters

        Returns
        -------
        np.ndarray
            Preprocessed spectra
        """
        X = np.asarray(X, dtype=np.float64)

        if self.preprocessing == PreprocessingMethod.NONE:
            return X

        if self.preprocessing == PreprocessingMethod.SNV:
            # SNV: normalize each spectrum independently
            row_mean = X.mean(axis=1, keepdims=True)
            row_std = X.std(axis=1, keepdims=True)
            row_std = np.where(row_std < 1e-10, 1.0, row_std)
            return (X - row_mean) / row_std

        # Column-wise preprocessing
        if fit:
            self._x_mean = X.mean(axis=0)
            self._x_std = X.std(axis=0, ddof=1)
            # Prevent division by zero for constant features
            self._x_std = np.where(self._x_std < 1e-10, 1.0, self._x_std)

        if self._x_mean is None or self._x_std is None:
            raise RuntimeError("Model must be fitted before preprocessing")

        X_centered = X - self._x_mean

        if self.preprocessing == PreprocessingMethod.CENTER:
            return X_centered
        elif self.preprocessing == PreprocessingMethod.AUTOSCALE:
            return X_centered / self._x_std
        elif self.preprocessing == PreprocessingMethod.PARETO:
            return X_centered / np.sqrt(self._x_std)
        else:
            raise ValueError(f"Unknown preprocessing: {self.preprocessing}")

    def _preprocess_y(self, Y: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Apply preprocessing to Y matrix (always mean-centered).

        Parameters
        ----------
        Y : np.ndarray
            Target concentrations (n_samples,) or (n_samples, n_targets)
        fit : bool
            If True, compute and store preprocessing parameters

        Returns
        -------
        np.ndarray
            Preprocessed Y (always 2D)
        """
        Y = np.asarray(Y, dtype=np.float64)

        # Ensure 2D
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        if fit:
            self._y_mean = Y.mean(axis=0)

        if self._y_mean is None:
            raise RuntimeError("Model must be fitted before preprocessing")

        return Y - self._y_mean

    def _postprocess_y(self, Y_pred: np.ndarray) -> np.ndarray:
        """Reverse Y preprocessing to get predictions in original scale."""
        if self._y_mean is None:
            raise RuntimeError("Model must be fitted before postprocessing")
        return Y_pred + self._y_mean

    def _nipals_pls(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        n_components: Optional[int] = None,
    ) -> PLSComponents:
        """
        NIPALS algorithm for PLS regression.

        Parameters
        ----------
        X : np.ndarray
            Preprocessed X matrix (n_samples x n_features)
        Y : np.ndarray
            Preprocessed Y matrix (n_samples x n_targets)

        Returns
        -------
        PLSComponents
            Fitted PLS decomposition
        """
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]
        requested_components = self.n_components if n_components is None else n_components
        n_comp = min(requested_components, n_samples, n_features)

        # Initialize storage
        T = np.zeros((n_samples, n_comp))  # X scores
        P = np.zeros((n_features, n_comp))  # X loadings
        Q = np.zeros((n_targets, n_comp))  # Y loadings
        W = np.zeros((n_features, n_comp))  # X weights

        # Working copies for deflation
        Xk = X.copy()
        Yk = Y.copy()

        # Variance tracking
        ss_x_total = np.sum(X**2)
        ss_y_total = np.sum(Y**2)
        x_var_explained = np.zeros(n_comp)
        y_var_explained = np.zeros(n_comp)

        for k in range(n_comp):
            # Initialize u as first column of Y
            u = Yk[:, 0].copy()

            # Initialize w and t for this component (in case of early termination)
            w = np.zeros(n_features)
            t = np.zeros(n_samples)
            q = np.zeros(n_targets)
            converged = False
            w_norm = 0.0
            diff = np.inf
            iteration = -1

            for iteration in range(self.max_iter):
                # X block: w = X'u / u'u
                w = Xk.T @ u
                w_norm = np.linalg.norm(w)
                if w_norm < 1e-10:
                    logger.warning(f"Component {k+1}: near-zero weight vector, stopping")
                    # No more useful components can be extracted
                    n_comp = k
                    break
                w = w / w_norm

                # X scores: t = Xw
                t = Xk @ w

                # Y block: q = Y't / t't
                t_sq = t @ t
                if t_sq < 1e-10:
                    logger.warning(f"Component {k+1}: near-zero score vector, stopping")
                    n_comp = k
                    break

                q = Yk.T @ t / t_sq

                # For PLS2, normalize q and update u
                if n_targets > 1:
                    q_norm = np.linalg.norm(q)
                    if q_norm > 1e-10:
                        q = q / q_norm
                    u_new = Yk @ q
                else:
                    u_new = Yk[:, 0].copy()

                # Check convergence
                diff = np.linalg.norm(u_new - u) / (np.linalg.norm(u) + 1e-10)
                u = u_new

                if diff < self.tol:
                    converged = True
                    break

            if not converged:
                logger.warning(
                    "Component %d: NIPALS did not converge in %d iterations (final diff=%.2e)",
                    k + 1,
                    self.max_iter,
                    diff,
                )

            # If we broke early due to numerical issues, stop extracting components
            if not converged and iteration == 0 and w_norm < 1e-10:
                break

            # Store component
            W[:, k] = w
            T[:, k] = t

            # X loadings: p = X't / t't
            t_sq_final = t @ t
            if t_sq_final < 1e-10:
                # Cannot compute loadings, treat as zero component
                P[:, k] = 0
                Q[:, k] = q
            else:
                P[:, k] = Xk.T @ t / t_sq_final
                Q[:, k] = Yk.T @ t / t_sq_final

                # Deflate X and Y
                Xk = Xk - np.outer(t, P[:, k])
                Yk = Yk - np.outer(t, Q[:, k])

            # Track explained variance
            x_var_explained[k] = 1 - np.sum(Xk**2) / ss_x_total if ss_x_total > 0 else 0
            y_var_explained[k] = 1 - np.sum(Yk**2) / ss_y_total if ss_y_total > 0 else 0

        # Track actual number of valid components extracted
        # (n_comp may have been reduced in the loop)
        actual_n_comp = n_comp
        for k in range(n_comp):
            if np.all(W[:, k] == 0) and np.all(T[:, k] == 0):
                actual_n_comp = k
                break

        # If no valid components were extracted, return minimal result
        if actual_n_comp == 0:
            logger.warning("No valid components could be extracted")
            actual_n_comp = 1  # Keep at least one (even if zero)

        # Trim arrays to actual components
        T = T[:, :actual_n_comp]
        P = P[:, :actual_n_comp]
        Q = Q[:, :actual_n_comp]
        W = W[:, :actual_n_comp]
        x_var_explained = x_var_explained[:actual_n_comp]
        y_var_explained = y_var_explained[:actual_n_comp]

        # Convert cumulative to per-component variance
        x_var_per_comp = np.diff(np.concatenate([[0], x_var_explained]))
        y_var_per_comp = np.diff(np.concatenate([[0], y_var_explained]))

        # Compute regression coefficients B = W(P'W)^-1 Q'
        PW_inv = self._safe_pw_inverse(P, W)
        B = W @ PW_inv @ Q.T

        return PLSComponents(
            n_components=actual_n_comp,
            x_scores=T,
            x_loadings=P,
            y_loadings=Q,
            x_weights=W,
            coefficients=B,
            x_explained_variance=x_var_per_comp,
            y_explained_variance=y_var_per_comp,
        )

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> "PLSRegression":
        """
        Fit PLS model to training data.

        Parameters
        ----------
        X : np.ndarray
            Training spectra (n_samples x n_features)
        Y : np.ndarray
            Target concentrations (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        PLSRegression
            Fitted model (self)

        Raises
        ------
        ValueError
            If X and Y have inconsistent sample sizes
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n_samples_x, n_features = X.shape
        n_samples_y, n_targets = Y.shape

        if n_samples_x != n_samples_y:
            raise ValueError(
                f"X and Y have different sample counts: {n_samples_x} vs {n_samples_y}"
            )

        n_samples = n_samples_x

        # Adjust n_components if needed
        max_components = min(n_samples - 1, n_features)
        n_components = self.n_components
        if n_components > max_components:
            logger.warning(
                "Reducing n_components from %d to %d for this fit",
                n_components,
                max_components,
            )
            n_components = max_components

        logger.info(
            f"Fitting PLS with {n_components} components on "
            f"{n_samples} samples x {n_features} features -> {n_targets} targets"
        )

        # Preprocess
        X_proc = self._preprocess_x(X, fit=True)
        Y_proc = self._preprocess_y(Y, fit=True)

        # Fit using selected algorithm
        if self.algorithm == PLSAlgorithm.NIPALS:
            self._components = self._nipals_pls(X_proc, Y_proc, n_components=n_components)
        else:
            # SIMPLS (not yet implemented, fall back to NIPALS)
            logger.warning("SIMPLS not implemented, using NIPALS")
            self._components = self._nipals_pls(X_proc, Y_proc, n_components=n_components)

        self._is_fitted = True
        logger.info(
            f"PLS fit complete. X variance explained: "
            f"{self._components.x_explained_variance.sum():.1%}, "
            f"Y variance explained: {self._components.y_explained_variance.sum():.1%}"
        )

        return self

    def predict(
        self,
        X: np.ndarray,
        Y: Optional[np.ndarray] = None,
        n_components: Optional[int] = None,
    ) -> PLSResult:
        """
        Predict Y values for new spectra.

        Parameters
        ----------
        X : np.ndarray
            Spectra to predict (n_samples x n_features)
        Y : np.ndarray, optional
            True Y values for computing metrics
        n_components : int, optional
            Number of components to use (default: all fitted components)

        Returns
        -------
        PLSResult
            Predictions and optional quality metrics

        Raises
        ------
        RuntimeError
            If model has not been fitted
        """
        if not self._is_fitted or self._components is None:
            raise RuntimeError("Model must be fitted before prediction")

        X = np.asarray(X, dtype=np.float64)
        X_proc = self._preprocess_x(X, fit=False)

        # Use specified number of components or all
        n_comp = n_components or self._components.n_components
        n_comp = min(n_comp, self._components.n_components)

        # Compute scores
        W = self._components.x_weights[:, :n_comp]
        P = self._components.x_loadings[:, :n_comp]

        # Scores via projection
        PW_inv = self._safe_pw_inverse(P, W)
        T = X_proc @ W @ PW_inv

        # Predict using coefficients with reduced components
        Q = self._components.y_loadings[:, :n_comp]
        Y_pred_proc = T @ Q.T

        # Reverse preprocessing
        Y_pred = self._postprocess_y(Y_pred_proc)

        # Compute metrics if Y provided
        residuals = None
        r2 = None
        rmse = None

        if Y is not None:
            Y = np.asarray(Y, dtype=np.float64)
            if Y.ndim == 1:
                Y = Y.reshape(-1, 1)

            residuals = Y - Y_pred
            ss_res = np.sum(residuals**2)
            ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)

            if ss_tot > 1e-10:
                r2 = float(1 - ss_res / ss_tot)
            else:
                r2 = 0.0

            rmse = float(np.sqrt(np.mean(residuals**2)))

        return PLSResult(
            predictions=Y_pred,
            scores=T,
            residuals=residuals,
            r2=r2,
            rmse=rmse,
        )

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project X onto the latent space (compute scores).

        Parameters
        ----------
        X : np.ndarray
            Spectra (n_samples x n_features)

        Returns
        -------
        np.ndarray
            Score matrix T (n_samples x n_components)
        """
        if not self._is_fitted or self._components is None:
            raise RuntimeError("Model must be fitted before transform")

        X_proc = self._preprocess_x(X, fit=False)

        W = self._components.x_weights
        P = self._components.x_loadings

        PW_inv = self._safe_pw_inverse(P, W)
        return X_proc @ W @ PW_inv

    def vip_scores(self) -> np.ndarray:
        """
        Compute Variable Importance in Projection (VIP) scores.

        VIP scores indicate the importance of each wavelength in the PLS model.
        Variables with VIP > 1 are considered important.

        Returns
        -------
        np.ndarray
            VIP scores (n_features,)

        References
        ----------
        Wold, S. (1994). PLS for multivariate linear modeling. In QSAR: chemometric
        methods in molecular design. VCH.
        """
        if not self._is_fitted or self._components is None:
            raise RuntimeError("Model must be fitted to compute VIP")

        W = self._components.x_weights
        Q = self._components.y_loadings
        T = self._components.x_scores

        n_features = W.shape[0]
        n_comp = W.shape[1]

        # Sum of squares of Y explained by each component
        ss_y = np.sum(T**2, axis=0) * np.sum(Q**2, axis=0)

        # VIP = sqrt(p * sum_k(w_k^2 * ss_y_k) / sum(ss_y))
        vip = np.zeros(n_features)
        ss_y_total = ss_y.sum()

        if ss_y_total > 1e-10:
            for k in range(n_comp):
                vip += (W[:, k] ** 2) * ss_y[k]
            vip = np.sqrt(n_features * vip / ss_y_total)

        return vip

    def get_loadings_spectrum(self, component: int = 0) -> np.ndarray:
        """
        Get X loadings for a specific component (interpretable as spectral pattern).

        Parameters
        ----------
        component : int
            Component index (0-indexed)

        Returns
        -------
        np.ndarray
            Loading vector (n_features,)
        """
        if not self._is_fitted or self._components is None:
            raise RuntimeError("Model must be fitted")

        if component >= self._components.n_components:
            raise ValueError(f"Component {component} not available")

        return self._components.x_loadings[:, component]

    @staticmethod
    def _safe_pw_inverse(P: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Compute (P.T @ W)^-1 with fallback to pseudoinverse.

        Parameters
        ----------
        P : np.ndarray
            X loadings (n_features x n_components)
        W : np.ndarray
            X weights (n_features x n_components)

        Returns
        -------
        np.ndarray
            Inverse or pseudoinverse of (P.T @ W)
        """
        PW = P.T @ W
        try:
            return np.linalg.inv(PW)
        except np.linalg.LinAlgError:
            logger.warning("Singular PW matrix, using pseudoinverse")
            return np.linalg.pinv(PW)


class PLSCrossValidator:
    """
    Cross-validation for PLS component selection.

    Determines the optimal number of PLS components using k-fold cross-validation
    with various selection criteria.

    Parameters
    ----------
    max_components : int
        Maximum number of components to test (default: 20)
    n_folds : int
        Number of cross-validation folds (default: 10)
    preprocessing : PreprocessingMethod
        Preprocessing method (default: AUTOSCALE)
    selection_criterion : str
        Criterion for selecting optimal components:
        - 'min_rmse': Minimum CV-RMSE
        - 'one_sigma': One standard error rule (more parsimonious)
    random_state : Optional[int]
        Random seed for fold assignment

    Examples
    --------
    >>> cv = PLSCrossValidator(max_components=15, n_folds=10)
    >>> result = cv.validate(spectra, concentrations)
    >>> print(f"Optimal components: {result.optimal_n_components}")
    >>> pls = PLSRegression(n_components=result.optimal_n_components)
    >>> pls.fit(spectra, concentrations)
    """

    def __init__(
        self,
        max_components: int = 20,
        n_folds: int = 10,
        preprocessing: PreprocessingMethod = PreprocessingMethod.AUTOSCALE,
        selection_criterion: str = "min_rmse",
        random_state: Optional[int] = None,
    ):
        if n_folds < 2:
            raise ValueError(f"n_folds must be >= 2, got {n_folds}")

        self.max_components = max_components
        self.n_folds = n_folds
        self.preprocessing = preprocessing
        self.selection_criterion = selection_criterion
        self.random_state = random_state

    def _create_folds(self, n_samples: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create train/test indices for each fold."""
        rng = np.random.default_rng(self.random_state)
        indices = rng.permutation(n_samples)

        fold_sizes = np.full(self.n_folds, n_samples // self.n_folds)
        fold_sizes[: n_samples % self.n_folds] += 1

        folds = []
        current = 0
        for fold_size in fold_sizes:
            test_idx = indices[current : current + fold_size]
            train_idx = np.concatenate([indices[:current], indices[current + fold_size :]])
            folds.append((train_idx, test_idx))
            current += fold_size

        return folds

    def validate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> CrossValidationResult:
        """
        Perform cross-validation to select optimal number of components.

        Parameters
        ----------
        X : np.ndarray
            Training spectra (n_samples x n_features)
        Y : np.ndarray
            Target concentrations (n_samples,) or (n_samples, n_targets)

        Returns
        -------
        CrossValidationResult
            CV results with optimal component count
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n_samples, n_features = X.shape
        n_targets = Y.shape[1]

        # Adjust max components
        max_comp = min(self.max_components, n_samples - 2, n_features)
        if max_comp < 1:
            raise ValueError(
                "Cannot perform cross-validation: need at least 3 samples and 1 feature "
                f"(got n_samples={n_samples}, n_features={n_features})."
            )

        logger.info(f"Running {self.n_folds}-fold CV for 1-{max_comp} components")

        folds = self._create_folds(n_samples)
        n_components_tested = np.arange(1, max_comp + 1)

        # Storage for results
        cv_predictions = np.zeros((n_samples, n_targets, max_comp))
        fold_predictions = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train = Y[train_idx]

            # Fit with max components
            pls = PLSRegression(
                n_components=max_comp,
                preprocessing=self.preprocessing,
            )
            pls.fit(X_train, Y_train)

            # Predict for each component count
            for n_comp in range(1, max_comp + 1):
                result = pls.predict(X_test, n_components=n_comp)
                cv_predictions[test_idx, :, n_comp - 1] = result.predictions

            fold_predictions.append(cv_predictions[test_idx].copy())

        # Compute metrics
        press = np.zeros(max_comp)  # Predicted Residual Error Sum of Squares
        cv_rmse = np.zeros(max_comp)
        cv_r2 = np.zeros(max_comp)

        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)

        for n_comp in range(1, max_comp + 1):
            Y_pred = cv_predictions[:, :, n_comp - 1]
            residuals = Y - Y_pred

            press[n_comp - 1] = np.sum(residuals**2)
            cv_rmse[n_comp - 1] = np.sqrt(np.mean(residuals**2))

            if ss_tot > 1e-10:
                cv_r2[n_comp - 1] = 1 - press[n_comp - 1] / ss_tot

        # Select optimal components
        if self.selection_criterion == "min_rmse":
            optimal_n = int(np.argmin(cv_rmse) + 1)
        elif self.selection_criterion == "one_sigma":
            # One standard error rule: simplest model within 1 SE of best
            min_rmse = cv_rmse.min()
            # Estimate SE from fold variability
            se_estimate = cv_rmse.std() / np.sqrt(self.n_folds)
            threshold = min_rmse + se_estimate
            # Find first model below threshold
            candidates = np.where(cv_rmse <= threshold)[0]
            optimal_n = (
                int(candidates[0] + 1) if len(candidates) > 0 else int(np.argmin(cv_rmse) + 1)
            )
        else:
            raise ValueError(f"Unknown selection criterion: {self.selection_criterion}")

        logger.info(
            f"CV complete. Optimal components: {optimal_n} "
            f"(RMSE={cv_rmse[optimal_n-1]:.4f}, R2={cv_r2[optimal_n-1]:.4f})"
        )

        return CrossValidationResult(
            n_components_tested=n_components_tested,
            cv_rmse=cv_rmse,
            cv_r2=cv_r2,
            optimal_n_components=optimal_n,
            fold_predictions=fold_predictions,
            press=press,
        )


@dataclass
class PLSCalibrationModel:
    """
    High-level PLS calibration model for LIBS quantitative analysis.

    Combines PLS regression with wavelength information, element metadata,
    and convenient methods for LIBS workflows.

    Attributes
    ----------
    elements : List[str]
        Target element symbols
    wavelengths : np.ndarray
        Wavelength grid for spectra
    pls_model : PLSRegression
        Fitted PLS model
    cv_result : Optional[CrossValidationResult]
        Cross-validation results (if used)
    calibration_range : Dict[str, Tuple[float, float]]
        Valid concentration range per element (from training data)
    """

    elements: List[str]
    wavelengths: np.ndarray
    pls_model: PLSRegression
    cv_result: Optional[CrossValidationResult] = None
    calibration_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def predict(
        self,
        spectrum: np.ndarray,
        return_scores: bool = False,
    ) -> Dict[str, float]:
        """
        Predict elemental concentrations from a spectrum.

        Parameters
        ----------
        spectrum : np.ndarray
            Input spectrum (n_wavelengths,) or (n_spectra, n_wavelengths)
        return_scores : bool
            If True, also return latent scores

        Returns
        -------
        Dict[str, float]
            Predicted concentrations by element
        """
        spectrum = np.asarray(spectrum)
        if spectrum.ndim == 1:
            spectrum = spectrum.reshape(1, -1)

        result = self.pls_model.predict(spectrum)

        concentrations = {}
        for i, el in enumerate(self.elements):
            conc = float(result.predictions[0, i])
            concentrations[el] = conc

            # Warn if outside calibration range
            if el in self.calibration_range:
                cmin, cmax = self.calibration_range[el]
                if conc < cmin or conc > cmax:
                    logger.warning(
                        f"Predicted {el}={conc:.4f} outside calibration range "
                        f"[{cmin:.4f}, {cmax:.4f}]"
                    )

        return concentrations

    def get_important_wavelengths(self, vip_threshold: float = 1.0) -> Dict[str, List[float]]:
        """
        Get wavelengths with VIP > threshold.

        Parameters
        ----------
        vip_threshold : float
            Minimum VIP score (default: 1.0)

        Returns
        -------
        Dict[str, List[float]]
            Important wavelengths with their VIP scores
        """
        vip = self.pls_model.vip_scores()
        important_idx = np.where(vip > vip_threshold)[0]

        return {
            "wavelengths": self.wavelengths[important_idx].tolist(),
            "vip_scores": vip[important_idx].tolist(),
        }

    def summary(self) -> str:
        """Generate a human-readable model summary."""
        lines = [
            "PLS Calibration Model",
            "=" * 50,
            f"Elements: {', '.join(self.elements)}",
            f"Wavelength range: {self.wavelengths.min():.1f}-{self.wavelengths.max():.1f} nm",
            f"Components: {self.pls_model.n_components}",
            f"Preprocessing: {self.pls_model.preprocessing.value}",
        ]

        if self.cv_result:
            lines.append("-" * 50)
            lines.append("Cross-Validation:")
            lines.append(f"  R2: {self.cv_result.cv_r2[self.cv_result.optimal_n_components-1]:.4f}")
            lines.append(
                f"  RMSE: {self.cv_result.cv_rmse[self.cv_result.optimal_n_components-1]:.4f}"
            )

        if self.calibration_range:
            lines.append("-" * 50)
            lines.append("Calibration Ranges:")
            for el, (cmin, cmax) in self.calibration_range.items():
                lines.append(f"  {el}: {cmin:.4f} - {cmax:.4f}")

        return "\n".join(lines)


def build_pls_calibration(
    spectra: np.ndarray,
    concentrations: Union[np.ndarray, Dict[str, np.ndarray]],
    wavelengths: Optional[np.ndarray] = None,
    elements: Optional[List[str]] = None,
    n_components: Optional[int] = None,
    use_cv: bool = True,
    cv_folds: int = 10,
    max_components: int = 20,
    preprocessing: PreprocessingMethod = PreprocessingMethod.AUTOSCALE,
    random_state: Optional[int] = None,
) -> PLSCalibrationModel:
    """
    Build a PLS calibration model from training data.

    Convenience function that handles cross-validation for component selection
    and returns a ready-to-use calibration model.

    Parameters
    ----------
    spectra : np.ndarray
        Training spectra (n_samples x n_wavelengths)
    concentrations : Union[np.ndarray, Dict[str, np.ndarray]]
        Target concentrations as array (n_samples x n_elements) or
        dictionary mapping element symbols to concentration arrays
    wavelengths : np.ndarray, optional
        Wavelength grid (n_wavelengths,)
    elements : List[str], optional
        Element symbols (required if concentrations is array)
    n_components : int, optional
        Fixed number of components (if None, use CV to select)
    use_cv : bool
        Whether to use cross-validation for component selection
    cv_folds : int
        Number of CV folds
    max_components : int
        Maximum components to test in CV
    preprocessing : PreprocessingMethod
        Preprocessing method
    random_state : int, optional
        Random seed for CV

    Returns
    -------
    PLSCalibrationModel
        Fitted calibration model

    Examples
    --------
    >>> model = build_pls_calibration(
    ...     spectra=train_spectra,
    ...     concentrations={"Fe": fe_conc, "Si": si_conc},
    ...     wavelengths=wl_grid,
    ... )
    >>> predictions = model.predict(test_spectrum)
    >>> print(f"Fe: {predictions['Fe']:.2%}")
    """
    spectra = np.asarray(spectra, dtype=np.float64)
    n_samples, n_wavelengths = spectra.shape

    # Handle concentrations input format
    if isinstance(concentrations, dict):
        elements = list(concentrations.keys())
        Y = np.column_stack([concentrations[el] for el in elements])
    else:
        Y = np.asarray(concentrations, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        if elements is None:
            elements = [f"Y{i}" for i in range(Y.shape[1])]

    # Wavelengths
    if wavelengths is None:
        wavelengths = np.arange(n_wavelengths, dtype=np.float64)
    else:
        wavelengths = np.asarray(wavelengths, dtype=np.float64)

    # Cross-validation for component selection
    cv_result = None
    if use_cv and n_components is None:
        cv = PLSCrossValidator(
            max_components=max_components,
            n_folds=cv_folds,
            preprocessing=preprocessing,
            random_state=random_state,
        )
        cv_result = cv.validate(spectra, Y)
        n_components = cv_result.optimal_n_components

    # Default if no CV
    if n_components is None:
        n_components = min(10, n_samples - 1, n_wavelengths)

    # Fit final model
    pls = PLSRegression(
        n_components=n_components,
        preprocessing=preprocessing,
    )
    pls.fit(spectra, Y)

    # Compute calibration ranges
    calibration_range = {}
    for i, el in enumerate(elements):
        calibration_range[el] = (float(Y[:, i].min()), float(Y[:, i].max()))

    return PLSCalibrationModel(
        elements=elements,
        wavelengths=wavelengths,
        pls_model=pls,
        cv_result=cv_result,
        calibration_range=calibration_range,
    )
