"""
Principal Component Analysis for LIBS spectra.

This module provides high-performance PCA using JAX SVD for dimensionality
reduction of LIBS spectral data. PCA is useful for:

- **Noise reduction**: Reconstructing spectra from top principal components
- **Feature extraction**: Reducing dimensionality for downstream analysis
- **Visualization**: Projecting high-dimensional spectra to 2D/3D
- **Anomaly detection**: Identifying spectra with unusual scores/residuals

The implementation supports both NumPy and JAX backends, with JAX providing
GPU acceleration for large datasets.

User Guide
----------
1. **Basic PCA workflow**:

    >>> pca = PCAPipeline(n_components=10)
    >>> result = pca.fit(spectra)
    >>> print(f"Explained variance: {result.explained_variance_ratio.sum():.2%}")
    >>> scores = result.transform(spectra)

2. **Noise reduction via reconstruction**:

    >>> pca = PCAPipeline(n_components=5)
    >>> result = pca.fit(spectra)
    >>> denoised = result.inverse_transform(result.transform(spectra))

3. **Determine optimal components**:

    >>> pca = PCAPipeline(n_components=50)
    >>> result = pca.fit(spectra)
    >>> n_optimal = result.n_components_for_variance(0.95)
    >>> print(f"Need {n_optimal} components for 95% variance")

References
----------
- Jolliffe (2002), Principal Component Analysis, Springer
- Sirven et al. (2006), "PCA and SIMCA for LIBS analysis"
- Zhang et al. (2015), "Multivariate analysis in LIBS"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, Union, Any
import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.pca")

# Check for JAX availability
try:
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    jnp = None

    def jit(f):
        return f


@dataclass
class PCAResult:
    """
    Results from Principal Component Analysis.

    Stores fitted PCA model parameters, statistics, and provides methods
    for transforming new data.

    Attributes
    ----------
    components : np.ndarray
        Principal component vectors, shape (n_components, n_features).
        Each row is a principal component (eigenvector of covariance matrix).
    explained_variance : np.ndarray
        Variance explained by each component, shape (n_components,).
    explained_variance_ratio : np.ndarray
        Fraction of total variance explained by each component.
        Values sum to <= 1.0 (equality when all components are kept).
    singular_values : np.ndarray
        Singular values from SVD decomposition.
    mean : np.ndarray
        Per-feature mean of training data, shape (n_features,).
    n_components : int
        Number of components fitted.
    n_features : int
        Number of features (wavelength channels) in input data.
    n_samples : int
        Number of samples used for fitting.
    total_variance : float
        Total variance in the original data.

    Methods
    -------
    transform(X)
        Project data onto principal components
    inverse_transform(scores)
        Reconstruct data from principal component scores
    n_components_for_variance(threshold)
        Find number of components for target explained variance
    reconstruction_error(X)
        Compute reconstruction error for data
    """

    # Core PCA results
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    singular_values: np.ndarray
    mean: np.ndarray

    # Metadata
    n_components: int
    n_features: int
    n_samples: int
    total_variance: float

    # Optional: keep JAX arrays for GPU operations
    _components_jax: Any = field(default=None, repr=False)
    _mean_jax: Any = field(default=None, repr=False)

    @property
    def cumulative_variance_ratio(self) -> np.ndarray:
        """Cumulative explained variance ratio."""
        return np.cumsum(self.explained_variance_ratio)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Project data onto principal components (scores).

        Parameters
        ----------
        X : np.ndarray
            Data matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Principal component scores, shape (n_samples, n_components).
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Feature count mismatch: got {X.shape[1]}, expected {self.n_features}"
            )

        # Center data and project
        X_centered = X - self.mean
        scores = X_centered @ self.components.T

        return scores

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from principal component scores.

        Parameters
        ----------
        scores : np.ndarray
            Principal component scores, shape (n_samples, n_components).

        Returns
        -------
        np.ndarray
            Reconstructed data, shape (n_samples, n_features).
        """
        scores = np.asarray(scores, dtype=np.float64)

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        if scores.shape[1] != self.n_components:
            raise ValueError(
                f"Component count mismatch: got {scores.shape[1]}, expected {self.n_components}"
            )

        # Reconstruct: scores @ components + mean
        reconstructed = scores @ self.components + self.mean

        return reconstructed

    def n_components_for_variance(self, threshold: float = 0.95) -> int:
        """
        Find minimum number of components for target explained variance.

        Parameters
        ----------
        threshold : float
            Target cumulative explained variance ratio (default: 0.95).

        Returns
        -------
        int
            Minimum number of components to achieve the threshold.
            Returns n_components if threshold cannot be reached.
        """
        if threshold <= 0 or threshold > 1:
            raise ValueError(f"Threshold must be in (0, 1], got {threshold}")

        cumvar = self.cumulative_variance_ratio
        idx = int(np.searchsorted(cumvar, threshold))

        # searchsorted returns the index where threshold would be inserted
        # We need at least that many components (1-indexed)
        return min(idx + 1, self.n_components)

    def reconstruction_error(
        self,
        X: np.ndarray,
        n_components: Optional[int] = None,
    ) -> Tuple[np.ndarray, float]:
        """
        Compute reconstruction error for data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix, shape (n_samples, n_features).
        n_components : int, optional
            Number of components to use for reconstruction.
            If None, uses all fitted components.

        Returns
        -------
        per_sample_error : np.ndarray
            Mean squared error per sample, shape (n_samples,).
        total_rmse : float
            Root mean squared error across all samples.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if n_components is None:
            n_components = self.n_components

        if n_components < 1:
            raise ValueError(f"n_components must be >= 1, got {n_components}")

        if n_components > self.n_components:
            raise ValueError(
                f"Requested {n_components} components but only {self.n_components} fitted"
            )

        # Use subset of components if requested
        components_subset = self.components[:n_components]

        # Center, project, reconstruct
        X_centered = X - self.mean
        scores = X_centered @ components_subset.T
        reconstructed_centered = scores @ components_subset
        reconstructed = reconstructed_centered + self.mean

        # Compute errors
        residuals = X - reconstructed
        per_sample_mse = np.mean(residuals**2, axis=1)
        total_rmse = float(np.sqrt(np.mean(per_sample_mse)))

        return per_sample_mse, total_rmse

    def scree_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get data for scree plot visualization.

        Returns
        -------
        component_indices : np.ndarray
            Component indices (1-based for plotting)
        variance_ratio : np.ndarray
            Individual explained variance ratios
        cumulative_ratio : np.ndarray
            Cumulative explained variance ratios
        """
        indices = np.arange(1, self.n_components + 1)
        return indices, self.explained_variance_ratio, self.cumulative_variance_ratio

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            "=" * 60,
            "PCA Results Summary",
            "=" * 60,
            f"Samples: {self.n_samples} | Features: {self.n_features} | "
            f"Components: {self.n_components}",
            "-" * 60,
            f"Total variance: {self.total_variance:.4e}",
            f"Explained variance (all components): {self.cumulative_variance_ratio[-1]:.2%}",
            "-" * 60,
            "Top Components:",
        ]

        # Show top 5 components
        n_show = min(5, self.n_components)
        for i in range(n_show):
            lines.append(
                f"  PC{i+1}: {self.explained_variance_ratio[i]:.2%} "
                f"(cumulative: {self.cumulative_variance_ratio[i]:.2%})"
            )

        if self.n_components > n_show:
            lines.append(f"  ... ({self.n_components - n_show} more components)")

        # Component requirements for common thresholds
        lines.append("-" * 60)
        for thresh in [0.90, 0.95, 0.99]:
            if self.cumulative_variance_ratio[-1] >= thresh:
                n_needed = self.n_components_for_variance(thresh)
                lines.append(f"Components for {thresh:.0%} variance: {n_needed}")

        lines.append("=" * 60)
        return "\n".join(lines)


class PCAPipeline:
    """
    Principal Component Analysis pipeline for LIBS spectra.

    Uses SVD decomposition for numerical stability and efficiency.
    Supports both NumPy (CPU) and JAX (GPU) backends.

    Parameters
    ----------
    n_components : int or float, optional
        Number of components to keep:
        - int: Keep exactly this many components
        - float (0 < n < 1): Keep components explaining this fraction of variance
        - None: Keep min(n_samples, n_features) components
    center : bool
        Whether to center data by subtracting mean (default: True).
        This is standard for PCA; disable only for special cases.
    use_jax : bool or None
        Force JAX (True) or NumPy (False) backend.
        If None (default), uses JAX if available.
    svd_solver : str
        SVD algorithm: 'full' or 'truncated' (default: 'full').
        'truncated' is faster for large data with few components.

    Attributes
    ----------
    result_ : PCAResult or None
        Fitted result (None before fit() is called)
    is_fitted : bool
        Whether the model has been fitted

    Examples
    --------
    >>> # Fit PCA and transform data
    >>> pca = PCAPipeline(n_components=10)
    >>> result = pca.fit(spectra)
    >>> scores = result.transform(spectra)
    >>> print(f"Shape: {scores.shape}")  # (n_samples, 10)

    >>> # Automatic component selection for 95% variance
    >>> pca = PCAPipeline(n_components=0.95)
    >>> result = pca.fit(spectra)
    >>> print(f"Selected {result.n_components} components")

    >>> # GPU-accelerated PCA with JAX
    >>> pca = PCAPipeline(n_components=20, use_jax=True)
    >>> result = pca.fit(large_dataset)
    """

    def __init__(
        self,
        n_components: Optional[Union[int, float]] = None,
        center: bool = True,
        use_jax: Optional[bool] = None,
        svd_solver: str = "full",
    ):
        self.n_components = n_components
        self.center = center
        self.svd_solver = svd_solver

        # Determine backend
        if use_jax is None:
            self.use_jax = HAS_JAX
        elif use_jax and not HAS_JAX:
            raise ImportError("JAX requested but not available")
        else:
            self.use_jax = use_jax

        self.result_: Optional[PCAResult] = None

        if svd_solver not in ("full", "truncated"):
            raise ValueError(f"Unknown svd_solver: {svd_solver}")

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self.result_ is not None

    def fit(self, X: np.ndarray) -> PCAResult:
        """
        Fit PCA model to data.

        Parameters
        ----------
        X : np.ndarray
            Data matrix, shape (n_samples, n_features).
            For LIBS: samples are spectra, features are wavelength channels.

        Returns
        -------
        PCAResult
            Fitted PCA model with components and statistics.

        Raises
        ------
        ValueError
            If input is invalid or n_components is too large.
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X.shape}")

        n_samples, n_features = X.shape

        if n_samples < 2:
            raise ValueError(f"Need at least 2 samples, got {n_samples}")

        # Determine number of components
        n_components = self._resolve_n_components(self.n_components, n_samples, n_features)

        logger.info(
            f"Fitting PCA: {n_samples} samples, {n_features} features, "
            f"{n_components} components, backend={'JAX' if self.use_jax else 'NumPy'}"
        )

        # Center data
        if self.center:
            mean = np.mean(X, axis=0)
            X_centered = X - mean
        else:
            mean = np.zeros(n_features, dtype=np.float64)
            X_centered = X

        # Total variance (before any dimensionality reduction)
        # When center=False, np.var() re-centers internally which gives wrong
        # results since SVD ran on uncentered data. Use sum(X^2)/(n-1) instead.
        if self.center:
            total_variance = float(np.var(X_centered, axis=0, ddof=1).sum())
        else:
            total_variance = float(np.sum(X_centered**2) / (n_samples - 1))

        # Perform SVD
        if self.use_jax:
            result = self._fit_jax(X_centered, n_components, mean, total_variance)
        else:
            result = self._fit_numpy(X_centered, n_components, mean, total_variance)

        # Update result with metadata
        result = PCAResult(
            components=result.components,
            explained_variance=result.explained_variance,
            explained_variance_ratio=result.explained_variance_ratio,
            singular_values=result.singular_values,
            mean=mean,
            n_components=result.n_components,
            n_features=n_features,
            n_samples=n_samples,
            total_variance=total_variance,
            _components_jax=result._components_jax if self.use_jax else None,
            _mean_jax=result._mean_jax if self.use_jax else None,
        )

        self.result_ = result

        logger.info(
            f"PCA fitted: {n_components} components, "
            f"{result.cumulative_variance_ratio[-1]:.1%} variance explained"
        )

        return result

    def _resolve_n_components(
        self,
        n_components: Optional[Union[int, float]],
        n_samples: int,
        n_features: int,
    ) -> int:
        """Resolve n_components parameter to actual component count."""
        max_components = min(n_samples, n_features)

        if n_components is None:
            return max_components

        if isinstance(n_components, float):
            if not (0 < n_components < 1):
                raise ValueError(f"n_components as float must be in (0, 1), got {n_components}")
            # Will be resolved after SVD based on variance
            # For now, fit all and truncate later
            return max_components

        if isinstance(n_components, int):
            if n_components < 1:
                raise ValueError(f"n_components must be >= 1, got {n_components}")
            if n_components > max_components:
                raise ValueError(
                    f"n_components={n_components} exceeds max({n_samples}, {n_features})={max_components}"
                )
            return n_components

        raise TypeError(f"n_components must be int, float, or None, got {type(n_components)}")

    def _apply_variance_selection(
        self,
        components: np.ndarray,
        singular_values: np.ndarray,
        explained_variance: np.ndarray,
        explained_variance_ratio: np.ndarray,
        n_components: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Apply variance selection if n_components is a float."""
        if isinstance(self.n_components, float):
            cumvar = np.cumsum(explained_variance_ratio)
            n_keep = np.searchsorted(cumvar, self.n_components) + 1
            n_keep = min(n_keep, len(components))

            components = components[:n_keep]
            singular_values = singular_values[:n_keep]
            explained_variance = explained_variance[:n_keep]
            explained_variance_ratio = explained_variance_ratio[:n_keep]
            n_components = n_keep

        return (
            components,
            singular_values,
            explained_variance,
            explained_variance_ratio,
            n_components,
        )

    def _fit_numpy(
        self,
        X_centered: np.ndarray,
        n_components: int,
        mean: np.ndarray,
        total_variance: float,
    ) -> PCAResult:
        """Fit PCA using NumPy SVD."""
        n_samples = X_centered.shape[0]

        # SVD: X = U @ S @ Vt
        # Components are rows of Vt
        if self.svd_solver == "truncated" and n_components < min(X_centered.shape):
            # Use randomized SVD for efficiency (requires scipy)
            try:
                from scipy.sparse.linalg import svds

                # svds returns components in reverse order
                U, S, Vt = svds(X_centered, k=n_components)
                # Reverse to get largest first
                idx = np.argsort(S)[::-1]
                S = S[idx]
                Vt = Vt[idx]
            except ImportError:
                logger.warning("scipy not available, falling back to full SVD")
                U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
                Vt = Vt[:n_components]
                S = S[:n_components]
        else:
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            Vt = Vt[:n_components]
            S = S[:n_components]

        # Components are rows of Vt
        components = Vt

        # Explained variance: s^2 / (n_samples - 1)
        explained_variance = (S**2) / (n_samples - 1)

        # Handle case where total_variance might be zero or very small
        if total_variance > 1e-10:
            explained_variance_ratio = explained_variance / total_variance
        else:
            explained_variance_ratio = np.zeros_like(explained_variance)

        # Handle variance selection if n_components was a float
        (
            components,
            singular_values,
            explained_variance,
            explained_variance_ratio,
            n_components,
        ) = self._apply_variance_selection(
            components, S, explained_variance, explained_variance_ratio, n_components
        )

        return PCAResult(
            components=components,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            singular_values=singular_values,
            mean=mean,
            n_components=n_components,
            n_features=X_centered.shape[1],
            n_samples=X_centered.shape[0],
            total_variance=total_variance,
        )

    def _fit_jax(
        self,
        X_centered: np.ndarray,
        n_components: int,
        mean: np.ndarray,
        total_variance: float,
    ) -> PCAResult:
        """Fit PCA using JAX SVD (GPU-accelerated)."""
        if not HAS_JAX:
            raise RuntimeError("JAX not available")

        n_samples = X_centered.shape[0]

        # Convert to JAX array; use float64 when x64 mode is enabled
        import jax

        _use_f64 = getattr(jax.config, "jax_enable_x64", False)
        _dtype = jnp.float64 if _use_f64 else jnp.float32
        X_jax = jnp.array(X_centered, dtype=_dtype)

        # JAX SVD
        U, S, Vt = jnp.linalg.svd(X_jax, full_matrices=False)

        # Truncate to n_components
        S = S[:n_components]
        Vt = Vt[:n_components]

        # Convert back to NumPy for storage
        components = np.array(Vt)
        singular_values = np.array(S)

        # Explained variance
        explained_variance = (singular_values**2) / (n_samples - 1)

        if total_variance > 1e-10:
            explained_variance_ratio = explained_variance / total_variance
        else:
            explained_variance_ratio = np.zeros_like(explained_variance)

        # Handle variance selection if n_components was a float
        (
            components,
            singular_values,
            explained_variance,
            explained_variance_ratio,
            n_components,
        ) = self._apply_variance_selection(
            components, singular_values, explained_variance, explained_variance_ratio, n_components
        )

        # Keep JAX arrays for fast transform
        components_jax = jnp.array(components)
        mean_jax = jnp.array(mean)

        return PCAResult(
            components=components,
            explained_variance=explained_variance,
            explained_variance_ratio=explained_variance_ratio,
            singular_values=singular_values,
            mean=mean,
            n_components=n_components,
            n_features=X_centered.shape[1],
            n_samples=X_centered.shape[0],
            total_variance=total_variance,
            _components_jax=components_jax,
            _mean_jax=mean_jax,
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one call.

        Parameters
        ----------
        X : np.ndarray
            Data matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Principal component scores, shape (n_samples, n_components).
        """
        result = self.fit(X)
        return result.transform(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted PCA model.

        Parameters
        ----------
        X : np.ndarray
            Data matrix, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Principal component scores, shape (n_samples, n_components).

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before transform()")
        assert self.result_ is not None
        return self.result_.transform(X)

    def inverse_transform(self, scores: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from principal component scores.

        Parameters
        ----------
        scores : np.ndarray
            Principal component scores, shape (n_samples, n_components).

        Returns
        -------
        np.ndarray
            Reconstructed data, shape (n_samples, n_features).

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before inverse_transform()")
        assert self.result_ is not None
        return self.result_.inverse_transform(scores)


# --- JAX-accelerated functions for batch operations ---

if HAS_JAX:

    @jit
    def pca_transform_jax(
        X: jnp.ndarray,
        components: jnp.ndarray,
        mean: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JAX-accelerated PCA transform.

        Parameters
        ----------
        X : jnp.ndarray
            Data matrix, shape (n_samples, n_features)
        components : jnp.ndarray
            Principal components, shape (n_components, n_features)
        mean : jnp.ndarray
            Feature means, shape (n_features,)

        Returns
        -------
        jnp.ndarray
            Scores, shape (n_samples, n_components)
        """
        X_centered = X - mean
        return X_centered @ components.T

    @jit
    def pca_inverse_transform_jax(
        scores: jnp.ndarray,
        components: jnp.ndarray,
        mean: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JAX-accelerated PCA inverse transform.

        Parameters
        ----------
        scores : jnp.ndarray
            Principal component scores, shape (n_samples, n_components)
        components : jnp.ndarray
            Principal components, shape (n_components, n_features)
        mean : jnp.ndarray
            Feature means, shape (n_features,)

        Returns
        -------
        jnp.ndarray
            Reconstructed data, shape (n_samples, n_features)
        """
        return scores @ components + mean

    @jit
    def pca_reconstruction_error_jax(
        X: jnp.ndarray,
        components: jnp.ndarray,
        mean: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        JAX-accelerated reconstruction error computation.

        Parameters
        ----------
        X : jnp.ndarray
            Original data, shape (n_samples, n_features)
        components : jnp.ndarray
            Principal components, shape (n_components, n_features)
        mean : jnp.ndarray
            Feature means, shape (n_features,)

        Returns
        -------
        jnp.ndarray
            Per-sample MSE, shape (n_samples,)
        """
        scores = pca_transform_jax(X, components, mean)
        reconstructed = pca_inverse_transform_jax(scores, components, mean)
        residuals = X - reconstructed
        return jnp.mean(residuals**2, axis=1)

else:
    # Stubs when JAX is not available
    def pca_transform_jax(*args, **kwargs):
        raise ImportError("JAX not available")

    def pca_inverse_transform_jax(*args, **kwargs):
        raise ImportError("JAX not available")

    def pca_reconstruction_error_jax(*args, **kwargs):
        raise ImportError("JAX not available")


# --- Convenience functions ---


def fit_pca(
    spectra: np.ndarray,
    n_components: Optional[Union[int, float]] = None,
    use_jax: bool = False,
) -> PCAResult:
    """
    Fit PCA to spectral data (convenience function).

    Parameters
    ----------
    spectra : np.ndarray
        Spectral data, shape (n_spectra, n_wavelengths)
    n_components : int, float, or None
        Number of components or variance fraction to retain
    use_jax : bool
        Whether to use JAX backend (default: False)

    Returns
    -------
    PCAResult
        Fitted PCA model

    Examples
    --------
    >>> result = fit_pca(spectra, n_components=10)
    >>> scores = result.transform(spectra)
    """
    pca = PCAPipeline(n_components=n_components, use_jax=use_jax)
    return pca.fit(spectra)


def denoise_spectra(
    spectra: np.ndarray,
    n_components: int = 10,
    use_jax: bool = False,
) -> Tuple[np.ndarray, PCAResult]:
    """
    Denoise spectra using PCA reconstruction.

    Projects spectra onto top principal components and reconstructs,
    effectively filtering out noise in the discarded components.

    Parameters
    ----------
    spectra : np.ndarray
        Spectral data, shape (n_spectra, n_wavelengths)
    n_components : int
        Number of components to retain (default: 10)
    use_jax : bool
        Whether to use JAX backend (default: False)

    Returns
    -------
    denoised : np.ndarray
        Denoised spectra, shape (n_spectra, n_wavelengths)
    result : PCAResult
        Fitted PCA model for inspection

    Examples
    --------
    >>> denoised, pca_result = denoise_spectra(noisy_spectra, n_components=5)
    >>> print(f"Retained {pca_result.cumulative_variance_ratio[-1]:.1%} of variance")
    """
    pca = PCAPipeline(n_components=n_components, use_jax=use_jax)
    result = pca.fit(spectra)
    scores = result.transform(spectra)
    denoised = result.inverse_transform(scores)

    return denoised, result


def explained_variance_curve(
    spectra: np.ndarray,
    max_components: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute explained variance curve for component selection.

    Useful for scree plots and determining optimal number of components.

    Parameters
    ----------
    spectra : np.ndarray
        Spectral data, shape (n_spectra, n_wavelengths)
    max_components : int, optional
        Maximum number of components to compute
        (default: min(n_spectra, n_wavelengths))

    Returns
    -------
    cumulative_variance : np.ndarray
        Cumulative explained variance ratio
    individual_variance : np.ndarray
        Per-component explained variance ratio

    Examples
    --------
    >>> cumvar, var = explained_variance_curve(spectra)
    >>> # Find elbow point
    >>> diffs = np.diff(cumvar)
    >>> elbow = np.argmax(diffs < 0.01) + 1  # Stop when gain < 1%
    """
    pca = PCAPipeline(n_components=max_components, use_jax=False)
    result = pca.fit(spectra)

    return result.cumulative_variance_ratio, result.explained_variance_ratio
