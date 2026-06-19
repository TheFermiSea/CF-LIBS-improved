"""
Boltzmann plot generation and fitting for CF-LIBS.

This module provides robust Boltzmann plot fitting with multiple outlier rejection
methods based on literature recommendations:

- **Sigma-clipping**: Iterative rejection based on residual standard deviation
- **RANSAC**: Random Sample Consensus for robust fitting with gross outliers
- **Huber**: M-estimation with Huber loss function for moderate outliers

References:
    - Huber, P.J. (1981): Robust Statistics
"""

from __future__ import annotations

import numpy as np
from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger

# Re-export data structures from their canonical location for backward compatibility.
# All new code should import these from cflibs.inversion.common.data_structures.
from cflibs.inversion.common.data_structures import (  # noqa: F401
    BoltzmannFitResult,
    FitMethod,
    LineObservation,
)

logger = get_logger("inversion.boltzmann")


try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = object  # Dummy for type hinting if missing


class BoltzmannPlotFitter:
    """
    Fitter for Boltzmann plots to determine excitation temperature.

    Supports multiple robust fitting methods:

    - **sigma_clip** (default): Iterative sigma-clipping of outliers
    - **ransac**: Random Sample Consensus for gross outlier rejection
    - **huber**: M-estimation with Huber loss function

    The inner linear-fit kernel of the sigma-clip loop defaults to weighted
    ordinary least squares (``numpy.polyfit``). Passing ``use_odr=True`` swaps
    it for an **opt-in** weighted orthogonal distance regression (errors-in-
    variables / total least squares via :mod:`scipy.odr`) that accounts for
    uncertainty on both axes — ``y = ln(I lambda / g A)`` and ``x = E_k`` — and
    removes the regression-dilution bias on the slope (hence temperature). The
    default remains weighted OLS; with zero x-noise the ODR result is
    ``allclose`` to OLS.

    Examples
    --------
    >>> fitter = BoltzmannPlotFitter(method=FitMethod.RANSAC)
    >>> result = fitter.fit(observations)
    >>> print(f"Temperature: {result.temperature_K:.0f} K")
    """

    def __init__(
        self,
        outlier_sigma: float = 2.5,
        max_iterations: int = 10,
        method: FitMethod = FitMethod.SIGMA_CLIP,
        ransac_min_samples: int = 2,
        ransac_residual_threshold: float | None = None,
        ransac_max_trials: int = 100,
        huber_epsilon: float = 1.2,
        use_jax: bool = False,
        use_odr: bool = False,
        odr_x_uncertainty: float = 0.0,
    ):
        """
        Initialize fitter.

        Parameters
        ----------
        outlier_sigma : float
            Sigma threshold for outlier rejection (sigma_clip method).
            Default: 2.5 (stricter than legacy 3.0 for mineral matrices).
        max_iterations : int
            Maximum iterations for sigma clipping. Default: 10.
        method : FitMethod
            Fitting method to use (SIGMA_CLIP, RANSAC, or HUBER)
        ransac_min_samples : int
            Minimum number of samples for RANSAC model fitting (default: 2)
        ransac_residual_threshold : float, optional
            Maximum residual for a point to be considered an inlier in RANSAC.
            If None, uses median absolute deviation of residuals.
        ransac_max_trials : int
            Maximum RANSAC iterations (default: 100)
        huber_epsilon : float
            Huber loss transition point. Points with standardized residuals
            below this use squared loss; above use linear loss. Default: 1.2
            (stricter than the legacy 1.35 setting).
        use_jax : bool
            When True and ``method == FitMethod.SIGMA_CLIP``, route the
            inner weighted-least-squares step through the JAX kernel
            :func:`cflibs.inversion.physics.boltzmann_jax.batched_boltzmann_fit`
            instead of :func:`numpy.polyfit`. Default ``False`` preserves
            byte-for-byte the existing CPU behavior. RANSAC and Huber
            methods always use the CPU path (those are non-bottleneck
            paths in the composition workflows). See
            ``docs/jax-port/iterative-boltzmann-consultation.md`` for
            rationale. Opt-in only; the call sites in
            ``cflibs/inversion/solve/iterative.py`` and
            ``cflibs/inversion/runtime/streaming.py`` flip this from the
            ``CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION=1`` env var.
        use_odr : bool
            **Opt-in** errors-in-variables fit. When True, the slope/intercept
            of the (already inlier-selected) Boltzmann plot are estimated by
            weighted orthogonal distance regression (ODR / total least squares)
            via :mod:`scipy.odr` instead of the default weighted ordinary
            least squares (``numpy.polyfit``). ODR accounts for uncertainty on
            **both** axes — ``y = ln(I lambda / g A)`` (intensity + A_ki) and
            ``x = E_k`` — and removes the regression-dilution (attenuation)
            bias that biases the OLS slope, hence the temperature, when the
            upper-level energies carry non-negligible error
            (Boggs & Rodgers 1990; Aragón & Aguilera 2008). Default ``False``
            preserves the existing weighted-OLS behavior **exactly**. The
            outlier-rejection scheme is unchanged — ODR only replaces the
            inner linear-fit kernel of the sigma-clip loop. With zero x-noise
            ODR reduces to OLS (``allclose``).
        odr_x_uncertainty : float
            Default 1-sigma uncertainty (in eV) on the upper-level energy
            ``E_k`` used by the ODR fit when a per-line value is unavailable.
            Per-line x-uncertainties take precedence and are read from an
            optional ``E_k_uncertainty`` attribute on each
            :class:`LineObservation` (via ``getattr``, so the dataclass need
            not define it). ``0.0`` means "no per-axis x-error supplied for
            this line" and the fit then falls back to this scalar; if it too
            is ``0.0`` ODR degenerates to OLS (the zero-x-noise limit). Only
            consulted when ``use_odr=True``. Default ``0.0``.
        """
        self.outlier_sigma = outlier_sigma
        self.max_iterations = max_iterations
        self.method = method
        self.ransac_min_samples = ransac_min_samples
        self.ransac_residual_threshold = ransac_residual_threshold
        self.ransac_max_trials = ransac_max_trials
        self.huber_epsilon = huber_epsilon
        self.use_jax = use_jax
        self.use_odr = use_odr
        self.odr_x_uncertainty = odr_x_uncertainty

    def fit(
        self,
        observations: list[LineObservation],
        aki_uncertainty_weighting: bool = True,
        multiplet_groups: list[str | int | None] | None = None,
    ) -> BoltzmannFitResult:
        """
        Perform robust linear regression on Boltzmann plot data.

        The fitting method is determined by the `method` parameter set during
        initialization. All methods return consistent results with outlier
        identification.

        Parameters
        ----------
        observations : list[LineObservation]
            List of line observations
        aki_uncertainty_weighting : bool, optional
            When True (default), inflate each per-line y-uncertainty by the
            atomic-data contribution sigma_y(A_ki) = sigma(A_ki)/A_ki, combined
            in quadrature with the intensity term. This down-weights NIST C/D
            graded transitions in the inverse-variance LSQ so that poorly
            characterized lines do not dominate the fitted slope. Lines whose
            ``aki_uncertainty`` is ``None`` keep their original intensity-only
            sigma_y. Set to False to reproduce the legacy intensity-only fit
            exactly.
        multiplet_groups : list[str | int | None], optional
            If provided, must be same length as observations. Observations with the
            same non-None group ID are aggregated (summing gA and intensity) into a
            single multiplet point before fitting, following Wakil et al. (2023).
            Observations assigned ``None`` as their group ID are treated as
            independent lines and are never aggregated.

        Returns
        -------
        BoltzmannFitResult
            Fit results including temperature, uncertainties, and outlier info.
            When *multiplet_groups* is provided, ``rejected_points`` and
            ``inlier_mask`` are expressed in terms of original observation indices
            (not aggregated-point indices) so that downstream consumers such as
            ``plot()`` can be used without special-casing.

        Raises
        ------
        ValueError
            If fewer than 2 valid observations (or aggregated points) are available
        """
        if len(observations) < 2:
            raise ValueError("Need at least 2 points for a fit")

        x_all, y_all, y_err_all, agg_to_obs_indices = self._prepare_fit_arrays(
            observations, aki_uncertainty_weighting, multiplet_groups
        )

        # Handle cases where y calculation failed (e.g. negative intensity)
        valid_mask = np.isfinite(y_all)
        if not np.all(valid_mask):
            logger.warning(f"Excluding {np.sum(~valid_mask)} points with invalid Y values")

        # Ensure at least 2 valid (aggregated) points remain before fitting.
        # Multiplet aggregation can reduce the effective point count below 2 even
        # when len(observations) >= 2.
        if int(np.sum(valid_mask)) < 2:
            raise ValueError(
                f"Need at least 2 valid points for a fit; only {int(np.sum(valid_mask))} "
                "remain after multiplet aggregation and invalid-y filtering."
            )

        # Route to appropriate fitting method
        if self.method == FitMethod.RANSAC:
            result = self._fit_ransac(x_all, y_all, y_err_all, valid_mask)
        elif self.method == FitMethod.HUBER:
            result = self._fit_huber(x_all, y_all, y_err_all, valid_mask)
        elif self.use_odr:  # SIGMA_CLIP loop, ODR inner kernel (opt-in EIV fit)
            x_err_all = self._build_sigma_x(observations, agg_to_obs_indices)
            result = self._fit_sigma_clip_odr(x_all, y_all, y_err_all, x_err_all, valid_mask)
        elif self.use_jax:  # SIGMA_CLIP + JAX kernel for inner WLS step
            result = self._fit_sigma_clip_jax(x_all, y_all, y_err_all, valid_mask)
        else:  # SIGMA_CLIP (default, CPU)
            result = self._fit_sigma_clip(x_all, y_all, y_err_all, valid_mask)

        # When multiplet_groups was provided, the fit operated on the aggregated
        # array.  Translate rejected_points and inlier_mask back to original
        # observation indices so downstream consumers (plot(), widgets) can use
        # them against the observations list without special-casing.
        if agg_to_obs_indices is not None:
            result = self._translate_multiplet_indices(
                result, agg_to_obs_indices, len(observations)
            )

        return result

    def _prepare_fit_arrays(
        self,
        observations: list[LineObservation],
        aki_uncertainty_weighting: bool,
        multiplet_groups: list[str | int | None] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[int]] | None]:
        """Build fit arrays, optionally aggregating unresolved multiplet groups."""
        if multiplet_groups is None:
            return (
                np.array([obs.E_k_ev for obs in observations]),
                np.array([obs.y_value for obs in observations]),
                self._build_sigma_y(observations, aki_uncertainty_weighting),
                None,
            )

        if len(multiplet_groups) != len(observations):
            raise ValueError("multiplet_groups must have same length as observations")

        groups = self._group_multiplet_indices(multiplet_groups)
        x_agg: list[float] = []
        y_agg: list[float] = []
        y_err_agg: list[float] = []
        agg_to_obs_indices: list[list[int]] = []

        for gid, idxs in groups:
            if gid is None or len(idxs) == 1:
                self._append_independent_observations(
                    observations,
                    idxs,
                    aki_uncertainty_weighting,
                    x_agg,
                    y_agg,
                    y_err_agg,
                    agg_to_obs_indices,
                )
                continue

            x_val, y_val, y_err = self._aggregate_multiplet_observations(
                observations, idxs, aki_uncertainty_weighting
            )
            x_agg.append(x_val)
            y_agg.append(y_val)
            y_err_agg.append(y_err)
            agg_to_obs_indices.append(list(idxs))

        return (
            np.array(x_agg),
            np.array(y_agg),
            np.array(y_err_agg),
            agg_to_obs_indices,
        )

    @staticmethod
    def _group_multiplet_indices(
        multiplet_groups: list[str | int | None],
    ) -> list[tuple[str | int | None, list[int]]]:
        """Group observation indices in first-seen order."""
        groups: dict[str | int | None, list[int]] = {}
        group_order: list[str | int | None] = []
        for idx, gid in enumerate(multiplet_groups):
            if gid not in groups:
                groups[gid] = []
                group_order.append(gid)
            groups[gid].append(idx)
        return [(gid, groups[gid]) for gid in group_order]

    def _append_independent_observations(
        self,
        observations: list[LineObservation],
        indices: list[int],
        aki_uncertainty_weighting: bool,
        x_agg: list[float],
        y_agg: list[float],
        y_err_agg: list[float],
        agg_to_obs_indices: list[list[int]],
    ) -> None:
        """Append non-aggregated observations to the fit arrays."""
        single_obs = [observations[idx] for idx in indices]
        single_sigma = self._build_sigma_y(single_obs, aki_uncertainty_weighting)
        for sigma_idx, obs_idx in enumerate(indices):
            obs = observations[obs_idx]
            x_agg.append(obs.E_k_ev)
            y_agg.append(obs.y_value)
            y_err_agg.append(float(single_sigma[sigma_idx]))
            agg_to_obs_indices.append([obs_idx])

    @staticmethod
    def _aggregate_multiplet_observations(
        observations: list[LineObservation],
        indices: list[int],
        aki_uncertainty_weighting: bool,
    ) -> tuple[float, float, float]:
        """Return one Wakil-style aggregate point for a multiplet group."""
        sum_ga = 0.0
        sum_i_lam = 0.0
        sum_e_ga = 0.0
        var_i_lam = 0.0
        var_ga = 0.0

        for idx in indices:
            obs = observations[idx]
            ga = obs.g_k * obs.A_ki
            if ga <= 0.0:
                continue
            i_lam = ga * np.exp(obs.y_value)
            sum_ga += ga
            sum_i_lam += i_lam
            sum_e_ga += obs.E_k_ev * ga
            var_i_lam += (i_lam * obs.y_uncertainty) ** 2

            unc = obs.aki_uncertainty
            if aki_uncertainty_weighting and unc is not None and np.isfinite(unc) and unc > 0:
                var_ga += (ga * unc) ** 2

        if sum_ga <= 0.0 or sum_i_lam <= 0.0:
            return np.nan, np.nan, np.nan

        rel_var_i = var_i_lam / (sum_i_lam**2)
        rel_var_ga = var_ga / (sum_ga**2)
        return sum_e_ga / sum_ga, np.log(sum_i_lam / sum_ga), np.sqrt(rel_var_i + rel_var_ga)

    def _fit_sigma_clip(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        y_err_all: np.ndarray,
        valid_mask: np.ndarray,
    ) -> BoltzmannFitResult:
        """Iterative sigma-clipping fit."""
        indices = np.arange(len(x_all))
        mask = valid_mask.copy()

        slope = 0.0
        intercept = 0.0
        slope_err = 0.0
        intercept_err = 0.0
        r_squared = 0.0
        n_iterations = 0
        covariance_matrix = None

        for iteration in range(self.max_iterations):
            n_iterations = iteration + 1
            x = x_all[mask]
            y = y_all[mask]
            y_err = y_err_all[mask]

            if len(x) < 2:
                logger.warning("Too few points remaining after rejection")
                break

            # Weighted least squares. ``weights`` are 1/sigma^2; polyfit wants
            # 1/sigma (it squares internally), so pass sqrt. See _compute_weights.
            weights = self._compute_weights(y_err)
            polyfit_w = np.sqrt(weights)

            try:
                # Need >2 points for covariance estimation
                if len(x) > 2:
                    (m, c), cov = np.polyfit(x, y, 1, w=polyfit_w, cov=True)
                    slope_err = np.sqrt(cov[0, 0])
                    intercept_err = np.sqrt(cov[1, 1])
                    covariance_matrix = cov
                else:
                    m, c = np.polyfit(x, y, 1, w=polyfit_w)
                    slope_err = np.inf
                    intercept_err = np.inf
                    covariance_matrix = None
            except np.linalg.LinAlgError:
                logger.error("Linear regression failed")
                return self._empty_result()

            slope = m
            intercept = c

            # Calculate R^2

            y_pred = m * x + c
            r_squared = self._compute_r_squared(y, y_pred, weights)

            # Check for outliers

            residuals = y - y_pred
            std_res = np.std(residuals)
            if std_res == 0:
                break

            bad_indices = np.abs(residuals) > self.outlier_sigma * std_res

            if not np.any(bad_indices):
                break

            current_indices = indices[mask]
            outlier_global_indices = current_indices[bad_indices]
            mask[outlier_global_indices] = False

            logger.debug(f"Iteration {iteration}: Rejected {len(outlier_global_indices)} outliers")

        return self._create_result(
            slope,
            slope_err,
            intercept,
            intercept_err,
            r_squared,
            mask,
            indices,
            "sigma_clip",
            n_iterations,
            covariance_matrix,
        )

    def _fit_sigma_clip_jax(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        y_err_all: np.ndarray,
        valid_mask: np.ndarray,
    ) -> BoltzmannFitResult:
        """JAX-accelerated iterative sigma-clip WLS fit.

        Algorithmically equivalent to :meth:`_fit_sigma_clip` — same
        weighted-LS normal equations (algebraically), same iterative
        residual-sigma outlier rejection, same fit_method label
        ``"sigma_clip"``. Only the inner per-iteration solve is delegated
        to :func:`cflibs.inversion.physics.boltzmann_jax.batched_boltzmann_fit`
        (a single-element "batch" of one), which avoids the ~10
        ``scipy.stats.linregress`` / ``np.polyfit`` round-trips that
        dominate Vrabel-50k wall time.

        Numerical agreement vs CPU path: ``rtol ~1e-8`` on slope /
        intercept / temperature on well-conditioned synthetic inputs;
        ``inlier_mask`` matches exactly when no residual sits within
        machine-epsilon of the rejection boundary. See
        ``tests/inversion/physics/test_boltzmann_jax_composition.py``.
        """
        # Local import — keeps JAX optional for the CPU-only path.
        from cflibs.inversion.physics.boltzmann_jax import (
            HAS_JAX,
            batched_boltzmann_fit,
        )

        if not HAS_JAX:
            logger.warning("use_jax=True but JAX is not installed; falling back to CPU sigma_clip.")
            return self._fit_sigma_clip(x_all, y_all, y_err_all, valid_mask)

        import jax.numpy as jnp

        indices = np.arange(len(x_all))
        mask = valid_mask.copy()

        slope = 0.0
        intercept = 0.0
        slope_err = 0.0
        intercept_err = 0.0
        r_squared = 0.0
        n_iterations = 0
        covariance_matrix: np.ndarray | None = None

        for iteration in range(self.max_iterations):
            n_iterations = iteration + 1
            x = x_all[mask]
            y = y_all[mask]
            y_err = y_err_all[mask]

            if len(x) < 2:
                logger.warning("Too few points remaining after rejection")
                break

            weights = self._compute_weights(y_err)

            # WEIGHTING CONVENTION (corrected): the inverse-variance weights
            # ``1/sigma^2`` are exactly the weight on the *squared* residual in
            # WLS. The closed-form 5-sum kernel minimises ``sum(w * r^2)``, so
            # it consumes these weights DIRECTLY (no squaring). The CPU
            # sigma-clip path achieves the same objective by passing
            # ``sqrt(weights) = 1/sigma`` to ``numpy.polyfit`` (which squares
            # its ``w`` internally). Both paths therefore minimise
            # ``sum((1/sigma^2) r^2)`` and agree. (Previously this squared the
            # weights to ``1/sigma^4`` to mirror a CPU-side polyfit bug; that
            # bug is now fixed — see _compute_weights.)
            kernel_weights = weights

            # Closed-form WLS via JAX kernel. Pack as a (1, N) batch-of-one.
            # NOTE: the kernel applies its own ``mask`` over weights, but we
            # already filtered to inliers above so the mask is all-True here.
            x_jax = jnp.asarray(x[None, :], dtype=jnp.float64)
            y_jax = jnp.asarray(y[None, :], dtype=jnp.float64)
            w_jax = jnp.asarray(kernel_weights[None, :], dtype=jnp.float64)
            m_jax = jnp.ones_like(x_jax, dtype=bool)

            kernel_result = batched_boltzmann_fit(x_jax, y_jax, w_jax, m_jax)

            m = float(kernel_result.slope[0])
            c = float(kernel_result.intercept[0])

            # NaN/inf from a degenerate solve (det≈0) -> mirror polyfit's
            # LinAlgError failure path on the CPU side.
            if not (np.isfinite(m) and np.isfinite(c)):
                logger.error("Linear regression failed (JAX kernel returned non-finite)")
                return self._empty_result()

            slope_err, intercept_err, covariance_matrix = self._jax_scaled_covariance(
                x, y, m, c, kernel_weights, kernel_result
            )

            slope = m
            intercept = c

            # R² — recompute on the host so we use the **same**
            # effective-weights convention as the CPU path
            # (``self._compute_r_squared`` consumes the polyfit-flavor
            # ``weights`` directly, not the squared ``kernel_weights``).
            # The kernel's own ``R_squared`` field is computed against
            # ``kernel_weights`` and would therefore disagree.
            y_pred = m * x + c
            r_squared = self._compute_r_squared(y, y_pred, weights)

            if self._jax_reject_outliers(x, y, m, c, mask, indices, iteration):
                break

        return self._create_result(
            slope,
            slope_err,
            intercept,
            intercept_err,
            r_squared,
            mask,
            indices,
            "sigma_clip",
            n_iterations,
            covariance_matrix,
        )

    def _jax_scaled_covariance(
        self,
        x: np.ndarray,
        y: np.ndarray,
        m: float,
        c: float,
        kernel_weights: np.ndarray,
        kernel_result: object,
    ) -> tuple[float, float, np.ndarray | None]:
        """Slope/intercept errors + covariance for the JAX sigma-clip path.

        Mirrors ``numpy.polyfit(..., cov=True)``'s chi^2/dof-scaled covariance
        using the same squared-residual weights as the corrected CPU path.
        Returns ``(slope_err, intercept_err, covariance_matrix)``.
        """
        if len(x) <= 2:
            return np.inf, np.inf, None

        # numpy.polyfit(..., cov=True) scales its returned cov by
        # chi^2/dof (the ``scale_cov=True`` default in older
        # polyfit). We mirror that here using the **same**
        # squared-residual weight as the corrected CPU path:
        # ``kernel_weights == weights == 1/sigma^2``. The JAX kernel
        # returns the *unscaled* formal cov (sigma_slope=sqrt(S_w/det));
        # we rescale by chi^2/dof so r_jax.slope_uncertainty matches
        # r_cpu.slope_uncertainty.
        w_masked = kernel_weights
        S_w = float(np.sum(w_masked))
        S_wx = float(np.sum(w_masked * x))
        S_wxx = float(np.sum(w_masked * x * x))
        det = S_w * S_wxx - S_wx * S_wx
        if abs(det) <= 1e-30:
            return (
                float(kernel_result.sigma_slope[0]),  # type: ignore[attr-defined]
                float(kernel_result.sigma_intercept[0]),  # type: ignore[attr-defined]
                None,
            )

        # Chi-squared per degree of freedom from the current
        # in-mask fit.
        y_pred_now = m * x + c
        resid_now = y - y_pred_now
        chi2 = float(np.sum(w_masked * resid_now * resid_now))
        dof = len(x) - 2
        chi2_dof = chi2 / dof if dof > 0 else 1.0

        var_slope = (S_w / det) * chi2_dof
        var_intercept = (S_wxx / det) * chi2_dof
        cov_si = (-S_wx / det) * chi2_dof
        slope_err = float(np.sqrt(max(var_slope, 0.0)))
        intercept_err = float(np.sqrt(max(var_intercept, 0.0)))
        covariance_matrix = np.array([[var_slope, cov_si], [cov_si, var_intercept]])
        return slope_err, intercept_err, covariance_matrix

    def _jax_reject_outliers(
        self,
        x: np.ndarray,
        y: np.ndarray,
        m: float,
        c: float,
        mask: np.ndarray,
        indices: np.ndarray,
        iteration: int,
    ) -> bool:
        """Reject sigma-clip outliers in-place on ``mask``; return stop flag.

        Identical predicate to the CPU path (computed in numpy, not JAX, so
        behavior is bit-exact with the CPU sigma_clip when residuals are
        well-separated). Returns ``True`` when the iteration loop should stop
        (zero residual spread or no remaining outliers).
        """
        y_pred = m * x + c
        residuals = y - y_pred
        std_res = float(np.std(residuals))
        if std_res == 0:
            return True

        bad_indices = np.abs(residuals) > self.outlier_sigma * std_res

        if not np.any(bad_indices):
            return True

        current_indices = indices[mask]
        outlier_global_indices = current_indices[bad_indices]
        mask[outlier_global_indices] = False

        logger.debug(
            "Iteration %d (JAX): Rejected %d outliers",
            iteration,
            len(outlier_global_indices),
        )
        return False

    def _fit_sigma_clip_odr(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        y_err_all: np.ndarray,
        x_err_all: np.ndarray,
        valid_mask: np.ndarray,
    ) -> BoltzmannFitResult:
        """Sigma-clip Boltzmann fit with an orthogonal-distance-regression kernel.

        Errors-in-variables (total least squares) replacement for the inner
        linear fit of :meth:`_fit_sigma_clip`. Identical iterative
        residual-sigma outlier rejection (so the inlier set is selected the
        same way), but the slope/intercept at each iteration come from
        weighted orthogonal distance regression rather than weighted OLS.

        Physics / statistics
        --------------------
        The Boltzmann/Saha-Boltzmann plot regresses ``y = ln(I lambda / g A)``
        on ``x = E_k`` with the slope ``m = -1 / (k_B T)``. Ordinary least
        squares assumes the predictor ``x`` is error-free; when ``E_k`` carries
        non-negligible uncertainty the OLS slope is *attenuated* toward zero
        (the classic regression-dilution / errors-in-variables bias), which
        systematically *overestimates* ``T`` because ``T = -1/(k_B m)`` and
        ``|m|`` is biased low. Orthogonal distance regression minimises the
        sum of squared distances measured orthogonally in the
        error-scaled metric ``sum_i [ (x_i - x*_i)^2 / sigma_x_i^2 +
        (y_i - (m x*_i + c))^2 / sigma_y_i^2 ]`` over both the line parameters
        and the latent true abscissae ``x*_i``, removing that bias to first
        order (Boggs & Rodgers 1990). In the limit ``sigma_x -> 0`` the
        orthogonal metric collapses to the vertical one and ODR reduces to
        weighted OLS, so this path is ``allclose`` to the default fit on data
        with no x-noise.

        References
        ----------
        Boggs, P.T. & Rodgers, J.E. (1990), "Orthogonal Distance Regression",
        Contemporary Mathematics 112, 183-194 (the ODRPACK algorithm wrapped
        by :mod:`scipy.odr`).
        Aragón, C. & Aguilera, J.A. (2008), "Characterization of laser induced
        plasmas by optical emission spectroscopy: A review of experiments and
        methods", Spectrochim. Acta B 63, 893-916 (Saha-Boltzmann plot and the
        role of E_k uncertainties in plasma-parameter retrieval).
        """
        indices = np.arange(len(x_all))
        mask = valid_mask.copy()

        slope = 0.0
        intercept = 0.0
        slope_err = 0.0
        intercept_err = 0.0
        r_squared = 0.0
        n_iterations = 0
        covariance_matrix: np.ndarray | None = None

        for iteration in range(self.max_iterations):
            n_iterations = iteration + 1
            x = x_all[mask]
            y = y_all[mask]
            y_err = y_err_all[mask]
            x_err = x_err_all[mask]

            if len(x) < 2:
                logger.warning("Too few points remaining after rejection")
                break

            fit = self._odr_linear_fit(x, y, x_err, y_err)
            if fit is None:
                logger.error("Orthogonal distance regression failed")
                return self._empty_result()
            m, c, slope_err, intercept_err, covariance_matrix = fit

            slope = m
            intercept = c

            # R^2 uses the same inverse-variance y-weights convention as the
            # CPU OLS path so the reported goodness-of-fit is comparable.
            weights = self._compute_weights(y_err)
            y_pred = m * x + c
            r_squared = self._compute_r_squared(y, y_pred, weights)

            # Outlier rejection: identical predicate to the OLS sigma-clip
            # path (vertical residuals vs. residual std), so the inlier set is
            # selected consistently across fit kernels.
            residuals = y - y_pred
            std_res = float(np.std(residuals))
            if std_res == 0:
                break

            bad_indices = np.abs(residuals) > self.outlier_sigma * std_res
            if not np.any(bad_indices):
                break

            current_indices = indices[mask]
            mask[current_indices[bad_indices]] = False
            logger.debug(
                "Iteration %d (ODR): Rejected %d outliers", iteration, int(np.sum(bad_indices))
            )

        return self._create_result(
            slope,
            slope_err,
            intercept,
            intercept_err,
            r_squared,
            mask,
            indices,
            "sigma_clip_odr",
            n_iterations,
            covariance_matrix,
        )

    def _odr_linear_fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        x_err: np.ndarray,
        y_err: np.ndarray,
    ) -> tuple[float, float, float, float, np.ndarray | None] | None:
        """Weighted orthogonal distance regression of ``y`` on ``x``.

        Returns ``(slope, intercept, slope_err, intercept_err, cov)`` where
        ``cov`` is the 2x2 covariance matrix ``[[var(m), cov(m,c)],
        [cov(m,c), var(c)]]``, scaled by the residual variance so it matches
        the chi^2/dof scaling that :func:`numpy.polyfit` applies on the OLS
        path. Returns ``None`` on solver failure (mirrors the OLS
        ``LinAlgError`` path so the caller can emit an empty result).

        Parameters
        ----------
        x_err, y_err : np.ndarray
            1-sigma uncertainties on the abscissae (``E_k``) and ordinates
            (``ln(I lambda / g A)``). Non-positive or non-finite entries are
            replaced by a small floor so ODR's reciprocal weights stay finite
            (a zero ``sigma_x`` for every point recovers the OLS limit).
        """
        # Local, lazily-imported so the default path never touches scipy.odr.
        # scipy.odr is deprecation-flagged (SciPy >=1.17) but still functional
        # and is the physics-only ODRPACK wrapper this method is specified to
        # use; suppress that one warning at the call site only.
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import scipy.odr as odr

        # Initial guess from weighted OLS (same kernel as the default path).
        polyfit_w = np.sqrt(self._compute_weights(y_err))
        try:
            m0, c0 = np.polyfit(x, y, 1, w=polyfit_w)
        except np.linalg.LinAlgError:
            return None

        # ODR consumes per-point standard deviations directly. Floor any
        # zero/invalid sigma so the implied weight 1/sigma^2 stays finite; a
        # vanishing sigma_x then makes the orthogonal metric collapse to the
        # vertical one (OLS limit).
        sx = self._sanitize_sigma(x_err, x)
        sy = self._sanitize_sigma(y_err, y)

        data = odr.RealData(x, y, sx=sx, sy=sy)
        model = odr.Model(lambda beta, xv: beta[0] * xv + beta[1])
        try:
            out = odr.ODR(data, model, beta0=[float(m0), float(c0)]).run()
        except Exception as exc:  # pragma: no cover - solver-internal failure
            logger.error("scipy.odr ODR.run() failed: %s", exc)
            return None

        m, c = float(out.beta[0]), float(out.beta[1])
        if not (np.isfinite(m) and np.isfinite(c)):
            return None

        # out.cov_beta is the *unscaled* parameter covariance; out.res_var is
        # the residual variance (chi^2/dof). The product is the scaled
        # covariance, matching numpy.polyfit(..., cov=True) so downstream
        # uncertainty propagation is convention-consistent. out.sd_beta is
        # already sqrt(diag(scaled cov)).
        if len(x) > 2 and out.cov_beta is not None:
            cov = np.asarray(out.cov_beta, dtype=float) * float(out.res_var)
            slope_err = float(out.sd_beta[0])
            intercept_err = float(out.sd_beta[1])
        else:
            cov = None
            slope_err = float("inf")
            intercept_err = float("inf")

        return m, c, slope_err, intercept_err, cov

    @staticmethod
    def _sanitize_sigma(sigma: np.ndarray, values: np.ndarray) -> np.ndarray:
        """Replace non-positive/non-finite sigmas with a tiny relative floor.

        A vanishing ``sigma_x`` is physically meaningful (errorless abscissa,
        the OLS limit) but ODR forms ``1/sigma`` weights internally, so an
        exact zero would be non-finite. We substitute a floor proportional to
        the data scale so the corresponding axis is treated as effectively
        error-free without overflowing the weight.
        """
        sigma = np.asarray(sigma, dtype=float)
        scale = float(np.max(np.abs(values))) if values.size else 1.0
        floor = max(scale, 1.0) * 1e-12
        out = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, floor)
        return out

    def _build_sigma_x(
        self,
        observations: list[LineObservation],
        agg_to_obs_indices: list[list[int]] | None,
    ) -> np.ndarray:
        """Per-point 1-sigma uncertainty on ``E_k`` aligned with the fit array.

        For non-aggregated fits each entry is the per-line value taken from an
        optional ``E_k_uncertainty`` attribute on the observation (read via
        ``getattr`` so the :class:`LineObservation` dataclass need not declare
        it), falling back to the scalar ``self.odr_x_uncertainty`` when a line
        provides no value. For multiplet-aggregated points the constituent
        ``E_k`` values are collapsed to a gA-weighted mean upstream; we
        conservatively assign the scalar ``self.odr_x_uncertainty`` to those
        aggregated abscissae rather than attempting to propagate the per-line
        x-errors through the weighting.
        """

        def per_obs_sigma(obs: LineObservation) -> float:
            val = getattr(obs, "E_k_uncertainty", None)
            if val is not None and np.isfinite(val) and val > 0.0:
                return float(val)
            return float(self.odr_x_uncertainty)

        if agg_to_obs_indices is None:
            return np.array([per_obs_sigma(obs) for obs in observations])

        sigma_x: list[float] = []
        for obs_idxs in agg_to_obs_indices:
            if len(obs_idxs) == 1:
                sigma_x.append(per_obs_sigma(observations[obs_idxs[0]]))
            else:
                sigma_x.append(float(self.odr_x_uncertainty))
        return np.array(sigma_x)

    def _fit_ransac(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        y_err_all: np.ndarray,
        valid_mask: np.ndarray,
    ) -> BoltzmannFitResult:
        """
        RANSAC (Random Sample Consensus) robust fitting.

        RANSAC is effective for data with gross outliers (self-absorbed lines,
        misidentified lines) that would severely bias least squares.
        """
        indices = np.arange(len(x_all))
        mask = valid_mask.copy()

        x = x_all[mask]
        y = y_all[mask]
        y_err = y_err_all[mask]
        n_valid = len(x)

        if n_valid < self.ransac_min_samples:
            logger.warning(f"Too few valid points ({n_valid}) for RANSAC")
            return self._empty_result()

        # Compute threshold if not specified
        threshold = self.ransac_residual_threshold
        if threshold is None:
            # Use median absolute deviation as robust scale estimate
            weights = self._compute_weights(y_err)
            # polyfit wants 1/sigma (= sqrt of inverse-variance weights).
            m_init, c_init = np.polyfit(x, y, 1, w=np.sqrt(weights))
            residuals_init = np.abs(y - (m_init * x + c_init))
            mad = np.median(residuals_init)
            # Use the instance outlier_sigma for consistency across methods
            threshold = self.outlier_sigma * mad * 1.4826  # Scale MAD to std dev estimate
            y_scale = max(float(np.max(np.abs(y))), 1.0)
            threshold = max(threshold, np.finfo(float).eps * y_scale)

        best_inliers = None
        best_n_inliers = 0

        rng = np.random.default_rng(42)  # Reproducible results

        for _ in range(self.ransac_max_trials):
            # Random sample
            sample_idx = rng.choice(n_valid, size=self.ransac_min_samples, replace=False)
            x_sample = x[sample_idx]
            y_sample = y[sample_idx]

            # Fit model to sample
            if len(np.unique(x_sample)) < 2:
                continue  # Degenerate sample

            try:
                m, c = np.polyfit(x_sample, y_sample, 1)
            except np.linalg.LinAlgError:
                continue

            # Count inliers
            residuals = np.abs(y - (m * x + c))
            inliers = residuals <= threshold
            n_inliers = np.sum(inliers)

            if n_inliers > best_n_inliers:
                best_n_inliers = n_inliers
                best_inliers = inliers

        if best_inliers is None or best_n_inliers < 2:
            logger.warning("RANSAC failed to find sufficient inliers")
            return self._empty_result()

        # Refit using all inliers with weights
        x_inliers = x[best_inliers]
        y_inliers = y[best_inliers]
        y_err_inliers = y_err[best_inliers]

        slope, intercept, slope_err, intercept_err, covariance_matrix = self._weighted_fit(
            x_inliers, y_inliers, y_err_inliers
        )

        # Compute R^2 on inliers
        weights = self._compute_weights(y_err_inliers)
        y_pred = slope * x_inliers + intercept
        r_squared = self._compute_r_squared(y_inliers, y_pred, weights)

        # Map inliers back to original indices
        valid_indices = indices[mask]
        final_mask = np.zeros_like(mask)
        final_mask[valid_indices[best_inliers]] = True

        return self._create_result(
            slope,
            slope_err,
            intercept,
            intercept_err,
            r_squared,
            final_mask,
            indices,
            "ransac",
            self.ransac_max_trials,
            covariance_matrix,
        )

    def _fit_huber(
        self,
        x_all: np.ndarray,
        y_all: np.ndarray,
        y_err_all: np.ndarray,
        valid_mask: np.ndarray,
    ) -> BoltzmannFitResult:
        """
        Huber M-estimation robust fitting.

        Uses iteratively reweighted least squares (IRLS) with Huber weights.
        More efficient than RANSAC for moderate outliers.
        """
        indices = np.arange(len(x_all))
        mask = valid_mask.copy()

        x = x_all[mask]
        y = y_all[mask]
        y_err = y_err_all[mask]

        if len(x) < 2:
            logger.warning("Too few valid points for Huber fit")
            return self._empty_result()

        # Initial weights from measurement uncertainties. ``base_weights`` and
        # ``combined_weights`` are squared-residual weights (1/sigma^2 times the
        # dimensionless Huber factor); polyfit wants their sqrt (= 1/sigma scale).
        base_weights = self._compute_weights(y_err)
        combined_weights = base_weights.copy()  # Initialize for case with no iterations

        # IRLS with Huber weights
        slope, intercept = np.polyfit(x, y, 1, w=np.sqrt(base_weights))
        n_iterations = 0

        for iteration in range(self.max_iterations):
            n_iterations = iteration + 1

            # Compute residuals and scale
            residuals = y - (slope * x + intercept)
            scale = np.median(np.abs(residuals)) * 1.4826  # MAD-based scale

            if scale < 1e-10:
                break  # Perfect fit

            # Standardized residuals
            u = residuals / scale

            # Huber weights: 1 for |u| <= epsilon, epsilon/|u| for |u| > epsilon
            huber_weights = np.where(
                np.abs(u) <= self.huber_epsilon, 1.0, self.huber_epsilon / np.abs(u)
            )

            # Combined weights
            combined_weights = base_weights * huber_weights

            # Refit
            try:
                new_slope, new_intercept = np.polyfit(x, y, 1, w=np.sqrt(combined_weights))
            except np.linalg.LinAlgError:
                break

            # Check convergence
            if abs(new_slope - slope) < 1e-8 * abs(slope) + 1e-12:
                slope, intercept = new_slope, new_intercept
                break

            slope, intercept = new_slope, new_intercept

        # Final fit with covariance (using final weights)
        covariance_matrix = None
        try:
            (slope, intercept), cov = np.polyfit(x, y, 1, w=np.sqrt(combined_weights), cov=True)
            slope_err = np.sqrt(cov[0, 0])
            intercept_err = np.sqrt(cov[1, 1])
            covariance_matrix = cov
        except np.linalg.LinAlgError:
            slope_err = np.inf
            intercept_err = np.inf

        # Identify outliers (points with very low Huber weight)
        residuals = y - (slope * x + intercept)
        scale = np.median(np.abs(residuals)) * 1.4826
        if scale > 0:
            u = np.abs(residuals / scale)
            inlier_points = u <= 3 * self.huber_epsilon  # Mark extreme outliers
        else:
            inlier_points = np.ones(len(x), dtype=bool)

        # Compute R^2 on all points
        y_pred = slope * x + intercept
        r_squared = self._compute_r_squared(y, y_pred, base_weights)

        # Map to original indices
        valid_indices = indices[mask]
        final_mask = np.zeros_like(mask)
        final_mask[valid_indices[inlier_points]] = True

        return self._create_result(
            slope,
            slope_err,
            intercept,
            intercept_err,
            r_squared,
            final_mask,
            indices,
            "huber",
            n_iterations,
            covariance_matrix,
        )

    def _translate_multiplet_indices(
        self,
        result: BoltzmannFitResult,
        agg_to_obs_indices: list[list[int]],
        n_observations: int,
    ) -> BoltzmannFitResult:
        """Translate aggregated-point indices back to original observation indices.

        When *multiplet_groups* is used the fit operates on a (possibly smaller)
        aggregated array.  This helper maps ``rejected_points`` and ``inlier_mask``
        from aggregated-point space back to original-observation space so that
        downstream consumers such as ``plot()`` and visualization widgets can use
        the result against the original *observations* list without special-casing
        the multiplet path.

        For a rejected aggregated point every constituent original observation is
        also marked as rejected (and vice-versa for inlier points).
        """
        obs_rejected: list[int] = []
        for agg_i in result.rejected_points:
            obs_rejected.extend(agg_to_obs_indices[agg_i])

        if result.inlier_mask is not None:
            obs_inlier_mask = np.zeros(n_observations, dtype=bool)
            for agg_i, is_inlier in enumerate(result.inlier_mask):
                for obs_i in agg_to_obs_indices[agg_i]:
                    obs_inlier_mask[obs_i] = is_inlier
        else:
            obs_inlier_mask = None

        n_points = (
            int(np.sum(obs_inlier_mask))
            if obs_inlier_mask is not None
            else n_observations - len(obs_rejected)
        )

        return BoltzmannFitResult(
            temperature_K=result.temperature_K,
            temperature_uncertainty_K=result.temperature_uncertainty_K,
            intercept=result.intercept,
            intercept_uncertainty=result.intercept_uncertainty,
            r_squared=result.r_squared,
            n_points=n_points,
            rejected_points=sorted(obs_rejected),
            slope=result.slope,
            slope_uncertainty=result.slope_uncertainty,
            fit_method=result.fit_method,
            n_iterations=result.n_iterations,
            inlier_mask=obs_inlier_mask,
            covariance_matrix=result.covariance_matrix,
        )

    def _build_sigma_y(
        self,
        observations: list[LineObservation],
        aki_uncertainty_weighting: bool,
    ) -> np.ndarray:
        """Compute per-line sigma_y, optionally folding in NIST A_ki uncertainty.

        For y = ln(I * lambda / (g * A_ki)) the linearized error propagation gives
        sigma_y^2 = (sigma_I / I)^2 + (sigma(A_ki) / A_ki)^2 (errors in lambda and
        g_k are negligible). When ``aki_uncertainty_weighting`` is False, only the
        intensity term is retained so the fitter reproduces the legacy result.

        Lines with ``aki_uncertainty is None`` contribute zero atomic-data variance
        (i.e. fall back to the intensity-only sigma); the count of such lines is
        logged at debug level so users can see how much of the fit is unweighted
        by atomic-data quality.
        """
        sigma_intensity = np.array([obs.y_uncertainty for obs in observations])
        if not aki_uncertainty_weighting:
            return sigma_intensity

        sigma_aki = np.zeros(len(observations))
        n_missing = 0
        for i, obs in enumerate(observations):
            unc = obs.aki_uncertainty
            if unc is None or not np.isfinite(unc) or unc <= 0:
                n_missing += 1
                continue
            sigma_aki[i] = unc
        if n_missing:
            logger.debug(
                "%d/%d lines lack aki_uncertainty; using intensity-only sigma_y for those.",
                n_missing,
                len(observations),
            )
        return np.sqrt(sigma_intensity**2 + sigma_aki**2)

    def _compute_weights(self, y_err: np.ndarray) -> np.ndarray:
        """Inverse-variance weights ``1/sigma^2`` for weighted least squares.

        These are the weights that multiply the *squared* residual in the WLS
        objective ``sum_i w_i (y_i - y_hat_i)^2`` — the correct form for
        Gaussian measurement errors (Bevington & Robinson 2003, §6). They are
        consumed directly by :meth:`_compute_r_squared`, by the JAX 5-sum
        kernel (which minimises ``sum(w * r^2)``), and by the covariance
        normal-equation sums.

        IMPORTANT — :func:`numpy.polyfit` convention: ``polyfit(x, y, deg, w=W)``
        applies ``W`` to the *unsquared* residual and minimises
        ``sum((W*r)^2) = sum(W^2 r^2)``. So polyfit's ``w`` must be ``1/sigma``,
        i.e. ``sqrt`` of these inverse-variance weights. Every polyfit call site
        in this class therefore passes ``np.sqrt(self._compute_weights(...))``.
        Passing these weights to polyfit directly would yield a non-physical
        ``1/sigma^4`` weighting that collapses leverage onto the few
        highest-SNR points and biases the fitted slope/temperature.
        """
        if np.all(y_err == 0):
            return np.ones_like(y_err)
        safe_err = np.where(y_err > 0, y_err, np.inf)
        return 1.0 / safe_err**2

    def _compute_r_squared(self, y: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
        """Compute weighted R-squared."""
        ss_res = np.sum(weights * (y - y_pred) ** 2)
        ss_tot = np.sum(weights * (y - np.average(y, weights=weights)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    def _weighted_fit(self, x: np.ndarray, y: np.ndarray, y_err: np.ndarray) -> tuple:
        """Perform weighted linear fit returning slope, intercept, uncertainties, and covariance.

        Returns
        -------
        tuple
            (slope, intercept, slope_err, intercept_err, covariance_matrix)
            covariance_matrix is 2x2: [[var(slope), cov], [cov, var(intercept)]]
        """
        weights = self._compute_weights(y_err)
        # polyfit wants 1/sigma (= sqrt of inverse-variance weights).
        polyfit_w = np.sqrt(weights)
        try:
            (m, c), cov = np.polyfit(x, y, 1, w=polyfit_w, cov=True)
            return m, c, np.sqrt(cov[0, 0]), np.sqrt(cov[1, 1]), cov
        except np.linalg.LinAlgError:
            m, c = np.polyfit(x, y, 1, w=polyfit_w)
            return m, c, np.inf, np.inf, None

    def _create_result(
        self,
        slope: float,
        slope_err: float,
        intercept: float,
        intercept_err: float,
        r_squared: float,
        mask: np.ndarray,
        indices: np.ndarray,
        method: str,
        n_iterations: int,
        covariance_matrix: np.ndarray | None = None,
    ) -> BoltzmannFitResult:
        """Create BoltzmannFitResult with temperature calculation.

        Parameters
        ----------
        covariance_matrix : np.ndarray, optional
            2x2 covariance matrix [[var(slope), cov], [cov, var(intercept)]].
            Used for propagating correlated uncertainties through Saha/closure.
        """
        if slope >= 0:
            logger.warning(
                f"Positive or zero slope ({slope}) detected. "
                "Population inversion or error. T set to infinity."
            )
            temperature_K = float("inf")
            temp_err_K = float("inf")
        else:
            temperature_K = -1.0 / (KB_EV * slope)
            temp_err_K = (temperature_K**2) * KB_EV * slope_err

        rejected_points = list(indices[~mask])

        return BoltzmannFitResult(
            temperature_K=temperature_K,
            temperature_uncertainty_K=temp_err_K,
            intercept=intercept,
            intercept_uncertainty=intercept_err,
            r_squared=r_squared,
            n_points=int(np.sum(mask)),
            rejected_points=rejected_points,
            slope=slope,
            slope_uncertainty=slope_err,
            fit_method=method,
            n_iterations=n_iterations,
            inlier_mask=mask.copy(),
            covariance_matrix=covariance_matrix,
        )

    def _empty_result(self) -> BoltzmannFitResult:
        """Return empty result for failed fits."""
        return BoltzmannFitResult(
            temperature_K=0.0,
            temperature_uncertainty_K=0.0,
            intercept=0.0,
            intercept_uncertainty=0.0,
            r_squared=0.0,
            n_points=0,
            rejected_points=[],
            slope=0.0,
            slope_uncertainty=0.0,
            fit_method=self.method.value,
            n_iterations=0,
            inlier_mask=None,
            covariance_matrix=None,
        )

    def plot(
        self, observations: list[LineObservation], result: BoltzmannFitResult, ax=None
    ) -> object | None:
        """
        Plot the Boltzmann plot.

        Parameters
        ----------
        observations : list[LineObservation]
            Data points
        result : BoltzmannFitResult
            Fit result
        ax : matplotlib.axes.Axes, optional
            Axes to plot on

        Returns
        -------
        Figure or None
        """
        if not HAS_MATPLOTLIB:
            logger.warning("Matplotlib not installed, cannot plot.")
            return None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure

        # Extract data
        x_all = np.array([obs.E_k_ev for obs in observations])
        y_all = np.array([obs.y_value for obs in observations])
        y_err_all = np.array([obs.y_uncertainty for obs in observations])

        # Split into accepted and rejected
        rejected_set = set(result.rejected_points)
        accepted_mask = np.array([i not in rejected_set for i in range(len(observations))])

        # Plot accepted
        ax.errorbar(
            x_all[accepted_mask],
            y_all[accepted_mask],
            yerr=y_err_all[accepted_mask],
            fmt="o",
            color="blue",
            label="Accepted",
            alpha=0.7,
            capsize=3,
        )

        # Plot rejected
        if np.any(~accepted_mask):
            ax.scatter(
                x_all[~accepted_mask],
                y_all[~accepted_mask],
                marker="x",
                color="red",
                label="Rejected",
                zorder=5,
            )

        # Plot fit line
        # Use range of x
        x_min, x_max = np.min(x_all), np.max(x_all)
        x_range = np.linspace(x_min, x_max, 100)
        y_fit = result.slope * x_range + result.intercept

        label_fit = f"Fit: T = {result.temperature_K:.0f} ± {result.temperature_uncertainty_K:.0f} K\n$R^2$={result.r_squared:.3f}"
        ax.plot(x_range, y_fit, "k--", label=label_fit)

        ax.set_xlabel("Upper Level Energy $E_k$ (eV)")
        ax.set_ylabel("$\\ln(I \\lambda / g_k A_{ki})$")
        ax.set_title(
            f"Boltzmann Plot - {observations[0].element} {observations[0].ionization_stage}"
        )
        ax.legend()
        ax.grid(True, linestyle=":", alpha=0.6)

        return fig
