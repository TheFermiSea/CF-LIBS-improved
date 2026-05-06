"""
Closure equation implementation for CF-LIBS.

Includes standard, matrix, oxide, ILR (Isometric Log-Ratio), and PWLR
(pairwise/pivot log-ratio) closure modes. The ILR/PWLR modes map
compositions from the D-simplex to R^(D-1), enabling unconstrained
optimization in coordinate space.

References
----------
Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for
Compositional Data Analysis." Mathematical Geology 35(3), 279-300.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("inversion.closure")

LOGRATIO_CLIP_FLOOR = 1e-10
PWLR_WEIGHT_FLOOR = 1e-12
PWLR_ADAPTIVE_SCALE_MAX = 50.0


class ClosureMode(Enum):
    """Closure equation modes for CF-LIBS normalization."""

    STANDARD = "standard"
    MATRIX = "matrix"
    OXIDE = "oxide"
    ILR = "ilr"
    PWLR = "pwlr"


# ---------------------------------------------------------------------------
# ILR (Isometric Log-Ratio) compositional transform functions
# ---------------------------------------------------------------------------


def _helmert_basis(D: int) -> np.ndarray:
    """
    Helmert sub-composition contrast matrix (D x D-1).

    The columns form an orthonormal basis for the (D-1)-dimensional
    hyperplane in CLR space, satisfying V^T V = I_{D-1}.

    Parameters
    ----------
    D : int
        Number of compositional parts (must be >= 2).

    Returns
    -------
    np.ndarray
        Contrast matrix of shape (D, D-1).
    """
    if D < 2:
        raise ValueError("Helmert basis requires D >= 2")
    V = np.zeros((D, D - 1))
    for i in range(1, D):
        V[:i, i - 1] = 1.0 / np.sqrt(i * (i + 1))
        V[i, i - 1] = -i / np.sqrt(i * (i + 1))
    return V


def clr_transform(composition: np.ndarray) -> np.ndarray:
    """
    Centered log-ratio (CLR) transform.

    Parameters
    ----------
    composition : np.ndarray
        Composition vector(s) on the simplex, shape (D,) or (N, D).
        Values must be positive.

    Returns
    -------
    np.ndarray
        CLR coordinates, same shape as input.
    """
    log_comp = np.log(np.clip(composition, LOGRATIO_CLIP_FLOOR, None))
    return log_comp - np.mean(log_comp, axis=-1, keepdims=True)


def ilr_transform(composition: np.ndarray) -> np.ndarray:
    """
    Isometric log-ratio (ILR) transform using the Helmert basis.

    Maps a D-part composition on the simplex to (D-1) real-valued
    coordinates suitable for unconstrained optimization.

    Parameters
    ----------
    composition : np.ndarray
        Composition vector(s) on the simplex, shape (D,) or (N, D).

    Returns
    -------
    np.ndarray
        ILR coordinates, shape (D-1,) or (N, D-1).
    """
    clr = clr_transform(composition)
    V = _helmert_basis(composition.shape[-1])
    return clr @ V  # (D,) @ (D, D-1) -> (D-1,)


def ilr_inverse(coords: np.ndarray, D: int) -> np.ndarray:
    """
    Inverse ILR transform: map from R^(D-1) back to the D-simplex.

    Parameters
    ----------
    coords : np.ndarray
        ILR coordinates, shape (D-1,) or (N, D-1).
    D : int
        Number of compositional parts.

    Returns
    -------
    np.ndarray
        Composition on the simplex (sums to 1), shape (D,) or (N, D).
    """
    V = _helmert_basis(D)
    clr = coords @ V.T  # (D-1,) @ (D-1, D) -> (D,)
    comp = np.exp(clr)
    return comp / np.sum(comp, axis=-1, keepdims=True)


def _pivot_permutation(D: int, pivot_index: int) -> np.ndarray:
    """Return index permutation with the selected pivot moved to position 0."""
    if pivot_index < 0 or pivot_index >= D:
        raise ValueError(f"pivot_index={pivot_index} out of bounds for D={D}")
    if pivot_index == 0:
        return np.arange(D)
    keep = [i for i in range(D) if i != pivot_index]
    return np.array([pivot_index, *keep], dtype=int)


def plr_transform(composition: np.ndarray, pivot_index: int = 0) -> np.ndarray:
    """
    Pivot log-ratio (PLR) transform (a form of ALR).

    Uses a pivot element (default first element). Maps a D-part composition
    to (D-1) log-ratio coordinates.

    Parameters
    ----------
    composition : np.ndarray
        Composition vector(s) on the simplex, shape (D,) or (N, D).

    pivot_index : int, optional
        Index of the pivot component.

    Returns
    -------
    np.ndarray
        PLR coordinates, shape (D-1,) or (N, D-1).
    """
    D = composition.shape[-1]
    perm = _pivot_permutation(D, pivot_index)
    log_comp = np.log(np.clip(composition[..., perm], LOGRATIO_CLIP_FLOOR, None))
    return log_comp[..., 1:] - log_comp[..., :1]


def plr_inverse(coords: np.ndarray, D: Optional[int] = None, pivot_index: int = 0) -> np.ndarray:
    """
    Inverse PLR transform: map from R^(D-1) back to the D-simplex.

    Parameters
    ----------
    coords : np.ndarray
        PLR coordinates, shape (D-1,) or (N, D-1).

    D : int, optional
        Number of compositional parts. If omitted, inferred as ``coords.shape[-1] + 1``.
    pivot_index : int, optional
        Index of the pivot component in output composition.

    Returns
    -------
    np.ndarray
        Composition on the simplex (sums to 1), shape (D,) or (N, D).
    """
    if D is None:
        D = coords.shape[-1] + 1
    ratios = np.exp(coords)
    ones_shape = list(ratios.shape)
    ones_shape[-1] = 1
    simplex_perm = np.concatenate([np.ones(ones_shape), ratios], axis=-1)
    simplex_perm = simplex_perm / np.sum(simplex_perm, axis=-1, keepdims=True)

    perm = _pivot_permutation(D, pivot_index)
    inv_perm = np.argsort(perm)
    return simplex_perm[..., inv_perm]


def optimize_pwlr_coordinates(
    simplex: np.ndarray,
    pivot_index: int,
    regularization_strength: float = 1e-4,
) -> np.ndarray:
    """
    Optimize concentrations in PWLR coordinates with adaptive ridge regularization.

    The optimization is performed directly in PWLR space:

        argmin_z  0.5 * sum_i w_i (z_i - z_target_i)^2 + 0.5 * λ ||z||_2^2

    where ``z_target`` is the PWLR transform of the normalized raw concentrations.
    The closed-form minimizer is solved as a small linear system. For near-zero
    components, an adaptive λ dampens extreme coordinates and improves numerical
    stability without adding concentration offsets in simplex space.
    """
    if regularization_strength < 0.0 or not np.isfinite(regularization_strength):
        raise ValueError("regularization_strength must be finite and >= 0")

    D = simplex.shape[-1]
    target = plr_transform(simplex, pivot_index=pivot_index)
    if D <= 2:
        return target

    perm = _pivot_permutation(D, pivot_index)
    simplex_perm = np.clip(simplex[perm], PWLR_WEIGHT_FLOOR, None)
    weights = np.ones(D - 1)
    min_component = float(np.clip(simplex_perm.min(), PWLR_WEIGHT_FLOOR, None))
    adaptive_scale = 1.0 - np.log(min_component)
    adaptive_lambda = regularization_strength * min(adaptive_scale, PWLR_ADAPTIVE_SCALE_MAX)

    A = np.diag(weights + adaptive_lambda)
    b = weights * target
    return np.linalg.solve(A, b)


def _validated_abundance_multiplier(
    abundance_multipliers: Optional[Dict[str, float]],
    element: str,
) -> float:
    """Return a finite positive abundance multiplier for an element."""
    multiplier = abundance_multipliers.get(element, 1.0) if abundance_multipliers else 1.0
    if not np.isfinite(multiplier) or multiplier <= 0.0:
        raise ValueError(f"abundance_multipliers[{element!r}] must be finite and positive")
    return float(multiplier)


@dataclass
class ClosureResult:
    """
    Result of applying the closure equation.
    """

    concentrations: Dict[str, float]  # element -> number (mole) fraction (sum=1)
    experimental_factor: float  # The eliminated factor F (scaling factor)
    total_measured: float  # Sum of relative concentrations before normalization
    mode: str  # Mode used ('standard', 'matrix', 'oxide')


@dataclass
class DirichletResidualResult:
    """
    Result from Dirichlet-residual closure.

    This closure mode adds a latent "dark element" residual category to absorb
    mass from undetected elements (e.g., S, P with VUV lines outside
    spectrometer range).  Instead of inflating all detected concentrations via
    standard sum-to-one normalization, detected elements are normalized to fill
    only ``(1 - residual_fraction)`` of the total mass budget.

    Attributes
    ----------
    concentrations : Dict[str, float]
        Detected element concentrations (sum to ``1 - residual_fraction``).
    residual_fraction : float
        Estimated missing mass fraction (gamma_residual).
    raw_closure_sum : float
        Sum of raw (un-normalized) concentrations before any adjustment.
    closure_diagnostic : float
        Absolute deviation ``|raw_closure_sum - 1|``.  Large values signal
        significant missing (positive) or over-counted (negative) mass.
    mode : str
        ``'simple'`` or ``'dirichlet'``.
    alpha_residual : float
        Prior strength for the residual category (Dirichlet mode only;
        meaningless in simple mode).
    experimental_factor : float
        The eliminated factor *F* used for normalization, consistent with
        the standard closure interface.
    """

    concentrations: Dict[str, float]
    residual_fraction: float
    raw_closure_sum: float
    closure_diagnostic: float
    mode: str
    alpha_residual: float
    experimental_factor: float = 0.0


class ClosureEquation:
    """
    Handles the closure equation (normalization) step of CF-LIBS.

    The fundamental equation is: C_s = (U_s(T) * exp(q_s)) / F
    where q_s is the intercept from the Boltzmann plot.

    The closure condition determines F.
    """

    @staticmethod
    def apply_standard(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        abundance_multipliers: Optional[Dict[str, float]] = None,
    ) -> ClosureResult:
        """
        Apply standard closure: sum(C_s) = 1.

        F = sum(U_s * exp(q_s))

        Parameters
        ----------
        intercepts : Dict[str, float]
            Boltzmann plot intercepts q_s for each element
        partition_funcs : Dict[str, float]
            Partition function values U_s(T) for each element
        abundance_multipliers : Dict[str, float], optional
            Optional per-element scaling that maps the fitted intercept from
            the neutral Saha-Boltzmann plane back to total elemental abundance.
            Defaults to unity for all elements.

        Returns
        -------
        ClosureResult
        """
        # Calculate relative concentrations (unscaled)
        # rel_C_s = U_s * exp(q_s)
        rel_concentrations = {}
        total_measured = 0.0

        for element, q_s in intercepts.items():
            if element not in partition_funcs:
                logger.warning(f"Missing partition function for {element} in closure")
                continue

            U_s = partition_funcs[element]
            multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
            rel_C = multiplier * U_s * np.exp(q_s)
            rel_concentrations[element] = rel_C
            total_measured += rel_C

        if total_measured == 0:
            logger.error("Total measured concentration is zero")
            return ClosureResult({}, 0.0, 0.0, "standard")

        # F is effectively total_measured
        F = total_measured

        concentrations = {el: val / F for el, val in rel_concentrations.items()}

        return ClosureResult(
            concentrations=concentrations,
            experimental_factor=F,
            total_measured=total_measured,
            mode="standard",
        )

    @staticmethod
    def apply_matrix_mode(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        matrix_element: str,
        matrix_fraction: float = 0.9,
        abundance_multipliers: Optional[Dict[str, float]] = None,
    ) -> ClosureResult:
        """
        Apply matrix closure: One element has fixed concentration.

        C_matrix = (U_m * exp(q_m)) / F = matrix_fraction
        => F = (U_m * exp(q_m)) / matrix_fraction

        Parameters
        ----------
        intercepts : Dict[str, float]
            Intercepts
        partition_funcs : Dict[str, float]
            Partition functions
        matrix_element : str
            Element with known concentration
        matrix_fraction : float
            Concentration of matrix element (0.0 to 1.0)
        abundance_multipliers : Dict[str, float], optional
            Optional per-element scaling that maps the fitted intercept from
            the neutral Saha-Boltzmann plane back to total elemental abundance.
            Keys should match the intercept and partition-function mappings.
            Defaults to unity for all elements when omitted.

        Returns
        -------
        ClosureResult
        """
        if matrix_element not in intercepts or matrix_element not in partition_funcs:
            logger.error(f"Matrix element {matrix_element} missing from data")
            return ClosureEquation.apply_standard(
                intercepts,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )

        U_m = partition_funcs[matrix_element]
        q_m = intercepts[matrix_element]
        matrix_multiplier = _validated_abundance_multiplier(abundance_multipliers, matrix_element)
        rel_C_m = matrix_multiplier * U_m * np.exp(q_m)

        F = rel_C_m / matrix_fraction

        concentrations = {}
        total_measured = 0.0

        for element, q_s in intercepts.items():
            if element in partition_funcs:
                U_s = partition_funcs[element]
                multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
                rel_C = multiplier * U_s * np.exp(q_s)
                total_measured += rel_C
                concentrations[element] = rel_C / F

        return ClosureResult(
            concentrations=concentrations,
            experimental_factor=F,
            total_measured=total_measured,
            mode=f"matrix({matrix_element}={matrix_fraction})",
        )

    @staticmethod
    def apply_oxide_mode(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        oxide_stoichiometry: Dict[str, float],
        abundance_multipliers: Optional[Dict[str, float]] = None,
    ) -> ClosureResult:
        """
        Apply oxide closure: Elements exist as oxides, sum(Oxides) = 1.

        Used for geological samples.
        e.g. Si -> SiO2 (factor ~2.14), Al -> Al2O3

        C_oxide_s = C_s * oxide_factor_s
        sum(C_oxide_s) = 1
        sum( (U_s * exp(q_s) / F) * oxide_factor_s ) = 1
        F = sum( U_s * exp(q_s) * oxide_factor_s )

        Parameters
        ----------
        intercepts : Dict[str, float]
            Boltzmann plot intercepts q_s for each element.
        partition_funcs : Dict[str, float]
            Partition function values U_s(T) for each element.
        oxide_stoichiometry : Dict[str, float]
            Map of element to oxide conversion factor (e.g. ``{"Si": 2.139}``
            for SiO2).
        abundance_multipliers : Dict[str, float], optional
            Optional per-element scaling that maps the fitted intercept from
            the neutral Saha-Boltzmann plane back to total elemental abundance
            before oxide weighting. Defaults to unity for all elements when
            omitted.

        Returns
        -------
        ClosureResult
            Concentrations are ELEMENTAL fractions, but normalized such that oxides sum to 1.
        """
        rel_concentrations = {}
        total_oxide_rel = 0.0

        for element, q_s in intercepts.items():
            if element not in partition_funcs:
                continue

            U_s = partition_funcs[element]
            multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
            rel_C = multiplier * U_s * np.exp(q_s)
            rel_concentrations[element] = rel_C

            factor = oxide_stoichiometry.get(element, 1.0)  # Default to metal if no oxide
            total_oxide_rel += rel_C * factor

        if total_oxide_rel == 0:
            return ClosureResult({}, 0.0, 0.0, "oxide")

        F = total_oxide_rel

        concentrations = {el: val / F for el, val in rel_concentrations.items()}

        return ClosureResult(
            concentrations=concentrations,
            experimental_factor=F,
            total_measured=total_oxide_rel,
            mode="oxide",
        )

    @staticmethod
    def apply_ilr(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        abundance_multipliers: Optional[Dict[str, float]] = None,
    ) -> ClosureResult:
        """
        Apply ILR-based closure: compute concentrations via the simplex.

        Instead of direct normalization (standard closure), this method:
        1. Computes raw relative concentrations (U_s * exp(q_s)).
        2. Normalizes to an initial simplex estimate.
        3. Transforms to ILR space (unconstrained R^{D-1}).
        4. Transforms back to the simplex, guaranteeing sum=1 and positivity
           through the ILR inverse (exp + closure).

        This is mathematically equivalent to standard closure for a single
        pass, but provides the ILR infrastructure for downstream optimizers
        that need to work in unconstrained coordinates (e.g., gradient-based
        Boltzmann fitting or joint optimization in compositional space).

        Parameters
        ----------
        intercepts : Dict[str, float]
            Boltzmann plot intercepts q_s for each element.
        partition_funcs : Dict[str, float]
            Partition function values U_s(T) for each element.
        abundance_multipliers : Dict[str, float], optional
            Optional per-element scaling factors. Defaults to unity.

        Returns
        -------
        ClosureResult
            Concentrations on the simplex (sum=1, all positive).
        """
        # Build deterministic ordered element list and raw relative concentrations
        elements = []
        rel_values = []
        total_measured = 0.0

        for element in sorted(intercepts):
            q_s = intercepts[element]
            if element not in partition_funcs:
                logger.warning(f"Missing partition function for {element} in ILR closure")
                continue

            U_s = partition_funcs[element]
            multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
            rel_C = multiplier * U_s * np.exp(q_s)
            elements.append(element)
            rel_values.append(rel_C)
            total_measured += rel_C

        if total_measured == 0 or len(elements) < 2:
            logger.error("ILR closure requires at least 2 elements with non-zero concentration")
            return ClosureResult({}, 0.0, 0.0, "ilr")

        # Normalize to simplex, then round-trip through ILR
        raw = np.array(rel_values)
        simplex = raw / np.sum(raw)
        ilr_coords = ilr_transform(simplex)
        final_simplex = ilr_inverse(ilr_coords, len(elements))

        F = total_measured  # experimental factor matches standard definition

        concentrations = {el: float(final_simplex[i]) for i, el in enumerate(elements)}

        return ClosureResult(
            concentrations=concentrations,
            experimental_factor=F,
            total_measured=total_measured,
            mode="ilr",
        )

    @staticmethod
    def apply_pwlr(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        abundance_multipliers: Optional[Dict[str, float]] = None,
        pivot_element: Optional[str] = None,
        regularization_strength: float = 1e-4,
    ) -> ClosureResult:
        """
        Apply PWLR (Pivot Log-Ratio) based closure.

        Uses the pairwise/pivot log-ratio transform as an alternative to ILR. PLR
        provides a quasi-isometric coordinate system that is easier to
        interpret than ILR and handles compositional zeros more gracefully in
        optimization.

        Parameters
        ----------
        intercepts : Dict[str, float]
            Boltzmann plot intercepts q_s for each element.
        partition_funcs : Dict[str, float]
            Partition function values U_s(T) for each element.
        abundance_multipliers : Dict[str, float], optional
            Optional per-element scaling factors. Defaults to unity.
        pivot_element : str, optional
            Optional pivot element. If omitted, the dominant raw component is
            used as pivot.
        regularization_strength : float, optional
            Base ridge strength for PWLR-space optimization.

        Returns
        -------
        ClosureResult
            Concentrations on the simplex (sum=1, all positive).
        """
        if regularization_strength < 0.0 or not np.isfinite(regularization_strength):
            raise ValueError("regularization_strength must be finite and >= 0")

        # Build deterministic ordered element list and raw relative concentrations
        elements = []
        rel_values = []
        total_measured = 0.0

        for element in sorted(intercepts):
            q_s = intercepts[element]
            if element not in partition_funcs:
                logger.warning(f"Missing partition function for {element} in PWLR closure")
                continue

            U_s = partition_funcs[element]
            multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
            rel_C = multiplier * U_s * np.exp(q_s)
            elements.append(element)
            rel_values.append(rel_C)
            total_measured += rel_C

        if total_measured == 0 or len(elements) < 2:
            logger.error("PWLR closure requires at least 2 elements with non-zero concentration")
            return ClosureResult({}, 0.0, 0.0, "pwlr")

        # Normalize to simplex, optimize in PWLR space, then map back.
        raw = np.array(rel_values)
        simplex = raw / np.sum(raw)

        if pivot_element is not None:
            if pivot_element not in elements:
                raise ValueError(f"pivot_element {pivot_element!r} not found in closure elements")
            pivot_index = elements.index(pivot_element)
        else:
            pivot_index = int(np.argmax(simplex))

        coords = optimize_pwlr_coordinates(
            simplex=simplex,
            pivot_index=pivot_index,
            regularization_strength=regularization_strength,
        )
        final_simplex = plr_inverse(coords, D=len(elements), pivot_index=pivot_index)

        F = total_measured  # experimental factor matches standard definition

        concentrations = {el: float(final_simplex[i]) for i, el in enumerate(elements)}

        return ClosureResult(
            concentrations=concentrations,
            experimental_factor=F,
            total_measured=total_measured,
            mode="pwlr",
        )

    @staticmethod
    def apply_dirichlet_residual(
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        abundance_multipliers: Optional[Dict[str, float]] = None,
        mode: str = "simple",
        alpha_residual: float = 2.0,
        alpha_detected: float = 1.0,
        residual_threshold: float = 0.05,
    ) -> "DirichletResidualResult":
        """
        Apply closure with a latent dark-element residual category.

        When major elements go undetected (e.g., S or P whose strongest lines
        fall outside the spectrometer window), the standard ``sum(C_s) = 1``
        closure inflates every detected concentration.  This method adds a
        residual category ``gamma_residual`` so that
        ``sum(C_detected) + gamma_residual = 1``.

        Two estimation modes are supported:

        * **simple** -- ``gamma_residual = max(0, 1 - sum(C_raw))`` when the
          raw sum falls below ``1 - residual_threshold``; otherwise 0.
        * **dirichlet** -- MAP estimate under a Dirichlet prior
          ``Dir(alpha_1, ..., alpha_D, alpha_residual)`` where the prior on
          the residual category controls how much mass it can absorb.

        Parameters
        ----------
        intercepts : Dict[str, float]
            Boltzmann plot intercepts ``q_s`` for each detected element.
        partition_funcs : Dict[str, float]
            Partition function values ``U_s(T)`` for each detected element.
        abundance_multipliers : Dict[str, float], optional
            Per-element Saha scaling (same semantics as ``apply_standard``).
        mode : str
            ``'simple'`` or ``'dirichlet'``.
        alpha_residual : float
            Dirichlet prior strength for the residual category (only used in
            ``'dirichlet'`` mode).  Higher values allow more missing mass.
            Default is 2.0 (mild preference for some residual).
        alpha_detected : float
            Dirichlet prior strength for each detected element (only used in
            ``'dirichlet'`` mode).  Default is 1.0 (uniform / non-informative).
        residual_threshold : float
            In ``'simple'`` mode the residual is only activated when
            ``1 - sum(C_raw) > residual_threshold``.

        Returns
        -------
        DirichletResidualResult
        """
        # --- 1. Compute raw (un-normalized) concentrations ----------------
        raw_concentrations: Dict[str, float] = {}
        for element, q_s in intercepts.items():
            if element not in partition_funcs:
                logger.warning(
                    "Missing partition function for %s in dirichlet closure",
                    element,
                )
                continue
            U_s = partition_funcs[element]
            multiplier = _validated_abundance_multiplier(abundance_multipliers, element)
            raw_concentrations[element] = multiplier * U_s * np.exp(q_s)

        raw_sum = sum(raw_concentrations.values())
        if raw_sum == 0:
            logger.error("Total measured concentration is zero")
            return DirichletResidualResult(
                concentrations={},
                residual_fraction=1.0,
                raw_closure_sum=0.0,
                closure_diagnostic=1.0,
                mode=mode,
                alpha_residual=alpha_residual,
                experimental_factor=0.0,
            )

        closure_diagnostic = abs(raw_sum - 1.0)

        # --- 2. Estimate residual fraction --------------------------------
        if mode == "dirichlet":
            n_detected = len(raw_concentrations)
            sum_alpha_detected = n_detected * (alpha_detected - 1.0)
            alpha_res_minus_one = max(alpha_residual - 1.0, 0.0)
            denom = raw_sum + sum_alpha_detected + alpha_res_minus_one
            if denom > 0:
                residual = alpha_res_minus_one / denom
            else:
                residual = 0.0
            residual = max(0.0, min(residual, 1.0))
        else:
            deficit = 1.0 - raw_sum
            if deficit > residual_threshold:
                residual = max(0.0, deficit)
            else:
                residual = 0.0

        # --- 3. Normalize detected elements to (1 - residual) -------------
        detected_budget = 1.0 - residual
        concentrations = {el: c / raw_sum * detected_budget for el, c in raw_concentrations.items()}

        return DirichletResidualResult(
            concentrations=concentrations,
            residual_fraction=residual,
            raw_closure_sum=raw_sum,
            closure_diagnostic=closure_diagnostic,
            mode=mode,
            alpha_residual=alpha_residual,
            experimental_factor=raw_sum,
        )
