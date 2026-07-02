"""
Closure equation implementation for CF-LIBS.

Includes standard, matrix, oxide, ILR (Isometric Log-Ratio), and PWLR
(pivot log-ratio) closure modes. The ILR/PWLR modes map compositions
from the D-simplex to R^(D-1), enabling unconstrained optimization in
coordinate space.

The PWLR mode uses *isometric pivot coordinates* (Hron et al. 2012), a
true orthonormal balance basis built around a chosen pivot part. These
are distinct from the additive log-ratio (ALR), which is not isometric
(the ALR contrast matrix is not orthonormal, so Euclidean distances in
ALR space do not equal Aitchison distances). Both transforms are exposed
separately so callers can pick the geometry they need.

References
----------
Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for
Compositional Data Analysis." Mathematical Geology 35(3), 279-300.
Hron, K., Filzmoser, P., Thompson, K. (2012). "Linear regression with
compositional explanatory variables." Journal of Applied Statistics
39(5), 1115-1128. (Pivot/isometric log-ratio coordinates.)
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


# ---------------------------------------------------------------------------
# Geological oxide stoichiometry (oxygen atoms bound per cation).
# ---------------------------------------------------------------------------
#
# Oxide closure (:meth:`ClosureEquation.apply_oxide_mode`) weights each element's
# relative concentration (a cation NUMBER density n_cation = U·exp(q)·(1+S)) by an
# ``oxide_stoichiometry`` factor before the sum-to-one normalization.  The
# physically-grounded factor that makes the closure a *molar-oxygen* balance —
# sum_s n_cation_s · (O atoms per cation_s) = n_O — is simply the number of
# bound oxygen atoms per cation, b/a for an oxide X_a O_b.  This molar-oxygen
# closure is the variant validated on real BHVO-2 (oxide RMSE 5.60 vs 10.33 for
# bare sum-of-cations closure; SB-graph + oxide RMSE 4.29).
#
# The ratios below are pure stoichiometric constants fixed by each cation's
# common geological oxidation state (the rock-forming major-element oxides):
#   Si^4+  -> SiO2   (2 O / Si)        P^5+  -> P2O5  (2.5 O / P)
#   Ti^4+  -> TiO2   (2 O / Ti)        Al^3+ -> Al2O3 (1.5 O / Al)
#   Fe^3+  -> Fe2O3  (1.5 O / Fe; total iron reported as Fe2O3 per Jochum 2016)
#   Mn^2+  -> MnO    (1 O / Mn)        Mg^2+ -> MgO   (1 O / Mg)
#   Ca^2+  -> CaO    (1 O / Ca)        Na^+  -> Na2O  (0.5 O / Na)
#   K^+    -> K2O    (0.5 O / K)
# Elements absent from this map fall back to factor 1.0 in ``apply_oxide_mode``
# (treated as elemental / metal), so the table is opt-in and dataset-agnostic.
OXIDE_OXYGEN_PER_CATION: Dict[str, float] = {
    "Si": 2.0,
    "Ti": 2.0,
    "Al": 1.5,
    "Fe": 1.5,
    "Mn": 1.0,
    "Mg": 1.0,
    "Ca": 1.0,
    "Na": 0.5,
    "K": 0.5,
    "P": 2.5,
}


def default_oxide_stoichiometry(elements: Optional[list] = None) -> Dict[str, float]:
    """Return the molar-oxygen oxide-closure factors (O atoms per cation).

    These are the ``oxide_stoichiometry`` factors consumed by
    :meth:`ClosureEquation.apply_oxide_mode`: weighting each cation number
    density by its bound-oxygen count turns the closure into a molar-oxygen
    balance.  See :data:`OXIDE_OXYGEN_PER_CATION` for the derivation.

    Parameters
    ----------
    elements : list of str, optional
        If given, restrict the returned mapping to these elements (entries
        without a known oxide stoichiometry are omitted, so ``apply_oxide_mode``
        treats them as elemental). If ``None`` the full table is returned.

    Returns
    -------
    dict
        Element -> oxygen-atoms-per-cation factor.
    """
    if elements is None:
        return dict(OXIDE_OXYGEN_PER_CATION)
    return {el: OXIDE_OXYGEN_PER_CATION[el] for el in elements if el in OXIDE_OXYGEN_PER_CATION}


def log_ratios(
    concentrations: Dict[str, float],
    reference: str,
    *,
    include_reference: bool = False,
) -> Dict[str, float]:
    """Aitchison additive log-ratios ``ln(C_s / C_ref)`` against a reference part.

    The CF-LIBS deliverable for constrained composition *tracking* (e.g. DED
    Ti-6Al-4V drift) is the pairwise ratio, **not** the closure-normalized
    weight percent. The standard closure normalizes ``sum_s C_s = 1`` over the
    *detected* set (:meth:`ClosureEquation.apply_standard`), so every ``C_s``
    shares the denominator ``F = sum_t rel_t`` and any per-element intensity /
    atomic-data / self-absorption error in one element sloshes into *every*
    other element's fraction (the observed V<->Ti mass-slosh). The ratio cancels
    that shared denominator exactly::

        C_s / C_ref = (rel_s / F) / (rel_ref / F) = rel_s / rel_ref

    so ``ln(C_s / C_ref)`` is provably matrix- and detected-set invariant
    (Aitchison 1986 subcompositional coherence; formalized in
    ``cflibs-formal`` ``MatrixEffects.lean`` theorem
    ``recoveredComposition_ratio_matrix_invariant`` — the recovered ratio of two
    detected species is identical under any two detected sets, while the absolute
    fractions carry the completeness-channel matrix effect
    ``recoveredComposition_absolute_matrix_dependent``).

    Because the denominator cancels, this value is identical whether computed
    from the pre-closure relatives ``rel_s = mult_s * U_s * exp(q_s)`` or from
    the post-closure normalized fractions ``C_s`` — only the *ratio* is needed
    and the closure output already carries it, so this helper operates directly
    on the normalized ``concentrations``. It is also identical, up to an additive
    constant ``ln(AW_s / AW_ref)`` that cancels in any predicted-minus-truth
    difference, whether the inputs are number/mole fractions or mass fractions.

    Parameters
    ----------
    concentrations : dict of str -> float
        Element -> concentration (number/mole fraction, mass fraction, or raw
        relative abundance — the log-ratio geometry is scale-invariant).
    reference : str
        The denominator element. For DED this is the dominant matrix element
        (e.g. ``"Ti"`` for Ti-6Al-4V).
    include_reference : bool, default False
        If True, include ``reference -> 0.0`` (``ln(C_ref / C_ref)``) in the
        output. Default omits the reference (standard additive-log-ratio
        convention).

    Returns
    -------
    dict of str -> float
        ``element -> ln(C_element / C_reference)`` for every element except the
        reference. A numerator that is missing, zero, negative, or non-finite
        maps to ``float('nan')`` — physics-honest "not measured / below
        detection". No clipping is applied, so a genuine zero is never reported
        as a finite ratio.

    Raises
    ------
    KeyError
        If ``reference`` is not a key of ``concentrations``: with no reference
        density no ratio is defined, so the function refuses rather than
        silently returning all-NaN.
    ValueError
        If the reference concentration is zero, negative, or non-finite: the
        denominator is invalid and every ratio would be undefined.
    """
    if reference not in concentrations:
        raise KeyError(
            f"reference element {reference!r} absent from concentrations "
            f"{sorted(concentrations)}; cannot form log-ratios"
        )
    ref_val = float(concentrations[reference])
    if not np.isfinite(ref_val) or ref_val <= 0.0:
        raise ValueError(
            f"reference concentration for {reference!r} is {ref_val!r}; "
            "must be finite and > 0 to form log-ratios"
        )
    log_ref = float(np.log(ref_val))
    out: Dict[str, float] = {}
    if include_reference:
        out[reference] = 0.0
    for element, value in concentrations.items():
        if element == reference:
            continue
        v = float(value)
        if not np.isfinite(v) or v <= 0.0:
            out[element] = float("nan")
        else:
            out[element] = float(np.log(v) - log_ref)
    return out


class ClosureMode(Enum):
    """Closure equation modes for CF-LIBS normalization."""

    STANDARD = "standard"
    MATRIX = "matrix"
    OXIDE = "oxide"
    ILR = "ilr"
    PWLR = "pwlr"
    DIRICHLET_RESIDUAL = "dirichlet_residual"


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


def ilr_propagate_covariance(
    composition: np.ndarray,
    covariance: np.ndarray,
    symmetrize: bool = True,
) -> np.ndarray:
    """
    Propagate a simplex covariance into full-rank ILR-coordinate covariance.

    A composition covariance expressed on the *closed* simplex (raw parts
    summing to one) is necessarily rank-deficient: the closure constraint
    ``1^T c = 1`` forces ``1^T Sigma_c = 0`` (every row/column sums to zero),
    so ``Sigma_c`` has rank at most ``D - 1`` and is singular. Mapping it into
    isometric log-ratio (ILR) coordinates removes the degenerate direction and
    yields a full-rank ``(D-1) x (D-1)`` covariance suitable for inference,
    confidence-region construction, or Mahalanobis distances.

    The propagation is the first-order (delta-method) push-forward of the ILR
    map ``z = ilr(c) = V^T clr(c)`` through the orthonormal Helmert basis ``V``
    (:func:`_helmert_basis`). The clr Jacobian is ``J_clr = G diag(1/c)`` with
    centering matrix ``G = I - (1/D) 1 1^T``; composing with ``V`` and using the
    fact that the Helmert columns already sum to zero (``V^T G = V^T``) gives

        J = V^T diag(1/c)        (shape (D-1, D))
        Sigma_z = J Sigma_c J^T  (shape (D-1, D-1), full rank).

    Parameters
    ----------
    composition : np.ndarray
        Composition vector on the simplex, shape ``(D,)`` (``D >= 2``). The
        expansion point for the linearization; values must be positive and are
        clipped at :data:`LOGRATIO_CLIP_FLOOR` for numerical safety.
    covariance : np.ndarray
        Covariance of the (closed) composition, shape ``(D, D)``. Typically
        rank ``D - 1`` with rows/columns summing to ~0, but any symmetric
        matrix is accepted (the closure-degenerate direction is projected out
        by the transform).
    symmetrize : bool, optional
        If ``True`` (default) the returned matrix is symmetrized as
        ``0.5 (Sigma_z + Sigma_z^T)`` to suppress floating-point asymmetry.

    Returns
    -------
    np.ndarray
        ILR-coordinate covariance, shape ``(D-1, D-1)``. Positive semidefinite
        and, for a full-rank ``Sigma_c`` restricted to the simplex tangent
        space, positive definite.

    References
    ----------
    Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for
    Compositional Data Analysis." Mathematical Geology 35(3), 279-300.
    Aitchison, J. (1986). "The Statistical Analysis of Compositional Data."
    Chapman & Hall. (Closure-induced singularity of the raw covariance and the
    log-ratio resolution thereof.)
    """
    comp = np.asarray(composition, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if comp.ndim != 1:
        raise ValueError("composition must be a 1-D vector for covariance propagation")
    D = comp.shape[-1]
    if D < 2:
        raise ValueError("ILR covariance propagation requires D >= 2")
    if cov.shape != (D, D):
        raise ValueError(f"covariance must have shape ({D}, {D}), got {cov.shape}")

    V = _helmert_basis(D)
    inv_comp = 1.0 / np.clip(comp, LOGRATIO_CLIP_FLOOR, None)
    # J = V^T @ diag(1/c); V^T G = V^T because Helmert columns sum to zero.
    jacobian = V.T * inv_comp  # (D-1, D): scales column k by 1/c_k
    sigma_z = jacobian @ cov @ jacobian.T
    if symmetrize:
        sigma_z = 0.5 * (sigma_z + sigma_z.T)
    return sigma_z


def simplex_covariance_from_ilr(
    coords: np.ndarray,
    covariance: np.ndarray,
    symmetrize: bool = True,
) -> np.ndarray:
    """
    Back-propagate an ILR-coordinate covariance to the closed simplex.

    Inverse of :func:`ilr_propagate_covariance`. Given a full-rank covariance
    ``Sigma_z`` in ILR coordinates, returns the closure-consistent simplex
    covariance ``Sigma_c`` obtained by the delta-method push-forward of the
    inverse ILR map ``c = C(exp(V z))`` (closure of the softmax over clr
    coordinates ``y = V z``):

        J_back = (diag(c) - c c^T) V    (shape (D, D-1))
        Sigma_c = J_back Sigma_z J_back^T.

    Because ``1^T (diag(c) - c c^T) = c^T - (1^T c) c^T = 0`` whenever
    ``1^T c = 1``, the result satisfies ``1^T Sigma_c = 0`` and ``Sigma_c 1 =
    0`` exactly: rows and columns sum to ~0, consistent with the simplex
    closure constraint. The matrix is positive semidefinite (congruence of the
    positive semidefinite ``Sigma_z``) and rank-deficient by construction
    (rank at most ``D - 1``), reflecting the unrecoverable closed-composition
    null direction.

    Parameters
    ----------
    coords : np.ndarray
        ILR coordinates ``z``, shape ``(D-1,)``. The expansion point; the
        composition ``c`` is recovered via :func:`ilr_inverse`.
    covariance : np.ndarray
        ILR-coordinate covariance, shape ``(D-1, D-1)``.
    symmetrize : bool, optional
        If ``True`` (default) the returned matrix is symmetrized as
        ``0.5 (Sigma_c + Sigma_c^T)`` to suppress floating-point asymmetry.

    Returns
    -------
    np.ndarray
        Closure-consistent simplex covariance, shape ``(D, D)``, with rows and
        columns summing to ~0 and positive semidefinite.

    References
    ----------
    Egozcue, J.J. et al. (2003). "Isometric Logratio Transformations for
    Compositional Data Analysis." Mathematical Geology 35(3), 279-300.
    Aitchison, J. (1986). "The Statistical Analysis of Compositional Data."
    Chapman & Hall.
    """
    z = np.asarray(coords, dtype=np.float64)
    sigma_z = np.asarray(covariance, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("coords must be a 1-D vector for covariance back-propagation")
    Dm1 = z.shape[-1]
    if Dm1 < 1:
        raise ValueError("ILR coordinates require at least one dimension (D >= 2)")
    if sigma_z.shape != (Dm1, Dm1):
        raise ValueError(f"covariance must have shape ({Dm1}, {Dm1}), got {sigma_z.shape}")

    D = Dm1 + 1
    comp = ilr_inverse(z, D)  # (D,) expansion point on the simplex
    V = _helmert_basis(D)
    # J_back = (diag(c) - c c^T) @ V; the softmax-on-clr Jacobian times V.
    softmax_jac = np.diag(comp) - np.outer(comp, comp)  # (D, D)
    jacobian = softmax_jac @ V  # (D, D-1)
    sigma_c = jacobian @ sigma_z @ jacobian.T
    if symmetrize:
        sigma_c = 0.5 * (sigma_c + sigma_c.T)
    return sigma_c


def _pivot_permutation(D: int, pivot_index: int) -> np.ndarray:
    """Return index permutation with the selected pivot moved to position 0."""
    if pivot_index < 0 or pivot_index >= D:
        raise ValueError(f"pivot_index={pivot_index} out of bounds for D={D}")
    if pivot_index == 0:
        return np.arange(D)
    keep = [i for i in range(D) if i != pivot_index]
    return np.array([pivot_index, *keep], dtype=int)


def _pivot_contrast(D: int) -> np.ndarray:
    """Hron 2012 isometric pivot-coordinate contrast matrix (D-1, D).

    Row ``i`` (1-based, ``i = 1 .. D-1``) contrasts the ``i``-th permuted
    part against the geometric mean of the remaining ``D - i`` parts:

        psi_i = sqrt((D - i) / (D - i + 1)) *
                ( e_i - (1 / (D - i)) * sum_{j > i} e_j )

    The rows are orthonormal (``Psi @ Psi.T == I_{D-1}``) and each row sums
    to zero, so the resulting coordinates form an *isometric* log-ratio
    (ILR) basis in CLR space — Euclidean distances between coordinate
    vectors equal Aitchison distances between compositions.

    Parameters
    ----------
    D : int
        Number of compositional parts (must be >= 2). Parts are assumed
        already permuted so the pivot occupies position 0.

    Returns
    -------
    np.ndarray
        Contrast matrix of shape (D-1, D).
    """
    if D < 2:
        raise ValueError("pivot contrast requires D >= 2")
    Psi = np.zeros((D - 1, D))
    for i in range(1, D):
        coef = np.sqrt((D - i) / (D - i + 1))
        Psi[i - 1, i - 1] = coef
        Psi[i - 1, i:] = -coef / (D - i)
    return Psi


def alr_transform(composition: np.ndarray, pivot_index: int = 0) -> np.ndarray:
    """
    Additive log-ratio (ALR) transform against a pivot/reference part.

    Maps a D-part composition to (D-1) plain log-ratio coordinates
    ``ln(x_j / x_pivot)``. ALR is *not* isometric (the implied contrast
    matrix is not orthonormal), so Euclidean distance in ALR space does
    not equal Aitchison distance — use :func:`plr_transform` (isometric
    pivot coordinates) or :func:`ilr_transform` when isometry matters.

    Parameters
    ----------
    composition : np.ndarray
        Composition vector(s) on the simplex, shape (D,) or (N, D).
    pivot_index : int, optional
        Index of the reference (denominator) component.

    Returns
    -------
    np.ndarray
        ALR coordinates, shape (D-1,) or (N, D-1).
    """
    D = composition.shape[-1]
    perm = _pivot_permutation(D, pivot_index)
    log_comp = np.log(np.clip(composition[..., perm], LOGRATIO_CLIP_FLOOR, None))
    return log_comp[..., 1:] - log_comp[..., :1]


def alr_inverse(coords: np.ndarray, D: Optional[int] = None, pivot_index: int = 0) -> np.ndarray:
    """
    Inverse ALR transform: map from R^(D-1) back to the D-simplex.

    Parameters
    ----------
    coords : np.ndarray
        ALR coordinates, shape (D-1,) or (N, D-1).
    D : int, optional
        Number of compositional parts. If omitted, inferred as
        ``coords.shape[-1] + 1``.
    pivot_index : int, optional
        Index of the reference component in the output composition.

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


def plr_transform(composition: np.ndarray, pivot_index: int = 0) -> np.ndarray:
    """
    Pivot log-ratio (PLR) transform — true isometric pivot coordinates.

    Implements the Hron et al. (2012) isometric pivot coordinates. After
    moving the pivot to the first position, the ``i``-th coordinate is the
    balance between the ``i``-th part and the geometric mean of all
    subsequent parts:

        z_i = sqrt((D - i) / (D - i + 1)) *
              ln( x_i / geomean(x_{i+1}, ..., x_D) ),   i = 1 .. D-1

    For example, with ``x = [0.7, 0.2, 0.1]`` and ``pivot_index = 0``:

        z_1 = sqrt(2/3) * ln(0.7 / sqrt(0.2 * 0.1)).

    Unlike the additive log-ratio (:func:`alr_transform`), this transform
    is isometric: Euclidean distances between coordinate vectors equal
    Aitchison distances between compositions.

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
    comp = np.asarray(composition, dtype=np.float64)
    D = comp.shape[-1]
    perm = _pivot_permutation(D, pivot_index)
    log_comp = np.log(np.clip(comp[..., perm], LOGRATIO_CLIP_FLOOR, None))
    Psi = _pivot_contrast(D)
    return log_comp @ Psi.T


def plr_inverse(coords: np.ndarray, D: Optional[int] = None, pivot_index: int = 0) -> np.ndarray:
    """
    Inverse PLR transform: map isometric pivot coordinates back to the
    D-simplex.

    Parameters
    ----------
    coords : np.ndarray
        PLR coordinates, shape (D-1,) or (N, D-1).
    D : int, optional
        Number of compositional parts. If omitted, inferred as
        ``coords.shape[-1] + 1``.
    pivot_index : int, optional
        Index of the pivot component in the output composition.

    Returns
    -------
    np.ndarray
        Composition on the simplex (sums to 1), shape (D,) or (N, D).
    """
    coords = np.asarray(coords, dtype=np.float64)
    if D is None:
        D = coords.shape[-1] + 1
    Psi = _pivot_contrast(D)
    clr = coords @ Psi  # (D-1,) @ (D-1, D) -> (D,)
    comp_perm = np.exp(clr)
    comp_perm = comp_perm / np.sum(comp_perm, axis=-1, keepdims=True)

    perm = _pivot_permutation(D, pivot_index)
    inv_perm = np.argsort(perm)
    return comp_perm[..., inv_perm]


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


def _stabilized_relative_concentrations(
    elements,
    intercepts: Dict[str, float],
    partition_funcs: Dict[str, float],
    abundance_multipliers: Optional[Dict[str, float]],
):
    """Overflow-safe relative abundances ``rel_s = mult_s * U_s * exp(q_s)``.

    The naive ``U_s * exp(q_s)`` form overflows to ``inf`` for large Boltzmann
    intercepts ``q_s`` (and ``inf/inf`` -> ``NaN`` once normalized), which
    silently produces a non-finite composition that downstream gates can
    mistake for a converged result. To avoid this, the relative abundances are
    formed in log space and shifted by their maximum (the classic log-sum-exp
    trick)::

        log_rel_s = log(mult_s) + log(U_s) + q_s
        rel_s     = exp(log_rel_s - max_s log_rel_s)

    Every closure mode normalizes ``rel`` by some weighted sum of its own
    entries, so the common ``exp(-offset)`` shift cancels exactly and the
    normalized composition is mathematically identical to the naive form while
    never overflowing for finite ``q_s``.

    Returns
    -------
    rel : np.ndarray
        Stabilized relative abundances aligned to ``elements``.
    offset : float
        The subtracted maximum log abundance, so callers can reconstruct the
        absolute scale (experimental factor / total_measured) as
        ``rel * exp(offset)`` when ``offset`` is finite.
    """
    log_rel = np.array(
        [
            np.log(_validated_abundance_multiplier(abundance_multipliers, el))
            + np.log(partition_funcs[el])
            + intercepts[el]
            for el in elements
        ],
        dtype=float,
    )
    if log_rel.size == 0:
        return log_rel, 0.0
    offset = float(np.max(log_rel))
    if not np.isfinite(offset):
        # offset == -inf: no measurable signal (every rel -> 0).
        # offset == +inf: pathological intercept; emit NaN so the caller's
        # finite-total guard flags it rather than reporting a fake composition.
        if offset < 0:
            return np.zeros_like(log_rel), offset
        return np.full_like(log_rel, np.nan), offset
    return np.exp(log_rel - offset), offset


def _abs_scale(offset: float) -> float:
    """Reconstruct the absolute scale ``exp(offset)`` removed by stabilization.

    Returns ``+inf`` when the maximum log abundance genuinely overflows float64
    (``offset > 709``) instead of letting ``np.exp`` emit a RuntimeWarning. The
    absolute scale only feeds the reported ``experimental_factor`` /
    ``total_measured`` fields, never the normalized composition.
    """
    if not np.isfinite(offset):
        return float("inf") if offset > 0 else 0.0
    if offset > 709.0:
        return float("inf")
    return float(np.exp(offset))


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
    def validate_degeneracy(
        concentrations: Dict[str, float],
        threshold: float = 0.8,
        min_elements: int = 2,
    ) -> bool:
        """Flag a degenerate (single-element-dominated) composition.

        Returns ``True`` when at least ``min_elements`` elements are present
        and any single element soaks more than ``threshold`` of the total
        closure mass — the "keystone collapse" signature in which the closure
        has lost discriminating power and the recovered composition is
        untrustworthy.

        A single-element solve (``len <= 1``) is never flagged: a pure sample
        legitimately closes to a single 1.0 concentration. Callers analyzing
        small candidate sets where one dominant element is physically
        plausible (binary alloys: brass is ~90% Cu) should raise
        ``min_elements`` — the iterative solver passes ``min_elements=4`` so
        the gate fires only on multi-element candidate sets where >0.8
        dominance is the collapse signature, not chemistry.

        Parameters
        ----------
        concentrations : Dict[str, float]
            Element -> number/mole fraction (typically summing to 1).
        threshold : float
            Dominance fraction above which the composition is degenerate
            (default 0.8).
        min_elements : int
            Minimum candidate-set size for the gate to apply (default 2,
            i.e. any multi-element composition is eligible).

        Returns
        -------
        bool
            ``True`` if degenerate, ``False`` otherwise.
        """
        # A non-finite concentration (NaN/inf from an overflowed closure) is
        # always degenerate: a NaN composition must never be reported as a
        # converged result, regardless of candidate-set size.
        if any(not np.isfinite(c) for c in concentrations.values()):
            return True
        if len(concentrations) < max(int(min_elements), 2):
            return False
        return any(c > threshold for c in concentrations.values())

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
        # Relative concentrations rel_C_s = U_s * exp(q_s), formed in log space
        # with max-subtraction so large intercepts cannot overflow to inf/NaN.
        elements_order = []
        for element in intercepts:
            if element not in partition_funcs:
                logger.warning(f"Missing partition function for {element} in closure")
                continue
            elements_order.append(element)

        rel, offset = _stabilized_relative_concentrations(
            elements_order, intercepts, partition_funcs, abundance_multipliers
        )
        total_stable = float(np.sum(rel))

        if total_stable <= 0.0 or not np.isfinite(total_stable):
            logger.error("Total measured concentration is zero or non-finite")
            return ClosureResult({}, 0.0, 0.0, "standard")

        concentrations = {el: float(r / total_stable) for el, r in zip(elements_order, rel)}

        # Reported absolute scale (may be +inf for genuine overflow; never used
        # for the normalized composition above).
        total_measured = total_stable * _abs_scale(offset)
        F = total_measured

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

        elements_order = [el for el in intercepts if el in partition_funcs]
        rel, offset = _stabilized_relative_concentrations(
            elements_order, intercepts, partition_funcs, abundance_multipliers
        )
        m_idx = elements_order.index(matrix_element)
        rel_m = float(rel[m_idx])
        F_stable = rel_m / matrix_fraction

        if F_stable <= 0.0 or not np.isfinite(F_stable):
            logger.error("Matrix element relative concentration is zero or non-finite")
            return ClosureResult({}, 0.0, 0.0, f"matrix({matrix_element}={matrix_fraction})")

        concentrations = {el: float(r / F_stable) for el, r in zip(elements_order, rel)}

        # Reported absolute scale (may be +inf for genuine overflow).
        scale = _abs_scale(offset)
        total_measured = float(np.sum(rel)) * scale
        F = F_stable * scale

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
        The oxide factor is OXYGEN ATOMS PER CATION (stoichiometric), NOT the
        molar-mass ratio: e.g. Si -> SiO2 has 2 O per Si = 2.0 (not 2.139), and
        Al -> Al2O3 has 1.5 O per Al = 1.5.

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
            Map of element to OXYGEN ATOMS PER CATION (e.g. ``{"Si": 2.0}`` for
            SiO2, ``{"Al": 1.5}`` for Al2O3); see ``OXIDE_OXYGEN_PER_CATION``.
            This is the stoichiometric O/cation ratio, NOT the molar-mass ratio
            (M(SiO2)/M(Si) = 2.139) — passing molar-mass ratios corrupts closure.
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
        elements_order = [el for el in intercepts if el in partition_funcs]
        rel, offset = _stabilized_relative_concentrations(
            elements_order, intercepts, partition_funcs, abundance_multipliers
        )
        factors = np.array(
            [oxide_stoichiometry.get(el, 1.0) for el in elements_order],  # metal if no oxide
            dtype=float,
        )
        total_oxide_stable = float(np.sum(rel * factors))

        if total_oxide_stable <= 0.0 or not np.isfinite(total_oxide_stable):
            return ClosureResult({}, 0.0, 0.0, "oxide")

        concentrations = {el: float(r / total_oxide_stable) for el, r in zip(elements_order, rel)}

        # Reported absolute scale (may be +inf for genuine overflow).
        total_oxide_rel = total_oxide_stable * _abs_scale(offset)
        F = total_oxide_rel

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
        # Build deterministic ordered element list; relative abundances are
        # formed in log space with max-subtraction (overflow-safe).
        elements = []
        for element in sorted(intercepts):
            if element not in partition_funcs:
                logger.warning(f"Missing partition function for {element} in ILR closure")
                continue
            elements.append(element)

        rel, offset = _stabilized_relative_concentrations(
            elements, intercepts, partition_funcs, abundance_multipliers
        )
        total_stable = float(np.sum(rel))

        if total_stable <= 0.0 or not np.isfinite(total_stable) or len(elements) < 2:
            logger.error("ILR closure requires at least 2 elements with non-zero concentration")
            return ClosureResult({}, 0.0, 0.0, "ilr")

        # Normalize to simplex (the common stabilization shift cancels here),
        # then round-trip through ILR
        simplex = rel / total_stable
        ilr_coords = ilr_transform(simplex)
        final_simplex = ilr_inverse(ilr_coords, len(elements))

        # experimental factor matches standard definition (absolute scale)
        total_measured = total_stable * _abs_scale(offset)
        F = total_measured

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

        # Build deterministic ordered element list; relative abundances are
        # formed in log space with max-subtraction (overflow-safe).
        elements = []
        for element in sorted(intercepts):
            if element not in partition_funcs:
                logger.warning(f"Missing partition function for {element} in PWLR closure")
                continue
            elements.append(element)

        rel, offset = _stabilized_relative_concentrations(
            elements, intercepts, partition_funcs, abundance_multipliers
        )
        total_stable = float(np.sum(rel))

        if total_stable <= 0.0 or not np.isfinite(total_stable) or len(elements) < 2:
            logger.error("PWLR closure requires at least 2 elements with non-zero concentration")
            return ClosureResult({}, 0.0, 0.0, "pwlr")

        # Normalize to simplex (the common stabilization shift cancels here),
        # optimize in PWLR space, then map back.
        simplex = rel / total_stable

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

        # experimental factor matches standard definition (absolute scale)
        total_measured = total_stable * _abs_scale(offset)
        F = total_measured

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
        experimental_factor: float = 1.0,
    ) -> "DirichletResidualResult":
        """
        Apply closure with a latent dark-element residual category.

        When major elements go undetected (e.g., S or P whose strongest lines
        fall outside the spectrometer window), the standard ``sum(C_s) = 1``
        closure inflates every detected concentration.  This method adds a
        residual category ``gamma_residual`` so that
        ``sum(C_detected) + gamma_residual = 1``.

        The raw relative concentrations ``rel_C_s = M_s · U_s · exp(q_s)`` are
        only known up to the experimental factor *F* (the global multiplicative
        constant eliminated by the closure condition; see
        :class:`ClosureResult`).  The "missing mass" is therefore measured in
        *calibrated* space, ``rho_s = rel_C_s / F``: the deficit and the
        closure diagnostic compare ``sum(rho_s)`` against unity, not the bare
        ``sum(rel_C_s)``.  Because ``F`` carries the same intensity units as
        ``rel_C_s``, scaling all line intensities by a constant *k* (which
        scales every ``rel_C_s`` and the data-derived ``F`` by *k*) leaves the
        residual and diagnostic unchanged — i.e. the closure is
        **scale-invariant** (Egozcue et al. 2003: compositional information is
        scale-free).  The default ``F = 1`` reproduces the convention where
        intercepts are already on a unit-calibrated scale.

        Two estimation modes are supported:

        * **simple** -- ``gamma_residual = max(0, 1 - sum(rho_s))`` when the
          calibrated raw sum falls below ``1 - residual_threshold``;
          otherwise 0.
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
            ``1 - sum(rho_s) > residual_threshold``.
        experimental_factor : float
            The intensity calibration scale *F* used to map raw relative
            concentrations onto the calibrated scale ``rho_s = rel_C_s / F``.
            Must be finite and positive.  Pass the same intensity-derived
            factor that scales the line intensities to keep the diagnostic
            scale-invariant; the default ``1.0`` assumes unit calibration.

        Returns
        -------
        DirichletResidualResult
        """
        if not np.isfinite(experimental_factor) or experimental_factor <= 0.0:
            raise ValueError("experimental_factor must be finite and positive")

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
        if raw_sum == 0 or not np.isfinite(raw_sum):
            logger.error("Total measured concentration is zero or non-finite")
            return DirichletResidualResult(
                concentrations={},
                residual_fraction=1.0,
                raw_closure_sum=0.0,
                closure_diagnostic=1.0,
                mode=mode,
                alpha_residual=alpha_residual,
                experimental_factor=experimental_factor,
            )

        # Calibrated raw sum: rho_s = rel_C_s / F.  Comparing this against 1
        # (rather than the bare raw_sum against the literal 1.0) makes the
        # deficit/diagnostic invariant to a global intensity rescaling, since
        # F carries the same intensity units as rel_C_s and scales with it.
        # With the default F = 1 this reproduces the unit-calibrated behavior.
        calibrated_sum = raw_sum / experimental_factor
        closure_diagnostic = abs(calibrated_sum - 1.0)

        # --- 2. Estimate residual fraction --------------------------------
        if mode == "dirichlet":
            n_detected = len(raw_concentrations)
            sum_alpha_detected = n_detected * (alpha_detected - 1.0)
            alpha_res_minus_one = max(alpha_residual - 1.0, 0.0)
            denom = calibrated_sum + sum_alpha_detected + alpha_res_minus_one
            if denom > 0:
                residual = alpha_res_minus_one / denom
            else:
                residual = 0.0
            residual = max(0.0, min(residual, 1.0))
        else:
            deficit = 1.0 - calibrated_sum
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
