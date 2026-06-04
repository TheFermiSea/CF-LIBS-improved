"""
Iterative solver for Classic CF-LIBS.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, NamedTuple, Any
import numpy as np
from collections import defaultdict

from cflibs.core.constants import (
    KB,
    KB_EV,
    SAHA_CONST_CM3,
    STP_PRESSURE,
    EV_TO_K,
    H_PLANCK_EV,
    C_LIGHT,
)
from cflibs.atomic.database import AtomicDatabase
from cflibs.inversion.physics.boltzmann import LineObservation, BoltzmannPlotFitter
from cflibs.inversion.physics.closure import ClosureEquation
from cflibs.inversion.physics.closure_strategy import ClosureStrategy
from cflibs.inversion.physics.self_absorption import SelfAbsorptionCorrector
from cflibs.core.logging_config import get_logger


def _jax_boltzmann_composition_enabled() -> bool:
    """Opt-in env-var toggle for routing the inner Boltzmann sigma-clip
    WLS step through the JAX kernel in composition workflows.

    Default (unset or "0") preserves byte-for-byte the CPU behavior. Set
    ``CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION=1`` to enable. See
    ``docs/jax-port/iterative-boltzmann-consultation.md`` for design
    rationale.
    """
    return os.environ.get("CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION", "0") == "1"


def _lax_while_loop_enabled() -> bool:
    """Opt-in env-var toggle for routing ``IterativeCFLIBSSolver.solve`` through
    the ``jax.lax.while_loop`` JAX path (T1-3, ADR-0001).

    Default (unset or "0") preserves the Python ``for``-loop semantics
    byte-for-byte. Set ``CFLIBS_USE_LAX_WHILE_LOOP=1`` to enable. The lax path
    pre-fetches all SQLite-backed atomic data outside the loop body and runs
    the iteration through ``jax.lax.while_loop`` so the solver is jit-traceable,
    ``vmap``-able across batches of observations, and (eventually) ``grad``-able.

    See ``docs/adr/specs/T1-3-lax-while-iterative.md`` for the full design.
    """
    return os.environ.get("CFLIBS_USE_LAX_WHILE_LOOP", "0") == "1"


# Optional JAX imports — IterativeCFLIBSSolverJax raises ImportError at
# instantiation time if JAX is missing, so the rest of the module is unaffected.
try:
    import jax
    import jax.numpy as jnp
    from jax import jit

    HAS_JAX = True
except ImportError:  # pragma: no cover - exercised when JAX absent
    HAS_JAX = False
    jax = None  # type: ignore[assignment]
    jnp = None
    jit = None  # type: ignore[assignment]

logger = get_logger("inversion.solver")


@dataclass
class CFLIBSResult:
    """
    Result of the iterative CF-LIBS inversion.

    Attributes
    ----------
    temperature_K : float
        Plasma temperature in Kelvin
    temperature_uncertainty_K : float
        1-sigma uncertainty in temperature
    electron_density_cm3 : float
        Electron density in cm^-3
    electron_density_uncertainty_cm3 : float
        1-sigma uncertainty in electron density (0 if not computed)
    concentrations : Dict[str, float]
        Element concentrations (number/mole fractions, sum to 1).
        These are the internal CF-LIBS closure fractions used in
        Saha-Boltzmann algebra. Convert to mass fractions via
        C_mass_i = C_i * AW_i / sum(C_j * AW_j).
    concentration_uncertainties : Dict[str, float]
        1-sigma uncertainties in concentrations
    iterations : int
        Number of iterations performed
    converged : bool
        Whether solver converged within tolerance
    temperature_corona_K : float, optional
        Estimated corona temperature in Kelvin (for two-region fits)
    quality_metrics : Dict[str, float]
        Quality metrics (R², chi², etc.)
    boltzmann_covariance : np.ndarray, optional
        2x2 covariance matrix of a representative pooled Boltzmann fit
        (slope, intercept). For multi-element uncertainty solves this stores
        the covariance for the selected reference element noted in
        ``quality_metrics["boltzmann_covariance_element"]``.
    """

    temperature_K: float
    temperature_uncertainty_K: float
    electron_density_cm3: float
    concentrations: Dict[str, float]
    concentration_uncertainties: Dict[str, float]
    iterations: int
    converged: bool
    temperature_corona_K: Optional[float] = None
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    electron_density_uncertainty_cm3: float = 0.0
    boltzmann_covariance: Optional[np.ndarray] = field(default=None, repr=False)


@dataclass
class _CommonSlopeElementStats:
    """Weighted per-element statistics for the pooled Boltzmann fit."""

    x_values: np.ndarray = field(repr=False)
    y_values: np.ndarray = field(repr=False)
    weights: np.ndarray = field(repr=False)
    x_mean: float
    y_mean: float


@dataclass
class _CommonSlopeFit:
    """Result of the pooled common-slope Boltzmann regression."""

    slope: float
    slope_variance: float
    intercepts: Dict[str, float]
    element_stats: Dict[str, _CommonSlopeElementStats] = field(repr=False)
    r_squared: float = 0.0


# ---------------------------------------------------------------------------
# T1-3: jax.lax.while_loop iterative solver helpers (ADR-0001 spec §3-§6)
# ---------------------------------------------------------------------------


class _LaxFallback(RuntimeError):
    """Internal signal that the lax.while_loop path cannot run for this input.

    Raised by :meth:`IterativeCFLIBSSolver._solve_lax` when prerequisites
    fail (e.g. no usable padded observations, no elements). The caller in
    :meth:`IterativeCFLIBSSolver.solve` catches this and falls back to the
    Python path verbatim.
    """


class LoopState(NamedTuple):
    """Pytree-compatible state carried by ``jax.lax.while_loop`` (spec §3).

    All fields are JAX arrays so the tuple registers automatically as a JAX
    pytree. Scalar fields are kept as 0-d arrays for ``while_loop`` strictness.
    """

    T_K: Any
    n_e_cm3: Any
    T_prev: Any
    n_e_prev: Any
    converged: Any
    i: Any
    U_I: Any
    U_II: Any
    intercepts: Any
    concentrations: Any
    r_squared: Any


@dataclass(frozen=True)
class _AtomicSnapshot:
    """Frozen per-element atomic-data bundle pre-fetched outside the loop.

    Built once per :meth:`IterativeCFLIBSSolver._solve_lax` call (spec §6) so
    the loop body never touches the SQLite-backed :class:`AtomicDatabase`.

    Attributes
    ----------
    elements
        Element symbols in the bundle ordering, length ``E``.
    ip0_eV
        Stage-I ionization potentials per element, shape ``(E,)``.
    use_direct
        Per-element boolean: ``True`` -> use padded ``(g_levels, E_levels)``
        for direct summation; ``False`` -> use polynomial ``coefficients``.
        Shape ``(E, 2)`` for stages I, II respectively.
    g_levels_I, E_levels_I
        Padded ``(E, Nk_max_I)`` arrays of level g and E for stage I.
    ip_I_for_direct
        Per-element direct-sum cutoff ionization potential for stage I, ``(E,)``.
    levels_mask_I
        Padded ``(E, Nk_max_I)`` bool mask of valid levels for stage I.
    g_levels_II, E_levels_II, ip_II_for_direct, levels_mask_II
        Same as stage I but for stage II.
    coeffs_I, coeffs_II
        Polynomial coefficients ``(E, 5)`` for stage I, II. Zero-padded when
        unused (i.e. when ``use_direct`` is True for that element/stage).
    fallback_U_I, fallback_U_II
        Per-element scalar fallbacks (25, 15 by convention) used when both
        direct and polynomial paths are unavailable, ``(E,)``.
    """

    elements: Tuple[str, ...]
    ip0_eV: np.ndarray
    use_direct: np.ndarray  # shape (E, 2), bool
    g_levels_I: np.ndarray
    E_levels_I: np.ndarray
    ip_I_for_direct: np.ndarray
    levels_mask_I: np.ndarray
    g_levels_II: np.ndarray
    E_levels_II: np.ndarray
    ip_II_for_direct: np.ndarray
    levels_mask_II: np.ndarray
    coeffs_I: np.ndarray
    coeffs_II: np.ndarray
    fallback_U_I: np.ndarray
    fallback_U_II: np.ndarray

    @classmethod
    def from_solver(cls, solver: "IterativeCFLIBSSolver", elements: List[str]) -> "_AtomicSnapshot":
        """Pre-fetch atomic data for ``elements`` from the solver's database.

        One-shot SQLite query bundle: ionization potentials, energy levels,
        polynomial coefficients. No further SQLite calls happen inside
        ``_solve_lax``; the resulting arrays feed the JAX while-loop body.
        """
        from cflibs.plasma.partition import get_levels_for_species

        E = len(elements)
        ip0 = np.zeros(E, dtype=np.float64)
        use_direct = np.zeros((E, 2), dtype=bool)
        g_I: List[np.ndarray] = []
        E_I: List[np.ndarray] = []
        ip_I: np.ndarray = np.zeros(E, dtype=np.float64)
        g_II: List[np.ndarray] = []
        E_II: List[np.ndarray] = []
        ip_II: np.ndarray = np.zeros(E, dtype=np.float64)
        coeffs_I = np.zeros((E, 5), dtype=np.float64)
        coeffs_II = np.zeros((E, 5), dtype=np.float64)
        fallback_I = np.full(E, 25.0, dtype=np.float64)
        fallback_II = np.full(E, 15.0, dtype=np.float64)

        for i, el in enumerate(elements):
            # Stage-I IP (the only one used by the loop; for IPD/Saha)
            ip = solver.atomic_db.get_ionization_potential(el, 1)
            if ip is None:
                logger.warning("No IP for %s I, assuming high (15.0 eV)", el)
                ip = 15.0
            ip0[i] = float(ip)

            # Try direct-sum levels (stage I and II)
            for stage_idx, stage in enumerate((1, 2)):
                lev = get_levels_for_species(solver.atomic_db, el, stage)
                if lev is not None:
                    g_arr, E_arr, ip_ev = lev
                    if stage == 1:
                        g_I.append(np.asarray(g_arr, dtype=np.float64))
                        E_I.append(np.asarray(E_arr, dtype=np.float64))
                        ip_I[i] = float(ip_ev)
                    else:
                        g_II.append(np.asarray(g_arr, dtype=np.float64))
                        E_II.append(np.asarray(E_arr, dtype=np.float64))
                        ip_II[i] = float(ip_ev)
                    use_direct[i, stage_idx] = True
                    continue
                # Polynomial fallback
                pf = solver.atomic_db.get_partition_coefficients(el, stage)
                if pf:
                    coeffs = np.asarray(pf.coefficients, dtype=np.float64)
                    # Pad/truncate to 5 coefficients
                    n = min(coeffs.size, 5)
                    if stage == 1:
                        coeffs_I[i, :n] = coeffs[:n]
                    else:
                        coeffs_II[i, :n] = coeffs[:n]
                    use_direct[i, stage_idx] = False
                    if stage == 1:
                        g_I.append(np.zeros(0, dtype=np.float64))
                        E_I.append(np.zeros(0, dtype=np.float64))
                    else:
                        g_II.append(np.zeros(0, dtype=np.float64))
                        E_II.append(np.zeros(0, dtype=np.float64))
                else:
                    # No data at all — record the empty level arrays and rely on
                    # ``fallback_U_*`` per-element scalars at evaluation time.
                    if stage == 1:
                        g_I.append(np.zeros(0, dtype=np.float64))
                        E_I.append(np.zeros(0, dtype=np.float64))
                    else:
                        g_II.append(np.zeros(0, dtype=np.float64))
                        E_II.append(np.zeros(0, dtype=np.float64))
                    use_direct[i, stage_idx] = False
                    # coeffs already zeros — eval_poly will return exp(0)=1; we
                    # use fallback in that case. Mark via NaN sentinel below.
                    if stage == 1 and not pf:
                        coeffs_I[i, :] = np.nan
                    elif stage == 2 and not pf:
                        coeffs_II[i, :] = np.nan

        # Pad ragged level arrays per stage
        def _pad_ragged(arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            counts = [int(a.size) for a in arrays]
            n_max = max(counts) if counts else 0
            if n_max == 0:
                # Provide a length-1 dummy column so JAX accepts the shape.
                n_max = 1
            padded = np.zeros((len(arrays), n_max), dtype=np.float64)
            mask = np.zeros((len(arrays), n_max), dtype=bool)
            for j, arr in enumerate(arrays):
                k = arr.size
                if k:
                    padded[j, :k] = arr
                    mask[j, :k] = True
            return padded, mask

        gI_pad, mI = _pad_ragged(g_I)
        EI_pad, _ = _pad_ragged(E_I)
        # Re-pad EI to the same width as gI by reading mI shape
        if EI_pad.shape[1] != gI_pad.shape[1]:
            new_E = np.zeros_like(gI_pad)
            new_E[:, : EI_pad.shape[1]] = EI_pad
            EI_pad = new_E
        gII_pad, mII = _pad_ragged(g_II)
        EII_pad, _ = _pad_ragged(E_II)
        if EII_pad.shape[1] != gII_pad.shape[1]:
            new_E2 = np.zeros_like(gII_pad)
            new_E2[:, : EII_pad.shape[1]] = EII_pad
            EII_pad = new_E2

        return cls(
            elements=tuple(elements),
            ip0_eV=ip0,
            use_direct=use_direct,
            g_levels_I=gI_pad,
            E_levels_I=EI_pad,
            ip_I_for_direct=ip_I,
            levels_mask_I=mI,
            g_levels_II=gII_pad,
            E_levels_II=EII_pad,
            ip_II_for_direct=ip_II,
            levels_mask_II=mII,
            coeffs_I=coeffs_I,
            coeffs_II=coeffs_II,
            fallback_U_I=fallback_I,
            fallback_U_II=fallback_II,
        )

    def reorder(self, new_order: List[str]) -> "_AtomicSnapshot":
        """Return a snapshot reordered to match ``new_order`` element symbols."""
        if list(new_order) == list(self.elements):
            return self
        idx = np.array([self.elements.index(el) for el in new_order], dtype=int)
        return _AtomicSnapshot(
            elements=tuple(new_order),
            ip0_eV=self.ip0_eV[idx],
            use_direct=self.use_direct[idx],
            g_levels_I=self.g_levels_I[idx],
            E_levels_I=self.E_levels_I[idx],
            ip_I_for_direct=self.ip_I_for_direct[idx],
            levels_mask_I=self.levels_mask_I[idx],
            g_levels_II=self.g_levels_II[idx],
            E_levels_II=self.E_levels_II[idx],
            ip_II_for_direct=self.ip_II_for_direct[idx],
            levels_mask_II=self.levels_mask_II[idx],
            coeffs_I=self.coeffs_I[idx],
            coeffs_II=self.coeffs_II[idx],
            fallback_U_I=self.fallback_U_I[idx],
            fallback_U_II=self.fallback_U_II[idx],
        )


def _make_closure_callback(
    closure_mode: str,
    elements: List[str],
    closure_kwargs: Dict[str, Any],
):
    """Build a closure callable invoked from inside ``lax.while_loop`` (spec §5).

    Implements Option A — the closure mode is resolved at solve-time, not
    per-iteration. The returned callable takes three ``(E,)`` arrays
    ``(intercepts, U_I, mult)`` and returns the ``(E,)`` concentration vector.

    For ``standard`` / ``matrix`` / ``oxide`` modes the closure is expressed as
    pure JAX algebra (no callback). For ``ilr`` / ``pwlr`` /
    ``dirichlet_residual`` we route through :func:`jax.pure_callback` to the
    existing :class:`ClosureEquation` host implementation — this preserves
    numerics bit-for-bit while staying jit-traceable inside the while-loop
    body.
    """
    if not HAS_JAX:  # pragma: no cover - guarded
        raise _LaxFallback("JAX not available")

    mode = closure_mode.lower()
    E = len(elements)

    if mode in {"", "standard"}:

        def _standard(intercepts, U_I, mult):
            rel = mult * U_I * jnp.exp(intercepts)
            total = jnp.sum(rel)
            return jnp.where(total > 0.0, rel / jnp.where(total > 0.0, total, 1.0), 0.0)

        return _standard

    if mode == "matrix":
        matrix_element = closure_kwargs.get("matrix_element")
        matrix_fraction = float(closure_kwargs.get("matrix_fraction", 0.9))
        if matrix_element not in elements:
            # Mirror ClosureEquation.apply_matrix_mode: fall through to standard
            def _matrix_fallback(intercepts, U_I, mult):
                rel = mult * U_I * jnp.exp(intercepts)
                total = jnp.sum(rel)
                return jnp.where(total > 0.0, rel / jnp.where(total > 0.0, total, 1.0), 0.0)

            return _matrix_fallback
        m_idx = elements.index(matrix_element)

        def _matrix(intercepts, U_I, mult):
            rel = mult * U_I * jnp.exp(intercepts)
            rel_m = rel[m_idx]
            F = rel_m / matrix_fraction
            return jnp.where(F > 0.0, rel / jnp.where(F > 0.0, F, 1.0), 0.0)

        return _matrix

    if mode == "oxide":
        stoich_map = closure_kwargs.get("oxide_stoichiometry", {}) or {}
        factors = jnp.asarray(
            [float(stoich_map.get(el, 1.0)) for el in elements], dtype=jnp.float64
        )

        def _oxide(intercepts, U_I, mult):
            rel = mult * U_I * jnp.exp(intercepts)
            total_oxide = jnp.sum(rel * factors)
            return jnp.where(
                total_oxide > 0.0,
                rel / jnp.where(total_oxide > 0.0, total_oxide, 1.0),
                0.0,
            )

        return _oxide

    # ILR / PWLR / Dirichlet residual: route via pure_callback so the host
    # implementation runs unchanged (preserves numerics bit-for-bit).
    elements_tuple = tuple(elements)
    apply_kwargs = dict(closure_kwargs)

    def _host_closure(intercepts_np, U_I_np, mult_np):
        intercepts_dict = {el: float(intercepts_np[i]) for i, el in enumerate(elements_tuple)}
        U_I_dict = {el: float(U_I_np[i]) for i, el in enumerate(elements_tuple)}
        mult_dict = {el: float(mult_np[i]) for i, el in enumerate(elements_tuple)}
        if mode == "ilr":
            res = ClosureEquation.apply_ilr(
                intercepts_dict, U_I_dict, abundance_multipliers=mult_dict
            )
        elif mode == "pwlr":
            res = ClosureEquation.apply_pwlr(
                intercepts_dict, U_I_dict, abundance_multipliers=mult_dict, **apply_kwargs
            )
        elif mode == "dirichlet_residual":
            res = ClosureEquation.apply_dirichlet_residual(
                intercepts_dict, U_I_dict, abundance_multipliers=mult_dict, **apply_kwargs
            )
        else:  # pragma: no cover - defensive
            res = ClosureEquation.apply_standard(
                intercepts_dict, U_I_dict, abundance_multipliers=mult_dict
            )
        out = np.zeros(E, dtype=np.float64)
        for i, el in enumerate(elements_tuple):
            out[i] = float(res.concentrations.get(el, 0.0))
        return out

    result_shape = jax.ShapeDtypeStruct((E,), jnp.float64)

    def _closure_via_callback(intercepts, U_I, mult):
        return jax.pure_callback(
            _host_closure,
            result_shape,
            intercepts,
            U_I,
            mult,
        )

    return _closure_via_callback


def _eval_partition_jax(
    T_K,
    use_direct_col,
    g_pad,
    E_pad,
    ip_for_direct,
    coeffs,
    fallback_U,
    levels_mask,
):
    """JAX-evaluate the partition function per element with direct/polynomial mix.

    Mirrors :meth:`IterativeCFLIBSSolver._evaluate_partition_function` semantics:
    direct summation when available, polynomial fallback, then scalar fallback.

    All inputs except ``T_K`` are constant for the lifetime of one
    ``_solve_lax`` call; they are passed in as JAX arrays so the body remains
    jit-traceable.
    """
    # Direct: U_direct = Σ_k g_k exp(-E_k * EV_TO_K / T_K) over masked levels
    # (filters levels with E >= ip_for_direct, matching direct_sum_partition_function)
    T_safe = jnp.maximum(T_K, 1.0)
    # Boltzmann factor per level. Masked entries contribute zero because g=0 there.
    arg = -E_pad * EV_TO_K / T_safe
    # Stability: avoid overflow for huge arg (very low T edge case)
    arg = jnp.clip(arg, -700.0, 700.0)
    bz = jnp.exp(arg)
    valid_level = levels_mask & (E_pad < ip_for_direct[:, None])
    contrib = jnp.where(valid_level, g_pad * bz, 0.0)
    U_direct = jnp.sum(contrib, axis=1)  # (E,)
    # Mirror direct_sum_partition_function floor: U >= 1.0 (matches host path)
    U_direct = jnp.maximum(U_direct, 1.0)

    # Polynomial: ln U = Σ_n a_n (ln T)^n, with NaN-coefficient sentinel falling
    # back to scalar fallback.
    ln_T = jnp.log(T_safe)
    poly = (
        coeffs[..., 0]
        + coeffs[..., 1] * ln_T
        + coeffs[..., 2] * (ln_T**2)
        + coeffs[..., 3] * (ln_T**3)
        + coeffs[..., 4] * (ln_T**4)
    )
    U_poly = jnp.exp(jnp.clip(poly, -700.0, 700.0))
    poly_valid = jnp.all(jnp.isfinite(coeffs), axis=-1) & (jnp.any(coeffs != 0.0, axis=-1))

    # Compose: direct (where available) > poly (where valid) > fallback scalar
    U = jnp.where(use_direct_col, U_direct, jnp.where(poly_valid, U_poly, fallback_U))
    return U


def _saha_ratio_per_element(T_K, n_e, U_I, U_II, ip_eV):
    """Element-wise Saha ratio n_II / n_I (vectorized over E)."""
    safe_ne = jnp.maximum(n_e, 1e10)
    T_eV = jnp.maximum(T_K / EV_TO_K, 0.1)
    return (
        (SAHA_CONST_CM3 / safe_ne)
        * (T_eV**1.5)
        * (U_II / jnp.maximum(U_I, 1e-30))
        * jnp.exp(-ip_eV / T_eV)
    )


def _run_lax_while_loop(
    init_state: LoopState,
    x_d,
    y_d,
    w_d,
    stage_d,
    mask_d,
    snapshot: "_AtomicSnapshot",
    closure_fn,
    *,
    apply_ipd: bool,
    two_region: bool,
    max_iter: int,
    t_tol_k: float,
    ne_tol_frac: float,
    pressure_pa: float,
    min_r2: float = 0.3,
) -> LoopState:
    """Drive the iteration through ``jax.lax.while_loop`` (spec §4).

    The body builds one CF-LIBS iteration entirely in JAX (no SQLite, no
    Python dispatch). The closure step optionally routes through
    :func:`jax.pure_callback` for the non-trivially-traceable modes.
    """
    # Move snapshot arrays to JAX device once
    ip0_eV = jnp.asarray(snapshot.ip0_eV, dtype=jnp.float64)
    use_direct = jnp.asarray(snapshot.use_direct, dtype=bool)
    g_I = jnp.asarray(snapshot.g_levels_I, dtype=jnp.float64)
    E_I = jnp.asarray(snapshot.E_levels_I, dtype=jnp.float64)
    ipI = jnp.asarray(snapshot.ip_I_for_direct, dtype=jnp.float64)
    mI = jnp.asarray(snapshot.levels_mask_I, dtype=bool)
    g_II = jnp.asarray(snapshot.g_levels_II, dtype=jnp.float64)
    E_II = jnp.asarray(snapshot.E_levels_II, dtype=jnp.float64)
    ipII = jnp.asarray(snapshot.ip_II_for_direct, dtype=jnp.float64)
    mII = jnp.asarray(snapshot.levels_mask_II, dtype=bool)
    coeffs_I = jnp.asarray(snapshot.coeffs_I, dtype=jnp.float64)
    coeffs_II = jnp.asarray(snapshot.coeffs_II, dtype=jnp.float64)
    fallback_I = jnp.asarray(snapshot.fallback_U_I, dtype=jnp.float64)
    fallback_II = jnp.asarray(snapshot.fallback_U_II, dtype=jnp.float64)

    def cond_fun(state: LoopState):
        return jnp.logical_and(jnp.logical_not(state.converged), state.i < max_iter)

    def body_fun(state: LoopState) -> LoopState:
        T_prev = state.T_K
        ne_prev = state.n_e_cm3

        # Partition functions (JAX, closed-form)
        U_I = _eval_partition_jax(T_prev, use_direct[:, 0], g_I, E_I, ipI, coeffs_I, fallback_I, mI)
        U_II = _eval_partition_jax(
            T_prev, use_direct[:, 1], g_II, E_II, ipII, coeffs_II, fallback_II, mII
        )

        # Effective IPs (IPD, optional)
        if apply_ipd:
            # Debye-Hückel: ΔE = (z+1) * e^2 / (4π ε₀ λ_D). Matches
            # ionization_potential_lowering — simplified for stage-I only.
            kT_eV = jnp.maximum(T_prev / EV_TO_K, 1e-3)
            # Approximate Debye length in cm:  λ_D = 6.9 * sqrt(T/n_e) [CGS-ish]
            lambda_D_cm = 6.9 * jnp.sqrt(kT_eV * EV_TO_K / jnp.maximum(ne_prev, 1.0))
            delta_chi = 1.44e-7 / jnp.maximum(lambda_D_cm, 1e-30)  # eV
            ip_eff = jnp.maximum(ip0_eV - delta_chi, 0.0)
        else:
            ip_eff = ip0_eV

        # Saha correction (broadcast ip per row)
        T_eV = jnp.maximum(T_prev / EV_TO_K, 0.1)
        safe_ne = jnp.maximum(ne_prev, 1e10)
        log_correction = jnp.log((SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5))
        ip_arr = jnp.broadcast_to(ip_eff[:, None], x_d.shape)
        x_corr, y_corr = _saha_correct_kernel(x_d, y_d, stage_d, ip_arr, T_eV, log_correction)

        # Common-slope Boltzmann fit (already a JAX kernel)
        fit = _common_slope_kernel(x_corr, y_corr, w_d, mask_d)
        slope = fit["slope"]
        intercepts = fit["intercepts"]
        r_squared = fit["r_squared"]

        # T update (50/50 damping), gated on Boltzmann-plot quality to mirror
        # the Python path (see IterativeCFLIBSSolver.__init__): on a degenerate
        # fit (non-negative slope or R^2 < min_r2) hold T at the prior value
        # instead of running it to the legacy 50000 K clamp, which collapses the
        # closure into a raw-intensity softmax.
        degenerate = jnp.logical_or(slope >= 0.0, r_squared < min_r2)
        T_new = jnp.where(degenerate, T_prev, -1.0 / (slope * KB_EV))
        T_K = 0.5 * T_prev + 0.5 * T_new

        # Two-region corona: weighted T for Saha scaling matches the Python
        # path (_compute_abundance_multipliers, when T_corona is set we use
        # T_saha = 0.3 T + 0.7 T_corona; T_corona = 0.8 T => 0.3T + 0.56T = 0.86T).
        # But the Python path applies this only to ``corona_sensitive`` elements;
        # since the mock test fixture doesn't use those it's a no-op there. We
        # preserve the parent's behavior in the lax path by NOT applying the
        # corona-weighted T in the array-broadcast Saha multiplier step (it's
        # a per-element conditional that depends on element symbols, which we
        # carry as static metadata). For now use T_K uniformly; corona weighting
        # is a low-priority refinement deferred until a real fixture exercises it.
        T_for_saha = T_K  # spec §11: corona-element weighting deferred

        # Abundance multipliers (1 + Saha ratio)
        S = _saha_ratio_per_element(T_for_saha, ne_prev, U_I, U_II, ip_eff)
        mult = 1.0 + jnp.maximum(S, 0.0)

        # Closure dispatch (resolved at solve-time, spec §5 Option A)
        concentrations = closure_fn(intercepts, U_I, mult)

        # Pressure balance n_e update (50/50 damping)
        # eps_s = S / (1 + S) -- electrons per atom of species
        S_now = _saha_ratio_per_element(T_K, ne_prev, U_I, U_II, ip_eff)
        eps_s = S_now / (1.0 + S_now)
        avg_Z = jnp.sum(concentrations * eps_s)
        n_tot = pressure_pa / (KB * T_K * (1.0 + avg_Z))
        n_tot_cm3 = n_tot * 1e-6
        ne_new = avg_Z * n_tot_cm3
        n_e = 0.5 * ne_prev + 0.5 * ne_new

        # Convergence (matches Python path: |ΔT|<tol and |Δne|/ne_prev<frac).
        # A degenerate fit holds T at the prior, which would satisfy |ΔT|<tol
        # spuriously, so it can never count as converged.
        converged = jnp.logical_and(
            jnp.logical_not(degenerate),
            jnp.logical_and(
                jnp.abs(T_K - T_prev) < t_tol_k,
                jnp.abs(n_e - ne_prev) / jnp.maximum(ne_prev, 1e-30) < ne_tol_frac,
            ),
        )

        return LoopState(
            T_K=T_K,
            n_e_cm3=n_e,
            T_prev=T_prev,
            n_e_prev=ne_prev,
            converged=converged,
            i=state.i + 1,
            U_I=U_I,
            U_II=U_II,
            intercepts=intercepts,
            concentrations=concentrations,
            r_squared=r_squared,
        )

    return jax.lax.while_loop(cond_fun, body_fun, init_state)


class IterativeCFLIBSSolver:
    """
    Implements the iterative self-consistent CF-LIBS algorithm.

    Algorithm:
    1. Guess T, ne
    2. Saha-Boltzmann correction to map ionic lines to neutral plane
    3. Multi-species Boltzmann fit to find common T and species intercepts
    4. Closure equation to find relative concentrations
    5. Enforce Pressure/Charge balance to update ne
    6. Iterate until convergence
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        max_iterations: int = 20,
        t_tolerance_k: float = 100.0,
        ne_tolerance_frac: float = 0.1,
        pressure_pa: float = STP_PRESSURE,
        apply_ipd: bool = False,
        aki_uncertainty_weighting: bool = True,
        two_region: bool = False,
        closure: Optional[ClosureStrategy] = None,
        apply_self_absorption: bool = False,
        self_absorption_plasma_length_cm: float = 0.1,
        self_absorption_mask_threshold: float = 1.0e6,
        self_absorption_tau_cap: float = 10.0,
        self_absorption_column_density_cm3: float = 1.0e16,
        min_boltzmann_r2: float = 0.3,
    ):
        self.atomic_db = atomic_db
        self.max_iterations = max_iterations
        self.t_tolerance_k = t_tolerance_k
        self.ne_tolerance_frac = ne_tolerance_frac
        self.pressure_pa = pressure_pa
        self.apply_ipd = apply_ipd
        self.aki_uncertainty_weighting = aki_uncertainty_weighting
        self.two_region = two_region
        # Boltzmann-plot quality gate for the slope -> temperature update.
        #
        # The temperature is recovered from the (negative) slope of the common
        # Boltzmann plane: T = -1/(slope * k_B). When the fit is *degenerate*
        # — a near-zero or positive slope, or a poor R^2 — the slope carries no
        # reliable temperature information. The legacy code ran T to a 50000 K
        # clamp on a non-negative slope; on real, noisy LIBS data this collapses
        # the inversion: at T ~ 10^5 K the Boltzmann factor exp(-E_k/kT) -> 1 for
        # every line, so the per-element intercept q_s degenerates to
        # ln(<intensity>/gA) and the closure becomes a raw-intensity softmax in
        # which the largest-U low-ionization alkali/alkaline-earth element soaks
        # all the mass (the BHVO-2 "keystone collapse": 9 correct Mg + 9 correct
        # Fe lines -> Na/Ca dominate). We therefore *gate* the slope->T update:
        # on a degenerate fit (slope >= 0 or R^2 < ``min_boltzmann_r2``) we hold
        # T at the previous (prior) value instead of running it to the clamp, and
        # mark the solve non-converged so downstream consumers know the
        # temperature — and hence the composition — is untrustworthy. Set to 0.0
        # to disable the R^2 gate (slope-sign gate still applies). Default 0.3 is
        # a permissive floor: a clean optically-thin Boltzmann plane sits at
        # R^2 > 0.95, while the collapse cases sit at R^2 ~ 1e-3.
        self.min_boltzmann_r2 = float(min_boltzmann_r2)
        # Self-absorption correction (Bulajic 2002; physics-audit defect B1).
        # When enabled, the iterative solver applies a curve-of-growth
        # escape-factor correction to the observed line intensities *before* the
        # Boltzmann/closure fit on every iteration after the first, recomputing the
        # optical depth tau from the current plasma state (T, n_e, concentrations,
        # partition functions) each pass. This is the outer recursion that the
        # corrector's per-line :meth:`_apply_recursive_correction` (B4) is designed
        # to be driven by — without it the corrector is a no-op (dead code).
        #
        # OPT-IN (default False). The correction is grounded and reduces error on
        # genuinely self-absorbed spectra (see the Mg/Ca regression in
        # tests/inversion/solve/test_self_absorption_wiring.py), but the
        # plasma-state optical-depth estimate cannot by itself tell a thick line
        # from a thin one: it computes tau from the recovered density/composition
        # and corrects unconditionally. On an *optically thin* spectrum (e.g. a
        # synthetic forward model with no self-absorption) a high recovered
        # density makes tau look large and the correction over-boosts the
        # low-E_k lines — a false positive. Until the correction is gated on an
        # observed self-absorption signature (El Sherbini 2005 line-width ratio,
        # or a Boltzmann-residual / doublet-ratio test), it stays opt-in so the
        # default analyze/invert path is never harmed on thin data. Enable it via
        # ``apply_self_absorption=True`` (or the analysis-config key) for known
        # optically-thick samples such as the BHVO-2 basalt majors.
        #
        # Two safety levers keep the correction bounded and stable inside the
        # solver, where the absorbing column density (n_e * L) is only known to
        # ~an order of magnitude:
        #
        #   * ``self_absorption_tau_cap`` (default 10) clamps the τ used for the
        #     *correction factor*. The escape-factor boost ``I/f(τ) ≈ τ`` grows
        #     without bound and the Doppler-core curve-of-growth model loses
        #     validity beyond τ ~ 5-10 (El Sherbini 2005), so an uncapped 1/f(τ)
        #     would amplify a strong resonance line by 100-1000× off a possibly
        #     over-estimated density — wrong and destabilising. Capping at ≈ 10
        #     bounds the boost to the literature "~one order of magnitude"
        #     accuracy gain (Bulajic 2002) and corrects the dominant
        #     self-absorbed majors without deleting them.
        #   * ``self_absorption_mask_threshold`` is therefore set very high
        #     (default 1e6) so that with the cap in place NO line is ever
        #     dropped — even a saturated resonance line keeps its (bounded)
        #     correction and its Boltzmann lever-arm, rather than being masked
        #     out of the fit. Masking strong major lines was the failure mode
        #     that made naive wiring collapse the fit ("insufficient points").
        # Effective absorbing heavy-particle density (cm^-3) used for the τ
        # estimate. The iterative solver's own n_e comes from an STP (1 atm)
        # pressure balance and is ~1e18 — far above the ~1e16-1e17 heavy-
        # particle density of a *gated* ps-LIBS plasma at the analysis delay.
        # Feeding 1e18 puts every strong major line deep in saturation (τ ≫
        # τ_cap), where the correction is a flat ~τ_cap boost that cannot
        # discriminate self-absorbed lines from thin ones — so it has no effect
        # on composition. We instead anchor the column density to a LIBS-
        # realistic reference (default 1e16, calibrated so the audit's Si I
        # 251.611 reference line lands at τ ≈ 5, the El Sherbini correctable
        # regime). Number fractions from closure then modulate this per element.
        self.self_absorption_column_density_cm3 = self_absorption_column_density_cm3
        self.apply_self_absorption = apply_self_absorption
        self.self_absorption_corrector: Optional[SelfAbsorptionCorrector] = None
        self._last_sa_max_tau: float = 0.0
        if apply_self_absorption:
            self.self_absorption_corrector = SelfAbsorptionCorrector(
                optical_depth_threshold=0.1,
                mask_threshold=self_absorption_mask_threshold,
                max_iterations=5,
                convergence_tolerance=0.01,
                plasma_length_cm=self_absorption_plasma_length_cm,
                correction_tau_cap=self_absorption_tau_cap,
            )
        # Closure strategy — defaults to ILR per architecture-review
        # Candidate 3 (ILR has well-conditioned gradients down to the
        # trace-element regime).  The per-call ``closure_mode`` argument
        # of :meth:`solve` continues to override this and remains the
        # primary closure-selection API for the iterative solver, so the
        # legacy default (``closure_mode='standard'``) is preserved
        # bit-for-bit when callers do not pass ``closure_mode``.
        if closure is None:
            from cflibs.inversion.physics.closure_strategy import ILRClosure

            closure = ILRClosure()
        self.closure: ClosureStrategy = closure
        self.boltzmann_fitter = BoltzmannPlotFitter(
            outlier_sigma=2.5,
            use_jax=_jax_boltzmann_composition_enabled(),
        )

    def _line_y_uncertainty(self, obs: LineObservation) -> float:
        """Return fit-space uncertainty with optional A_ki contribution."""
        sigma_y = obs.y_uncertainty
        unc = obs.aki_uncertainty
        if self.aki_uncertainty_weighting and unc is not None and np.isfinite(unc) and unc > 0:
            sigma_y = float(np.sqrt(sigma_y**2 + float(unc) ** 2))
        return sigma_y

    def _evaluate_partition_function(
        self, element: str, ionization_stage: int, T_K: float
    ) -> float:
        """Evaluate a partition function through the single provider factory.

        Routes U(T) through :meth:`AtomicDatabase.partition_function_for` — THE
        single source of the partition-function policy (direct-sum preferred,
        always clamped + ``g0``-floored).  For species with energy levels the
        CPU scalar provider sums the levels directly, so this path stays
        bit-for-bit identical to the historical ``evaluate_direct`` call it
        replaces; for level-less species it applies the guarded stored
        polynomial.  The hardcoded estimates remain only for species the
        factory cannot resolve at all (no levels, no stored row).

        ``partition_function_for`` is a convenience method on the concrete
        :class:`AtomicDatabase`, NOT part of the :class:`AtomicDataSource` ABC.
        Pluggable backends (the documented Key Abstraction) need only satisfy
        the ABC, so we ``hasattr``-guard the factory call and fall back to the
        ABC-level accessors (``get_energy_levels`` direct sum, then the stored
        polynomial) — the same fallback ladder this method used before the
        provider unification, and mirroring the guard in
        :meth:`SahaBoltzmannSolver.calculate_partition_function`.
        """
        if hasattr(self.atomic_db, "partition_function_for"):
            provider = self.atomic_db.partition_function_for(element, ionization_stage)
            if provider is not None:
                return float(provider.at(T_K))
        else:
            # ABC-only backend: reproduce the pre-unification fallback ladder
            # (direct sum over energy levels, then the stored polynomial).
            from cflibs.plasma.partition import (
                PartitionFunctionEvaluator,
                get_levels_for_species,
            )

            levels = get_levels_for_species(self.atomic_db, element, ionization_stage)
            if levels is not None:
                g_arr, E_arr, ip_ev = levels
                return PartitionFunctionEvaluator.evaluate_direct(T_K, g_arr, E_arr, ip_ev)

            pf = self.atomic_db.get_partition_coefficients(element, ionization_stage)
            if pf:
                return PartitionFunctionEvaluator.evaluate(T_K, pf.coefficients)

        if ionization_stage == 1:
            return 25.0
        if ionization_stage == 2:
            return 15.0
        return 2.0

    def _compute_saha_ratio(
        self,
        element: str,
        T_K: float,
        n_e_cm3: float,
        U_I: float,
        U_II: float,
        ip_ev: float,
    ) -> float:
        """Compute n_II / n_I using the first Saha ratio."""
        safe_ne = max(float(n_e_cm3), 1e10)
        T_eV = max(T_K / EV_TO_K, 0.1)
        return (SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5) * (U_II / U_I) * np.exp(-ip_ev / T_eV)

    def _compute_abundance_multipliers(
        self,
        elements: List[str],
        T_K: float,
        n_e_cm3: float,
        partition_funcs_I: Dict[str, float],
        partition_funcs_II: Dict[str, float],
        ips: Dict[str, float],
        T_corona: Optional[float] = None,
    ) -> Dict[str, float]:
        """
        Map the neutral-plane intercept back to total elemental abundance.

        The pooled Saha-Boltzmann fit returns q_s proportional to N_I / U_I.
        Closure must scale by (1 + n_II / n_I) to recover total elemental
        abundance before normalization.
        """
        multipliers: Dict[str, float] = {}
        # Per Hermann (2017), high-Z elements (Si, Fe, Ca, Al, Mg) in Aalto
        # are sensitive to the corona. We use T_corona for their neutral-plane
        # scaling if provided.
        corona_sensitive = {"Si", "Fe", "Ca", "Al", "Mg"}
        for el in elements:
            U_I = partition_funcs_I.get(el, 25.0)
            U_II = partition_funcs_II.get(el, 15.0)

            T_saha = T_K
            if T_corona is not None and el in corona_sensitive:
                # Weighted temperature for Saha-Boltzmann scaling
                T_saha = 0.3 * T_K + 0.7 * T_corona

            S = self._compute_saha_ratio(el, T_saha, n_e_cm3, U_I, U_II, ips[el])
            multipliers[el] = 1.0 + max(S, 0.0)
        return multipliers

    def _compute_effective_ips(
        self,
        ips: Dict[str, float],
        n_e: float,
        T_K: float,
    ) -> Dict[str, float]:
        """Compute ionization potentials with optional plasma screening applied."""
        if not self.apply_ipd:
            return ips

        from cflibs.plasma.saha_boltzmann import ionization_potential_lowering

        delta_chi = ionization_potential_lowering(n_e, T_K)
        return {el: max(ip - delta_chi, 0.0) for el, ip in ips.items()}

    def _apply_saha_correction(
        self,
        obs_by_element: Dict[str, List[LineObservation]],
        T_K: float,
        n_e: float,
        ips: Dict[str, float],
    ) -> Dict[str, List[LineObservation]]:
        """
        Map ionic lines to the neutral energy plane via the Saha-Boltzmann transform.

        For each ionic (stage-2) line, applies:
          y* = y - log(SAHA_CONST * T^1.5 / n_e)
          x* = E_k + IP

        Neutral lines are passed through unchanged.

        Parameters
        ----------
        obs_by_element : dict
            Raw observations grouped by element symbol
        T_K : float
            Plasma temperature [K]
        n_e : float
            Electron density [cm^-3]
        ips : dict
            First ionization potentials by element [eV]

        Returns
        -------
        dict
            Corrected observations grouped by element symbol
        """
        T_eV = max(T_K / EV_TO_K, 0.1)
        safe_ne = max(n_e, 1e10)
        correction_term = np.log((SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5))
        scale = np.exp(-correction_term)
        corrected: Dict[str, List[LineObservation]] = defaultdict(list)

        for el, obs_list in obs_by_element.items():
            ip = ips.get(el, 15.0)
            for obs in obs_list:
                if obs.ionization_stage == 2:
                    new_obs = LineObservation(
                        wavelength_nm=obs.wavelength_nm,
                        intensity=obs.intensity * scale,
                        intensity_uncertainty=obs.intensity_uncertainty * scale,
                        element=obs.element,
                        ionization_stage=obs.ionization_stage,
                        E_k_ev=obs.E_k_ev + ip,
                        g_k=obs.g_k,
                        A_ki=obs.A_ki,
                    )
                else:
                    new_obs = LineObservation(
                        wavelength_nm=obs.wavelength_nm,
                        intensity=obs.intensity,
                        intensity_uncertainty=obs.intensity_uncertainty,
                        element=obs.element,
                        ionization_stage=obs.ionization_stage,
                        E_k_ev=obs.E_k_ev,
                        g_k=obs.g_k,
                        A_ki=obs.A_ki,
                    )
                corrected[el].append(new_obs)

        return dict(corrected)

    @staticmethod
    def _lower_level_energy_ev(obs: LineObservation) -> float:
        """Lower-level energy ``E_i`` of a line, in eV.

        ``LineObservation`` carries only the upper-level energy ``E_k`` and the
        wavelength, but the curve-of-growth optical-depth estimate needs the
        *lower*-level energy (the absorbing level). Energy conservation for the
        transition gives ``E_i = E_k - hc/lambda`` exactly, so we recover it
        rather than defaulting every line to the ground state (which would make
        every line look like a maximally self-absorbed resonance line).

        Clamped to be non-negative to absorb small wavelength/energy rounding in
        the atomic data.
        """
        photon_ev = (H_PLANCK_EV * C_LIGHT) / (obs.wavelength_nm * 1e-9)
        return max(0.0, obs.E_k_ev - photon_ev)

    def _apply_self_absorption_correction(
        self,
        obs_by_element: Dict[str, List[LineObservation]],
        T_K: float,
        concentrations: Dict[str, float],
        total_n_cm3: float,
        partition_funcs: Dict[str, float],
    ) -> Dict[str, List[LineObservation]]:
        """Curve-of-growth self-absorption correction of observed intensities.

        Implements the outer recursion of the Bulajic et al. (2002,
        *Spectrochim. Acta B* 57, 339, doi:10.1016/S0584-8547(01)00398-6)
        self-absorption algorithm (physics-audit defect **B1**). For each line
        the optical depth ``tau_0`` is recomputed from the *current* plasma
        state (T, concentrations, total number density, partition functions) and
        the observed integrated intensity is divided by the Gaussian escape
        factor ``f(tau) = (1 - e^-tau) / tau`` to recover the optically-thin
        intensity that the Boltzmann/closure fit assumes.

        This is applied to the *raw per-element observations* (before the Saha
        ion->neutral remap) so that the corrected intensities feed the existing
        ``_apply_saha_correction`` -> ``_fit_common_boltzmann_plane`` -> closure
        chain unchanged. ``concentrations`` are the number fractions from the
        previous iteration's closure; on the first iteration they are empty, so
        every ``tau`` is zero and this is a transparent no-op — the correction
        bootstraps once the loop has a composition estimate, exactly as in the
        reference recursive algorithm.

        Lines whose ``tau`` exceeds ``mask_threshold`` are dropped (they are
        black at line centre and carry no recoverable column-density info);
        every other line is returned with its corrected intensity. The
        ``LineObservation`` identity (element, stage, E_k, g_k, A_ki, wavelength)
        is preserved so downstream Saha mapping and Boltzmann ``y`` values stay
        consistent.

        Returns the corrected ``obs_by_element`` mapping (same structure as the
        input). Elements with no surviving lines are omitted.
        """
        self._last_sa_max_tau = 0.0
        if self.self_absorption_corrector is None:
            return obs_by_element
        if not concentrations or total_n_cm3 <= 0 or T_K <= 0:
            return obs_by_element

        flat_obs: List[LineObservation] = []
        lower_level_energies: Dict[float, float] = {}
        for obs_list in obs_by_element.values():
            for obs in obs_list:
                flat_obs.append(obs)
                lower_level_energies[obs.wavelength_nm] = self._lower_level_energy_ev(obs)

        result = self.self_absorption_corrector.correct(
            flat_obs,
            temperature_K=T_K,
            concentrations=concentrations,
            total_number_density_cm3=total_n_cm3,
            partition_funcs=partition_funcs,
            lower_level_energies=lower_level_energies,
        )
        self._last_sa_max_tau = result.max_optical_depth

        corrected_by_element: Dict[str, List[LineObservation]] = defaultdict(list)
        for obs in result.corrected_observations:
            corrected_by_element[obs.element].append(obs)
        return dict(corrected_by_element)

    def _fit_common_boltzmann_plane(
        self,
        corrected_obs_map: Dict[str, List[LineObservation]],
    ) -> Optional[_CommonSlopeFit]:
        """
        Compute a pooled Boltzmann slope common to multiple elements by fitting a single weighted linear slope to per-element centered Boltzmann data.

        For each element with at least two valid corrected observations, this routine computes weighted means in the Boltzmann plane, centers the element's points by those means, and pools the centered points across elements to fit a single slope. The result includes the fitted slope, its variance (accounting for one common slope plus one intercept per contributing element), per-element intercepts, per-element statistics (original values, weights, and means), and an R² goodness-of-fit metric.

        Returns:
            _CommonSlopeFit | None: A _CommonSlopeFit with fields:
                - slope: fitted common slope
                - slope_variance: estimated variance of the slope
                - intercepts: mapping from element to fitted intercept on the original (uncentered) scale
                - element_stats: per-element _CommonSlopeElementStats containing x/y values, weights, and means
                - r_squared: weighted R² of the pooled centered fit
            Returns None if there is insufficient valid data to perform the pooled fit.
        """
        pooled_x_parts: List[np.ndarray] = []
        pooled_y_parts: List[np.ndarray] = []
        pooled_w_parts: List[np.ndarray] = []
        element_stats: Dict[str, _CommonSlopeElementStats] = {}

        for el, obs_list in corrected_obs_map.items():
            if len(obs_list) < 2:
                continue

            xs = np.array([o.E_k_ev for o in obs_list], dtype=float)
            ys = np.array([o.y_value for o in obs_list], dtype=float)
            y_uncertainties = np.array([self._line_y_uncertainty(o) for o in obs_list])
            ws = np.array(
                [1.0 / sigma_y**2 if sigma_y > 0 else 1.0 for sigma_y in y_uncertainties],
                dtype=float,
            )

            valid_mask = np.isfinite(xs) & np.isfinite(ys) & np.isfinite(ws) & (ws > 0.0)
            xs = xs[valid_mask]
            ys = ys[valid_mask]
            ws = ws[valid_mask]

            if xs.size < 2:
                continue

            x_mean = float(np.average(xs, weights=ws))
            y_mean = float(np.average(ys, weights=ws))

            element_stats[el] = _CommonSlopeElementStats(
                x_values=xs,
                y_values=ys,
                weights=ws,
                x_mean=x_mean,
                y_mean=y_mean,
            )
            pooled_x_parts.append(xs - x_mean)
            pooled_y_parts.append(ys - y_mean)
            pooled_w_parts.append(ws)

        if not pooled_x_parts:
            return None

        pooled_x = np.concatenate(pooled_x_parts)
        pooled_y = np.concatenate(pooled_y_parts)
        pooled_w = np.concatenate(pooled_w_parts)

        if pooled_x.size < 3:
            return None

        denom = float(np.sum(pooled_w * pooled_x**2))
        if not np.isfinite(denom) or denom <= 0.0:
            return None

        slope = float(np.sum(pooled_w * pooled_x * pooled_y) / denom)
        residuals = pooled_y - slope * pooled_x
        ss_res = float(np.sum(pooled_w * residuals**2))
        ss_tot = float(np.sum(pooled_w * pooled_y**2))
        r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0.0 else 1.0

        # The centered pooled fit is equivalent to y = a_element + m x, so the
        # residual variance must account for one common slope plus one intercept
        # per contributing element.
        dof = max(int(pooled_x.size) - (1 + len(element_stats)), 1)
        slope_variance = ss_res / (dof * denom)
        if not np.isfinite(slope_variance) or slope_variance <= 0.0:
            slope_variance = 1.0 / denom

        intercepts = {
            el: stats.y_mean - slope * stats.x_mean for el, stats in element_stats.items()
        }

        return _CommonSlopeFit(
            slope=slope,
            slope_variance=slope_variance,
            intercepts=intercepts,
            element_stats=element_stats,
            r_squared=r_squared,
        )

    def solve(
        self, observations: List[LineObservation], closure_mode: str = "standard", **closure_kwargs
    ) -> CFLIBSResult:
        """
        Estimate plasma temperature, electron density, and elemental concentrations from spectral line observations using the iterative CF-LIBS algorithm.

        Routes to ``_solve_lax`` (``jax.lax.while_loop`` path, T1-3) when both
        :func:`_lax_while_loop_enabled` and ``HAS_JAX`` are true; otherwise
        runs the Python ``for``-loop reference path in ``_solve_python``.

        Parameters:
            observations (List[LineObservation]): Spectral line observations to invert; lines are grouped by element.
            closure_mode (str): Closure method for converting Boltzmann intercepts to concentrations. One of "standard", "matrix", "oxide", "ilr", "pwlr", or "dirichlet_residual".
            **closure_kwargs: Additional keyword arguments forwarded to the chosen closure method (for example, a matrix_element for "matrix" mode).

        Returns:
            CFLIBSResult: Final inversion result containing:
                - temperature_K: Estimated plasma temperature (Kelvin).
                - temperature_uncertainty_K: Set to 0.0 in this routine (see solve_with_uncertainty for propagated uncertainties).
                - electron_density_cm3: Estimated electron density (cm^-3).
                - concentrations: Dictionary of elemental concentrations (relative units returned by the chosen closure).
                - concentration_uncertainties: Empty in this routine (see solve_with_uncertainty).
                - iterations: Number of iterations performed.
                - converged: Whether the iterative solver met convergence criteria.
                - quality_metrics: Diagnostics including the last Boltzmann fit R^2 and LTE validation metrics.
                - electron_density_uncertainty_cm3: Set to 0.0 here.
                - boltzmann_covariance: None in this routine; covariance information is produced by solve_with_uncertainty.
        """
        # The self-absorption correction (B1) applies a per-line curve-of-growth
        # correction whose escape factor is not part of the traced lax body, so
        # when it is enabled we run the reference Python loop (which is the
        # default production path anyway). Disable SA to opt back into lax.
        if HAS_JAX and _lax_while_loop_enabled() and not self.apply_self_absorption:
            try:
                return self._solve_lax(observations, closure_mode, **closure_kwargs)
            except _LaxFallback as exc:
                logger.info("lax.while_loop path bailed out (%s); using Python loop", exc)
                return self._solve_python(observations, closure_mode, **closure_kwargs)
        return self._solve_python(observations, closure_mode, **closure_kwargs)

    def _solve_python(
        self, observations: List[LineObservation], closure_mode: str = "standard", **closure_kwargs
    ) -> CFLIBSResult:
        """Reference Python ``for``-loop implementation of :meth:`solve`.

        Bit-for-bit equivalent to the pre-T1-3 ``solve`` body; the public
        :meth:`solve` routes through here when ``CFLIBS_USE_LAX_WHILE_LOOP`` is
        unset (default) or when JAX is unavailable.
        """
        # 1. Initialization
        T_K = 10000.0
        T_corona = None
        n_e = 1.0e17

        # Cache static data (IPs, atomic data)
        # Group observations by element
        obs_by_element = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        elements = list(obs_by_element.keys())

        # Pre-fetch Ionization Potentials
        ips = {}
        for el in elements:
            # Need IP of neutral (I -> II)
            ip = self.atomic_db.get_ionization_potential(el, 1)
            if ip is None:
                logger.warning(f"No IP for {el} I, assuming high")
                ip = 15.0  # Fallback
            ips[el] = ip

        # Iteration loop
        converged = False
        history = []
        concentrations: Dict[str, float] = {}  # Initialize before loop
        last_common_fit: Optional[_CommonSlopeFit] = None
        last_max_tau = 0.0

        for _ in range(1, self.max_iterations + 1):
            T_prev = T_K
            ne_prev = n_e

            T_eV = T_K / EV_TO_K
            if T_eV < 0.1:
                T_eV = 0.1  # clamp

            # 2. Calculate Partition Functions & Saha Corrections
            partition_funcs = {}  # U_I for each element
            partition_funcs_II = {}

            for el in elements:
                partition_funcs[el] = self._evaluate_partition_function(el, 1, T_K)
                partition_funcs_II[el] = self._evaluate_partition_function(el, 2, T_K)

            effective_ips = self._compute_effective_ips(ips, n_e, T_K)

            # 2b. Self-absorption correction (Bulajic 2002; physics-audit B1).
            # Recompute tau from the *current* plasma state and divide observed
            # intensities by the curve-of-growth escape factor before the
            # Boltzmann/closure fit. No-op on iteration 1 (concentrations empty).
            #
            # Column density: the optical depth scales with the absorbing heavy-
            # particle column n_heavy * L. We anchor it to a LIBS-realistic
            # reference (``self_absorption_column_density_cm3``, default 1e16)
            # rather than the solver's STP pressure-balance n_e (~1e18), which
            # would push every strong major line into uniform saturation and
            # erase the per-line discrimination the correction needs. See the
            # extended note in ``__init__``. Per-element number fractions
            # (``concentrations``) still modulate τ inside the corrector.
            sa_obs_by_element = obs_by_element
            if self.apply_self_absorption:
                sa_obs_by_element = self._apply_self_absorption_correction(
                    obs_by_element,
                    T_K,
                    concentrations,
                    self.self_absorption_column_density_cm3,
                    partition_funcs,
                )
                last_max_tau = self._last_sa_max_tau

            # Map ionic lines to the neutral energy plane
            corrected_obs_map = self._apply_saha_correction(
                sa_obs_by_element, T_K, n_e, effective_ips
            )

            # 3. Multi-species Boltzmann Fit
            common_fit = self._fit_common_boltzmann_plane(corrected_obs_map)
            if common_fit is None:
                logger.warning("Insufficient points for fit")
                break

            last_common_fit = common_fit
            slope = common_fit.slope
            fit_r2 = float(getattr(common_fit, "r_squared", 0.0))

            # Update T — gated on Boltzmann-plot quality (see __init__ note).
            #
            # A non-negative slope is unphysical (the populations would *rise*
            # with E_k) and a low-R^2 fit means the slope is not estimable. In
            # either case the legacy "T_new = 50000" clamp is what triggers the
            # keystone collapse, so instead we hold T at the prior value and flag
            # the fit as degenerate. Holding T (rather than clamping high) keeps
            # the Boltzmann factor exp(-E_k/kT) discriminating between lines so
            # the intercepts stay physically meaningful for the closure step.
            boltzmann_degenerate = slope >= 0 or fit_r2 < self.min_boltzmann_r2
            if boltzmann_degenerate:
                T_new = T_prev  # hold at prior; slope carries no usable T
            else:
                T_new = -1.0 / (slope * KB_EV)

            # Damping
            T_K = 0.5 * T_prev + 0.5 * T_new

            if self.two_region:
                # Per Hermann (2017), corona temperature is typically 70-90% of core.
                # We use a fixed ratio of 0.8 for the iterative update.
                T_corona = 0.8 * T_K

            # Calculate Intercepts
            intercepts = common_fit.intercepts

            abundance_multipliers = self._compute_abundance_multipliers(
                list(intercepts.keys()),
                T_K,
                n_e,
                partition_funcs,
                partition_funcs_II,
                effective_ips,
                T_corona=T_corona,
            )

            # 4. Closure
            if closure_mode == "matrix":
                closure_res = ClosureEquation.apply_matrix_mode(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            elif closure_mode == "oxide":
                closure_res = ClosureEquation.apply_oxide_mode(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            elif closure_mode == "ilr":
                closure_res = ClosureEquation.apply_ilr(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                )
            elif closure_mode == "pwlr":
                closure_res = ClosureEquation.apply_pwlr(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            elif closure_mode == "dirichlet_residual":
                closure_res = ClosureEquation.apply_dirichlet_residual(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                    **closure_kwargs,
                )
            else:
                closure_res = ClosureEquation.apply_standard(
                    intercepts,
                    partition_funcs,
                    abundance_multipliers=abundance_multipliers,
                )

            concentrations = closure_res.concentrations

            # 5. Update electron density via pressure balance

            # Calculate avg_Z based on Saha ratios
            # n_II / n_I = S(T, ne)
            # Z_bar_s = (1*n_I + 2*n_II) / (n_I + n_II) - 1  (since Z=0 for neutral? No, Z=1 neutral in our code)
            # In code sp_num=1 is neutral (charge 0), sp_num=2 is ion (charge +1)
            # So electron contribution: Neutral=0, Ion=1
            # Avg electrons per atom of species s:
            # eps_s = (n_II) / (n_I + n_II) = S / (1+S)

            total_eps = 0.0
            for el, C_s in concentrations.items():
                U_I = partition_funcs.get(el, 25.0)
                U_II = partition_funcs_II.get(el, 15.0)
                S = self._compute_saha_ratio(el, T_K, n_e, U_I, U_II, effective_ips[el])
                eps_s = S / (1.0 + S)
                total_eps += C_s * eps_s

            avg_Z = total_eps

            # Total particle density (nuclei)
            # n_tot = P / (k T (1 + avg_Z))
            # n_e = avg_Z * n_tot

            n_tot = self.pressure_pa / (KB * T_K * (1.0 + avg_Z))
            # Convert to cm^-3
            n_tot_cm3 = n_tot * 1e-6

            ne_new = avg_Z * n_tot_cm3

            # Damping
            n_e = 0.5 * ne_prev + 0.5 * ne_new

            history.append((T_K, n_e))

            # Check convergence. A degenerate Boltzmann fit holds T at the prior,
            # which would otherwise satisfy |ΔT| < tol spuriously — so a
            # degenerate fit can never be reported as converged. The composition
            # from such a fit is untrustworthy and the False flag tells callers
            # (and round-trip/NIST tooling) to treat T and C as unconstrained.
            if (
                not boltzmann_degenerate
                and abs(T_K - T_prev) < self.t_tolerance_k
                and abs(n_e - ne_prev) / ne_prev < self.ne_tolerance_frac
            ):
                converged = True
                break
            converged = False

        # LTE validity check
        from cflibs.plasma.lte_validator import LTEValidator

        lte_validator = LTEValidator()
        lte_report = lte_validator.validate(
            T_K=T_K,
            n_e_cm3=n_e,
            observations=observations,
        )
        quality_metrics = {
            "r_squared_last": last_common_fit.r_squared if last_common_fit is not None else 0.0
        }
        quality_metrics.update(lte_report.quality_metrics)
        # Self-absorption diagnostics: max optical depth seen on the final
        # correction pass (0.0 when SA is disabled or never fired).
        quality_metrics["self_absorption_applied"] = float(self.apply_self_absorption)
        quality_metrics["self_absorption_max_tau"] = float(last_max_tau)

        if self.two_region and T_corona is None:
            T_corona = 0.8 * T_K

        return CFLIBSResult(
            temperature_K=T_K,
            temperature_uncertainty_K=0.0,  # See solve_with_uncertainty for propagation
            electron_density_cm3=n_e,
            concentrations=concentrations,
            concentration_uncertainties={},  # See solve_with_uncertainty for propagation
            iterations=len(history),
            converged=converged,
            temperature_corona_K=T_corona,
            quality_metrics=quality_metrics,
            electron_density_uncertainty_cm3=0.0,
            boltzmann_covariance=None,
        )

    def _solve_lax(
        self, observations: List[LineObservation], closure_mode: str = "standard", **closure_kwargs
    ) -> CFLIBSResult:
        """JAX ``lax.while_loop`` implementation of :meth:`solve` (T1-3).

        Pre-fetches every atomic-DB-backed value (IPs, partition coefficients,
        per-element abundance scales) outside the loop body, builds padded
        ``(E, N_max)`` observation arrays, and runs the iteration through
        ``jax.lax.while_loop`` with the closure equation routed via
        :func:`jax.pure_callback` to call the existing :class:`ClosureEquation`
        functions (preserving numerics bit-for-bit while keeping the body
        jit-traceable).

        Raises
        ------
        _LaxFallback
            If observations cannot be padded into a usable array (e.g. zero
            valid lines), routing back to :meth:`_solve_python`.
        """
        if not HAS_JAX:  # pragma: no cover - guarded by caller
            raise _LaxFallback("JAX not available")

        # 1. Initialization
        T_init = 10000.0
        ne_init = 1.0e17

        # Group observations by element (preserve insertion order)
        obs_by_element: Dict[str, List[LineObservation]] = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        elements_seq = list(obs_by_element.keys())
        if not elements_seq:
            raise _LaxFallback("no elements with observations")

        # 2. Pre-fetch atomic data outside the loop (spec §6)
        snapshot = _AtomicSnapshot.from_solver(self, elements_seq)

        # 3. Build padded observation arrays
        elements_ord, x_raw, y_raw, w_raw, stage_arr, mask_arr = _build_padded_arrays_from_obs(
            dict(obs_by_element)
        )
        if x_raw is None:
            raise _LaxFallback("no usable observations after padding")

        # Reorder snapshot to match the element order produced by _build_padded_arrays_from_obs.
        if elements_ord != elements_seq:
            snapshot = snapshot.reorder(elements_ord)

        # 4. Build/resolve the closure callable at solve-time (spec §5 Option A)
        closure_callback = _make_closure_callback(closure_mode, elements_ord, closure_kwargs)

        # 5. Convert to JAX arrays
        x_d = jnp.asarray(x_raw, dtype=jnp.float64)
        y_d = jnp.asarray(y_raw, dtype=jnp.float64)
        w_d = jnp.asarray(w_raw, dtype=jnp.float64)
        stage_d = jnp.asarray(stage_arr, dtype=jnp.int32)
        mask_d = jnp.asarray(mask_arr, dtype=bool)

        # 6. Initial state
        init_state = LoopState(
            T_K=jnp.asarray(T_init, dtype=jnp.float64),
            n_e_cm3=jnp.asarray(ne_init, dtype=jnp.float64),
            T_prev=jnp.asarray(T_init, dtype=jnp.float64),
            n_e_prev=jnp.asarray(ne_init, dtype=jnp.float64),
            converged=jnp.asarray(False),
            i=jnp.asarray(0, dtype=jnp.int32),
            U_I=jnp.zeros(len(elements_ord), dtype=jnp.float64),
            U_II=jnp.zeros(len(elements_ord), dtype=jnp.float64),
            intercepts=jnp.zeros(len(elements_ord), dtype=jnp.float64),
            concentrations=jnp.zeros(len(elements_ord), dtype=jnp.float64),
            r_squared=jnp.asarray(0.0, dtype=jnp.float64),
        )

        # 7. Run the while loop
        final_state = _run_lax_while_loop(
            init_state,
            x_d,
            y_d,
            w_d,
            stage_d,
            mask_d,
            snapshot,
            closure_callback,
            apply_ipd=self.apply_ipd,
            two_region=self.two_region,
            max_iter=self.max_iterations,
            t_tol_k=self.t_tolerance_k,
            ne_tol_frac=self.ne_tolerance_frac,
            pressure_pa=self.pressure_pa,
            min_r2=self.min_boltzmann_r2,
        )

        # 8. Host-side assembly
        T_K = float(final_state.T_K)
        n_e = float(final_state.n_e_cm3)
        converged_bool = bool(final_state.converged)
        iterations = int(final_state.i)
        r_squared = float(final_state.r_squared)
        conc_arr = np.asarray(final_state.concentrations)
        concentrations = {el: float(conc_arr[i]) for i, el in enumerate(elements_ord)}

        # Corona post-loop assembly (matches Python path)
        T_corona = 0.8 * T_K if self.two_region else None

        # LTE validity check
        from cflibs.plasma.lte_validator import LTEValidator

        lte_validator = LTEValidator()
        lte_report = lte_validator.validate(
            T_K=T_K,
            n_e_cm3=n_e,
            observations=observations,
        )
        quality_metrics: Dict[str, float] = {"r_squared_last": r_squared}
        quality_metrics.update(lte_report.quality_metrics)

        return CFLIBSResult(
            temperature_K=T_K,
            temperature_uncertainty_K=0.0,
            electron_density_cm3=n_e,
            concentrations=concentrations,
            concentration_uncertainties={},
            iterations=iterations,
            converged=converged_bool,
            temperature_corona_K=T_corona,
            quality_metrics=quality_metrics,
            electron_density_uncertainty_cm3=0.0,
            boltzmann_covariance=None,
        )

    def solve_with_uncertainty(
        self,
        observations: List[LineObservation],
        closure_mode: str = "standard",
        **closure_kwargs,
    ) -> CFLIBSResult:
        """
        Compute plasma parameters while propagating measurement and fit uncertainties.

        Performs uncertainty propagation through the pooled Boltzmann fit, Saha
        correction, and the chosen closure equation, returning the same result
        structure as solve() augmented with uncertainty fields.

        Parameters:
            observations (List[LineObservation]): Spectral lines with intensity uncertainties.
            closure_mode (str): Closure algorithm to use ('standard', 'matrix', 'oxide', 'ilr', 'pwlr', or 'dirichlet_residual').
            **closure_kwargs: Arguments passed to the chosen closure routine (e.g. 'matrix_element',
                'matrix_fraction', or 'oxide_stoichiometry').

        Returns:
            CFLIBSResult: Solver result including populated uncertainty fields:
                - temperature_uncertainty_K: estimated standard deviation of temperature (K)
                - concentration_uncertainties: per-element concentration uncertainties
                - boltzmann_covariance: selected 2x2 covariance matrix for slope/intercept (or None)

        Raises:
            ImportError: If the external `uncertainties`-based utilities are not available.
        """
        # First run the standard solver to convergence
        result = self.solve(observations, closure_mode, **closure_kwargs)

        # Import uncertainty utilities (will raise ImportError if not available)
        from cflibs.inversion.physics.uncertainty import (
            propagate_through_closure_oxide,
            propagate_through_closure_standard,
            propagate_through_closure_matrix,
            extract_values_and_uncertainties,
        )
        from uncertainties import ufloat

        # Group observations by element
        obs_by_element: Dict[str, list] = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        elements = list(obs_by_element.keys())

        # Use converged plasma state
        T_K = result.temperature_K
        n_e = result.electron_density_cm3

        # Pre-fetch ionization potentials (same as solve())
        ips = {}
        for el in elements:
            ip = self.atomic_db.get_ionization_potential(el, 1)
            ips[el] = ip if ip is not None else 15.0

        effective_ips = self._compute_effective_ips(ips, n_e, T_K)

        # Apply Saha correction so intercepts match those from solve()
        corrected_obs_map = self._apply_saha_correction(obs_by_element, T_K, n_e, effective_ips)

        # Get partition functions at converged T
        partition_funcs = {}
        partition_funcs_II = {}
        for el in elements:
            partition_funcs[el] = self._evaluate_partition_function(el, 1, T_K)
            partition_funcs_II[el] = self._evaluate_partition_function(el, 2, T_K)

        abundance_multipliers = self._compute_abundance_multipliers(
            elements,
            T_K,
            n_e,
            partition_funcs,
            partition_funcs_II,
            effective_ips,
            T_corona=result.temperature_corona_K,
        )

        common_fit = self._fit_common_boltzmann_plane(corrected_obs_map)
        if common_fit is None:
            return result

        # Propagate the same common-slope model used by solve()
        intercepts_u = {}
        covariances = {}
        slope_err = (
            float(np.sqrt(common_fit.slope_variance))
            if np.isfinite(common_fit.slope_variance) and common_fit.slope_variance > 0.0
            else 0.0
        )
        slope_u = ufloat(common_fit.slope, slope_err)

        for el, stats in common_fit.element_stats.items():
            weight_sum = float(np.sum(stats.weights))
            y_mean_err = np.sqrt(1.0 / weight_sum) if weight_sum > 0.0 else 0.0
            y_mean_u = ufloat(stats.y_mean, y_mean_err)
            intercept_u = y_mean_u - slope_u * stats.x_mean
            intercepts_u[el] = intercept_u

            intercept_var = y_mean_err**2 + (stats.x_mean**2) * common_fit.slope_variance
            covariances[el] = np.array(
                [
                    [common_fit.slope_variance, -stats.x_mean * common_fit.slope_variance],
                    [-stats.x_mean * common_fit.slope_variance, intercept_var],
                ],
                dtype=float,
            )

        # Propagate through closure
        if closure_mode == "matrix" and "matrix_element" in closure_kwargs:
            concentrations_u = propagate_through_closure_matrix(
                intercepts_u,
                partition_funcs,
                closure_kwargs["matrix_element"],
                closure_kwargs.get("matrix_fraction", 0.9),
                abundance_multipliers=abundance_multipliers,
            )
        elif closure_mode == "oxide":
            concentrations_u = propagate_through_closure_oxide(
                intercepts_u,
                partition_funcs,
                closure_kwargs.get("oxide_stoichiometry", {}),
                abundance_multipliers=abundance_multipliers,
            )
        elif closure_mode in {"ilr", "pwlr", "dirichlet_residual"}:
            concentrations_u = propagate_through_closure_standard(
                intercepts_u,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )
        else:
            concentrations_u = propagate_through_closure_standard(
                intercepts_u,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )

        # Extract nominal values and uncertainties
        conc_nominal, conc_uncert = extract_values_and_uncertainties(concentrations_u)

        # Temperature uncertainty from pooled slope estimate
        from cflibs.inversion.physics.uncertainty import temperature_from_slope

        T_err = 0.0
        if slope_err > 0.0:
            T_K_u = temperature_from_slope(slope_u)
            T_err = float(T_K_u.std_dev) if np.isfinite(T_K_u.std_dev) else 0.0

        selected_covariance = None
        covariance_element = None
        if covariances:
            preferred_element = (
                closure_kwargs.get("matrix_element") if closure_mode == "matrix" else None
            )
            if preferred_element in covariances:
                covariance_element = preferred_element
            else:
                covariance_element = sorted(covariances)[0]
            selected_covariance = covariances[covariance_element]

        quality_metrics = dict(result.quality_metrics)
        if covariance_element is not None:
            quality_metrics["boltzmann_covariance_element"] = covariance_element

        return CFLIBSResult(
            temperature_K=result.temperature_K,
            temperature_uncertainty_K=T_err,
            electron_density_cm3=result.electron_density_cm3,
            concentrations=conc_nominal if conc_nominal else result.concentrations,
            concentration_uncertainties=conc_uncert if conc_uncert else {},
            iterations=result.iterations,
            converged=result.converged,
            temperature_corona_K=result.temperature_corona_K,
            quality_metrics=quality_metrics,
            electron_density_uncertainty_cm3=0.0,  # Would need iterative uncertainty
            boltzmann_covariance=selected_covariance,
        )


# ---------------------------------------------------------------------------
# JAX-accelerated iterative solver
# ---------------------------------------------------------------------------


if HAS_JAX:

    @jit
    def _saha_correct_kernel(
        x: jnp.ndarray,
        y: jnp.ndarray,
        stage: jnp.ndarray,
        ip: jnp.ndarray,
        T_eV: jnp.ndarray,
        log_correction: jnp.ndarray,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Vectorized Saha correction kernel.

        For each line: if stage==2 (ionic), shift y by ``-log_correction`` and
        x by ``+ip``. Neutral lines are passed through unchanged.

        All arrays are shape (B, N_max). ``stage`` is 1 or 2.
        ``ip`` and ``T_eV`` are scalars or shape (B, N_max) broadcastable.
        """
        is_ionic = stage == 2
        y_corr = jnp.where(is_ionic, y - log_correction, y)
        x_corr = jnp.where(is_ionic, x + ip, x)
        return x_corr, y_corr

    @jit
    def _common_slope_kernel(
        x: jnp.ndarray,  # (E, N_max) - per-element padded x
        y: jnp.ndarray,  # (E, N_max) - per-element padded y
        w: jnp.ndarray,  # (E, N_max) - per-element padded weights
        mask: jnp.ndarray,  # (E, N_max) - bool mask
    ) -> Dict[str, jnp.ndarray]:
        """
        JAX kernel for pooled common-slope Boltzmann fit across E elements.

        Mirrors ``IterativeCFLIBSSolver._fit_common_boltzmann_plane`` math
        exactly: per-element weighted means, centered points, single pooled
        weighted slope, per-element intercepts on uncentered scale.

        Returns dict with keys: slope, slope_variance, r_squared,
        intercepts (E,), x_means (E,), y_means (E,), n_valid_per_el (E,).
        """
        mf = mask.astype(jnp.float64)
        w_eff = w * mf  # zero out padded entries

        # Per-element weight totals
        S_w = jnp.sum(w_eff, axis=1)  # (E,)
        # Per-element weighted means; guard zero-weight elements
        denom_safe = jnp.where(S_w > 0.0, S_w, 1.0)
        x_means = jnp.sum(w_eff * x, axis=1) / denom_safe  # (E,)
        y_means = jnp.sum(w_eff * y, axis=1) / denom_safe

        # Centered points (still padded; padded entries get w_eff=0 so they
        # don't contribute regardless of the centering values)
        xc = x - x_means[:, None]
        yc = y - y_means[:, None]

        # Pooled sums (across E and N_max simultaneously)
        sum_wxx = jnp.sum(w_eff * xc * xc)  # scalar
        sum_wxy = jnp.sum(w_eff * xc * yc)  # scalar

        # Single common slope
        denom = sum_wxx
        slope = jnp.where(denom > 0.0, sum_wxy / jnp.where(denom > 0.0, denom, 1.0), 0.0)

        # Pooled residuals and r_squared
        y_pred_centered = slope * xc
        residuals = yc - y_pred_centered
        ss_res = jnp.sum(w_eff * residuals * residuals)
        ss_tot = jnp.sum(w_eff * yc * yc)
        r_squared = jnp.where(
            ss_tot > 0.0, 1.0 - ss_res / jnp.where(ss_tot > 0.0, ss_tot, 1.0), 1.0
        )

        # Per-element valid counts (for DOF)
        n_valid_per_el = jnp.sum(mf, axis=1)
        # DOF accounts for one slope plus one intercept per contributing element
        n_total = jnp.sum(n_valid_per_el)
        n_elements_active = jnp.sum(n_valid_per_el >= 2.0)
        dof = jnp.maximum(n_total - (1.0 + n_elements_active), 1.0)
        slope_variance = jnp.where(
            denom > 0.0, ss_res / (dof * jnp.where(denom > 0.0, denom, 1.0)), 1.0
        )
        # Fall back to inverse Fisher information when residual variance is degenerate
        slope_variance = jnp.where(
            (slope_variance > 0.0) & jnp.isfinite(slope_variance),
            slope_variance,
            jnp.where(denom > 0.0, 1.0 / jnp.where(denom > 0.0, denom, 1.0), 1.0),
        )

        intercepts = y_means - slope * x_means

        return {
            "slope": slope,
            "slope_variance": slope_variance,
            "r_squared": r_squared,
            "intercepts": intercepts,
            "x_means": x_means,
            "y_means": y_means,
            "n_valid_per_el": n_valid_per_el,
        }


def _build_padded_arrays_from_obs(
    obs_by_element: Dict[str, List[LineObservation]],
):
    """
    Build padded (E, N_max) numpy arrays from per-element observation lists.

    Returns:
        elements: list of element symbols (length E, in dict-iteration order)
        x: (E, N_max) E_k_ev padded with 0
        y: (E, N_max) y_value padded with 0
        w: (E, N_max) inverse-variance weights padded with 0
        stage: (E, N_max) ionization stage (1 or 2) padded with 1
        mask: (E, N_max) bool mask
    """
    elements = list(obs_by_element.keys())
    if not elements:
        return [], None, None, None, None, None
    counts = [len(obs_by_element[el]) for el in elements]
    n_max = max(counts) if counts else 0
    E = len(elements)
    if n_max == 0:
        return elements, None, None, None, None, None
    x = np.zeros((E, n_max), dtype=np.float64)
    y = np.zeros((E, n_max), dtype=np.float64)
    w = np.zeros((E, n_max), dtype=np.float64)
    stage = np.ones((E, n_max), dtype=np.int32)
    mask = np.zeros((E, n_max), dtype=bool)
    for i, el in enumerate(elements):
        for j, obs in enumerate(obs_by_element[el]):
            x[i, j] = obs.E_k_ev
            y[i, j] = obs.y_value
            sigma = obs.y_uncertainty
            w[i, j] = 1.0 / (sigma**2) if sigma > 0 else 1.0
            stage[i, j] = obs.ionization_stage
            mask[i, j] = (
                np.isfinite(obs.y_value)
                and np.isfinite(obs.E_k_ev)
                and np.isfinite(w[i, j])
                and w[i, j] > 0.0
            )
    # Zero-out invalid entries in x/y/w so masked sums are clean
    x = np.where(mask, x, 0.0)
    y = np.where(mask, y, 0.0)
    w = np.where(mask, w, 0.0)
    return elements, x, y, w, stage, mask


class IterativeCFLIBSSolverJax(IterativeCFLIBSSolver):
    """
    JAX-accelerated iterative CF-LIBS solver.

    Drop-in replacement for :class:`IterativeCFLIBSSolver` with the same
    constructor signature and ``solve()``/``solve_with_uncertainty()`` API.
    The hot path (Saha correction + pooled Boltzmann fit) is dispatched to
    JAX kernels; the partition-function evaluation, closure equation, and
    pressure-balance update remain in numpy/Python because they require
    ``AtomicDatabase`` lookups and per-element dictionary plumbing.

    The solver falls back gracefully to the numpy parent implementation if
    JAX is not available at instantiation time -- callers can always use
    ``IterativeCFLIBSSolverJax`` without paying the import-error cost.

    Numerical equivalence: the same closed-form pooled WLS algebra is used
    here and in the numpy parent; composition outputs match to ``rtol=1e-3``
    on representative multi-element fixtures (see
    ``tests/inversion/test_solver_jax_parity.py``).
    """

    backend: str  # "jax" or "numpy_fallback"

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        max_iterations: int = 20,
        t_tolerance_k: float = 100.0,
        ne_tolerance_frac: float = 0.1,
        pressure_pa: float = STP_PRESSURE,
        apply_ipd: bool = False,
    ):
        super().__init__(
            atomic_db=atomic_db,
            max_iterations=max_iterations,
            t_tolerance_k=t_tolerance_k,
            ne_tolerance_frac=ne_tolerance_frac,
            pressure_pa=pressure_pa,
            apply_ipd=apply_ipd,
        )
        if HAS_JAX:
            try:
                self._jax_backend = jax.default_backend()
            except Exception:  # pragma: no cover
                self._jax_backend = "cpu"
            self.backend = "jax"
        else:
            logger.info(
                "JAX unavailable; IterativeCFLIBSSolverJax will fall back to the "
                "numpy parent implementation."
            )
            self._jax_backend = None
            self.backend = "numpy_fallback"

    @property
    def jax_backend(self) -> Optional[str]:
        """Active JAX backend ('cpu', 'gpu', 'tpu') or None when JAX is absent."""
        return self._jax_backend

    # -- Hot path: vectorized Saha + common-slope Boltzmann fit ---------------

    def _saha_and_fit_jax(
        self,
        elements: List[str],
        x_raw: np.ndarray,
        y_raw: np.ndarray,
        w_raw: np.ndarray,
        stage_arr: np.ndarray,
        mask_arr: np.ndarray,
        T_K: float,
        n_e: float,
        ips: Dict[str, float],
    ) -> Optional[_CommonSlopeFit]:
        """
        Apply Saha correction + pooled common-slope fit using JAX kernels.

        Returns a ``_CommonSlopeFit`` populated from JAX-side reductions, or
        None when there is insufficient valid data.
        """
        if not HAS_JAX:
            return None

        # Saha correction inputs
        T_eV = max(T_K / EV_TO_K, 0.1)
        safe_ne = max(float(n_e), 1e10)
        log_correction = float(np.log((SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5)))

        # Build per-element ip array broadcast to (E, N_max)
        ip_per_el = np.array([ips.get(el, 15.0) for el in elements], dtype=np.float64)
        ip_arr = np.broadcast_to(ip_per_el[:, None], x_raw.shape).astype(np.float64)

        # Move to device
        x_d = jnp.asarray(x_raw, dtype=jnp.float64)
        y_d = jnp.asarray(y_raw, dtype=jnp.float64)
        w_d = jnp.asarray(w_raw, dtype=jnp.float64)
        stage_d = jnp.asarray(stage_arr, dtype=jnp.int32)
        ip_d = jnp.asarray(ip_arr, dtype=jnp.float64)
        mask_d = jnp.asarray(mask_arr, dtype=bool)
        log_correction_d = jnp.asarray(log_correction, dtype=jnp.float64)

        # Kernel 1: Saha correction
        x_c, y_c = _saha_correct_kernel(
            x_d, y_d, stage_d, ip_d, jnp.asarray(T_eV, dtype=jnp.float64), log_correction_d
        )

        # Kernel 2: pooled common-slope fit
        fit = _common_slope_kernel(x_c, y_c, w_d, mask_d)

        # Pull scalars/arrays back to host (single sync point per iteration)
        slope = float(fit["slope"])
        slope_variance = float(fit["slope_variance"])
        r_squared = float(fit["r_squared"])
        intercepts_arr = np.asarray(fit["intercepts"])
        x_means_arr = np.asarray(fit["x_means"])
        y_means_arr = np.asarray(fit["y_means"])
        n_valid_per_el = np.asarray(fit["n_valid_per_el"])

        # Validity check: need at least one element with >=2 points and >=3 total
        active_elements = [el for el, n in zip(elements, n_valid_per_el) if n >= 2]
        if not active_elements:
            return None
        if int(n_valid_per_el.sum()) < 3:
            return None
        if not np.isfinite(slope) or not np.isfinite(slope_variance):
            return None

        # Build _CommonSlopeFit dataclass with per-element stats so downstream
        # consumers (e.g. solve_with_uncertainty) keep working.
        element_stats: Dict[str, _CommonSlopeElementStats] = {}
        intercepts_dict: Dict[str, float] = {}
        for i, el in enumerate(elements):
            if n_valid_per_el[i] < 2:
                continue
            row_mask = mask_arr[i]
            xs = x_raw[i][row_mask].astype(np.float64)
            # NOTE: x_raw stores pre-Saha E_k; for ionic lines apply the
            # IP shift on the host side so element_stats reflects the
            # corrected (neutral-plane) coordinates the downstream
            # uncertainty propagation expects.
            xs_corrected = xs.copy()
            stage_row = stage_arr[i][row_mask]
            ip_el = ips.get(el, 15.0)
            xs_corrected[stage_row == 2] += ip_el
            ys_raw = y_raw[i][row_mask].astype(np.float64)
            ys_corrected = ys_raw.copy()
            ys_corrected[stage_row == 2] -= log_correction
            ws = w_raw[i][row_mask].astype(np.float64)
            element_stats[el] = _CommonSlopeElementStats(
                x_values=xs_corrected,
                y_values=ys_corrected,
                weights=ws,
                x_mean=float(x_means_arr[i]),
                y_mean=float(y_means_arr[i]),
            )
            intercepts_dict[el] = float(intercepts_arr[i])

        if not element_stats:
            return None

        return _CommonSlopeFit(
            slope=slope,
            slope_variance=slope_variance,
            intercepts=intercepts_dict,
            element_stats=element_stats,
            r_squared=r_squared,
        )

    # -- Public solve() -------------------------------------------------------

    def solve(
        self,
        observations: List[LineObservation],
        closure_mode: str = "standard",
        **closure_kwargs,
    ) -> CFLIBSResult:
        """Deprecated thin shim for the JAX iterative path (T1-3).

        .. deprecated:: T1-3
            ``IterativeCFLIBSSolverJax`` is superseded by the
            ``CFLIBS_USE_LAX_WHILE_LOOP=1`` env flag on the parent
            :class:`IterativeCFLIBSSolver`, which selects the
            :func:`jax.lax.while_loop` path with the same numerics. This
            subclass now delegates to that path (or to the parent's Python
            loop) and is retained as an alias for one release; new callers
            should use :class:`IterativeCFLIBSSolver` directly with the env
            flag set.

        Falls back to the parent Python implementation when JAX is not
        available, preserving the prior behavior contract.
        """
        warnings.warn(
            "IterativeCFLIBSSolverJax is deprecated; use IterativeCFLIBSSolver "
            "with CFLIBS_USE_LAX_WHILE_LOOP=1 instead (T1-3, ADR-0001).",
            DeprecationWarning,
            stacklevel=2,
        )
        if not HAS_JAX:
            return super().solve(observations, closure_mode, **closure_kwargs)
        # Route through the new lax.while_loop path; fall back to the parent
        # Python path on any internal bailout.
        try:
            result = self._solve_lax(observations, closure_mode, **closure_kwargs)
        except _LaxFallback as exc:
            logger.info("lax.while_loop path bailed out (%s); using Python loop", exc)
            return super().solve(observations, closure_mode, **closure_kwargs)
        # Augment quality_metrics with the legacy backend-reporting keys so
        # downstream consumers of IterativeCFLIBSSolverJax keep working.
        result.quality_metrics.setdefault("backend", self.backend)
        result.quality_metrics.setdefault("jax_backend", self._jax_backend or "n/a")
        return result
