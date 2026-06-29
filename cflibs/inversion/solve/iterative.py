"""
Iterative solver for Classic CF-LIBS.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence, Tuple, NamedTuple, Any
import numpy as np
from collections import defaultdict

from cflibs.core.constants import (
    KB,
    KB_EV,
    SAHA_CONST_CM3,
    STP_PRESSURE,
    EV_TO_K,
)
from cflibs.atomic.database import AtomicDatabase
from cflibs.radiation.stark import estimate_ne_from_stark
from cflibs.inversion.physics.boltzmann import LineObservation
from cflibs.inversion.physics.line_selection import compute_db_isolation_weights
from cflibs.inversion.physics.closure import ClosureEquation
from cflibs.inversion.physics.self_absorption_observable import (
    ObservableSAResult,
    ObservableSelfAbsorptionCorrector,
    normalize_self_absorption_mode,
)
from cflibs.inversion.physics.self_absorption_inputs import (
    evaluate_partition_function as _build_evaluate_partition_function,
)
from cflibs.plasma.partition import canonical_partition_fallback, lookup_partition_function
from cflibs.core.logging_config import get_logger

# Physical temperature window for the Boltzmann slope→T inversion. T = -1/(slope*k)
# blows up as the slope → 0⁻, so a shallow-but-negative slope that passes the R^2
# gate can still yield a runaway T (>1e5 K, even ~1e6 K). Such a T is unphysical
# for LIBS plasmas; outside this window the fit is flagged degenerate (T held at
# the prior) and can never be reported as converged. This is NOT an R^2 bound —
# it bounds the *inverted* temperature, the quantity the runaway actually escapes.
T_PHYSICAL_MIN_K = 2000.0
T_PHYSICAL_MAX_K = 50000.0


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


def _reliability_from_uncertainty_enabled() -> bool:
    """M8 Lever 7: opt-in coupling of per-element CI width to the reliability flag.

    Controlled by ``CFLIBS_RELIABILITY_FROM_UNCERTAINTY`` (default OFF == legacy:
    no per-element reliability labels, quality_flag/overall_reliable unchanged).
    """
    return os.environ.get("CFLIBS_RELIABILITY_FROM_UNCERTAINTY", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


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
    overall_reliable : bool
        M7 refuse-to-report verdict (Lever 6, Cristoforetti 2010). True only
        when n_e was measured from a literature Stark-width line
        (``ne_from_stark``) AND the McWhirter LTE criterion is satisfied AND
        the Cristoforetti multi-check ``quality_flag`` is acceptable-or-better.
        The Stark-provenance term is decisive: a pressure-balance fallback n_e
        is physically invalid, and the McWhirter check runs on that same n_e so
        it cannot catch the error. Defaults False (conservative): a
        fallback/degenerate solve that never reached the quality assessment is
        correctly not-reliable. Consumed by the CLI refuse-to-report gate when
        ``CFLIBS_REFUSE_TO_REPORT`` is set; always computed (a pure annotation —
        never alters T/n_e/composition).
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
    overall_reliable: bool = False
    per_element_reliability: Dict[str, str] = field(default_factory=dict)
    #: Mass (weight) fractions, sum to 1. ``concentrations`` are number/mole
    #: fractions on the peak-based path; this parallel field carries the
    #: mass-fraction view so consumers compare wt% like-for-like (DED Gap 4).
    #: Populated by ``run_pipeline``; empty if not yet computed.
    mass_fractions: Dict[str, float] = field(default_factory=dict)


@dataclass
class StarkDiagnosticLine:
    """A measured isolated line used to diagnose ``n_e`` from its Stark width.

    Stark broadening is the canonical electron-density diagnostic in LIBS
    (Tognoni 2010; Aragón & Aguilera 2010, *Spectrochim. Acta B* 65, 395 —
    Hα / Fe I / Si II). The solver inverts the project-wide electron-impact
    width law (:func:`cflibs.radiation.stark.estimate_ne_from_stark`) to obtain
    ``n_e`` from a measured FWHM, after deconvolving the Gaussian instrument and
    Doppler contributions.

    Attributes
    ----------
    measured_fwhm_nm : float
        Measured (Voigt) FWHM of the diagnostic line in nm.
    stark_w_ref_nm : float
        Reference electron-impact Stark FWHM for this line at
        ``REF_NE = 1e17 cm^-3``, ``T = 10000 K`` (nm). Same convention as the
        stored ``lines.stark_w`` column.
    stark_alpha : float, optional
        Temperature-scaling exponent (default 0.5).
    instrument_fwhm_nm : float
        Instrument response FWHM in nm (Gaussian), removed in quadrature.
    doppler_fwhm_nm : float
        Doppler (thermal) FWHM in nm (Gaussian), removed in quadrature.
    wavelength_nm : float, optional
        Diagnostic line wavelength (informational/logging only).
    """

    measured_fwhm_nm: float
    stark_w_ref_nm: float
    stark_alpha: float = 0.5
    instrument_fwhm_nm: float = 0.0
    doppler_fwhm_nm: float = 0.0
    wavelength_nm: Optional[float] = None


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


class _PythonIterationResult(NamedTuple):
    """Carried state produced by one ``_solve_python`` iteration step.

    ``should_break`` signals the ``common_fit is None`` early exit (before the
    history append); ``converged`` is the per-iteration convergence verdict.
    """

    T_K: float
    n_e: float
    T_corona: Optional[float]
    concentrations: Dict[str, float]
    last_common_fit: Optional[_CommonSlopeFit]
    boltzmann_degenerate: bool
    closure_degenerate: bool
    ne_from_stark: bool
    converged: bool
    should_break: bool


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
    # Boltzmann-fit degeneracy flag from the LAST iteration (slope >= 0 or
    # R^2 < min_boltzmann_r2). Initialized True ("until a clean fit proves
    # otherwise", mirroring the Python path) so a loop that never runs a
    # usable fit reports degenerate. Consumed host-side by ``_solve_lax`` for
    # the quality_metrics/converged parity gates (audit 02-F8).
    boltzmann_degenerate: Any


def _snapshot_record_levels(
    g_arr: np.ndarray,
    E_arr: np.ndarray,
    ip_ev: float,
    stage: int,
    i: int,
    g_I: List[np.ndarray],
    E_I: List[np.ndarray],
    ip_I: np.ndarray,
    g_II: List[np.ndarray],
    E_II: List[np.ndarray],
    ip_II: np.ndarray,
) -> None:
    """Record direct-sum level arrays for one element/stage (``from_solver`` helper)."""
    if stage == 1:
        g_I.append(np.asarray(g_arr, dtype=np.float64))
        E_I.append(np.asarray(E_arr, dtype=np.float64))
        ip_I[i] = float(ip_ev)
    else:
        g_II.append(np.asarray(g_arr, dtype=np.float64))
        E_II.append(np.asarray(E_arr, dtype=np.float64))
        ip_II[i] = float(ip_ev)


def _snapshot_append_empty_levels(
    stage: int,
    g_I: List[np.ndarray],
    E_I: List[np.ndarray],
    g_II: List[np.ndarray],
    E_II: List[np.ndarray],
) -> None:
    """Append empty level arrays for one element/stage (``from_solver`` helper)."""
    if stage == 1:
        g_I.append(np.zeros(0, dtype=np.float64))
        E_I.append(np.zeros(0, dtype=np.float64))
    else:
        g_II.append(np.zeros(0, dtype=np.float64))
        E_II.append(np.zeros(0, dtype=np.float64))


def _snapshot_record_coeffs(
    coeffs: np.ndarray,
    stage: int,
    i: int,
    coeffs_I: np.ndarray,
    coeffs_II: np.ndarray,
) -> None:
    """Record padded polynomial coefficients for one element/stage (``from_solver`` helper)."""
    # Pad/truncate to 5 coefficients
    n = min(coeffs.size, 5)
    if stage == 1:
        coeffs_I[i, :n] = coeffs[:n]
    else:
        coeffs_II[i, :n] = coeffs[:n]


def _snapshot_fetch_element_stage(
    solver: "IterativeCFLIBSSolver",
    el: str,
    stage: int,
    stage_idx: int,
    i: int,
    use_direct: np.ndarray,
    g_I: List[np.ndarray],
    E_I: List[np.ndarray],
    ip_I: np.ndarray,
    g_II: List[np.ndarray],
    E_II: List[np.ndarray],
    ip_II: np.ndarray,
    coeffs_I: np.ndarray,
    coeffs_II: np.ndarray,
) -> None:
    """Populate snapshot arrays for a single ``(element, stage)`` (``from_solver`` helper).

    Mirrors the per-stage branch of :meth:`_AtomicSnapshot.from_solver` exactly:
    direct-sum levels when available, else polynomial coefficients, else empty
    level arrays plus a NaN sentinel marking the scalar-fallback regime.
    """
    from cflibs.plasma.partition import get_levels_for_species

    lev = get_levels_for_species(solver.atomic_db, el, stage)
    if lev is not None:
        g_arr, E_arr, ip_ev = lev
        _snapshot_record_levels(g_arr, E_arr, ip_ev, stage, i, g_I, E_I, ip_I, g_II, E_II, ip_II)
        use_direct[i, stage_idx] = True
        return
    # Polynomial fallback
    pf = solver.atomic_db.get_partition_coefficients(el, stage)
    if pf:
        coeffs = np.asarray(pf.coefficients, dtype=np.float64)
        _snapshot_record_coeffs(coeffs, stage, i, coeffs_I, coeffs_II)
        use_direct[i, stage_idx] = False
        _snapshot_append_empty_levels(stage, g_I, E_I, g_II, E_II)
    else:
        # No data at all — record the empty level arrays and rely on
        # ``fallback_U_*`` per-element scalars at evaluation time.
        _snapshot_append_empty_levels(stage, g_I, E_I, g_II, E_II)
        use_direct[i, stage_idx] = False
        # coeffs already zeros — eval_poly will return exp(0)=1; we
        # use fallback in that case. Mark via NaN sentinel below.
        if stage == 1 and not pf:
            coeffs_I[i, :] = np.nan
        elif stage == 2 and not pf:
            coeffs_II[i, :] = np.nan


def _pad_ragged_arrays(arrays: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Pad ragged per-element level arrays to a uniform width (``from_solver`` helper)."""
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


def _align_E_width(E_pad: np.ndarray, g_pad: np.ndarray) -> np.ndarray:
    """Re-pad an E array to match the g array's width (``from_solver`` helper)."""
    if E_pad.shape[1] == g_pad.shape[1]:
        return E_pad
    new_E = np.zeros_like(g_pad)
    new_E[:, : E_pad.shape[1]] = E_pad
    return new_E


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
                _snapshot_fetch_element_stage(
                    solver,
                    el,
                    stage,
                    stage_idx,
                    i,
                    use_direct,
                    g_I,
                    E_I,
                    ip_I,
                    g_II,
                    E_II,
                    ip_II,
                    coeffs_I,
                    coeffs_II,
                )

        # Canonical per-species fallbacks for the scalar-fallback regime, set
        # lazily ONLY where _eval_partition_jax will actually read them:
        # neither direct levels nor a polynomial resolved (the NaN-coefficient
        # sentinel). Closed-shell ions get their exact U here (e.g. Na II →
        # 1.0, not 15.0), and the helper warns once per species. Eager
        # evaluation would add g0 probes (get_energy_levels) for every
        # element and break the no-sqlite-inside-loop prefetch budget.
        for i, el in enumerate(elements):
            if not use_direct[i, 0] and np.any(np.isnan(coeffs_I[i])):
                fallback_I[i] = canonical_partition_fallback(el, 1, solver.atomic_db, warn=True)
            if not use_direct[i, 1] and np.any(np.isnan(coeffs_II[i])):
                fallback_II[i] = canonical_partition_fallback(el, 2, solver.atomic_db, warn=True)

        # Pad ragged level arrays per stage
        gI_pad, mI = _pad_ragged_arrays(g_I)
        EI_pad, _ = _pad_ragged_arrays(E_I)
        # Re-pad EI to the same width as gI by reading mI shape
        EI_pad = _align_E_width(EI_pad, gI_pad)
        gII_pad, mII = _pad_ragged_arrays(g_II)
        EII_pad, _ = _pad_ragged_arrays(E_II)
        EII_pad = _align_E_width(EII_pad, gII_pad)

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

    def _stable_rel(intercepts, U_I, mult):
        """Overflow-safe rel = mult * U_I * exp(intercepts) via log-sum-exp.

        Mirrors :func:`closure._stabilized_relative_concentrations`: forming the
        relative abundances in log space and subtracting the max keeps large
        intercepts from overflowing ``jnp.exp`` to ``inf`` (the partition
        ``jnp.exp`` is already clipped to ``[-700, 700]``; the intercept ``exp``
        was not). The common shift cancels under every closure normalization
        below, so the normalized composition is unchanged.
        """
        log_rel = jnp.log(mult) + jnp.log(U_I) + intercepts
        return jnp.exp(log_rel - jnp.max(log_rel))

    if mode in {"", "standard"}:

        def _standard(intercepts, U_I, mult):
            rel = _stable_rel(intercepts, U_I, mult)
            total = jnp.sum(rel)
            return jnp.where(total > 0.0, rel / jnp.where(total > 0.0, total, 1.0), 0.0)

        return _standard

    if mode == "matrix":
        matrix_element = closure_kwargs.get("matrix_element")
        matrix_fraction = float(closure_kwargs.get("matrix_fraction", 0.9))
        if matrix_element not in elements:
            # Mirror ClosureEquation.apply_matrix_mode: fall through to standard
            def _matrix_fallback(intercepts, U_I, mult):
                rel = _stable_rel(intercepts, U_I, mult)
                total = jnp.sum(rel)
                return jnp.where(total > 0.0, rel / jnp.where(total > 0.0, total, 1.0), 0.0)

            return _matrix_fallback
        m_idx = elements.index(matrix_element)

        def _matrix(intercepts, U_I, mult):
            rel = _stable_rel(intercepts, U_I, mult)
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
            rel = _stable_rel(intercepts, U_I, mult)
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
        T_candidate = -1.0 / (slope * KB_EV)
        # Flag a runaway/unphysical inverted T (shallow-but-negative slope) the
        # same way as slope>=0 / low-R^2: hold T at the prior instead of letting
        # it escape the physical window. Mirrors the Python path.
        t_out_of_window = jnp.logical_or(
            T_candidate < T_PHYSICAL_MIN_K, T_candidate > T_PHYSICAL_MAX_K
        )
        degenerate = jnp.logical_or(
            jnp.logical_or(slope >= 0.0, r_squared < min_r2), t_out_of_window
        )
        T_new = jnp.where(degenerate, T_prev, T_candidate)
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
            boltzmann_degenerate=degenerate,
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
        apply_self_absorption: "bool | str" = False,
        min_boltzmann_r2: float = 0.3,
        boltzmann_weight_cap: float = 5.0,
        saha_boltzmann_graph: bool = False,
        use_jax_boltzmann: Optional[bool] = None,
        use_lax_while_loop: Optional[bool] = None,
        degeneracy_dominance_threshold: float = 0.8,
        degeneracy_min_elements: int = 4,
        assess_quality: bool = True,
        fixed_temperature_K: Optional[float] = None,
        db_isolation_gate: bool = False,
        db_isolation_fwhm_nm: float = 0.1,
        db_isolation_window_n_fwhm: float = 1.5,
        db_isolation_blend_fraction: float = 0.15,
        db_isolation_min_lines_per_element: int = 3,
    ):
        # JAX numerical-path selectors lifted onto the interface (arch review
        # c5-solver-flags). ``use_lax_while_loop`` chooses between the CPU
        # reference path and the JAX ``jax.lax.while_loop`` kernel for the whole
        # iterative solve loop (T1-3 / ADR-0001) and is read by ``solve``.
        # ``use_jax_boltzmann`` records the
        # ``CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION`` env selection on the
        # interface, mirroring the streaming FastAnalyzer's flag of the same
        # name; it is retained as the queryable record of the env-selected path.
        # Both flags were previously read silently from the process environment
        # (``CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION`` / ``CFLIBS_USE_LAX_WHILE_LOOP``)
        # at construction/solve time, which made the active numerical path
        # invisible to callers and to tests.
        #
        # Default ``None`` SEEDS each flag from its env var, so default
        # construction is byte-identical to the historical env-driven behavior
        # and every existing ``CFLIBS_USE_*`` invocation is unchanged. Passing
        # an explicit ``True``/``False`` is AUTHORITATIVE and overrides the env
        # var.
        self.use_jax_boltzmann = (
            _jax_boltzmann_composition_enabled()
            if use_jax_boltzmann is None
            else bool(use_jax_boltzmann)
        )
        self.use_lax_while_loop = (
            _lax_while_loop_enabled() if use_lax_while_loop is None else bool(use_lax_while_loop)
        )
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
        # Per-element Boltzmann-weight dynamic-range cap.
        #
        # The pooled common-slope fit weights each line by inverse variance,
        # w = 1/sigma_y^2. Under the Poisson intensity model sigma_I ~ sqrt(I),
        # so in fit space (y = ln(I/gA)) sigma_y ~ 1/sqrt(I) and w ~ I — the WLS
        # becomes intensity-weighted, and an element's fitted intercept q_s is
        # set almost entirely by its single brightest line. On real ChemCam
        # BHVO-2 the Fe I 382.0 nm line carries ~133x the weight of the next Fe
        # line and sits HIGH in the Boltzmann plane, lifting Fe's weighted
        # intercept from an honest unweighted ~18.3 to ~20.5 (+2.2) while Si
        # stays near its ensemble mean; this collapses the physical
        # q_Si - q_Fe ~ +3 spread to ~+0.2, and the closure
        # C_s ∝ U_s·exp(q_s)·mult then lets the large U_Fe drive Fe to ~72%
        # (cert 8.6%). Capping each element's weights at
        # ``boltzmann_weight_cap`` × median(valid weights) neutralizes the
        # single-bright-line concentration while preserving inverse-variance
        # weighting in the well-behaved regime (lines within K× of the median
        # are untouched). The cap is applied identically on the numpy host path
        # (``_fit_common_boltzmann_plane``) and the JAX padded-array path
        # (``_build_padded_arrays_from_obs``), so the CPU and lax kernels
        # consume bit-for-bit identical weights.
        #
        # Anti-overfit / generality guard: the cap is dataset-agnostic (no
        # element list, no certified value enters it) and its effect is
        # CSA-invariant — on the CSA BHVO-2 spectrum, whose per-element weight
        # spreads are already narrow, the cap is a near no-op (RMSE flat across
        # K), while it monotonically corrects the ChemCam over-attribution.
        # Set to 0 (or any non-positive value) to disable.
        self.boltzmann_weight_cap = float(boltzmann_weight_cap)
        # DB-aware spectral-isolation gate on the solver's Boltzmann line set.
        # OPT-IN (default False) -> byte-identical legacy behaviour when off.
        #
        # The per-element intercept q_s = y_mean - slope*x_mean is a weighted mean
        # of y = ln(I*lambda/(g*A)). When a measured peak integrates flux from >=2
        # transitions separated by less than the instrument FWHM, the numerator I is
        # the SUM of the blended lines, so every blended point is shifted *up* by
        # dy = ln(I_blend/I_true) > 0. Because the closure maps C_s ∝ exp(q_s), this
        # additive shift on the densest-spectrum elements (transition metals Fe/Ti,
        # which have the most DB lines per nm) becomes multiplicative
        # over-attribution. Inverse-variance A_ki weighting cannot remove it (a
        # blended line's A_ki is well characterized even though its *intensity* is
        # corrupted), and the existing detected-peak isolation test is blind to the
        # unresolved DB neighbors that do the blending. The
        # ``boltzmann_weight_cap`` bounds a single bright line's leverage but does
        # NOT see DB blends, so the two levers are orthogonal (the cap caps weight
        # magnitude; this gate inflates sigma for DB-contaminated lines).
        #
        # When enabled, ``_apply_db_isolation_gate`` queries the atomic DB for every
        # transition (of any solved element) within
        # ``±db_isolation_window_n_fwhm`` FWHM of each candidate line and
        # down-weights lines whose strongest predicted neighbor exceeds
        # ``db_isolation_blend_fraction`` of the candidate's own predicted
        # emissivity, recomputed against the prior-iteration (T, n_e,
        # concentrations) each pass. It is element-agnostic (triggers on local
        # spectral density, never identity) and down-weights (rather than
        # hard-dropping) so sparse cations near the min-lines floor are preserved.
        #
        # HOST-ONLY: enabling the gate forces the Python while-loop path (it is not
        # traced into the lax kernel), mirroring the saha_boltzmann_graph / Stark
        # diagnostics handling in ``solve``. Default-off keeps the lax path
        # byte-identical, so tests/jitpipe host/lax parity is unaffected.
        self.db_isolation_gate = bool(db_isolation_gate)
        self.db_isolation_fwhm_nm = float(db_isolation_fwhm_nm)
        self.db_isolation_window_n_fwhm = float(db_isolation_window_n_fwhm)
        self.db_isolation_blend_fraction = float(db_isolation_blend_fraction)
        self.db_isolation_min_lines_per_element = int(db_isolation_min_lines_per_element)
        # Saha-Boltzmann GRAPH intercept extraction (Aguilera & Aragon 2004,
        # *Spectrochim. Acta B* 59, 1861, "saha-boltzmann plot" / CD-SB graph).
        #
        # OPT-IN (default False). When enabled, the per-iteration intercept
        # extraction is swapped from the per-element-centered common-slope plane
        # (``_apply_saha_correction`` + ``_fit_common_boltzmann_plane``) to a
        # single POOLED global least-squares regression over ALL lines of ALL
        # species/stages at once (``_fit_saha_boltzmann_graph``): one shared
        # slope (-1/kT) plus one intercept dummy per element. Ion (stage z>1)
        # lines are shifted onto the neutral plane exactly as the existing Saha
        # correction does (x += IP1*(z-1); y -= ln(S)*(z-1)), so the fitted
        # per-element intercept is q_s = ln(n_I / U_I) on the neutral plane —
        # the SAME quantity ``_fit_common_boltzmann_plane`` produces, and the
        # SAME quantity the closure step (standard / oxide / matrix / ...) and
        # ``_compute_abundance_multipliers`` consume. The method is therefore
        # ORTHOGONAL to ``closure_mode``: SB-graph + oxide closure stack.
        #
        # Why it helps: the high-E_k ion lines (Fe II / Ti II land at
        # x = E_k + IP ~ 15-23 eV after the shift) give a long lever arm that
        # well-conditions each element's neutral intercept. Neutral-only Fe spans
        # ~1 eV, so its per-element centered fit is unstable (R^2 ~ 0); pooling
        # the shifted ion lines onto one graph stabilises the slope AND every
        # intercept simultaneously. On real ChemCam BHVO-2 this moves Fe from a
        # ~39 wt% over-attribution toward ~18 (cert 8.6) with global R^2 ~ 0.95.
        # See ``scripts/probe_saha_boltzmann_graph.py`` for the validated probe.
        self.saha_boltzmann_graph = bool(saha_boltzmann_graph)
        # Self-absorption correction — OBSERVABLE-GATED (bead 0jvr; audit
        # 02-F4 + 06-Q2).
        #
        # ``apply_self_absorption`` accepts a mode: ``'off'`` (or ``False``,
        # the default) applies no correction; ``'observable'`` (or ``True``)
        # corrects the measured line intensities ONCE, BEFORE the
        # Boltzmann / Saha-Boltzmann-graph fit, using only observables of the
        # measured spectrum (doublet intensity ratios — Pace 2025 — with the
        # Völker & Gornushkin 2023 Planck-ceiling closed form available when
        # calibrated peak radiances exist). Lines with no usable observable
        # that match the published SA-risk signature (bright resonance /
        # low-E_i lines, Fayyaz 2023) are DOWN-WEIGHTED in the fit via
        # uncertainty inflation, never boosted.
        #
        # The previous implementation recomputed the optical depth from the
        # *recovered* composition on every iteration — a positive feedback
        # loop (over-attributed element -> bigger tau -> bigger intensity
        # boost -> bigger intercept -> more mass at closure) that measurably
        # WORSENED intercept inflation on real ChemCam BHVO-2. That path was
        # deleted; see audit 02-inversion-solver.md F4 and git history for
        # the archive.
        self.self_absorption_mode: str = normalize_self_absorption_mode(apply_self_absorption)
        # Back-compat boolean view of the mode (read by quality metrics,
        # scripts and tests that predate the mode knob).
        self.apply_self_absorption: bool = self.self_absorption_mode != "off"
        self.self_absorption_corrector: Optional[ObservableSelfAbsorptionCorrector] = None
        self._last_sa_result: Optional[ObservableSAResult] = None
        if self.self_absorption_mode == "observable":
            self.self_absorption_corrector = ObservableSelfAbsorptionCorrector()
        # Stark n_e diagnostic stats from the most recent _update_ne_python
        # call (line count + MAD scatter), surfaced in quality_metrics.
        self._last_stark_stats: Dict[str, float] = {"n_lines": 0, "scatter_cm3": 0.0}
        # Composition-degeneracy ("keystone collapse") gate, bead
        # CF-LIBS-improved-tpkm / -cxxq. A composition where ONE element soaks
        # more than ``degeneracy_dominance_threshold`` of the closure mass out
        # of a candidate set of >= ``degeneracy_min_elements`` elements is the
        # collapse signature (the historical Na=98% blow-up from a basalt-like
        # 10-element set): the closure has lost discriminating power and the
        # result is untrustworthy, so the solve is reported converged=False
        # and ``quality_metrics['degenerate_composition']`` is set on BOTH the
        # Python and the lax solve paths. Small candidate sets are exempt
        # (binary alloys legitimately exceed 0.8: brass is ~90% Cu).
        self.degeneracy_dominance_threshold = float(degeneracy_dominance_threshold)
        self.degeneracy_min_elements = int(degeneracy_min_elements)
        # Post-loop reliability re-fit gate (perf knob). When True (default,
        # byte-identical legacy behaviour) ``_assemble_quality_metrics`` runs the
        # Cristoforetti multi-check (``_assess_reliability``), which re-fits a
        # per-element Boltzmann plot and re-evaluates U_I/U_II for every element
        # — pure annotation that never touches T/n_e/composition but costs
        # ~26-34% of a solve. When False the re-fit is SKIPPED and the
        # reliability keys are emitted conservatively (quality_flag='unknown',
        # consistency/scatter NaN, overall_reliable=False); the result-dict
        # key-set is IDENTICAL across both paths so downstream / refuse-to-report
        # consumers never KeyError. Default-False on the ded/batch/streaming
        # presets where the annotation layer is not consumed.
        self.assess_quality = bool(assess_quality)
        # Optimal-temperature / OPC lever (real-steel L1; Zhao 2018, Plasma Sci.
        # Technol. 20 035502). When set, the iterative solve HOLDS the plasma
        # temperature at this fixed value instead of recovering it from the
        # Boltzmann-plot slope. CF-LIBS composition is hypersensitive to T —
        # an ion-only-biased fit drifts T low (~6700 K) and the Saha ion->total
        # back-correction exp(E_ion/kT) then over-estimates trace minors. Fixing
        # T at the matrix's optimal value (found by minimizing a matrix-matched
        # standard's composition error) cuts that error substantially while the
        # rest of the iteration (Saha shift, intercepts, closure, n_e) stays
        # self-consistent at the held T. Default ``None`` => byte-identical
        # legacy slope-recovered-T behaviour. Physics-only (a plasma temperature
        # is a physical state variable, not a learned parameter).
        self.fixed_temperature_K = (
            float(fixed_temperature_K) if fixed_temperature_K is not None else None
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

        The fallback-ladder implementation was moved verbatim to
        :func:`cflibs.inversion.physics.self_absorption_inputs.evaluate_partition_function`
        so the solver and the self-absorption tooling scripts share one
        partition-function policy; this method delegates to it with
        ``self.atomic_db``, preserving byte-identical behavior.
        """
        return _build_evaluate_partition_function(self.atomic_db, element, ionization_stage, T_K)

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
        # Empirically, the refractory high-Z majors (Si, Fe, Ca, Al, Mg) are the
        # most corona-sensitive, so when a two-region corona temperature is
        # provided we use it (weighted) for their neutral-plane Saha scaling.
        # This element set and the weighting are empirical stabilization choices,
        # not a specific literature prescription.
        corona_sensitive = {"Si", "Fe", "Ca", "Al", "Mg"}
        for el in elements:
            U_I = lookup_partition_function(partition_funcs_I, el, 1, self.atomic_db)
            U_II = lookup_partition_function(partition_funcs_II, el, 2, self.atomic_db)

            T_saha = T_K
            if T_corona is not None and el in corona_sensitive:
                # Weighted temperature for Saha-Boltzmann scaling
                T_saha = 0.3 * T_K + 0.7 * T_corona

            S = self._compute_saha_ratio(el, T_saha, n_e_cm3, U_I, U_II, ips[el])
            multipliers[el] = 1.0 + max(S, 0.0)
        return multipliers

    def _compute_abundance_multipliers_uncertain(
        self,
        elements: List[str],
        T_K: float,
        n_e_cm3: float,
        n_e_relative_uncertainty: float,
        partition_funcs_I: Dict[str, float],
        partition_funcs_II: Dict[str, float],
        ips: Dict[str, float],
        T_corona: Optional[float] = None,
    ) -> Dict[str, "Any"]:
        """
        Neutral-plane abundance multipliers ``(1 + n_II/n_I)`` as UFloats.

        Mirrors :meth:`_compute_abundance_multipliers` but propagates an n_e
        uncertainty through the Saha factor. The electron density is wrapped as
        a UFloat with the given relative (1-sigma) uncertainty and threaded
        through :func:`saha_factor_with_uncertainty` so that each multiplier
        carries the resulting variance. Temperature is treated as exact here
        (its variance is handled by the Boltzmann slope path).
        """
        from uncertainties import ufloat
        from cflibs.inversion.physics.uncertainty import saha_factor_with_uncertainty

        safe_ne = max(float(n_e_cm3), 1e10)
        n_e_u = ufloat(safe_ne, n_e_relative_uncertainty * safe_ne)

        corona_sensitive = {"Si", "Fe", "Ca", "Al", "Mg"}
        multipliers: Dict[str, "Any"] = {}
        for el in elements:
            U_I = lookup_partition_function(partition_funcs_I, el, 1, self.atomic_db)
            U_II = lookup_partition_function(partition_funcs_II, el, 2, self.atomic_db)

            T_saha = T_K
            if T_corona is not None and el in corona_sensitive:
                T_saha = 0.3 * T_K + 0.7 * T_corona

            T_eV_exact = max(T_saha / EV_TO_K, 0.1)
            # Exact-T UFloat so saha_factor_with_uncertainty only injects the
            # n_e variance into this multiplier (T variance is propagated via
            # the pooled Boltzmann slope, not here, to avoid double counting).
            # std_dev==0 is intentional; suppress the (benign) library warning.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                T_eV_u = ufloat(T_eV_exact, 0.0)
                S = saha_factor_with_uncertainty(
                    T_eV_u,
                    n_e_u,
                    ips[el],
                    U_I,
                    U_II,
                    SAHA_CONST_CM3,
                )
            multipliers[el] = 1.0 + S
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

    def _estimate_ne_from_stark(
        self,
        stark_diagnostic: Optional["StarkDiagnosticLine"],
        T_K: float,
    ) -> Optional[float]:
        """Estimate ``n_e`` from a measured Stark line FWHM (primary diagnostic).

        Returns ``None`` when no usable Stark line is supplied, so the caller
        can fall back to the (physically weaker) pressure-balance estimate.
        """
        if stark_diagnostic is None:
            return None
        return estimate_ne_from_stark(
            measured_fwhm_nm=stark_diagnostic.measured_fwhm_nm,
            T_K=T_K,
            stark_w_ref=stark_diagnostic.stark_w_ref_nm,
            stark_alpha=stark_diagnostic.stark_alpha,
            instrument_fwhm_nm=stark_diagnostic.instrument_fwhm_nm,
            doppler_fwhm_nm=stark_diagnostic.doppler_fwhm_nm,
        )

    def _estimate_ne_from_stark_multi(
        self,
        stark_diagnostics: Sequence["StarkDiagnosticLine"],
        T_K: float,
    ) -> Tuple[Optional[float], int, float]:
        """Combine several Stark diagnostic lines into one robust ``n_e``.

        Each line is inverted independently at the *current* iteration
        temperature (the ``(T/T_ref)^alpha`` factor is the only T-dependent
        term left once the Gaussian components are deconvolved), then the
        per-line densities are combined with the median; the scatter is the
        robust 1.4826*MAD (std for n=2, 0 for a single line).

        Returns ``(ne_median, n_lines_used, scatter_cm3)``; ``ne_median`` is
        ``None`` when no line yields a usable density.
        """
        values: List[float] = []
        for diag in stark_diagnostics:
            ne = self._estimate_ne_from_stark(diag, T_K)
            if ne is not None:
                values.append(float(ne))
        if not values:
            return None, 0, 0.0
        arr = np.asarray(values, dtype=float)
        ne_median = float(np.median(arr))
        if len(arr) >= 2:
            mad = float(np.median(np.abs(arr - ne_median)))
            scatter = 1.4826 * mad if mad > 0 else float(np.std(arr))
        else:
            scatter = 0.0
        return ne_median, len(values), scatter

    def _apply_db_isolation_gate(
        self,
        obs_by_element: Dict[str, List[LineObservation]],
        T_K: float,
        n_e: float,
        concentrations: Dict[str, float],
    ) -> Dict[str, List[LineObservation]]:
        """Down-weight DB-blended lines before the Boltzmann/closure fit.

        Computes a per-line isolation weight against the full atomic-database
        transition list (see
        :func:`cflibs.inversion.physics.line_selection.compute_db_isolation_weights`)
        and folds it into each line's intensity uncertainty. Inflating ``sigma_I``
        by ``1/sqrt(w)`` lowers the line's inverse-variance fit weight
        ``w_fit = 1/sigma_y^2`` in :meth:`_fit_common_boltzmann_plane` (and the
        Saha-Boltzmann graph fit) by exactly the factor ``w`` — because
        ``LineObservation.y_uncertainty = intensity_uncertainty / intensity`` — so a
        blended line's additive ``+dy`` pull on the weighted intercept ``q_s`` is
        suppressed in proportion to how badly it is contaminated. Lines the gate
        leaves at weight 1.0 are returned unchanged (the common case).

        When ``db_isolation_gate`` is False this returns the *same* mapping object
        unchanged, so the default solve path is byte-identical. Otherwise a new
        mapping is returned; the input is never mutated. ``n_e`` is accepted for
        signature symmetry with the other per-iteration transforms and to allow a
        future n_e-dependent window; it is currently unused.
        """
        if not self.db_isolation_gate or self.atomic_db is None:
            return obs_by_element

        flat = [o for lst in obs_by_element.values() for o in lst]
        if not flat:
            return obs_by_element

        try:
            weights = compute_db_isolation_weights(
                flat,
                self.atomic_db,
                elements=list(obs_by_element.keys()),
                T_K=T_K,
                concentrations=concentrations,
                fwhm_nm=self.db_isolation_fwhm_nm,
                window_n_fwhm=self.db_isolation_window_n_fwhm,
                blend_fraction=self.db_isolation_blend_fraction,
                min_lines_per_element=self.db_isolation_min_lines_per_element,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("DB isolation gate failed (%s); skipping", exc)
            return obs_by_element

        gated: Dict[str, List[LineObservation]] = defaultdict(list)
        for el, obs_list in obs_by_element.items():
            for obs in obs_list:
                w = weights.get(id(obs), 1.0)
                if w >= 1.0 or w <= 0.0:
                    gated[el].append(obs)
                    continue
                # Inflate sigma_I by 1/sqrt(w) so the fit weight w_fit ~ 1/sigma^2
                # is multiplied by w. y/x/atomic data are untouched -- only the
                # line's influence on the weighted intercept is reduced.
                inflate = 1.0 / float(np.sqrt(w))
                gated[el].append(
                    LineObservation(
                        wavelength_nm=obs.wavelength_nm,
                        intensity=obs.intensity,
                        intensity_uncertainty=obs.intensity_uncertainty * inflate,
                        element=obs.element,
                        ionization_stage=obs.ionization_stage,
                        E_k_ev=obs.E_k_ev,
                        g_k=obs.g_k,
                        A_ki=obs.A_ki,
                        aki_uncertainty=obs.aki_uncertainty,
                    )
                )
        return dict(gated)

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

            # Bound the per-element weight dynamic range so the pooled intercept
            # is not dominated by a single bright line (see
            # ``self.boltzmann_weight_cap``). Applied here on the numpy host path
            # and identically in ``_build_padded_arrays_from_obs`` for the JAX
            # path, on the same post-valid-mask weights, so both kernels consume
            # bit-for-bit identical weights.
            ws = _cap_boltzmann_weights(ws, self.boltzmann_weight_cap)

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

    def _fit_saha_boltzmann_graph(
        self,
        obs_by_element: Dict[str, List[LineObservation]],
        T_K: float,
        n_e: float,
        ips: Dict[str, float],
    ) -> Optional[_CommonSlopeFit]:
        """Pooled Saha-Boltzmann GRAPH fit (Aguilera & Aragon 2004).

        Pools EVERY line of EVERY element/stage onto one graph and fits a single
        shared slope ``m = -1/(k_B T)`` together with one intercept dummy per
        element via a global least-squares system. Ion (stage ``z > 1``) lines
        are mapped onto the neutral plane with the *same* Saha transform the
        per-element path uses (:meth:`_apply_saha_correction`):

            neutral (z=1): x = E_k,                 y = ln(I*lambda/(g_k*A_ki))
            ion     (z>1): x = E_k + IP1*(z-1),     y = (...) - ln(S)*(z-1)

        with ``ln(S) = ln(SAHA_CONST_CM3 / n_e * T_eV**1.5)`` (no partition
        functions — these are absorbed into the fitted neutral intercept). The
        per-element intercept returned is therefore ``q_s = ln(n_I / U_I)`` on
        the neutral plane, identical in meaning to the intercept from
        :meth:`_fit_common_boltzmann_plane`, so the downstream
        :meth:`_compute_abundance_multipliers` ``(1 + n_II/n_I)`` scaling and the
        :class:`ClosureEquation` step (standard / oxide / matrix / ...) consume
        it unchanged.  This makes the SB-graph orthogonal to ``closure_mode``.

        Unlike the centered common-slope plane, this is a *global* fit: every
        element's intercept is estimated jointly against the shared slope, so the
        long lever arm contributed by the high-``x`` shifted ion lines
        well-conditions all intercepts at once. Inverse-variance weighting
        (with the same :attr:`boltzmann_weight_cap` dynamic-range bound applied
        per element as the common-slope path) is retained.

        Parameters
        ----------
        obs_by_element : dict
            RAW observations grouped by element (pre Saha remap). Both neutral
            and ionic lines are passed; the ion shift is applied internally.
        T_K, n_e : float
            Current plasma temperature [K] and electron density [cm^-3]; set the
            ``ln(S)`` ion->neutral shift.
        ips : dict
            First ionization potentials by element [eV] (effective IPs, already
            IPD-corrected by the caller when applicable).

        Returns
        -------
        _CommonSlopeFit | None
            Slope, per-element neutral intercepts, per-element stats and the
            global (pooled) weighted R^2. ``None`` when fewer than three usable
            lines survive or the global system is under-determined.
        """
        T_eV = max(T_K / EV_TO_K, 0.1)
        safe_ne = max(float(n_e), 1e10)
        ln_S = float(np.log((SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5)))

        # Collect shifted (x, y, w) rows per element across ALL species/stages.
        per_el_x, per_el_y, per_el_w = _collect_sb_graph_rows(obs_by_element, ips, ln_S)

        elements: List[str] = [el for el in obs_by_element.keys() if per_el_x.get(el)]
        if not elements:
            return None

        # Assemble the global design matrix A = [x | element-dummies] and solve
        # the weighted normal equations once. Columns: 0 = shared slope,
        # 1..E = per-element intercepts.
        el_index = {el: i for i, el in enumerate(elements)}
        solved = _solve_sb_graph_lstsq(elements, el_index, per_el_x, per_el_y, per_el_w)
        if solved is None:
            # Under-determined global system; let the caller fall back.
            return None
        slope, slope_variance, r_squared, intercepts = solved

        # Per-element stats on the SHIFTED (neutral-plane) coordinates so that
        # ``solve_with_uncertainty`` and any consumer of ``element_stats`` see
        # the same x/y values the global fit used.
        element_stats = _build_sb_graph_element_stats(elements, per_el_x, per_el_y, per_el_w)

        return _CommonSlopeFit(
            slope=slope,
            slope_variance=slope_variance,
            intercepts=intercepts,
            element_stats=element_stats,
            r_squared=r_squared,
        )

    def solve(
        self,
        observations: List[LineObservation],
        closure_mode: str = "standard",
        stark_diagnostic: Optional["StarkDiagnosticLine"] = None,
        stark_diagnostics: Optional[Sequence["StarkDiagnosticLine"]] = None,
        **closure_kwargs,
    ) -> CFLIBSResult:
        """
        Estimate plasma temperature, electron density, and elemental concentrations from spectral line observations using the iterative CF-LIBS algorithm.

        Routes to ``_solve_lax`` (``jax.lax.while_loop`` path, T1-3) when both
        ``self.use_lax_while_loop`` (seeded from
        :func:`_lax_while_loop_enabled` at construction, overridable via the
        constructor) and ``HAS_JAX`` are true; otherwise runs the Python
        ``for``-loop reference path in ``_solve_python``.

        Parameters:
            observations (List[LineObservation]): Spectral line observations to invert; lines are grouped by element.
            closure_mode (str): Closure method for converting Boltzmann intercepts to concentrations. One of "standard", "matrix", "oxide", "ilr", "pwlr", or "dirichlet_residual".
            stark_diagnostic (StarkDiagnosticLine, optional): Single measured isolated line used as the PRIMARY per-iteration ``n_e`` diagnostic via its Stark width (Tognoni 2010; Aragón & Aguilera 2010). When no diagnostic is available, ``n_e`` falls back to the (physically invalid for LIBS) 1-atm pressure balance, which logs a warning.
            stark_diagnostics (Sequence[StarkDiagnosticLine], optional): Multiple measured Stark diagnostic lines; the per-line densities are combined robustly (median, MAD scatter) each iteration. May be combined with ``stark_diagnostic``. Typically produced by :func:`cflibs.inversion.physics.stark_ne.measure_stark_ne`.
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
        # Observable-gated self-absorption correction (bead 0jvr): applied
        # ONCE, BEFORE the Boltzmann / SB-graph fit, on the measured line
        # list. It is a pure observation transform — every correction factor
        # derives from observables (doublet intensity ratios), never from
        # the recovered composition — so both the Python and the lax solve
        # paths consume the same corrected observations. SA-suspect lines
        # without a usable observable are down-weighted (uncertainty
        # inflated), not boosted; see ObservableSelfAbsorptionCorrector.
        self._last_sa_result = None
        if self.self_absorption_corrector is not None:
            sa_result = self.self_absorption_corrector.correct(observations)
            self._last_sa_result = sa_result
            observations = sa_result.observations

        # The Saha-Boltzmann graph intercept extraction is only implemented
        # on the Python path (its global lstsq is not traced into the lax
        # common-slope kernel), so it forces the Python loop.
        # The Stark n_e diagnostic is only wired into the Python reference loop
        # (the lax body's n_e update is a traced pressure-balance kernel), so a
        # supplied stark diagnostic forces the Python path.
        #
        # The DB-isolation gate is implemented only on the Python iteration body
        # (it queries the atomic DB and rebuilds per-line uncertainties each pass,
        # which is not traced into the lax common-slope kernel), so enabling it
        # forces the Python loop — keeping the lax path byte-identical when the
        # gate is off (default).
        diags: List["StarkDiagnosticLine"] = []
        if stark_diagnostic is not None:
            diags.append(stark_diagnostic)
        if stark_diagnostics:
            diags.extend(stark_diagnostics)
        if (
            HAS_JAX
            and self.use_lax_while_loop
            and not self.saha_boltzmann_graph
            and not self.db_isolation_gate
            and not diags
        ):
            try:
                return self._solve_lax(observations, closure_mode, **closure_kwargs)
            except _LaxFallback as exc:
                logger.info("lax.while_loop path bailed out (%s); using Python loop", exc)
                return self._solve_python(
                    observations, closure_mode, stark_diagnostics=diags, **closure_kwargs
                )
        return self._solve_python(
            observations, closure_mode, stark_diagnostics=diags, **closure_kwargs
        )

    def _prefetch_ips_python(self, elements: List[str]) -> Dict[str, float]:
        """Pre-fetch first ionization potentials for ``elements`` (``_solve_python`` helper)."""
        ips = {}
        for el in elements:
            # Need IP of neutral (I -> II)
            ip = self.atomic_db.get_ionization_potential(el, 1)
            if ip is None:
                logger.warning(f"No IP for {el} I, assuming high")
                ip = 15.0  # Fallback
            ips[el] = ip
        return ips

    def _evaluate_partition_functions(
        self, elements: List[str], T_K: float
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Evaluate stage-I and stage-II partition functions for ``elements`` at ``T_K``."""
        partition_funcs = {}  # U_I for each element
        partition_funcs_II = {}
        for el in elements:
            partition_funcs[el] = self._evaluate_partition_function(el, 1, T_K)
            partition_funcs_II[el] = self._evaluate_partition_function(el, 2, T_K)
        return partition_funcs, partition_funcs_II

    def _select_common_fit(
        self,
        sa_obs_by_element: Dict[str, List[LineObservation]],
        T_K: float,
        n_e: float,
        effective_ips: Dict[str, float],
    ) -> Optional[_CommonSlopeFit]:
        """Choose the intercept-extraction mode and run it (``_solve_python`` helper).

        Saha-Boltzmann graph (opt-in) vs. the default common-slope plane (with
        ionic lines pre-mapped to the neutral plane). Both yield per-element
        neutral intercepts feeding the same closure step.
        """
        if self.saha_boltzmann_graph:
            return self._fit_saha_boltzmann_graph(sa_obs_by_element, T_K, n_e, effective_ips)
        # Map ionic lines to the neutral energy plane, then fit.
        corrected_obs_map = self._apply_saha_correction(sa_obs_by_element, T_K, n_e, effective_ips)
        return self._fit_common_boltzmann_plane(corrected_obs_map)

    def _dispatch_closure(
        self,
        closure_mode: str,
        intercepts: Dict[str, float],
        partition_funcs: Dict[str, float],
        abundance_multipliers: Dict[str, float],
        closure_kwargs: Dict[str, Any],
    ):
        """Apply the selected closure equation (``_solve_python`` helper).

        Identical dispatch to the inline if/elif chain it replaces; numerics
        are unchanged.
        """
        if closure_mode == "matrix":
            return ClosureEquation.apply_matrix_mode(
                intercepts,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
                **closure_kwargs,
            )
        elif closure_mode == "oxide":
            return ClosureEquation.apply_oxide_mode(
                intercepts,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
                **closure_kwargs,
            )
        elif closure_mode == "ilr":
            return ClosureEquation.apply_ilr(
                intercepts,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )
        elif closure_mode == "pwlr":
            return ClosureEquation.apply_pwlr(
                intercepts,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
                **closure_kwargs,
            )
        elif closure_mode == "dirichlet_residual":
            return ClosureEquation.apply_dirichlet_residual(
                intercepts,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
                **closure_kwargs,
            )
        else:
            return ClosureEquation.apply_standard(
                intercepts,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )

    def _pressure_balance_ne(
        self,
        concentrations: Dict[str, float],
        T_K: float,
        n_e: float,
        partition_funcs: Dict[str, float],
        partition_funcs_II: Dict[str, float],
        effective_ips: Dict[str, float],
    ) -> float:
        """Isobaric 1-atm pressure-balance ``n_e`` fallback (``_solve_python`` helper).

        Physically non-standard for LIBS — used only when no usable Stark
        diagnostic is available. Numerics identical to the inline fallback.
        """
        # Calculate avg_Z based on Saha ratios.
        # eps_s = n_II / (n_I + n_II) = S / (1+S) (electrons per atom).
        total_eps = 0.0
        for el, C_s in concentrations.items():
            U_I = lookup_partition_function(partition_funcs, el, 1, self.atomic_db)
            U_II = lookup_partition_function(partition_funcs_II, el, 2, self.atomic_db)
            S = self._compute_saha_ratio(el, T_K, n_e, U_I, U_II, effective_ips[el])
            eps_s = S / (1.0 + S)
            total_eps += C_s * eps_s

        avg_Z = total_eps
        # n_tot = P / (k T (1 + avg_Z));  n_e = avg_Z * n_tot
        n_tot = self.pressure_pa / (KB * T_K * (1.0 + avg_Z))
        n_tot_cm3 = n_tot * 1e-6  # m^-3 -> cm^-3
        return avg_Z * n_tot_cm3

    def _update_ne_python(
        self,
        stark_diagnostics: Sequence["StarkDiagnosticLine"],
        T_K: float,
        n_e: float,
        concentrations: Dict[str, float],
        partition_funcs: Dict[str, float],
        partition_funcs_II: Dict[str, float],
        effective_ips: Dict[str, float],
    ) -> Tuple[float, bool]:
        """Compute the next ``n_e`` and its provenance flag (``_solve_python`` helper).

        PRIMARY: Stark-width diagnostic (canonical LIBS n_e; multiple lines
        are combined by the median with MAD scatter — Ciucci 1999 / Tognoni
        2010 treat the Stark n_e as an INPUT to the Saha terms, not a closure
        iterate). FALLBACK: the physically-invalid isobaric 1-atm pressure
        balance, which logs a warning. Returns ``(ne_new, ne_from_stark)``;
        the per-call line count and scatter are stashed on
        ``self._last_stark_stats`` for the quality metrics.
        """
        ne_stark, n_lines, scatter = self._estimate_ne_from_stark_multi(stark_diagnostics, T_K)
        if ne_stark is not None:
            self._last_stark_stats = {"n_lines": n_lines, "scatter_cm3": scatter}
            return ne_stark, True
        self._last_stark_stats = {"n_lines": 0, "scatter_cm3": 0.0}
        if stark_diagnostics:
            logger.warning(
                "Stark diagnostic line(s) supplied but yielded no usable n_e "
                "(width fully accounted for by instrument+Doppler, or no "
                "reference Stark width); falling back to 1-atm pressure balance."
            )
        logger.warning(
            "No usable Stark n_e diagnostic; using the isobaric 1-atm "
            "(STP) pressure-balance fallback for n_e. This is physically "
            "non-standard for LIBS (hypersonic shock, never static 1 atm) "
            "and should be treated as a coarse last-resort estimate."
        )
        ne_new = self._pressure_balance_ne(
            concentrations, T_K, n_e, partition_funcs, partition_funcs_II, effective_ips
        )
        return ne_new, False

    def _run_python_iteration(
        self,
        elements: List[str],
        obs_by_element: Dict[str, List[LineObservation]],
        ips: Dict[str, float],
        closure_mode: str,
        stark_diagnostics: Sequence["StarkDiagnosticLine"],
        closure_kwargs: Dict[str, Any],
        *,
        T_K: float,
        n_e: float,
        T_corona: Optional[float],
        concentrations: Dict[str, float],
        last_common_fit: Optional[_CommonSlopeFit],
    ) -> _PythonIterationResult:
        """Run one self-consistent CF-LIBS iteration (``_solve_python`` loop body).

        Byte-for-byte equivalent to the original inline loop body: partition
        functions, Boltzmann fit, T update, closure, n_e update, and the
        convergence verdict. Returns the carried state for the next iteration
        plus the ``should_break`` / ``converged`` flags. The observable-gated
        self-absorption correction is NOT part of the iteration: it is a pure
        observation transform applied once in :meth:`solve`, before the loop
        (audit 02-F4 — the per-iteration composition-fed placement was the
        feedback problem).
        """
        T_prev = T_K
        ne_prev = n_e

        T_eV = T_K / EV_TO_K
        if T_eV < 0.1:
            T_eV = 0.1  # clamp

        # 2. Calculate Partition Functions & Saha Corrections
        partition_funcs, partition_funcs_II = self._evaluate_partition_functions(elements, T_K)

        effective_ips = self._compute_effective_ips(ips, n_e, T_K)

        # DB-aware spectral-isolation gate (additive-bias removal at the source):
        # down-weight lines blended by unresolved DB neighbors BEFORE the
        # Boltzmann/closure fit, recomputed against the current (prior-iteration)
        # plasma state so the gate tracks T/n_e/composition rather than the
        # iteration-0 defaults. No-op (returns the same object) when
        # ``db_isolation_gate`` is False or no line has a stronger predicted DB
        # neighbor, so the default path is byte-identical.
        sa_obs_by_element = self._apply_db_isolation_gate(obs_by_element, T_K, n_e, concentrations)

        # 3. Multi-species Boltzmann Fit.
        #
        # Two intercept-extraction modes (both yield per-element neutral
        # intercepts q_s = ln(n_I/U_I) that feed the SAME closure step):
        #   * common-slope plane (default): per-element centered WLS with
        #     ionic lines pre-mapped to the neutral plane by _apply_saha_-
        #     correction, then centered points pooled for ONE slope.
        #   * Saha-Boltzmann graph (opt-in, self.saha_boltzmann_graph):
        #     a single GLOBAL regression over all lines of all species at
        #     once (shared slope + per-element intercept dummies), the ion
        #     shift applied inside _fit_saha_boltzmann_graph. The high-x
        #     shifted ion lines give the lever arm that stabilises every
        #     intercept jointly (see __init__ note).
        common_fit = self._select_common_fit(sa_obs_by_element, T_K, n_e, effective_ips)
        if common_fit is None:
            logger.warning("Insufficient points for fit")
            return _PythonIterationResult(
                T_K=T_K,
                n_e=n_e,
                T_corona=T_corona,
                concentrations=concentrations,
                last_common_fit=last_common_fit,
                boltzmann_degenerate=True,
                closure_degenerate=False,
                ne_from_stark=False,
                converged=False,
                should_break=True,
            )

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
        if self.fixed_temperature_K is not None:
            # L1 optimal-T lever: hold T at the fixed value (no slope->T update,
            # no damping). The Boltzmann fit still supplies the per-element
            # intercepts; only the temperature is pinned. The slope is not used
            # for T, so it does not gate degeneracy here.
            boltzmann_degenerate = False
            T_new = self.fixed_temperature_K
            T_K = self.fixed_temperature_K
        else:
            boltzmann_degenerate = slope >= 0 or fit_r2 < self.min_boltzmann_r2
            if boltzmann_degenerate:
                T_new = T_prev  # hold at prior; slope carries no usable T
            else:
                T_new = -1.0 / (slope * KB_EV)
                # Upper/lower physical clamp: a shallow-but-negative slope passing
                # the R^2 gate can still invert to a runaway T (>1e5 K). Flag it
                # degenerate and hold T at the prior rather than reporting an
                # unphysical T as converged (do NOT silently clamp to 50000).
                if not (T_PHYSICAL_MIN_K <= T_new <= T_PHYSICAL_MAX_K):
                    boltzmann_degenerate = True
                    T_new = T_prev

            # Damping
            T_K = 0.5 * T_prev + 0.5 * T_new

        if self.two_region:
            # Empirical two-region DOF reduction: take the cooler outer/corona
            # zone at ~0.8 of the core temperature. The 0.8 is a common
            # stabilization choice for two-region LTE fits; it has NO specific
            # literature attribution (it is not a Hermann 2017 value).
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
        closure_res = self._dispatch_closure(
            closure_mode,
            intercepts,
            partition_funcs,
            abundance_multipliers,
            closure_kwargs,
        )

        concentrations = closure_res.concentrations

        # 5. Update electron density.
        #
        # PRIMARY: Stark broadening of a measured isolated line is the
        # canonical n_e diagnostic in LIBS (Tognoni 2010; Aragón & Aguilera
        # 2010, Spectrochim. Acta B 65, 395 — Hα / Fe I / Si II). When a
        # usable Stark line is supplied we invert the electron-impact width
        # law for n_e (instrument + Doppler deconvolved) and use it directly.
        #
        # FALLBACK: the isobaric 1-atm (STP) pressure balance. This is
        # *physically invalid* for a LIBS plasma — the laser-induced plasma
        # is a hypersonic shock at ~1e11 Pa initially and is NEVER at static
        # 1 atm in the analysis window — so it is demoted to a last resort
        # and emits a warning whenever it drives the n_e update.
        ne_new, ne_from_stark = self._update_ne_python(
            stark_diagnostics,
            T_K,
            n_e,
            concentrations,
            partition_funcs,
            partition_funcs_II,
            effective_ips,
        )

        # Damping
        n_e = 0.5 * ne_prev + 0.5 * ne_new

        # Composition degeneracy gate (supersedes PR #220, which only
        # *reported* the flag). A single element soaking more than
        # ``degeneracy_dominance_threshold`` of the closure mass out of a
        # >= ``degeneracy_min_elements`` candidate set is the "keystone
        # collapse" signature: the closure has lost discriminating power and
        # the composition is untrustworthy. Acting on it (not just reporting)
        # means such a solve can NEVER be flagged converged.
        closure_degenerate = self._validate_composition_degeneracy(concentrations)

        # Check convergence. A degenerate Boltzmann fit holds T at the prior,
        # which would otherwise satisfy |ΔT| < tol spuriously — so a
        # degenerate fit can never be reported as converged. Likewise a
        # degenerate composition is never reported as converged. The False
        # flag tells callers (and round-trip/NIST tooling) to treat T and C
        # as unconstrained.
        converged = (
            not boltzmann_degenerate
            and not closure_degenerate
            and abs(T_K - T_prev) < self.t_tolerance_k
            and abs(n_e - ne_prev) / ne_prev < self.ne_tolerance_frac
        )

        return _PythonIterationResult(
            T_K=T_K,
            n_e=n_e,
            T_corona=T_corona,
            concentrations=concentrations,
            last_common_fit=last_common_fit,
            boltzmann_degenerate=boltzmann_degenerate,
            closure_degenerate=closure_degenerate,
            ne_from_stark=ne_from_stark,
            converged=converged,
            should_break=False,
        )

    def _validate_composition_degeneracy(self, concentrations: Dict[str, float]) -> bool:
        """Apply the keystone-collapse gate with the solver's configured knobs.

        One element soaking more than ``self.degeneracy_dominance_threshold``
        of the closure mass out of a candidate set of at least
        ``self.degeneracy_min_elements`` elements means the closure has lost
        discriminating power (bead CF-LIBS-improved-tpkm). Used identically by
        the Python and lax solve paths.
        """
        return ClosureEquation.validate_degeneracy(
            concentrations,
            threshold=self.degeneracy_dominance_threshold,
            min_elements=self.degeneracy_min_elements,
        )

    def _warn_degenerate_composition(self, concentrations: Dict[str, float]) -> None:
        """Log the keystone-collapse warning for a degenerate final composition."""
        if not concentrations:
            return
        dominant = max(concentrations, key=lambda el: concentrations[el])
        logger.warning(
            "Degenerate composition: %s soaks %.1f%% of the closure mass out of "
            "%d candidate elements (> %.0f%% dominance threshold). The closure "
            "has lost discriminating power (keystone collapse); reporting "
            "converged=False and quality_metrics['degenerate_composition']=1.0.",
            dominant,
            100.0 * concentrations[dominant],
            len(concentrations),
            100.0 * self.degeneracy_dominance_threshold,
        )

    def _mcwhirter_delta_e_resonance(self, observations: List[LineObservation]) -> Optional[float]:
        """Resonance-line McWhirter delta_E from the lines table (M7 sub-lever b).

        The physically-correct McWhirter delta_E is the energy of the resonance
        transition — the dipole-allowed line out of the ground state, whose fast
        radiative decay is precisely what electron collisions must overcome for
        LTE to hold (Cristoforetti et al. 2010, Spectrochim. Acta B 65, 86-95).
        For each (element, ionization_stage) present in the observations we take
        the upper-level energy of the STRONGEST (max A_ki) resonance line
        (``is_resonance``, i.e. E_lower ~ 0); the binding delta_E for the
        multi-element plasma is the MAX over species (the hardest-to-thermalise
        species sets the LTE floor). Validated against literature resonance
        energies: Ca I 2.93, Na I 2.10, Mg I 4.35, Si I 4.93 eV (DB matches).

        Returns ``None`` if no species exposes a resonance line, so the caller
        falls back to the observation-derived delta_E (legacy ``max(E_k)``).

        Why not the largest adjacent level gap, nor ``max(E_k)``: the largest
        adjacent gap lands on low-lying SAME-PARITY (forbidden) terms that do
        not stress LTE (Fe I -> 0.74 eV, too lax); ``max(E_k)`` models an
        implausible single ground->highest-level collision (Fe I -> 7.5 eV,
        too strict -> false-rejects valid LTE via the cubic n_e floor).
        """
        seen: set = set()
        best: Optional[float] = None
        for obs in observations:
            key = (obs.element, obs.ionization_stage)
            if key in seen:
                continue
            seen.add(key)
            # Defensive per-species: a malformed / pluggable AtomicDataSource
            # backend (transition missing A_ki, non-numeric E_k_ev, raising
            # get_transitions) must degrade to "skip this species", never abort
            # the solve. Mirrors _assess_reliability's broad-guard posture, so
            # the docstring's "returns None when no resonance line" promise holds
            # for any backend, not just the well-formed AtomicDatabase rows.
            try:
                transitions = self.atomic_db.get_transitions(obs.element, obs.ionization_stage)
            except Exception:  # pragma: no cover - defensive backend guard
                continue
            best_species_aki: Optional[float] = None
            best_species_de: Optional[float] = None
            for transition in transitions:
                try:
                    if not getattr(transition, "is_resonance", False):
                        continue
                    aki = float(getattr(transition, "A_ki", float("nan")))
                    de = float(getattr(transition, "E_k_ev", float("nan")))
                except Exception:
                    continue
                if not (np.isfinite(aki) and aki > 0.0 and np.isfinite(de) and de > 0.0):
                    continue
                if best_species_aki is None or aki > best_species_aki:
                    best_species_aki = aki
                    best_species_de = de
            if best_species_de is None:
                continue
            de = best_species_de
            if best is None or de > best:
                best = de
        return best

    def _assess_reliability(
        self,
        observations: List[LineObservation],
        T_K: float,
        n_e: float,
        concentrations: Dict[str, float],
    ) -> Dict[str, object]:
        """Run the Cristoforetti multi-check (``QualityAssessor.assess``).

        M7 Lever 6: wire the previously-dead ``QualityAssessor`` into the
        production solver so every result carries a Saha-Boltzmann-consistency
        / inter-element-T-scatter / closure ``quality_flag`` (Cristoforetti et
        al. 2010, Spectrochim. Acta B 65, 86-95 — necessary, not sufficient).

        DEFENSIVE BY DESIGN: reliability is an *annotation*, never core math.
        Any failure (missing IP/partition data, a mock DB, a numerical edge)
        yields a conservative ``quality_flag='unknown'`` rather than breaking
        the solve. The IP/partition inputs are sourced from the same atomic-DB
        provider the solve itself used, evaluated at the fitted ``T_K``.
        """
        unknown = {
            "quality_flag": "unknown",
            "saha_boltzmann_consistency": float("nan"),
            "inter_element_t_std_frac": float("nan"),
        }
        try:
            from cflibs.inversion.physics.quality import QualityAssessor

            ips: Dict[str, float] = {}
            u_i: Dict[str, float] = {}
            u_ii: Dict[str, float] = {}
            concentrations_for_assess = dict(concentrations)
            for el in concentrations_for_assess:
                ip = self.atomic_db.get_ionization_potential(el, 1)
                if ip is None:
                    continue
                ips[el] = float(ip)
                u_i[el] = float(self._evaluate_partition_function(el, 1, T_K))
                u_ii[el] = float(self._evaluate_partition_function(el, 2, T_K))
            metrics = QualityAssessor().assess(
                observations=observations,
                temperature_K=T_K,
                electron_density_cm3=n_e,
                concentrations=concentrations_for_assess,
                ionization_potentials=ips,
                partition_funcs_I=u_i,
                partition_funcs_II=u_ii,
            )
            return {
                "quality_flag": str(metrics.quality_flag),
                "saha_boltzmann_consistency": float(metrics.saha_boltzmann_consistency),
                "inter_element_t_std_frac": float(metrics.inter_element_t_std_frac),
            }
        except Exception as exc:  # pragma: no cover - defensive annotation path
            logger.debug("Reliability assessment failed (annotation only): %s", exc)
            return dict(unknown)

    def _assemble_quality_metrics(
        self,
        observations: List[LineObservation],
        T_K: float,
        n_e: float,
        concentrations: Dict[str, float],
        fit_r2: float,
        boltzmann_degenerate: bool,
        closure_degenerate: bool,
        ne_from_stark: bool,
        stark_n_lines: int = 0,
        stark_ne_scatter_cm3: float = 0.0,
    ) -> Dict[str, float]:
        """Assemble the post-loop ``quality_metrics`` dict.

        SINGLE source of truth for the quality-metric key set: both
        ``_solve_python`` and ``_solve_lax`` route through here, so the two
        paths emit identical keys by construction (audit 02-F8: the lax path
        previously emitted only ``r_squared_last`` + LTE keys, letting
        ``converged=True`` coexist with ``boltzmann_r_squared=None``).
        """
        # LTE validity check
        from cflibs.plasma.lte_validator import LTEValidator

        lte_validator = LTEValidator()
        # M7 sub-lever b (opt-in): use the strongest resonance-line upper
        # energy from the transitions table for the McWhirter delta_E instead
        # of the legacy max(E_k). Gated behind CFLIBS_MCWHIRTER_RESONANCE_DE
        # (default OFF == byte-identical legacy LTE verdict); falls back to the
        # observation-derived delta_E when no resonance line is available.
        delta_e_override: Optional[float] = None
        if os.environ.get("CFLIBS_MCWHIRTER_RESONANCE_DE", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            delta_e_override = self._mcwhirter_delta_e_resonance(observations)
        lte_report = lte_validator.validate(
            T_K=T_K,
            n_e_cm3=n_e,
            observations=observations,
            delta_E_eV=delta_e_override,
        )
        quality_metrics = {
            # CANONICAL fit quality: R^2 of the common-slope plane / SB-graph
            # fit, read from the fit where it happens (the final
            # _CommonSlopeFit on the Python path; the final lax kernel fit on
            # the lax path).
            "boltzmann_r_squared": fit_r2,
            # Compatibility alias of ``boltzmann_r_squared`` with the same
            # provenance (do not read both). Still load-bearing: the CLI reads
            # it as the fallback for ``boltzmann_r_squared`` (cli/main.py) and
            # several active tests assert on it, so it is not removable.
            "r_squared_last": fit_r2,
            # Number of elements that received an intercept from the fit and
            # entered the closure.
            "n_elements_fit": float(len(concentrations)),
            # Silent-failure gates (supersedes PR #220 — incorporates its
            # additive metrics AND acts on them). These let downstream consumers
            # detect a non-physical Boltzmann slope or a collapsed composition
            # even though the solver already refuses to flag such a solve as
            # converged.
            "degenerate_composition": float(closure_degenerate),
            # Compatibility alias of ``degenerate_composition`` (pre-cxxq key).
            # Still load-bearing: the CLI reads ``closure_degenerate`` directly
            # (cli/main.py) and closed_form.py emits it, so it is not removable.
            "closure_degenerate": float(closure_degenerate),
            "boltzmann_degenerate": float(boltzmann_degenerate),
            # n_e provenance: 1.0 when the canonical Stark-width diagnostic drove
            # the final n_e, 0.0 when the physically-invalid 1-atm pressure
            # balance fallback was used.
            "ne_from_stark": float(ne_from_stark),
            # Number of literature-grade Stark lines combined for the final n_e
            # and their robust (1.4826*MAD) scatter — the n_e trust surface.
            "stark_n_lines": float(stark_n_lines),
            "stark_ne_scatter_cm3": float(stark_ne_scatter_cm3),
        }
        quality_metrics.update(lte_report.quality_metrics)
        # Self-absorption diagnostics from the observable-gated pre-fit
        # correction (bead 0jvr). All zero when SA mode is 'off' or no
        # observable fired. ``max_tau_estimate`` and the legacy
        # ``self_absorption_max_tau`` alias both report the largest
        # observable-derived optical depth (doublet-ratio / Planck-ceiling).
        sa = self._last_sa_result
        quality_metrics["self_absorption_applied"] = float(self.apply_self_absorption)
        quality_metrics["self_absorption_max_tau"] = float(sa.max_tau) if sa else 0.0
        quality_metrics["max_tau_estimate"] = float(sa.max_tau) if sa else 0.0
        quality_metrics["n_lines_sa_corrected"] = float(sa.n_corrected) if sa else 0.0
        quality_metrics["n_lines_sa_suspect"] = float(sa.n_suspect) if sa else 0.0

        # M7 Lever 6 refuse-to-report: Cristoforetti multi-check + overall_reliable.
        # Pure additive annotation — these keys never alter T/n_e/composition.
        # Computed on BOTH solve paths (shared builder) so key-set parity holds.
        #
        # PERF GATE (assess_quality, default True): ``_assess_reliability`` re-fits
        # a per-element Boltzmann plot and re-evaluates U_I/U_II for every element
        # on EVERY solve (~26-34% of solve cost). When ``assess_quality`` is False
        # (ded/batch/streaming presets) we SKIP that re-fit and emit conservative
        # unknown/NaN reliability keys instead. The emitted KEY-SET is byte-for-byte
        # identical across both branches — only the *values* differ — so downstream
        # consumers and refuse-to-report logic never KeyError. T/n_e/composition are
        # untouched either way (this is the annotation layer only).
        if self.assess_quality:
            reliability = self._assess_reliability(observations, T_K, n_e, dict(concentrations))
            quality_metrics["quality_flag"] = reliability["quality_flag"]
            quality_metrics["saha_boltzmann_consistency"] = reliability[
                "saha_boltzmann_consistency"
            ]
            quality_metrics["inter_element_t_std_frac"] = reliability["inter_element_t_std_frac"]
            # overall_reliable = {n_e from a Stark line} AND {McWhirter satisfied}
            # AND {quality_flag >= acceptable} (roadmap M7c).
            # 'unknown'/'poor'/'reject' flags are NOT reliable. The n_e-provenance
            # term is decisive: when the canonical Stark-width diagnostic was
            # unavailable the solver falls back to a physically-invalid 1-atm
            # pressure balance, and the McWhirter LTE check is itself evaluated
            # ON that fallback n_e, so it cannot catch a bad n_e. A result whose
            # n_e is a pressure-balance guess is therefore never trustworthy.
            mcwhirter_ok = bool(lte_report.mcwhirter.satisfied)
            quality_metrics["overall_reliable"] = bool(
                ne_from_stark
                and mcwhirter_ok
                and reliability["quality_flag"] in ("excellent", "good", "acceptable")
            )
        else:
            quality_metrics["quality_flag"] = "unknown"
            quality_metrics["saha_boltzmann_consistency"] = float("nan")
            quality_metrics["inter_element_t_std_frac"] = float("nan")
            # 'unknown' is never trustworthy, so the conservative verdict is False
            # regardless of the (still-cheap) McWhirter check.
            quality_metrics["overall_reliable"] = False
        return quality_metrics

    def _build_python_quality_metrics(
        self,
        observations: List[LineObservation],
        T_K: float,
        n_e: float,
        concentrations: Dict[str, float],
        last_common_fit: Optional[_CommonSlopeFit],
        boltzmann_degenerate: bool,
        closure_degenerate: bool,
        ne_from_stark: bool,
    ) -> Dict[str, float]:
        """Assemble the post-loop ``quality_metrics`` dict (``_solve_python`` helper)."""
        fit_r2_final = last_common_fit.r_squared if last_common_fit is not None else 0.0
        stark_stats = getattr(self, "_last_stark_stats", None) or {}
        return self._assemble_quality_metrics(
            observations,
            T_K,
            n_e,
            concentrations,
            fit_r2=fit_r2_final,
            boltzmann_degenerate=boltzmann_degenerate,
            closure_degenerate=closure_degenerate,
            ne_from_stark=ne_from_stark,
            stark_n_lines=int(stark_stats.get("n_lines", 0)) if ne_from_stark else 0,
            stark_ne_scatter_cm3=(
                float(stark_stats.get("scatter_cm3", 0.0)) if ne_from_stark else 0.0
            ),
        )

    def _solve_python(
        self,
        observations: List[LineObservation],
        closure_mode: str = "standard",
        stark_diagnostic: Optional["StarkDiagnosticLine"] = None,
        stark_diagnostics: Optional[Sequence["StarkDiagnosticLine"]] = None,
        **closure_kwargs,
    ) -> CFLIBSResult:
        """Reference Python ``for``-loop implementation of :meth:`solve`.

        Bit-for-bit equivalent to the pre-T1-3 ``solve`` body; the public
        :meth:`solve` routes through here when ``CFLIBS_USE_LAX_WHILE_LOOP`` is
        unset (default) or when JAX is unavailable.
        """
        # 1. Initialization
        # L1 fixed-T lever: seed the loop at the held temperature so the first
        # Saha correction / closure already use it (default None => 10000 K seed).
        T_K = 10000.0 if self.fixed_temperature_K is None else self.fixed_temperature_K
        T_corona = None
        n_e = 1.0e17

        # Normalise the single/multi Stark-diagnostic inputs into one list.
        diags: List["StarkDiagnosticLine"] = []
        if stark_diagnostic is not None:
            diags.append(stark_diagnostic)
        if stark_diagnostics:
            diags.extend(stark_diagnostics)

        # Cache static data (IPs, atomic data)
        # Group observations by element
        obs_by_element = defaultdict(list)
        for obs in observations:
            obs_by_element[obs.element].append(obs)

        elements = list(obs_by_element.keys())

        # Pre-fetch Ionization Potentials
        ips = self._prefetch_ips_python(elements)

        # Iteration loop
        converged = False
        history = []
        concentrations: Dict[str, float] = {}  # Initialize before loop
        last_common_fit: Optional[_CommonSlopeFit] = None
        # Diagnostics tracked across iterations for the post-loop quality_metrics
        # (also guards against an early break leaving them unbound).
        boltzmann_degenerate = True  # until a clean fit proves otherwise
        closure_degenerate = False
        ne_from_stark = False

        for _ in range(1, self.max_iterations + 1):
            step = self._run_python_iteration(
                elements,
                obs_by_element,
                ips,
                closure_mode,
                diags,
                closure_kwargs,
                T_K=T_K,
                n_e=n_e,
                T_corona=T_corona,
                concentrations=concentrations,
                last_common_fit=last_common_fit,
            )
            if step.should_break:
                break

            T_K = step.T_K
            n_e = step.n_e
            T_corona = step.T_corona
            concentrations = step.concentrations
            last_common_fit = step.last_common_fit
            boltzmann_degenerate = step.boltzmann_degenerate
            closure_degenerate = step.closure_degenerate
            ne_from_stark = step.ne_from_stark

            history.append((T_K, n_e))

            if step.converged:
                converged = True
                break
            converged = False

        quality_metrics = self._build_python_quality_metrics(
            observations,
            T_K,
            n_e,
            concentrations,
            last_common_fit,
            boltzmann_degenerate,
            closure_degenerate,
            ne_from_stark,
        )

        # Defensive: a degenerate Boltzmann slope (non-physical T) or a
        # collapsed composition must never report converged, even if an early
        # break left the loop's convergence flag set on a prior clean iteration.
        if closure_degenerate:
            self._warn_degenerate_composition(concentrations)
        # Final-state T window guard: even if the loop's per-iteration flags were
        # clean, a reported T outside the physical window is degenerate and must
        # never be reported converged (mirrors the closure-degeneracy guard).
        if not (T_PHYSICAL_MIN_K <= T_K <= T_PHYSICAL_MAX_K):
            boltzmann_degenerate = True
        if boltzmann_degenerate or closure_degenerate:
            converged = False

        if self.two_region and T_corona is None:
            # Empirical two-region DOF-reduction: the cooler outer/corona zone is
            # taken as ~0.8 of the core temperature. This 0.8 is a common
            # stabilization choice for two-region LTE fits, NOT a value with a
            # specific literature attribution.
            T_corona = 0.8 * T_K

        # When the Stark diagnostic drove n_e, the multi-line scatter is a real
        # measurement uncertainty — surface it (0.0 for a single line or the
        # pressure-balance fallback, whose uncertainty is unquantifiable).
        ne_uncertainty = float(quality_metrics.get("stark_ne_scatter_cm3", 0.0))

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
            electron_density_uncertainty_cm3=ne_uncertainty,
            boltzmann_covariance=None,
            overall_reliable=bool(quality_metrics.get("overall_reliable", False)),
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
            dict(obs_by_element), weight_cap=self.boltzmann_weight_cap
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
            # True until a clean fit proves otherwise (Python-path parity).
            boltzmann_degenerate=jnp.asarray(True),
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
        boltzmann_degenerate = bool(final_state.boltzmann_degenerate)
        conc_arr = np.asarray(final_state.concentrations)
        concentrations = {el: float(conc_arr[i]) for i, el in enumerate(elements_ord)}

        # Corona post-loop assembly (matches Python path)
        T_corona = 0.8 * T_K if self.two_region else None

        # Quality gates + metrics: PARITY with the Python path (audit 02-F8).
        # The lax loop already refuses to converge on a degenerate Boltzmann
        # fit (the in-body gate), but the composition-degeneracy (keystone
        # collapse) gate and the full quality_metrics key set were missing —
        # the lax path could report converged=True with
        # quality_metrics.get('boltzmann_r_squared') is None on a collapsed
        # composition. Compute both gates host-side after the loop and route
        # the metric assembly through the SAME builder as the Python path so
        # the key sets cannot drift.
        closure_degenerate = self._validate_composition_degeneracy(concentrations)
        if closure_degenerate:
            self._warn_degenerate_composition(concentrations)
        # Final-state T window guard (mirrors the Python path): a reported T
        # outside the physical window is degenerate and never converged.
        if not (T_PHYSICAL_MIN_K <= T_K <= T_PHYSICAL_MAX_K):
            boltzmann_degenerate = True
        if boltzmann_degenerate or closure_degenerate:
            converged_bool = False

        quality_metrics = self._assemble_quality_metrics(
            observations,
            T_K,
            n_e,
            concentrations,
            fit_r2=r_squared,
            boltzmann_degenerate=boltzmann_degenerate,
            closure_degenerate=closure_degenerate,
            # The lax body's n_e update is always the traced pressure-balance
            # kernel (a supplied stark_diagnostic forces the Python path).
            # The observable-gated self-absorption correction is a pure
            # observation transform applied in solve() BEFORE routing, so its
            # diagnostics are read from self._last_sa_result inside the
            # metric builder for both paths.
            ne_from_stark=False,
        )

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
            overall_reliable=bool(quality_metrics.get("overall_reliable", False)),
        )

    def _build_uncertainty_abundance_multipliers(
        self,
        elements: List[str],
        T_K: float,
        n_e: float,
        n_e_relative_uncertainty: float,
        partition_funcs: Dict[str, float],
        partition_funcs_II: Dict[str, float],
        effective_ips: Dict[str, float],
        T_corona: Optional[float],
    ):
        """Build abundance multipliers, optionally as UFloats (``solve_with_uncertainty`` helper).

        When ``n_e_relative_uncertainty > 0`` the neutral-plane multipliers
        ``(1 + n_II/n_I)`` are rebuilt as UFloats so the n_e variance propagates
        through closure into sigma_C. Numerics identical to the inline block.
        """
        abundance_multipliers = self._compute_abundance_multipliers(
            elements,
            T_K,
            n_e,
            partition_funcs,
            partition_funcs_II,
            effective_ips,
            T_corona=T_corona,
        )

        # When an n_e uncertainty is supplied, rebuild the neutral-plane
        # multipliers (1 + n_II/n_I) as UFloats so the n_e variance propagates
        # through closure into sigma_C. The Saha ratio scales as 1/n_e, so a
        # UFloat n_e carries its relative uncertainty into each multiplier.
        if n_e_relative_uncertainty and n_e_relative_uncertainty > 0.0:
            abundance_multipliers = self._compute_abundance_multipliers_uncertain(
                elements,
                T_K,
                n_e,
                n_e_relative_uncertainty,
                partition_funcs,
                partition_funcs_II,
                effective_ips,
                T_corona=T_corona,
            )
        return abundance_multipliers

    @staticmethod
    def _build_intercept_covariances(common_fit: _CommonSlopeFit):
        """Build per-element UFloat intercepts and covariances (``solve_with_uncertainty`` helper).

        Returns ``(intercepts_u, covariances, slope_err, slope_u)`` using the
        same common-slope model as :meth:`solve`. Numerics unchanged.
        """
        from uncertainties import ufloat

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
        return intercepts_u, covariances, slope_err, slope_u

    @staticmethod
    def _propagate_closure_uncertainty(
        closure_mode: str,
        intercepts_u: Dict[str, Any],
        partition_funcs: Dict[str, float],
        abundance_multipliers,
        closure_kwargs: Dict[str, Any],
    ):
        """Propagate intercept uncertainties through closure (``solve_with_uncertainty`` helper).

        Identical dispatch to the inline if/elif chain it replaces.
        """
        from cflibs.inversion.physics.uncertainty import (
            propagate_through_closure_oxide,
            propagate_through_closure_standard,
            propagate_through_closure_matrix,
        )

        if closure_mode == "matrix" and "matrix_element" in closure_kwargs:
            return propagate_through_closure_matrix(
                intercepts_u,
                partition_funcs,
                closure_kwargs["matrix_element"],
                closure_kwargs.get("matrix_fraction", 0.9),
                abundance_multipliers=abundance_multipliers,
            )
        elif closure_mode == "oxide":
            return propagate_through_closure_oxide(
                intercepts_u,
                partition_funcs,
                closure_kwargs.get("oxide_stoichiometry", {}),
                abundance_multipliers=abundance_multipliers,
            )
        elif closure_mode in {"ilr", "pwlr", "dirichlet_residual"}:
            return propagate_through_closure_standard(
                intercepts_u,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )
        else:
            return propagate_through_closure_standard(
                intercepts_u,
                partition_funcs,
                abundance_multipliers=abundance_multipliers,
            )

    @staticmethod
    def _select_boltzmann_covariance(
        covariances: Dict[str, np.ndarray],
        closure_mode: str,
        closure_kwargs: Dict[str, Any],
    ) -> Tuple[Optional[np.ndarray], Optional[str]]:
        """Pick the representative slope/intercept covariance (``solve_with_uncertainty`` helper).

        Returns ``(selected_covariance, covariance_element)``; numerics unchanged.
        """
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
        return selected_covariance, covariance_element

    def solve_with_uncertainty(
        self,
        observations: List[LineObservation],
        closure_mode: str = "standard",
        n_e_relative_uncertainty: float = 0.0,
        stark_diagnostics: Optional[Sequence["StarkDiagnosticLine"]] = None,
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
            n_e_relative_uncertainty (float): Optional fractional (1-sigma) uncertainty on the
                converged electron density. When > 0, the neutral-plane abundance multipliers
                ``(1 + n_II/n_I)`` are built as UFloats via ``saha_factor_with_uncertainty`` so
                the n_e variance propagates into the per-element concentration uncertainty.
                Defaults to 0.0 (n_e treated as exact, preserving prior behaviour).
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
        result = self.solve(
            observations, closure_mode, stark_diagnostics=stark_diagnostics, **closure_kwargs
        )

        # Import uncertainty utilities (will raise ImportError if not available).
        # The closure-propagation and ufloat helpers are imported inside the
        # private helpers below; importing this module here preserves the
        # documented early ImportError when the uncertainties stack is missing.
        from cflibs.inversion.physics.uncertainty import extract_values_and_uncertainties

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
            if ip is None:
                logger.warning(
                    "No ionization potential for %s I in the DB; using 15.0 eV fallback. "
                    "The complete ASD DB covers all I/II/III species, so this signals a "
                    "data gap, not a normal fallback.",
                    el,
                )
                ip = 15.0
            ips[el] = float(ip)

        effective_ips = self._compute_effective_ips(ips, n_e, T_K)

        # Apply Saha correction so intercepts match those from solve()
        # (common-slope path only; the SB-graph path applies the ion shift
        # internally and consumes the raw observations).
        corrected_obs_map = self._apply_saha_correction(obs_by_element, T_K, n_e, effective_ips)

        # Get partition functions at converged T
        partition_funcs, partition_funcs_II = self._evaluate_partition_functions(elements, T_K)

        abundance_multipliers = self._build_uncertainty_abundance_multipliers(
            elements,
            T_K,
            n_e,
            n_e_relative_uncertainty,
            partition_funcs,
            partition_funcs_II,
            effective_ips,
            result.temperature_corona_K,
        )

        if self.saha_boltzmann_graph:
            common_fit = self._fit_saha_boltzmann_graph(obs_by_element, T_K, n_e, effective_ips)
        else:
            common_fit = self._fit_common_boltzmann_plane(corrected_obs_map)
        if common_fit is None:
            return result

        # Propagate the same common-slope model used by solve()
        intercepts_u, covariances, slope_err, slope_u = self._build_intercept_covariances(
            common_fit
        )

        # Propagate through closure
        concentrations_u = self._propagate_closure_uncertainty(
            closure_mode,
            intercepts_u,
            partition_funcs,
            abundance_multipliers,
            closure_kwargs,
        )

        # Extract nominal values and uncertainties
        conc_nominal, conc_uncert = extract_values_and_uncertainties(concentrations_u)

        # Temperature uncertainty from pooled slope estimate
        from cflibs.inversion.physics.uncertainty import temperature_from_slope

        T_err = 0.0
        if slope_err > 0.0:
            T_K_u = temperature_from_slope(slope_u)
            T_err = float(T_K_u.std_dev) if np.isfinite(T_K_u.std_dev) else 0.0

        selected_covariance, covariance_element = self._select_boltzmann_covariance(
            covariances, closure_mode, closure_kwargs
        )

        quality_metrics = dict(result.quality_metrics)
        if covariance_element is not None:
            quality_metrics["boltzmann_covariance_element"] = covariance_element

        # M8 Lever 7 (opt-in): couple per-element CI width into the reliability
        # flag. A weak emitter with a huge relative CI is downgraded even if its
        # fit metrics look fine. Gated CFLIBS_RELIABILITY_FROM_UNCERTAINTY
        # (default OFF == bit-identical: empty labels, flag/overall_reliable
        # unchanged). Pure annotation -- never alters composition/T/n_e.
        final_conc = conc_nominal if conc_nominal else result.concentrations
        per_element_reliability: Dict[str, str] = {}
        if _reliability_from_uncertainty_enabled() and conc_uncert:
            from cflibs.inversion.physics.quality import (
                downgrade_quality_flag,
                per_element_reliability_from_uncertainty,
            )

            per_element_reliability = per_element_reliability_from_uncertainty(
                final_conc, conc_uncert
            )
            qf = quality_metrics.get("quality_flag")
            if isinstance(qf, str):
                new_qf = downgrade_quality_flag(qf, per_element_reliability)
                quality_metrics["quality_flag"] = new_qf
                mcw = bool(quality_metrics.get("lte_mcwhirter_satisfied", False))
                # Preserve the n_e-provenance gate from _assemble_quality_metrics:
                # a pressure-balance fallback n_e is never trustworthy (the
                # McWhirter check runs on that same fallback n_e).
                ne_ok = bool(quality_metrics.get("ne_from_stark", 0.0))
                quality_metrics["overall_reliable"] = bool(
                    ne_ok and mcw and new_qf in ("excellent", "good", "acceptable")
                )

        return CFLIBSResult(
            temperature_K=result.temperature_K,
            temperature_uncertainty_K=T_err,
            electron_density_cm3=result.electron_density_cm3,
            concentrations=final_conc,
            concentration_uncertainties=conc_uncert if conc_uncert else {},
            iterations=result.iterations,
            converged=result.converged,
            temperature_corona_K=result.temperature_corona_K,
            quality_metrics=quality_metrics,
            electron_density_uncertainty_cm3=0.0,  # Would need iterative uncertainty
            boltzmann_covariance=selected_covariance,
            overall_reliable=bool(quality_metrics.get("overall_reliable", False)),
            per_element_reliability=per_element_reliability,
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


def _collect_sb_graph_rows(
    obs_by_element: Dict[str, List[LineObservation]],
    ips: Dict[str, float],
    ln_S: float,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]], Dict[str, List[float]]]:
    """Collect shifted ``(x, y, w)`` rows per element (``_fit_saha_boltzmann_graph`` helper).

    WEIGHTING: the validated SB-graph (Aguilera & Aragon 2004; the probe in
    scripts/probe_saha_boltzmann_graph.py) uses an UNWEIGHTED global lstsq.
    That is essential to the method: with inverse-variance (≈ intensity)
    weighting the single brightest line of each element dominates its
    intercept dummy, lifting bright-resonance elements (Fe) by ~2 in
    ln-space (≈ 7x in C) and re-creating the over-attribution the global
    fit is meant to cure (Fe 16.7 -> 18.5 intercept; RMSE 8 -> 14 on real
    ChemCam BHVO-2). The pooled global geometry already conditions every
    intercept jointly via the shifted ion lines, so per-line weighting is
    neither needed nor helpful here. ``boltzmann_weight_cap`` is therefore
    not applied on this path. Lines are still inverse-variance *screened*
    for validity (finite, positive y-uncertainty -> usable) but contribute
    with unit weight.
    """
    per_el_x: Dict[str, List[float]] = defaultdict(list)
    per_el_y: Dict[str, List[float]] = defaultdict(list)
    per_el_w: Dict[str, List[float]] = defaultdict(list)
    for el, obs_list in obs_by_element.items():
        ip = ips.get(el, 15.0)
        for obs in obs_list:
            if obs.A_ki <= 0 or obs.g_k <= 0 or obs.intensity <= 0:
                continue
            y = obs.y_value
            z = obs.ionization_stage
            x = obs.E_k_ev + (ip * (z - 1) if z > 1 else 0.0)
            if z > 1:
                y = y - ln_S * (z - 1)
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            per_el_x[el].append(x)
            per_el_y[el].append(y)
            per_el_w[el].append(1.0)  # unweighted (validated SB-graph)
    return per_el_x, per_el_y, per_el_w


def _solve_sb_graph_lstsq(
    elements: List[str],
    el_index: Dict[str, int],
    per_el_x: Dict[str, List[float]],
    per_el_y: Dict[str, List[float]],
    per_el_w: Dict[str, List[float]],
) -> Optional[Tuple[float, float, float, Dict[str, float]]]:
    """Solve the pooled SB-graph WLS system (``_fit_saha_boltzmann_graph`` helper).

    Assembles the global design matrix ``A = [x | element-dummies]``, solves
    the weighted normal equations, and returns
    ``(slope, slope_variance, r_squared, intercepts)`` — or ``None`` when the
    global system is under-determined. Numerics identical to the inline block.
    """
    rows_x: List[float] = []
    rows_y: List[float] = []
    rows_w: List[float] = []
    rows_el: List[int] = []
    for el in elements:
        rows_x.extend(per_el_x[el])
        rows_y.extend(per_el_y[el])
        rows_w.extend(per_el_w[el])
        rows_el.extend([el_index[el]] * len(per_el_x[el]))

    n_rows = len(rows_x)
    E = len(elements)
    if n_rows < 3 or n_rows < (1 + E):
        # Under-determined global system; let the caller fall back.
        return None

    x_vec = np.asarray(rows_x, dtype=float)
    y_vec = np.asarray(rows_y, dtype=float)
    w_vec = np.asarray(rows_w, dtype=float)
    el_vec = np.asarray(rows_el, dtype=int)

    A = np.zeros((n_rows, 1 + E), dtype=float)
    A[:, 0] = x_vec
    A[np.arange(n_rows), 1 + el_vec] = 1.0

    # Weighted least squares via sqrt(w) row scaling -> ordinary lstsq.
    sw = np.sqrt(w_vec)
    Aw = A * sw[:, None]
    yw = y_vec * sw
    coef, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    slope = float(coef[0])

    # Global weighted R^2 of the pooled fit.
    y_pred = A @ coef
    resid = y_vec - y_pred
    ss_res = float(np.sum(w_vec * resid**2))
    y_wmean = float(np.average(y_vec, weights=w_vec))
    ss_tot = float(np.sum(w_vec * (y_vec - y_wmean) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0

    # Slope variance from the weighted normal-equation covariance.
    dof = max(n_rows - (1 + E), 1)
    sigma2 = ss_res / dof
    slope_variance = float("nan")
    try:
        cov = np.linalg.inv(Aw.T @ Aw) * sigma2
        slope_variance = float(cov[0, 0])
    except np.linalg.LinAlgError:
        slope_variance = float("nan")
    if not np.isfinite(slope_variance) or slope_variance <= 0.0:
        denom = float(np.sum(w_vec * (x_vec - np.average(x_vec, weights=w_vec)) ** 2))
        slope_variance = 1.0 / denom if denom > 0.0 else 1.0

    intercepts = {el: float(coef[1 + el_index[el]]) for el in elements}
    return slope, slope_variance, r_squared, intercepts


def _build_sb_graph_element_stats(
    elements: List[str],
    per_el_x: Dict[str, List[float]],
    per_el_y: Dict[str, List[float]],
    per_el_w: Dict[str, List[float]],
) -> Dict[str, "_CommonSlopeElementStats"]:
    """Build per-element shifted-coordinate stats (``_fit_saha_boltzmann_graph`` helper)."""
    element_stats: Dict[str, _CommonSlopeElementStats] = {}
    for el in elements:
        xs = np.asarray(per_el_x[el], dtype=float)
        ys = np.asarray(per_el_y[el], dtype=float)
        ws = np.asarray(per_el_w[el], dtype=float)
        element_stats[el] = _CommonSlopeElementStats(
            x_values=xs,
            y_values=ys,
            weights=ws,
            x_mean=float(np.average(xs, weights=ws)),
            y_mean=float(np.average(ys, weights=ws)),
        )
    return element_stats


def _cap_boltzmann_weights(weights: np.ndarray, cap: float) -> np.ndarray:
    """Clip a single element's inverse-variance weights to ``cap`` × their median.

    Bounds the per-element Boltzmann-weight dynamic range so the pooled
    common-slope intercept is no longer dominated by a single bright line (see
    ``IterativeCFLIBSSolver.boltzmann_weight_cap``). The median is taken over the
    finite, strictly-positive entries only; entries at or below the cap (the
    well-behaved regime) are left untouched, preserving inverse-variance
    weighting there. A non-positive ``cap`` disables the clip and returns the
    input unchanged.

    Parameters
    ----------
    weights:
        1-D array of weights for one element's lines (only the valid entries
        should be passed in; padded/zero entries do not affect the median but
        would skew it if included as positive values, so callers must mask first).
    cap:
        Multiplier ``K`` such that no weight exceeds ``K × median(valid weights)``.

    Returns
    -------
    np.ndarray
        A clipped copy of ``weights`` (or ``weights`` itself when disabled).
    """
    if cap <= 0.0:
        return weights
    finite_pos = weights[np.isfinite(weights) & (weights > 0.0)]
    if finite_pos.size == 0:
        return weights
    med = float(np.median(finite_pos))
    if not np.isfinite(med) or med <= 0.0:
        return weights
    return np.minimum(weights, cap * med)


def _fill_obs_row(
    i: int,
    el: str,
    obs_by_element: Dict[str, List[LineObservation]],
    x: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    stage: np.ndarray,
    mask: np.ndarray,
    weight_cap: float,
) -> None:
    """Fill row ``i`` of the padded arrays for element ``el`` (``_build_padded_arrays_from_obs`` helper).

    Mirrors the inner observation loop exactly: per-line x/y/weight/stage and a
    finite, positive-weight validity mask, then the optional per-element weight
    dynamic-range cap. Numerics unchanged.
    """
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
    # Bound the per-element weight dynamic range using ONLY this row's valid
    # weights for the median (padded zeros excluded by the mask), matching
    # the numpy host path in _fit_common_boltzmann_plane so the lax kernel
    # receives identically-capped weights.
    if weight_cap > 0.0:
        row_mask = mask[i]
        if row_mask.any():
            w[i, row_mask] = _cap_boltzmann_weights(w[i, row_mask], weight_cap)


def _build_padded_arrays_from_obs(
    obs_by_element: Dict[str, List[LineObservation]],
    weight_cap: float = 0.0,
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
        _fill_obs_row(i, el, obs_by_element, x, y, w, stage, mask, weight_cap)
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
        boltzmann_weight_cap: float = 5.0,
    ):
        super().__init__(
            atomic_db=atomic_db,
            max_iterations=max_iterations,
            t_tolerance_k=t_tolerance_k,
            ne_tolerance_frac=ne_tolerance_frac,
            pressure_pa=pressure_pa,
            apply_ipd=apply_ipd,
            boltzmann_weight_cap=boltzmann_weight_cap,
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
        stark_diagnostic: Optional["StarkDiagnosticLine"] = None,
        stark_diagnostics: Optional[Sequence["StarkDiagnosticLine"]] = None,
        **closure_kwargs,
    ) -> CFLIBSResult:
        """Deprecated thin shim for the JAX iterative path (T1-3).

        .. deprecated:: T1-3
            ``IterativeCFLIBSSolverJax`` is superseded by the
            ``CFLIBS_USE_LAX_WHILE_LOOP=1`` env flag on the parent
            :class:`IterativeCFLIBSSolver`, which selects the
            :func:`jax.lax.while_loop` path with the same numerics. This
            subclass now delegates to that path (or to the parent's Python
            loop). It is still wired into the benchmark harness and exported
            from ``cflibs.inversion``, so it remains a live (if deprecated)
            class rather than dead code; new callers should use
            :class:`IterativeCFLIBSSolver` directly with the env flag set.

        Falls back to the parent Python implementation when JAX is not
        available, preserving the prior behavior contract.
        """
        warnings.warn(
            "IterativeCFLIBSSolverJax is deprecated; use IterativeCFLIBSSolver "
            "with CFLIBS_USE_LAX_WHILE_LOOP=1 instead (T1-3, ADR-0001).",
            DeprecationWarning,
            stacklevel=2,
        )
        # The Stark-width n_e diagnostic is implemented only on the parent
        # Python path; force it when any diagnostic line is supplied.
        if stark_diagnostic is not None or stark_diagnostics:
            return super().solve(
                observations,
                closure_mode,
                stark_diagnostic=stark_diagnostic,
                stark_diagnostics=stark_diagnostics,
                **closure_kwargs,
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
