"""Full-spectrum CF-LIBS solver (memory-efficient + SVD-conditioned).

This module implements the literature-validated recipe that makes the
gradient-based full-spectrum fit *actually converge* on real, many-element,
high-pixel data (e.g. SuperCam, 7933 px x ~6000 lines) where the naive
implementation OOMs on reverse-mode AD and/or diverges to a degenerate
single-element minimum.

Two root causes, two fixes (both applied together):

(A) MEMORY — the unified per-line forward kernel
    (:func:`cflibs.radiation.kernels.forward_model`) materialises a dense
    ``(n_lines x n_wavelength)`` profile matrix.  Under reverse-mode AD this
    blows past device memory past ~3 elements.

    FIX — route the differentiable forward through the *chunked* kernel
    :func:`cflibs.radiation.kernels.forward_model_chunked`, which scans the
    per-line broadening over ``nstitch`` wavelength chunks under
    :func:`jax.checkpoint` (rematerialisation) and recombines via
    overlap-and-add.  Peak transient + backward-pass activation memory drop
    from ``O(n_lines * n_wl)`` to ``O(n_lines * chunk_width)``.  This is the
    same memory-efficient differentiable radiative-transfer idea as ExoJAX2
    (arXiv:2410.06900).

(B) CONDITIONING — fitting the raw ~thousands-of-pixels spectrum (intensities
    spanning ~10 orders of magnitude) with a per-pixel least-squares loss has
    a degenerate global minimum (a single bright element can soak the whole
    fit) and overflow-prone gradients.

    FIX — per Hebert et al. 2020 (arXiv:2008.04982, ChemCam LIBS Bayesian
    calibration): compress the spectrum to a small SVD/PCA basis (~15-30
    components capturing >99.99% variance) built from candidate forward-model
    spectra, and fit the *projection* of the (model - observed) residual in
    that low-dimensional, well-conditioned space.  The fit is run on
    area-normalised spectra so the absolute radiometric scale (arbitrary on
    real data) drops out.

The iterative CF-LIBS solver supplies the warm start (T, n_e, composition) and
informative priors; bounded/transformed (T, log n_e) and an entropy / Dirichlet
regulariser on the composition keep the optimiser inside the physical regime.

The public entry point is :func:`solve_full_spectrum`.  It returns a
:class:`FullSpectrumResult` carrying BOTH the warm-start and the converged-fit
diagnostics so callers can honestly report whether a *real* full-spectrum
optimum was reached (and whether it improved on the iterative warm start) vs a
fall-back to the warm start.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cflibs.atomic.masses import STANDARD_ATOMIC_MASSES
from cflibs.core.constants import EV_TO_K, K_TO_EV
from cflibs.core.jax_runtime import HAS_JAX
from cflibs.core.logging_config import get_logger
from cflibs.inversion.common.strict import (
    IllConditioned,
    NonPhysicalResult,
    OptimizerFailure,
    SolveDiagnostics,
    require_atomic_data,
    resolve_strict,
)
from cflibs.inversion.physics.self_absorption_observable import (
    ObservableSelfAbsorptionCorrector,
    ThickLineMask,
    build_observed_thick_line_mask,
    normalize_self_absorption_mode,
)

logger = get_logger("inversion.solve.full_spectrum")

# Forward-model intensity clip bounds (negatives -> 0, overflow -> _FORWARD_CLIP_HI).
# In strict mode a value pinned at _FORWARD_CLIP_HI is treated as a divergence
# (overflow masked by the clamp) rather than a legitimate spectrum.
_FORWARD_CLIP_LO = 0.0
_FORWARD_CLIP_HI = 1e12


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class FullSpectrumResult:
    """Outcome of a full-spectrum fit, with warm-start vs converged accounting.

    Attributes
    ----------
    temperature_K, electron_density_cm3 : float
        Adopted plasma parameters (converged fit when it improved, else the
        warm start).
    concentrations : dict[str, float]
        Adopted MASS fractions (sum to 1) keyed by element symbol.
    warm_start_temperature_K, warm_start_electron_density_cm3 : float
        Iterative warm-start plasma parameters.
    warm_start_concentrations : dict[str, float]
        Iterative warm-start MASS fractions.
    fit_concentrations : dict[str, float]
        The converged full-spectrum fit MASS fractions (regardless of whether
        adopted).
    fit_temperature_K, fit_electron_density_cm3 : float
        Converged full-spectrum fit plasma parameters.
    converged : bool
        ``True`` iff a *real* full-spectrum optimisation step ran and reduced
        the reduced (PC-space) objective relative to the warm start AND
        produced finite parameters — i.e. NOT a fall-back to the warm start.
    adopted_fit : bool
        ``True`` iff the converged fit was actually adopted as the result.
    initial_objective, final_objective : float
        SVD-space objective at the warm start and at the fit optimum.
    iterations : int
        Optimiser iterations actually run.
    gradient_norm : float
        Norm of the objective gradient at the fit optimum.
    diagnostics : dict
        Free-form extra diagnostics (n_lines, n_pc, nstitch, ...).
    """

    temperature_K: float
    electron_density_cm3: float
    concentrations: Dict[str, float]

    warm_start_temperature_K: float
    warm_start_electron_density_cm3: float
    warm_start_concentrations: Dict[str, float]

    fit_temperature_K: float
    fit_electron_density_cm3: float
    fit_concentrations: Dict[str, float]

    converged: bool
    adopted_fit: bool
    initial_objective: float
    final_objective: float
    iterations: int
    gradient_norm: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    # --- strict / no-fallback visibility fields (additive, default-safe) ---
    # ``failure_reason`` names the gate that would have triggered a silent
    # fallback (set in both modes for visibility; only acted on in strict mode).
    # ``fit_valid``/``fit_source`` make the provenance of ``fit_*`` explicit so
    # ``fit_concentrations == warm_start_concentrations`` is never ambiguous
    # ('fit genuinely landed at warm' vs 'fit never ran / was rejected').
    failure_reason: Optional[str] = None
    fit_valid: bool = True
    fit_source: str = "optimizer"  # 'optimizer' | 'optimizer_rejected' | 'warm_fallback'


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def _atomic_weight(element: str, *, strict: bool) -> float:
    """Atomic weight for ``element``.

    In strict mode a missing element is a data-completeness failure
    (``MissingAtomicData``) — the ``AW=50.0`` default is a silent substitution of
    a fabricated mass and is refused.  Non-strict mode keeps the production
    ``.get(el, 50.0)`` default.
    """
    if strict:
        require_atomic_data(
            "atomic_mass", STANDARD_ATOMIC_MASSES.get(element), element, strict=True
        )
        return float(STANDARD_ATOMIC_MASSES[element])
    return float(STANDARD_ATOMIC_MASSES.get(element, 50.0))


def _number_to_mass_fractions(
    number_fractions: Dict[str, float],
    *,
    strict: bool = False,
) -> Dict[str, float]:
    """Convert number/mole fractions to mass fractions (sum to 1).

    ``C_mass_i = C_i * AW_i / sum_j (C_j * AW_j)``.

    Strict mode refuses the fabricated ``AW=50.0`` default (``MissingAtomicData``)
    and raises ``NonPhysicalResult`` on a non-normalisable (weighted total <= 0)
    composition instead of returning an all-zero dict.  Non-strict mode is the
    byte-identical production path.
    """
    weights = {
        el: float(c) * _atomic_weight(el, strict=strict) for el, c in number_fractions.items()
    }
    total = sum(weights.values())
    if total <= 0:
        if strict:
            raise NonPhysicalResult(
                f"non-normalisable composition: weighted total {total:.3g} <= 0 "
                f"(number fractions {number_fractions!r})"
            )
        return {el: 0.0 for el in number_fractions}
    return {el: w / total for el, w in weights.items()}


def _mass_to_number_fractions(
    mass_fractions: Dict[str, float],
    elements: Sequence[str],
    *,
    strict: bool = False,
) -> np.ndarray:
    """Convert mass fractions to a number-fraction array over ``elements``.

    ``C_i = (m_i / AW_i) / sum_j (m_j / AW_j)``.  Missing elements seed a small
    floor so the softmax warm start is non-degenerate.

    Strict mode keeps genuinely-absent elements at zero (no ``1e-6`` phantom
    mass), refuses the fabricated ``AW=50.0`` default (``MissingAtomicData``),
    and raises ``NonPhysicalResult`` on a degenerate (total moles <= 0) warm
    composition instead of fabricating a uniform mix.  Non-strict mode is the
    byte-identical production path.
    """
    if strict:
        moles = np.array(
            [
                float(mass_fractions.get(el, 0.0)) / _atomic_weight(el, strict=True)
                for el in elements
            ],
            dtype=np.float64,
        )
        total = moles.sum()
        if total <= 0:
            raise NonPhysicalResult(
                "degenerate warm composition: total moles <= 0 " f"({dict(mass_fractions)!r})"
            )
        return moles / total
    moles = np.array(
        [
            max(float(mass_fractions.get(el, 0.0)), 1e-6)
            / float(STANDARD_ATOMIC_MASSES.get(el, 50.0))
            for el in elements
        ],
        dtype=np.float64,
    )
    total = moles.sum()
    if total <= 0:
        return np.full(len(elements), 1.0 / max(len(elements), 1))
    return moles / total


# ---------------------------------------------------------------------------
# SVD basis (Hebert et al. 2020 dimensionality reduction)
# ---------------------------------------------------------------------------


def _normalise_spectrum(
    spec: np.ndarray, *, strict: bool = False, what: str = "spectrum"
) -> np.ndarray:
    """Area-normalise a spectrum (drop arbitrary radiometric scale).

    Strict mode raises ``NonPhysicalResult`` on a non-normalisable (all-zero or
    fully-cancelling, ``area <= 0``) spectrum — a dead forward model / empty
    observed spectrum — instead of silently returning the un-normalised zeros
    that would corrupt the SVD basis and projection.  Non-strict mode is the
    byte-identical production path.
    """
    spec = np.asarray(spec, dtype=np.float64)
    area = float(np.sum(np.abs(spec)))
    if area <= 0:
        if strict:
            raise NonPhysicalResult(
                f"non-normalisable {what}: area {area:.3g} <= 0 (dead forward / empty spectrum)"
            )
        return spec
    return spec / area


def build_svd_basis(
    library: np.ndarray,
    observed: np.ndarray,
    *,
    n_components: int = 20,
    variance_target: float = 0.99995,
    strict: bool = False,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Build an SVD/PCA basis from a small library of candidate model spectra.

    The library is the set of forward-model spectra produced by a coarse
    parameter sweep around the warm start (plus the observed spectrum itself),
    each AREA-NORMALISED.  We mean-centre, SVD, and keep the smallest number of
    right-singular vectors reaching ``variance_target`` (capped at
    ``n_components``).  Per Hebert et al. 2020 q=15 typically clears 99.995%.

    Returns
    -------
    basis : (k, n_wl) ndarray
        Orthonormal projection rows (right singular vectors).
    mean : (n_wl,) ndarray
        Column mean subtracted before projection.
    k : int
        Number of components retained.
    """
    lib = np.asarray(library, dtype=np.float64)
    if lib.ndim != 2:
        raise ValueError(f"library must be 2D (n_samples, n_wl); got {lib.shape}")
    obs_n = _normalise_spectrum(observed, strict=strict, what="observed spectrum")
    rows = [_normalise_spectrum(row, strict=strict, what="library row") for row in lib]
    rows.append(obs_n)
    X = np.vstack(rows)
    mean = X.mean(axis=0)
    Xc = X - mean
    # full_matrices=False → Vt is (min(n,p), p); rows are PC directions.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2
    total_var = float(np.sum(var))
    if strict and (total_var < 1e-300 or not np.isfinite(total_var)):
        # The seed library spans no informative direction (rows identical /
        # all-zero forward): the conditioned fit would run in a junk subspace.
        raise IllConditioned(
            f"degenerate SVD library: total singular-value variance {total_var:.3g} "
            f"< floor — warm-start sweep produced indistinguishable spectra"
        )
    cum = np.cumsum(var) / max(total_var, 1e-300)
    k_var = int(np.searchsorted(cum, variance_target) + 1)
    if strict and k_var < 1:
        raise IllConditioned(
            "degenerate SVD library: variance target reached in 0 components "
            "(library does not span an informative direction)"
        )
    k = max(1, min(n_components, k_var, Vt.shape[0]))
    return Vt[:k].copy(), mean, k


# ---------------------------------------------------------------------------
# Differentiable forward (chunked → memory-efficient AD)
# ---------------------------------------------------------------------------


class _ChunkedForward:
    """Memory-efficient differentiable forward built on a BayesianForwardModel.

    Wraps :class:`cflibs.inversion.solve.bayesian.forward.BayesianForwardModel`
    but replaces its dense ``forward_model`` call with
    :func:`cflibs.radiation.kernels.forward_model_chunked` so reverse-mode AD
    activation memory is ``O(n_lines * chunk_width)`` rather than
    ``O(n_lines * n_wl)``.

    The chunk plan is built once at construction (static across the fit).
    """

    def __init__(
        self,
        db_path: str,
        elements: Sequence[str],
        wavelength_grid: np.ndarray,
        *,
        resolving_power: Optional[float] = None,
        instrument_fwhm_nm: Optional[float] = None,
        nstitch: Optional[int] = None,
        overlap_sigma_nm: float = 2.0,
        strict: bool = False,
    ) -> None:
        if not HAS_JAX:
            raise ImportError("JAX required for the full-spectrum solver")
        import jax.numpy as jnp  # noqa: PLC0415

        from cflibs.inversion.solve.bayesian.forward import (  # noqa: PLC0415
            BayesianForwardModel,
        )
        from cflibs.radiation.host import build_chunk_plan  # noqa: PLC0415
        from cflibs.radiation.profiles import BroadeningMode  # noqa: PLC0415

        self.elements = list(elements)
        self._jnp = jnp
        self._BroadeningMode = BroadeningMode
        self._strict = bool(strict)

        wl = np.asarray(wavelength_grid, dtype=np.float64)
        self.fm = BayesianForwardModel(
            db_path,
            self.elements,
            (float(wl.min()), float(wl.max())),
            wavelength_grid=wl,
            resolving_power=resolving_power,
            instrument_fwhm_nm=instrument_fwhm_nm,
        )
        self.snapshot = self.fm.snapshot
        self.instrument = self.fm.instrument
        self.wavelength = self.fm.wavelength
        self.n_lines = int(np.asarray(self.snapshot.line_wavelengths_nm).shape[0])
        self.n_wl = int(wl.shape[0])

        # Auto chunk count: keep the per-chunk profile matrix near ~32 MB
        # (chunk_width * n_lines * 8 bytes). Always at least 4 chunks so AD
        # memory is bounded even for modest grids.
        if nstitch is None:
            target_cells = 4_000_000  # ~32 MB fp64 per chunk matrix
            chunk_width = max(target_cells // max(self.n_lines, 1), 64)
            nstitch = max(4, int(np.ceil(self.n_wl / chunk_width)))
        self.nstitch = int(nstitch)
        self.plan = build_chunk_plan(
            wl,
            np.asarray(self.snapshot.line_wavelengths_nm),
            nstitch=self.nstitch,
            max_sigma_nm=overlap_sigma_nm,
        )
        logger.info(
            "ChunkedForward: %d lines x %d wl, nstitch=%d overlap=%d "
            "(dense matrix would be %.0f MB; chunked ~%.0f MB)",
            self.n_lines,
            self.n_wl,
            self.plan.nstitch,
            self.plan.overlap,
            self.n_lines * self.n_wl * 8 / 1e6,
            self.n_lines * np.asarray(self.plan.chunk_wavelength_grids).shape[1] * 8 / 1e6,
        )

    def spectrum(self, T_eV, log_ne, number_fractions):
        """Differentiable forward: (T_eV, log10 n_e, number-frac array) -> spectrum."""
        jnp = self._jnp
        from cflibs.core.constants import EV_TO_K as _EV_TO_K  # noqa: PLC0415
        from cflibs.inversion.solve.bayesian.atomic import (  # noqa: PLC0415
            _resolve_total_species_density_cm3,
        )
        from cflibs.plasma.state import SingleZoneLTEPlasma  # noqa: PLC0415
        from cflibs.radiation.kernels import forward_model_chunked  # noqa: PLC0415

        n_e = jnp.power(jnp.asarray(10.0, dtype=jnp.asarray(log_ne).dtype), log_ne)
        total_density = _resolve_total_species_density_cm3(n_e, None)

        ps = object.__new__(SingleZoneLTEPlasma)
        ps.T_e = T_eV * _EV_TO_K
        ps.n_e = n_e
        dens = number_fractions * total_density
        ps.species = {el: dens[i] for i, el in enumerate(self.elements)}
        ps.T_g = None
        ps.pressure = None

        intensity = forward_model_chunked(
            ps,
            self.snapshot,
            self.instrument,
            self.wavelength,
            plan=self.plan,
            broadening_mode=self._BroadeningMode.PHYSICAL_DOPPLER,
            path_length_m=0.0,
            apply_self_absorption=False,
            fold_instrument_sigma=True,
            apply_stark=True,
            total_species_density_cm3=total_density,
        )
        return jnp.clip(intensity, _FORWARD_CLIP_LO, _FORWARD_CLIP_HI)

    def spectrum_jit(self):
        """Return a cached ``jax.jit`` of :meth:`spectrum` (compile once, reuse).

        The warm-start library sweep evaluates the forward ~15 times; without a
        shared compiled graph each call re-traces. Caching the jit collapses
        that to a single XLA compile.
        """
        cached = getattr(self, "_spectrum_jit_fn", None)
        if cached is None:
            import jax  # noqa: PLC0415

            cached = jax.jit(self.spectrum)
            self._spectrum_jit_fn = cached
        return cached

    def spectrum_numpy(
        self, T_eV: float, log_ne: float, number_fractions: np.ndarray
    ) -> np.ndarray:
        jnp = self._jnp
        out = self.spectrum_jit()(
            jnp.asarray(float(T_eV)),
            jnp.asarray(float(log_ne)),
            jnp.asarray(np.asarray(number_fractions, dtype=np.float64)),
        )
        arr = np.asarray(out, dtype=np.float64)
        if self._strict:
            # The forward clip ([_FORWARD_CLIP_LO, _FORWARD_CLIP_HI]) turns an
            # overflow into a finite plateau that the convergence checks accept.
            # In strict mode treat saturation / non-finite as a forward
            # divergence at the offending (T, n_e) rather than a valid spectrum.
            n_saturated = int(np.count_nonzero(arr >= _FORWARD_CLIP_HI))
            if n_saturated > 0 or not np.all(np.isfinite(arr)):
                raise NonPhysicalResult(
                    f"forward-model divergence: {n_saturated} pixels pinned at clip "
                    f"bound {_FORWARD_CLIP_HI:.3g} (or non-finite) at "
                    f"T_eV={float(T_eV):.4g}, log_ne={float(log_ne):.4g}"
                )
        return arr


# ---------------------------------------------------------------------------
# Strict / no-fallback decision helpers (pure — no JAX / DB; unit-testable)
# ---------------------------------------------------------------------------


@dataclass
class _ConvergenceDecision:
    """Whether the optimizer endpoint is a *real* converged fit, and why not."""

    real_fit: bool
    reason: Optional[str]  # None | 'nonfinite' | 'zero_iters' | 'no_move' | 'no_improvement'
    fit_source: str  # 'optimizer' | 'optimizer_rejected'


def _decide_convergence(
    *, finite: bool, moved: bool, improved_obj: bool, iterations: int
) -> _ConvergenceDecision:
    """Classify the BFGS endpoint (site 619-636).

    A *real fit* requires the optimizer to be finite, to have moved off the warm
    start, to have improved the conditioned objective, and to have run >=1
    iteration.  When it is not a real fit, the failed predicate is named so the
    indistinguishable 'kept warm start' outcome becomes a typed reason — instead
    of the coarse logged booleans the production path discards.
    """
    real_fit = bool(finite and moved and improved_obj and iterations > 0)
    if real_fit:
        return _ConvergenceDecision(True, None, "optimizer")
    if not finite:
        reason = "nonfinite"
    elif iterations <= 0:
        reason = "zero_iters"
    elif not moved:
        reason = "no_move"
    else:
        reason = "no_improvement"
    # The optimizer ran (this branch is only reached when no exception was
    # raised), so the real endpoint exists and should be surfaced, not warm.
    return _ConvergenceDecision(False, reason, "optimizer_rejected")


@dataclass
class _AdoptionDecision:
    """Adopted (T, n_e, composition) plus whether the fit was trusted."""

    adopted: bool
    temperature_K: float
    electron_density_cm3: float
    concentrations: Dict[str, float]
    failure_reason: Optional[str]


def _resolve_adoption(
    *,
    real_fit: bool,
    physically_near: bool,
    strict: bool,
    fit_T_K: float,
    fit_ne: float,
    fit_mass: Dict[str, float],
    warm_T_K: float,
    warm_ne: float,
    warm_mass: Dict[str, float],
    T_ratio: float,
    ne_ratio: float,
    t_threshold: float,
    ne_threshold: float,
) -> _AdoptionDecision:
    """Physical-plausibility adoption gate (site 653-684).

    * A converged, physically-near fit is adopted (both modes).
    * A converged fit that rode (T, n_e) to a box edge is the single most
      diagnostic event for this solver (LIBS T<->composition degeneracy). The
      production path silently reverts to the warm-start composition under a
      ``converged=True`` banner; **strict mode surfaces the (untrusted) FIT
      composition** with a ``failure_reason`` so the degenerate optimum is
      visible — it never launders warm into the adopted slot.
    * Otherwise (not a real fit, or non-strict implausible) keep the warm start
      exactly as production does.
    """
    if real_fit and physically_near:
        return _AdoptionDecision(True, fit_T_K, fit_ne, dict(fit_mass), None)
    if real_fit and not physically_near and strict:
        reason = (
            f"adoption_degenerate: fit rode T/n_e to box edge "
            f"(T_ratio={T_ratio:.3g} vs<{t_threshold:.3g}, "
            f"ne_ratio={ne_ratio:.3g} vs<{ne_threshold:.3g}); "
            f"surfacing untrusted fit composition instead of reverting to warm"
        )
        return _AdoptionDecision(False, fit_T_K, fit_ne, dict(fit_mass), reason)
    return _AdoptionDecision(False, warm_T_K, warm_ne, dict(warm_mass), None)


def _handle_optimizer_exception(
    exc: BaseException, *, strict: bool, diagnostics: Optional[SolveDiagnostics] = None
) -> None:
    """Bare-except policy (site 637-638).

    Production collapses every hard solver failure (crash, non-finite gradient,
    device OOM, XLA compile error) into one indistinguishable 'kept warm start'
    with the type/traceback only in a log line.  This records the failure on the
    diagnostics in both modes and, in strict mode, re-raises a typed
    :class:`OptimizerFailure` chaining the original instead of degrading.
    """
    reason = f"optimizer_exception: {type(exc).__name__}: {exc}"
    if diagnostics is not None:
        diagnostics.failure_reason = diagnostics.failure_reason or reason
        diagnostics.extra["optimizer_exception_type"] = type(exc).__name__
    if strict:
        raise OptimizerFailure(reason, diagnostics) from exc


# ---------------------------------------------------------------------------
# Observable self-absorption mask (Issue 3: thin model vs thick data)
# ---------------------------------------------------------------------------


def _self_absorption_line_mask(
    fwd: "_ChunkedForward",
    fit_wl: np.ndarray,
    measure_wl: np.ndarray,
    measure_obs: np.ndarray,
    *,
    temperature_K: float,
    mask_tau_min: float,
    corrector: Optional[ObservableSelfAbsorptionCorrector] = None,
) -> ThickLineMask:
    """Observable-anchored optically-thick line mask for the raw-spectrum fit.

    Unpacks the forward's atomic snapshot into per-line arrays and delegates to
    :func:`cflibs.inversion.physics.self_absorption_observable.build_observed_thick_line_mask`,
    which measures each in-band line from the OBSERVED spectrum and runs the
    same observable corrector wired into the iterative path (doublet intensity
    ratios; no composition-derived optical depth, no fitted tau DOF).

    Line intensities are measured on the NATIVE (high-resolution) spectrum
    ``measure_wl``/``measure_obs`` — a coarsely-resampled fit grid under-samples
    narrow lines and corrupts the integrated-intensity ratio — while the
    returned ``keep`` mask is built on the fit grid ``fit_wl``.

    Best-effort: any failure (a stub forward without a snapshot, a degenerate
    catalog, etc.) yields an empty mask so the fit proceeds exactly as the
    optically-thin path would — the correction never crashes the solver.
    """
    snap = getattr(fwd, "snapshot", None)
    if snap is None:
        return ThickLineMask(None, [], 0.0, 0, 0, {}, ["no atomic snapshot; mask skipped"])
    try:
        species = list(snap.species)
        line_sp_idx = np.asarray(snap.line_species_index)
        line_elements = [species[j][0] for j in line_sp_idx]
        line_ion_stages = [int(species[j][1]) for j in line_sp_idx]
        line_wl = np.asarray(snap.line_wavelengths_nm, dtype=np.float64)

        # Per-line instrument sigma: gives each line a wavelength-scaled
        # measurement window that captures its full profile at both the narrow
        # (blue) and broad (red) ends of a wide spectrum — a single fixed window
        # mis-measures one end and fakes doublet-ratio deviations on thin data.
        inst = getattr(fwd, "instrument", None)
        line_sigma_nm: Optional[np.ndarray] = None
        if inst is not None:
            try:
                if getattr(inst, "is_resolving_power_mode", False):
                    line_sigma_nm = np.array(
                        [float(inst.sigma_at_wavelength(float(w))) for w in line_wl],
                        dtype=np.float64,
                    )
                else:
                    s = float(getattr(inst, "resolution_sigma_nm", 0.0) or 0.0)
                    if s > 0:
                        line_sigma_nm = np.full(line_wl.shape, s, dtype=np.float64)
            except Exception:  # noqa: BLE001 — instrument model is advisory here
                line_sigma_nm = None
        fit_pixel_nm = (
            float(np.median(np.abs(np.diff(np.sort(fit_wl))))) if fit_wl.size > 1 else 0.0
        )
        # Mask windows on the fit grid: a few fit pixels, floored to cover a line.
        mask_hw = max(5.0 * fit_pixel_nm, 0.15)

        return build_observed_thick_line_mask(
            measure_wl,
            measure_obs,
            line_wavelengths_nm=line_wl,
            line_elements=line_elements,
            line_ion_stages=line_ion_stages,
            line_E_k_ev=np.asarray(snap.line_E_k_ev),
            line_g_k=np.asarray(snap.line_g_k),
            line_A_ki=np.asarray(snap.line_A_ki),
            mask_wavelength_grid=fit_wl,
            corrector=corrector,
            temperature_K=temperature_K,
            line_sigma_nm=line_sigma_nm,
            measure_half_width_nm=0.2,
            mask_half_width_nm=mask_hw,
            mask_tau_min=mask_tau_min,
        )
    except Exception as exc:  # noqa: BLE001 — mask is a best-effort preprocessing step
        logger.warning("Observable self-absorption mask failed (%r); fitting thin.", exc)
        return ThickLineMask(None, [], 0.0, 0, 0, {}, [f"mask error: {exc!r}"])


# ---------------------------------------------------------------------------
# Public solver
# ---------------------------------------------------------------------------


def solve_full_spectrum(
    wavelength: np.ndarray,
    intensity: np.ndarray,
    elements: Sequence[str],
    db_path: str,
    *,
    warm_start_T_K: float,
    warm_start_ne_cm3: float,
    warm_start_concentrations: Dict[str, float],
    resolving_power: Optional[float] = None,
    instrument_fwhm_nm: Optional[float] = None,
    n_components: int = 20,
    max_iterations: int = 40,
    sweep_points: int = 9,
    fit_pixels: Optional[int] = 1500,
    method: str = "bayesian",
    apply_self_absorption: "bool | str" = "observable",
    sa_mask_tau_min: float = 0.5,
    strict: Optional[bool] = None,
) -> FullSpectrumResult:
    """Run the memory-efficient, SVD-conditioned full-spectrum fit.

    Parameters
    ----------
    wavelength, intensity : ndarray
        Observed spectrum (nm, arbitrary radiometric units).
    elements : sequence of str
        Elements to fit (the warm-start element set).
    db_path : str
        Atomic database path.
    warm_start_* :
        Iterative CF-LIBS warm start.  ``warm_start_concentrations`` are MASS
        fractions (the iterative-solver / scoreboard convention).
    resolving_power, instrument_fwhm_nm :
        Instrument model (mutually exclusive).
    n_components : int
        Max SVD components for the conditioned loss (Hebert q~15-30).
    max_iterations : int
        Optimiser iteration cap.
    sweep_points : int
        Total target for the warm-start-centred SVD seed library. The base
        library is always a 3x3 T/ne grid plus one composition-boost row per
        element (so composition directions are always spanned); values above
        ``9`` add that many extra warm-centred T-refinement rows.
    fit_pixels : int or None
        Resample the spectrum onto a uniform ``fit_pixels``-point grid for the
        differentiable fit.  The chunked-forward XLA graph (a ``lax.scan`` of
        per-line Voigt over the wavelength axis) compiles and evaluates with
        cost ~``O(n_wl)``, so a few-thousand-point fit grid is dramatically
        cheaper to optimise on CPU than the native ~8000-px SuperCam axis while
        preserving the SVD-compressed composition signal (the basis is
        peak-dominated, not per-pixel).  ``None`` keeps the native grid.
    method : {'bayesian', 'joint'}
        ``'bayesian'`` adds informative Gaussian priors on (T, log n_e) from the
        warm start and an entropy (Dirichlet-like) regulariser on composition;
        ``'joint'`` runs the same conditioned data term without the priors
        (pure MAP-of-likelihood).  Both use the identical chunked forward and
        SVD loss — the difference is purely the regularisation, mirroring the
        joint-optimizer (no prior) vs Bayesian-MAP (informative prior) split.
    apply_self_absorption : bool or str, optional
        Observable-anchored self-absorption handling for the thin forward vs
        thick data mismatch (physics-first-principles audit, Issue 3). The
        optically-thin forward here has no saturation term, so a prior-free fit
        fakes the missing saturation by riding ``T`` to a box edge. When
        ``'observable'``/``True`` (default) the strongest OBSERVED lines that
        the observable corrector (doublet intensity ratios — the same corrector
        wired into the iterative path) measures as optically thick are EXCLUDED
        from the SVD residual on both the observed and model side, so the fit is
        never asked to reproduce a saturated line with a thin model. ``'off'``/
        ``False`` disables the mask (byte-identical optically-thin fit) for A/B.
        The flag mirrors the iterative solver's ``apply_self_absorption`` knob;
        it is data-side only — no composition-derived optical depth and no
        fitted optical-depth degree of freedom are introduced (the audited-
        harmful F4 loop is explicitly avoided).
    sa_mask_tau_min : float, optional
        Minimum observable line-center optical depth for a line to be masked
        (default 0.5). A tiny measured tau is not worth excluding a window for.
    strict : bool, optional
        Strict / no-fallback mode (resolved via
        :func:`cflibs.inversion.common.strict.resolve_strict` — defaults to the
        ``CFLIBS_NO_FALLBACK`` env var, else ``False``).  When ``False`` (the
        default) every path is byte-identical to production: silent warm-start
        substitution, clamps, and the swallowed optimizer exception are
        preserved (the new visibility fields are merely populated).  When
        ``True`` the silent fallbacks become honest failures — a crashed
        optimizer raises :class:`OptimizerFailure`, missing atomic masses raise
        :class:`MissingAtomicData`, a dead forward / degenerate seed library
        raises :class:`NonPhysicalResult` / :class:`IllConditioned`, a
        non-converged endpoint is surfaced (not fabricated to equal warm) with
        ``fit_valid=False``, and a box-edge degenerate optimum surfaces the
        untrusted fit composition rather than reverting to warm under a
        ``converged=True`` banner.

    Returns
    -------
    FullSpectrumResult
    """
    if not HAS_JAX:
        raise ImportError("JAX required for the full-spectrum solver")
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from jax.scipy.optimize import minimize as jax_minimize  # noqa: PLC0415

    strict = resolve_strict(strict)
    diag = SolveDiagnostics(solver="full_spectrum", strict=strict)

    elements = list(elements)
    n_el = len(elements)
    wl = np.asarray(wavelength, dtype=np.float64)
    obs = np.asarray(intensity, dtype=np.float64)
    # Native (pre-resample) spectrum for observable line MEASUREMENT — the SA
    # mask must measure narrow lines at full resolution even though the fit runs
    # on the coarser resampled grid.
    native_wl = wl.copy()
    native_obs = obs.copy()

    # Resample onto a coarser uniform fit grid (CPU-tractable XLA compile/eval).
    # Real SuperCam axes have inter-spectrometer gaps; a uniform grid that spans
    # the same range with linear interpolation of the observed spectrum keeps
    # the chunked forward's wavelength axis contiguous (the kernel assumes a
    # uniform grid for overlap-and-add).
    if fit_pixels is not None and fit_pixels < wl.shape[0]:
        order = np.argsort(wl)
        wl_sorted = wl[order]
        obs_sorted = obs[order]
        fit_wl = np.linspace(float(wl_sorted[0]), float(wl_sorted[-1]), int(fit_pixels))
        obs = np.interp(fit_wl, wl_sorted, obs_sorted)
        wl = fit_wl

    warm_mass = dict(warm_start_concentrations)
    warm_T_eV = float(warm_start_T_K) * K_TO_EV
    # Warm-start input floors (sites 464 / 531-533): a non-physical iterative
    # warm start (n_e<=1e10, T<=0.05 eV) is silently floored into a barely-
    # physical seed in production. In strict mode validate it up front instead.
    if strict and (not np.isfinite(warm_start_ne_cm3) or warm_start_ne_cm3 <= 1e10):
        raise NonPhysicalResult(
            f"non-physical warm-start n_e={warm_start_ne_cm3!r} (<=1e10); "
            "refusing to floor a failed iterative warm start",
            diag,
        )
    if strict and (not np.isfinite(warm_T_eV) or warm_T_eV <= 0.05):
        raise NonPhysicalResult(
            f"non-physical warm-start T={warm_start_T_K!r} K ({warm_T_eV:.3g} eV, <=0.05); "
            "refusing to floor a failed iterative warm start",
            diag,
        )
    warm_log_ne = float(np.log10(max(warm_start_ne_cm3, 1e10)))
    warm_numfrac = _mass_to_number_fractions(warm_mass, elements, strict=strict)

    fwd = _ChunkedForward(
        db_path,
        elements,
        wl,
        resolving_power=resolving_power,
        instrument_fwhm_nm=instrument_fwhm_nm,
        strict=strict,
    )

    # ---- Observable self-absorption exclusion mask (audit Issue 3) ----------
    # The forward here is optically THIN (apply_self_absorption=False); real
    # data is optically thick in the strong resonance lines. Without a
    # saturation term a prior-free fit fakes the missing saturation by riding T
    # to a box edge. We exclude the observed-and-modelled windows of lines the
    # OBSERVABLE corrector (doublet intensity ratios) measures as thick, so the
    # fit is never asked to reproduce a saturated line with a thin model. This
    # is data-side only: no composition-derived tau, no fitted optical-depth DOF
    # (the audited-harmful F4 loop). ``keep_np is None`` (no line flagged, or SA
    # off) means an EXACT no-op — the masking multiply is skipped entirely so a
    # thin spectrum reproduces the un-masked fit bit-for-bit.
    sa_mode = normalize_self_absorption_mode(apply_self_absorption)
    keep_np: Optional[np.ndarray] = None
    keep_jnp = None
    sa_mask_diag: Dict[str, Any] = {"mode": sa_mode, "n_flagged": 0, "max_tau": 0.0}
    if sa_mode != "off":
        sa_mask = _self_absorption_line_mask(
            fwd,
            wl,
            native_wl,
            native_obs,
            temperature_K=float(warm_start_T_K),
            mask_tau_min=sa_mask_tau_min,
        )
        sa_mask_diag.update(
            {
                "n_lines_measured": sa_mask.n_lines_measured,
                "n_flagged": sa_mask.n_flagged,
                "max_tau": sa_mask.max_tau,
                "flagged_wavelengths_nm": [round(w, 3) for w in sa_mask.flagged_wavelengths],
            }
        )
        if sa_mask.warnings:
            sa_mask_diag["warnings"] = list(sa_mask.warnings)
        if sa_mask.keep is not None:
            keep_np = sa_mask.keep.astype(np.float64)
            keep_jnp = jnp.asarray(keep_np)
            obs = obs * keep_np  # masked observed drives the basis + projection
            logger.info(
                "Full-spectrum SA mask active: %d/%d thick lines excluded (max_tau=%.2f)",
                sa_mask.n_flagged,
                sa_mask.n_lines_measured,
                sa_mask.max_tau,
            )
    diag.extra["self_absorption_mask"] = sa_mask_diag

    def _mask_np(spec: np.ndarray) -> np.ndarray:
        """Apply the SA keep-mask to a pixel-space spectrum (no-op when None)."""
        return spec if keep_np is None else spec * keep_np

    # ---- Build the SVD library from a coarse warm-start-centred sweep ----
    # Vary T (+/-30%) and log n_e (+/-0.5 dex) and a few single-element-boosted
    # compositions so the basis spans the directions the fit will move along.
    lib_rows: List[np.ndarray] = []
    T_factors = np.linspace(0.7, 1.3, 3)
    ne_offsets = np.linspace(-0.5, 0.5, 3)
    for tf in T_factors:
        for dne in ne_offsets:
            lib_rows.append(
                _mask_np(fwd.spectrum_numpy(warm_T_eV * tf, warm_log_ne + dne, warm_numfrac))
            )
    # Composition perturbations: boost each element in turn. ALWAYS generated
    # (one per element) so the SVD basis spans composition directions. The 3x3
    # T/ne grid alone does not — without these rows the optimizer moves in
    # composition directions orthogonal to the basis (treated as zero residual).
    # Audit C3: previously gated behind `extra = sweep_points - len(lib_rows)`,
    # which is 0 at the default sweep_points=9 (the T/ne grid already fills it),
    # so the basis never saw composition and the fit ran in a T/ne-only subspace.
    # Hebert et al. 2020 §3 uses single-element spectra precisely for this.
    for i in range(n_el):
        boosted = warm_numfrac.copy()
        boosted[i] = boosted[i] * 3.0 + 0.05
        boosted = boosted / boosted.sum()
        lib_rows.append(_mask_np(fwd.spectrum_numpy(warm_T_eV, warm_log_ne, boosted)))
    # `sweep_points` now requests OPTIONAL extra warm-centred T-refinement rows
    # beyond the base 3x3 T/ne grid + per-element composition rows.
    extra_tne = max(0, int(sweep_points) - 9)
    for tf in np.linspace(0.8, 1.2, extra_tne) if extra_tne else ():
        lib_rows.append(
            _mask_np(fwd.spectrum_numpy(warm_T_eV * float(tf), warm_log_ne, warm_numfrac))
        )
    library = np.vstack(lib_rows)

    basis_np, mean_np, k = build_svd_basis(library, obs, n_components=n_components, strict=strict)
    obs_norm = _normalise_spectrum(obs, strict=strict, what="observed spectrum")
    obs_proj_np = (obs_norm - mean_np) @ basis_np.T  # (k,)

    basis = jnp.asarray(basis_np)
    mean = jnp.asarray(mean_np)
    obs_proj = jnp.asarray(obs_proj_np)

    # PC-residual noise scale. The data term is a proper Gaussian chi-square
    # ``0.5 * sum((resid/sigma_d)^2)`` that must be COMMENSURATE with the
    # ``0.5*(Delta/sigma)^2`` priors below, otherwise the data term dwarfs the
    # prior and the fit rides T to the box edge (observed). We anchor sigma_d to
    # the *warm-start* PC residual: the model at the warm parameters already
    # matches the data to within this residual, so chi-square ~ k (the PC count)
    # at a good fit and a prior with sigma_logT~0.15 genuinely competes with a
    # T excursion. A floor keeps it from collapsing to zero on a perfect seed.
    warm_model = _mask_np(fwd.spectrum_numpy(warm_T_eV, warm_log_ne, warm_numfrac))
    warm_proj = (_normalise_spectrum(warm_model, strict=strict) - mean_np) @ basis_np.T
    warm_resid = warm_proj - obs_proj_np
    sigma_d = float(
        max(np.sqrt(np.mean(warm_resid**2)), 0.05 * (np.std(obs_proj_np) + 1e-12), 1e-12)
    )

    # ---- Bounded/transformed parameter packing ----
    # x[0], x[1] are unconstrained ``tanh`` pre-images for log T_eV / log10 n_e
    # (see ``_params``); x[2:] are softmax logits for composition. The warm
    # start sits at tanh-preimage 0 (= warm value) and the warm composition.
    log_T0 = float(np.log(max(warm_T_eV, 0.05)))
    theta0 = np.log(np.maximum(warm_numfrac, 1e-8))
    x0 = jnp.asarray(np.concatenate([[0.0, 0.0], theta0]))

    # Box half-widths: the ``tanh`` reparameterisation in ``_params`` confines T
    # to [warm/3, warm*3] and n_e to +/- box_logne dex of the warm start *by
    # construction*, so the prior-free SVD objective can no longer run T/n_e off
    # to an unphysical global minimum (the classic LIBS T<->composition
    # degeneracy). The Bayesian path additionally pulls (T, n_e) toward the warm
    # start with informative Gaussian priors (~15% in T, ~0.4 dex in n_e) and
    # adds a max-entropy composition regulariser; the joint path uses only the
    # bounded data term.
    use_prior = method.lower() == "bayesian"
    sigma_logT = 0.15
    sigma_logne = 0.40
    entropy_weight = 1e-3 if use_prior else 0.0
    box_logT = float(np.log(3.0))  # +/- a factor of 3 in T
    box_logne = 1.5  # +/- 1.5 dex in n_e

    def _proj(spec_norm):
        return (spec_norm - mean) @ basis.T

    def _softmax(logits):
        # Numerically stable softmax, hand-rolled: the physics-only constraint
        # (ruff TID251) bans the neural-network helper module, so we implement
        # the closure map directly instead of importing a library softmax.
        shifted = logits - jnp.max(logits)
        exp = jnp.exp(shifted)
        return exp / jnp.sum(exp)

    def _params(x):
        # T and log n_e use a ``tanh`` box reparameterisation so they CANNOT
        # leave the physical neighbourhood of the warm start by construction
        # (T within [warm/3, warm*3]; n_e within +/- box_logne dex). This is
        # what prevents the prior-free SVD objective from running T/n_e off to
        # an unphysical global minimum -- the classic failure mode for both the
        # joint and Bayesian full-spectrum fits. ``x[0]``/``x[1]`` are the
        # unconstrained tanh pre-images; ``x[2:]`` are softmax logits.
        log_T = log_T0 + box_logT * jnp.tanh(x[0])
        log_ne = warm_log_ne + box_logne * jnp.tanh(x[1])
        T_eV = jnp.exp(log_T)
        conc = _softmax(x[2:])
        return T_eV, log_ne, conc

    def objective(x):
        T_eV, log_ne, conc = _params(x)
        model = fwd.spectrum(T_eV, log_ne, conc)
        if keep_jnp is not None:
            # Exclude observable-thick line windows from BOTH sides of the
            # residual (obs_proj was built from the masked observed spectrum),
            # so the thin model is never scored against a saturated line.
            model = model * keep_jnp
        area = jnp.sum(jnp.abs(model)) + 1e-30
        model_norm = model / area
        resid = _proj(model_norm) - obs_proj
        # Gaussian negative-log-likelihood data term (0.5 * chi-square), in the
        # same units as the 0.5*(Delta/sigma)^2 priors below so they compete on
        # equal footing.
        data = 0.5 * jnp.sum((resid / sigma_d) ** 2)
        if use_prior:
            # Informative Gaussian priors pulling (T, n_e) toward the warm
            # start. The actual log-deviation is ``box_log* * tanh(x[i])``
            # (see ``_params``); sigma_log* is in nats.
            prior = 0.5 * (
                (box_logT * jnp.tanh(x[0]) / sigma_logT) ** 2
                + (box_logne * jnp.tanh(x[1]) / sigma_logne) ** 2
            )
            # Maximum-entropy (Dirichlet alpha->1) push away from degenerate
            # single-element minima: maximise entropy => subtract it.
            entropy = -jnp.sum(conc * jnp.log(conc + 1e-12))
            return data + prior - entropy_weight * entropy
        return data

    objective_jit = jax.jit(objective)
    initial_objective = float(objective_jit(x0))
    diag.objective_initial = initial_objective

    iterations = 0
    grad_norm = float("inf")
    final_x = x0
    final_objective = initial_objective
    real_fit = False
    cand_x = None  # the actual optimizer endpoint (None iff it crashed)
    moved_norm = float("nan")
    delta_obj = float("nan")
    bfgs_status = None
    bfgs_success = None
    convergence_reason: Optional[str] = None
    try:
        res = jax_minimize(
            objective_jit,
            x0,
            method="bfgs",
            options={"maxiter": max_iterations},
        )
        cand_x = res.x
        cand_obj = float(res.fun)
        iterations = int(res.nit) if hasattr(res, "nit") else max_iterations
        # The optimizer's own status/success — production NEVER consults these
        # (convergence is re-derived from coarse heuristics). Capture them so a
        # 'maxiter reached, still descending' is distinguishable from 'true min'.
        bfgs_status = int(res.status) if hasattr(res, "status") else None
        bfgs_success = bool(res.success) if hasattr(res, "success") else None
        grad_fn = jax.grad(objective_jit)
        grad_norm = float(jnp.linalg.norm(grad_fn(cand_x)))
        finite = bool(np.all(np.isfinite(np.asarray(cand_x)))) and np.isfinite(cand_obj)
        # A "real fit" means the optimiser actually moved and reduced the
        # conditioned objective (not a no-op / fall-back to the warm start).
        moved_norm = float(np.linalg.norm(np.asarray(cand_x) - np.asarray(x0)))
        moved = moved_norm > 1e-6
        delta_obj = initial_objective - cand_obj
        improved_obj = cand_obj < initial_objective - 1e-9
        decision = _decide_convergence(
            finite=finite, moved=moved, improved_obj=improved_obj, iterations=iterations
        )
        real_fit = decision.real_fit
        convergence_reason = decision.reason
        diag.optimizer_status = bfgs_status
        diag.optimizer_success = bfgs_success
        diag.objective_final = cand_obj
        diag.extra.update(
            {
                "moved_norm": moved_norm,
                "delta_obj": delta_obj,
                "grad_norm": grad_norm,
                "bfgs_status": bfgs_status,
                "bfgs_success": bfgs_success,
            }
        )
        if real_fit:
            final_x = cand_x
            final_objective = cand_obj
        else:
            diag.failure_reason = diag.failure_reason or f"not_converged: {decision.reason}"
            if not strict:
                # Production path: log the coarse booleans and keep warm start.
                logger.warning(
                    "Full-spectrum fit did not produce a real converged optimum "
                    "(finite=%s moved=%s improved=%s iters=%d); keeping warm start.",
                    finite,
                    moved,
                    improved_obj,
                    iterations,
                )
    except Exception as exc:  # noqa: BLE001 — degrade to warm start, never crash
        # Strict: re-raise a typed OptimizerFailure (chained). Non-strict: the
        # production path logs once and keeps the warm start.
        _handle_optimizer_exception(exc, strict=strict, diagnostics=diag)
        logger.warning("Full-spectrum optimisation failed (%r); keeping warm start.", exc)

    # fit_* provenance (sites 640-645): production fabricates fit_* = warm when
    # not real_fit, destroying the warm-vs-fit accounting. Strict mode never
    # fabricates warm — it surfaces the actual optimizer endpoint (cand_x) and
    # tags ``fit_source``/``fit_valid``. (A crash already raised in strict, so
    # cand_x is guaranteed present on the strict ``elif`` branch.)
    if real_fit:
        T_eV_fit, log_ne_fit, conc_fit = jax.jit(_params)(final_x)
        fit_source = "optimizer"
        fit_valid = True
    elif strict and cand_x is not None:
        T_eV_fit, log_ne_fit, conc_fit = jax.jit(_params)(cand_x)
        fit_source = "optimizer_rejected"
        fit_valid = False
    else:
        T_eV_fit = jnp.asarray(warm_T_eV)
        log_ne_fit = jnp.asarray(warm_log_ne)
        conc_fit = jnp.asarray(warm_numfrac)
        fit_source = "warm_fallback"
        fit_valid = False
    fit_numfrac = {el: float(np.asarray(conc_fit)[i]) for i, el in enumerate(elements)}
    fit_mass = _number_to_mass_fractions(fit_numfrac, strict=strict)
    fit_T_K = float(np.asarray(T_eV_fit)) * EV_TO_K
    # log_ne_fit is tanh-bounded to warm_log_ne +/- box_logne, but clamp
    # defensively so 10**log_ne can never overflow a Python float. (Site 649-651)
    log_ne_raw = float(np.asarray(log_ne_fit))
    log_ne_clamped = float(np.clip(log_ne_raw, 1.0, 25.0))
    if strict and log_ne_clamped != log_ne_raw:
        # The tanh box already bounds log_ne; a fired clamp means that invariant
        # was violated, so surface it instead of silently substituting.
        raise NonPhysicalResult(
            f"tanh-box invariant violated: fit log_ne={log_ne_raw:.4g} outside [1, 25]",
            diag,
        )
    fit_ne = float(10.0**log_ne_clamped)

    # Physical-plausibility adoption gate (truth-free). A *real* converged fit
    # is only ADOPTED into the returned composition when its (T, n_e) stayed
    # physically near the warm start — i.e. it did not ride to a box edge. The
    # prior-free ``joint`` objective on the SVD-compressed spectrum is degenerate
    # and rides T to whichever box edge minimises the spectral residual (a
    # classic LIBS T<->composition degeneracy); that is a real optimum of the
    # spectral objective but NOT a trustworthy composition, so it falls back to
    # the warm start here (non-strict). The informative-prior ``bayesian`` fit,
    # which stays in the warm-start neighbourhood, is adopted. ``converged``
    # still records that a real fit ran; ``adopted_fit`` records whether it was
    # trusted. Strict mode surfaces the untrusted fit instead of laundering warm.
    t_threshold = float(np.log(1.8))
    ne_threshold = 0.7
    T_ratio = abs(np.log(max(fit_T_K, 1.0) / max(float(warm_start_T_K), 1.0)))
    ne_ratio = abs(np.log10(max(fit_ne, 1e1) / max(float(warm_start_ne_cm3), 1e1)))
    physically_near = T_ratio < t_threshold and ne_ratio < ne_threshold
    if real_fit and not physically_near and not strict:
        logger.warning(
            "Full-spectrum fit converged but rode T/n_e to a degenerate "
            "edge (T %.0f vs warm %.0f K, n_e %.2e vs %.2e); not adopting "
            "(keeping warm start composition).",
            fit_T_K,
            float(warm_start_T_K),
            fit_ne,
            float(warm_start_ne_cm3),
        )
    adoption = _resolve_adoption(
        real_fit=real_fit,
        physically_near=physically_near,
        strict=strict,
        fit_T_K=fit_T_K,
        fit_ne=fit_ne,
        fit_mass=fit_mass,
        warm_T_K=float(warm_start_T_K),
        warm_ne=float(warm_start_ne_cm3),
        warm_mass=warm_mass,
        T_ratio=T_ratio,
        ne_ratio=ne_ratio,
        t_threshold=t_threshold,
        ne_threshold=ne_threshold,
    )
    adopted = adoption.adopted
    adopted_T_K = adoption.temperature_K
    adopted_ne = adoption.electron_density_cm3
    adopted_conc = adoption.concentrations
    if adoption.failure_reason:
        diag.failure_reason = diag.failure_reason or adoption.failure_reason
    diag.converged = real_fit
    diag.adopted = adopted
    diag.extra.update(
        {
            "T_ratio": T_ratio,
            "ne_ratio": ne_ratio,
            "physically_near": physically_near,
            "fit_source": fit_source,
            "fit_valid": fit_valid,
            "convergence_reason": convergence_reason,
        }
    )

    return FullSpectrumResult(
        temperature_K=adopted_T_K,
        electron_density_cm3=adopted_ne,
        concentrations=adopted_conc,
        warm_start_temperature_K=float(warm_start_T_K),
        warm_start_electron_density_cm3=float(warm_start_ne_cm3),
        warm_start_concentrations=dict(warm_mass),
        fit_temperature_K=fit_T_K,
        fit_electron_density_cm3=fit_ne,
        fit_concentrations=fit_mass,
        converged=real_fit,
        adopted_fit=adopted,
        initial_objective=initial_objective,
        final_objective=final_objective,
        iterations=iterations,
        gradient_norm=grad_norm,
        failure_reason=diag.failure_reason,
        fit_valid=fit_valid,
        fit_source=fit_source,
        diagnostics={
            "n_lines": fwd.n_lines,
            "n_wl": fwd.n_wl,
            "n_pc": int(k),
            "nstitch": fwd.plan.nstitch,
            "overlap": fwd.plan.overlap,
            "method": method.lower(),
            "dense_matrix_mb": fwd.n_lines * fwd.n_wl * 8 / 1e6,
            # Additive nested strict / no-fallback visibility (populated in both
            # modes; the existing keys above keep their exact production values).
            "strict_diagnostics": diag.to_dict(),
        },
    )
