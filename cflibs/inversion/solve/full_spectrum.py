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

logger = get_logger("inversion.solve.full_spectrum")


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


# ---------------------------------------------------------------------------
# Composition helpers
# ---------------------------------------------------------------------------


def _number_to_mass_fractions(
    number_fractions: Dict[str, float],
) -> Dict[str, float]:
    """Convert number/mole fractions to mass fractions (sum to 1).

    ``C_mass_i = C_i * AW_i / sum_j (C_j * AW_j)``.
    """
    weights = {
        el: float(c) * float(STANDARD_ATOMIC_MASSES.get(el, 50.0))
        for el, c in number_fractions.items()
    }
    total = sum(weights.values())
    if total <= 0:
        return {el: 0.0 for el in number_fractions}
    return {el: w / total for el, w in weights.items()}


def _mass_to_number_fractions(
    mass_fractions: Dict[str, float],
    elements: Sequence[str],
) -> np.ndarray:
    """Convert mass fractions to a number-fraction array over ``elements``.

    ``C_i = (m_i / AW_i) / sum_j (m_j / AW_j)``.  Missing elements seed a small
    floor so the softmax warm start is non-degenerate.
    """
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


def _normalise_spectrum(spec: np.ndarray) -> np.ndarray:
    """Area-normalise a spectrum (drop arbitrary radiometric scale)."""
    spec = np.asarray(spec, dtype=np.float64)
    area = float(np.sum(np.abs(spec)))
    if area <= 0:
        return spec
    return spec / area


def build_svd_basis(
    library: np.ndarray,
    observed: np.ndarray,
    *,
    n_components: int = 20,
    variance_target: float = 0.99995,
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
    obs_n = _normalise_spectrum(observed)
    rows = [_normalise_spectrum(row) for row in lib]
    rows.append(obs_n)
    X = np.vstack(rows)
    mean = X.mean(axis=0)
    Xc = X - mean
    # full_matrices=False → Vt is (min(n,p), p); rows are PC directions.
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var = S**2
    cum = np.cumsum(var) / max(float(np.sum(var)), 1e-300)
    k_var = int(np.searchsorted(cum, variance_target) + 1)
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
        return jnp.clip(intensity, 0.0, 1e12)

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
        return np.asarray(out, dtype=np.float64)


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
        Number of warm-start-centred forward spectra used to seed the SVD basis.
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

    Returns
    -------
    FullSpectrumResult
    """
    if not HAS_JAX:
        raise ImportError("JAX required for the full-spectrum solver")
    import jax  # noqa: PLC0415
    import jax.numpy as jnp  # noqa: PLC0415
    from jax.scipy.optimize import minimize as jax_minimize  # noqa: PLC0415

    elements = list(elements)
    n_el = len(elements)
    wl = np.asarray(wavelength, dtype=np.float64)
    obs = np.asarray(intensity, dtype=np.float64)

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
    warm_log_ne = float(np.log10(max(warm_start_ne_cm3, 1e10)))
    warm_numfrac = _mass_to_number_fractions(warm_mass, elements)

    fwd = _ChunkedForward(
        db_path,
        elements,
        wl,
        resolving_power=resolving_power,
        instrument_fwhm_nm=instrument_fwhm_nm,
    )

    # ---- Build the SVD library from a coarse warm-start-centred sweep ----
    # Vary T (+/-30%) and log n_e (+/-0.5 dex) and a few single-element-boosted
    # compositions so the basis spans the directions the fit will move along.
    lib_rows: List[np.ndarray] = []
    T_factors = np.linspace(0.7, 1.3, 3)
    ne_offsets = np.linspace(-0.5, 0.5, 3)
    for tf in T_factors:
        for dne in ne_offsets:
            lib_rows.append(fwd.spectrum_numpy(warm_T_eV * tf, warm_log_ne + dne, warm_numfrac))
    # A few composition perturbations (boost each element in turn) up to the
    # sweep budget so the residual's composition directions are represented.
    extra = max(0, sweep_points - len(lib_rows))
    for i in range(min(extra, n_el)):
        boosted = warm_numfrac.copy()
        boosted[i] = boosted[i] * 3.0 + 0.05
        boosted = boosted / boosted.sum()
        lib_rows.append(fwd.spectrum_numpy(warm_T_eV, warm_log_ne, boosted))
    library = np.vstack(lib_rows)

    basis_np, mean_np, k = build_svd_basis(library, obs, n_components=n_components)
    obs_norm = _normalise_spectrum(obs)
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
    warm_model = fwd.spectrum_numpy(warm_T_eV, warm_log_ne, warm_numfrac)
    warm_proj = (_normalise_spectrum(warm_model) - mean_np) @ basis_np.T
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
        # Numerically stable softmax (physics-only: jax.nn is banned, so this
        # is hand-rolled rather than ``jax.nn.softmax``).
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

    iterations = 0
    grad_norm = float("inf")
    final_x = x0
    final_objective = initial_objective
    real_fit = False
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
        grad_fn = jax.grad(objective_jit)
        grad_norm = float(jnp.linalg.norm(grad_fn(cand_x)))
        finite = bool(np.all(np.isfinite(np.asarray(cand_x)))) and np.isfinite(cand_obj)
        # A "real fit" means the optimiser actually moved and reduced the
        # conditioned objective (not a no-op / fall-back to the warm start).
        moved = float(np.linalg.norm(np.asarray(cand_x) - np.asarray(x0))) > 1e-6
        improved_obj = cand_obj < initial_objective - 1e-9
        if finite and moved and improved_obj and iterations > 0:
            real_fit = True
            final_x = cand_x
            final_objective = cand_obj
        else:
            logger.warning(
                "Full-spectrum fit did not produce a real converged optimum "
                "(finite=%s moved=%s improved=%s iters=%d); keeping warm start.",
                finite,
                moved,
                improved_obj,
                iterations,
            )
    except Exception as exc:  # noqa: BLE001 — degrade to warm start, never crash
        logger.warning("Full-spectrum optimisation failed (%r); keeping warm start.", exc)

    if real_fit:
        T_eV_fit, log_ne_fit, conc_fit = jax.jit(_params)(final_x)
    else:
        T_eV_fit = jnp.asarray(warm_T_eV)
        log_ne_fit = jnp.asarray(warm_log_ne)
        conc_fit = jnp.asarray(warm_numfrac)
    fit_numfrac = {el: float(np.asarray(conc_fit)[i]) for i, el in enumerate(elements)}
    fit_mass = _number_to_mass_fractions(fit_numfrac)
    fit_T_K = float(np.asarray(T_eV_fit)) * EV_TO_K
    # log_ne_fit is tanh-bounded to warm_log_ne +/- box_logne, but clamp
    # defensively so 10**log_ne can never overflow a Python float.
    fit_ne = float(10.0 ** float(np.clip(np.asarray(log_ne_fit), 1.0, 25.0)))

    # Physical-plausibility adoption gate (truth-free). A *real* converged fit
    # is only ADOPTED into the returned composition when its (T, n_e) stayed
    # physically near the warm start — i.e. it did not ride to a box edge. The
    # prior-free ``joint`` objective on the SVD-compressed spectrum is degenerate
    # and rides T to whichever box edge minimises the spectral residual (a
    # classic LIBS T<->composition degeneracy); that is a real optimum of the
    # spectral objective but NOT a trustworthy composition, so it falls back to
    # the warm start here. The informative-prior ``bayesian`` fit, which stays
    # in the warm-start neighbourhood, is adopted. ``converged`` still records
    # that a real fit ran; ``adopted_fit`` records whether it was trusted.
    T_ratio = abs(np.log(max(fit_T_K, 1.0) / max(float(warm_start_T_K), 1.0)))
    ne_ratio = abs(np.log10(max(fit_ne, 1e1) / max(float(warm_start_ne_cm3), 1e1)))
    physically_near = T_ratio < np.log(1.8) and ne_ratio < 0.7
    adopted = bool(real_fit and physically_near)
    if real_fit and not physically_near:
        logger.warning(
            "Full-spectrum fit converged but rode T/n_e to a degenerate "
            "edge (T %.0f vs warm %.0f K, n_e %.2e vs %.2e); not adopting "
            "(keeping warm start composition).",
            fit_T_K,
            float(warm_start_T_K),
            fit_ne,
            float(warm_start_ne_cm3),
        )
    if adopted:
        adopted_T_K = fit_T_K
        adopted_ne = fit_ne
        adopted_conc = fit_mass
    else:
        adopted_T_K = float(warm_start_T_K)
        adopted_ne = float(warm_start_ne_cm3)
        adopted_conc = dict(warm_mass)

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
        diagnostics={
            "n_lines": fwd.n_lines,
            "n_wl": fwd.n_wl,
            "n_pc": int(k),
            "nstitch": fwd.plan.nstitch,
            "overlap": fwd.plan.overlap,
            "method": method.lower(),
            "dense_matrix_mb": fwd.n_lines * fwd.n_wl * 8 / 1e6,
        },
    )
