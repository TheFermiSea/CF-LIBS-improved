"""J10 — Gornushkin-style population forward-fitting identification (ADR-0004 §8.1 track C).

Batched, fixed-shape, vmap-clean forward-fitting element identifier. The idea
(Gornushkin & Völker 2022, PMC9573556; Demidov et al. 2016; ADR-0004 §9): instead
of matching detected peaks against a catalog one element at a time, *forward model*
a large population of candidate plasma configurations — each a draw of
``(T, n_e, candidate-element concentration vector)`` — push every draw through the
**existing, frozen** forward kernel
(:func:`cflibs.radiation.kernels.forward_model`), and score each synthetic spectrum
against the measured one with an element-weighted full-spectrum correlation cost.
An element is *called present* when configurations that include it score materially
better than configurations that exclude it (a leave-one-element-out marginal-evidence
contrast, made discrete by a BIC-style residual criterion that mirrors
:func:`cflibs.inversion.identify.model_selection._compute_bic`).

Why this stage exists (the recall payoff)
------------------------------------------
The audit (``docs/audit/2026-06-09-overhaul/03-identification.md`` F3) found the
production identifier has *no multi-line intensity-coherence test* on the path that
feeds the solver: single-line coincidences leak through, and recall sits at
0.27-0.58 against precision 0.89-1.00. Forward-fitting is exactly that missing
coherence test, run at full forward-physics fidelity. The acceptance bar
(ADR-0004 §8.3 item 3 / J10 spec §3) is micro-F1 >= +0.03 on the optimization split
with precision loss <= 0.02.

Fixed-shape contract (the whole point — ADR-0004 §1.1)
------------------------------------------------------
Everything inside the jit graph is fixed-shape: the candidate population is a padded
``(B_eval, ...)`` batch with a per-config validity mask; the element axis is the
padded snapshot element superset (``E`` = ``len(snapshot.element_symbols)``); subset
membership is an ``(B_eval, E)`` float mask, never a data-dependent reslice. No draw
count, element count, or line count depends on the data, so a single compiled graph
serves every spectrum (no 7394-cache-entry pathology). Failures surface as quality
flags / masked-out configs, never exceptions.

Device boundary
---------------
The *only* per-spectrum host<->device seam is candidate-population assembly
(:func:`build_candidate_population`) and the host-side stacking of candidate plasma
states into one batched :class:`~cflibs.plasma.state.SingleZoneLTEPlasma` pytree —
exactly the seam ``snapshot.py`` documents. The scoring core
(:func:`evaluate_population`, :func:`correlation_cost`,
:func:`forward_fit_presence_scores`) is pure ``jit``/``vmap`` over arrays and the
``PipelineSnapshot`` pytree. No SQLite, no host imports — import-hygiene clean.

This module does NOT duplicate any forward physics: it calls
:func:`cflibs.radiation.kernels.forward_model` verbatim (the single source of truth,
ADR-0004 §4 row 11). The bridge from :class:`PipelineSnapshot` to the kernel's
``AtomicSnapshot`` is :meth:`PipelineSnapshot.to_atomic_snapshot`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from cflibs.core.constants import KB_EV
from cflibs.radiation.kernels import BroadeningMode, forward_model

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.instrument.model import InstrumentModel
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot
    from cflibs.plasma.state import SingleZoneLTEPlasma


# ---------------------------------------------------------------------------
# Broadening-mode resolution (StaticConfig.broadening_mode is a hashable str;
# the kernel wants the BroadeningMode enum — resolved on the host at trace time).
# ---------------------------------------------------------------------------

_BROADENING_BY_NAME: dict[str, BroadeningMode] = {
    "gaussian": BroadeningMode.NIST_PARITY,
    "nist_parity": BroadeningMode.NIST_PARITY,
    "resolving_power": BroadeningMode.NIST_PARITY,
    "doppler": BroadeningMode.PHYSICAL_DOPPLER,
    "physical_doppler": BroadeningMode.PHYSICAL_DOPPLER,
    "legacy": BroadeningMode.LEGACY,
}


def _resolve_broadening_mode(mode: str | BroadeningMode) -> BroadeningMode:
    """Map a :class:`StaticConfig.broadening_mode` string to the kernel enum.

    Host-side, resolved at jit-trace time so the mode keys the compiled graph
    (it is a static dispatch, never a traced value).
    """
    if isinstance(mode, BroadeningMode):
        return mode
    try:
        return _BROADENING_BY_NAME[str(mode).lower()]
    except KeyError as exc:  # pragma: no cover - defensive
        raise ValueError(
            f"unknown broadening_mode {mode!r}; expected one of "
            f"{sorted(_BROADENING_BY_NAME)} or a BroadeningMode"
        ) from exc


# ---------------------------------------------------------------------------
# Result container (NamedTuple => a pytree; all leaves fixed-shape arrays).
# ---------------------------------------------------------------------------


class ForwardFitResult(NamedTuple):
    """Fixed-shape result of a population forward-fit identification pass.

    Attributes
    ----------
    element_present : ndarray of float, shape (E,)
        ``1.0`` where the element is called present, ``0.0`` otherwise. ``E`` is
        the padded snapshot element-superset length. Real elements live in
        ``[0, E_valid)``; padding entries are forced to ``0.0`` via
        ``element_valid_mask``.
    presence_score : ndarray, shape (E,)
        Per-element marginal-evidence contrast (best correlation among configs
        that *include* the element minus best among configs that *exclude* it).
        Higher => stronger evidence of presence. Padding entries are ``-inf``.
    best_bic : ndarray, shape (E,)
        BIC of the best-fitting config that includes the element (lower is
        better), mirroring :func:`model_selection._compute_bic`.
    best_correlation : float scalar
        Best (highest) correlation achieved over the whole population — a
        global goodness-of-fit canary for the spectrum.
    best_config_index : int scalar
        Index into the population of the globally best config (argmax
        correlation), for downstream (T, n_e, C) read-out / gradient polish.
    n_valid_configs : int scalar
        Number of non-padded candidate configs that contributed (truncation /
        K-saturation canary).
    """

    element_present: Any
    presence_score: Any
    best_bic: Any
    best_correlation: Any
    best_config_index: Any
    n_valid_configs: Any


class PolishResult(NamedTuple):
    """Fixed-shape result of the gradient (Gauss-Newton/LM) polish loop.

    The optional refinement step (J10 spec §1 last clause; ADR-0004 §6.1) that the
    MC-CF prior art lacks entirely: starting from a surviving candidate's
    coarse ``(T, n_e, C)`` draw, run a fixed-K Levenberg-Marquardt descent on the
    shape-correlation residual to snap each candidate onto its locally best fit
    *before* the present/absent call. Every leaf is a fixed-shape array; a leading
    axis appears when :func:`polish_candidates` ``vmap``-s over a candidate batch.

    Attributes
    ----------
    params : ndarray, shape (..., P)
        Polished parameter vector in the optimizer parameterization
        ``[ln T_eV, log10 n_e, theta_0..theta_{E-1}]`` — bit-for-bit the
        ``joint_optimizer.py:16-23`` packing (``P = 2 + E``).
    temperature_k : ndarray, shape (...)
        Polished temperature, K (``exp(params[0]) / KB_EV``).
    electron_density : ndarray, shape (...)
        Polished electron density, cm^-3 (``10 ** params[1]``).
    concentrations : ndarray, shape (..., E)
        Polished number fractions, ``softmax(theta)`` (sum-to-one simplex).
    correlation : ndarray, shape (...)
        Shape-correlation of the polished fit vs the measured spectrum
        (the maximized objective; ``cost = 1 - correlation``).
    init_correlation : ndarray, shape (...)
        Correlation at the *start* point — lets the caller assert the polish
        never regresses (monotone-improvement canary, asserted in tests).
    n_steps : int scalar
        Fixed LM step count actually run (a static canary; equals the requested
        ``max_steps`` since the loop is fixed-K with no early exit).
    """

    params: Any
    temperature_k: Any
    electron_density: Any
    concentrations: Any
    correlation: Any
    init_correlation: Any
    n_steps: Any


# ---------------------------------------------------------------------------
# Core scoring kernels (pure jit/vmap over arrays + the snapshot pytree).
# ---------------------------------------------------------------------------


def evaluate_population(
    batched_plasma: "SingleZoneLTEPlasma",
    snapshot: "PipelineSnapshot",
    instrument: "InstrumentModel",
    wavelengths_nm: Any,
    *,
    broadening_mode: BroadeningMode = BroadeningMode.NIST_PARITY,
    path_length_m: float = 0.01,
    apply_self_absorption: bool = False,
    apply_stark: bool = False,
) -> Any:
    """Forward model an entire candidate population in one ``vmap``.

    ``vmap(forward_model, in_axes=(0, None, None, None))`` over the batched
    plasma pytree, sharing the snapshot / instrument / wavelength grid (the
    manifold batch pattern, ``batch_forward.py:13``, generalized to the unified
    snapshot). The forward physics is the frozen kernel verbatim — no
    duplication (ADR-0004 §4 row 11).

    Parameters
    ----------
    batched_plasma : SingleZoneLTEPlasma
        A *batched* plasma pytree: every leaf has a leading axis of length
        ``B_eval``. Build it on the host by stacking per-config plasma states
        (:func:`stack_plasma_states`).
    snapshot : PipelineSnapshot
        Unified atomic snapshot (the candidate catalog). Bridged to the
        kernel's ``AtomicSnapshot`` via
        :meth:`PipelineSnapshot.to_atomic_snapshot`.
    instrument : InstrumentModel
        Shared instrument model.
    wavelengths_nm : array, shape (N_wl,)
        Shared output wavelength grid, nm.
    broadening_mode : BroadeningMode, **static**
        Forward-model broadening dispatch.
    path_length_m : float
        Plasma path length, m. Consulted only when ``apply_self_absorption``.
    apply_self_absorption, apply_stark : bool, **static**
        Forward-kernel toggles.

    Returns
    -------
    spectra : ndarray, shape (B_eval, N_wl)
        Synthetic spectra, one per candidate config. Fixed shape.
    """
    atomic = snapshot.to_atomic_snapshot(include_levels=True)
    wl = jnp.asarray(wavelengths_nm)

    def _one(plasma: "SingleZoneLTEPlasma") -> Any:
        return forward_model(
            plasma,
            atomic,
            instrument,
            wl,
            broadening_mode=broadening_mode,
            path_length_m=path_length_m,
            apply_self_absorption=apply_self_absorption,
            apply_stark=apply_stark,
        )

    return jax.vmap(_one)(batched_plasma)


def correlation_cost(
    synthetic: Any,
    measured: Any,
    *,
    weights: Any = None,
    eps: float = 1e-30,
) -> Any:
    """Continuum-removed weighted Pearson correlation, per synthetic spectrum.

    The Gornushkin-Völker 2022 cost shape (ADR-0004 §9): both spectra are
    continuum-removed (mean-subtracted over the wavelength axis, optionally
    weighted) and the normalized cross-correlation is taken. ``+1`` is a perfect
    shape match; values near ``0`` mean no coherence. We return the *correlation*
    (higher is better); a cost is ``1 - correlation``.

    NaN-safe and division-safe: a zero-variance synthetic spectrum (e.g. a config
    with all-zero composition) scores ``0.0`` rather than ``NaN``. Fully
    vmap-clean: ``synthetic`` may carry a leading batch axis.

    Parameters
    ----------
    synthetic : array, shape (..., N_wl)
        Synthetic spectra; a leading batch axis is correlated row-by-row.
    measured : array, shape (N_wl,)
        Measured spectrum (shared across the batch).
    weights : array, shape (N_wl,), optional
        Per-wavelength weights (e.g. an element-emphasis or validity mask).
        ``None`` => uniform weights.
    eps : float
        Variance floor guarding the normalization.

    Returns
    -------
    correlation : array, shape (...)
        Weighted Pearson correlation per row in ``[-1, 1]``.
    """
    syn = jnp.asarray(synthetic)
    meas = jnp.asarray(measured)
    if weights is None:
        w = jnp.ones_like(meas)
    else:
        w = jnp.asarray(weights)
    w = w / jnp.maximum(jnp.sum(w), eps)

    meas_mean = jnp.sum(w * meas)
    meas_c = meas - meas_mean
    meas_var = jnp.sum(w * meas_c * meas_c)
    meas_norm = jnp.sqrt(jnp.maximum(meas_var, eps))

    def _corr_row(row: Any) -> Any:
        row_mean = jnp.sum(w * row)
        row_c = row - row_mean
        row_var = jnp.sum(w * row_c * row_c)
        cov = jnp.sum(w * row_c * meas_c)
        denom = jnp.sqrt(jnp.maximum(row_var, eps)) * meas_norm
        corr = cov / jnp.maximum(denom, eps)
        # Zero-variance synthetic => no coherence (not a perfect match).
        good = (row_var > eps) & (meas_var > eps)
        return jnp.where(good, corr, 0.0)

    if syn.ndim == 1:
        return _corr_row(syn)
    return jax.vmap(_corr_row)(syn.reshape(-1, syn.shape[-1])).reshape(syn.shape[:-1])


def bic_cost(synthetic: Any, measured: Any, *, n_params: Any) -> Any:
    """BIC over each synthetic spectrum vs the measured one (lower is better).

    ``BIC = n * ln(RSS / n) + k * ln(n)`` — bit-for-bit the criterion in
    :func:`cflibs.inversion.identify.model_selection._compute_bic`, but the
    spectra are L2-normalized first so amplitude (which forward-fit does not
    constrain — only shape) does not dominate the residual. ``k`` is the number
    of free parameters of each config (e.g. the count of included elements + 2
    for ``T, n_e``).

    Vmap-clean: ``synthetic`` may carry a leading batch axis; ``n_params`` then
    broadcasts row-wise.

    Returns
    -------
    bic : array, shape (...)
        BIC per row.
    """
    syn = jnp.asarray(synthetic)
    meas = jnp.asarray(measured)
    n = meas.shape[-1]
    tiny = jnp.finfo(jnp.result_type(meas.dtype, jnp.float32)).tiny

    meas_norm = jnp.maximum(jnp.sqrt(jnp.sum(meas * meas)), tiny)
    meas_u = meas / meas_norm

    def _bic_row(row: Any, k: Any) -> Any:
        row_norm = jnp.maximum(jnp.sqrt(jnp.sum(row * row)), tiny)
        row_u = row / row_norm
        resid = meas_u - row_u
        rss = jnp.maximum(jnp.sum(resid * resid), tiny)
        return n * jnp.log(rss / n) + jnp.asarray(k, dtype=meas.dtype) * jnp.log(jnp.asarray(n))

    if syn.ndim == 1:
        return _bic_row(syn, jnp.asarray(n_params))
    k_arr = jnp.broadcast_to(jnp.asarray(n_params), (syn.shape[0],))
    return jax.vmap(_bic_row)(syn.reshape(syn.shape[0], -1), k_arr)


def forward_fit_presence_scores(
    correlations: Any,
    bics: Any,
    element_membership: Any,
    config_valid: Any,
    element_valid: Any,
    *,
    presence_threshold: float = 0.0,
    require_bic: bool = False,
    bic_margin: float = 0.0,
) -> ForwardFitResult:
    """Marginal-evidence element calls from a scored candidate population.

    The leave-one-element-out contrast that *is* the multi-line coherence test
    (audit F3): for each element ``e`` we compare the best correlation achieved
    by configs that *include* ``e`` against the best achieved by configs that
    *exclude* ``e``. A large positive gap means the data demand ``e`` — its lines
    cannot be coherently explained without it. The discrete call is gated on this
    gap (``presence_threshold``); BIC of the best including-config is returned as
    a secondary, model-comparison-grounded score.

    Optional BIC-margin presence gate (opt-in, default off => bit-identical to the
    pure correlation-gap call). The same leave-one-element-out contrast is also
    available as a *model-selection* test: a confounder pays ``~k ln n`` BIC per
    extra element and rarely earns it back from coincidental lines, whereas a true
    major's residual reduction clears it easily. When ``require_bic`` is set, an
    element is called present only if it *both* clears the correlation gap **and**
    its inclusion lowers BIC by more than ``bic_margin`` (``bic_improvement =
    best_bic_exc - best_bic_inc``; BIC lower-is-better, so a positive improvement
    means including the element fits better even after the complexity penalty).
    The BIC gate can only *remove* calls (an AND with the existing decision), never
    add them — it raises precision without dropping recall.

    Fully fixed-shape and branch-free: ``element_membership`` is a dense
    ``(B_eval, E)`` float mask, so adding/removing an element from a subset is a
    mask flip, never a reslice (no per-subset recompiles, J10 spec §2). Invalid
    (padded) configs and elements are neutralized with ``-inf`` sentinels.

    Parameters
    ----------
    correlations : array, shape (B_eval,)
        Per-config correlation (output of :func:`correlation_cost`).
    bics : array, shape (B_eval,)
        Per-config BIC (output of :func:`bic_cost`).
    element_membership : array, shape (B_eval, E)
        ``1.0`` where config ``b`` includes element ``e`` with non-trivial
        concentration, else ``0.0``.
    config_valid : array, shape (B_eval,)
        ``1.0`` for a real (non-padded) config, ``0.0`` for padding.
    element_valid : array, shape (E,)
        ``1.0`` for a real snapshot element, ``0.0`` for padding.
    presence_threshold : float
        Minimum include-minus-exclude correlation gap to call an element present.
    require_bic : bool
        When ``True``, additionally require ``bic_improvement > bic_margin`` to
        call an element present (a BIC-margin presence gate). Default ``False``
        yields exactly the existing correlation-gap decision (bit-identical).
    bic_margin : float
        Minimum BIC improvement (``best_bic_exc - best_bic_inc``) required when
        ``require_bic`` is set. ``0.0`` requires only that including the element
        not raise BIC.

    Returns
    -------
    ForwardFitResult
    """
    corr = jnp.asarray(correlations)
    bic = jnp.asarray(bics)
    member = jnp.asarray(element_membership)
    cfg_ok = jnp.asarray(config_valid).astype(corr.dtype)
    el_ok = jnp.asarray(element_valid).astype(corr.dtype)

    neg_inf = jnp.asarray(-jnp.inf, dtype=corr.dtype)
    pos_inf = jnp.asarray(jnp.inf, dtype=corr.dtype)

    # Mask invalid configs out of every reduction.
    corr_valid = jnp.where(cfg_ok > 0, corr, neg_inf)  # (B,)
    bic_valid = jnp.where(cfg_ok > 0, bic, pos_inf)  # (B,)

    inc = (member > 0) & (cfg_ok[:, None] > 0)  # (B, E)
    exc = (member <= 0) & (cfg_ok[:, None] > 0)  # (B, E)

    # Best correlation among including / excluding configs, per element.
    best_inc = jnp.max(jnp.where(inc, corr_valid[:, None], neg_inf), axis=0)  # (E,)
    best_exc = jnp.max(jnp.where(exc, corr_valid[:, None], neg_inf), axis=0)  # (E,)
    # Best (lowest) BIC among including configs, per element.
    best_bic = jnp.min(jnp.where(inc, bic_valid[:, None], pos_inf), axis=0)  # (E,)
    # Best (lowest) BIC among EXCLUDING configs, per element (mirrors best_exc).
    best_bic_exc = jnp.min(jnp.where(exc, bic_valid[:, None], pos_inf), axis=0)  # (E,)

    # An element with no including config has best_inc = -inf; the gap is then
    # -inf and it is (correctly) not called. Elements never excluded (best_exc
    # = -inf) get a +inf gap => strongly called, which is right: every viable
    # config needs them.
    gap = best_inc - best_exc  # (E,)
    present_corr = (gap > presence_threshold) & (el_ok > 0) & jnp.isfinite(best_inc)

    # BIC-margin presence gate (opt-in). BIC lower-is-better, so including the
    # element pays off when the best excluding BIC exceeds the best including BIC
    # by more than the margin. An AND with the correlation call => only removes.
    bic_improvement = best_bic_exc - best_bic  # (E,)
    present_bic = bic_improvement > bic_margin
    present = jnp.where(require_bic, present_corr & present_bic, present_corr)

    presence_score = jnp.where(el_ok > 0, gap, neg_inf)
    best_bic = jnp.where(el_ok > 0, best_bic, pos_inf)

    best_corr = jnp.max(corr_valid)
    best_idx = jnp.argmax(corr_valid)
    n_valid = jnp.sum(cfg_ok).astype(jnp.int32)

    return ForwardFitResult(
        element_present=present.astype(corr.dtype),
        presence_score=presence_score,
        best_bic=best_bic,
        best_correlation=best_corr,
        best_config_index=best_idx.astype(jnp.int32),
        n_valid_configs=n_valid,
    )


# ---------------------------------------------------------------------------
# Gradient polish (fixed-K Gauss-Newton / Levenberg-Marquardt) — the ~30 % of
# J10 that the MC-CF prior art lacks (J10 spec §1 last clause; ADR-0004 §6.1).
#
# Parameterization (bit-for-bit joint_optimizer.py:16-23, the reference oracle):
#     x[0] = ln(T_eV),   x[1] = log10(n_e),   x[2:] = theta (softmax logits).
# The polished composition is C = softmax(theta) — the simplex reparameterization
# ADR-0004 §6.4 mandates ("physical constraints by reparameterization not
# relaxation: log-T, log-n_e, softmax/ILR simplex"; §6.1 notes standard ≡ ILR).
#
# Objective: forward-fit constrains *shape*, not amplitude (Gornushkin-Volker;
# correlation_cost above). So we minimize the continuum-removed, unit-norm
# residual r = meas_u - syn_u, whose ||r||^2 is monotone in (1 - correlation).
# That gives a genuine (N_wl,)-residual least-squares problem over the
# P = 2 + E parameters, so LM is real Gauss-Newton (not a rank-1 scalar fit):
#     (J^T J + lambda diag(J^T J)) delta = J^T r,    x <- x - delta  (if accepted)
# solved with core-JAX jnp.linalg.solve — NO jaxopt/optimistix/lineax/equinox
# (banned, ADR-0004 §6.1). Fixed-K lax.scan, accept/reject via jnp.where: no
# data-dependent control flow, fully vmap-clean and reverse-differentiable.
# ---------------------------------------------------------------------------


def pack_polish_params(temperature_k: Any, electron_density: Any, concentrations: Any) -> Any:
    """Pack ``(T_K, n_e, C)`` into the optimizer vector ``[lnT_eV, log10 n_e, theta]``.

    Mirrors :meth:`JointOptimizer._pack_params` (``joint_optimizer.py:599``): the
    temperature enters as ``ln(T_eV)`` with ``T_eV = T_K * KB_EV``, the electron
    density as ``log10(n_e)``, and the composition as softmax logits
    ``theta = ln(C)`` (shift-invariant; softmax recovers ``C``). Concentrations are
    floored at ``1e-10`` exactly as the reference does, so ``ln`` is finite.

    Parameters
    ----------
    temperature_k : array-like scalar
        Temperature, K.
    electron_density : array-like scalar
        Electron density, cm^-3.
    concentrations : array, shape (E,)
        Number fractions over the element superset.

    Returns
    -------
    x : ndarray, shape (E + 2,)
        Packed parameter vector.
    """
    t_k = jnp.asarray(temperature_k)
    ne = jnp.asarray(electron_density)
    conc = jnp.asarray(concentrations)
    t_ev = jnp.maximum(t_k * KB_EV, 0.1 * KB_EV)
    log_t = jnp.log(jnp.maximum(t_ev, 1e-30))
    log_ne = jnp.log10(jnp.maximum(ne, 1e10))
    theta = jnp.log(jnp.maximum(conc, 1e-10))
    return jnp.concatenate([jnp.stack([log_t, log_ne]), theta])


def unpack_polish_params(x: Any) -> tuple[Any, Any, Any]:
    """Invert :func:`pack_polish_params` => ``(T_K, n_e, C)``.

    ``T_K = exp(x[0]) / KB_EV``, ``n_e = 10 ** x[1]``, ``C = softmax(x[2:])``
    (log-sum-exp-stable softmax, the simplex map of ``softmax_closure`` written
    as explicit ``jnp`` so no banned ``jax.nn`` import is needed — ADR-0004 §6.4).

    Returns
    -------
    temperature_k, electron_density, concentrations
    """
    xx = jnp.asarray(x)
    t_ev = jnp.exp(xx[0])
    t_k = t_ev / KB_EV
    ne = jnp.power(10.0, xx[1])
    theta = xx[2:]
    theta_max = jnp.max(theta)
    w = jnp.exp(theta - theta_max)
    conc = w / jnp.sum(w)
    return t_k, ne, conc


def _polish_plasma_from_params(x: Any, element_symbols: tuple[str, ...]) -> "SingleZoneLTEPlasma":
    """Build a traced :class:`SingleZoneLTEPlasma` from a polish parameter vector.

    Constructed via ``object.__new__`` + field set (the pattern the grad-smoke
    path already uses, ``test_parity_j10.py:300``) so ``__init__``'s f-string
    logging never runs on a tracer. Differentiable end-to-end in ``x``.
    """
    from cflibs.plasma.state import SingleZoneLTEPlasma

    t_k, ne, conc = unpack_polish_params(x)
    species = {el: conc[j] for j, el in enumerate(element_symbols)}
    plasma = object.__new__(SingleZoneLTEPlasma)
    plasma.T_e = t_k
    plasma.n_e = ne
    plasma.species = species
    plasma.T_g = None
    plasma.pressure = None
    return plasma


def _unit_centered(vec: Any, weights: Any, eps: float) -> Any:
    """Weighted continuum-removal + unit-norm of a single spectrum.

    ``v_c = v - <v>_w`` then ``v_c / ||sqrt(w) v_c||``. The squared Euclidean
    distance between two such unit vectors is ``2(1 - corr)``, so least-squares on
    the difference *is* correlation maximization — the bridge that lets the
    Gornushkin shape cost be polished by Gauss-Newton.
    """
    v = jnp.asarray(vec)
    w = weights / jnp.maximum(jnp.sum(weights), eps)
    v_mean = jnp.sum(w * v)
    v_c = v - v_mean
    sw = jnp.sqrt(w)
    scaled = sw * v_c
    norm = jnp.sqrt(jnp.maximum(jnp.sum(scaled * scaled), eps))
    return scaled / norm


def gauss_newton_polish(
    x0: Any,
    measured: Any,
    snapshot: "PipelineSnapshot",
    instrument: "InstrumentModel",
    wavelengths_nm: Any,
    *,
    element_symbols: tuple[str, ...],
    broadening_mode: BroadeningMode = BroadeningMode.NIST_PARITY,
    path_length_m: float = 0.01,
    apply_self_absorption: bool = False,
    apply_stark: bool = False,
    max_steps: int = 6,
    lm_init: float = 1e-2,
    lm_up: float = 3.0,
    lm_down: float = 0.5,
    weights: Any = None,
    eps: float = 1e-30,
) -> PolishResult:
    """Fixed-K Levenberg-Marquardt polish of one candidate ``(T, n_e, C)`` draw.

    The differentiable refinement ADR-0004 §6.1 specifies: ``minimize over
    theta = (ln T, alpha, beta)`` ... ``by fixed-K Gauss-Newton/LM on <= (E+2)
    parameters``. Here ``alpha`` is the softmax-logit composition block and the
    objective is the shape-correlation residual (forward-fit fits shape, not the
    free amplitude ``beta``, so ``beta`` is absorbed by the unit-norm). Each step:

    1. ``r(x) = meas_u - syn_u(x)``  — both continuum-removed + unit-norm, ``(N_wl,)``.
    2. ``J = dr/dx``  via :func:`jax.jacfwd` (``P`` columns; forward-mode is cheap
       when ``P << N_wl``), ``(N_wl, P)``.
    3. damped normal equations ``(J^T J + lambda diag(J^T J)) delta = J^T r``
       solved by core-JAX :func:`jnp.linalg.solve`.
    4. trial ``x_new = x - delta``; **accept** (and shrink ``lambda``) iff the
       correlation strictly improves, else **reject** (keep ``x``, grow ``lambda``)
       — both branches via :func:`jnp.where`, so the graph is fixed-shape.

    No data-dependent control flow (fixed ``max_steps`` :func:`jax.lax.scan`),
    so the whole routine is ``vmap``/``jit``/``grad`` clean. NaN-guarded: a
    non-finite step is rejected, so a degenerate candidate simply keeps ``x0``.

    Parameters
    ----------
    x0 : array, shape (P,)
        Start vector from :func:`pack_polish_params` (``P = E + 2``).
    measured : array, shape (N_wl,)
        Measured spectrum.
    snapshot, instrument, wavelengths_nm :
        Forward-model inputs (frozen kernel, no duplication).
    element_symbols : tuple of str
        Element superset; column order of the ``theta`` block.
    broadening_mode, path_length_m, apply_self_absorption, apply_stark :
        Forward-kernel toggles (static).
    max_steps : int
        Fixed LM iteration count (keys the compiled graph; no early exit).
    lm_init, lm_up, lm_down : float
        LM damping schedule (initial damping, reject-grow, accept-shrink factors).
    weights : array, shape (N_wl,), optional
        Per-wavelength weights for continuum-removal + correlation.
    eps : float
        Numerical floor.

    Returns
    -------
    PolishResult
        Single-candidate (no leading batch axis). ``vmap`` this over a candidate
        batch (:func:`polish_candidates`).
    """
    atomic = snapshot.to_atomic_snapshot(include_levels=True)
    wl = jnp.asarray(wavelengths_nm)
    meas = jnp.asarray(measured)
    x0 = jnp.asarray(x0)
    p = x0.shape[0]
    if weights is None:
        w = jnp.ones_like(meas)
    else:
        w = jnp.asarray(weights)

    meas_u = _unit_centered(meas, w, eps)

    def _forward(x: Any) -> Any:
        plasma = _polish_plasma_from_params(x, element_symbols)
        return forward_model(
            plasma,
            atomic,
            instrument,
            wl,
            broadening_mode=broadening_mode,
            path_length_m=path_length_m,
            apply_self_absorption=apply_self_absorption,
            apply_stark=apply_stark,
        )

    def _residual(x: Any) -> Any:
        syn_u = _unit_centered(_forward(x), w, eps)
        return meas_u - syn_u  # (N_wl,)

    def _correlation(x: Any) -> Any:
        # corr = 1 - 0.5 * ||r||^2 for unit vectors; computed directly for clarity.
        syn_u = _unit_centered(_forward(x), w, eps)
        return jnp.sum(meas_u * syn_u)

    eye = jnp.eye(p, dtype=x0.dtype)

    def _step(carry: Any, _: Any) -> Any:
        x, lam, corr = carry
        r = _residual(x)  # (N_wl,)
        jac = jax.jacfwd(_residual)(x)  # (N_wl, P)
        jtj = jac.T @ jac  # (P, P)
        jtr = jac.T @ r  # (P,)
        # LM: scale the damping by the diagonal of J^T J (Marquardt's scaling).
        diag = jnp.clip(jnp.diagonal(jtj), eps, None)
        a = jtj + lam * (diag[:, None] * eye)
        delta = jnp.linalg.solve(a, jtr)
        x_trial = x - delta
        # Reject non-finite trials outright (keep x, grow damping).
        finite = jnp.all(jnp.isfinite(x_trial))
        corr_trial = jnp.where(finite, _correlation(x_trial), jnp.asarray(-jnp.inf, x0.dtype))
        improved = corr_trial > corr
        x_new = jnp.where(improved, x_trial, x)
        corr_new = jnp.where(improved, corr_trial, corr)
        lam_new = jnp.where(improved, lam * lm_down, lam * lm_up)
        lam_new = jnp.clip(lam_new, eps, 1e12)
        return (x_new, lam_new, corr_new), None

    corr0 = _correlation(x0)
    lam0 = jnp.asarray(lm_init, dtype=x0.dtype)
    (x_final, _, corr_final), _ = jax.lax.scan(
        _step, (x0, lam0, corr0), None, length=int(max_steps)
    )

    t_k, ne, conc = unpack_polish_params(x_final)
    return PolishResult(
        params=x_final,
        temperature_k=t_k,
        electron_density=ne,
        concentrations=conc,
        correlation=corr_final,
        init_correlation=corr0,
        n_steps=jnp.asarray(int(max_steps), dtype=jnp.int32),
    )


def polish_candidates(
    x0_batch: Any,
    measured: Any,
    snapshot: "PipelineSnapshot",
    instrument: "InstrumentModel",
    wavelengths_nm: Any,
    *,
    element_symbols: tuple[str, ...],
    broadening_mode: BroadeningMode = BroadeningMode.NIST_PARITY,
    path_length_m: float = 0.01,
    apply_self_absorption: bool = False,
    apply_stark: bool = False,
    max_steps: int = 6,
    lm_init: float = 1e-2,
    weights: Any = None,
) -> PolishResult:
    """``vmap`` :func:`gauss_newton_polish` over a fixed batch of start vectors.

    The candidate axis is fixed-shape ``(M, P)`` — the surviving-candidate count
    ``M`` is a *static* selection (the host keeps the top-``M`` by coarse
    correlation, padding to ``M`` when fewer survive), so a single compiled graph
    polishes every spectrum. Returns a :class:`PolishResult` whose leaves carry a
    leading ``(M, ...)`` axis.

    Parameters
    ----------
    x0_batch : array, shape (M, P)
        Stacked start vectors (one per surviving candidate).
    measured, snapshot, instrument, wavelengths_nm, element_symbols :
        As :func:`gauss_newton_polish`.
    broadening_mode, path_length_m, apply_self_absorption, apply_stark,
    max_steps, lm_init, weights :
        Polish knobs (shared across the batch).

    Returns
    -------
    PolishResult
        Leaves shaped ``(M, ...)``.
    """

    def _one(x0: Any) -> PolishResult:
        return gauss_newton_polish(
            x0,
            measured,
            snapshot,
            instrument,
            wavelengths_nm,
            element_symbols=element_symbols,
            broadening_mode=broadening_mode,
            path_length_m=path_length_m,
            apply_self_absorption=apply_self_absorption,
            apply_stark=apply_stark,
            max_steps=max_steps,
            lm_init=lm_init,
            weights=weights,
        )

    return jax.vmap(_one)(jnp.asarray(x0_batch))


# ---------------------------------------------------------------------------
# Host-side candidate-population assembly (the per-spectrum host<->device seam).
# ---------------------------------------------------------------------------


def _ilr_simplex_samples(key: Any, n: int, e: int) -> Any:
    """``n`` Dirichlet-like draws on the ``e``-element simplex via softmax-of-normal.

    A counter-based, fixed-shape composition sampler that stays on the simplex
    interior (positive, sum-to-one) — the ILR-coordinate spirit of J10 spec §2
    without importing any banned softmax (this is a plain exp/normalize). Fixed
    output shape ``(n, e)`` regardless of values.
    """
    z = jax.random.normal(key, (n, e))
    z = z - jnp.max(z, axis=1, keepdims=True)
    w = jnp.exp(z)
    return w / jnp.sum(w, axis=1, keepdims=True)


def build_candidate_population(
    key: Any,
    element_symbols: tuple[str, ...],
    *,
    n_configs: int,
    t_range_k: tuple[float, float] = (6000.0, 14000.0),
    log10_ne_range: tuple[float, float] = (16.0, 18.0),
    subset_keep_prob: float = 0.5,
    min_concentration: float = 1e-3,
) -> dict[str, Any]:
    """Stratified, fixed-shape candidate population over (T, n_e, C, subset mask).

    Counter-based ``jax.random`` (deterministic per key — J10 spec AC4), fixed
    population shape ``(n_configs, ...)`` with no data-dependent control flow.
    Each config draws a temperature, an electron density, a full-simplex
    composition over the element superset, and an element-subset mask (each
    element kept with probability ``subset_keep_prob``). The effective
    composition is the full-simplex draw times the subset mask, renormalized;
    membership marks elements whose effective concentration clears
    ``min_concentration``.

    Returns a dict (not a device call): this is the host seam. The caller stacks
    the per-config compositions into a batched plasma (:func:`stack_plasma_states`)
    and feeds :func:`evaluate_population`.

    Parameters
    ----------
    key : jax.random.PRNGKey
    element_symbols : tuple of str
        The padded element superset (snapshot ``element_symbols``); ``E`` =
        ``len(element_symbols)``.
    n_configs : int
        Population size ``B_eval`` (fixed; keys the jit cache via batch shape).
    t_range_k, log10_ne_range : tuple of float
        Stratification ranges (J10 spec §2 "preset T/n_e ranges").
    subset_keep_prob : float
        Per-element Bernoulli keep probability for the subset mask.
    min_concentration : float
        Membership threshold on the renormalized effective concentration.

    Returns
    -------
    dict with keys
        ``temperatures_k`` (n_configs,), ``electron_densities`` (n_configs,),
        ``concentrations`` (n_configs, E), ``membership`` (n_configs, E),
        ``config_valid`` (n_configs,), ``n_params`` (n_configs,).
    """
    e = len(element_symbols)
    k_t, k_ne, k_c, k_sub = jax.random.split(key, 4)

    t_lo, t_hi = t_range_k
    temperatures = jax.random.uniform(k_t, (n_configs,), minval=t_lo, maxval=t_hi)

    ne_lo, ne_hi = log10_ne_range
    log_ne = jax.random.uniform(k_ne, (n_configs,), minval=ne_lo, maxval=ne_hi)
    electron_densities = jnp.power(10.0, log_ne)

    comp_full = _ilr_simplex_samples(k_c, n_configs, max(e, 1))
    subset = (jax.random.uniform(k_sub, (n_configs, max(e, 1))) < subset_keep_prob).astype(
        comp_full.dtype
    )
    # Guarantee every config keeps at least one element (else zero spectrum):
    # if a row's subset is all-zero, fall back to keeping the argmax element.
    row_sum = jnp.sum(subset, axis=1, keepdims=True)
    am = jnp.argmax(comp_full, axis=1)
    one_hot = (jnp.arange(max(e, 1))[None, :] == am[:, None]).astype(comp_full.dtype)
    subset = jnp.where(row_sum > 0, subset, one_hot)

    eff = comp_full * subset
    eff = eff / jnp.maximum(jnp.sum(eff, axis=1, keepdims=True), 1e-300)
    membership = (eff >= min_concentration).astype(comp_full.dtype)
    n_params = jnp.sum(membership, axis=1) + 2.0  # +T, +n_e

    return {
        "temperatures_k": temperatures,
        "electron_densities": electron_densities,
        "concentrations": eff,
        "membership": membership,
        "config_valid": jnp.ones((n_configs,), dtype=comp_full.dtype),
        "n_params": n_params,
    }


def stack_plasma_states(
    temperatures_k: Any,
    electron_densities: Any,
    concentrations: Any,
    element_symbols: tuple[str, ...],
) -> "SingleZoneLTEPlasma":
    """Stack per-config plasma states into one batched plasma pytree (host seam).

    Builds one :class:`~cflibs.plasma.state.SingleZoneLTEPlasma` per config (cheap
    dict construction — the documented per-spectrum host<->device seam) and
    ``tree_map(jnp.stack)`` them into a single batched pytree whose every leaf has
    a leading ``B_eval`` axis, ready for ``vmap(forward_model, in_axes=(0, ...))``.

    The species dict carries *every* element in ``element_symbols`` (the forward
    kernel iterates the snapshot element set and reads ``species[el]`` for each),
    so excluded elements simply carry concentration ~0.

    Parameters
    ----------
    temperatures_k : array, shape (B_eval,)
        Temperatures, K.
    electron_densities : array, shape (B_eval,)
        Electron densities, cm^-3.
    concentrations : array, shape (B_eval, E)
        Per-config number fractions over ``element_symbols``.
    element_symbols : tuple of str
        Element superset; column order of ``concentrations``.

    Returns
    -------
    SingleZoneLTEPlasma
        Batched plasma pytree (leading axis ``B_eval``).
    """
    from cflibs.plasma.state import SingleZoneLTEPlasma

    t = np.asarray(temperatures_k, dtype=float)
    ne = np.asarray(electron_densities, dtype=float)
    comp = np.asarray(concentrations, dtype=float)
    b = t.shape[0]

    states = []
    for i in range(b):
        species = {el: float(comp[i, j]) for j, el in enumerate(element_symbols)}
        states.append(SingleZoneLTEPlasma(T_e=float(t[i]), n_e=float(ne[i]), species=species))
    return jax.tree_util.tree_map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *states)


# ---------------------------------------------------------------------------
# Orchestrator — host builds the population + batched plasma, device scores it.
# ---------------------------------------------------------------------------


def forward_fit_identify(
    measured_spectrum: Any,
    wavelengths_nm: Any,
    snapshot: "PipelineSnapshot",
    instrument: "InstrumentModel",
    params: "PipelineParams | None" = None,
    static: "StaticConfig | None" = None,
    *,
    key: Any = None,
    n_configs: int = 1024,
    presence_threshold: float = 0.05,
    t_range_k: tuple[float, float] = (6000.0, 14000.0),
    log10_ne_range: tuple[float, float] = (16.0, 18.0),
    subset_keep_prob: float = 0.5,
    element_weights: Any = None,
    polish_steps: int = 0,
    top_m_polish: int = 8,
    polish_lm_init: float = 1e-2,
    require_bic: bool = False,
    bic_margin: float = 0.0,
) -> ForwardFitResult:
    """Run a full population forward-fit identification pass for one spectrum.

    Host: build a fixed-shape candidate population over the snapshot element
    superset (:func:`build_candidate_population`) and stack the candidate plasma
    states (:func:`stack_plasma_states`). Device: forward-model the population
    (:func:`evaluate_population`), score it (:func:`correlation_cost`,
    :func:`bic_cost`), and call elements (:func:`forward_fit_presence_scores`).

    Optional gradient polish (J10 spec §1 / ADR-0004 §6.1): when
    ``polish_steps > 0``, the top-``top_m_polish`` candidates by coarse correlation
    are refined with a fixed-K Levenberg-Marquardt descent on the shape-correlation
    residual (:func:`polish_candidates`), and their *polished* correlations replace
    the coarse ones before the present/absent call — this is the refinement the
    MC-CF prior art lacks and is the binding recall lever (J10 §3 item 1).

    Determinism: identical ``(key, snapshot, n_configs, ranges)`` give an
    identical result (counter-based RNG; J10 spec AC4). ``key=None`` uses a fixed
    default seed so the call is reproducible by default.

    Parameters
    ----------
    measured_spectrum : array, shape (N_wl,)
        Measured (preprocessed) spectrum.
    wavelengths_nm : array, shape (N_wl,)
        Wavelength grid, nm.
    snapshot : PipelineSnapshot
        Unified atomic snapshot.
    instrument : InstrumentModel
        Instrument model.
    params : PipelineParams, optional
        Traced knobs. Currently unused by this stage (forward-fit has its own
        population knobs); accepted for signature parity with the other stages.
    static : StaticConfig, optional
        Static config. ``broadening_mode`` / ``apply_self_absorption`` are read
        when present; otherwise sensible defaults are used.
    key : jax.random.PRNGKey, optional
        RNG key. ``None`` => ``jax.random.PRNGKey(0)`` (reproducible default).
    n_configs : int
        Population size ``B_eval``.
    presence_threshold : float
        Include-minus-exclude correlation gap to call an element present.
    t_range_k, log10_ne_range, subset_keep_prob :
        Population stratification knobs (see :func:`build_candidate_population`).
    element_weights : array, shape (N_wl,), optional
        Per-wavelength correlation weights (element-emphasis / validity mask).
    polish_steps : int
        Fixed LM iteration count for the gradient polish. ``0`` (default) skips
        polishing entirely (pure population forward-fit). Static (keys the graph).
    top_m_polish : int
        Number of top-correlation candidates to polish (fixed-shape selection).
        Clamped to the population size.
    polish_lm_init : float
        Initial LM damping for the polish (see :func:`gauss_newton_polish`).
    require_bic : bool
        When ``True``, gate presence on a BIC-improvement margin in addition to
        the correlation gap (see :func:`forward_fit_presence_scores`). Default
        ``False`` is bit-identical to the pure correlation-gap call.
    bic_margin : float
        Minimum BIC improvement required when ``require_bic`` is set.

    Returns
    -------
    ForwardFitResult
        ``element_present`` indexes ``snapshot.element_symbols``. When polishing is
        active, ``best_correlation`` / ``best_config_index`` reflect the polished
        improvement on the refined candidates.
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    if static is not None and getattr(static, "broadening_mode", None) is not None:
        broadening_mode = _resolve_broadening_mode(static.broadening_mode)
        apply_self_absorption = bool(getattr(static, "apply_self_absorption", False))
    else:
        broadening_mode = BroadeningMode.NIST_PARITY
        apply_self_absorption = False

    element_symbols = tuple(snapshot.element_symbols)

    pop = build_candidate_population(
        key,
        element_symbols,
        n_configs=n_configs,
        t_range_k=t_range_k,
        log10_ne_range=log10_ne_range,
        subset_keep_prob=subset_keep_prob,
    )
    batched_plasma = stack_plasma_states(
        pop["temperatures_k"],
        pop["electron_densities"],
        pop["concentrations"],
        element_symbols,
    )

    spectra = evaluate_population(
        batched_plasma,
        snapshot,
        instrument,
        wavelengths_nm,
        broadening_mode=broadening_mode,
        apply_self_absorption=apply_self_absorption,
    )
    corr = correlation_cost(spectra, measured_spectrum, weights=element_weights)
    bic = bic_cost(spectra, measured_spectrum, n_params=pop["n_params"])

    if polish_steps and polish_steps > 0:
        corr, bic = _polish_population_scores(
            corr,
            bic,
            pop,
            measured_spectrum,
            wavelengths_nm,
            snapshot,
            instrument,
            element_symbols,
            broadening_mode=broadening_mode,
            apply_self_absorption=apply_self_absorption,
            element_weights=element_weights,
            polish_steps=int(polish_steps),
            top_m_polish=int(top_m_polish),
            polish_lm_init=float(polish_lm_init),
        )

    element_valid = jnp.ones((len(element_symbols),), dtype=spectra.dtype)
    return forward_fit_presence_scores(
        corr,
        bic,
        pop["membership"],
        pop["config_valid"],
        element_valid,
        presence_threshold=presence_threshold,
        require_bic=require_bic,
        bic_margin=bic_margin,
    )


def _polish_population_scores(
    corr: Any,
    bic: Any,
    pop: dict[str, Any],
    measured_spectrum: Any,
    wavelengths_nm: Any,
    snapshot: "PipelineSnapshot",
    instrument: "InstrumentModel",
    element_symbols: tuple[str, ...],
    *,
    broadening_mode: BroadeningMode,
    apply_self_absorption: bool,
    element_weights: Any,
    polish_steps: int,
    top_m_polish: int,
    polish_lm_init: float,
) -> tuple[Any, Any]:
    """Refine the top-``M`` candidates by LM polish; scatter improved scores back.

    Fixed-shape: ``M = min(top_m_polish, n_configs)`` is a *static* int, so the
    polished-candidate batch ``(M, P)`` and the scatter are a single compiled
    graph per spectrum. Polishing can only *raise* a candidate's correlation (LM
    accepts steps only when correlation strictly improves), so the scatter uses
    ``max`` and is order-independent. BIC of polished candidates is recomputed from
    the polished spectra so the secondary model-selection score stays consistent.
    """
    n_configs = int(corr.shape[0])
    m = min(int(top_m_polish), n_configs)

    # Static top-M selection by coarse correlation (host-side argsort on values).
    top_idx = jnp.argsort(corr)[::-1][:m]  # (M,)

    t_top = jnp.asarray(pop["temperatures_k"])[top_idx]
    ne_top = jnp.asarray(pop["electron_densities"])[top_idx]
    conc_top = jnp.asarray(pop["concentrations"])[top_idx]

    x0_batch = jax.vmap(pack_polish_params)(t_top, ne_top, conc_top)  # (M, P)

    polished = polish_candidates(
        x0_batch,
        measured_spectrum,
        snapshot,
        instrument,
        wavelengths_nm,
        element_symbols=element_symbols,
        broadening_mode=broadening_mode,
        apply_self_absorption=apply_self_absorption,
        max_steps=polish_steps,
        lm_init=polish_lm_init,
        weights=element_weights,
    )
    polished_corr = polished.correlation  # (M,)

    # Recompute BIC of the polished candidates from their refined spectra so the
    # secondary score reflects the refinement (uses the same n_params bookkeeping).
    # Re-forward at the *polished* (T, n_e, C) — the full refined config.
    polished_plasma = stack_plasma_states(
        np.asarray(polished.temperature_k),
        np.asarray(polished.electron_density),
        np.asarray(polished.concentrations),
        element_symbols,
    )
    polished_spectra = evaluate_population(
        polished_plasma,
        snapshot,
        instrument,
        wavelengths_nm,
        broadening_mode=broadening_mode,
        apply_self_absorption=apply_self_absorption,
    )
    n_params_top = jnp.asarray(pop["n_params"])[top_idx]
    polished_bic = bic_cost(polished_spectra, measured_spectrum, n_params=n_params_top)

    # Scatter the improvements back: correlation takes the max (polish never
    # regresses); BIC takes the min (lower is better) so the polished, better-fit
    # config wins the secondary score too.
    corr_new = corr.at[top_idx].max(polished_corr)
    bic_new = bic.at[top_idx].min(polished_bic)
    return corr_new, bic_new
