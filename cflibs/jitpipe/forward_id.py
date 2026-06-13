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
) -> ForwardFitResult:
    """Marginal-evidence element calls from a scored candidate population.

    The leave-one-element-out contrast that *is* the multi-line coherence test
    (audit F3): for each element ``e`` we compare the best correlation achieved
    by configs that *include* ``e`` against the best achieved by configs that
    *exclude* ``e``. A large positive gap means the data demand ``e`` — its lines
    cannot be coherently explained without it. The discrete call is gated on this
    gap (``presence_threshold``); BIC of the best including-config is returned as
    a secondary, model-comparison-grounded score.

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

    # An element with no including config has best_inc = -inf; the gap is then
    # -inf and it is (correctly) not called. Elements never excluded (best_exc
    # = -inf) get a +inf gap => strongly called, which is right: every viable
    # config needs them.
    gap = best_inc - best_exc  # (E,)
    present = (gap > presence_threshold) & (el_ok > 0) & jnp.isfinite(best_inc)

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
) -> ForwardFitResult:
    """Run a full population forward-fit identification pass for one spectrum.

    Host: build a fixed-shape candidate population over the snapshot element
    superset (:func:`build_candidate_population`) and stack the candidate plasma
    states (:func:`stack_plasma_states`). Device: forward-model the population
    (:func:`evaluate_population`), score it (:func:`correlation_cost`,
    :func:`bic_cost`), and call elements (:func:`forward_fit_presence_scores`).

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

    Returns
    -------
    ForwardFitResult
        ``element_present`` indexes ``snapshot.element_symbols``.
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

    element_valid = jnp.ones((len(element_symbols),), dtype=spectra.dtype)
    return forward_fit_presence_scores(
        corr,
        bic,
        pop["membership"],
        pop["config_valid"],
        element_valid,
        presence_threshold=presence_threshold,
    )
