"""Stage 6a — jittable observable-gated self-absorption (J5; ADR-0004 §4 row 7).

Fixed-shape, ``vmap``-clean JAX port of
:meth:`cflibs.inversion.physics.self_absorption_observable.ObservableSelfAbsorptionCorrector.correct`
restricted to the two *observable-only* ladder steps that run pre-fit as a pure
``(L,)`` array transform: the **doublet intensity-ratio** correction (ladder
step (a)) and the **SA-suspect down-weighting** pass (ladder step (c)). The
Planck-ceiling step (b) is omitted here because it requires per-line absolute
peak spectral radiance + a temperature estimate, neither of which is part of
the pre-fit pure array transform (see ``remaining_todo`` in the J5 verdict).

Design (ADR-0004 §1.1, J5 spec §3)
----------------------------------
* **No data-dependent shapes.** Every array is padded to a fixed bucket and
  carries a validity mask; failures are quality flags, never exceptions. The
  pathological 7394-cache-entry behaviour of the host root-finder is avoided.
* **Root solve -> fixed-K bisection.** ``scipy.optimize.brentq`` on
  ``f(tau_1)/f(tau_1/rho) = r_meas/r_thin`` over the pre-bracketed
  ``[1e-4, 30]`` becomes :data:`BISECTION_STEPS` branchless bisection steps,
  ``vmap``-ped over all pairs at once. The same-sign-endpoints "observably
  thin" test of the reference (``self_absorption.py:421``) is reproduced as a
  mask returning ``tau = 1e-4``.
* **First-usable-pair-wins -> deterministic claim resolution.** The reference
  iterates pairs in ``find_doublet_pairs`` order and skips a pair if either
  line is already corrected/cleared (``self_absorption_observable.py:480``).
  Here each line is claimed by the *lowest-priority-index usable pair* that
  touches it (scatter-min over the pair->line incidence). Whenever every line
  appears in at most one *usable* pair (the common case) this reproduces the
  sequential semantics exactly; the multi-claim contract is the documented,
  order-free scatter-min rule (J5 spec §3, acceptance §5.2).

No SQLite, no ``cflibs.jitpipe.host`` import (import-hygiene test); JAX only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

from cflibs.core.constants import C_LIGHT, H_PLANCK_EV
from cflibs.core.jax_runtime import HAS_JAX

if HAS_JAX:
    import jax.numpy as jnp
else:  # pragma: no cover - jitpipe requires JAX (see cflibs.jitpipe.__init__)
    raise ImportError("cflibs.jitpipe.selfabs requires JAX")

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


# ---------------------------------------------------------------------------
# Constants — pinned to the reference oracle (do not retune; parity contract).
# ---------------------------------------------------------------------------

#: Fixed bisection steps replacing brentq's adaptive bracketing. The residual
#: is smooth and pre-bracketed on [1e-4, 30]; ``ceil(log2((30-1e-4)/1e-6)) =
#: 25`` steps already beat brentq's ``xtol=1e-6``. 60 steps drive the bracket
#: to ~3e-17 (well below f64 round-off), so the recovered tau matches brentq to
#: the §4 ``atol 1e-6`` with margin.
BISECTION_STEPS: int = 60

#: Root-solve bracket (reference ``self_absorption.py:412``).
TAU_LOW: float = 1.0e-4
TAU_HIGH: float = 30.0

#: Doublet "observably thin" ceiling on max(tau_1, tau_2)
#: (``self_absorption_observable.py:493``).
THIN_TAU_CEIL: float = 1.0e-3

#: Doublet validity ceiling; pairs implying a larger tau are force-suspect,
#: never boosted (``DOUBLET_TAU_VALIDITY_MAX`` = 5.0).
DOUBLET_TAU_VALIDITY_MAX: float = 5.0

#: Absolute floor on the fractional ratio deviation ``|1 - r_meas/r_thin|``
#: (``min_ratio_deviation`` default, ``self_absorption_observable.py:337``).
MIN_RATIO_DEVIATION: float = 0.10

#: Significance threshold in propagated sigmas (``min_ratio_significance_sigma``
#: default, ``self_absorption_observable.py:336``).
MIN_RATIO_SIGNIFICANCE_SIGMA: float = 1.0

#: Fayyaz 2023 SA-risk lower-level energy cut (``SUSPECT_E_I_MAX_EV``).
SUSPECT_E_I_MAX_EV: float = 0.74

#: Bright-line factor for the suspect signature (``suspect_intensity_factor``).
SUSPECT_INTENSITY_FACTOR: float = 1.0

#: Suspect uncertainty inflation factor (``suspect_uncertainty_inflation``).
SUSPECT_UNCERTAINTY_INFLATION: float = 3.0

#: ``hc`` in eV*nm — photon energy ``E = HC_EV_NM / lambda_nm`` (matches
#: ``self_absorption_inputs.lower_level_energy_ev``: ``H_PLANCK_EV * C_LIGHT /
#: (lambda_nm * 1e-9)`` with C_LIGHT in m/s).
HC_EV_NM: float = float(H_PLANCK_EV * C_LIGHT * 1.0e9)

#: Per-line method codes (host-side warning strings decode these, J5 spec §3).
METHOD_NONE: int = 0
METHOD_DOUBLET: int = 1
METHOD_DOUBLET_THIN: int = 2
METHOD_SUSPECT: int = 3


class SelfAbsorptionResult(NamedTuple):
    """Fixed-shape output of :func:`correct_self_absorption_arrays`.

    All per-line arrays share the padded line axis ``(L,)`` (or ``(B, L)`` when
    vmapped). Invalid (padded) positions carry pass-through / zero values and
    are excluded from the masked counters.

    Attributes
    ----------
    intensity : array, shape (..., L)
        Self-absorption-corrected integrated intensities. Equal to the input
        for cleared / suspect / untouched lines.
    intensity_unc : array, shape (..., L)
        Updated intensity uncertainties: ``sigma * factor`` for doublet-
        corrected lines, ``sigma * inflation`` (or ``intensity * inflation``
        when ``sigma == 0``) for suspects, unchanged otherwise.
    tau : array, shape (..., L)
        Per-line recovered line-center optical depth (``nan`` where undefined,
        matching the reference suspect record).
    method : array, shape (..., L), int32
        Per-line method code (:data:`METHOD_NONE` / ``_DOUBLET`` /
        ``_DOUBLET_THIN`` / ``_SUSPECT``).
    suspect : array, shape (..., L), bool
        ``True`` where the line was flagged SA-suspect (down-weighted).
    n_corrected : array, shape (...)
        Count of doublet-corrected lines (``factor > 1``, not suspect) —
        matches ``ObservableSAResult.n_corrected``.
    n_suspect : array, shape (...)
        Count of down-weighted suspect lines.
    max_tau : array, shape (...)
        Maximum finite recovered tau across all pairs/lines (0 when none).
    """

    intensity: Any
    intensity_unc: Any
    tau: Any
    method: Any
    suspect: Any
    n_corrected: Any
    n_suspect: Any
    max_tau: Any


# ---------------------------------------------------------------------------
# Core scalar physics (vmap-clean: pure jnp, branchless).
# ---------------------------------------------------------------------------


def escape_factor(tau: Any) -> Any:
    """Photon escape factor ``f(tau) = (1 - exp(-tau)) / tau``, branchless.

    Reproduces :func:`cflibs.inversion.physics.self_absorption._escape_factor`
    (which special-cases ``tau < 1e-10 -> 1`` and ``tau > 50 -> 1/tau``) with a
    single smooth expression. ``expm1`` keeps the small-tau limit ``f -> 1``
    accurate to round-off, so the host's ``tau < 1e-10`` branch is matched
    without a ``where``; ``tau`` is clamped to a tiny positive floor so the
    division is finite under ``jax.grad`` even at ``tau == 0``.
    """
    tau_safe = jnp.maximum(tau, 1.0e-300)
    return -jnp.expm1(-tau_safe) / tau_safe


def _escape_ref(tau: float) -> float:
    """Scalar reference escape factor (host branchy form) for property tests.

    Bit-for-bit mirror of
    :func:`cflibs.inversion.physics.self_absorption._escape_factor`, kept here
    so the parity test can bracket/compare against brentq without importing the
    SQLite-adjacent reference module's private helper. Not used by the kernel.
    """
    import math

    if tau < 1e-10:
        return 1.0
    if tau > 50:
        return 1.0 / tau
    return (1.0 - math.exp(-tau)) / tau


def _residual(tau_1: Any, rho: Any, ratio_of_ratios: Any) -> Any:
    """Reference doublet residual ``f(tau_1)/f(tau_1/rho) - r_meas/r_thin``."""
    f1 = escape_factor(tau_1)
    f2 = escape_factor(tau_1 / rho)
    return f1 / f2 - ratio_of_ratios


def _bisect_tau(rho: Any, ratio_of_ratios: Any) -> Any:
    """Fixed-:data:`BISECTION_STEPS` bisection for ``tau_1`` on ``[1e-4, 30]``.

    Branchless ``vmap``-clean replacement for ``scipy.optimize.brentq``. Assumes
    a sign change across the bracket (the caller's ``same_sign`` mask handles
    the no-root "observably thin" case and overrides this output with
    ``tau = 1e-4``). The midpoint update is the standard bisection invariant:
    move the endpoint whose residual shares the midpoint residual's sign.
    """
    lo = jnp.full_like(rho, TAU_LOW)
    hi = jnp.full_like(rho, TAU_HIGH)
    f_lo = _residual(lo, rho, ratio_of_ratios)

    def body(carry, _):
        lo, hi, f_lo = carry
        mid = 0.5 * (lo + hi)
        f_mid = _residual(mid, rho, ratio_of_ratios)
        same_side = (f_mid * f_lo) > 0.0
        lo = jnp.where(same_side, mid, lo)
        hi = jnp.where(same_side, hi, mid)
        f_lo = jnp.where(same_side, f_mid, f_lo)
        return (lo, hi, f_lo), None

    import jax

    (lo, hi, _), _ = jax.lax.scan(body, (lo, hi, f_lo), None, length=BISECTION_STEPS)
    return 0.5 * (lo + hi)


def solve_doublet_tau(rho: Any, r_meas: Any, r_thin: Any) -> Any:
    """Recover ``tau_1`` for a batch of doublets (reference root + thin branch).

    Mirrors ``correct_via_doublet_ratio`` (``self_absorption.py:406-448``):
    compute the residual bracket endpoints, take the ``tau = 1e-4`` "observably
    thin" branch when the endpoints share a sign (no root) or
    ``|r_meas/r_thin - 1| < 1e-12``, otherwise the bisection root. Fully
    branchless via masks.

    Parameters
    ----------
    rho : array, shape (P,)
        Optical-depth ratio ``tau_1 / tau_2`` per pair.
    r_meas, r_thin : array, shape (P,)
        Measured and optically-thin emission ratios per pair.

    Returns
    -------
    tau_1 : array, shape (P,)
    """
    ratio_of_ratios = r_meas / r_thin
    res_low = _residual(jnp.full_like(rho, TAU_LOW), rho, ratio_of_ratios)
    res_high = _residual(jnp.full_like(rho, TAU_HIGH), rho, ratio_of_ratios)
    same_sign = (res_low * res_high) > 0.0
    no_dev = jnp.abs(ratio_of_ratios - 1.0) < 1.0e-12
    thin = jnp.logical_or(same_sign, no_dev)
    tau_root = _bisect_tau(rho, ratio_of_ratios)
    return jnp.where(thin, jnp.full_like(rho, TAU_LOW), tau_root)


# ---------------------------------------------------------------------------
# Per-line atomic gathers.
# ---------------------------------------------------------------------------


def _gather_line_atomics(
    line_index: Any, snapshot: "PipelineSnapshot"
) -> tuple[Any, Any, Any, Any, Any]:
    """Gather ``(lambda_nm, A_ki, g_k, E_k_ev, species_index)`` per padded line.

    ``line_index`` indexes the superset snapshot catalog. Padding entries (any
    negative index) clamp to 0 and are excluded by the validity mask the caller
    carries; the gathered values are only consumed where ``line_valid`` holds.
    """
    idx = jnp.maximum(jnp.asarray(line_index), 0)
    wl = jnp.asarray(snapshot.line_wavelength_nm)[idx]
    aki = jnp.asarray(snapshot.line_A_ki)[idx]
    gk = jnp.asarray(snapshot.line_g_k)[idx]
    ek = jnp.asarray(snapshot.line_E_k_ev)[idx]
    sp = jnp.asarray(snapshot.line_species_index)[idx]
    return wl, aki, gk, ek, sp


def _lower_level_energy_ev(wl_nm: Any, e_k_ev: Any) -> Any:
    """``E_i = max(0, E_k - hc/lambda)`` (reference ``lower_level_energy_ev``)."""
    photon = HC_EV_NM / wl_nm
    return jnp.maximum(0.0, e_k_ev - photon)


# ---------------------------------------------------------------------------
# Public kernel.
# ---------------------------------------------------------------------------


def correct_self_absorption_arrays(
    intensity: Any,
    intensity_unc: Any,
    aki_unc: Any,
    line_index: Any,
    line_valid: Any,
    line_element: Any,
    pair_idx: Any,
    pair_valid: Any,
    snapshot: "PipelineSnapshot",
    *,
    doublet_tau_max: float = DOUBLET_TAU_VALIDITY_MAX,
    min_ratio_deviation: float = MIN_RATIO_DEVIATION,
    min_ratio_significance_sigma: float = MIN_RATIO_SIGNIFICANCE_SIGMA,
    suspect_e_i_max_ev: float = SUSPECT_E_I_MAX_EV,
    suspect_intensity_factor: float = SUSPECT_INTENSITY_FACTOR,
    suspect_uncertainty_inflation: float = SUSPECT_UNCERTAINTY_INFLATION,
) -> SelfAbsorptionResult:
    """Jittable observable-gated self-absorption correction (doublet + suspect).

    Fixed-shape, ``vmap``/``grad``-clean. Reproduces the doublet pass (a) and
    the suspect pass (c) of
    :meth:`ObservableSelfAbsorptionCorrector.correct` to the J5 §4 tolerance.

    Parameters
    ----------
    intensity, intensity_unc, aki_unc : array, shape (L,)
        Per padded line: measured integrated intensity, its absolute
        uncertainty, and the *fractional* A_ki uncertainty (0 when unknown,
        mirroring ``aki_uncertainty or 0.0`` in the reference).
    line_index : array, shape (L,), int
        Catalog index of each padded line into ``snapshot`` (negative = pad).
    line_valid : array, shape (L,), bool
        ``True`` for real lines, ``False`` for padding.
    line_element : array, shape (L,), int
        Per-line element-group id for the per-element suspect median (any
        stable integer labelling; padding entries are ignored via the mask).
    pair_idx : array, shape (P, 2), int
        Local padded line indices of each candidate doublet (both members lie
        on the ``(L,)`` axis). Order within a pair is irrelevant — the kernel
        re-orders to shorter-wavelength-first to match the reference.
    pair_valid : array, shape (P,), bool
        ``True`` for real candidate pairs, ``False`` for padding.
    snapshot : PipelineSnapshot
        Atomic-data snapshot (per-line ``g_k`` / ``A_ki`` / ``wavelength_nm`` /
        ``E_k_ev``).
    doublet_tau_max, min_ratio_deviation, min_ratio_significance_sigma, \
suspect_e_i_max_ev, suspect_intensity_factor, suspect_uncertainty_inflation : float
        Continuous knobs; defaults mirror the reference corrector.

    Returns
    -------
    SelfAbsorptionResult
    """
    intensity = jnp.asarray(intensity)
    intensity_unc = jnp.asarray(intensity_unc)
    aki_unc = jnp.asarray(aki_unc)
    line_valid = jnp.asarray(line_valid).astype(bool)
    line_element = jnp.asarray(line_element)
    pair_idx = jnp.asarray(pair_idx)
    pair_valid = jnp.asarray(pair_valid).astype(bool)

    n_lines = intensity.shape[0]

    wl, aki, gk, ek, _sp = _gather_line_atomics(line_index, snapshot)

    # --- (a) doublet pass: per-pair physics ------------------------------
    a = pair_idx[:, 0]
    b = pair_idx[:, 1]
    # Reference orders the shorter-wavelength line as line1; swap when needed.
    swap = wl[b] < wl[a]
    i1 = jnp.where(swap, b, a)
    i2 = jnp.where(swap, a, b)

    wl1, wl2 = wl[i1], wl[i2]
    aki1, aki2 = aki[i1], aki[i2]
    gk1, gk2 = gk[i1], gk[i2]
    I1, I2 = intensity[i1], intensity[i2]
    sig1, sig2 = intensity_unc[i1], intensity_unc[i2]
    aki_u1, aki_u2 = aki_unc[i1], aki_unc[i2]

    # r_thin = (g_k1 A_1 / lambda_1) / (g_k2 A_2 / lambda_2)
    r_thin = (gk1 * aki1 / wl1) / (gk2 * aki2 / wl2)
    # rho = (g_k1 A_1 lambda_1^3) / (g_k2 A_2 lambda_2^3)
    rho = (gk1 * aki1 * wl1**3) / (gk2 * aki2 * wl2**3)
    r_meas = I1 / I2
    ratio_of_ratios = r_meas / r_thin

    tau1 = solve_doublet_tau(rho, r_meas, r_thin)
    tau2 = tau1 / rho
    f1 = escape_factor(tau1)
    f2 = escape_factor(tau2)
    tau_pair_max = jnp.maximum(tau1, tau2)

    # Significance gate (_ratio_deviation_significant): two required gates.
    deviation = jnp.abs(1.0 - ratio_of_ratios)
    rel1 = sig1 / jnp.where(I1 > 0, I1, 1.0)
    rel2 = sig2 / jnp.where(I2 > 0, I2, 1.0)
    sigma_rel = jnp.sqrt(rel1**2 + rel2**2 + aki_u1**2 + aki_u2**2)
    floor_ok = deviation >= min_ratio_deviation
    # When sigma_rel <= 0 the reference returns True (floor already passed);
    # else require deviation/sigma_rel >= threshold.
    sig_ok = jnp.where(
        sigma_rel <= 0.0,
        True,
        deviation >= min_ratio_significance_sigma * sigma_rel,
    )
    positive_I = jnp.logical_and(I1 > 0, I2 > 0)
    significant = jnp.logical_and(jnp.logical_and(floor_ok, sig_ok), positive_I)

    # Pair-level usability + classification (mirror the reference branch order).
    pair_usable = pair_valid  # both members real by construction; mask carries it
    thin_branch = jnp.logical_or(tau_pair_max <= THIN_TAU_CEIL, jnp.logical_not(significant))
    over_ceiling = jnp.logical_and(jnp.logical_not(thin_branch), tau_pair_max > doublet_tau_max)
    correct_branch = jnp.logical_and(jnp.logical_not(thin_branch), jnp.logical_not(over_ceiling))

    # --- claim resolution: lowest pair index wins each line --------------
    # Priority = pair position (the reference iterates find_doublet_pairs in
    # order; lower index = earlier = wins). Invalid pairs get +inf priority.
    #
    # The reference's first-usable-pair-wins skips a pair if EITHER endpoint is
    # already corrected/cleared by an earlier usable pair. We reproduce this
    # without sequential state: a pair "wins" a line iff it is the lowest-
    # priority usable pair touching that line, AND it wins BOTH its endpoints.
    # When a pair loses (because an earlier pair already claimed one of its
    # lines), it must not write anything — so each LINE gathers its outcome
    # from the single pair that claims it (a positional gather, dedup-safe),
    # never a scatter with duplicate indices (which is last-write-wins and would
    # let a losing pair clobber the winner).
    P = pair_idx.shape[0]
    prio = jnp.where(pair_usable, jnp.arange(P), P + jnp.arange(P) + 1)

    BIG = jnp.int32(P * 4 + 10)
    # scatter-min of priority onto each line via both pair endpoints.
    line_claim = jnp.full((n_lines,), BIG, dtype=jnp.int32)
    line_claim = line_claim.at[i1].min(prio.astype(jnp.int32))
    line_claim = line_claim.at[i2].min(prio.astype(jnp.int32))

    # A pair "wins" a line only if it is the claiming (min-priority) pair for it.
    win1 = jnp.logical_and(pair_usable, line_claim[i1] == prio)
    win2 = jnp.logical_and(pair_usable, line_claim[i2] == prio)
    pair_wins = jnp.logical_and(win1, win2)

    # Effective per-pair branch only fires for the winning pair of BOTH lines.
    eff_correct = jnp.logical_and(correct_branch, pair_wins)
    eff_thin = jnp.logical_and(thin_branch, pair_wins)
    eff_suspect_forced = jnp.logical_and(over_ceiling, pair_wins)

    factor1 = jnp.where(f1 > 0, 1.0 / f1, 1.0)
    factor2 = jnp.where(f2 > 0, 1.0 / f2, 1.0)
    i1_corr = I1 / f1
    i2_corr = I2 / f2
    tau1_thin = jnp.minimum(tau1, THIN_TAU_CEIL)
    tau2_thin = jnp.minimum(tau2, THIN_TAU_CEIL)

    # Per-WINNING-pair, per-endpoint outcome arrays, all length P. We then map
    # each line to its claiming pair + endpoint role and gather (no scatter
    # collisions). A pair that does not win contributes nothing.
    def _endpoint_outcome(corr_val, thin_val, forced_val):
        """Branch-select the per-pair value for one endpoint role."""
        return jnp.where(
            eff_correct,
            corr_val,
            jnp.where(eff_thin, thin_val, jnp.where(eff_suspect_forced, forced_val, jnp.nan)),
        )

    # Endpoint 1 (the shorter-wavelength reference line1).
    p_int1 = jnp.where(eff_correct, i1_corr, I1)  # only correct boosts intensity
    p_unc1 = jnp.where(eff_correct, sig1 * factor1, sig1)
    p_tau1 = _endpoint_outcome(tau1, tau1_thin, tau1)
    # Endpoint 2.
    p_int2 = jnp.where(eff_correct, i2_corr, I2)
    p_unc2 = jnp.where(eff_correct, sig2 * factor2, sig2)
    p_tau2 = _endpoint_outcome(tau2, tau2_thin, tau2)

    p_method = jnp.where(
        eff_correct,
        jnp.int32(METHOD_DOUBLET),
        jnp.where(
            eff_thin,
            jnp.int32(METHOD_DOUBLET_THIN),
            jnp.where(eff_suspect_forced, jnp.int32(METHOD_SUSPECT), jnp.int32(METHOD_NONE)),
        ),
    )
    p_cleared = jnp.logical_or(eff_correct, eff_thin)
    p_forced = eff_suspect_forced

    # For each line, the claiming pair index + whether it is endpoint 1 or 2.
    # We scatter the *pair index* (deduped via the claim min) so a line reads
    # back exactly one pair. Lines with no claim keep -1.
    claim_pair = jnp.full((n_lines,), -1, dtype=jnp.int32)
    # only WINNING pairs write their index; for the claimed endpoints the
    # min-priority pair is the unique winner, so .set has no real collision,
    # but we guard with where(pair_wins) and rely on the claim equality.
    wins_idx = jnp.where(pair_wins, jnp.arange(P, dtype=jnp.int32), jnp.int32(-1))
    claim_pair = claim_pair.at[i1].max(jnp.where(win1, wins_idx, jnp.int32(-1)))
    claim_pair = claim_pair.at[i2].max(jnp.where(win2, wins_idx, jnp.int32(-1)))

    has_claim = claim_pair >= 0
    cp = jnp.maximum(claim_pair, 0)
    # is this line the endpoint-1 of its claiming pair?
    is_ep1 = i1[cp] == jnp.arange(n_lines)

    def _gather_line(ep1_vals, ep2_vals, default):
        chosen = jnp.where(is_ep1, ep1_vals[cp], ep2_vals[cp])
        return jnp.where(has_claim, chosen, default)

    out_intensity = _gather_line(p_int1, p_int2, intensity)
    out_unc = _gather_line(p_unc1, p_unc2, intensity_unc)
    out_tau = _gather_line(p_tau1, p_tau2, jnp.full((n_lines,), jnp.nan))
    # method is a per-pair (not per-endpoint) classification.
    out_method = jnp.where(has_claim, p_method[cp], jnp.int32(METHOD_NONE))
    out_cleared = jnp.logical_and(has_claim, p_cleared[cp])
    out_forced = jnp.logical_and(has_claim, p_forced[cp])

    # --- (c) suspect pass ------------------------------------------------
    # Per-element median intensity over positive intensities (real lines only).
    median_int = _per_element_median(intensity, line_element, line_valid)

    e_i = _lower_level_energy_ev(wl, ek)
    low_ei = e_i <= suspect_e_i_max_ev
    bright = intensity >= suspect_intensity_factor * median_int
    sa_risk = jnp.logical_and(low_ei, bright)

    # Eligible for the suspect pass: real, not cleared, not already corrected.
    eligible = jnp.logical_and(line_valid, jnp.logical_not(out_cleared))
    # corrected lines are exactly those cleared by the correct branch; the
    # reference excludes both `cleared` and `corrected`, but every `corrected`
    # line is also added to `cleared`, so `cleared` alone is the exclusion set.
    suspect = jnp.logical_and(eligible, jnp.logical_or(out_forced, sa_risk))

    inflated = jnp.where(
        intensity_unc > 0.0,
        intensity_unc * suspect_uncertainty_inflation,
        intensity * suspect_uncertainty_inflation,
    )
    out_unc = jnp.where(suspect, inflated, out_unc)
    # Suspects that were NOT force-suspect (no doublet tau) get method=SUSPECT,
    # tau=nan; force-suspect already carry method/tau from the doublet pass.
    plain_suspect = jnp.logical_and(suspect, jnp.logical_not(out_forced))
    out_method = jnp.where(plain_suspect, jnp.int32(METHOD_SUSPECT), out_method)
    out_suspect = jnp.logical_and(suspect, line_valid)

    # --- counters (masked reductions) ------------------------------------
    is_corrected_line = jnp.logical_and(
        line_valid, jnp.logical_and(out_method == METHOD_DOUBLET, out_tau > 0.0)
    )
    # n_corrected counts lines whose correction factor > 1 (tau > 0 => f < 1).
    n_corrected = jnp.sum(is_corrected_line.astype(jnp.int32))
    n_suspect = jnp.sum(out_suspect.astype(jnp.int32))

    # max_tau over finite recovered taus on real lines (reference uses the
    # correction records' finite taus; thin/correct/forced lines all record a
    # finite tau, plain suspects record nan).
    finite_tau = jnp.where(jnp.logical_and(line_valid, jnp.isfinite(out_tau)), out_tau, 0.0)
    max_tau = jnp.max(jnp.concatenate([finite_tau, jnp.zeros((1,))]))

    # Zero-out padded positions for clean downstream consumption.
    out_intensity = jnp.where(line_valid, out_intensity, intensity)
    out_unc = jnp.where(line_valid, out_unc, intensity_unc)
    out_method = jnp.where(line_valid, out_method, jnp.int32(METHOD_NONE))
    out_suspect = jnp.logical_and(out_suspect, line_valid)

    return SelfAbsorptionResult(
        intensity=out_intensity,
        intensity_unc=out_unc,
        tau=out_tau,
        method=out_method,
        suspect=out_suspect,
        n_corrected=n_corrected,
        n_suspect=n_suspect,
        max_tau=max_tau,
    )


def _per_element_median(intensity: Any, line_element: Any, line_valid: Any) -> Any:
    """Per-line broadcast of its element's median *positive* intensity.

    Matches the reference suspect-pass median (``self_absorption_observable.py``
    ``_apply_suspect_pass``): ``median`` over each element's strictly-positive
    intensities, with elements having no positive line falling back to 0.0.
    Fixed-shape: computed against the full ``(L,)`` axis using an element-id
    equality matrix, so it is ``vmap``-clean.

    The median of an even-length set is the mean of the two central order
    statistics (NumPy convention), reproduced here by sorting the masked column
    and averaging the two middle valid entries.
    """
    n = intensity.shape[0]
    # element-equality matrix (L, L)
    same = line_element[:, None] == line_element[None, :]
    valid_pos = jnp.logical_and(line_valid, intensity > 0.0)
    members = jnp.logical_and(same, valid_pos[None, :])  # (L, L): row i's element members

    # For each row, gather that element's positive intensities, push non-members
    # to +inf so they sort to the end, then average the two central valid ones.
    big = jnp.max(intensity) + 1.0
    masked = jnp.where(members, intensity[None, :], big)
    sorted_vals = jnp.sort(masked, axis=1)  # ascending; valid first, +inf padding
    counts = jnp.sum(members, axis=1)  # (L,) number of positive members per row

    # median index handling (0-based): for count c, lower = (c-1)//2, upper = c//2.
    c = counts
    safe_c = jnp.maximum(c, 1)
    lower_idx = (safe_c - 1) // 2
    upper_idx = safe_c // 2
    rows = jnp.arange(n)
    lo_val = sorted_vals[rows, lower_idx]
    hi_val = sorted_vals[rows, upper_idx]
    med = 0.5 * (lo_val + hi_val)
    return jnp.where(c > 0, med, 0.0)


def correct_self_absorption(
    line_intensities: Any,
    line_index: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """J0-skeleton entry point — thin adapter to :func:`correct_self_absorption_arrays`.

    The J0 stub signature (``line_intensities, line_index, snapshot, params,
    static``) is retained for pipeline wiring at integration. The real
    fixed-shape kernel is :func:`correct_self_absorption_arrays`, which also
    takes the per-line uncertainties, element ids, and the local doublet-pair
    table the J5 spec §3 requires. Until the J2/J4 stages emit those padded
    arrays, this adapter raises a clear error rather than guessing them.

    Raises
    ------
    NotImplementedError
        The pipeline wiring (uncertainty + pair-table plumbing) lands at
        integration; call :func:`correct_self_absorption_arrays` directly with
        explicit padded inputs (as the parity test does).
    """
    raise NotImplementedError(
        "Use correct_self_absorption_arrays with explicit padded inputs; the "
        "5-arg pipeline adapter is wired at J-integration once J2/J4 emit the "
        "per-line uncertainty + local doublet-pair arrays."
    )
