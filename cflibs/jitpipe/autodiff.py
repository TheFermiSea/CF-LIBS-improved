"""J11 — Differentiability payoffs: implicit-diff spike + gradient knob-tuning.

ADR-0004 §6 / J11 spec §1 (deliverables 1 & 2). This module de-risks the two
biggest unknowns of the differentiable-pipeline program, *both implemented in
core JAX only* (no jaxopt / optimistix / lineax — equinox-dependent, banned per
ADR-0004 §9; no ``jax.nn`` per the physics-only TID251 rule):

1. **Implicit-diff spike (§1.1, deferred T4-1).** Differentiate the converged
   ``(T*, n_e*)`` fixed point of the damped Saha-Boltzmann update map (the
   reference scan body, ``solve/iterative.py:764-863``) w.r.t. its continuous
   inputs *without unrolling the loop*. Two routes are provided and must agree:

   * :func:`fixed_point_custom_root` — ``jax.lax.custom_root`` around a
     residual ``g(z) = z - F(z) = 0``, with a hand-supplied 2x2 Newton/tangent
     solver (``custom_root`` needs a linear-solve closure; the banned successors
     are exactly what we avoid by writing it ourselves).
   * :func:`fixed_point_custom_vjp` — an explicit Implicit-Function-Theorem
     ``custom_vjp``: forward returns ``z*``; backward solves the 2x2 adjoint
     ``(I - dF/dz)^T u = ḡ`` and contracts ``u`` with ``dF/dθ``.

   Backward memory is O(1) in the iteration count for both (the whole point —
   vs the unrolled scan whose tape grows with ``max_iters``).

2. **Gradient knob-tuning scaffolding (§6.3 / §6.4).** The *train-soft /
   eval-hard* relaxation map. Continuous knobs of ``AnalysisPipelineConfig``
   (here: presence cutoff, SNR gate, top-K cap) are made differentiable via
   explicit ``jnp`` smooth surrogates so **one backward pass** over a batch
   yields ``d(soft-F1)/d(knobs)``. The hard scoreboard counts are recovered
   exactly as ``tau -> 0`` (acceptance: ``test_soft_relax`` τ→0 limit), so the
   board never sees a relaxed metric.

   * presence rule ``C >= presence_cutoff`` -> :func:`soft_presence`
     (``sigmoid((C - cut) / tau)``, explicit ``jnp``);
   * SNR / prominence gate ``s >= thr`` -> :func:`soft_gate`;
   * top-K cap -> :func:`soft_top_k` (smooth fractional membership);
   * objective -> :func:`soft_f1` (differentiable F1 from soft TP/FP/FN).

This is a **research spike**: it consumes the reference physics (constants,
softmax closure) but is a self-contained, fixed-shape kernel — no SQLite, no
host imports, no data-dependent shapes (J11 import-hygiene + fixed-shape rules).
The full HMC deliverable (§1.3) depends on J7's joint residual and is tracked
separately; see ``remaining_todo`` in the J11 verdict.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp

from cflibs.core.constants import EV_TO_K, KB, SAHA_CONST_CM3
from cflibs.inversion.physics.softmax_closure import softmax_closure

# ---------------------------------------------------------------------------
# Section 1 — Implicit-diff spike through the (T*, n_e*) fixed point.
# ---------------------------------------------------------------------------
#
# The reference iterative solve (solve/iterative.py:_run_lax_while_loop) is a
# damped fixed-point iteration on z = (T_K, n_e_cm3). Per ADR-0004 §6.1 / the
# J11 spec, we isolate the converged-state map z* = F(z*) and differentiate
# through it implicitly. The full reference body folds in a Boltzmann common-
# slope fit; for the *spike* the relevant differentiability question is the
# coupled Saha / pressure-balance fixed point in (T, n_e), which is exactly the
# part that the deferred custom_root item (T4-1) targets. We reproduce that
# coupling faithfully (same SAHA_CONST_CM3, KB, EV_TO_K constants and the same
# 50/50 damping) in a closed, fixed-shape form so the implicit gradient can be
# FD-verified against the reference constants.


class FixedPointInputs(NamedTuple):
    """Continuous inputs to the (T*, n_e*) fixed-point map (all pytree leaves).

    Fixed shapes throughout: ``E`` is the padded element axis; validity is
    carried by ``element_mask`` (never by ragged shapes), per ADR-0004 §5.1.

    Attributes
    ----------
    intercepts : array, shape (E,)
        Common-slope Boltzmann-plot intercepts per element (``ln`` scale). In
        the reference these come from the fit; for the spike they are a frozen,
        well-converged input (J11 AC1 uses golden fixtures so the fixed point is
        well-conditioned).
    ip_eV : array, shape (E,)
        First-ionization potential per element, eV.
    U_I, U_II : array, shape (E,)
        Stage-I / stage-II partition functions (held constant over the spike's
        fixed point; the reference re-evaluates them per iter but they vary
        weakly near convergence — see ``conditioning`` note in the test).
    element_mask : array of bool, shape (E,)
        ``True`` for real elements, ``False`` for padding. Padding contributes
        zero everywhere (masked intercepts -> ``-inf`` logits -> zero weight).
    pressure_pa : scalar
        Plasma pressure for the charge/pressure balance, Pa.
    """

    intercepts: Any
    ip_eV: Any
    U_I: Any
    U_II: Any
    element_mask: Any
    pressure_pa: Any


def _saha_ratio(T_K, n_e, U_I, U_II, ip_eV):
    """Element-wise Saha ratio ``n_II / n_I`` — mirrors ``iterative._saha_ratio_per_element``."""
    safe_ne = jnp.maximum(n_e, 1e10)
    T_eV = jnp.maximum(T_K / EV_TO_K, 0.1)
    return (
        (SAHA_CONST_CM3 / safe_ne)
        * (T_eV**1.5)
        * (U_II / jnp.maximum(U_I, 1e-30))
        * jnp.exp(-ip_eV / T_eV)
    )


def fixed_point_step(z, inp: FixedPointInputs):
    """One damped Saha / pressure-balance update ``z -> F(z)`` on ``z = (T_K, n_e)``.

    This is the (T, n_e)-coupling slice of the reference body
    (``iterative.py:820-835``): standard-closure concentrations from the
    intercepts, abundance multipliers ``1 + S``, the average ionization
    ``avg_Z``, and the pressure-balance ``n_e`` update — all with the reference
    constants and the same 50/50 damping. ``T`` is held by its self-consistent
    Saha/pressure relation (the spike pins ``T`` to the closure-consistent value
    so the map is a contraction in both coordinates; the production loop drives
    ``T`` from the Boltzmann slope, deferred to J7).

    Parameters
    ----------
    z : tuple(scalar, scalar)
        Current ``(T_K, n_e_cm3)``.
    inp : FixedPointInputs
        Frozen continuous inputs.

    Returns
    -------
    tuple(scalar, scalar)
        Damped next ``(T_K, n_e_cm3)``.
    """
    T_prev, ne_prev = z
    mask = inp.element_mask.astype(jnp.float64)

    # Standard closure: relative number density n_s ~ mult * U_I * exp(intercept)
    # (reference _standard closure path). softmax over masked logits gives the
    # simplex concentrations; padding (mask=0) is sent to -inf -> weight 0.
    S = _saha_ratio(T_prev, ne_prev, inp.U_I, inp.U_II, inp.ip_eV)
    mult = 1.0 + jnp.maximum(S, 0.0)
    logits = jnp.log(jnp.maximum(mult * inp.U_I, 1e-300)) + inp.intercepts
    logits = jnp.where(inp.element_mask, logits, -1e30)
    concentrations = softmax_closure(logits)

    # Pressure-balance n_e update (reference :827-835), 50/50 damped.
    eps_s = S / (1.0 + S)  # electrons per atom of species
    avg_Z = jnp.sum(concentrations * eps_s * mask)
    n_tot = inp.pressure_pa / (KB * jnp.maximum(T_prev, 1.0) * (1.0 + avg_Z))
    n_tot_cm3 = n_tot * 1e-6
    ne_new = jnp.maximum(avg_Z * n_tot_cm3, 1e10)
    n_e = 0.5 * ne_prev + 0.5 * ne_new

    # T update: pin to the Saha-consistent temperature implied by the closure.
    # We use a smooth, contraction-preserving relaxation toward the closure-
    # weighted mean ionization energy scale (a stand-in for the Boltzmann slope
    # the J7 loop will supply). 50/50 damped to match the reference cadence.
    ip_bar = jnp.sum(concentrations * inp.ip_eV * mask) / jnp.maximum(
        jnp.sum(concentrations * mask), 1e-30
    )
    T_target = ip_bar * EV_TO_K / 3.0  # ~ kT ≈ ip/3 self-consistency scale
    T_new = jnp.clip(T_target, 2000.0, 60000.0)
    T_K = 0.5 * T_prev + 0.5 * T_new
    return (T_K, n_e)


def _residual(z, inp: FixedPointInputs):
    """Root residual ``g(z) = z - F(z)`` as a flat length-2 vector."""
    T1, n1 = fixed_point_step(z, inp)
    return jnp.array([z[0] - T1, z[1] - n1])


def solve_forward(inp: FixedPointInputs, *, max_iter: int = 200, tol: float = 1e-10):
    """Iterate ``z -> F(z)`` to convergence with a fixed-trip ``lax.scan``.

    Fixed shapes / fixed trip count (no data-dependent ``while_loop`` in a
    differentiated region, ADR-0004 §6.1). Returns the converged ``z* =
    (T_K, n_e)`` as a length-2 array.
    """
    z0 = jnp.array([8000.0, 1e16])

    def body(z, _):
        T1, n1 = fixed_point_step((z[0], z[1]), inp)
        return jnp.array([T1, n1]), None

    z_star, _ = jax.lax.scan(body, z0, None, length=max_iter)
    return z_star


# -- Route A: jax.lax.custom_root ------------------------------------------


def _newton_2x2_solve(matvec: Callable, b):
    """Solve ``J x = b`` for a 2x2 system given a matvec closure.

    ``custom_root`` hands us ``matvec = lambda x: J @ x`` (the linearized
    residual) and the rhs ``b``; we build the 2x2 ``J`` explicitly by probing
    with the basis vectors and invert in closed form. This is the hand-written
    linear solver that replaces the banned lineax dependency.
    """
    e0 = jnp.array([1.0, 0.0])
    e1 = jnp.array([0.0, 1.0])
    col0 = matvec(e0)
    col1 = matvec(e1)
    jac = jnp.stack([col0, col1], axis=1)  # columns are J @ e_i
    return jnp.linalg.solve(jac, b)


def fixed_point_custom_root(inp: FixedPointInputs, *, max_iter: int = 200):
    """Converged ``z* = (T_K, n_e)`` via ``jax.lax.custom_root`` (implicit diff).

    ``custom_root(f, initial_guess, solve, tangent_solve)`` returns the root of
    ``f`` and differentiates it *implicitly* — the backward pass solves the
    linearized residual via ``tangent_solve`` rather than unrolling ``solve``.
    Backward memory is therefore O(1) in ``max_iter`` (J11 AC1).

    Returns
    -------
    array, shape (2,)
        ``[T_K, n_e_cm3]``.
    """

    def f(z):
        return _residual(z, inp)

    def solve(f_inner, z0):
        # Forward solver: plain fixed-point iteration (not differentiated through).
        def body(z, _):
            T1, n1 = fixed_point_step((z[0], z[1]), inp)
            return jnp.array([T1, n1]), None

        z_star, _ = jax.lax.scan(body, z0, None, length=max_iter)
        return z_star

    def tangent_solve(g_lin, y):
        # g_lin(x) is the JVP of the residual at the root; solve g_lin(x) = y.
        return _newton_2x2_solve(g_lin, y)

    z0 = jnp.array([8000.0, 1e16])
    return jax.lax.custom_root(f, z0, solve, tangent_solve)


# -- Route B: hand-written custom_vjp via the Implicit Function Theorem -----


@jax.custom_vjp
def fixed_point_custom_vjp(inp: FixedPointInputs):
    """Converged ``z*`` with a hand-written IFT ``custom_vjp`` (implicit diff).

    Forward: run the fixed-point solve to ``z*``. Backward: by the IFT, for
    ``g(z*, θ) = z* - F(z*, θ) = 0``,

        dz*/dθ = -(∂g/∂z)^{-1} ∂g/∂θ = (I - ∂F/∂z)^{-1} ∂F/∂θ.

    The cotangent pullback solves the 2x2 adjoint ``(I - ∂F/∂z)^T u = z̄`` then
    contracts ``u`` with ``∂F/∂θ`` (a single VJP of ``F`` at the *fixed* root).
    No loop is unrolled => O(1) backward memory. Returns ``[T_K, n_e]``.
    """
    return solve_forward(inp)


def _fp_fwd(inp: FixedPointInputs):
    z_star = solve_forward(inp)
    return z_star, (z_star, inp)


def _fp_bwd(res, z_bar):
    z_star, inp = res

    # F as a function of z (theta fixed) and of theta (z fixed at the root).
    def F_of_z(z):
        T1, n1 = fixed_point_step((z[0], z[1]), inp)
        return jnp.array([T1, n1])

    def F_of_theta(theta_inp):
        T1, n1 = fixed_point_step((z_star[0], z_star[1]), theta_inp)
        return jnp.array([T1, n1])

    # Adjoint solve: (I - dF/dz)^T u = z_bar. Build the 2x2 dF/dz by Jacobian.
    dFdz = jax.jacobian(F_of_z)(z_star)  # (2, 2)
    A = jnp.eye(2) - dFdz
    u = jnp.linalg.solve(A.T, z_bar)

    # Contract u with dF/dtheta via one VJP of F_of_theta at the root.
    _, vjp_theta = jax.vjp(F_of_theta, inp)
    (inp_bar,) = vjp_theta(u)
    return (inp_bar,)


fixed_point_custom_vjp.defvjp(_fp_fwd, _fp_bwd)


# ---------------------------------------------------------------------------
# Section 2 — Gradient knob-tuning: train-soft / eval-hard relaxation map.
# ---------------------------------------------------------------------------
#
# ADR-0004 §6.4 relaxation map. Every smooth surrogate is explicit jnp (no
# jax.nn). As tau -> 0 each surrogate recovers its hard counterpart EXACTLY
# (test_soft_relax), so the scoreboard never sees a relaxed metric.

PRESENCE_CUTOFF: float = 0.005  # reference scoreboard presence rule C >= 0.005


def _sigmoid(x):
    """Numerically stable logistic, written explicitly (no ``jax.nn.sigmoid``)."""
    return jnp.where(x >= 0.0, 1.0 / (1.0 + jnp.exp(-x)), jnp.exp(x) / (1.0 + jnp.exp(x)))


def soft_presence(concentration, cutoff: float = PRESENCE_CUTOFF, tau: float = 1e-3):
    """Soft membership of the presence rule ``C >= cutoff``.

    ``sigmoid((C - cutoff) / tau)`` — explicit ``jnp`` (ADR-0004 §6.4). As
    ``tau -> 0`` this tends to the hard indicator ``1{C >= cutoff}`` exactly
    (away from the boundary), which the τ→0 parity test asserts.
    """
    return _sigmoid((concentration - cutoff) / tau)


def soft_gate(score, threshold, tau: float = 1e-3):
    """Soft membership of a generic gate ``score >= threshold`` (SNR / prominence).

    Same explicit-``jnp`` sigmoid relaxation as :func:`soft_presence`, with the
    threshold a *traced* continuous knob so ``d(objective)/d(threshold)`` is one
    backward pass.
    """
    return _sigmoid((score - threshold) / tau)


def hard_presence(concentration, cutoff: float = PRESENCE_CUTOFF):
    """Hard presence indicator ``1{C >= cutoff}`` — the scoreboard rule."""
    return (concentration >= cutoff).astype(jnp.float64)


def soft_top_k(scores, k, tau: float = 1e-2, mask=None):
    """Smooth fractional top-K membership over a fixed-length ``scores`` vector.

    Returns a soft mask in ``[0, 1]`` per element approximating "is this score
    among the top ``k``". Construction (all fixed-shape, vmap-clean): each
    score's *rank from the top* is ``rank_above_i = sum_j sigmoid((s_j - s_i)/
    tau)`` (a smooth count of how many other scores strictly beat it; the
    diagonal self-term is removed). Membership is
    ``sigmoid((k - 0.5 - rank_above)/tau)`` — the top score has ``rank_above ≈
    0`` and is selected; the (k+1)-th has ``rank_above ≈ k`` and is rejected.
    As ``tau -> 0`` this selects exactly the ``round(k)`` largest scores. ``k``
    is a *continuous* (traced) knob, so the top-K cap is differentiable.

    Padding is handled with a validity mask, NOT ``-inf`` sentinels (a pairwise
    ``(-inf) - (-inf)`` would be ``NaN``): invalid entries are pulled far below
    every valid score by a finite offset, never selected, and zeroed in the
    returned mask.

    Parameters
    ----------
    scores : array, shape (N,)
        Per-candidate scores.
    k : scalar
        Continuous top-K cap.
    tau : float
        Relaxation temperature.
    mask : array of bool, shape (N,), optional
        Validity mask; invalid entries are excluded from the ranking and from
        the returned membership.

    Returns
    -------
    array, shape (N,)
        Soft membership mask (zero on invalid entries).
    """
    if mask is None:
        valid = jnp.ones_like(scores, dtype=bool)
    else:
        valid = mask.astype(bool)
    # Finite "push-down" for invalid entries: below the smallest valid score by
    # a wide, finite margin so they never rank in the top-k and never produce
    # NaN in the pairwise differences.
    finite_scores = jnp.where(jnp.isfinite(scores), scores, 0.0)
    lo = jnp.min(jnp.where(valid, finite_scores, jnp.max(finite_scores)))
    span = jnp.maximum(jnp.max(finite_scores) - lo, 1.0)
    s = jnp.where(valid, finite_scores, lo - 1e6 * span)

    s_i = s[:, None]
    s_j = s[None, :]
    # Soft count of scores STRICTLY GREATER than s_i (rank from the top).
    greater = _sigmoid((s_j - s_i) / tau)
    rank_above = jnp.sum(greater, axis=1) - jnp.diagonal(greater)
    membership = _sigmoid((k - 0.5 - rank_above) / tau)
    return membership * valid.astype(membership.dtype)


def soft_confusion_counts(
    pred_conc, truth_present, *, cutoff: float = PRESENCE_CUTOFF, tau: float = 1e-3, mask=None
):
    """Soft (TP, FP, FN) over a fixed-length element axis.

    ``truth_present`` is the hard ground-truth presence (0/1, never relaxed —
    it is data, not a decision). ``pred_conc`` are the predicted concentrations
    whose presence is *soft*-thresholded so the counts are differentiable in the
    pipeline knobs that move ``pred_conc`` and in ``cutoff``.

    Parameters
    ----------
    pred_conc : array, shape (E,)
        Predicted concentrations.
    truth_present : array, shape (E,)
        Ground-truth presence indicators (0/1).
    cutoff : float
        Presence cutoff knob (continuous).
    tau : float
        Relaxation temperature.
    mask : array of bool, shape (E,), optional
        Validity mask (padding excluded from the counts).

    Returns
    -------
    tuple(scalar, scalar, scalar)
        ``(soft_TP, soft_FP, soft_FN)``.
    """
    p = soft_presence(pred_conc, cutoff=cutoff, tau=tau)
    t = truth_present.astype(jnp.float64)
    if mask is None:
        m = jnp.ones_like(p)
    else:
        m = mask.astype(jnp.float64)
    tp = jnp.sum(p * t * m)
    fp = jnp.sum(p * (1.0 - t) * m)
    fn = jnp.sum((1.0 - p) * t * m)
    return tp, fp, fn


def soft_f1(
    pred_conc, truth_present, *, cutoff: float = PRESENCE_CUTOFF, tau: float = 1e-3, mask=None
):
    """Differentiable soft-F1 objective (J11 §6.3 knob-tuning target).

    ``F1 = 2 TP / (2 TP + FP + FN)`` over the soft confusion counts. One reverse
    pass yields ``d(soft-F1)/d(knobs)`` (presence cutoff, plus any upstream knob
    that moves ``pred_conc``). As ``tau -> 0`` this tends to the hard F1 the
    scoreboard reports.
    """
    tp, fp, fn = soft_confusion_counts(pred_conc, truth_present, cutoff=cutoff, tau=tau, mask=mask)
    denom = 2.0 * tp + fp + fn
    return jnp.where(denom > 0.0, 2.0 * tp / jnp.maximum(denom, 1e-30), 0.0)


def hard_f1(pred_conc, truth_present, *, cutoff: float = PRESENCE_CUTOFF, mask=None):
    """Hard F1 the scoreboard computes (the ``tau -> 0`` limit of :func:`soft_f1`)."""
    p = hard_presence(pred_conc, cutoff=cutoff)
    t = truth_present.astype(jnp.float64)
    if mask is None:
        m = jnp.ones_like(p)
    else:
        m = mask.astype(jnp.float64)
    tp = jnp.sum(p * t * m)
    fp = jnp.sum(p * (1.0 - t) * m)
    fn = jnp.sum((1.0 - p) * t * m)
    denom = 2.0 * tp + fp + fn
    return jnp.where(denom > 0.0, 2.0 * tp / jnp.maximum(denom, 1e-30), 0.0)


class KnobSlice(NamedTuple):
    """The 3-knob continuous slice the J11 gradient harness tunes (§6.3).

    These are *traced* leaves — moving them never recompiles (the design's
    zero-recompile property). The full 20-knob surface lives in
    :class:`cflibs.jitpipe.params.PipelineParams`; the spike tunes a slice so the
    gradient is FD-verifiable cheaply (J11 AC2 "3-knob slice").

    Attributes
    ----------
    presence_cutoff : scalar
        Presence rule cutoff (continuous; soft in training, hard on the board).
    min_snr : scalar
        Per-line SNR gate.
    top_k : scalar
        Soft top-K cap per element.
    """

    presence_cutoff: Any
    min_snr: Any
    top_k: Any


def knob_objective(knobs: KnobSlice, batch: dict, *, tau: float = 1e-2):
    """Soft-F1 objective over a fixed-shape batch as a function of the 3-knob slice.

    The objective threads all three continuous knobs through their soft
    surrogates and returns the *negative* mean soft-F1 (a loss to minimise),
    so ``jax.grad(knob_objective)`` is the gradient inner loop of §6.3. Fully
    fixed-shape and ``vmap``-clean across the batch axis ``B``.

    Parameters
    ----------
    knobs : KnobSlice
        The traced continuous knobs.
    batch : dict
        Fixed-shape arrays, batch axis first:
        ``pred_conc`` (B, E), ``truth_present`` (B, E), ``snr`` (B, E),
        ``element_mask`` (B, E), ``cand_scores`` (B, E).
    tau : float
        Relaxation temperature (train-soft; eval uses τ→0 / hard).

    Returns
    -------
    scalar
        Negative mean soft-F1 over the batch.
    """
    pred_conc = batch["pred_conc"]
    truth_present = batch["truth_present"]
    snr = batch["snr"]
    element_mask = batch["element_mask"]
    cand_scores = batch["cand_scores"]

    def per_spectrum(pc, tp_truth, s, m, cs):
        # SNR gate softly attenuates candidates below min_snr.
        snr_keep = soft_gate(s, knobs.min_snr, tau=tau)
        # Soft top-K cap on candidate scores (padding excluded via mask).
        topk_keep = soft_top_k(cs, knobs.top_k, tau=tau, mask=m)
        # Effective predicted presence weight = conc gated by SNR & top-K.
        eff_conc = pc * snr_keep * topk_keep
        return soft_f1(eff_conc, tp_truth, cutoff=knobs.presence_cutoff, tau=tau, mask=m)

    f1s = jax.vmap(per_spectrum)(pred_conc, truth_present, snr, element_mask, cand_scores)
    return -jnp.mean(f1s)


def knob_gradient(knobs: KnobSlice, batch: dict, *, tau: float = 1e-2):
    """One backward pass: ``(loss, d loss / d knobs)`` for the 3-knob slice.

    This is the §6.3 gradient inner loop — ~2-3x a forward board for the full
    gradient, vs one full board per TPE point with no gradient.
    """
    return jax.value_and_grad(knob_objective)(knobs, batch, tau=tau)
