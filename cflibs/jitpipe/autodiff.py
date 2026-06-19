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

3. **HMC / NUTS over the jit model (§1.3 / ADR-0004 §6.2).** NumPyro NUTS over
   ``(T, n_e, α)`` consuming **J7's joint-WLS residual** (``solve.py
   joint_wls_solve``) as the differentiable potential — the ExoJAX pattern
   (forward outside the model graph; sampler loop host-side, ADR-0004 §5.1.3).
   NUTS *replaces* the iterative solve (it does not wrap it): the same
   weighted-least-squares Boltzmann-plane residual that ``joint_wls_solve``
   minimises by Gauss–Newton becomes the log-potential a Hamiltonian sampler
   explores. The global intercept ``β`` (a nuisance) is profiled out
   analytically per draw — the concentrated residual is a clean function of
   ``(lnT, log10 n_e, α)`` alone, reverse-differentiated by NUTS. fp64 is
   mandatory (NUTS on fp32 plasma exponentials diverges, §1.3). The mandatory
   NNLS top-K candidate prefilter
   (:func:`cflibs.inversion.candidate_prefilter.select_candidate_elements`,
   CLAUDE.md) bounds the element (``E``) axis; priors come from the reused
   ``solve/bayesian/priors.PriorConfig``; the Chebyshev baseline of
   ``BayesianForwardModel`` reduces, on the Boltzmann plane, to the additive
   intercept ``β`` already absorbed by the profile-out (see
   :func:`joint_wls_potential`). See :func:`run_joint_nuts`.

This is a **research spike**: it consumes the reference physics (constants,
softmax closure, the J7 joint-WLS design) but is a self-contained, fixed-shape
kernel — no SQLite, no host imports, no data-dependent shapes (J11
import-hygiene + fixed-shape rules). numpyro is the shipped CF-LIBS sampler
(``solve/bayesian/``) and is the only ML-adjacent dep used here; jaxopt /
optimistix / lineax / equinox stay TID251-banned and ``jax.nn`` is never used.

Consumption status (2026-06)
----------------------------
This is the documented differentiability surface (ADR-0004 §6.2). None of its
public symbols (``knob_gradient``, ``knob_objective``, ``run_joint_nuts``,
``soft_f1``, ``fixed_point_custom_vjp``, …) are referenced from any production
``cflibs/`` module or script; the only importer is the J11 parity test
(``tests/jitpipe/test_parity_j11.py``). It is intentional research
infrastructure, not cruft — kept as the differentiable-pipeline payoff spike.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple

import jax
import jax.numpy as jnp

from cflibs.core.constants import EV_TO_K, KB, KB_EV, SAHA_CONST_CM3
from cflibs.inversion.physics.softmax_closure import softmax_closure
from cflibs.jitpipe.solve import (
    LaxKernelInputs,
    _eval_partition_jax,
    _saha_ratio_per_element,
    helmert_basis,
    ilr_inverse,
)

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


# ---------------------------------------------------------------------------
# Section 3 — HMC / NUTS over the jit model (§1.3 / ADR-0004 §6.2).
# ---------------------------------------------------------------------------
#
# NumPyro NUTS over (T, n_e, α) consuming J7's joint-WLS residual as the
# differentiable potential. The ExoJAX pattern (ADR-0004 §5.1.3): the forward
# physics live OUTSIDE the model graph (here, the device-pure
# :func:`joint_wls_potential` built from the FROZEN `solve.py` kernels), and
# NumPyro's sampler loop runs host-side. NUTS *replaces* the iterative
# Gauss-Newton solve: the same weighted-least-squares Boltzmann-plane residual
# that :func:`cflibs.jitpipe.solve.joint_wls_solve` minimises becomes the
# log-potential a Hamiltonian sampler explores. fp64 throughout (§1.3).


class JointPotentialOutputs(NamedTuple):
    """Diagnostics returned alongside :func:`joint_wls_potential`.

    Attributes
    ----------
    rss : scalar
        Weighted residual sum of squares ``Σ_l w_l r_l²`` at the profiled-out
        intercept ``β*`` — the joint-WLS objective evaluated at ``(T, n_e, α)``.
    beta : scalar
        The analytically profiled-out global intercept (the
        :class:`BayesianForwardModel` Chebyshev-baseline DC term on the
        Boltzmann plane).
    T_K : scalar
        The sampled temperature in K (echoed for convenience).
    n_e_cm3 : scalar
        The sampled electron density in cm^-3 (echoed).
    concentrations : (E,)
        ``ilr_inverse(α)`` simplex composition (sum 1) — the SAME map
        :func:`joint_wls_solve` uses to read off its recovered composition.
    """

    rss: Any
    beta: Any
    T_K: Any
    n_e_cm3: Any
    concentrations: Any


def joint_wls_potential(
    ln_T: Any,
    log10_ne: Any,
    alpha: Any,
    inp: LaxKernelInputs,
    *,
    sb_graph: bool = False,
) -> JointPotentialOutputs:
    """Weighted joint-WLS Boltzmann-plane residual as a function of ``(T, n_e, α)``.

    This is the **exact** residual that :func:`cflibs.jitpipe.solve.joint_wls_solve`
    minimises by Gauss-Newton (ADR-0004 §6.1), but parameterised by the *plasma*
    coordinates ``(ln T, log10 n_e, α)`` instead of the regression vector
    ``θ = (m, α, β)``. The slope ``m = -1/(k_B T)`` is pinned by ``T``; ``α`` are
    the ILR composition coordinates; the global intercept ``β`` is a nuisance
    profiled out analytically (its closed-form WLS optimum given the rest), so
    the returned ``rss`` is a clean differentiable function of ``(ln T, log10
    n_e, α)`` alone. NUTS reverse-differentiates this potential directly.

    Construction is byte-faithful to ``joint_wls_solve`` (same flattening,
    same ``x_shift = E_k + IP·(z-1)`` ion lever arm, same ``y_adj = y -
    ln_S·(z-1) + ln U_s(T) + ln M_s(T, n_e)`` neutral-plane pre-adjust, same
    Helmert ILR columns, same masked weights). It imports the FROZEN reference
    partition / Saha kernels (``_eval_partition_jax``, ``_saha_ratio_per_element``)
    rather than re-deriving them, so the potential and the GN solver share one
    physics by construction (parity asserted in ``test_parity_j11``).

    Parameters
    ----------
    ln_T : scalar
        Natural log of the temperature in K (NUTS samples in log space so the
        physical-positivity constraint is reparameterised, ADR-0004 §6.4
        "reparameterized physical constraints hard").
    log10_ne : scalar
        ``log10`` of the electron density in cm^-3 (the Jeffreys / log-uniform
        coordinate the ``solve/bayesian`` priors use).
    alpha : (D-1,)
        ILR composition coordinates; ``concentrations = ilr_inverse(alpha)``.
    inp : LaxKernelInputs
        The padded per-bucket candidate line block (same object the J7 solve
        kernels consume).
    sb_graph : bool
        Unit weights (validated SB-graph) when True; inverse-variance obs
        weights otherwise (default — the ClosedFormILRSolver path), matching
        ``joint_wls_solve``'s ``sb_graph`` switch.

    Returns
    -------
    JointPotentialOutputs
    """
    T_K = jnp.exp(ln_T)
    n_e_cm3 = jnp.power(10.0, log10_ne)

    E = inp.x.shape[0]
    Nmax = inp.x.shape[1]
    D = E

    # Flatten the (E, N_max) line block to rows; species index per row (== joint_wls_solve).
    sp_idx = jnp.repeat(jnp.arange(E), Nmax)
    x_flat = inp.x.reshape(-1)
    y_flat = inp.y.reshape(-1)
    w_flat = inp.w.reshape(-1)
    stage_flat = inp.stage.reshape(-1)
    mask_flat = inp.mask.reshape(-1)
    ip_per_row = inp.ip0[sp_idx]

    if sb_graph:
        W = jnp.where(mask_flat, 1.0, 0.0)
    else:
        W = jnp.where(mask_flat, w_flat, 0.0)

    # Ion -> neutral plane shift (matches joint_wls_solve / _saha_correct_kernel).
    z_minus_1 = jnp.where(stage_flat > 1, (stage_flat - 1).astype(jnp.float64), 0.0)
    x_shift = x_flat + ip_per_row * z_minus_1

    # Helmert ILR design columns (same basis the GN solver / ClosedForm use).
    if D >= 2:
        V = helmert_basis(D)
        ilr_cols = V[sp_idx, :]  # (rows, D-1)
        conc = ilr_inverse(alpha, V)
    else:
        ilr_cols = jnp.zeros((x_flat.shape[0], 0), dtype=jnp.float64)
        conc = jnp.ones(1, dtype=jnp.float64)

    # y_adj = y_shift + ln U_s(T) + ln M_s(T, n_e)  (joint_wls_solve._y_adj_for).
    T_eV = jnp.maximum(T_K / EV_TO_K, 0.1)
    safe_ne = jnp.maximum(n_e_cm3, 1e10)
    ln_S = jnp.log((SAHA_CONST_CM3 / safe_ne) * (T_eV**1.5))
    y_shift = y_flat - ln_S * z_minus_1
    U_I = _eval_partition_jax(
        T_K,
        inp.use_direct[:, 0],
        inp.g_I,
        inp.E_I,
        inp.ip_I,
        inp.coeffs_I,
        inp.fallback_I,
        inp.mask_I,
    )
    U_II = _eval_partition_jax(
        T_K,
        inp.use_direct[:, 1],
        inp.g_II,
        inp.E_II,
        inp.ip_II,
        inp.coeffs_II,
        inp.fallback_II,
        inp.mask_II,
    )
    S = _saha_ratio_per_element(T_K, n_e_cm3, U_I, U_II, inp.ip0)
    M = 1.0 + jnp.maximum(S, 0.0)
    lnU = jnp.log(jnp.maximum(U_I, 1e-30))  # (E,)
    lnM = jnp.log(jnp.maximum(M, 1e-30))  # (E,)
    y_adj = y_shift + lnU[sp_idx] + lnM[sp_idx]

    # m = -1/(k_B T) is the slope column lever-arm coefficient (joint_wls_solve:
    # T_run = -1/(m * KB_EV)  =>  m = -1/(T * KB_EV)).
    m = -1.0 / (T_K * KB_EV)

    # Per-row model without the intercept; β is profiled out by weighted-LS.
    fitted_no_beta = m * x_shift + (ilr_cols @ alpha if D >= 2 else 0.0)
    resid_no_beta = y_adj - fitted_no_beta
    Wsum = jnp.maximum(jnp.sum(W), 1e-30)
    beta = jnp.sum(W * resid_no_beta) / Wsum  # WLS-optimal intercept.

    residuals = resid_no_beta - beta
    rss = jnp.sum(W * residuals**2)

    return JointPotentialOutputs(
        rss=rss,
        beta=beta,
        T_K=T_K,
        n_e_cm3=n_e_cm3,
        concentrations=conc,
    )


def joint_potential_rss(ln_T: Any, log10_ne: Any, alpha: Any, inp: LaxKernelInputs) -> Any:
    """Scalar potential ``Σ_l w_l r_l²`` — the FD-verifiable joint-WLS residual.

    Thin scalar adapter over :func:`joint_wls_potential` for ``jax.grad`` /
    finite-difference checks (J11 AC: FD-verified gradients on the joint
    potential). The Gauss-Newton fixed point of ``joint_wls_solve`` is, by
    construction, a stationary point of this scalar (``∇_{(m,α,β)} rss = 0`` at
    the WLS optimum), so the two estimators share one objective.
    """
    return joint_wls_potential(ln_T, log10_ne, alpha, inp).rss


def make_joint_nuts_model(
    inp: LaxKernelInputs,
    *,
    n_elements: int,
    T_eV_range: Tuple[float, float] = (0.5, 3.0),
    log_ne_range: Tuple[float, float] = (15.0, 19.0),
    alpha_scale: float = 5.0,
    sb_graph: bool = False,
    sigma: float = 0.05,
) -> Callable[[], None]:
    """Build a NumPyro model whose potential is the joint-WLS residual.

    The model samples ``(T_eV, log_ne, α)`` from the reused
    ``solve/bayesian`` prior conventions (uniform-in-eV temperature, log-uniform
    ``n_e``, a weakly-informative Normal on the ILR coordinates — the simplex
    reparameterisation kept "hard" per ADR-0004 §6.4) and adds the
    Boltzmann-plane Gaussian log-likelihood ``-rss / (2 σ²)`` via
    :func:`numpyro.factor`. The residual is :func:`joint_wls_potential` — the
    forward physics evaluated *outside* the NumPyro graph proper (ExoJAX
    pattern). NUTS reverse-differentiates it for its Hamiltonian dynamics.

    Returns a zero-arg callable (the NumPyro model) suitable for ``NUTS(model)``.

    Parameters
    ----------
    inp : LaxKernelInputs
        Padded candidate line block (E == ``n_elements`` after the mandatory
        candidate prefilter has fixed the element axis).
    n_elements : int
        Number of (prefiltered) elements ``E``; the ILR axis is ``E - 1``.
    T_eV_range, log_ne_range : tuple
        Prior bounds (``PriorConfig`` defaults).
    alpha_scale : float
        Prior std on each ILR coordinate (broad; the data dominate).
    sb_graph : bool
        Forward the SB-graph weight switch to :func:`joint_wls_potential`.
    sigma : float
        Boltzmann-plane residual noise scale (the ``ln`` y-residual std).
    """
    import numpyro
    import numpyro.distributions as dist

    T_lo, T_hi = T_eV_range
    ne_lo, ne_hi = log_ne_range
    D = n_elements
    inv_two_sig2 = 1.0 / (2.0 * sigma**2)

    def model() -> None:
        # T in eV (PriorConfig convention) -> K for the potential.
        T_eV = numpyro.sample("T_eV", dist.Uniform(T_lo, T_hi))
        T_K = numpyro.deterministic("T_K", T_eV * EV_TO_K)
        ln_T = jnp.log(T_K)
        log_ne = numpyro.sample("log_ne", dist.Uniform(ne_lo, ne_hi))
        if D >= 2:
            alpha = numpyro.sample("alpha", dist.Normal(jnp.zeros(D - 1), alpha_scale).to_event(1))
        else:
            alpha = jnp.zeros(0, dtype=jnp.float64)
        out = joint_wls_potential(ln_T, log_ne, alpha, inp, sb_graph=sb_graph)
        # Record the simplex composition as a derived quantity for posterior read-off.
        numpyro.deterministic("concentrations", out.concentrations)
        # Gaussian Boltzmann-plane likelihood: log p = -rss / (2 σ²).
        numpyro.factor("joint_wls_loglik", -inv_two_sig2 * out.rss)

    return model


def run_joint_nuts(
    inp: LaxKernelInputs,
    *,
    n_elements: int,
    candidate_elements: Optional[List[str]] = None,
    num_warmup: int = 200,
    num_samples: int = 200,
    num_chains: int = 1,
    seed: int = 0,
    T_eV_range: Tuple[float, float] = (0.5, 3.0),
    log_ne_range: Tuple[float, float] = (15.0, 19.0),
    alpha_scale: float = 5.0,
    sb_graph: bool = False,
    sigma: float = 0.05,
    target_accept_prob: float = 0.9,
    max_tree_depth: int = 8,
    progress_bar: bool = False,
) -> Dict[str, Any]:
    """Run NumPyro NUTS over the joint-WLS potential and return posterior draws.

    Host driver mirroring ``solve/bayesian/samplers.py::MCMCSampler.run``: it
    constructs a :class:`~numpyro.infer.NUTS` kernel over
    :func:`make_joint_nuts_model` and drives the host-side sampler loop. The
    mandatory NNLS candidate prefilter
    (:func:`cflibs.inversion.candidate_prefilter.select_candidate_elements`) is
    expected to have already fixed the element axis of ``inp``; pass the chosen
    symbols as ``candidate_elements`` for provenance (recorded in the result).

    Parameters
    ----------
    inp : LaxKernelInputs
        Padded candidate line block (E == ``n_elements``).
    n_elements : int
        Number of prefiltered elements.
    candidate_elements : list of str, optional
        Prefilter output (recorded in the returned ``metadata`` for provenance).
    num_warmup, num_samples, num_chains, seed : int
        Standard NUTS controls.
    T_eV_range, log_ne_range, alpha_scale, sb_graph, sigma
        Forwarded to :func:`make_joint_nuts_model`.
    target_accept_prob, max_tree_depth, progress_bar
        Standard NUTS kernel knobs.

    Returns
    -------
    dict
        ``{"samples": {site: array}, "diagnostics": {...}, "metadata": {...},
        "num_divergences": int}``.
    """
    import jax.random as random
    from numpyro.infer import MCMC, NUTS, init_to_uniform

    model = make_joint_nuts_model(
        inp,
        n_elements=n_elements,
        T_eV_range=T_eV_range,
        log_ne_range=log_ne_range,
        alpha_scale=alpha_scale,
        sb_graph=sb_graph,
        sigma=sigma,
    )
    kernel = NUTS(
        model,
        init_strategy=init_to_uniform(radius=0.5),
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
    )
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="sequential" if num_chains == 1 else "vectorized",
        progress_bar=progress_bar,
    )
    mcmc.run(random.PRNGKey(seed), extra_fields=("diverging",))

    samples = {k: jnp.asarray(v) for k, v in mcmc.get_samples().items()}
    # Divergence count (a NUTS health probe; fp64 keeps plasma exponentials sane).
    try:
        num_div = int(jnp.sum(mcmc.get_extra_fields()["diverging"]))
    except Exception:  # pragma: no cover - extra fields not collected
        num_div = 0

    return {
        "samples": samples,
        "diagnostics": {"num_samples": num_samples, "num_chains": num_chains},
        "num_divergences": num_div,
        "metadata": {
            "sampler": "joint_wls_nuts",
            "n_elements": n_elements,
            "candidate_elements": list(candidate_elements) if candidate_elements else None,
        },
    }
