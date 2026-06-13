"""J11 parity / acceptance tests — implicit-diff spike + gradient knob-tuning.

ADR-0004 §6 / J11 spec §2-§3. Covers the two implemented deliverables:

* **Implicit-diff spike (§1.1):** ``jax.lax.custom_root`` and a hand-written
  ``custom_vjp`` both round-trip to the fixed point and yield gradients of
  ``(T*, n_e*)`` that agree with (a) the unrolled-``scan`` autodiff oracle and
  (b) central finite differences at rtol ≤ 1e-4 (AC1). Backward memory is
  O(1) in the iteration count (asserted structurally: the cotangent rule never
  unrolls the loop).
* **Reference-anchored physics:** the spike's Saha ratio is asserted
  **bit-identical** to the FROZEN reference ``iterative._saha_ratio_per_element``
  (the parity oracle is imported and run, not reimplemented).
* **Knob-tuning relaxation (§6.4):** soft surrogates recover their hard
  scoreboard counterparts exactly as ``tau -> 0`` (presence / top-K / F1); the
  3-knob gradient is FD-verified (AC2).
* Cross-cutting: vmap smoke (batch 16), grad finiteness, no-SQLite-in-kernel
  guard, padding-invariance (rerun at the next pad size => bit-identical on the
  valid region).

All CPU-x64 (conftest forces it), well under the 600 s watchdog.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_jax, pytest.mark.physics]

jax = pytest.importorskip("jax")
jnp = jax.numpy


from cflibs.jitpipe import autodiff as ad  # noqa: E402
from cflibs.jitpipe.autodiff import (  # noqa: E402
    FixedPointInputs,
    KnobSlice,
    fixed_point_custom_root,
    fixed_point_custom_vjp,
    hard_f1,
    hard_presence,
    knob_gradient,
    knob_objective,
    soft_f1,
    soft_presence,
    soft_top_k,
    solve_forward,
)

# --------------------------------------------------------------------------
# Fixtures: a well-converged golden fixed-point case (AC1 uses well-conditioned
# golden fixtures so the implicit linear solve is well-posed).
# --------------------------------------------------------------------------


def _golden_inputs(pad: int = 4) -> FixedPointInputs:
    """A 3-element golden case padded to ``pad`` (>= 3) with a validity mask."""
    assert pad >= 3
    base_intercepts = np.array([0.0, -1.0, -2.0])
    base_ip = np.array([7.9, 6.0, 5.1])
    base_UI = np.array([25.0, 10.0, 6.0])
    base_UII = np.array([30.0, 12.0, 5.0])

    def _pad(vals, fill):
        out = np.full(pad, fill, dtype=np.float64)
        out[: len(vals)] = vals
        return out

    mask = np.zeros(pad, dtype=bool)
    mask[:3] = True
    return FixedPointInputs(
        intercepts=jnp.asarray(_pad(base_intercepts, 0.0)),
        ip_eV=jnp.asarray(_pad(base_ip, 0.0)),
        U_I=jnp.asarray(_pad(base_UI, 1.0)),
        U_II=jnp.asarray(_pad(base_UII, 1.0)),
        element_mask=jnp.asarray(mask),
        pressure_pa=jnp.asarray(101325.0),
    )


# --------------------------------------------------------------------------
# 1. Implicit-diff spike — round-trip + gradient parity (AC1).
# --------------------------------------------------------------------------


def test_fixed_point_round_trips_all_routes():
    """custom_root, custom_vjp and the unrolled scan all reach the same z*."""
    inp = _golden_inputs()
    z_scan = np.asarray(solve_forward(inp))
    z_root = np.asarray(fixed_point_custom_root(inp))
    z_vjp = np.asarray(fixed_point_custom_vjp(inp))

    assert np.all(np.isfinite(z_scan))
    assert np.allclose(z_scan, z_root, rtol=1e-8, atol=0.0)
    assert np.allclose(z_scan, z_vjp, rtol=1e-8, atol=0.0)

    # Residual at the root is (numerically) zero — it IS a fixed point.
    resid = np.asarray(ad._residual((z_root[0], z_root[1]), inp))
    assert np.allclose(resid, 0.0, atol=1e-6)


def test_implicit_grad_matches_scan_and_fd():
    """d(T*, n_e*)/d(intercepts) via implicit diff == unrolled scan == central FD."""
    inp = _golden_inputs()
    intercepts0 = inp.intercepts

    def _with(ic):
        return inp._replace(intercepts=ic)

    # Grad of T* and of log10(n_e*) for the two coordinates' scales.
    for coord, scale in ((0, lambda z: z[0]), (1, lambda z: jnp.log10(z[1]))):
        f_root = lambda ic: scale(fixed_point_custom_root(_with(ic)))  # noqa: E731
        f_vjp = lambda ic: scale(fixed_point_custom_vjp(_with(ic)))  # noqa: E731
        f_scan = lambda ic: scale(solve_forward(_with(ic)))  # noqa: E731

        g_root = np.asarray(jax.grad(f_root)(intercepts0))
        g_vjp = np.asarray(jax.grad(f_vjp)(intercepts0))
        g_scan = np.asarray(jax.grad(f_scan)(intercepts0))

        # Implicit routes agree with the unrolled-autodiff oracle to ~fp64.
        assert np.allclose(g_root, g_scan, rtol=1e-6, atol=1e-9), coord
        assert np.allclose(g_vjp, g_scan, rtol=1e-6, atol=1e-9), coord

        # Central finite differences on the (real) element axis, rtol <= 1e-4.
        eps = 1e-5
        fd = np.zeros_like(g_scan)
        for i in range(3):  # only the 3 real elements (padding grad == 0)
            p = intercepts0.at[i].add(eps)
            m = intercepts0.at[i].add(-eps)
            fd[i] = (float(f_scan(p)) - float(f_scan(m))) / (2 * eps)
        assert np.allclose(g_root[:3], fd[:3], rtol=1e-4, atol=1e-6), (coord, g_root[:3], fd[:3])
        assert np.all(np.isfinite(g_root))
        assert np.all(np.isfinite(g_vjp))


def test_implicit_diff_backward_is_loop_free():
    """The custom_vjp backward rule contains no scan/while (O(1)-in-iters memory).

    Structural guarantee of the IFT route: the cotangent pullback solves a fixed
    2x2 adjoint system at the root, never re-traversing the forward iteration.
    We assert this by jaxpr inspection of the backward rule.
    """
    inp = _golden_inputs()
    _, vjp = jax.vjp(fixed_point_custom_vjp, inp)
    jaxpr = jax.make_jaxpr(vjp)(jnp.array([1.0, 0.0]))
    prim_names = {str(e.primitive) for e in jaxpr.jaxpr.eqns}
    # No looping primitives in the backward pass (memory independent of iters).
    assert "scan" not in prim_names
    assert "while" not in prim_names


# --------------------------------------------------------------------------
# 2. Reference-anchored physics — bit-identical Saha ratio vs the FROZEN oracle.
# --------------------------------------------------------------------------


def test_saha_ratio_bit_identical_to_reference():
    """Spike Saha ratio == reference ``iterative._saha_ratio_per_element`` (imported)."""
    from cflibs.inversion.solve.iterative import _saha_ratio_per_element

    T_K = jnp.asarray(9000.0)
    n_e = jnp.asarray(1.3e16)
    U_I = jnp.asarray([25.0, 10.0, 6.0])
    U_II = jnp.asarray([30.0, 12.0, 5.0])
    ip = jnp.asarray([7.9, 6.0, 5.1])

    ref = np.asarray(_saha_ratio_per_element(T_K, n_e, U_I, U_II, ip))
    spike = np.asarray(ad._saha_ratio(T_K, n_e, U_I, U_II, ip))
    # Bit-identical: same constants, same expression.
    assert np.array_equal(ref, spike)


# --------------------------------------------------------------------------
# 3. Soft / hard relaxation — tau -> 0 recovers the scoreboard exactly (§6.4).
# --------------------------------------------------------------------------


def test_soft_presence_recovers_hard_as_tau_to_zero():
    C = jnp.array([0.001, 0.004, 0.006, 0.5, 0.0, 0.02])
    soft = np.asarray(soft_presence(C, tau=1e-7))
    hard = np.asarray(hard_presence(C))
    assert np.allclose(soft, hard, atol=1e-6)


def test_soft_top_k_recovers_hard_selection_and_no_nan():
    # Padding has the highest raw score but must be excluded by the mask.
    scores = jnp.array([5.0, 1.0, 3.0, 2.0, 4.0, 99.0])
    mask = jnp.array([True, True, True, True, True, False])
    tk = np.asarray(soft_top_k(scores, 3.0, tau=1e-3, mask=mask))
    assert np.all(np.isfinite(tk))  # no (-inf)-(-inf) NaN
    # The 3 largest VALID scores (5, 4, 3 -> indices 0, 4, 2) selected.
    expected = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    assert np.allclose(tk, expected, atol=1e-3)


def test_soft_f1_recovers_hard_f1_as_tau_to_zero():
    pred = jnp.array([0.5, 0.001, 0.02, 0.3, 0.004])
    truth = jnp.array([1.0, 0.0, 1.0, 0.0, 1.0])
    soft = float(soft_f1(pred, truth, tau=1e-7))
    hard = float(hard_f1(pred, truth))
    assert abs(soft - hard) < 1e-6


# --------------------------------------------------------------------------
# 4. Knob gradient — one backward pass, FD-verified on the 3-knob slice (AC2).
# --------------------------------------------------------------------------


def _knob_batch(B: int = 16, E: int = 6) -> dict:
    rng = np.random.default_rng(0)
    return {
        "pred_conc": jnp.asarray(rng.uniform(0.0, 0.4, (B, E))),
        "truth_present": jnp.asarray((rng.uniform(0, 1, (B, E)) > 0.5).astype(float)),
        "snr": jnp.asarray(rng.uniform(5.0, 30.0, (B, E))),
        "element_mask": jnp.asarray(np.ones((B, E), dtype=bool)),
        "cand_scores": jnp.asarray(rng.uniform(0.0, 1.0, (B, E))),
    }


def test_knob_gradient_fd_verified():
    """One backward pass gives d(soft-F1 loss)/d(3 knobs), FD-agreeing."""
    batch = _knob_batch()
    knobs = KnobSlice(
        presence_cutoff=jnp.asarray(0.005),
        min_snr=jnp.asarray(10.0),
        top_k=jnp.asarray(4.0),
    )
    tau = 5e-2
    loss, grad = knob_gradient(knobs, batch, tau=tau)
    assert np.isfinite(float(loss))

    def L(k):
        return float(knob_objective(k, batch, tau=tau))

    eps = 1e-5
    checks = {
        "presence_cutoff": (float(grad.presence_cutoff), knobs.presence_cutoff),
        "min_snr": (float(grad.min_snr), knobs.min_snr),
        "top_k": (float(grad.top_k), knobs.top_k),
    }
    for name, (g, val) in checks.items():
        kp = knobs._replace(**{name: val + eps})
        km = knobs._replace(**{name: val - eps})
        fd = (L(kp) - L(km)) / (2 * eps)
        assert np.isfinite(g)
        # Larger atol for the structurally-flat SNR knob slope (~1e-7 magnitude).
        assert np.isclose(g, fd, rtol=2e-3, atol=1e-6), (name, g, fd)


def test_knob_gradient_descent_reduces_soft_objective():
    """A few gradient steps on the continuous knobs reduce the soft-F1 loss (§6.3 demo)."""
    batch = _knob_batch()
    knobs = KnobSlice(
        presence_cutoff=jnp.asarray(0.05),
        min_snr=jnp.asarray(20.0),
        top_k=jnp.asarray(2.0),
    )
    tau = 5e-2
    lr = jnp.asarray([1e-4, 1e-2, 1e-1])  # per-knob step (scales differ)
    loss0, _ = knob_gradient(knobs, batch, tau=tau)
    for _ in range(25):
        _, grad = knob_gradient(knobs, batch, tau=tau)
        knobs = KnobSlice(
            presence_cutoff=jnp.clip(
                knobs.presence_cutoff - lr[0] * grad.presence_cutoff, 1e-4, 0.2
            ),
            min_snr=jnp.clip(knobs.min_snr - lr[1] * grad.min_snr, 0.0, 40.0),
            top_k=jnp.clip(knobs.top_k - lr[2] * grad.top_k, 1.0, 6.0),
        )
    loss1, _ = knob_gradient(knobs, batch, tau=tau)
    assert float(loss1) <= float(loss0) + 1e-9


# --------------------------------------------------------------------------
# 5. vmap smoke (batch 16) + grad finiteness.
# --------------------------------------------------------------------------


def _batched_inputs(B: int = 16, E: int = 4) -> FixedPointInputs:
    rng = np.random.default_rng(2)
    return FixedPointInputs(
        intercepts=jnp.asarray(rng.uniform(-3.0, 1.0, (B, E))),
        ip_eV=jnp.tile(jnp.asarray([7.9, 6.0, 5.1, 0.0]), (B, 1)),
        U_I=jnp.tile(jnp.asarray([25.0, 10.0, 6.0, 1.0]), (B, 1)),
        U_II=jnp.tile(jnp.asarray([30.0, 12.0, 5.0, 1.0]), (B, 1)),
        element_mask=jnp.tile(jnp.asarray([True, True, True, False]), (B, 1)),
        pressure_pa=jnp.full((B,), 101325.0),
    )


def test_vmap_batch16_and_grad_finite():
    inp = _batched_inputs(16)
    z_root = jax.jit(jax.vmap(fixed_point_custom_root))(inp)
    z_vjp = jax.jit(jax.vmap(fixed_point_custom_vjp))(inp)
    assert z_root.shape == (16, 2)
    assert np.all(np.isfinite(np.asarray(z_root)))
    assert np.allclose(np.asarray(z_root), np.asarray(z_vjp), rtol=1e-6)

    # Batched grad of sum(T*) w.r.t. the continuous intercepts leaf is finite.
    ic0 = inp.intercepts

    def total_T(ic):
        b = inp._replace(intercepts=ic)
        return jnp.sum(jax.vmap(lambda x: fixed_point_custom_root(x)[0])(b))

    g = jax.jit(jax.grad(total_T))(ic0)
    assert g.shape == (16, 4)
    assert np.all(np.isfinite(np.asarray(g)))


# --------------------------------------------------------------------------
# 6. No-SQLite-in-kernel guard.
# --------------------------------------------------------------------------


def test_kernel_imports_no_sqlite_or_host():
    """The autodiff kernel must not pull in sqlite3 / atomic.database / jitpipe.host."""
    import importlib
    import sys

    for mod in ("sqlite3", "cflibs.atomic.database", "cflibs.jitpipe.host"):
        sys.modules.pop(mod, None)
    # Re-import the kernel fresh and assert no SQLite-touching module loaded.
    importlib.reload(ad)
    assert "sqlite3" not in sys.modules, "kernel imported sqlite3"
    assert "cflibs.atomic.database" not in sys.modules, "kernel imported atomic.database"
    assert "cflibs.jitpipe.host" not in sys.modules, "kernel imported jitpipe.host"
    # Source-level guard: no banned strings in the kernel module file.
    import cflibs.jitpipe.autodiff as fresh

    src = open(fresh.__file__).read()
    assert "import sqlite3" not in src
    assert "atomic.database" not in src
    assert "jitpipe.host" not in src
    assert "jitpipe import host" not in src


# --------------------------------------------------------------------------
# 7. Padding-invariance — next pad size => bit-identical on the valid region.
# --------------------------------------------------------------------------


def test_padding_invariance_fixed_point_and_grad():
    """z* and its implicit gradient are bit-identical at pad=4 vs pad=8."""
    inp4 = _golden_inputs(pad=4)
    inp8 = _golden_inputs(pad=8)

    z4 = np.asarray(fixed_point_custom_root(inp4))
    z8 = np.asarray(fixed_point_custom_root(inp8))
    # Same converged (T*, n_e*) regardless of padding width.
    assert np.array_equal(z4, z8)

    # Gradient on the real (first 3) element axis is identical too.
    def g_of(inp):
        f = lambda ic: fixed_point_custom_root(inp._replace(intercepts=ic))[0]  # noqa: E731
        return np.asarray(jax.grad(f)(inp.intercepts))

    g4 = g_of(inp4)
    g8 = g_of(inp8)
    assert np.array_equal(g4[:3], g8[:3])
    # Padding entries carry zero gradient.
    assert np.allclose(g4[3:], 0.0)
    assert np.allclose(g8[3:], 0.0)


def test_soft_f1_padding_invariance():
    """soft-F1 is bit-identical when extra padded (masked) elements are appended."""
    pred = np.array([0.5, 0.001, 0.02, 0.3])
    truth = np.array([1.0, 0.0, 1.0, 0.0])
    mask = np.array([True, True, True, True])

    f_small = float(
        soft_f1(jnp.asarray(pred), jnp.asarray(truth), tau=1e-2, mask=jnp.asarray(mask))
    )
    # Append two padded entries with arbitrary junk values + mask=False.
    predP = np.concatenate([pred, [0.9, 0.0]])
    truthP = np.concatenate([truth, [1.0, 0.0]])
    maskP = np.concatenate([mask, [False, False]])
    f_big = float(
        soft_f1(jnp.asarray(predP), jnp.asarray(truthP), tau=1e-2, mask=jnp.asarray(maskP))
    )
    assert f_small == f_big
