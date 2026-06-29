"""Structured-Jacobian K=1 Gauss-Newton inversion for the ExoJAX-grade CF-LIBS
reference forward model.

Reuses the STRUCTURED strategy from /tmp/varpro_bench.py (the 354-408 us config):
  - exploit that the spectrum S is LINEAR in the N_elem concentrations:
        S = B(T, ne) @ comp,  dS/dcomp = B   (returned by forward_with_basis, NO AD)
  - only dS/dT and dS/dlog_ne are nonlinear -> 2 JVP columns through `forward`
  - assemble the (G, N_params) Jacobian and take K Gauss-Newton steps with
    Marquardt (relative, diagonal) damping -- scale-invariant across T(K), log_ne,
    and the softmax-raw concentration directions.

theta = [T(K), log10(ne), raw_0 .. raw_{Nelem-1}],  composition = softmax(raw).
Operates on a PEAK-NORMALIZED target so the GN conditioning is identical in every
parameter direction (the varpro_bench fairness convention).

inversion(spectrum, bundle, instr, ...) -> {composition, T, ne, theta, latency}.
"""

from __future__ import annotations

import time
import statistics

import numpy as np
import jax
import jax.numpy as jnp
from jax import jit

LAMBDA = 1e-2        # Marquardt relative (diagonal) damping
DIAG_EPS = 1e-9      # absolute SPD floor


def make_structured_inversion(fwd_bundle, k=1, lam=LAMBDA):
    """Build a jit-compiled structured-GN step closure for a given forward bundle.

    fwd_bundle: dict from make_reference_forward(...).
    Returns a jitted fn(theta0, target_norm) -> theta.
    """
    forward_with_basis = fwd_bundle["forward_with_basis"]
    conc_from_theta = fwd_bundle["conc_from_theta"]
    n_params = 2 + fwd_bundle["n_elem"]
    # finite-difference steps for the 2 nonlinear directions
    DT = 25.0          # K
    DLOGNE = 0.01      # dex

    def step(th, target):
        T = th[0]
        ne = 10.0 ** th[1]
        conc = conc_from_theta(th)
        B = forward_with_basis(T, ne)                 # (G, E)  basis (ExoJAX opacity)
        S0 = B @ conc                                 # current spectrum
        r = S0 - target                               # residual

        # 2 nonlinear columns (T, log_ne) via FORWARD DIFFERENCES on the ExoJAX
        # forward (= basis @ conc). We deliberately AVOID jvp through hjert: its
        # custom_jvp + Faddeeva branch makes a single JVP ~3.4 ms (9x a forward),
        # which alone blows the latency budget. Two extra basis evals (~0.4 ms each)
        # give the same Gauss-Newton columns at a fraction of the cost.
        B_T = forward_with_basis(T + DT, ne)
        col_T = (B_T @ conc - S0) / DT
        B_ne = forward_with_basis(T, 10.0 ** (th[1] + DLOGNE))
        col_ne = (B_ne @ conc - S0) / DLOGNE

        # N_elem concentration columns WITHOUT AD: dS/draw = B @ Jsoftmax
        Jsoft = jnp.diag(conc) - jnp.outer(conc, conc)  # (E, E)
        cols_conc = B @ Jsoft                           # (G, E)

        J = jnp.concatenate([col_T[:, None], col_ne[:, None], cols_conc], axis=1)
        JtJ = J.T @ J
        A = (
            JtJ
            + lam * jnp.diag(jnp.diag(JtJ))
            + DIAG_EPS * jnp.eye(n_params, dtype=th.dtype)
        )
        return th - jnp.linalg.solve(A, J.T @ r)

    def run(theta0, target):
        th = theta0
        for _ in range(k):
            th = step(th, target)
        return th

    return jit(run)


def _peak_norm(y):
    y = jnp.asarray(y, dtype=jnp.float32)
    return y / (jnp.max(y) + 1e-30)


def invert(
    spectrum,
    fwd_bundle,
    theta0=None,
    T0=9000.0,
    logne0=17.0,
    k=1,
    lam=LAMBDA,
    n_timing=120,
    warmup=5,
    runner=None,
):
    """Invert one PEAK-NORMALIZED spectrum -> composition + plasma params.

    Returns dict: composition (E,), T, ne, theta, latency_us {median,min,mean}.
    Pass a pre-built `runner` (from make_structured_inversion) to reuse compilation.
    """
    n_elem = fwd_bundle["n_elem"]
    n_params = 2 + n_elem
    target = _peak_norm(spectrum)
    if theta0 is None:
        theta0 = jnp.asarray(
            np.concatenate([[T0, logne0], np.zeros(n_elem)]).astype(np.float32)
        )
    if runner is None:
        runner = make_structured_inversion(fwd_bundle, k=k, lam=lam)

    # warmup (trigger compile) + timing (strict single-shot, block_until_ready each)
    for _ in range(warmup):
        jax.block_until_ready(runner(theta0, target))
    samples = []
    out = None
    for _ in range(n_timing):
        t0 = time.perf_counter()
        out = runner(theta0, target)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - t0) * 1e6)

    theta = np.asarray(out)
    comp = np.asarray(fwd_bundle["conc_from_theta"](jnp.asarray(theta)))
    return dict(
        composition=comp,
        T=float(theta[0]),
        ne=float(10.0 ** theta[1]),
        theta=theta,
        latency_us=dict(
            median=statistics.median(samples),
            min=min(samples),
            mean=statistics.mean(samples),
        ),
        n_params=n_params,
        k=k,
    )
