#!/usr/bin/env python
"""J9/M2 batched-inversion GPU benchmark for the ``cflibs.jitpipe`` JAX port.

ADR-0004 §1.2 (the GPU payoff) / M2: the jittable inversion units are
fixed-shape and ``vmap``-clean, so a single compiled graph can score a whole
*batch* of spectra at once. This script measures that batched throughput
(spectra/s) on **whatever JAX backend is active** — CPU locally, a V100S on the
cluster — so the same file CPU-verifies here and runs under ``--gres=gpu:1``
there. It does NOT hardcode CPU; it reads :func:`jax.devices` and runs on the
backend JAX selects.

What it times (each via ``jit(vmap(...))`` over ``B`` synthetic spectra):

(a) **Forward model** — :func:`cflibs.jitpipe.forward_id.evaluate_population`,
    which is ``vmap(cflibs.radiation.kernels.forward_model)`` over a batched
    plasma. This is the manifold-style baseline (spectra generated per device
    call) and the per-spectrum-equivalent throughput floor.

(b) **J10 forward-fitting population scoring** — the GPU-payoff identifier
    (ADR-0004 §1.2): for each measured spectrum, score a *shared* population of
    candidate configs (``evaluate_population`` -> :func:`correlation_cost`
    -> :func:`bic_cost` -> :func:`forward_fit_presence_scores`). The population
    is forwarded once; the per-spectrum cost is the ``B``-way batched scoring of
    that population against ``B`` measured spectra. This is the 10^3-10^4
    forward-eval-per-spectrum stage where the GPU earns its keep.

(c) **Solve** — :func:`cflibs.jitpipe.solve.scan_solve` (fixed-K lax.scan
    initializer) and :func:`cflibs.jitpipe.solve.joint_wls_solve` (joint WLS GN
    production estimator), each ``jit(vmap(...))`` over a batch of perturbed
    padded observation blocks (the J7 ``test_vmap_batched_scan`` pattern).

Front-end (detect/identify) is intentionally SKIPPED: the per-spectrum
candidate-set gather is host-side (the documented host<->device seam in
``snapshot.py`` / ``host.py``), so a clean device ``vmap`` over the front-end
would need the host gather inside the trace. Per the J9 spec this is an explicit
PARTIAL demo with the front-end host-side; see the printed SKIPPED note.

Determinism: synthetic spectra are generated from the frozen reference kernel
with a fixed RNG seed, so the run is reproducible. fp64 throughout (mandatory,
ADR-0004 §5.3).

Memory note (V100S = 32 GB): the default ``--batch-size 64`` is modest. The
Voigt/forward path's intermediates scale with ``B * N_lines * N_wl``; very large
batches (the forward path OOMs near ``B ~ 2048`` on a 32 GB V100S) should be
swept upward cautiously.

Usage
-----
Local CPU verification (small batch, asserts the vmap == loop sanity check)::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$PWD \
        .venv/bin/python scripts/jitpipe_gpu_bench.py --batch-size 8

Cluster V100S::

    srun --gres=gpu:1 --cpus-per-task=4 --mem=24G --time=00:20:00 bash -lc \
      'cd <repo> && JAX_ENABLE_X64=1 PYTHONPATH=$PWD \
       <cflibs-gpu-python> scripts/jitpipe_gpu_bench.py --batch-size 256'
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

# Enable float64 BEFORE any JAX import (mandatory: ADR-0004 §5.3). Do not force a
# platform here — JAX_PLATFORMS (if set by the caller) selects the backend; the
# default lets JAX pick the GPU when present.
os.environ.setdefault("JAX_ENABLE_X64", "True")

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cflibs.atomic.database import AtomicDatabase
from cflibs.core.jax_runtime import _ensure_pytrees_registered
from cflibs.instrument.model import InstrumentModel
from cflibs.inversion.solve import iterative as iterative_mod
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver, LineObservation
from cflibs.jitpipe import forward_id as ff
from cflibs.jitpipe.snapshot import PipelineSnapshot
from cflibs.jitpipe.solve import LaxKernelInputs, joint_wls_solve, scan_solve
from cflibs.radiation.kernels import BroadeningMode

# ---------------------------------------------------------------------------
# Fixed problem definition (deterministic; a handful of elements, T~9000 K).
# ---------------------------------------------------------------------------

ELEMENTS = ["Fe", "Ca", "Ti"]
WL_RANGE = (300.0, 420.0)
DEFAULT_DB = "ASD_da/libs_production.db"
T_NOMINAL_K = 9000.0
NE_NOMINAL_CM3 = 2.0e16
PATH_LENGTH_M = 0.01
FWHM_NM = 0.1
RNG_SEED = 20260612
N_POPULATION = 256  # candidate configs forwarded once for the J10 scoring stage.


def _block_until_ready(tree):
    """Force device computation to finish so timing is honest (async dispatch)."""
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return tree


def _time_unit(name, fn, *, warmup=True):
    """Compile (warmup) then time one device call; return (seconds, result).

    JAX dispatches asynchronously, so we ``block_until_ready`` on every leaf
    before stopping the clock. The warmup call pays the (one-time) XLA compile
    so the timed call measures execution, not compilation.
    """
    if warmup:
        t0 = time.perf_counter()
        _block_until_ready(fn())
        compile_s = time.perf_counter() - t0
    else:
        compile_s = float("nan")
    t0 = time.perf_counter()
    result = _block_until_ready(fn())
    run_s = time.perf_counter() - t0
    return name, compile_s, run_s, result


# ---------------------------------------------------------------------------
# Setup: real atomic snapshot + instrument + wavelength grid.
# ---------------------------------------------------------------------------


def build_setup(db_path, grid_size):
    """Build (snapshot, wavelengths, instrument) from the real atomic DB."""
    _ensure_pytrees_registered()
    db = AtomicDatabase(db_path)
    asnap = db.snapshot(elements=ELEMENTS, wavelength_range=WL_RANGE, include_levels=True)
    snap = PipelineSnapshot.from_atomic_snapshot(asnap)
    wl = jnp.linspace(WL_RANGE[0], WL_RANGE[1], grid_size, dtype=jnp.float64)
    instr = InstrumentModel(resolution_fwhm_nm=FWHM_NM)
    return snap, wl, instr


# ---------------------------------------------------------------------------
# Batched synthetic spectra (the measured batch) from the frozen kernel.
# ---------------------------------------------------------------------------


def build_spectrum_batch(snap, wl, instr, batch_size):
    """Deterministic batch of ``B`` measured spectra + their batched plasma.

    Compositions/T/n_e are drawn once (fixed seed) on the host; the batched
    plasma pytree is assembled via the documented host seam
    (:func:`forward_id.stack_plasma_states`) and forwarded on-device by
    :func:`forward_id.evaluate_population`.

    Returns
    -------
    batched_plasma : SingleZoneLTEPlasma
        Batched plasma pytree (leading axis ``B``).
    measured : ndarray, shape (B, N_wl)
        Synthetic measured spectra (frozen kernel, the oracle).
    """
    rng = np.random.default_rng(RNG_SEED)
    e = len(snap.element_symbols)
    temps = rng.uniform(T_NOMINAL_K * 0.85, T_NOMINAL_K * 1.15, batch_size)
    nes = NE_NOMINAL_CM3 * 10.0 ** rng.uniform(-0.3, 0.3, batch_size)
    comps = rng.random((batch_size, e))
    comps /= comps.sum(axis=1, keepdims=True)

    batched_plasma = ff.stack_plasma_states(temps, nes, comps, tuple(snap.element_symbols))
    measured = ff.evaluate_population(
        batched_plasma,
        snap,
        instr,
        wl,
        broadening_mode=BroadeningMode.NIST_PARITY,
        path_length_m=PATH_LENGTH_M,
    )
    return batched_plasma, _block_until_ready(measured)


# ---------------------------------------------------------------------------
# Unit (a): forward model — jit(vmap(forward_model)) over a batched plasma.
# ---------------------------------------------------------------------------


def make_forward_unit(snap, wl, instr, batched_plasma):
    """``jit`` the batched forward model (manifold-style baseline)."""

    @jax.jit
    def _forward(plasma):
        return ff.evaluate_population(
            plasma,
            snap,
            instr,
            wl,
            broadening_mode=BroadeningMode.NIST_PARITY,
            path_length_m=PATH_LENGTH_M,
        )

    return lambda: _forward(batched_plasma)


# ---------------------------------------------------------------------------
# Unit (b): J10 forward-fit population scoring — the GPU payoff (ADR-0004 §1.2).
# A shared candidate population is forwarded once; the batched, per-measured
# scoring (correlation + BIC + presence call) is what scales with B.
# ---------------------------------------------------------------------------


def make_forward_id_unit(snap, wl, instr, measured, *, n_population):
    """``jit(vmap(...))`` the J10 device scoring core over the measured batch.

    Builds one candidate population (host seam) and forwards it once. The timed
    unit is ``vmap`` over the ``B`` measured spectra of:
    ``correlation_cost -> bic_cost -> forward_fit_presence_scores`` against the
    shared population's spectra — the multi-line coherence test ADR-0004 §1.2
    points at the GPU for.
    """
    element_symbols = tuple(snap.element_symbols)
    key = jax.random.PRNGKey(RNG_SEED)
    pop = ff.build_candidate_population(key, element_symbols, n_configs=int(n_population))
    pop_plasma = ff.stack_plasma_states(
        pop["temperatures_k"],
        pop["electron_densities"],
        pop["concentrations"],
        element_symbols,
    )
    pop_spectra = _block_until_ready(
        ff.evaluate_population(
            pop_plasma,
            snap,
            instr,
            wl,
            broadening_mode=BroadeningMode.NIST_PARITY,
            path_length_m=PATH_LENGTH_M,
        )
    )
    membership = pop["membership"]
    config_valid = pop["config_valid"]
    n_params = pop["n_params"]
    element_valid = jnp.ones((len(element_symbols),), dtype=pop_spectra.dtype)

    def _score_one(meas):
        corr = ff.correlation_cost(pop_spectra, meas)
        bic = ff.bic_cost(pop_spectra, meas, n_params=n_params)
        return ff.forward_fit_presence_scores(
            corr, bic, membership, config_valid, element_valid, presence_threshold=0.05
        )

    scorer = jax.jit(jax.vmap(_score_one))
    return lambda: scorer(measured), _score_one, measured


# ---------------------------------------------------------------------------
# Unit (c): solve — jit(vmap(scan_solve / joint_wls_solve)) over obs blocks.
# One LaxKernelInputs is built from a real DB-backed solver + a synthetic obs
# block; the batch perturbs the Boltzmann y-values (the J7 vmap pattern).
# ---------------------------------------------------------------------------


def _make_solve_obs():
    """Multi-element neutral+ionic Boltzmann observations (~T=9000 K)."""
    from cflibs.core.constants import EV_TO_K, SAHA_CONST_CM3

    t_ev = T_NOMINAL_K / EV_TO_K
    ip = 7.6
    saha_offset = float(np.log((SAHA_CONST_CM3 / NE_NOMINAL_CM3) * (t_ev**1.5)))
    intercepts = {"Fe": 10.0, "Ca": 9.5, "Ti": 9.0}
    rng = np.random.default_rng(RNG_SEED)
    obs = []
    for el, intercept in intercepts.items():
        for e_k in (1.0, 2.5, 4.0, 5.5):
            y = intercept - e_k / t_ev
            intensity = float(np.exp(y) * (1.0 + rng.normal(0.0, 0.005)))
            obs.append(
                LineObservation(
                    wavelength_nm=400.0,
                    intensity=intensity / 400.0,
                    intensity_uncertainty=max(intensity * 0.005 / 400.0, 1e-8),
                    element=el,
                    ionization_stage=1,
                    E_k_ev=e_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )
        for e_k in (3.0, 4.0):
            y = intercept + saha_offset - (ip + e_k) / t_ev
            intensity = float(np.exp(y) * (1.0 + rng.normal(0.0, 0.005)))
            obs.append(
                LineObservation(
                    wavelength_nm=400.0,
                    intensity=intensity / 400.0,
                    intensity_uncertainty=max(intensity * 0.005 / 400.0, 1e-8),
                    element=el,
                    ionization_stage=2,
                    E_k_ev=e_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )
    return obs


def build_solve_inputs(db_path):
    """Build one LaxKernelInputs from a real DB-backed solver + synthetic obs."""
    db = AtomicDatabase(db_path)
    solver = IterativeCFLIBSSolver(db, max_iterations=20)
    obs = _make_solve_obs()
    obs_by = {el: [o for o in obs if o.element == el] for el in {o.element for o in obs}}
    elements_ord, x, y, w, stage, mask = iterative_mod._build_padded_arrays_from_obs(
        obs_by, weight_cap=solver.boltzmann_weight_cap
    )
    snapshot = iterative_mod._AtomicSnapshot.from_solver(solver, elements_ord)
    if elements_ord != list(obs_by.keys()):
        snapshot = snapshot.reorder(elements_ord)
    inp = LaxKernelInputs.from_snapshot(snapshot, x, y, w, stage, mask)
    return solver, inp, np.asarray(y)


def make_solve_batch(inp, y_base, batch_size):
    """Batch of perturbed Boltzmann y-blocks (one LaxKernelInputs per row)."""
    rng = np.random.default_rng(RNG_SEED)
    y_batch = jnp.asarray(
        np.stack([y_base + rng.normal(0.0, 0.01, size=y_base.shape) for _ in range(batch_size)])
    )
    return y_batch


def make_scan_solve_unit(inp, solver, y_batch):
    """``jit(vmap(scan_solve))`` over the perturbed y-block batch."""

    def _run_one(y_one):
        return scan_solve(
            inp._replace(y=y_one),
            max_iters=20,
            closure_mode="standard",
            t_tol_k=solver.t_tolerance_k,
            ne_tol_frac=solver.ne_tolerance_frac,
            pressure_pa=solver.pressure_pa,
            min_r2=solver.min_boltzmann_r2,
        )

    unit = jax.jit(jax.vmap(_run_one))
    return lambda: unit(y_batch)


def make_joint_wls_unit(inp, y_batch):
    """``jit(vmap(joint_wls_solve))`` over the perturbed y-block batch."""

    def _run_one(y_one):
        return joint_wls_solve(
            inp._replace(y=y_one),
            init_T_K=T_NOMINAL_K,
            n_gn_steps=3,
            closure_mode="standard",
        )

    unit = jax.jit(jax.vmap(_run_one))
    return lambda: unit(y_batch)


# ---------------------------------------------------------------------------
# CPU sanity assertion: vmap'd unit == per-item (loop) result.
# ---------------------------------------------------------------------------


def sanity_check_forward_id(score_one, measured):
    """Assert the vmap'd J10 scorer matches a per-item Python loop (no nonsense).

    Scores each measured spectrum individually through the SAME ``_score_one``
    closure and compares to the batched ``vmap`` output, element by element.
    """
    vmapped = jax.vmap(score_one)(measured)
    n = int(measured.shape[0])
    loop_present = np.stack([np.asarray(score_one(measured[i]).element_present) for i in range(n)])
    loop_score = np.stack([np.asarray(score_one(measured[i]).presence_score) for i in range(n)])
    batch_present = np.asarray(vmapped.element_present)
    batch_score = np.asarray(vmapped.presence_score)
    np.testing.assert_array_equal(batch_present, loop_present)
    # presence_score can be -inf for padded/never-included elements; compare the
    # finite entries exactly and require the inf-pattern to match.
    np.testing.assert_array_equal(np.isfinite(batch_score), np.isfinite(loop_score))
    finite = np.isfinite(batch_score)
    np.testing.assert_allclose(batch_score[finite], loop_score[finite], rtol=1e-12, atol=0.0)
    return True


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Number of spectra B per batched device call (default 64; V100S "
        "has 32GB — the forward path OOMs near B~2048).",
    )
    p.add_argument(
        "--grid-size",
        type=int,
        default=600,
        help="Wavelength grid points N_wl (default 600).",
    )
    p.add_argument(
        "--population-size",
        type=int,
        default=N_POPULATION,
        help="Candidate population size for the J10 forward-fit scoring stage "
        f"(default {N_POPULATION}; forwarded once, shared across the batch).",
    )
    p.add_argument("--db", default=DEFAULT_DB, help=f"Atomic DB path (default {DEFAULT_DB}).")
    p.add_argument(
        "--no-sanity",
        action="store_true",
        help="Skip the vmap==loop CPU sanity assertion (timing only).",
    )
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    b = int(args.batch_size)
    n_wl = int(args.grid_size)
    n_pop = int(args.population_size)

    devices = jax.devices()
    backend = jax.default_backend()
    print("=" * 72)
    print("J9/M2 jitpipe batched-inversion GPU benchmark")
    print("=" * 72)
    print(f"JAX version      : {jax.__version__}")
    print(f"JAX backend      : {backend}")
    print(f"JAX devices      : {devices}")
    print(f"x64 enabled      : {jax.config.jax_enable_x64}")
    print(f"batch size B     : {b}")
    print(f"grid size N_wl   : {n_wl}")
    print(f"population size  : {n_pop}")
    print(f"elements         : {ELEMENTS}")
    print(f"atomic DB        : {args.db}")
    print("-" * 72)

    if not Path(args.db).exists():
        raise SystemExit(f"atomic DB not found: {args.db} (run from the repo root)")

    snap, wl, instr = build_setup(args.db, n_wl)
    batched_plasma, measured = build_spectrum_batch(snap, wl, instr, b)

    results = []  # (name, compile_s, run_s, spectra_per_s)

    # (a) forward model baseline.
    fwd = make_forward_unit(snap, wl, instr, batched_plasma)
    name, c, r, _ = _time_unit("forward_model (a)", fwd)
    results.append((name, c, r, b / r if r > 0 else float("inf")))

    # (b) J10 forward-fit population scoring — the GPU payoff.
    fid, score_one, meas = make_forward_id_unit(snap, wl, instr, measured, n_population=n_pop)
    name, c, r, _ = _time_unit("forward_id score (b)", fid)
    results.append((name, c, r, b / r if r > 0 else float("inf")))

    # (c) solve: scan_solve + joint_wls_solve.
    solver, inp, y_base = build_solve_inputs(args.db)
    y_batch = make_solve_batch(inp, y_base, b)
    scan_unit = make_scan_solve_unit(inp, solver, y_batch)
    name, c, r, _ = _time_unit("scan_solve (c)", scan_unit)
    results.append((name, c, r, b / r if r > 0 else float("inf")))

    jwls_unit = make_joint_wls_unit(inp, y_batch)
    name, c, r, _ = _time_unit("joint_wls_solve (c)", jwls_unit)
    results.append((name, c, r, b / r if r > 0 else float("inf")))

    # Report.
    print(f"{'unit':<24}{'compile s':>12}{'run s':>12}{'spectra/s':>16}")
    print("-" * 72)
    total_run = 0.0
    for name, c, r, sps in results:
        total_run += r
        print(f"{name:<24}{c:>12.4f}{r:>12.5f}{sps:>16.1f}")
    print("-" * 72)
    print(f"{'TOTAL (timed run)':<24}{'':>12}{total_run:>12.5f}{b / total_run:>16.1f}")
    print("-" * 72)
    print("BATCHED units : forward_model, forward_id score, scan_solve, joint_wls_solve")
    print(
        "SKIPPED units : detect / identify front-end (per-spectrum candidate-set "
        "gather is\n                host-side — the documented host<->device seam; "
        "J9 is an explicit\n                PARTIAL demo with the front-end host-side)."
    )

    # CPU sanity assertion (vmap == per-item loop) — proves the bench isn't
    # measuring nonsense. Cheap; runs by default.
    if not args.no_sanity:
        ok = sanity_check_forward_id(score_one, meas)
        print("-" * 72)
        print(f"SANITY (vmap == loop, forward_id score): {'PASS' if ok else 'FAIL'}")

    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
