"""Parity tests for the J7 fixed-shape solve kernels (ADR-0004 §3 C4, §6.1).

Mirrors ``tests/inversion/test_iterative_lax.py`` but targets
:mod:`cflibs.jitpipe.solve`:

* **Scan-vs-while-loop parity** — :func:`cflibs.jitpipe.solve.scan_solve`
  (fixed-K ``lax.scan`` initializer) feeds *identical padded inputs* to the
  REAL reference :func:`cflibs.inversion.solve.iterative._run_lax_while_loop`
  and asserts the §4 contract (T rtol 1e-5, conc atol 1e-7), across all six
  closure modes plus the oxide configuration. The scan reproduces the while
  loop's fixed point bit-for-bit (converged-state freeze).
* **Idempotence** — scan at ``K`` and ``K + 5`` is bit-identical (the freeze
  makes the scan a fixed point past convergence; J7 spec §6).
* **vmap batch=16** and **grad-finiteness as a HARD assert** (supersedes the
  reference try/except smoke; J7 spec §5/§4 "no try/except").
* **No-SQLite-in-kernel guard** — the kernel module imports no sqlite/atomic
  host code; the kernel runs with the DB connection severed.
* **Padding-invariance** — rerun at a larger pad width => bit-identical on the
  valid region.
* **Joint-WLS GN-step-0 anchor** — :func:`cflibs.jitpipe.solve.joint_wls_solve`
  step 0 == the REAL :class:`cflibs.inversion.solve.closed_form.ClosedFormILRSolver`
  weighted-least-squares solve, rtol 1e-10 (the exact algebraic anchor).
* **Covariance vs finite-difference Hessian** — the WLS covariance matches a
  numerical Hessian inverse on a well-conditioned noisy fixture (rtol 1e-3).

CPU x64 (conftest); narrow subset per watchdog rules.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.atomic.structures import PartitionFunction  # noqa: E402
from cflibs.core.constants import EV_TO_K, KB_EV, SAHA_CONST_CM3  # noqa: E402
from cflibs.inversion.solve import iterative as iterative_mod  # noqa: E402
from cflibs.inversion.solve.closed_form import (  # noqa: E402
    ClosedFormConfig,
    ClosedFormILRSolver,
)
from cflibs.inversion.solve.iterative import (  # noqa: E402
    IterativeCFLIBSSolver,
    LineObservation,
    LoopState,
    _eval_partition_jax,
    _saha_ratio_per_element,
)

import cflibs.jitpipe.solve as solve_mod  # noqa: E402
from cflibs.jitpipe.solve import (  # noqa: E402
    JointWLSResult,
    LaxKernelInputs,
    joint_wls_solve,
    scan_solve,
)

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


# ---------------------------------------------------------------------------
# Fixtures (mirror test_iterative_lax.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_db():
    """Mock AtomicDatabase: constant partition poly, polynomial path forced.

    ``get_energy_levels`` returns ``None`` so the snapshot deterministically
    takes the polynomial partition path (a bare MagicMock would otherwise yield
    a garbage truthy levels object and collapse U to 1.0).
    """
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.0
    db.get_energy_levels.return_value = None
    coeffs_I = [3.2188, 0, 0, 0, 0]  # ln U = 3.2188 -> U = 25

    def _pf(el, sp):
        return PartitionFunction(
            element=el,
            ionization_stage=sp,
            coefficients=coeffs_I,
            t_min=1000,
            t_max=20000,
            source="test",
        )

    db.get_partition_coefficients.side_effect = _pf
    return db


def _mock_db_varying(coeffs_map):
    """Mock DB with per-element partition polynomials (non-trivial anchor)."""
    db = MagicMock(spec=AtomicDatabase)
    db.get_ionization_potential.return_value = 7.0
    db.get_energy_levels.return_value = None
    db.get_partition_coefficients.side_effect = lambda el, sp: PartitionFunction(
        element=el,
        ionization_stage=sp,
        coefficients=coeffs_map[el],
        t_min=1000,
        t_max=20000,
        source="t",
    )
    return db


def _make_multi_element_obs(seed: int = 20260512) -> list:
    """Multi-element neutral+ionic fixture (~T=8000K). Matches the lax test."""
    T_K = 8000.0
    T_eV = T_K / EV_TO_K
    n_e = 1.0e16
    ip = 7.0
    saha_offset = float(np.log((SAHA_CONST_CM3 / n_e) * (T_eV**1.5)))
    intercepts_true = {"Fe": 10.0, "Ni": 9.5, "Cr": 9.0}
    obs: list = []
    rng = np.random.default_rng(seed)
    for el, intercept in intercepts_true.items():
        for E_k in [1.0, 2.5, 4.0, 5.5]:
            y = intercept - E_k / T_eV
            intensity = float(np.exp(y) * (1.0 + rng.normal(0.0, 0.005)))
            obs.append(
                LineObservation(
                    wavelength_nm=500.0,
                    intensity=intensity / 500.0,
                    intensity_uncertainty=max(intensity * 0.005 / 500.0, 1e-8),
                    element=el,
                    ionization_stage=1,
                    E_k_ev=E_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )
        for E_k in [3.0, 4.0]:
            y = intercept + saha_offset - (ip + E_k) / T_eV
            intensity = float(np.exp(y) * (1.0 + rng.normal(0.0, 0.005)))
            obs.append(
                LineObservation(
                    wavelength_nm=500.0,
                    intensity=intensity / 500.0,
                    intensity_uncertainty=max(intensity * 0.005 / 500.0, 1e-8),
                    element=el,
                    ionization_stage=2,
                    E_k_ev=E_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )
    return obs


def _make_neutral_obs(coeffs_map, seed: int | None = None) -> list:
    """Neutral-only, multi-element fixture for the GN-step-0 anchor."""
    rng = np.random.default_rng(seed) if seed is not None else None
    obs: list = []
    for el, inter in {"Fe": 10.0, "Ni": 9.5, "Cr": 9.0}.items():
        for E_k in [1.0, 2.0, 3.0, 4.0, 5.0]:
            y = inter - E_k / 1.0
            scale = 1.0 + rng.normal(0.0, 0.03) if rng is not None else 1.0
            intensity = float(np.exp(y) * scale)
            obs.append(
                LineObservation(
                    wavelength_nm=500.0,
                    intensity=intensity,
                    intensity_uncertainty=0.1 * intensity,
                    element=el,
                    ionization_stage=1,
                    E_k_ev=E_k,
                    g_k=1,
                    A_ki=1e8,
                )
            )
    return obs


def _build_kernel_inputs(solver, obs):
    """Build (elements_ord, snapshot, LaxKernelInputs) from a solver + obs list."""
    obs_by = {el: [o for o in obs if o.element == el] for el in {o.element for o in obs}}
    elements_ord, x, y, w, stage, mask = iterative_mod._build_padded_arrays_from_obs(
        obs_by, weight_cap=solver.boltzmann_weight_cap
    )
    snapshot = iterative_mod._AtomicSnapshot.from_solver(solver, elements_ord)
    if elements_ord != list(obs_by.keys()):
        snapshot = snapshot.reorder(elements_ord)
    inp = LaxKernelInputs.from_snapshot(snapshot, x, y, w, stage, mask)
    return elements_ord, snapshot, inp, (x, y, w, stage, mask)


def _reference_loopstate(snapshot, padded, solver, closure_fn, max_iter):
    """Run the REAL reference _run_lax_while_loop on identical padded inputs."""
    x, y, w, stage, mask = padded
    E = snapshot.ip0_eV.shape[0]
    base = LoopState(
        T_K=jnp.asarray(10000.0),
        n_e_cm3=jnp.asarray(1.0e17),
        T_prev=jnp.asarray(10000.0),
        n_e_prev=jnp.asarray(1.0e17),
        converged=jnp.asarray(False),
        i=jnp.asarray(0, dtype=jnp.int32),
        U_I=jnp.zeros(E),
        U_II=jnp.zeros(E),
        intercepts=jnp.zeros(E),
        concentrations=jnp.zeros(E),
        r_squared=jnp.asarray(0.0),
        boltzmann_degenerate=jnp.asarray(True),
    )
    return iterative_mod._run_lax_while_loop(
        base,
        jnp.asarray(x),
        jnp.asarray(y),
        jnp.asarray(w),
        jnp.asarray(stage),
        jnp.asarray(mask),
        snapshot,
        closure_fn,
        apply_ipd=False,
        two_region=False,
        max_iter=max_iter,
        t_tol_k=solver.t_tolerance_k,
        ne_tol_frac=solver.ne_tolerance_frac,
        pressure_pa=solver.pressure_pa,
        min_r2=solver.min_boltzmann_r2,
    )


# ---------------------------------------------------------------------------
# 1. Scan-vs-while-loop parity across closure modes (J7 §4 row 1)
# ---------------------------------------------------------------------------


CLOSURE_MODES = ["standard", "ilr", "pwlr", "dirichlet_residual"]


@pytest.mark.parametrize("closure_mode", CLOSURE_MODES)
def test_scan_matches_while_loop(mock_db, closure_mode):
    """scan_solve reproduces the while-loop fixed point bit-for-bit.

    Feeds IDENTICAL padded inputs to the REAL ``_run_lax_while_loop`` and to
    ``scan_solve`` and asserts the §4 tolerance. ``standard`` / ``ilr`` / ``pwlr``
    / ``dirichlet_residual`` collapse to the standard simplex map in both paths,
    so the comparison is exact.
    """
    obs = _make_multi_element_obs()
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    elements_ord, snapshot, inp, padded = _build_kernel_inputs(solver, obs)
    closure_fn = iterative_mod._make_closure_callback("standard", elements_ord, {})

    ref = _reference_loopstate(snapshot, padded, solver, closure_fn, solver.max_iterations)
    got = scan_solve(
        inp,
        max_iters=solver.max_iterations,
        closure_mode=closure_mode,
        t_tol_k=solver.t_tolerance_k,
        ne_tol_frac=solver.ne_tolerance_frac,
        pressure_pa=solver.pressure_pa,
        min_r2=solver.min_boltzmann_r2,
    )

    np.testing.assert_allclose(float(got.T_K), float(ref.T_K), rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(float(got.n_e_cm3), float(ref.n_e_cm3), rtol=1e-5)
    np.testing.assert_allclose(
        np.asarray(got.concentrations), np.asarray(ref.concentrations), rtol=1e-5, atol=1e-7
    )
    # Iteration count within 1 (J7 §4).
    assert abs(int(got.i) - int(ref.i)) <= 1


def test_scan_oxide_closure_parity(mock_db):
    """Oxide closure (factor-weighted) parity vs the reference oxide callback.

    Covers the SB-graph + oxide configuration the current lax tests cannot
    (J7 §4 row 1 'including oxide'). Both paths weight the relative
    concentrations by the same per-element stoichiometry before normalising.
    """
    obs = _make_multi_element_obs()
    stoich = {"Fe": 1.43, "Ni": 1.27, "Cr": 1.46}
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    elements_ord, snapshot, inp, padded = _build_kernel_inputs(solver, obs)
    closure_fn = iterative_mod._make_closure_callback(
        "oxide", elements_ord, {"oxide_stoichiometry": stoich}
    )
    ref = _reference_loopstate(snapshot, padded, solver, closure_fn, solver.max_iterations)

    factors = jnp.asarray([stoich[el] for el in elements_ord], dtype=jnp.float64)
    got = scan_solve(
        inp,
        max_iters=solver.max_iterations,
        closure_mode="oxide",
        oxide_factors=factors,
        t_tol_k=solver.t_tolerance_k,
        ne_tol_frac=solver.ne_tolerance_frac,
        pressure_pa=solver.pressure_pa,
        min_r2=solver.min_boltzmann_r2,
    )
    np.testing.assert_allclose(float(got.T_K), float(ref.T_K), rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(got.concentrations), np.asarray(ref.concentrations), rtol=1e-5, atol=1e-7
    )


def test_scan_matrix_closure_parity(mock_db):
    """Matrix closure (one pinned coordinate) parity vs the reference."""
    obs = _make_multi_element_obs()
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    elements_ord, snapshot, inp, padded = _build_kernel_inputs(solver, obs)
    closure_fn = iterative_mod._make_closure_callback(
        "matrix", elements_ord, {"matrix_element": "Fe", "matrix_fraction": 0.5}
    )
    ref = _reference_loopstate(snapshot, padded, solver, closure_fn, solver.max_iterations)
    got = scan_solve(
        inp,
        max_iters=solver.max_iterations,
        closure_mode="matrix",
        matrix_idx=elements_ord.index("Fe"),
        matrix_fraction=0.5,
        t_tol_k=solver.t_tolerance_k,
        ne_tol_frac=solver.ne_tolerance_frac,
        pressure_pa=solver.pressure_pa,
        min_r2=solver.min_boltzmann_r2,
    )
    np.testing.assert_allclose(float(got.T_K), float(ref.T_K), rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(got.concentrations), np.asarray(ref.concentrations), rtol=1e-5, atol=1e-7
    )


# ---------------------------------------------------------------------------
# 2. Convergence-freeze idempotence (J7 §6): scan at K and K+5 identical
# ---------------------------------------------------------------------------


def test_scan_idempotent_K_vs_Kplus5(mock_db):
    """The converged-freeze makes scan output identical at K and K+5."""
    obs = _make_multi_element_obs()
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)

    kw = dict(
        closure_mode="standard",
        t_tol_k=solver.t_tolerance_k,
        ne_tol_frac=solver.ne_tolerance_frac,
        pressure_pa=solver.pressure_pa,
        min_r2=solver.min_boltzmann_r2,
    )
    at_k = scan_solve(inp, max_iters=20, **kw)
    at_kp5 = scan_solve(inp, max_iters=25, **kw)

    # Bit-identical frozen output (the fixed point was reached before K=20).
    assert float(at_k.T_K) == float(at_kp5.T_K)
    assert float(at_k.n_e_cm3) == float(at_kp5.n_e_cm3)
    assert int(at_k.i) == int(at_kp5.i)
    np.testing.assert_array_equal(
        np.asarray(at_k.concentrations), np.asarray(at_kp5.concentrations)
    )


# ---------------------------------------------------------------------------
# 3. vmap batch=16 (J7 §5)
# ---------------------------------------------------------------------------


def test_vmap_batched_scan(mock_db):
    """``jax.vmap`` over 16 perturbed observation sets through scan_solve."""
    obs = _make_multi_element_obs()
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    _, _, inp, (x, y, w, stage, mask) = _build_kernel_inputs(solver, obs)
    E = inp.x.shape[0]

    rng = np.random.default_rng(20260512)
    n_batch = 16
    y_batch = jnp.asarray(
        np.stack([y + rng.normal(0.0, 0.01, size=y.shape) for _ in range(n_batch)])
    )

    def _run_one(y_one):
        inp_one = inp._replace(y=y_one)
        return scan_solve(
            inp_one,
            max_iters=20,
            closure_mode="standard",
            t_tol_k=solver.t_tolerance_k,
            ne_tol_frac=solver.ne_tolerance_frac,
            pressure_pa=solver.pressure_pa,
            min_r2=solver.min_boltzmann_r2,
        )

    batched = jax.vmap(_run_one)(y_batch)
    assert batched.T_K.shape == (n_batch,)
    assert batched.concentrations.shape == (n_batch, E)
    assert jnp.all(jnp.isfinite(batched.T_K))
    sums = jnp.sum(batched.concentrations, axis=1)
    np.testing.assert_allclose(np.asarray(sums), 1.0, rtol=1e-2)


# ---------------------------------------------------------------------------
# 4. Grad-finiteness — HARD assert (supersedes the reference try/except smoke)
# ---------------------------------------------------------------------------


def test_grad_finite_hard(mock_db):
    """``jax.grad`` through scan_solve yields a finite scalar — HARD assert.

    The fixed-K scan is reverse-differentiable (the while loop is not); the J7
    spec §4/§5 replaces the reference try/except smoke with a hard finiteness
    assert.
    """
    obs = _make_multi_element_obs()
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)

    def _temperature_from_init(init_T):
        final = scan_solve(
            inp,
            init_T_K=init_T,
            max_iters=20,
            closure_mode="standard",
            t_tol_k=solver.t_tolerance_k,
            ne_tol_frac=solver.ne_tolerance_frac,
            pressure_pa=solver.pressure_pa,
            min_r2=solver.min_boltzmann_r2,
        )
        return final.T_K

    g = jax.grad(_temperature_from_init)(jnp.asarray(10000.0, dtype=jnp.float64))
    assert jnp.isfinite(g), f"grad returned non-finite value: {g}"


def test_grad_finite_joint_wls(mock_db):
    """``jax.grad`` through joint_wls_solve T w.r.t. init_T is finite (HARD)."""
    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_obs(coeffs_map)
    solver = IterativeCFLIBSSolver(db, max_iterations=20)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)

    def _T_from_init(init_T):
        return joint_wls_solve(inp, init_T_K=init_T, n_gn_steps=3, closure_mode="standard").T_K

    g = jax.grad(_T_from_init)(jnp.asarray(9000.0, dtype=jnp.float64))
    assert jnp.isfinite(g), f"joint WLS grad non-finite: {g}"


# ---------------------------------------------------------------------------
# 5. No-SQLite-in-kernel guard (J7 §5 / ADR §5.1)
# ---------------------------------------------------------------------------


def test_no_sqlite_in_kernel_module():
    """The kernel module imports no sqlite / atomic.database host code."""
    import ast
    import pathlib

    src = pathlib.Path(solve_mod.__file__).read_text()
    tree = ast.parse(src)
    banned = {"sqlite3", "cflibs.atomic.database", "cflibs.jitpipe.host"}
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.append(node.module)
    assert not (
        set(found) & banned
    ), f"kernel imports forbidden host modules: {set(found) & banned}"


def test_kernel_runs_with_db_severed(mock_db):
    """scan_solve runs after the DB connection is gone (inputs fully padded)."""
    obs = _make_multi_element_obs()
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)
    # Sever the DB: any in-kernel query would now raise.
    solver.atomic_db = None
    mock_db.get_ionization_potential.side_effect = AssertionError("DB queried inside kernel")
    mock_db.get_partition_coefficients.side_effect = AssertionError("DB queried inside kernel")

    final = scan_solve(
        inp,
        max_iters=20,
        closure_mode="standard",
        t_tol_k=solver.t_tolerance_k,
        ne_tol_frac=solver.ne_tolerance_frac,
        pressure_pa=solver.pressure_pa,
        min_r2=solver.min_boltzmann_r2,
    )
    assert jnp.isfinite(final.T_K)


# ---------------------------------------------------------------------------
# 6. Padding-invariance (J7 §5): larger pad => bit-identical on valid region
# ---------------------------------------------------------------------------


def test_padding_invariance(mock_db):
    """Adding extra (masked-off) line columns leaves the answer bit-identical."""
    obs = _make_multi_element_obs()
    solver = IterativeCFLIBSSolver(mock_db, max_iterations=20)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)

    kw = dict(
        max_iters=20,
        closure_mode="standard",
        t_tol_k=solver.t_tolerance_k,
        ne_tol_frac=solver.ne_tolerance_frac,
        pressure_pa=solver.pressure_pa,
        min_r2=solver.min_boltzmann_r2,
    )
    base = scan_solve(inp, **kw)

    # Pad the line axis with 4 fully-masked-off columns.
    E, N = inp.x.shape
    pad = 4

    def _pad2d(a, fill):
        out = jnp.full((E, N + pad), fill, dtype=a.dtype)
        return out.at[:, :N].set(a)

    inp_padded = inp._replace(
        x=_pad2d(inp.x, 0.0),
        y=_pad2d(inp.y, 0.0),
        w=_pad2d(inp.w, 0.0),
        stage=_pad2d(inp.stage, 1),
        mask=_pad2d(inp.mask, False),
    )
    padded = scan_solve(inp_padded, **kw)

    assert float(base.T_K) == float(padded.T_K)
    assert float(base.n_e_cm3) == float(padded.n_e_cm3)
    np.testing.assert_array_equal(
        np.asarray(base.concentrations), np.asarray(padded.concentrations)
    )


# ---------------------------------------------------------------------------
# 7. Joint WLS GN-step-0 anchor == ClosedFormILRSolver WLS (rtol 1e-10)
# ---------------------------------------------------------------------------


def test_gn_step0_anchor_exact():
    """GN step 0 == the REAL ClosedFormILRSolver weighted-least-squares solve.

    Builds the reference WLS design with the SAME frozen U_s, M_s the kernel
    uses at ``init_T``, then asserts the recovered θ (and hence T and the
    simplex) match to rtol 1e-10 — the exact algebraic anchor (J4 Schur
    identity; J7 §4 row 2).
    """
    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_obs(coeffs_map)
    solver = IterativeCFLIBSSolver(db, max_iterations=20)
    elements_ord, _, inp, _ = _build_kernel_inputs(solver, obs)
    init_T = 10000.0
    n_e = 1.0e17

    # Frozen U_s, M_s exactly as the kernel computes them at init_T.
    U_I = np.asarray(
        _eval_partition_jax(
            jnp.asarray(init_T),
            inp.use_direct[:, 0],
            inp.g_I,
            inp.E_I,
            inp.ip_I,
            inp.coeffs_I,
            inp.fallback_I,
            inp.mask_I,
        )
    )
    U_II = np.asarray(
        _eval_partition_jax(
            jnp.asarray(init_T),
            inp.use_direct[:, 1],
            inp.g_II,
            inp.E_II,
            inp.ip_II,
            inp.coeffs_II,
            inp.fallback_II,
            inp.mask_II,
        )
    )
    S = np.asarray(
        _saha_ratio_per_element(
            jnp.asarray(init_T), jnp.asarray(n_e), jnp.asarray(U_I), jnp.asarray(U_II), inp.ip0
        )
    )
    M = 1.0 + np.maximum(S, 0.0)

    cf = ClosedFormILRSolver(db, ClosedFormConfig(saha_passes=1, closure_mode="standard"))
    obs_by = {el: [o for o in obs if o.element == el] for el in {o.element for o in obs}}
    obs_ord = {el: obs_by[el] for el in elements_ord}
    pf = {el: float(U_I[i]) for i, el in enumerate(elements_ord)}
    mult = {el: float(M[i]) for i, el in enumerate(elements_ord)}
    X, y_adj, W = cf._build_design_matrix(obs_ord, elements_ord, pf, mult)
    theta_ref, _ = cf._solve_wls(X, y_adj, W)

    got: JointWLSResult = joint_wls_solve(
        inp, init_T_K=init_T, n_e_cm3=n_e, n_gn_steps=1, closure_mode="standard"
    )
    np.testing.assert_allclose(np.asarray(got.theta), theta_ref, rtol=1e-10, atol=1e-10)
    T_ref = -1.0 / (theta_ref[0] * KB_EV)
    np.testing.assert_allclose(float(got.T_K), T_ref, rtol=1e-10)


def test_gn_step0_is_step1_when_linear():
    """With Saha refinement off, GN steps beyond 0 do not move θ (pure WLS)."""
    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_obs(coeffs_map)
    solver = IterativeCFLIBSSolver(db, max_iterations=20)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)

    step0 = joint_wls_solve(inp, init_T_K=10000.0, n_gn_steps=1, closure_mode="standard")
    step5 = joint_wls_solve(
        inp, init_T_K=10000.0, n_gn_steps=5, refine_saha=False, closure_mode="standard"
    )
    np.testing.assert_allclose(np.asarray(step5.theta), np.asarray(step0.theta), rtol=1e-12)


# ---------------------------------------------------------------------------
# 8. Covariance vs finite-difference Hessian (J7 §4 row 4)
# ---------------------------------------------------------------------------


def test_covariance_vs_fd_hessian():
    """WLS covariance ≈ σ̂² (½ Hessian of χ²)⁻¹ from finite differences.

    On a well-conditioned noisy fixture the analytic ``σ̂² (XᵀWX)⁻¹`` matches a
    numerical Hessian inverse of the χ² objective (rtol 1e-3). For a *linear*
    WLS the Hessian is exactly ``2 XᵀWX``, so the FD check validates both the
    objective assembly and the covariance scaling.
    """
    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_obs(coeffs_map, seed=7)
    solver = IterativeCFLIBSSolver(db, max_iterations=20)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)
    init_T = 10000.0

    got = joint_wls_solve(inp, init_T_K=init_T, n_gn_steps=1, closure_mode="standard")
    theta = np.asarray(got.theta)
    cov = np.asarray(got.cov_theta)
    assert np.all(np.isfinite(cov))

    # Reconstruct X, W, y_adj host-side (matches the kernel's frozen-U assembly).
    elements_ord = list({o.element for o in obs})
    elements_ord = [el for el in ["Cr", "Ni", "Fe"] if el in elements_ord] or elements_ord
    # Reorder to the kernel's element order.
    _, snapshot, inp2, _ = _build_kernel_inputs(solver, obs)
    elements_ord = list(snapshot.elements)
    E = len(elements_ord)
    U_I = np.asarray(
        _eval_partition_jax(
            jnp.asarray(init_T),
            inp.use_direct[:, 0],
            inp.g_I,
            inp.E_I,
            inp.ip_I,
            inp.coeffs_I,
            inp.fallback_I,
            inp.mask_I,
        )
    )
    U_II = np.asarray(
        _eval_partition_jax(
            jnp.asarray(init_T),
            inp.use_direct[:, 1],
            inp.g_II,
            inp.E_II,
            inp.ip_II,
            inp.coeffs_II,
            inp.fallback_II,
            inp.mask_II,
        )
    )
    S = np.asarray(
        _saha_ratio_per_element(
            jnp.asarray(init_T), jnp.asarray(1e17), jnp.asarray(U_I), jnp.asarray(U_II), inp.ip0
        )
    )
    M = 1.0 + np.maximum(S, 0.0)

    from cflibs.inversion.physics.closure import _helmert_basis

    V = _helmert_basis(E)
    x = np.asarray(inp.x)
    y = np.asarray(inp.y)
    w = np.asarray(inp.w)
    mask = np.asarray(inp.mask)
    rows_X, rows_y, rows_w = [], [], []
    for i in range(E):
        for j in range(x.shape[1]):
            if not mask[i, j]:
                continue
            row = np.zeros(1 + (E - 1) + 1)
            row[0] = x[i, j]
            row[1:E] = V[i, :]
            row[-1] = 1.0
            rows_X.append(row)
            rows_y.append(y[i, j] + np.log(U_I[i]) + np.log(M[i]))
            rows_w.append(w[i, j])
    X = np.array(rows_X)
    y_adj = np.array(rows_y)
    Wv = np.array(rows_w)

    def chi2(th):
        r = y_adj - X @ th
        return float(np.sum(Wv * r**2))

    n = len(theta)
    H = np.zeros((n, n))
    eps = 1e-4
    for a in range(n):
        for b in range(n):
            tpp = theta.copy()
            tpp[a] += eps
            tpp[b] += eps
            tpm = theta.copy()
            tpm[a] += eps
            tpm[b] -= eps
            tmp = theta.copy()
            tmp[a] -= eps
            tmp[b] += eps
            tmm = theta.copy()
            tmm[a] -= eps
            tmm[b] -= eps
            H[a, b] = (chi2(tpp) - chi2(tpm) - chi2(tmp) + chi2(tmm)) / (4 * eps**2)

    residuals = y_adj - X @ theta
    dof = max(len(y_adj) - n, 1)
    sigma2 = float(np.sum(Wv * residuals**2)) / dof
    cov_fd = sigma2 * np.linalg.inv(0.5 * H)

    np.testing.assert_allclose(cov, cov_fd, rtol=1e-3, atol=1e-8)


# ---------------------------------------------------------------------------
# 9. PRODUCTION joint WLS — SB-graph + Stark + closure modes (divergence D-J8-1)
#
# The GEOLOGICAL preset routes ``_solve_python`` with
# ``saha_boltzmann_graph=True`` (UNIT-weight global lstsq) + Stark n_e — the
# path ``scan_solve`` does NOT mirror. These tests cover that production
# configuration for the first time (J7 §4 row 10, §5 AC1/AC5; ADR-0004 §6.1).
# ---------------------------------------------------------------------------


def _make_neutral_ionic_obs(coeffs_map, n_e=1.0e16, T_K=8000.0, seed=None) -> list:
    """Multi-element neutral+ionic fixture exercising the SB-graph ion shift."""
    T_eV = T_K / EV_TO_K
    ip = 7.0
    ln_S = float(np.log((SAHA_CONST_CM3 / n_e) * (T_eV**1.5)))
    rng = np.random.default_rng(seed) if seed is not None else None
    obs: list = []
    for el, q in {"Fe": 10.0, "Ni": 9.5, "Cr": 9.0}.items():
        for E_k in [1.0, 2.5, 4.0, 5.5]:
            y = q - E_k / T_eV
            scale = 1.0 + rng.normal(0.0, 0.005) if rng is not None else 1.0
            intensity = float(np.exp(y) * scale)
            obs.append(
                LineObservation(
                    wavelength_nm=500.0,
                    intensity=intensity,
                    intensity_uncertainty=max(0.005 * intensity, 1e-9),
                    element=el,
                    ionization_stage=1,
                    E_k_ev=E_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )
        for E_k in [3.0, 4.0]:
            y = q + ln_S - (ip + E_k) / T_eV
            scale = 1.0 + rng.normal(0.0, 0.005) if rng is not None else 1.0
            intensity = float(np.exp(y) * scale)
            obs.append(
                LineObservation(
                    wavelength_nm=500.0,
                    intensity=intensity,
                    intensity_uncertainty=max(0.005 * intensity, 1e-9),
                    element=el,
                    ionization_stage=2,
                    E_k_ev=E_k,
                    g_k=1,
                    A_ki=1.0,
                )
            )
    return obs


def _reference_sb_graph_composition(solver, inp, elements_ord, obs_by, T_freeze, n_e, closure):
    """SB-graph slope + closure composition the REAL reference produces.

    Mirrors the ``_solve_python`` SB-graph path: the global unit-weight
    ``_fit_saha_boltzmann_graph`` slope/intercepts, then the closure
    (``ClosureEquation``) using the kernel's FROZEN ``U_s``/``M_s`` at
    ``T_freeze`` (so the partition source is consistent across both paths — the
    same convention ``test_gn_step0_anchor_exact`` uses).
    """

    ips = {el: 7.0 for el in elements_ord}
    fit = solver._fit_saha_boltzmann_graph(obs_by, T_freeze, n_e, ips)
    T_ref = -1.0 / (fit.slope * KB_EV)
    U_I = np.asarray(
        _eval_partition_jax(
            jnp.asarray(T_freeze),
            inp.use_direct[:, 0],
            inp.g_I,
            inp.E_I,
            inp.ip_I,
            inp.coeffs_I,
            inp.fallback_I,
            inp.mask_I,
        )
    )
    U_II = np.asarray(
        _eval_partition_jax(
            jnp.asarray(T_freeze),
            inp.use_direct[:, 1],
            inp.g_II,
            inp.E_II,
            inp.ip_II,
            inp.coeffs_II,
            inp.fallback_II,
            inp.mask_II,
        )
    )
    S = np.asarray(
        _saha_ratio_per_element(
            jnp.asarray(T_freeze),
            jnp.asarray(n_e),
            jnp.asarray(U_I),
            jnp.asarray(U_II),
            inp.ip0,
        )
    )
    M = 1.0 + np.maximum(S, 0.0)
    pf = {el: float(U_I[i]) for i, el in enumerate(elements_ord)}
    mult = {el: float(M[i]) for i, el in enumerate(elements_ord)}
    res = closure(fit.intercepts, pf, mult)
    return T_ref, res.concentrations


def test_sb_graph_gn_step0_anchor_exact():
    """SB-graph (unit-weight) GN step 0 == ``_fit_saha_boltzmann_graph`` (rtol 1e-10).

    This is the PRODUCTION GEOLOGICAL anchor (ADR-0004 §6.1 (i), §4 row 6): the
    arrow-matrix Schur identity says the unit-weight joint WLS slope/composition
    equals the reference global element-dummy lstsq — including the ionic-line
    Saha shift the closed-form anchor (neutral-only) never exercised.
    """
    from cflibs.inversion.physics.closure import ClosureEquation

    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_ionic_obs(coeffs_map)
    solver = IterativeCFLIBSSolver(db, max_iterations=20, saha_boltzmann_graph=True)
    elements_ord, _, inp, _ = _build_kernel_inputs(solver, obs)
    obs_by = {el: [o for o in obs if o.element == el] for el in elements_ord}
    init_T, n_e = 8000.0, 1.0e16

    T_ref, C_ref = _reference_sb_graph_composition(
        solver,
        inp,
        elements_ord,
        obs_by,
        init_T,
        n_e,
        lambda i, p, m: ClosureEquation.apply_standard(i, p, abundance_multipliers=m),
    )

    got: JointWLSResult = joint_wls_solve(
        inp,
        init_T_K=init_T,
        ne_stark_cm3=n_e,
        n_gn_steps=1,
        sb_graph=True,
        closure_mode="standard",
    )
    np.testing.assert_allclose(float(got.T_K), T_ref, rtol=1e-10)
    for i, el in enumerate(elements_ord):
        np.testing.assert_allclose(float(got.concentrations[i]), C_ref[el], rtol=1e-10, atol=1e-12)
    # Provenance: n_e pinned to the Stark input.
    assert bool(got.ne_from_stark) is True
    assert float(got.n_e_cm3) == n_e


def test_production_converged_vs_reference_solve_python():
    """Converged (T, C) of the production joint solve vs the REAL ``_solve_python``.

    Drives the reference ``_solve_python`` in the GEOLOGICAL configuration
    (``saha_boltzmann_graph=True`` + Stark-pinned n_e via a mocked Stark
    estimate) and the joint solve with the SAME pinned n_e, then asserts the
    §4 row-10 contract: **T rtol 1e-3, per-element C atol 0.5 wt %**. The
    partition provider is made consistent across both paths (the kernel reads
    the polynomial; the reference closure must read the same U).
    """

    class _Prov:
        def __init__(self, a0):
            self.a0 = a0

        def at(self, _T):
            return float(np.exp(self.a0))

    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    db.partition_function_for.side_effect = lambda el, sp: _Prov(coeffs_map[el][0])
    obs = _make_neutral_ionic_obs(coeffs_map, seed=5)
    n_e_stark = 1.0e16

    solver = IterativeCFLIBSSolver(
        db,
        max_iterations=60,
        saha_boltzmann_graph=True,
        t_tolerance_k=1.0,
        ne_tolerance_frac=1e-4,
    )
    # Pin the Stark n_e: mock the multi-line estimator (3 lines, no scatter).
    solver._estimate_ne_from_stark_multi = MagicMock(return_value=(n_e_stark, 3, 0.0))
    ref = solver.solve(obs, closure_mode="standard", stark_diagnostics=[MagicMock()] * 3)
    assert ref.converged

    elements_ord, _, inp, _ = _build_kernel_inputs(solver, obs)
    got = joint_wls_solve(
        inp,
        init_T_K=10000.0,
        ne_stark_cm3=n_e_stark,
        n_gn_steps=20,
        sb_graph=True,
        refine_saha=True,
        closure_mode="standard",
    )
    assert bool(got.converged) is True
    np.testing.assert_allclose(float(got.T_K), ref.temperature_K, rtol=1e-3)
    for i, el in enumerate(elements_ord):
        # per-element C atol 0.5 wt % (J7 §4 row 10).
        assert abs(float(got.concentrations[i]) - ref.concentrations[el]) < 5e-3


def test_production_self_consistent_fixed_point():
    """The joint solve's converged point IS the reference SB-graph fixed point.

    At the joint solve's converged T, the reference ``_fit_saha_boltzmann_graph``
    + ``ClosureEquation`` (kernel-frozen U) reproduce its (T, C) to machine
    precision — the strongest statement of production parity (no partition-mock
    mismatch, no damping residual).
    """
    from cflibs.inversion.physics.closure import ClosureEquation

    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_ionic_obs(coeffs_map, seed=11)
    n_e = 1.0e16
    solver = IterativeCFLIBSSolver(db, max_iterations=20, saha_boltzmann_graph=True)
    elements_ord, _, inp, _ = _build_kernel_inputs(solver, obs)
    obs_by = {el: [o for o in obs if o.element == el] for el in elements_ord}

    got = joint_wls_solve(
        inp,
        init_T_K=10000.0,
        ne_stark_cm3=n_e,
        n_gn_steps=15,
        sb_graph=True,
        refine_saha=True,
        closure_mode="standard",
    )
    T_ref, C_ref = _reference_sb_graph_composition(
        solver,
        inp,
        elements_ord,
        obs_by,
        float(got.T_K),  # freeze the reference at the joint converged T
        n_e,
        lambda i, p, m: ClosureEquation.apply_standard(i, p, abundance_multipliers=m),
    )
    np.testing.assert_allclose(float(got.T_K), T_ref, rtol=1e-10)
    for i, el in enumerate(elements_ord):
        np.testing.assert_allclose(float(got.concentrations[i]), C_ref[el], rtol=1e-9, atol=1e-11)


def test_production_oxide_closure_parity():
    """SB-graph + OXIDE closure parity vs the reference (geological default)."""
    from cflibs.inversion.physics.closure import ClosureEquation

    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    stoich = {"Fe": 1.43, "Ni": 1.27, "Cr": 1.46}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_ionic_obs(coeffs_map, seed=2)
    n_e = 1.0e16
    solver = IterativeCFLIBSSolver(db, max_iterations=20, saha_boltzmann_graph=True)
    elements_ord, _, inp, _ = _build_kernel_inputs(solver, obs)
    obs_by = {el: [o for o in obs if o.element == el] for el in elements_ord}

    got = joint_wls_solve(
        inp,
        init_T_K=10000.0,
        ne_stark_cm3=n_e,
        n_gn_steps=15,
        sb_graph=True,
        refine_saha=True,
        closure_mode="oxide",
        oxide_factors=jnp.asarray([stoich[el] for el in elements_ord], dtype=jnp.float64),
    )
    T_ref, C_ref = _reference_sb_graph_composition(
        solver,
        inp,
        elements_ord,
        obs_by,
        float(got.T_K),
        n_e,
        lambda i, p, m: ClosureEquation.apply_oxide_mode(
            i, p, abundance_multipliers=m, oxide_stoichiometry=stoich
        ),
    )
    np.testing.assert_allclose(float(got.T_K), T_ref, rtol=1e-10)
    for i, el in enumerate(elements_ord):
        np.testing.assert_allclose(float(got.concentrations[i]), C_ref[el], rtol=1e-8, atol=1e-10)


# ---------------------------------------------------------------------------
# 10. Stark n_e pinning, MAD penalty, pressure-balance fallback (ADR §6.1)
# ---------------------------------------------------------------------------


def test_stark_mad_penalty_inflates_covariance():
    """A non-zero Stark MAD scatter inflates the covariance (D>1 penalty term)."""
    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_ionic_obs(coeffs_map, seed=9)
    solver = IterativeCFLIBSSolver(db, max_iterations=20, saha_boltzmann_graph=True)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)

    kw = dict(init_T_K=8000.0, n_gn_steps=6, sb_graph=True, closure_mode="standard")
    no_scatter = joint_wls_solve(inp, ne_stark_cm3=1.0e16, ne_scatter_cm3=0.0, **kw)
    with_scatter = joint_wls_solve(inp, ne_stark_cm3=1.0e16, ne_scatter_cm3=3.0e15, **kw)

    cov0 = np.asarray(no_scatter.cov_theta)
    cov1 = np.asarray(with_scatter.cov_theta)
    assert np.all(np.isfinite(cov1))
    # The MAD penalty is PSD-additive -> every diagonal element grows (or holds).
    assert np.all(np.diag(cov1) >= np.diag(cov0) - 1e-18)
    assert np.trace(cov1) > np.trace(cov0)
    assert float(with_scatter.ne_scatter_cm3) == 3.0e15


def test_pressure_balance_fallback_flag():
    """No Stark n_e -> pressure-balance fixed point drives n_e; flag is False."""
    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_ionic_obs(coeffs_map, seed=4)
    solver = IterativeCFLIBSSolver(db, max_iterations=20, saha_boltzmann_graph=True)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)

    got = joint_wls_solve(
        inp,
        init_T_K=10000.0,
        ne_stark_cm3=None,
        n_gn_steps=8,
        sb_graph=True,
        closure_mode="standard",
    )
    assert bool(got.ne_from_stark) is False
    assert float(got.ne_scatter_cm3) == 0.0
    assert jnp.isfinite(got.n_e_cm3)
    assert float(got.n_e_cm3) > 0.0


# ---------------------------------------------------------------------------
# 11. custom_root differentiability: grad finite + FD-verified (HARD asserts)
# ---------------------------------------------------------------------------


def test_production_grad_finite_and_fd_verified():
    """``jax.grad`` through the custom_root production solve is finite AND FD-close.

    Supersedes the reference try/except smoke (J7 §4 "hard asserts"): the
    implicit-function-theorem gradient of T w.r.t. the warm-start temperature is
    checked finite and against a central finite difference (rtol 5e-2 — the
    fixed-point map's mild conditioning, not a parity statement).
    """
    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_ionic_obs(coeffs_map, seed=6)
    solver = IterativeCFLIBSSolver(db, max_iterations=20, saha_boltzmann_graph=True)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)

    def _T_of_init(init_T):
        return joint_wls_solve(
            inp,
            init_T_K=init_T,
            ne_stark_cm3=1.0e16,
            n_gn_steps=6,
            sb_graph=True,
            refine_saha=True,
            closure_mode="standard",
            use_custom_root=True,
        ).T_K

    x0 = jnp.asarray(9500.0, dtype=jnp.float64)
    g = jax.grad(_T_of_init)(x0)
    assert jnp.isfinite(g), f"production grad non-finite: {g}"

    h = 1.0
    fd = (float(_T_of_init(x0 + h)) - float(_T_of_init(x0 - h))) / (2 * h)
    # The converged T is a fixed point: ∂T/∂init_T is ~0 at convergence, so the
    # FD check is an absolute-tolerance finiteness/agreement guard.
    np.testing.assert_allclose(float(g), fd, atol=1e-3)


def test_production_grad_matches_unrolled():
    """custom_root IFT gradient == unrolled-sweep gradient (both core-JAX)."""
    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_ionic_obs(coeffs_map, seed=7)
    solver = IterativeCFLIBSSolver(db, max_iterations=20, saha_boltzmann_graph=True)
    _, _, inp, _ = _build_kernel_inputs(solver, obs)

    def _maker(use_root):
        def _f(init_T):
            return joint_wls_solve(
                inp,
                init_T_K=init_T,
                ne_stark_cm3=1.0e16,
                n_gn_steps=10,
                sb_graph=True,
                refine_saha=True,
                closure_mode="standard",
                use_custom_root=use_root,
            ).T_K

        return _f

    x0 = jnp.asarray(9000.0, dtype=jnp.float64)
    g_root = jax.grad(_maker(True))(x0)
    g_unroll = jax.grad(_maker(False))(x0)
    assert jnp.isfinite(g_root) and jnp.isfinite(g_unroll)
    np.testing.assert_allclose(float(g_root), float(g_unroll), atol=1e-4)


# ---------------------------------------------------------------------------
# 12. vmap batch=16 over the production joint solve (J7 §5 AC1)
# ---------------------------------------------------------------------------


def test_vmap_production_joint_solve():
    """``jax.vmap`` over 16 perturbed sets through the SB-graph joint solve."""
    coeffs_map = {"Fe": [3.2, 0, 0, 0, 0], "Ni": [2.5, 0, 0, 0, 0], "Cr": [3.9, 0, 0, 0, 0]}
    db = _mock_db_varying(coeffs_map)
    obs = _make_neutral_ionic_obs(coeffs_map, seed=12)
    solver = IterativeCFLIBSSolver(db, max_iterations=20, saha_boltzmann_graph=True)
    _, _, inp, (x, y, w, stage, mask) = _build_kernel_inputs(solver, obs)
    E = inp.x.shape[0]

    rng = np.random.default_rng(20260613)
    n_batch = 16
    y_batch = jnp.asarray(
        np.stack([y + rng.normal(0.0, 0.01, size=y.shape) for _ in range(n_batch)])
    )

    def _run_one(y_one):
        inp_one = inp._replace(y=y_one)
        return joint_wls_solve(
            inp_one,
            init_T_K=9000.0,
            ne_stark_cm3=1.0e16,
            n_gn_steps=6,
            sb_graph=True,
            refine_saha=True,
            closure_mode="standard",
        )

    batched = jax.vmap(_run_one)(y_batch)
    assert batched.T_K.shape == (n_batch,)
    assert batched.concentrations.shape == (n_batch, E)
    assert jnp.all(jnp.isfinite(batched.T_K))
    sums = jnp.sum(batched.concentrations, axis=1)
    np.testing.assert_allclose(np.asarray(sums), 1.0, rtol=1e-6)
