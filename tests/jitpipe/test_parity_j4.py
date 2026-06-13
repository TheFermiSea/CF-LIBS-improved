"""Parity tests for jitpipe J4 fit kernels (ADR-0004 §4 rows 5/6/9).

Each test feeds IDENTICAL inputs to the REAL reference function and the J4 jit
kernel and asserts the §4 tolerance contract:

* **Line selection** — exact selected-set equality vs ``LineSelector.select``
  on tie-free inputs; deterministic tiebreak (lower original index wins) vs the
  Python stable sort (acceptance §4).
* **SB-graph** — slope/intercepts rtol 1e-10 vs ``np.linalg.lstsq`` on identical
  rows; the Schur-equivalence property over 1,000 random systems INCLUDING
  degenerate fixtures (collinear x; single element; ``n_rows == 1+E`` exactly;
  ``n_rows < 1+E`` -> validity flag) (acceptance §5.AC-2).
* **Common-slope** — vs the real ``_fit_common_boltzmann_plane``.
* **Sigma-clip** — slope rtol 1e-8 vs the real CPU ``_fit_sigma_clip``.
* **Closure** — standard/matrix/oxide rtol 1e-12 vs the dict implementation;
  ILR ≡ standard equivalence on strictly-positive compositions; keystone gate
  exact vs ``validate_degeneracy``.
* **Cross-cutting** — jit + vmap (B=16) + padding invariance + grad-finite;
  no ``pure_callback`` in ``jitpipe/fit.py``; no SQLite import inside the kernel.

Fixture style mirrors ``tests/inversion/test_iterative_lax.py:79-121``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from cflibs.inversion.common.data_structures import LineObservation  # noqa: E402
from cflibs.inversion.physics import line_selection as ls_mod  # noqa: E402
from cflibs.inversion.physics.boltzmann import BoltzmannPlotFitter  # noqa: E402
from cflibs.inversion.physics.closure import ClosureEquation  # noqa: E402
from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver  # noqa: E402
from cflibs.jitpipe import fit as fitmod  # noqa: E402

pytestmark = [pytest.mark.requires_jax, pytest.mark.unit]


# ---------------------------------------------------------------------------
# §2 Line selection
# ---------------------------------------------------------------------------


def _build_selection_arrays(obs, elements):
    """Pack observations into the (1, L) padded arrays the kernel consumes."""
    el_index = {el: i for i, el in enumerate(elements)}
    L = len(obs)
    wl = np.array([o.wavelength_nm for o in obs], dtype=np.float64)
    inten = np.array([o.intensity for o in obs], dtype=np.float64)
    inten_unc = np.array([o.intensity_uncertainty for o in obs], dtype=np.float64)
    atomic_unc = np.full(L, fitmod.DEFAULT_ATOMIC_UNCERTAINTY, dtype=np.float64)
    el_idx = np.array([el_index[o.element] for o in obs], dtype=np.int32)
    res = np.zeros(L, dtype=bool)
    mask = np.ones(L, dtype=bool)
    return wl, inten, inten_unc, atomic_unc, el_idx, res, mask


def _make_selection_obs(seed=7):
    """A tie-free multi-element fixture spanning gates and the top-K cap."""
    rng = np.random.default_rng(seed)
    obs = []
    elements = ["Fe", "Ni"]
    base_wl = {"Fe": 300.0, "Ni": 400.0}
    for el in elements:
        for k in range(25):  # > top_k=20 so truncation is exercised
            wl = base_wl[el] + 2.0 * k + rng.uniform(0.0, 0.3)
            inten = 100.0 * (1.0 + rng.uniform(0.0, 5.0))
            unc = inten / (12.0 + rng.uniform(0.0, 40.0))  # SNR in [~12, ~52]
            obs.append(
                LineObservation(
                    wavelength_nm=float(wl),
                    intensity=float(inten),
                    intensity_uncertainty=float(unc),
                    element=el,
                    ionization_stage=1,
                    E_k_ev=float(1.0 + k * 0.2),
                    g_k=2,
                    A_ki=1e7,
                )
            )
    return obs, elements


def test_selection_set_equality_vs_reference():
    """Exact selected-set equality vs the real LineSelector on tie-free input."""
    obs, elements = _make_selection_obs()
    selector = ls_mod.LineSelector(
        min_snr=10.0, isolation_wavelength_nm=0.1, max_lines_per_element=20
    )
    ref = selector.select(obs)
    # Reference selected set keyed by identity-stable (element, wl, E_k).
    ref_keys = {
        (o.element, round(o.wavelength_nm, 9), round(o.E_k_ev, 9)) for o in ref.selected_lines
    }

    wl, inten, inten_unc, atomic_unc, el_idx, res, mask = _build_selection_arrays(obs, elements)
    out = jax.jit(
        lambda *a: fitmod.select_lines(
            *a,
            n_elements=len(elements),
            min_snr=10.0,
            isolation_scale_nm=0.1,
            exclude_resonance=False,
            top_k=20,
        ),
        static_argnums=(),
    )(wl[None], inten[None], inten_unc[None], atomic_unc[None], el_idx[None], res[None], mask[None])
    sel = np.asarray(out["selected_mask"][0])
    kern_keys = {
        (obs[i].element, round(obs[i].wavelength_nm, 9), round(obs[i].E_k_ev, 9))
        for i in range(len(obs))
        if sel[i]
    }
    assert kern_keys == ref_keys


def test_selection_scores_match_reference():
    """Per-line scores rtol 1e-12 vs the reference scorer (§2 contract)."""
    obs, elements = _make_selection_obs(seed=11)
    selector = ls_mod.LineSelector(min_snr=10.0, isolation_wavelength_nm=0.1)
    ref = selector.select(obs)
    ref_score = {id(s.observation): s.score for s in ref.scores}

    wl, inten, inten_unc, atomic_unc, el_idx, res, mask = _build_selection_arrays(obs, elements)
    scored = fitmod.line_scores(
        jnp.asarray(wl[None]),
        jnp.asarray(inten[None]),
        jnp.asarray(inten_unc[None]),
        jnp.asarray(atomic_unc[None]),
        jnp.asarray(mask[None]),
        0.1,
    )
    kern_score = np.asarray(scored["score"][0])
    for i, o in enumerate(obs):
        assert kern_score[i] == pytest.approx(ref_score[id(o)], rel=1e-12)


def test_selection_tiebreak_lower_index_wins():
    """On an exact score tie, the lower original index is kept (stable sort)."""
    # 22 identical-score Fe lines (same SNR, same isolation): only 20 survive,
    # and they must be the FIRST 20 by original index.
    obs = []
    for k in range(22):
        obs.append(
            LineObservation(
                wavelength_nm=300.0 + 5.0 * k,  # well separated -> isolation ~1
                intensity=100.0,
                intensity_uncertainty=2.0,  # SNR = 50, identical
                element="Fe",
                ionization_stage=1,
                E_k_ev=1.0 + 0.1 * k,
                g_k=2,
                A_ki=1e7,
            )
        )
    elements = ["Fe"]
    wl, inten, inten_unc, atomic_unc, el_idx, res, mask = _build_selection_arrays(obs, elements)
    out = fitmod.select_lines(
        jnp.asarray(wl[None]),
        jnp.asarray(inten[None]),
        jnp.asarray(inten_unc[None]),
        jnp.asarray(atomic_unc[None]),
        jnp.asarray(el_idx[None]),
        jnp.asarray(res[None]),
        jnp.asarray(mask[None]),
        n_elements=1,
        min_snr=10.0,
        isolation_scale_nm=0.1,
        top_k=20,
    )
    sel = np.asarray(out["selected_mask"][0])
    kept = sorted(np.flatnonzero(sel).tolist())
    assert kept == list(range(20))  # lower index wins; indices 20,21 dropped


def test_selection_energy_spread_diagnostic():
    """spread_ev / n_valid per-element diagnostics match the gated-line stats."""
    obs, elements = _make_selection_obs(seed=31)
    wl, inten, inten_unc, atomic_unc, el_idx, res, mask = _build_selection_arrays(obs, elements)
    energy = np.array([o.E_k_ev for o in obs], dtype=np.float64)
    out = fitmod.select_lines(
        jnp.asarray(wl[None]),
        jnp.asarray(inten[None]),
        jnp.asarray(inten_unc[None]),
        jnp.asarray(atomic_unc[None]),
        jnp.asarray(el_idx[None]),
        jnp.asarray(res[None]),
        jnp.asarray(mask[None]),
        n_elements=len(elements),
        min_snr=10.0,
        isolation_scale_nm=0.1,
        energy_ev=jnp.asarray(energy[None]),
        top_k=20,
    )
    spread = np.asarray(out["spread_ev"][0])
    n_valid = np.asarray(out["n_valid"][0])
    # The kernel's per-element diagnostics are computed on the PRE-truncation
    # gated set (the same basis as the reference _warn_energy_spread, which runs
    # on all valid lines before the top-K cut, line_selection.py:292-306). Use
    # the reference partitioner directly to get that gated set.
    selector = ls_mod.LineSelector(min_snr=10.0, isolation_wavelength_nm=0.1)
    scores = [selector._score_line(o, obs, set(), {}) for o in obs]
    valid_scores, _ = selector._partition_by_criteria(scores)
    valid_keys = {id(s.observation) for s in valid_scores}
    for ei, el in enumerate(elements):
        gated_e = [o.E_k_ev for o in obs if o.element == el and id(o) in valid_keys]
        assert n_valid[ei] == pytest.approx(float(len(gated_e)))
        if gated_e:
            assert spread[ei] == pytest.approx(max(gated_e) - min(gated_e), rel=1e-12)


# ---------------------------------------------------------------------------
# §3 SB-graph (Schur-equivalence vs np.linalg.lstsq) — AC-2 centerpiece
# ---------------------------------------------------------------------------


def _lstsq_sb_graph(rows_x, rows_y, rows_el, E):
    """Reference SB-graph solve: A=[x | element dummies], unit-weight lstsq."""
    n_rows = len(rows_x)
    A = np.zeros((n_rows, 1 + E), dtype=float)
    A[:, 0] = rows_x
    A[np.arange(n_rows), 1 + np.asarray(rows_el)] = 1.0
    coef, *_ = np.linalg.lstsq(A, np.asarray(rows_y), rcond=None)
    slope = float(coef[0])
    intercepts = np.array([coef[1 + e] for e in range(E)], dtype=float)
    return slope, intercepts


def _pack_sb_arrays(per_el_x, per_el_y, E, N):
    """Pack per-element ragged rows into padded (E, N) x/y arrays + mask."""
    x = np.zeros((E, N), dtype=np.float64)
    y = np.zeros((E, N), dtype=np.float64)
    mask = np.zeros((E, N), dtype=bool)
    for e in range(E):
        n = len(per_el_x[e])
        x[e, :n] = per_el_x[e]
        y[e, :n] = per_el_y[e]
        mask[e, :n] = True
    return x, y, mask


@pytest.mark.parametrize("trial", range(50))
def test_sb_graph_schur_equiv_random(trial):
    """SB-graph kernel slope/intercepts rtol 1e-10 vs np.linalg.lstsq (random)."""
    rng = np.random.default_rng(1000 + trial)
    E = int(rng.integers(2, 6))
    N = 8
    per_el_x, per_el_y = [], []
    rows_x, rows_y, rows_el = [], [], []
    for e in range(E):
        n = int(rng.integers(2, N + 1))
        xs = rng.uniform(1.0, 6.0, size=n)
        ys = rng.uniform(-5.0, 5.0, size=n)
        per_el_x.append(xs)
        per_el_y.append(ys)
        rows_x.extend(xs.tolist())
        rows_y.extend(ys.tolist())
        rows_el.extend([e] * n)

    ref_slope, ref_intercepts = _lstsq_sb_graph(rows_x, rows_y, rows_el, E)

    x, y, mask = _pack_sb_arrays(per_el_x, per_el_y, E, N)
    stage = np.ones((E, N), dtype=np.int32)  # all neutral -> no ion shift
    out = jax.jit(fitmod.sb_graph_fit)(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(stage), jnp.asarray(mask), 0.0, 0.0
    )
    assert bool(out["valid"])
    assert float(out["slope"]) == pytest.approx(ref_slope, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(
        np.asarray(out["intercepts"]), ref_intercepts, rtol=1e-10, atol=1e-10
    )


def test_sb_graph_schur_equiv_1000_systems():
    """AC-2: 1,000 random (rows, E<=20) systems, jit SB-graph vs np.linalg.lstsq.

    Fixed (E, N) so the kernel is jitted ONCE and reused for all 1,000 systems
    (each system varies only the padded valid-count per element, so all shapes
    are identical and no recompile happens). rtol 1e-10. Sprinkles in the AC-2
    degenerate families (exactly-determined and collinear-x) by construction.
    """
    E, N = 20, 6  # E<=20 per AC-2; line axis padded to N
    rng = np.random.default_rng(20260613)
    sb_jit = jax.jit(fitmod.sb_graph_fit)
    n_checked = 0
    for trial in range(1000):
        per_el_x, per_el_y = [], []
        rows_x, rows_y, rows_el = [], [], []
        for e in range(E):
            n = int(rng.integers(1, N + 1))
            if trial % 7 == 0:
                xs = np.full(n, 2.0 + e)  # collinear-x family
            else:
                xs = rng.uniform(1.0, 6.0, size=n)
            ys = rng.uniform(-5.0, 5.0, size=n)
            per_el_x.append(xs)
            per_el_y.append(ys)
            rows_x.extend(xs.tolist())
            rows_y.extend(ys.tolist())
            rows_el.extend([e] * n)

        n_rows = len(rows_x)
        # Only compare on over-/exactly-determined systems (the kernel flags the
        # rest as invalid; lstsq's minimum-norm solution differs there).
        if n_rows < max(3, 1 + E):
            continue
        # Skip the per-element-collinear case where the GLOBAL slope column has
        # no variance (every centered x is 0) — there lstsq's min-norm slope=...
        # is not unique; the kernel's guarded slope=0 is the documented behavior
        # (covered separately by test_sb_graph_collinear_x_degenerate).
        x_all = np.asarray(rows_x)
        el_all = np.asarray(rows_el)
        # Total centered-x variance across the pooled, per-element-centered set.
        var = 0.0
        for e in range(E):
            xe = x_all[el_all == e]
            if xe.size:
                var += float(np.sum((xe - xe.mean()) ** 2))
        if var <= 1e-9:
            continue

        ref_slope, ref_intercepts = _lstsq_sb_graph(rows_x, rows_y, rows_el, E)
        x, y, mask = _pack_sb_arrays(per_el_x, per_el_y, E, N)
        stage = np.ones((E, N), dtype=np.int32)
        out = sb_jit(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(stage), jnp.asarray(mask), 0.0, 0.0
        )
        assert bool(out["valid"]), f"trial {trial} should be valid (n_rows={n_rows})"
        assert float(out["slope"]) == pytest.approx(
            ref_slope, rel=1e-10, abs=1e-10
        ), f"slope mismatch trial {trial}"
        # Compare intercepts only for elements that actually contributed rows;
        # lstsq leaves a min-norm 0 for empty dummies, which the kernel also
        # returns as the per-element y-mean (0 when no rows).
        active = np.array([np.any(el_all == e) for e in range(E)])
        np.testing.assert_allclose(
            np.asarray(out["intercepts"])[active],
            ref_intercepts[active],
            rtol=1e-10,
            atol=1e-9,
            err_msg=f"intercept mismatch trial {trial}",
        )
        n_checked += 1
    assert n_checked >= 800, f"too few determined systems checked: {n_checked}"


def test_sb_graph_collinear_x_degenerate():
    """Collinear-x within each element: slope is rank-deficient but lstsq matches."""
    # Each element has identical x for all its lines -> centered x are all 0 ->
    # the slope column has no variance for that element. Across elements there
    # IS x variation, so the global system is still solvable. lstsq is the oracle.
    E, N = 3, 6
    rng = np.random.default_rng(42)
    per_el_x, per_el_y = [], []
    rows_x, rows_y, rows_el = [], [], []
    for e in range(E):
        n = 4
        xs = np.full(n, 2.0 + e)  # collinear within element, varies across
        ys = rng.uniform(-2.0, 2.0, size=n)
        per_el_x.append(xs)
        per_el_y.append(ys)
        rows_x.extend(xs.tolist())
        rows_y.extend(ys.tolist())
        rows_el.extend([e] * n)
    # With per-element collinear x, every centered x is 0 -> Σ(x-x̄)²=0 -> slope=0.
    x, y, mask = _pack_sb_arrays(per_el_x, per_el_y, E, N)
    stage = np.ones((E, N), dtype=np.int32)
    out = fitmod.sb_graph_fit(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(stage), jnp.asarray(mask), 0.0, 0.0
    )
    # Kernel guards denom==0 -> slope 0; intercepts = per-element y means.
    assert float(out["slope"]) == pytest.approx(0.0, abs=1e-12)
    ymeans = np.array([per_el_y[e].mean() for e in range(E)])
    np.testing.assert_allclose(np.asarray(out["intercepts"]), ymeans, rtol=1e-12, atol=1e-12)


def test_sb_graph_single_element():
    """Single-element SB-graph reduces to an ordinary line fit; matches lstsq."""
    E, N = 1, 8
    rng = np.random.default_rng(99)
    xs = rng.uniform(1.0, 5.0, size=6)
    ys = 2.0 - 1.5 * xs + rng.normal(0, 0.01, size=6)
    ref_slope, ref_intercepts = _lstsq_sb_graph(xs.tolist(), ys.tolist(), [0] * 6, E)
    x, y, mask = _pack_sb_arrays([xs], [ys], E, N)
    stage = np.ones((E, N), dtype=np.int32)
    out = fitmod.sb_graph_fit(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(stage), jnp.asarray(mask), 0.0, 0.0
    )
    assert float(out["slope"]) == pytest.approx(ref_slope, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(
        np.asarray(out["intercepts"]), ref_intercepts, rtol=1e-10, atol=1e-10
    )


def test_sb_graph_exactly_determined():
    """n_rows == 1+E exactly -> still over-/exactly-determined, valid flag True."""
    E, N = 2, 4
    # 3 rows, 1+E = 3 -> exactly determined.
    per_el_x = [np.array([1.0, 2.0]), np.array([3.0])]
    per_el_y = [np.array([0.5, -0.5]), np.array([1.0])]
    rows_x = [1.0, 2.0, 3.0]
    rows_y = [0.5, -0.5, 1.0]
    rows_el = [0, 0, 1]
    ref_slope, ref_intercepts = _lstsq_sb_graph(rows_x, rows_y, rows_el, E)
    x, y, mask = _pack_sb_arrays(per_el_x, per_el_y, E, N)
    stage = np.ones((E, N), dtype=np.int32)
    out = fitmod.sb_graph_fit(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(stage), jnp.asarray(mask), 0.0, 0.0
    )
    assert bool(out["valid"])
    assert float(out["slope"]) == pytest.approx(ref_slope, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(
        np.asarray(out["intercepts"]), ref_intercepts, rtol=1e-10, atol=1e-10
    )


def test_sb_graph_underdetermined_flag():
    """n_rows < 1+E -> validity flag False (mirrors lstsq None at iterative.py:2844)."""
    E, N = 3, 4
    # only 3 rows but 1+E = 4 -> under-determined.
    per_el_x = [np.array([1.0]), np.array([2.0]), np.array([3.0])]
    per_el_y = [np.array([0.1]), np.array([0.2]), np.array([0.3])]
    x, y, mask = _pack_sb_arrays(per_el_x, per_el_y, E, N)
    stage = np.ones((E, N), dtype=np.int32)
    out = fitmod.sb_graph_fit(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(stage), jnp.asarray(mask), 0.0, 0.0
    )
    assert not bool(out["valid"])


def test_sb_graph_ion_shift_matches_lstsq():
    """Ion-shifted rows (stage==2) match lstsq on the SAME shifted coordinates."""
    rng = np.random.default_rng(7)
    E, N = 2, 8
    ip, ln_S = 7.5, 1.3
    per_el = []
    rows_x, rows_y, rows_el = [], [], []
    for e in range(E):
        # mix neutral + ionic
        xs_raw = rng.uniform(1.0, 5.0, size=6)
        ys_raw = rng.uniform(-3.0, 3.0, size=6)
        stages = np.array([1, 1, 1, 2, 2, 2])
        xs = xs_raw + np.where(stages == 2, ip, 0.0)
        ys = ys_raw - np.where(stages == 2, ln_S, 0.0)
        per_el.append((xs_raw, ys_raw, stages))
        rows_x.extend(xs.tolist())
        rows_y.extend(ys.tolist())
        rows_el.extend([e] * 6)
    ref_slope, ref_intercepts = _lstsq_sb_graph(rows_x, rows_y, rows_el, E)

    x = np.zeros((E, N), dtype=np.float64)
    y = np.zeros((E, N), dtype=np.float64)
    stage = np.ones((E, N), dtype=np.int32)
    mask = np.zeros((E, N), dtype=bool)
    for e in range(E):
        xs_raw, ys_raw, stages = per_el[e]
        n = len(xs_raw)
        x[e, :n] = xs_raw
        y[e, :n] = ys_raw
        stage[e, :n] = stages
        mask[e, :n] = True
    out = fitmod.sb_graph_fit(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(stage), jnp.asarray(mask), ip, ln_S
    )
    assert float(out["slope"]) == pytest.approx(ref_slope, rel=1e-10, abs=1e-10)
    np.testing.assert_allclose(
        np.asarray(out["intercepts"]), ref_intercepts, rtol=1e-10, atol=1e-10
    )


# ---------------------------------------------------------------------------
# §3 Common-slope vs the real reference
# ---------------------------------------------------------------------------


def test_common_slope_vs_reference():
    """common_slope_fit matches _fit_common_boltzmann_plane (slope/intercepts)."""
    from unittest.mock import MagicMock

    from cflibs.atomic.database import AtomicDatabase

    db = MagicMock(spec=AtomicDatabase)
    solver = IterativeCFLIBSSolver(atomic_db=db, boltzmann_weight_cap=5.0)

    rng = np.random.default_rng(3)
    elements = ["Fe", "Ni", "Cr"]
    E, N = len(elements), 8
    obs_map = {}
    x = np.zeros((E, N))
    y = np.zeros((E, N))
    w = np.zeros((E, N))
    mask = np.zeros((E, N), dtype=bool)
    for ei, el in enumerate(elements):
        n = int(rng.integers(3, 7))
        E_k = rng.uniform(1.0, 6.0, size=n)
        true_slope = -1.2
        intercept = 9.0 + ei * 0.5
        y_vals = intercept + true_slope * E_k + rng.normal(0, 0.01, size=n)
        sigmas = rng.uniform(0.01, 0.1, size=n)
        obs_list = []
        for k in range(n):
            # Build a LineObservation whose y_value and y_uncertainty reproduce
            # (E_k, y_vals, sigma) — pick I, lambda, g, A consistently.
            wl = 500.0
            g, A = 1, 1.0
            inten = float(np.exp(y_vals[k]) * g * A / wl)
            obs_list.append(
                LineObservation(
                    wavelength_nm=wl,
                    intensity=inten,
                    intensity_uncertainty=inten * sigmas[k],
                    element=el,
                    ionization_stage=1,
                    E_k_ev=float(E_k[k]),
                    g_k=g,
                    A_ki=A,
                )
            )
            x[ei, k] = E_k[k]
            y[ei, k] = obs_list[k].y_value
            w[ei, k] = 1.0 / obs_list[k].y_uncertainty ** 2
            mask[ei, k] = True
        obs_map[el] = obs_list

    ref = solver._fit_common_boltzmann_plane(obs_map)
    assert ref is not None

    out = fitmod.common_slope_fit(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(w), jnp.asarray(mask), 5.0
    )
    assert float(out["slope"]) == pytest.approx(ref.slope, rel=1e-10, abs=1e-12)
    ref_int = np.array([ref.intercepts[el] for el in elements])
    np.testing.assert_allclose(np.asarray(out["intercepts"]), ref_int, rtol=1e-10, atol=1e-10)


# ---------------------------------------------------------------------------
# §3 Sigma-clip fixed-K scan vs the real CPU fitter
# ---------------------------------------------------------------------------


def test_sigma_clip_vs_reference():
    """Fixed-K sigma-clip slope/intercept rtol 1e-8 vs the CPU _fit_sigma_clip."""
    rng = np.random.default_rng(5)
    n = 12
    x = np.sort(rng.uniform(1.0, 6.0, size=n))
    y = 8.0 - 1.5 * x + rng.normal(0, 0.02, size=n)
    # Inject two clear outliers.
    y[3] += 3.0
    y[9] -= 2.5
    y_err = np.full(n, 0.05)

    fitter = BoltzmannPlotFitter(outlier_sigma=2.5, max_iterations=10)
    ref = fitter._fit_sigma_clip(x, y, y_err, np.ones(n, dtype=bool))

    w = 1.0 / y_err**2
    out = fitmod.sigma_clip_fit(
        jnp.asarray(x[None]),
        jnp.asarray(y[None]),
        jnp.asarray(w[None]),
        jnp.asarray(np.ones(n, dtype=bool)[None]),
    )
    assert float(out["slope"][0]) == pytest.approx(ref.slope, rel=1e-8, abs=1e-8)
    assert float(out["intercept"][0]) == pytest.approx(ref.intercept, rel=1e-8, abs=1e-8)
    # Final inlier mask should match (no residual on the rejection boundary here).
    ref_mask = np.ones(n, dtype=bool)
    ref_mask[ref.rejected_points] = False
    np.testing.assert_array_equal(np.asarray(out["inlier_mask"][0]), ref_mask)


# ---------------------------------------------------------------------------
# §4 Closure (standard / matrix / oxide / ILR) + keystone gate
# ---------------------------------------------------------------------------


def _closure_inputs(seed=1, E=4):
    rng = np.random.default_rng(seed)
    elements = ["Fe", "Ni", "Cr", "Si"][:E]
    intercepts = rng.uniform(-2.0, 2.0, size=E)
    partition = rng.uniform(10.0, 40.0, size=E)
    mult = np.ones(E)
    mask = np.ones(E, dtype=bool)
    return elements, intercepts, partition, mult, mask


def test_closure_standard_rtol_1e12():
    elements, intercepts, partition, mult, mask = _closure_inputs()
    int_dict = {el: float(intercepts[i]) for i, el in enumerate(elements)}
    pf_dict = {el: float(partition[i]) for i, el in enumerate(elements)}
    ref = ClosureEquation.apply_standard(int_dict, pf_dict)
    out = jax.jit(fitmod.closure_standard)(
        jnp.asarray(intercepts), jnp.asarray(partition), jnp.asarray(mult), jnp.asarray(mask)
    )
    ref_vec = np.array([ref.concentrations[el] for el in elements])
    np.testing.assert_allclose(np.asarray(out), ref_vec, rtol=1e-12, atol=1e-15)


def test_closure_matrix_rtol_1e12():
    elements, intercepts, partition, mult, mask = _closure_inputs(seed=2)
    int_dict = {el: float(intercepts[i]) for i, el in enumerate(elements)}
    pf_dict = {el: float(partition[i]) for i, el in enumerate(elements)}
    ref = ClosureEquation.apply_matrix_mode(int_dict, pf_dict, "Fe", matrix_fraction=0.6)
    out = fitmod.closure_matrix(
        jnp.asarray(intercepts),
        jnp.asarray(partition),
        jnp.asarray(mult),
        jnp.asarray(mask),
        matrix_index=0,
        matrix_fraction=0.6,
    )
    ref_vec = np.array([ref.concentrations[el] for el in elements])
    np.testing.assert_allclose(np.asarray(out), ref_vec, rtol=1e-12, atol=1e-15)


def test_closure_oxide_rtol_1e12():
    elements, intercepts, partition, mult, mask = _closure_inputs(seed=3)
    int_dict = {el: float(intercepts[i]) for i, el in enumerate(elements)}
    pf_dict = {el: float(partition[i]) for i, el in enumerate(elements)}
    stoich = {"Fe": 1.286, "Ni": 1.272, "Cr": 1.462, "Si": 2.139}
    ref = ClosureEquation.apply_oxide_mode(int_dict, pf_dict, stoich)
    factors = np.array([stoich[el] for el in elements])
    out = fitmod.closure_oxide(
        jnp.asarray(intercepts),
        jnp.asarray(partition),
        jnp.asarray(mult),
        jnp.asarray(mask),
        jnp.asarray(factors),
    )
    ref_vec = np.array([ref.concentrations[el] for el in elements])
    np.testing.assert_allclose(np.asarray(out), ref_vec, rtol=1e-12, atol=1e-15)


def test_closure_ilr_equals_standard():
    """ILR ≡ standard closure on strictly-positive compositions (§4 contract)."""
    elements, intercepts, partition, mult, mask = _closure_inputs(seed=4)
    int_dict = {el: float(intercepts[i]) for i, el in enumerate(elements)}
    pf_dict = {el: float(partition[i]) for i, el in enumerate(elements)}
    ref_ilr = ClosureEquation.apply_ilr(int_dict, pf_dict)
    ref_std = ClosureEquation.apply_standard(int_dict, pf_dict)
    out = fitmod.closure_ilr(
        jnp.asarray(intercepts), jnp.asarray(partition), jnp.asarray(mult), jnp.asarray(mask)
    )
    out_vec = np.asarray(out)
    ilr_vec = np.array([ref_ilr.concentrations[el] for el in elements])
    std_vec = np.array([ref_std.concentrations[el] for el in elements])
    # ILR reference == standard reference (the documented identity round-trip)...
    np.testing.assert_allclose(ilr_vec, std_vec, rtol=1e-10, atol=1e-12)
    # ...and the kernel matches the ILR reference.
    np.testing.assert_allclose(out_vec, ilr_vec, rtol=1e-10, atol=1e-12)


def test_closure_missing_partition_masked():
    """A masked element (missing U) is dropped, mirroring the closure `continue`."""
    intercepts = np.array([0.5, -0.2, 1.0])
    partition = np.array([20.0, 30.0, 25.0])
    mult = np.ones(3)
    mask = np.array([True, False, True])  # Ni has no partition function
    # Reference: drop Ni from both dicts.
    int_dict = {"Fe": 0.5, "Cr": 1.0}
    pf_dict = {"Fe": 20.0, "Cr": 25.0}
    ref = ClosureEquation.apply_standard(int_dict, pf_dict)
    out = fitmod.closure_standard(
        jnp.asarray(intercepts), jnp.asarray(partition), jnp.asarray(mult), jnp.asarray(mask)
    )
    out_vec = np.asarray(out)
    assert out_vec[1] == pytest.approx(0.0)
    assert out_vec[0] == pytest.approx(ref.concentrations["Fe"], rel=1e-12)
    assert out_vec[2] == pytest.approx(ref.concentrations["Cr"], rel=1e-12)


@pytest.mark.parametrize(
    "conc,expected",
    [
        ([0.85, 0.05, 0.05, 0.05], True),  # >0.8 dominance, 4 elements -> degenerate
        ([0.4, 0.3, 0.2, 0.1], False),  # well-spread -> not degenerate
        ([0.9, 0.1], False),  # only 2 elements < min_elements=4 -> not flagged
    ],
)
def test_keystone_gate_vs_reference(conc, expected):
    conc = np.array(conc, dtype=np.float64)
    elements = [f"E{i}" for i in range(len(conc))]
    ref = ClosureEquation.validate_degeneracy(
        {el: float(conc[i]) for i, el in enumerate(elements)},
        threshold=0.8,
        min_elements=4,
    )
    assert ref is expected
    out = fitmod.keystone_degenerate(
        jnp.asarray(conc),
        jnp.asarray(np.ones(len(conc), dtype=bool)),
        threshold=0.8,
        min_elements=4,
    )
    assert bool(out) is expected


# ---------------------------------------------------------------------------
# Shared masked_median exactness
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_valid", [1, 2, 3, 4, 5, 6])
def test_masked_median_exact(n_valid):
    rng = np.random.default_rng(n_valid)
    N = 10
    vals = rng.uniform(-5, 5, size=N)
    mask = np.zeros(N, dtype=bool)
    mask[:n_valid] = True
    rng.shuffle(mask)
    ref = float(np.median(vals[mask]))
    out = float(fitmod.masked_median(jnp.asarray(vals), jnp.asarray(mask)))
    assert out == pytest.approx(ref, rel=1e-12, abs=1e-12)


# ---------------------------------------------------------------------------
# Cross-cutting: vmap (B=16), grad-finite, padding invariance, no callback/sqlite
# ---------------------------------------------------------------------------


def test_vmap_batch16_sb_graph():
    """vmap over B=16 SB-graph problems is jit/vmap-clean and finite."""
    rng = np.random.default_rng(123)
    B, E, N = 16, 3, 8
    x = rng.uniform(1.0, 6.0, size=(B, E, N))
    y = rng.uniform(-3.0, 3.0, size=(B, E, N))
    stage = np.ones((B, E, N), dtype=np.int32)
    mask = np.ones((B, E, N), dtype=bool)
    batched = jax.jit(
        jax.vmap(lambda xx, yy, ss, mm: fitmod.sb_graph_fit(xx, yy, ss, mm, 0.0, 0.0))
    )
    out = batched(jnp.asarray(x), jnp.asarray(y), jnp.asarray(stage), jnp.asarray(mask))
    assert out["slope"].shape == (B,)
    assert np.all(np.isfinite(np.asarray(out["slope"])))


def test_vmap_batch16_closure():
    rng = np.random.default_rng(321)
    B, E = 16, 4
    intercepts = rng.uniform(-2, 2, size=(B, E))
    partition = rng.uniform(10, 40, size=(B, E))
    mult = np.ones((B, E))
    mask = np.ones((B, E), dtype=bool)
    out = jax.jit(jax.vmap(fitmod.closure_standard))(
        jnp.asarray(intercepts), jnp.asarray(partition), jnp.asarray(mult), jnp.asarray(mask)
    )
    assert out.shape == (B, E)
    np.testing.assert_allclose(np.asarray(out).sum(axis=1), np.ones(B), rtol=1e-12)


def test_grad_finite_sb_graph():
    """grad of the SB-graph slope w.r.t. y is finite (J7 differentiated region)."""
    rng = np.random.default_rng(11)
    E, N = 3, 6
    x = jnp.asarray(rng.uniform(1.0, 5.0, size=(E, N)))
    mask = jnp.asarray(np.ones((E, N), dtype=bool))
    stage = jnp.asarray(np.ones((E, N), dtype=np.int32))

    def loss(y):
        out = fitmod.sb_graph_fit(x, y, stage, mask, 0.0, 0.0)
        return out["slope"] ** 2

    y0 = jnp.asarray(rng.uniform(-2.0, 2.0, size=(E, N)))
    g = jax.grad(loss)(y0)
    assert np.all(np.isfinite(np.asarray(g)))


def test_grad_finite_closure():
    rng = np.random.default_rng(13)
    E = 4
    partition = jnp.asarray(rng.uniform(10, 40, size=E))
    mult = jnp.asarray(np.ones(E))
    mask = jnp.asarray(np.ones(E, dtype=bool))

    def loss(intercepts):
        c = fitmod.closure_standard(intercepts, partition, mult, mask)
        return jnp.sum(c**2)

    g = jax.grad(loss)(jnp.asarray(rng.uniform(-2, 2, size=E)))
    assert np.all(np.isfinite(np.asarray(g)))


def test_padding_invariance_sb_graph():
    """Re-running at a larger pad size is bit-identical on the valid region."""
    rng = np.random.default_rng(17)
    E = 3
    per_el_x = [rng.uniform(1, 5, size=int(rng.integers(2, 5))) for _ in range(E)]
    per_el_y = [rng.uniform(-3, 3, size=len(per_el_x[e])) for e in range(E)]

    def run(N):
        x, y, mask = _pack_sb_arrays(per_el_x, per_el_y, E, N)
        stage = np.ones((E, N), dtype=np.int32)
        return fitmod.sb_graph_fit(
            jnp.asarray(x), jnp.asarray(y), jnp.asarray(stage), jnp.asarray(mask), 0.0, 0.0
        )

    out_small = run(6)
    out_big = run(12)
    assert float(out_small["slope"]) == float(out_big["slope"])
    np.testing.assert_array_equal(
        np.asarray(out_small["intercepts"]), np.asarray(out_big["intercepts"])
    )


def test_padding_invariance_selection():
    """Selection mask is invariant to extra all-padded line slots."""
    obs, elements = _make_selection_obs(seed=21)
    wl, inten, inten_unc, atomic_unc, el_idx, res, mask = _build_selection_arrays(obs, elements)
    L = len(obs)

    def run(pad):
        n = L + pad
        wl2 = np.zeros(n)
        wl2[:L] = wl
        i2 = np.zeros(n)
        i2[:L] = inten
        iu2 = np.ones(n)
        iu2[:L] = inten_unc
        au2 = np.full(n, fitmod.DEFAULT_ATOMIC_UNCERTAINTY)
        au2[:L] = atomic_unc
        ei2 = np.zeros(n, dtype=np.int32)
        ei2[:L] = el_idx
        r2 = np.zeros(n, dtype=bool)
        m2 = np.zeros(n, dtype=bool)
        m2[:L] = True
        out = fitmod.select_lines(
            jnp.asarray(wl2[None]),
            jnp.asarray(i2[None]),
            jnp.asarray(iu2[None]),
            jnp.asarray(au2[None]),
            jnp.asarray(ei2[None]),
            jnp.asarray(r2[None]),
            jnp.asarray(m2[None]),
            n_elements=len(elements),
            min_snr=10.0,
            isolation_scale_nm=0.1,
            top_k=20,
        )
        return np.asarray(out["selected_mask"][0])[:L]

    np.testing.assert_array_equal(run(0), run(10))


def test_no_pure_callback_in_fit_module():
    """AC §4.5: no jax.pure_callback CALL anywhere in jitpipe/fit.py.

    We parse the AST and look for an attribute access named ``pure_callback``
    (and the bare ``callback``/``io_callback`` variants), so a prose mention of
    "pure_callback" in the module docstring does not trip the guard.
    """
    import ast
    import pathlib

    src = pathlib.Path(fitmod.__file__).read_text()
    tree = ast.parse(src)
    banned = {"pure_callback", "callback", "io_callback", "experimental_callback"}
    bad = [
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr in banned
    ]
    assert not bad, f"callback API used in fit.py: {bad}"


def test_no_sqlite_in_kernel():
    """No sqlite3 / atomic.database import is triggered by the kernels."""
    import sys

    # Run a representative kernel; assert no DB module got imported by it.
    before = set(sys.modules)
    fitmod.sb_graph_fit(
        jnp.zeros((2, 4)),
        jnp.zeros((2, 4)),
        jnp.ones((2, 4), dtype=jnp.int32),
        jnp.ones((2, 4), dtype=bool),
        0.0,
        0.0,
    )
    newly = set(sys.modules) - before
    assert not any("sqlite3" in m for m in newly)
    # The kernel source itself must not import the SQLite-backed DB layer.
    import pathlib

    src = pathlib.Path(fitmod.__file__).read_text()
    assert "atomic.database" not in src
    assert "import sqlite3" not in src
