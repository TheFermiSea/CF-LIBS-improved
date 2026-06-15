"""Parity + contract tests for the J10 forward-fitting identifier (ADR-0004 §8.1 C).

These tests feed *identical* inputs to the FROZEN reference forward kernel
(:func:`cflibs.radiation.kernels.forward_model`) and to the J10 stage
(:mod:`cflibs.jitpipe.forward_id`), and assert the §4 / §5.4 tolerance contracts:

* **Tier-K forward-eval parity** — the J10 ``vmap`` population evaluator equals a
  per-config Python loop over the real reference kernel (rtol 1e-12; it *is* the
  same function, so the only thing under test is that vmap/batching introduces no
  drift).
* **BIC parity** — :func:`forward_id.bic_cost` reproduces
  :func:`cflibs.inversion.identify.model_selection._compute_bic` on the same
  (L2-normalized) residuals.
* **Recall-driven F1 payoff (the acceptance mechanism, §3 item 1)** — on a
  synthetic confounder panel built from the reference kernel with known truth,
  forward-fit identification beats a peak-coincidence baseline (the audit-F3
  failure mode) by micro-F1 ≥ +0.03 with precision loss ≤ 0.02, scored with the
  exact scoreboard metric math.
* **Fixed-shape contracts** — vmap smoke (batch 16), finite-gradient smoke,
  no-sqlite-in-kernel guard, padding invariance (bit-identical on the valid region
  at the next pad size), and seed determinism.

The reference oracle is imported and *run*, never reimplemented (HARD RULE).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.benchmark.synthetic_eval import (  # noqa: E402
    compute_binary_metrics,
    confusion_counts,
)
from cflibs.core.jax_runtime import _ensure_pytrees_registered  # noqa: E402
from cflibs.inversion.identify.model_selection import _compute_bic  # noqa: E402
from cflibs.instrument.model import InstrumentModel  # noqa: E402
from cflibs.jitpipe import forward_id as ff  # noqa: E402
from cflibs.jitpipe.snapshot import PipelineSnapshot  # noqa: E402
from cflibs.radiation.kernels import BroadeningMode, forward_model  # noqa: E402

pytestmark = [pytest.mark.requires_jax, pytest.mark.requires_db]

DB_PATH = "ASD_da/libs_production.db"
WL_RANGE = (300.0, 420.0)
N_WL = 600


# ---------------------------------------------------------------------------
# Shared fixtures (built once — the snapshot scan is the expensive bit).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _pytrees() -> None:
    _ensure_pytrees_registered()


@pytest.fixture(scope="module")
def setup():
    """A small, real snapshot over a few elements + a shared wavelength grid."""
    elements = ["Fe", "Cu", "Ca"]
    db = AtomicDatabase(DB_PATH)
    asnap = db.snapshot(elements=elements, wavelength_range=WL_RANGE, include_levels=True)
    snap = PipelineSnapshot.from_atomic_snapshot(asnap)
    wl = jnp.linspace(WL_RANGE[0], WL_RANGE[1], N_WL)
    instr = InstrumentModel(resolution_fwhm_nm=0.1)
    return snap, wl, instr


def _reference_spectrum(snap, wl, instr, comp: dict[str, float], T_K=10000.0, ne=1e17):
    """One measured spectrum from the FROZEN reference kernel (the oracle)."""
    from cflibs.plasma.state import SingleZoneLTEPlasma

    plasma = SingleZoneLTEPlasma(T_e=T_K, n_e=ne, species=comp)
    spec = forward_model(
        plasma,
        snap.to_atomic_snapshot(),
        instr,
        wl,
        broadening_mode=BroadeningMode.NIST_PARITY,
        path_length_m=0.01,
    )
    return np.asarray(spec)


# ---------------------------------------------------------------------------
# Tier-K: vmap population evaluator == per-config reference loop.
# ---------------------------------------------------------------------------


def test_forward_eval_parity_vs_reference_loop(setup):
    """``evaluate_population`` (vmap) matches a host loop over ``forward_model``."""
    snap, wl, instr = setup
    elements = snap.element_symbols

    rng = np.random.default_rng(0)
    n = 8
    temps = rng.uniform(7000.0, 13000.0, n)
    nes = 10.0 ** rng.uniform(16.0, 18.0, n)
    comps = rng.random((n, len(elements)))
    comps /= comps.sum(axis=1, keepdims=True)

    # Reference: explicit per-config loop over the frozen kernel.
    ref = np.stack(
        [
            _reference_spectrum(
                snap,
                wl,
                instr,
                {el: float(comps[i, j]) for j, el in enumerate(elements)},
                T_K=float(temps[i]),
                ne=float(nes[i]),
            )
            for i in range(n)
        ]
    )

    # J10: build the batched plasma + one vmap.
    batched = ff.stack_plasma_states(temps, nes, comps, elements)
    got = np.asarray(ff.evaluate_population(batched, snap, instr, wl))

    assert got.shape == ref.shape == (n, N_WL)
    # Same function under vmap => Tier-K-tight (effectively exact in fp64).
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# BIC parity against the reference model-selection oracle.
# ---------------------------------------------------------------------------


def test_bic_parity_vs_reference(setup):
    """``bic_cost`` reproduces ``model_selection._compute_bic`` (normalized)."""
    snap, wl, instr = setup
    meas = _reference_spectrum(snap, wl, instr, {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0})
    syn = _reference_spectrum(snap, wl, instr, {"Fe": 0.6, "Cu": 0.4, "Ca": 0.0})

    k = 4  # Fe + Cu + T + n_e
    got = float(ff.bic_cost(jnp.asarray(syn), jnp.asarray(meas), n_params=k))

    # Reference: L2-normalize both (forward-fit constrains shape, not amplitude),
    # then the exact reference BIC.
    meas_u = meas / np.linalg.norm(meas)
    syn_u = syn / np.linalg.norm(syn)
    ref = _compute_bic(meas_u, syn_u, k)

    assert np.isfinite(got)
    np.testing.assert_allclose(got, ref, rtol=1e-10, atol=1e-8)


# ---------------------------------------------------------------------------
# §3 item 1 — recall-driven F1 payoff vs the peak-coincidence baseline.
# ---------------------------------------------------------------------------


def _peak_coincidence_baseline(meas, wl, snap, *, tol_nm=0.15, min_lines=1):
    """The audit-F3 failure mode: call any element with >= ``min_lines`` catalog
    lines coinciding (within ``tol_nm``) with a measured peak. This is exactly the
    single-line-coincidence path that inflates FPs / starves recall — the thing
    forward-fitting is meant to beat.
    """
    from scipy.signal import find_peaks

    meas = np.asarray(meas)
    norm = meas / max(meas.max(), 1e-30)
    peak_idx, _ = find_peaks(norm, height=0.02)
    peak_wl = np.asarray(wl)[peak_idx]

    line_wl = np.asarray(snap.line_wavelength_nm)
    line_el = np.asarray(snap.line_element_index)
    called = set()
    for ei, el in enumerate(snap.element_symbols):
        el_lines = line_wl[line_el == ei]
        if el_lines.size == 0:
            continue
        # count catalog lines within tol of ANY peak
        d = np.abs(el_lines[:, None] - peak_wl[None, :])
        hits = int(np.any(d <= tol_nm, axis=1).sum())
        if hits >= min_lines:
            called.add(el)
    return called


def test_recall_driven_f1_payoff(setup):
    """Forward-fit micro-F1 >= baseline + 0.03 with precision loss <= 0.02.

    Panel: each spectrum has Fe and/or Cu truly present; Ca is a *confounder* that
    is never present but whose catalog lines coincide with Fe/Cu peaks closely
    enough to fool the coincidence baseline. The forward-fit coherence test should
    reject Ca (precision) and recover the true majors (recall).
    """
    snap, wl, instr = setup
    candidates = list(snap.element_symbols)  # Fe, Cu, Ca

    panel = [
        ({"Fe", "Cu"}, {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0}),
        ({"Fe"}, {"Fe": 1.0, "Cu": 0.0, "Ca": 0.0}),
        ({"Cu"}, {"Fe": 0.0, "Cu": 1.0, "Ca": 0.0}),
        ({"Fe", "Cu"}, {"Fe": 0.5, "Cu": 0.5, "Ca": 0.0}),
    ]

    base = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    fit = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}

    for s, (truth, comp) in enumerate(panel):
        meas = _reference_spectrum(snap, wl, instr, comp)

        called_base = _peak_coincidence_baseline(meas, wl, snap)

        res = ff.forward_fit_identify(
            meas,
            wl,
            snap,
            instr,
            key=jax.random.PRNGKey(100 + s),
            n_configs=512,
            presence_threshold=0.02,
        )
        present = np.asarray(res.element_present)
        called_fit = {el for ei, el in enumerate(snap.element_symbols) if present[ei] > 0.5}

        cb = confusion_counts(truth, called_base, candidates)
        cf = confusion_counts(truth, called_fit, candidates)
        for kk in base:
            base[kk] += cb[kk]
            fit[kk] += cf[kk]

    m_base = compute_binary_metrics(base["tp"], base["fp"], base["fn"], base["tn"])
    m_fit = compute_binary_metrics(fit["tp"], fit["fp"], fit["fn"], fit["tn"])

    # The acceptance mechanism (ADR-0004 §8.3 item 3 / J10 §3 item 1).
    assert m_fit["f1"] - m_base["f1"] >= 0.03, (
        f"micro-F1 gain {m_fit['f1'] - m_base['f1']:.3f} < 0.03 " f"(fit={m_fit}, base={m_base})"
    )
    assert m_base["precision"] - m_fit["precision"] <= 0.02, (
        f"precision loss {m_base['precision'] - m_fit['precision']:.3f} > 0.02 "
        f"(fit={m_fit}, base={m_base})"
    )


# ---------------------------------------------------------------------------
# Fixed-shape contracts.
# ---------------------------------------------------------------------------


def test_vmap_smoke_batch16(setup):
    """vmap the whole scoring core over a batch of 16 measured spectra."""
    snap, wl, instr = setup
    # One shared candidate population (host seam), reused across spectra.
    pop = ff.build_candidate_population(jax.random.PRNGKey(7), snap.element_symbols, n_configs=128)
    batched_plasma = ff.stack_plasma_states(
        pop["temperatures_k"],
        pop["electron_densities"],
        pop["concentrations"],
        snap.element_symbols,
    )
    spectra = ff.evaluate_population(batched_plasma, snap, instr, wl)

    # 16 measured spectra (just reuse some population rows as "measured").
    measured_batch = jnp.stack([spectra[i] for i in range(16)])
    el_valid = jnp.ones((len(snap.element_symbols),), dtype=spectra.dtype)

    def score_one(meas):
        corr = ff.correlation_cost(spectra, meas)
        bic = ff.bic_cost(spectra, meas, n_params=pop["n_params"])
        return ff.forward_fit_presence_scores(
            corr,
            bic,
            pop["membership"],
            pop["config_valid"],
            el_valid,
            presence_threshold=0.02,
        ).element_present

    out = jax.vmap(score_one)(measured_batch)
    assert out.shape == (16, len(snap.element_symbols))
    assert np.all(np.isfinite(np.asarray(out)))


def test_grad_smoke_finite(setup):
    """Gradient of the correlation cost wrt composition is finite (enables
    the spec's optional gradient polish of surviving candidates)."""
    snap, wl, instr = setup
    from cflibs.plasma.state import SingleZoneLTEPlasma

    meas = _reference_spectrum(snap, wl, instr, {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0})
    meas = jnp.asarray(meas)
    elements = snap.element_symbols
    atomic = snap.to_atomic_snapshot()

    def loss(comp_vec):
        species = {el: comp_vec[j] for j, el in enumerate(elements)}
        plasma = SingleZoneLTEPlasma.__new__(SingleZoneLTEPlasma)
        plasma.T_e = jnp.asarray(10000.0)
        plasma.n_e = jnp.asarray(1e17)
        plasma.species = species
        plasma.T_g = None
        plasma.pressure = None
        spec = forward_model(
            plasma,
            atomic,
            instr,
            wl,
            broadening_mode=BroadeningMode.NIST_PARITY,
            path_length_m=0.01,
        )
        return 1.0 - ff.correlation_cost(spec, meas)

    c0 = jnp.asarray([0.5, 0.4, 0.1])
    g = jax.grad(loss)(c0)
    g = np.asarray(g)
    assert g.shape == (3,)
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)


def test_no_sqlite_in_kernel():
    """The J10 stage module imports no sqlite / host / atomic.database code."""
    import cflibs.jitpipe.forward_id as mod

    src = open(mod.__file__).read()
    assert "sqlite3" not in src
    assert "atomic.database" not in src
    assert "import sqlite3" not in src
    # No host import (the SQLite-touching carve-out) inside the stage.
    assert "from cflibs.jitpipe import host" not in src
    assert "jitpipe.host" not in src


def test_padding_invariance(setup):
    """Re-running at a larger candidate-population pad size leaves the valid
    region's scores bit-identical (the padded configs are validity-masked out)."""
    snap, wl, instr = setup
    el = snap.element_symbols

    # Population of 64 real configs.
    key = jax.random.PRNGKey(42)
    pop_small = ff.build_candidate_population(key, el, n_configs=64)
    pl_small = ff.stack_plasma_states(
        pop_small["temperatures_k"],
        pop_small["electron_densities"],
        pop_small["concentrations"],
        el,
    )
    spec_small = ff.evaluate_population(pl_small, snap, instr, wl)

    meas = _reference_spectrum(snap, wl, instr, {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0})
    corr_small = np.asarray(ff.correlation_cost(spec_small, meas))

    # Same 64 configs + 64 PADDING configs marked invalid. The valid region's
    # correlations must be bit-identical, and the presence calls must match.
    pad = 64
    cfg_valid = np.concatenate([np.ones(64), np.zeros(pad)]).astype(corr_small.dtype)
    # Pad correlations / membership / bic with arbitrary junk in the invalid tail.
    corr_pad = np.concatenate([corr_small, np.full(pad, 0.999, dtype=corr_small.dtype)])
    bic_small = np.asarray(ff.bic_cost(spec_small, meas, n_params=pop_small["n_params"]))
    bic_pad = np.concatenate([bic_small, np.full(pad, -1e9, dtype=bic_small.dtype)])
    member_small = np.asarray(pop_small["membership"])
    member_pad = np.concatenate([member_small, np.ones((pad, len(el)))], axis=0)
    el_valid = np.ones(len(el))

    res_small = ff.forward_fit_presence_scores(
        jnp.asarray(corr_small),
        jnp.asarray(bic_small),
        jnp.asarray(member_small),
        jnp.asarray(np.ones(64)),
        jnp.asarray(el_valid),
        presence_threshold=0.02,
    )
    res_pad = ff.forward_fit_presence_scores(
        jnp.asarray(corr_pad),
        jnp.asarray(bic_pad),
        jnp.asarray(member_pad),
        jnp.asarray(cfg_valid),
        jnp.asarray(el_valid),
        presence_threshold=0.02,
    )

    # Valid-region invariance: the padded (masked) configs must not change calls.
    np.testing.assert_array_equal(
        np.asarray(res_small.element_present), np.asarray(res_pad.element_present)
    )
    np.testing.assert_allclose(
        np.asarray(res_small.presence_score),
        np.asarray(res_pad.presence_score),
        rtol=0.0,
        atol=0.0,
    )


def test_seed_determinism(setup):
    """Identical (key, snapshot, n_configs) => identical result (§3 item 4)."""
    snap, wl, instr = setup
    meas = _reference_spectrum(snap, wl, instr, {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0})

    a = ff.forward_fit_identify(meas, wl, snap, instr, key=jax.random.PRNGKey(5), n_configs=256)
    b = ff.forward_fit_identify(meas, wl, snap, instr, key=jax.random.PRNGKey(5), n_configs=256)
    np.testing.assert_array_equal(np.asarray(a.element_present), np.asarray(b.element_present))
    np.testing.assert_array_equal(np.asarray(a.presence_score), np.asarray(b.presence_score))
    assert int(a.best_config_index) == int(b.best_config_index)


def test_module_importable_without_circular(setup):
    """Import-hygiene canary: forward_id imports cleanly and ForwardFitResult is a
    registered pytree (NamedTuple) usable as a vmap output."""
    assert "cflibs.jitpipe.forward_id" in sys.modules
    res = ff.ForwardFitResult(
        element_present=jnp.zeros(3),
        presence_score=jnp.zeros(3),
        best_bic=jnp.zeros(3),
        best_correlation=jnp.asarray(0.0),
        best_config_index=jnp.asarray(0),
        n_valid_configs=jnp.asarray(0),
    )
    leaves = jax.tree_util.tree_leaves(res)
    assert len(leaves) == 6


# ---------------------------------------------------------------------------
# Gradient polish (fixed-K Gauss-Newton / LM) — the J10 §1 / ADR-0004 §6.1 piece.
#
# The reference oracle for the PARAMETERIZATION is the shipped joint optimizer
# (joint_optimizer.py:16-23, _pack_params/_unpack_params): log-T, log10-n_e,
# softmax(theta) simplex. We instantiate the REAL JointOptimizer and assert our
# pack/unpack reproduce its packing bit-for-bit. The reference oracle for the
# REFINED PHYSICS is the frozen forward_model (run, never reimplemented).
# ---------------------------------------------------------------------------


def _ref_joint_optimizer(elements):
    """A real JointOptimizer instance (default SoftmaxClosure) — the packing
    oracle. ``forward_model`` is a never-called stub: _pack/_unpack don't use it."""
    from cflibs.inversion.solve.joint_optimizer import JointOptimizer

    return JointOptimizer(
        forward_model=lambda *a, **k: None,  # unused by _pack_params/_unpack_params
        elements=list(elements),
        wavelength=np.linspace(300.0, 420.0, N_WL),
    )


def test_pack_params_parity_vs_joint_optimizer(setup):
    """``pack_polish_params`` == ``JointOptimizer._pack_params`` (the oracle).

    Tier-K (pure algebra): log-T / log10-n_e / softmax-logit packing must agree to
    f64 round-off with the shipped reference parameterization (ADR-0004 §6.1
    "parameterization matches joint_optimizer.py:16-23").
    """
    snap, wl, instr = setup
    elements = list(snap.element_symbols)
    opt = _ref_joint_optimizer(elements)

    T_K, n_e = 9300.0, 4.7e16
    conc = {el: c for el, c in zip(elements, [0.55, 0.30, 0.15] + [1e-6] * (len(elements) - 3))}
    conc_vec = jnp.asarray([conc[el] for el in elements])

    from cflibs.core.constants import KB_EV

    got = np.asarray(ff.pack_polish_params(T_K, n_e, conc_vec))
    ref = np.asarray(opt._pack_params(T_K * KB_EV, n_e, conc))

    # log-T and log10-n_e blocks: exact algebra.
    np.testing.assert_allclose(got[:2], ref[:2], rtol=1e-12, atol=0.0)
    # theta block is shift-invariant under softmax; compare the *compositions* they
    # decode to (the physically meaningful, gauge-fixed quantity).
    c_got = np.asarray(ff.unpack_polish_params(jnp.asarray(got))[2])
    c_ref = np.asarray(opt.closure.apply(jnp.asarray(ref[2:])))
    np.testing.assert_allclose(c_got, c_ref, rtol=1e-10, atol=1e-12)


def test_unpack_params_parity_vs_joint_optimizer(setup):
    """``unpack_polish_params`` == ``JointOptimizer._unpack_params`` (the oracle).

    Tier-K: exp(log-T)->T_eV->T_K, 10**log10-n_e->n_e, softmax(theta)->C must all
    match the reference inverse map to f64 round-off.
    """
    snap, wl, instr = setup
    elements = list(snap.element_symbols)
    opt = _ref_joint_optimizer(elements)

    from cflibs.core.constants import KB_EV

    rng = np.random.default_rng(3)
    x = np.concatenate(
        [
            [np.log(1.1)],  # log(T_eV)
            [16.8],  # log10(n_e)
            rng.normal(size=len(elements)),  # theta
        ]
    )
    t_k, ne, conc = ff.unpack_polish_params(jnp.asarray(x))
    T_eV_ref, n_e_ref, conc_ref = opt._unpack_params(jnp.asarray(x))

    np.testing.assert_allclose(float(t_k) * KB_EV, float(T_eV_ref), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(float(ne), float(n_e_ref), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(conc), np.asarray(conc_ref), rtol=1e-12, atol=0.0)
    # The simplex constraint the parameterization enforces.
    np.testing.assert_allclose(float(jnp.sum(conc)), 1.0, rtol=0.0, atol=1e-12)


def test_polish_recovers_truth_within_one_percent(setup):
    """Accuracy sanity (J10 §3 item 3): the LM polish recovers the major-element
    concentrations of a reference-generated spectrum to ~1 % relative.

    Start far from truth in T, n_e, AND composition; assert the polished fit is
    near-perfect (correlation -> 1) and the recovered majors match truth — the
    MC-CF prior-art benchmark, which the polish (absent in MC-CF) achieves.
    """
    snap, wl, instr = setup
    els = tuple(snap.element_symbols)
    truth = {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0}
    meas = _reference_spectrum(snap, wl, instr, truth, T_K=10000.0, ne=1e17)

    # Deliberately bad start: wrong T, wrong n_e, near-uniform composition.
    bad = jnp.asarray([0.34, 0.33, 0.33] + [1e-6] * (len(els) - 3))
    x0 = ff.pack_polish_params(8000.0, 3e17, bad)
    res = ff.gauss_newton_polish(
        x0, jnp.asarray(meas), snap, instr, wl, element_symbols=els, max_steps=20
    )

    assert np.isfinite(float(res.correlation))
    # Polish drove correlation essentially to 1 from a poor start.
    assert float(res.correlation) > 0.999, f"polished corr {float(res.correlation):.4f}"
    assert float(res.correlation) >= float(res.init_correlation)

    c = np.asarray(res.concentrations)
    idx = {el: i for i, el in enumerate(els)}
    # Major elements within ~1 % relative of truth (MC-CF benchmark).
    assert abs(c[idx["Fe"]] - 0.7) <= 0.01, f"Fe {c[idx['Fe']]:.4f}"
    assert abs(c[idx["Cu"]] - 0.3) <= 0.01, f"Cu {c[idx['Cu']]:.4f}"
    # Recovered T / n_e back near truth too.
    assert abs(float(res.temperature_k) - 10000.0) / 10000.0 <= 0.02
    assert abs(np.log10(float(res.electron_density)) - 17.0) <= 0.2


def test_polish_monotone_never_regresses(setup):
    """LM accept/reject guarantees correlation never decreases (the whole point of
    the damped step gating) — across a batch of varied, even pathological, starts.
    """
    snap, wl, instr = setup
    els = tuple(snap.element_symbols)
    meas = _reference_spectrum(snap, wl, instr, {"Fe": 0.5, "Cu": 0.5, "Ca": 0.0})

    starts = jnp.stack(
        [
            ff.pack_polish_params(
                7000.0, 1e16, jnp.asarray([0.5, 0.4, 0.1] + [1e-6] * (len(els) - 3))
            ),
            ff.pack_polish_params(
                13000.0, 1e18, jnp.asarray([0.2, 0.2, 0.6] + [1e-6] * (len(els) - 3))
            ),
            ff.pack_polish_params(
                10000.0, 1e17, jnp.asarray([0.5, 0.5, 0.0] + [1e-6] * (len(els) - 3))
            ),
        ]
    )
    res = ff.polish_candidates(
        starts, jnp.asarray(meas), snap, instr, wl, element_symbols=els, max_steps=6
    )
    init = np.asarray(res.init_correlation)
    final = np.asarray(res.correlation)
    assert np.all(np.isfinite(final))
    # Hard contract: the polish never makes any candidate worse.
    assert np.all(final >= init - 1e-12), f"regressed: init={init} final={final}"


def test_polish_grad_finite_hard_assert(setup):
    """ADR-0004 §5.4: grad-finiteness is a **hard** assert for the solve/polish.

    Differentiate the polish residual's ||r||^2 (the LM objective) wrt the FULL
    polish parameter vector (lnT, log10-n_e, theta) through the frozen forward
    kernel; gradient must be finite and non-trivial.
    """
    snap, wl, instr = setup
    els = tuple(snap.element_symbols)
    meas = jnp.asarray(_reference_spectrum(snap, wl, instr, {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0}))
    atomic = snap.to_atomic_snapshot()

    def sq_resid(x):
        plasma = ff._polish_plasma_from_params(x, els)
        spec = forward_model(
            plasma,
            atomic,
            instr,
            wl,
            broadening_mode=BroadeningMode.NIST_PARITY,
            path_length_m=0.01,
        )
        return 1.0 - ff.correlation_cost(spec, meas)

    x0 = ff.pack_polish_params(9000.0, 2e17, jnp.asarray([0.5, 0.4, 0.1] + [1e-6] * (len(els) - 3)))
    g = np.asarray(jax.grad(sq_resid)(x0))
    assert g.shape == (len(els) + 2,)
    assert np.all(np.isfinite(g)), "non-finite polish gradient (HARD)"
    assert np.any(g != 0.0)


def test_polish_vmap_smoke_fixed_shape(setup):
    """vmap the whole polish over a fixed (M, P) start batch; output (M, ...)."""
    snap, wl, instr = setup
    els = tuple(snap.element_symbols)
    meas = _reference_spectrum(snap, wl, instr, {"Fe": 0.6, "Cu": 0.4, "Ca": 0.0})

    m = 8
    rng = np.random.default_rng(11)
    starts = []
    for _ in range(m):
        c = rng.random(len(els))
        c /= c.sum()
        starts.append(
            ff.pack_polish_params(
                rng.uniform(8000, 12000), 10 ** rng.uniform(16, 18), jnp.asarray(c)
            )
        )
    starts = jnp.stack(starts)

    res = ff.polish_candidates(
        starts, jnp.asarray(meas), snap, instr, wl, element_symbols=els, max_steps=4
    )
    assert res.params.shape == (m, len(els) + 2)
    assert res.concentrations.shape == (m, len(els))
    assert res.correlation.shape == (m,)
    assert np.all(np.isfinite(np.asarray(res.params)))
    # Simplex preserved for every polished candidate.
    sums = np.asarray(jnp.sum(res.concentrations, axis=1))
    np.testing.assert_allclose(sums, 1.0, rtol=0.0, atol=1e-10)


def test_polish_determinism(setup):
    """Identical inputs => identical polish (no RNG inside the polish; §3 item 4)."""
    snap, wl, instr = setup
    els = tuple(snap.element_symbols)
    meas = jnp.asarray(_reference_spectrum(snap, wl, instr, {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0}))
    x0 = ff.pack_polish_params(8500.0, 2e17, jnp.asarray([0.5, 0.4, 0.1] + [1e-6] * (len(els) - 3)))

    a = ff.gauss_newton_polish(x0, meas, snap, instr, wl, element_symbols=els, max_steps=5)
    b = ff.gauss_newton_polish(x0, meas, snap, instr, wl, element_symbols=els, max_steps=5)
    np.testing.assert_array_equal(np.asarray(a.params), np.asarray(b.params))
    assert float(a.correlation) == float(b.correlation)


def test_polish_improves_identify_best_correlation(setup):
    """Wiring contract: enabling the polish in ``forward_fit_identify`` raises the
    population's best correlation (the polish only ever helps) without changing the
    correct present/absent calls on a clean two-element spectrum."""
    snap, wl, instr = setup
    meas = _reference_spectrum(snap, wl, instr, {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0})

    common = dict(key=jax.random.PRNGKey(13), n_configs=256, presence_threshold=0.02)
    no_polish = ff.forward_fit_identify(meas, wl, snap, instr, polish_steps=0, **common)
    polished = ff.forward_fit_identify(
        meas, wl, snap, instr, polish_steps=5, top_m_polish=8, **common
    )

    assert float(polished.best_correlation) >= float(no_polish.best_correlation) - 1e-9
    # The polish should push the best fit very close to a perfect shape match.
    assert float(polished.best_correlation) > 0.99
    # Correct calls preserved (Fe, Cu present; Ca absent).
    present = np.asarray(polished.element_present)
    called = {el for ei, el in enumerate(snap.element_symbols) if present[ei] > 0.5}
    assert "Fe" in called and "Cu" in called
    assert "Ca" not in called


# ---------------------------------------------------------------------------
# J10 BIC-margin presence gate (opt-in, additive, default-off bit-identical).
# ---------------------------------------------------------------------------


def test_bic_gate_default_off_is_bit_identical(setup):
    """``require_bic=False`` (default) reproduces the pure correlation-gap call
    bit-for-bit — the frozen-core parity guard for the new gate."""
    snap, wl, instr = setup
    meas = _reference_spectrum(snap, wl, instr, {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0})

    pop = ff.build_candidate_population(jax.random.PRNGKey(21), snap.element_symbols, n_configs=256)
    pl = ff.stack_plasma_states(
        pop["temperatures_k"],
        pop["electron_densities"],
        pop["concentrations"],
        snap.element_symbols,
    )
    spectra = ff.evaluate_population(pl, snap, instr, wl)
    corr = ff.correlation_cost(spectra, meas)
    bic = ff.bic_cost(spectra, meas, n_params=pop["n_params"])
    el_valid = jnp.ones((len(snap.element_symbols),), dtype=spectra.dtype)

    base = ff.forward_fit_presence_scores(
        corr, bic, pop["membership"], pop["config_valid"], el_valid, presence_threshold=0.02
    )
    default = ff.forward_fit_presence_scores(
        corr,
        bic,
        pop["membership"],
        pop["config_valid"],
        el_valid,
        presence_threshold=0.02,
        require_bic=False,
        bic_margin=0.0,
    )
    np.testing.assert_array_equal(
        np.asarray(base.element_present), np.asarray(default.element_present)
    )
    np.testing.assert_array_equal(
        np.asarray(base.presence_score), np.asarray(default.presence_score)
    )
    np.testing.assert_array_equal(np.asarray(base.best_bic), np.asarray(default.best_bic))


def test_bic_gate_present_set_is_subset(setup):
    """``require_bic=True`` with a large ``bic_margin`` yields a present set that is
    a SUBSET of the ``require_bic=False`` present set — the gate only ever removes
    calls (an AND on top of the correlation decision), never adds."""
    snap, wl, instr = setup
    meas = _reference_spectrum(snap, wl, instr, {"Fe": 0.7, "Cu": 0.3, "Ca": 0.0})

    pop = ff.build_candidate_population(jax.random.PRNGKey(22), snap.element_symbols, n_configs=256)
    pl = ff.stack_plasma_states(
        pop["temperatures_k"],
        pop["electron_densities"],
        pop["concentrations"],
        snap.element_symbols,
    )
    spectra = ff.evaluate_population(pl, snap, instr, wl)
    corr = ff.correlation_cost(spectra, meas)
    bic = ff.bic_cost(spectra, meas, n_params=pop["n_params"])
    el_valid = jnp.ones((len(snap.element_symbols),), dtype=spectra.dtype)

    no_gate = ff.forward_fit_presence_scores(
        corr,
        bic,
        pop["membership"],
        pop["config_valid"],
        el_valid,
        presence_threshold=0.02,
        require_bic=False,
    )
    gated = ff.forward_fit_presence_scores(
        corr,
        bic,
        pop["membership"],
        pop["config_valid"],
        el_valid,
        presence_threshold=0.02,
        require_bic=True,
        bic_margin=1e9,  # impossibly large => the BIC gate must trim aggressively
    )

    present_no_gate = np.asarray(no_gate.element_present) > 0.5
    present_gated = np.asarray(gated.element_present) > 0.5
    # SUBSET: every gated call was already a no-gate call (gate only removes).
    assert np.all(present_gated <= present_no_gate), (
        f"gate added a call: no_gate={present_no_gate} gated={present_gated}"
    )
    # The huge margin removes everything (no element earns +1e9 BIC improvement).
    assert not np.any(present_gated)
