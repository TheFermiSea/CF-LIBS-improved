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
