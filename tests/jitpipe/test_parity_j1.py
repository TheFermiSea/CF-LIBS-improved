"""J1 parity — fixed-shape baseline/noise + exact peak detection (ADR-0004 §4 rows 1-2).

Every test feeds **identical inputs** to the FROZEN REFERENCE
(``cflibs.inversion.preprocess.preprocessing`` + ``identify.line_detection._find_peaks``,
which delegate to ``scipy``) and to the jit kernels in ``cflibs.jitpipe.{preprocess,detect}``,
then asserts the §4 tolerance contract:

* baselines SNIP/MEDIAN/PERCENTILE rtol 1e-12; ALS ``max|Δz| ≤ 1e-6·scale(y)``;
* noise exact (rtol 1e-12);
* peak index lists **byte-identical** vs ``scipy.signal.find_peaks`` for BOTH
  detector parameterisations, on a corpus of synthetic spectra and ≥1,000
  randomized property cases (float-valued — the regime of real LIBS spectra);
* AC3: jit + vmap (B=16) smoke, grad smoke (finite), no-SQLite-in-kernel guard,
  padding invariance (next pad size bit-identical on the valid region);
* AC4: truncation flag on a synthetic overflow fixture.

CPU x64 (conftest forces it); the whole file runs well under the 600 s watchdog.
Run only this file (never the full suite):

    PYTHONPATH=$PWD JAX_PLATFORMS=cpu .venv/bin/python -m pytest \
        tests/jitpipe/test_parity_j1.py -q --timeout=300
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_jax

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.inversion.identify.line_detection import _find_peaks  # noqa: E402
from cflibs.inversion.preprocess import preprocessing as REF  # noqa: E402
from cflibs.jitpipe import detect as D  # noqa: E402
from cflibs.jitpipe import preprocess as P  # noqa: E402

# ---------------------------------------------------------------------------- #
# Fixtures: synthetic spectra spanning realistic LIBS shapes + edge corners.   #
# ---------------------------------------------------------------------------- #


def _make_spectrum(seed: int, n: int = 2000, wl0: float = 240.0, wl1: float = 265.0):
    rng = np.random.default_rng(seed)
    wl = np.linspace(wl0, wl1, n)
    y = np.zeros(n)
    for _ in range(rng.integers(20, 60)):
        c = rng.uniform(wl0, wl1)
        a = rng.uniform(40.0, 1200.0)
        w = rng.uniform(0.02, 0.10)
        y += a * np.exp(-0.5 * ((wl - c) / w) ** 2)
    # slowly varying continuum + Gaussian noise (float64 -> tie-free).
    y += 30.0 + 0.4 * (wl - wl0) + 0.01 * (wl - 250.0) ** 2 + rng.normal(0.0, 7.0, n)
    y = np.maximum(y, 0.0)
    return wl, y


# A small "corpus" of distinct synthetic spectra (one per seed) standing in for
# the per-dataset real spectra; CPU-cheap, deterministic, tie-free.
_CORPUS = [_make_spectrum(s) for s in range(8)]


def _spacing(wl: np.ndarray) -> float:
    return float(np.median(np.abs(np.diff(wl))))


# ---------------------------------------------------------------------------- #
# Baselines                                                                     #
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(8))
def test_snip_baseline_rtol_1e12(seed):
    wl, y = _CORPUS[seed]
    ref = REF.estimate_baseline_snip(wl, y)
    jit = np.asarray(P.snip_baseline(jnp.asarray(y)))
    np.testing.assert_allclose(jit, ref, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("seed", range(8))
def test_median_baseline_exact(seed):
    wl, y = _CORPUS[seed]
    win = P._window_pts(len(y), _spacing(wl), 10.0)
    ref = REF.estimate_baseline(wl, y, window_nm=10.0)
    jit = np.asarray(P.median_baseline(jnp.asarray(y), win))
    # sort-based median is exact vs scipy.ndimage.median_filter.
    np.testing.assert_array_equal(jit, ref)


@pytest.mark.parametrize("percentile", [5.0, 10.0, 25.0, 50.0, 90.0])
def test_percentile_baseline_exact(percentile):
    wl, y = _CORPUS[0]
    win = P._window_pts(len(y), _spacing(wl), 10.0)
    ref = REF.estimate_baseline_percentile(wl, y, window_nm=10.0, percentile=percentile)
    jit = np.asarray(P.percentile_baseline(jnp.asarray(y), win, percentile))
    np.testing.assert_array_equal(jit, ref)


@pytest.mark.parametrize("seed", range(8))
def test_als_baseline_within_contract(seed):
    wl, y = _CORPUS[seed]
    ref = REF.estimate_baseline_als(wl, y)
    jit = np.asarray(P.als_baseline(jnp.asarray(y)))
    scale = float(np.max(np.abs(y)))
    assert np.max(np.abs(jit - ref)) <= 1e-6 * scale


def test_als_frozen_iteration_idempotence():
    """AC1: ALS on an early-converging (flat) spectrum freezes -> stable result."""
    wl = np.linspace(200.0, 400.0, 500)
    y = np.full_like(wl, 5.0)  # perfectly flat -> converges on iteration 1
    a = np.asarray(P.als_baseline(jnp.asarray(y), max_iters=20))
    b = np.asarray(P.als_baseline(jnp.asarray(y), max_iters=40))
    # Frozen-iteration: extending max_iters must not change the converged result.
    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-8)


# ---------------------------------------------------------------------------- #
# Noise (exact)                                                                 #
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("method", ["snip", "median", "als"])
def test_noise_exact(seed, method):
    wl, y = _CORPUS[seed]
    if method == "snip":
        bl = REF.estimate_baseline_snip(wl, y)
    elif method == "median":
        bl = REF.estimate_baseline(wl, y, window_nm=10.0)
    else:
        bl = REF.estimate_baseline_als(wl, y)
    ref = REF.estimate_noise(y, bl)
    jit = float(P.estimate_noise(jnp.asarray(y), jnp.asarray(bl)))
    np.testing.assert_allclose(jit, ref, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------------------- #
# Peak detection — byte-identical index lists, both parameterisations          #
# ---------------------------------------------------------------------------- #


def _jit_idx(res) -> np.ndarray:
    return np.sort(np.asarray(res.indices)[np.asarray(res.mask)])


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("method", ["median", "snip", "percentile", "als"])
def test_peaks_calibration_byte_identical(seed, method):
    """Calibration detector (``detect_peaks``) vs the real reference, byte-identical."""
    wl, y = _CORPUS[seed]
    bm = {
        "median": REF.BaselineMethod.MEDIAN,
        "snip": REF.BaselineMethod.SNIP,
        "percentile": REF.BaselineMethod.PERCENTILE,
        "als": REF.BaselineMethod.ALS,
    }[method]
    ref_peaks, baseline, noise = REF.detect_peaks_auto(wl, y, baseline_method=bm)
    ref_idx = np.sort([p[0] for p in ref_peaks])

    # Feed the SAME reference baseline + noise into the jit detector so the only
    # thing under test is the find_peaks port (baselines are tested above).
    dist = D.min_peak_distance_calibration(_spacing(wl))
    res = D.detect_peaks_calibration(
        jnp.asarray(y), jnp.asarray(baseline), jnp.asarray(noise), distance_px=dist
    )
    np.testing.assert_array_equal(_jit_idx(res), ref_idx)


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("mph,pwnm", [(0.01, 0.05), (0.05, 0.03), (0.1, 0.08)])
def test_peaks_detection_byte_identical(seed, mph, pwnm):
    """Detection detector (``_find_peaks``) vs the real reference, byte-identical."""
    wl, y = _CORPUS[seed]
    ref_peaks = _find_peaks(wl, y, mph, pwnm)
    ref_idx = np.sort([p[0] for p in ref_peaks])
    dist = D.min_peak_distance_detection(_spacing(wl), pwnm)
    res = D.detect_peaks_detection(jnp.asarray(y), mph, distance_px=dist)
    np.testing.assert_array_equal(_jit_idx(res), ref_idx)


def test_peaks_property_1000_float_cases():
    """≥1,000 randomized float-valued property cases vs scipy.find_peaks.

    Float64 spectra are tie-free, so the contract is *byte-identical* (the
    documented Jaccard ≥0.995 fallback applies only to quantised/integer ties,
    which do not occur in real preprocessed LIBS data — see the NMS docstring).
    """
    from scipy.signal import find_peaks

    rng = np.random.default_rng(2024)
    n_cases = 1000
    p_cand = 512
    p_max = 512
    n_fixed = 256  # fixed N so the kernel compiles once
    mism = 0
    for _ in range(n_cases):
        kind = int(rng.integers(0, 3))
        if kind == 0:  # noisy sinusoid
            x = np.maximum(
                rng.normal(0, 1, n_fixed)
                + np.sin(np.linspace(0, rng.uniform(5, 40), n_fixed)) * rng.uniform(1, 4),
                0,
            )
        elif kind == 1:  # sparse spikes on flat
            x = np.zeros(n_fixed)
            k = int(rng.integers(1, 8))
            x[rng.choice(n_fixed, k, replace=False)] = rng.uniform(1, 10, k)
        else:  # overlapping gaussians (LIBS-like)
            x = np.zeros(n_fixed)
            for _ in range(int(rng.integers(1, 12))):
                c = rng.uniform(0, n_fixed)
                a = rng.uniform(1, 10)
                w = rng.uniform(0.5, 5)
                x += a * np.exp(-0.5 * ((np.arange(n_fixed) - c) / w) ** 2)
            x += rng.normal(0, 0.1, n_fixed)
        h = float(rng.uniform(0, 3))
        pr = float(rng.uniform(0, 2))
        d = int(rng.integers(1, 6))
        ref, _ = find_peaks(x, height=h, prominence=pr, distance=d)
        res = D.find_peaks_fixed(
            jnp.asarray(x),
            jnp.asarray(h),
            jnp.asarray(pr),
            jnp.asarray(float(d)),
            p_cand=p_cand,
            p_max=p_max,
        )
        jit = np.sort(np.asarray(res.indices)[np.asarray(res.mask)])
        if not np.array_equal(np.sort(ref), jit):
            mism += 1
    assert mism == 0, f"{mism}/{n_cases} float property cases diverged from scipy"


def test_peaks_plateau_and_tie_corners():
    """Crafted plateaus, equal-height ties, distance-suppression chains vs scipy."""
    from scipy.signal import find_peaks

    cases = [
        (np.array([0, 2, 2, 2, 0, 5, 5, 0, 3, 3, 3, 3, 0, 1, 1, 0], float), 0.5, 0.5, 1),
        (np.array([0, 1, 0, 2, 0, 3, 2, 5, 0, 4, 0, 1, 6, 0], float), 1.5, 0.5, 1),
        # leading/trailing edge plateaus must be excluded by both:
        (np.array([5, 5, 3, 4, 3, 5, 5], float), 0.5, 0.1, 1),
        # single isolated peak
        (np.array([0, 0, 9, 0, 0], float), 1.0, 1.0, 1),
    ]
    for x, h, pr, d in cases:
        ref, _ = find_peaks(x, height=h, prominence=pr, distance=d)
        res = D.find_peaks_fixed(
            jnp.asarray(x),
            jnp.asarray(h),
            jnp.asarray(pr),
            jnp.asarray(float(d)),
            p_cand=64,
            p_max=64,
        )
        jit = np.sort(np.asarray(res.indices)[np.asarray(res.mask)])
        np.testing.assert_array_equal(jit, np.sort(ref))


# ---------------------------------------------------------------------------- #
# AC3 — jit, vmap (B=16), grad, no-SQLite, padding invariance                  #
# ---------------------------------------------------------------------------- #


def test_vmap_batch16_smoke():
    """vmap over a batch of 16 spectra (the manifold pattern) compiles + runs."""
    batch = jnp.stack([jnp.asarray(_make_spectrum(100 + i, n=512)[1]) for i in range(16)])

    def one(y):
        bl = P.snip_baseline(y)
        ns = P.estimate_noise(y, bl)
        return D.detect_peaks_calibration(y, bl, ns, distance_px=3, p_cand=2048, p_max=512)

    res = jax.vmap(one)(batch)
    assert res.indices.shape == (16, 512)
    assert res.mask.shape == (16, 512)
    # every batch member finds at least one peak on these dense spectra.
    assert bool(jnp.all(res.mask.sum(axis=1) > 0))


def test_grad_smoke_finite():
    """grad of a scalar through baseline+noise is finite (hard assert, §5.4)."""
    wl, y = _make_spectrum(7, n=512)
    yj = jnp.asarray(y)

    def scalar(y):
        bl = P.snip_baseline(y)
        ns = P.estimate_noise(y, bl)
        # smooth scalar objective through the differentiable kernels.
        return jnp.sum((y - bl) ** 2) + ns

    g = jax.grad(scalar)(yj)
    assert bool(jnp.all(jnp.isfinite(g)))

    # ALS path too (banded LDLT scan must be differentiable).
    g2 = jax.grad(lambda y: jnp.sum(P.als_baseline(y) ** 2))(yj)
    assert bool(jnp.all(jnp.isfinite(g2)))


def test_no_sqlite_in_kernel():
    """The stage takes arrays only — no DB connection is opened during a run.

    Mirrors ``tests/inversion/test_iterative_lax.py``'s no-SQLite guard: patch
    ``sqlite3.connect`` to raise and assert the kernels still run.
    """
    import sqlite3

    orig = sqlite3.connect

    def boom(*a, **k):  # pragma: no cover - only fires on a regression
        raise AssertionError("kernel opened a SQLite connection")

    sqlite3.connect = boom
    try:
        wl, y = _make_spectrum(0, n=512)
        bl = P.snip_baseline(jnp.asarray(y))
        ns = P.estimate_noise(jnp.asarray(y), bl)
        res = D.detect_peaks_calibration(jnp.asarray(y), bl, ns, distance_px=3, p_max=512)
        assert int(res.count) > 0
    finally:
        sqlite3.connect = orig


def test_median_row_chunk_invariance():
    """The ``lax.map`` row-chunk knob is a memory choice, not a parity choice.

    Changing ``row_chunk`` must leave the median baseline bit-identical (catches
    chunk-seam / mask bugs in the row-chunked sliding-window kernel).
    """
    wl, y = _CORPUS[0]
    win = P._window_pts(len(y), _spacing(wl), 10.0)
    a = np.asarray(P.median_baseline(jnp.asarray(y), win, row_chunk=4096))
    b = np.asarray(P.median_baseline(jnp.asarray(y), win, row_chunk=256))
    np.testing.assert_array_equal(a, b)


def test_peak_candidate_pad_invariance():
    """Padding invariance: peak indices are bit-identical at the next pad size.

    The dominant failure mode of fixed-shape rewrites is a mask bug that lets the
    padded region leak into a reduction. Re-running at a larger ``p_cand`` /
    ``p_max`` must reproduce the identical valid-region peak set and count.
    """
    wl, y = _make_spectrum(1, n=1024)
    bl = P.snip_baseline(jnp.asarray(y))
    ns = P.estimate_noise(jnp.asarray(y), bl)
    r_small = D.detect_peaks_calibration(
        jnp.asarray(y), bl, ns, distance_px=3, p_cand=2048, p_max=512
    )
    r_big = D.detect_peaks_calibration(
        jnp.asarray(y), bl, ns, distance_px=3, p_cand=4096, p_max=1024
    )
    a = np.asarray(r_small.indices)[np.asarray(r_small.mask)]
    b = np.asarray(r_big.indices)[np.asarray(r_big.mask)]
    np.testing.assert_array_equal(np.sort(a), np.sort(b))
    assert int(r_small.count) == int(r_big.count)


# ---------------------------------------------------------------------------- #
# AC4 — truncation flag on a synthetic overflow fixture                        #
# ---------------------------------------------------------------------------- #


def test_truncation_flag_on_overflow():
    """A spectrum with more peaks than ``P_max`` keeps the top-priority set + flags."""
    # Comb of strong, well-separated peaks: one every 4 samples -> ~N/4 peaks.
    n = 4096
    x = np.zeros(n)
    x[2::4] = np.linspace(1.0, 5.0, len(x[2::4]))  # distinct heights -> tie-free
    p_max = 256
    res = D.find_peaks_fixed(
        jnp.asarray(x),
        jnp.asarray(0.5),
        jnp.asarray(0.1),
        jnp.asarray(1.0),
        p_cand=2048,
        p_max=p_max,
    )
    assert bool(res.truncated)
    assert int(res.count) > p_max
    assert int(res.mask.sum()) == p_max
    # The kept set must be the top-priority (highest-x) peaks.
    kept = np.asarray(res.indices)[np.asarray(res.mask)]
    assert x[kept].min() >= np.sort(x[x > 0.5])[-p_max]
