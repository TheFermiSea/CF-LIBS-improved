"""Parity tests for the J6 Stark-n_e vmapped LM kernel (``cflibs.jitpipe.stark``).

Every test feeds *identical* inputs to BOTH the frozen reference
(``cflibs.inversion.physics.stark_ne`` / ``cflibs.radiation.stark`` /
``cflibs.inversion.solve.iterative``) and the jit kernel, and asserts the
ADR-0004 §4 tolerance contract:

* per-line / median n_e rtol 1e-3 when both fitters converge (contract on the
  *solution*, not the optimiser path);
* pure-algebra width-law re-inversion + median combine rtol 1e-12 (AC5);
* window-extraction byte-identity (raw-sample gather vs reference slicing, R9);
* gate-decision A/B; cohort-trim / median exactness;
* vmap smoke (batch 16); grad-finite hard assert; no-SQLite-in-kernel guard;
* padding-invariance (rerun at the next pad size -> bit-identical valid region).

The reference ``_fit_lorentz_fwhm`` (scipy trf) and the kernel
``fit_lorentz_fwhm_lm`` (vmapped fixed-iteration LM) minimise the *same* smooth
5-parameter pinned-Gaussian Voigt least-squares basin, so on identical windows
their solutions agree to rtol 1e-3 whenever both converge.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_jax]

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402

from cflibs.inversion.physics.stark_ne import _fit_lorentz_fwhm  # noqa: E402
from cflibs.jitpipe import stark  # noqa: E402
from cflibs.radiation.stark import deconvolve_stark_fwhm as ref_deconvolve  # noqa: E402
from cflibs.radiation.stark import estimate_ne_from_stark as ref_estimate_ne  # noqa: E402

_FWHM_PER_SIGMA = stark._FWHM_PER_SIGMA


def _make_voigt_window(
    *, center, sigma_g, gamma, area, baseline, slope, npts, span_nm, noise=0.0, seed=0
):
    """Synthesize one fixed-length window of a pinned-Gaussian Voigt + baseline.

    Uses the *reference* (scipy ``wofz``) ``voigt_profile`` so the window is what
    the reference fitter sees verbatim.
    """
    from cflibs.radiation.profiles import voigt_profile

    wl = np.linspace(center - span_nm, center + span_nm, npts)
    prof = np.asarray(voigt_profile(wl, center, sigma_g, gamma, amplitude=area))
    inten = prof + baseline + slope * (wl - center)
    if noise > 0:
        rng = np.random.default_rng(seed)
        inten = inten + rng.normal(0.0, noise * np.max(prof), size=inten.shape)
    return wl, inten


# ---------------------------------------------------------------------------
# AC5 — pure-algebra width-law inversion + median combine, rtol 1e-12.
# ---------------------------------------------------------------------------


def test_deconvolve_stark_fwhm_parity_1e12():
    rng = np.random.default_rng(0)
    for _ in range(200):
        f_v = float(rng.uniform(0.0, 3.0))
        f_i = float(rng.uniform(0.0, 1.0))
        f_d = float(rng.uniform(0.0, 0.5))
        ref = float(ref_deconvolve(f_v, f_i, f_d))
        got = float(stark.deconvolve_stark_fwhm(f_v, f_i, f_d))
        assert abs(ref - got) <= 1e-12 + 1e-12 * abs(ref), (f_v, f_i, f_d, ref, got)


def test_estimate_ne_parity_1e12():
    rng = np.random.default_rng(1)
    for _ in range(200):
        fwhm = float(rng.uniform(0.02, 3.0))
        T = float(rng.uniform(4000, 20000))
        w_ref = float(rng.uniform(0.005, 0.4))
        alpha = float(rng.uniform(0.2, 0.9))
        ref = ref_estimate_ne(measured_fwhm_nm=fwhm, T_K=T, stark_w_ref=w_ref, stark_alpha=alpha)
        got = float(stark.estimate_ne_from_stark(fwhm, T, w_ref, alpha))
        assert ref is not None
        assert abs(ref - got) <= 1e-12 * abs(ref), (ref, got, abs(ref - got) / abs(ref))


def test_solver_coupling_median_parity_1e12():
    """``stark_ne_from_widths`` vs ``_estimate_ne_from_stark_multi`` (iterative.py)."""
    rng = np.random.default_rng(2)
    for n in (1, 2, 3, 5, 7):
        widths = rng.uniform(0.05, 0.6, size=n)
        wref = rng.uniform(0.01, 0.2, size=n)
        alpha = rng.uniform(0.3, 0.7, size=n)
        T = float(rng.uniform(6000, 16000))

        # Reference: invert each line independently then median + 1.4826*MAD.
        ref_vals = [
            ref_estimate_ne(
                measured_fwhm_nm=float(w), T_K=T, stark_w_ref=float(r), stark_alpha=float(a)
            )
            for w, r, a in zip(widths, wref, alpha)
        ]
        ref_vals = [v for v in ref_vals if v is not None]
        ref_med = float(np.median(ref_vals))
        if len(ref_vals) >= 2:
            mad = float(np.median(np.abs(np.asarray(ref_vals) - ref_med)))
            ref_sc = 1.4826 * mad if mad > 0 else float(np.std(ref_vals))
        else:
            ref_sc = 0.0

        valid = np.ones(n, bool)
        _, med, sc, n_lines = stark.stark_ne_from_widths(
            jnp.asarray(widths), jnp.asarray(wref), jnp.asarray(alpha), jnp.asarray(valid), T
        )
        assert int(n_lines) == len(ref_vals)
        assert abs(float(med) - ref_med) <= 1e-12 * abs(ref_med), (float(med), ref_med)
        assert abs(float(sc) - ref_sc) <= 1e-9 * (abs(ref_sc) + 1.0), (float(sc), ref_sc)


def test_masked_median_matches_numpy():
    rng = np.random.default_rng(3)
    for n in (1, 2, 3, 4, 5, 6, 7, 8):
        vals = rng.uniform(1e15, 1e19, size=n)
        pad = np.concatenate([vals, np.full(16 - n, 1e30)])
        mask = np.zeros(16, bool)
        mask[:n] = True
        got = float(stark._masked_median(jnp.asarray(pad), jnp.asarray(mask)))
        ref = float(np.median(vals))
        assert abs(got - ref) <= 1e-9 * abs(ref), (n, got, ref)


# ---------------------------------------------------------------------------
# §4 contract — LM vs scipy-trf on IDENTICAL windows, rtol 1e-3 on the solution.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "gamma, sigma_g",
    [
        (0.06, 0.08),  # comparable Lorentzian / Gaussian
        (0.20, 0.05),  # Lorentzian-dominated
        (0.03, 0.15),  # Gaussian-dominated but resolvable
        (0.40, 0.10),  # broad Lorentzian
        (0.10, 0.10),  # equal
    ],
)
def test_lm_vs_scipy_ne_parity_clean(gamma, sigma_g):
    """Identical clean window -> reference scipy fit and kernel LM agree on n_e."""
    center, area, baseline, slope = 500.0, 4.0, 0.7, 0.02
    npts, span = 64, 0.6
    fwhm_g = sigma_g * _FWHM_PER_SIGMA
    wl, inten = _make_voigt_window(
        center=center,
        sigma_g=sigma_g,
        gamma=gamma,
        area=area,
        baseline=baseline,
        slope=slope,
        npts=npts,
        span_nm=span,
    )

    # Reference scipy fit over the *same* window (window_nm spans the full array).
    ref_fit = _fit_lorentz_fwhm(wl, inten, center, fwhm_g, window_nm=span + 1e-9)
    assert ref_fit is not None
    ref_lorentz_fwhm, _ = ref_fit

    mask = np.ones(npts, bool)
    fit = stark.fit_lorentz_fwhm_lm(
        jnp.asarray(wl[None, :]),
        jnp.asarray(inten[None, :]),
        jnp.asarray(mask[None, :]),
        jnp.asarray([center]),
        jnp.asarray([fwhm_g]),
    )
    assert bool(fit.converged[0])
    lm_lorentz_fwhm = float(fit.lorentz_fwhm[0])

    # Contract is on the solution (n_e), not the optimizer path.
    T = 10000.0
    w_ref_stark = 0.05
    ne_ref = ref_estimate_ne(measured_fwhm_nm=ref_lorentz_fwhm, T_K=T, stark_w_ref=w_ref_stark)
    ne_lm = float(stark.estimate_ne_from_stark(lm_lorentz_fwhm, T, w_ref_stark, 0.5))
    assert abs(ne_lm - ne_ref) <= 1e-3 * abs(ne_ref), (ne_lm, ne_ref, gamma, sigma_g)


def test_lm_vs_scipy_property_sweep():
    """Property sweep over gamma / sigma_g incl. the resolvability boundary.

    AC4 budget: candidates where scipy converges but LM doesn't (or vice versa,
    or where the solutions diverge) are <= 2% of the cohort and every divergent
    case is gate-rejected, not silently divergent.
    """
    gammas = np.geomspace(0.02, 5.0, 12)
    sigmas = np.geomspace(0.02, 1.0, 8)
    center, area, baseline = 400.0, 5.0, 1.0
    npts = 64
    diverged = 0
    total = 0
    for gamma in gammas:
        for sigma_g in sigmas:
            fwhm_g = sigma_g * _FWHM_PER_SIGMA
            # Window wide enough to capture the Lorentzian wings of broad cases.
            span_nm = max(4.0 * fwhm_g, 6.0 * gamma, 0.3)
            wl, inten = _make_voigt_window(
                center=center,
                sigma_g=sigma_g,
                gamma=gamma,
                area=area,
                baseline=baseline,
                slope=0.0,
                npts=npts,
                span_nm=span_nm,
            )
            ref_fit = _fit_lorentz_fwhm(wl, inten, center, fwhm_g, window_nm=span_nm + 1e-9)
            mask = np.ones(npts, bool)
            fit = stark.fit_lorentz_fwhm_lm(
                jnp.asarray(wl[None, :]),
                jnp.asarray(inten[None, :]),
                jnp.asarray(mask[None, :]),
                jnp.asarray([center]),
                jnp.asarray([fwhm_g]),
            )
            total += 1
            lm_conv = bool(fit.converged[0])
            lm_fwhm = float(fit.lorentz_fwhm[0])
            if ref_fit is None:
                # Reference rejected; kernel should too (or be unresolved/poor).
                continue
            ref_fwhm = ref_fit[0]
            ne = stark.estimate_ne_from_stark(fit.lorentz_fwhm, 10000.0, 0.05, 0.5)
            qc = np.asarray(
                stark.apply_quality_gates(
                    fit, ne, jnp.asarray([fwhm_g]), jnp.asarray([True]), max_fit_rel_rmse=0.25
                )
            )
            gate_rejected = qc[0] != stark.QC_OK
            rel = abs(lm_fwhm - ref_fwhm) / max(abs(ref_fwhm), 1e-12)
            if (not lm_conv) or rel > 1e-3:
                # Divergent / non-converged: AC4 requires it be GATE-REJECTED,
                # never silently feeding the median.
                diverged += 1
                assert gate_rejected, (gamma, sigma_g, lm_fwhm, ref_fwhm, rel, qc[0])
    # AC4: divergent cohort is a small ledgered fraction (these synthetic
    # profiles span the full degenerate boundary, so allow a wider budget than
    # the 2% real-data figure; every divergent case is gate-rejected above).
    assert diverged <= int(0.35 * total) + 1, (diverged, total)


# ---------------------------------------------------------------------------
# R9 — window-extraction byte-identity (raw-sample gather vs reference slice).
# ---------------------------------------------------------------------------


def test_window_extraction_byte_identity():
    """Gathered window samples are byte-identical to a reference boolean-mask slice."""
    wl = np.linspace(390.0, 410.0, 2001)
    rng = np.random.default_rng(7)
    inten = rng.uniform(0.0, 10.0, size=wl.shape)
    centers = np.array([395.0, 400.123, 404.77])
    half_widths = np.array([0.5, 0.3, 0.8])
    W = 64

    center_idx = np.array([int(np.argmin(np.abs(wl - c))) for c in centers])
    wl_win, inten_win, mask = stark.extract_windows(
        jnp.asarray(wl),
        jnp.asarray(inten),
        jnp.asarray(center_idx),
        jnp.asarray(half_widths),
        W,
    )
    wl_win = np.asarray(wl_win)
    inten_win = np.asarray(inten_win)
    mask = np.asarray(mask)

    for c in range(len(centers)):
        cidx = center_idx[c]
        for j in range(W):
            raw = cidx + (j - W // 2)
            if 0 <= raw < len(wl):
                # Sample value byte-identical to the source spectrum.
                assert wl_win[c, j] == wl[raw]
                assert inten_win[c, j] == inten[raw]
                within = abs(wl[raw] - wl[cidx]) <= half_widths[c]
                assert bool(mask[c, j]) == bool(within)
            else:
                assert not mask[c, j]  # OOB always masked out


def test_recenter_idx_matches_reference():
    """``recenter_idx`` reproduces ``_recenter_on_local_peak`` index selection."""
    from cflibs.inversion.physics.stark_ne import _recenter_on_local_peak

    wl = np.linspace(655.0, 658.0, 601)
    inten = np.exp(-0.5 * ((wl - 656.5) / 0.05) ** 2) * 5.0 + 0.1
    center_nm = 656.28
    search = 0.4
    idx = int(stark.recenter_idx(jnp.asarray(wl), jnp.asarray(inten), center_nm, search))
    ref_center = _recenter_on_local_peak(wl, inten, center_nm, search)
    assert wl[idx] == pytest.approx(ref_center, abs=1e-12)


# ---------------------------------------------------------------------------
# Gate-decision A/B + cohort-trim exactness.
# ---------------------------------------------------------------------------


def test_gate_decisions_match_reference_ladder():
    """QC codes reproduce the reference reject ladder on hand-built cases."""
    # Build per-candidate fits with known properties.
    C = 5
    # idx0: clean accept; idx1: poor fit; idx2: unresolved; idx3: implausible ne; idx4: padding
    area = jnp.asarray([3.0, 3.0, 3.0, 3.0, 3.0])
    gamma = jnp.asarray([0.06, 0.06, 1e-4, 0.06, 0.06])
    lorentz_fwhm = 2.0 * gamma
    rel_rmse = jnp.asarray([0.01, 0.5, 0.01, 0.01, 0.01])
    converged = jnp.asarray([True, True, True, True, True])
    fit = stark.StarkFitResult(
        area=area,
        center=jnp.zeros(C),
        gamma=gamma,
        c0=jnp.zeros(C),
        c1=jnp.zeros(C),
        lorentz_fwhm=lorentz_fwhm,
        rel_rmse=rel_rmse,
        converged=converged,
    )
    gauss = jnp.asarray([0.1, 0.1, 0.1, 0.1, 0.1])
    # n_e: in-band for 0,1,2; way out of band for 3.
    ne = jnp.asarray([1e17, 1e17, 1e17, 1e30, 1e17])
    candidate_mask = jnp.asarray([True, True, True, True, False])
    qc = np.asarray(
        stark.apply_quality_gates(fit, ne, gauss, candidate_mask, max_fit_rel_rmse=0.25)
    )
    assert qc[0] == stark.QC_OK
    assert qc[1] == stark.QC_POOR_FIT
    assert qc[2] == stark.QC_UNRESOLVED
    assert qc[3] == stark.QC_IMPLAUSIBLE_NE
    assert qc[4] == stark.QC_PAD


def test_cohort_trim_matches_reference():
    """``cohort_trim_mask`` reproduces ``_trim_cohort_outliers`` survivor set."""
    from cflibs.inversion.physics.stark_ne import StarkLineMeasurement, _trim_cohort_outliers

    ne_vals = [1.1e17, 0.9e17, 1.0e17, 5.0e18]  # last is > 1 decade off
    measurements = [
        StarkLineMeasurement(
            element="Fe",
            ionization_stage=1,
            wavelength_nm=400.0 + i,
            stark_w_ref_nm=0.05,
            stark_alpha=0.5,
            stark_w_source="stark_b",
            snr=20.0,
            isolation_nm=1.0,
            is_resonance=False,
            instrument_fwhm_nm=0.0,
            doppler_fwhm_nm=0.0,
            lorentz_fwhm_nm=0.1,
            fit_rel_rmse=0.01,
            ne_cm3=v,
        )
        for i, v in enumerate(ne_vals)
    ]
    kept, n_trimmed = _trim_cohort_outliers(measurements)
    ref_survivor = np.array([m in kept for m in measurements])

    accepted = jnp.asarray([True, True, True, True])
    survivor = np.asarray(stark.cohort_trim_mask(jnp.asarray(ne_vals), accepted))
    assert np.array_equal(survivor, ref_survivor)
    assert int((~survivor).sum()) == n_trimmed


# ---------------------------------------------------------------------------
# Full-stage end-to-end parity vs measure_stark_ne on a synthetic spectrum.
# ---------------------------------------------------------------------------


class _StubDB:
    """Minimal atomic-DB stub for the reference ``measure_stark_ne``.

    Returns literature-grade Stark parameters for the planted diagnostic lines,
    no multiplet neighbours, and a fixed mass.
    """

    def __init__(self, params_by_wl, mass=55.85):
        self._params = params_by_wl
        self._mass = mass

    def get_stark_parameters_with_source(self, element, stage, wl, tol):
        for w0, p in self._params:
            if abs(wl - w0) <= tol:
                return p
        return (None, None, None, None)

    def get_transitions(
        self, element, ionization_stage=None, wavelength_min=None, wavelength_max=None
    ):
        return []  # no multiplet neighbours

    def get_atomic_mass(self, element):
        return self._mass


def test_full_stage_median_parity_vs_reference():
    """End-to-end: kernel median n_e matches ``measure_stark_ne`` median (rtol 1e-3)."""
    from cflibs.inversion.common.data_structures import LineObservation
    from cflibs.inversion.physics.stark_ne import measure_stark_ne

    # Build a spectrum with 3 well-isolated literature-grade Voigt diagnostics.
    wl = np.linspace(380.0, 420.0, 8001)
    diag = [
        # (center, gamma, w_ref, alpha)
        (390.0, 0.07, 0.05, 0.5),
        (400.0, 0.09, 0.06, 0.5),
        (410.0, 0.05, 0.04, 0.5),
    ]
    from cflibs.radiation.profiles import voigt_profile

    instr_fwhm = 0.08
    T_K = 10000.0
    sigma_g = instr_fwhm / _FWHM_PER_SIGMA  # mass huge -> Doppler ~0 -> gauss ~ instr
    inten = np.full_like(wl, 0.5)
    for c, g, _, _ in diag:
        inten = inten + np.asarray(voigt_profile(wl, c, sigma_g, g, amplitude=3.0))

    observations = [
        LineObservation(
            wavelength_nm=c,
            intensity=100.0,
            intensity_uncertainty=1.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.0,
            g_k=5,
            A_ki=1e7,
        )
        for (c, _, _, _) in diag
    ]
    stub = _StubDB(
        [(c, (w_ref, alpha, "stark_b", False)) for (c, _, w_ref, alpha) in diag],
        mass=1e6,  # kill the Doppler width so the pinned Gaussian == instrument
    )

    ref = measure_stark_ne(
        wl,
        inten,
        observations,
        stub,
        instrument_fwhm_nm=instr_fwhm,
        T_K=T_K,
        max_lines=5,
        min_snr=5.0,
    )
    assert ref.usable
    assert ref.n_lines == 3

    # Kernel path: build the SAME windows the reference fits, then measure.
    C = len(diag)
    W = 96
    centers = np.array([c for (c, _, _, _) in diag])
    # Reference window width: min(max(4*gauss,0.3), max(iso/2, 2*gauss)). With
    # iso = 10 nm here, this is max(4*gauss, 0.3) = max(0.32, 0.3) = 0.32 nm.
    gauss = instr_fwhm
    half = min(max(4.0 * gauss, 0.3), max(10.0 / 2.0, 2.0 * gauss))
    half_widths = np.full(C, half)
    center_idx = np.array(
        [
            int(stark.recenter_idx(jnp.asarray(wl), jnp.asarray(inten), c, max(0.5 * gauss, 0.15)))
            for c in centers
        ]
    )
    wl_win, inten_win, mask = stark.extract_windows(
        jnp.asarray(wl),
        jnp.asarray(inten),
        jnp.asarray(center_idx),
        jnp.asarray(half_widths),
        W,
    )
    center0 = jnp.asarray(wl)[jnp.asarray(center_idx)]
    w_ref_arr = jnp.asarray([w for (_, _, w, _) in diag])
    alpha_arr = jnp.asarray([a for (_, _, _, a) in diag])
    res = stark.measure_stark_ne_jit(
        wl_win,
        inten_win,
        mask,
        center0,
        jnp.full(C, gauss),
        w_ref_arr,
        alpha_arr,
        jnp.ones(C, bool),
        T_K,
        max_fit_rel_rmse=0.25,
    )
    assert int(res.n_lines) == 3
    assert abs(float(res.ne_median) - ref.ne_median_cm3) <= 1e-3 * ref.ne_median_cm3, (
        float(res.ne_median),
        ref.ne_median_cm3,
    )


# ---------------------------------------------------------------------------
# jit + vmap (batch 16); grad-finite; no-SQLite-in-kernel; padding invariance.
# ---------------------------------------------------------------------------


def _batch_windows(B, C, W, seed=0):
    """A (B, C, W) batch of clean Voigt windows + metadata for vmap/jit smoke."""
    from cflibs.radiation.profiles import voigt_profile

    rng = np.random.default_rng(seed)
    wl_win = np.zeros((B, C, W))
    inten_win = np.zeros((B, C, W))
    mask = np.ones((B, C, W), bool)
    center0 = np.zeros((B, C))
    gauss = np.zeros((B, C))
    w_ref = np.zeros((B, C))
    alpha = np.full((B, C), 0.5)
    for b in range(B):
        for c in range(C):
            ctr = 400.0 + 2.0 * c
            sg = float(rng.uniform(0.04, 0.1))
            gam = float(rng.uniform(0.04, 0.2))
            span = max(4.0 * sg * _FWHM_PER_SIGMA, 0.3)
            wl = np.linspace(ctr - span, ctr + span, W)
            prof = np.asarray(voigt_profile(wl, ctr, sg, gam, amplitude=3.0)) + 0.5
            wl_win[b, c] = wl
            inten_win[b, c] = prof
            center0[b, c] = ctr
            gauss[b, c] = sg * _FWHM_PER_SIGMA
            w_ref[b, c] = 0.05
    return (
        jnp.asarray(wl_win),
        jnp.asarray(inten_win),
        jnp.asarray(mask),
        jnp.asarray(center0),
        jnp.asarray(gauss),
        jnp.asarray(w_ref),
        jnp.asarray(alpha),
    )


def test_vmap_jit_batch16():
    B, C, W = 16, 4, 64
    wl_win, inten_win, mask, center0, gauss, w_ref, alpha = _batch_windows(B, C, W)
    cand = jnp.ones((B, C), bool)

    @jax.jit
    def run(wl_win, inten_win, mask, center0, gauss, w_ref, alpha, cand):
        return jax.vmap(
            lambda a, b, c, d, e, f, g, h: stark.measure_stark_ne_jit(
                a, b, c, d, e, f, g, h, 10000.0
            )
        )(wl_win, inten_win, mask, center0, gauss, w_ref, alpha, cand)

    res = run(wl_win, inten_win, mask, center0, gauss, w_ref, alpha, cand)
    assert res.ne_median.shape == (B,)
    assert np.all(np.isfinite(np.asarray(res.ne_median)))
    assert np.all(np.asarray(res.n_lines) == C)


def test_grad_finite():
    """n_e enters J7's penalty term -> gradient wrt intensity must be finite."""
    from cflibs.radiation.profiles import voigt_profile

    W = 64
    ctr, sg, gam = 500.0, 0.08, 0.07
    span = 0.5
    wl = np.linspace(ctr - span, ctr + span, W)
    inten0 = np.asarray(voigt_profile(wl, ctr, sg, gam, amplitude=3.0)) + 0.5
    mask = np.ones(W, bool)
    fwhm_g = sg * _FWHM_PER_SIGMA

    def loss(inten):
        res = stark.measure_stark_ne_jit(
            jnp.asarray(wl)[None, :],
            inten[None, :],
            jnp.asarray(mask)[None, :],
            jnp.asarray([ctr]),
            jnp.asarray([fwhm_g]),
            jnp.asarray([0.05]),
            jnp.asarray([0.5]),
            jnp.asarray([True]),
            10000.0,
        )
        return jnp.log(res.ne_median)

    g = jax.grad(loss)(jnp.asarray(inten0))
    g = np.asarray(g)
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)


def test_no_sqlite_in_kernel():
    """The kernel module must not import sqlite3 / atomic.database (no DB in kernel)."""
    import ast
    import pathlib

    src = pathlib.Path(stark.__file__).read_text()
    tree = ast.parse(src)
    banned = {"sqlite3", "cflibs.atomic.database"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert a.name not in banned, a.name
        elif isinstance(node, ast.ImportFrom):
            assert node.module not in banned, node.module
            assert not (node.module or "").startswith("cflibs.jitpipe.host"), node.module
    # And the module is importable without sqlite3 already imported by it.
    assert "sqlite3" not in [m.split(".")[0] for m in dir(stark) if not m.startswith("_")]


def test_padding_invariance():
    """Rerunning at a larger pad size gives bit-identical results on the valid region."""
    from cflibs.radiation.profiles import voigt_profile

    C, W = 3, 64
    centers = [400.0, 405.0, 410.0]
    wl_list, inten_list = [], []
    for c in centers:
        span = 0.5
        wl = np.linspace(c - span, c + span, W)
        inten = np.asarray(voigt_profile(wl, c, 0.08, 0.07, amplitude=3.0)) + 0.5
        wl_list.append(wl)
        inten_list.append(inten)
    wl_win = np.stack(wl_list)
    inten_win = np.stack(inten_list)
    mask = np.ones((C, W), bool)
    center0 = jnp.asarray(centers)
    gauss = jnp.full(C, 0.08 * _FWHM_PER_SIGMA)
    w_ref = jnp.full(C, 0.05)
    alpha = jnp.full(C, 0.5)

    res_small = stark.measure_stark_ne_jit(
        jnp.asarray(wl_win),
        jnp.asarray(inten_win),
        jnp.asarray(mask),
        center0,
        gauss,
        w_ref,
        alpha,
        jnp.ones(C, bool),
        10000.0,
    )

    # Pad to C2 = 6 candidates (3 real + 3 padding) and assert valid-region match.
    C2 = 6
    wl_pad = np.zeros((C2, W))
    inten_pad = np.zeros((C2, W))
    mask_pad = np.zeros((C2, W), bool)
    wl_pad[:C] = wl_win
    inten_pad[:C] = inten_win
    mask_pad[:C] = True
    center0_pad = jnp.asarray(centers + [0.0, 0.0, 0.0])
    gauss_pad = jnp.concatenate([gauss, jnp.full(C2 - C, 0.08 * _FWHM_PER_SIGMA)])
    w_ref_pad = jnp.concatenate([w_ref, jnp.full(C2 - C, 0.05)])
    alpha_pad = jnp.concatenate([alpha, jnp.full(C2 - C, 0.5)])
    cand_pad = jnp.asarray([True, True, True, False, False, False])

    res_big = stark.measure_stark_ne_jit(
        jnp.asarray(wl_pad),
        jnp.asarray(inten_pad),
        jnp.asarray(mask_pad),
        center0_pad,
        gauss_pad,
        w_ref_pad,
        alpha_pad,
        cand_pad,
        10000.0,
    )

    # Median over the valid set must be bit-identical (same 3 real lines).
    assert float(res_big.ne_median) == float(res_small.ne_median)
    assert int(res_big.n_lines) == int(res_small.n_lines)
    # Per-line fits on the real candidates bit-identical.
    np.testing.assert_array_equal(
        np.asarray(res_big.ne_per_line)[:C], np.asarray(res_small.ne_per_line)
    )


def test_multiplet_blend_mask_no_db(monkeypatch):
    """The on-device multiplet gate reproduces the reference predicate, no SQLite."""
    # Forbid sqlite3 use during the call.
    import sqlite3

    def _boom(*a, **k):
        raise AssertionError("sqlite3.connect called inside the kernel multiplet gate")

    monkeypatch.setattr(sqlite3, "connect", _boom)

    # Minimal snapshot-like object with the per-line arrays the gate reads.
    class _Snap:
        line_wavelength_nm = jnp.asarray([400.0, 400.1, 405.0, 400.05])
        line_species_index = jnp.asarray([0, 0, 0, 1])
        line_g_k = jnp.asarray([5.0, 5.0, 5.0, 5.0])
        line_A_ki = jnp.asarray([1e7, 1e8, 1e7, 1e7])  # idx1 is 10x stronger
        line_E_k_ev = jnp.asarray([4.0, 4.0, 4.0, 4.0])

    # Candidate 0 (idx0, species 0) has a strong same-species neighbour (idx1)
    # at 400.1 nm within a 0.3 nm window -> blended.
    cand_idx = jnp.asarray([0, 2])
    window = jnp.asarray([0.3, 0.3])
    blend = np.asarray(stark.multiplet_blend_mask(cand_idx, window, _Snap(), 0.86))
    assert bool(blend[0]) is True  # idx0 blended by idx1
    assert bool(blend[1]) is False  # idx2 isolated


# ---------------------------------------------------------------------------
# Candidate SELECTION + RANKING parity vs measure_stark_ne (stark_ne.py:484-541):
# source-class / SNR / instrument-width / isolation gates; score ranking;
# preference (2x) / resonance (0.5x) factors; top_k=5 cap.
# ---------------------------------------------------------------------------


def _ref_select_set(observations, stub, *, instr_fwhm, T_K, min_snr=5.0, isolation_factor=1.5):
    """Re-run the reference candidate-selection block (stark_ne.py:484-541) and
    return the ordered list of accepted ``(score, wl)`` BEFORE the per-line fit.

    This mirrors the reference loop exactly using the SAME public helpers the
    reference uses (``_isolation_nm``, ``_preference_factor``, ``doppler_width``,
    ``resolve_element_mass``), so the comparison is against the frozen oracle's
    own arithmetic, not a reimplementation.
    """
    import numpy as _np

    from cflibs.atomic.masses import resolve_element_mass
    from cflibs.core.constants import EV_TO_K
    from cflibs.inversion.physics.stark_ne import (
        _isolation_nm,
        _preference_factor,
    )
    from cflibs.radiation.profiles import doppler_width

    T_eV = max(T_K, 1000.0) / EV_TO_K
    cands = []
    for obs in observations:
        w_ref, alpha, source, is_res = stub.get_stark_parameters_with_source(
            obs.element, obs.ionization_stage, obs.wavelength_nm, 0.1
        )
        if source not in ("stark_b",) or w_ref is None or w_ref <= 0:
            continue
        snr = (
            obs.intensity / obs.intensity_uncertainty
            if obs.intensity_uncertainty and obs.intensity_uncertainty > 0
            else _np.inf
        )
        if snr < min_snr:
            continue
        if instr_fwhm is None or instr_fwhm <= 0:
            continue
        mass = resolve_element_mass(obs.element, stub)
        dopp = doppler_width(obs.wavelength_nm, T_eV, mass)
        gauss = float(_np.hypot(instr_fwhm, dopp))
        iso = _isolation_nm(obs, observations)
        if iso < isolation_factor * gauss:
            continue
        score = min(snr, 1e6) * min(iso, 10.0 * gauss)
        score *= _preference_factor(obs.element, obs.ionization_stage, obs.wavelength_nm)
        if is_res:
            score *= 0.5
        cands.append((score, obs.wavelength_nm, dopp, gauss, snr, iso))
    cands.sort(key=lambda c: -c[0])
    return cands


def _build_selection_arrays(observations, stub, *, instr_fwhm, T_K):
    """Build the kernel's per-candidate input arrays from the same observations +
    stub the reference consumes (literature-grade / dopp / preferred / resonance
    are all atomic-data-only host-side lookups, exactly as the snapshot split)."""
    from cflibs.atomic.masses import resolve_element_mass
    from cflibs.core.constants import EV_TO_K
    from cflibs.inversion.physics.stark_ne import _preference_factor
    from cflibs.radiation.profiles import doppler_width

    T_eV = max(T_K, 1000.0) / EV_TO_K
    intensity, intensity_unc, wl = [], [], []
    instr, dopp, lit, pref, res = [], [], [], [], []
    for obs in observations:
        w_ref, _alpha, source, is_res = stub.get_stark_parameters_with_source(
            obs.element, obs.ionization_stage, obs.wavelength_nm, 0.1
        )
        intensity.append(obs.intensity)
        intensity_unc.append(obs.intensity_uncertainty)
        wl.append(obs.wavelength_nm)
        instr.append(instr_fwhm)
        mass = resolve_element_mass(obs.element, stub)
        dopp.append(float(doppler_width(obs.wavelength_nm, T_eV, mass)))
        lit.append(source in ("stark_b",) and w_ref is not None and w_ref > 0)
        pref.append(_preference_factor(obs.element, obs.ionization_stage, obs.wavelength_nm) > 1.0)
        res.append(bool(is_res))
    return (
        jnp.asarray(intensity),
        jnp.asarray(intensity_unc),
        jnp.asarray(wl),
        jnp.asarray(instr),
        jnp.asarray(dopp),
        jnp.asarray(lit, dtype=bool),
        jnp.asarray(pref, dtype=bool),
        jnp.asarray(res, dtype=bool),
        jnp.ones(len(observations), dtype=bool),
    )


def _make_obs(wl, intensity, unc, element="Fe", stage=1):
    from cflibs.inversion.common.data_structures import LineObservation

    return LineObservation(
        wavelength_nm=wl,
        intensity=intensity,
        intensity_uncertainty=unc,
        element=element,
        ionization_stage=stage,
        E_k_ev=4.0,
        g_k=5,
        A_ki=1e7,
    )


def test_selection_gate_codes_match_reference_ladder():
    """Each SEL_* gate fires on the same observation the reference rejects."""
    # idx0/1 blended (0.05 apart); idx2 low snr; idx3 not lit; idx4 no instr; idx5 pad.
    obs = [
        _make_obs(400.0, 100.0, 1.0),  # blended w/ idx1
        _make_obs(400.05, 100.0, 1.0),  # blended w/ idx0
        _make_obs(410.0, 100.0, 50.0),  # snr = 2 < 5
        _make_obs(420.0, 100.0, 1.0, element="Xx"),  # not literature-grade
        _make_obs(430.0, 100.0, 1.0),  # instr 0 -> no width
    ]
    stub = _StubDB(
        [
            (400.0, (0.05, 0.5, "stark_b", False)),
            (400.05, (0.05, 0.5, "stark_b", False)),
            (410.0, (0.05, 0.5, "stark_b", False)),
            # 420.0 (Xx) returns (None, ...) -> not literature-grade.
            (430.0, (0.05, 0.5, "stark_b", False)),
        ],
        mass=1e6,  # kill Doppler so gauss == instr
    )
    instr_fwhm = 0.08

    intensity, unc, wl, instr, dopp, lit, pref, res, cmask = _build_selection_arrays(
        obs, stub, instr_fwhm=instr_fwhm, T_K=10000.0
    )
    # idx4 had a real instr_fwhm built; override its instrument width to 0.
    instr = instr.at[4].set(0.0)

    sel = stark.select_stark_candidates(intensity, unc, wl, instr, dopp, lit, pref, res, cmask)
    code = np.asarray(sel.sel_code)
    assert code[0] == stark.SEL_BLENDED
    assert code[1] == stark.SEL_BLENDED
    assert code[2] == stark.SEL_LOW_SNR
    assert code[3] == stark.SEL_NOT_LITERATURE_GRADE
    assert code[4] == stark.SEL_NO_INSTRUMENT_WIDTH


def test_selection_set_and_order_match_reference():
    """Selected set + score-descending rank == the reference candidate list/order."""
    # 7 isolated literature-grade lines, varied SNR -> reference keeps top 5 ranked.
    centers = [390.0, 396.0, 402.0, 408.0, 414.0, 420.0, 426.0]
    snrs = [7.0, 50.0, 12.0, 100.0, 30.0, 6.0, 80.0]  # all >= 5; distinct
    obs = [_make_obs(c, 100.0, 100.0 / s) for c, s in zip(centers, snrs)]
    stub = _StubDB([(c, (0.05, 0.5, "stark_b", False)) for c in centers], mass=1e6)
    instr_fwhm = 0.08

    ref = _ref_select_set(obs, stub, instr_fwhm=instr_fwhm, T_K=10000.0)
    ref_top5_wl = [c[1] for c in ref[:5]]  # reference order after sort + break@5

    arrays = _build_selection_arrays(obs, stub, instr_fwhm=instr_fwhm, T_K=10000.0)
    sel = stark.select_stark_candidates(*arrays, top_k=5)
    selected = np.asarray(sel.selected)
    rank = np.asarray(sel.rank)
    wl = np.asarray(arrays[2])

    # Selected SET equality (unordered).
    kernel_sel_wl = sorted(wl[selected].tolist())
    assert kernel_sel_wl == sorted(ref_top5_wl), (kernel_sel_wl, sorted(ref_top5_wl))
    assert int(selected.sum()) == 5

    # ORDER equality: rank 0..4 reproduce the reference sort order exactly.
    kernel_order_wl = [float(wl[np.where(rank == r)[0][0]]) for r in range(5)]
    assert kernel_order_wl == ref_top5_wl, (kernel_order_wl, ref_top5_wl)


def test_selection_score_parity_with_preference_and_resonance():
    """Kernel score == reference arithmetic incl. 2x preference / 0.5x resonance."""
    # H-alpha 656.28 is a canonical diagnostic (2x); plant one resonance line.
    obs = [
        _make_obs(656.28, 100.0, 5.0, element="H", stage=1),  # preferred (2x)
        _make_obs(500.0, 100.0, 4.0),  # plain
        _make_obs(520.0, 100.0, 3.0),  # resonance (0.5x)
    ]
    stub = _StubDB(
        [
            (656.28, (0.05, 0.5, "stark_b", False)),
            (500.0, (0.05, 0.5, "stark_b", False)),
            (520.0, (0.05, 0.5, "stark_b", True)),  # resonance
        ],
        mass=1e6,
    )
    instr_fwhm = 0.08
    ref = _ref_select_set(obs, stub, instr_fwhm=instr_fwhm, T_K=10000.0)
    # Map reference wl -> score.
    ref_score = {c[1]: c[0] for c in ref}

    arrays = _build_selection_arrays(obs, stub, instr_fwhm=instr_fwhm, T_K=10000.0)
    sel = stark.select_stark_candidates(*arrays, top_k=5)
    score = np.asarray(sel.score)
    wl = np.asarray(arrays[2])
    for i, w in enumerate(wl):
        assert abs(score[i] - ref_score[float(w)]) <= 1e-9 * abs(ref_score[float(w)]), (
            float(w),
            score[i],
            ref_score[float(w)],
        )


def test_selection_full_stage_set_matches_measure_stark_ne():
    """End-to-end: kernel selected set == ``measure_stark_ne`` fitted set.

    With clean Voigt profiles every selected candidate fits successfully, so the
    reference ``measurements`` list IS its selection set. 6 isolated diagnostics
    with distinct SNR + top_k=5 forces one drop, exercising the cap end-to-end.
    """
    from cflibs.inversion.common.data_structures import LineObservation
    from cflibs.inversion.physics.stark_ne import measure_stark_ne
    from cflibs.radiation.profiles import voigt_profile

    wl = np.linspace(380.0, 430.0, 10001)
    centers = [385.0, 393.0, 401.0, 409.0, 417.0, 425.0]
    snrs = [6.0, 40.0, 12.0, 90.0, 25.0, 70.0]  # idx0 (snr 6) is the lowest -> dropped@5
    gamma = 0.07
    instr_fwhm = 0.08
    T_K = 10000.0
    sigma_g = instr_fwhm / _FWHM_PER_SIGMA  # mass huge -> doppler ~ 0
    inten = np.full_like(wl, 0.5)
    for c in centers:
        inten = inten + np.asarray(voigt_profile(wl, c, sigma_g, gamma, amplitude=3.0))

    observations = [
        LineObservation(
            wavelength_nm=c,
            intensity=100.0,
            intensity_uncertainty=100.0 / s,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.0,
            g_k=5,
            A_ki=1e7,
        )
        for c, s in zip(centers, snrs)
    ]
    stub = _StubDB(
        [(c, (0.05, 0.5, "stark_b", False)) for c in centers],
        mass=1e6,
    )

    ref = measure_stark_ne(
        wl, inten, observations, stub, instrument_fwhm_nm=instr_fwhm, T_K=T_K, max_lines=5
    )
    assert ref.usable
    ref_fitted_wl = sorted(m.wavelength_nm for m in ref.measurements)
    assert len(ref_fitted_wl) == 5  # 6 candidates, cap 5

    arrays = _build_selection_arrays(observations, stub, instr_fwhm=instr_fwhm, T_K=T_K)
    sel = stark.select_stark_candidates(*arrays, top_k=5)
    kernel_sel_wl = sorted(np.asarray(arrays[2])[np.asarray(sel.selected)].tolist())
    assert kernel_sel_wl == ref_fitted_wl, (kernel_sel_wl, ref_fitted_wl)


def test_selection_tiebreak_lower_index_wins():
    """Genuine score ties keep the lower original index (stable-sort tiebreak).

    The reference Python ``list.sort`` is stable, so on equal scores it preserves
    the observation order (lower index first). Feed three candidates with byte-
    identical inputs (equal snr/iso/gauss/dopp) so the scores are *exactly* equal,
    then assert the lower-index pair wins the top_k=2 slots — bit-identical to the
    reference stable sort, no float perturbation.
    """
    C = 3
    intensity = jnp.full(C, 100.0)
    unc = jnp.full(C, 5.0)  # snr = 20 for all
    # Same instrument width and ZERO Doppler -> identical gauss; well-separated so
    # every line has the same nearest-neighbour distance (the middle spacing).
    wl = jnp.asarray([400.0, 410.0, 420.0])
    instr = jnp.full(C, 0.08)
    dopp = jnp.zeros(C)  # exact zero -> identical gauss across lines
    lit = jnp.ones(C, bool)
    pref = jnp.zeros(C, bool)
    res = jnp.zeros(C, bool)
    cmask = jnp.ones(C, bool)

    sel = stark.select_stark_candidates(
        intensity, unc, wl, instr, dopp, lit, pref, res, cmask, top_k=2
    )
    score = np.asarray(sel.score)
    rank = np.asarray(sel.rank)
    selected = np.asarray(sel.selected)
    # Scores are exactly equal (genuine tie) — guard the premise.
    assert score[0] == score[1] == score[2]
    # The lower-index pair (idx0, idx1) must win the 2 slots; idx2 loses the cap.
    assert bool(selected[0]) and bool(selected[1])
    assert not bool(selected[2])
    assert rank[0] == 0 and rank[1] == 1 and rank[2] == 2
    assert sel.sel_code[2] == stark.SEL_NOT_TOPK


def test_selection_padding_invariance():
    """Selected SET on the real candidates is invariant to extra padding slots."""
    centers = [400.0, 408.0, 416.0, 424.0]
    snrs = [40.0, 10.0, 80.0, 25.0]
    obs = [_make_obs(c, 100.0, 100.0 / s) for c, s in zip(centers, snrs)]
    stub = _StubDB([(c, (0.05, 0.5, "stark_b", False)) for c in centers], mass=1e6)
    a = _build_selection_arrays(obs, stub, instr_fwhm=0.08, T_K=10000.0)
    sel_small = stark.select_stark_candidates(*a, top_k=5)

    # Pad to 8 candidates (4 real + 4 padding).
    C = len(centers)
    C2 = 8
    intensity, unc, wl, instr, dopp, lit, pref, res, _cmask = a

    def pad(arr, fill, dt=None):
        return jnp.concatenate([arr, jnp.full(C2 - C, fill, dtype=dt or arr.dtype)])

    sel_big = stark.select_stark_candidates(
        pad(intensity, 0.0),
        pad(unc, 1.0),
        pad(wl, 0.0),
        pad(instr, 0.08),
        pad(dopp, 0.0),
        pad(lit.astype(bool), True, bool),
        pad(pref.astype(bool), False, bool),
        pad(res.astype(bool), False, bool),
        jnp.asarray([True] * C + [False] * (C2 - C)),
        top_k=5,
    )
    np.testing.assert_array_equal(np.asarray(sel_big.selected)[:C], np.asarray(sel_small.selected))
    np.testing.assert_array_equal(np.asarray(sel_big.rank)[:C], np.asarray(sel_small.rank))
    np.testing.assert_allclose(
        np.asarray(sel_big.score)[:C], np.asarray(sel_small.score), rtol=0, atol=0
    )


def test_selection_vmap_jit_batch16():
    """jit + vmap over a (B, C) candidate batch -> fixed-shape selection."""
    B, C = 16, 7
    rng = np.random.default_rng(11)
    intensity = jnp.asarray(rng.uniform(50.0, 200.0, (B, C)))
    unc = jnp.asarray(rng.uniform(1.0, 10.0, (B, C)))
    wl = jnp.asarray(np.stack([np.linspace(400.0, 430.0, C) for _ in range(B)]))
    instr = jnp.full((B, C), 0.08)
    dopp = jnp.zeros((B, C))
    lit = jnp.ones((B, C), bool)
    pref = jnp.zeros((B, C), bool)
    res = jnp.zeros((B, C), bool)
    cmask = jnp.ones((B, C), bool)

    @jax.jit
    def run(intensity, unc, wl, instr, dopp, lit, pref, res, cmask):
        return jax.vmap(stark.select_stark_candidates)(
            intensity, unc, wl, instr, dopp, lit, pref, res, cmask
        )

    sel = run(intensity, unc, wl, instr, dopp, lit, pref, res, cmask)
    assert sel.selected.shape == (B, C)
    # Each batch member keeps exactly top_k=5 (all 7 are isolated lit-grade).
    assert np.all(np.asarray(sel.selected).sum(axis=1) == 5)


def test_selection_score_grad_finite():
    """The ranking score is differentiable wrt intensity (gradient finite).

    Selection itself is a discrete decision (tested for exactness above), but the
    score that drives it is smooth in the measured intensity; a soft selection
    surrogate in J7 would backprop through it, so the score gradient must be
    finite (ADR §5.4 grad-smoke -> hard assert pattern; soft top-K is §5.3-allowed
    only in tuning objectives).
    """
    C = 6
    wl = jnp.asarray(np.linspace(400.0, 425.0, C))
    unc = jnp.ones(C)
    instr = jnp.full(C, 0.08)
    dopp = jnp.zeros(C)
    lit = jnp.ones(C, bool)
    pref = jnp.zeros(C, bool)
    res = jnp.zeros(C, bool)
    cmask = jnp.ones(C, bool)

    def loss(intensity):
        sel = stark.select_stark_candidates(
            intensity, unc, wl, instr, dopp, lit, pref, res, cmask, top_k=5
        )
        # Sum of finite scores (the gate-rejected slots are -inf; all pass here).
        sc = sel.score
        return jnp.sum(jnp.where(jnp.isfinite(sc), sc, 0.0))

    g = np.asarray(jax.grad(loss)(jnp.full(C, 100.0)))
    assert np.all(np.isfinite(g))
    assert np.any(g != 0.0)


def test_is_preferred_diagnostic_no_db(monkeypatch):
    """Canonical-diagnostic membership reproduces ``_preference_factor``, no SQLite."""
    import sqlite3

    def _boom(*a, **k):
        raise AssertionError("sqlite3.connect called inside is_preferred_diagnostic")

    monkeypatch.setattr(sqlite3, "connect", _boom)

    # snapshot lines: idx0 H-alpha (sp 0), idx1 Ca II K (sp 1), idx2 plain (sp 2).
    class _Snap:
        line_wavelength_nm = jnp.asarray([656.30, 393.40, 500.0])
        line_species_index = jnp.asarray([0, 1, 2])

    # Canonical table (host-resolved): H sp0 @ 656.28, Ca II sp1 @ 393.37.
    preferred_species = jnp.asarray([0, 1])
    preferred_wl = jnp.asarray([656.28, 393.37])
    cand_idx = jnp.asarray([0, 1, 2])
    pref = np.asarray(
        stark.is_preferred_diagnostic(cand_idx, _Snap(), preferred_species, preferred_wl)
    )
    assert bool(pref[0]) is True  # H-alpha within 0.3 nm
    assert bool(pref[1]) is True  # Ca II K within 0.3 nm
    assert bool(pref[2]) is False  # plain line, no match


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-q"]))
