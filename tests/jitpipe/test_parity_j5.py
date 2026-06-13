"""J5 parity — jittable self-absorption kernel vs the frozen reference oracle.

Feeds IDENTICAL padded inputs to the REAL reference
:class:`cflibs.inversion.physics.self_absorption_observable.ObservableSelfAbsorptionCorrector`
and to :func:`cflibs.jitpipe.selfabs.correct_self_absorption_arrays`, then
asserts the ADR-0004 §4 tolerance contract:

* tau atol 1e-6 per pair (= brentq xtol);
* corrected intensities rtol 1e-6;
* identical corrected / suspect / cleared sets under the documented
  pair-priority (scatter-min) contract;
* quality counters (n_corrected / n_suspect / max_tau) exact.

Plus: bisection-vs-brentq property test over randomized (rho, r_meas/r_thin)
grids, the adversarial shared-upper-level triplet fixture (one line in two
usable pairs), the never-boost / suspect-only-inflate invariants, vmap (B=16),
grad-finite, padding invariance, and a no-sqlite-in-kernel import guard.
"""

from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)

from cflibs.inversion.common.data_structures import LineObservation  # noqa: E402
from cflibs.inversion.physics.self_absorption import (  # noqa: E402
    correct_via_doublet_ratio,
    find_doublet_pairs,
)
from cflibs.inversion.physics.self_absorption_observable import (  # noqa: E402
    ObservableSelfAbsorptionCorrector,
)
from cflibs.jitpipe import selfabs  # noqa: E402
from cflibs.jitpipe.snapshot import PipelineSnapshot, _LEAF_FIELDS  # noqa: E402

# ---------------------------------------------------------------------------
# Fixture builders.
# ---------------------------------------------------------------------------


def _snapshot_from_obs(obs_list, *, element_map=None):
    """Build a minimal PipelineSnapshot whose line-i atomics match obs i.

    ``line_index = arange(L)`` then maps each padded line to catalog row i, so
    the kernel's gather returns exactly the observation's atomic constants.
    """
    n = len(obs_list)
    kw = {}
    kw["line_wavelength_nm"] = np.array([o.wavelength_nm for o in obs_list], dtype=np.float64)
    kw["line_A_ki"] = np.array([o.A_ki for o in obs_list], dtype=np.float64)
    kw["line_E_k_ev"] = np.array([o.E_k_ev for o in obs_list], dtype=np.float64)
    kw["line_g_k"] = np.array([float(o.g_k) for o in obs_list], dtype=np.float64)
    kw["line_E_i_ev"] = np.zeros(n)
    kw["line_g_i"] = np.ones(n)
    kw["line_element_index"] = np.zeros(n, np.int16)
    kw["line_sp_num"] = np.array([o.ionization_stage for o in obs_list], dtype=np.int8)
    kw["line_species_index"] = np.zeros(n, np.int32)
    kw["line_stark_w"] = np.zeros(n)
    kw["line_stark_alpha"] = np.zeros(n)
    kw["line_stark_shift"] = np.zeros(n)
    kw["line_aki_uncertainty"] = np.array(
        [o.aki_uncertainty or 0.0 for o in obs_list], dtype=np.float64
    )
    kw["line_is_resonance"] = np.zeros(n, bool)
    kw["line_stark_source_class"] = np.zeros(n, np.uint8)
    kw["line_gamma_vdw_log"] = np.zeros(n)

    ns = 1
    kw["level_g"] = np.zeros((ns, 1))
    kw["level_E_ev"] = np.zeros((ns, 1))
    kw["level_mask"] = np.zeros((ns, 1), bool)
    kw["partition_coeffs"] = np.zeros((ns, 5))
    kw["partition_coeffs_stored"] = np.zeros((ns, 5))
    kw["canonical_fallback"] = np.zeros((ns, 2))
    kw["species_physics"] = np.zeros((ns, 2))
    kw["partition_t_min"] = np.zeros(ns)
    kw["partition_t_max"] = np.zeros(ns)
    kw["partition_g0"] = np.zeros(ns)
    kw["partition_from_direct_sum"] = np.zeros(ns)
    kw["oxide_stoichiometry"] = np.zeros(ns)
    kw["doublet_pairs"] = np.zeros((0, 2), np.int32)
    kw["doublet_rho"] = np.zeros(0)
    kw["doublet_r_thin"] = np.zeros(0)

    assert set(kw) == set(_LEAF_FIELDS)
    return PipelineSnapshot(species=(("Fe", 1),), element_symbols=("Fe",), **kw)


def _local_pairs(obs_list):
    """Local (P, 2) pair table over the obs L-axis from ``find_doublet_pairs``.

    Matches the reference's pair set + iteration order exactly: each pair is
    (i, j) with the shorter-wavelength line first, in find_doublet_pairs order
    (so the kernel's scatter-min priority equals the reference's first-wins).
    """
    id_to_idx = {id(o): i for i, o in enumerate(obs_list)}
    pairs = find_doublet_pairs(obs_list)
    rows = [(id_to_idx[id(p[0])], id_to_idx[id(p[1])]) for p in pairs]
    if not rows:
        return np.zeros((0, 2), np.int32)
    return np.array(rows, dtype=np.int32)


def _kernel_inputs(obs_list, *, element_ids=None, pad_to=None):
    """Build the padded array inputs (+ snapshot) the kernel consumes."""
    n = len(obs_list)
    snap = _snapshot_from_obs(obs_list)
    intensity = np.array([o.intensity for o in obs_list], dtype=np.float64)
    intensity_unc = np.array([o.intensity_uncertainty for o in obs_list], dtype=np.float64)
    aki_unc = np.array([o.aki_uncertainty or 0.0 for o in obs_list], dtype=np.float64)
    line_index = np.arange(n, dtype=np.int32)
    line_valid = np.ones(n, dtype=bool)
    if element_ids is None:
        # map element symbol -> int id
        syms = sorted({o.element for o in obs_list})
        sym_id = {s: i for i, s in enumerate(syms)}
        element_ids = np.array([sym_id[o.element] for o in obs_list], dtype=np.int32)
    pair_idx = _local_pairs(obs_list)
    pair_valid = np.ones(pair_idx.shape[0], dtype=bool)

    if pad_to is not None and pad_to > n:
        pad = pad_to - n
        # pad lines with a benign dummy (intensity 0, distinct element id).
        intensity = np.concatenate([intensity, np.zeros(pad)])
        intensity_unc = np.concatenate([intensity_unc, np.zeros(pad)])
        aki_unc = np.concatenate([aki_unc, np.zeros(pad)])
        line_index = np.concatenate([line_index, np.zeros(pad, dtype=np.int32)])
        line_valid = np.concatenate([line_valid, np.zeros(pad, dtype=bool)])
        element_ids = np.concatenate(
            [element_ids, (element_ids.max() + 1 + np.arange(pad)).astype(np.int32)]
        )

    return dict(
        intensity=intensity,
        intensity_unc=intensity_unc,
        aki_unc=aki_unc,
        line_index=line_index,
        line_valid=line_valid,
        line_element=element_ids,
        pair_idx=pair_idx,
        pair_valid=pair_valid,
        snapshot=snap,
    )


def _run_kernel(obs_list, **overrides):
    inp = _kernel_inputs(obs_list, **overrides.pop("inputs", {}))
    return selfabs.correct_self_absorption_arrays(**inp, **overrides)


def _run_reference(obs_list, **kw):
    correct_kw = {}
    for k in ("temperature_K", "peak_spectral_radiance"):
        if k in kw:
            correct_kw[k] = kw.pop(k)
    corr = ObservableSelfAbsorptionCorrector(**kw)
    return corr.correct(obs_list, **correct_kw)


def _peak_radiance_arrays(obs_list, peak_by_wl):
    """Build the kernel's ``(peak_spectral_radiance, peak_radiance_valid)`` pair.

    ``peak_by_wl`` mirrors the reference ``peak_spectral_radiance`` dict
    (``{wavelength_nm: peak}``); lines absent from it get ``valid=False`` exactly
    like the reference's ``peak_spectral_radiance.get(wl) is None`` skip.
    """
    n = len(obs_list)
    peak = np.zeros(n, dtype=np.float64)
    valid = np.zeros(n, dtype=bool)
    for i, o in enumerate(obs_list):
        if o.wavelength_nm in peak_by_wl:
            peak[i] = peak_by_wl[o.wavelength_nm]
            valid[i] = True
    return peak, valid


# ---------------------------------------------------------------------------
# Concrete fixtures.
# ---------------------------------------------------------------------------


def _doublet_self_absorbed():
    """A genuinely self-absorbed Fe I doublet (shared upper level)."""
    # Two lines from the same upper level. Atomic ratio r_thin known; the
    # measured intensities are perturbed so r_meas deviates well past the floor.
    g_k = 9
    e_k = 4.30
    l1 = LineObservation(
        wavelength_nm=370.0,
        intensity=80.0,
        intensity_uncertainty=1.0,
        element="Fe",
        ionization_stage=1,
        E_k_ev=e_k,
        g_k=g_k,
        A_ki=1.0e8,
        aki_uncertainty=0.05,
    )
    l2 = LineObservation(
        wavelength_nm=380.0,
        intensity=70.0,
        intensity_uncertainty=1.0,
        element="Fe",
        ionization_stage=1,
        E_k_ev=e_k,
        g_k=g_k,
        A_ki=5.0e7,
        aki_uncertainty=0.05,
    )
    return [l1, l2]


def _suspect_line():
    """A bright low-E_i Fe line with no doublet partner (SA-suspect)."""
    # E_i = E_k - hc/lambda. Pick E_k so E_i < 0.74 eV and intensity is bright.
    wl = 248.0
    photon = selfabs.HC_EV_NM / wl  # ~5.0 eV
    e_k = photon + 0.3  # E_i = 0.3 eV < 0.74
    bright = LineObservation(
        wavelength_nm=wl,
        intensity=1000.0,
        intensity_uncertainty=2.0,
        element="Fe",
        ionization_stage=1,
        E_k_ev=e_k,
        g_k=5,
        A_ki=1.0e8,
        aki_uncertainty=0.05,
    )
    # two faint companions to set the element median below `bright`.
    faint1 = LineObservation(
        wavelength_nm=300.0,
        intensity=10.0,
        intensity_uncertainty=1.0,
        element="Fe",
        ionization_stage=1,
        E_k_ev=6.0,
        g_k=5,
        A_ki=1.0e8,
        aki_uncertainty=0.05,
    )
    faint2 = LineObservation(
        wavelength_nm=310.0,
        intensity=12.0,
        intensity_uncertainty=1.0,
        element="Fe",
        ionization_stage=1,
        E_k_ev=6.0,
        g_k=5,
        A_ki=1.0e8,
        aki_uncertainty=0.05,
    )
    return [bright, faint1, faint2]


def _planck_lines():
    """Two isolated (no-doublet) lines for the Planck-ceiling pass.

    No shared upper level => find_doublet_pairs yields nothing, so the doublet
    pass is a no-op and the Planck pass is the only correction path. E_k is set
    high so E_i = E_k - hc/lambda exceeds the suspect cut (these lines are NOT
    SA-suspect; the Planck pass is what touches them).
    """
    common = dict(
        element="Fe",
        ionization_stage=1,
        g_k=5,
        aki_uncertainty=0.05,
        intensity_uncertainty=1.0,
    )
    # high E_k => high E_i => not suspect.
    l0 = LineObservation(wavelength_nm=400.0, intensity=500.0, E_k_ev=10.0, A_ki=1.0e8, **common)
    l1 = LineObservation(wavelength_nm=500.0, intensity=300.0, E_k_ev=9.0, A_ki=2.0e7, **common)
    return [l0, l1]


def _peak_for_tau(wavelength_nm, temperature_K, tau_target):
    """Peak spectral radiance producing exactly ``tau_target`` at the ceiling.

    Inverts ``tau = -ln(1 - peak/B)`` => ``peak = B * (1 - exp(-tau))`` using the
    REAL reference Planck radiance so reference and kernel see identical peaks.
    """
    from cflibs.inversion.physics.self_absorption_observable import planck_spectral_radiance

    b = planck_spectral_radiance(wavelength_nm, temperature_K)
    return b * (1.0 - np.exp(-tau_target))


def _adversarial_triplet():
    """Three Fe I lines sharing ONE upper level -> one line in two usable pairs.

    A genuine multiplet (k -> i1, k -> i2, k -> i3) all from the same upper
    level. find_doublet_pairs yields three pairs (01, 02, 12); each line
    appears in two pairs, so the first-usable-pair-wins / scatter-min claim
    rule is exercised. Intensities are perturbed so all three deviations clear
    the significance floor.
    """
    g_k = 11
    e_k = 5.10
    base = dict(
        element="Fe",
        ionization_stage=1,
        E_k_ev=e_k,
        g_k=g_k,
        aki_uncertainty=0.05,
        intensity_uncertainty=1.0,
    )
    l0 = LineObservation(wavelength_nm=350.0, intensity=200.0, A_ki=2.0e8, **base)
    l1 = LineObservation(wavelength_nm=360.0, intensity=150.0, A_ki=1.2e8, **base)
    l2 = LineObservation(wavelength_nm=372.0, intensity=90.0, A_ki=6.0e7, **base)
    return [l0, l1, l2]


# ---------------------------------------------------------------------------
# Reference-extraction helpers.
# ---------------------------------------------------------------------------


def _ref_sets(result, obs_list):
    """Extract per-index corrected/suspect/cleared sets from the reference run.

    The reference keys corrections by wavelength; we map back to obs index.
    A line is 'corrected' if its output intensity differs from input AND its
    correction method is 'doublet'/'planck'; 'suspect' if its correction is a
    suspect record; 'cleared' if a doublet observable touched it (doublet,
    doublet-thin, or doublet correction).
    """
    corrected = set()
    suspect = set()
    cleared = set()
    wl_to_idx = {}
    for i, o in enumerate(obs_list):
        wl_to_idx.setdefault(o.wavelength_nm, i)
    for wl, c in result.corrections.items():
        i = wl_to_idx.get(wl)
        if i is None:
            continue
        if c.method == "doublet":
            corrected.add(i)
            cleared.add(i)
        elif c.method == "planck":
            corrected.add(i)
            cleared.add(i)
        elif c.method == "doublet-thin":
            cleared.add(i)
        elif c.method == "suspect":
            suspect.add(i)
    return corrected, suspect, cleared


# ---------------------------------------------------------------------------
# Tests — §4 contract.
# ---------------------------------------------------------------------------


def test_doublet_tau_and_intensity_parity():
    """tau atol 1e-6 + corrected intensity rtol 1e-6 vs the real reference."""
    obs = _doublet_self_absorbed()
    ref = _run_reference(obs)
    out = _run_kernel(obs)

    # The reference produced a genuine doublet correction.
    res = correct_via_doublet_ratio(obs[0], obs[1])
    assert res.tau_1 > 1e-3  # genuinely thick fixture

    ref_out = ref.observations
    for i in range(len(obs)):
        np.testing.assert_allclose(
            float(out.intensity[i]), ref_out[i].intensity, rtol=1e-6, atol=0.0
        )
        np.testing.assert_allclose(
            float(out.intensity_unc[i]),
            ref_out[i].intensity_uncertainty,
            rtol=1e-6,
            atol=0.0,
        )
    # per-pair tau atol 1e-6 (kernel records per-line tau; line0 is the
    # shorter-wavelength = line1 of the reference pair).
    np.testing.assert_allclose(float(out.tau[0]), res.tau_1, atol=1e-6)
    np.testing.assert_allclose(float(out.tau[1]), res.tau_2, atol=1e-6)


def test_doublet_sets_and_counters_parity():
    """Identical corrected/suspect/cleared sets + exact counters."""
    obs = _doublet_self_absorbed()
    ref = _run_reference(obs)
    out = _run_kernel(obs)

    ref_corrected, ref_suspect, ref_cleared = _ref_sets(ref, obs)

    k_corrected = {i for i in range(len(obs)) if int(out.method[i]) == selfabs.METHOD_DOUBLET}
    k_suspect = {i for i in range(len(obs)) if bool(out.suspect[i])}
    # cleared = corrected or doublet-thin in the kernel.
    k_cleared = {
        i
        for i in range(len(obs))
        if int(out.method[i]) in (selfabs.METHOD_DOUBLET, selfabs.METHOD_DOUBLET_THIN)
    }

    assert k_corrected == ref_corrected
    assert k_suspect == ref_suspect
    assert k_cleared == ref_cleared

    assert int(out.n_corrected) == ref.n_corrected
    assert int(out.n_suspect) == ref.n_suspect
    np.testing.assert_allclose(float(out.max_tau), ref.max_tau, atol=1e-6)


def test_suspect_pass_parity():
    """Suspect down-weighting matches the reference (inflate sigma, never boost)."""
    obs = _suspect_line()
    ref = _run_reference(obs)
    out = _run_kernel(obs)

    ref_out = ref.observations
    for i in range(len(obs)):
        # intensities are NEVER changed by the suspect pass.
        np.testing.assert_allclose(
            float(out.intensity[i]), ref_out[i].intensity, rtol=1e-6, atol=0.0
        )
        np.testing.assert_allclose(
            float(out.intensity_unc[i]),
            ref_out[i].intensity_uncertainty,
            rtol=1e-6,
            atol=0.0,
        )
    # the bright low-E_i line (index 0) is suspect; faint ones are not.
    assert bool(out.suspect[0])
    assert not bool(out.suspect[1])
    assert not bool(out.suspect[2])
    assert int(out.n_suspect) == ref.n_suspect == 1


def test_adversarial_triplet_claim_resolution():
    """Triplet sharing one upper level: one line in two usable pairs.

    The kernel's scatter-min claim rule must reproduce the reference's
    first-usable-pair-wins outcome on this multiplet (J5 spec §5.2).
    """
    obs = _adversarial_triplet()
    pairs = find_doublet_pairs(obs)
    assert len(pairs) == 3  # (0,1), (0,2), (1,2) all share the upper level

    ref = _run_reference(obs)
    out = _run_kernel(obs)

    ref_corrected, ref_suspect, ref_cleared = _ref_sets(ref, obs)
    k_corrected = {i for i in range(len(obs)) if int(out.method[i]) == selfabs.METHOD_DOUBLET}
    k_cleared = {
        i
        for i in range(len(obs))
        if int(out.method[i]) in (selfabs.METHOD_DOUBLET, selfabs.METHOD_DOUBLET_THIN)
    }
    k_suspect = {i for i in range(len(obs)) if bool(out.suspect[i])}

    assert k_corrected == ref_corrected, (k_corrected, ref_corrected)
    assert k_cleared == ref_cleared, (k_cleared, ref_cleared)
    assert k_suspect == ref_suspect, (k_suspect, ref_suspect)

    # And the corrected intensities still match where both agree.
    ref_out = ref.observations
    for i in ref_corrected:
        np.testing.assert_allclose(float(out.intensity[i]), ref_out[i].intensity, rtol=1e-6)
    assert int(out.n_corrected) == ref.n_corrected


def test_observably_thin_doublet_cleared():
    """A doublet with r_meas == r_thin is cleared (tau ~ 0), not corrected."""
    g_k, e_k = 7, 3.5
    # set intensities exactly to the thin ratio so deviation = 0.
    a1, a2 = 1.0e8, 4.0e7
    wl1, wl2 = 400.0, 410.0
    r_thin = (g_k * a1 / wl1) / (g_k * a2 / wl2)
    i2 = 100.0
    i1 = r_thin * i2  # makes r_meas == r_thin
    common = dict(
        element="Fe",
        ionization_stage=1,
        E_k_ev=e_k,
        g_k=g_k,
        aki_uncertainty=0.05,
        intensity_uncertainty=1.0,
    )
    l1 = LineObservation(wavelength_nm=wl1, intensity=i1, A_ki=a1, **common)
    l2 = LineObservation(wavelength_nm=wl2, intensity=i2, A_ki=a2, **common)
    obs = [l1, l2]
    out = _run_kernel(obs)
    # Both cleared as doublet-thin, none corrected.
    assert int(out.method[0]) == selfabs.METHOD_DOUBLET_THIN
    assert int(out.method[1]) == selfabs.METHOD_DOUBLET_THIN
    assert int(out.n_corrected) == 0
    # intensities unchanged.
    np.testing.assert_allclose(float(out.intensity[0]), i1, rtol=1e-12)


# ---------------------------------------------------------------------------
# Tests — Planck-ceiling pass (ladder step (b)).
# ---------------------------------------------------------------------------


def test_planck_radiance_and_tau_match_reference():
    """Kernel Planck radiance / ceiling-tau scalars match the real reference."""
    from cflibs.inversion.physics.self_absorption_observable import (
        planck_ceiling_optical_depth,
        planck_spectral_radiance,
    )

    rng = np.random.default_rng(7)
    for _ in range(80):
        wl = float(rng.uniform(200.0, 800.0))
        T = float(rng.uniform(4000.0, 12000.0))
        b_ref = planck_spectral_radiance(wl, T)
        b_ker = float(selfabs.planck_spectral_radiance_arr(jnp.array(wl), jnp.array(T)))
        np.testing.assert_allclose(b_ker, b_ref, rtol=1e-12, atol=0.0)

        tau_target = float(rng.uniform(0.01, 2.9))
        peak = b_ref * (1.0 - np.exp(-tau_target))
        tau_ref = planck_ceiling_optical_depth(peak, wl, T)
        tau_ker, det = selfabs.planck_ceiling_optical_depth_arr(
            jnp.array(peak), jnp.array(wl), jnp.array(T)
        )
        assert bool(det)
        np.testing.assert_allclose(float(tau_ker), tau_ref, atol=1e-6)


def test_planck_ceiling_none_branches():
    """Saturated (peak>=B) and B<=0 inputs are 'undeterminable' (ref None)."""
    from cflibs.inversion.physics.self_absorption_observable import (
        planck_ceiling_optical_depth,
        planck_spectral_radiance,
    )

    wl, T = 400.0, 8000.0
    b = planck_spectral_radiance(wl, T)
    # peak >= B => ref returns None.
    _, det = selfabs.planck_ceiling_optical_depth_arr(
        jnp.array(b * 1.5), jnp.array(wl), jnp.array(T)
    )
    assert not bool(det)
    assert planck_ceiling_optical_depth(b * 1.5, wl, T) is None
    # T <= 0 => B == 0 => ref returns None.
    _, det0 = selfabs.planck_ceiling_optical_depth_arr(
        jnp.array(b * 0.5), jnp.array(wl), jnp.array(0.0)
    )
    assert not bool(det0)
    # peak <= 0 => ref returns 0.0 (determinable).
    tau_z, det_z = selfabs.planck_ceiling_optical_depth_arr(
        jnp.array(0.0), jnp.array(wl), jnp.array(T)
    )
    assert bool(det_z)
    np.testing.assert_allclose(float(tau_z), 0.0, atol=0.0)
    assert planck_ceiling_optical_depth(0.0, wl, T) == 0.0


def test_doppler_cog_escape_factor_matches_reference():
    """64-term COG series matches the reference over the validity range."""
    from cflibs.inversion.physics.self_absorption_observable import doppler_cog_escape_factor

    taus = np.concatenate([[0.0, 1e-12, 1e-9], np.linspace(1e-4, 3.0, 60)])
    ker = np.asarray(selfabs.doppler_cog_escape_factor_arr(jnp.asarray(taus)))
    for i, t in enumerate(taus):
        ref = doppler_cog_escape_factor(float(t))
        np.testing.assert_allclose(ker[i], ref, rtol=1e-12, atol=0.0)


def test_planck_pass_intensity_and_counter_parity():
    """Full kernel Planck pass vs reference: intensity rtol 1e-6, tau atol 1e-6."""
    obs = _planck_lines()
    T = 8000.0
    # land line0 in the valid range (tau~0.8); line1 just over the ceiling.
    peak0 = _peak_for_tau(obs[0].wavelength_nm, T, 0.8)
    peak1 = _peak_for_tau(obs[1].wavelength_nm, T, 3.5)  # > PLANCK_TAU_VALIDITY_MAX
    peak_by_wl = {obs[0].wavelength_nm: peak0, obs[1].wavelength_nm: peak1}

    ref = _run_reference(obs, temperature_K=T, peak_spectral_radiance=peak_by_wl)

    peak, valid = _peak_radiance_arrays(obs, peak_by_wl)
    inp = _kernel_inputs(obs)
    out = selfabs.correct_self_absorption_arrays(
        **inp,
        peak_spectral_radiance=peak,
        peak_radiance_valid=valid,
        temperature_K=T,
    )

    ref_out = ref.observations
    for i in range(len(obs)):
        np.testing.assert_allclose(
            float(out.intensity[i]), ref_out[i].intensity, rtol=1e-6, atol=0.0
        )
        np.testing.assert_allclose(
            float(out.intensity_unc[i]), ref_out[i].intensity_uncertainty, rtol=1e-6, atol=0.0
        )

    # line0 corrected via Planck; line1 over-ceiling => untouched (not corrected).
    assert int(out.method[0]) == selfabs.METHOD_PLANCK
    assert int(out.method[1]) != selfabs.METHOD_PLANCK
    np.testing.assert_allclose(float(out.intensity[1]), obs[1].intensity, rtol=1e-12)

    # tau parity for the valid line.
    ref_corr0 = ref.corrections[obs[0].wavelength_nm]
    np.testing.assert_allclose(float(out.tau[0]), ref_corr0.tau, atol=1e-6)

    # counters exact.
    assert int(out.n_corrected) == ref.n_corrected == 1
    assert int(out.n_suspect) == ref.n_suspect
    np.testing.assert_allclose(float(out.max_tau), ref.max_tau, atol=1e-6)


def test_planck_pass_off_when_no_temperature():
    """temperature_K<=0 disables the Planck pass entirely (reference no-op)."""
    obs = _planck_lines()
    peak_by_wl = {obs[0].wavelength_nm: _peak_for_tau(obs[0].wavelength_nm, 8000.0, 1.0)}
    peak, valid = _peak_radiance_arrays(obs, peak_by_wl)
    inp = _kernel_inputs(obs)
    # T=0 => off. Reference with temperature_K=None also runs no Planck pass.
    out = selfabs.correct_self_absorption_arrays(
        **inp, peak_spectral_radiance=peak, peak_radiance_valid=valid, temperature_K=0.0
    )
    ref = _run_reference(obs)  # no temperature => no Planck pass
    for i in range(len(obs)):
        np.testing.assert_allclose(
            float(out.intensity[i]), ref.observations[i].intensity, rtol=1e-12
        )
        assert int(out.method[i]) != selfabs.METHOD_PLANCK
    assert int(out.n_corrected) == ref.n_corrected == 0


def _doublet_corrected_in_range():
    """A doublet whose recovered tau lands inside the validity range (<5).

    Unlike ``_doublet_self_absorbed`` (tau_1 ~ 6.6 => force-suspect), this pair
    is genuinely *corrected* (method='doublet', added to `cleared`), so it must
    be protected from the later Planck pass.
    """
    g_k, e_k = 9, 4.30
    a1, a2 = 1.0e8, 5.0e7
    wl1, wl2 = 370.0, 380.0
    r_thin = (g_k * a1 / wl1) / (g_k * a2 / wl2)
    common = dict(
        element="Fe",
        ionization_stage=1,
        E_k_ev=e_k,
        g_k=g_k,
        aki_uncertainty=0.05,
        intensity_uncertainty=1.0,
    )
    # deviation clears the 0.10 floor but recovered tau stays below the ceiling.
    i2 = 100.0
    i1 = r_thin * i2 * 0.8  # 20% below the thin ratio
    l1 = LineObservation(wavelength_nm=wl1, intensity=i1, A_ki=a1, **common)
    l2 = LineObservation(wavelength_nm=wl2, intensity=i2, A_ki=a2, **common)
    return [l1, l2]


def test_planck_does_not_touch_doublet_cleared_lines():
    """Doublet-cleared lines are exempt from the Planck pass (ref ordering)."""
    obs = _doublet_corrected_in_range()
    # confirm the fixture is genuinely doublet-corrected (tau < 5).
    res = correct_via_doublet_ratio(obs[0], obs[1])
    assert 1e-3 < max(res.tau_1, res.tau_2) < selfabs.DOUBLET_TAU_VALIDITY_MAX
    T = 9000.0
    # supply (large) peak radiance for BOTH doublet members; the doublet pass
    # claims them first, so the Planck pass must leave them as doublet results.
    peak_by_wl = {
        obs[0].wavelength_nm: _peak_for_tau(obs[0].wavelength_nm, T, 2.0),
        obs[1].wavelength_nm: _peak_for_tau(obs[1].wavelength_nm, T, 2.0),
    }
    ref = _run_reference(obs, temperature_K=T, peak_spectral_radiance=peak_by_wl)
    peak, valid = _peak_radiance_arrays(obs, peak_by_wl)
    inp = _kernel_inputs(obs)
    out = selfabs.correct_self_absorption_arrays(
        **inp, peak_spectral_radiance=peak, peak_radiance_valid=valid, temperature_K=T
    )
    for i in range(len(obs)):
        # method stays DOUBLET (Planck never overrides a doublet-cleared line).
        assert int(out.method[i]) == selfabs.METHOD_DOUBLET
        np.testing.assert_allclose(
            float(out.intensity[i]), ref.observations[i].intensity, rtol=1e-6
        )
    assert int(out.n_corrected) == ref.n_corrected


def test_planck_grad_finite():
    """grad of corrected intensity wrt (intensity, peak, T) is finite (J7 hook)."""
    obs = _planck_lines()
    T0 = 8000.0
    peak_by_wl = {obs[0].wavelength_nm: _peak_for_tau(obs[0].wavelength_nm, T0, 1.0)}
    peak, valid = _peak_radiance_arrays(obs, peak_by_wl)
    inp = _kernel_inputs(obs)

    def loss(intensity, peak_arr, temp):
        out = selfabs.correct_self_absorption_arrays(
            intensity,
            inp["intensity_unc"],
            inp["aki_unc"],
            inp["line_index"],
            inp["line_valid"],
            inp["line_element"],
            inp["pair_idx"],
            inp["pair_valid"],
            inp["snapshot"],
            peak_spectral_radiance=peak_arr,
            peak_radiance_valid=valid,
            temperature_K=temp,
        )
        return jnp.sum(out.intensity) + jnp.sum(jnp.nan_to_num(out.tau))

    g = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.asarray(inp["intensity"]), jnp.asarray(peak), jnp.asarray(T0)
    )
    for gi in g:
        assert np.all(np.isfinite(np.asarray(gi)))


def test_planck_vmap_batch16():
    """vmap (B=16) over the Planck path; every row matches the single run."""
    obs = _planck_lines()
    T = 8000.0
    peak_by_wl = {obs[0].wavelength_nm: _peak_for_tau(obs[0].wavelength_nm, T, 0.9)}
    peak, valid = _peak_radiance_arrays(obs, peak_by_wl)
    inp = _kernel_inputs(obs)
    single = selfabs.correct_self_absorption_arrays(
        **inp, peak_spectral_radiance=peak, peak_radiance_valid=valid, temperature_K=T
    )

    B = 16

    def _call(intensity, peak_arr, valid_arr):
        return selfabs.correct_self_absorption_arrays(
            intensity,
            jnp.asarray(inp["intensity_unc"]),
            jnp.asarray(inp["aki_unc"]),
            jnp.asarray(inp["line_index"]),
            jnp.asarray(inp["line_valid"]),
            jnp.asarray(inp["line_element"]),
            jnp.asarray(inp["pair_idx"]),
            jnp.asarray(inp["pair_valid"]),
            inp["snapshot"],
            peak_spectral_radiance=peak_arr,
            peak_radiance_valid=valid_arr,
            temperature_K=T,
        )

    bi = jnp.broadcast_to(jnp.asarray(inp["intensity"]), (B,) + inp["intensity"].shape)
    bp = jnp.broadcast_to(jnp.asarray(peak), (B,) + peak.shape)
    bv = jnp.broadcast_to(jnp.asarray(valid), (B,) + valid.shape)
    res = jax.jit(jax.vmap(_call))(bi, bp, bv)
    assert res.intensity.shape == (B, len(obs))
    for b in range(B):
        np.testing.assert_allclose(
            np.asarray(res.intensity[b]), np.asarray(single.intensity), rtol=1e-12
        )
        assert int(res.n_corrected[b]) == int(single.n_corrected)


# ---------------------------------------------------------------------------
# Property: bisection vs brentq.
# ---------------------------------------------------------------------------


def test_bisection_matches_brentq_property():
    """Randomized (rho, ratio_of_ratios) grid: bisection tau == brentq tau."""
    from scipy.optimize import brentq

    rng = np.random.default_rng(0)
    n_ok = 0
    for _ in range(200):
        rho = float(rng.uniform(1.01, 5.0))  # rho >= 1 by construction
        # pick a true tau in the validity range, derive the consistent RoR so a
        # root exists in the bracket.
        tau_true = float(rng.uniform(0.05, 8.0))

        def f(t, _rho=rho):
            return selfabs._escape_ref(t) / selfabs._escape_ref(t / _rho)

        ror = f(tau_true)
        # reference brentq path
        res_low = selfabs._escape_ref(1e-4) / selfabs._escape_ref(1e-4 / rho) - ror
        res_high = selfabs._escape_ref(30.0) / selfabs._escape_ref(30.0 / rho) - ror
        if res_low * res_high > 0:
            continue
        tau_brentq = brentq(
            lambda t, _rho=rho, _ror=ror: f(t, _rho) - _ror,
            1e-4,
            30.0,
            xtol=1e-6,
            rtol=1e-8,
            maxiter=100,
        )
        tau_kernel = float(
            selfabs.solve_doublet_tau(jnp.array([rho]), jnp.array([ror]), jnp.array([1.0]))[0]
        )
        np.testing.assert_allclose(tau_kernel, tau_brentq, atol=1e-6)
        n_ok += 1
    assert n_ok > 50  # the grid actually exercised the root path


# ---------------------------------------------------------------------------
# Invariants.
# ---------------------------------------------------------------------------


def test_never_boost_and_suspect_only_inflates():
    """Corrected intensity <= boosted bound; suspects only inflate sigma."""
    obs = _adversarial_triplet() + _suspect_line()
    out = _run_kernel(obs)
    for i in range(len(obs)):
        if int(out.method[i]) == selfabs.METHOD_DOUBLET:
            # boosted: I_corr = I / f(tau) with f<=1 -> I_corr >= I, and tau>0.
            assert float(out.intensity[i]) >= obs[i].intensity - 1e-9
        if bool(out.suspect[i]):
            # never boosted: intensity unchanged for suspects.
            np.testing.assert_allclose(float(out.intensity[i]), obs[i].intensity, rtol=1e-12)
            # sigma inflated.
            assert float(out.intensity_unc[i]) >= obs[i].intensity_uncertainty - 1e-9


# ---------------------------------------------------------------------------
# jit / vmap / grad / padding invariance.
# ---------------------------------------------------------------------------


def _stack_inputs(inp, B):
    """Replicate a single-spectrum input dict B times along a leading axis."""
    batched = {}
    for k, v in inp.items():
        if k == "snapshot":
            batched[k] = v
            continue
        arr = jnp.asarray(v)
        batched[k] = jnp.broadcast_to(arr, (B,) + arr.shape)
    return batched


def test_jit_and_vmap_batch16():
    obs = _adversarial_triplet() + _suspect_line()
    inp = _kernel_inputs(obs)
    single = selfabs.correct_self_absorption_arrays(**inp)

    B = 16
    batched = _stack_inputs(inp, B)

    def _call(
        intensity,
        intensity_unc,
        aki_unc,
        line_index,
        line_valid,
        line_element,
        pair_idx,
        pair_valid,
    ):
        return selfabs.correct_self_absorption_arrays(
            intensity,
            intensity_unc,
            aki_unc,
            line_index,
            line_valid,
            line_element,
            pair_idx,
            pair_valid,
            inp["snapshot"],
        )

    vfun = jax.jit(jax.vmap(_call))
    res = vfun(
        batched["intensity"],
        batched["intensity_unc"],
        batched["aki_unc"],
        batched["line_index"],
        batched["line_valid"],
        batched["line_element"],
        batched["pair_idx"],
        batched["pair_valid"],
    )
    assert res.intensity.shape == (B, len(obs))
    # every batch row matches the single-spectrum result.
    for b in range(B):
        np.testing.assert_allclose(
            np.asarray(res.intensity[b]), np.asarray(single.intensity), rtol=1e-12
        )
        assert int(res.n_corrected[b]) == int(single.n_corrected)


def test_grad_finite():
    """grad of a scalar reduction of corrected intensity wrt inputs is finite."""
    obs = _doublet_self_absorbed()
    inp = _kernel_inputs(obs)

    def loss(intensity, intensity_unc, aki_unc):
        out = selfabs.correct_self_absorption_arrays(
            intensity,
            intensity_unc,
            aki_unc,
            inp["line_index"],
            inp["line_valid"],
            inp["line_element"],
            inp["pair_idx"],
            inp["pair_valid"],
            inp["snapshot"],
        )
        return jnp.sum(out.intensity) + jnp.sum(jnp.nan_to_num(out.tau))

    g = jax.grad(loss, argnums=(0, 1, 2))(
        jnp.asarray(inp["intensity"]),
        jnp.asarray(inp["intensity_unc"]),
        jnp.asarray(inp["aki_unc"]),
    )
    for gi in g:
        assert np.all(np.isfinite(np.asarray(gi)))


def test_padding_invariance():
    """Re-running at a larger pad size is bit-identical on the valid region."""
    obs = _adversarial_triplet() + _suspect_line()
    base = selfabs.correct_self_absorption_arrays(**_kernel_inputs(obs))
    padded = selfabs.correct_self_absorption_arrays(**_kernel_inputs(obs, pad_to=64))
    n = len(obs)
    np.testing.assert_array_equal(np.asarray(base.intensity), np.asarray(padded.intensity[:n]))
    np.testing.assert_array_equal(
        np.asarray(base.intensity_unc), np.asarray(padded.intensity_unc[:n])
    )
    np.testing.assert_array_equal(np.asarray(base.method), np.asarray(padded.method[:n]))
    np.testing.assert_array_equal(np.asarray(base.suspect), np.asarray(padded.suspect[:n]))
    assert int(base.n_corrected) == int(padded.n_corrected)
    assert int(base.n_suspect) == int(padded.n_suspect)
    np.testing.assert_allclose(float(base.max_tau), float(padded.max_tau), atol=0)


# ---------------------------------------------------------------------------
# Import hygiene — no SQLite in the kernel module.
# ---------------------------------------------------------------------------


def test_no_sqlite_in_kernel_module():
    """The kernel module imports no SQLite-backed code (J5 spec hard rule)."""
    path = Path(selfabs.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    banned = {"sqlite3", "cflibs.atomic.database", "cflibs.io", "cflibs.jitpipe.host"}
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name in banned or a.name.startswith("cflibs.io."):
                    found.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod in banned or mod == "cflibs.io" or mod.startswith("cflibs.io."):
                found.add(mod)
            if mod == "cflibs.jitpipe" and any(a.name == "host" for a in node.names):
                found.add("cflibs.jitpipe.host")
    assert not found, found
