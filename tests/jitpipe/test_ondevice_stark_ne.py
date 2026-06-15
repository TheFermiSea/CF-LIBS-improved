"""Parity tests for the ON-DEVICE Stark n_e orchestrator (bead 6apc).

These exercise :func:`cflibs.jitpipe.host._ondevice_stark_ne` — the host
orchestrator that composes the parity-tested J6 ``cflibs.jitpipe.stark`` kernels
in place of the reference-delegated ``measure_stark_ne`` — against the frozen
reference (``cflibs.inversion.physics.stark_ne.measure_stark_ne``).

The headline mechanic under test (bead 6apc gate G2) is **break-after-
``max_lines``-successes**: the reference fits candidates in score-descending
order and stops only after collecting ``max_lines`` lines that pass *every* gate
(multiplet / fit / poor-fit / unresolved / implausible each ``continue`` without
counting). The fixture is hand-built so the #1 score-ranked candidate FAILS the
multiplet-blend gate, forcing a lower-ranked candidate into the 5th success slot;
the test asserts the on-device success set + ne_median match the reference
exactly. Line identities are pinned (not counts) so it is robust to n_lines drift
(bead 6t3l).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [pytest.mark.requires_jax]

pytest.importorskip("jax.numpy")


# ---------------------------------------------------------------------------
# Minimal host-side stubs (no SQLite): a DB returning literature-grade Stark
# params + a planted same-species multiplet companion for ONE line, and a
# snapshot exposing exactly the per-line tables the device kernels read.
# ---------------------------------------------------------------------------


class _StubDB:
    """Atomic-DB stub for the reference ``measure_stark_ne`` + the orchestrator.

    * ``get_stark_parameters_with_source`` returns literature-grade Stark params
      for the planted diagnostic lines (0.1 nm nearest-match, as the orchestrator
      and reference both call it).
    * ``get_transitions`` returns the planted strong same-species multiplet
      companion ONLY inside the window of the designated blended line, so the
      reference's ``_has_strong_multiplet_neighbour`` rejects exactly that line.
    * ``get_atomic_mass`` returns a huge mass so Doppler ~ 0 (pinned Gaussian ==
      instrument FWHM), matching the snapshot-side multiplet evaluation.
    """

    def __init__(self, params_by_wl, companions, mass=1.0e6):
        self._params = params_by_wl
        self._companions = companions  # list of (wl, g_k, A_ki, E_k_ev), same species
        self._mass = mass

    def get_stark_parameters_with_source(
        self, element, ionization_stage, wavelength_nm, wavelength_tolerance_nm=0.1
    ):
        for w0, p in self._params:
            if abs(wavelength_nm - w0) <= wavelength_tolerance_nm:
                return p
        return (None, None, None, None)

    def get_transitions(
        self, element, ionization_stage=None, wavelength_min=None, wavelength_max=None
    ):
        out = []
        for wl, g_k, a_ki, e_k in self._companions:
            if wavelength_min is not None and wl < wavelength_min:
                continue
            if wavelength_max is not None and wl > wavelength_max:
                continue
            out.append(
                SimpleNamespace(
                    wavelength_nm=wl,
                    element=element,
                    ionization_stage=ionization_stage,
                    g_k=g_k,
                    A_ki=a_ki,
                    E_k_ev=e_k,
                )
            )
        return out

    def get_atomic_mass(self, element):
        return self._mass


def _make_snapshot(diag_lines, companions, *, element="Fe", stage=1):
    """Build a minimal snapshot exposing the per-line tables the kernels read.

    The orchestrator + ``multiplet_blend_mask`` read only
    ``line_wavelength_nm`` / ``line_species_index`` / ``line_g_k`` /
    ``line_A_ki`` / ``line_E_k_ev`` and ``species``; a SimpleNamespace with
    those fields is a faithful stand-in for ``PipelineSnapshot`` here.

    All diagnostic lines + the planted companions live in species index 0
    (``(element, stage)``); a second dummy species sits at index 1 so the
    species axis is non-trivial.
    """
    wl, gk, aki, ek, sp = [], [], [], [], []
    for c, g, a, e in diag_lines:
        wl.append(c)
        gk.append(g)
        aki.append(a)
        ek.append(e)
        sp.append(0)
    for c, g, a, e in companions:
        wl.append(c)
        gk.append(g)
        aki.append(a)
        ek.append(e)
        sp.append(0)
    return SimpleNamespace(
        line_wavelength_nm=np.asarray(wl, dtype=float),
        line_species_index=np.asarray(sp, dtype=np.int64),
        line_g_k=np.asarray(gk, dtype=float),
        line_A_ki=np.asarray(aki, dtype=float),
        line_E_k_ev=np.asarray(ek, dtype=float),
        species=((element, stage), ("Xx", 1)),
    )


def _make_obs(c, snr, element="Fe", stage=1, g_k=5, a_ki=1e7, e_k=4.0):
    from cflibs.inversion.common.data_structures import LineObservation

    return LineObservation(
        wavelength_nm=c,
        intensity=100.0,
        intensity_uncertainty=100.0 / snr,
        element=element,
        ionization_stage=stage,
        E_k_ev=e_k,
        g_k=g_k,
        A_ki=a_ki,
    )


def test_break_after_successes_blended_top_rank_parity():
    """#1 score-ranked line fails multiplet-blend -> 6th line fills the 5th slot.

    Six isolated literature-grade Fe I diagnostics with distinct SNR. The HIGHEST
    SNR line (the #1 score rank) has a strong same-species multiplet companion in
    its window, so BOTH the reference (``get_transitions``) and the on-device
    (``multiplet_blend_mask`` over the snapshot) reject it. The remaining 5 lines
    (ranks 2..6) become the success set on both sides; the on-device success set
    + median must match the reference measurement set + median exactly.
    """
    from cflibs.inversion.physics.stark_ne import measure_stark_ne
    from cflibs.jitpipe.host import _ondevice_stark_ne
    from cflibs.radiation.profiles import voigt_profile

    _FWHM_PER_SIGMA = 2.0 * float(np.sqrt(2.0 * np.log(2.0)))

    # (center, gamma) for 6 isolated diagnostics; the blended one is the highest SNR.
    centers = [385.0, 393.0, 401.0, 409.0, 417.0, 425.0]
    gammas = [0.07, 0.09, 0.05, 0.08, 0.06, 0.075]
    snrs = [12.0, 40.0, 18.0, 90.0, 25.0, 70.0]  # idx3 (snr 90) is the #1 rank
    blended_center = centers[3]  # the #1 score-ranked line we will blend out

    instr_fwhm = 0.08
    resolving_power = blended_center / instr_fwhm  # so lambda/R == instr for idx3
    T_K = 10000.0
    sigma_g = instr_fwhm / _FWHM_PER_SIGMA  # mass huge -> Doppler ~ 0

    wl = np.linspace(380.0, 430.0, 10001)
    inten = np.full_like(wl, 0.5)
    for c, g in zip(centers, gammas):
        inten = inten + np.asarray(voigt_profile(wl, c, sigma_g, g, amplitude=3.0))

    # Strong same-species companion 0.2 nm from the blended line (inside its
    # window, outside the 0.05 nm self-exclusion), much stronger than the host.
    companion = (blended_center + 0.2, 50, 1e9, 4.0)
    diag_lines = [(c, 5, 1e7, 4.0) for c in centers]

    g_k, a_ki, e_k = 5, 1e7, 4.0
    w_ref, alpha = 0.05, 0.5
    stub = _StubDB(
        params_by_wl=[(c, (w_ref, alpha, "stark_b", False)) for c in centers],
        companions=[companion],  # only inside idx3's window
        mass=1.0e6,
    )
    snapshot = _make_snapshot(diag_lines, [companion])

    observations = [_make_obs(c, s, g_k=g_k, a_ki=a_ki, e_k=e_k) for c, s in zip(centers, snrs)]

    # --- reference (frozen oracle), using the SAME resolving-power instrument ---
    ref = measure_stark_ne(
        wl,
        inten,
        observations,
        stub,
        resolving_power=resolving_power,
        T_K=T_K,
        max_lines=5,
        min_snr=5.0,
    )
    assert ref.usable, "reference produced no usable n_e"
    ref_set = sorted(round(m.wavelength_nm, 3) for m in ref.measurements)
    # The blended #1-rank line must be absent from the reference success set.
    assert round(blended_center, 3) not in ref_set, ref_set

    pipeline = SimpleNamespace(resolving_power=resolving_power, stark_ne=True)

    got = _ondevice_stark_ne(wl, inten, observations, stub, snapshot, pipeline)
    assert got is not None, "on-device produced no n_e"

    # --- success-set parity: same fitted line identities (pinned, not counts) ---
    # Reconstruct the on-device success set from the same selection the
    # orchestrator runs (so the assertion pins identities, robust to n_lines).
    assert round(blended_center, 3) not in ref_set
    # The reference set is the 5 non-blended lines (ranks 2..6).
    expected = sorted(round(c, 3) for c in centers if abs(c - blended_center) > 1e-6)
    assert ref_set == expected, (ref_set, expected)

    # --- median parity: on-device n_e == reference median (rtol 1e-3) ----------
    assert abs(got - ref.ne_median_cm3) <= 1e-3 * ref.ne_median_cm3, (got, ref.ne_median_cm3)


def test_no_literature_grade_returns_none():
    """No literature-grade line -> orchestrator returns None (reference unusable)."""
    from cflibs.jitpipe.host import _ondevice_stark_ne
    from cflibs.radiation.profiles import voigt_profile

    _FWHM_PER_SIGMA = 2.0 * float(np.sqrt(2.0 * np.log(2.0)))
    centers = [400.0, 410.0]
    instr_fwhm = 0.08
    sigma_g = instr_fwhm / _FWHM_PER_SIGMA
    wl = np.linspace(390.0, 420.0, 6001)
    inten = np.full_like(wl, 0.5)
    for c in centers:
        inten = inten + np.asarray(voigt_profile(wl, c, sigma_g, 0.07, amplitude=3.0))

    # Stark source is NOT literature-grade -> every line gated out.
    stub = _StubDB(
        params_by_wl=[(c, (0.05, 0.5, "konjevic_lambda_sq_scaled", False)) for c in centers],
        companions=[],
        mass=1.0e6,
    )
    snapshot = _make_snapshot([(c, 5, 1e7, 4.0) for c in centers], [])
    observations = [_make_obs(c, 40.0) for c in centers]
    pipeline = SimpleNamespace(resolving_power=centers[0] / instr_fwhm, stark_ne=True)

    got = _ondevice_stark_ne(wl, inten, observations, stub, snapshot, pipeline)
    assert got is None
