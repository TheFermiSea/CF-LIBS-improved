"""Adapter contract test for the J10 forward-fitting identifier (wiring only).

Verifies that :class:`cflibs.jitpipe.forward_id_identifier.ForwardFitIdentifier`
— the host bridge from the jit forward-fit core to the scoreboard's
``IdentifierProtocol`` — produces a well-formed
:class:`~cflibs.inversion.common.element_id.ElementIdentificationResult` on a
small real 3-element (Fe/Cu/Ca) snapshot.

This is a *wiring* test, not an accuracy test: it asserts the result type, the
score/confidence bounds, the protocol conformance, and that the detected set is
a subset of the candidate elements. The recall/F1 *payoff* of the algorithm is
covered by ``tests/jitpipe/test_parity_j10.py`` (which exercises the jit core).

Kept small (n_configs=256, 3 elements, 400-point grid) so it runs in well under
a minute on CPU.
"""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

from cflibs.atomic.database import AtomicDatabase  # noqa: E402
from cflibs.core.jax_runtime import _ensure_pytrees_registered  # noqa: E402
from cflibs.inversion.common.element_id import (  # noqa: E402
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.inversion.identify._protocol import IdentifierProtocol  # noqa: E402
from cflibs.instrument.model import InstrumentModel  # noqa: E402
from cflibs.jitpipe.forward_id_identifier import ForwardFitIdentifier  # noqa: E402
from cflibs.jitpipe.snapshot import PipelineSnapshot  # noqa: E402
from cflibs.radiation.kernels import BroadeningMode, forward_model  # noqa: E402

pytestmark = [pytest.mark.requires_jax, pytest.mark.requires_db]

DB_PATH = "ASD_da/libs_production.db"
WL_RANGE = (300.0, 420.0)
N_WL = 400
CANDIDATES = ["Fe", "Cu", "Ca"]


@pytest.fixture(scope="module", autouse=True)
def _pytrees() -> None:
    _ensure_pytrees_registered()


@pytest.fixture(scope="module")
def setup():
    """Small real Fe/Cu/Ca snapshot + a measured spectrum from the frozen kernel."""
    from cflibs.plasma.state import SingleZoneLTEPlasma

    db = AtomicDatabase(DB_PATH)
    asnap = db.snapshot(elements=CANDIDATES, wavelength_range=WL_RANGE, include_levels=True)
    snap = PipelineSnapshot.from_atomic_snapshot(asnap)
    wl = jnp.linspace(WL_RANGE[0], WL_RANGE[1], N_WL)
    instr = InstrumentModel(resolution_fwhm_nm=0.1)

    # Measured spectrum: Fe + Cu present, Ca absent (a confounder).
    plasma = SingleZoneLTEPlasma(T_e=10000.0, n_e=1e17, species={"Fe": 0.7, "Cu": 0.3, "Ca": 0.0})
    meas = np.asarray(
        forward_model(
            plasma,
            snap.to_atomic_snapshot(),
            instr,
            wl,
            broadening_mode=BroadeningMode.NIST_PARITY,
            path_length_m=0.01,
        )
    )
    return snap, np.asarray(wl), instr, meas


def test_forward_fit_adapter_returns_well_formed_result(setup):
    """``identify`` returns a well-formed ElementIdentificationResult."""
    snap, wl, instr, meas = setup

    identifier = ForwardFitIdentifier(
        CANDIDATES,
        snapshot=snap,
        instrument=instr,
        n_configs=256,
        presence_threshold=0.02,
        seed=7,
    )

    # Protocol conformance (structural).
    assert isinstance(identifier, IdentifierProtocol)

    result = identifier.identify(wl, meas)

    # Right type and algorithm tag.
    assert isinstance(result, ElementIdentificationResult)
    assert result.algorithm == "forward_fit"

    # all_elements covers exactly the snapshot element superset.
    all_syms = {e.element for e in result.all_elements}
    assert all_syms == set(snap.element_symbols)
    assert len(result.all_elements) == len(snap.element_symbols)

    # detected + rejected partition all_elements.
    assert len(result.detected_elements) + len(result.rejected_elements) == len(result.all_elements)

    # Scores / confidences are finite and in [0, 1] for every element.
    for e in result.all_elements:
        assert isinstance(e, ElementIdentification)
        assert np.isfinite(e.score) and 0.0 <= e.score <= 1.0
        assert np.isfinite(e.confidence) and 0.0 <= e.confidence <= 1.0
        # Raw unbounded gap + BIC are carried in metadata.
        assert "presence_score" in e.metadata
        assert "best_bic" in e.metadata

    # detected subset of candidates.
    detected = {e.element for e in result.detected_elements}
    assert detected <= set(CANDIDATES)

    # Parameters round-trip the candidate list and global canaries.
    assert result.parameters["candidate_elements"] == CANDIDATES
    assert np.isfinite(result.parameters["best_correlation"])
    assert result.parameters["n_valid_configs"] == 256


def test_forward_fit_adapter_accepts_raw_atomic_snapshot(setup):
    """Regression: a raw ``AtomicSnapshot`` (no ``element_symbols``) must work.

    ``db.snapshot(...)`` returns the raw ``AtomicSnapshot``; the benchmark
    ``_run_forward_fit`` runner passed it straight through, hitting
    ``AttributeError: 'AtomicSnapshot' object has no attribute 'element_symbols'``
    (the adapter's own ``identify`` round-trip test used the *converted*
    ``PipelineSnapshot`` and missed this). The adapter now lifts a raw snapshot to
    a ``PipelineSnapshot`` defensively; this pins that path.
    """
    _snap, wl, instr, meas = setup

    db = AtomicDatabase(DB_PATH)
    raw = db.snapshot(elements=CANDIDATES, wavelength_range=WL_RANGE, include_levels=True)
    assert not hasattr(raw, "element_symbols"), "fixture must be the raw AtomicSnapshot"

    identifier = ForwardFitIdentifier(
        CANDIDATES, snapshot=raw, instrument=instr, n_configs=64, presence_threshold=0.02, seed=7
    )
    result = identifier.identify(wl, meas)

    assert isinstance(result, ElementIdentificationResult)
    assert result.algorithm == "forward_fit"
    assert {e.element for e in result.all_elements} == set(CANDIDATES)
    assert {e.element for e in result.detected_elements} <= set(CANDIDATES)


def _stub_snapshot(line_wl, line_el, element_symbols):
    """Minimal duck-typed snapshot carrying only the host arrays the diagnostic
    weight reads. Each line gets unit A_ki/g_k and zero E_k (constant strength)
    so the weight is driven purely by line placement and element membership."""
    from types import SimpleNamespace

    line_wl = np.asarray(line_wl, dtype=np.float64)
    n = line_wl.shape[0]
    return SimpleNamespace(
        element_symbols=tuple(element_symbols),
        line_element_index=np.asarray(line_el, dtype=np.int64),
        line_wavelength_nm=line_wl,
        line_A_ki=np.ones(n, dtype=np.float64),
        line_g_k=np.ones(n, dtype=np.float64),
        line_E_k_ev=np.zeros(n, dtype=np.float64),
    )


def test_ief_penalizes_shared_bins_below_isolated_bins():
    """With ``use_ief=True``, a bin where several elements emit gets a strictly
    lower diagnostic weight than an isolated single-element bin.

    Construct three elements (A/B/C) on a clean 1000-point grid: A, B, C each
    place a line at 410 nm (a *shared* crowded bin), while A alone places a line
    at 320 nm (an *isolated* bin). The IEF factor ``log((E+1)/f_i)`` is
    ``log(4/3)`` at the shared bin vs ``log(4/1)`` at the isolated bin, so the
    isolated bin must out-weigh the shared one.
    """
    wl = np.linspace(300.0, 430.0, 1000)
    instr = InstrumentModel(resolution_fwhm_nm=0.1)
    snap = _stub_snapshot(
        line_wl=[320.0, 410.0, 410.0, 410.0],
        line_el=[0, 0, 1, 2],  # A@320 isolated; A,B,C@410 shared
        element_symbols=("A", "B", "C"),
    )

    identifier = ForwardFitIdentifier(
        ["A", "B", "C"], snapshot=snap, instrument=instr, use_ief=True, ief_floor_frac=0.25
    )
    w = identifier._diagnostic_wavelength_weights(wl, snap, instr)

    assert np.all(np.isfinite(w))
    assert w.shape == wl.shape
    np.testing.assert_allclose(float(np.mean(w)), 1.0, rtol=1e-12)

    iso_bin = int(np.argmin(np.abs(wl - 320.0)))
    shared_bin = int(np.argmin(np.abs(wl - 410.0)))
    # The isolated single-element bin out-weighs the shared/crowded one.
    assert w[iso_bin] > w[shared_bin]


def test_ief_disabled_reproduces_diag_only_weights_exactly():
    """``use_ief=False`` reproduces the prior diagnostic-only weight array exactly.

    Two identifiers over the *same* snapshot/grid that differ only in ``use_ief``;
    the IEF factor is the only change to the weight pipeline, so the disabled path
    must be bit-identical (``assert_allclose`` rtol/atol 0) to the diag-only weight
    that predated this feature.
    """
    wl = np.linspace(300.0, 430.0, 1000)
    instr = InstrumentModel(resolution_fwhm_nm=0.1)
    snap = _stub_snapshot(
        line_wl=[320.0, 410.0, 410.0, 410.0],
        line_el=[0, 0, 1, 2],
        element_symbols=("A", "B", "C"),
    )

    diag_only = ForwardFitIdentifier(
        ["A", "B", "C"], snapshot=snap, instrument=instr, use_ief=False, weight_gamma=2.0
    )
    with_ief = ForwardFitIdentifier(
        ["A", "B", "C"], snapshot=snap, instrument=instr, use_ief=True, weight_gamma=2.0
    )

    w_diag = diag_only._diagnostic_wavelength_weights(wl, snap, instr)
    w_ief = with_ief._diagnostic_wavelength_weights(wl, snap, instr)

    # IEF off vs on differ (the feature does something) ...
    assert not np.allclose(w_diag, w_ief)

    # ... and IEF-off is exactly the legacy diag-only weight: base + total scaled
    # by distinct**gamma, normalized to mean 1, with no IEF factor applied.
    expected = _legacy_diag_only_weight(wl, snap, instr, weight_gamma=2.0)
    np.testing.assert_allclose(w_diag, expected, rtol=0.0, atol=0.0)


def _legacy_diag_only_weight(wl, snap, instr, *, weight_gamma):
    """Reference re-implementation of the pre-IEF diagnostic weight (no IEF factor).

    Mirrors ``_diagnostic_wavelength_weights`` exactly *minus* the IEF block, so a
    bit-identical match proves ``use_ief=False`` is a true no-op extension.
    """
    wl = np.asarray(wl, dtype=np.float64)
    n_wl = int(wl.shape[0])
    eps = 1e-30
    ones = np.ones(n_wl, dtype=np.float64)
    wl_min, wl_max = float(np.min(wl)), float(np.max(wl))

    line_wl = np.asarray(snap.line_wavelength_nm, dtype=np.float64)
    line_A = np.asarray(snap.line_A_ki, dtype=np.float64)
    line_g = np.asarray(snap.line_g_k, dtype=np.float64)
    line_E = np.asarray(snap.line_E_k_ev, dtype=np.float64)
    line_el = np.asarray(snap.line_element_index, dtype=np.int64)
    n_elements = len(snap.element_symbols)

    in_band = (line_wl >= wl_min) & (line_wl <= wl_max)
    valid_el = (line_el >= 0) & (line_el < n_elements)
    finite = np.isfinite(line_wl) & np.isfinite(line_A) & np.isfinite(line_g)
    mask = in_band & valid_el & finite

    lw, la, lg = line_wl[mask], line_A[mask], line_g[mask]
    le = np.where(np.isfinite(line_E[mask]), line_E[mask], 0.0)
    lel = line_el[mask]

    strength = lg * la * np.exp(-le / 1.0)
    strength = np.where(np.isfinite(strength) & (strength > 0.0), strength, 0.0)

    if instr.resolving_power is not None and instr.resolving_power > 0:
        fwhm = lw / float(instr.resolving_power)
    else:
        fwhm = np.full_like(lw, float(instr.resolution_fwhm_nm))
    sigma = fwhm / 2.3548
    sigma = np.where(np.isfinite(sigma) & (sigma > 0.0), sigma, eps)
    dwl = float(wl_max - wl_min) / max(n_wl - 1, 1)
    sigma = np.maximum(sigma, dwl)

    profiles = np.zeros((n_elements, n_wl), dtype=np.float64)
    for j in range(lw.shape[0]):
        s = strength[j]
        if s <= 0.0:
            continue
        sig = sigma[j]
        lo = np.searchsorted(wl, lw[j] - 3.0 * sig, side="left")
        hi = np.searchsorted(wl, lw[j] + 3.0 * sig, side="right")
        if hi <= lo:
            continue
        g = np.exp(-0.5 * ((wl[lo:hi] - lw[j]) / sig) ** 2)
        np.add.at(profiles[lel[j]], np.arange(lo, hi), s * g)

    total = profiles.sum(axis=0)
    dom = profiles.max(axis=0)
    distinct = dom / np.maximum(total, eps)
    if float(np.max(total)) <= 0.0:
        return ones
    base = 0.05 * float(np.max(total))
    w = (base + total) * (distinct**weight_gamma)
    w = np.where(np.isfinite(w) & (w > 0.0), w, eps)
    mean_w = float(np.mean(w))
    return w / mean_w if mean_w > 0.0 else ones


def test_bic_gate_detected_set_is_subset(setup):
    """``require_bic=True`` with a large ``bic_margin`` yields a detected set that
    is a SUBSET of the ``require_bic=False`` detected set — the BIC presence gate
    only removes false positives, never adds detections (precision lever)."""
    snap, wl, instr, meas = setup

    common = dict(
        snapshot=snap,
        instrument=instr,
        n_configs=256,
        presence_threshold=0.02,
        seed=7,
    )
    no_gate = ForwardFitIdentifier(CANDIDATES, require_bic=False, **common)
    gated = ForwardFitIdentifier(CANDIDATES, require_bic=True, bic_margin=1e9, **common)

    res_no_gate = no_gate.identify(wl, meas)
    res_gated = gated.identify(wl, meas)

    det_no_gate = {e.element for e in res_no_gate.detected_elements}
    det_gated = {e.element for e in res_gated.detected_elements}

    # SUBSET: the gated detections were all already detected without the gate.
    assert det_gated <= det_no_gate, f"gate added a detection: {det_gated - det_no_gate}"
    # The huge margin removes everything (no element earns +1e9 BIC improvement).
    assert det_gated == set()
    # The knobs round-trip into result.parameters.
    assert res_gated.parameters["require_bic"] is True
    assert res_gated.parameters["bic_margin"] == 1e9
    assert res_no_gate.parameters["require_bic"] is False
