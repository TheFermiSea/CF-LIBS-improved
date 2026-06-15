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
