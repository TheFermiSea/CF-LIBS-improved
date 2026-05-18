"""Conformance tests for :class:`IdentifierProtocol`.

These tests pin down the *external* contract of every identifier in
``cflibs.inversion.identify``: each must expose an ``identify`` method
whose first two positional parameters are ``wavelength`` and
``intensity`` and whose return type is
:class:`~cflibs.inversion.common.element_id.ElementIdentificationResult`.

The protocol is purely structural (``@runtime_checkable``); these tests
verify both runtime ``isinstance`` conformance and the precise
positional signature via ``inspect``.

We intentionally do *not* exercise ``identify()`` end-to-end here -- that
is the job of the per-identifier integration tests. The point of this
file is to catch silent API drift: if someone renames the first
positional arg from ``wavelength`` to ``wl`` in (say) ``ALIASIdentifier``
without touching the others, this test fails immediately.
"""

from __future__ import annotations

import inspect
from typing import Sequence, Set

import numpy as np
import pytest

from cflibs.inversion.common.element_id import (
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.inversion.identify._protocol import IdentifierProtocol
from cflibs.inversion.identify.alias import ALIASIdentifier
from cflibs.inversion.identify.comb import CombIdentifier
from cflibs.inversion.identify.correlation import CorrelationIdentifier
from cflibs.inversion.identify.hybrid_consensus import HybridConsensusIdentifier
from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Cheap stubs so we can construct each identifier without loading a real
# atomic database / basis library. ``IdentifierProtocol`` is structural;
# attribute presence is all that ``isinstance`` actually checks. We do
# need real instances though, because Protocol structural checks apply
# to instances, not classes.
# ---------------------------------------------------------------------------


class _DummyAtomicDB:
    """Minimal stand-in for :class:`AtomicDatabase` -- attribute-free.

    The four identifiers that take an ``atomic_db`` argument (ALIAS,
    Comb, Correlation, and indirectly the hybrid wrappers) don't dispatch
    on it during construction, so a bare object suffices.
    """


class _DummyBasisLibrary:
    """Minimal stand-in for :class:`BasisLibrary` used by SpectralNNLS.

    ``SpectralNNLSIdentifier.__init__`` only reads
    ``basis_library.config.ionization_stages`` when present; absence is
    handled via ``hasattr``.
    """


class _StubIdentifier:
    """Hand-rolled identifier that emits a fixed ``ElementIdentificationResult``.

    Used as a building block for ``HybridConsensusIdentifier`` so we can
    assemble a hybrid without instantiating three real backends.
    """

    def __init__(self, detected: Sequence[str] = ()):
        self._detected: Set[str] = set(detected)

    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> ElementIdentificationResult:
        detected_elements = [
            ElementIdentification(
                element=el,
                detected=True,
                score=1.0,
                confidence=1.0,
                n_matched_lines=0,
                n_total_lines=0,
                matched_lines=[],
                unmatched_lines=[],
                metadata={},
            )
            for el in sorted(self._detected)
        ]
        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=[],
            all_elements=detected_elements,
            experimental_peaks=[],
            n_peaks=0,
            n_matched_peaks=0,
            n_unmatched_peaks=0,
            algorithm="stub",
        )


# ---------------------------------------------------------------------------
# Fixtures: one instance per identifier under test.
# ---------------------------------------------------------------------------


@pytest.fixture
def alias_identifier() -> ALIASIdentifier:
    return ALIASIdentifier(_DummyAtomicDB())


@pytest.fixture
def comb_identifier() -> CombIdentifier:
    return CombIdentifier(atomic_db=_DummyAtomicDB())


@pytest.fixture
def correlation_identifier() -> CorrelationIdentifier:
    return CorrelationIdentifier(atomic_db=_DummyAtomicDB())


@pytest.fixture
def spectral_nnls_identifier() -> SpectralNNLSIdentifier:
    return SpectralNNLSIdentifier(basis_library=_DummyBasisLibrary())


@pytest.fixture
def hybrid_consensus_identifier() -> HybridConsensusIdentifier:
    stubs = [_StubIdentifier(detected={"Fe"}), _StubIdentifier(detected={"Fe"})]
    return HybridConsensusIdentifier(stubs, elements=["Fe"], min_agreeing=2)


# ---------------------------------------------------------------------------
# Tests.
# ---------------------------------------------------------------------------


def test_alias_satisfies_protocol(alias_identifier):
    assert isinstance(alias_identifier, IdentifierProtocol)


def test_comb_satisfies_protocol(comb_identifier):
    assert isinstance(comb_identifier, IdentifierProtocol)


def test_correlation_satisfies_protocol(correlation_identifier):
    assert isinstance(correlation_identifier, IdentifierProtocol)


def test_spectral_nnls_satisfies_protocol(spectral_nnls_identifier):
    assert isinstance(spectral_nnls_identifier, IdentifierProtocol)


def test_hybrid_consensus_satisfies_protocol(hybrid_consensus_identifier):
    assert isinstance(hybrid_consensus_identifier, IdentifierProtocol)


@pytest.mark.parametrize(
    "identifier_cls",
    [
        ALIASIdentifier,
        CombIdentifier,
        CorrelationIdentifier,
        SpectralNNLSIdentifier,
        HybridConsensusIdentifier,
    ],
    ids=["alias", "comb", "correlation", "spectral_nnls", "hybrid_consensus"],
)
def test_identify_first_two_positional_params_are_wavelength_intensity(identifier_cls):
    """Every identifier's ``identify`` must accept ``(self, wavelength, intensity, ...)``.

    The protocol is structural (so ``isinstance`` only sees attribute
    presence) -- this test backs it up by inspecting the actual
    signature, which is what callers like the unified benchmark
    predictor builders rely on.
    """
    sig = inspect.signature(identifier_cls.identify)
    params = list(sig.parameters.values())
    # params[0] is ``self``.
    assert len(params) >= 3, (
        f"{identifier_cls.__name__}.identify must take at least "
        f"(self, wavelength, intensity); got {params}"
    )
    assert params[1].name == "wavelength", (
        f"{identifier_cls.__name__}.identify first non-self param must be "
        f"'wavelength', got {params[1].name!r}"
    )
    assert params[2].name == "intensity", (
        f"{identifier_cls.__name__}.identify second non-self param must be "
        f"'intensity', got {params[2].name!r}"
    )


def test_protocol_rejects_non_identifiers():
    """``IdentifierProtocol`` should NOT match arbitrary objects.

    Guards against the protocol degenerating into ``object`` if someone
    later drops the ``identify`` requirement.
    """

    class _NotAnIdentifier:
        def classify(self, wavelength, intensity):  # wrong name on purpose
            return None

    assert not isinstance(_NotAnIdentifier(), IdentifierProtocol)
    assert not isinstance(object(), IdentifierProtocol)
