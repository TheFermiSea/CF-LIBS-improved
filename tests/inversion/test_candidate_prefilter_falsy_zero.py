"""Regression tests for the candidate prefilter falsy-zero fallback bug.

Audit Family P: in ``select_candidate_elements`` the multi-T base estimate was
read via ``getattr(identifier, "_estimated_T", None) or identifier.fallback_T_K``.
A *valid* numeric estimate of exactly 0.0 is falsy, so ``or fallback`` silently
discarded it and substituted the default. These tests pin the corrected
``is not None`` behavior with an independent oracle.

Oracle (does not reuse the production base_T computation):
  The multi-T loop deep-copies the identifier and sets
  ``id_copy.fallback_T_K = max(base_T + offset, 3000.0)`` and
  ``id_copy.fallback_ne_cm3 = base_ne`` before re-running ``identify``. By
  recording the ``(fallback_T_K, fallback_ne_cm3)`` each copy is queried with,
  we can reconstruct which ``base_T`` / ``base_ne`` the prefilter actually used,
  independently of the (previously buggy) expression under test.
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from cflibs.inversion.candidate_prefilter import select_candidate_elements


@dataclass
class _FakeEID:
    element: str
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass
class _FakeResult:
    all_elements: List[_FakeEID]


class _RecordingIdentifier:
    """Fake identifier that records every (T, ne) it is queried at.

    ``_estimated_T`` / ``_estimated_ne`` mimic the values an NNLS identifier
    caches after its base ``identify`` call. The base call appends its own
    ``fallback_T_K`` so the recorded list also reflects pre-copy state.
    """

    def __init__(
        self,
        estimated_T: Optional[float],
        estimated_ne: Optional[float],
        fallback_T_K: float = 8000.0,
        fallback_ne_cm3: float = 1e17,
    ):
        self._estimated_T = estimated_T
        self._estimated_ne = estimated_ne
        self.fallback_T_K = fallback_T_K
        self.fallback_ne_cm3 = fallback_ne_cm3
        self.basis_index = object()
        # Shared across deepcopies so offset-copy queries are visible here.
        self.queries: List[Tuple[float, float]] = []

    def __deepcopy__(self, memo):
        # The prefilter deepcopies the identifier for each multi-T offset and
        # mutates the copy's fallback_T_K / fallback_ne_cm3. Deepcopy every
        # attribute EXCEPT `queries`, which we deliberately share so the
        # offset-copy queries land in the original's record (our oracle).
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            if key == "queries":
                new.queries = self.queries  # shared reference
            else:
                setattr(new, key, copy.deepcopy(value, memo))
        return new

    def identify(self, wavelength, intensity):
        self.queries.append((self.fallback_T_K, self.fallback_ne_cm3))
        return _FakeResult(
            all_elements=[
                _FakeEID(
                    element="Fe I",
                    metadata={"nnls_snr": 10.0, "nnls_coefficient": 5.0},
                )
            ]
        )


def _wl_int():
    return np.linspace(200.0, 400.0, 64), np.ones(64)


def _offset_query_temps(ident: _RecordingIdentifier) -> List[float]:
    """Recorded fallback_T_K from the multi-T offset copies (skip base call)."""
    return [T for (T, _ne) in ident.queries[1:]]


def _offset_query_nes(ident: _RecordingIdentifier) -> List[float]:
    return [ne for (_T, ne) in ident.queries[1:]]


class TestFalsyZeroEstimate:
    """A valid estimate must be honored, not replaced by the fallback."""

    def test_nonzero_estimate_still_used(self):
        ident = _RecordingIdentifier(estimated_T=6000.0, estimated_ne=2e16, fallback_T_K=8000.0)
        wl, intens = _wl_int()
        select_candidate_elements(ident, wl, intens, multi_t_offsets=[-1500.0, 1500.0])
        offset_temps = _offset_query_temps(ident)
        # base_T == 6000.0 -> max(4500, 3000)=4500 and max(7500, 3000)=7500.
        assert sorted(offset_temps) == [4500.0, 7500.0]
        assert _offset_query_nes(ident) == [2e16, 2e16]


class TestNoneEstimateRoutesToFallback:
    """None estimates must still route to the fallback defaults."""

    def test_none_estimated_T_uses_fallback(self):
        ident = _RecordingIdentifier(estimated_T=None, estimated_ne=None, fallback_T_K=8000.0)
        wl, intens = _wl_int()
        select_candidate_elements(ident, wl, intens, multi_t_offsets=[-1500.0, 1500.0])
        offset_temps = _offset_query_temps(ident)
        # base_T == fallback 8000.0 -> max(6500,3000)=6500 and max(9500,3000)=9500.
        assert sorted(offset_temps) == [6500.0, 9500.0]

    def test_none_estimated_ne_uses_fallback(self):
        ident = _RecordingIdentifier(estimated_T=None, estimated_ne=None, fallback_ne_cm3=1e17)
        wl, intens = _wl_int()
        select_candidate_elements(ident, wl, intens, multi_t_offsets=[-1500.0, 1500.0])
        assert _offset_query_nes(ident) == [1e17, 1e17]

    def test_missing_attribute_uses_fallback(self):
        # An identifier without the cached estimate attributes at all.
        class _NoEstimateIdentifier(_RecordingIdentifier):
            def __init__(self):
                super().__init__(estimated_T=None, estimated_ne=None)
                del self._estimated_T
                del self._estimated_ne

        ident = _NoEstimateIdentifier()
        wl, intens = _wl_int()
        select_candidate_elements(ident, wl, intens, multi_t_offsets=[-1500.0, 1500.0])
        offset_temps = _offset_query_temps(ident)
        assert sorted(offset_temps) == [6500.0, 9500.0]
        assert _offset_query_nes(ident) == [1e17, 1e17]
