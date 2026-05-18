"""Structural protocol for element identifiers.

This module defines :class:`IdentifierProtocol`, the minimal duck-typed
contract that every concrete identifier in ``cflibs.inversion.identify``
already satisfies in practice. The protocol exists so that

* consumers (e.g. ``cflibs/benchmark/unified.py`` predictor builders) can
  type-annotate their inputs against a single explicit shape rather than
  carrying ``Union[ALIASIdentifier, CombIdentifier, ...]`` everywhere;
* downstream callers can ``isinstance(x, IdentifierProtocol)`` at
  runtime to discover whether an arbitrary object can be plugged into
  the identification pipeline without importing every concrete class;
* future identifier additions have a single, machine-checkable
  conformance check, so a new "alias-v3" or "ensemble" identifier can be
  added without anyone having to manually audit five constructor
  signatures.

Design notes
------------
1. The protocol is **structural** (``@runtime_checkable``). No identifier
   needs to subclass it -- conformance is detected by attribute presence
   at ``isinstance`` time. This keeps blast radius minimal: we do *not*
   touch the internals of the five existing identifiers.
2. The protocol intentionally captures only the **common shape** of
   ``identify`` across the existing implementations. As of 2026-05 every
   concrete identifier in this package accepts two positional
   ``np.ndarray`` arguments -- ``wavelength`` and ``intensity`` (both in
   nanometres / arbitrary intensity units respectively) -- and returns
   an :class:`~cflibs.inversion.common.element_id.ElementIdentificationResult`.
   Identifier-specific kwargs (``spectrum_id`` on ALIAS / Comb /
   Correlation / Hybrid; ``mode`` on Correlation) are NOT part of the
   protocol; callers that need them must dispatch on the concrete class.
3. ``runtime_checkable`` Protocols only verify *attribute presence*, not
   the exact signature of the bound method. The accompanying test
   (``tests/inversion/test_identifier_protocol.py``) backs this up with
   an actual ``inspect.signature`` shape check for each concrete class.

If you are adding a sixth identifier, follow these rules:

* Provide ``identify(self, wavelength, intensity) ->
  ElementIdentificationResult`` (extra ``*args``/``**kwargs`` after the
  two required positional args are fine).
* Return an ``ElementIdentificationResult`` (or a subclass) -- the
  benchmark layer assumes that exact dataclass.
* Re-run ``tests/inversion/test_identifier_protocol.py`` to confirm the
  new class is picked up by the conformance scan.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from cflibs.inversion.common.element_id import ElementIdentificationResult


@runtime_checkable
class IdentifierProtocol(Protocol):
    """Structural contract for element identifiers.

    Every implementation in ``cflibs.inversion.identify`` (ALIAS, Comb,
    Correlation, SpectralNNLS, HybridConsensus -- and the
    legacy two-stage ``HybridIdentifier`` wrapper) satisfies this
    protocol as of 2026-05.

    Implementations MAY accept additional keyword arguments after
    ``intensity`` (for example ``spectrum_id`` for coverage telemetry or
    ``mode`` for the correlation identifier). Those identifier-specific
    knobs are intentionally NOT part of the protocol; callers that need
    them must dispatch on the concrete class.
    """

    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> ElementIdentificationResult:
        """Identify elements present in a 1-D LIBS spectrum.

        Parameters
        ----------
        wavelength
            1-D ``np.ndarray`` of wavelengths in nanometres.
        intensity
            1-D ``np.ndarray`` of intensities (arbitrary units), the
            same length as ``wavelength``.

        Returns
        -------
        ElementIdentificationResult
            Detected + rejected elements with per-line metadata. The
            ``algorithm`` field on the result names the concrete
            identifier (``"alias"``, ``"comb"``, ``"correlation"``,
            ``"spectral_nnls"``, ``"hybrid_nnls_alias"``,
            ``"hybrid_consensus"``).
        """
        ...


__all__ = ["IdentifierProtocol"]
