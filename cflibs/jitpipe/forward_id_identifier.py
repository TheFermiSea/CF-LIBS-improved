"""Host adapter exposing the J10 forward-fitting identifier to the scoreboard.

The jit core (:mod:`cflibs.jitpipe.forward_id`) is a fixed-shape, vmap-clean
population forward-fitter that returns a :class:`~cflibs.jitpipe.forward_id.ForwardFitResult`
indexed by the snapshot element superset. The benchmark harness
(``cflibs/benchmark/unified.py``) and every production identifier instead speak
the duck-typed
:class:`~cflibs.inversion.identify._protocol.IdentifierProtocol`:

    ``identify(wavelength, intensity) -> ElementIdentificationResult``

This module is the thin **host-only** bridge between the two. It does *not*
touch the jit algorithm (J10 spec / ADR-0004 §4 row 11: forward physics has a
single source of truth) — it only:

1. builds the :class:`~cflibs.jitpipe.snapshot.PipelineSnapshot` once (over the
   candidate element list) and an :class:`~cflibs.instrument.model.InstrumentModel`,
2. calls :func:`cflibs.jitpipe.forward_id.forward_fit_identify`,
3. maps the fixed-shape ``ForwardFitResult`` onto the
   :class:`~cflibs.inversion.common.element_id.ElementIdentificationResult`
   dataclass the scoreboard consumes.

Score mapping (open-question #2 in the J10 scope)
-------------------------------------------------
``ForwardFitResult.presence_score`` is the **include-minus-exclude correlation
gap**: for each element, the best correlation achieved by configs that include
it minus the best achieved by configs that exclude it. It is *unbounded* — an
element that every viable config needs (never excluded) scores ``+inf``; an
element no config ever included scores ``-inf``. The scoreboard wants a
comparable ``score``/``confidence`` in ``[0, 1]``.

We map the raw gaps to ``[0, 1]`` by a **min-max normalization over the finite,
valid-element scores** of this spectrum:

    score(e) = (gap(e) - g_min) / (g_max - g_min)

where ``g_min``/``g_max`` are taken over the finite gaps of the *valid* (real,
non-padded) elements. Non-finite gaps are clamped first: ``+inf`` -> the finite
max (strongest possible evidence), ``-inf`` -> the finite min (weakest). Edge
cases: a single valid element, or all gaps equal, yields ``score = 1.0`` for any
*detected* element (``element_present > 0.5``) and ``0.0`` otherwise, so a clear
detection is never reported as zero-confidence. This is a per-spectrum relative
score (like the correlation/comb identifiers'), **not** an absolute probability;
it is monotone in the underlying coherence evidence, which is what the
precision/recall scoreboard needs. The unbounded raw gap and the best BIC are
carried verbatim in each element's ``metadata`` for downstream inspection.

``detected`` is taken directly from ``ForwardFitResult.element_present`` (the
``gap > presence_threshold`` call inside the jit core), so the detection set is
the physics decision, not a re-thresholding of the normalized score.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from cflibs.inversion.common.element_id import (
    ElementIdentification,
    ElementIdentificationResult,
)
from cflibs.jitpipe.forward_id import forward_fit_identify

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.instrument.model import InstrumentModel
    from cflibs.jitpipe.snapshot import PipelineSnapshot


ALGORITHM_NAME = "forward_fit"


def _minmax_scores(
    presence_score: np.ndarray,
    element_valid: np.ndarray,
    detected: np.ndarray,
) -> np.ndarray:
    """Map unbounded include-minus-exclude gaps to ``[0, 1]`` per spectrum.

    See the module docstring for the rationale. ``presence_score`` may carry
    ``+/-inf`` sentinels (never-excluded / never-included elements) and ``nan``
    for padded entries; only the *valid* finite scores set the normalization
    range.

    Parameters
    ----------
    presence_score : ndarray, shape (E,)
        Raw include-minus-exclude correlation gaps from the jit core.
    element_valid : ndarray of bool, shape (E,)
        ``True`` for real (non-padded) snapshot elements.
    detected : ndarray of bool, shape (E,)
        ``element_present > 0.5`` per element.

    Returns
    -------
    scores : ndarray of float, shape (E,)
        Normalized scores in ``[0, 1]``; padded entries are ``0.0``.
    """
    gaps = np.asarray(presence_score, dtype=float)
    valid = np.asarray(element_valid, dtype=bool)
    det = np.asarray(detected, dtype=bool)
    scores = np.zeros_like(gaps)

    finite_valid = valid & np.isfinite(gaps)
    if not np.any(finite_valid):
        # No finite evidence at all: fall back to the discrete detection call.
        scores[valid] = np.where(det[valid], 1.0, 0.0)
        return scores

    g_min = float(np.min(gaps[finite_valid]))
    g_max = float(np.max(gaps[finite_valid]))

    # Clamp the +/-inf sentinels onto the finite range before normalizing.
    clamped = np.clip(gaps, g_min, g_max)

    span = g_max - g_min
    if span <= 0.0:
        # Degenerate (single valid element or all gaps equal): a detection is
        # full-confidence, a non-detection is zero — never report a clear
        # detection as zero confidence.
        scores[valid] = np.where(det[valid], 1.0, 0.0)
        return scores

    norm = (clamped - g_min) / span
    scores[valid] = norm[valid]
    return scores


class ForwardFitIdentifier:
    """``IdentifierProtocol``-conformant adapter over the J10 forward-fitter.

    Builds the :class:`PipelineSnapshot` over a fixed candidate element list and
    an :class:`InstrumentModel`, then routes ``identify`` through
    :func:`cflibs.jitpipe.forward_id.forward_fit_identify` and maps the result
    onto :class:`ElementIdentificationResult`.

    Parameters
    ----------
    candidate_elements : sequence of str
        Element superset to search. Defines the snapshot's element axis and the
        only elements that can ever be called present.
    db_path : str or os.PathLike, optional
        SQLite atomic database. Used to build the snapshot when ``snapshot`` is
        not supplied. One of ``db_path`` or ``snapshot`` is required.
    snapshot : PipelineSnapshot, optional
        Pre-built snapshot over (at least) ``candidate_elements``. When given,
        the DB is not opened; the snapshot's ``element_symbols`` define the
        element axis (``candidate_elements`` is then ignored for axis ordering
        but still recorded in ``parameters``).
    instrument : InstrumentModel, optional
        Instrument model. Defaults to a resolving-power model at
        ``resolving_power`` when set, else a fixed-FWHM model at
        ``resolution_fwhm_nm``.
    resolving_power : float, optional
        Resolving power for the default instrument (``R = lambda / FWHM``).
    resolution_fwhm_nm : float
        Fixed instrument FWHM (nm) used when ``resolving_power`` is not set.
    wavelength_range : tuple of float, optional
        ``(min_nm, max_nm)`` for the snapshot scan when building from ``db_path``.
        Defaults to the measured spectrum's span at ``identify`` time when not
        given (the snapshot is then built lazily on first ``identify`` call).
    n_configs : int
        Forward-fit population size (``B_eval``).
    presence_threshold : float
        Include-minus-exclude correlation gap to call an element present.
    polish_steps : int
        Fixed Levenberg-Marquardt polish iterations (``0`` disables polishing).
    seed : int
        RNG seed for the (counter-based, deterministic) candidate population.
    """

    def __init__(
        self,
        candidate_elements: Sequence[str],
        *,
        db_path: Optional[str | os.PathLike] = None,
        snapshot: Optional["PipelineSnapshot"] = None,
        instrument: Optional["InstrumentModel"] = None,
        resolving_power: Optional[float] = None,
        resolution_fwhm_nm: float = 0.1,
        wavelength_range: Optional[tuple[float, float]] = None,
        n_configs: int = 1024,
        presence_threshold: float = 0.05,
        polish_steps: int = 0,
        seed: int = 0,
    ) -> None:
        if snapshot is None and db_path is None:
            raise ValueError("ForwardFitIdentifier requires either `snapshot` or `db_path`.")

        self.candidate_elements = list(candidate_elements)
        self.db_path = os.fspath(db_path) if db_path is not None else None
        self._snapshot = snapshot
        self._instrument = instrument
        self.resolving_power = resolving_power
        self.resolution_fwhm_nm = float(resolution_fwhm_nm)
        self.wavelength_range = wavelength_range
        self.n_configs = int(n_configs)
        self.presence_threshold = float(presence_threshold)
        self.polish_steps = int(polish_steps)
        self.seed = int(seed)

    # ------------------------------------------------------------------ helpers
    def _build_instrument(self) -> "InstrumentModel":
        from cflibs.instrument.model import InstrumentModel

        if self._instrument is not None:
            return self._instrument
        if self.resolving_power is not None and self.resolving_power > 0:
            inst = InstrumentModel.from_resolving_power(float(self.resolving_power))
        else:
            inst = InstrumentModel(resolution_fwhm_nm=self.resolution_fwhm_nm)
        self._instrument = inst
        return inst

    def _build_snapshot(self, wavelength_nm: np.ndarray) -> "PipelineSnapshot":
        if self._snapshot is not None:
            return self._snapshot

        from cflibs.atomic.database import AtomicDatabase
        from cflibs.jitpipe.snapshot import PipelineSnapshot

        wl_range = self.wavelength_range
        if wl_range is None:
            wl = np.asarray(wavelength_nm, dtype=float)
            wl_range = (float(np.min(wl)), float(np.max(wl)))

        with AtomicDatabase(self.db_path) as db:
            asnap = db.snapshot(
                elements=self.candidate_elements,
                wavelength_range=wl_range,
                include_levels=True,
            )
        self._snapshot = PipelineSnapshot.from_atomic_snapshot(asnap)
        return self._snapshot

    # ------------------------------------------------------------------ protocol
    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> ElementIdentificationResult:
        """Identify elements via population forward-fitting.

        Parameters
        ----------
        wavelength : ndarray, shape (N_wl,)
            Wavelengths in nm.
        intensity : ndarray, shape (N_wl,)
            Measured intensities (arbitrary units), same length as ``wavelength``.

        Returns
        -------
        ElementIdentificationResult
            ``detected_elements`` are those with ``element_present > 0.5`` from
            the jit core; ``score``/``confidence`` are the min-max-normalized
            include-minus-exclude gaps (see module docstring). The raw gap and
            best BIC per element live in ``metadata``.
        """
        import jax

        wl = np.asarray(wavelength, dtype=float)
        meas = np.asarray(intensity, dtype=float)

        snapshot = self._build_snapshot(wl)
        instrument = self._build_instrument()
        element_symbols = tuple(snapshot.element_symbols)

        result = forward_fit_identify(
            meas,
            wl,
            snapshot,
            instrument,
            key=jax.random.PRNGKey(self.seed),
            n_configs=self.n_configs,
            presence_threshold=self.presence_threshold,
            polish_steps=self.polish_steps,
        )

        present = np.asarray(result.element_present, dtype=float)
        presence_score = np.asarray(result.presence_score, dtype=float)
        best_bic = np.asarray(result.best_bic, dtype=float)
        detected_mask = present > 0.5
        # A snapshot built over a candidate list (``db.snapshot(elements=...)``)
        # or supplied directly carries only real elements — there is no padding
        # axis here, so every snapshot element is valid. (``forward_fit_identify``
        # builds ``element_valid`` as all-ones over ``snapshot.element_symbols``;
        # the +/-inf gap sentinels for never-included / never-excluded elements
        # are handled by the finite-range clamp inside ``_minmax_scores``.)
        element_valid = np.ones(len(element_symbols), dtype=bool)

        scores = _minmax_scores(presence_score, element_valid, detected_mask)

        all_elements: list[ElementIdentification] = []
        for ei, el in enumerate(element_symbols):
            raw_gap = float(presence_score[ei])
            score = float(np.clip(scores[ei], 0.0, 1.0))
            detected = bool(detected_mask[ei])
            ident = ElementIdentification(
                element=el,
                detected=detected,
                score=score,
                confidence=score,
                n_matched_lines=0,
                n_total_lines=0,
                matched_lines=[],
                unmatched_lines=[],
                metadata={
                    "presence_score": raw_gap,
                    "best_bic": float(best_bic[ei]),
                },
            )
            all_elements.append(ident)

        detected_elements = [e for e in all_elements if e.detected]
        rejected_elements = [e for e in all_elements if not e.detected]

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_elements,
            experimental_peaks=[],
            n_peaks=0,
            n_matched_peaks=0,
            n_unmatched_peaks=0,
            algorithm=ALGORITHM_NAME,
            parameters={
                "candidate_elements": list(self.candidate_elements),
                "n_configs": self.n_configs,
                "presence_threshold": self.presence_threshold,
                "polish_steps": self.polish_steps,
                "best_correlation": float(np.asarray(result.best_correlation)),
                "best_config_index": int(np.asarray(result.best_config_index)),
                "n_valid_configs": int(np.asarray(result.n_valid_configs)),
                "seed": self.seed,
            },
        )


__all__ = ["ForwardFitIdentifier", "ALGORITHM_NAME"]
