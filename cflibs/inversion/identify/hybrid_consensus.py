"""
Majority-vote consensus identifier across alias/comb/correlation line-matchers.

Where :class:`HybridIdentifier` (NNLS + ALIAS, two-stage) targets a *single*
two-stage pipeline, :class:`HybridConsensusIdentifier` is a *combiner* over
the three independent line-matching identifiers (ALIAS, Comb, Correlation).

Each contributing identifier votes "detected" or "not detected" per element.
An element is reported as detected if at least ``min_agreeing`` identifiers
agree.

Why
---
The default :func:`hybrid_union` workflow used in the benchmark leaderboard
takes the *union* of per-identifier detections, which maximizes recall at
the cost of precision (one over-eager identifier can flip an element to
"detected"). The 2-of-3 confirmation rule reverses that trade-off: an
element must be confirmed by a *majority* of the line-matchers, which
suppresses false positives at the cost of some recall.

This class is opt-in. Nothing in the codebase silently switches to consensus
mode — callers must explicitly construct ``HybridConsensusIdentifier`` (or
register the ``hybrid_consensus_2of3`` workflow alongside the existing
``hybrid_union`` / ``hybrid_intersect``).

Spectral NNLS is intentionally excluded from the vote. It is a
regression-based identifier with a fundamentally different detection-noise
profile than the three line-matchers; mixing it into a majority vote
distorts the semantics. Callers who want NNLS in the loop should use
:class:`HybridIdentifier` (NNLS+ALIAS) instead.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from cflibs.inversion.element_id import (
    ElementIdentification,
    ElementIdentificationResult,
)

_DEFAULT_NAMES: Tuple[str, str, str] = ("alias", "comb", "correlation")


class HybridConsensusIdentifier:
    """
    Combine 2-or-more line-matching identifiers via majority voting.

    The identifier accepts already-constructed instances of ALIAS, Comb, and
    Correlation identifiers (or any duck-typed object with a compatible
    ``identify(wavelength, intensity) -> ElementIdentificationResult``
    method). It runs each one, then applies an N-of-M vote.

    Parameters
    ----------
    identifiers : sequence of identifier instances
        Typically 3 identifiers (ALIAS, Comb, Correlation), but the
        consensus rule generalizes: pass any number ``M >= min_agreeing``.
    elements : list of str
        Element symbols expected to appear in the per-identifier results.
        Used to enumerate the final all_elements list.
    min_agreeing : int, default 2
        Minimum number of identifiers that must agree (``detected == True``)
        for an element to be reported as detected. Default is 2, giving
        2-of-3 semantics when three identifiers are passed.
    names : sequence of str, optional
        Human-readable names for each identifier in ``identifiers``. Used
        only for the per-element metadata (``votes_by`` mapping). When
        omitted, defaults to ``("alias", "comb", "correlation")`` for
        len(identifiers)==3, else ``("id0", "id1", ...)``.

    Notes
    -----
    *Score aggregation.* Per element, the consensus identifier reports the
    *mean* score across identifiers that voted "detected" (zero when none
    did). This is a deliberately conservative aggregation — taking the
    max would over-promote any single confident vote, defeating the
    precision-improving purpose of the consensus rule.

    *Matched lines.* The matched-line list is the union of matched lines
    across all *voting* identifiers (those that returned detected==True).
    Non-voting identifiers' lines are dropped to keep the downstream
    Boltzmann pipeline aligned with the consensus decision.

    *Default semantics preserved.* This class does NOT subclass or modify
    :class:`HybridIdentifier`. Callers using the existing union-mode
    workflow get byte-identical behavior; only callers that *explicitly*
    construct ``HybridConsensusIdentifier`` see the new vote logic.
    """

    def __init__(
        self,
        identifiers: Sequence[object],
        elements: List[str],
        *,
        min_agreeing: int = 2,
        names: Optional[Sequence[str]] = None,
    ):
        if len(identifiers) < 2:
            raise ValueError(
                f"HybridConsensusIdentifier requires at least 2 identifiers, "
                f"got {len(identifiers)}."
            )
        if min_agreeing < 1:
            raise ValueError(f"min_agreeing must be >= 1, got {min_agreeing}")
        if min_agreeing > len(identifiers):
            raise ValueError(
                f"min_agreeing={min_agreeing} cannot exceed the number of "
                f"identifiers ({len(identifiers)})"
            )

        self.identifiers = list(identifiers)
        self.elements = list(elements)
        self.min_agreeing = int(min_agreeing)

        if names is None:
            if len(identifiers) == 3:
                self.names: List[str] = list(_DEFAULT_NAMES)
            else:
                self.names = [f"id{i}" for i in range(len(identifiers))]
        else:
            if len(names) != len(identifiers):
                raise ValueError(
                    f"names length ({len(names)}) must equal identifiers "
                    f"length ({len(identifiers)})"
                )
            self.names = [str(n) for n in names]

    # ------------------------------------------------------------------ #
    # Pure combine path: no identifier execution, just merge results.    #
    # ------------------------------------------------------------------ #
    def combine(
        self,
        results: Sequence[ElementIdentificationResult],
    ) -> ElementIdentificationResult:
        """
        Combine pre-computed per-identifier results via majority vote.

        Useful when the contributing identifiers have already been run
        (e.g. inside the benchmark harness where each one is executed in
        its own ``IDWorkflowSpec``). Skips the cost of re-running them.
        """
        if len(results) != len(self.identifiers):
            raise ValueError(
                f"combine() expected {len(self.identifiers)} results, " f"got {len(results)}"
            )
        return self._aggregate(results)

    # ------------------------------------------------------------------ #
    # Execute-then-combine path.                                         #
    # ------------------------------------------------------------------ #
    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
    ) -> ElementIdentificationResult:
        """
        Run all configured identifiers and combine their outputs.

        Returns
        -------
        ElementIdentificationResult
            Element detections that pass the N-of-M consensus threshold,
            with per-element metadata recording how each identifier voted.
        """
        per_results: List[ElementIdentificationResult] = []
        for identifier in self.identifiers:
            res = identifier.identify(wavelength, intensity)
            per_results.append(res)
        return self._aggregate(per_results)

    # ------------------------------------------------------------------ #
    # Internals                                                          #
    # ------------------------------------------------------------------ #
    def _aggregate(
        self,
        per_results: Sequence[ElementIdentificationResult],
    ) -> ElementIdentificationResult:
        # Per-identifier detection sets keyed by element symbol
        per_detected: List[Set[str]] = [
            {eid.element for eid in r.detected_elements} for r in per_results
        ]

        # Per-identifier score & matched-line lookup tables, keyed by element.
        per_score: List[Dict[str, float]] = []
        per_eid: List[Dict[str, ElementIdentification]] = []
        for r in per_results:
            score_map: Dict[str, float] = {}
            eid_map: Dict[str, ElementIdentification] = {}
            for eid in r.all_elements:
                score_map[eid.element] = float(eid.score)
                eid_map[eid.element] = eid
            per_score.append(score_map)
            per_eid.append(eid_map)

        all_element_ids: List[ElementIdentification] = []
        for element in self.elements:
            votes_by: Dict[str, bool] = {}
            for name, detected_set in zip(self.names, per_detected):
                votes_by[name] = element in detected_set
            vote_count = sum(1 for v in votes_by.values() if v)
            detected = vote_count >= self.min_agreeing

            # Mean score across *voting* identifiers; 0 when none voted yes.
            voting_scores = [
                per_score[i].get(element, 0.0) for i, voted in enumerate(votes_by.values()) if voted
            ]
            score = float(np.mean(voting_scores)) if voting_scores else 0.0

            # Matched lines = union across voting identifiers; falls back to
            # the first identifier's entry when nothing voted yes (so the
            # downstream consumer still gets *some* line metadata for
            # rejected elements).
            matched_lines = []
            unmatched_lines = []
            n_matched = 0
            n_total = 0
            voting_indices = [i for i, v in enumerate(votes_by.values()) if v]
            if voting_indices:
                seen = set()
                for i in voting_indices:
                    eid = per_eid[i].get(element)
                    if eid is None:
                        continue
                    for line in eid.matched_lines:
                        key = (
                            line.element,
                            line.ionization_stage,
                            line.wavelength_th_nm,
                        )
                        if key in seen:
                            continue
                        seen.add(key)
                        matched_lines.append(line)
                    # Track the *largest* observed line counts to give the
                    # downstream consumer a representative summary.
                    n_matched = max(n_matched, eid.n_matched_lines)
                    n_total = max(n_total, eid.n_total_lines)
                # Use the first voting identifier's unmatched lines verbatim
                # — these are theoretical, not experimental, so deduping is
                # unnecessary.
                first_eid = per_eid[voting_indices[0]].get(element)
                if first_eid is not None:
                    unmatched_lines = list(first_eid.unmatched_lines)
            else:
                # No identifier voted detected. Pull metadata from id 0 if
                # available so the rejected-elements section stays
                # informative.
                eid = per_eid[0].get(element)
                if eid is not None:
                    matched_lines = list(eid.matched_lines)
                    unmatched_lines = list(eid.unmatched_lines)
                    n_matched = eid.n_matched_lines
                    n_total = eid.n_total_lines

            metadata = {
                "consensus_mode": f"{self.min_agreeing}_of_{len(self.identifiers)}",
                "votes_by": dict(votes_by),
                "vote_count": vote_count,
                "min_agreeing": self.min_agreeing,
                "scores_by": {
                    name: per_score[i].get(element, 0.0) for i, name in enumerate(self.names)
                },
            }

            all_element_ids.append(
                ElementIdentification(
                    element=element,
                    detected=detected,
                    score=score,
                    confidence=score,
                    n_matched_lines=n_matched,
                    n_total_lines=n_total,
                    matched_lines=matched_lines,
                    unmatched_lines=unmatched_lines,
                    metadata=metadata,
                )
            )

        detected_elements = [e for e in all_element_ids if e.detected]
        rejected_elements = [e for e in all_element_ids if not e.detected]

        # Aggregate peak counts from the first result (peaks are a property
        # of the input spectrum, not the per-identifier decision).
        first = per_results[0]
        n_peaks = first.n_peaks
        experimental_peaks = list(first.experimental_peaks)
        n_matched_peaks = first.n_matched_peaks
        n_unmatched_peaks = first.n_unmatched_peaks

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_element_ids,
            experimental_peaks=experimental_peaks,
            n_peaks=n_peaks,
            n_matched_peaks=n_matched_peaks,
            n_unmatched_peaks=n_unmatched_peaks,
            algorithm=f"hybrid_consensus_{self.min_agreeing}of{len(self.identifiers)}",
            parameters={
                "min_agreeing": self.min_agreeing,
                "n_identifiers": len(self.identifiers),
                "identifier_names": list(self.names),
                "n_detected": len(detected_elements),
            },
        )


def consensus_detected_elements(
    per_results: Iterable[ElementIdentificationResult],
    min_agreeing: int = 2,
) -> Set[str]:
    """
    Lightweight helper: return the set of elements that pass the N-of-M
    vote across the provided per-identifier results.

    Exposed for benchmark scripts that already have the per-identifier
    outputs in hand and only need the consensus *decision* (no merging of
    matched-line lists, no score aggregation). The full ``identify`` /
    ``combine`` workflow on :class:`HybridConsensusIdentifier` does both
    the decision and the metadata bookkeeping.
    """
    if min_agreeing < 1:
        raise ValueError(f"min_agreeing must be >= 1, got {min_agreeing}")

    counts: Dict[str, int] = {}
    n_results = 0
    for result in per_results:
        n_results += 1
        for eid in result.detected_elements:
            counts[eid.element] = counts.get(eid.element, 0) + 1

    if min_agreeing > n_results:
        raise ValueError(
            f"min_agreeing={min_agreeing} cannot exceed the number of " f"results ({n_results})"
        )

    return {element for element, count in counts.items() if count >= min_agreeing}
