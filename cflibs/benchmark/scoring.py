"""The one don't-care-aware confusion rule, shared across scoring paths.

Confusion (TP/FP/FN/TN) is computed in several places — the synthetic-corpus
benchmark (:mod:`cflibs.benchmark.synthetic_eval`) and the observability
per-element aggregator (:mod:`cflibs.observability.element_confusion`). This
module is the single home for the *rule* that maps ``(truth, predicted,
don't-care band)`` to a per-element label, so those callers cannot drift on the
semantics — in particular the **don't-care band**: real-but-sub-detection-floor
traces that are neither rewarded (TP) nor penalised (FP), and never an FN.

Callers differ in *scope*, not rule. The synthetic-corpus path iterates a fixed
candidate panel (so it has a TN cell); set-based callers iterate the elements
that appear (``predicted | truth``) and have no TN. Both classify each element
through :func:`classify_element`.

:class:`ScoringRow` wraps the three element sets a benchmark row carries and
offers panel confusion + per-element labelling on the synthetic_eval compute
path. The benchmark row dict stays the wire format (``per_spectrum.jsonl`` /
Parquet); build a :class:`ScoringRow` from one with :meth:`ScoringRow.from_row`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# Per-element confusion as a (tp, fp, fn, tn) tuple.
Counts = Tuple[int, int, int, int]

TP = "tp"
FP = "fp"
FN = "fn"
TN = "tn"


def classify_element(
    element: str,
    truth: Set[str] | FrozenSet[str],
    predicted: Set[str] | FrozenSet[str],
    ignore: Set[str] | FrozenSet[str] = frozenset(),
) -> Optional[str]:
    """Classify one element as ``"tp"``/``"fp"``/``"fn"``/``"tn"``, or ``None``.

    ``None`` means the element is in the **don't-care band** (``element in
    ignore``) — a real-but-sub-detection-floor trace, neither rewarded nor
    penalised, and never counted. Otherwise the standard truth x predicted
    matrix applies.
    """
    if element in ignore:
        return None
    t = element in truth
    p = element in predicted
    if t and p:
        return TP
    if p:
        return FP
    if t:
        return FN
    return TN


def confusion_counts(
    true_elements: Iterable[str],
    predicted_elements: Iterable[str],
    candidate_elements: Sequence[str],
    ignore_elements: Iterable[str] = (),
) -> Dict[str, int]:
    """Return ``{"tp","fp","fn","tn"}`` over ``candidate_elements`` (the panel).

    Elements in the don't-care band (``ignore_elements``) are skipped entirely —
    neither rewarded nor penalised. Returns a dict so callers can splat it into a
    row record with ``**confusion_counts(...)``.
    """
    truth = set(true_elements)
    predicted = set(predicted_elements)
    ignore = set(ignore_elements)
    tp = fp = fn = tn = 0
    for element in candidate_elements:
        label = classify_element(element, truth, predicted, ignore)
        if label == TP:
            tp += 1
        elif label == FP:
            fp += 1
        elif label == FN:
            fn += 1
        elif label == TN:
            tn += 1
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def per_element_tally(
    rows: Iterable[Mapping[str, Any]],
    candidate_elements: Sequence[str],
) -> Dict[str, Counts]:
    """Tally ``{element: (tp, fp, fn, tn)}`` for every candidate in one pass.

    Each row's truth/predicted/ignore sets are built once (not once per element),
    so the cost is O(rows) set builds rather than O(rows x elements). Rows are
    plain benchmark dicts carrying ``true_elements`` / ``predicted_elements`` /
    (optionally) ``ignore_elements`` list values. Rows whose don't-care band
    contains an element are skipped for that element.
    """
    counts: Dict[str, List[int]] = {el: [0, 0, 0, 0] for el in candidate_elements}
    for row in rows:
        truth = set(row.get("true_elements", ()) or ())
        predicted = set(row.get("predicted_elements", ()) or ())
        ignore = set(row.get("ignore_elements", ()) or ())
        for element in candidate_elements:
            label = classify_element(element, truth, predicted, ignore)
            if label is None:
                continue
            c = counts[element]
            if label == TP:
                c[0] += 1
            elif label == FP:
                c[1] += 1
            elif label == FN:
                c[2] += 1
            else:
                c[3] += 1
    return {el: (c[0], c[1], c[2], c[3]) for el, c in counts.items()}


@dataclass(frozen=True)
class ScoringRow:
    """The confusion core of one spectrum x algorithm benchmark row.

    Wraps the three element sets and computes confusion over any panel, honoring
    the per-spectrum don't-care band. The row dict stays the wire format — build
    one with :meth:`from_row`.
    """

    true_elements: FrozenSet[str]
    predicted_elements: FrozenSet[str]
    ignore_elements: FrozenSet[str]

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ScoringRow":
        """Build from a benchmark row dict (tolerant of missing keys / None)."""
        return cls(
            frozenset(row.get("true_elements", ()) or ()),
            frozenset(row.get("predicted_elements", ()) or ()),
            frozenset(row.get("ignore_elements", ()) or ()),
        )

    def confusion(self, candidate_elements: Sequence[str]) -> Dict[str, int]:
        """``{"tp","fp","fn","tn"}`` over a panel, honoring the don't-care band."""
        return confusion_counts(
            self.true_elements,
            self.predicted_elements,
            candidate_elements,
            self.ignore_elements,
        )

    def label_for(self, element: str) -> Optional[str]:
        """Classify one element (``None`` if it is in the don't-care band)."""
        return classify_element(
            element, self.true_elements, self.predicted_elements, self.ignore_elements
        )
