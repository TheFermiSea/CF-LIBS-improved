"""Scoreboard dataset registry: the shared truth-bearing adapter contract (bead A1).

The goal-metric scoreboard (``cflibs scoreboard``) measures the only things
that matter for CF-LIBS — element-identification accuracy, composition
accuracy and runtime — across every dataset that carries ground truth. This
module is the *contract* both the harness and the dataset adapters build to:

An **adapter** is a zero-argument generator function yielding tuples
``(spectrum_id, wavelength_nm, intensity, truth)`` where

* ``spectrum_id`` is a ``str`` unique within its dataset,
* ``wavelength_nm`` / ``intensity`` are 1-D ``np.ndarray`` of equal length,
* ``truth`` is a :class:`SpectrumTruth`.

Adapters MUST be lazy (no heavy I/O at import time), deterministic, and
**skip-with-log** — an adapter whose data files are absent logs the gap and
yields nothing instead of raising. Element symbols use standard
capitalization (``Fe``, ``Si``). Trace elements below the adapter's
documented cutoff (~0.01 wt%) may be excluded from ``elements_present`` with
the cutoff recorded in ``notes``.

This module is intentionally tiny and dependency-free (stdlib only) so both
the harness branch and the datasets branch can build against it without
import-order coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Iterator, Optional, Tuple

#: What one adapter yield looks like: (spectrum_id, wavelength_nm, intensity, truth).
#: Arrays are typed ``Any`` here to keep this module numpy-free.
AdapterYield = Tuple[str, Any, Any, "SpectrumTruth"]

#: A zero-argument generator function producing the dataset's spectra.
AdapterFactory = Callable[[], Iterator[AdapterYield]]

#: Valid values of :attr:`SpectrumTruth.composition_basis`.
COMPOSITION_BASES = ("element_wt", "presence_only")


@dataclass(frozen=True)
class SpectrumTruth:
    """Ground truth for one spectrum.

    Attributes
    ----------
    elements_present : frozenset[str]
        Ground-truth element set (REQUIRED; the identification-scoring
        target). Standard capitalization (``Fe``, ``Si``).
    composition_wt : dict[str, float] | None
        ELEMENT wt% (0-100). Adapters convert oxide certificates to element
        basis before yielding. ``None`` when only presence truth exists.
    composition_basis : str
        ``"element_wt"`` when ``composition_wt`` is given, else
        ``"presence_only"``. Derived automatically when left empty.
    resolving_power : float | None
        Instrument resolving power hint when known.
    notes : str
        Provenance: instrument, reference, caveats, trace-element cutoff.
    """

    elements_present: frozenset[str]
    composition_wt: Optional[dict[str, float]] = None
    composition_basis: str = ""
    resolving_power: Optional[float] = None
    notes: str = ""

    def __post_init__(self) -> None:
        derived = "element_wt" if self.composition_wt is not None else "presence_only"
        if not self.composition_basis:
            object.__setattr__(self, "composition_basis", derived)
        elif self.composition_basis != derived:
            raise ValueError(
                f"composition_basis={self.composition_basis!r} inconsistent with "
                f"composition_wt={'given' if self.composition_wt is not None else 'None'} "
                f"(expected {derived!r})"
            )
        if self.composition_basis not in COMPOSITION_BASES:
            raise ValueError(
                f"composition_basis must be one of {COMPOSITION_BASES}, "
                f"got {self.composition_basis!r}"
            )


@dataclass(frozen=True)
class DatasetEntry:
    """One registered scoreboard dataset."""

    name: str
    adapter_factory: AdapterFactory
    tags: frozenset[str] = field(default_factory=frozenset)


#: Module-level registry, insertion-ordered. Tests monkeypatch this dict.
_REGISTRY: dict[str, DatasetEntry] = {}


def register_dataset(
    name: str,
    adapter_factory: AdapterFactory,
    *,
    tags: Iterable[str] = (),
    replace: bool = False,
) -> DatasetEntry:
    """Register a dataset adapter under ``name``.

    Parameters
    ----------
    name : str
        Unique dataset name (e.g. ``"bhvo2_chemcam"``).
    adapter_factory : AdapterFactory
        Zero-argument generator function (see module docstring contract).
    tags : Iterable[str]
        Filter tags, e.g. ``("real", "geological")`` or ``("synthetic",)``.
    replace : bool
        Allow overwriting an existing registration (default False: duplicate
        names raise ``ValueError`` so two branches cannot silently shadow
        each other's datasets at integration time).
    """
    if name in _REGISTRY and not replace:
        raise ValueError(f"Scoreboard dataset {name!r} is already registered (use replace=True).")
    entry = DatasetEntry(name=name, adapter_factory=adapter_factory, tags=frozenset(tags))
    _REGISTRY[name] = entry
    return entry


def iter_datasets(
    *,
    names: Optional[Iterable[str]] = None,
    tags: Optional[Iterable[str]] = None,
) -> Iterator[DatasetEntry]:
    """Iterate registered datasets in registration order, optionally filtered.

    Parameters
    ----------
    names : Iterable[str] or None
        Keep only datasets whose name is in this set. Unknown names raise
        ``KeyError`` (a typo must not silently produce an empty board).
    tags : Iterable[str] or None
        Keep only datasets having at least one of these tags.
    """
    name_filter = set(names) if names is not None else None
    tag_filter = set(tags) if tags is not None else None
    if name_filter is not None:
        unknown = name_filter - set(_REGISTRY)
        if unknown:
            raise KeyError(
                f"Unknown scoreboard dataset(s): {sorted(unknown)}. "
                f"Registered: {sorted(_REGISTRY)}"
            )
    for entry in _REGISTRY.values():
        if name_filter is not None and entry.name not in name_filter:
            continue
        if tag_filter is not None and not (entry.tags & tag_filter):
            continue
        yield entry


def registered_names() -> list[str]:
    """Names of all registered datasets, in registration order."""
    return list(_REGISTRY)
