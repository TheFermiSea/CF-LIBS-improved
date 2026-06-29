"""Unified, jit-friendly :class:`PipelineSnapshot` for ``cflibs.jitpipe``.

ADR-0004 §5.1 / J0 spec §2. ONE frozen, ``jax.tree_util``-registered
struct-of-arrays bundle holding the whole atomic database (~5-6 MB) so the
jittable inversion pipeline never touches SQLite inside a trace. It *unifies*
the two pre-existing snapshot types — :class:`cflibs.core.jax_runtime.AtomicSnapshot`
(forward kernel) and :class:`cflibs.inversion.solve.iterative._AtomicSnapshot`
(lax solver) — without modifying either, and provides bridge methods to/from
both (J0 AC4).

Build path
----------
``ASD_da/libs_production.db`` --(one SQLite scan, host.py)--> RawScan
--(pure NumPy)--> PipelineSnapshot. The result is cached as a byte-stable
``.npz`` keyed by the DB content hash; a cache hit skips the SQLite scan
entirely (AC3). This module and :mod:`cflibs.jitpipe.host` /
:mod:`cflibs.jitpipe.parity` are the only ``jitpipe`` modules permitted to
import SQLite-touching code.

Per-bucket candidate-set assembly (element masks over the superset snapshot +
the ``_build_padded_arrays_from_obs`` / ``reorder`` gather) is the only
per-spectrum host<->device seam; the bridge methods below implement that gather
against both legacy consumers.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from cflibs.jitpipe import host as _host
from cflibs.plasma.partition import bound_levels_sorted

# Field groups -------------------------------------------------------------

#: Per-line array fields, length ``N_lines`` (all pytree leaves).
_LINE_FIELDS: tuple[str, ...] = (
    "line_element_index",
    "line_sp_num",
    "line_wavelength_nm",
    "line_A_ki",
    "line_E_i_ev",
    "line_E_k_ev",
    "line_g_i",
    "line_g_k",
    "line_species_index",
    "line_stark_w",
    "line_stark_alpha",
    "line_stark_shift",
    "line_aki_uncertainty",
    "line_is_resonance",
    "line_stark_source_class",
    "line_gamma_vdw_log",
)

#: Per-species / level / partition / metadata array fields (pytree leaves).
_OTHER_ARRAY_FIELDS: tuple[str, ...] = (
    "level_g",
    "level_E_ev",
    "level_mask",
    "partition_coeffs",
    "partition_coeffs_stored",
    "partition_t_min",
    "partition_t_max",
    "partition_g0",
    "partition_from_direct_sum",
    "canonical_fallback",
    "species_physics",
    "doublet_pairs",
    "doublet_rho",
    "doublet_r_thin",
    "oxide_stoichiometry",
)

#: Every array leaf, in a fixed flatten order.
_LEAF_FIELDS: tuple[str, ...] = _LINE_FIELDS + _OTHER_ARRAY_FIELDS

#: Static (aux) fields — NOT pytree leaves; carried in the treedef.
_AUX_FIELDS: tuple[str, ...] = ("species", "element_symbols")


@dataclass(frozen=True)
class PipelineSnapshot:
    """Frozen, pytree-registered struct-of-arrays bundle of the whole DB.

    The single source of atomic data for the jittable pipeline. Built once
    per process (host-only) and flows through ``jit``/``vmap`` as a pytree:
    all array fields are leaves; ``species`` and ``element_symbols`` are static
    aux carried in the treedef.

    Attributes
    ----------
    species : tuple[tuple[str, int], ...]
        ``((element, sp_num), ...)`` — the canonical species axis. Static
        (aux), not a pytree leaf. Indexed by ``line_species_index`` and by
        every per-species block.
    element_symbols : tuple[str, ...]
        Distinct element symbols indexed by ``line_element_index``. Static.
    line_element_index : ndarray, shape (N_lines,), int16
        Index into ``element_symbols`` for each line.
    line_sp_num : ndarray, shape (N_lines,), int8
        Ionisation stage number (1 = neutral) of each line.
    line_wavelength_nm : ndarray, shape (N_lines,)
        Line-center wavelengths, nm.
    line_A_ki : ndarray, shape (N_lines,)
        Einstein A coefficient, s^-1.
    line_E_i_ev, line_E_k_ev : ndarray, shape (N_lines,)
        Lower / upper level energies, eV.
    line_g_i, line_g_k : ndarray, shape (N_lines,)
        Lower / upper level statistical weights.
    line_species_index : ndarray, shape (N_lines,), int32
        Index into ``species`` for each line (``-1`` when unknown).
    line_stark_w : ndarray, shape (N_lines,)
        Stark width parameter; 0 when missing.
    line_stark_alpha : ndarray, shape (N_lines,)
        Stark temperature-power-law exponent; 0 collapses the factor to 1.
    line_stark_shift : ndarray, shape (N_lines,)
        Signed Stark line-center shift at the reference density; 0 when missing.
    line_aki_uncertainty : ndarray, shape (N_lines,)
        Relative A_ki uncertainty (catalogued); 0 when missing.
    line_is_resonance : ndarray of bool, shape (N_lines,)
        Resonance-line flag.
    line_stark_source_class : ndarray, shape (N_lines,), uint8
        Provenance class of ``stark_w`` (0 null / 1 konjevic-λ²-scaled /
        2 interpolated / 3 hydrogenic / 4 stark_b).
    line_gamma_vdw_log : ndarray, shape (N_lines,)
        log10 van-der-Waals broadening coefficient; 0 when missing.
    level_g, level_E_ev : ndarray, shape (N_species, L_max)
        Padded per-species level statistical weights / energies (eV).
    level_mask : ndarray of bool, shape (N_species, L_max)
        ``True`` where a real level sits, ``False`` in padding.
    partition_coeffs : ndarray, shape (N_species, 5)
        CANONICAL ``log U(T)`` polynomial from ``partition_spec_for`` —
        direct-sum-derived re-fit when levels exist (matches the forward
        kernel exactly, AC4). Concrete (no NaN): missing-data species carry
        ``[ln(U_fallback), 0, 0, 0, 0]``.
    partition_coeffs_stored : ndarray, shape (N_species, 5)
        RAW ``partition_functions`` table poly; NaN sentinel when absent
        (the lax-solver convention).
    partition_t_min, partition_t_max : ndarray, shape (N_species,)
        Validity window of the partition fit, K.
    partition_g0 : ndarray, shape (N_species,)
        Ground-state statistical weight (clamp floor for ``log U``).
    partition_from_direct_sum : ndarray, shape (N_species,)
        1.0 where ``partition_coeffs`` was re-fit from a direct level sum.
    canonical_fallback : ndarray, shape (N_species, 2)
        Eager canonical scalar partition fallbacks ``[U_I, U_II]``.
    species_physics : ndarray, shape (N_species, 2)
        ``[ip_ev, atomic_mass]`` per species (mass feeds Doppler widths in J6).
    doublet_pairs : ndarray, shape (P, 2), int32
        Global line-index pairs sharing an upper level (for J5).
    doublet_rho, doublet_r_thin : ndarray, shape (P,)
        Density-coupling ratio and optically-thin emission ratio per pair.
    oxide_stoichiometry : ndarray, shape (N_species,)
        O atoms per cation per species (oxide-closure weighting, J7).
    db_content_hash : str
        sha256 of the source DB file (provenance / cache key). Static, not a
        leaf.
    """

    species: tuple[tuple[str, int], ...]
    element_symbols: tuple[str, ...]

    line_element_index: Any
    line_sp_num: Any
    line_wavelength_nm: Any
    line_A_ki: Any
    line_E_i_ev: Any
    line_E_k_ev: Any
    line_g_i: Any
    line_g_k: Any
    line_species_index: Any
    line_stark_w: Any
    line_stark_alpha: Any
    line_stark_shift: Any
    line_aki_uncertainty: Any
    line_is_resonance: Any
    line_stark_source_class: Any
    line_gamma_vdw_log: Any

    level_g: Any
    level_E_ev: Any
    level_mask: Any
    partition_coeffs: Any
    partition_coeffs_stored: Any
    partition_t_min: Any
    partition_t_max: Any
    partition_g0: Any
    partition_from_direct_sum: Any
    canonical_fallback: Any
    species_physics: Any
    doublet_pairs: Any
    doublet_rho: Any
    doublet_r_thin: Any
    oxide_stoichiometry: Any

    db_content_hash: str = ""

    # -- shape conveniences ------------------------------------------------

    @property
    def n_lines(self) -> int:
        """Number of lines in the snapshot."""
        return int(np.asarray(self.line_wavelength_nm).shape[0])

    @property
    def n_species(self) -> int:
        """Number of species rows in the snapshot."""
        return len(self.species)

    @property
    def level_pad(self) -> tuple[int, int]:
        """``(N_species, L_max)`` shape of the padded level blocks."""
        arr = np.asarray(self.level_g)
        return (int(arr.shape[0]), int(arr.shape[1]) if arr.ndim > 1 else 0)

    # -- construction ------------------------------------------------------

    @classmethod
    def from_raw_scan(
        cls, raw: "_host.RawScan", *, db_content_hash: str = ""
    ) -> "PipelineSnapshot":
        """Build a snapshot from a pure-NumPy :class:`~cflibs.jitpipe.host.RawScan`.

        No SQLite is touched here — :func:`cflibs.jitpipe.host.scan_database`
        already did the single scan.
        """
        kwargs: dict[str, Any] = {name: getattr(raw, name) for name in _LEAF_FIELDS}
        return cls(
            species=tuple(raw.species),
            element_symbols=tuple(raw.element_symbols),
            db_content_hash=db_content_hash,
            **kwargs,
        )

    # -- (de)serialization to dict / npz ----------------------------------

    def to_payload(self) -> dict[str, np.ndarray]:
        """Return a flat ``{name: ndarray}`` payload for ``.npz`` caching.

        Static metadata (``species``, ``element_symbols``, ``db_content_hash``)
        is JSON-serialized into a single ``__meta__`` string array so the cache
        is fully self-describing and byte-stable.
        """
        payload: dict[str, np.ndarray] = {
            name: np.asarray(getattr(self, name)) for name in _LEAF_FIELDS
        }
        meta = {
            "species": [list(s) for s in self.species],
            "element_symbols": list(self.element_symbols),
            "db_content_hash": self.db_content_hash,
        }
        payload["__meta__"] = np.asarray(json.dumps(meta, sort_keys=True))
        return payload

    @classmethod
    def from_payload(cls, payload: dict[str, np.ndarray]) -> "PipelineSnapshot":
        """Reconstruct a snapshot from a ``.npz`` payload (inverse of :meth:`to_payload`)."""
        meta = json.loads(str(payload["__meta__"]))
        kwargs = {name: payload[name] for name in _LEAF_FIELDS}
        return cls(
            species=tuple((str(e), int(s)) for e, s in meta["species"]),
            element_symbols=tuple(meta["element_symbols"]),
            db_content_hash=meta["db_content_hash"],
            **kwargs,
        )

    # -- bridge: PipelineSnapshot -> AtomicSnapshot (forward kernel) -------

    def to_atomic_snapshot(self, *, include_levels: bool = True) -> Any:
        """Convert to a :class:`cflibs.core.jax_runtime.AtomicSnapshot`.

        Maps the unified line / partition / level blocks onto the forward
        kernel's field names (canonical ``_nm`` / ``_ev`` suffixes). Used by
        ``forward.py`` so it can feed :func:`cflibs.radiation.kernels.forward_model`
        verbatim (J7). Does not modify ``AtomicSnapshot`` (J0 AC4).

        Parameters
        ----------
        include_levels : bool, optional
            Populate the optional padded ``level_*`` arrays. Default True.

        Returns
        -------
        AtomicSnapshot
        """
        from cflibs.core.jax_runtime import AtomicSnapshot

        ip = np.asarray(self.species_physics)[:, 0]
        kwargs: dict[str, Any] = {
            "species": tuple(self.species),
            "line_wavelengths_nm": self.line_wavelength_nm,
            "line_A_ki": self.line_A_ki,
            "line_E_k_ev": self.line_E_k_ev,
            "line_g_k": self.line_g_k,
            "line_E_i_ev": self.line_E_i_ev,
            "line_g_i": self.line_g_i,
            "line_species_index": np.asarray(self.line_species_index, dtype=np.int32),
            "line_stark_w": self.line_stark_w,
            "line_stark_alpha": self.line_stark_alpha,
            "line_natural_w": np.zeros(self.n_lines, dtype=np.float64),
            "line_stark_d": self.line_stark_shift,
            "partition_coeffs": self.partition_coeffs,
            "ionization_potential_ev": np.asarray(ip, dtype=np.float64),
            "partition_t_min": self.partition_t_min,
            "partition_t_max": self.partition_t_max,
            "partition_g0": self.partition_g0,
        }
        if include_levels:
            kwargs["level_g"] = self.level_g
            kwargs["level_E_ev"] = self.level_E_ev
            kwargs["level_mask"] = self.level_mask
        return AtomicSnapshot(**kwargs)

    @classmethod
    def from_atomic_snapshot(cls, snap: Any, *, db_content_hash: str = "") -> "PipelineSnapshot":
        """Build a (partial) snapshot from a forward-kernel ``AtomicSnapshot``.

        Bridge for the transition (J0 AC4): fields absent from ``AtomicSnapshot``
        (stark shift/source class, aki uncertainty, doublets, oxide, masses) are
        zero/identity-filled. ``line_element_index`` is derived from the species
        axis. Used by tests to assert round-trip field parity.
        """
        species = tuple(snap.species)
        n_species = len(species)
        wl = np.asarray(snap.line_wavelengths_nm)
        n = wl.shape[0]
        sp_index = np.asarray(snap.line_species_index, dtype=np.int32)

        # Element symbol table + per-line element index from species axis.
        el_for_species = [s[0] for s in species]
        seen: dict[str, int] = {}
        for el in el_for_species:
            if el not in seen:
                seen[el] = len(seen)
        element_symbols = tuple(sorted(seen, key=lambda e: seen[e]))
        el_index = np.array(
            [seen[el_for_species[i]] if 0 <= i < n_species else -1 for i in sp_index],
            dtype=np.int16,
        )
        sp_num = np.array(
            [species[i][1] if 0 <= i < n_species else 0 for i in sp_index],
            dtype=np.int8,
        )

        def _opt(name: str, fill: np.ndarray) -> np.ndarray:
            val = getattr(snap, name, None)
            return fill if val is None else np.asarray(val)

        ip = np.asarray(snap.ionization_potential_ev, dtype=np.float64)
        species_physics = np.zeros((n_species, 2), dtype=np.float64)
        species_physics[:, 0] = ip[:n_species] if ip.shape[0] >= n_species else 0.0

        level_g = _opt("level_g", np.zeros((n_species, 1)))
        level_E = _opt("level_E_ev", np.zeros((n_species, 1)))
        level_mask = _opt("level_mask", np.zeros((n_species, 1), dtype=bool))

        return cls(
            species=species,
            element_symbols=element_symbols,
            line_element_index=el_index,
            line_sp_num=sp_num,
            line_wavelength_nm=wl,
            line_A_ki=np.asarray(snap.line_A_ki),
            line_E_i_ev=np.asarray(snap.line_E_i_ev),
            line_E_k_ev=np.asarray(snap.line_E_k_ev),
            line_g_i=np.asarray(snap.line_g_i),
            line_g_k=np.asarray(snap.line_g_k),
            line_species_index=sp_index,
            line_stark_w=np.asarray(snap.line_stark_w),
            line_stark_alpha=np.asarray(snap.line_stark_alpha),
            line_stark_shift=_opt("line_stark_d", np.zeros(n)),
            line_aki_uncertainty=np.zeros(n),
            line_is_resonance=np.zeros(n, dtype=bool),
            line_stark_source_class=np.zeros(n, dtype=np.uint8),
            line_gamma_vdw_log=np.zeros(n),
            level_g=np.asarray(level_g),
            level_E_ev=np.asarray(level_E),
            level_mask=np.asarray(level_mask, dtype=bool),
            partition_coeffs=np.asarray(snap.partition_coeffs),
            partition_coeffs_stored=np.asarray(snap.partition_coeffs),
            partition_t_min=_opt("partition_t_min", np.full(n_species, 2000.0)),
            partition_t_max=_opt("partition_t_max", np.full(n_species, 25000.0)),
            partition_g0=_opt("partition_g0", np.ones(n_species)),
            partition_from_direct_sum=_opt("partition_from_direct_sum", np.zeros(n_species)),
            canonical_fallback=np.tile(
                np.array([_host.FALLBACK_U_I, _host.FALLBACK_U_II]), (n_species, 1)
            ),
            species_physics=species_physics,
            doublet_pairs=np.zeros((0, 2), dtype=np.int32),
            doublet_rho=np.zeros(0),
            doublet_r_thin=np.zeros(0),
            oxide_stoichiometry=np.ones(n_species),
            db_content_hash=db_content_hash,
        )

    # -- bridge: PipelineSnapshot -> _AtomicSnapshot (lax solver) ----------

    def to_lax_snapshot(self, elements: list[str]) -> Any:
        """Convert to a per-element :class:`~cflibs.inversion.solve.iterative._AtomicSnapshot`.

        Implements the per-bucket candidate-set assembly seam: gathers the
        stage-I/II partition coefficients, IPs, eager scalar fallbacks and
        padded level blocks for ``elements`` from the superset snapshot, in the
        exact field layout the lax solver consumes inside ``_run_lax_while_loop``
        (J7 reuses the kernels verbatim). Does not modify ``_AtomicSnapshot``
        (J0 AC4).

        Parameters
        ----------
        elements : list of str
            Element symbols, defining the bundle ordering ``E``.

        Returns
        -------
        _AtomicSnapshot
        """
        from cflibs.inversion.solve.iterative import _AtomicSnapshot

        sp_to_row = {sp: i for i, sp in enumerate(self.species)}
        # The lax solver consumes the RAW stored poly with the NaN sentinel
        # (it prefers direct levels and falls back to the stored polynomial),
        # NOT the canonical re-fit used by the forward kernel.
        coeffs = np.asarray(self.partition_coeffs_stored)
        ip_col = np.asarray(self.species_physics)[:, 0]
        fallback = np.asarray(self.canonical_fallback)
        level_g = np.asarray(self.level_g)
        level_E = np.asarray(self.level_E_ev)
        level_mask = np.asarray(self.level_mask)

        E = len(elements)
        ip0 = np.zeros(E, dtype=np.float64)
        use_direct = np.zeros((E, 2), dtype=bool)
        coeffs_I = np.zeros((E, _host.N_POLY), dtype=np.float64)
        coeffs_II = np.zeros((E, _host.N_POLY), dtype=np.float64)
        fb_I = np.full(E, _host.FALLBACK_U_I, dtype=np.float64)
        fb_II = np.full(E, _host.FALLBACK_U_II, dtype=np.float64)

        gI_rows: list[np.ndarray] = []
        EI_rows: list[np.ndarray] = []
        gII_rows: list[np.ndarray] = []
        EII_rows: list[np.ndarray] = []
        ipI = np.zeros(E, dtype=np.float64)
        ipII = np.zeros(E, dtype=np.float64)

        def _row_levels(row: int | None) -> tuple[np.ndarray, np.ndarray, bool]:
            if row is None:
                return np.empty(0), np.empty(0), False
            m = level_mask[row]
            if not np.any(m):
                return np.empty(0), np.empty(0), False
            g = level_g[row][m]
            E = level_E[row][m]
            # Mirror the reference direct-sum level set
            # (``cflibs.plasma.partition.get_levels_for_species``): sort by
            # energy ascending and DROP autoionizing levels at/above the
            # species' ionization potential. The superset block from
            # ``_scan_levels`` keeps every tabulated level (the complete-DB /
            # Kurucz M5 ingest now stores levels above the IP), so without this
            # cut ``to_lax_snapshot`` would carry bound+autoionizing levels
            # while ``_AtomicSnapshot.from_solver`` carries bound-only — a
            # field-for-field parity break (J0 AC4). The lax kernel already
            # masks ``E < ip`` at eval time, so this is representation-only and
            # leaves the partition value unchanged.
            g, E = bound_levels_sorted(g, E, float(ip_col[row]))
            return g, E, True

        for i, el in enumerate(elements):
            r1 = sp_to_row.get((el, 1))
            r2 = sp_to_row.get((el, 2))
            # ``ip0_eV`` is always the stage-I IP (used by IPD/Saha).
            if r1 is not None:
                ip0[i] = float(ip_col[r1])

            gI, EI, hasI = _row_levels(r1)
            gII, EII, hasII = _row_levels(r2)
            use_direct[i, 0] = hasI
            use_direct[i, 1] = hasII

            # ``ip_*_for_direct`` are the DIRECT-SUM cutoff IPs — the reference
            # only populates them when the level probe succeeds (use_direct);
            # they stay 0 otherwise.
            if hasI and r1 is not None:
                ipI[i] = float(ip_col[r1])
            if hasII and r2 is not None:
                ipII[i] = float(ip_col[r2])
            gI_rows.append(gI)
            EI_rows.append(EI)
            gII_rows.append(gII)
            EII_rows.append(EII)

            if not hasI:
                if r1 is not None and not np.any(np.isnan(coeffs[r1])):
                    coeffs_I[i] = coeffs[r1]
                else:
                    coeffs_I[i] = np.nan
                    fb_I[i] = float(fallback[r1, 0]) if r1 is not None else _host.FALLBACK_U_I
            if not hasII:
                if r2 is not None and not np.any(np.isnan(coeffs[r2])):
                    coeffs_II[i] = coeffs[r2]
                else:
                    coeffs_II[i] = np.nan
                    fb_II[i] = float(fallback[r2, 1]) if r2 is not None else _host.FALLBACK_U_II

        gI_pad, mI = _host.pad_ragged(gI_rows)
        EI_pad, _ = _host.pad_ragged(EI_rows)
        EI_pad = _align_width(EI_pad, gI_pad)
        gII_pad, mII = _host.pad_ragged(gII_rows)
        EII_pad, _ = _host.pad_ragged(EII_rows)
        EII_pad = _align_width(EII_pad, gII_pad)

        return _AtomicSnapshot(
            elements=tuple(elements),
            ip0_eV=ip0,
            use_direct=use_direct,
            g_levels_I=gI_pad,
            E_levels_I=EI_pad,
            ip_I_for_direct=ipI,
            levels_mask_I=mI,
            g_levels_II=gII_pad,
            E_levels_II=EII_pad,
            ip_II_for_direct=ipII,
            levels_mask_II=mII,
            coeffs_I=coeffs_I,
            coeffs_II=coeffs_II,
            fallback_U_I=fb_I,
            fallback_U_II=fb_II,
        )


def _align_width(narrow: np.ndarray, wide: np.ndarray) -> np.ndarray:
    """Re-pad ``narrow`` to ``wide``'s column count (mirrors ``_align_E_width``)."""
    if narrow.shape[1] == wide.shape[1]:
        return narrow
    out = np.zeros_like(wide)
    out[:, : narrow.shape[1]] = narrow
    return out


# ---------------------------------------------------------------------------
# Public build entry point — DB -> snapshot with byte-stable .npz cache.
# ---------------------------------------------------------------------------


def build_snapshot(
    db_path: str | os.PathLike | None = None,
    *,
    cache: bool = True,
    cache_dir: Path | None = None,
    _scan_counter: list[int] | None = None,
) -> PipelineSnapshot:
    """Build (or load from cache) the unified :class:`PipelineSnapshot`.

    Cache flow (J0 AC3): the DB content hash keys a ``.npz`` cache. On a hit
    the SQLite scan is skipped entirely; on a miss (or hash mismatch =
    invalidation) a single scan rebuilds and re-writes the cache. Two builds
    from the same DB produce byte-identical ``.npz`` (byte stability).

    Parameters
    ----------
    db_path : str or os.PathLike, optional
        SQLite atomic database. Defaults to
        :data:`cflibs.jitpipe.host.DEFAULT_DB_PATH`.
    cache : bool, optional
        When True (default) read/write the ``.npz`` cache; False forces a
        fresh scan and skips persistence.
    cache_dir : Path, optional
        Override the cache directory (tests use a tmp dir).
    _scan_counter : list[int], optional
        Test hook: appended-to once per actual SQLite scan, so tests can assert
        a cache hit skipped the scan (no-SQLite-after-build guard).

    Returns
    -------
    PipelineSnapshot
    """
    db_path = os.fspath(db_path) if db_path is not None else _host.DEFAULT_DB_PATH
    content_hash = _host.db_content_hash(db_path)

    if cache:
        cache_path = _host.default_cache_path(content_hash, cache_dir=cache_dir)
        if cache_path.exists():
            payload = _host.load_npz_cache(cache_path)
            meta = json.loads(str(payload["__meta__"]))
            if meta.get("db_content_hash") == content_hash:
                return PipelineSnapshot.from_payload(payload)
            # Hash mismatch -> stale cache; fall through to a rebuild.

    if _scan_counter is not None:
        _scan_counter.append(1)
    raw = _host.scan_database(db_path)
    snap = PipelineSnapshot.from_raw_scan(raw, db_content_hash=content_hash)

    if cache:
        cache_path = _host.default_cache_path(content_hash, cache_dir=cache_dir)
        _host.save_npz_cache(cache_path, snap.to_payload())

    return snap


# ---------------------------------------------------------------------------
# Pytree registration — all array fields are leaves; species/element_symbols/
# db_content_hash are static aux carried in the treedef.
# ---------------------------------------------------------------------------


def _flatten(snap: PipelineSnapshot):
    children = tuple(getattr(snap, name) for name in _LEAF_FIELDS)
    aux = (snap.species, snap.element_symbols, snap.db_content_hash)
    return children, aux


def _unflatten(aux: tuple, children: tuple) -> PipelineSnapshot:
    species, element_symbols, db_content_hash = aux
    kwargs = dict(zip(_LEAF_FIELDS, children))
    return PipelineSnapshot(
        species=species,
        element_symbols=element_symbols,
        db_content_hash=db_content_hash,
        **kwargs,
    )


try:  # pragma: no cover - exercised whenever JAX is installed (the norm)
    import jax

    jax.tree_util.register_pytree_node(PipelineSnapshot, _flatten, _unflatten)
except ImportError:  # pragma: no cover - jitpipe requires JAX; see __init__
    pass


# Sanity: every dataclass array field is accounted for as either a leaf or aux.
_DECLARED = {f.name for f in fields(PipelineSnapshot)} - {"db_content_hash"}
assert _DECLARED == set(_LEAF_FIELDS) | set(_AUX_FIELDS), (
    "PipelineSnapshot field/leaf bookkeeping drifted: "
    f"{_DECLARED ^ (set(_LEAF_FIELDS) | set(_AUX_FIELDS))}"
)
