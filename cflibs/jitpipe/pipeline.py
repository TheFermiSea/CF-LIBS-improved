"""End-to-end jittable pipeline orchestration (J7 stub; ADR-0004 §5.1.1).

Wires the stage modules (preprocess -> detect -> calibrate -> identify -> fit
-> selfabs -> stark -> solve) into single-spectrum (:func:`run_one`) and
batched (:func:`run_batch`, ``vmap`` over ``B``) entry points. The full
orchestration logic lands in bead J7; J0 ships importable signatures so the
public API in ``__init__.py`` resolves and the boundary tests can run. No
SQLite, no host imports (import-hygiene test): the snapshot is built once on
the host (``snapshot.build_snapshot``) and passed in as a pytree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


def run_one(
    intensities: Any,
    wavelengths_nm: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Run the full jittable inversion pipeline on ONE spectrum (J7).

    Parameters
    ----------
    intensities : array
        Raw measured intensities, shape ``(N_pix,)``.
    wavelengths_nm : array
        Wavelength axis, nm, shape ``(N_pix,)``.
    snapshot : PipelineSnapshot
        Atomic-data snapshot (host-built, passed as a pytree).
    params : PipelineParams
        Traced continuous knobs.
    static : StaticConfig
        Static config — the jit cache key.

    Returns
    -------
    object
        Inversion result (T, n_e, concentrations, diagnostics). Concrete type
        defined in J7.

    Raises
    ------
    NotImplementedError
        Orchestration logic is implemented in bead J7.
    """
    raise NotImplementedError("jitpipe.run_one is a J0 skeleton; logic lands in J7")


def run_batch(
    intensities: Any,
    wavelengths_nm: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Run the pipeline on a BATCH of spectra (``vmap`` over ``B``) (J7).

    Parameters
    ----------
    intensities : array
        Raw intensities, shape ``(B, N_pix)``.
    wavelengths_nm : array
        Wavelength axis, nm, shape ``(N_pix,)`` (shared) or ``(B, N_pix)``.
    snapshot : PipelineSnapshot
        Atomic-data snapshot (host-built, passed as a pytree; broadcast).
    params : PipelineParams
        Traced continuous knobs (shared across the batch).
    static : StaticConfig
        Static config — the jit cache key (carries ``batch_size`` ``B``).

    Returns
    -------
    object
        Batched inversion results. Concrete type defined in J7.

    Raises
    ------
    NotImplementedError
        Orchestration logic is implemented in bead J7.
    """
    raise NotImplementedError("jitpipe.run_batch is a J0 skeleton; logic lands in J7")
