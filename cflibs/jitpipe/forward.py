"""Forward model adapter (J7 stub; ADR-0004 §5.1.1).

Thin jittable wrapper that feeds the unified :class:`PipelineSnapshot` to the
existing forward kernel :func:`cflibs.radiation.kernels.forward_model` (which
consumes an :class:`~cflibs.core.jax_runtime.AtomicSnapshot`) via
:meth:`cflibs.jitpipe.snapshot.PipelineSnapshot.to_atomic_snapshot`. Logic
lands in bead J7. No SQLite, no host imports (import-hygiene test) — the bridge
to ``AtomicSnapshot`` lives in ``snapshot.py``, which is a carve-out module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from cflibs.jitpipe.params import PipelineParams, StaticConfig
    from cflibs.jitpipe.snapshot import PipelineSnapshot


def forward_spectrum(
    temperature_eV: Any,
    electron_density: Any,
    composition: Any,
    wavelengths_nm: Any,
    snapshot: "PipelineSnapshot",
    params: "PipelineParams",
    static: "StaticConfig",
) -> Any:
    """Synthesize a spectrum from plasma parameters (J7).

    Bridges to :func:`cflibs.radiation.kernels.forward_model` via
    :meth:`PipelineSnapshot.to_atomic_snapshot`.

    Parameters
    ----------
    temperature_eV : array
        Plasma temperature, eV.
    electron_density : array
        Electron density, cm^-3.
    composition : array
        Number/mass fractions per species or element.
    wavelengths_nm : array
        Output wavelength grid, nm.
    snapshot : PipelineSnapshot
        Atomic-data snapshot.
    params : PipelineParams
        Traced knobs.
    static : StaticConfig
        Static config (``broadening_mode``, ``apply_self_absorption``).

    Returns
    -------
    array
        Synthetic spectrum on ``wavelengths_nm``.

    Raises
    ------
    NotImplementedError
        Stage logic is implemented in bead J7.
    """
    raise NotImplementedError("jitpipe.forward is a J0 skeleton; logic lands in J7")
