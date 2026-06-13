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
    *,
    instrument: Any = None,
    path_length_m: float = 0.01,
) -> Any:
    """Synthesize a spectrum from plasma parameters (J8).

    Bridges the unified :class:`PipelineSnapshot` to the existing forward kernel
    :func:`cflibs.radiation.kernels.forward_model` via
    :meth:`PipelineSnapshot.to_atomic_snapshot`. The forward stage is off the
    M1 inversion-parity critical path (it is the synthetic-generation /
    differentiability surface, ADR §6.2); this thin wrapper makes the
    snapshot->forward bridge callable end to end.

    Parameters
    ----------
    temperature_eV : array
        Plasma temperature, eV.
    electron_density : array
        Electron density, cm^-3.
    composition : dict[str, float]
        Per-element number densities (cm^-3) for the plasma state.
    wavelengths_nm : array
        Output wavelength grid, nm.
    snapshot : PipelineSnapshot
        Atomic-data snapshot.
    params : PipelineParams
        Traced knobs (unused by the forward bridge; reserved).
    static : StaticConfig
        Static config (``broadening_mode``, ``apply_self_absorption``).
    instrument : InstrumentModel, optional
        Instrument model; defaults to a 0.05 nm fixed-FWHM Gaussian.
    path_length_m : float, optional
        Plasma path length (m) for the optional self-absorption branch.

    Returns
    -------
    array
        Synthetic spectrum on ``wavelengths_nm``.
    """
    del params  # reserved; the forward bridge is parameter-free for M1.
    from cflibs.core.constants import EV_TO_K
    from cflibs.instrument.model import InstrumentModel
    from cflibs.plasma.state import SingleZoneLTEPlasma
    from cflibs.radiation.kernels import BroadeningMode, forward_model

    if instrument is None:
        instrument = InstrumentModel(resolution_fwhm_nm=0.05)

    T_K = float(temperature_eV) * EV_TO_K
    atomic = snapshot.to_atomic_snapshot()
    # The forward kernel reads a density for EVERY element in the snapshot
    # (the full-DB superset); elements absent from ``composition`` carry zero
    # density (they emit nothing). This keeps the bridge usable against the
    # superset snapshot without a per-element subset rebuild.
    snap_elements = sorted({el for el, _sp in snapshot.species})
    species = {el: float(dict(composition).get(el, 0.0)) for el in snap_elements}
    plasma = SingleZoneLTEPlasma(T_e=T_K, n_e=float(electron_density), species=species)
    mode = getattr(BroadeningMode, str(static.broadening_mode).upper(), BroadeningMode.LEGACY)
    return forward_model(
        plasma,
        atomic,
        instrument,
        wavelengths_nm,
        broadening_mode=mode,
        path_length_m=path_length_m,
        apply_self_absorption=bool(static.apply_self_absorption),
    )
