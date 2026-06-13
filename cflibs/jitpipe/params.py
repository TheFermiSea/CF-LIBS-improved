"""Traced :class:`PipelineParams` + hashable :class:`StaticConfig` (ADR-0004 §5.1).

The static-vs-traced split is the jit cache key contract, mirroring the
documented split at ``cflibs/radiation/kernels.py:28-38``:

* **traced** (``PipelineParams``) — every *continuous* knob of
  :class:`cflibs.inversion.pipeline.AnalysisPipelineConfig` the stage specs
  name: detection thresholds, windows, tolerances, score weights, solver
  tolerances. A traced pytree of ``float`` leaves, so changing any leaf flows
  as a new traced value WITHOUT retriggering compilation (J0 AC6).
* **static** (``StaticConfig``) — hashable structural choices that DO key the
  jit cache: the shape bucket id, broadening mode, padding constants,
  ``max_iters``, batch size ``B``. Changing any of these is a recompile by
  design.

Discrete / structural knobs of ``AnalysisPipelineConfig`` (e.g.
``closure_mode``, ``saha_boltzmann_graph``, ``wavelength_calibration``) belong
in ``StaticConfig`` (or are resolved on the host before the traced region),
because branching on them inside a trace would change the compiled graph.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any


@dataclass(frozen=True)
class PipelineParams:
    """Traced pytree of every continuous CF-LIBS pipeline knob.

    All leaves are scalar ``float`` (boxed to JAX scalars when the snapshot
    flows through ``jit``). Changing any leaf is a *value* change, not a shape
    or static change, so it reuses the compiled graph (J0 AC6). Field defaults
    mirror :class:`cflibs.inversion.pipeline.AnalysisPipelineConfig`
    (``cflibs/inversion/pipeline.py:124-180``).

    Attributes
    ----------
    min_relative_intensity : float
        Detection floor on catalogued relative line intensity.
    top_k_per_element : float
        Soft cap on candidate lines kept per element (continuous so it can be
        traced; rounded where a count is needed).
    wavelength_tolerance_nm : float
        Peak<->catalog match window half-width, nm.
    min_peak_height : float
        Detection threshold on normalized peak height.
    peak_width_nm : float
        Integration window width per line, nm.
    isolation_wavelength_nm : float
        Minimum neighbour spacing for a line to count as isolated, nm.
    min_snr : float
        Minimum per-line signal-to-noise to retain a line.
    min_energy_spread_ev : float
        Minimum upper-level energy spread for a usable Boltzmann fit, eV.
    min_lines_per_element : float
        Minimum retained lines per element (continuous; rounded for counts).
    max_lines_per_element : float
        Maximum retained lines per element.
    residual_shift_scan_nm : float
        Residual comb shift-scan half-width after a quality-passed calibration.
    global_shift_scan_nm : float
        Global comb shift-scan half-width when calibration is skipped/failed.
    max_iterations : float
        Iterative-solver iteration cap (continuous; the *static* loop bound
        lives in :class:`StaticConfig.max_iters`; this is the convergence
        budget passed as a traced value).
    t_tolerance_k : float
        Temperature convergence tolerance, K.
    ne_tolerance_frac : float
        Electron-density convergence tolerance (fractional).
    pressure_pa : float
        Plasma pressure for charge/pressure balance, Pa.
    boltzmann_weight_cap : float
        Cap on per-point Boltzmann-fit weights.
    min_boltzmann_r2 : float
        Minimum Boltzmann-plot R^2 to accept a species' temperature.
    """

    # Detection + selection knobs (continuous).
    min_relative_intensity: float = 0.0
    top_k_per_element: float = 60.0
    wavelength_tolerance_nm: float = 0.1
    min_peak_height: float = 0.01
    peak_width_nm: float = 0.2
    isolation_wavelength_nm: float = 0.1
    min_snr: float = 10.0
    min_energy_spread_ev: float = 2.0
    min_lines_per_element: float = 3.0
    max_lines_per_element: float = 20.0

    # Calibration shift-scan knobs (continuous).
    residual_shift_scan_nm: float = 0.0
    global_shift_scan_nm: float = 0.5

    # Iterative-solver knobs (continuous).
    max_iterations: float = 20.0
    t_tolerance_k: float = 100.0
    ne_tolerance_frac: float = 0.1
    pressure_pa: float = 101325.0
    boltzmann_weight_cap: float = 5.0
    min_boltzmann_r2: float = 0.3

    @classmethod
    def from_analysis_config(cls, cfg: Any) -> "PipelineParams":
        """Build from an :class:`~cflibs.inversion.pipeline.AnalysisPipelineConfig`.

        Pulls only the continuous knobs; ``None`` values fall back to this
        class's defaults (e.g. ``min_relative_intensity=None`` -> 0.0). Discrete
        / structural fields of the analysis config are ignored here — they
        belong in :class:`StaticConfig`.
        """

        def _g(name: str, default: float) -> float:
            val = getattr(cfg, name, None)
            return default if val is None else float(val)

        return cls(
            min_relative_intensity=_g("min_relative_intensity", 0.0),
            top_k_per_element=_g("top_k_per_element", 60.0),
            wavelength_tolerance_nm=_g("wavelength_tolerance_nm", 0.1),
            min_peak_height=_g("min_peak_height", 0.01),
            peak_width_nm=_g("peak_width_nm", 0.2),
            isolation_wavelength_nm=_g("isolation_wavelength_nm", 0.1),
            min_snr=_g("min_snr", 10.0),
            min_energy_spread_ev=_g("min_energy_spread_ev", 2.0),
            min_lines_per_element=_g("min_lines_per_element", 3.0),
            max_lines_per_element=_g("max_lines_per_element", 20.0),
            residual_shift_scan_nm=_g("residual_shift_scan_nm", 0.0),
            global_shift_scan_nm=_g("global_shift_scan_nm", 0.5),
            max_iterations=_g("max_iterations", 20.0),
            t_tolerance_k=_g("t_tolerance_k", 100.0),
            ne_tolerance_frac=_g("ne_tolerance_frac", 0.1),
            pressure_pa=_g("pressure_pa", 101325.0),
            boltzmann_weight_cap=_g("boltzmann_weight_cap", 5.0),
            min_boltzmann_r2=_g("min_boltzmann_r2", 0.3),
        )


#: The ordered tuple of traced leaf field names (the pytree flatten order).
_PARAM_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(PipelineParams))


def _params_flatten(p: PipelineParams):
    return tuple(getattr(p, name) for name in _PARAM_FIELDS), None


def _params_unflatten(_aux: Any, children: tuple) -> PipelineParams:
    return PipelineParams(**dict(zip(_PARAM_FIELDS, children)))


@dataclass(frozen=True)
class StaticConfig:
    """Hashable static config — the jit cache key (ADR-0004 §5.4).

    Every field is hashable so an instance can be a ``static_argnums`` argument
    or part of a ``functools.lru_cache`` key. Changing any field is a recompile
    by design (it changes the compiled graph's shapes or branches).

    Attributes
    ----------
    bucket_id : int
        Padded line-count bucket (see
        :func:`cflibs.jitpipe.host.bucket_for_n_lines`). Selects the line-axis
        shape, so it keys the cache.
    n_species : int
        Padded species-axis length for the candidate set.
    level_pad : int
        Padded level count ``L_max`` of the per-species level blocks.
    broadening_mode : str
        Forward-model broadening mode (``'gaussian'`` / ``'doppler'`` /
        ``'resolving_power'`` ...). Static dispatch, so it keys the cache.
    apply_self_absorption : bool
        Whether the optical-depth correction branch is compiled in.
    max_iters : int
        Static upper bound on the iterative solve loop (``lax.while_loop``
        trip count cap). Distinct from the traced convergence budget
        ``PipelineParams.max_iterations``.
    batch_size : int
        Batch dimension ``B`` for :func:`cflibs.jitpipe.run_batch`. ``0`` for
        the unbatched :func:`run_one` path.
    closure_mode : str
        Closure strategy (``'standard'`` / ``'matrix'`` / ``'oxide'`` /
        ``'ilr'``). Selects a static code path.
    """

    bucket_id: int
    n_species: int
    level_pad: int
    broadening_mode: str = "gaussian"
    apply_self_absorption: bool = False
    max_iters: int = 20
    batch_size: int = 0
    closure_mode: str = "oxide"

    def __post_init__(self) -> None:
        # Fail fast if a caller tries to stuff a non-hashable in — the whole
        # point of this type is that it is a valid jit cache key.
        hash(self)


# Register PipelineParams as a pytree so every leaf is a traced value and
# changing a leaf reuses the compiled graph (J0 AC6). StaticConfig is NOT a
# pytree: it is the hashable cache key, passed as a static argument.
try:  # pragma: no cover - exercised whenever JAX is installed (the norm)
    import jax

    jax.tree_util.register_pytree_node(PipelineParams, _params_flatten, _params_unflatten)
except ImportError:  # pragma: no cover - jitpipe requires JAX; see __init__
    pass
