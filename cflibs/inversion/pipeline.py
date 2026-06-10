"""
Shared CF-LIBS analysis pipeline: configuration resolution and execution.

This module is the SINGLE source of truth for the detection -> selection ->
iterative-solve pipeline behind the ``cflibs analyze``/``invert``/``batch``
CLI commands AND the benchmark harnesses under ``scripts/`` (e.g.
``scripts/measure_bhvo2_presence.py``). It was extracted from
``cflibs/cli/main.py`` (bead vj82) because the BHVO-2 gate script used to
hand-build its own pipeline: two harnesses produced incomparable baselines
for the *same* knobs (CLI RMSE ~5.8 wt% vs script 4.03 wt% on ChemCam
BHVO-2), making every default-tuning decision ambiguous.

Anything that measures the CF-LIBS pipeline must build its configuration
through :func:`build_pipeline_config` and execute through
:func:`run_pipeline`. Knobs where harnesses have historically diverged
(per-spectrum wavelength calibration, the shift-coherence veto) are explicit
fields on :class:`AnalysisPipelineConfig` so any difference between two runs
is visible in the resolved config rather than buried in wiring.

The CLI keeps thin backward-compatible aliases (``_build_pipeline_config``,
``_detect_and_select_lines``, ``_run_pipeline``) in ``cflibs.cli.main``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from cflibs.core.logging_config import get_logger
from cflibs.inversion.physics.self_absorption_observable import normalize_self_absorption_mode

logger = get_logger("inversion.pipeline")


# ---------------------------------------------------------------------------
# Shared analyze/invert/batch pipeline configuration (bead l4a8)
# ---------------------------------------------------------------------------

#: Closure modes accepted by the iterative solver / CLI.
CLOSURE_MODES = ("standard", "matrix", "oxide", "ilr", "pwlr", "dirichlet_residual")

#: Preset bundles for the accuracy-critical solver knobs, measured on real
#: ChemCam BHVO-2 (docs/audit/2026-06-09-overhaul/04-pipeline-defaults.md):
#: the default ``analyze`` path scored RMSE 10.29 wt% (Fe 39 vs certified
#: 8.6) while ``--saha-boltzmann-graph --closure-mode oxide`` scored 4.03.
#: ``geological`` (the default) IS that measured-best configuration;
#: ``metallic`` keeps the pooled SB-graph but uses the standard closure
#: (oxide stoichiometry is wrong physics for alloys); ``raw`` reproduces the
#: legacy defaults for comparison runs.
ANALYSIS_PRESETS = {
    "geological": {"saha_boltzmann_graph": True, "closure_mode": "oxide", "stark_ne": True},
    "metallic": {"saha_boltzmann_graph": True, "closure_mode": "standard", "stark_ne": True},
    "raw": {"saha_boltzmann_graph": False, "closure_mode": "standard", "stark_ne": False},
}

DEFAULT_ANALYSIS_PRESET = "geological"


@dataclass
class AnalysisPipelineConfig:
    """Fully resolved configuration for the shared CF-LIBS analysis pipeline.

    One instance describes everything ``analyze``, ``invert`` and ``batch``
    feed into detection, selection and the iterative solver. Building it
    through :func:`build_pipeline_config` is what guarantees the three
    entry points cannot drift apart (the pre-fix ``batch`` wiring kept the
    exact raw-detection + bare-``LineSelector`` path whose drift caused the
    Na=98% blowup).
    """

    preset: str
    elements: list
    # Detection + selection knobs (mirror ``detect_and_select_lines``).
    min_relative_intensity: Optional[float] = None
    top_k_per_element: Optional[int] = 60
    resolving_power: Optional[float] = None
    wavelength_tolerance_nm: float = 0.1
    min_peak_height: float = 0.01
    peak_width_nm: float = 0.2
    #: Self-absorption mode: ``'off'`` or ``'observable'`` (bead 0jvr).
    #: Booleans are accepted on input and normalized (True -> 'observable').
    apply_self_absorption: str = "off"
    exclude_resonance: Optional[bool] = None
    min_snr: float = 10.0
    min_energy_spread_ev: float = 2.0
    min_lines_per_element: int = 3
    isolation_wavelength_nm: float = 0.1
    max_lines_per_element: int = 20
    wavelength_calibration: bool = True
    shift_coherence_veto: bool = True
    #: Optional path to a spectral-response curve E(lambda); identity when None
    #: (audit 02-F5 — ChemCam CCS data arrive response-corrected upstream).
    response_curve: Optional[str] = None
    # Iterative-solver knobs (mirror ``IterativeCFLIBSSolver``).
    max_iterations: int = 20
    t_tolerance_k: float = 100.0
    ne_tolerance_frac: float = 0.1
    pressure_pa: float = 101325.0
    boltzmann_weight_cap: float = 5.0
    min_boltzmann_r2: float = 0.3
    saha_boltzmann_graph: bool = True
    closure_mode: str = "oxide"
    closure_kwargs: dict = field(default_factory=dict)
    matrix_element: Optional[str] = None
    oxide_elements: Optional[list] = None
    #: Stark-broadening n_e diagnostic (bead pxex): measure n_e from observed
    #: literature-grade line widths; falls back (with warning) when none qualify.
    stark_ne: bool = True


def _first_not_none(*values):
    """Return the first value that is not ``None`` (all-None -> ``None``)."""
    for value in values:
        if value is not None:
            return value
    return None


def build_pipeline_config(
    elements,
    *,
    preset: Optional[str] = None,
    analysis_cfg: Optional[dict] = None,
    saha_boltzmann_graph: Optional[bool] = None,
    closure_mode: Optional[str] = None,
    apply_self_absorption: Optional["bool | str"] = None,
    min_relative_intensity: Optional[float] = None,
    resolving_power: Optional[float] = None,
    wavelength_tolerance_nm: Optional[float] = None,
    min_peak_height: Optional[float] = None,
    peak_width_nm: Optional[float] = None,
    exclude_resonance: Optional[bool] = None,
    wavelength_calibration: Optional[bool] = None,
    shift_coherence_veto: Optional[bool] = None,
    response_curve: Optional[str] = None,
    stark_ne: Optional[bool] = None,
) -> AnalysisPipelineConfig:
    """Resolve the shared pipeline configuration for analyze/invert/batch.

    Precedence per knob, highest first:

    1. explicit CLI flags (the keyword arguments; ``None`` = not given),
    2. YAML ``analysis.*`` keys (``analysis_cfg``, used by ``invert``),
    3. the resolved preset bundle (``--preset`` / ``analysis.preset``,
       default ``geological``),
    4. built-in defaults.

    The resolved preset and every knob are logged at INFO so each run
    records exactly which configuration produced its numbers.
    """
    cfg = dict(analysis_cfg or {})

    preset_name = _first_not_none(preset, cfg.get("preset"), DEFAULT_ANALYSIS_PRESET)
    if preset_name not in ANALYSIS_PRESETS:
        raise ValueError(
            f"Unknown analysis preset {preset_name!r}. "
            f"Valid presets: {sorted(ANALYSIS_PRESETS)}"
        )
    preset_knobs = ANALYSIS_PRESETS[preset_name]

    resolved_closure_mode = _first_not_none(
        closure_mode, cfg.get("closure_mode"), preset_knobs["closure_mode"]
    )
    if resolved_closure_mode not in CLOSURE_MODES:
        raise ValueError(
            f"Unknown closure mode {resolved_closure_mode!r}. "
            f"Valid modes: {list(CLOSURE_MODES)}"
        )

    pipeline = AnalysisPipelineConfig(
        preset=preset_name,
        elements=list(elements),
        min_relative_intensity=_first_not_none(
            min_relative_intensity, cfg.get("min_relative_intensity")
        ),
        top_k_per_element=cfg.get("top_k_per_element", 60),
        resolving_power=_first_not_none(resolving_power, cfg.get("resolving_power")),
        wavelength_tolerance_nm=_first_not_none(
            wavelength_tolerance_nm, cfg.get("wavelength_tolerance_nm"), 0.1
        ),
        min_peak_height=_first_not_none(min_peak_height, cfg.get("min_peak_height"), 0.01),
        peak_width_nm=_first_not_none(peak_width_nm, cfg.get("peak_width_nm"), 0.2),
        apply_self_absorption=normalize_self_absorption_mode(
            _first_not_none(apply_self_absorption, cfg.get("apply_self_absorption"), False)
        ),
        exclude_resonance=_first_not_none(exclude_resonance, cfg.get("exclude_resonance")),
        min_snr=cfg.get("min_snr", 10.0),
        min_energy_spread_ev=cfg.get("min_energy_spread_ev", 2.0),
        min_lines_per_element=cfg.get("min_lines_per_element", 3),
        isolation_wavelength_nm=cfg.get("isolation_wavelength_nm", 0.1),
        max_lines_per_element=cfg.get("max_lines_per_element", 20),
        wavelength_calibration=bool(
            _first_not_none(wavelength_calibration, cfg.get("wavelength_calibration"), True)
        ),
        shift_coherence_veto=bool(
            _first_not_none(shift_coherence_veto, cfg.get("shift_coherence_veto"), True)
        ),
        response_curve=_first_not_none(response_curve, cfg.get("response_curve")),
        max_iterations=cfg.get("max_iterations", 20),
        t_tolerance_k=cfg.get("t_tolerance_k", 100.0),
        ne_tolerance_frac=cfg.get("ne_tolerance_frac", 0.1),
        pressure_pa=cfg.get("pressure_pa", None) or cfg.get("pressure", 101325.0),
        boltzmann_weight_cap=cfg.get("boltzmann_weight_cap", 5.0),
        min_boltzmann_r2=cfg.get("min_boltzmann_r2", 0.3),
        saha_boltzmann_graph=bool(
            _first_not_none(
                saha_boltzmann_graph,
                cfg.get("saha_boltzmann_graph"),
                preset_knobs["saha_boltzmann_graph"],
            )
        ),
        closure_mode=resolved_closure_mode,
        closure_kwargs=dict(cfg.get("closure_kwargs", {})),
        matrix_element=cfg.get("matrix_element"),
        oxide_elements=cfg.get("oxide_elements"),
        stark_ne=bool(_first_not_none(stark_ne, cfg.get("stark_ne"), preset_knobs["stark_ne"])),
    )
    _log_pipeline_config(pipeline)
    return pipeline


def _log_pipeline_config(pipeline: AnalysisPipelineConfig) -> None:
    """Log the resolved preset and every pipeline knob at INFO."""
    logger.info(
        "Resolved analysis preset '%s': saha_boltzmann_graph=%s, closure_mode=%s, "
        "stark_ne=%s, "
        "apply_self_absorption=%s, exclude_resonance=%s, min_relative_intensity=%s, "
        "top_k_per_element=%s, resolving_power=%s, wavelength_calibration=%s, "
        "shift_coherence_veto=%s, "
        "wavelength_tolerance_nm=%s, min_peak_height=%s, peak_width_nm=%s, "
        "min_snr=%s, min_energy_spread_ev=%s, min_lines_per_element=%s, "
        "max_lines_per_element=%s, isolation_wavelength_nm=%s, max_iterations=%s, "
        "t_tolerance_k=%s, ne_tolerance_frac=%s, pressure_pa=%s, "
        "boltzmann_weight_cap=%s, min_boltzmann_r2=%s, response_curve=%s, elements=%s",
        pipeline.preset,
        pipeline.saha_boltzmann_graph,
        pipeline.closure_mode,
        pipeline.stark_ne,
        pipeline.apply_self_absorption,
        pipeline.exclude_resonance,
        pipeline.min_relative_intensity,
        pipeline.top_k_per_element,
        pipeline.resolving_power,
        pipeline.wavelength_calibration,
        pipeline.shift_coherence_veto,
        pipeline.wavelength_tolerance_nm,
        pipeline.min_peak_height,
        pipeline.peak_width_nm,
        pipeline.min_snr,
        pipeline.min_energy_spread_ev,
        pipeline.min_lines_per_element,
        pipeline.max_lines_per_element,
        pipeline.isolation_wavelength_nm,
        pipeline.max_iterations,
        pipeline.t_tolerance_k,
        pipeline.ne_tolerance_frac,
        pipeline.pressure_pa,
        pipeline.boltzmann_weight_cap,
        pipeline.min_boltzmann_r2,
        pipeline.response_curve,
        pipeline.elements,
    )


def detect_and_select_lines(
    wavelength,
    intensity,
    atomic_db,
    elements,
    *,
    min_relative_intensity: float | None = None,
    top_k_per_element: int | None = 60,
    resolving_power: float | None = None,
    wavelength_tolerance_nm: float = 0.1,
    min_peak_height: float = 0.01,
    peak_width_nm: float = 0.2,
    #: Self-absorption mode: ``'off'`` or ``'observable'`` (bead 0jvr).
    #: Booleans are accepted on input and normalized (True -> 'observable').
    apply_self_absorption: str = "off",
    exclude_resonance: bool | None = None,
    min_snr: float = 10.0,
    min_energy_spread_ev: float = 2.0,
    min_lines_per_element: int = 3,
    isolation_wavelength_nm: float = 0.1,
    max_lines_per_element: int = 20,
    wavelength_calibration: bool = True,
    shift_coherence_veto: bool = True,
    residual_shift_scan_nm: float = 0.05,
    return_diagnostics: bool = False,
):
    """
    Detect spectral lines and apply the line-selection quality gate.

    This is the single shared detection+selection path for both the
    ``invert`` and ``analyze`` CLI entry points. Keeping it in one helper
    prevents the two paths from drifting: a prior drift left ``analyze`` with
    no relative-intensity floor and a bare ``LineSelector()``, which admitted
    spurious weak Na Rydberg lines and produced a catastrophic Na-dominated
    composition (RMSE 33.69 wt%, Na ~98 % on BHVO-2). Both paths now resolve
    to identical detection-floor + selector behaviour for the same inputs.

    The defaults here are the *good* defaults for the ``invert``/``analyze``
    paths: no absolute relative-intensity floor (replaced by element-relative
    top-K + a shift-coherence veto in ``detect_line_observations``), the tuned
    ``LineSelector`` gates, self-absorption off. ``exclude_resonance`` now
    defaults (``None``) to ``False`` — resonance lines are the brightest,
    most persistent LIBS lines and the only detectable lines for some majors
    (Al I 394.4/396.2 nm), so they are kept; the gA-Boltzmann comb strength and
    coherence veto guard against the spurious weak-line matches that resonance
    exclusion + the rel_int floor used to protect against. With
    self-absorption on the solver *corrects* resonance lines (Aragón &
    Aguilera 2008 §7).

    Parameters
    ----------
    wavelength, intensity : np.ndarray
        Spectrum axes (nm, intensity).
    atomic_db : AtomicDatabase
        Atomic database instance.
    elements : list[str]
        Elements to match against.
    min_relative_intensity : float or None
        Absolute relative-intensity floor for database lines. Default ``None``
        (disabled): the floor deletes whole real elements whose tabulated
        rel_int is small or NULL, so detection now bounds richness with
        ``top_k_per_element`` instead. Pass a number to re-enable the legacy
        floor.
    top_k_per_element : int or None
        Keep only each element's ``K`` strongest in-band lines (by gA-Boltzmann
        strength) instead of an absolute rel_int floor (default 60). ``None``
        keeps all in-band lines.
    resolving_power : float or None
        Instrument resolving power; enables adaptive tolerance/width when set.
    apply_self_absorption : bool
        Whether the downstream solver applies the curve-of-growth correction.
        Controls the default ``exclude_resonance`` tie.
    exclude_resonance : bool or None
        Override the resonance-exclusion policy. ``None`` ties it to
        ``not apply_self_absorption``.
    wavelength_calibration : bool
        If True (default), run robust per-spectrum/per-segment wavelength
        calibration (:func:`calibrate_wavelength_axis_segmented`) before
        detection. This replaces the single-constant global comb shift-scan's
        job for instruments with a per-spectrum or piecewise dispersion error;
        the global scan is reduced to a small residual mop-up. When calibration
        cannot be confidently estimated, it is skipped and the full global
        shift-scan is used (legacy behaviour preserved).
    shift_coherence_veto : bool
        If True (default), reject accepted elements whose matched lines do not
        agree on the single instrument residual shift (see
        ``detect_line_observations``). Exposed as a pipeline knob (bead vj82)
        so benchmark harnesses can ablate it explicitly instead of forking the
        pipeline wiring.
    residual_shift_scan_nm : float
        Half-width (nm) of the residual comb shift-scan applied *after* a
        successful wavelength calibration. Small by design (default 0.05 nm):
        the calibration has already aligned the axis, so only sub-tolerance
        jitter remains.
    return_diagnostics : bool
        When True, also return a diagnostics dict recording which requested
        elements were dropped and at which stage (``detection`` = no matched
        lines; ``selection`` = matched lines failed the quality gate). This
        feeds the CLI trust report: requested-but-dropped elements used to
        vanish silently while the run still printed ``converged=True``.

    Returns
    -------
    list or (list, dict)
        The selected ``LineObservation`` list (``selection.selected_lines``);
        with ``return_diagnostics=True``, a ``(selected_lines, diagnostics)``
        tuple.
    """
    import numpy as np

    from cflibs.inversion.line_detection import detect_line_observations
    from cflibs.inversion.line_selection import LineSelector
    from cflibs.inversion.preprocess.wavelength_calibration import (
        calibrate_wavelength_axis_segmented,
    )

    # Robust per-spectrum wavelength calibration BEFORE detection. Real LIBS
    # instruments carry a per-spectrum (and, for stitched multi-channel
    # spectrometers, per-channel) dispersion error that the single global comb
    # shift-scan inside ``detect_line_observations`` cannot represent: it can
    # only apply one constant offset, so on a piecewise-dispersion instrument it
    # compromises and leaves real majors mis-aligned beyond tolerance (measured
    # on BHVO-2: ChemCam Al 396.152 misses by 0.109 nm; CSA abandons Si vs Mg).
    # ``calibrate_wavelength_axis_segmented`` detects detector seams and fits each
    # channel independently (degrading to a single global affine when there are
    # no seams), aligning the axis robustly. After calibration only a small
    # residual ``shift_scan_nm`` is needed to mop up sub-tolerance jitter.
    shift_scan_nm = 0.5  # legacy global scan (used when calibration disabled)
    if wavelength_calibration:
        try:
            cal = calibrate_wavelength_axis_segmented(
                wavelength=np.asarray(wavelength, dtype=float),
                intensity=np.asarray(intensity, dtype=float),
                atomic_db=atomic_db,
                elements=elements,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Wavelength calibration failed ({exc!r}); using raw axis.")
            cal = None
        if cal is not None and cal.success and cal.quality_passed:
            wavelength = cal.corrected_wavelength
            shift_scan_nm = residual_shift_scan_nm
            logger.info(
                "Applied wavelength calibration: model=%s segments=%s "
                "n_inliers=%d rmse=%.4f nm (residual shift-scan +/-%.3f nm)",
                cal.model,
                cal.details.get("segments", 1),
                cal.n_inliers,
                cal.rmse_nm,
                residual_shift_scan_nm,
            )
        else:
            reason = cal.quality_reason if cal is not None else "exception"
            logger.info(
                "Wavelength calibration not applied (%s); "
                "falling back to global comb shift-scan.",
                reason,
            )

    detection = detect_line_observations(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=elements,
        wavelength_tolerance_nm=wavelength_tolerance_nm,
        resolving_power=resolving_power,
        min_peak_height=min_peak_height,
        peak_width_nm=peak_width_nm,
        min_relative_intensity=min_relative_intensity,
        top_k_per_element=top_k_per_element,
        shift_scan_nm=shift_scan_nm,
        shift_coherence_veto=shift_coherence_veto,
    )

    for warning in detection.warnings:
        logger.warning(f"Line detection warning: {warning}")

    # Resonance lines are the brightest, most persistent lines in a LIBS
    # plasma (low E_k, large gA) and are the *only* detectable lines for some
    # majors (e.g. the Al I 394.4/396.2 nm doublet). Keep them by default so
    # those elements reach the solver. The legacy tie to
    # ``not apply_self_absorption`` dropped them whenever self-absorption
    # correction was off, deleting whole elements at the selection gate; the
    # gA-Boltzmann comb strength + shift-coherence veto now guard against the
    # spurious weak-line matches that exclusion was protecting against.
    if exclude_resonance is None:
        exclude_resonance = False

    selector = LineSelector(
        min_snr=min_snr,
        min_energy_spread_ev=min_energy_spread_ev,
        min_lines_per_element=min_lines_per_element,
        exclude_resonance=exclude_resonance,
        isolation_wavelength_nm=isolation_wavelength_nm,
        max_lines_per_element=max_lines_per_element,
    )

    selection = selector.select(
        detection.observations,
        resonance_lines=detection.resonance_lines,
    )
    if not return_diagnostics:
        return selection.selected_lines

    detected_elements = {o.element for o in detection.observations}
    selected_elements = {o.element for o in selection.selected_lines}
    dropped: dict = {}
    for el in elements:
        if el not in detected_elements:
            dropped[el] = "detection"
        elif el not in selected_elements:
            dropped[el] = "selection"
    diagnostics = {
        "requested_elements": list(elements),
        "detected_elements": sorted(detected_elements),
        "selected_elements": sorted(selected_elements),
        "dropped_elements": dropped,
    }
    return selection.selected_lines, diagnostics


def _finalize_closure_kwargs(pipeline: AnalysisPipelineConfig, observations) -> dict:
    """Build the closure kwargs for a solve, applying matrix/oxide defaults."""
    closure_kwargs = dict(pipeline.closure_kwargs)
    if pipeline.closure_mode == "matrix" and pipeline.matrix_element is not None:
        closure_kwargs.setdefault("matrix_element", pipeline.matrix_element)
    if pipeline.closure_mode == "oxide":
        if pipeline.oxide_elements is not None:
            closure_kwargs.setdefault("oxide_elements", pipeline.oxide_elements)
        # Default geological molar-oxygen stoichiometry (O atoms per cation) when
        # the config does not supply an explicit oxide_stoichiometry map.
        if "oxide_stoichiometry" not in closure_kwargs:
            from cflibs.inversion.physics.closure import default_oxide_stoichiometry

            els = [o.element for o in observations]
            closure_kwargs["oxide_stoichiometry"] = default_oxide_stoichiometry(els)
    return closure_kwargs


def _solve_analyze_result(
    solver,
    observations,
    closure_mode: str,
    closure_kwargs: dict,
    uncertainty_mode: str,
    stark_diagnostics=None,
):
    """Run the solver for ``analyze`` honouring the requested uncertainty mode."""
    if uncertainty_mode == "analytical":
        try:
            return solver.solve_with_uncertainty(
                observations,
                closure_mode=closure_mode,
                stark_diagnostics=stark_diagnostics,
                **closure_kwargs,
            )
        except ImportError:
            logger.warning(
                "uncertainties package not installed; falling back to solve() without UQ. "
                "Install with: pip install uncertainties"
            )
            return solver.solve(
                observations,
                closure_mode=closure_mode,
                stark_diagnostics=stark_diagnostics,
                **closure_kwargs,
            )
    elif uncertainty_mode == "mc":
        from cflibs.inversion.physics.uncertainty import MonteCarloUQ

        mc = MonteCarloUQ(solver, n_samples=200)
        mc_result = mc.run(observations)
        result = solver.solve(
            observations,
            closure_mode=closure_mode,
            stark_diagnostics=stark_diagnostics,
            **closure_kwargs,
        )
        return result.__class__(
            temperature_K=mc_result.T_mean,
            temperature_uncertainty_K=mc_result.T_std,
            electron_density_cm3=mc_result.ne_mean,
            concentrations=mc_result.concentrations_mean,
            concentration_uncertainties=mc_result.concentrations_std,
            iterations=result.iterations,
            converged=result.converged,
            quality_metrics=result.quality_metrics,
        )
    else:
        return solver.solve(
            observations,
            closure_mode=closure_mode,
            stark_diagnostics=stark_diagnostics,
            **closure_kwargs,
        )


def run_pipeline(
    wavelength,
    intensity,
    atomic_db,
    pipeline: AnalysisPipelineConfig,
    uncertainty_mode: str = "none",
):
    """Run the shared detection -> selection -> iterative-solve pipeline.

    The single execution path behind ``analyze``, ``invert``, ``batch`` and
    the benchmark harnesses. Returns ``(result, diagnostics)`` where
    ``diagnostics`` carries the resolved preset, the requested-but-dropped
    element map for the trust report, and per-element observation counts
    (``observation_counts``) for benchmark scoring.
    """
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

    # Spectral-response correction FIRST (audit 02-F5): divide the measured
    # spectrum by the relative detection efficiency E(lambda) before any
    # observation building, so integrated line intensities, their shot-noise
    # uncertainties and the Boltzmann ordinates are all computed from
    # response-corrected data. Identity (no-op) when no curve is configured —
    # ChemCam CCS spectra arrive response-corrected upstream and must NOT be
    # corrected twice. Path resolution (config-relative etc.) happens at the
    # CLI layer; here the path is used as given.
    if pipeline.response_curve:
        from cflibs.inversion.preprocess.response_correction import SpectralResponseCorrection

        correction = SpectralResponseCorrection.from_file(pipeline.response_curve)
        intensity = correction.apply(wavelength, intensity)
        logger.info(
            "Applied spectral-response correction from %s (coverage %.1f-%.1f nm).",
            pipeline.response_curve,
            *correction.coverage_nm,
        )

    observations, diagnostics = detect_and_select_lines(
        wavelength,
        intensity,
        atomic_db,
        pipeline.elements,
        min_relative_intensity=pipeline.min_relative_intensity,
        top_k_per_element=pipeline.top_k_per_element,
        resolving_power=pipeline.resolving_power,
        wavelength_tolerance_nm=pipeline.wavelength_tolerance_nm,
        min_peak_height=pipeline.min_peak_height,
        peak_width_nm=pipeline.peak_width_nm,
        apply_self_absorption=pipeline.apply_self_absorption,
        exclude_resonance=pipeline.exclude_resonance,
        min_snr=pipeline.min_snr,
        min_energy_spread_ev=pipeline.min_energy_spread_ev,
        min_lines_per_element=pipeline.min_lines_per_element,
        isolation_wavelength_nm=pipeline.isolation_wavelength_nm,
        max_lines_per_element=pipeline.max_lines_per_element,
        wavelength_calibration=pipeline.wavelength_calibration,
        shift_coherence_veto=pipeline.shift_coherence_veto,
        return_diagnostics=True,
    )
    diagnostics["preset"] = pipeline.preset
    diagnostics["saha_boltzmann_graph"] = pipeline.saha_boltzmann_graph
    diagnostics["closure_mode"] = pipeline.closure_mode
    diagnostics["response_curve"] = pipeline.response_curve

    # Per-element observation counts (which elements survived detection +
    # selection, and with how many lines). Benchmark harnesses score these;
    # the CLI trust report ignores them.
    observation_counts: dict = {}
    for obs in observations:
        element = getattr(obs, "element", None)
        if element is not None:
            observation_counts[element] = observation_counts.get(element, 0) + 1
    diagnostics["observation_counts"] = observation_counts
    diagnostics["n_observations"] = len(observations)

    if len(observations) == 0:
        raise ValueError("No usable spectral lines detected for inversion.")

    # Stark-broadening n_e diagnostic (bead pxex; audit 02-F2): measure n_e
    # from the widths of observed literature-grade lines so it enters the Saha
    # terms as a measurement, not the 1-atm pressure-balance assumption. When
    # no line qualifies, stark_diagnostics stays None and the solver keeps the
    # existing (warned) fallback path — behaviour unchanged.
    stark_diagnostics = None
    if pipeline.stark_ne:
        from cflibs.inversion.physics.stark_ne import measure_stark_ne

        try:
            stark_result = measure_stark_ne(
                wavelength,
                intensity,
                observations,
                atomic_db,
                resolving_power=pipeline.resolving_power,
            )
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"Stark n_e diagnostic failed ({exc!r}); using solver fallback n_e.")
            stark_result = None
        if stark_result is not None:
            diagnostics["stark_ne"] = {
                "n_lines": stark_result.n_lines,
                "ne_cm3": stark_result.ne_median_cm3,
                "ne_scatter_cm3": stark_result.ne_scatter_cm3,
                "instrument_fwhm_source": stark_result.instrument_fwhm_source,
                "lines": [
                    {
                        "element": m.element,
                        "ionization_stage": m.ionization_stage,
                        "wavelength_nm": m.wavelength_nm,
                        "lorentz_fwhm_nm": m.lorentz_fwhm_nm,
                        "ne_cm3": m.ne_cm3,
                    }
                    for m in stark_result.measurements
                ],
                "rejected": dict(stark_result.rejected),
            }
            if stark_result.usable:
                stark_diagnostics = stark_result.diagnostics

    solver = IterativeCFLIBSSolver(
        atomic_db=atomic_db,
        max_iterations=pipeline.max_iterations,
        t_tolerance_k=pipeline.t_tolerance_k,
        ne_tolerance_frac=pipeline.ne_tolerance_frac,
        pressure_pa=pipeline.pressure_pa,
        apply_self_absorption=pipeline.apply_self_absorption,
        min_boltzmann_r2=pipeline.min_boltzmann_r2,
        boltzmann_weight_cap=pipeline.boltzmann_weight_cap,
        saha_boltzmann_graph=pipeline.saha_boltzmann_graph,
    )

    closure_kwargs = _finalize_closure_kwargs(pipeline, observations)
    result = _solve_analyze_result(
        solver,
        observations,
        pipeline.closure_mode,
        closure_kwargs,
        uncertainty_mode,
        stark_diagnostics=stark_diagnostics,
    )

    # Solver-stage drops: requested elements that survived detection and
    # selection but ended the solve with no mass attributed.
    dropped = diagnostics["dropped_elements"]
    for el in pipeline.elements:
        if el not in dropped and result.concentrations.get(el, 0.0) <= 0.0:
            dropped[el] = "solve"
    return result, diagnostics
