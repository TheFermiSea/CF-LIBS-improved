"""
Main CLI entry point for CF-LIBS.
"""

import argparse
import importlib.util
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cflibs.core.logging_config import setup_logging, get_logger

logger = get_logger("cli.main")


def _repo_root() -> Path:
    """Return the repository root when running from a source checkout."""
    return Path(__file__).resolve().parents[2]


def _resolve_existing_path(
    path_value: str | Path, *, relative_to: str | Path | None = None
) -> Path:
    """
    Resolve a user-provided path using beginner-friendly fallbacks.

    Config examples often use short relative paths. Check the current working
    directory first, then the config file directory, then the source checkout.
    """
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path

    candidates = [Path.cwd() / path]
    if relative_to is not None:
        candidates.append(Path(relative_to).resolve().parent / path)

    root = _repo_root()
    candidates.extend(
        [
            root / path,
            root / "ASD_da" / path.name,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _resolve_db_path(
    path_value: str | Path | None = None, *, relative_to: str | Path | None = None
) -> Path:
    """Resolve the atomic database path, preferring the bundled sample database if present."""
    return _resolve_existing_path(path_value or "libs_production.db", relative_to=relative_to)


def _missing_db_message(db_path: Path) -> str:
    return (
        f"Atomic database not found: {db_path}. "
        "If you are using the source checkout, try ASD_da/libs_production.db. "
        "To build a new database, run: cflibs generate-db --db-path libs_production.db"
    )


def _float_config_value(config: dict, key: str, section: str) -> float:
    """Read numeric YAML values, including scientific notation parsed as strings."""
    try:
        return float(config[key])
    except KeyError as exc:
        raise ValueError(f"{section} config missing required field: {key}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{section}.{key} must be numeric; got {config.get(key)!r}") from exc


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
    "geological": {"saha_boltzmann_graph": True, "closure_mode": "oxide"},
    "metallic": {"saha_boltzmann_graph": True, "closure_mode": "standard"},
    "raw": {"saha_boltzmann_graph": False, "closure_mode": "standard"},
}

DEFAULT_ANALYSIS_PRESET = "geological"


@dataclass
class AnalysisPipelineConfig:
    """Fully resolved configuration for the shared CF-LIBS analysis pipeline.

    One instance describes everything ``analyze``, ``invert`` and ``batch``
    feed into detection, selection and the iterative solver. Building it
    through :func:`_build_pipeline_config` is what guarantees the three
    entry points cannot drift apart (the pre-fix ``batch`` wiring kept the
    exact raw-detection + bare-``LineSelector`` path whose drift caused the
    Na=98% blowup).
    """

    preset: str
    elements: list
    # Detection + selection knobs (mirror ``_detect_and_select_lines``).
    min_relative_intensity: Optional[float] = None
    top_k_per_element: Optional[int] = 60
    resolving_power: Optional[float] = None
    wavelength_tolerance_nm: float = 0.1
    min_peak_height: float = 0.01
    peak_width_nm: float = 0.2
    apply_self_absorption: bool = False
    exclude_resonance: Optional[bool] = None
    min_snr: float = 10.0
    min_energy_spread_ev: float = 2.0
    min_lines_per_element: int = 3
    isolation_wavelength_nm: float = 0.1
    max_lines_per_element: int = 20
    wavelength_calibration: bool = True
    # Iterative-solver knobs (mirror ``IterativeCFLIBSSolver``).
    max_iterations: int = 20
    t_tolerance_k: float = 100.0
    ne_tolerance_frac: float = 0.1
    pressure_pa: float = 101325.0
    self_absorption_column_density_cm3: float = 1.0e16
    self_absorption_plasma_length_cm: float = 0.1
    boltzmann_weight_cap: float = 5.0
    min_boltzmann_r2: float = 0.3
    saha_boltzmann_graph: bool = True
    closure_mode: str = "oxide"
    closure_kwargs: dict = field(default_factory=dict)
    matrix_element: Optional[str] = None
    oxide_elements: Optional[list] = None


def _first_not_none(*values):
    """Return the first value that is not ``None`` (all-None -> ``None``)."""
    for value in values:
        if value is not None:
            return value
    return None


def _build_pipeline_config(
    elements,
    *,
    preset: Optional[str] = None,
    analysis_cfg: Optional[dict] = None,
    saha_boltzmann_graph: Optional[bool] = None,
    closure_mode: Optional[str] = None,
    apply_self_absorption: Optional[bool] = None,
    min_relative_intensity: Optional[float] = None,
    resolving_power: Optional[float] = None,
    wavelength_tolerance_nm: Optional[float] = None,
    min_peak_height: Optional[float] = None,
    peak_width_nm: Optional[float] = None,
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
        apply_self_absorption=bool(
            _first_not_none(apply_self_absorption, cfg.get("apply_self_absorption"), False)
        ),
        exclude_resonance=cfg.get("exclude_resonance"),
        min_snr=cfg.get("min_snr", 10.0),
        min_energy_spread_ev=cfg.get("min_energy_spread_ev", 2.0),
        min_lines_per_element=cfg.get("min_lines_per_element", 3),
        isolation_wavelength_nm=cfg.get("isolation_wavelength_nm", 0.1),
        max_lines_per_element=cfg.get("max_lines_per_element", 20),
        wavelength_calibration=bool(cfg.get("wavelength_calibration", True)),
        max_iterations=cfg.get("max_iterations", 20),
        t_tolerance_k=cfg.get("t_tolerance_k", 100.0),
        ne_tolerance_frac=cfg.get("ne_tolerance_frac", 0.1),
        pressure_pa=cfg.get("pressure_pa", None) or cfg.get("pressure", 101325.0),
        self_absorption_column_density_cm3=cfg.get("self_absorption_column_density_cm3", 1.0e16),
        self_absorption_plasma_length_cm=cfg.get("self_absorption_plasma_length_cm", 0.1),
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
    )
    _log_pipeline_config(pipeline)
    return pipeline


def _log_pipeline_config(pipeline: AnalysisPipelineConfig) -> None:
    """Log the resolved preset and every pipeline knob at INFO."""
    logger.info(
        "Resolved analysis preset '%s': saha_boltzmann_graph=%s, closure_mode=%s, "
        "apply_self_absorption=%s, exclude_resonance=%s, min_relative_intensity=%s, "
        "top_k_per_element=%s, resolving_power=%s, wavelength_calibration=%s, "
        "wavelength_tolerance_nm=%s, min_peak_height=%s, peak_width_nm=%s, "
        "min_snr=%s, min_energy_spread_ev=%s, min_lines_per_element=%s, "
        "max_lines_per_element=%s, isolation_wavelength_nm=%s, max_iterations=%s, "
        "t_tolerance_k=%s, ne_tolerance_frac=%s, pressure_pa=%s, "
        "boltzmann_weight_cap=%s, min_boltzmann_r2=%s, elements=%s",
        pipeline.preset,
        pipeline.saha_boltzmann_graph,
        pipeline.closure_mode,
        pipeline.apply_self_absorption,
        pipeline.exclude_resonance,
        pipeline.min_relative_intensity,
        pipeline.top_k_per_element,
        pipeline.resolving_power,
        pipeline.wavelength_calibration,
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
        pipeline.elements,
    )


def forward_model_cmd(args):
    """Forward modeling command."""
    from cflibs.core.config import load_config, validate_plasma_config, validate_instrument_config
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.plasma.state import SingleZoneLTEPlasma
    from cflibs.instrument.model import InstrumentModel
    from cflibs.radiation.spectrum_model import SpectrumModel
    import numpy as np

    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)

    # Validate configuration
    validate_plasma_config(config)
    validate_instrument_config(config)

    # Load atomic database
    db_path = _resolve_db_path(config.get("atomic_database"), relative_to=args.config)
    if not db_path.exists():
        raise FileNotFoundError(_missing_db_message(db_path))
    atomic_db = AtomicDatabase(db_path)

    # Create plasma state
    plasma_config = config["plasma"]
    plasma = SingleZoneLTEPlasma(
        T_e=_float_config_value(plasma_config, "Te", "plasma"),
        n_e=_float_config_value(plasma_config, "ne", "plasma"),
        species={
            s["element"]: _float_config_value(s, "number_density", f"species {s['element']}")
            for s in plasma_config["species"]
        },
        T_g=float(plasma_config["Tg"]) if plasma_config.get("Tg") is not None else None,
        pressure=(
            float(plasma_config["pressure"]) if plasma_config.get("pressure") is not None else None
        ),
    )

    # Create instrument model
    instrument = InstrumentModel.from_file(Path(args.config))

    # Get spectrum parameters
    spectrum_config = config.get("spectrum", {})
    lambda_min = float(spectrum_config.get("lambda_min_nm", 200.0))
    lambda_max = float(spectrum_config.get("lambda_max_nm", 800.0))
    delta_lambda = float(spectrum_config.get("delta_lambda_nm", 0.01))
    path_length = float(spectrum_config.get("path_length_m", 0.01))

    # Create spectrum model
    model = SpectrumModel(
        plasma=plasma,
        atomic_db=atomic_db,
        instrument=instrument,
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        delta_lambda=delta_lambda,
        path_length_m=path_length,
    )

    # Compute spectrum
    logger.info("Computing spectrum...")
    wavelength, intensity = model.compute_spectrum()

    # Output results
    if args.output:
        output_path = Path(args.output)
        logger.info(f"Saving spectrum to {output_path}")
        # Save as CSV
        np.savetxt(
            output_path,
            np.column_stack([wavelength, intensity]),
            delimiter=",",
            header="wavelength_nm,intensity_W_m2_nm_sr",
            comments="",
        )
        print(f"Spectrum saved to {output_path}")
    else:
        # Print to stdout
        print("# Wavelength (nm), Intensity (W m^-2 nm^-1 sr^-1)")
        for wl, intensity in zip(wavelength[::10], intensity[::10]):  # Sample every 10th point
            print(f"{wl:.3f},{intensity:.6e}")

    logger.info("Forward modeling complete")


def _detect_and_select_lines(
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
    apply_self_absorption: bool = False,
    exclude_resonance: bool | None = None,
    min_snr: float = 10.0,
    min_energy_spread_ev: float = 2.0,
    min_lines_per_element: int = 3,
    isolation_wavelength_nm: float = 0.1,
    max_lines_per_element: int = 20,
    wavelength_calibration: bool = True,
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


def _resolve_invert_elements(args, config: dict, analysis_cfg: dict):
    """Resolve the element list for ``invert`` from CLI args then config fallbacks."""
    cli_elements = getattr(args, "elements", None)
    config_elements = config.get("elements") if isinstance(config, dict) else None
    elements = cli_elements or analysis_cfg.get("elements") or config_elements
    if elements is None:
        raise ValueError(
            "Elements must be specified via --elements, config 'analysis.elements', "
            "or config 'elements'."
        )
    if isinstance(elements, str):
        elements = [elements]
    return elements


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


def _run_pipeline(
    wavelength,
    intensity,
    atomic_db,
    pipeline: AnalysisPipelineConfig,
    uncertainty_mode: str = "none",
):
    """Run the shared detection -> selection -> iterative-solve pipeline.

    The single execution path behind ``analyze``, ``invert`` and ``batch``.
    Returns ``(result, diagnostics)`` where ``diagnostics`` carries the
    resolved preset and the requested-but-dropped element map for the trust
    report.
    """
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver

    observations, diagnostics = _detect_and_select_lines(
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
        return_diagnostics=True,
    )
    diagnostics["preset"] = pipeline.preset
    diagnostics["saha_boltzmann_graph"] = pipeline.saha_boltzmann_graph
    diagnostics["closure_mode"] = pipeline.closure_mode

    if len(observations) == 0:
        raise ValueError("No usable spectral lines detected for inversion.")

    solver = IterativeCFLIBSSolver(
        atomic_db=atomic_db,
        max_iterations=pipeline.max_iterations,
        t_tolerance_k=pipeline.t_tolerance_k,
        ne_tolerance_frac=pipeline.ne_tolerance_frac,
        pressure_pa=pipeline.pressure_pa,
        apply_self_absorption=pipeline.apply_self_absorption,
        self_absorption_column_density_cm3=pipeline.self_absorption_column_density_cm3,
        self_absorption_plasma_length_cm=pipeline.self_absorption_plasma_length_cm,
        min_boltzmann_r2=pipeline.min_boltzmann_r2,
        boltzmann_weight_cap=pipeline.boltzmann_weight_cap,
        saha_boltzmann_graph=pipeline.saha_boltzmann_graph,
    )

    closure_kwargs = _finalize_closure_kwargs(pipeline, observations)
    result = _solve_analyze_result(
        solver, observations, pipeline.closure_mode, closure_kwargs, uncertainty_mode
    )

    # Solver-stage drops: requested elements that survived detection and
    # selection but ended the solve with no mass attributed.
    dropped = diagnostics["dropped_elements"]
    for el in pipeline.elements:
        if el not in dropped and result.concentrations.get(el, 0.0) <= 0.0:
            dropped[el] = "solve"
    return result, diagnostics


def _ne_source_label(quality_metrics: dict) -> Optional[str]:
    """Map the ``ne_from_stark`` provenance flag to a human-readable source."""
    ne_from_stark = quality_metrics.get("ne_from_stark")
    if ne_from_stark is None:
        return None
    return "stark" if ne_from_stark else "pressure_balance_fallback"


def _trust_report(result, diagnostics: Optional[dict] = None) -> tuple[list, list]:
    """Build ``(info_lines, warning_lines)`` for the CLI trust/quality report.

    Surfaces what the solver already knows but stdout never showed: the
    convergence verdict, the Boltzmann-plane R^2, the n_e provenance (a
    Stark-width measurement vs the physically-non-standard 1-atm
    pressure-balance fallback), the degeneracy gates, and any requested
    elements that were dropped before the fit.
    """
    qm = result.quality_metrics or {}
    info: list = []
    warnings_out: list = []

    if diagnostics and diagnostics.get("preset"):
        info.append(
            f"Preset      : {diagnostics['preset']} "
            f"(saha_boltzmann_graph={diagnostics.get('saha_boltzmann_graph')}, "
            f"closure_mode={diagnostics.get('closure_mode')})"
        )
    r2 = qm.get("boltzmann_r_squared", qm.get("r_squared_last"))
    if r2 is not None:
        info.append(f"Boltzmann R2: {r2:.3f}")
    n_fit = qm.get("n_elements_fit")
    if n_fit is not None:
        info.append(f"Elements fit: {int(n_fit)}")

    ne_source = _ne_source_label(qm)
    if ne_source == "stark":
        info.append("n_e source  : Stark-width diagnostic (measured)")
    elif ne_source == "pressure_balance_fallback":
        info.append("n_e source  : 1-atm pressure-balance fallback (ASSUMED)")
        warnings_out.append(
            "WARNING: n_e was ASSUMED from the 1-atm pressure-balance fallback, not "
            "measured (no Stark-width diagnostic available). Treat n_e as a coarse "
            "order-of-magnitude estimate."
        )

    if not result.converged:
        warnings_out.append(f"WARNING: solver did NOT converge ({result.iterations} iterations).")
    if qm.get("boltzmann_degenerate"):
        warnings_out.append(
            "WARNING: Boltzmann fit DEGENERATE (non-physical slope or R^2 below the "
            "quality gate); the temperature is unconstrained."
        )
    if qm.get("closure_degenerate"):
        warnings_out.append(
            "WARNING: closure DEGENERATE (one element soaks >80% of the closure "
            "mass); the composition is untrustworthy."
        )
    lte_ok = qm.get("lte_mcwhirter_satisfied")
    if lte_ok is not None and not lte_ok:
        warnings_out.append(
            f"WARNING: McWhirter criterion NOT satisfied "
            f"(n_e ratio = {qm.get('lte_n_e_ratio', 0):.2f})"
        )

    dropped = (diagnostics or {}).get("dropped_elements") or {}
    if dropped:
        detail = ", ".join(f"{el} ({stage})" for el, stage in sorted(dropped.items()))
        warnings_out.append(f"WARNING: requested elements dropped: {detail}")

    if not result.converged or qm.get("boltzmann_degenerate") or qm.get("closure_degenerate"):
        warnings_out.append("RESULT UNRELIABLE: see warnings above.")
    return info, warnings_out


def _json_default(obj):
    """``json.dumps`` fallback: unwrap numpy scalars (bool_/float64/int64).

    The solver's quality gates produce numpy scalars (e.g. ``converged`` is a
    numpy bool on the Saha-Boltzmann-graph path, now the default), which the
    stdlib json encoder rejects.
    """
    item = getattr(obj, "item", None)
    if callable(item):
        return item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _trust_json(result, diagnostics: Optional[dict] = None) -> dict:
    """Machine-readable trust block for JSON output paths."""
    qm = result.quality_metrics or {}
    _, warning_lines = _trust_report(result, diagnostics)
    r2 = qm.get("boltzmann_r_squared", qm.get("r_squared_last"))
    n_fit = qm.get("n_elements_fit")
    return {
        "converged": bool(result.converged),
        "boltzmann_r_squared": float(r2) if r2 is not None else None,
        "n_elements_fit": int(n_fit) if n_fit is not None else None,
        "ne_source": _ne_source_label(qm),
        "boltzmann_degenerate": bool(qm.get("boltzmann_degenerate", 0.0)),
        "closure_degenerate": bool(qm.get("closure_degenerate", 0.0)),
        "dropped_elements": (diagnostics or {}).get("dropped_elements", {}),
        "warnings": warning_lines,
    }


def _output_invert_result(result, args, diagnostics: Optional[dict] = None) -> None:
    """Export or print the ``invert`` result depending on ``args.output``."""
    from cflibs.io.exporters import create_exporter

    info_lines, warning_lines = _trust_report(result, diagnostics)

    if args.output:
        output_path = Path(args.output)
        ext = output_path.suffix.lower().lstrip(".")
        if ext not in {"csv", "json", "h5", "hdf5"}:
            ext = "json"
            output_path = output_path.with_suffix(".json")

        exporter = create_exporter(ext)
        exporter.export(result, str(output_path))
        print(f"Inversion results saved to {output_path}")
        for line in warning_lines:
            print(line)
        return

    print("CF-LIBS inversion results:")
    print(f"  Temperature: {result.temperature_K:.0f} ± {result.temperature_uncertainty_K:.0f} K")
    print(f"  Electron density: {result.electron_density_cm3:.3e} cm^-3")
    print(f"  Converged: {result.converged} ({result.iterations} iterations)")
    for line in info_lines:
        print(f"  {line}")
    print("  Concentrations:")
    for element, concentration in result.concentrations.items():
        unc = result.concentration_uncertainties.get(element, 0.0)
        print(f"    {element}: {concentration:.4f} ± {unc:.4f}")
    for line in warning_lines:
        print(line)


def invert_cmd(args):
    """
    Inversion command.

    Runs classic CF-LIBS inversion using detected spectral lines, through the
    same shared pipeline as ``analyze``/``batch`` (see
    :func:`_build_pipeline_config` / :func:`_run_pipeline`). YAML
    ``analysis.*`` keys and CLI flags resolve through the same precedence
    rules, so the flag and config paths cannot drift apart.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.core.config import load_config, validate_analysis_config
    from cflibs.io.spectrum import load_spectrum

    logger.info("Inversion command (classic CF-LIBS)")
    logger.info(f"Spectrum file: {args.spectrum}")
    logger.info(f"Config file: {args.config}")

    config = {}
    if args.config:
        config = load_config(args.config)
        # Hard-error on unknown analysis.* keys: a typo'd knob
        # (``saha_boltzman_graph``) used to silently revert the run to
        # defaults with no indication anything was wrong.
        validate_analysis_config(config)

    analysis_cfg = config.get("analysis", {}) if isinstance(config, dict) else {}

    elements = _resolve_invert_elements(args, config, analysis_cfg)

    db_path = _resolve_db_path(config.get("atomic_database"), relative_to=args.config)
    if not db_path.exists():
        raise FileNotFoundError(_missing_db_message(db_path))

    wavelength, intensity = load_spectrum(args.spectrum)
    atomic_db = AtomicDatabase(db_path)

    pipeline = _build_pipeline_config(
        elements,
        preset=getattr(args, "preset", None),
        analysis_cfg=analysis_cfg,
        saha_boltzmann_graph=getattr(args, "saha_boltzmann_graph", None),
        closure_mode=getattr(args, "closure_mode", None),
        apply_self_absorption=getattr(args, "apply_self_absorption", None),
        resolving_power=getattr(args, "resolving_power", None),
        wavelength_tolerance_nm=getattr(args, "tolerance_nm", None),
        min_peak_height=getattr(args, "min_peak_height", None),
        peak_width_nm=getattr(args, "peak_width_nm", None),
    )

    result, diagnostics = _run_pipeline(wavelength, intensity, atomic_db, pipeline)

    _output_invert_result(result, args, diagnostics)


def analyze_cmd(args):
    """
    End-to-end analysis command with the validated defaults.

    Loads a CSV spectrum, runs the shared detection + selection + iterative
    CF-LIBS pipeline (:func:`_run_pipeline`), and outputs results in the
    requested format. With no flags, the ``geological`` preset resolves to
    the measured-best configuration on real geological standards: pooled
    Saha-Boltzmann graph intercepts + oxide closure (+ the always-on
    Boltzmann weight cap and per-spectrum wavelength calibration).
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.io.spectrum import load_spectrum

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(_missing_db_message(db_path))

    elements = [e.strip() for e in args.elements.split(",")]
    wavelength, intensity = load_spectrum(args.spectrum)
    atomic_db = AtomicDatabase(db_path)

    pipeline = _build_pipeline_config(
        elements,
        preset=getattr(args, "preset", None),
        saha_boltzmann_graph=getattr(args, "saha_boltzmann_graph", None),
        closure_mode=getattr(args, "closure_mode", None),
        apply_self_absorption=getattr(args, "apply_self_absorption", None),
        min_relative_intensity=getattr(args, "min_relative_intensity", None),
        resolving_power=getattr(args, "resolving_power", None),
    )

    uncertainty_mode = getattr(args, "uncertainty", "none")
    result, diagnostics = _run_pipeline(
        wavelength, intensity, atomic_db, pipeline, uncertainty_mode=uncertainty_mode
    )

    fmt = getattr(args, "output_format", "table")
    _output_analyze_result(result, fmt, diagnostics)


def _solve_analyze_result(
    solver, observations, closure_mode: str, closure_kwargs: dict, uncertainty_mode: str
):
    """Run the solver for ``analyze`` honouring the requested uncertainty mode."""
    if uncertainty_mode == "analytical":
        try:
            return solver.solve_with_uncertainty(
                observations, closure_mode=closure_mode, **closure_kwargs
            )
        except ImportError:
            logger.warning(
                "uncertainties package not installed; falling back to solve() without UQ. "
                "Install with: pip install uncertainties"
            )
            return solver.solve(observations, closure_mode=closure_mode, **closure_kwargs)
    elif uncertainty_mode == "mc":
        from cflibs.inversion.physics.uncertainty import MonteCarloUQ

        mc = MonteCarloUQ(solver, n_samples=200)
        mc_result = mc.run(observations)
        result = solver.solve(observations, closure_mode=closure_mode, **closure_kwargs)
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
        return solver.solve(observations, closure_mode=closure_mode, **closure_kwargs)


def _output_analyze_result(result, fmt: str, diagnostics: Optional[dict] = None) -> None:
    """Print the ``analyze`` result in the requested format (json/csv/table).

    Every format carries the trust report (:func:`_trust_report`): JSON gets a
    machine-readable ``trust`` block, CSV prints warnings to stderr (keeping
    stdout parseable), and the table prints the quality lines and warnings
    inline.
    """
    info_lines, warning_lines = _trust_report(result, diagnostics)
    if fmt == "json":
        import json

        output = {
            "temperature_K": result.temperature_K,
            "temperature_uncertainty_K": result.temperature_uncertainty_K,
            "electron_density_cm3": result.electron_density_cm3,
            "concentrations": result.concentrations,
            "concentration_uncertainties": result.concentration_uncertainties,
            "converged": result.converged,
            "iterations": result.iterations,
            "quality_metrics": result.quality_metrics,
            "trust": _trust_json(result, diagnostics),
        }
        print(json.dumps(output, indent=2, default=_json_default))
    elif fmt == "csv":
        print("element,concentration,uncertainty")
        for el, conc in result.concentrations.items():
            unc = result.concentration_uncertainties.get(el, 0.0)
            print(f"{el},{conc:.6f},{unc:.6f}")
        for line in warning_lines:
            print(line, file=sys.stderr)
    else:
        print(
            f"Temperature : {result.temperature_K:.0f} ± {result.temperature_uncertainty_K:.0f} K"
        )
        print(f"n_e         : {result.electron_density_cm3:.3e} cm^-3")
        print(f"Converged   : {result.converged} ({result.iterations} iterations)")
        for line in info_lines:
            print(line)
        for line in warning_lines:
            print(line)
        print("\nConcentrations:")
        for el, conc in sorted(result.concentrations.items()):
            unc = result.concentration_uncertainties.get(el, 0.0)
            print(f"  {el:4s}: {conc:.4f} ± {unc:.4f}")


def _bayesian_prefilter_elements(args, elements: list, wavelength, intensity) -> list:
    """Apply the mandatory NNLS candidate prefilter before Bayesian MCMC.

    Full-element MCMC is intractable (CLAUDE.md: ``select_candidate_elements``
    is mandatory), so the candidate set must be bounded by ``k_max`` before it
    reaches :class:`BayesianForwardModel`. Small requested sets pass through
    unchanged; oversized sets are reduced with the NNLS prefilter (requires a
    pre-computed basis library) or rejected with instructions.
    """
    k_max = int(getattr(args, "prefilter_k", None) or 15)
    if len(elements) <= k_max:
        logger.info(
            "Candidate prefilter: %d requested element(s) <= k_max=%d; all kept.",
            len(elements),
            k_max,
        )
        return elements

    basis_path = getattr(args, "basis_library", None)
    if not basis_path:
        raise ValueError(
            f"{len(elements)} candidate elements exceed the tractable MCMC limit "
            f"(k_max={k_max}); full-element MCMC is intractable and the NNLS "
            "candidate prefilter is mandatory. Provide --basis-library <basis.h5> "
            "so the prefilter can run, raise --prefilter-k, or reduce --elements."
        )

    from cflibs.inversion.candidate_prefilter import select_candidate_elements
    from cflibs.inversion.identify.spectral_nnls import SpectralNNLSIdentifier
    from cflibs.manifold.basis_library import BasisLibrary

    identifier = SpectralNNLSIdentifier(
        basis_library=BasisLibrary(str(basis_path)),
        detection_snr=3.0,
        continuum_degree=3,
        fallback_T_K=8000.0,
        fallback_ne_cm3=1e17,
    )
    prefiltered = select_candidate_elements(
        identifier=identifier,
        wavelength=wavelength,
        intensity=intensity,
        force_include=[],
        k_max=k_max,
        k_min=3,
    )
    requested = set(elements)
    prefiltered = [el for el in prefiltered if el in requested]
    if not prefiltered:
        logger.warning(
            "Candidate prefilter returned no requested elements; "
            "falling back to the first k_max=%d requested.",
            k_max,
        )
        prefiltered = elements[:k_max]
    dropped = [el for el in elements if el not in prefiltered]
    logger.info(
        "Candidate prefilter (NNLS) kept %d/%d elements: %s",
        len(prefiltered),
        len(elements),
        prefiltered,
    )
    if dropped:
        print(f"Candidate prefilter (NNLS) dropped before MCMC: {', '.join(dropped)}")
    return prefiltered


def bayesian_cmd(args):
    """
    Bayesian CF-LIBS inversion via MCMC.

    Requires: pip install cflibs[bayesian]
    """
    try:
        import numpyro  # noqa: F401
    except ImportError:
        print(
            "ERROR: Bayesian inference requires the [bayesian] optional group.\n"
            "Install with: pip install cflibs[bayesian]"
        )
        sys.exit(1)

    import numpy as np
    from cflibs.inversion.solve.bayesian import BayesianForwardModel, MCMCSampler
    from cflibs.io.spectrum import load_spectrum

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Atomic database not found: {db_path}.")

    elements = [e.strip() for e in args.elements.split(",")]
    wavelength, intensity = load_spectrum(args.spectrum)

    # Mandatory tractability guard: bound the element set before MCMC.
    elements = _bayesian_prefilter_elements(args, elements, wavelength, intensity)

    wl_min, wl_max = float(np.min(wavelength)), float(np.max(wavelength))

    forward_model_kwargs = {}
    resolving_power = getattr(args, "resolving_power", None)
    if resolving_power is not None:
        forward_model_kwargs["resolving_power"] = float(resolving_power)

    forward_model = BayesianForwardModel(
        db_path=db_path,
        elements=elements,
        wavelength_range=(wl_min, wl_max),
        **forward_model_kwargs,
    )
    sampler = MCMCSampler(forward_model=forward_model)

    logger.info("Running MCMC...")
    result = sampler.run(
        observed=intensity,
        num_samples=args.samples,
        num_chains=args.chains,
    )

    output_path = getattr(args, "output", None)
    if output_path and result.inference_data is not None:
        result.inference_data.to_netcdf(output_path)
        print(f"MCMC trace saved to {output_path}")
    else:
        try:
            import arviz as az

            if result.inference_data is not None:
                print(az.summary(result.inference_data))
            else:
                print(result.summary_table())
        except ImportError:
            print(result.summary_table())


def _batch_row(filename: str, result, diagnostics: dict, elements: list) -> dict:
    """Build one per-spectrum batch summary row including the trust fields."""
    qm = result.quality_metrics or {}
    r2 = qm.get("boltzmann_r_squared", qm.get("r_squared_last"))
    n_fit = qm.get("n_elements_fit")
    dropped = diagnostics.get("dropped_elements") or {}
    return {
        "file": filename,
        "temperature_K": result.temperature_K,
        "electron_density_cm3": result.electron_density_cm3,
        "converged": result.converged,
        "boltzmann_r_squared": r2,
        "n_elements_fit": int(n_fit) if n_fit is not None else None,
        "ne_source": _ne_source_label(qm) or "",
        "boltzmann_degenerate": bool(qm.get("boltzmann_degenerate", 0.0)),
        "closure_degenerate": bool(qm.get("closure_degenerate", 0.0)),
        "dropped_elements": ";".join(f"{el}:{stage}" for el, stage in sorted(dropped.items())),
        **{f"C_{el}": result.concentrations.get(el, 0.0) for el in elements},
    }


def _print_batch_summary(aggregate: list, n_failed: int, stream) -> None:
    """Print the aggregate batch trust summary to ``stream``."""
    n = len(aggregate)
    n_converged = sum(1 for row in aggregate if row["converged"])
    n_assumed_ne = sum(1 for row in aggregate if row["ne_source"] == "pressure_balance_fallback")
    n_dropped = sum(1 for row in aggregate if row["dropped_elements"])
    n_degenerate = sum(
        1 for row in aggregate if row["boltzmann_degenerate"] or row["closure_degenerate"]
    )
    print(
        f"Batch summary: {n} spectra processed ({n_failed} failed), "
        f"{n_converged}/{n} converged.",
        file=stream,
    )
    if n_assumed_ne:
        print(
            f"WARNING: {n_assumed_ne}/{n} spectra used the 1-atm pressure-balance n_e "
            "fallback - n_e was ASSUMED, not measured.",
            file=stream,
        )
    if n_degenerate:
        print(
            f"WARNING: {n_degenerate}/{n} spectra hit a degeneracy gate "
            "(boltzmann_degenerate/closure_degenerate columns); those results are "
            "unreliable.",
            file=stream,
        )
    if n_dropped:
        print(
            f"WARNING: {n_dropped}/{n} spectra dropped requested elements before the "
            "fit (see the 'dropped_elements' column).",
            file=stream,
        )


def batch_cmd(args):
    """
    Batch analysis: process all CSV files in a directory.

    Runs every spectrum through the SAME shared pipeline as ``analyze``
    (:func:`_build_pipeline_config` + :func:`_run_pipeline`), so per-spectrum
    results are identical to single-spectrum ``analyze`` runs with the same
    flags. The previous wiring bypassed the shared helper entirely — raw
    ``detect_line_observations`` (no top-K bound, no wavelength calibration)
    plus a bare ``LineSelector()`` — i.e. the exact pre-fix path whose drift
    caused the Na=98% blowup.
    """
    import csv as csv_mod
    import json

    from cflibs.atomic.database import AtomicDatabase
    from cflibs.io.spectrum import load_spectrum

    directory = Path(args.directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(_missing_db_message(db_path))

    elements = [e.strip() for e in args.elements.split(",")]
    atomic_db = AtomicDatabase(db_path)

    pipeline = _build_pipeline_config(
        elements,
        preset=getattr(args, "preset", None),
        saha_boltzmann_graph=getattr(args, "saha_boltzmann_graph", None),
        closure_mode=getattr(args, "closure_mode", None),
        apply_self_absorption=getattr(args, "apply_self_absorption", None),
        min_relative_intensity=getattr(args, "min_relative_intensity", None),
        resolving_power=getattr(args, "resolving_power", None),
    )

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    aggregate = []
    n_failed = 0
    for csv_path in csv_files:
        try:
            wavelength, intensity = load_spectrum(str(csv_path))
            result, diagnostics = _run_pipeline(wavelength, intensity, atomic_db, pipeline)
            aggregate.append(_batch_row(csv_path.name, result, diagnostics, elements))
            _, warning_lines = _trust_report(result, diagnostics)
            for line in warning_lines:
                logger.warning(f"{csv_path.name}: {line}")
            logger.info(
                f"{csv_path.name}: T={result.temperature_K:.0f} K, converged={result.converged}"
            )
        except Exception as e:
            n_failed += 1
            logger.error(f"{csv_path.name}: {e}")

    if not aggregate:
        print(f"No spectra processed successfully ({n_failed} failed).")
        return

    output_path = getattr(args, "output", None)
    if output_path and str(output_path).endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(aggregate, fh, indent=2, default=_json_default)
        print(f"Results written to {output_path}")
        _print_batch_summary(aggregate, n_failed, sys.stdout)
    elif output_path:
        fieldnames = list(aggregate[0].keys())
        with open(output_path, "w", newline="") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate)
        print(f"Results written to {output_path}")
        _print_batch_summary(aggregate, n_failed, sys.stdout)
    else:
        # Default: CSV to stdout; the trust summary goes to stderr so the
        # CSV stream stays machine-parseable.
        fieldnames = list(aggregate[0].keys())
        writer = csv_mod.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate)
        _print_batch_summary(aggregate, n_failed, sys.stderr)


def dbgen_cmd(args):
    """Database generation command."""
    from cflibs.atomic.database_generator import generate_database

    logger.info(f"Generating atomic database: {args.db_path}")
    print(f"\nGenerating atomic database: {args.db_path}")
    print("This may take a long time (hours) for all elements.")
    print("The script will fetch data from NIST and cache it locally.\n")

    try:
        generate_database(
            db_path=args.db_path,
            elements=args.elements,
        )
        print("\nDatabase generation complete!")
        print(f"Database saved to: {args.db_path}")
    except KeyboardInterrupt:
        print("\nDatabase generation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during database generation: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)


def doctor_cmd(args):
    """Check whether the local environment is ready for common beginner workflows."""
    import cflibs

    root = _repo_root()
    db_path = _resolve_db_path(args.db_path)
    example_config = root / "examples" / "config_example.yaml"

    required_packages = ["numpy", "scipy", "pandas", "yaml", "matplotlib"]
    optional_packages = ["jax", "h5py", "faiss", "numpyro", "arviz"]

    print("CF-LIBS setup doctor")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  cflibs: {cflibs.__version__}")
    print(f"  Working directory: {Path.cwd()}")
    print()

    print("Core checks:")
    print(f"  {'OK' if db_path.exists() else 'MISSING'} atomic database: {db_path}")
    print(f"  {'OK' if example_config.exists() else 'MISSING'} example config: {example_config}")
    for package in required_packages:
        status = "OK" if importlib.util.find_spec(package) is not None else "MISSING"
        print(f"  {status} Python package: {package}")

    print()
    print("Optional capabilities:")
    for package in optional_packages:
        status = "OK" if importlib.util.find_spec(package) is not None else "not installed"
        print(f"  {status}: {package}")

    print()
    print("Try this first:")
    print("  cflibs forward examples/config_example.yaml --output spectrum.csv")
    print(
        "  cflibs analyze data/aalto_libs/elements/Fe_spectrum.csv "
        "--elements Fe --db-path ASD_da/libs_production.db"
    )


def manifold_cmd(args):
    """Manifold generation command."""
    from cflibs.manifold.config import ManifoldConfig
    from cflibs.manifold.generator import ManifoldGenerator

    logger.info(f"Loading manifold configuration from {args.config}")

    try:
        config = ManifoldConfig.from_file(Path(args.config))
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        print(f"ERROR: Failed to load configuration: {e}")
        sys.exit(1)

    logger.info("Initializing manifold generator...")
    print("\nGenerating spectral manifold:")
    print(f"  Output: {config.output_path}")
    print(f"  Elements: {config.elements}")
    print(
        f"  Wavelength: {float(config.wavelength_range[0]):.1f} - "
        f"{float(config.wavelength_range[1]):.1f} nm"
    )
    print(
        f"  Temperature: {float(config.temperature_range[0]):.2f} - "
        f"{float(config.temperature_range[1]):.2f} eV"
    )
    print(
        f"  Density: {float(config.density_range[0]):.2e} - "
        f"{float(config.density_range[1]):.2e} cm^-3"
    )
    print("\nThis may take a long time depending on grid size...\n")

    try:
        generator = ManifoldGenerator(config)

        # Progress callback
        def progress(completed, total, percentage):
            if args.progress or completed % (total // 10) == 0:
                print(f"Progress: {completed}/{total} ({percentage:.1%})")

        generator.generate_manifold(progress_callback=progress if args.progress else None)

        print("\nManifold generation complete!")
        print(f"Output saved to: {config.output_path}")
        logger.info("Manifold generation complete")

    except ImportError as e:
        logger.error(f"JAX not available: {e}")
        print("ERROR: JAX is required for manifold generation.")
        print("Install with: pip install jax jaxlib")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Manifold generation failed: {e}", exc_info=True)
        print(f"ERROR: {e}")
        sys.exit(1)


def _add_pipeline_flags(parser) -> None:
    """Add the shared preset/solver flags to an analysis-capable subcommand.

    Used by ``analyze``, ``invert`` and ``batch`` so the three entry points
    expose identical knobs. All defaults are ``None`` ("not given"): the
    actual values resolve in :func:`_build_pipeline_config` with precedence
    CLI flag > YAML ``analysis.*`` key > preset > built-in default.
    """
    parser.add_argument(
        "--preset",
        type=str,
        choices=sorted(ANALYSIS_PRESETS),
        default=None,
        help=(
            "Analysis preset bundling the validated solver knobs (default: "
            f"{DEFAULT_ANALYSIS_PRESET}). 'geological' = pooled Saha-Boltzmann "
            "graph + oxide closure (measured best on real geological standards: "
            "BHVO-2 RMSE 10.29 -> 4.03 wt%%); 'metallic' = Saha-Boltzmann graph "
            "+ standard closure; 'raw' = legacy defaults (no SB-graph, standard "
            "closure). Explicit --closure-mode / --saha-boltzmann-graph "
            "override the preset."
        ),
    )
    parser.add_argument(
        "--closure-mode",
        type=str,
        default=None,
        choices=list(CLOSURE_MODES),
        help=(
            "Closure equation used to normalize Boltzmann intercepts to "
            "concentrations (default: from --preset; 'oxide' for the default "
            "geological preset). 'oxide' applies the default molar-oxygen "
            "stoichiometry automatically."
        ),
    )
    parser.add_argument(
        "--saha-boltzmann-graph",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use the pooled Saha-Boltzmann graph intercept extraction "
            "(Aguilera & Aragon 2004): one global regression over all lines of "
            "all species, shifting ion lines onto the neutral plane. Orthogonal "
            "to --closure-mode; stacks with oxide closure (default: from "
            "--preset; ON for geological/metallic)."
        ),
    )
    parser.add_argument(
        "--apply-self-absorption",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply the curve-of-growth self-absorption correction in the solver "
            "and retain strong resonance lines (default: off)"
        ),
    )


def _min_relative_intensity_arg(value: str):
    """Parse ``--min-relative-intensity`` ('none' keeps the floor disabled)."""
    return None if str(value).lower() == "none" else float(value)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="CF-LIBS: Computational Framework for Laser-Induced Breakdown Spectroscopy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Forward modeling command
    forward_parser = subparsers.add_parser(
        "forward", help="Generate synthetic spectrum from configuration"
    )
    forward_parser.add_argument(
        "config", type=str, help="Path to configuration file (YAML or JSON)"
    )
    forward_parser.add_argument(
        "--output", type=str, default=None, help="Output file path (default: print to stdout)"
    )
    forward_parser.set_defaults(func=forward_model_cmd)

    # Inversion command
    invert_parser = subparsers.add_parser(
        "invert", help="Infer plasma parameters from measured spectrum"
    )
    invert_parser.add_argument("spectrum", type=str, help="Path to spectrum file (CSV or similar)")
    invert_parser.add_argument(
        "--config", type=str, default=None, help="Path to inversion configuration file"
    )
    invert_parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=None,
        help="Elements to include in inversion (overrides config)",
    )
    invert_parser.add_argument(
        "--tolerance-nm",
        type=float,
        default=None,
        help="Wavelength matching tolerance in nm (default: config or 0.1)",
    )
    invert_parser.add_argument(
        "--min-peak-height",
        type=float,
        default=None,
        help="Minimum peak height (fraction of max intensity, default: config or 0.01)",
    )
    invert_parser.add_argument(
        "--peak-width-nm",
        type=float,
        default=None,
        help="Peak integration width in nm (default: config or 0.2)",
    )
    invert_parser.add_argument(
        "--resolving-power",
        type=float,
        default=None,
        help="Instrument resolving power lambda/delta_lambda (default: config value)",
    )
    invert_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (default: print to stdout)",
    )
    _add_pipeline_flags(invert_parser)
    invert_parser.set_defaults(func=invert_cmd)

    # Analyze command (end-to-end with defaults)
    analyze_parser = subparsers.add_parser(
        "analyze", help="End-to-end CF-LIBS analysis with sensible defaults"
    )
    analyze_parser.add_argument("spectrum", type=str, help="Path to spectrum CSV file")
    analyze_parser.add_argument(
        "--elements", type=str, required=True, help="Comma-separated element list, e.g. Fe,Si,Ca"
    )
    analyze_parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to atomic database (default: libs_production.db)",
    )
    analyze_parser.add_argument(
        "--format",
        type=str,
        choices=["json", "csv", "table"],
        default="table",
        dest="output_format",
        help="Output format (default: table)",
    )
    analyze_parser.add_argument(
        "--uncertainty",
        type=str,
        choices=["analytical", "mc", "none"],
        default="none",
        help="Uncertainty quantification method (default: none)",
    )
    analyze_parser.add_argument(
        "--resolving-power",
        type=float,
        default=None,
        help="Instrument resolving power lambda/delta_lambda (default: None)",
    )
    analyze_parser.add_argument(
        "--min-relative-intensity",
        type=_min_relative_intensity_arg,
        default=None,
        help=(
            "Absolute relative-intensity floor for database lines. Default None: "
            "the detection path instead bounds catalog richness with an "
            "element-relative top-K (by gA-Boltzmann strength) plus a "
            "shift-coherence veto, which restores real majors (Al/Mg/Ca/Na/K) "
            "that an absolute floor silently deletes. Pass a number to re-enable "
            "the legacy floor, or 'none' to keep it disabled."
        ),
    )
    _add_pipeline_flags(analyze_parser)
    analyze_parser.set_defaults(func=analyze_cmd)

    # Bayesian command
    bayesian_parser = subparsers.add_parser(
        "bayesian", help="Bayesian CF-LIBS inversion via MCMC (requires [bayesian] extras)"
    )
    bayesian_parser.add_argument("spectrum", type=str, help="Path to spectrum CSV file")
    bayesian_parser.add_argument(
        "--elements", type=str, required=True, help="Comma-separated element list"
    )
    bayesian_parser.add_argument(
        "--db-path", type=str, default=None, help="Path to atomic database"
    )
    bayesian_parser.add_argument(
        "--samples", type=int, default=1000, help="MCMC samples per chain (default: 1000)"
    )
    bayesian_parser.add_argument(
        "--chains", type=int, default=4, help="Number of MCMC chains (default: 4)"
    )
    bayesian_parser.add_argument(
        "--output", type=str, default=None, help="Output NetCDF path for ArviZ trace"
    )
    bayesian_parser.add_argument(
        "--resolving-power",
        type=float,
        default=None,
        help=(
            "Instrument resolving power lambda/delta_lambda; enables the "
            "resolving-power broadening mode of the forward model (default: "
            "fixed-FWHM mode)"
        ),
    )
    bayesian_parser.add_argument(
        "--basis-library",
        type=str,
        default=None,
        help=(
            "Pre-computed single-element basis library (HDF5) used by the "
            "mandatory NNLS candidate prefilter when more elements are "
            "requested than --prefilter-k allows"
        ),
    )
    bayesian_parser.add_argument(
        "--prefilter-k",
        type=int,
        default=15,
        help=(
            "Maximum tractable number of candidate elements for MCMC "
            "(default: 15). Larger requested sets are reduced with the NNLS "
            "candidate prefilter (requires --basis-library)."
        ),
    )
    bayesian_parser.set_defaults(func=bayesian_cmd)

    # Batch command
    batch_parser = subparsers.add_parser("batch", help="Process all CSV spectra in a directory")
    batch_parser.add_argument("directory", type=str, help="Directory containing CSV spectra")
    batch_parser.add_argument(
        "--elements", type=str, required=True, help="Comma-separated element list"
    )
    batch_parser.add_argument("--db-path", type=str, default=None, help="Path to atomic database")
    batch_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path (.json for JSON, else CSV to stdout)",
    )
    batch_parser.add_argument(
        "--resolving-power",
        type=float,
        default=None,
        help="Instrument resolving power lambda/delta_lambda (default: None)",
    )
    batch_parser.add_argument(
        "--min-relative-intensity",
        type=_min_relative_intensity_arg,
        default=None,
        help=(
            "Absolute relative-intensity floor for database lines "
            "(default None; see 'cflibs analyze --help')."
        ),
    )
    _add_pipeline_flags(batch_parser)
    batch_parser.set_defaults(func=batch_cmd)

    # Setup diagnostics command
    doctor_parser = subparsers.add_parser(
        "doctor", help="Check setup and print beginner-friendly next commands"
    )
    doctor_parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Path to atomic database to check (default: bundled sample or libs_production.db)",
    )
    doctor_parser.set_defaults(func=doctor_cmd)

    # Database generation command
    dbgen_parser = subparsers.add_parser(
        "generate-db", help="Generate atomic database from NIST data"
    )
    dbgen_parser.add_argument(
        "--db-path",
        type=str,
        default="libs_production.db",
        help="Path to output database file (default: libs_production.db)",
    )
    dbgen_parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        default=None,
        help="Specific elements to include (default: all elements)",
    )
    dbgen_parser.set_defaults(func=dbgen_cmd)

    # Manifold generation command
    manifold_parser = subparsers.add_parser(
        "generate-manifold", help="Generate pre-computed spectral manifold for fast inference"
    )
    manifold_parser.add_argument(
        "config", type=str, help="Path to manifold configuration file (YAML)"
    )
    manifold_parser.add_argument("--progress", action="store_true", help="Show progress updates")
    manifold_parser.set_defaults(func=manifold_cmd)

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    # Execute command
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except Exception as e:
        if args.log_level == "DEBUG":
            logger.error(f"Error executing command: {e}", exc_info=True)
        else:
            print(f"ERROR: {e}", file=sys.stderr)
            print("Run again with --log-level DEBUG to show the traceback.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
