"""
Main CLI entry point for CF-LIBS.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

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

    Returns
    -------
    list
        The selected ``LineObservation`` list (``selection.selected_lines``).
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
    return selection.selected_lines


def invert_cmd(args):
    """
    Inversion command.

    Runs classic CF-LIBS inversion using detected spectral lines.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.core.config import load_config
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
    from cflibs.io.exporters import create_exporter
    from cflibs.io.spectrum import load_spectrum

    logger.info("Inversion command (classic CF-LIBS)")
    logger.info(f"Spectrum file: {args.spectrum}")
    logger.info(f"Config file: {args.config}")

    config = {}
    if args.config:
        config = load_config(args.config)

    analysis_cfg = config.get("analysis", {}) if isinstance(config, dict) else {}

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

    db_path = _resolve_db_path(config.get("atomic_database"), relative_to=args.config)
    if not db_path.exists():
        raise FileNotFoundError(_missing_db_message(db_path))

    wavelength, intensity = load_spectrum(args.spectrum)
    atomic_db = AtomicDatabase(db_path)

    cli_tolerance = getattr(args, "tolerance_nm", None)
    cli_min_peak_height = getattr(args, "min_peak_height", None)
    cli_peak_width_nm = getattr(args, "peak_width_nm", None)
    wavelength_tolerance = (
        cli_tolerance
        if cli_tolerance is not None
        else analysis_cfg.get("wavelength_tolerance_nm", 0.1)
    )
    min_peak_height = (
        cli_min_peak_height
        if cli_min_peak_height is not None
        else analysis_cfg.get("min_peak_height", 0.01)
    )
    peak_width_nm = (
        cli_peak_width_nm
        if cli_peak_width_nm is not None
        else analysis_cfg.get("peak_width_nm", 0.2)
    )
    # Default to a non-None relative-intensity floor (physics-audit /
    # composition-pipeline-diagnosis: IDENT-RYDBERG). With no floor, weak
    # high-lying (Rydberg) transitions with relative_intensity ~ 0 (e.g. the
    # Na I 413-421 nm lines, E_k ~ 5 eV, unobservable in a ~1 eV ps-LIBS
    # plasma) are matched to bright wrong-element peaks and, because the
    # Boltzmann ordinate divides by their tiny A_ki, extrapolate the closure to
    # a huge spurious abundance (Na ~ 77-98 wt% vs certified 1.65 on BHVO-2).
    # As of the detection-cascade fix the default floor is OFF (None): an
    # absolute rel_int floor deletes whole real elements (Mg/K and the Al I
    # resonance doublet all sit below 100), so it is replaced by an
    # element-relative top-K (gA-Boltzmann strength) plus a shift-coherence
    # veto, which suppress the weak-Rydberg false matches the floor was
    # guarding against without deleting real majors. Set a numeric
    # ``min_relative_intensity`` in the analysis config to re-enable the
    # legacy floor.
    min_relative_intensity = analysis_cfg.get("min_relative_intensity", None)
    resolving_power = (
        args.resolving_power
        if args.resolving_power is not None
        else analysis_cfg.get("resolving_power")
    )

    # Self-absorption correction (physics-audit 2026-05-27 defects B1/B2).
    # ``apply_self_absorption`` (default False) wires the curve-of-growth
    # correction into the iterative solver for known optically-thick samples
    # (e.g. the BHVO-2 basalt majors). When it is enabled we ALSO retain the
    # strong low-E_i resonance lines (Ca II H/K, Na D, Mg I 285, Al I 396) that
    # dominate the majors, because the solver now *corrects* them rather than
    # the selector dropping them for "self-absorption risk" (B2; Aragón &
    # Aguilera 2008 §7). When SA is off we keep the original safe behaviour of
    # excluding resonance lines (uncorrected resonance lines are self-absorbed
    # and bias the Boltzmann plot). Both are config-overridable.
    apply_self_absorption = bool(analysis_cfg.get("apply_self_absorption", False))
    # Default ``None`` -> ``_detect_and_select_lines`` keeps resonance lines
    # (the brightest persistent LIBS lines, sole detectable lines for some
    # majors). Set ``exclude_resonance`` in the analysis config to override.
    exclude_resonance = analysis_cfg.get("exclude_resonance", None)

    # Shared detection + selection path (identical to ``analyze``; see
    # ``_detect_and_select_lines``). Keeping both CLI entry points on one
    # helper prevents the default-path Na-blowup regression from re-emerging.
    observations = _detect_and_select_lines(
        wavelength,
        intensity,
        atomic_db,
        elements,
        min_relative_intensity=min_relative_intensity,
        resolving_power=resolving_power,
        wavelength_tolerance_nm=wavelength_tolerance,
        min_peak_height=min_peak_height,
        peak_width_nm=peak_width_nm,
        apply_self_absorption=apply_self_absorption,
        exclude_resonance=exclude_resonance,
        min_snr=analysis_cfg.get("min_snr", 10.0),
        min_energy_spread_ev=analysis_cfg.get("min_energy_spread_ev", 2.0),
        min_lines_per_element=analysis_cfg.get("min_lines_per_element", 3),
        isolation_wavelength_nm=analysis_cfg.get("isolation_wavelength_nm", 0.1),
        max_lines_per_element=analysis_cfg.get("max_lines_per_element", 20),
    )

    if len(observations) == 0:
        raise ValueError("No usable spectral lines detected for inversion.")

    solver = IterativeCFLIBSSolver(
        atomic_db=atomic_db,
        max_iterations=analysis_cfg.get("max_iterations", 20),
        t_tolerance_k=analysis_cfg.get("t_tolerance_k", 100.0),
        ne_tolerance_frac=analysis_cfg.get("ne_tolerance_frac", 0.1),
        pressure_pa=analysis_cfg.get("pressure_pa", None) or analysis_cfg.get("pressure", 101325.0),
        apply_self_absorption=apply_self_absorption,
        self_absorption_column_density_cm3=analysis_cfg.get(
            "self_absorption_column_density_cm3", 1.0e16
        ),
        self_absorption_plasma_length_cm=analysis_cfg.get("self_absorption_plasma_length_cm", 0.1),
    )

    closure_mode = analysis_cfg.get("closure_mode", "standard")
    closure_kwargs = dict(analysis_cfg.get("closure_kwargs", {}))
    if closure_mode == "matrix" and "matrix_element" in analysis_cfg:
        closure_kwargs.setdefault("matrix_element", analysis_cfg["matrix_element"])
    if closure_mode == "oxide" and "oxide_elements" in analysis_cfg:
        closure_kwargs.setdefault("oxide_elements", analysis_cfg["oxide_elements"])

    result = solver.solve(observations, closure_mode=closure_mode, **closure_kwargs)

    if args.output:
        output_path = Path(args.output)
        ext = output_path.suffix.lower().lstrip(".")
        if ext not in {"csv", "json", "h5", "hdf5"}:
            ext = "json"
            output_path = output_path.with_suffix(".json")

        exporter = create_exporter(ext)
        exporter.export(result, str(output_path))
        print(f"Inversion results saved to {output_path}")
        return

    print("CF-LIBS inversion results:")
    print(f"  Temperature: {result.temperature_K:.0f} ± {result.temperature_uncertainty_K:.0f} K")
    print(f"  Electron density: {result.electron_density_cm3:.3e} cm^-3")
    print("  Concentrations:")
    for element, concentration in result.concentrations.items():
        unc = result.concentration_uncertainties.get(element, 0.0)
        print(f"    {element}: {concentration:.4f} ± {unc:.4f}")


def analyze_cmd(args):
    """
    End-to-end analysis command with sensible defaults.

    Loads a CSV spectrum, runs line detection + iterative CF-LIBS inversion,
    and outputs results in the requested format.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
    from cflibs.io.spectrum import load_spectrum

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(_missing_db_message(db_path))

    elements = [e.strip() for e in args.elements.split(",")]
    wavelength, intensity = load_spectrum(args.spectrum)
    atomic_db = AtomicDatabase(db_path)

    # Shared detection + selection path — identical to ``invert`` (see
    # ``_detect_and_select_lines``). Previously ``analyze`` called
    # ``detect_line_observations(...)`` with no relative-intensity floor and a
    # bare ``LineSelector()``, which admitted spurious weak Na Rydberg lines
    # and produced a catastrophic Na-dominated composition (RMSE 33.69 wt%, Na
    # ~98 % on BHVO-2). Using the same helper with the good defaults kills that
    # default-path blowup. ``--min-relative-intensity`` / ``--resolving-power``
    # / ``--apply-self-absorption`` opt-in flags override the defaults.
    apply_self_absorption = bool(getattr(args, "apply_self_absorption", False))
    min_relative_intensity = getattr(args, "min_relative_intensity", None)
    resolving_power = getattr(args, "resolving_power", None)

    observations = _detect_and_select_lines(
        wavelength,
        intensity,
        atomic_db,
        elements,
        min_relative_intensity=min_relative_intensity,
        resolving_power=resolving_power,
        apply_self_absorption=apply_self_absorption,
    )

    if len(observations) == 0:
        raise ValueError("No usable spectral lines detected.")

    solver = IterativeCFLIBSSolver(
        atomic_db=atomic_db,
        apply_self_absorption=apply_self_absorption,
    )

    uncertainty_mode = getattr(args, "uncertainty", "none")
    if uncertainty_mode == "analytical":
        try:
            result = solver.solve_with_uncertainty(observations)
        except ImportError:
            logger.warning(
                "uncertainties package not installed; falling back to solve() without UQ. "
                "Install with: pip install uncertainties"
            )
            result = solver.solve(observations)
    elif uncertainty_mode == "mc":
        from cflibs.inversion.physics.uncertainty import MonteCarloUQ

        mc = MonteCarloUQ(solver, n_samples=200)
        mc_result = mc.run(observations)
        result = solver.solve(observations)
        result = result.__class__(
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
        result = solver.solve(observations)

    fmt = getattr(args, "output_format", "table")
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
        }
        print(json.dumps(output, indent=2))
    elif fmt == "csv":
        print("element,concentration,uncertainty")
        for el, conc in result.concentrations.items():
            unc = result.concentration_uncertainties.get(el, 0.0)
            print(f"{el},{conc:.6f},{unc:.6f}")
    else:
        print(
            f"Temperature : {result.temperature_K:.0f} ± {result.temperature_uncertainty_K:.0f} K"
        )
        print(f"n_e         : {result.electron_density_cm3:.3e} cm^-3")
        print(f"Converged   : {result.converged} ({result.iterations} iterations)")
        lte_ok = result.quality_metrics.get("lte_mcwhirter_satisfied", None)
        if lte_ok is not None and not lte_ok:
            print(
                f"WARNING: McWhirter criterion NOT satisfied "
                f"(n_e ratio = {result.quality_metrics.get('lte_n_e_ratio', 0):.2f})"
            )
        print("\nConcentrations:")
        for el, conc in sorted(result.concentrations.items()):
            unc = result.concentration_uncertainties.get(el, 0.0)
            print(f"  {el:4s}: {conc:.4f} ± {unc:.4f}")


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

    wl_min, wl_max = float(np.min(wavelength)), float(np.max(wavelength))

    forward_model = BayesianForwardModel(
        db_path=db_path,
        elements=elements,
        wavelength_range=(wl_min, wl_max),
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


def batch_cmd(args):
    """
    Batch analysis: process all CSV files in a directory.
    """
    import csv as csv_mod
    import json

    from cflibs.atomic.database import AtomicDatabase
    from cflibs.inversion.line_detection import detect_line_observations
    from cflibs.inversion.line_selection import LineSelector
    from cflibs.inversion.solve.iterative import IterativeCFLIBSSolver
    from cflibs.io.spectrum import load_spectrum

    directory = Path(args.directory)
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Atomic database not found: {db_path}.")

    elements = [e.strip() for e in args.elements.split(",")]
    atomic_db = AtomicDatabase(db_path)
    solver = IterativeCFLIBSSolver(atomic_db=atomic_db)

    csv_files = sorted(directory.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {directory}")
        return

    aggregate = []
    for csv_path in csv_files:
        try:
            wavelength, intensity = load_spectrum(str(csv_path))
            detection = detect_line_observations(
                wavelength=wavelength,
                intensity=intensity,
                atomic_db=atomic_db,
                elements=elements,
            )
            selector = LineSelector()
            selection = selector.select(
                detection.observations, resonance_lines=detection.resonance_lines
            )
            if not selection.selected_lines:
                logger.warning(f"{csv_path.name}: no lines detected, skipping")
                continue
            result = solver.solve(selection.selected_lines)
            row = {
                "file": csv_path.name,
                "temperature_K": result.temperature_K,
                "electron_density_cm3": result.electron_density_cm3,
                "converged": result.converged,
                **{f"C_{el}": result.concentrations.get(el, 0.0) for el in elements},
            }
            aggregate.append(row)
            logger.info(
                f"{csv_path.name}: T={result.temperature_K:.0f} K, converged={result.converged}"
            )
        except Exception as e:
            logger.error(f"{csv_path.name}: {e}")

    if not aggregate:
        print("No spectra processed successfully.")
        return

    output_path = getattr(args, "output", None)
    if output_path and str(output_path).endswith(".json"):
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(aggregate, fh, indent=2)
        print(f"Results written to {output_path}")
    elif output_path:
        fieldnames = list(aggregate[0].keys())
        with open(output_path, "w", newline="") as fh:
            writer = csv_mod.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate)
        print(f"Results written to {output_path}")
    else:
        # Default: CSV to stdout
        fieldnames = list(aggregate[0].keys())
        writer = csv_mod.DictWriter(sys.stdout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(aggregate)


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
        type=lambda s: None if str(s).lower() == "none" else float(s),
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
    analyze_parser.add_argument(
        "--apply-self-absorption",
        action="store_true",
        help=(
            "Apply the curve-of-growth self-absorption correction in the solver "
            "and retain strong resonance lines (default: off)"
        ),
    )
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
