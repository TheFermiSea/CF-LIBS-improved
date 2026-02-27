"""
Main CLI entry point for CF-LIBS.
"""

import argparse
import sys
from pathlib import Path

from cflibs.core.logging_config import setup_logging, get_logger

logger = get_logger("cli.main")


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
    db_path = config.get("atomic_database", "libs_production.db")
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Atomic database not found: {db_path}. "
            "Please run datagen_v2.py to generate the database."
        )
    atomic_db = AtomicDatabase(db_path)

    # Create plasma state
    plasma_config = config["plasma"]
    plasma = SingleZoneLTEPlasma(
        T_e=plasma_config["Te"],
        n_e=plasma_config["ne"],
        species={s["element"]: s["number_density"] for s in plasma_config["species"]},
        T_g=plasma_config.get("Tg"),
        pressure=plasma_config.get("pressure"),
    )

    # Create instrument model
    instrument = InstrumentModel.from_file(Path(args.config))

    # Get spectrum parameters
    spectrum_config = config.get("spectrum", {})
    lambda_min = spectrum_config.get("lambda_min_nm", 200.0)
    lambda_max = spectrum_config.get("lambda_max_nm", 800.0)
    delta_lambda = spectrum_config.get("delta_lambda_nm", 0.01)
    path_length = spectrum_config.get("path_length_m", 0.01)

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
            comments="#",
        )
        print(f"Spectrum saved to {output_path}")
    else:
        # Print to stdout
        print("# Wavelength (nm), Intensity (W m^-2 nm^-1 sr^-1)")
        for wl, intensity in zip(wavelength[::10], intensity[::10]):  # Sample every 10th point
            print(f"{wl:.3f},{intensity:.6e}")

    logger.info("Forward modeling complete")


def invert_cmd(args):
    """
    Inversion command.

    Runs classic CF-LIBS inversion using detected spectral lines.
    """
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.core.config import load_config
    from cflibs.inversion.line_detection import detect_line_observations
    from cflibs.inversion.line_selection import LineSelector
    from cflibs.inversion.solver import IterativeCFLIBSSolver
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

    db_path = config.get("atomic_database", "libs_production.db")
    if not Path(db_path).exists():
        raise FileNotFoundError(
            f"Atomic database not found: {db_path}. "
            "Please run datagen_v2.py to generate the database."
        )

    wavelength, intensity = load_spectrum(args.spectrum)
    atomic_db = AtomicDatabase(db_path)

    cli_tolerance = getattr(args, "tolerance_nm", None)
    cli_min_peak_height = getattr(args, "min_peak_height", None)
    cli_peak_width_nm = getattr(args, "peak_width_nm", None)
    wavelength_tolerance = (
        cli_tolerance if cli_tolerance is not None else analysis_cfg.get("wavelength_tolerance_nm", 0.1)
    )
    min_peak_height = (
        cli_min_peak_height
        if cli_min_peak_height is not None
        else analysis_cfg.get("min_peak_height", 0.01)
    )
    peak_width_nm = (
        cli_peak_width_nm if cli_peak_width_nm is not None else analysis_cfg.get("peak_width_nm", 0.2)
    )
    min_relative_intensity = analysis_cfg.get("min_relative_intensity")

    detection = detect_line_observations(
        wavelength=wavelength,
        intensity=intensity,
        atomic_db=atomic_db,
        elements=elements,
        wavelength_tolerance_nm=wavelength_tolerance,
        min_peak_height=min_peak_height,
        peak_width_nm=peak_width_nm,
        min_relative_intensity=min_relative_intensity,
    )

    for warning in detection.warnings:
        logger.warning(f"Line detection warning: {warning}")

    selector = LineSelector(
        min_snr=analysis_cfg.get("min_snr", 10.0),
        min_energy_spread_ev=analysis_cfg.get("min_energy_spread_ev", 2.0),
        min_lines_per_element=analysis_cfg.get("min_lines_per_element", 3),
        exclude_resonance=analysis_cfg.get("exclude_resonance", True),
        isolation_wavelength_nm=analysis_cfg.get("isolation_wavelength_nm", 0.1),
        max_lines_per_element=analysis_cfg.get("max_lines_per_element", 20),
    )

    selection = selector.select(
        detection.observations,
        resonance_lines=detection.resonance_lines,
    )
    observations = selection.selected_lines

    if len(observations) == 0:
        raise ValueError("No usable spectral lines detected for inversion.")

    solver = IterativeCFLIBSSolver(
        atomic_db=atomic_db,
        max_iterations=analysis_cfg.get("max_iterations", 20),
        t_tolerance_k=analysis_cfg.get("t_tolerance_k", 100.0),
        ne_tolerance_frac=analysis_cfg.get("ne_tolerance_frac", 0.1),
        pressure_pa=analysis_cfg.get("pressure_pa", None) or analysis_cfg.get("pressure", 101325.0),
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


def dbgen_cmd(args):
    """Database generation command."""
    from pathlib import Path
    import subprocess

    datagen_script = Path(__file__).parent.parent.parent / "datagen_v2.py"

    if not datagen_script.exists():
        logger.error(f"datagen_v2.py not found at {datagen_script}")
        print("ERROR: datagen_v2.py script not found.")
        print("Please ensure datagen_v2.py is in the project root directory.")
        sys.exit(1)

    logger.info(f"Generating atomic database: {args.db_path}")
    print(f"\nGenerating atomic database: {args.db_path}")
    print("This may take a long time (hours) for all elements.")
    print("The script will fetch data from NIST and cache it locally.\n")

    # Build command
    cmd = [sys.executable, str(datagen_script)]

    # Note: datagen_v2.py uses hardcoded DB_NAME, so we'd need to modify it
    # or pass environment variable. For now, just run it and let user know.
    if args.db_path != "libs_production.db":
        print("WARNING: datagen_v2.py currently generates 'libs_production.db'.")
        print("To use a different path, modify datagen_v2.py or rename the file after generation.")

    try:
        result = subprocess.run(cmd, cwd=str(datagen_script.parent))
        if result.returncode == 0:
            print("\nDatabase generation complete!")
            print(f"Database saved to: {args.db_path}")
        else:
            logger.error("Database generation failed")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\nDatabase generation interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error during database generation: {e}")
        print(f"ERROR: {e}")
        sys.exit(1)


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
    print(f"  Wavelength: {config.wavelength_range[0]:.1f} - {config.wavelength_range[1]:.1f} nm")
    print(
        f"  Temperature: {config.temperature_range[0]:.2f} - {config.temperature_range[1]:.2f} eV"
    )
    print(f"  Density: {config.density_range[0]:.2e} - {config.density_range[1]:.2e} cm^-3")
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
        "--output",
        type=str,
        default=None,
        help="Output file path for results (default: print to stdout)",
    )
    invert_parser.set_defaults(func=invert_cmd)

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
        logger.error(f"Error executing command: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
