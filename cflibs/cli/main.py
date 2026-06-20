"""
Main CLI entry point for CF-LIBS.
"""

import argparse
import importlib.util
import os
import sys
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
# Shared analyze/invert/batch pipeline configuration + execution.
#
# The pipeline itself lives in ``cflibs.inversion.pipeline`` (bead vj82) so
# the CLI and the benchmark harnesses under ``scripts/`` measure the SAME
# code path. The underscore aliases below preserve the historical private
# names that tests (``tests/cli/test_pipeline_defaults.py``) and probe
# scripts import or monkeypatch.
# ---------------------------------------------------------------------------

from cflibs.inversion.pipeline import (  # noqa: E402
    ANALYSIS_PRESETS as ANALYSIS_PRESETS,
    CLOSURE_MODES as CLOSURE_MODES,
    DEFAULT_ANALYSIS_PRESET as DEFAULT_ANALYSIS_PRESET,
    AnalysisPipelineConfig as AnalysisPipelineConfig,
    build_pipeline_config,
    detect_and_select_lines,
    run_pipeline,
)

#: Backward-compatible aliases for the pre-vj82 private names.
_build_pipeline_config = build_pipeline_config
_detect_and_select_lines = detect_and_select_lines
_run_pipeline = run_pipeline

__all_pipeline_names__ = (
    "ANALYSIS_PRESETS",
    "CLOSURE_MODES",
    "DEFAULT_ANALYSIS_PRESET",
    "AnalysisPipelineConfig",
    "_build_pipeline_config",
    "_detect_and_select_lines",
    "_run_pipeline",
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


def _ne_source_label(quality_metrics: dict) -> Optional[str]:
    """Map the ``ne_from_stark`` provenance flag to a human-readable source."""
    ne_from_stark = quality_metrics.get("ne_from_stark")
    if ne_from_stark is None:
        return None
    return "stark" if ne_from_stark else "pressure_balance_fallback"


def _refuse_to_report_enabled() -> bool:
    """M7 Lever 6: opt-in CLI refuse-to-report gate.

    Controlled by the ``CFLIBS_REFUSE_TO_REPORT`` env flag (default OFF). When
    OFF the CLI trust report is byte-for-byte identical to legacy behaviour
    (non-regression); when ON, a result that fails the LTE/quality gate
    (``overall_reliable`` is False) is additionally marked RESULT UNRELIABLE.
    """
    return os.environ.get("CFLIBS_REFUSE_TO_REPORT", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


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
        n_lines = int(qm.get("stark_n_lines", 0) or 0)
        scatter = float(qm.get("stark_ne_scatter_cm3", 0.0) or 0.0)
        detail = f"{n_lines} line{'s' if n_lines != 1 else ''}" if n_lines else "supplied"
        if scatter > 0:
            detail += f", scatter {scatter:.1e} cm^-3"
        info.append(f"n_e source  : Stark-width diagnostic (measured, {detail})")
        ne_value = float(getattr(result, "electron_density_cm3", 0.0) or 0.0)
        if scatter > 0 and ne_value > 0 and scatter > ne_value:
            warnings_out.append(
                f"WARNING: Stark n_e line-to-line scatter ({scatter:.1e} cm^-3) exceeds "
                f"the measured median ({ne_value:.1e} cm^-3); the diagnostic lines "
                "disagree — treat n_e as an order-of-magnitude measurement."
            )
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

    # M7 Lever 6: surface the Cristoforetti multi-check quality flag.
    quality_flag = qm.get("quality_flag")
    if quality_flag is not None:
        info.append(f"Quality flag: {quality_flag}")
    overall_reliable = getattr(result, "overall_reliable", None)

    dropped = (diagnostics or {}).get("dropped_elements") or {}
    if dropped:
        detail = ", ".join(f"{el} ({stage})" for el, stage in sorted(dropped.items()))
        warnings_out.append(f"WARNING: requested elements dropped: {detail}")

    # Always-on hard gates (convergence + degeneracy) — unchanged.
    unreliable = bool(
        not result.converged or qm.get("boltzmann_degenerate") or qm.get("closure_degenerate")
    )
    # M7 refuse-to-report (opt-in): also refuse when the LTE/quality gate fails.
    if _refuse_to_report_enabled() and overall_reliable is False:
        unreliable = True
        warnings_out.append(
            "WARNING: result fails the refuse-to-report gate "
            f"(quality_flag={quality_flag}, McWhirter satisfied={bool(lte_ok)}); "
            "composition withheld as non-quantitative."
        )
    if unreliable:
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

    # A response-curve path in the YAML is resolved relative to the config
    # file (like atomic_database), so shipped configs can use short paths.
    if analysis_cfg.get("response_curve") and args.config:
        analysis_cfg = dict(analysis_cfg)
        analysis_cfg["response_curve"] = str(
            _resolve_existing_path(analysis_cfg["response_curve"], relative_to=args.config)
        )

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
        stark_ne=getattr(args, "stark_ne", None),
        resolving_power=getattr(args, "resolving_power", None),
        wavelength_tolerance_nm=getattr(args, "tolerance_nm", None),
        min_peak_height=getattr(args, "min_peak_height", None),
        peak_width_nm=getattr(args, "peak_width_nm", None),
        response_curve=getattr(args, "response_curve", None),
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
        stark_ne=getattr(args, "stark_ne", None),
        min_relative_intensity=getattr(args, "min_relative_intensity", None),
        resolving_power=getattr(args, "resolving_power", None),
        response_curve=getattr(args, "response_curve", None),
    )

    uncertainty_mode = getattr(args, "uncertainty", "none")
    result, diagnostics = _run_pipeline(
        wavelength, intensity, atomic_db, pipeline, uncertainty_mode=uncertainty_mode
    )

    fmt = getattr(args, "output_format", "table")
    _output_analyze_result(result, fmt, diagnostics)


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
        stark_ne=getattr(args, "stark_ne", None),
        min_relative_intensity=getattr(args, "min_relative_intensity", None),
        resolving_power=getattr(args, "resolving_power", None),
        response_curve=getattr(args, "response_curve", None),
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
            "Apply the observable-gated self-absorption correction "
            "(doublet intensity ratios; SA-suspect resonance lines are "
            "down-weighted) to the measured line intensities before the "
            "Boltzmann fit (default: off)"
        ),
    )
    parser.add_argument(
        "--response-curve",
        type=str,
        default=None,
        help=(
            "Path to a spectral-response curve (CSV 'wavelength_nm,relative_efficiency' "
            "or YAML with those two keys). Measured intensities are divided by the "
            "interpolated relative detection efficiency E(lambda) before line "
            "extraction — mandatory for CF-LIBS on uncalibrated spectrometers "
            "(Tognoni et al. 2010). Only the relative shape matters; the curve is "
            "normalized to max=1. Default: no correction. Do NOT use on ChemCam CCS "
            "data (already response-corrected upstream)."
        ),
    )
    parser.add_argument(
        "--stark-ne",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Measure n_e from the Stark widths of observed literature-grade "
            "lines (pinned-Gaussian Voigt fit; Gigosos 2014, Konjevic 2002) "
            "instead of assuming the 1-atm pressure-balance fallback. Degrades "
            "gracefully to the (warned) fallback when no line qualifies "
            "(default: from --preset; ON for geological/metallic, OFF for raw)."
        ),
    )


def _min_relative_intensity_arg(value: str):
    """Parse ``--min-relative-intensity`` ('none' keeps the floor disabled)."""
    return None if str(value).lower() == "none" else float(value)


def scoreboard_cmd(args):
    """Goal-metric scoreboard: ID accuracy, composition accuracy, runtime."""
    from cflibs.atomic.database import AtomicDatabase
    from cflibs.benchmark.scoreboard import run_scoreboard, write_artifacts
    from cflibs.benchmark.scoreboard_registry import ensure_default_datasets

    ensure_default_datasets()

    db_path = _resolve_db_path(args.db_path)
    if not db_path.exists():
        raise FileNotFoundError(_missing_db_message(db_path))
    atomic_db = AtomicDatabase(db_path)

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()] if args.datasets else None
    tags = [s.strip() for s in args.tags.split(",") if s.strip()] if args.tags else None

    board = run_scoreboard(
        atomic_db,
        datasets=datasets,
        tags=tags,
        max_spectra=args.max_spectra,
        seed=args.seed,
        include_holdout=args.include_holdout,
        pipeline_impl=args.pipeline,
    )
    json_path, md_path = write_artifacts(board, args.output_dir)
    print(md_path.read_text())
    print(f"JSON artifact: {json_path}")
    print(f"Markdown table: {md_path}")


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

    # Goal-metric scoreboard command (bead A1)
    scoreboard_parser = subparsers.add_parser(
        "scoreboard",
        help=(
            "Goal-metric scoreboard: element-ID accuracy, composition accuracy "
            "and runtime across every truth-bearing dataset"
        ),
    )
    scoreboard_parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset names (default: all registered)",
    )
    scoreboard_parser.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Comma-separated tag filter, e.g. 'real' or 'synthetic' (default: all)",
    )
    scoreboard_parser.add_argument(
        "--db-path", type=str, default=None, help="Path to atomic database"
    )
    scoreboard_parser.add_argument(
        "--output-dir",
        type=str,
        default="output/scoreboard",
        help="Directory for scoreboard.json + scoreboard.md (default: output/scoreboard)",
    )
    scoreboard_parser.add_argument(
        "--max-spectra",
        type=int,
        default=None,
        help=(
            "Per-dataset spectrum cap; larger datasets are sampled with a "
            "seeded rng (default: run everything)"
        ),
    )
    from cflibs.benchmark.scoreboard import DEFAULT_SEED as _SCOREBOARD_DEFAULT_SEED

    scoreboard_parser.add_argument(
        "--seed",
        type=int,
        default=_SCOREBOARD_DEFAULT_SEED,
        help=f"Sampling seed used with --max-spectra (default: {_SCOREBOARD_DEFAULT_SEED})",
    )
    scoreboard_parser.add_argument(
        "--include-holdout",
        action="store_true",
        help=(
            "Also run holdout-tier datasets (the campaign adoption gate, e.g. "
            "bhvo2_chemcam, emslibs2019). Off by default so casual boards cannot "
            "leak the gate; vault-tier datasets never run."
        ),
    )
    scoreboard_parser.add_argument(
        "--pipeline",
        choices=["reference", "jit"],
        default="reference",
        help=(
            "Inversion pipeline implementation: 'reference' (run_pipeline, the "
            "parity oracle) or 'jit' (cflibs.jitpipe.run_one, the JAX port). "
            "J12/M3 superiority runs compare the two (default: reference)."
        ),
    )
    scoreboard_parser.set_defaults(func=scoreboard_cmd)

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
