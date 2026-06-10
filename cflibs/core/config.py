"""
Configuration management for CF-LIBS.

Provides utilities for loading and validating YAML/JSON configuration files
for plasma models, instrument settings, and inversion parameters.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from typing import Any, Dict, Union

# Type alias for yaml module to avoid mypy [import-untyped] error when stubs unavailable
# Install type stubs via: pip install pyyaml-stubs
# For now, we use type: ignore comments to suppress the warning
try:
    import yaml  # type: ignore[import-untyped]

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None  # type: ignore[assignment, unused-ignore]

# Type aliases for common structures
ConfigDict = Dict[str, Any]

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.

    Parameters
    ----------
    config_path : str or Path
        Path to configuration file (.yaml, .yml, or .json)

    Returns
    -------
    dict
        Configuration dictionary

    Raises
    ------
    FileNotFoundError
        If config file does not exist
    ValueError
        If file format is not supported
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    suffix = config_path.suffix.lower()

    with open(config_path, "r") as f:
        if suffix in [".yaml", ".yml"]:
            if not HAS_YAML:
                raise ImportError(
                    "PyYAML is required for YAML config files. " "Install with: pip install pyyaml"
                )
            config = yaml.safe_load(f)  # type: ignore[attr-defined, union-attr]
        elif suffix == ".json":
            import json

            config = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix} (use .yaml, .yml, or .json)"
            )

    logger.info(f"Loaded configuration from {config_path}")
    return config


def validate_plasma_config(config: Dict[str, Any]) -> bool:
    """
    Validate plasma configuration structure.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    if "plasma" not in config:
        raise ValueError("Configuration must contain 'plasma' section")

    plasma = config["plasma"]

    # Check required fields
    required = ["model", "Te", "ne"]
    for field in required:
        if field not in plasma:
            raise ValueError(f"Plasma config missing required field: {field}")

    # Validate model type
    valid_models = ["single_zone_lte", "multi_zone_lte"]
    if plasma["model"] not in valid_models:
        raise ValueError(
            f"Invalid plasma model: {plasma['model']}. " f"Must be one of: {valid_models}"
        )

    # Validate species if present
    if "species" in plasma:
        if not isinstance(plasma["species"], (list, dict)):
            raise ValueError("'species' must be a list or dict")

    return True


#: Keys accepted under the ``analysis:`` section of an inversion config.
#: Single source of truth shared by :func:`validate_analysis_config` and the
#: CLI (``cflibs invert --config``). Unknown keys are a HARD ERROR: a typo'd
#: key (e.g. ``saha_boltzman_graph``) used to be silently ignored, reverting
#: the run to defaults with no indication anything was wrong.
VALID_ANALYSIS_KEYS = frozenset(
    {
        "preset",
        "elements",
        "wavelength_tolerance_nm",
        "min_peak_height",
        "peak_width_nm",
        "min_relative_intensity",
        "top_k_per_element",
        "resolving_power",
        "apply_self_absorption",
        "exclude_resonance",
        "min_snr",
        "min_energy_spread_ev",
        "min_lines_per_element",
        "isolation_wavelength_nm",
        "max_lines_per_element",
        "wavelength_calibration",
        "shift_coherence_veto",
        "residual_shift_scan_nm",
        "affine_coverage_gate",
        "line_residual_gate",
        "response_curve",
        "max_iterations",
        "t_tolerance_k",
        "ne_tolerance_frac",
        "pressure_pa",
        "pressure",
        "boltzmann_weight_cap",
        "min_boltzmann_r2",
        "saha_boltzmann_graph",
        "closure_mode",
        "closure_kwargs",
        "matrix_element",
        "oxide_elements",
        "stark_ne",
    }
)


def validate_analysis_config(config: Dict[str, Any]) -> bool:
    """
    Validate the ``analysis`` section of an inversion configuration.

    Unknown keys raise a hard error listing the valid keys, so a typo'd
    knob (``saha_boltzman_graph``) cannot silently fall back to defaults.

    Parameters
    ----------
    config : dict
        Full configuration dictionary (the ``analysis`` section is optional).

    Returns
    -------
    bool
        True if valid (or no ``analysis`` section is present).

    Raises
    ------
    ValueError
        If the ``analysis`` section is not a mapping or contains unknown keys.
    """
    analysis = config.get("analysis")
    if analysis is None:
        return True
    if not isinstance(analysis, dict):
        raise ValueError(f"'analysis' section must be a mapping; got {type(analysis).__name__}")

    unknown = sorted(set(analysis) - VALID_ANALYSIS_KEYS)
    if unknown:
        raise ValueError(
            f"Unknown analysis config key(s): {unknown}. "
            f"Valid keys: {sorted(VALID_ANALYSIS_KEYS)}"
        )
    return True


def validate_instrument_config(config: Dict[str, Any]) -> bool:
    """
    Validate instrument configuration structure.

    Parameters
    ----------
    config : dict
        Configuration dictionary

    Returns
    -------
    bool
        True if valid

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    if "instrument" not in config:
        raise ValueError("Configuration must contain 'instrument' section")

    instr = config["instrument"]

    # Resolution is required
    if "resolution_fwhm_nm" not in instr:
        raise ValueError("Instrument config missing 'resolution_fwhm_nm'")

    # Validate resolution is positive
    if instr["resolution_fwhm_nm"] <= 0:
        raise ValueError("Instrument resolution must be positive")

    return True


def save_config(config: Dict[str, Any], config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML or JSON file.

    Parameters
    ----------
    config : dict
        Configuration dictionary
    config_path : str or Path
        Path to output file
    """
    config_path = Path(config_path)
    suffix = config_path.suffix.lower()

    # Validate suffix before opening file to avoid truncating on error
    if suffix in [".yaml", ".yml"]:
        if not HAS_YAML:
            raise ImportError(
                "PyYAML is required for YAML config files. " "Install with: pip install pyyaml"
            )
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)  # type: ignore[attr-defined, union-attr]
    elif suffix == ".json":
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported config file format: {suffix}. " "Use .yaml, .yml, or .json")

    logger.info(f"Saved configuration to {config_path}")
