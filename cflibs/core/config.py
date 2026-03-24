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
try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None  # type: ignore[assignment, import-untyped]

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
            config = yaml.safe_load(f)
        elif suffix == ".json":
            config = json.load(f)
        else:
            raise ValueError(
                f"Unsupported config file format: {suffix}. " "Use .yaml, .yml, or .json"
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
    elif suffix != ".json":
        raise ValueError(f"Unsupported config file format: {suffix}. " "Use .yaml, .yml, or .json")

    with open(config_path, "w") as f:
        if suffix in [".yaml", ".yml"]:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif suffix == ".json":
            json.dump(config, f, indent=2)

    logger.info(f"Saved configuration to {config_path}")
