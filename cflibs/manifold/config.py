"""
Configuration for manifold generation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from pathlib import Path

from cflibs.core.logging_config import get_logger
from cflibs.radiation.profiles import BroadeningMode

logger = get_logger("manifold.config")


def _coerce_broadening_mode(raw) -> BroadeningMode:
    """Coerce a YAML-loaded broadening_mode value into a ``BroadeningMode``.

    Accepts:
      * ``BroadeningMode`` instances (returned as-is)
      * Case-insensitive strings matching enum values: ``"ldm_gaussian"``,
        ``"physical_doppler"``, ``"legacy"``, ``"nist_parity"``.

    Raises ``ValueError`` with the list of valid values otherwise.
    """
    if isinstance(raw, BroadeningMode):
        return raw
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        try:
            return BroadeningMode(normalized)
        except ValueError as exc:
            valid = ", ".join(repr(m.value) for m in BroadeningMode)
            raise ValueError(f"Invalid broadening_mode {raw!r}; expected one of {valid}") from exc
    raise ValueError(
        f"broadening_mode must be a string or BroadeningMode; got {type(raw).__name__}"
    )


@dataclass
class ManifoldConfig:
    """
    Configuration for generating a spectral manifold.

    The manifold is a pre-computed lookup table of synthetic spectra
    covering the parameter space of interest.

    Attributes
    ----------
    db_path : str
        Path to atomic database
    output_path : str
        Path to output HDF5 file
    elements : List[str]
        Elements to include in manifold
    wavelength_range : Tuple[float, float]
        (min, max) wavelength in nm
    temperature_range : Tuple[float, float]
        (min, max) temperature in eV
    temperature_steps : int
        Number of temperature grid points
    density_range : Tuple[float, float]
        (min, max) electron density in cm^-3
    density_steps : int
        Number of density grid points (log space)
    concentration_steps : int
        Resolution for concentration grid
    pixels : int
        Number of wavelength pixels in output spectrum
    gate_delay_s : float
        ICCD gate delay in seconds
    gate_width_s : float
        ICCD gate width in seconds
    time_steps : int
        Number of time steps for integration
    batch_size : int
        Batch size for GPU processing
    """

    db_path: str
    output_path: str
    elements: List[str]
    wavelength_range: Tuple[float, float] = (250.0, 550.0)
    temperature_range: Tuple[float, float] = (0.5, 2.0)
    temperature_steps: int = 50
    density_range: Tuple[float, float] = (1e16, 1e19)
    density_steps: int = 20
    concentration_steps: int = 20
    pixels: int = 4096
    gate_delay_s: float = 300e-9
    gate_width_s: float = 5e-6
    time_steps: int = 20
    batch_size: int = 1000

    # Phase 2 physics options
    use_voigt_profile: bool = True
    use_stark_broadening: bool = True
    instrument_fwhm_nm: float = 0.05
    physics_version: int = 2

    # Broadening kernel dispatch (ADR-0001 T1-4 / bead 8n4i).
    # Default preserves the current per-line Voigt path; users can opt in to
    # ``BroadeningMode.LDM_GAUSSIAN`` for the O(N_σ · N_λ log N_λ) Line
    # Distribution Method (van den Bekerom & Pannier 2021) which is ~19×
    # faster at N_lines=1500 but currently 1-D Gaussian-only (no Stark).
    broadening_mode: BroadeningMode = field(
        default=BroadeningMode.PHYSICAL_DOPPLER,
    )

    @classmethod
    def from_file(cls, config_path: Path) -> "ManifoldConfig":
        """
        Load manifold configuration from YAML file.

        Parameters
        ----------
        config_path : Path
            Path to configuration file

        Returns
        -------
        ManifoldConfig
            Configuration instance
        """
        from cflibs.core.config import load_config

        config = load_config(config_path)

        if "manifold" not in config:
            raise ValueError("Configuration must contain 'manifold' section")

        manifold_config = config["manifold"]

        # YAML 1.1 (PyYAML default) parses unsigned-exponent scientific
        # notation like ``1e16`` as a STRING, not a float — only ``1.0e+16``
        # and ``1e+16`` parse as floats. Coerce numeric ranges so user configs
        # work either way.
        def _float_pair(raw, default):
            seq = raw if raw is not None else default
            return tuple(float(x) for x in seq)

        broadening_mode_raw = manifold_config.get("broadening_mode")
        if broadening_mode_raw is None:
            broadening_mode = BroadeningMode.PHYSICAL_DOPPLER
        else:
            broadening_mode = _coerce_broadening_mode(broadening_mode_raw)

        return cls(
            db_path=manifold_config.get("db_path", "libs_production.db"),
            output_path=manifold_config.get("output_path", "spectral_manifold.h5"),
            elements=manifold_config.get("elements", ["Ti", "Al", "V", "Fe"]),
            wavelength_range=_float_pair(manifold_config.get("wavelength_range"), [250.0, 550.0]),
            temperature_range=_float_pair(manifold_config.get("temperature_range"), [0.5, 2.0]),
            temperature_steps=manifold_config.get("temperature_steps", 50),
            density_range=_float_pair(manifold_config.get("density_range"), [1e16, 1e19]),
            density_steps=manifold_config.get("density_steps", 20),
            concentration_steps=manifold_config.get("concentration_steps", 20),
            pixels=manifold_config.get("pixels", 4096),
            gate_delay_s=float(manifold_config.get("gate_delay_s", 300e-9)),
            gate_width_s=float(manifold_config.get("gate_width_s", 5e-6)),
            time_steps=manifold_config.get("time_steps", 20),
            batch_size=manifold_config.get("batch_size", 1000),
            use_voigt_profile=manifold_config.get("use_voigt_profile", True),
            use_stark_broadening=manifold_config.get("use_stark_broadening", True),
            instrument_fwhm_nm=float(manifold_config.get("instrument_fwhm_nm", 0.05)),
            physics_version=manifold_config.get("physics_version", 2),
            broadening_mode=broadening_mode,
        )

    def validate(self) -> bool:
        """
        Validate configuration.

        Returns
        -------
        bool
            True if valid

        Raises
        ------
        ValueError
            If configuration is invalid
        """
        if not Path(self.db_path).exists():
            raise ValueError(f"Database file not found: {self.db_path}")

        if len(self.elements) == 0:
            raise ValueError("At least one element must be specified")

        if self.wavelength_range[0] >= self.wavelength_range[1]:
            raise ValueError("Invalid wavelength range")

        if self.temperature_range[0] >= self.temperature_range[1]:
            raise ValueError("Invalid temperature range")

        if self.density_range[0] >= self.density_range[1]:
            raise ValueError("Invalid density range")

        if self.temperature_steps < 2:
            raise ValueError("temperature_steps must be >= 2")

        if self.density_steps < 2:
            raise ValueError("density_steps must be >= 2")

        if self.pixels < 10:
            raise ValueError("pixels must be >= 10")

        return True
