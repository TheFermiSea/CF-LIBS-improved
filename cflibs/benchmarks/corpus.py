"""
Synthetic test corpus generator for CF-LIBS benchmark experiments.

Generates LIBS spectra with known ground truth for systematic evaluation
of inversion pipelines.  Three generation strategies are attempted in
order:

1. **Forward model** -- uses :class:`cflibs.radiation.spectrum_model.SpectrumModel`
   with the project's atomic database (requires DB on disk).
2. **Saved fixtures** -- loads pre-saved NumPy arrays from ``tests/fixtures/``.
3. **Gaussian fallback** -- places Gaussian peaks at tabulated wavelengths
   with intensities derived from simplified Boltzmann statistics.  No
   external data required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkSpectrum:
    """A single synthetic spectrum with known ground truth.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength grid in nm.
    intensity : np.ndarray
        Spectral intensity (arbitrary units).
    ground_truth : Dict
        Known parameters: ``temperature_K``, ``electron_density_cm3``,
        ``concentrations`` (Dict[str, float] summing to 1).
    label : str
        Human-readable label (e.g. ``"T=10000K_ne=1e17_clean"``).
    snr : Optional[float]
        Signal-to-noise ratio (``None`` for clean spectra).
    metadata : Dict
        Arbitrary additional info.
    """

    wavelength: np.ndarray
    intensity: np.ndarray
    ground_truth: Dict
    label: str = ""
    snr: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Reference line tables for the Gaussian fallback
# ---------------------------------------------------------------------------

# Prominent lines for common LIBS elements:
#   (wavelength_nm, upper_level_eV, degeneracy, log10_A_ki)
_REFERENCE_LINES: Dict[str, List[Tuple[float, float, int, float]]] = {
    "Fe": [
        (238.20, 5.22, 9, 7.23),
        (239.56, 5.18, 7, 7.10),
        (240.49, 5.07, 5, 6.90),
        (248.33, 5.09, 9, 7.30),
        (252.28, 4.99, 7, 7.08),
        (259.94, 4.77, 11, 7.40),
        (271.90, 4.61, 9, 7.15),
        (275.57, 4.60, 7, 7.00),
        (302.06, 4.11, 9, 6.85),
        (358.12, 4.42, 9, 6.90),
        (373.49, 4.18, 11, 6.88),
        (382.04, 4.10, 9, 6.62),
    ],
    "Cu": [
        (324.75, 3.82, 4, 7.89),
        (327.40, 3.79, 2, 7.59),
        (217.89, 5.69, 4, 7.60),
        (219.96, 5.64, 2, 7.27),
        (249.21, 5.07, 4, 6.40),
        (282.44, 7.74, 4, 7.20),
    ],
    "Al": [
        (308.22, 4.02, 4, 7.65),
        (309.27, 4.02, 2, 7.35),
        (394.40, 3.14, 4, 7.70),
        (396.15, 3.14, 2, 7.40),
        (236.71, 5.24, 4, 7.10),
        (237.31, 5.22, 2, 6.80),
    ],
    "Ti": [
        (334.94, 3.70, 11, 7.50),
        (336.12, 3.69, 9, 7.42),
        (337.28, 3.68, 7, 7.33),
        (363.55, 3.44, 9, 6.90),
        (364.27, 3.43, 7, 6.80),
        (365.35, 3.43, 5, 6.70),
    ],
    "Ni": [
        (341.48, 3.66, 9, 7.30),
        (344.63, 3.61, 7, 7.10),
        (349.30, 3.54, 5, 6.90),
        (351.51, 3.54, 7, 7.15),
        (352.45, 3.52, 5, 7.00),
        (361.94, 3.42, 3, 6.65),
    ],
    "Cr": [
        (357.87, 3.46, 9, 7.30),
        (359.35, 3.44, 7, 7.20),
        (360.53, 3.42, 5, 7.08),
        (425.44, 2.91, 9, 7.50),
        (427.48, 2.90, 7, 7.40),
        (428.97, 2.89, 5, 7.27),
    ],
    "Mn": [
        (257.61, 4.81, 8, 7.50),
        (259.37, 4.77, 6, 7.40),
        (260.57, 4.75, 4, 7.25),
        (403.08, 3.07, 8, 7.70),
        (403.31, 3.07, 6, 7.53),
        (403.45, 3.07, 4, 7.30),
    ],
}

# Boltzmann constant in eV/K (avoid importing cflibs.core.constants so
# this module stays self-contained for the fallback path).
_KB_EV = 8.617333262e-5


# ---------------------------------------------------------------------------
# Corpus builder
# ---------------------------------------------------------------------------


class BenchmarkCorpus:
    """Generate a collection of :class:`BenchmarkSpectrum` instances.

    Parameters
    ----------
    wavelength_range : Tuple[float, float]
        (min, max) wavelength in nm.
    delta_lambda : float
        Wavelength step in nm.
    temperatures_K : Sequence[float]
        Temperatures to sweep.
    electron_densities_cm3 : Sequence[float]
        Electron densities to sweep.
    compositions : Sequence[Dict[str, float]]
        Composition dicts to sweep (each should sum to ~1).
    snr_values : Optional[Sequence[float]]
        SNR values for noisy variants.  ``None`` means clean only.
    missing_element_specs : Optional[Sequence[Dict[str, float]]]
        Additional compositions with intentionally missing elements
        (for dark-element testing).
    seed : int
        Base random seed for reproducibility.
    """

    def __init__(
        self,
        wavelength_range: Tuple[float, float] = (200.0, 450.0),
        delta_lambda: float = 0.02,
        temperatures_K: Sequence[float] = (8000.0, 10000.0, 12000.0, 15000.0),
        electron_densities_cm3: Sequence[float] = (1e16, 1e17, 1e18),
        compositions: Optional[Sequence[Dict[str, float]]] = None,
        snr_values: Optional[Sequence[float]] = None,
        missing_element_specs: Optional[Sequence[Dict[str, float]]] = None,
        seed: int = 42,
    ):
        self.wavelength_range = wavelength_range
        self.delta_lambda = delta_lambda
        self.temperatures_K = list(temperatures_K)
        self.electron_densities_cm3 = list(electron_densities_cm3)
        self.snr_values: List[Optional[float]] = (
            list(snr_values) if snr_values is not None else [None]
        )
        self.seed = seed

        if compositions is None:
            self.compositions: List[Dict[str, float]] = [
                {"Fe": 0.70, "Cu": 0.20, "Al": 0.10},
                {"Fe": 0.50, "Ni": 0.30, "Cr": 0.20},
                {"Ti": 0.60, "Al": 0.25, "Fe": 0.15},
            ]
        else:
            self.compositions = list(compositions)

        if missing_element_specs is not None:
            self.missing_element_specs = list(missing_element_specs)
        else:
            # Default dark-element test: Fe-Cu binary labeled as Fe-Cu-Al
            # so that Al is a "dark" element (present in label but absent from
            # the actual spectrum).
            self.missing_element_specs = [
                {"Fe": 0.70, "Cu": 0.30},  # Al missing
            ]

        self.wavelength = np.arange(
            wavelength_range[0],
            wavelength_range[1] + delta_lambda,
            delta_lambda,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> List[BenchmarkSpectrum]:
        """Generate the full corpus.

        Tries the forward-model path first, falls back to Gaussian peaks
        if the atomic database is not available.

        Returns
        -------
        List[BenchmarkSpectrum]
            All generated spectra (clean + noisy variants).
        """
        spectra: List[BenchmarkSpectrum] = []

        # Pick generation backend
        generator = self._try_forward_model() or self._try_fixtures() or self._gaussian_fallback

        rng = np.random.default_rng(self.seed)

        # Main parameter sweep
        for comp in self.compositions:
            for T in self.temperatures_K:
                for ne in self.electron_densities_cm3:
                    base_wl, base_int = generator(T, ne, comp)
                    base_label = self._make_label(T, ne, comp, snr=None)
                    ground_truth = {
                        "temperature_K": T,
                        "electron_density_cm3": ne,
                        "concentrations": dict(comp),
                    }
                    for snr in self.snr_values:
                        if snr is None:
                            spectra.append(
                                BenchmarkSpectrum(
                                    wavelength=base_wl.copy(),
                                    intensity=base_int.copy(),
                                    ground_truth=ground_truth,
                                    label=base_label,
                                    snr=None,
                                )
                            )
                        else:
                            noisy = self._add_noise(base_int, snr, rng)
                            spectra.append(
                                BenchmarkSpectrum(
                                    wavelength=base_wl.copy(),
                                    intensity=noisy,
                                    ground_truth=ground_truth,
                                    label=self._make_label(T, ne, comp, snr),
                                    snr=snr,
                                )
                            )

        # Dark-element spectra
        for comp in self.missing_element_specs:
            T = self.temperatures_K[len(self.temperatures_K) // 2]
            ne = self.electron_densities_cm3[len(self.electron_densities_cm3) // 2]
            base_wl, base_int = generator(T, ne, comp)
            ground_truth = {
                "temperature_K": T,
                "electron_density_cm3": ne,
                "concentrations": dict(comp),
                "dark_element_test": True,
            }
            spectra.append(
                BenchmarkSpectrum(
                    wavelength=base_wl.copy(),
                    intensity=base_int.copy(),
                    ground_truth=ground_truth,
                    label=f"dark_elem_{'-'.join(sorted(comp.keys()))}",
                    metadata={"dark_element_test": True},
                )
            )

        return spectra

    # ------------------------------------------------------------------
    # Generation backends
    # ------------------------------------------------------------------

    def _try_forward_model(self):
        """Attempt to build a generator backed by SpectrumModel + AtomicDatabase."""
        try:
            from cflibs.radiation.spectrum_model import SpectrumModel
            from cflibs.plasma.state import SingleZoneLTEPlasma
            from cflibs.atomic.database import AtomicDatabase
            from cflibs.instrument.model import InstrumentModel

            # Try the default production database path
            db_paths = [
                Path("ASD_da/libs_production.db"),
                Path("ASD_da/libs_v2.db"),
            ]
            db = None
            for p in db_paths:
                if p.exists():
                    db = AtomicDatabase(str(p))
                    break
            if db is None:
                return None

            instrument = InstrumentModel(fwhm_nm=0.05)

            def _gen(T: float, ne: float, comp: Dict[str, float]):
                total_density = ne  # rough approximation
                plasma = SingleZoneLTEPlasma.from_number_fractions(
                    T_e=T,
                    n_e=ne,
                    number_fractions=comp,
                    total_species_density_cm3=total_density,
                )
                model = SpectrumModel(
                    plasma=plasma,
                    atomic_db=db,
                    instrument=instrument,
                    lambda_min=self.wavelength_range[0],
                    lambda_max=self.wavelength_range[1],
                    delta_lambda=self.delta_lambda,
                )
                return model.compute_spectrum()

            return _gen

        except Exception:
            return None

    def _try_fixtures(self):
        """Attempt to load pre-saved fixture spectra."""
        fixtures_dir = Path("tests/fixtures/benchmark_spectra")
        if not fixtures_dir.exists():
            return None

        try:
            index_file = fixtures_dir / "index.json"
            if not index_file.exists():
                return None
            import json

            with open(index_file) as f:
                index = json.load(f)

            def _gen(T: float, ne: float, comp: Dict[str, float]):
                key = f"T{T:.0f}_ne{ne:.0e}_{'_'.join(sorted(comp.keys()))}"
                if key in index:
                    data = np.load(fixtures_dir / index[key])
                    return data["wavelength"], data["intensity"]
                # Fall through to None → will use Gaussian fallback
                return self._gaussian_fallback(T, ne, comp)

            return _gen

        except Exception:
            return None

    def _gaussian_fallback(
        self,
        T: float,
        ne: float,
        comp: Dict[str, float],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate a spectrum from tabulated Gaussian peaks.

        Uses simplified Boltzmann statistics:
            I ~ C * g_k * A_ki * exp(-E_k / kT)
        convolved with a Gaussian of width proportional to the instrument
        resolution.
        """
        wl = self.wavelength
        intensity = np.zeros_like(wl)
        sigma_nm = 0.03  # ~0.07 nm FWHM

        T_eV = T * _KB_EV

        for element, fraction in comp.items():
            lines = _REFERENCE_LINES.get(element, [])
            for wl_center, E_k, g_k, log_A in lines:
                if wl_center < wl[0] or wl_center > wl[-1]:
                    continue
                A_ki = 10.0**log_A
                boltzmann = np.exp(-E_k / T_eV)
                peak_intensity = fraction * g_k * A_ki * boltzmann
                intensity += peak_intensity * np.exp(-0.5 * ((wl - wl_center) / sigma_nm) ** 2)

        # Normalize to have a reasonable max (like real spectra)
        max_val = intensity.max()
        if max_val > 0:
            intensity = intensity / max_val * 1e4

        return wl, intensity

    # ------------------------------------------------------------------
    # Noise injection
    # ------------------------------------------------------------------

    @staticmethod
    def _add_noise(
        intensity: np.ndarray,
        snr: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Add Gaussian white noise to achieve a target peak SNR.

        Parameters
        ----------
        intensity : np.ndarray
            Clean intensity.
        snr : float
            Desired signal-to-noise ratio (peak_signal / noise_std).
        rng : np.random.Generator
            Random number generator.

        Returns
        -------
        np.ndarray
            Noisy intensity (clipped to non-negative).
        """
        peak = intensity.max()
        if peak <= 0 or snr <= 0:
            return intensity.copy()
        noise_std = peak / snr
        noisy = intensity + rng.normal(0, noise_std, size=intensity.shape)
        return np.clip(noisy, 0.0, None)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_label(
        T: float,
        ne: float,
        comp: Dict[str, float],
        snr: Optional[float],
    ) -> str:
        elements = "-".join(sorted(comp.keys()))
        snr_str = "clean" if snr is None else f"SNR{snr:.0f}"
        return f"T={T:.0f}K_ne={ne:.0e}_{elements}_{snr_str}"
