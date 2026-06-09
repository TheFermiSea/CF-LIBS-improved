"""
SuperCam PDS data parser and loader.

Parses SuperCam calibrated LIBS data products from the Mars 2020 PDS
archive. SuperCam products use PDS4 labeling with CSV data files.

SuperCam has three spectrometers:
- UV:   ~245-340 nm (reflection spectrometer)
- VIO:  ~385-465 nm (reflection spectrometer)
- VNIR: ~536-853 nm (transmission spectrometer)

Note: SuperCam VNIR range differs significantly from ChemCam
(starts at 536 nm vs 474 nm) due to the transmission spectrometer design.

References
----------
- Maurice et al. (2021), Space Science Reviews 217, 47
- Wiens et al. (2021), Space Science Reviews 217, 4
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("pds.supercam")


@dataclass
class SuperCamSpectrum:
    """Parsed SuperCam LIBS spectrum.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength array in nm (concatenated across all spectrometers).
    intensity : np.ndarray
        Calibrated intensity.
    product_id : str
        PDS product identifier.
    sol : int
        Mars sol number.
    target : str
        Target name.
    n_shots : int
        Number of shots averaged.
    metadata : Dict
        Additional metadata from the PDS label.
    spectrometer_ranges : List[Tuple[int, int]]
        Index ranges for each spectrometer in the concatenated arrays.
    """

    wavelength: np.ndarray
    intensity: np.ndarray
    product_id: str = ""
    sol: int = 0
    target: str = ""
    n_shots: int = 0
    metadata: Dict = field(default_factory=dict)
    spectrometer_ranges: List[Tuple[int, int]] = field(default_factory=list)


class SuperCamParser:
    """Parser for SuperCam calibrated LIBS PDS data products.

    SuperCam data uses a different format than ChemCam, with PDS4-style
    labels and potentially different CSV column layouts.

    Examples
    --------
    >>> parser = SuperCamParser()
    >>> spectrum = parser.parse("path/to/SC3_product.csv")
    """

    def parse(self, file_path: Path | str) -> SuperCamSpectrum:
        """Parse a SuperCam calibrated LIBS CSV file.

        Parameters
        ----------
        file_path : Path or str
            Path to the calibrated CSV data product.

        Returns
        -------
        SuperCamSpectrum
            Parsed spectrum with metadata.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        ValueError
            If the file format is not recognized as SuperCam LIBS.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"SuperCam file not found: {path}")

        logger.debug("Parsing SuperCam file: %s", path)

        text = path.read_text(encoding="utf-8", errors="replace")
        lines = text.strip().split("\n")

        metadata: Dict = {"source_file": str(path)}
        product_id = path.stem

        # Parse header
        data_start = self._parse_header(lines, metadata)

        # Parse numeric data
        wavelengths, intensities = self._parse_numeric_data(lines, data_start)

        if not wavelengths:
            raise ValueError(f"No spectral data found in {path}")

        wl_arr = np.array(wavelengths)
        int_arr = np.array(intensities)

        # Identify spectrometer boundaries
        spec_ranges = self._find_spectrometer_ranges(wl_arr)

        # Extract sol from product ID
        sol = self._extract_sol(product_id, metadata)

        spectrum = SuperCamSpectrum(
            wavelength=wl_arr,
            intensity=int_arr,
            product_id=product_id,
            sol=sol,
            target=metadata.get("target", ""),
            n_shots=metadata.get("n_shots", 0),
            metadata=metadata,
            spectrometer_ranges=spec_ranges,
        )

        logger.info(
            "Parsed SuperCam spectrum: %d points, %d spectrometers, sol %d",
            len(wl_arr),
            len(spec_ranges),
            sol,
        )
        return spectrum

    def _parse_header(self, lines: List[str], metadata: Dict) -> int:
        """Parse header lines into ``metadata`` and return the data start index."""
        data_start = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith('"'):
                self._parse_header_line(stripped, metadata)
                data_start = i + 1
            elif "," in stripped:
                parts = stripped.split(",")
                try:
                    float(parts[0])
                    data_start = i
                    break
                except ValueError:
                    data_start = i + 1
            else:
                data_start = i + 1
        return data_start

    def _parse_numeric_data(
        self, lines: List[str], data_start: int
    ) -> Tuple[List[float], List[float]]:
        """Parse numeric wavelength/intensity rows starting at ``data_start``."""
        wavelengths: List[float] = []
        intensities: List[float] = []

        for line in lines[data_start:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split(",")
            if len(parts) >= 2:
                try:
                    wl = float(parts[0])
                    intensity = float(parts[1])
                    wavelengths.append(wl)
                    intensities.append(intensity)
                except ValueError:
                    continue

        return wavelengths, intensities

    def _parse_header_line(self, line: str, metadata: Dict) -> None:
        """Extract metadata from a header line."""
        clean = line.lstrip("#").strip().strip('"')
        if "=" in clean:
            key, _, val = clean.partition("=")
            key = key.strip().lower().replace(" ", "_")
            val = val.strip().strip('"')
            try:
                metadata[key] = int(val)
            except ValueError:
                try:
                    metadata[key] = float(val)
                except ValueError:
                    metadata[key] = val

    def _find_spectrometer_ranges(
        self, wavelength: np.ndarray, gap_threshold_nm: float = 30.0
    ) -> List[Tuple[int, int]]:
        """Identify spectrometer boundaries from wavelength gaps.

        SuperCam has a larger gap between VIO and VNIR (~70 nm) than
        ChemCam, so we use a larger default threshold.
        """
        if len(wavelength) < 2:
            return [(0, len(wavelength))]

        diffs = np.diff(wavelength)
        gap_indices = np.where(diffs > gap_threshold_nm)[0]

        ranges = []
        start = 0
        for gap_idx in gap_indices:
            ranges.append((start, gap_idx + 1))
            start = gap_idx + 1
        ranges.append((start, len(wavelength)))

        return ranges

    def _extract_sol(self, product_id: str, metadata: Dict) -> int:
        """Extract sol number from product ID or metadata."""
        if "sol" in metadata:
            return int(metadata["sol"])

        # SuperCam product IDs: SC3_SSSS_... where SSSS is the sol
        try:
            if product_id.startswith("SC3_"):
                sol_str = product_id.split("_")[1]
                return int(sol_str)
        except (IndexError, ValueError):
            pass
        return 0
