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
from cflibs.pds._csv_spectrum import parse_metadata_csv

logger = get_logger("pds.supercam")

# SuperCam has a larger VIO->VNIR gap (~70 nm) than ChemCam, so it uses a
# larger threshold to avoid splitting on small within-spectrometer gaps.
_GAP_THRESHOLD_NM = 30.0


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
        parsed = parse_metadata_csv(text, gap_threshold_nm=_GAP_THRESHOLD_NM)

        if parsed.wavelength.size == 0:
            raise ValueError(f"No spectral data found in {path}")

        metadata: Dict = {"source_file": str(path), **parsed.metadata}
        product_id = path.stem

        wl_arr = parsed.wavelength
        int_arr = parsed.intensity
        spec_ranges = parsed.spectrometer_ranges

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
