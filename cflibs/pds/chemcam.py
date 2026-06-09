"""
ChemCam PDS data parser and loader.

Parses ChemCam CL5 (calibrated Level 5) LIBS data products from the
MSL PDS archive. CL5 products contain mean spectra averaged over
multiple laser shots, with wavelength calibration and instrument
response correction already applied.

ChemCam has three spectrometers:
- UV:   ~240-342 nm (Body unit)
- VIO:  ~382-469 nm (Body unit)
- VNIR: ~474-906 nm (Body unit)

File format: CSV with header rows containing metadata, followed by
wavelength,intensity pairs for each spectrometer.

References
----------
- Wiens et al. (2012), Space Science Reviews 170, 167-227
- Maurice et al. (2012), Space Science Reviews 170, 95-166
- ChemCam PDS SIS: msl_ccam_sis.pdf
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.pds._csv_spectrum import parse_metadata_csv

logger = get_logger("pds.chemcam")

# ChemCam has a ~40 nm gap (UV->VIO) and a ~5 nm gap (VIO->VNIR), so the
# threshold is set low enough to catch the smaller of the two gaps.
_GAP_THRESHOLD_NM = 3.0


@dataclass
class ChemCamSpectrum:
    """Parsed ChemCam LIBS spectrum.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength array in nm (concatenated across all spectrometers).
    intensity : np.ndarray
        Calibrated intensity (photon counts or radiance, depending on level).
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


class ChemCamParser:
    """Parser for ChemCam CL5 PDS data products.

    Handles the CSV format used in the MSL ChemCam PDS archive.
    The CL5 products contain calibrated, averaged spectra.

    Examples
    --------
    >>> parser = ChemCamParser()
    >>> spectrum = parser.parse("path/to/CL5_product.csv")
    >>> print(spectrum.wavelength.shape, spectrum.intensity.shape)
    """

    def parse(self, file_path: Path | str) -> ChemCamSpectrum:
        """Parse a ChemCam CL5 CSV file.

        Parameters
        ----------
        file_path : Path or str
            Path to the CL5 CSV data product.

        Returns
        -------
        ChemCamSpectrum
            Parsed spectrum with metadata.

        Raises
        ------
        FileNotFoundError
            If the file doesn't exist.
        ValueError
            If the file format is not recognized as ChemCam CL5.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"ChemCam file not found: {path}")

        logger.debug("Parsing ChemCam file: %s", path)

        # Read the file and parse the shared metadata-CSV layout.
        text = path.read_text(encoding="utf-8", errors="replace")
        parsed = parse_metadata_csv(text, gap_threshold_nm=_GAP_THRESHOLD_NM)

        if parsed.wavelength.size == 0:
            raise ValueError(f"No spectral data found in {path}")

        metadata: Dict = {"source_file": str(path), **parsed.metadata}
        product_id = path.stem

        wl_arr = parsed.wavelength
        int_arr = parsed.intensity
        spec_ranges = parsed.spectrometer_ranges

        # Extract metadata
        sol = metadata.get("sol", 0)
        target = metadata.get("target", "")
        n_shots = metadata.get("n_shots", 0)

        # Try to extract sol from product_id if not in metadata
        if sol == 0 and "sol" not in metadata:
            sol = self._extract_sol_from_product_id(product_id)

        spectrum = ChemCamSpectrum(
            wavelength=wl_arr,
            intensity=int_arr,
            product_id=product_id,
            sol=sol,
            target=target,
            n_shots=n_shots,
            metadata=metadata,
            spectrometer_ranges=spec_ranges,
        )

        logger.info(
            "Parsed ChemCam spectrum: %d points, %d spectrometers, sol %d",
            len(wl_arr),
            len(spec_ranges),
            sol,
        )
        return spectrum

    def _extract_sol_from_product_id(self, product_id: str) -> int:
        """Try to extract sol number from product ID naming convention.

        ChemCam CL5 product IDs follow the pattern:
            CL5_<sclk>RCE_F<seq>CCAM<TT><SSS>P1
        where TT is the target number (2 digits) and SSS is the sol
        (variable-width integer before the trailing ``P1``).
        """
        try:
            # Match: CCAM<target_digits><sol_digits>P<version>
            match = re.search(r"CCAM(\d{2})(\d+)P\d", product_id)
            if match:
                return int(match.group(2))
        except (IndexError, ValueError):
            pass
        return 0
