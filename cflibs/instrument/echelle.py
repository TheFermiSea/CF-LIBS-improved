"""
Echellogram extraction for 2D spectral images.

This module provides tools for extracting 1D spectra from 2D echellogram images
produced by echelle spectrometers (e.g., Andor Mechelle 5000).
"""

from typing import Dict, Tuple, Optional, List
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import interp1d

from cflibs.core.logging_config import get_logger

logger = get_logger("instrument.echelle")


class EchelleExtractor:
    """
    Extracts 1D spectra from 2D echellogram images.

    The Andor Mechelle 5000 and similar echelle spectrometers produce 2D images
    where multiple spectral orders are stacked vertically. This class implements
    a "trace and sum" extraction algorithm to convert these into 1D spectra.

    Algorithm:
    1. Order Tracing: Each order center is defined by a polynomial y_m(x)
    2. Flux Extraction: Sum intensity in a window around the trace
    3. Wavelength Mapping: Map pixel position to wavelength using calibration
    4. Order Merging: Interpolate all orders onto a common wavelength grid

    Attributes
    ----------
    orders : Dict[str, Dict]
        Dictionary mapping order names to calibration coefficients
        Format: {"order_N": {"y_coeffs": [...], "wl_coeffs": [...]}}
    extraction_window : int
        Number of pixels above/below trace center to include in extraction
    """

    def __init__(self, calibration_file: Optional[str] = None, extraction_window: int = 5):
        """
        Initialize echelle extractor.

        Parameters
        ----------
        calibration_file : str, optional
            Path to JSON calibration file containing order polynomials
        extraction_window : int
            Number of pixels above/below trace center to extract (default: 5)
        """
        self.orders: Dict[str, Dict] = {}
        self.extraction_window = extraction_window

        if calibration_file:
            self.load_calibration(calibration_file)

    def load_calibration(self, filepath: str) -> None:
        """
        Load order trace and wavelength calibration polynomials.

        The calibration file should be a JSON file with the following format:
        {
            "order_50": {
                "y_coeffs": [c0, c1, c2],  # Polynomial coefficients for y(x)
                "wl_coeffs": [k0, k1, k2]  # Polynomial coefficients for λ(x)
            },
            "order_49": {...},
            ...
        }

        Note: Polynomial coefficients are in order [highest_order, ..., lowest_order]
        as expected by numpy.polyval.

        Parameters
        ----------
        filepath : str
            Path to JSON calibration file

        Raises
        ------
        FileNotFoundError
            If calibration file does not exist
        ValueError
            If calibration file format is invalid
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        try:
            with open(path, "r") as f:
                self.orders = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in calibration file: {e}")

        # Validate calibration structure
        for order_name, coeffs in self.orders.items():
            if "y_coeffs" not in coeffs or "wl_coeffs" not in coeffs:
                raise ValueError(
                    f"Invalid calibration format for {order_name}: "
                    "must contain 'y_coeffs' and 'wl_coeffs'"
                )

        logger.info(f"Loaded calibration for {len(self.orders)} orders from {path}")

    def save_calibration(self, filepath: str) -> None:
        """
        Save calibration to JSON file.

        Parameters
        ----------
        filepath : str
            Output file path
        """
        path = Path(filepath)
        with open(path, "w") as f:
            json.dump(self.orders, f, indent=2)
        logger.info(f"Saved calibration to {path}")

    def extract_order(
        self, image_2d: np.ndarray, order_name: str, background_subtract: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract spectrum from a single order.

        Parameters
        ----------
        image_2d : array
            2D image array (height, width)
        order_name : str
            Name of the order to extract
        background_subtract : bool
            Whether to subtract background (default: True)

        Returns
        -------
        wavelengths : array
            Wavelength array in nm
        flux : array
            Extracted flux array

        Raises
        ------
        KeyError
            If order_name is not in calibration
        """
        if order_name not in self.orders:
            raise KeyError(f"Order '{order_name}' not found in calibration")

        coeffs = self.orders[order_name]
        height, width = image_2d.shape
        x_pixels = np.arange(width)

        # Calculate trace position y(x)
        y_coeffs = coeffs["y_coeffs"]
        y_trace = np.polyval(y_coeffs, x_pixels)

        # Calculate wavelength solution λ(x)
        wl_coeffs = coeffs["wl_coeffs"]
        wl_trace = np.polyval(wl_coeffs, x_pixels)

        # Extract flux by summing pixels in window around trace
        flux = np.zeros(width, dtype=np.float32)

        for i, x in enumerate(x_pixels):
            y_center = int(np.round(y_trace[i]))
            y_min = max(0, y_center - self.extraction_window)
            y_max = min(height, y_center + self.extraction_window + 1)

            if y_min >= y_max:
                continue

            # Sum vertical column
            flux[i] = np.sum(image_2d[y_min:y_max, x])

            # Background subtraction (simple: use median of nearby pixels)
            if background_subtract:
                flux[i] -= self._estimate_column_background(
                    image_2d, height, x, y_center, y_min, y_max
                )

        return wl_trace, flux

    def _estimate_column_background(
        self,
        image_2d: np.ndarray,
        height: int,
        x: int,
        y_center: int,
        y_min: int,
        y_max: int,
    ) -> float:
        """
        Estimate background contribution for a single extracted column.

        Uses the median of pixels outside the extraction window, scaled by the
        number of summed pixels. Returns 0.0 when no background pixels remain.
        """
        # Use pixels outside the extraction window
        bg_y_min = max(0, y_center - 2 * self.extraction_window)
        bg_y_max = min(height, y_center + 2 * self.extraction_window + 1)
        bg_mask = np.ones(bg_y_max - bg_y_min, dtype=bool)
        if y_min - bg_y_min > 0:
            bg_mask[y_min - bg_y_min : y_max - bg_y_min] = False
        if bg_y_max - y_max > 0:
            bg_mask[y_max - bg_y_min : bg_y_max - bg_y_min] = False

        if np.any(bg_mask):
            bg_pixels = image_2d[bg_y_min:bg_y_max, x][bg_mask]
            return float(np.median(bg_pixels) * (y_max - y_min))
        return 0.0

    def extract_spectrum(
        self,
        image_2d: np.ndarray,
        wavelength_step_nm: float = 0.05,
        merge_method: str = "weighted_average",
        min_valid_pixels: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract complete 1D spectrum from 2D echellogram.

        This method:
        1. Extracts flux from each order
        2. Interpolates all orders onto a common wavelength grid
        3. Merges overlapping regions

        Parameters
        ----------
        image_2d : array
            2D echellogram image (height, width)
        wavelength_step_nm : float
            Wavelength step for output grid in nm (default: 0.05)
        merge_method : str
            Method for merging overlapping orders:
            - 'weighted_average': Weight by number of contributing orders
            - 'simple_average': Simple average
            - 'max': Take maximum value
        min_valid_pixels : int
            Minimum number of valid pixels required for an order to be included

        Returns
        -------
        wavelengths : array
            Wavelength grid in nm
        intensity : array
            Merged intensity spectrum

        Raises
        ------
        ValueError
            If no orders are calibrated or merge_method is invalid
        """
        if not self.orders:
            raise ValueError("No orders calibrated. Load calibration first.")

        if merge_method not in ["weighted_average", "simple_average", "max"]:
            raise ValueError(
                f"Invalid merge_method: {merge_method}. "
                "Must be 'weighted_average', 'simple_average', or 'max'"
            )

        # Extract all orders
        extracted_orders = self._extract_valid_orders(image_2d, min_valid_pixels)

        if not extracted_orders:
            raise ValueError("No orders could be extracted")

        # Create master wavelength grid
        all_wls = np.concatenate([o[0] for o in extracted_orders])
        min_wl = np.min(all_wls)
        max_wl = np.max(all_wls)

        master_grid = np.arange(min_wl, max_wl + wavelength_step_nm, wavelength_step_nm)
        master_flux = np.zeros_like(master_grid)
        weights = np.zeros_like(master_grid)

        # Interpolate each order onto master grid
        for wl_arr, flux_arr in extracted_orders:
            self._merge_order_onto_grid(
                wl_arr,
                flux_arr,
                master_grid,
                master_flux,
                weights,
                merge_method,
                min_valid_pixels,
            )

        # Normalize
        if merge_method in ["weighted_average", "simple_average"]:
            weights[weights == 0] = 1.0  # Avoid division by zero
            final_spectrum = master_flux / weights
        else:  # max
            final_spectrum = master_flux

        logger.info(
            f"Extracted spectrum: {len(master_grid)} points, "
            f"λ=[{min_wl:.1f}, {max_wl:.1f}] nm, "
            f"{len(extracted_orders)} orders merged"
        )

        return master_grid, final_spectrum

    def _extract_valid_orders(
        self, image_2d: np.ndarray, min_valid_pixels: int
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Extract every calibrated order, keeping only those with enough valid pixels.

        Orders are processed in sorted name order. Orders that raise during
        extraction, or that have fewer than ``min_valid_pixels`` positive-flux
        pixels, are skipped.
        """
        extracted_orders: List[Tuple[np.ndarray, np.ndarray]] = []

        for order_name in sorted(self.orders.keys()):
            try:
                wl_arr, flux_arr = self.extract_order(image_2d, order_name)

                # Filter out orders with insufficient valid data
                valid_mask = flux_arr > 0
                if np.sum(valid_mask) < min_valid_pixels:
                    logger.debug(f"Skipping order {order_name}: insufficient valid pixels")
                    continue

                extracted_orders.append((wl_arr, flux_arr))
                logger.debug(f"Extracted order {order_name}: {np.sum(valid_mask)} valid pixels")
            except Exception as e:
                logger.warning(f"Failed to extract order {order_name}: {e}")
                continue

        return extracted_orders

    def _merge_order_onto_grid(
        self,
        wl_arr: np.ndarray,
        flux_arr: np.ndarray,
        master_grid: np.ndarray,
        master_flux: np.ndarray,
        weights: np.ndarray,
        merge_method: str,
        min_valid_pixels: int,
    ) -> None:
        """
        Resample a single order onto the master grid and accumulate it in place.

        Mutates ``master_flux`` and ``weights`` according to ``merge_method``.
        Orders with too few valid pixels, or that fail to interpolate, are skipped.
        """
        # Only interpolate where flux is valid
        mask = flux_arr > 0
        if np.sum(mask) < min_valid_pixels:
            return

        try:
            interp_func = interp1d(
                wl_arr[mask], flux_arr[mask], kind="linear", bounds_error=False, fill_value=0.0
            )
            resampled_flux = interp_func(master_grid)

            # Apply merge method
            if merge_method in ("weighted_average", "simple_average"):
                order_weight = (resampled_flux > 0).astype(float)
                master_flux += resampled_flux
                weights += order_weight
            elif merge_method == "max":
                # Take maximum where multiple orders contribute
                mask_overlap = resampled_flux > 0
                master_flux[mask_overlap] = np.maximum(
                    master_flux[mask_overlap], resampled_flux[mask_overlap]
                )
                weights[mask_overlap] = np.maximum(weights[mask_overlap], 1.0)
        except Exception as e:
            logger.warning(f"Failed to interpolate order: {e}")
            return

    def create_mock_calibration(
        self,
        width: int = 2048,
        num_orders: int = 3,
        wavelength_range: Tuple[float, float] = (300.0, 400.0),
    ) -> None:
        """
        Generate mock calibration for testing.

        Creates a simple calibration with polynomial traces and linear
        wavelength dispersion. Useful for testing without real calibration data.

        Parameters
        ----------
        width : int
            Image width in pixels
        num_orders : int
            Number of orders to simulate
        wavelength_range : tuple
            (min_wavelength, max_wavelength) in nm
        """
        wl_min, wl_max = wavelength_range
        wl_per_order = (wl_max - wl_min) / num_orders

        self.orders = {}
        y_start = 500

        for i in range(num_orders):
            order_num = 50 - i
            order_name = str(order_num)

            # Curved trace: y(x) = a*x^2 + b*x + c
            y_coeffs = [
                0.0001,  # quadratic term
                0.1,  # linear term
                y_start + i * 200,  # constant (vertical offset)
            ]

            # Linear wavelength: λ(x) = k*x + λ0
            wl_coeffs = [
                wl_per_order / width,  # dispersion
                wl_min + i * wl_per_order,  # starting wavelength
            ]

            self.orders[order_name] = {"y_coeffs": y_coeffs, "wl_coeffs": wl_coeffs}

        logger.info(
            f"Created mock calibration: {num_orders} orders, " f"λ=[{wl_min:.1f}, {wl_max:.1f}] nm"
        )
