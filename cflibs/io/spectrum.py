"""
I/O utilities for spectra.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple

from cflibs.core.logging_config import get_logger

logger = get_logger("io.spectrum")


def _find_column(df: pd.DataFrame, candidates: list[str], kind: str) -> str:
    """Return the first candidate column present in ``df`` or raise.

    Parameters
    ----------
    df : pandas.DataFrame
        Loaded CSV data.
    candidates : list of str
        Candidate column names, in priority order.
    kind : str
        Human-readable column kind used in the error message.
    """
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(f"Could not find {kind} column in CSV")


def _load_spectrum_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a spectrum from a CSV file by resolving common column names."""
    df = pd.read_csv(path, comment="#")
    # Try common column names
    wl_col = _find_column(
        df, ["wavelength", "wavelength_nm", "wl", "lambda", "lambda_nm"], "wavelength"
    )
    int_col = _find_column(
        df,
        [
            "intensity",
            "intensity_W_m2_nm_sr",
            "I",
            "counts",
            "signal",
            "spectrum",
            "flux",
        ],
        "intensity",
    )
    return df[wl_col].values, df[int_col].values


def _load_spectrum_text(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a spectrum from a whitespace-delimited text file via numpy."""
    # Try numpy loadtxt as fallback
    data = np.loadtxt(path)
    if data.ndim == 1:
        raise ValueError("Spectrum file must have at least 2 columns")
    return data[:, 0], data[:, 1]


def load_spectrum(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load spectrum from file.

    Supports CSV files with columns: wavelength, intensity

    Parameters
    ----------
    file_path : str
        Path to spectrum file

    Returns
    -------
    wavelength : array
        Wavelength array in nm
    intensity : array
        Intensity array
    """
    path = Path(file_path)

    if path.suffix.lower() == ".csv":
        wavelength, intensity = _load_spectrum_csv(path)
    else:
        wavelength, intensity = _load_spectrum_text(path)

    logger.info(f"Loaded spectrum from {path}: {len(wavelength)} points")
    return wavelength, intensity


def save_spectrum(
    file_path: str,
    wavelength: np.ndarray,
    intensity: np.ndarray,
    header: Optional[str] = None,
) -> None:
    """
    Save spectrum to file.

    Parameters
    ----------
    file_path : str
        Output file path
    wavelength : array
        Wavelength array in nm
    intensity : array
        Intensity array
    header : str, optional
        Header comment
    """
    path = Path(file_path)

    if path.suffix.lower() == ".csv":
        if header is None:
            header = "wavelength_nm,intensity"

        np.savetxt(
            path,
            np.column_stack([wavelength, intensity]),
            delimiter=",",
            header=header,
            comments="",
        )
    else:
        # Default to space-separated
        np.savetxt(
            path,
            np.column_stack([wavelength, intensity]),
            header=header if header is not None else "",
        )

    logger.info(f"Saved spectrum to {path}")
