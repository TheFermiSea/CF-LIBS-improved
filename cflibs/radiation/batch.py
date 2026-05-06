"""
Batch processing utilities for multiple spectra.
"""

from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from cflibs.radiation.spectrum_model import SpectrumModel
from cflibs.core.logging_config import get_logger

logger = get_logger("radiation.batch")


def compute_spectrum_batch(
    models: List[SpectrumModel], n_workers: Optional[int] = None, use_processes: bool = False
) -> List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]:
    """
    Compute multiple spectra in parallel.

    Parameters
    ----------
    models : List[SpectrumModel]
        List of spectrum models to compute
    n_workers : int, optional
        Number of worker threads/processes. If None, uses CPU count.
    use_processes : bool
        If True, use processes instead of threads (for CPU-bound work)

    Returns
    -------
    List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]]
        List of (wavelength, intensity) tuples. Failed spectra produce
        ``(None, None)`` so callers can filter without losing index alignment.
    """
    if not models:
        return []

    if n_workers is None:
        import os

        n_workers = os.cpu_count() or 1

    logger.info(f"Computing {len(models)} spectra with {n_workers} workers")

    executor_class = ProcessPoolExecutor if use_processes else ThreadPoolExecutor

    results = []
    with executor_class(max_workers=n_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(model.compute_spectrum): i for i, model in enumerate(models)}

        # Collect results in order. Successful entries hold (wavelength,
        # intensity) ndarrays; failed entries hold (None, None) and are
        # filtered/handled by callers.
        completed: Dict[int, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                completed[idx] = result
            except Exception as e:
                logger.error(f"Error computing spectrum {idx}: {e}")
                completed[idx] = (None, None)

        # Return in original order
        results = [completed[i] for i in range(len(models))]

    logger.info(f"Completed batch computation of {len(results)} spectra")
    return results


def _apply_params_to_plasma(plasma, params: dict) -> None:
    """Apply parameter dictionary to a plasma object.

    Parameters
    ----------
    plasma : SingleZoneLTEPlasma
        Plasma state to modify in-place
    params : dict
        Parameter values keyed by name (T_e_eV, n_e, or species names)
    """
    if "T_e_eV" in params:
        plasma.T_e_eV = params["T_e_eV"]
    if "n_e" in params:
        plasma.n_e = params["n_e"]

    for key, value in params.items():
        if key not in ("T_e_eV", "n_e") and key in plasma.species:
            plasma.species[key] = value


def compute_spectrum_grid(
    base_model: SpectrumModel,
    parameter_grid: Dict[str, List[float]],
    n_workers: Optional[int] = None,
) -> Tuple[
    List[Dict[str, float]],
    List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
]:
    """
    Compute spectra for a grid of parameter values.

    Parameters
    ----------
    base_model : SpectrumModel
        Base model to use as template
    parameter_grid : Dict[str, List[float]]
        Dictionary mapping parameter names to lists of values.
        Supported parameters: 'T_e_eV', 'n_e', and species names.
    n_workers : int, optional
        Number of worker threads

    Returns
    -------
    parameters : List[Dict[str, float]]
        List of parameter dictionaries for each spectrum
    spectra : List[Tuple[np.ndarray, np.ndarray]]
        List of (wavelength, intensity) tuples

    Example
    -------
    >>> base = SpectrumModel(...)
    >>> grid = {
    ...     'T_e_eV': [0.8, 1.0, 1.2],
    ...     'n_e': [1e16, 1e17],
    ...     'Ti': [1e15, 2e15]
    ... }
    >>> params, spectra = compute_spectrum_grid(base, grid)
    """
    from itertools import product
    from copy import deepcopy

    # Generate all parameter combinations
    param_names = list(parameter_grid.keys())
    param_values = list(parameter_grid.values())

    all_combinations = list(product(*param_values))

    logger.info(f"Computing grid with {len(all_combinations)} parameter combinations")

    # Create models for each combination
    models = []
    parameter_list = []

    for combo in all_combinations:
        params = dict(zip(param_names, combo))
        parameter_list.append(params)

        # Create new model with modified plasma
        model = deepcopy(base_model)
        plasma = model.plasma

        _apply_params_to_plasma(plasma, params)

        models.append(model)

    # Compute all spectra
    spectra = compute_spectrum_batch(models, n_workers=n_workers)

    return parameter_list, spectra


def compute_spectrum_ensemble(
    base_model: SpectrumModel,
    n_samples: int,
    parameter_distributions: Dict[str, Callable[..., float]],
    n_workers: Optional[int] = None,
) -> Tuple[
    List[Dict[str, float]],
    List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]],
]:
    """
    Compute spectra for an ensemble of random parameter samples.

    Parameters
    ----------
    base_model : SpectrumModel
        Base model to use as template
    n_samples : int
        Number of samples to generate
    parameter_distributions : Dict[str, callable]
        Dictionary mapping parameter names to distribution functions.
        Each function should take no arguments and return a float.
    n_workers : int, optional
        Number of worker threads

    Returns
    -------
    parameters : List[Dict[str, float]]
        List of sampled parameter dictionaries
    spectra : List[Tuple[np.ndarray, np.ndarray]]
        List of (wavelength, intensity) tuples

    Example
    -------
    >>> import numpy as np
    >>> base = SpectrumModel(...)
    >>> dists = {
    ...     'T_e_eV': lambda: np.random.normal(1.0, 0.1),
    ...     'n_e': lambda: 10**np.random.uniform(16, 18)
    ... }
    >>> params, spectra = compute_spectrum_ensemble(base, 100, dists)
    """
    from copy import deepcopy

    logger.info(f"Generating ensemble with {n_samples} samples")

    # Generate parameter samples
    parameter_list = []
    models = []

    for i in range(n_samples):
        params = {}
        for name, dist_func in parameter_distributions.items():
            params[name] = dist_func()
        parameter_list.append(params)

        # Create model with sampled parameters
        model = deepcopy(base_model)
        plasma = model.plasma

        _apply_params_to_plasma(plasma, params)

        models.append(model)

    # Compute all spectra
    spectra = compute_spectrum_batch(models, n_workers=n_workers)

    return parameter_list, spectra
