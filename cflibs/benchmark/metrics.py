"""
Evaluation metrics for LIBS benchmark comparison.

This module provides standardized metrics for comparing algorithm performance
on LIBS benchmark datasets, supporting both CF-LIBS and ML-based approaches.

Metrics Overview
----------------
- RMSEP: Root Mean Square Error of Prediction (most common in LIBS literature)
- MAE: Mean Absolute Error (robust to outliers)
- MAPE: Mean Absolute Percentage Error (relative performance)
- Bias: Systematic over/under-estimation
- R-squared: Coefficient of determination
- LOD: Limit of Detection estimation

References
----------
- Hahn & Omenetto (2010): Standardized LIBS metrics
- Tognoni et al. (2010): CF-LIBS evaluation criteria
- Chaney et al. (2023): ML-LIBS benchmarking recommendations
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("benchmark.metrics")


class MetricType(Enum):
    """Types of evaluation metrics."""

    RMSEP = "rmsep"  # Root Mean Square Error of Prediction
    MAE = "mae"  # Mean Absolute Error
    MAPE = "mape"  # Mean Absolute Percentage Error
    BIAS = "bias"  # Mean bias (signed)
    R_SQUARED = "r_squared"  # Coefficient of determination
    PEARSON_R = "pearson_r"  # Pearson correlation coefficient
    LOD = "lod"  # Limit of Detection (3-sigma)
    LOQ = "loq"  # Limit of Quantification (10-sigma)
    RELATIVE_RMSEP = "relative_rmsep"  # RMSEP / mean true value


@dataclass
class ElementMetrics:
    """
    Evaluation metrics for a single element.

    Attributes
    ----------
    element : str
        Element symbol
    n_samples : int
        Number of samples evaluated
    rmsep : float
        Root Mean Square Error of Prediction (mass fraction)
    mae : float
        Mean Absolute Error (mass fraction)
    mape : float
        Mean Absolute Percentage Error (%)
    bias : float
        Mean bias (predicted - true) in mass fraction
    r_squared : float
        Coefficient of determination
    pearson_r : float
        Pearson correlation coefficient
    lod : float
        Estimated Limit of Detection (3-sigma, mass fraction)
    loq : float
        Estimated Limit of Quantification (10-sigma, mass fraction)
    relative_rmsep : float
        RMSEP relative to mean true concentration
    true_range : Tuple[float, float]
        Range of true values (min, max)
    """

    element: str
    n_samples: int
    rmsep: float
    mae: float
    mape: float
    bias: float
    r_squared: float
    pearson_r: float
    lod: float
    loq: float
    relative_rmsep: float
    true_range: Tuple[float, float]

    @property
    def rmsep_pct(self) -> float:
        """RMSEP as percentage (wt%)."""
        return self.rmsep * 100

    @property
    def mae_pct(self) -> float:
        """MAE as percentage (wt%)."""
        return self.mae * 100

    @property
    def bias_pct(self) -> float:
        """Bias as percentage (wt%)."""
        return self.bias * 100

    @property
    def lod_ppm(self) -> float:
        """LOD in parts per million."""
        return self.lod * 1e6

    def summary(self) -> str:
        """Generate summary string."""
        return (
            f"{self.element}: RMSEP={self.rmsep_pct:.3f}%, "
            f"MAE={self.mae_pct:.3f}%, Bias={self.bias_pct:+.3f}%, "
            f"R^2={self.r_squared:.3f}, LOD={self.lod_ppm:.0f} ppm"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "element": self.element,
            "n_samples": self.n_samples,
            "rmsep": self.rmsep,
            "mae": self.mae,
            "mape": self.mape,
            "bias": self.bias,
            "r_squared": self.r_squared,
            "pearson_r": self.pearson_r,
            "lod": self.lod,
            "loq": self.loq,
            "relative_rmsep": self.relative_rmsep,
            "true_range": self.true_range,
        }


@dataclass
class EvaluationResult:
    """
    Complete evaluation results for a benchmark run.

    Attributes
    ----------
    dataset_name : str
        Name of benchmark dataset
    split_name : str
        Name of data split used
    n_spectra : int
        Total number of spectra evaluated
    element_metrics : Dict[str, ElementMetrics]
        Per-element evaluation metrics
    overall_rmsep : float
        Averaged RMSEP across all elements
    overall_mae : float
        Averaged MAE across all elements
    overall_r_squared : float
        Averaged R-squared across all elements
    algorithm_name : str
        Name of algorithm being evaluated
    algorithm_version : str
        Version of algorithm
    metadata : Dict
        Additional metadata
    """

    dataset_name: str
    split_name: str
    n_spectra: int
    element_metrics: Dict[str, ElementMetrics]
    overall_rmsep: float
    overall_mae: float
    overall_r_squared: float
    algorithm_name: str = "unknown"
    algorithm_version: str = "unknown"
    metadata: Dict = field(default_factory=dict)

    @property
    def elements(self) -> List[str]:
        """List of evaluated elements."""
        return sorted(self.element_metrics.keys())

    @property
    def n_elements(self) -> int:
        """Number of elements evaluated."""
        return len(self.element_metrics)

    def get_metric(self, element: str, metric: MetricType) -> float:
        """
        Get a specific metric for an element.

        Parameters
        ----------
        element : str
            Element symbol
        metric : MetricType
            Metric type

        Returns
        -------
        float
            Metric value
        """
        if element not in self.element_metrics:
            raise KeyError(f"Element '{element}' not in evaluation results")

        elem_metrics = self.element_metrics[element]
        metric_map = {
            MetricType.RMSEP: elem_metrics.rmsep,
            MetricType.MAE: elem_metrics.mae,
            MetricType.MAPE: elem_metrics.mape,
            MetricType.BIAS: elem_metrics.bias,
            MetricType.R_SQUARED: elem_metrics.r_squared,
            MetricType.PEARSON_R: elem_metrics.pearson_r,
            MetricType.LOD: elem_metrics.lod,
            MetricType.LOQ: elem_metrics.loq,
            MetricType.RELATIVE_RMSEP: elem_metrics.relative_rmsep,
        }
        return metric_map[metric]

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Evaluation Results: {self.algorithm_name} v{self.algorithm_version}",
            f"  Dataset: {self.dataset_name} ({self.split_name})",
            f"  Spectra: {self.n_spectra}, Elements: {self.n_elements}",
            f"  Overall RMSEP: {self.overall_rmsep*100:.3f}%",
            f"  Overall MAE: {self.overall_mae*100:.3f}%",
            f"  Overall R^2: {self.overall_r_squared:.3f}",
            "",
            "Per-element results:",
        ]

        for elem in self.elements:
            lines.append(f"  {self.element_metrics[elem].summary()}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "dataset_name": self.dataset_name,
            "split_name": self.split_name,
            "n_spectra": self.n_spectra,
            "element_metrics": {el: m.to_dict() for el, m in self.element_metrics.items()},
            "overall_rmsep": self.overall_rmsep,
            "overall_mae": self.overall_mae,
            "overall_r_squared": self.overall_r_squared,
            "algorithm_name": self.algorithm_name,
            "algorithm_version": self.algorithm_version,
            "metadata": self.metadata,
        }

    def to_csv_row(self) -> str:
        """Generate CSV row for tabular comparison."""
        cols = [
            self.algorithm_name,
            self.algorithm_version,
            self.dataset_name,
            str(self.n_spectra),
            f"{self.overall_rmsep*100:.4f}",
            f"{self.overall_mae*100:.4f}",
            f"{self.overall_r_squared:.4f}",
        ]
        return ",".join(cols)


class BenchmarkMetrics:
    """
    Calculator for benchmark evaluation metrics.

    Provides standardized metric computation for comparing algorithm
    predictions against ground truth compositions from benchmark datasets.

    Parameters
    ----------
    min_concentration : float
        Minimum concentration to include in MAPE calculation (avoids div/0)
    lod_method : str
        Method for LOD calculation: "3sigma" or "calibration"

    Example
    -------
    >>> metrics = BenchmarkMetrics()
    >>> predictions = {"Fe": [0.65, 0.72], "Cu": [0.08, 0.12]}
    >>> true_values = {"Fe": [0.68, 0.70], "Cu": [0.10, 0.10]}
    >>> result = metrics.evaluate(
    ...     predictions, true_values,
    ...     dataset_name="test", split_name="default"
    ... )
    >>> print(result.summary())
    """

    def __init__(
        self,
        min_concentration: float = 0.001,  # 0.1% minimum for MAPE
        lod_method: str = "3sigma",
    ):
        self.min_concentration = min_concentration
        self.lod_method = lod_method

    def evaluate(
        self,
        predictions: Dict[str, List[float]],
        true_values: Dict[str, List[float]],
        dataset_name: str = "unknown",
        split_name: str = "unknown",
        algorithm_name: str = "unknown",
        algorithm_version: str = "unknown",
        metadata: Optional[Dict] = None,
    ) -> EvaluationResult:
        """
        Evaluate predictions against ground truth.

        Parameters
        ----------
        predictions : Dict[str, List[float]]
            Predicted compositions {element: [values]}
        true_values : Dict[str, List[float]]
            True compositions {element: [values]}
        dataset_name : str
            Name of benchmark dataset
        split_name : str
            Name of data split
        algorithm_name : str
            Algorithm identifier
        algorithm_version : str
            Algorithm version
        metadata : Dict, optional
            Additional metadata

        Returns
        -------
        EvaluationResult
            Complete evaluation results
        """
        # Validate inputs
        elements = sorted(set(predictions.keys()) & set(true_values.keys()))
        if not elements:
            raise ValueError("No common elements between predictions and true values")

        # Check array lengths
        n_spectra = None
        for elem in elements:
            pred_len = len(predictions[elem])
            true_len = len(true_values[elem])
            if pred_len != true_len:
                raise ValueError(
                    f"Element {elem}: prediction length ({pred_len}) != "
                    f"true values length ({true_len})"
                )
            if n_spectra is None:
                n_spectra = pred_len
            elif pred_len != n_spectra:
                raise ValueError("Inconsistent array lengths across elements")

        # n_spectra is set by the loop above; the elements list is non-empty.
        assert n_spectra is not None

        # Calculate per-element metrics
        element_metrics = {}
        for elem in elements:
            pred = np.array(predictions[elem])
            true = np.array(true_values[elem])
            element_metrics[elem] = self._calculate_element_metrics(elem, pred, true)

        # Calculate overall metrics (weighted by element presence). np.mean
        # returns np.floating[Any]; coerce to plain float for the dataclass.
        overall_rmsep = float(np.mean([m.rmsep for m in element_metrics.values()]))
        overall_mae = float(np.mean([m.mae for m in element_metrics.values()]))
        overall_r2 = float(np.mean([m.r_squared for m in element_metrics.values()]))

        return EvaluationResult(
            dataset_name=dataset_name,
            split_name=split_name,
            n_spectra=n_spectra,
            element_metrics=element_metrics,
            overall_rmsep=overall_rmsep,
            overall_mae=overall_mae,
            overall_r_squared=overall_r2,
            algorithm_name=algorithm_name,
            algorithm_version=algorithm_version,
            metadata=metadata or {},
        )

    def _calculate_element_metrics(
        self,
        element: str,
        predictions: np.ndarray,
        true_values: np.ndarray,
    ) -> ElementMetrics:
        """Calculate all metrics for a single element."""
        n = len(predictions)
        residuals = predictions - true_values

        # RMSEP
        rmsep = np.sqrt(np.mean(residuals**2))

        # MAE
        mae = np.mean(np.abs(residuals))

        # MAPE (only for concentrations above threshold)
        mask = true_values >= self.min_concentration
        if mask.sum() > 0:
            mape = np.mean(np.abs(residuals[mask]) / true_values[mask]) * 100
        else:
            mape = np.nan

        # Bias
        bias = np.mean(residuals)

        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)
        if ss_tot > 0:
            r_squared = 1 - ss_res / ss_tot
        else:
            r_squared = 0.0

        # Pearson correlation
        if np.std(predictions) > 0 and np.std(true_values) > 0:
            pearson_r = np.corrcoef(predictions, true_values)[0, 1]
        else:
            pearson_r = 0.0

        # LOD estimation (3-sigma of residuals at low concentrations)
        low_mask = true_values < np.percentile(true_values, 25)
        if low_mask.sum() >= 3:
            lod = 3 * np.std(residuals[low_mask])
        else:
            lod = 3 * np.std(residuals)

        loq = lod * 10 / 3  # LOQ = 10-sigma

        # Relative RMSEP
        mean_true = np.mean(true_values)
        if mean_true > 0:
            relative_rmsep = rmsep / mean_true
        else:
            relative_rmsep = np.inf

        return ElementMetrics(
            element=element,
            n_samples=n,
            rmsep=rmsep,
            mae=mae,
            mape=mape,
            bias=bias,
            r_squared=r_squared,
            pearson_r=pearson_r,
            lod=lod,
            loq=loq,
            relative_rmsep=relative_rmsep,
            true_range=(float(true_values.min()), float(true_values.max())),
        )

    def compare_algorithms(
        self,
        results: List[EvaluationResult],
        metric: MetricType = MetricType.RMSEP,
        element: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Compare multiple algorithm results.

        Parameters
        ----------
        results : List[EvaluationResult]
            Results from different algorithms
        metric : MetricType
            Metric to compare (default: RMSEP)
        element : str, optional
            Specific element to compare (default: overall)

        Returns
        -------
        Dict[str, float]
            Algorithm name -> metric value
        """
        comparison = {}

        for result in results:
            key = f"{result.algorithm_name} v{result.algorithm_version}"

            if element is not None:
                value = result.get_metric(element, metric)
            else:
                # Get overall metric
                if metric == MetricType.RMSEP:
                    value = result.overall_rmsep
                elif metric == MetricType.MAE:
                    value = result.overall_mae
                elif metric == MetricType.R_SQUARED:
                    value = result.overall_r_squared
                else:
                    # Average across elements
                    values = [result.get_metric(el, metric) for el in result.elements]
                    value = float(np.mean(values))

            comparison[key] = value

        return comparison

    @staticmethod
    def rmsep(
        predictions: np.ndarray,
        true_values: np.ndarray,
    ) -> float:
        """
        Calculate Root Mean Square Error of Prediction.

        Parameters
        ----------
        predictions : np.ndarray
            Predicted values
        true_values : np.ndarray
            Ground truth values

        Returns
        -------
        float
            RMSEP value
        """
        predictions = np.asarray(predictions)
        true_values = np.asarray(true_values)
        return np.sqrt(np.mean((predictions - true_values) ** 2))

    @staticmethod
    def mae(
        predictions: np.ndarray,
        true_values: np.ndarray,
    ) -> float:
        """
        Calculate Mean Absolute Error.

        Parameters
        ----------
        predictions : np.ndarray
            Predicted values
        true_values : np.ndarray
            Ground truth values

        Returns
        -------
        float
            MAE value
        """
        predictions = np.asarray(predictions)
        true_values = np.asarray(true_values)
        return np.mean(np.abs(predictions - true_values))

    @staticmethod
    def mape(
        predictions: np.ndarray,
        true_values: np.ndarray,
        epsilon: float = 1e-6,
    ) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        Parameters
        ----------
        predictions : np.ndarray
            Predicted values
        true_values : np.ndarray
            Ground truth values
        epsilon : float
            Small value to avoid division by zero

        Returns
        -------
        float
            MAPE value (as percentage)
        """
        predictions = np.asarray(predictions)
        true_values = np.asarray(true_values)
        return np.mean(np.abs(predictions - true_values) / (true_values + epsilon)) * 100

    @staticmethod
    def bias(
        predictions: np.ndarray,
        true_values: np.ndarray,
    ) -> float:
        """
        Calculate mean bias (systematic error).

        Parameters
        ----------
        predictions : np.ndarray
            Predicted values
        true_values : np.ndarray
            Ground truth values

        Returns
        -------
        float
            Mean bias (positive = over-estimation)
        """
        predictions = np.asarray(predictions)
        true_values = np.asarray(true_values)
        return np.mean(predictions - true_values)

    @staticmethod
    def r_squared(
        predictions: np.ndarray,
        true_values: np.ndarray,
    ) -> float:
        """
        Calculate coefficient of determination.

        Parameters
        ----------
        predictions : np.ndarray
            Predicted values
        true_values : np.ndarray
            Ground truth values

        Returns
        -------
        float
            R-squared value
        """
        predictions = np.asarray(predictions)
        true_values = np.asarray(true_values)

        ss_res = np.sum((true_values - predictions) ** 2)
        ss_tot = np.sum((true_values - np.mean(true_values)) ** 2)

        if ss_tot == 0:
            return 0.0
        return 1 - ss_res / ss_tot


def calculate_figure_of_merit(
    result: EvaluationResult,
    weights: Optional[Dict[str, float]] = None,
) -> float:
    """
    Calculate a single figure of merit for algorithm ranking.

    Combines multiple metrics into a single score for algorithm comparison.
    Lower scores are better.

    Parameters
    ----------
    result : EvaluationResult
        Evaluation results
    weights : Dict[str, float], optional
        Metric weights (default: {"rmsep": 0.5, "mae": 0.3, "r_squared": 0.2})

    Returns
    -------
    float
        Figure of merit score (lower is better)
    """
    if weights is None:
        weights = {"rmsep": 0.5, "mae": 0.3, "r_squared": 0.2}

    # Normalize weights
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    score = 0.0

    if "rmsep" in weights:
        score += weights["rmsep"] * result.overall_rmsep

    if "mae" in weights:
        score += weights["mae"] * result.overall_mae

    if "r_squared" in weights:
        # Invert R-squared since higher is better but we want lower scores to be better
        score += weights["r_squared"] * (1 - result.overall_r_squared)

    return score


def create_comparison_table(
    results: List[EvaluationResult],
    elements: Optional[List[str]] = None,
    metric: MetricType = MetricType.RMSEP,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Create a comparison table of algorithm performance.

    Parameters
    ----------
    results : List[EvaluationResult]
        Results from multiple algorithms
    elements : List[str], optional
        Elements to include (default: all common elements)
    metric : MetricType
        Metric to compare (default: RMSEP)

    Returns
    -------
    table : np.ndarray
        Shape (n_algorithms, n_elements) comparison table
    algorithm_names : List[str]
        Algorithm labels for rows
    element_names : List[str]
        Element labels for columns
    """
    if elements is None:
        # Find common elements
        element_sets = [set(r.elements) for r in results]
        elements = sorted(set.intersection(*element_sets))

    n_algs = len(results)
    n_elems = len(elements)

    table = np.zeros((n_algs, n_elems))
    algorithm_names = []

    for i, result in enumerate(results):
        algorithm_names.append(f"{result.algorithm_name} v{result.algorithm_version}")
        for j, elem in enumerate(elements):
            table[i, j] = result.get_metric(elem, metric)

    return table, algorithm_names, elements
