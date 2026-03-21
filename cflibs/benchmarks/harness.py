"""
Experiment runner for CF-LIBS benchmark evaluation.

Registers named pipeline functions, runs them on a
:class:`~cflibs.benchmarks.corpus.BenchmarkCorpus`, and collects timing
and compositional accuracy data into a structured
:class:`BenchmarkReport`.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from cflibs.benchmarks.corpus import BenchmarkSpectrum
from cflibs.benchmarks.metrics import aitchison_distance, per_element_error, rmse_composition

# ---------------------------------------------------------------------------
# Accuracy tiers
# ---------------------------------------------------------------------------


class AccuracyTier(Enum):
    """Compositional accuracy tier based on Aitchison distance."""

    EXCELLENT = "EXCELLENT"  # < 5%
    GOOD = "GOOD"  # < 10%
    ACCEPTABLE = "ACCEPTABLE"  # < 20%
    POOR = "POOR"  # >= 20%

    @classmethod
    def from_aitchison(cls, dist: float) -> "AccuracyTier":
        """Classify an Aitchison distance into an accuracy tier.

        Parameters
        ----------
        dist : float
            Aitchison distance value.

        Returns
        -------
        AccuracyTier
        """
        if dist < 0.05:
            return cls.EXCELLENT
        elif dist < 0.10:
            return cls.GOOD
        elif dist < 0.20:
            return cls.ACCEPTABLE
        else:
            return cls.POOR


# ---------------------------------------------------------------------------
# Result data structures
# ---------------------------------------------------------------------------


@dataclass
class SpectrumResult:
    """Result for a single spectrum evaluated by a single pipeline.

    Attributes
    ----------
    label : str
        Spectrum label.
    aitchison : float
        Aitchison distance between true and predicted compositions.
    rmse : float
        RMSE between true and predicted compositions.
    element_errors : Dict[str, tuple]
        Per-element (absolute_error, relative_error).
    tier : AccuracyTier
        Accuracy tier.
    elapsed_ns : int
        Wall-clock time in nanoseconds.
    temperature_error_frac : Optional[float]
        Fractional error in temperature recovery.
    ne_error_frac : Optional[float]
        Fractional error in electron density recovery.
    predicted : Dict[str, Any]
        Raw pipeline output.
    error : Optional[str]
        Error message if the pipeline raised an exception.
    """

    label: str
    aitchison: float
    rmse: float
    element_errors: Dict
    tier: AccuracyTier
    elapsed_ns: int
    temperature_error_frac: Optional[float] = None
    ne_error_frac: Optional[float] = None
    predicted: Dict = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Aggregated results for a single pipeline across all spectra.

    Attributes
    ----------
    name : str
        Pipeline name.
    spectrum_results : List[SpectrumResult]
        Per-spectrum results.
    """

    name: str
    spectrum_results: List[SpectrumResult] = field(default_factory=list)

    @property
    def n_spectra(self) -> int:
        return len(self.spectrum_results)

    @property
    def n_errors(self) -> int:
        return sum(1 for r in self.spectrum_results if r.error is not None)

    @property
    def aitchison_values(self) -> np.ndarray:
        return np.array([r.aitchison for r in self.spectrum_results if r.error is None])

    @property
    def timing_ns(self) -> np.ndarray:
        return np.array([r.elapsed_ns for r in self.spectrum_results])

    def tier_distribution(self) -> Dict[str, int]:
        """Count of spectra in each accuracy tier."""
        counts: Dict[str, int] = {t.value: 0 for t in AccuracyTier}
        for r in self.spectrum_results:
            if r.error is None:
                counts[r.tier.value] += 1
        return counts


@dataclass
class BenchmarkReport:
    """Full benchmark report across pipelines and spectra.

    Attributes
    ----------
    results : List[PipelineResult]
        Per-pipeline aggregated results.
    metadata : Dict
        Report metadata (timestamp, corpus size, etc.).
    """

    results: List[PipelineResult] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        """Aggregate statistics for each pipeline.

        Returns
        -------
        Dict[str, Any]
            Mapping of pipeline name to summary dict containing:
            ``mean_aitchison``, ``median_aitchison``, ``p95_aitchison``,
            ``mean_time_ms``, ``p95_time_ms``, ``tier_distribution``,
            ``n_errors``.
        """
        out: Dict[str, Any] = {}
        for pr in self.results:
            ad = pr.aitchison_values
            t_ms = pr.timing_ns / 1e6
            summary_entry: Dict[str, Any] = {
                "n_spectra": pr.n_spectra,
                "n_errors": pr.n_errors,
                "tier_distribution": pr.tier_distribution(),
            }
            if len(ad) > 0:
                summary_entry["mean_aitchison"] = float(np.mean(ad))
                summary_entry["median_aitchison"] = float(np.median(ad))
                summary_entry["p95_aitchison"] = float(np.percentile(ad, 95))
            if len(t_ms) > 0:
                summary_entry["mean_time_ms"] = float(np.mean(t_ms))
                summary_entry["p95_time_ms"] = float(np.percentile(t_ms, 95))
            out[pr.name] = summary_entry
        return out

    def to_json(self, indent: int = 2) -> str:
        """Serialize the report to JSON.

        Returns
        -------
        str
            JSON string.
        """
        payload: Dict[str, Any] = {
            "metadata": self.metadata,
            "summary": self.summary(),
            "pipelines": {},
        }
        for pr in self.results:
            pipeline_data = []
            for sr in pr.spectrum_results:
                entry: Dict[str, Any] = {
                    "label": sr.label,
                    "aitchison": sr.aitchison,
                    "rmse": sr.rmse,
                    "tier": sr.tier.value,
                    "elapsed_ms": sr.elapsed_ns / 1e6,
                }
                if sr.temperature_error_frac is not None:
                    entry["temperature_error_frac"] = sr.temperature_error_frac
                if sr.ne_error_frac is not None:
                    entry["ne_error_frac"] = sr.ne_error_frac
                if sr.error is not None:
                    entry["error"] = sr.error
                pipeline_data.append(entry)
            payload["pipelines"][pr.name] = pipeline_data
        return json.dumps(payload, indent=indent)

    def compare(self, other: "BenchmarkReport") -> Dict[str, Any]:
        """Side-by-side comparison of two reports.

        Parameters
        ----------
        other : BenchmarkReport
            Another report to compare against.

        Returns
        -------
        Dict[str, Any]
            Comparison dict with delta values for each shared pipeline.
        """
        self_summary = self.summary()
        other_summary = other.summary()

        comparison: Dict[str, Any] = {}
        all_names = set(self_summary.keys()) | set(other_summary.keys())
        for name in sorted(all_names):
            s = self_summary.get(name, {})
            o = other_summary.get(name, {})
            entry: Dict[str, Any] = {"self": s, "other": o}
            if "mean_aitchison" in s and "mean_aitchison" in o:
                entry["delta_mean_aitchison"] = s["mean_aitchison"] - o["mean_aitchison"]
            if "mean_time_ms" in s and "mean_time_ms" in o:
                entry["delta_mean_time_ms"] = s["mean_time_ms"] - o["mean_time_ms"]
            comparison[name] = entry

        return comparison


# ---------------------------------------------------------------------------
# Pipeline type
# ---------------------------------------------------------------------------

# A pipeline function takes (wavelengths, intensities, elements) and returns
# a dict with at least 'concentrations' (Dict[str, float]).
PipelineFn = Callable[
    [np.ndarray, np.ndarray, List[str]],
    Dict[str, Any],
]


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class BenchmarkHarness:
    """Register and run inversion pipelines on a benchmark corpus.

    Example
    -------
    >>> harness = BenchmarkHarness()
    >>> harness.register_pipeline("my_solver", my_pipeline_fn)
    >>> corpus = BenchmarkCorpus().generate()
    >>> report = harness.run(corpus)
    >>> print(json.dumps(report.summary(), indent=2))
    """

    def __init__(self) -> None:
        self._pipelines: Dict[str, PipelineFn] = {}

    def register_pipeline(self, name: str, pipeline_fn: PipelineFn) -> None:
        """Register a named pipeline function.

        Parameters
        ----------
        name : str
            Unique pipeline name.
        pipeline_fn : Callable
            Function with signature
            ``(wavelengths: ndarray, intensities: ndarray, elements: List[str])
            -> Dict`` returning at least ``{'concentrations': Dict[str, float]}``.
            May also return ``'temperature_K'`` and ``'electron_density_cm3'``.
        """
        if name in self._pipelines:
            raise ValueError(f"Pipeline {name!r} is already registered")
        self._pipelines[name] = pipeline_fn

    def run(
        self,
        corpus: Sequence[BenchmarkSpectrum],
        pipelines: Optional[Sequence[str]] = None,
    ) -> BenchmarkReport:
        """Run pipelines on a corpus and collect results.

        Parameters
        ----------
        corpus : Sequence[BenchmarkSpectrum]
            Spectra to evaluate.
        pipelines : Optional[Sequence[str]]
            Pipeline names to run.  ``None`` runs all registered pipelines.

        Returns
        -------
        BenchmarkReport
        """
        names = list(pipelines) if pipelines is not None else list(self._pipelines.keys())
        if not names:
            raise ValueError("No pipelines to run.  Register at least one first.")

        report = BenchmarkReport(
            metadata={
                "n_spectra": len(corpus),
                "pipelines": names,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        for name in names:
            fn = self._pipelines[name]
            pr = PipelineResult(name=name)

            for spec in corpus:
                sr = self._evaluate_one(fn, spec)
                pr.spectrum_results.append(sr)

            report.results.append(pr)

        return report

    def report(
        self,
        corpus: Sequence[BenchmarkSpectrum],
        pipelines: Optional[Sequence[str]] = None,
    ) -> BenchmarkReport:
        """Alias for :meth:`run` (convenience)."""
        return self.run(corpus, pipelines)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_one(fn: PipelineFn, spec: BenchmarkSpectrum) -> SpectrumResult:
        """Run a single pipeline on a single spectrum and measure accuracy."""
        gt = spec.ground_truth
        true_conc = gt.get("concentrations", {})
        elements = sorted(true_conc.keys())

        # Time the pipeline
        t0 = time.perf_counter_ns()
        try:
            result = fn(spec.wavelength, spec.intensity, elements)
            error_msg = None
        except Exception as exc:
            result = {"concentrations": {}}
            error_msg = f"{type(exc).__name__}: {exc}"
        elapsed = time.perf_counter_ns() - t0

        pred_conc = result.get("concentrations", {})

        # Compositional metrics
        ad = aitchison_distance(true_conc, pred_conc) if pred_conc else float("inf")
        rmse = rmse_composition(true_conc, pred_conc)
        el_err = per_element_error(true_conc, pred_conc)
        tier = AccuracyTier.from_aitchison(ad)

        # Temperature / ne error
        temp_err = None
        if "temperature_K" in result and "temperature_K" in gt:
            t_true = gt["temperature_K"]
            t_pred = result["temperature_K"]
            if t_true > 0:
                temp_err = abs(t_pred - t_true) / t_true

        ne_err = None
        if "electron_density_cm3" in result and "electron_density_cm3" in gt:
            ne_true = gt["electron_density_cm3"]
            ne_pred = result["electron_density_cm3"]
            if ne_true > 0:
                ne_err = abs(ne_pred - ne_true) / ne_true

        return SpectrumResult(
            label=spec.label,
            aitchison=ad,
            rmse=rmse,
            element_errors=el_err,
            tier=tier,
            elapsed_ns=elapsed,
            temperature_error_frac=temp_err,
            ne_error_frac=ne_err,
            predicted=result,
            error=error_msg,
        )
