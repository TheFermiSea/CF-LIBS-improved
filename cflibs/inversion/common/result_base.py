"""
Shared result formatting utilities for CF-LIBS inversion results.

This module provides mixins for common result formatting patterns used across:
- MCMCResult (bayesian.py)
- NestedSamplingResult (bayesian.py)
- MonteCarloResult (uncertainty.py)

The mixins eliminate code duplication while maintaining consistent output format.
"""

from typing import Dict, Optional, Tuple
import numpy as np

# Table formatting constants
TABLE_WIDTH = 70
TABLE_SEP = "-" * TABLE_WIDTH
TABLE_HEADER = "=" * TABLE_WIDTH


class ResultTableMixin:
    """
    Mixin providing shared table formatting for result classes.

    Classes using this mixin should have attributes:
    - concentrations_mean: Dict[str, float]
    - concentrations_std: Dict[str, float]
    """

    @staticmethod
    def _format_header(title: str) -> str:
        """Format a table header with title."""
        return f"{TABLE_HEADER}\n{title}\n{TABLE_HEADER}"

    @staticmethod
    def _format_separator() -> str:
        """Return a horizontal separator line."""
        return TABLE_SEP

    @staticmethod
    def _format_footer() -> str:
        """Return a table footer."""
        return TABLE_HEADER

    @staticmethod
    def _format_param_row(
        label: str,
        mean: float,
        std: float,
        ci: Optional[Tuple[float, float]] = None,
        fmt: str = ".4f",
        include_ci: bool = True,
    ) -> str:
        """
        Format a single parameter row.

        Parameters
        ----------
        label : str
            Parameter label (e.g., "T [eV]")
        mean : float
            Mean value
        std : float
            Standard deviation
        ci : tuple, optional
            (lower, upper) confidence interval
        fmt : str
            Format string for values (default: ".4f")
        include_ci : bool
            Whether to include confidence interval column
        """
        if include_ci and ci is not None:
            ci_str = f"[{ci[0]:{fmt}}, {ci[1]:{fmt}}]"
            return f"{label:<20} {mean:>12{fmt}} {std:>12{fmt}} {ci_str:>20}"
        else:
            return f"{label:<20} {mean:>12{fmt}} {std:>12{fmt}}"

    @staticmethod
    def _format_param_row_exp(
        label: str,
        mean: float,
        std: Optional[float] = None,
        ci: Optional[Tuple[float, float]] = None,
        include_ci: bool = True,
    ) -> str:
        """Format a parameter row with exponential notation."""
        base = f"{label:<20} {mean:>12.2e}"
        if std is not None:
            base += f" {std:>12.2e}"
        if include_ci and ci is not None:
            base += f" [{ci[0]:.2e}, {ci[1]:.2e}]"
        return base

    @staticmethod
    def _format_param_row_int(
        label: str,
        mean: float,
        std: float,
        ci: Optional[Tuple[float, float]] = None,
        include_ci: bool = True,
    ) -> str:
        """Format a parameter row with integer formatting (for temperature K)."""
        if include_ci and ci is not None:
            ci_str = f"[{ci[0]:.0f}, {ci[1]:.0f}]"
            return f"{label:<20} {mean:>12.0f} {std:>12.0f} {ci_str:>20}"
        else:
            return f"{label:<20} {mean:>12.0f} {std:>12.0f}"

    def _format_concentration_table(
        self,
        conc_mean: Dict[str, float],
        conc_std: Dict[str, float],
        conc_ci: Optional[Dict[str, Tuple[float, float]]] = None,
        include_ci: bool = True,
    ) -> list:
        """
        Format concentration table rows.

        Parameters
        ----------
        conc_mean : dict
            Mean concentrations by element
        conc_std : dict
            Standard deviations by element
        conc_ci : dict, optional
            95% confidence intervals by element
        include_ci : bool
            Whether to include CI column

        Returns
        -------
        list of str
            Formatted table rows
        """
        lines = []
        lines.append(TABLE_SEP)

        has_ci = include_ci and bool(conc_ci)

        if has_ci:
            lines.append(f"{'Element':<20} {'Conc.':<12} {'Std':>12} {'95% CI':>20}")
        else:
            lines.append(f"{'Element':<20} {'Conc.':<12} {'Std':>12}")

        lines.append(TABLE_SEP)

        for el in sorted(conc_mean.keys()):
            mean = conc_mean[el]
            std = conc_std.get(el, 0.0)

            if has_ci:
                ci = conc_ci.get(el, (mean - 2 * std, mean + 2 * std))
                lines.append(f"{el:<20} {mean:>12.4f} {std:>12.4f} [{ci[0]:.4f}, {ci[1]:.4f}]")
            else:
                lines.append(f"{el:<20} {mean:>12.4f} {std:>12.4f}")

        return lines


class StatisticsMixin:
    """
    Mixin providing shared statistical calculations.

    Provides methods for computing confidence intervals and percentiles.
    """

    @staticmethod
    def compute_ci(samples: np.ndarray, level: float = 0.95) -> Tuple[float, float]:
        """
        Compute confidence interval from samples.

        Parameters
        ----------
        samples : array
            Sample array
        level : float
            Confidence level (default: 0.95 for 95% CI)

        Returns
        -------
        tuple
            (lower, upper) bounds
        """
        if len(samples) < 2:
            return (float("nan"), float("nan"))

        alpha = (1 - level) / 2
        lower = float(np.percentile(samples, alpha * 100))
        upper = float(np.percentile(samples, (1 - alpha) * 100))
        return (lower, upper)

    @staticmethod
    def compute_ci_68(samples: np.ndarray) -> Tuple[float, float]:
        """Compute 68% CI (1-sigma equivalent)."""
        return StatisticsMixin.compute_ci(samples, 0.68)

    @staticmethod
    def compute_ci_95(samples: np.ndarray) -> Tuple[float, float]:
        """Compute 95% CI (2-sigma equivalent)."""
        return StatisticsMixin.compute_ci(samples, 0.95)

    @staticmethod
    def compute_quantiles(
        samples: np.ndarray,
    ) -> Tuple[float, float, float, float]:
        """
        Compute standard quantiles.

        Returns
        -------
        tuple
            (q025, q16, q84, q975) - 2.5%, 16%, 84%, 97.5% quantiles
        """
        if len(samples) < 2:
            nan = float("nan")
            return (nan, nan, nan, nan)

        q025 = float(np.percentile(samples, 2.5))
        q16 = float(np.percentile(samples, 16))
        q84 = float(np.percentile(samples, 84))
        q975 = float(np.percentile(samples, 97.5))
        return (q025, q16, q84, q975)
