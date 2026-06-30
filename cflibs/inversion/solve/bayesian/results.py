"""Result containers for Bayesian CF-LIBS inference (T1-6).

Hosts the dataclasses that samplers return:

* :class:`MCMCResult` -- single-zone MCMC posterior + convergence diagnostics.
* :class:`NestedSamplingResult` -- single-zone nested-sampling posterior with
  evidence ``ln(Z)`` for model comparison.
* :class:`TwoZoneMCMCResult` -- two-zone (core+shell) MCMC posterior.

These are intentionally separated from :mod:`samplers` to keep each file
under the 800 LOC limit imposed by ADR-0001 / T1-6 spec section 6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from cflibs.core.constants import EV_TO_K

from .priors import ConvergenceStatus

# ---------------------------------------------------------------------------
# Single-zone MCMC result
# ---------------------------------------------------------------------------


@dataclass
class MCMCResult:
    """Result container for MCMC sampling with convergence diagnostics.

    Stores posterior samples, summary statistics, and convergence diagnostics
    from Bayesian CF-LIBS inference.
    """

    samples: Dict[str, np.ndarray]

    T_eV_mean: float
    T_eV_std: float
    T_eV_q025: float
    T_eV_q975: float

    log_ne_mean: float
    log_ne_std: float
    log_ne_q025: float
    log_ne_q975: float

    concentrations_mean: Dict[str, float]
    concentrations_std: Dict[str, float]
    concentrations_q025: Dict[str, float]
    concentrations_q975: Dict[str, float]

    r_hat: Dict[str, float] = field(default_factory=dict)
    ess: Dict[str, float] = field(default_factory=dict)
    convergence_status: ConvergenceStatus = ConvergenceStatus.UNKNOWN

    n_samples: int = 0
    n_chains: int = 1
    n_warmup: int = 0

    inference_data: Any = None

    #: Number of divergent NUTS transitions (``None`` when not collected). Now
    #: surfaced unconditionally; the strict gate raises when it is > 0.
    n_divergences: Optional[int] = None

    @property
    def n_e_mean(self) -> float:
        """Mean electron density [cm^-3]."""
        return 10.0**self.log_ne_mean

    @property
    def T_K_mean(self) -> float:
        """Mean temperature [K]."""
        return self.T_eV_mean * EV_TO_K

    @property
    def is_converged(self) -> bool:
        """Check if MCMC has converged (R-hat < 1.01 for all parameters)."""
        return self.convergence_status == ConvergenceStatus.CONVERGED

    def summary_table(self) -> str:
        """Generate a publication-ready summary table."""
        lines = [
            "=" * 70,
            "CF-LIBS Bayesian Inference Results",
            "=" * 70,
            f"Samples: {self.n_samples} | Chains: {self.n_chains} | Warmup: {self.n_warmup}",
            f"Convergence: {self.convergence_status.value}",
            "-" * 70,
            f"{'Parameter':<20} {'Mean':>12} {'Std':>12} {'95% CI':>20}",
            "-" * 70,
            f"{'T [eV]':<20} {self.T_eV_mean:>12.4f} {self.T_eV_std:>12.4f} "
            f"[{self.T_eV_q025:.4f}, {self.T_eV_q975:.4f}]",
            f"{'T [K]':<20} {self.T_K_mean:>12.0f} {self.T_eV_std * EV_TO_K:>12.0f} "
            f"[{self.T_eV_q025 * EV_TO_K:.0f}, {self.T_eV_q975 * EV_TO_K:.0f}]",
            f"{'log10(n_e)':<20} {self.log_ne_mean:>12.4f} {self.log_ne_std:>12.4f} "
            f"[{self.log_ne_q025:.4f}, {self.log_ne_q975:.4f}]",
            f"{'n_e [cm^-3]':<20} {self.n_e_mean:>12.2e}",
        ]

        lines.append("-" * 70)
        lines.append(f"{'Element':<20} {'Conc.':<12} {'Std':>12} {'95% CI':>20}")
        lines.append("-" * 70)

        for el in self.concentrations_mean:
            mean = self.concentrations_mean[el]
            std = self.concentrations_std[el]
            q025 = self.concentrations_q025.get(el, mean - 2 * std)
            q975 = self.concentrations_q975.get(el, mean + 2 * std)
            lines.append(f"{el:<20} {mean:>12.4f} {std:>12.4f} [{q025:.4f}, {q975:.4f}]")

        if self.r_hat:
            lines.append("-" * 70)
            lines.append("Convergence Diagnostics:")
            for param, rhat in self.r_hat.items():
                ess_val = self.ess.get(param, float("nan"))
                status = "OK" if rhat < 1.01 else "FAIL"
                lines.append(f"  {param}: R-hat={rhat:.3f} {status}, ESS={ess_val:.0f}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def correlation_matrix(self, include_concentrations: bool = True) -> Dict[str, Any]:
        """Compute correlation matrix between posterior parameters.

        Correlation analysis helps identify parameter degeneracies and
        understand how uncertainties are coupled (e.g., T-n_e correlation).
        """
        T_samples = np.array(self.samples["T_eV"]).flatten()
        log_ne_samples = np.array(self.samples["log_ne"]).flatten()

        param_names = ["T_eV", "log_ne"]
        param_data = [T_samples, log_ne_samples]

        if include_concentrations and "concentrations" in self.samples:
            conc_samples = np.array(self.samples["concentrations"])
            if conc_samples.ndim == 3:
                conc_samples = conc_samples.reshape(-1, conc_samples.shape[-1])
            for i, el in enumerate(self.concentrations_mean.keys()):
                param_names.append(f"C_{el}")
                param_data.append(conc_samples[:, i])

        data_matrix = np.vstack(param_data)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr_matrix = np.atleast_2d(np.corrcoef(data_matrix))
        # A zero-variance (stuck/degenerate) chain has undefined Pearson
        # correlation and would poison its whole row/column with NaN. Report
        # 0 against every other parameter (no detectable linear association)
        # and 1 on its own diagonal so the result stays a valid, finite
        # correlation matrix. ``~(std > 0)`` also catches NaN samples.
        degenerate = ~(data_matrix.std(axis=1) > 0.0)
        if degenerate.any():
            corr_matrix[degenerate, :] = 0.0
            corr_matrix[:, degenerate] = 0.0
            np.fill_diagonal(corr_matrix, 1.0)
        T_log_ne_corr = corr_matrix[0, 1]

        return {
            "matrix": corr_matrix,
            "labels": param_names,
            "T_log_ne_corr": float(T_log_ne_corr),
        }

    def correlation_table(self) -> str:
        """Generate a formatted correlation table."""
        corr_data = self.correlation_matrix()
        matrix = corr_data["matrix"]
        labels = corr_data["labels"]

        lines = [
            "=" * 70,
            "Parameter Correlations",
            "=" * 70,
        ]

        header = f"{'':>12}" + "".join(f"{lbl:>10}" for lbl in labels)
        lines.append(header)
        lines.append("-" * len(header))

        for i, label in enumerate(labels):
            row = f"{label:>12}"
            for j in range(len(labels)):
                val = matrix[i, j]
                if i == j:
                    row += f"{'1.000':>10}"
                else:
                    row += f"{val:>10.3f}"
            lines.append(row)

        lines.append("-" * 70)
        lines.append(f"T - log_ne correlation: {corr_data['T_log_ne_corr']:.3f}")
        lines.append("=" * 70)

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Nested sampling result (with evidence for model comparison)
# ---------------------------------------------------------------------------


@dataclass
class NestedSamplingResult:
    """Result container for nested sampling with model evidence.

    Provides both posterior samples and marginal likelihood (evidence)
    for model comparison.
    """

    samples: Dict[str, np.ndarray]
    weights: np.ndarray

    log_evidence: float
    log_evidence_err: float
    information: float  # KL divergence

    T_eV_mean: float
    T_eV_std: float
    log_ne_mean: float
    log_ne_std: float
    concentrations_mean: Dict[str, float]
    concentrations_std: Dict[str, float]

    n_live: int = 100
    n_iterations: int = 0
    n_calls: int = 0

    #: Count of log-likelihood evaluations that raised an exception inside the
    #: forward model (non-strict only — strict re-raises immediately). A non-zero
    #: value means a real code/data failure was silently treated as -inf.
    n_loglike_exceptions: int = 0
    #: Count of non-finite log-likelihood evaluations (divergence regions).
    n_nonfinite_loglike: int = 0

    @property
    def n_e_mean(self) -> float:
        """Mean electron density [cm^-3]."""
        return 10.0**self.log_ne_mean

    @property
    def T_K_mean(self) -> float:
        """Mean temperature [K]."""
        return self.T_eV_mean * EV_TO_K

    @property
    def evidence(self) -> float:
        """Marginal likelihood Z = exp(log_evidence)."""
        return np.exp(self.log_evidence)

    @property
    def bayes_factor_vs(self) -> str:
        """Interpretation helper for Bayes factors (Kass & Raftery 1995)."""
        return (
            "Bayes factor interpretation (Kass & Raftery 1995):\n"
            "  |Delta ln(Z)| < 1:    Not worth more than a bare mention\n"
            "  1 < |Delta ln(Z)| < 3:  Positive evidence\n"
            "  3 < |Delta ln(Z)| < 5:  Strong evidence\n"
            "  |Delta ln(Z)| > 5:      Very strong evidence"
        )

    def summary_table(self) -> str:
        """Generate a publication-ready summary table."""
        lines = [
            "=" * 70,
            "CF-LIBS Nested Sampling Results",
            "=" * 70,
            f"Live points: {self.n_live} | Iterations: {self.n_iterations} | "
            f"Likelihood calls: {self.n_calls}",
            "-" * 70,
            "MODEL EVIDENCE:",
            f"  ln(Z) = {self.log_evidence:.2f} +/- {self.log_evidence_err:.2f}",
            f"  Information (H) = {self.information:.2f} nats",
            "-" * 70,
            f"{'Parameter':<20} {'Mean':>12} {'Std':>12}",
            "-" * 70,
            f"{'T [eV]':<20} {self.T_eV_mean:>12.4f} {self.T_eV_std:>12.4f}",
            f"{'T [K]':<20} {self.T_K_mean:>12.0f} {self.T_eV_std * EV_TO_K:>12.0f}",
            f"{'log10(n_e)':<20} {self.log_ne_mean:>12.4f} {self.log_ne_std:>12.4f}",
            f"{'n_e [cm^-3]':<20} {self.n_e_mean:>12.2e}",
        ]

        lines.append("-" * 70)
        lines.append(f"{'Element':<20} {'Conc.':<12} {'Std':>12}")
        lines.append("-" * 70)

        for el in self.concentrations_mean:
            mean = self.concentrations_mean[el]
            std = self.concentrations_std[el]
            lines.append(f"{el:<20} {mean:>12.4f} {std:>12.4f}")

        lines.append("=" * 70)
        return "\n".join(lines)

    @staticmethod
    def compare_models(
        result_a: "NestedSamplingResult",
        result_b: "NestedSamplingResult",
        name_a: str = "Model A",
        name_b: str = "Model B",
    ) -> str:
        """Compare two models using Bayes factor."""
        delta_ln_z = result_a.log_evidence - result_b.log_evidence
        err = np.sqrt(result_a.log_evidence_err**2 + result_b.log_evidence_err**2)

        if abs(delta_ln_z) < 1:
            interpretation = "No significant preference"
        elif abs(delta_ln_z) < 3:
            preferred = name_a if delta_ln_z > 0 else name_b
            interpretation = f"Weak evidence for {preferred}"
        elif abs(delta_ln_z) < 5:
            preferred = name_a if delta_ln_z > 0 else name_b
            interpretation = f"Strong evidence for {preferred}"
        else:
            preferred = name_a if delta_ln_z > 0 else name_b
            interpretation = f"Very strong evidence for {preferred}"

        lines = [
            "=" * 60,
            "Bayesian Model Comparison",
            "=" * 60,
            f"{name_a}: ln(Z) = {result_a.log_evidence:.2f} +/- {result_a.log_evidence_err:.2f}",
            f"{name_b}: ln(Z) = {result_b.log_evidence:.2f} +/- {result_b.log_evidence_err:.2f}",
            "-" * 60,
            f"Delta ln(Z) = {delta_ln_z:.2f} +/- {err:.2f}",
            f"Bayes factor K = {np.exp(delta_ln_z):.2e}",
            "-" * 60,
            f"Interpretation: {interpretation}",
            "=" * 60,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Two-zone MCMC result
# ---------------------------------------------------------------------------


@dataclass
class TwoZoneMCMCResult:
    """Result container for two-zone MCMC sampling.

    Extends the single-zone result pattern with parameters for the two-zone
    model (core/shell temperatures, shell fraction, optical depth scale).
    """

    samples: Dict[str, np.ndarray]

    T_core_eV_mean: float
    T_core_eV_std: float
    T_core_eV_q025: float
    T_core_eV_q975: float

    T_shell_eV_mean: float
    T_shell_eV_std: float
    T_shell_eV_q025: float
    T_shell_eV_q975: float

    log_ne_mean: float
    log_ne_std: float
    log_ne_q025: float
    log_ne_q975: float

    shell_fraction_mean: float
    shell_fraction_std: float
    optical_depth_scale_mean: float
    optical_depth_scale_std: float

    concentrations_mean: Dict[str, float]
    concentrations_std: Dict[str, float]
    concentrations_q025: Dict[str, float] = field(default_factory=dict)
    concentrations_q975: Dict[str, float] = field(default_factory=dict)

    r_hat: Dict[str, float] = field(default_factory=dict)
    ess: Dict[str, float] = field(default_factory=dict)
    convergence_status: ConvergenceStatus = ConvergenceStatus.UNKNOWN

    n_samples: int = 0
    n_chains: int = 1
    n_warmup: int = 0
    inference_data: Any = None

    #: Number of divergent NUTS transitions (``None`` when not collected).
    n_divergences: Optional[int] = None

    @property
    def n_e_mean(self) -> float:
        """Mean electron density [cm^-3]."""
        return 10.0**self.log_ne_mean

    @property
    def T_core_K_mean(self) -> float:
        """Mean core temperature [K]."""
        return self.T_core_eV_mean * EV_TO_K

    @property
    def T_shell_K_mean(self) -> float:
        """Mean shell temperature [K]."""
        return self.T_shell_eV_mean * EV_TO_K

    @property
    def is_converged(self) -> bool:
        """Check MCMC convergence (R-hat < 1.01 for all parameters)."""
        return self.convergence_status == ConvergenceStatus.CONVERGED

    def summary_table(self) -> str:
        """Generate a publication-ready summary table."""
        lines = [
            "=" * 70,
            "Two-Zone CF-LIBS Bayesian Inference Results",
            "=" * 70,
            f"Samples: {self.n_samples} | Chains: {self.n_chains} | Warmup: {self.n_warmup}",
            f"Convergence: {self.convergence_status.value}",
            "-" * 70,
            f"{'Parameter':<25} {'Mean':>10} {'Std':>10} {'95% CI':>20}",
            "-" * 70,
            f"{'T_core [eV]':<25} {self.T_core_eV_mean:>10.4f} {self.T_core_eV_std:>10.4f} "
            f"[{self.T_core_eV_q025:.4f}, {self.T_core_eV_q975:.4f}]",
            f"{'T_shell [eV]':<25} {self.T_shell_eV_mean:>10.4f} {self.T_shell_eV_std:>10.4f} "
            f"[{self.T_shell_eV_q025:.4f}, {self.T_shell_eV_q975:.4f}]",
            f"{'log10(n_e)':<25} {self.log_ne_mean:>10.4f} {self.log_ne_std:>10.4f} "
            f"[{self.log_ne_q025:.4f}, {self.log_ne_q975:.4f}]",
            f"{'shell_fraction':<25} {self.shell_fraction_mean:>10.4f} "
            f"{self.shell_fraction_std:>10.4f}",
            f"{'optical_depth_scale':<25} {self.optical_depth_scale_mean:>10.4f} "
            f"{self.optical_depth_scale_std:>10.4f}",
        ]
        lines.append("-" * 70)
        for el in self.concentrations_mean:
            mean = self.concentrations_mean[el]
            std = self.concentrations_std[el]
            lines.append(f"{el:<25} {mean:>10.4f} {std:>10.4f}")
        lines.append("=" * 70)
        return "\n".join(lines)


__all__ = ["MCMCResult", "NestedSamplingResult", "TwoZoneMCMCResult"]
