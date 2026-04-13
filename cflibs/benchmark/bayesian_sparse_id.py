"""
Bayesian sparse element identification workflow for the unified benchmark.

Integrates the candidate prefilter (NNLS top-K) with the Bayesian forward
model (NumPyro MCMC) to produce ElementIdentificationResult compatible with
the benchmark harness.

Detection decision: Since the Dirichlet prior on concentrations is strictly
positive, "credible interval excludes zero" cannot be used. Instead, detection
is defined as P(c_i > presence_floor) >= posterior_probability_threshold.
This approach was validated by Codex (2026-04-13).

Design reviewed by Codex GPT 5.4:
- Separate module (not inline in unified.py) to isolate JAX/NumPyro deps
- No new UnifiedBenchmarkContext fields needed (db_path + basis_for_rp suffice)
- Configs include MCMC hyperparameters + detection thresholds
- Lazy registration in build_id_workflow_registry()
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("benchmark.bayesian_sparse_id")


def bayesian_sparse_workflow_configs(quick: bool = False) -> List[Dict[str, Any]]:
    """Configuration grid for the bayesian_sparse workflow.

    Parameters
    ----------
    quick : bool
        If True, return a minimal config for fast smoke testing.
    """
    if quick:
        return [
            {
                "num_warmup": 200,
                "num_samples": 200,
                "num_chains": 1,
                "target_accept_prob": 0.8,
                "baseline_degree": 0,
                "k_max": 10,
                "presence_floor": 0.01,
                "posterior_prob_threshold": 0.90,
            }
        ]
    return [
        {
            "num_warmup": 500,
            "num_samples": 1000,
            "num_chains": 1,
            "target_accept_prob": 0.8,
            "baseline_degree": 0,
            "k_max": 15,
            "presence_floor": 0.01,
            "posterior_prob_threshold": 0.90,
        },
        {
            "num_warmup": 500,
            "num_samples": 1000,
            "num_chains": 1,
            "target_accept_prob": 0.8,
            "baseline_degree": 3,
            "k_max": 15,
            "presence_floor": 0.01,
            "posterior_prob_threshold": 0.90,
        },
    ]


def bayesian_sparse_config_name(config: Dict[str, Any]) -> str:
    """Human-readable name for a bayesian_sparse config."""
    bd = config.get("baseline_degree", 0)
    k = config.get("k_max", 15)
    ns = config.get("num_samples", 1000)
    return f"bl{bd}_k{k}_n{ns}"


def build_bayesian_sparse_predictor(
    context: Any,
    candidate_elements: List[str],
    config: Dict[str, Any],
) -> Callable:
    """Build a bayesian_sparse predictor compatible with IDWorkflowSpec.

    Parameters
    ----------
    context : UnifiedBenchmarkContext
        Benchmark context with db_path and basis_for_rp().
    candidate_elements : list of str
        Full candidate element list from the benchmark dataset.
    config : dict
        MCMC and detection hyperparameters.

    Returns
    -------
    predictor : callable
        Function mapping BenchmarkSpectrum → ElementIdentificationResult.
    """

    def predictor(spectrum) -> Any:
        from cflibs.inversion.bayesian import (
            BayesianForwardModel,
            MCMCSampler,
            PriorConfig,
        )
        from cflibs.inversion.candidate_prefilter import select_candidate_elements
        from cflibs.inversion.element_id import (
            ElementIdentification,
            ElementIdentificationResult,
        )
        from cflibs.inversion.spectral_nnls_identifier import SpectralNNLSIdentifier

        t_start = time.perf_counter()

        # --- Step 1: Prefilter via NNLS ---
        basis, basis_fwhm, mismatch = context.basis_for_rp(spectrum.rp_estimate)
        nnls_identifier = SpectralNNLSIdentifier(
            basis_library=basis,
            detection_snr=3.0,
            continuum_degree=3,
            fallback_T_K=8000.0,
            fallback_ne_cm3=1e17,
        )

        k_max = int(config.get("k_max", 15))
        prefiltered = select_candidate_elements(
            identifier=nnls_identifier,
            wavelength=spectrum.wavelength_nm,
            intensity=spectrum.intensity,
            force_include=[],
            k_max=k_max,
            k_min=3,
        )

        if not prefiltered:
            logger.warning("Prefilter returned empty set; using full candidates")
            prefiltered = list(candidate_elements)[:k_max]

        t_prefilter = time.perf_counter() - t_start

        # --- Step 2: Bayesian Forward Model ---
        wl_range = (float(spectrum.wavelength_nm[0]), float(spectrum.wavelength_nm[-1]))
        prior_config = PriorConfig(
            baseline_degree=int(config.get("baseline_degree", 0)),
        )

        # Pass the spectrum's instrument broadening to match the prefilter basis
        rp_estimate = getattr(spectrum, "rp_estimate", None)
        fm_kwargs: dict = {}
        if rp_estimate and rp_estimate > 0:
            fm_kwargs["resolving_power"] = float(rp_estimate)
        else:
            fm_kwargs["instrument_fwhm_nm"] = basis_fwhm

        try:
            forward_model = BayesianForwardModel(
                db_path=str(context.db_path),
                elements=prefiltered,
                wavelength_range=wl_range,
                wavelength_grid=spectrum.wavelength_nm,
                **fm_kwargs,
            )
        except Exception as exc:
            logger.error("BayesianForwardModel init failed: %s", exc)
            return _empty_result(candidate_elements, prefiltered, config, str(exc))

        # --- Step 3: MCMC Sampling ---
        sampler = MCMCSampler(
            forward_model=forward_model,
            prior_config=prior_config,
        )

        try:
            mcmc_result = sampler.run(
                observed=spectrum.intensity,
                num_warmup=int(config.get("num_warmup", 500)),
                num_samples=int(config.get("num_samples", 1000)),
                num_chains=int(config.get("num_chains", 1)),
                target_accept_prob=float(config.get("target_accept_prob", 0.8)),
            )
        except Exception as exc:
            logger.error("MCMC sampling failed: %s", exc)
            return _empty_result(candidate_elements, prefiltered, config, str(exc))

        t_mcmc = time.perf_counter() - t_start - t_prefilter

        # --- Step 4: Convert posteriors to ElementIdentificationResult ---
        # Detection: P(c_i > presence_floor) >= posterior_prob_threshold
        # Dirichlet prior makes all concentrations strictly positive,
        # so "CI excludes zero" is always true — use physical presence floor.
        presence_floor = float(config.get("presence_floor", 0.01))
        prob_threshold = float(config.get("posterior_prob_threshold", 0.90))

        conc_samples = mcmc_result.samples.get("concentrations", None)
        conc_mean = mcmc_result.concentrations_mean
        conc_q025 = mcmc_result.concentrations_q025
        conc_q975 = mcmc_result.concentrations_q975

        all_element_ids: List[ElementIdentification] = []

        for i, element in enumerate(prefiltered):
            # concentrations_mean/q025/q975 are Dict[str, float] keyed by element name
            mean_c = float(conc_mean.get(element, 0.0))
            q025 = float(conc_q025.get(element, 0.0))
            q975 = float(conc_q975.get(element, 0.0))

            # Compute detection probability: P(c_i > presence_floor)
            # samples["concentrations"] is a 2D array (n_samples, n_elements)
            # indexed by position matching forward_model.elements order
            if conc_samples is not None and i < conc_samples.shape[1]:
                samples_i = conc_samples[:, i]
                prob_present = float(np.mean(samples_i > presence_floor))
            else:
                # Fallback: use mean
                prob_present = 1.0 if mean_c > presence_floor else 0.0

            detected = prob_present >= prob_threshold

            element_id = ElementIdentification(
                element=element,
                detected=detected,
                score=prob_present,
                confidence=prob_present,
                n_matched_lines=0,
                n_total_lines=0,
                matched_lines=[],
                unmatched_lines=[],
                metadata={
                    "posterior_mean_concentration": mean_c,
                    "posterior_q025": q025,
                    "posterior_q975": q975,
                    "prob_above_floor": prob_present,
                    "presence_floor": presence_floor,
                    "posterior_prob_threshold": prob_threshold,
                },
            )
            all_element_ids.append(element_id)

        # Also mark non-prefiltered candidate elements as rejected
        prefiltered_set = set(prefiltered)
        for element in candidate_elements:
            if element not in prefiltered_set:
                all_element_ids.append(
                    ElementIdentification(
                        element=element,
                        detected=False,
                        score=0.0,
                        confidence=0.0,
                        n_matched_lines=0,
                        n_total_lines=0,
                        matched_lines=[],
                        unmatched_lines=[],
                        metadata={"excluded_by_prefilter": True},
                    )
                )

        t_total = time.perf_counter() - t_start

        # --- Step 5: Build result with diagnostics ---
        detected_elements = [e for e in all_element_ids if e.detected]
        rejected_elements = [e for e in all_element_ids if not e.detected]

        # Convergence diagnostics (r_hat, ess, convergence_status are
        # guaranteed fields on MCMCResult — no hasattr needed)
        max_r_hat = max(mcmc_result.r_hat.values()) if mcmc_result.r_hat else float("nan")
        min_ess = min(mcmc_result.ess.values()) if mcmc_result.ess else float("nan")
        convergence_status = mcmc_result.convergence_status.value

        # Posterior predictive check (optional method on MCMCSampler)
        ppc_p_value = float("nan")
        try:
            ppc = sampler.posterior_predictive_check(mcmc_result, spectrum.intensity, n_samples=100)
            ppc_p_value = ppc.get("p_value", float("nan"))
        except (AttributeError, Exception):
            pass

        result = ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=all_element_ids,
            experimental_peaks=[],
            n_peaks=0,
            n_matched_peaks=0,
            n_unmatched_peaks=0,
            algorithm="bayesian_sparse",
            parameters={
                "config": config,
                "prefiltered_elements": prefiltered,
                "candidate_elements": list(candidate_elements),
                "n_prefiltered": len(prefiltered),
                "max_r_hat": max_r_hat,
                "min_ess": min_ess,
                "posterior_predictive_p_value": ppc_p_value,
                "convergence_status": convergence_status,
                "mcmc_wall_seconds": t_mcmc,
                "prefilter_wall_seconds": t_prefilter,
                "total_wall_seconds": t_total,
                "T_eV_mean": mcmc_result.T_eV_mean,
                "log_ne_mean": mcmc_result.log_ne_mean,
                "basis_fwhm_nm": basis_fwhm,
            },
        )
        return result

    return predictor


def _empty_result(
    candidate_elements: Sequence[str],
    prefiltered: List[str],
    config: Dict[str, Any],
    error_msg: str,
) -> Any:
    """Return an all-rejected result when the Bayesian pipeline fails."""
    from cflibs.inversion.element_id import (
        ElementIdentification,
        ElementIdentificationResult,
    )

    all_rejected = [
        ElementIdentification(
            element=el,
            detected=False,
            score=0.0,
            confidence=0.0,
            n_matched_lines=0,
            n_total_lines=0,
            matched_lines=[],
            unmatched_lines=[],
            metadata={"error": error_msg},
        )
        for el in candidate_elements
    ]
    return ElementIdentificationResult(
        detected_elements=[],
        rejected_elements=all_rejected,
        all_elements=all_rejected,
        experimental_peaks=[],
        n_peaks=0,
        n_matched_peaks=0,
        n_unmatched_peaks=0,
        algorithm="bayesian_sparse",
        parameters={
            "config": config,
            "prefiltered_elements": prefiltered,
            "error": error_msg,
        },
    )
