"""Two-zone (core+shell) MCMC sampler for self-reversed LIBS plasmas (T1-6).

Hosts :class:`TwoZoneMCMCSampler`. Separated from :mod:`samplers` to keep
each file under the 800-LOC limit imposed by ADR-0001 / T1-6 spec section 6.

See :mod:`forward` for :class:`TwoZoneBayesianForwardModel` and
:func:`two_zone_bayesian_model`.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.inversion.common.strict import resolve_strict

from .atomic import _as_jax_real
from .forward import TwoZoneBayesianForwardModel, two_zone_bayesian_model
from .priors import HAS_NUMPYRO, NoiseParameters, TwoZonePriorConfig
from .results import TwoZoneMCMCResult
from .samplers import (
    _assess_convergence,
    _count_divergences,
    _diagnostics_from_mcmc,
    _strict_convergence_gate,
    _to_arviz,
)

logger = get_logger("inversion.bayesian.two_zone")


if HAS_NUMPYRO:
    from numpyro.infer import MCMC, NUTS, init_to_uniform
else:  # pragma: no cover
    MCMC = None  # type: ignore[assignment]
    NUTS = None  # type: ignore[assignment]
    init_to_uniform = None  # type: ignore[assignment]


class TwoZoneMCMCSampler:
    """MCMC sampler for two-zone Bayesian CF-LIBS inference."""

    def __init__(
        self,
        forward_model: TwoZoneBayesianForwardModel,
        prior_config: TwoZonePriorConfig = TwoZonePriorConfig(),
        noise_params: NoiseParameters = NoiseParameters(),
        strict: Optional[bool] = None,
    ):
        if not HAS_NUMPYRO:
            raise ImportError("NumPyro required. Install with: pip install numpyro")

        self.forward_model = forward_model
        self.prior_config = prior_config
        self.noise_params = noise_params
        self.elements = forward_model.elements
        # No-fallback mode (resolved via CFLIBS_NO_FALLBACK when None).
        self.strict = resolve_strict(strict)

    def run(
        self,
        observed: np.ndarray,
        num_warmup: int = 500,
        num_samples: int = 1000,
        num_chains: int = 1,
        seed: int = 0,
        target_accept_prob: float = 0.8,
        max_tree_depth: int = 10,
        progress_bar: bool = True,
    ) -> TwoZoneMCMCResult:
        """Run MCMC sampling with the two-zone model."""
        import jax.random as random

        observed_jax = _as_jax_real(observed)

        def model(obs):
            two_zone_bayesian_model(
                self.forward_model,
                obs,
                self.prior_config,
                self.noise_params,
                strict=self.strict,
            )

        kernel = NUTS(
            model,
            init_strategy=init_to_uniform(radius=0.5),
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
        )

        rng_key = random.PRNGKey(seed)
        logger.info(
            f"Starting two-zone MCMC: {num_chains} chains, "
            f"{num_warmup} warmup, {num_samples} samples"
        )
        # Collect divergent-transition diagnostics (does not alter the posterior).
        mcmc.run(rng_key, observed_jax, extra_fields=("diverging",))
        n_divergences = _count_divergences(mcmc)

        samples = mcmc.get_samples(group_by_chain=(num_chains > 1))
        n_el = len(self.elements)

        T_core_flat = np.array(samples["T_core_eV"]).flatten()
        T_shell_flat = np.array(samples["T_shell_eV"]).flatten()
        log_ne_flat = np.array(samples["log_ne"]).flatten()
        conc_flat = np.array(samples["concentrations"]).reshape(-1, n_el)
        sf_flat = np.array(samples["shell_fraction"]).flatten()
        ods_flat = np.array(samples["optical_depth_scale"]).flatten()

        r_hat, ess = _diagnostics_from_mcmc(
            mcmc, num_chains, variables=("T_core_eV", "T_shell_eV", "log_ne")
        )
        status = _assess_convergence(r_hat, ess, num_samples)

        result = TwoZoneMCMCResult(
            samples={k: np.array(v) for k, v in samples.items()},
            T_core_eV_mean=float(np.mean(T_core_flat)),
            T_core_eV_std=float(np.std(T_core_flat)),
            T_core_eV_q025=float(np.percentile(T_core_flat, 2.5)),
            T_core_eV_q975=float(np.percentile(T_core_flat, 97.5)),
            T_shell_eV_mean=float(np.mean(T_shell_flat)),
            T_shell_eV_std=float(np.std(T_shell_flat)),
            T_shell_eV_q025=float(np.percentile(T_shell_flat, 2.5)),
            T_shell_eV_q975=float(np.percentile(T_shell_flat, 97.5)),
            log_ne_mean=float(np.mean(log_ne_flat)),
            log_ne_std=float(np.std(log_ne_flat)),
            log_ne_q025=float(np.percentile(log_ne_flat, 2.5)),
            log_ne_q975=float(np.percentile(log_ne_flat, 97.5)),
            shell_fraction_mean=float(np.mean(sf_flat)),
            shell_fraction_std=float(np.std(sf_flat)),
            optical_depth_scale_mean=float(np.mean(ods_flat)),
            optical_depth_scale_std=float(np.std(ods_flat)),
            concentrations_mean={
                el: float(np.mean(conc_flat[:, i])) for i, el in enumerate(self.elements)
            },
            concentrations_std={
                el: float(np.std(conc_flat[:, i])) for i, el in enumerate(self.elements)
            },
            concentrations_q025={
                el: float(np.percentile(conc_flat[:, i], 2.5)) for i, el in enumerate(self.elements)
            },
            concentrations_q975={
                el: float(np.percentile(conc_flat[:, i], 97.5))
                for i, el in enumerate(self.elements)
            },
            r_hat=r_hat,
            ess=ess,
            convergence_status=status,
            n_samples=num_samples,
            n_chains=num_chains,
            n_warmup=num_warmup,
            inference_data=_to_arviz(mcmc),
            n_divergences=n_divergences,
        )

        logger.info(
            f"Two-zone MCMC complete: T_core={result.T_core_eV_mean:.3f} eV, "
            f"T_shell={result.T_shell_eV_mean:.3f} eV, "
            f"n_e={result.n_e_mean:.2e} cm^-3"
        )
        # Strict / no-fallback gate (off by default -> result returned as-is).
        if self.strict:
            _strict_convergence_gate(
                status=status,
                r_hat=r_hat,
                ess=ess,
                n_divergences=n_divergences,
                num_samples=num_samples,
                solver="bayesian.two_zone_mcmc",
                strict=True,
            )
        return result


__all__ = ["TwoZoneMCMCSampler"]
