"""Distributed MCMC via MPI for multi-node CF-LIBS Bayesian inference.

Each MPI rank runs independent NUTS chains.  Rank 0 gathers all samples and
computes cross-chain convergence diagnostics (R-hat, ESS) via ArviZ.

MPI was chosen over Ray because:
- SLURM infrastructure already exists on target clusters.
- ``mpi4py`` is already in the ``[cluster]`` dependency group.
- The communication pattern is simple scatter/gather of independent chains.

Usage (SLURM)::

    srun --ntasks=4 python -m cflibs.hpc.distributed_mcmc \\
        --db-path atomic.db --elements Fe Cu --wl-range 200 600

Or programmatically::

    from cflibs.hpc.distributed_mcmc import DistributedMCMCSampler
    sampler = DistributedMCMCSampler(forward_model, config=config)
    result = sampler.run(observed)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from cflibs.core.logging_config import get_logger

logger = get_logger("hpc.distributed_mcmc")

try:
    from mpi4py import MPI

    HAS_MPI = True
except ImportError:
    HAS_MPI = False
    MPI = None

# JAX, NumPyro, and ArviZ are imported lazily in run() so that
# CUDA_VISIBLE_DEVICES can be set before JAX initialisation.
HAS_JAX = False
HAS_ARVIZ = False
HAS_NUMPYRO = False

try:
    import importlib.util

    HAS_JAX = importlib.util.find_spec("jax") is not None
    HAS_ARVIZ = importlib.util.find_spec("arviz") is not None
    HAS_NUMPYRO = importlib.util.find_spec("numpyro") is not None
except Exception:
    pass


@dataclass
class DistributedMCMCConfig:
    """Configuration for distributed MCMC sampling.

    Parameters
    ----------
    chains_per_rank : int
        Number of NUTS chains each MPI rank runs (default: 1).
    num_warmup : int
        Warmup samples per chain (default: 500).
    num_samples : int
        Posterior samples per chain (default: 1000).
    use_gpu : bool
        If True, pin each rank to a GPU (default: False).
    target_accept_prob : float
        NUTS target acceptance probability (default: 0.8).
    max_tree_depth : int
        Maximum NUTS tree depth (default: 10).
    seed_offset : int
        Base seed; each rank adds its rank number (default: 0).
    """

    chains_per_rank: int = 1
    num_warmup: int = 500
    num_samples: int = 1000
    use_gpu: bool = False
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    seed_offset: int = 0


@dataclass
class DistributedMCMCResult:
    """Result from distributed MCMC (rank-0 only).

    Attributes
    ----------
    samples : dict
        Merged posterior samples ``{name: array}`` from all ranks.
    r_hat : dict
        Cross-chain R-hat for each parameter.
    ess : dict
        Effective sample sizes for each parameter.
    total_chains : int
        Total number of chains across all ranks.
    total_samples : int
        Total posterior samples (all chains combined).
    rank_chain_counts : list of int
        Number of chains contributed by each rank.
    """

    samples: Dict[str, np.ndarray]
    r_hat: Dict[str, float] = field(default_factory=dict)
    ess: Dict[str, float] = field(default_factory=dict)
    total_chains: int = 0
    total_samples: int = 0
    rank_chain_counts: List[int] = field(default_factory=list)


class DistributedMCMCSampler:
    """MPI-distributed MCMC sampler for CF-LIBS Bayesian inference.

    Each MPI rank runs ``chains_per_rank`` independent NUTS chains.
    Rank 0 gathers all samples and computes cross-chain diagnostics.

    Parameters
    ----------
    forward_model : BayesianForwardModel
        The forward model instance (must be picklable or available on all
        ranks via shared filesystem).
    model_fn : callable
        NumPyro model function (e.g. ``bayesian_model`` or
        ``two_zone_bayesian_model``).
    config : DistributedMCMCConfig
        Distributed sampling configuration.
    comm : MPI.Comm, optional
        MPI communicator (defaults to ``MPI.COMM_WORLD``).
    """

    def __init__(
        self,
        forward_model: Any,
        model_fn: Any,
        config: DistributedMCMCConfig = DistributedMCMCConfig(),
        comm: Any = None,
        **model_kwargs: Any,
    ):
        if not HAS_MPI:
            raise ImportError("mpi4py required. Install with: pip install mpi4py")
        if not HAS_NUMPYRO:
            raise ImportError("NumPyro required. Install with: pip install numpyro")
        if not HAS_JAX:
            raise ImportError("JAX required. Install with: pip install jax")

        self.forward_model = forward_model
        self.model_fn = model_fn
        self.config = config
        self.model_kwargs = model_kwargs
        self.comm = comm if comm is not None else MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def run(self, observed: np.ndarray) -> Optional[DistributedMCMCResult]:
        """Run distributed MCMC.

        All ranks participate in sampling.  Only rank 0 returns a
        :class:`DistributedMCMCResult`; other ranks return ``None``.

        Parameters
        ----------
        observed : np.ndarray
            Observed spectrum (broadcast from rank 0).

        Returns
        -------
        DistributedMCMCResult or None
            Merged result on rank 0; ``None`` on other ranks.
        """
        cfg = self.config

        # GPU pinning must happen BEFORE JAX import to set CUDA_VISIBLE_DEVICES
        if cfg.use_gpu:
            from cflibs.hpc.gpu_config import configure_gpu

            local_rank = int(self.comm.Get_rank())
            configure_gpu(device_id=local_rank, enable_x64=True)

        import jax.numpy as jnp
        import jax.random as random
        from numpyro.infer import MCMC, NUTS, init_to_uniform

        # Broadcast observed spectrum from rank 0
        observed = self.comm.bcast(observed, root=0)
        observed_jax = jnp.array(observed)

        # Per-rank seed for reproducibility
        rank_seed = cfg.seed_offset + self.rank

        # Build model closure (forward extra kwargs like prior_config, noise_params)
        model_kwargs = self.model_kwargs

        def model(obs):
            self.model_fn(self.forward_model, obs, **model_kwargs)

        # Create and run local MCMC
        kernel = NUTS(
            model,
            init_strategy=init_to_uniform(radius=0.5),
            target_accept_prob=cfg.target_accept_prob,
            max_tree_depth=cfg.max_tree_depth,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=cfg.num_warmup,
            num_samples=cfg.num_samples,
            num_chains=cfg.chains_per_rank,
            progress_bar=(self.rank == 0),
        )

        rng_key = random.PRNGKey(rank_seed)
        logger.info(
            f"Rank {self.rank}/{self.size}: running {cfg.chains_per_rank} chains "
            f"(warmup={cfg.num_warmup}, samples={cfg.num_samples})"
        )

        mcmc.run(rng_key, observed_jax)
        local_samples = {k: np.array(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}

        # Gather all samples to rank 0
        all_samples_list = self.comm.gather(local_samples, root=0)

        if self.rank != 0:
            return None

        # Merge on rank 0
        return self._merge_results(all_samples_list, cfg)

    def _merge_results(
        self,
        all_samples_list: List[Dict[str, np.ndarray]],
        cfg: DistributedMCMCConfig,
    ) -> DistributedMCMCResult:
        """Merge samples from all ranks and compute diagnostics."""
        merged: Dict[str, List[np.ndarray]] = {}
        rank_counts = []

        for rank_samples in all_samples_list:
            n_chains = 0
            for key, arr in rank_samples.items():
                if key not in merged:
                    merged[key] = []
                # arr shape: (n_chains, n_samples, ...) or (n_samples, ...)
                if arr.ndim >= 2:
                    merged[key].append(arr)
                    n_chains = arr.shape[0]
                else:
                    merged[key].append(arr[np.newaxis, :])
                    n_chains = 1
            rank_counts.append(n_chains)

        # Concatenate along chain axis
        combined = {}
        for key, arrs in merged.items():
            combined[key] = np.concatenate(arrs, axis=0)

        total_chains = sum(rank_counts)
        total_samples = total_chains * cfg.num_samples

        # Compute cross-chain diagnostics with ArviZ
        r_hat: Dict[str, float] = {}
        ess: Dict[str, float] = {}

        if HAS_ARVIZ and total_chains > 1:
            try:
                import arviz as az

                # Build InferenceData manually from combined chains
                posterior_dict = {}
                for key, arr in combined.items():
                    # Shape: (n_chains, n_samples, ...)
                    posterior_dict[key] = arr

                idata = az.from_dict(posterior=posterior_dict)
                rhat_data = az.rhat(idata)
                ess_data = az.ess(idata)

                for var in combined.keys():
                    if var in rhat_data:
                        val = rhat_data[var]
                        if hasattr(val, "values"):
                            r_hat[var] = float(np.mean(val.values))
                        else:
                            r_hat[var] = float(val)
                    if var in ess_data:
                        val = ess_data[var]
                        if hasattr(val, "values"):
                            ess[var] = float(np.mean(val.values))
                        else:
                            ess[var] = float(val)
            except Exception as e:
                logger.warning(f"ArviZ diagnostics failed: {e}")

        # Flatten for the result
        flat_samples = {}
        for key, arr in combined.items():
            flat_samples[key] = arr.reshape(-1, *arr.shape[2:])

        logger.info(
            f"Distributed MCMC complete: {total_chains} chains, "
            f"{total_samples} total samples from {len(rank_counts)} ranks"
        )

        return DistributedMCMCResult(
            samples=flat_samples,
            r_hat=r_hat,
            ess=ess,
            total_chains=total_chains,
            total_samples=total_samples,
            rank_chain_counts=rank_counts,
        )