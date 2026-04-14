"""
SLURM job management for HPC cluster integration.

Provides classes and utilities for submitting, monitoring, and managing
SLURM jobs for CF-LIBS model spectrum generation.
"""

import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class SlurmJobState(Enum):
    """SLURM job states."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    TIMEOUT = "TIMEOUT"
    UNKNOWN = "UNKNOWN"


@dataclass
class SlurmJobConfig:
    """
    Configuration for a SLURM batch job.

    Parameters
    ----------
    job_name : str
        Name for the job (default: "cflibs")
    partition : str
        SLURM partition to submit to (default: "default")
    nodes : int
        Number of nodes to request (default: 1)
    ntasks : int
        Number of tasks (default: 1)
    cpus_per_task : int
        CPUs per task (default: 1)
    mem_gb : int
        Memory in GB (default: 4)
    time_limit : str
        Time limit in HH:MM:SS format (default: "01:00:00")
    account : Optional[str]
        SLURM account name (default: None)
    output_path : str
        Path for stdout (default: "slurm-%j.out")
    error_path : str
        Path for stderr (default: "slurm-%j.err")
    extra_sbatch : Dict[str, str]
        Extra SBATCH directives as key-value pairs
    env_vars : Dict[str, str]
        Environment variables to set
    modules : List[str]
        Modules to load
    """

    job_name: str = "cflibs"
    partition: str = "default"
    nodes: int = 1
    ntasks: int = 1
    cpus_per_task: int = 1
    mem_gb: int = 4
    time_limit: str = "01:00:00"
    account: Optional[str] = None
    output_path: str = "slurm-%j.out"
    error_path: str = "slurm-%j.err"
    extra_sbatch: Dict[str, str] = field(default_factory=dict)
    env_vars: Dict[str, str] = field(default_factory=dict)
    modules: List[str] = field(default_factory=list)


@dataclass
class ArrayJobConfig(SlurmJobConfig):
    """
    Configuration for a SLURM array job.

    Inherits all fields from SlurmJobConfig and adds array-specific parameters.

    Parameters
    ----------
    array_size : int
        Number of array tasks (default: 1)
    max_concurrent : int
        Maximum concurrent array tasks, 0 for unlimited (default: 0)
    """

    array_size: int = 1
    max_concurrent: int = 0

    def __post_init__(self) -> None:
        """Validate array job configuration parameters."""
        if self.array_size < 1:
            raise ValueError(f"array_size must be >= 1, got {self.array_size}")
        if self.max_concurrent < 0:
            raise ValueError(f"max_concurrent must be >= 0, got {self.max_concurrent}")


@dataclass
class SlurmJobStatus:
    """
    Status information for a SLURM job.

    Parameters
    ----------
    job_id : str
        SLURM job ID
    state : SlurmJobState
        Current job state
    submit_time : Optional[str]
        Job submission timestamp
    start_time : Optional[str]
        Job start timestamp
    end_time : Optional[str]
        Job end timestamp
    exit_code : Optional[int]
        Exit code if job completed
    """

    job_id: str
    state: SlurmJobState
    submit_time: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    exit_code: Optional[int] = None


class SlurmJobManager:
    """
    Manages SLURM job submission, monitoring, and cancellation.

    Parameters
    ----------
    dry_run : bool
        If True, print commands instead of executing (default: False)
    """

    _ENV_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    _SBATCH_KEY_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")

    def __init__(self, dry_run: bool = False) -> None:
        self.dry_run = dry_run

    def _validate_env_key(self, key: str) -> None:
        if not self._ENV_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"Invalid environment variable name: {key!r}")

    def _validate_sbatch_key(self, key: str) -> None:
        if not self._SBATCH_KEY_PATTERN.fullmatch(key):
            raise ValueError(f"Invalid SBATCH directive key: {key!r}")

    def generate_sbatch_script(self, config: SlurmJobConfig, script_content: str) -> str:
        """
        Generate SBATCH script content with directives and commands.

        Parameters
        ----------
        config : SlurmJobConfig
            Job configuration
        script_content : str
            Script commands to execute

        Returns
        -------
        str
            Complete SBATCH script content
        """
        lines = ["#!/bin/bash"]

        # Standard SBATCH directives
        lines.append(f"#SBATCH --job-name={config.job_name}")
        lines.append(f"#SBATCH --partition={config.partition}")
        lines.append(f"#SBATCH --nodes={config.nodes}")
        lines.append(f"#SBATCH --ntasks={config.ntasks}")
        lines.append(f"#SBATCH --cpus-per-task={config.cpus_per_task}")
        lines.append(f"#SBATCH --mem={config.mem_gb}G")
        lines.append(f"#SBATCH --time={config.time_limit}")
        lines.append(f"#SBATCH --output={config.output_path}")
        lines.append(f"#SBATCH --error={config.error_path}")

        if config.account is not None:
            lines.append(f"#SBATCH --account={config.account}")

        # Array job directives
        if isinstance(config, ArrayJobConfig):
            array_str = f"0-{config.array_size - 1}"
            if config.max_concurrent > 0:
                array_str += f"%{config.max_concurrent}"
            lines.append(f"#SBATCH --array={array_str}")

        # Extra SBATCH directives
        for key, value in config.extra_sbatch.items():
            self._validate_sbatch_key(key)
            if "\n" in value:
                raise ValueError(f"SBATCH directive value for {key!r} must not contain newlines")
            lines.append(f"#SBATCH --{key}={value}")

        lines.append("")

        # Module loads
        for module in config.modules:
            lines.append(f"module load {shlex.quote(module)}")

        if config.modules:
            lines.append("")

        # Environment variables
        for key, value in config.env_vars.items():
            self._validate_env_key(key)
            if "\n" in value:
                raise ValueError(
                    f"Environment variable value for {key!r} must not contain newlines"
                )
            lines.append(f"export {key}={shlex.quote(value)}")

        if config.env_vars:
            lines.append("")

        # Script content
        lines.append(script_content)

        return "\n".join(lines)

    def submit(
        self,
        config: SlurmJobConfig,
        script_content: str,
        work_dir: Optional[str] = None,
    ) -> str:
        """
        Submit a SLURM job.

        Parameters
        ----------
        config : SlurmJobConfig
            Job configuration
        script_content : str
            Script commands to execute
        work_dir : Optional[str]
            Working directory for job submission

        Returns
        -------
        str
            Job ID (or "DRY_RUN_<jobname>" if dry_run=True)
        """
        script = self.generate_sbatch_script(config, script_content)

        if self.dry_run:
            print("=== DRY RUN: Would submit job ===")
            print(script)
            print("=================================")
            return f"DRY_RUN_{config.job_name}"

        # Submit via sbatch
        cmd = ["sbatch"]
        if work_dir:
            cmd.extend(["-D", work_dir])

        returncode, stdout, stderr = self._run_command(cmd, input_text=script, cwd=work_dir)

        if returncode != 0:
            raise RuntimeError(f"sbatch failed: {stderr}")

        # Parse job ID from output: "Submitted batch job 12345"
        for line in stdout.strip().split("\n"):
            if "Submitted batch job" in line:
                job_id = line.split()[-1]
                return job_id

        raise RuntimeError(f"Failed to parse job ID from sbatch output: {stdout}")

    def submit_with_dependency(
        self,
        config: SlurmJobConfig,
        script_content: str,
        depends_on: List[str],
        dependency_type: str = "afterok",
        work_dir: Optional[str] = None,
    ) -> str:
        """
        Submit a SLURM job with dependencies on other jobs.

        Parameters
        ----------
        config : SlurmJobConfig
            Job configuration
        script_content : str
            Script commands to execute
        depends_on : List[str]
            List of job IDs this job depends on
        dependency_type : str
            Dependency type (default: "afterok")
        work_dir : Optional[str]
            Working directory for job submission

        Returns
        -------
        str
            Job ID (or "DRY_RUN_<jobname>" if dry_run=True)
        """
        if not depends_on:
            raise ValueError("depends_on must contain at least one job ID")

        # Add dependency directive to extra_sbatch
        extra_sbatch_copy = config.extra_sbatch.copy()
        dependency_str = f"{dependency_type}:" + ":".join(depends_on)
        extra_sbatch_copy["dependency"] = dependency_str

        config_copy = replace(
            config,
            extra_sbatch=extra_sbatch_copy,
            env_vars=config.env_vars.copy(),
            modules=config.modules.copy(),
        )

        return self.submit(config_copy, script_content, work_dir)

    def status(self, job_id: str) -> SlurmJobStatus:
        """
        Query the status of a SLURM job.

        Parameters
        ----------
        job_id : str
            SLURM job ID

        Returns
        -------
        SlurmJobStatus
            Job status information
        """
        if self.dry_run:
            return SlurmJobStatus(job_id=job_id, state=SlurmJobState.COMPLETED, exit_code=0)

        # Try squeue first (for running/pending jobs)
        cmd = ["squeue", "-j", job_id, "-h", "-o", "%T"]
        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0 and stdout.strip():
            state_str = stdout.strip()
            state = self._parse_state(state_str)
            return SlurmJobStatus(job_id=job_id, state=state)

        # Try sacct for completed jobs
        cmd = [
            "sacct",
            "-j",
            job_id,
            "-n",
            "-o",
            "State,Submit,Start,End,ExitCode",
            "--delimiter=|",
        ]
        returncode, stdout, stderr = self._run_command(cmd)

        if returncode == 0 and stdout.strip():
            lines = stdout.strip().split("\n")
            if lines:
                # Take first line (main job, not steps)
                parts = lines[0].split("|")
                if len(parts) >= 5:
                    state = self._parse_state(parts[0].strip())
                    submit_time = parts[1].strip() or None
                    start_time = parts[2].strip() or None
                    end_time = parts[3].strip() or None
                    exit_code_str = parts[4].strip()
                    exit_code = None
                    if exit_code_str and ":" in exit_code_str:
                        exit_code = int(exit_code_str.split(":")[0])

                    return SlurmJobStatus(
                        job_id=job_id,
                        state=state,
                        submit_time=submit_time,
                        start_time=start_time,
                        end_time=end_time,
                        exit_code=exit_code,
                    )

        return SlurmJobStatus(job_id=job_id, state=SlurmJobState.UNKNOWN)

    def wait(
        self, job_id: str, poll_interval: float = 10.0, timeout: float = 3600.0
    ) -> SlurmJobStatus:
        """
        Wait for a job to complete.

        Parameters
        ----------
        job_id : str
            SLURM job ID
        poll_interval : float
            Seconds between status checks (default: 10.0)
        timeout : float
            Maximum seconds to wait (default: 3600.0)

        Returns
        -------
        SlurmJobStatus
            Final job status

        Raises
        ------
        TimeoutError
            If job does not complete within timeout
        """
        start = time.time()
        while time.time() - start < timeout:
            status = self.status(job_id)
            if status.state in [
                SlurmJobState.COMPLETED,
                SlurmJobState.FAILED,
                SlurmJobState.CANCELLED,
                SlurmJobState.TIMEOUT,
            ]:
                return status
            time.sleep(poll_interval)

        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    def cancel(self, job_id: str) -> bool:
        """
        Cancel a SLURM job.

        Parameters
        ----------
        job_id : str
            SLURM job ID

        Returns
        -------
        bool
            True if cancellation succeeded
        """
        if self.dry_run:
            print(f"=== DRY RUN: Would cancel job {job_id} ===")
            return True

        cmd = ["scancel", job_id]
        returncode, stdout, stderr = self._run_command(cmd)
        return returncode == 0

    @staticmethod
    def _run_command(
        cmd: List[str],
        input_text: Optional[str] = None,
        cwd: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> Tuple[int, str, str]:
        """
        Run a shell command and return (returncode, stdout, stderr).

        Parameters
        ----------
        cmd : List[str]
            Command and arguments
        input_text : Optional[str]
            Text to send to stdin
        cwd : Optional[str]
            Working directory
        timeout : Optional[float]
            Command timeout in seconds (default: None)

        Returns
        -------
        Tuple[int, str, str]
            (returncode, stdout, stderr)
        """
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr

    @staticmethod
    def _parse_state(state_str: str) -> SlurmJobState:
        """
        Parse SLURM state string to SlurmJobState enum.

        Parameters
        ----------
        state_str : str
            SLURM state string

        Returns
        -------
        SlurmJobState
            Parsed state
        """
        state_upper = state_str.upper()
        if "PEND" in state_upper:
            return SlurmJobState.PENDING
        elif "RUN" in state_upper:
            return SlurmJobState.RUNNING
        elif "COMP" in state_upper:
            return SlurmJobState.COMPLETED
        elif "FAIL" in state_upper or "OUT_OF_MEMORY" in state_upper:
            return SlurmJobState.FAILED
        elif "CANC" in state_upper or "PREEMPT" in state_upper:
            return SlurmJobState.CANCELLED
        elif "TIMEOUT" in state_upper or "TO" == state_upper:
            return SlurmJobState.TIMEOUT
        else:
            return SlurmJobState.UNKNOWN


def generate_distributed_mcmc_script(
    db_path: str,
    elements: List[str],
    wavelength_range: Tuple[float, float],
    observed_path: str,
    output_dir: str = ".",
    ntasks: int = 4,
    gpus_per_task: int = 0,
    chains_per_rank: int = 1,
    num_warmup: int = 500,
    num_samples: int = 1000,
    partition: str = "default",
    time_limit: str = "04:00:00",
    mem_gb: int = 16,
    account: Optional[str] = None,
    modules: Optional[List[str]] = None,
    conda_env: Optional[str] = None,
) -> str:
    """Generate a SLURM SBATCH script content for distributed MCMC.

    Parameters
    ----------
    db_path : str
        Path to the atomic database.
    elements : list of str
        Elements to include in the analysis.
    wavelength_range : tuple of float
        ``(wl_min, wl_max)`` in nm.
    observed_path : str
        Path to the observed spectrum file (NumPy ``.npy``).
    output_dir : str
        Directory for results (default: ``"."``).
    ntasks : int
        Number of MPI tasks (ranks) (default: 4).
    gpus_per_task : int
        GPUs per task (0 = CPU only) (default: 0).
    chains_per_rank : int
        NUTS chains per MPI rank (default: 1).
    num_warmup : int
        Warmup samples per chain (default: 500).
    num_samples : int
        Posterior samples per chain (default: 1000).
    partition : str
        SLURM partition (default: ``"default"``).
    time_limit : str
        Time limit in ``HH:MM:SS`` (default: ``"04:00:00"``).
    mem_gb : int
        Memory per node in GB (default: 16).
    account : str, optional
        SLURM account.
    modules : list of str, optional
        Environment modules to load.
    conda_env : str, optional
        Conda environment to activate.

    Returns
    -------
    str
        The SBATCH script content.
    """
    wl_min, wl_max = wavelength_range

    lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=cflibs-mcmc",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --ntasks={ntasks}",
        "#SBATCH --cpus-per-task=2",
        f"#SBATCH --mem={mem_gb}G",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={output_dir}/cflibs-mcmc-%j.out",
        f"#SBATCH --error={output_dir}/cflibs-mcmc-%j.err",
    ]

    if account:
        lines.append(f"#SBATCH --account={account}")
    if gpus_per_task > 0:
        lines.append(f"#SBATCH --gpus-per-task={gpus_per_task}")

    lines.append("")

    # Module loading
    if modules:
        for mod in modules:
            lines.append(f"module load {shlex.quote(mod)}")
        lines.append("")

    # Conda env
    if conda_env:
        lines.append(f"conda activate {shlex.quote(conda_env)}")
        lines.append("")

    # JAX environment variables
    lines.extend(
        [
            "# JAX configuration",
            "export JAX_ENABLE_X64=True",
            "export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9",
        ]
    )
    if gpus_per_task > 0:
        lines.append("export JAX_PLATFORMS=gpu")
    else:
        lines.append("export JAX_PLATFORMS=cpu")
    lines.append("")

    # Python inline script — use repr() for safe string escaping in Python literals
    use_gpu = "True" if gpus_per_task > 0 else "False"
    db_path_repr = repr(db_path)
    observed_path_repr = repr(observed_path)
    output_dir_repr = repr(output_dir)
    lines.extend(
        [
            "# Run distributed MCMC",
            f'mpirun -n {ntasks} python -c "',
            "import numpy as np",
            "from cflibs.inversion.bayesian import BayesianForwardModel, bayesian_model",
            "from cflibs.hpc.distributed_mcmc import DistributedMCMCSampler, DistributedMCMCConfig",
            "",
            f"model = BayesianForwardModel({db_path_repr}, {elements!r}, ({wl_min}, {wl_max}))",
            f"observed = np.load({observed_path_repr})",
            "config = DistributedMCMCConfig(",
            f"    chains_per_rank={chains_per_rank},",
            f"    num_warmup={num_warmup},",
            f"    num_samples={num_samples},",
            f"    use_gpu={use_gpu},",
            ")",
            "sampler = DistributedMCMCSampler(model, bayesian_model, config=config)",
            "result = sampler.run(observed)",
            "if result is not None:",
            f"    np.savez({output_dir_repr} + '/mcmc_result.npz', **result.samples)",
            "    print(f'R-hat: {{result.r_hat}}')",
            "    print(f'ESS: {{result.ess}}')",
            '"',
        ]
    )

    script_content = "\n".join(lines) + "\n"
    return script_content


def create_distributed_mcmc_job(script_path: str, **kwargs: Any) -> str:
    """Generate a SLURM SBATCH script for distributed MCMC.

    This function wraps :func:`generate_distributed_mcmc_script` to also
    write the generated script content to a file.

    Parameters
    ----------
    script_path : str
        Path to write the generated SBATCH script.
    **kwargs : Any
        Arguments passed to :func:`generate_distributed_mcmc_script`.

    Returns
    -------
    str
        The SBATCH script content (also written to *script_path*).
    """
    script_content = generate_distributed_mcmc_script(**kwargs)

    with open(script_path, "w") as f:
        f.write(script_content)

    return script_content
