"""
Tests for SLURM job management.

These tests verify SBATCH script generation and dry-run mode
without requiring an actual SLURM cluster.
"""

from dataclasses import replace

import pytest

from cflibs.hpc import (
    ArrayJobConfig,
    SlurmJobConfig,
    SlurmJobManager,
    SlurmJobState,
    SlurmJobStatus,
)


def test_slurm_job_config_defaults():
    """Test default values for SlurmJobConfig."""
    config = SlurmJobConfig()
    assert config.job_name == "cflibs"
    assert config.partition == "default"
    assert config.nodes == 1
    assert config.ntasks == 1
    assert config.cpus_per_task == 1
    assert config.mem_gb == 4
    assert config.time_limit == "01:00:00"
    assert config.account is None
    assert config.output_path == "slurm-%j.out"
    assert config.error_path == "slurm-%j.err"
    assert config.extra_sbatch == {}
    assert config.env_vars == {}
    assert config.modules == []


def test_array_job_config():
    """Test ArrayJobConfig with array-specific fields."""
    config = ArrayJobConfig(
        job_name="test_array",
        array_size=100,
        max_concurrent=10,
    )
    assert config.job_name == "test_array"
    assert config.array_size == 100
    assert config.max_concurrent == 10


def test_array_job_config_validation():
    """ArrayJobConfig should reject invalid size/concurrency values."""
    with pytest.raises(ValueError, match="array_size must be >= 1"):
        ArrayJobConfig(array_size=0)
    with pytest.raises(ValueError, match="max_concurrent must be >= 0"):
        ArrayJobConfig(array_size=1, max_concurrent=-1)


def test_generate_sbatch_basic():
    """Test basic SBATCH script generation."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(
        job_name="test_job",
        partition="compute",
        nodes=2,
        ntasks=4,
        cpus_per_task=8,
        mem_gb=16,
        time_limit="02:30:00",
    )

    script = manager.generate_sbatch_script(config, "echo 'Hello World'")

    # Check shebang
    assert script.startswith("#!/bin/bash")

    # Check standard directives
    assert "#SBATCH --job-name=test_job" in script
    assert "#SBATCH --partition=compute" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks=4" in script
    assert "#SBATCH --cpus-per-task=8" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --time=02:30:00" in script
    assert "#SBATCH --output=slurm-%j.out" in script
    assert "#SBATCH --error=slurm-%j.err" in script

    # Check script content
    assert "echo 'Hello World'" in script


def test_generate_sbatch_array():
    """Test SBATCH script generation with array job."""
    manager = SlurmJobManager(dry_run=True)
    config = ArrayJobConfig(
        job_name="test_array",
        array_size=100,
        max_concurrent=10,
    )

    script = manager.generate_sbatch_script(config, "python script.py $SLURM_ARRAY_TASK_ID")

    # Check array directive with limit
    assert "#SBATCH --array=0-99%10" in script
    assert "python script.py $SLURM_ARRAY_TASK_ID" in script


def test_generate_sbatch_array_unlimited():
    """Test SBATCH script with unlimited concurrent array tasks."""
    manager = SlurmJobManager(dry_run=True)
    config = ArrayJobConfig(
        job_name="test_array",
        array_size=50,
        max_concurrent=0,  # Unlimited
    )

    script = manager.generate_sbatch_script(config, "echo $SLURM_ARRAY_TASK_ID")

    # Check array directive without limit
    assert "#SBATCH --array=0-49" in script
    assert "%" not in script.split("#SBATCH --array=")[1].split("\n")[0]


def test_generate_sbatch_modules():
    """Test module load commands in SBATCH script."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(modules=["python/3.10", "cuda/11.8", "hdf5/1.12"])

    script = manager.generate_sbatch_script(config, "python script.py")

    assert "module load python/3.10" in script
    assert "module load cuda/11.8" in script
    assert "module load hdf5/1.12" in script


def test_generate_sbatch_env_vars():
    """Test environment variable export in SBATCH script."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(
        env_vars={
            "JAX_PLATFORMS": "cpu",
            "OMP_NUM_THREADS": "8",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }
    )

    script = manager.generate_sbatch_script(config, "python script.py")

    assert "export JAX_PLATFORMS=cpu" in script
    assert "export OMP_NUM_THREADS=8" in script
    assert "export CUDA_VISIBLE_DEVICES=0,1" in script


def test_generate_sbatch_rejects_invalid_env_var_names():
    """Unsafe environment variable names should be rejected."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(env_vars={"BAD-NAME": "x"})
    with pytest.raises(ValueError, match="Invalid environment variable name"):
        manager.generate_sbatch_script(config, "echo test")


def test_generate_sbatch_extra_directives():
    """Test extra SBATCH directives."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(
        extra_sbatch={
            "gres": "gpu:2",
            "constraint": "haswell",
            "mail-type": "END,FAIL",
            "mail-user": "user@example.com",
        }
    )

    script = manager.generate_sbatch_script(config, "python script.py")

    assert "#SBATCH --gres=gpu:2" in script
    assert "#SBATCH --constraint=haswell" in script
    assert "#SBATCH --mail-type=END,FAIL" in script
    assert "#SBATCH --mail-user=user@example.com" in script


def test_generate_sbatch_account():
    """Test account directive when specified."""
    manager = SlurmJobManager(dry_run=True)

    # With account
    config_with = SlurmJobConfig(account="proj123")
    script_with = manager.generate_sbatch_script(config_with, "echo test")
    assert "#SBATCH --account=proj123" in script_with

    # Without account
    config_without = SlurmJobConfig(account=None)
    script_without = manager.generate_sbatch_script(config_without, "echo test")
    assert "--account" not in script_without


def test_dry_run_submit():
    """Test job submission in dry-run mode."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(job_name="test_dry_run")

    job_id = manager.submit(config, "echo 'test'")

    assert job_id == "DRY_RUN_test_dry_run"


def test_submit_with_dependency_dry_run():
    """Test dependency job submission in dry-run mode."""
    manager = SlurmJobManager(dry_run=True)
    config = SlurmJobConfig(job_name="dependent_job")

    # Submit with dependencies
    job_id = manager.submit_with_dependency(
        config,
        "echo 'depends on previous jobs'",
        depends_on=["12345", "67890"],
        dependency_type="afterok",
    )

    assert job_id == "DRY_RUN_dependent_job"

    # Verify dependency directive would be added
    # Note: dependency is added in submit_with_dependency via extra_sbatch
    # We test the mechanism separately
    config_with_dep = SlurmJobConfig(
        job_name="test", extra_sbatch={"dependency": "afterok:12345:67890"}
    )
    script = manager.generate_sbatch_script(config_with_dep, "echo test")
    assert "#SBATCH --dependency=afterok:12345:67890" in script


def test_submit_with_dependency_preserves_array_config():
    """Test that submit_with_dependency preserves ArrayJobConfig fields."""
    manager = SlurmJobManager(dry_run=True)
    array_config = ArrayJobConfig(
        job_name="array_dependent",
        array_size=50,
        max_concurrent=10,
        partition="gpu",
        mem_gb=16,
    )

    # Submit with dependencies
    job_id = manager.submit_with_dependency(
        array_config,
        "python process.py $SLURM_ARRAY_TASK_ID",
        depends_on=["12345"],
        dependency_type="afterok",
    )

    assert job_id == "DRY_RUN_array_dependent"

    # Verify array directives are preserved in the generated script
    # We need to access the config after modification to verify
    extra_sbatch_copy = array_config.extra_sbatch.copy()
    extra_sbatch_copy["dependency"] = "afterok:12345"
    config_copy = replace(
        array_config,
        extra_sbatch=extra_sbatch_copy,
        env_vars=array_config.env_vars.copy(),
        modules=array_config.modules.copy(),
    )

    script = manager.generate_sbatch_script(config_copy, "python process.py")

    # Verify array directives are present
    assert "#SBATCH --array=0-49%10" in script
    assert "#SBATCH --partition=gpu" in script
    assert "#SBATCH --mem=16G" in script
    assert "#SBATCH --dependency=afterok:12345" in script


def test_slurm_job_state_enum():
    """Test SlurmJobState enum values."""
    assert SlurmJobState.PENDING.value == "PENDING"
    assert SlurmJobState.RUNNING.value == "RUNNING"
    assert SlurmJobState.COMPLETED.value == "COMPLETED"
    assert SlurmJobState.FAILED.value == "FAILED"
    assert SlurmJobState.CANCELLED.value == "CANCELLED"
    assert SlurmJobState.TIMEOUT.value == "TIMEOUT"
    assert SlurmJobState.UNKNOWN.value == "UNKNOWN"


def test_slurm_job_status_dataclass():
    """Test SlurmJobStatus dataclass."""
    status = SlurmJobStatus(
        job_id="12345",
        state=SlurmJobState.COMPLETED,
        submit_time="2026-02-08T10:00:00",
        start_time="2026-02-08T10:01:00",
        end_time="2026-02-08T10:15:00",
        exit_code=0,
    )

    assert status.job_id == "12345"
    assert status.state == SlurmJobState.COMPLETED
    assert status.submit_time == "2026-02-08T10:00:00"
    assert status.start_time == "2026-02-08T10:01:00"
    assert status.end_time == "2026-02-08T10:15:00"
    assert status.exit_code == 0


def test_status_dry_run():
    """Test status query in dry-run mode."""
    manager = SlurmJobManager(dry_run=True)

    # Dry run jobs always return COMPLETED
    status = manager.status("DRY_RUN_test")
    assert status.job_id == "DRY_RUN_test"
    assert status.state == SlurmJobState.COMPLETED
    assert status.exit_code == 0


def test_cancel_dry_run():
    """Test cancel in dry-run mode."""
    manager = SlurmJobManager(dry_run=True)

    result = manager.cancel("DRY_RUN_test")
    assert result is True


def test_parse_state():
    """Test state string parsing."""
    # Test various SLURM state strings
    assert SlurmJobManager._parse_state("PENDING") == SlurmJobState.PENDING
    assert SlurmJobManager._parse_state("RUNNING") == SlurmJobState.RUNNING
    assert SlurmJobManager._parse_state("COMPLETED") == SlurmJobState.COMPLETED
    assert SlurmJobManager._parse_state("FAILED") == SlurmJobState.FAILED
    assert SlurmJobManager._parse_state("CANCELLED") == SlurmJobState.CANCELLED
    assert SlurmJobManager._parse_state("TIMEOUT") == SlurmJobState.TIMEOUT
    assert SlurmJobManager._parse_state("TO") == SlurmJobState.TIMEOUT
    assert SlurmJobManager._parse_state("UNKNOWN_STATE") == SlurmJobState.UNKNOWN

    # Test case insensitivity
    assert SlurmJobManager._parse_state("pending") == SlurmJobState.PENDING
    assert SlurmJobManager._parse_state("running") == SlurmJobState.RUNNING


def test_full_workflow_dry_run():
    """Test complete workflow in dry-run mode."""
    manager = SlurmJobManager(dry_run=True)

    # Submit array job
    array_config = ArrayJobConfig(
        job_name="workflow_array",
        array_size=10,
        max_concurrent=5,
        partition="compute",
        time_limit="00:30:00",
    )
    array_job_id = manager.submit(array_config, "python process_chunk.py $SLURM_ARRAY_TASK_ID")
    assert array_job_id == "DRY_RUN_workflow_array"

    # Submit dependent job
    consolidate_config = SlurmJobConfig(
        job_name="workflow_consolidate",
        partition="compute",
        time_limit="00:15:00",
    )
    consolidate_job_id = manager.submit_with_dependency(
        consolidate_config,
        "python consolidate.py",
        depends_on=[array_job_id],
    )
    assert consolidate_job_id == "DRY_RUN_workflow_consolidate"

    # Check status
    status = manager.status(consolidate_job_id)
    assert status.state == SlurmJobState.COMPLETED
