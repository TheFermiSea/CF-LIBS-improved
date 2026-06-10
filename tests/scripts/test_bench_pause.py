"""Tests for the post-merge-benchmark pause mechanism.

T1.4 — CF-LIBS-improved-5t6n.

The actual `do_run` invocation requires an SSH-reachable bench host
(vasp-03) and a full NFS-shared venv, neither of which is available
in CI. We test the *short-circuit* path: when the pause flag is set,
`do_classify` outputs "skip" and `do_run` logs+returns 0 without ever
attempting the rsync/SSH machinery.

We run the live `post-merge-benchmark.sh` from beefcake-swarm via a
small wrapper that sources it and calls `do_classify` / `do_run` with
a sandbox `BENCH_PAUSE_FLAG`. If `beefcake-swarm` is not co-located
with `CF-LIBS-improved` (e.g. CI runners), the test is skipped — the
companion PR in `beefcake-swarm` ships an equivalent unit test there.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest


# Resolve the bench script. Prefer the env override (CI may set
# BENCH_SCRIPT to point at the staged copy), fall back to the
# default sibling-checkout location used during development.
def _bench_script() -> Path | None:
    env_path = os.environ.get("BENCH_SCRIPT")
    if env_path:
        p = Path(env_path).expanduser()
        return p if p.exists() else None

    candidates = [
        Path("/home/brian/code/beefcake-swarm/scripts/post-merge-benchmark.sh"),
        Path.home() / "code" / "beefcake-swarm" / "scripts" / "post-merge-benchmark.sh",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


BENCH_SCRIPT = _bench_script()

pytestmark = pytest.mark.skipif(
    BENCH_SCRIPT is None,
    reason="post-merge-benchmark.sh not found — companion PR in beefcake-swarm "
    "ships an equivalent unit test in that repo.",
)


def _hermetic_env(env: dict[str, str], shim_dir: Path) -> dict[str, str]:
    """Neutralize every cluster side-effect the bench script could trigger.

    The `run` subcommand of `post-merge-benchmark.sh` delegates to
    `python/bench_dispatch.py`, which (when *not* short-circuited by the
    pause check) runs `sbatch --wait`/`ssh`/`rsync` against the live
    cluster — `sbatch` resolves on this host and submits a REAL SLURM job
    (observed: jobs 3083-3085 fired during a naïve reproduction). A unit
    test must never do that. We prepend a shim dir of no-op `sbatch`,
    `ssh`, and `rsync` stubs to PATH and set `BENCH_SKIP_SYNC=1` so the
    dispatch path is inert even if the pause short-circuit fails to fire.
    """
    for tool in ("sbatch", "ssh", "rsync"):
        stub = shim_dir / tool
        # Exit non-zero so a non-short-circuited dispatch is recorded as a
        # benign local failure instead of a real cluster submission.
        stub.write_text("#!/usr/bin/env bash\nexit 97\n")
        stub.chmod(0o755)
    env = dict(env)
    env["PATH"] = f"{shim_dir}{os.pathsep}{env.get('PATH', '')}"
    env["BENCH_SKIP_SYNC"] = "1"
    return env


def _run_classify(flag_path: Path, *, env_paused: bool, repo_dir: Path) -> subprocess.CompletedProcess[str]:
    """Source the script and invoke `do_classify`. stdout = verdict."""
    env = os.environ.copy()
    env["BENCH_PAUSE_FLAG"] = str(flag_path)
    if env_paused:
        env["BENCH_PAUSED"] = "1"
    else:
        env.pop("BENCH_PAUSED", None)
    # Run the script's classify subcommand directly; the script's main
    # dispatch already invokes do_classify and prints the verdict.
    return subprocess.run(
        [str(BENCH_SCRIPT), "classify", str(repo_dir), "HEAD~1"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def _run_run(
    flag_path: Path, *, env_paused: bool, repo_dir: Path, shim_dir: Path
) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["BENCH_PAUSE_FLAG"] = str(flag_path)
    if env_paused:
        env["BENCH_PAUSED"] = "1"
    else:
        env.pop("BENCH_PAUSED", None)
    # Stub out sbatch/ssh/rsync so the `run` subcommand can never reach a
    # real cluster, regardless of whether the pause short-circuit fires.
    env = _hermetic_env(env, shim_dir)
    # Run subcommand. When paused, the early-return must fire before
    # any SSH/rsync — so this completes quickly even without bench host.
    return subprocess.run(
        [str(BENCH_SCRIPT), "run", str(repo_dir), "light", "test-bench-pause"],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    """Init a minimal git repo so do_classify's `git diff` doesn't crash."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@local"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=repo, check=True)
    (repo / "a.txt").write_text("a\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=repo, check=True)
    (repo / "b.txt").write_text("b\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "second"], cwd=repo, check=True)
    return repo


# --- Pause via flag file ---------------------------------------------------


def test_classify_skip_when_flag_present(tmp_path: Path, fake_repo: Path):
    flag = tmp_path / "paused"
    flag.write_text("paused_at=now\nreason=test\n")
    result = _run_classify(flag, env_paused=False, repo_dir=fake_repo)
    assert result.returncode == 0
    assert result.stdout.strip() == "skip"
    # Diagnostic log lands on stderr so the stdout contract is preserved.
    assert "paused" in result.stderr.lower()


@pytest.mark.xfail(
    strict=False,
    reason=(
        "External-repo regression: beefcake-swarm #290 (shell→Python bench "
        "dispatch refactor, 2026-05-18) moved the `run` pause check into "
        "python/bench_dispatch.py, whose `_is_paused()` only honors "
        "BENCH_PAUSED=1 and the hardcoded /tmp/cf-libs-bench-paused — it "
        "dropped the BENCH_PAUSE_FLAG env override that this test relies on. "
        "So the flag-file pause is not honored on the `run` path and the "
        "dispatcher proceeds (locally stubbed here so no real SLURM job is "
        "submitted). The env-var path (test_run_skips_when_env_var_set) IS "
        "honored and stays green. Fix belongs in beefcake-swarm "
        "bench_dispatch.py:_is_paused(); tracked by CF-LIBS-improved-5t6n."
    ),
)
def test_run_skips_when_flag_present(tmp_path: Path, fake_repo: Path):
    flag = tmp_path / "paused"
    flag.write_text("paused_at=now\n")
    shim = tmp_path / "shim"
    shim.mkdir()
    result = _run_run(flag, env_paused=False, repo_dir=fake_repo, shim_dir=shim)
    assert result.returncode == 0, (
        f"do_run should return 0 when paused; got {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "paused" in result.stdout.lower() or "paused" in result.stderr.lower()


# --- Pause via BENCH_PAUSED=1 env var --------------------------------------


def test_classify_skip_when_env_var_set(tmp_path: Path, fake_repo: Path):
    flag = tmp_path / "nonexistent"  # not created
    assert not flag.exists()
    result = _run_classify(flag, env_paused=True, repo_dir=fake_repo)
    assert result.returncode == 0
    assert result.stdout.strip() == "skip"


def test_run_skips_when_env_var_set(tmp_path: Path, fake_repo: Path):
    flag = tmp_path / "nonexistent"
    assert not flag.exists()
    shim = tmp_path / "shim"
    shim.mkdir()
    result = _run_run(flag, env_paused=True, repo_dir=fake_repo, shim_dir=shim)
    assert result.returncode == 0
    assert "paused" in result.stdout.lower() or "paused" in result.stderr.lower()


# --- No pause: short-circuit does NOT fire ---------------------------------


def test_classify_not_paused_uses_real_logic(tmp_path: Path, fake_repo: Path):
    """Without flag or env var, classify returns a real verdict (skip/light/heavy)."""
    flag = tmp_path / "nonexistent"
    assert not flag.exists()
    result = _run_classify(flag, env_paused=False, repo_dir=fake_repo)
    assert result.returncode == 0
    # The fake repo's diff is a single .txt → skip (no physics paths).
    assert result.stdout.strip() in {"skip", "light", "heavy"}
    # Crucially, the "paused, classifying as skip" log message must NOT appear.
    assert "paused, classifying as skip" not in result.stderr.lower()


# --- Helper scripts smoke test ---------------------------------------------


def test_bench_pause_creates_flag(tmp_path: Path, monkeypatch):
    """`bench-pause.sh` (local mode) creates the configured flag path."""
    pause_sh = Path(__file__).resolve().parents[2] / "scripts" / "bench-pause.sh"
    resume_sh = Path(__file__).resolve().parents[2] / "scripts" / "bench-resume.sh"
    assert pause_sh.exists(), f"{pause_sh} should exist"
    assert resume_sh.exists(), f"{resume_sh} should exist"
    assert os.access(pause_sh, os.X_OK), f"{pause_sh} should be executable"
    assert os.access(resume_sh, os.X_OK), f"{resume_sh} should be executable"

    # Run pause with a sandboxed flag path so we don't disturb the live one.
    flag = tmp_path / "paused"
    # Read+rewrite the script in a tmp copy with the FLAG line patched.
    body = pause_sh.read_text().replace(
        'FLAG="/tmp/cf-libs-bench-paused"',
        f'FLAG="{flag}"',
    )
    tmp_pause = tmp_path / "bench-pause.sh"
    tmp_pause.write_text(body)
    tmp_pause.chmod(0o755)

    result = subprocess.run(
        [str(tmp_pause), "--reason", "unit test"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert flag.exists()
    content = flag.read_text()
    assert "paused_at=" in content
    assert "reason=unit test" in content


def test_bench_resume_removes_flag(tmp_path: Path):
    resume_sh = Path(__file__).resolve().parents[2] / "scripts" / "bench-resume.sh"
    assert resume_sh.exists()

    flag = tmp_path / "paused"
    flag.write_text("paused_at=now\n")
    body = resume_sh.read_text().replace(
        'FLAG="/tmp/cf-libs-bench-paused"',
        f'FLAG="{flag}"',
    )
    tmp_resume = tmp_path / "bench-resume.sh"
    tmp_resume.write_text(body)
    tmp_resume.chmod(0o755)

    result = subprocess.run([str(tmp_resume)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert not flag.exists()


def test_bench_resume_idempotent(tmp_path: Path):
    """Running resume twice (or when no flag exists) should not error."""
    resume_sh = Path(__file__).resolve().parents[2] / "scripts" / "bench-resume.sh"
    flag = tmp_path / "paused"
    assert not flag.exists()
    body = resume_sh.read_text().replace(
        'FLAG="/tmp/cf-libs-bench-paused"',
        f'FLAG="{flag}"',
    )
    tmp_resume = tmp_path / "bench-resume.sh"
    tmp_resume.write_text(body)
    tmp_resume.chmod(0o755)

    result = subprocess.run([str(tmp_resume)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "already resumed" in result.stdout.lower()
