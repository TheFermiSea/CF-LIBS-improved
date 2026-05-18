#!/usr/bin/env python3
"""Autonomous Exp 1 → 2 → 3 → 4 chain driver.

Polls Exp 1 shard processes until they stop, aggregates results, picks
top-N identifiers by F1 mean, builds Exp 2 cells JSON, launches Exp 2
shards. In parallel launches Exp 3 (basis-FWHM rebuild). When Exp 2
finishes, launches Exp 4 (winning config, equal allocation).

Usage:
    python scripts/exp_chain.py --start exp001
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

EXP1_RESULTS = "/cluster/shared/cf-libs-bench/results/exp001"
EXP2_RESULTS = "/cluster/shared/cf-libs-bench/results/exp002"
EXP4_RESULTS = "/cluster/shared/cf-libs-bench/results/exp004"
SHARDS = ("shard1", "shard2", "shard3")
HOSTS = {"shard1": "10.0.0.20", "shard2": "10.0.0.21", "shard3": "10.0.0.22"}
COMP_WORKFLOWS = ("iterative_jax", "bayesian")


def _ssh(host: str, cmd: str) -> tuple[int, str]:
    r = subprocess.run(
        ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
         f"root@{host}", cmd],
        capture_output=True, text=True, timeout=60,
    )
    return r.returncode, r.stdout


def _all_shards_stopped() -> bool:
    for shard in SHARDS:
        host = HOSTS[shard]
        rc, out = _ssh(host, "pgrep -f 'python.*parameter_sweep' | head -1")
        if out.strip():
            return False
    return True


def _wait_for_stop(label: str, poll_sec: int = 60) -> None:
    print(f"[chain] waiting for {label} to stop (poll every {poll_sec}s)…", flush=True)
    while not _all_shards_stopped():
        time.sleep(poll_sec)
    print(f"[chain] {label} all shards stopped at "
          f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}", flush=True)


def _aggregate_exp1() -> dict:
    """Pull per-iter id_summary.json from all shards, return top identifiers."""
    print("[chain] aggregating Exp 1 results…", flush=True)
    workflow_scores: dict[str, list[float]] = {}
    for shard in SHARDS:
        host = HOSTS[shard]
        rc, listing = _ssh(host,
            f"ls -d {EXP1_RESULTS}/{shard}/iter-* 2>/dev/null")
        if not listing:
            continue
        for iter_path in listing.strip().splitlines():
            rc, content = _ssh(host, f"cat {iter_path}/id_summary.json 2>/dev/null")
            if not content:
                continue
            try:
                summary = json.loads(content)
            except Exception:
                continue
            overall = summary.get("overall", {})
            for workflow, metrics in overall.items():
                f1 = metrics.get("micro_f1")
                if isinstance(f1, (int, float)):
                    workflow_scores.setdefault(workflow, []).append(float(f1))
    ranked = sorted(
        workflow_scores.items(),
        key=lambda kv: -(sum(kv[1]) / len(kv[1]) if kv[1] else 0.0),
    )
    print("[chain] Exp 1 workflow rankings by mean F1:", flush=True)
    for wf, scores in ranked:
        mean = sum(scores) / len(scores) if scores else 0.0
        print(f"  {wf:24s} F1={mean:.3f}  n={len(scores)}", flush=True)
    return {wf: (sum(s) / len(s) if s else 0.0, len(s)) for wf, s in ranked}


def _build_exp2_cells(top_n: int = 3) -> list[dict]:
    """After Exp 1 aggregation, build Exp 2 cells: top-N × {iterative_jax,bayesian}."""
    rankings = _aggregate_exp1()
    top = list(rankings.keys())[:top_n]
    print(f"[chain] Exp 2 will use top-{top_n} identifiers: {top}", flush=True)
    cells = []
    for wf_id in top:
        # Map back to identifier args. The Exp 1 workflow_name in id_summary.json
        # is the identifier name (e.g., 'alias', 'comb', 'correlation', 'spectral_nnls', 'hybrid_union').
        # Tag as either jax or basis-driven.
        is_basis = wf_id in ("spectral_nnls", "hybrid_union", "hybrid_intersect",
                             "nnls_concentration_threshold")
        basis_arg = " --basis-dir /cluster/shared/cf-libs-bench/basis_libraries" if is_basis else ""
        jax_arg = " --jax-identifier" if wf_id != "spectral_nnls" else ""
        for comp_wf in COMP_WORKFLOWS:
            # Bayesian is slow; use half as many shots for bayesian cells.
            shots = 2 if comp_wf == "iterative_jax" else 1
            cell = {
                "name": f"{wf_id}__{comp_wf}",
                "config_args": (
                    f"--quick --max-outer-folds 1 --sections all "
                    f"--id-workflows {wf_id} --composition-workflows {comp_wf}"
                    f"{jax_arg}{basis_arg} --vrabel-max-shots {shots}"
                ),
            }
            cells.append(cell)
    return cells


def _launch_shard(shard: str, config_path: str, results_dir: str, n_iters: int = 15) -> None:
    """Launch one parameter_sweep shard on its assigned vasp node."""
    host = HOSTS[shard]
    _shard_num = shard[-1]  # noqa: F841 — kept for debug inspection
    # Patch the config file's dataset-shard tail for this shard.
    cmd = (
        f"cd /scratch/cf-libs-exp001 && "
        f"NV_LIBS=$(find .venv/lib/python3.11/site-packages/nvidia -maxdepth 3 "
        f"-name lib -type d 2>/dev/null | tr '\\n' ':' | sed 's/:$//') && "
        f"export LD_LIBRARY_PATH=\"${{NV_LIBS}}:/usr/local/cuda/lib64:/usr/local/cuda/lib\" && "
        f"export JAX_PLATFORMS=cuda JAX_ENABLE_X64=0 PYTHONWARNINGS='ignore::UserWarning' "
        f"CFLIBS_USE_JAX_BOLTZMANN_COMPOSITION=1 "
        f"CFLIBS_DATA_DIR=/cluster/shared/cf-libs-bench/data "
        f"CFLIBS_BASIS_DIR=/cluster/shared/cf-libs-bench/basis_libraries "
        f"JAX_COMPILATION_CACHE_DIR=/cluster/shared/jax-cache && "
        f"setsid nohup bash -c \".venv/bin/python -u scripts/parameter_sweep.py "
        f"--cells {config_path} --bandit 2 --n-iters {n_iters} "
        f"--output-dir {results_dir}/{shard} 2>&1 "
        f"| tee /tmp/{Path(results_dir).name}-{shard}.log\" "
        f"</dev/null >/tmp/{Path(results_dir).name}-{shard}-wrap.log 2>&1 & disown"
    )
    rc, _ = _ssh(host, cmd)
    if rc != 0:
        print(f"[chain] WARNING: shard {shard} launch returned rc={rc}", flush=True)


def _launch_exp2() -> None:
    """Build Exp 2 cells JSON per shard and launch."""
    cells = _build_exp2_cells(top_n=3)
    print(f"[chain] Exp 2 cells (per shard): {len(cells)}", flush=True)
    # Reset NFS output dirs
    _ssh("10.0.0.5", f"rm -rf {EXP2_RESULTS}; mkdir -p {EXP2_RESULTS}/shard1 "
         f"{EXP2_RESULTS}/shard2 {EXP2_RESULTS}/shard3; chmod -R 1777 {EXP2_RESULTS}")
    repo = Path("/home/brian/code/CF-LIBS-improved")
    exp2_cfg = repo / "configs" / "exp002"
    exp2_cfg.mkdir(parents=True, exist_ok=True)
    for shard_idx, shard in enumerate(SHARDS, 1):
        # Per-shard cells: append --dataset-shard <N>/3 to each cell's config_args
        shard_cells = []
        for c in cells:
            shard_cells.append({
                "name": c["name"],
                "config_args": c["config_args"] + f" --dataset-shard {shard_idx}/3",
            })
        cfg_path = exp2_cfg / f"shard{shard_idx}.json"
        cfg_path.write_text(json.dumps(shard_cells, indent=2))
        # rsync to all nodes
        for h in HOSTS.values():
            subprocess.run(
                ["rsync", "-aL",
                 str(repo) + "/",
                 f"root@{h}:/scratch/cf-libs-exp001/",
                 "--exclude=.git", "--exclude=output", "--exclude=.venv",
                 "--exclude=data", "--exclude=cache_basis"],
                check=False, capture_output=True,
            )
        _launch_shard(shard, f"configs/exp002/shard{shard_idx}.json",
                      EXP2_RESULTS, n_iters=15)
        print(f"[chain] Exp 2 shard {shard} launched", flush=True)


def _launch_exp3() -> None:
    """Trigger Exp 3 basis-FWHM rebuild on vasp-02 in parallel with Exp 2."""
    print("[chain] launching Exp 3 (basis FWHM rebuild on vasp-02)…", flush=True)
    rc, _ = _ssh("10.0.0.21",
        "cd /scratch/cf-libs-build-temp && "
        "setsid nohup bash -c \"export PYTHONPATH=. JAX_PLATFORMS=cpu; "
        ".venv/bin/python scripts/build_basis_library.py --fwhm 0.03 0.04 0.06 0.08 2>&1 "
        "| tee /tmp/exp003-basis-rebuild.log\" </dev/null >/tmp/exp003-wrap.log 2>&1 "
        "& disown")
    print(f"[chain] Exp 3 launch rc={rc}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-exp", choices=("exp1", "exp2"), default="exp1",
                    help="Which experiment to wait-for-then-advance from.")
    ap.add_argument("--poll-sec", type=int, default=120,
                    help="Polling interval while waiting (default 120 s).")
    args = ap.parse_args()

    if args.from_exp == "exp1":
        _wait_for_stop("Exp 1", args.poll_sec)
        _launch_exp2()
        _launch_exp3()
        print("[chain] Exp 2 + Exp 3 launched. Will not auto-advance to Exp 4 "
              "(needs human judgment on winner). Run again with --from-exp exp2 "
              "to wait + decide.", flush=True)
    elif args.from_exp == "exp2":
        _wait_for_stop("Exp 2", args.poll_sec)
        print("[chain] Exp 2 done. Manual review required to pick winner for Exp 4.", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
