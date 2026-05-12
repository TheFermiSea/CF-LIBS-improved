#!/usr/bin/env python3
"""Exp 1 live dashboard — reads partial outputs across all 3 shards and prints
an aggregate view of cell × dataset metrics, bandit state, and runtime status.

Safe to run while shards are writing (reads only, no locks).

Usage:
    python scripts/exp001_dashboard.py
    python scripts/exp001_dashboard.py --output /cluster/shared/.../live_dashboard.md
    python scripts/exp001_dashboard.py --watch 60  # refresh every 60 s
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

RESULTS_ROOT = Path("/cluster/shared/cf-libs-bench/results/exp001")
SHARDS = ("shard1", "shard2", "shard3")
HOSTS = {"shard1": "10.0.0.20", "shard2": "10.0.0.21", "shard3": "10.0.0.22"}


def _ssh_read(host: str, remote_path: str) -> str | None:
    """Read a file from a remote host via SSH. Returns None if missing or error."""
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"root@{host}", f"cat {remote_path} 2>/dev/null || true"],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout if r.returncode == 0 and r.stdout else None
    except Exception:
        return None


def _ssh_lines(host: str, remote_cmd: str) -> list[str]:
    try:
        r = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes",
             f"root@{host}", remote_cmd],
            capture_output=True, text=True, timeout=15,
        )
        return r.stdout.splitlines() if r.returncode == 0 else []
    except Exception:
        return []


def _read_iter_summary(host: str, shard: str, iter_dir: str) -> dict | None:
    """Read id_summary.json + composition_summary.json from one iter on the remote NFS."""
    remote_base = f"{RESULTS_ROOT}/{shard}/{iter_dir}"
    id_summary = _ssh_read(host, f"{remote_base}/id_summary.json")
    comp_summary = _ssh_read(host, f"{remote_base}/composition_summary.json")
    out = {"iter": iter_dir}
    if id_summary:
        try:
            out["id"] = json.loads(id_summary)
        except Exception:
            pass
    if comp_summary:
        try:
            out["composition"] = json.loads(comp_summary)
        except Exception:
            pass
    return out if ("id" in out or "composition" in out) else None


def _read_manifest(host: str, shard: str) -> list[dict]:
    """Read manifest.jsonl from a shard (small file)."""
    raw = _ssh_read(host, f"{RESULTS_ROOT}/{shard}/manifest.jsonl")
    if not raw:
        return []
    out = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _list_iter_dirs(host: str, shard: str) -> list[str]:
    lines = _ssh_lines(
        host, f"ls -d {RESULTS_ROOT}/{shard}/iter-* 2>/dev/null | xargs -n1 basename"
    )
    return sorted(lines)


def _proc_state(host: str) -> dict:
    """Quick proc state for the shard host. Picks the highest-%CPU python (the
    actual worker, not the wrapper bash/timeout)."""
    out = {"running": False, "elapsed": "?", "cpu_pct": "?"}
    lines = _ssh_lines(
        host,
        "for pid in $(pgrep -f 'python.*parameter_sweep'); do "
        "ps -p $pid -o pid,etime,%cpu --no-headers 2>/dev/null; done "
        "| sort -k3 -nr | head -1",
    )
    if lines:
        parts = lines[0].split()
        if len(parts) >= 3:
            out["running"] = True
            out["elapsed"] = parts[1]
            out["cpu_pct"] = parts[2]
    return out


def _aggregate_iter(records: list[dict]) -> dict:
    """Aggregate per-iter summary records into (cell, dataset) -> metric dict.

    Each iter_summary['id'] looks like {"overall": {workflow: {metric: value, ...}},
                                         "per_dataset": {workflow: {dataset: {metric: ...}}}}.
    """
    by_cd: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in records:
        id_sum = r.get("id", {})
        per_ds = id_sum.get("per_dataset", {})
        for workflow, ds_map in per_ds.items():
            for ds, metrics in ds_map.items():
                for mk in ("micro_f1", "micro_precision", "micro_recall",
                           "false_positives_per_spectrum", "latency_mean_s"):
                    v = metrics.get(mk)
                    if isinstance(v, (int, float)) and not math.isnan(v):
                        by_cd[(workflow, ds)][mk].append(float(v))
                n_spec = metrics.get("n_spectra")
                if isinstance(n_spec, int):
                    by_cd[(workflow, ds)]["n_spectra"].append(float(n_spec))

        comp_sum = r.get("composition", {})
        comp_per_ds = comp_sum.get("per_dataset", {}) if isinstance(comp_sum, dict) else {}
        for workflow, ds_map in comp_per_ds.items():
            for ds, metrics in ds_map.items():
                v = metrics.get("mean_aitchison")
                if isinstance(v, (int, float)) and not math.isnan(v):
                    by_cd[(workflow, ds)]["d_a"].append(float(v))

    out = {}
    for (workflow, ds), metrics in by_cd.items():
        agg = {}
        for mk, vals in metrics.items():
            if not vals:
                continue
            agg[f"{mk}_mean"] = sum(vals) / len(vals)
            if len(vals) > 1:
                m = agg[f"{mk}_mean"]
                var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
                agg[f"{mk}_std"] = math.sqrt(var)
            agg[f"{mk}_n"] = len(vals)
        out[(workflow, ds)] = agg
    return out


def _bandit_state(manifests: dict[str, list[dict]]) -> dict[str, dict]:
    """From manifests across shards, extract latest bandit state per shard."""
    out = {}
    for shard, rows in manifests.items():
        if not rows:
            out[shard] = {"latest": None, "arms_pulled": {}}
            continue
        arms_pulled: dict[str, int] = defaultdict(int)
        latest_posteriors = None
        latest_iter = -1
        for r in rows:
            cell = r.get("cell_name") or r.get("cell_id")
            if cell:
                arms_pulled[cell] += 1
            if "arm_posteriors" in r and r.get("iter_index", -1) >= latest_iter:
                latest_iter = r["iter_index"]
                latest_posteriors = r["arm_posteriors"]
        out[shard] = {
            "latest_iter": latest_iter,
            "arms_pulled": dict(arms_pulled),
            "latest_posteriors": latest_posteriors,
        }
    return out


def _markdown_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    """Render a list of dicts as a Markdown table. cols=[(key, header)]."""
    out = ["| " + " | ".join(h for _, h in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "—")
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 100 else f"{v:.1f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def render(now_iso: str) -> str:
    parts = [f"# Exp 1 live dashboard — {now_iso}", ""]

    # 1) Per-shard process status
    parts.append("## Shard processes")
    proc_rows = []
    for shard in SHARDS:
        host = HOSTS[shard]
        ps = _proc_state(host)
        iters = _list_iter_dirs(host, shard)
        proc_rows.append({
            "shard": f"{shard} ({host})",
            "running": "✅ alive" if ps["running"] else "❌ stopped",
            "elapsed": ps["elapsed"],
            "cpu_pct": ps["cpu_pct"],
            "iters_started": len(iters),
            "latest_iter": iters[-1] if iters else "—",
        })
    parts.append(_markdown_table(proc_rows, [
        ("shard", "shard"), ("running", "running"),
        ("elapsed", "elapsed"), ("cpu_pct", "%CPU"),
        ("iters_started", "iters"),
        ("latest_iter", "latest"),
    ]))
    parts.append("")

    # 2) Pull per-iter summaries from each shard, aggregate
    all_records = []
    manifests = {}
    for shard in SHARDS:
        host = HOSTS[shard]
        manifests[shard] = _read_manifest(host, shard)
        for d in _list_iter_dirs(host, shard):
            rec = _read_iter_summary(host, shard, d)
            if rec:
                rec["shard"] = shard
                all_records.append(rec)

    parts.append(f"## Per-iter summaries collected: {len(all_records)} "
                 f"(from {sum(len(_list_iter_dirs(HOSTS[s], s)) for s in SHARDS)} iter dirs)")
    parts.append("")

    # 3) Aggregate by (workflow, dataset)
    agg = _aggregate_iter(all_records)
    parts.append("## Identification metrics (workflow × dataset)")
    rows = []
    for (workflow, ds), metrics in sorted(agg.items(), key=lambda kv: -(kv[1].get("micro_f1_mean") or 0)):
        rows.append({
            "workflow": workflow,
            "dataset": ds,
            "n_iters": int(metrics.get("micro_f1_n") or 0),
            "f1": metrics.get("micro_f1_mean"),
            "precision": metrics.get("micro_precision_mean"),
            "recall": metrics.get("micro_recall_mean"),
            "fp_per_spec": metrics.get("false_positives_per_spectrum_mean"),
            "latency_s": metrics.get("latency_mean_s_mean"),
            "n_spectra": int(metrics.get("n_spectra_mean") or 0),
            "d_a": metrics.get("d_a_mean"),
        })
    parts.append(_markdown_table(rows, [
        ("workflow", "workflow"), ("dataset", "dataset"),
        ("n_iters", "n_iters"),
        ("f1", "F1"), ("precision", "P"), ("recall", "R"),
        ("fp_per_spec", "FP/spec"), ("latency_s", "lat_s"),
        ("n_spectra", "n_spec"), ("d_a", "d_A"),
    ]))
    parts.append("")

    # 4) Bandit state
    parts.append("## Bandit state per shard")
    bandit = _bandit_state(manifests)
    for shard in SHARDS:
        st = bandit[shard]
        parts.append(f"### {shard}")
        if not st.get("arms_pulled"):
            parts.append("_no completed iters yet_")
            parts.append("")
            continue
        ap = st["arms_pulled"]
        parts.append("Arms pulled so far:")
        for cell, n in sorted(ap.items(), key=lambda kv: -kv[1]):
            parts.append(f"- `{cell}`: {n}")
        if st.get("latest_posteriors"):
            parts.append("")
            parts.append("Latest posteriors (last iter):")
            rows = []
            for ap in st["latest_posteriors"]:
                rows.append({
                    "arm": ap.get("arm_id", "?"),
                    "name": ap.get("cell_name", "?"),
                    "mean": ap.get("mean"),
                    "var": ap.get("variance"),
                    "n_pulls": ap.get("n_pulls"),
                    "prob_best": ap.get("prob_best"),
                })
            parts.append(_markdown_table(rows, [
                ("arm", "arm"), ("name", "name"),
                ("mean", "post_mean"), ("var", "post_var"),
                ("n_pulls", "pulls"), ("prob_best", "P(best)"),
            ]))
        parts.append("")

    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=str, default=None,
                    help="Write Markdown to this path (otherwise stdout).")
    ap.add_argument("--watch", type=int, default=0,
                    help="If >0, refresh every N seconds (Ctrl-C to stop).")
    args = ap.parse_args()

    while True:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        md = render(now)
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(md)
            print(f"[{now}] dashboard written to {args.output} "
                  f"({len(md)} bytes)", flush=True)
        else:
            print(md, flush=True)
        if args.watch <= 0:
            return 0
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
