#!/usr/bin/env python3
"""telemetry_report.py — Query the cluster telemetry stream for a time range.

Reads JSONL telemetry shards written by ``scripts/telemetry_sampler.sh`` and
emits a Markdown utilization report with per-host summary stats and a tiny
unicode-block sparkline of GPU utilization.

Usage::

    python scripts/telemetry_report.py \
        --since 2026-05-12T20:55:00Z \
        --until 2026-05-12T23:30:00Z \
        --hosts vasp-01,vasp-02,vasp-03 \
        --output /cluster/shared/cf-libs-bench/results/exp001/utilization.md

If ``--output`` is omitted, the report is printed to stdout. ``--root`` lets you
point the tool at a different telemetry root for testing.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Iterator


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

ISO_FMT = "%Y-%m-%dT%H:%M:%SZ"


def parse_ts(s: str) -> datetime:
    """Parse an RFC3339/ISO-8601 UTC timestamp.

    Accepts both "Z" and "+00:00" suffixes for robustness.
    """
    if s.endswith("Z"):
        return datetime.strptime(s, ISO_FMT).replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(s).astimezone(timezone.utc)


def hours_between(start: datetime, end: datetime) -> Iterator[tuple[str, str]]:
    """Yield (YYYY-MM-DD, HH) UTC tuples covering [start, end] inclusive.

    Each tuple corresponds to one telemetry shard file
    (``<root>/<host>/<date>/<hour>.jsonl``). Returns the start hour, every
    hour between, and the end hour — boundary inclusive so partial-hour
    queries still pick up the right files.
    """
    cur = start.replace(minute=0, second=0, microsecond=0)
    last = end.replace(minute=0, second=0, microsecond=0)
    while cur <= last:
        yield cur.strftime("%Y-%m-%d"), cur.strftime("%H")
        cur += timedelta(hours=1)


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def iter_samples(
    root: Path,
    host: str,
    since: datetime,
    until: datetime,
) -> Iterator[dict]:
    """Yield JSON sample dicts for ``host`` within ``[since, until]``.

    Silently skips missing shard files and malformed lines — telemetry should
    never crash the report tool.
    """
    host_root = root / host
    for date_str, hour_str in hours_between(since, until):
        shard = host_root / date_str / f"{hour_str}.jsonl"
        if not shard.exists():
            continue
        try:
            with shard.open("r") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    ts_str = rec.get("ts")
                    if not ts_str:
                        continue
                    try:
                        ts = parse_ts(ts_str)
                    except ValueError:
                        continue
                    if since <= ts <= until:
                        rec["_ts"] = ts
                        yield rec
        except OSError:
            continue


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class HostStats:
    """Aggregate stats over one host's samples for the query window."""

    host: str
    samples: int = 0
    first_ts: datetime | None = None
    last_ts: datetime | None = None
    gpu_util: list[float] = field(default_factory=list)
    gpu_mem_mb: list[float] = field(default_factory=list)
    cpu_load1: list[float] = field(default_factory=list)
    rx_bytes_first: int | None = None
    rx_bytes_last: int | None = None
    tx_bytes_first: int | None = None
    tx_bytes_last: int | None = None


def _safe_float(v) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def collect_stats(samples: Iterable[dict], host: str) -> HostStats:
    """Fold an iterator of samples into a :class:`HostStats`."""
    s = HostStats(host=host)
    for rec in samples:
        s.samples += 1
        ts = rec.get("_ts")
        if ts is not None:
            if s.first_ts is None or ts < s.first_ts:
                s.first_ts = ts
            if s.last_ts is None or ts > s.last_ts:
                s.last_ts = ts
        gu = _safe_float(rec.get("gpu_util"))
        if gu is not None:
            s.gpu_util.append(gu)
        gm = _safe_float(rec.get("gpu_mem_mb"))
        if gm is not None:
            s.gpu_mem_mb.append(gm)
        la = rec.get("loadavg") or []
        if la and _safe_float(la[0]) is not None:
            s.cpu_load1.append(float(la[0]))
        rx = _safe_float(rec.get("net_rx_bytes"))
        tx = _safe_float(rec.get("net_tx_bytes"))
        if rx is not None:
            if s.rx_bytes_first is None:
                s.rx_bytes_first = int(rx)
            s.rx_bytes_last = int(rx)
        if tx is not None:
            if s.tx_bytes_first is None:
                s.tx_bytes_first = int(tx)
            s.tx_bytes_last = int(tx)
    return s


def percentile(xs: list[float], q: float) -> float | None:
    """Linear-interpolation percentile (q in [0,100]). None for empty input."""
    if not xs:
        return None
    if len(xs) == 1:
        return xs[0]
    s = sorted(xs)
    k = (len(s) - 1) * (q / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    frac = k - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def mean(xs: list[float]) -> float | None:
    return (sum(xs) / len(xs)) if xs else None


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------

# 8 unicode block-elements, low → high.
SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 60) -> str:
    """Return a unicode-block sparkline of ``values`` resampled to ``width``.

    Values are normalized to [0, 100] (GPU utilization range). Empty → "(no data)".
    """
    if not values:
        return "(no data)"
    if len(values) <= width:
        resampled = values
    else:
        # Bucket-average down to `width` points.
        bucket = len(values) / width
        resampled = []
        for i in range(width):
            lo = int(i * bucket)
            hi = int((i + 1) * bucket)
            chunk = values[lo:hi] if hi > lo else [values[lo]]
            resampled.append(sum(chunk) / len(chunk))
    out = []
    for v in resampled:
        v = max(0.0, min(100.0, v))
        idx = min(len(SPARK_BLOCKS) - 1, int(v / 100.0 * len(SPARK_BLOCKS)))
        out.append(SPARK_BLOCKS[idx])
    return "".join(out)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    return f"{x:.0f}%" if x is not None else "—"


def _fmt_num(x: float | None, fmt: str = "{:.0f}") -> str:
    return fmt.format(x) if x is not None else "—"


def _fmt_duration(td: timedelta | None) -> str:
    if td is None:
        return "—"
    secs = td.total_seconds()
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs/60:.1f} min"
    return f"{secs/3600:.2f} h"


def render_report(
    since: datetime,
    until: datetime,
    host_stats: list[HostStats],
    *,
    sparkline_width: int = 60,
) -> str:
    """Render a Markdown utilization report."""
    lines: list[str] = []
    lines.append(
        f"## Utilization report: {since.strftime(ISO_FMT)} -> {until.strftime(ISO_FMT)}"
    )
    lines.append("")

    lines.append(
        "| host    | samples | duration | gpu_util_mean | gpu_util_p95 |"
        " gpu_mem_max_mb | cpu_load_max | rx_mb | tx_mb |"
    )
    lines.append(
        "|---------|---------|----------|---------------|--------------|"
        "----------------|--------------|-------|-------|"
    )

    for s in host_stats:
        dur = (
            s.last_ts - s.first_ts
            if s.first_ts is not None and s.last_ts is not None
            else None
        )
        rx_mb = (
            (s.rx_bytes_last - s.rx_bytes_first) / 1024.0 / 1024.0
            if s.rx_bytes_first is not None and s.rx_bytes_last is not None
            else None
        )
        tx_mb = (
            (s.tx_bytes_last - s.tx_bytes_first) / 1024.0 / 1024.0
            if s.tx_bytes_first is not None and s.tx_bytes_last is not None
            else None
        )
        lines.append(
            "| {host:<7s} | {n:>7d} | {dur:<8s} | {mean:>13s} | {p95:>12s} |"
            " {mem_max:>14s} | {load_max:>12s} | {rx:>5s} | {tx:>5s} |".format(
                host=s.host,
                n=s.samples,
                dur=_fmt_duration(dur),
                mean=_fmt_pct(mean(s.gpu_util)),
                p95=_fmt_pct(percentile(s.gpu_util, 95)),
                mem_max=_fmt_num(max(s.gpu_mem_mb) if s.gpu_mem_mb else None),
                load_max=_fmt_num(
                    max(s.cpu_load1) if s.cpu_load1 else None, fmt="{:.1f}"
                ),
                rx=_fmt_num(rx_mb),
                tx=_fmt_num(tx_mb),
            )
        )

    lines.append("")
    lines.append("### GPU utilization sparkline (left = oldest, right = newest)")
    lines.append("")
    lines.append("```")
    for s in host_stats:
        lines.append(f"{s.host}: {sparkline(s.gpu_util, sparkline_width)}")
    lines.append("```")
    lines.append("")

    # Quick interpretation footer.
    lines.append("### Quick read")
    lines.append("")
    any_gpu = False
    for s in host_stats:
        m = mean(s.gpu_util)
        if m is None:
            lines.append(f"- **{s.host}**: no samples in window")
            continue
        any_gpu = True
        p95 = percentile(s.gpu_util, 95)
        if m < 5 and (p95 or 0) < 20:
            lines.append(
                f"- **{s.host}**: GPU effectively idle ({m:.0f}% mean) — "
                "kernels probably falling back to CPU"
            )
        elif m < 30:
            lines.append(
                f"- **{s.host}**: GPU lightly used ({m:.0f}% mean, p95 {p95:.0f}%)"
            )
        else:
            lines.append(
                f"- **{s.host}**: GPU active ({m:.0f}% mean, p95 {p95:.0f}%)"
            )
    if not any_gpu:
        lines.append("- (no GPU samples found — was the sampler running?)")

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--since", required=True, help="Start UTC ISO timestamp (Z)")
    p.add_argument("--until", required=True, help="End UTC ISO timestamp (Z)")
    p.add_argument(
        "--hosts",
        default="vasp-01,vasp-02,vasp-03",
        help="Comma-separated host list",
    )
    p.add_argument(
        "--root",
        default="/cluster/shared/cf-libs-bench/telemetry",
        help="Telemetry root dir",
    )
    p.add_argument(
        "--output", "-o", default=None,
        help="Output Markdown file (default: stdout)",
    )
    p.add_argument(
        "--sparkline-width", type=int, default=60,
        help="Sparkline width in characters (default: 60)",
    )
    args = p.parse_args(argv)

    try:
        since = parse_ts(args.since)
        until = parse_ts(args.until)
    except ValueError as e:
        print(f"ERROR: bad timestamp: {e}", file=sys.stderr)
        return 2
    if until < since:
        print("ERROR: --until is before --since", file=sys.stderr)
        return 2

    root = Path(args.root)
    if not root.exists():
        print(f"ERROR: telemetry root {root} does not exist", file=sys.stderr)
        return 2

    hosts = [h.strip() for h in args.hosts.split(",") if h.strip()]
    host_stats = [
        collect_stats(iter_samples(root, h, since, until), h) for h in hosts
    ]

    md = render_report(since, until, host_stats, sparkline_width=args.sparkline_width)
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(md)
        print(f"wrote {out_path}", file=sys.stderr)
    else:
        sys.stdout.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
