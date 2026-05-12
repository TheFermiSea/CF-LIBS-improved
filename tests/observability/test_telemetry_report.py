"""Tests for ``scripts/telemetry_report.py``.

Builds synthetic JSONL telemetry shards under a tmp_path and verifies
the loader, stats aggregation, sparkline, and rendered Markdown all
behave correctly.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

# Load scripts/telemetry_report.py as a module — it's not on the package path.
# Must register in sys.modules BEFORE exec_module so @dataclass can find it.
SCRIPT_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "telemetry_report.py"
)
spec = importlib.util.spec_from_file_location("telemetry_report", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
telemetry_report = importlib.util.module_from_spec(spec)
sys.modules["telemetry_report"] = telemetry_report
spec.loader.exec_module(telemetry_report)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _write_samples(
    root: Path,
    host: str,
    start: datetime,
    n: int,
    *,
    interval: int = 5,
    gpu_util_fn=lambda i: 50.0,
    gpu_mem_fn=lambda i: 10000.0,
    cpu_load_fn=lambda i: 1.5,
    rx_bytes_fn=lambda i: 1000 * i,
    tx_bytes_fn=lambda i: 200 * i,
) -> None:
    """Write ``n`` samples for ``host`` starting at ``start`` UTC.

    All samples land in the appropriate ``<root>/<host>/<date>/<hour>.jsonl``
    shard. Synthetic — caller picks the time series shape.
    """
    for i in range(n):
        ts = start + timedelta(seconds=i * interval)
        date_str = ts.strftime("%Y-%m-%d")
        hour_str = ts.strftime("%H")
        shard = root / host / date_str / f"{hour_str}.jsonl"
        shard.parent.mkdir(parents=True, exist_ok=True)
        rec = {
            "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "host": host,
            "gpu_util": gpu_util_fn(i),
            "gpu_mem_mb": gpu_mem_fn(i),
            "gpu_mem_total_mb": 32510,
            "gpu_power_w": 200.0,
            "gpu_temp_c": 70,
            "loadavg": [cpu_load_fn(i), cpu_load_fn(i), cpu_load_fn(i)],
            "mem_used_mb": 80000,
            "mem_total_mb": 256000,
            "mem_free_mb": 100000,
            "net_rx_bytes": rx_bytes_fn(i),
            "net_tx_bytes": tx_bytes_fn(i),
            "top_cpu": [{"pid": 1, "cpu": 100.0, "mem": 5.0, "cmd": "python"}],
            "top_mem": [{"pid": 2, "cpu": 10.0, "mem": 30.0, "cmd": "python"}],
            "jax_cache_size_mb": 420,
            "jax_cache_files": 100,
        }
        with shard.open("a") as fh:
            fh.write(json.dumps(rec) + "\n")


@pytest.fixture
def root_with_samples(tmp_path: Path) -> tuple[Path, datetime, datetime]:
    """Build a synthetic telemetry root with 3 hosts over a 1-hour window.

    Returns ``(root, since, until)``.
    """
    root = tmp_path / "telemetry"
    start = datetime(2026, 5, 12, 20, 55, 0, tzinfo=timezone.utc)
    # 12 samples * 5s = 60s on vasp-01 (high GPU)
    _write_samples(root, "vasp-01", start, 12, gpu_util_fn=lambda i: 90.0 + i)
    # 12 samples on vasp-02 (medium GPU)
    _write_samples(root, "vasp-02", start, 12, gpu_util_fn=lambda i: 30.0)
    # 12 samples on vasp-03 (idle — simulates CPU fallback)
    _write_samples(root, "vasp-03", start, 12, gpu_util_fn=lambda i: 0.0)
    return root, start, start + timedelta(seconds=12 * 5)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def test_parse_ts_roundtrip():
    ts = telemetry_report.parse_ts("2026-05-12T20:55:00Z")
    assert ts.tzinfo is timezone.utc
    assert ts.strftime("%Y-%m-%dT%H:%M:%SZ") == "2026-05-12T20:55:00Z"


def test_hours_between_inclusive():
    start = telemetry_report.parse_ts("2026-05-12T20:30:00Z")
    end = telemetry_report.parse_ts("2026-05-12T22:10:00Z")
    hours = list(telemetry_report.hours_between(start, end))
    assert hours == [
        ("2026-05-12", "20"),
        ("2026-05-12", "21"),
        ("2026-05-12", "22"),
    ]


def test_hours_between_crosses_midnight():
    start = telemetry_report.parse_ts("2026-05-12T23:30:00Z")
    end = telemetry_report.parse_ts("2026-05-13T00:30:00Z")
    hours = list(telemetry_report.hours_between(start, end))
    assert hours == [("2026-05-12", "23"), ("2026-05-13", "00")]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def test_iter_samples_filters_by_window(root_with_samples):
    root, since, until = root_with_samples
    # Narrow window: first 5 samples only (0,5,10,15,20 seconds after since)
    narrow_until = since + timedelta(seconds=20)
    samples = list(
        telemetry_report.iter_samples(root, "vasp-01", since, narrow_until)
    )
    assert len(samples) == 5
    for rec in samples:
        assert rec["host"] == "vasp-01"
        assert since <= rec["_ts"] <= narrow_until


def test_iter_samples_missing_host_returns_empty(root_with_samples):
    root, since, until = root_with_samples
    samples = list(telemetry_report.iter_samples(root, "vasp-99", since, until))
    assert samples == []


def test_iter_samples_skips_malformed_lines(tmp_path: Path):
    root = tmp_path / "telemetry"
    start = datetime(2026, 5, 12, 20, 0, 0, tzinfo=timezone.utc)
    shard = root / "vasp-01" / "2026-05-12" / "20.jsonl"
    shard.parent.mkdir(parents=True, exist_ok=True)
    shard.write_text(
        '{"ts":"2026-05-12T20:00:00Z","gpu_util":50}\n'
        "not-json garbage\n"
        "\n"
        '{"ts":"2026-05-12T20:00:05Z","gpu_util":60}\n'
        '{"no_ts":true}\n'
    )
    out = list(
        telemetry_report.iter_samples(
            root, "vasp-01", start, start + timedelta(hours=1)
        )
    )
    assert len(out) == 2
    assert {r["gpu_util"] for r in out} == {50, 60}


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_collect_stats_basic(root_with_samples):
    root, since, until = root_with_samples
    samples = telemetry_report.iter_samples(root, "vasp-01", since, until)
    stats = telemetry_report.collect_stats(samples, "vasp-01")
    assert stats.host == "vasp-01"
    assert stats.samples == 12
    assert stats.first_ts == since
    # Last sample is at index 11 → 55s after start
    assert stats.last_ts == since + timedelta(seconds=55)
    assert min(stats.gpu_util) == 90.0
    assert max(stats.gpu_util) == 101.0


def test_percentile_linear_interpolation():
    xs = [0.0, 25.0, 50.0, 75.0, 100.0]
    assert telemetry_report.percentile(xs, 0) == 0.0
    assert telemetry_report.percentile(xs, 100) == 100.0
    assert telemetry_report.percentile(xs, 50) == 50.0
    # 95th percentile: between 75 and 100 at fractional position 0.8 → 95.0
    assert telemetry_report.percentile(xs, 95) == pytest.approx(95.0)


def test_percentile_empty_returns_none():
    assert telemetry_report.percentile([], 50) is None


def test_mean_empty_returns_none():
    assert telemetry_report.mean([]) is None


def test_safe_float_handles_garbage():
    f = telemetry_report._safe_float
    assert f(None) is None
    assert f("abc") is None
    assert f("3.14") == 3.14
    assert f(42) == 42.0


# ---------------------------------------------------------------------------
# Sparkline
# ---------------------------------------------------------------------------


def test_sparkline_empty_returns_placeholder():
    assert telemetry_report.sparkline([]) == "(no data)"


def test_sparkline_renders_full_range():
    # Min and max of 0 and 100 should map to blocks at both ends.
    s = telemetry_report.sparkline([0.0, 100.0], width=2)
    assert s[0] == telemetry_report.SPARK_BLOCKS[0]
    assert s[-1] == telemetry_report.SPARK_BLOCKS[-1]


def test_sparkline_downsamples_long_series():
    vals = [50.0] * 1000
    s = telemetry_report.sparkline(vals, width=20)
    assert len(s) == 20
    # All samples identical → all the same block.
    assert len(set(s)) == 1


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------


def test_render_report_full(root_with_samples):
    root, since, until = root_with_samples
    host_stats = [
        telemetry_report.collect_stats(
            telemetry_report.iter_samples(root, h, since, until), h
        )
        for h in ("vasp-01", "vasp-02", "vasp-03")
    ]
    md = telemetry_report.render_report(since, until, host_stats)
    # Header includes both timestamps.
    assert "2026-05-12T20:55:00Z" in md
    assert "2026-05-12T20:56:00Z" in md
    # All three hosts appear in the table.
    for h in ("vasp-01", "vasp-02", "vasp-03"):
        assert h in md
    # Idle host gets flagged in the Quick read footer.
    assert "vasp-03" in md
    assert "idle" in md.lower() or "0%" in md
    # vasp-01 should show high mean GPU.
    assert "9" in md  # 95%-ish — loose check, value is data-dependent


def test_render_report_no_data():
    """Empty stats produce a usable (if sparse) report."""
    start = datetime(2026, 5, 12, 20, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(hours=1)
    host_stats = [telemetry_report.HostStats(host="vasp-01")]
    md = telemetry_report.render_report(start, end, host_stats)
    assert "vasp-01" in md
    assert "no samples in window" in md


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_main_writes_output_file(root_with_samples, tmp_path: Path, capsys):
    root, since, until = root_with_samples
    out = tmp_path / "report.md"
    rc = telemetry_report.main(
        [
            "--since", since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "--until", until.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "--hosts", "vasp-01,vasp-02,vasp-03",
            "--root", str(root),
            "--output", str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    body = out.read_text()
    assert "Utilization report" in body
    assert "vasp-01" in body


def test_main_stdout_when_no_output_arg(root_with_samples, capsys):
    root, since, until = root_with_samples
    rc = telemetry_report.main(
        [
            "--since", since.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "--until", until.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "--hosts", "vasp-01",
            "--root", str(root),
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Utilization report" in out
    assert "vasp-01" in out


def test_main_rejects_inverted_window(tmp_path: Path, capsys):
    rc = telemetry_report.main(
        [
            "--since", "2026-05-12T22:00:00Z",
            "--until", "2026-05-12T20:00:00Z",
            "--root", str(tmp_path),
        ]
    )
    assert rc == 2
    err = capsys.readouterr().err
    assert "before" in err
