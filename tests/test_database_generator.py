"""Tests for the atomic database generation wrapper."""

from __future__ import annotations

from types import SimpleNamespace

from cflibs.atomic.database_generator import generate_database


def test_generate_database_forwards_cli_arguments(monkeypatch):
    """The wrapper should forward output/filter arguments into datagen_v2.py."""

    captured = {}

    def fake_run(cmd, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("cflibs.atomic.database_generator.subprocess.run", fake_run)

    generate_database(
        db_path="custom.db",
        elements=["Fe", "Cu"],
        max_ionization_stage=3,
        max_upper_energy_ev=15.5,
        min_relative_intensity=25.0,
    )

    cmd = captured["cmd"]
    assert cmd[0]
    assert cmd[1].endswith("datagen_v2.py")
    assert cmd[2:] == [
        "--db-path",
        "custom.db",
        "--max-ionization-stage",
        "3",
        "--max-upper-energy-ev",
        "15.5",
        "--min-relative-intensity",
        "25.0",
        "--elements",
        "Fe",
        "Cu",
    ]
