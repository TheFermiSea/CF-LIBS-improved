"""
Tests for CLI module.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from cflibs.cli.main import (
    _float_config_value,
    _resolve_db_path,
    doctor_cmd,
    forward_model_cmd,
    invert_cmd,
    dbgen_cmd,
    main,
)


def test_forward_model_cmd_missing_config():
    """Test forward command with missing config file."""

    class Args:
        config = "nonexistent.yaml"
        output = None

    args = Args()

    with pytest.raises(FileNotFoundError):
        forward_model_cmd(args)


def test_forward_model_cmd_invalid_config():
    """Test forward command with invalid config."""
    config_fd, config_path = tempfile.mkstemp(suffix=".yaml")
    os.close(config_fd)  # Close file descriptor to prevent leaks

    try:
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        class Args:
            config = config_path
            output = None

        args = Args()

        # Should raise an error during config loading
        with pytest.raises(Exception):
            forward_model_cmd(args)
    finally:
        Path(config_path).unlink()


def test_invert_cmd_requires_elements():
    """Invert command should fail with a clear error when no elements are provided."""

    class Args:
        spectrum = "spectrum.csv"
        config = None
        output = None

    args = Args()

    with pytest.raises(ValueError, match="Elements must be specified"):
        invert_cmd(args)


def test_dbgen_cmd_missing_script():
    """Test database generation command when generator setup fails."""

    class Args:
        db_path = "test.db"
        elements = None

    args = Args()

    with patch("cflibs.atomic.database_generator.generate_database") as mock_generate:
        mock_generate.side_effect = FileNotFoundError("missing datagen_v2.py")

        # Should exit with error
        with pytest.raises(SystemExit):
            dbgen_cmd(args)


def test_dbgen_cmd_forwards_args():
    """Database generation CLI should forward db path and elements to the generator."""

    class Args:
        db_path = "custom.db"
        elements = ["Fe", "Cu"]

    args = Args()

    with patch("cflibs.atomic.database_generator.generate_database") as mock_generate:
        dbgen_cmd(args)

    mock_generate.assert_called_once_with(db_path="custom.db", elements=["Fe", "Cu"])


def test_resolve_db_path_finds_bundled_sample_database():
    """Default database resolution should find the checked-in sample database."""
    db_path = _resolve_db_path()

    assert db_path.name == "libs_production.db"
    assert db_path.exists()
    assert db_path.parent.name == "ASD_da"


def test_float_config_value_accepts_scientific_notation_string():
    """YAML may parse values like 1.0e17 as strings; CLI configs should still work."""
    assert _float_config_value({"ne": "1.0e17"}, "ne", "plasma") == 1.0e17


def test_doctor_cmd_reports_next_steps(capsys):
    """Doctor command should provide a setup summary and first commands."""

    class Args:
        db_path = None

    doctor_cmd(Args())

    captured = capsys.readouterr()
    assert "CF-LIBS setup doctor" in captured.out
    assert "atomic database" in captured.out
    assert "cflibs forward examples/config_example.yaml" in captured.out


def test_main_no_command(capsys):
    """Test main function with no command."""
    with patch("sys.argv", ["cflibs"]):
        try:
            main()
        except SystemExit:
            pass

    captured = capsys.readouterr()
    assert "forward" in captured.out or "invert" in captured.out


def test_main_version(capsys):
    """Test version flag."""
    with patch("sys.argv", ["cflibs", "--version"]):
        try:
            main()
        except SystemExit:
            pass

    captured = capsys.readouterr()
    assert "0.1.0" in captured.out or "version" in captured.out.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
