"""
Tests for CLI module.
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch

from cflibs.cli.main import forward_model_cmd, invert_cmd, dbgen_cmd, main


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
    """Test database generation command when script is missing."""

    class Args:
        db_path = "test.db"
        elements = None

    args = Args()

    # Mock the path to not exist
    with patch("cflibs.cli.main.Path") as mock_path:
        mock_path.return_value.exists.return_value = False

        # Should exit with error
        with pytest.raises(SystemExit):
            dbgen_cmd(args)


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
