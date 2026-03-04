"""Tests for cflibs.core.platform_config."""

import logging
import os
import sys
from unittest.mock import patch

from cflibs.core.platform_config import AcceleratorBackend, configure_jax


class TestAcceleratorBackend:
    def test_values(self):
        assert AcceleratorBackend.CPU.value == "cpu"
        assert AcceleratorBackend.CUDA.value == "cuda"


class TestConfigureJax:
    def test_returns_accelerator_backend(self):
        result = configure_jax()
        assert isinstance(result, AcceleratorBackend)

    @patch("cflibs.core.platform_config._platform")
    def test_darwin_forces_cpu_env(self, mock_platform):
        """On macOS, JAX_PLATFORMS must be unconditionally set to 'cpu'."""
        mock_platform.system.return_value = "Darwin"
        # Even if JAX_PLATFORMS was set to something else, Darwin overrides it
        with patch.dict(os.environ, {"JAX_PLATFORMS": "metal"}, clear=False):
            result = configure_jax()
            assert result == AcceleratorBackend.CPU
            assert os.environ["JAX_PLATFORMS"] == "cpu"

    @patch("cflibs.core.platform_config._platform")
    def test_darwin_returns_cpu(self, mock_platform):
        mock_platform.system.return_value = "Darwin"
        assert configure_jax() == AcceleratorBackend.CPU

    @patch("cflibs.core.platform_config._platform")
    def test_linux_no_gpu_returns_cpu(self, mock_platform):
        mock_platform.system.return_value = "Linux"
        result = configure_jax(prefer_gpu=False)
        assert result == AcceleratorBackend.CPU

    @patch("cflibs.core.platform_config._platform")
    def test_linux_gpu_exception_falls_back_to_cpu(self, mock_platform):
        """When jax.devices('gpu') raises, gracefully fall back."""
        mock_platform.system.return_value = "Linux"
        import jax

        with patch.object(jax, "devices", side_effect=ValueError("no GPU")):
            result = configure_jax(prefer_gpu=True)
            assert result == AcceleratorBackend.CPU

    def test_enables_x64(self):
        """configure_jax should enable float64 by default."""
        configure_jax(enable_x64=True)
        import jax

        assert jax.config.jax_enable_x64 is True

    def test_warns_if_jax_already_imported(self, caplog):
        """Should warn when JAX is already in sys.modules."""
        assert "jax" in sys.modules
        with caplog.at_level(logging.WARNING, logger="cflibs.core.platform_config"):
            configure_jax()
        assert any("already imported" in r.message for r in caplog.records)

    @patch.dict(sys.modules, {"jax": None})
    def test_jax_not_installed_returns_cpu(self):
        """When JAX is not installed (None in sys.modules), returns CPU."""
        result = configure_jax()
        assert result == AcceleratorBackend.CPU

    def test_no_spurious_warning_when_jax_is_none_in_modules(self, caplog):
        """sys.modules['jax'] = None should NOT trigger the 'already imported' warning."""
        with patch.dict(sys.modules, {"jax": None}):
            with caplog.at_level(logging.WARNING, logger="cflibs.core.platform_config"):
                configure_jax()
            assert not any("already imported" in r.message for r in caplog.records)
