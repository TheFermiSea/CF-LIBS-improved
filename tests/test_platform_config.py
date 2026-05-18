"""Tests for cflibs.core.platform_config.

Note on imports
---------------
``tests/scripts/test_jax_compile_cache.py`` calls
``importlib.reload(cflibs.core.platform_config)`` to test cache-dir
behavior. After that reload, the ``AcceleratorBackend`` class in
``sys.modules`` is a fresh object, but module-level symbols bound here
at collection time still point to the *original* class. ``configure_jax``
returns the new class, so ``isinstance(result, AcceleratorBackend)``
silently fails -- the repr looks identical (``<AcceleratorBackend.CPU:
'cpu'>``) but the class identities differ.

Workaround: use a fixture that re-imports the symbols inside each test
so the bindings always reflect the current ``sys.modules`` state.
"""

import logging
import os
import sys
from unittest.mock import patch

import pytest


@pytest.fixture
def platform_config_symbols():
    """Re-import ``AcceleratorBackend`` and ``configure_jax`` at test time.

    Returns a ``(AcceleratorBackend, configure_jax)`` tuple bound to the
    *current* ``cflibs.core.platform_config`` module object, immune to any
    earlier ``importlib.reload`` from sibling tests.
    """
    from cflibs.core.platform_config import AcceleratorBackend, configure_jax

    return AcceleratorBackend, configure_jax


# Module-level imports kept for tests that don't go through the fixture
# (and to keep the import-side-effect surface identical for back-compat).
from cflibs.core.platform_config import AcceleratorBackend, configure_jax  # noqa: E402


class TestAcceleratorBackend:
    def test_values(self):
        assert AcceleratorBackend.CPU.value == "cpu"
        assert AcceleratorBackend.CUDA.value == "cuda"


class TestConfigureJax:
    def test_returns_accelerator_backend(self, platform_config_symbols):
        AcceleratorBackend, configure_jax = platform_config_symbols
        result = configure_jax()
        assert isinstance(result, AcceleratorBackend)

    @patch("cflibs.core.platform_config._platform")
    def test_darwin_forces_cpu_env(self, mock_platform, platform_config_symbols):
        """On macOS, JAX_PLATFORMS must be unconditionally set to 'cpu'."""
        AcceleratorBackend, configure_jax = platform_config_symbols
        mock_platform.system.return_value = "Darwin"
        # Even if JAX_PLATFORMS was set to something else, Darwin overrides it
        with patch.dict(os.environ, {"JAX_PLATFORMS": "metal"}, clear=False):
            result = configure_jax()
            assert result == AcceleratorBackend.CPU
            assert os.environ["JAX_PLATFORMS"] == "cpu"

    @patch("cflibs.core.platform_config._platform")
    def test_darwin_returns_cpu(self, mock_platform, platform_config_symbols):
        AcceleratorBackend, configure_jax = platform_config_symbols
        mock_platform.system.return_value = "Darwin"
        assert configure_jax() == AcceleratorBackend.CPU

    @patch("cflibs.core.platform_config._platform")
    def test_linux_no_gpu_returns_cpu(self, mock_platform, platform_config_symbols):
        AcceleratorBackend, configure_jax = platform_config_symbols
        mock_platform.system.return_value = "Linux"
        result = configure_jax(prefer_gpu=False)
        assert result == AcceleratorBackend.CPU

    @pytest.mark.requires_jax
    @patch("cflibs.core.platform_config._platform")
    def test_linux_gpu_exception_falls_back_to_cpu(
        self, mock_platform, platform_config_symbols
    ):
        """When jax.devices('gpu') raises, gracefully fall back."""
        AcceleratorBackend, configure_jax = platform_config_symbols
        mock_platform.system.return_value = "Linux"
        jax = pytest.importorskip("jax")

        with patch.object(jax, "devices", side_effect=ValueError("no GPU")):
            result = configure_jax(prefer_gpu=True)
            assert result == AcceleratorBackend.CPU

    @pytest.mark.requires_jax
    def test_enables_x64(self):
        """configure_jax should enable float64 by default."""
        configure_jax(enable_x64=True)
        jax = pytest.importorskip("jax")

        assert jax.config.jax_enable_x64 is True

    @pytest.mark.requires_jax
    def test_warns_if_jax_already_imported(self, caplog):
        """Should warn when JAX is already in sys.modules."""
        pytest.importorskip("jax")
        assert "jax" in sys.modules
        with caplog.at_level(logging.WARNING, logger="cflibs.core.platform_config"):
            configure_jax()
        assert any("already imported" in r.message for r in caplog.records)

    @patch.dict(sys.modules, {"jax": None})
    def test_jax_not_installed_returns_cpu(self, platform_config_symbols):
        """When JAX is not installed (None in sys.modules), returns CPU."""
        AcceleratorBackend, configure_jax = platform_config_symbols
        result = configure_jax()
        assert result == AcceleratorBackend.CPU

    def test_no_spurious_warning_when_jax_is_none_in_modules(self, caplog):
        """sys.modules['jax'] = None should NOT trigger the 'already imported' warning."""
        with patch.dict(sys.modules, {"jax": None}):
            with caplog.at_level(logging.WARNING, logger="cflibs.core.platform_config"):
                configure_jax()
            assert not any("already imported" in r.message for r in caplog.records)
