"""Tests for visualization optional-dependency gates and widget safeguards."""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_visualization_exports_match_widgets_availability() -> None:
    import cflibs.visualization as visualization
    from cflibs.visualization import widgets as widgets_module

    widget_symbols = {
        "SpectrumViewer",
        "BoltzmannPlotWidget",
        "PosteriorViewer",
        "QualityDashboard",
    }

    assert visualization.HAS_WIDGETS is widgets_module.HAS_WIDGETS
    exported = set(visualization.__all__)

    if visualization.HAS_WIDGETS:
        assert widget_symbols.issubset(exported)
        for name in widget_symbols:
            assert hasattr(visualization, name)
    else:
        assert exported == {"HAS_WIDGETS"}
        for name in widget_symbols:
            assert not hasattr(visualization, name)


def test_visualization_reload_clears_widget_exports() -> None:
    import cflibs.visualization as visualization
    from cflibs.visualization import widgets as widgets_module

    original = widgets_module.HAS_WIDGETS
    try:
        widgets_module.HAS_WIDGETS = False
        reloaded = importlib.reload(visualization)
        assert reloaded.HAS_WIDGETS is False
        assert reloaded.__all__ == ["HAS_WIDGETS"]
        assert not hasattr(reloaded, "SpectrumViewer")
        assert not hasattr(reloaded, "BoltzmannPlotWidget")
        assert not hasattr(reloaded, "PosteriorViewer")
        assert not hasattr(reloaded, "QualityDashboard")
    finally:
        widgets_module.HAS_WIDGETS = original
        importlib.reload(visualization)


def test_escape_html_escapes_user_input() -> None:
    from cflibs.visualization.widgets import _escape_html

    escaped = _escape_html("<script>alert('x')</script>")
    assert escaped == "&lt;script&gt;alert(&#x27;x&#x27;)&lt;/script&gt;"


def test_posterior_viewer_rejects_missing_parameter_before_plot_build() -> None:
    from cflibs.visualization.widgets import PosteriorViewer

    viewer = PosteriorViewer.__new__(PosteriorViewer)
    viewer._samples = {"T_eV": np.asarray([1.0, 2.0])}
    viewer._param_labels = {}
    viewer._weights = None
    viewer.title = "Posterior"
    viewer.height = 700
    viewer.width = 800

    with pytest.raises(ValueError, match="not available in samples"):
        viewer._build_figure(params=["log_ne"])


def test_posterior_viewer_rejects_empty_samples_before_plot_build() -> None:
    from cflibs.visualization.widgets import PosteriorViewer

    viewer = PosteriorViewer.__new__(PosteriorViewer)
    viewer._samples = {"T_eV": np.asarray([])}
    viewer._param_labels = {}
    viewer._weights = None
    viewer.title = "Posterior"
    viewer.height = 700
    viewer.width = 800

    with pytest.raises(ValueError, match="empty posterior samples"):
        viewer._build_figure(params=["T_eV"])
