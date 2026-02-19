"""
Interactive visualization widgets for CF-LIBS analysis.

This module provides Jupyter-compatible interactive widgets for visualizing
CF-LIBS spectra, Boltzmann plots, Bayesian posteriors, and quality metrics.

Optional Dependencies
---------------------
The widgets require ipywidgets and plotly:

    pip install cflibs[widgets]

Usage
-----
>>> from cflibs.visualization import SpectrumViewer, BoltzmannPlotWidget
>>> viewer = SpectrumViewer()
>>> viewer.add_spectrum(wavelength, intensity, label="Sample 1")
>>> viewer.show()
"""

from __future__ import annotations

# pyright: reportMissingImports=false

from . import widgets as _widgets

# Use the same availability signal as the widgets module itself.
HAS_WIDGETS = _widgets.HAS_WIDGETS

if HAS_WIDGETS:
    SpectrumViewer = _widgets.SpectrumViewer
    BoltzmannPlotWidget = _widgets.BoltzmannPlotWidget
    PosteriorViewer = _widgets.PosteriorViewer
    QualityDashboard = _widgets.QualityDashboard
    __all__ = [
        "HAS_WIDGETS",
        "SpectrumViewer",
        "BoltzmannPlotWidget",
        "PosteriorViewer",
        "QualityDashboard",
    ]
else:
    for _name in ("SpectrumViewer", "BoltzmannPlotWidget", "PosteriorViewer", "QualityDashboard"):
        globals().pop(_name, None)
    __all__ = ["HAS_WIDGETS"]
