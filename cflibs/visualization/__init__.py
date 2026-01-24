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

# Check for optional dependencies
HAS_WIDGETS = False

try:
    import ipywidgets
    import plotly

    HAS_WIDGETS = True
except ImportError:
    pass

# Public API (only if widgets available)
__all__ = ["HAS_WIDGETS"]

if HAS_WIDGETS:
    from cflibs.visualization.widgets import (
        SpectrumViewer,
        BoltzmannPlotWidget,
        PosteriorViewer,
        QualityDashboard,
    )

    __all__.extend([
        "SpectrumViewer",
        "BoltzmannPlotWidget",
        "PosteriorViewer",
        "QualityDashboard",
    ])
