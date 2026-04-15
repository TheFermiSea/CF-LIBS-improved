"""
Interactive Jupyter widgets for CF-LIBS visualization.

This module provides plotly + ipywidgets-based interactive visualizations
for exploring CF-LIBS data, including:

- SpectrumViewer: Interactive spectrum viewer with zoom/pan/overlay
- BoltzmannPlotWidget: Boltzmann plot with regression fit visualization
- PosteriorViewer: Posterior distributions for Bayesian results
- QualityDashboard: Summary metrics display for CF-LIBS analysis

Requirements
------------
- ipywidgets >= 8.0
- plotly >= 5.0

Install with: pip install cflibs[widgets]
"""

# pyright: reportMissingImports=false
# pyright: reportOptionalMemberAccess=false
# pyright: reportOptionalCall=false
# pyright: reportInvalidTypeForm=false

from __future__ import annotations

from html import escape as html_escape
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional
import numpy as np

from cflibs.core.constants import EV_TO_K

if TYPE_CHECKING:
    import ipywidgets as widgets
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

try:
    import ipywidgets as widgets
    from IPython.display import display

    HAS_IPYWIDGETS = True
except ImportError:
    HAS_IPYWIDGETS = False
    widgets = None  # type: ignore[assignment]
    display = None  # type: ignore[assignment]

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    go = None  # type: ignore[assignment]
    make_subplots = None  # type: ignore[assignment]

HAS_WIDGETS = HAS_IPYWIDGETS and HAS_PLOTLY
_SAFE_COLOR_RE = re.compile(r"^(#[0-9a-fA-F]{3,8}|[a-zA-Z]{1,20})$")


def _escape_html(value: Any) -> str:
    """Escape dynamic values rendered via widgets.HTML."""
    return html_escape(str(value), quote=True)


def _sanitize_css_color(value: Any, default: str = "#1f77b4") -> str:
    """Allow simple CSS color tokens and reject unsafe style fragments."""
    text = str(value).strip()
    return text if _SAFE_COLOR_RE.fullmatch(text) else default


def _weighted_percentiles(
    samples: np.ndarray, weights: Optional[np.ndarray], percentiles: np.ndarray
) -> np.ndarray:
    """Compute weighted percentiles; falls back to unweighted when weights are absent."""
    if weights is None:
        return np.percentile(samples, percentiles)

    if samples.size == 0:
        return np.array([np.nan] * len(percentiles), dtype=float)

    sorter = np.argsort(samples)
    sorted_samples = samples[sorter]
    sorted_weights = weights[sorter]
    cdf = np.cumsum(sorted_weights)
    if cdf[-1] <= 0:
        return np.percentile(samples, percentiles)
    cdf = cdf / cdf[-1]
    return np.interp(percentiles / 100.0, cdf, sorted_samples)


def _require_widgets() -> None:
    """Raise ImportError if widgets dependencies are not available."""
    if not HAS_WIDGETS:
        missing = []
        if not HAS_IPYWIDGETS:
            missing.append("ipywidgets")
        if not HAS_PLOTLY:
            missing.append("plotly")
        raise ImportError(
            f"Widget dependencies not available: {', '.join(missing)}. "
            "Install with: pip install cflibs[widgets]"
        )


class SpectrumViewer:
    """
    Interactive spectrum viewer with zoom, pan, and multi-spectrum overlay.

    Features:
    - Pan and zoom via plotly controls
    - Overlay multiple spectra for comparison
    - Toggle visibility of individual spectra
    - Adjustable line width and colors
    - Export to static image for publications

    Parameters
    ----------
    title : str, optional
        Figure title (default: "LIBS Spectrum Viewer")
    height : int, optional
        Figure height in pixels (default: 500)
    width : int, optional
        Figure width in pixels (default: 900)

    Examples
    --------
    >>> viewer = SpectrumViewer()
    >>> viewer.add_spectrum(wavelength1, intensity1, label="Sample A")
    >>> viewer.add_spectrum(wavelength2, intensity2, label="Sample B", color="red")
    >>> viewer.show()
    """

    def __init__(
        self,
        title: str = "LIBS Spectrum Viewer",
        height: int = 500,
        width: int = 900,
    ):
        _require_widgets()

        self.title = title
        self.height = height
        self.width = width
        self._spectra: List[Dict[str, Any]] = []
        self._fig: Optional["go.FigureWidget"] = None
        self._controls: Optional["widgets.VBox"] = None

    def add_spectrum(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        label: str = "Spectrum",
        color: Optional[str] = None,
        line_width: float = 1.0,
        opacity: float = 1.0,
    ) -> "SpectrumViewer":
        """
        Add a spectrum to the viewer.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength values in nm
        intensity : np.ndarray
            Intensity values (arbitrary units)
        label : str, optional
            Label for the spectrum (default: "Spectrum")
        color : str, optional
            Line color (default: auto-assigned)
        line_width : float, optional
            Line width in pixels (default: 1.0)
        opacity : float, optional
            Line opacity 0-1 (default: 1.0)

        Returns
        -------
        SpectrumViewer
            Self for method chaining
        """
        wavelength_arr = np.asarray(wavelength)
        intensity_arr = np.asarray(intensity)

        if wavelength_arr.ndim != 1 or intensity_arr.ndim != 1:
            raise ValueError("wavelength and intensity must be 1D arrays")
        if wavelength_arr.shape != intensity_arr.shape:
            raise ValueError("wavelength and intensity must have matching shapes")
        if wavelength_arr.size == 0:
            raise ValueError("wavelength and intensity must not be empty")

        self._spectra.append(
            {
                "wavelength": wavelength_arr,
                "intensity": intensity_arr,
                "label": label,
                "color": color,
                "line_width": line_width,
                "opacity": opacity,
            }
        )
        return self

    def _build_figure(self) -> "go.FigureWidget":
        """Build the plotly figure with all spectra."""
        fig = go.FigureWidget()

        # Default color cycle
        colors = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]

        for i, spec in enumerate(self._spectra):
            color = spec["color"] or colors[i % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=spec["wavelength"],
                    y=spec["intensity"],
                    mode="lines",
                    name=spec["label"],
                    line=dict(
                        color=color,
                        width=spec["line_width"],
                    ),
                    opacity=spec["opacity"],
                )
            )

        fig.update_layout(
            title=self.title,
            xaxis_title="Wavelength (nm)",
            yaxis_title="Intensity (a.u.)",
            height=self.height,
            width=self.width,
            hovermode="x unified",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        # Enable zoom and pan
        fig.update_xaxes(rangeslider_visible=True)

        return fig

    def _build_controls(self) -> "widgets.VBox":
        """Build control widgets for the viewer."""
        # Wavelength range controls
        if self._spectra:
            all_wl = np.concatenate([s["wavelength"] for s in self._spectra])
            wl_min, wl_max = float(all_wl.min()), float(all_wl.max())
        else:
            wl_min, wl_max = 200.0, 800.0

        range_slider = widgets.FloatRangeSlider(
            value=[wl_min, wl_max],
            min=wl_min,
            max=wl_max,
            step=0.1,
            description="Range (nm):",
            continuous_update=False,
            style={"description_width": "100px"},
            layout=widgets.Layout(width="400px"),
        )

        def on_range_change(change: Dict[str, Any]) -> None:
            if self._fig is not None:
                self._fig.update_xaxes(range=list(change["new"]))

        range_slider.observe(on_range_change, names="value")

        # Log scale toggle
        log_toggle = widgets.Checkbox(
            value=False,
            description="Log Y-axis",
        )

        def on_log_toggle(change: Dict[str, Any]) -> None:
            if self._fig is not None:
                yaxis_type = "log" if change["new"] else "linear"
                self._fig.update_yaxes(type=yaxis_type)

        log_toggle.observe(on_log_toggle, names="value")

        # Reset button
        reset_btn = widgets.Button(
            description="Reset View",
            button_style="info",
            icon="refresh",
        )

        def on_reset(_: Any) -> None:
            if self._fig is not None:
                self._fig.update_xaxes(range=[wl_min, wl_max])
                self._fig.update_yaxes(autorange=True)
                range_slider.value = [wl_min, wl_max]

        reset_btn.on_click(on_reset)

        return widgets.VBox(
            [
                widgets.HBox([range_slider, log_toggle, reset_btn]),
            ]
        )

    def show(self) -> "widgets.VBox":
        """
        Display the interactive spectrum viewer.

        Returns
        -------
        widgets.VBox
            The complete widget including figure and controls
        """
        if not self._spectra:
            raise ValueError("No spectra added. Use add_spectrum() first.")

        self._fig = self._build_figure()
        self._controls = self._build_controls()

        container = widgets.VBox([self._controls, self._fig])
        display(container)
        return container

    def to_static(self, **kwargs: Any) -> "go.Figure":
        """
        Export to a static plotly Figure for publication.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to Figure.update_layout()

        Returns
        -------
        go.Figure
            Static figure suitable for export
        """
        fig = go.Figure(self._build_figure())

        # Remove interactive elements
        fig.update_xaxes(rangeslider_visible=False)

        # Apply any custom layout
        if kwargs:
            fig.update_layout(**kwargs)

        return fig


class BoltzmannPlotWidget:
    """
    Interactive Boltzmann plot visualization with regression fit.

    Displays upper level energy (E_k) vs ln(I*lambda/gA) with:
    - Interactive hover showing line parameters
    - Fit line with temperature annotation
    - Highlighted outlier points
    - Residual subplot

    Parameters
    ----------
    title : str, optional
        Figure title (default: "Boltzmann Plot")
    height : int, optional
        Figure height in pixels (default: 600)
    width : int, optional
        Figure width in pixels (default: 800)
    show_residuals : bool, optional
        Show residuals subplot (default: True)

    Examples
    --------
    >>> from cflibs.inversion import BoltzmannPlotFitter, LineObservation
    >>> fitter = BoltzmannPlotFitter()
    >>> result = fitter.fit(observations)
    >>> widget = BoltzmannPlotWidget()
    >>> widget.plot(observations, result)
    >>> widget.show()
    """

    def __init__(
        self,
        title: str = "Boltzmann Plot",
        height: int = 600,
        width: int = 800,
        show_residuals: bool = True,
    ):
        _require_widgets()

        self.title = title
        self.height = height
        self.width = width
        self.show_residuals = show_residuals
        self._fig: Optional["go.FigureWidget"] = None
        self._observations: Optional[List[Any]] = None
        self._result: Optional[Any] = None

    def plot(
        self,
        observations: List[Any],
        result: Any,
    ) -> "BoltzmannPlotWidget":
        """
        Create the Boltzmann plot from observations and fit result.

        Parameters
        ----------
        observations : List[LineObservation]
            List of spectral line observations
        result : BoltzmannFitResult
            Result from BoltzmannPlotFitter.fit()

        Returns
        -------
        BoltzmannPlotWidget
            Self for method chaining
        """
        self._observations = observations
        self._result = result
        return self

    def _build_figure(self) -> "go.FigureWidget":
        """Build the plotly figure."""
        if self._observations is None or self._result is None:
            raise ValueError("No data. Call plot() first.")

        obs = self._observations
        result = self._result

        if len(obs) == 0:
            raise ValueError("No observations provided. Call plot() with at least one line.")

        # Extract data
        x_all = np.array([o.E_k_ev for o in obs])
        y_all = np.array([o.y_value for o in obs])
        y_err = np.array([o.y_uncertainty for o in obs])

        # Separate inliers and outliers
        rejected_set = set(result.rejected_points)
        inlier_mask = np.array([i not in rejected_set for i in range(len(obs))])
        inlier_indices = np.where(inlier_mask)[0]
        outlier_indices = np.where(~inlier_mask)[0]

        # Create subplots if showing residuals
        if self.show_residuals:
            fig = make_subplots(
                rows=2,
                cols=1,
                row_heights=[0.7, 0.3],
                shared_xaxes=True,
                vertical_spacing=0.08,
            )
        else:
            fig = go.FigureWidget()

        # Hover text with line info
        hover_text = []
        for i, o in enumerate(obs):
            status = "Rejected" if i in rejected_set else "Included"
            hover_text.append(
                f"<b>{o.element} {o.ionization_stage}</b><br>"
                f"lambda: {o.wavelength_nm:.2f} nm<br>"
                f"E_k: {o.E_k_ev:.3f} eV<br>"
                f"gA: {o.g_k * o.A_ki:.2e} s^-1<br>"
                f"Status: {status}"
            )

        # Plot inliers
        fig.add_trace(
            go.Scatter(
                x=x_all[inlier_mask],
                y=y_all[inlier_mask],
                mode="markers",
                name="Inliers",
                marker=dict(
                    size=10,
                    color="#1f77b4",
                    symbol="circle",
                ),
                error_y=dict(
                    type="data",
                    array=y_err[inlier_mask],
                    visible=True,
                    color="#1f77b4",
                    thickness=1,
                ),
                text=[hover_text[i] for i in inlier_indices],
                hovertemplate="%{text}<extra></extra>",
            ),
            row=1 if self.show_residuals else None,
            col=1 if self.show_residuals else None,
        )

        # Plot outliers
        if np.any(~inlier_mask):
            fig.add_trace(
                go.Scatter(
                    x=x_all[~inlier_mask],
                    y=y_all[~inlier_mask],
                    mode="markers",
                    name="Outliers",
                    marker=dict(
                        size=10,
                        color="#d62728",
                        symbol="x",
                    ),
                    text=[hover_text[i] for i in outlier_indices],
                    hovertemplate="%{text}<extra></extra>",
                ),
                row=1 if self.show_residuals else None,
                col=1 if self.show_residuals else None,
            )

        # Plot fit line
        x_fit = np.linspace(x_all.min(), x_all.max(), 100)
        y_fit = result.slope * x_fit + result.intercept

        temp_str = f"{result.temperature_K:.0f}"
        temp_err_str = f"{result.temperature_uncertainty_K:.0f}"

        fig.add_trace(
            go.Scatter(
                x=x_fit,
                y=y_fit,
                mode="lines",
                name=f"Fit: T = {temp_str} +/- {temp_err_str} K",
                line=dict(color="black", dash="dash", width=2),
                hoverinfo="skip",
            ),
            row=1 if self.show_residuals else None,
            col=1 if self.show_residuals else None,
        )

        # Add residuals subplot
        if self.show_residuals:
            y_pred = result.slope * x_all + result.intercept
            residuals = y_all - y_pred

            fig.add_trace(
                go.Scatter(
                    x=x_all[inlier_mask],
                    y=residuals[inlier_mask],
                    mode="markers",
                    name="Residuals",
                    marker=dict(size=6, color="#1f77b4"),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            if np.any(~inlier_mask):
                fig.add_trace(
                    go.Scatter(
                        x=x_all[~inlier_mask],
                        y=residuals[~inlier_mask],
                        mode="markers",
                        name="Outlier Residuals",
                        marker=dict(size=6, color="#d62728", symbol="x"),
                        showlegend=False,
                    ),
                    row=2,
                    col=1,
                )

            # Add zero line
            fig.add_hline(
                y=0,
                line_dash="dot",
                line_color="gray",
                row=2,
                col=1,
            )

            fig.update_yaxes(title_text="Residuals", row=2, col=1)
            fig.update_xaxes(title_text="Upper Level Energy E_k (eV)", row=2, col=1)

        # Update layout
        fig.update_layout(
            title=dict(
                text=f"{self.title}<br><sup>R^2 = {result.r_squared:.4f}, "
                f"Method: {result.fit_method}</sup>",
            ),
            height=self.height,
            width=self.width,
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
        )

        if self.show_residuals:
            fig.update_yaxes(title_text="ln(I * lambda / (g * A))", row=1, col=1)
        else:
            fig.update_xaxes(title_text="Upper Level Energy E_k (eV)")
            fig.update_yaxes(title_text="ln(I * lambda / (g * A))")

        if isinstance(fig, go.FigureWidget):
            return fig
        return go.FigureWidget(fig)

    def show(self) -> "go.FigureWidget":
        """
        Display the interactive Boltzmann plot.

        Returns
        -------
        go.FigureWidget
            The interactive figure widget
        """
        self._fig = self._build_figure()
        display(self._fig)
        return self._fig

    def to_static(self, **kwargs: Any) -> "go.Figure":
        """
        Export to a static plotly Figure for publication.

        Parameters
        ----------
        **kwargs
            Additional arguments passed to Figure.update_layout()

        Returns
        -------
        go.Figure
            Static figure suitable for export
        """
        fig = go.Figure(self._build_figure())
        if kwargs:
            fig.update_layout(**kwargs)
        return fig


class PosteriorViewer:
    """
    Interactive posterior distribution viewer for Bayesian CF-LIBS results.

    Displays marginal distributions and 2D corner plots for posterior samples
    from MCMC or nested sampling results.

    Features:
    - 1D marginal histograms with KDE
    - 2D scatter/contour plots for parameter pairs
    - Credible interval annotations
    - Interactive parameter selection

    Parameters
    ----------
    title : str, optional
        Figure title (default: "Posterior Distributions")
    height : int, optional
        Figure height in pixels (default: 700)
    width : int, optional
        Figure width in pixels (default: 800)

    Examples
    --------
    >>> from cflibs.inversion import MCMCSampler
    >>> result = sampler.run(observed_spectrum)
    >>> viewer = PosteriorViewer()
    >>> viewer.from_mcmc_result(result)
    >>> viewer.show()
    """

    def __init__(
        self,
        title: str = "Posterior Distributions",
        height: int = 700,
        width: int = 800,
    ):
        _require_widgets()

        self.title = title
        self.height = height
        self.width = width
        self._samples: Dict[str, np.ndarray] = {}
        self._weights: Optional[np.ndarray] = None
        self._param_labels: Dict[str, str] = {}
        self._fig: Optional["go.FigureWidget"] = None

    def from_mcmc_result(self, result: Any) -> "PosteriorViewer":
        """
        Load samples from an MCMCResult.

        Parameters
        ----------
        result : MCMCResult
            Result from MCMCSampler.run()

        Returns
        -------
        PosteriorViewer
            Self for method chaining
        """
        samples = result.samples
        self._samples = {}

        # Extract and flatten samples
        self._samples["T_eV"] = np.asarray(samples["T_eV"]).flatten()
        self._samples["log_ne"] = np.asarray(samples["log_ne"]).flatten()

        # Handle concentrations
        if "concentrations" in samples:
            conc = np.asarray(samples["concentrations"])
            if conc.ndim == 3:
                conc = conc.reshape(-1, conc.shape[-1])
            for i, el in enumerate(result.concentrations_mean.keys()):
                self._samples[f"C_{el}"] = conc[:, i]

        # Labels for display
        self._param_labels = {
            "T_eV": "Temperature (eV)",
            "log_ne": "log10(n_e) [cm^-3]",
        }
        for el in result.concentrations_mean.keys():
            self._param_labels[f"C_{el}"] = f"C_{el}"

        self._weights = None
        return self

    def from_nested_result(self, result: Any) -> "PosteriorViewer":
        """
        Load samples from a NestedSamplingResult.

        Parameters
        ----------
        result : NestedSamplingResult
            Result from NestedSampler.run()

        Returns
        -------
        PosteriorViewer
            Self for method chaining
        """
        samples = result.samples
        self._samples = {}

        self._samples["T_eV"] = np.asarray(samples["T_eV"]).flatten()
        self._samples["log_ne"] = np.asarray(samples["log_ne"]).flatten()

        if "concentrations" in samples:
            conc = np.asarray(samples["concentrations"])
            if conc.ndim == 1:
                conc = conc.reshape(-1, 1)
            for i, el in enumerate(result.concentrations_mean.keys()):
                self._samples[f"C_{el}"] = conc[:, i]

        self._param_labels = {
            "T_eV": "Temperature (eV)",
            "log_ne": "log10(n_e) [cm^-3]",
        }
        for el in result.concentrations_mean.keys():
            self._param_labels[f"C_{el}"] = f"C_{el}"

        self._weights = result.weights
        return self

    def from_samples(
        self,
        samples: Dict[str, np.ndarray],
        weights: Optional[np.ndarray] = None,
        labels: Optional[Dict[str, str]] = None,
    ) -> "PosteriorViewer":
        """
        Load samples from a dictionary.

        Parameters
        ----------
        samples : Dict[str, np.ndarray]
            Dictionary of parameter name -> samples array
        weights : np.ndarray, optional
            Sample weights (for nested sampling)
        labels : Dict[str, str], optional
            Display labels for parameters

        Returns
        -------
        PosteriorViewer
            Self for method chaining
        """
        self._samples = {k: np.asarray(v).flatten() for k, v in samples.items()}
        self._weights = weights
        self._param_labels = labels or {k: k for k in samples}
        return self

    def _validate_plot_params(self, params: Optional[List[str]]) -> List[str]:
        """Validate input parameters for building a posterior plot."""
        if not self._samples:
            raise ValueError("No samples loaded. Call from_*() first.")

        valid_params = params or list(self._samples.keys())

        if not valid_params:
            raise ValueError("At least one parameter required.")

        missing = [p for p in valid_params if p not in self._samples]
        if missing:
            missing_str = ", ".join(sorted(missing))
            raise ValueError(f"Requested parameters not available in samples: {missing_str}")

        empty = [p for p in valid_params if np.asarray(self._samples[p]).size == 0]
        if empty:
            empty_str = ", ".join(sorted(empty))
            raise ValueError(f"Cannot plot empty posterior samples for parameter(s): {empty_str}")

        lengths = {p: np.asarray(self._samples[p]).size for p in valid_params}
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            lengths_str = ", ".join(f"{k}={v}" for k, v in lengths.items())
            raise ValueError(
                "All selected posterior parameters must have equal sample length "
                f"(got: {lengths_str})"
            )
        return valid_params

    def _get_normalized_weights(self, base_n: int) -> Optional[np.ndarray]:
        """Validate and normalize sample weights."""
        if self._weights is None:
            return None

        candidate = np.asarray(self._weights, dtype=float).reshape(-1)
        if candidate.size != base_n:
            raise ValueError(
                "weights length must match sample length when plotting posterior samples"
            )
        total_weight = float(np.sum(candidate))
        if total_weight > 0:
            return candidate / total_weight
        return None

    def _add_lower_triangle_plot(
        self,
        fig: "go.FigureWidget",
        y_param: str,
        x_param: str,
        row: int,
        col: int,
        weights: Optional[np.ndarray],
    ) -> None:
        """Add a 2D scatter plot to the lower triangle of a corner plot."""
        xi = self._samples[x_param]
        yi = self._samples[y_param]

        # Subsample for performance
        n_samples = len(xi)
        n_plot = min(2000, n_samples)
        if n_samples == 0:
            idx = np.array([], dtype=int)
        elif n_plot == n_samples:
            idx = np.arange(n_samples)
        elif weights is None:
            idx = np.random.choice(n_samples, n_plot, replace=False)
        else:
            positive = np.flatnonzero(weights > 0)
            if positive.size == 0:
                idx = np.random.choice(n_samples, n_plot, replace=False)
            elif n_plot >= positive.size:
                idx = positive
            else:
                sample_weights = weights[positive]
                sample_weights = sample_weights / np.sum(sample_weights)
                idx = np.random.choice(
                    positive,
                    n_plot,
                    replace=False,
                    p=sample_weights,
                )

        fig.add_trace(
            go.Scatter(
                x=xi[idx],
                y=yi[idx],
                mode="markers",
                marker=dict(
                    size=2,
                    color="#1f77b4",
                    opacity=0.3,
                ),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

    def _add_diagonal_plot(
        self,
        fig: "go.FigureWidget",
        param: str,
        row: int,
        col: int,
        weights: Optional[np.ndarray],
    ) -> None:
        """Add a 1D marginal histogram to the diagonal of a corner plot."""
        samples = self._samples[param]
        if weights is None:
            fig.add_trace(
                go.Histogram(
                    x=samples,
                    nbinsx=30,
                    showlegend=False,
                    marker_color="#1f77b4",
                    opacity=0.7,
                ),
                row=row,
                col=col,
            )
        else:
            counts, edges = np.histogram(samples, bins=30, weights=weights)
            centers = 0.5 * (edges[:-1] + edges[1:])
            fig.add_trace(
                go.Bar(
                    x=centers,
                    y=counts,
                    showlegend=False,
                    marker_color="#1f77b4",
                    opacity=0.7,
                ),
                row=row,
                col=col,
            )

        # Add credible interval lines
        q025, q50, q975 = _weighted_percentiles(
            np.asarray(samples, dtype=float), weights, np.array([2.5, 50.0, 97.5])
        )
        for q, dash in [(q025, "dot"), (q50, "solid"), (q975, "dot")]:
            fig.add_vline(
                x=q,
                line_dash=dash,
                line_color="red",
                line_width=1,
                row=row,
                col=col,
            )

    def _build_single_marginal(self, param: str, weights: Optional[np.ndarray]) -> "go.FigureWidget":
        """Build a single marginal distribution plot."""
        fig = go.FigureWidget()
        samples = self._samples[param]
        label = self._param_labels.get(param, param)

        if weights is None:
            fig.add_trace(
                go.Histogram(
                    x=samples,
                    name=label,
                    opacity=0.7,
                    nbinsx=50,
                )
            )
        else:
            counts, edges = np.histogram(samples, bins=50, weights=weights)
            centers = 0.5 * (edges[:-1] + edges[1:])
            fig.add_trace(
                go.Bar(
                    x=centers,
                    y=counts,
                    name=label,
                    opacity=0.7,
                )
            )

        # Add credible interval lines
        q025, q50, q975 = _weighted_percentiles(
            np.asarray(samples, dtype=float), weights, np.array([2.5, 50.0, 97.5])
        )
        for q, style in [(q025, "dot"), (q50, "solid"), (q975, "dot")]:
            fig.add_vline(x=q, line_dash=style, line_color="red")

        fig.update_layout(
            title=f"{self.title}: {label}",
            xaxis_title=label,
            yaxis_title="Count",
            height=400,
            width=500,
        )

        return fig

    def _build_figure(self, params: Optional[List[str]] = None) -> "go.FigureWidget":
        """Build the corner plot figure."""
        params = self._validate_plot_params(params)
        n_params = len(params)
        base_n = np.asarray(self._samples[params[0]]).size
        weights = self._get_normalized_weights(base_n)

        if n_params == 1:
            return self._build_single_marginal(params[0], weights)

        fig = make_subplots(
            rows=n_params,
            cols=n_params,
            shared_xaxes=False,
            shared_yaxes=False,
            vertical_spacing=0.02,
            horizontal_spacing=0.02,
        )

        for i, pi in enumerate(params):
            for j, pj in enumerate(params):
                row, col = i + 1, j + 1
                if i == j:
                    self._add_diagonal_plot(fig, pi, row, col, weights)
                elif i > j:
                    self._add_lower_triangle_plot(fig, pi, pj, row, col, weights)

                # Axis labels
                if i == n_params - 1:
                    fig.update_xaxes(title_text=self._param_labels.get(pj, pj), row=row, col=col)
                if j == 0 and i > 0:
                    fig.update_yaxes(title_text=self._param_labels.get(pi, pi), row=row, col=col)

        fig.update_layout(
            title=self.title, height=self.height, width=self.width, showlegend=False
        )
        return go.FigureWidget(fig)

    def show(self, params: Optional[List[str]] = None) -> "go.FigureWidget":
        """
        Display the interactive corner plot.

        Parameters
        ----------
        params : List[str], optional
            Parameters to include (default: all)

        Returns
        -------
        go.FigureWidget
            The interactive figure widget
        """
        self._fig = self._build_figure(params)
        display(self._fig)
        return self._fig

    def marginal(self, param: str) -> "go.FigureWidget":
        """
        Display a single marginal distribution.

        Parameters
        ----------
        param : str
            Parameter name

        Returns
        -------
        go.FigureWidget
            The histogram figure widget
        """
        fig = self._build_figure([param])
        display(fig)
        return fig

    def to_static(self, params: Optional[List[str]] = None, **kwargs: Any) -> "go.Figure":
        """
        Export to a static plotly Figure for publication.

        Parameters
        ----------
        params : List[str], optional
            Parameters to include (default: all)
        **kwargs
            Additional arguments passed to Figure.update_layout()

        Returns
        -------
        go.Figure
            Static figure suitable for export
        """
        fig = go.Figure(self._build_figure(params))
        if kwargs:
            fig.update_layout(**kwargs)
        return fig


class QualityDashboard:
    """
    Summary dashboard for CF-LIBS quality metrics.

    Displays key quality indicators from CF-LIBS analysis including:
    - Temperature and uncertainties
    - Electron density
    - Composition breakdown
    - Goodness-of-fit metrics (R^2, chi^2)
    - Convergence status

    Parameters
    ----------
    title : str, optional
        Dashboard title (default: "CF-LIBS Analysis Summary")

    Examples
    --------
    >>> from cflibs.inversion import IterativeCFLIBSSolver
    >>> result = solver.solve(observations)
    >>> dashboard = QualityDashboard()
    >>> dashboard.from_cflibs_result(result)
    >>> dashboard.show()
    """

    def __init__(self, title: str = "CF-LIBS Analysis Summary"):
        _require_widgets()

        self.title = title
        self._metrics: Dict[str, Any] = {}
        self._widget: Optional["widgets.VBox"] = None

    def from_cflibs_result(self, result: Any) -> "QualityDashboard":
        """
        Load metrics from a CFLIBSResult.

        Parameters
        ----------
        result : CFLIBSResult
            Result from IterativeCFLIBSSolver.solve()

        Returns
        -------
        QualityDashboard
            Self for method chaining
        """
        self._metrics = {
            "temperature_K": result.temperature_K,
            "temperature_uncertainty_K": result.temperature_uncertainty_K,
            "electron_density_cm3": result.electron_density_cm3,
            "electron_density_uncertainty_cm3": getattr(
                result, "electron_density_uncertainty_cm3", 0.0
            ),
            "concentrations": result.concentrations,
            "concentration_uncertainties": result.concentration_uncertainties,
            "iterations": result.iterations,
            "converged": result.converged,
            "quality_metrics": getattr(result, "quality_metrics", {}),
            "result_type": "CFLIBSResult",
        }
        return self

    def from_mcmc_result(self, result: Any) -> "QualityDashboard":
        """
        Load metrics from an MCMCResult.

        Parameters
        ----------
        result : MCMCResult
            Result from MCMCSampler.run()

        Returns
        -------
        QualityDashboard
            Self for method chaining
        """
        self._metrics = {
            "temperature_K": result.T_K_mean,
            "temperature_uncertainty_K": result.T_eV_std * EV_TO_K,
            "T_eV": result.T_eV_mean,
            "T_eV_std": result.T_eV_std,
            "log_ne": result.log_ne_mean,
            "log_ne_std": result.log_ne_std,
            "electron_density_cm3": result.n_e_mean,
            "concentrations": result.concentrations_mean,
            "concentration_uncertainties": result.concentrations_std,
            "n_samples": result.n_samples,
            "n_chains": result.n_chains,
            "n_warmup": result.n_warmup,
            "r_hat": result.r_hat,
            "ess": result.ess,
            "converged": result.is_converged,
            "convergence_status": result.convergence_status.value,
            "result_type": "MCMCResult",
        }
        return self

    def from_dict(self, metrics: Dict[str, Any]) -> "QualityDashboard":
        """
        Load metrics from a dictionary.

        Parameters
        ----------
        metrics : Dict[str, Any]
            Dictionary of metric name -> value

        Returns
        -------
        QualityDashboard
            Self for method chaining
        """
        self._metrics = metrics
        return self

    def _create_metric_card(
        self,
        title: str,
        value: str,
        subtitle: Optional[str] = None,
        color: str = "#1f77b4",
    ) -> "widgets.VBox":
        """Create a styled metric card widget."""
        safe_color = _sanitize_css_color(color)
        title_text = _escape_html(title)
        value_text = _escape_html(value)
        title_label = widgets.HTML(
            f"<div style='color: #666; font-size: 12px; margin-bottom: 2px;'>{title_text}</div>"
        )
        value_label = widgets.HTML(
            f"<div style='color: {safe_color}; font-size: 24px; font-weight: bold;'>{value_text}</div>"
        )
        children = [title_label, value_label]

        if subtitle:
            subtitle_text = _escape_html(subtitle)
            subtitle_label = widgets.HTML(
                f"<div style='color: #999; font-size: 11px;'>{subtitle_text}</div>"
            )
            children.append(subtitle_label)

        return widgets.VBox(
            children,
            layout=widgets.Layout(
                border="1px solid #ddd",
                border_radius="8px",
                padding="12px",
                margin="4px",
                min_width="150px",
            ),
        )

    def _build_widget(self) -> "widgets.VBox":
        """Build the dashboard widget."""
        if not self._metrics:
            raise ValueError("No metrics loaded. Call from_*() first.")

        cards = []

        # Title
        safe_title = _escape_html(self.title)
        title_html = widgets.HTML(f"<h2 style='margin: 0 0 16px 0; color: #333;'>{safe_title}</h2>")

        # Temperature card
        T_K = self._metrics.get("temperature_K", 0)
        T_err = self._metrics.get("temperature_uncertainty_K", 0)
        temp_card = self._create_metric_card(
            "Temperature",
            f"{T_K:,.0f} K",
            f"+/- {T_err:,.0f} K" if T_err else None,
            color="#d62728",
        )
        cards.append(temp_card)

        # Electron density card
        ne = self._metrics.get("electron_density_cm3", 0)
        ne_err = self._metrics.get("electron_density_uncertainty_cm3", 0)
        ne_card = self._create_metric_card(
            "Electron Density",
            f"{ne:.2e} cm^-3",
            f"+/- {ne_err:.2e}" if ne_err else None,
            color="#1f77b4",
        )
        cards.append(ne_card)

        # Convergence card
        converged = self._metrics.get("converged", None)
        if converged is not None:
            conv_text = "Converged" if converged else "Not Converged"
            conv_color = "#2ca02c" if converged else "#d62728"
            subtitle = self._metrics.get("convergence_status", "")
            if "iterations" in self._metrics:
                subtitle = f"{self._metrics['iterations']} iterations"
            conv_card = self._create_metric_card(
                "Status",
                conv_text,
                subtitle,
                color=conv_color,
            )
            cards.append(conv_card)

        # Quality metrics
        quality = self._metrics.get("quality_metrics", {})
        if quality:
            r2 = quality.get("r_squared", quality.get("R2", None))
            if r2 is not None:
                r2_card = self._create_metric_card(
                    "R-squared",
                    f"{r2:.4f}",
                    color="#9467bd",
                )
                cards.append(r2_card)

        # MCMC-specific
        if self._metrics.get("result_type") == "MCMCResult":
            n_samples = self._metrics.get("n_samples", 0)
            n_chains = self._metrics.get("n_chains", 1)
            samples_card = self._create_metric_card(
                "Samples",
                f"{n_samples:,}",
                f"{n_chains} chain(s)",
                color="#17becf",
            )
            cards.append(samples_card)

        # Top row of cards
        top_row = widgets.HBox(
            cards,
            layout=widgets.Layout(flex_flow="row wrap"),
        )

        # Composition breakdown
        concentrations = self._metrics.get("concentrations", {})
        conc_uncertainties = self._metrics.get("concentration_uncertainties", {})

        if concentrations:
            comp_html = "<h4 style='margin: 16px 0 8px 0; color: #666;'>Composition</h4>"
            comp_html += "<table style='border-collapse: collapse; width: 100%;'>"
            comp_html += "<tr style='background: #f5f5f5;'>"
            comp_html += "<th style='padding: 8px; text-align: left;'>Element</th>"
            comp_html += "<th style='padding: 8px; text-align: right;'>Concentration</th>"
            comp_html += "<th style='padding: 8px; text-align: right;'>Uncertainty</th>"
            comp_html += "</tr>"

            for el, conc in concentrations.items():
                unc = conc_uncertainties.get(el, 0)
                safe_element = _escape_html(el)
                comp_html += f"<tr><td style='padding: 8px;'><b>{safe_element}</b></td>"
                comp_html += f"<td style='padding: 8px; text-align: right;'>{conc:.4f}</td>"
                comp_html += f"<td style='padding: 8px; text-align: right;'>+/- {unc:.4f}</td>"
                comp_html += "</tr>"

            comp_html += "</table>"
            comp_widget = widgets.HTML(comp_html)
        else:
            comp_widget = widgets.HTML("")

        # R-hat diagnostics for MCMC
        r_hat = self._metrics.get("r_hat", {})
        ess = self._metrics.get("ess", {})

        if r_hat:
            diag_html = (
                "<h4 style='margin: 16px 0 8px 0; color: #666;'>Convergence Diagnostics</h4>"
            )
            diag_html += "<table style='border-collapse: collapse; width: 100%;'>"
            diag_html += "<tr style='background: #f5f5f5;'>"
            diag_html += "<th style='padding: 8px; text-align: left;'>Parameter</th>"
            diag_html += "<th style='padding: 8px; text-align: right;'>R-hat</th>"
            diag_html += "<th style='padding: 8px; text-align: right;'>ESS</th>"
            diag_html += "<th style='padding: 8px; text-align: center;'>Status</th>"
            diag_html += "</tr>"

            for param, rhat in r_hat.items():
                ess_val = ess.get(param, float("nan"))
                status = "ok" if rhat < 1.01 else "warning"
                status_icon = "+" if status == "ok" else "!"
                status_color = "#2ca02c" if status == "ok" else "#d62728"
                safe_param = _escape_html(param)

                diag_html += f"<tr><td style='padding: 8px;'>{safe_param}</td>"
                diag_html += f"<td style='padding: 8px; text-align: right;'>{rhat:.3f}</td>"
                diag_html += f"<td style='padding: 8px; text-align: right;'>{ess_val:.0f}</td>"
                diag_html += (
                    f"<td style='padding: 8px; text-align: center; color: {status_color};'>"
                    f"{status_icon}</td>"
                )
                diag_html += "</tr>"

            diag_html += "</table>"
            diag_widget = widgets.HTML(diag_html)
        else:
            diag_widget = widgets.HTML("")

        return widgets.VBox(
            [
                title_html,
                top_row,
                comp_widget,
                diag_widget,
            ]
        )

    def show(self) -> "widgets.VBox":
        """
        Display the quality dashboard.

        Returns
        -------
        widgets.VBox
            The dashboard widget
        """
        self._widget = self._build_widget()
        display(self._widget)
        return self._widget

    def summary_text(self) -> str:
        """
        Generate a text summary of the metrics.

        Returns
        -------
        str
            Formatted text summary
        """
        if not self._metrics:
            return "No metrics loaded."

        lines = [
            "=" * 60,
            self.title,
            "=" * 60,
        ]

        T_K = self._metrics.get("temperature_K", 0)
        T_err = self._metrics.get("temperature_uncertainty_K", 0)
        lines.append(f"Temperature: {T_K:,.0f} +/- {T_err:,.0f} K")

        ne = self._metrics.get("electron_density_cm3", 0)
        lines.append(f"Electron Density: {ne:.2e} cm^-3")

        converged = self._metrics.get("converged", None)
        if converged is not None:
            lines.append(f"Converged: {converged}")

        concentrations = self._metrics.get("concentrations", {})
        if concentrations:
            lines.append("-" * 60)
            lines.append("Composition:")
            for el, conc in concentrations.items():
                unc = self._metrics.get("concentration_uncertainties", {}).get(el, 0)
                lines.append(f"  {el}: {conc:.4f} +/- {unc:.4f}")

        lines.append("=" * 60)
        return "\n".join(lines)
