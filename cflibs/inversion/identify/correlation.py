"""
Correlation-based element identification (modernized Labutin method).

Implements classic and vector-accelerated modes for identifying elements
from experimental spectra using model spectrum correlation matching.
"""

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple
import math
import numpy as np
from scipy.stats import pearsonr

from cflibs.inversion.common.element_id import (
    IdentifiedLine,
    ElementIdentification,
    ElementIdentificationResult,
    get_wavelength_tolerance,
)
from cflibs.inversion.preprocess.preprocessing import detect_peaks_auto
from cflibs.atomic.database import AtomicDatabase
from cflibs.atomic.structures import Transition
from cflibs.plasma.saha_boltzmann import SahaBoltzmannSolver
from cflibs.core.constants import KB_EV
from cflibs.core.logging_config import get_logger
from cflibs.inversion.identify._coverage import (
    CoverageTracker,
    merge_coverage_into_parameters,
)

logger = get_logger("inversion.correlation_identifier")

# JAX is an optional fast path for the (T, n_e) grid model-spectrum
# correlation hot loop inside ``CorrelationIdentifier._identify_classic``.
# The default path keeps SciPy so behavior is unchanged unless
# ``use_jax_classic=True`` is passed to the constructor.
try:
    import jax
    import jax.numpy as jnp

    _HAS_JAX = True
except ImportError:  # pragma: no cover - exercised only when jax missing
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    _HAS_JAX = False


if _HAS_JAX:

    @jax.jit
    def _jax_model_grid_correlations(
        wavelength: "jnp.ndarray",  # (W,)
        intensity: "jnp.ndarray",  # (W,)
        line_wl: "jnp.ndarray",  # (L,)
        line_A_g: "jnp.ndarray",  # (L,)  A_ki * g_k
        line_E: "jnp.ndarray",  # (L,)  E_k in eV
        line_U: "jnp.ndarray",  # (G, L) partition function per (T, line)
        line_W_q: "jnp.ndarray",  # (G, L) Saha weight per (T, n_e, line)
        line_sigma: "jnp.ndarray",  # (L,) Gaussian sigma in nm
        T_eV_grid: "jnp.ndarray",  # (G,) T in eV, flat grid
        exp_scale: "jnp.ndarray",  # () percentile-95 of intensity
        peak_region_threshold: float,
        peak_region_min_points: int,
    ) -> "jnp.ndarray":
        """Compute Pearson correlation of model spectrum vs. ``intensity``
        for each ``(T, n_e)`` grid point.

        Replicates ``CorrelationIdentifier._generate_model_spectrum`` +
        the peak-region masked Pearson logic inside
        ``_identify_classic`` in a single fused JAX kernel.

        Inputs are shaped to flatten the ``T x n_e`` outer product into
        a single ``G`` axis so the kernel runs in one vmap-free pass.
        """
        W = wavelength.shape[0]
        L = line_wl.shape[0]  # noqa: F841 -- shape doc alongside W and G for readers
        G = T_eV_grid.shape[0]

        # Boltzmann factor per (G, L). exp(-E/T) on the flat grid.
        boltz = jnp.exp(-line_E[None, :] / T_eV_grid[:, None])  # (G, L)
        # Per-line emissivity, including Saha W_q and partition function.
        # eps shape (G, L). U shape (G, L); W_q shape (G, L).
        eps = line_W_q * line_A_g[None, :] * boltz / jnp.maximum(line_U, 1e-30)

        # Gaussian profile per (L, W). sigma broadcast to (L, 1).
        diff = (wavelength[None, :] - line_wl[:, None]) / line_sigma[:, None]
        gauss = jnp.exp(-0.5 * diff * diff)  # (L, W)

        # Sum over lines: model[g, w] = sum_l eps[g, l] * gauss[l, w].
        model = eps @ gauss  # (G, W)

        # Robust normalization: 95th percentile per row (matches the CPU
        # path which divides by model_scale, then multiplies by
        # exp_scale).
        model_scale = jnp.percentile(model, 95.0, axis=-1)  # (G,)
        # Guard div-by-zero: leave model unchanged if scale tiny.
        safe = (model_scale > 1e-10) & (exp_scale > 1e-10)
        scale = jnp.where(safe, exp_scale / jnp.where(safe, model_scale, 1.0), 1.0)
        model = model * scale[:, None]  # (G, W)

        # Peak-region mask: per-row normalize, threshold AND/OR fallback.
        i_min = intensity.min()
        i_max = intensity.max()
        m_min = model.min(axis=-1, keepdims=True)  # (G, 1)
        m_max = model.max(axis=-1, keepdims=True)  # (G, 1)
        i_range = i_max - i_min
        m_range = m_max - m_min  # (G, 1)
        sigma_th = peak_region_threshold

        # exp_norm shape (W,), mod_norm shape (G, W).
        exp_norm = jnp.where(
            i_range > 1e-10, (intensity - i_min) / jnp.where(i_range > 1e-10, i_range, 1.0), 0.0
        )
        mod_norm = jnp.where(
            m_range > 1e-10, (model - m_min) / jnp.where(m_range > 1e-10, m_range, 1.0), 0.0
        )
        and_mask = (exp_norm[None, :] >= sigma_th) & (mod_norm >= sigma_th)  # (G, W)
        or_mask = (exp_norm[None, :] >= sigma_th) | (mod_norm >= sigma_th)

        # If model row has no dynamic range, fall back to all-ones mask
        # (CPU path's else branch).
        no_dyn = jnp.broadcast_to((i_range <= 1e-10) | (m_range[:, 0] <= 1e-10), (G,))

        # Per-row: pick AND mask unless it has fewer than min_points
        # support, in which case pick OR (matches CPU).
        and_count = and_mask.sum(axis=-1)  # (G,)
        use_and = and_count >= peak_region_min_points
        chosen = jnp.where(use_and[:, None], and_mask, or_mask)
        # And the no_dyn-range rows fall back to all-ones.
        all_ones = jnp.ones_like(chosen)
        mask = jnp.where(no_dyn[:, None], all_ones, chosen)
        m = mask.astype(jnp.float64)

        # Masked Pearson per row.
        x = jnp.broadcast_to(intensity[None, :], (G, W)).astype(jnp.float64)
        y = model.astype(jnp.float64)
        n = jnp.sum(m, axis=-1)
        n_safe = jnp.maximum(n, 1.0)
        mx = jnp.sum(m * x, axis=-1) / n_safe
        my = jnp.sum(m * y, axis=-1) / n_safe
        xc = x - mx[:, None]
        yc = y - my[:, None]
        cov = jnp.sum(m * xc * yc, axis=-1)
        vx = jnp.sum(m * xc * xc, axis=-1)
        vy = jnp.sum(m * yc * yc, axis=-1)
        denom = jnp.sqrt(vx * vy)
        corr = jnp.where(denom > 1e-20, cov / jnp.where(denom > 1e-20, denom, 1.0), 0.0)
        valid = (vx > 1e-20) & (vy > 1e-20) & (n > 2)
        return jnp.where(valid, corr, 0.0)


class CorrelationIdentifier:
    """
    Correlation-based element identification using model spectra.

    Supports two modes:
    - **classic**: Grid search over (T, n_e) with Pearson correlation
    - **vector**: Fast ANN search via FAISS VectorIndex with multi-model consensus

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database for transitions
    vector_index : VectorIndex, optional
        Pre-built vector index for vector mode
    vector_embedder : object, optional
        Fitted embedder with ``transform()`` method for vector-mode queries.
    library_metadata : List[dict], optional
        Per-spectrum metadata aligned with the vector index. Each item must
        expose element membership via ``element``, ``elements``, ``species``,
        or ``composition``.
    library_spectra : np.ndarray, optional
        Spectra aligned with ``library_metadata``. When provided, vector mode
        refines ANN scores with direct spectrum correlation.
    elements : List[str], optional
        Elements to consider (default: None, uses all elements from database)
    wavelength_tolerance_nm : float
        Wavelength matching tolerance in nm (default: 0.1)
    top_k : int
        Number of nearest neighbors for vector mode (default: 10)
    min_confidence : float
        Minimum confidence threshold for detection (default: 0.03)
    T_range_K : Tuple[float, float]
        Temperature range for classic mode in Kelvin (default: (8000, 12000))
    n_e_range_cm3 : Tuple[float, float]
        Electron density range for classic mode in cm⁻³ (default: (3e16, 3e17))
    T_steps : int
        Temperature grid steps for classic mode (default: 5)
    n_e_steps : int
        Density grid steps for classic mode (default: 3)
    resolving_power : float, optional
        Instrument resolving power (λ/Δλ). If set, per-line sigma is
        wavelength/resolving_power instead of fixed instrument_fwhm_nm (default: None).
    instrument_fwhm_nm : float
        Instrument spectral FWHM in nm (default: 0.05). Used when
        resolving_power is None to derive Gaussian sigma = FWHM / 2.355.
    max_lines_per_element : int
        Cap transitions per element by emissivity (default: 100)
    min_line_strength : float
        Minimum observable line strength A_ki * g_k (default: 1e4)
    reference_temperature : float
        Reference temperature in K for emissivity ranking (default: 10000.0)
    relative_threshold_scale : float
        Scale factor applied to median non-zero score for adaptive rejection
        (default: 1.5). Lower values increase recall; higher values reduce
        false positives.
    peak_region_threshold : float
        Normalized intensity threshold used to define peak-region masks for
        Pearson correlation in classic mode (default: 0.15).
    peak_region_min_points : int
        Minimum mask support before fallback from AND-mask to OR-mask in
        classic mode (default: 5).

    Attributes
    ----------
    atomic_db : AtomicDatabase
        Atomic database
    vector_index : VectorIndex or None
        Vector index for fast search
    saha_solver : SahaBoltzmannSolver
        Solver for partition functions

    Examples
    --------
    >>> # Classic mode
    >>> identifier = CorrelationIdentifier(atomic_db, elements=['Fe', 'Ti', 'Cr'])
    >>> result = identifier.identify(wavelength, intensity, mode="classic")
    >>> print(f"Detected: {[e.element for e in result.detected_elements]}")

    >>> # Vector mode with pre-built index
    >>> identifier = CorrelationIdentifier(atomic_db, vector_index=index)
    >>> result = identifier.identify(wavelength, intensity, mode="vector")
    """

    def __init__(
        self,
        atomic_db: AtomicDatabase,
        vector_index=None,
        vector_embedder=None,
        library_metadata: Optional[List[Mapping[str, Any]]] = None,
        library_spectra: Optional[np.ndarray] = None,
        elements: Optional[List[str]] = None,
        resolving_power: Optional[float] = None,
        wavelength_tolerance_nm: float = 0.08,
        top_k: int = 10,
        min_confidence: float = 0.03,
        T_range_K: Tuple[float, float] = (8000, 12000),
        n_e_range_cm3: Tuple[float, float] = (3e16, 3e17),
        T_steps: int = 5,
        n_e_steps: int = 3,
        instrument_fwhm_nm: float = 0.05,
        max_lines_per_element: int = 100,
        min_line_strength: float = 1e4,
        reference_temperature: float = 10000.0,
        relative_threshold_scale: float = 1.5,
        peak_region_threshold: float = 0.15,
        peak_region_min_points: int = 5,
        use_jax_classic: bool = False,
    ):
        self.use_jax_classic = bool(use_jax_classic)
        if self.use_jax_classic and not _HAS_JAX:  # pragma: no cover
            raise ImportError(
                "use_jax_classic=True requires JAX. " "Install with: pip install jax jaxlib"
            )
        # Fail fast when a JAX path is requested without x64 (bead jbfg.1 /
        # arch review #2 candidate 2). See cflibs.core.jax_runtime.
        if self.use_jax_classic:
            from cflibs.core.jax_runtime import check_jax64bit

            check_jax64bit()
        self.atomic_db = atomic_db
        self.resolving_power = resolving_power
        self.vector_index = vector_index
        self.vector_embedder = vector_embedder
        self.library_metadata = list(library_metadata) if library_metadata is not None else None
        self.library_spectra = None if library_spectra is None else np.asarray(library_spectra)
        self.elements = elements
        self.wavelength_tolerance_nm = wavelength_tolerance_nm
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.T_range_K = T_range_K
        self.n_e_range_cm3 = n_e_range_cm3
        self.T_steps = T_steps
        self.n_e_steps = n_e_steps
        self.instrument_fwhm_nm = instrument_fwhm_nm
        self.max_lines_per_element = max_lines_per_element
        self.min_line_strength = min_line_strength
        self.reference_temperature = reference_temperature
        self.relative_threshold_scale = relative_threshold_scale
        self.peak_region_threshold = peak_region_threshold
        self.peak_region_min_points = max(1, int(peak_region_min_points))

        self.saha_solver = SahaBoltzmannSolver(atomic_db)

    def identify(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        mode: str = "auto",
        spectrum_id: Optional[str] = None,
    ) -> ElementIdentificationResult:
        """
        Identify elements from experimental spectrum.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array
        mode : str
            Identification mode: "auto", "classic", or "vector"
            (default: "auto" uses vector if available, otherwise classic)
        spectrum_id : str, optional
            Caller-supplied identifier for this spectrum.  Threaded
            through to detection-coverage log records (L2/L3/L4) so
            downstream parsing can correlate per-element coverage data
            with the source spectrum.  Identifier behaviour is
            unchanged when this is left ``None``.

        Returns
        -------
        ElementIdentificationResult
            Identification results with detected and rejected elements

        Raises
        ------
        ValueError
            If mode is "vector" but the full vector workflow is not configured
        """
        # Resolve mode
        if mode == "auto":
            mode = "vector" if self._has_vector_workflow() else "classic"

        if mode == "vector" and not self._has_vector_workflow():
            raise ValueError(
                "mode='vector' requires vector_index, vector_embedder, and library_metadata"
            )

        logger.info(f"Running correlation identifier in {mode} mode")

        # Detection-coverage tracker -- additive telemetry only.
        coverage = CoverageTracker(
            spectrum_id=spectrum_id if spectrum_id is not None else "<unset>",
            identifier_name="correlation",
        )

        # Detect experimental peaks using canonical baseline-subtracted pipeline
        experimental_peaks, _, _ = detect_peaks_auto(
            wavelength,
            intensity,
            resolving_power=self.resolving_power,
        )
        coverage.set_n_peaks(len(experimental_peaks))

        logger.info(f"Detected {len(experimental_peaks)} experimental peaks")

        # Run identification
        if mode == "classic":
            element_scores = self._identify_classic(
                wavelength, intensity, experimental_peaks, coverage=coverage
            )
        elif mode == "vector":
            element_scores = self._identify_vector(
                wavelength, intensity, experimental_peaks, coverage=coverage
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Relative score filter: require score to stand out from median
        # Only apply when comparing 3+ elements (fewer can't form a noise floor)
        non_zero_scores = [s for _, s, _, _, _ in element_scores if s > 0]
        if len(non_zero_scores) >= 3:
            median_score = np.median(non_zero_scores)
            relative_threshold = min(1.0, self.relative_threshold_scale * median_score)
        else:
            relative_threshold = 0.0

        # Build result
        detected_elements = []
        rejected_elements = []

        for element, score, confidence, matched_lines, unmatched_lines in element_scores:
            detected_flag = confidence >= self.min_confidence and confidence >= relative_threshold
            elem_id = ElementIdentification(
                element=element,
                detected=detected_flag,
                score=score,
                confidence=confidence,
                n_matched_lines=len(matched_lines),
                n_total_lines=len(matched_lines) + len(unmatched_lines),
                matched_lines=matched_lines,
                unmatched_lines=unmatched_lines,
                metadata={"correlation": score, "relative_threshold": relative_threshold},
            )

            # L4 -- per-element fingerprint pass.  Correlation's
            # "fingerprint" is the confidence vs. min_confidence /
            # relative_threshold gate.
            coverage.record_fingerprint(
                element,
                passed=bool(detected_flag),
                score=float(confidence),
                floor=float(max(self.min_confidence, relative_threshold)),
            )

            if elem_id.detected:
                detected_elements.append(elem_id)
            else:
                rejected_elements.append(elem_id)

        # Count matched peaks
        matched_peak_wavelengths = set()
        for elem in detected_elements:
            for line in elem.matched_lines:
                matched_peak_wavelengths.add(line.wavelength_exp_nm)

        n_matched_peaks = len(matched_peak_wavelengths)
        n_unmatched_peaks = len(experimental_peaks) - n_matched_peaks

        # Detection-coverage finalisation -- additive telemetry only.
        coverage.emit_summary()

        base_parameters = {
            "mode": mode,
            "wavelength_tolerance_nm": self.wavelength_tolerance_nm,
            "min_confidence": self.min_confidence,
            "peak_region_threshold": self.peak_region_threshold,
            "peak_region_min_points": float(self.peak_region_min_points),
        }

        return ElementIdentificationResult(
            detected_elements=detected_elements,
            rejected_elements=rejected_elements,
            all_elements=detected_elements + rejected_elements,
            experimental_peaks=experimental_peaks,
            n_peaks=len(experimental_peaks),
            n_matched_peaks=n_matched_peaks,
            n_unmatched_peaks=n_unmatched_peaks,
            algorithm="correlation",
            parameters=merge_coverage_into_parameters(base_parameters, coverage.build_payload()),
        )

    def _identify_classic(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple],
        coverage: Optional[CoverageTracker] = None,
    ) -> List[Tuple[str, float, float, List[IdentifiedLine], List[Transition]]]:
        """
        Classic mode: grid search over (T, n_e) with Pearson correlation.

        Parameters
        ----------
        coverage
            Optional detection-coverage tracker.  When provided, the
            method records L2 (per-element line count in window) and
            L3 (per-element peak matches) without altering matching
            behaviour.

        Returns
        -------
        List[Tuple[str, float, float, List[IdentifiedLine], List[Transition]]]
            List of (element, score, confidence, matched_lines, unmatched_lines)
        """
        wl_min, wl_max = wavelength.min(), wavelength.max()

        # Generate (T, n_e) grid
        T_grid = np.linspace(self.T_range_K[0], self.T_range_K[1], self.T_steps)
        n_e_grid = np.linspace(self.n_e_range_cm3[0], self.n_e_range_cm3[1], self.n_e_steps)

        element_scores: List[Tuple[str, float, float, List[Any], List[Any]]] = []

        elements_to_search = (
            self.elements if self.elements is not None else self.atomic_db.get_available_elements()
        )
        for element in elements_to_search:
            transitions = self._get_transitions_for_element(element, wl_min, wl_max)
            # L2 -- per-element line presence in DB (after correlation's
            # min_line_strength + max_lines_per_element selection).
            if coverage is not None:
                coverage.record_db_lines(element, len(transitions))
            if not transitions:
                logger.debug(f"No transitions for {element} in wavelength range")
                element_scores.append((element, 0.0, 0.0, [], []))
                if coverage is not None:
                    coverage.record_peak_matches(element, 0)
                continue

            element_scores.append(
                self._classic_score_element(
                    element,
                    transitions,
                    wavelength,
                    intensity,
                    peaks,
                    T_grid,
                    n_e_grid,
                    coverage,
                )
            )

        return element_scores

    def _classic_score_element(
        self,
        element: str,
        transitions: List[Transition],
        wavelength: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple],
        T_grid: np.ndarray,
        n_e_grid: np.ndarray,
        coverage: Optional[CoverageTracker],
    ) -> Tuple[str, float, float, List[IdentifiedLine], List[Transition]]:
        """Score a single element in classic mode (JAX or CPU grid path).

        Pure extraction of the per-element body of ``_identify_classic``;
        chooses the JAX fast path or CPU grid loop, matches lines, records
        L3 coverage, and returns the element-score tuple.
        """
        # Compute correlations for each (T, n_e) point.
        # Paper (Labutin et al. 2013): correlate only in peak regions, not full spectrum.
        if self.use_jax_classic:
            # JAX fast path: vectorize the whole (T, n_e) grid + line
            # sum + peak-region masked Pearson in one fused kernel.
            correlations = self._classic_correlations_jax(
                wavelength, intensity, element, transitions, T_grid, n_e_grid
            )
            best_corr = float(np.max(correlations)) if len(correlations) else 0.0
            score: Any = float(np.clip(best_corr, 0.0, 1.0))
        else:
            correlations = self._classic_grid_correlations(
                element, transitions, wavelength, intensity, T_grid, n_e_grid
            )
            # Best correlation = element score
            best_corr = max(correlations) if correlations else 0.0
            score = np.clip(best_corr, 0.0, 1.0)
        confidence = score  # Simple mapping for now

        # Match lines to experimental peaks (unchanged).
        matched_lines, unmatched_lines = self._match_lines_to_peaks(
            element, transitions, wavelength, intensity, peaks
        )
        # L3 -- per-element peak match.
        if coverage is not None:
            coverage.record_peak_matches(element, len(matched_lines))

        return (element, score, confidence, matched_lines, unmatched_lines)

    def _classic_grid_correlations(
        self,
        element: str,
        transitions: List[Transition],
        wavelength: np.ndarray,
        intensity: np.ndarray,
        T_grid: np.ndarray,
        n_e_grid: np.ndarray,
    ) -> List[float]:
        """CPU ``(T, n_e)`` grid Pearson correlations for one element.

        Mirrors the inner T-outer/n_e-inner loop of ``_identify_classic``
        exactly, returning the list of per-grid-point correlations.
        """
        correlations: List[float] = []
        for T_K in T_grid:
            T_eV = T_K * KB_EV
            for n_e in n_e_grid:
                model_spectrum = self._generate_model_spectrum(
                    intensity, element, transitions, wavelength, T_eV, n_e
                )
                correlations.append(
                    self._classic_peak_region_correlation(intensity, model_spectrum)
                )
        return correlations

    def _classic_peak_region_correlation(
        self, intensity: np.ndarray, model_spectrum: np.ndarray
    ) -> float:
        """Masked Pearson correlation between intensity and model spectrum.

        Pure extraction of the single-grid-point body: build the
        peak-region mask, then correlate only on masked points.
        """
        peak_mask = self._classic_peak_region_mask(intensity, model_spectrum)

        # Pearson correlation on peak regions only
        exp_peaks = intensity[peak_mask]
        mod_peaks = model_spectrum[peak_mask]
        if len(exp_peaks) > 2 and np.std(mod_peaks) > 1e-10 and np.std(exp_peaks) > 1e-10:
            corr, _ = pearsonr(exp_peaks, mod_peaks)
            return corr
        return 0.0

    def _classic_peak_region_mask(
        self, intensity: np.ndarray, model_spectrum: np.ndarray
    ) -> np.ndarray:
        """Build the peak-region boolean mask for masked Pearson.

        Correlate only where BOTH spectra have significant signal to
        avoid baseline-dominated correlations; fall back from AND to OR
        when the AND mask is too restrictive, or to all-ones when neither
        spectrum has dynamic range.
        """
        i_min, i_max = intensity.min(), intensity.max()
        m_min, m_max = model_spectrum.min(), model_spectrum.max()
        sigma_threshold = float(self.peak_region_threshold)
        if (i_max - i_min) > 1e-10 and (m_max - m_min) > 1e-10:
            exp_norm = (intensity - i_min) / (i_max - i_min)
            mod_norm = (model_spectrum - m_min) / (m_max - m_min)
            peak_mask = (exp_norm >= sigma_threshold) & (mod_norm >= sigma_threshold)
            # Fallback: if AND is too restrictive, use OR.
            if np.sum(peak_mask) < self.peak_region_min_points:
                peak_mask = (exp_norm >= sigma_threshold) | (mod_norm >= sigma_threshold)
        else:
            peak_mask = np.ones(len(intensity), dtype=bool)
        return peak_mask

    def _identify_vector(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple],
        coverage: Optional[CoverageTracker] = None,
    ) -> List[Tuple[str, float, float, List[IdentifiedLine], List[Transition]]]:
        """
        Vector mode: ANN search via FAISS with multi-model consensus.

        Parameters
        ----------
        coverage
            Optional detection-coverage tracker.  When provided, the
            method records L2 (per-element line count in window) and
            L3 (per-element peak matches) without altering matching
            behaviour.

        Returns
        -------
        List[Tuple[str, float, float, List[IdentifiedLine], List[Transition]]]
            List of (element, score, confidence, matched_lines, unmatched_lines)

        Notes
        -----
        The vector workflow requires:
        - ``vector_index`` with ``search()``
        - ``vector_embedder`` with ``transform()``
        - ``library_metadata`` aligned with the index rows
        """
        if not self._has_vector_workflow():
            raise ValueError(
                "Vector mode requires vector_index, vector_embedder, and library_metadata"
            )

        query_embedding = np.asarray(self.vector_embedder.transform(np.asarray(intensity)[None, :]))
        distances, indices = self.vector_index.search(query_embedding, k=self.top_k)

        (
            candidate_weights,
            candidate_counts,
            candidate_best_distance,
        ) = self._vector_accumulate_candidates(intensity, distances, indices)

        if self.elements is not None:
            elements_to_search = list(self.elements)
        else:
            elements_to_search = sorted(candidate_weights.keys())

        if not elements_to_search:
            return []

        total_weight = sum(candidate_weights.values())
        wl_min, wl_max = wavelength.min(), wavelength.max()
        element_scores: List[Tuple[str, float, float, List[Any], List[Any]]] = []

        for element in elements_to_search:
            element_scores.append(
                self._vector_score_element(
                    element,
                    candidate_weights,
                    candidate_counts,
                    candidate_best_distance,
                    total_weight,
                    wavelength,
                    intensity,
                    peaks,
                    wl_min,
                    wl_max,
                    coverage,
                )
            )

        return element_scores

    def _vector_accumulate_candidates(
        self,
        intensity: np.ndarray,
        distances: np.ndarray,
        indices: np.ndarray,
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float]]:
        """Accumulate per-element ANN consensus weights from neighbors.

        Pure extraction of the neighbor loop in ``_identify_vector``;
        returns ``(candidate_weights, candidate_counts,
        candidate_best_distance)``.
        """
        library_metadata = self.library_metadata or []
        candidate_weights: Dict[str, float] = defaultdict(float)
        candidate_counts: Dict[str, int] = defaultdict(int)
        candidate_best_distance: Dict[str, float] = {}

        for distance, neighbor_idx in zip(np.ravel(distances), np.ravel(indices)):
            idx = int(neighbor_idx)
            if idx < 0 or idx >= len(library_metadata):
                continue

            similarity = self._vector_neighbor_similarity(idx, float(distance), intensity)

            for element in self._metadata_elements(library_metadata[idx]):
                if self.elements is not None and element not in self.elements:
                    continue
                candidate_weights[element] += similarity
                candidate_counts[element] += 1
                best_distance = candidate_best_distance.get(element, float("inf"))
                candidate_best_distance[element] = min(best_distance, float(distance))

        return candidate_weights, candidate_counts, candidate_best_distance

    def _vector_neighbor_similarity(
        self, idx: int, distance: float, intensity: np.ndarray
    ) -> float:
        """Compute the (optionally spectrum-refined) similarity of a neighbor.

        Pure extraction of the per-neighbor similarity computation in
        ``_identify_vector``.
        """
        similarity = 1.0 / (1.0 + max(distance, 0.0))
        if self.library_spectra is not None and idx < len(self.library_spectra):
            candidate_spectrum = np.asarray(self.library_spectra[idx], dtype=np.float64)
            if candidate_spectrum.shape == intensity.shape:
                if np.std(candidate_spectrum) > 1e-12 and np.std(intensity) > 1e-12:
                    corr, _ = pearsonr(intensity, candidate_spectrum)
                    similarity = 0.5 * similarity + 0.5 * np.clip((corr + 1.0) / 2.0, 0.0, 1.0)
        return similarity

    def _vector_score_element(
        self,
        element: str,
        candidate_weights: Dict[str, float],
        candidate_counts: Dict[str, int],
        candidate_best_distance: Dict[str, float],
        total_weight: float,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        peaks: List[Tuple],
        wl_min: float,
        wl_max: float,
        coverage: Optional[CoverageTracker],
    ) -> Tuple[str, float, float, List[IdentifiedLine], List[Transition]]:
        """Score a single element in vector mode.

        Pure extraction of the per-element body of ``_identify_vector``;
        computes the consensus score/confidence, matches lines, applies
        the ``score <= 0`` reset, records L2/L3 coverage, and returns the
        element-score tuple.
        """
        score = candidate_weights.get(element, 0.0) / max(total_weight, 1e-12)
        count_weight = candidate_counts.get(element, 0) / max(self.top_k, 1)
        best_similarity = 0.0
        if element in candidate_best_distance:
            best_similarity = 1.0 / (1.0 + max(candidate_best_distance[element], 0.0))
        confidence = np.clip(0.4 * score + 0.4 * count_weight + 0.2 * best_similarity, 0.0, 1.0)
        transitions = self._get_transitions_for_element(element, wl_min, wl_max)
        # L2 -- per-element line presence in DB (vector path).
        if coverage is not None:
            coverage.record_db_lines(element, len(transitions))
        matched_lines, unmatched_lines = self._match_lines_to_peaks(
            element, transitions, wavelength, intensity, peaks
        )

        if score <= 0.0:
            matched_lines = []
            unmatched_lines = transitions

        # L3 -- per-element peak match (after the score==0 reset).
        if coverage is not None:
            coverage.record_peak_matches(element, len(matched_lines))

        return (element, score, confidence, matched_lines, unmatched_lines)

    def _has_vector_workflow(self) -> bool:
        """Return True when the full vector-mode workflow is configured."""
        if self.vector_index is None or self.vector_embedder is None:
            return False
        if self.library_metadata is None or len(self.library_metadata) == 0:
            return False
        return True

    def _metadata_elements(self, metadata: Mapping[str, Any]) -> List[str]:
        """Extract element symbols from a library metadata record."""
        if (
            "elements" in metadata
            and not isinstance(metadata["elements"], str)
            and isinstance(metadata["elements"], Iterable)
        ):
            return [str(element) for element in metadata["elements"]]
        if "element" in metadata:
            return [str(metadata["element"])]
        for key in ("species", "composition"):
            value = metadata.get(key)
            if isinstance(value, Mapping):
                return [
                    str(element) for element, fraction in value.items() if float(fraction) > 0.0
                ]
        return []

    def _get_transitions_for_element(
        self, element: str, wl_min: float, wl_max: float
    ) -> List[Transition]:
        """Load and rank transitions for a candidate element."""
        transitions = self.atomic_db.get_transitions(
            element, wavelength_min=wl_min, wavelength_max=wl_max
        )
        transitions = [t for t in transitions if t.A_ki * t.g_k >= self.min_line_strength]
        if len(transitions) > self.max_lines_per_element:
            kT = KB_EV * self.reference_temperature
            transitions = sorted(
                transitions,
                key=lambda t: t.A_ki * t.g_k * math.exp(-t.E_k_ev / kT),
                reverse=True,
            )
            transitions = transitions[: self.max_lines_per_element]
        return transitions

    def _generate_model_spectrum(
        self,
        intensity: np.ndarray,
        element: str,
        transitions: List[Transition],
        wavelength: np.ndarray,
        T_eV: float,
        n_e: float,
    ) -> np.ndarray:
        """
        Generate model spectrum for element at (T, n_e).

        Uses Boltzmann distribution with Gaussian stick spectrum.

        Parameters
        ----------
        intensity : np.ndarray
            Experimental intensity array for normalization
        element : str
            Element symbol
        transitions : List[Transition]
            Atomic transitions
        wavelength : np.ndarray
            Wavelength grid in nm
        T_eV : float
            Temperature in eV
        n_e : float
            Electron density in cm⁻³

        Returns
        -------
        np.ndarray
            Model spectrum intensity on wavelength grid
        """

        model_spectrum = np.zeros_like(wavelength, dtype=np.float64)

        # Compute ionization fractions using Saha equation
        total_density = 1e15  # arbitrary reference density
        try:
            stage_densities = self.saha_solver.solve_ionization_balance(
                element, T_eV, n_e, total_density
            )
        except Exception:
            stage_densities = None

        default_sigma = self.instrument_fwhm_nm / 2.355

        for trans in transitions:
            # Partition function
            U = self.saha_solver.calculate_partition_function(element, trans.ionization_stage, T_eV)

            # Ion-stage population fraction from Saha balance
            if stage_densities is not None:
                W_q = stage_densities.get(trans.ionization_stage, 1.0) / max(total_density, 1e-30)
            else:
                W_q = 1.0  # Fallback: avoid zeroing model when Saha fails

            # Boltzmann factor weighted by ionization fraction
            eps = W_q * trans.A_ki * trans.g_k * np.exp(-trans.E_k_ev / T_eV) / U

            # Per-transition sigma from RP if available, else fixed default
            if self.resolving_power:
                sigma = (trans.wavelength_nm / self.resolving_power) / 2.355
            else:
                sigma = default_sigma

            gaussian = np.exp(-0.5 * ((wavelength - trans.wavelength_nm) / sigma) ** 2)
            model_spectrum += eps * gaussian

        # Robust normalization: 95th percentile instead of max (resistant to spikes)
        exp_scale = np.percentile(intensity, 95.0)
        model_scale = np.percentile(model_spectrum, 95.0)
        if model_scale > 1e-10 and exp_scale > 1e-10:
            model_spectrum = model_spectrum * (exp_scale / model_scale)

        return model_spectrum

    def _classic_correlations_jax(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        element: str,
        transitions: List[Transition],
        T_grid: np.ndarray,
        n_e_grid: np.ndarray,
    ) -> np.ndarray:
        """JAX-vectorized counterpart of the inner ``(T, n_e)`` loop.

        Returns
        -------
        correlations : np.ndarray, shape (T_steps * n_e_steps,)
            Pearson correlation per (T, n_e) grid point, mirroring the
            CPU implementation's ordering (T outer, n_e inner).
        """
        if not _HAS_JAX:  # pragma: no cover
            raise ImportError(
                "JAX is required for _classic_correlations_jax. "
                "Install with: pip install jax jaxlib"
            )

        L = len(transitions)
        # Per-line data (numpy)
        line_wl = np.fromiter((t.wavelength_nm for t in transitions), dtype=np.float64, count=L)
        line_A_g = np.fromiter((t.A_ki * t.g_k for t in transitions), dtype=np.float64, count=L)
        line_E = np.fromiter((t.E_k_ev for t in transitions), dtype=np.float64, count=L)
        line_stage = np.fromiter((t.ionization_stage for t in transitions), dtype=np.int64, count=L)

        # Per-line sigma (matches the CPU per-transition logic).
        if self.resolving_power:
            line_sigma = (line_wl / self.resolving_power) / 2.355
        else:
            default_sigma = self.instrument_fwhm_nm / 2.355
            line_sigma = np.full(L, default_sigma, dtype=np.float64)

        # Build the flat (T, n_e) grid in T-outer, n_e-inner order so the
        # output matches the CPU list order.
        G = T_grid.shape[0] * n_e_grid.shape[0]
        T_eV_grid = np.empty(G, dtype=np.float64)
        n_e_grid_flat = np.empty(G, dtype=np.float64)
        for ti, T_K in enumerate(T_grid):
            for ni, n_e in enumerate(n_e_grid):
                g_idx = ti * n_e_grid.shape[0] + ni
                T_eV_grid[g_idx] = T_K * KB_EV
                n_e_grid_flat[g_idx] = n_e

        # Per-grid-point Saha balance + partition functions. We compute
        # these in Python to keep parity with the CPU path (SahaBoltzmann-
        # Solver isn't JAX-native). Caching avoids redundant work.
        total_density = 1e15
        line_U = np.ones((G, L), dtype=np.float64)
        line_W_q = np.ones((G, L), dtype=np.float64)

        # Cache partition-function evaluations per (stage, T) to avoid
        # redundant Solver calls when multiple lines share a stage.
        stages = np.unique(line_stage)
        for g_idx in range(G):
            T_eV = float(T_eV_grid[g_idx])
            n_e = float(n_e_grid_flat[g_idx])
            try:
                stage_densities = self.saha_solver.solve_ionization_balance(
                    element, T_eV, n_e, total_density
                )
            except Exception:
                stage_densities = None
            U_by_stage = {
                int(stage): self.saha_solver.calculate_partition_function(element, int(stage), T_eV)
                for stage in stages
            }
            for l_idx in range(L):
                stage = int(line_stage[l_idx])
                line_U[g_idx, l_idx] = max(float(U_by_stage[stage]), 1e-30)
                if stage_densities is not None:
                    line_W_q[g_idx, l_idx] = stage_densities.get(stage, 1.0) / max(
                        total_density, 1e-30
                    )
                else:
                    line_W_q[g_idx, l_idx] = 1.0

        exp_scale = float(np.percentile(intensity, 95.0))

        # Run the fused JAX kernel.
        corr = _jax_model_grid_correlations(
            jnp.asarray(wavelength, dtype=jnp.float64),
            jnp.asarray(intensity, dtype=jnp.float64),
            jnp.asarray(line_wl, dtype=jnp.float64),
            jnp.asarray(line_A_g, dtype=jnp.float64),
            jnp.asarray(line_E, dtype=jnp.float64),
            jnp.asarray(line_U, dtype=jnp.float64),
            jnp.asarray(line_W_q, dtype=jnp.float64),
            jnp.asarray(line_sigma, dtype=jnp.float64),
            jnp.asarray(T_eV_grid, dtype=jnp.float64),
            jnp.asarray(exp_scale, dtype=jnp.float64),
            float(self.peak_region_threshold),
            int(self.peak_region_min_points),
        )
        return np.asarray(corr)

    def _match_lines_to_peaks(
        self,
        element: str,
        transitions: List[Transition],
        wavelength: np.ndarray,
        intensity: np.ndarray,
        peaks: Optional[List[Tuple]] = None,
    ) -> Tuple[List[IdentifiedLine], List[Transition]]:
        """
        Match theoretical transitions to experimental peaks.

        Uses canonical peak detection and one-to-one greedy matching
        (closest distance first; each peak and transition used at most once).

        Parameters
        ----------
        element : str
            Element symbol
        transitions : List[Transition]
            Theoretical transitions
        wavelength : np.ndarray
            Experimental wavelength in nm
        intensity : np.ndarray
            Experimental intensity
        peaks : List[Tuple], optional
            Pre-detected peaks. If None, peaks are detected fresh.

        Returns
        -------
        matched_lines : List[IdentifiedLine]
            Matched lines
        unmatched_lines : List[Transition]
            Transitions with no experimental match
        """
        if peaks is None:
            peaks, _, _ = detect_peaks_auto(
                wavelength, intensity, resolving_power=self.resolving_power
            )
        if not peaks:
            return [], list(transitions)

        peak_wavelengths = np.array([p[1] for p in peaks])
        peak_intensities = np.array([intensity[p[0]] for p in peaks])

        candidates = self._match_build_candidates(transitions, peak_wavelengths)

        matched_lines, claimed_trans = self._match_greedy_assign(
            candidates, transitions, peak_wavelengths, peak_intensities, element
        )

        unmatched_lines = [
            transitions[i] for i in range(len(transitions)) if i not in claimed_trans
        ]

        return matched_lines, unmatched_lines

    def _match_line_tolerance(self, trans: Transition) -> float:
        """Per-line matching tolerance in nm.

        Stark-aware per-line tolerance — only WIDENS the config default,
        never tightens. Picking ``max(config, helper)`` is deliberate:
        naive replacement with the helper value would tighten tolerance
        for short-lambda lines (e.g. 200 nm at R=10000 → 0.02 nm, vs
        the 0.1 default), tanking correlation's already-low recall
        (baseline 0.10). Stark-broadened lines (omega_stark > 0) get
        genuinely wider windows; non-Stark lines stay at the config
        default. PR #133 added the helper, PR #151 wired it into alias,
        PR #153 into comb. See CF-LIBS-improved-orej.
        """
        if self.resolving_power:
            return max(
                self.wavelength_tolerance_nm,
                get_wavelength_tolerance(
                    trans.wavelength_nm,
                    transition=trans,
                    resolving_power=self.resolving_power,
                    fallback=self.wavelength_tolerance_nm,
                    T_K=self.reference_temperature,
                ),
            )
        return self.wavelength_tolerance_nm

    def _match_build_candidates(
        self, transitions: List[Transition], peak_wavelengths: np.ndarray
    ) -> List[Tuple[float, int, int]]:
        """Build candidate ``(distance, peak_idx, trans_idx)`` matches.

        Pure extraction of the candidate-building loop in
        ``_match_lines_to_peaks``.
        """
        candidates: List[Tuple[float, int, int]] = []
        for t_idx, trans in enumerate(transitions):
            distances = np.abs(peak_wavelengths - trans.wavelength_nm)
            tol = self._match_line_tolerance(trans)
            for p_idx in range(len(peak_wavelengths)):
                if distances[p_idx] <= tol:
                    candidates.append((distances[p_idx], p_idx, t_idx))
        return candidates

    def _match_greedy_assign(
        self,
        candidates: List[Tuple[float, int, int]],
        transitions: List[Transition],
        peak_wavelengths: np.ndarray,
        peak_intensities: np.ndarray,
        element: str,
    ) -> Tuple[List[IdentifiedLine], set]:
        """Greedy one-to-one assignment of candidate matches.

        Pure extraction of the greedy-assignment loop in
        ``_match_lines_to_peaks``; returns ``(matched_lines,
        claimed_trans)``.
        """
        # Greedy one-to-one: sort by distance, assign first-come
        candidates.sort(key=lambda c: c[0])
        claimed_peaks: set = set()
        claimed_trans: set = set()

        matched_lines = []
        for _dist, p_idx, t_idx in candidates:
            if p_idx in claimed_peaks or t_idx in claimed_trans:
                continue
            claimed_peaks.add(p_idx)
            claimed_trans.add(t_idx)
            trans = transitions[t_idx]
            matched_lines.append(
                IdentifiedLine(
                    wavelength_exp_nm=float(peak_wavelengths[p_idx]),
                    wavelength_th_nm=trans.wavelength_nm,
                    element=element,
                    ionization_stage=trans.ionization_stage,
                    intensity_exp=float(peak_intensities[p_idx]),
                    emissivity_th=0.0,
                    transition=trans,
                    correlation=0.0,
                    is_interfered=False,
                    interfering_elements=[],
                )
            )

        return matched_lines, claimed_trans
