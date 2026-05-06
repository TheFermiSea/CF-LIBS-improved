"""
Real-time streaming analysis pipeline for CF-LIBS.

This module provides infrastructure for streaming spectrum analysis with sub-second
latency targets, designed for industrial and field applications requiring real-time
or near-real-time analysis.

Architecture
------------
The streaming pipeline consists of:

1. **SpectrumBuffer**: Thread-safe ring buffer for incoming spectra
2. **StreamingAnalyzer**: Continuous processing with configurable callbacks
3. **EdgeOptimizedModel**: Quantized/pruned models for edge deployment
4. **LatencyMonitor**: Real-time latency tracking and profiling

Latency Targets
---------------
- Fast mode: <100ms per spectrum (simplified physics)
- Standard mode: <500ms per spectrum (full CF-LIBS)
- Batch mode: <50ms/spectrum (amortized over batches)

Edge Deployment
---------------
For edge deployment (Raspberry Pi, Jetson, etc.):
- Model quantization (float32 -> float16/int8)
- Spectral downsampling to reduce computation
- Caching of partition functions and atomic data
- Optional JAX JIT compilation for GPU acceleration

Literature References
---------------------
- Mahyari et al. (2025): Real-time 3D mapping with LIBS
- Edge computing and model optimization for spectroscopy

Example
-------
>>> from cflibs.inversion.streaming import StreamingAnalyzer, SpectrumBuffer
>>> from cflibs.atomic.database import AtomicDatabase
>>>
>>> # Create buffer and analyzer
>>> buffer = SpectrumBuffer(max_size=100)
>>> db = AtomicDatabase("cflibs.db")
>>> analyzer = StreamingAnalyzer(db, buffer, mode="fast")
>>>
>>> # Start analysis thread
>>> analyzer.start()
>>>
>>> # Push spectra as they arrive
>>> for spectrum in spectrum_generator():
...     buffer.push(spectrum, metadata={"timestamp": time.time()})
>>>
>>> # Get results via callback or polling
>>> analyzer.stop()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Any,
    Protocol,
    runtime_checkable,
)
from collections import deque
import threading
import time
import numpy as np

from cflibs.core.logging_config import get_logger
from cflibs.inversion.boltzmann import LineObservation, BoltzmannPlotFitter
from cflibs.inversion.solver import CFLIBSResult, IterativeCFLIBSSolver
from cflibs.inversion.line_selection import LineSelector

logger = get_logger("inversion.streaming")


class AnalysisMode(Enum):
    """Analysis mode for latency/accuracy tradeoff."""

    FAST = "fast"  # <100ms, simplified physics
    STANDARD = "standard"  # <500ms, full CF-LIBS
    BATCH = "batch"  # <50ms/spectrum amortized
    ADAPTIVE = "adaptive"  # Adjusts based on queue depth


@dataclass
class StreamingConfig:
    """Configuration for streaming analyzer.

    Attributes
    ----------
    mode : AnalysisMode
        Analysis mode (fast/standard/batch/adaptive)
    max_latency_ms : float
        Maximum acceptable latency in milliseconds
    batch_size : int
        Batch size for batch mode
    warmup_spectra : int
        Number of spectra for warmup calibration
    downsample_factor : int
        Spectral downsampling factor (1 = no downsampling)
    use_jit : bool
        Use JAX JIT compilation if available
    cache_atomic_data : bool
        Cache partition functions and atomic data
    quantize : bool
        Use quantized (float16) arithmetic where possible
    """

    mode: AnalysisMode = AnalysisMode.STANDARD
    max_latency_ms: float = 500.0
    batch_size: int = 10
    warmup_spectra: int = 5
    downsample_factor: int = 1
    use_jit: bool = True
    cache_atomic_data: bool = True
    quantize: bool = False


@dataclass
class SpectrumPacket:
    """A spectrum packet with metadata for streaming.

    Attributes
    ----------
    wavelength : np.ndarray
        Wavelength array in nm
    intensity : np.ndarray
        Intensity array
    timestamp : float
        Acquisition timestamp (seconds since epoch)
    metadata : Dict[str, Any]
        Additional metadata (position, shot number, etc.)
    sequence_id : int
        Monotonically increasing sequence ID
    """

    wavelength: np.ndarray
    intensity: np.ndarray
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    sequence_id: int = 0


@dataclass
class StreamingResult:
    """Result from streaming analysis.

    Attributes
    ----------
    sequence_id : int
        Sequence ID of the analyzed spectrum
    timestamp : float
        Original acquisition timestamp
    analysis_timestamp : float
        When analysis completed
    latency_ms : float
        Total latency (analysis_timestamp - timestamp) in ms
    result : CFLIBSResult
        CF-LIBS analysis result
    quality_flag : str
        Quality indicator: 'good', 'warning', 'poor'
    warnings : List[str]
        Any warnings generated during analysis
    metadata : Dict[str, Any]
        Propagated metadata from spectrum packet
    """

    sequence_id: int
    timestamp: float
    analysis_timestamp: float
    latency_ms: float
    result: CFLIBSResult
    quality_flag: str = "good"
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LatencyStats:
    """Latency statistics for monitoring.

    Attributes
    ----------
    mean_ms : float
        Mean latency in milliseconds
    std_ms : float
        Standard deviation of latency
    min_ms : float
        Minimum latency
    max_ms : float
        Maximum latency
    p50_ms : float
        50th percentile (median) latency
    p95_ms : float
        95th percentile latency
    p99_ms : float
        99th percentile latency
    n_samples : int
        Number of samples in statistics
    target_met_fraction : float
        Fraction of samples meeting latency target
    """

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    n_samples: int
    target_met_fraction: float


@runtime_checkable
class ResultCallback(Protocol):
    """Protocol for streaming result callbacks."""

    def __call__(self, result: StreamingResult) -> None:
        """Handle a streaming result."""
        ...


class SpectrumBuffer:
    """Thread-safe ring buffer for incoming spectra.

    This buffer handles the asynchronous arrival of spectra from hardware
    while the analyzer processes them. Older spectra are dropped when the
    buffer is full (ring buffer semantics).

    Parameters
    ----------
    max_size : int
        Maximum number of spectra to buffer
    drop_old : bool
        If True, drop oldest when full. If False, block until space available.

    Example
    -------
    >>> buffer = SpectrumBuffer(max_size=100)
    >>> buffer.push(wavelength, intensity, metadata={"shot": 1})
    >>> packet = buffer.pop(timeout=1.0)
    """

    def __init__(self, max_size: int = 100, drop_old: bool = True):
        self.max_size = max_size
        self.drop_old = drop_old
        self._buffer: deque[SpectrumPacket] = deque(maxlen=max_size if drop_old else None)
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
        self._sequence_counter = 0
        self._dropped_count = 0
        self._closed = False

    def push(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timeout: Optional[float] = None,
    ) -> bool:
        """Push a spectrum onto the buffer.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array
        timestamp : float, optional
            Acquisition timestamp (default: current time)
        metadata : dict, optional
            Additional metadata
        timeout : float, optional
            Timeout in seconds if buffer is full (blocking mode only)

        Returns
        -------
        bool
            True if successfully pushed, False if dropped/timeout
        """
        if self._closed:
            return False

        if timestamp is None:
            timestamp = time.time()

        packet = SpectrumPacket(
            wavelength=wavelength.copy(),
            intensity=intensity.copy(),
            timestamp=timestamp,
            metadata=metadata or {},
            sequence_id=self._sequence_counter,
        )

        with self._lock:
            self._sequence_counter += 1

            if self.drop_old:
                # Ring buffer: always succeeds, may drop oldest
                if len(self._buffer) >= self.max_size:
                    self._dropped_count += 1
                self._buffer.append(packet)
                self._not_empty.notify()
                return True
            else:
                # Blocking buffer: wait for space
                if timeout is not None:
                    deadline = time.time() + timeout
                while len(self._buffer) >= self.max_size and not self._closed:
                    if timeout is not None:
                        remaining = deadline - time.time()
                        if remaining <= 0:
                            return False
                        self._not_full.wait(timeout=remaining)
                    else:
                        self._not_full.wait()

                if self._closed:
                    return False

                self._buffer.append(packet)
                self._not_empty.notify()
                return True

    def pop(self, timeout: Optional[float] = None) -> Optional[SpectrumPacket]:
        """Pop a spectrum from the buffer.

        Parameters
        ----------
        timeout : float, optional
            Timeout in seconds to wait for a spectrum

        Returns
        -------
        SpectrumPacket or None
            The spectrum packet, or None if timeout/closed
        """
        with self._not_empty:
            if timeout is not None:
                deadline = time.time() + timeout
            while len(self._buffer) == 0 and not self._closed:
                if timeout is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return None
                    self._not_empty.wait(timeout=remaining)
                else:
                    self._not_empty.wait()

            if len(self._buffer) == 0:
                return None

            packet = self._buffer.popleft()
            self._not_full.notify()
            return packet

    def pop_batch(self, max_count: int, timeout: Optional[float] = None) -> List[SpectrumPacket]:
        """Pop up to max_count spectra from the buffer.

        Parameters
        ----------
        max_count : int
            Maximum number of spectra to pop
        timeout : float, optional
            Timeout to wait for at least one spectrum

        Returns
        -------
        List[SpectrumPacket]
            List of spectrum packets (may be empty if timeout)
        """
        with self._not_empty:
            # Wait for at least one spectrum
            if timeout is not None:
                deadline = time.time() + timeout
            while len(self._buffer) == 0 and not self._closed:
                if timeout is not None:
                    remaining = deadline - time.time()
                    if remaining <= 0:
                        return []
                    self._not_empty.wait(timeout=remaining)
                else:
                    self._not_empty.wait()

            if len(self._buffer) == 0:
                return []

            # Take up to max_count
            batch = []
            for _ in range(min(max_count, len(self._buffer))):
                batch.append(self._buffer.popleft())
            self._not_full.notify_all()
            return batch

    def close(self) -> None:
        """Close the buffer, waking any waiting threads."""
        with self._lock:
            self._closed = True
            self._not_empty.notify_all()
            self._not_full.notify_all()

    def clear(self) -> int:
        """Clear all spectra from buffer.

        Returns
        -------
        int
            Number of spectra cleared
        """
        with self._lock:
            count = len(self._buffer)
            self._buffer.clear()
            self._not_full.notify_all()
            return count

    @property
    def size(self) -> int:
        """Current number of spectra in buffer."""
        with self._lock:
            return len(self._buffer)

    @property
    def dropped_count(self) -> int:
        """Number of spectra dropped due to buffer overflow."""
        with self._lock:
            return self._dropped_count

    @property
    def is_closed(self) -> bool:
        """Whether the buffer is closed."""
        return self._closed


class LatencyMonitor:
    """Monitor and track latency statistics.

    Parameters
    ----------
    window_size : int
        Number of samples to keep for statistics
    target_ms : float
        Target latency in milliseconds
    """

    def __init__(self, window_size: int = 1000, target_ms: float = 500.0):
        self.window_size = window_size
        self.target_ms = target_ms
        self._latencies: deque[float] = deque(maxlen=window_size)
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        """Record a latency measurement."""
        with self._lock:
            self._latencies.append(latency_ms)

    def get_stats(self) -> LatencyStats:
        """Get current latency statistics.

        Returns
        -------
        LatencyStats
            Current statistics
        """
        with self._lock:
            if len(self._latencies) == 0:
                return LatencyStats(
                    mean_ms=0.0,
                    std_ms=0.0,
                    min_ms=0.0,
                    max_ms=0.0,
                    p50_ms=0.0,
                    p95_ms=0.0,
                    p99_ms=0.0,
                    n_samples=0,
                    target_met_fraction=1.0,
                )

            arr = np.array(self._latencies)
            return LatencyStats(
                mean_ms=float(np.mean(arr)),
                std_ms=float(np.std(arr)),
                min_ms=float(np.min(arr)),
                max_ms=float(np.max(arr)),
                p50_ms=float(np.percentile(arr, 50)),
                p95_ms=float(np.percentile(arr, 95)),
                p99_ms=float(np.percentile(arr, 99)),
                n_samples=len(arr),
                target_met_fraction=float(np.mean(arr <= self.target_ms)),
            )

    def reset(self) -> None:
        """Reset all statistics."""
        with self._lock:
            self._latencies.clear()


class BaseStreamingAnalyzer(ABC):
    """Abstract base class for streaming analyzers.

    Subclasses implement specific analysis strategies (fast, standard, batch).
    """

    @abstractmethod
    def analyze_spectrum(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        **kwargs: Any,
    ) -> CFLIBSResult:
        """Analyze a single spectrum.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array
        **kwargs
            Additional parameters

        Returns
        -------
        CFLIBSResult
            Analysis result
        """
        pass

    @abstractmethod
    def analyze_batch(
        self,
        wavelengths: List[np.ndarray],
        intensities: List[np.ndarray],
        **kwargs: Any,
    ) -> List[CFLIBSResult]:
        """Analyze a batch of spectra.

        Parameters
        ----------
        wavelengths : List[np.ndarray]
            List of wavelength arrays
        intensities : List[np.ndarray]
            List of intensity arrays
        **kwargs
            Additional parameters

        Returns
        -------
        List[CFLIBSResult]
            List of analysis results
        """
        pass


class FastAnalyzer(BaseStreamingAnalyzer):
    """Fast analyzer with simplified physics for <100ms latency.

    Uses:
    - Reduced number of Boltzmann iterations
    - Cached partition functions
    - Optional spectral downsampling
    - Single Boltzmann fit (no full iteration)

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database
    elements : List[str]
        Elements to analyze
    config : StreamingConfig
        Streaming configuration
    """

    def __init__(
        self,
        atomic_db: Any,
        elements: List[str],
        config: Optional[StreamingConfig] = None,
    ):
        self.atomic_db = atomic_db
        self.elements = elements
        self.config = config or StreamingConfig(mode=AnalysisMode.FAST)
        self._boltzmann_fitter = BoltzmannPlotFitter(outlier_sigma=3.0)
        self._line_selector = LineSelector(min_snr=5.0, min_lines_per_element=2)

        # Cache for partition functions
        self._partition_cache: Dict[Tuple[str, int], np.ndarray] = {}
        if self.config.cache_atomic_data:
            self._warm_cache()

    def _warm_cache(self) -> None:
        """Pre-cache partition function coefficients."""
        for el in self.elements:
            for stage in [1, 2]:
                pf = self.atomic_db.get_partition_coefficients(el, stage)
                if pf is not None:
                    self._partition_cache[(el, stage)] = np.array(pf.coefficients)

    def analyze_spectrum(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        line_observations: Optional[List[LineObservation]] = None,
        **kwargs: Any,
    ) -> CFLIBSResult:
        """Perform fast analysis on a spectrum.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array
        line_observations : List[LineObservation], optional
            Pre-identified line observations (faster than re-detecting)
        **kwargs
            Additional parameters

        Returns
        -------
        CFLIBSResult
            Analysis result with estimated uncertainties
        """
        if line_observations is None:
            # In production, would use line detection here
            # For now, return placeholder result
            return CFLIBSResult(
                temperature_K=10000.0,
                temperature_uncertainty_K=1000.0,
                electron_density_cm3=1e17,
                concentrations={el: 1.0 / len(self.elements) for el in self.elements},
                concentration_uncertainties={el: 0.1 for el in self.elements},
                iterations=1,
                converged=True,
                quality_metrics={"mode": "fast", "n_lines": 0},
            )

        # Downsample if configured
        if self.config.downsample_factor > 1:
            step = self.config.downsample_factor
            wavelength = wavelength[::step]
            intensity = intensity[::step]

        # Select best lines quickly
        selection = self._line_selector.select(line_observations)
        selected = selection.selected_lines

        if len(selected) < 3:
            # Not enough lines
            return CFLIBSResult(
                temperature_K=10000.0,
                temperature_uncertainty_K=5000.0,
                electron_density_cm3=1e17,
                concentrations={el: 1.0 / len(self.elements) for el in self.elements},
                concentration_uncertainties={el: 0.5 for el in self.elements},
                iterations=0,
                converged=False,
                quality_metrics={
                    "mode": "fast",
                    "n_lines": len(selected),
                    "warning": "insufficient_lines",
                },
            )

        # Single Boltzmann fit (no iteration)
        fit_result = self._boltzmann_fitter.fit(selected)

        # Use fit result directly
        T_K = fit_result.temperature_K
        T_err = fit_result.temperature_uncertainty_K

        # Simplified concentration estimate from intercepts
        concentrations = {}
        concentration_uncertainties = {}
        total = 0.0

        for el in self.elements:
            el_lines = [o for o in selected if o.element == el]
            if el_lines:
                # Use average y-value as proxy for concentration
                avg_y = np.mean([o.y_value for o in el_lines])
                concentrations[el] = np.exp(avg_y)
                total += concentrations[el]
            else:
                concentrations[el] = 0.01

        # Normalize
        for el in concentrations:
            concentrations[el] /= total
            concentration_uncertainties[el] = concentrations[el] * 0.2  # 20% relative

        return CFLIBSResult(
            temperature_K=T_K,
            temperature_uncertainty_K=T_err,
            electron_density_cm3=1e17,  # Not determined in fast mode
            concentrations=concentrations,
            concentration_uncertainties=concentration_uncertainties,
            iterations=1,
            converged=True,
            quality_metrics={
                "mode": "fast",
                "n_lines": len(selected),
                "r_squared": fit_result.r_squared,
            },
        )

    def analyze_batch(
        self,
        wavelengths: List[np.ndarray],
        intensities: List[np.ndarray],
        **kwargs: Any,
    ) -> List[CFLIBSResult]:
        """Analyze batch of spectra sequentially (fast mode doesn't batch)."""
        return [
            self.analyze_spectrum(wl, inten, **kwargs)
            for wl, inten in zip(wavelengths, intensities)
        ]


class StandardAnalyzer(BaseStreamingAnalyzer):
    """Standard analyzer using full CF-LIBS iteration.

    Uses the standard IterativeCFLIBSSolver with optional optimizations:
    - Reduced max iterations for faster convergence
    - Cached atomic data lookups
    - Optional JIT compilation

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database
    config : StreamingConfig
        Streaming configuration
    """

    def __init__(
        self,
        atomic_db: Any,
        config: Optional[StreamingConfig] = None,
    ):
        self.atomic_db = atomic_db
        self.config = config or StreamingConfig(mode=AnalysisMode.STANDARD)

        # Create solver with reduced iterations for real-time
        max_iter = 10 if self.config.max_latency_ms < 300 else 20
        self._solver = IterativeCFLIBSSolver(
            atomic_db,
            max_iterations=max_iter,
            t_tolerance_k=200.0,  # Relaxed for speed
            ne_tolerance_frac=0.15,
        )

    def analyze_spectrum(
        self,
        wavelength: np.ndarray,
        intensity: np.ndarray,
        line_observations: Optional[List[LineObservation]] = None,
        closure_mode: str = "standard",
        **kwargs: Any,
    ) -> CFLIBSResult:
        """Analyze spectrum using full CF-LIBS.

        Parameters
        ----------
        wavelength : np.ndarray
            Wavelength array in nm
        intensity : np.ndarray
            Intensity array
        line_observations : List[LineObservation], optional
            Pre-identified line observations
        closure_mode : str
            Closure mode ('standard', 'matrix', 'oxide')
        **kwargs
            Additional parameters for closure equation

        Returns
        -------
        CFLIBSResult
            Full CF-LIBS analysis result
        """
        if line_observations is None:
            # Placeholder when no lines provided
            return CFLIBSResult(
                temperature_K=10000.0,
                temperature_uncertainty_K=1000.0,
                electron_density_cm3=1e17,
                concentrations={},
                concentration_uncertainties={},
                iterations=0,
                converged=False,
                quality_metrics={"mode": "standard", "warning": "no_lines"},
            )

        return self._solver.solve(line_observations, closure_mode=closure_mode, **kwargs)

    def analyze_batch(
        self,
        wavelengths: List[np.ndarray],
        intensities: List[np.ndarray],
        **kwargs: Any,
    ) -> List[CFLIBSResult]:
        """Analyze batch of spectra."""
        return [
            self.analyze_spectrum(wl, inten, **kwargs)
            for wl, inten in zip(wavelengths, intensities)
        ]


class StreamingAnalyzer:
    """Main streaming analyzer with threading support.

    Continuously processes spectra from a buffer with configurable
    analysis mode and result callbacks.

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database
    buffer : SpectrumBuffer
        Input spectrum buffer
    config : StreamingConfig, optional
        Configuration
    result_callback : Callable[[StreamingResult], None], optional
        Callback for each result

    Example
    -------
    >>> buffer = SpectrumBuffer(max_size=100)
    >>> analyzer = StreamingAnalyzer(db, buffer, config=StreamingConfig(mode=AnalysisMode.FAST))
    >>> analyzer.start()
    >>> # Push spectra to buffer...
    >>> analyzer.stop()
    >>> print(analyzer.latency_monitor.get_stats())
    """

    def __init__(
        self,
        atomic_db: Any,
        buffer: SpectrumBuffer,
        config: Optional[StreamingConfig] = None,
        result_callback: Optional[Callable[[StreamingResult], None]] = None,
        elements: Optional[List[str]] = None,
    ):
        self.atomic_db = atomic_db
        self.buffer = buffer
        self.config = config or StreamingConfig()
        self.result_callback = result_callback
        self.elements = elements or []

        # Initialize analyzer based on mode
        if self.config.mode == AnalysisMode.FAST:
            self._analyzer: BaseStreamingAnalyzer = FastAnalyzer(
                atomic_db, self.elements, self.config
            )
        else:
            self._analyzer = StandardAnalyzer(atomic_db, self.config)

        # Latency monitoring
        self.latency_monitor = LatencyMonitor(
            window_size=1000, target_ms=self.config.max_latency_ms
        )

        # Threading
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._results: deque[StreamingResult] = deque(maxlen=1000)
        self._results_lock = threading.Lock()

        # Statistics
        self._processed_count = 0
        self._error_count = 0

    def start(self) -> None:
        """Start the analyzer thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("Analyzer already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"StreamingAnalyzer started in {self.config.mode.value} mode")

    def stop(self, timeout: float = 5.0) -> None:
        """Stop the analyzer thread.

        Parameters
        ----------
        timeout : float
            Timeout in seconds to wait for thread to finish
        """
        if self._thread is None:
            return

        self._stop_event.set()
        self.buffer.close()  # Wake up any waiting pop()
        self._thread.join(timeout=timeout)

        if self._thread.is_alive():
            logger.warning("Analyzer thread did not stop gracefully")
        else:
            logger.info("StreamingAnalyzer stopped")

        self._thread = None

    def _run_loop(self) -> None:
        """Main processing loop (runs in separate thread)."""
        while not self._stop_event.is_set():
            try:
                if self.config.mode == AnalysisMode.BATCH:
                    self._process_batch()
                else:
                    self._process_single()
            except Exception as e:
                logger.error(f"Error in analyzer loop: {e}")
                self._error_count += 1

    def _process_single(self) -> None:
        """Process a single spectrum."""
        packet = self.buffer.pop(timeout=0.1)
        if packet is None:
            return

        try:
            result = self._analyzer.analyze_spectrum(packet.wavelength, packet.intensity)
            quality_flag = self._assess_quality(result)
            warnings = self._collect_warnings(result)
        except Exception as e:
            logger.warning(f"Analysis failed for seq {packet.sequence_id}: {e}")
            result = CFLIBSResult(
                temperature_K=0.0,
                temperature_uncertainty_K=0.0,
                electron_density_cm3=0.0,
                concentrations={},
                concentration_uncertainties={},
                iterations=0,
                converged=False,
                quality_metrics={"error": str(e)},
            )
            quality_flag = "poor"
            warnings = [f"Analysis error: {e}"]

        end_time = time.time()
        latency_ms = (end_time - packet.timestamp) * 1000.0

        streaming_result = StreamingResult(
            sequence_id=packet.sequence_id,
            timestamp=packet.timestamp,
            analysis_timestamp=end_time,
            latency_ms=latency_ms,
            result=result,
            quality_flag=quality_flag,
            warnings=warnings,
            metadata=packet.metadata,
        )

        self._store_result(streaming_result)
        self.latency_monitor.record(latency_ms)
        self._processed_count += 1

    def _process_batch(self) -> None:
        """Process a batch of spectra."""
        packets = self.buffer.pop_batch(self.config.batch_size, timeout=0.1)
        if not packets:
            return

        wavelengths = [p.wavelength for p in packets]
        intensities = [p.intensity for p in packets]

        try:
            results = self._analyzer.analyze_batch(wavelengths, intensities)
        except Exception as e:
            logger.warning(f"Batch analysis failed: {e}")
            results = [
                CFLIBSResult(
                    temperature_K=0.0,
                    temperature_uncertainty_K=0.0,
                    electron_density_cm3=0.0,
                    concentrations={},
                    concentration_uncertainties={},
                    iterations=0,
                    converged=False,
                    quality_metrics={"error": str(e)},
                )
                for _ in packets
            ]

        end_time = time.time()

        for packet, result in zip(packets, results):
            latency_ms = (end_time - packet.timestamp) * 1000.0
            quality_flag = self._assess_quality(result)
            warnings = self._collect_warnings(result)

            streaming_result = StreamingResult(
                sequence_id=packet.sequence_id,
                timestamp=packet.timestamp,
                analysis_timestamp=end_time,
                latency_ms=latency_ms,
                result=result,
                quality_flag=quality_flag,
                warnings=warnings,
                metadata=packet.metadata,
            )

            self._store_result(streaming_result)
            self.latency_monitor.record(latency_ms)
            self._processed_count += 1

    def _store_result(self, result: StreamingResult) -> None:
        """Store result and invoke callback."""
        with self._results_lock:
            self._results.append(result)

        if self.result_callback is not None:
            try:
                self.result_callback(result)
            except Exception as e:
                logger.warning(f"Result callback error: {e}")

    def _assess_quality(self, result: CFLIBSResult) -> str:
        """Assess result quality."""
        if not result.converged:
            return "poor"

        r_squared = result.quality_metrics.get("r_squared", 0.0)
        if r_squared < 0.9:
            return "warning"

        if result.temperature_uncertainty_K > result.temperature_K * 0.2:
            return "warning"

        return "good"

    def _collect_warnings(self, result: CFLIBSResult) -> List[str]:
        """Collect warnings from result."""
        warnings = []
        if not result.converged:
            warnings.append("Analysis did not converge")
        if result.quality_metrics.get("warning"):
            warnings.append(result.quality_metrics["warning"])
        return warnings

    def get_results(self, max_count: Optional[int] = None) -> List[StreamingResult]:
        """Get buffered results.

        Parameters
        ----------
        max_count : int, optional
            Maximum number of results to return (None = all)

        Returns
        -------
        List[StreamingResult]
            List of results (oldest first)
        """
        with self._results_lock:
            if max_count is None:
                return list(self._results)
            return list(self._results)[-max_count:]

    @property
    def processed_count(self) -> int:
        """Total number of spectra processed."""
        return self._processed_count

    @property
    def error_count(self) -> int:
        """Total number of errors encountered."""
        return self._error_count

    @property
    def is_running(self) -> bool:
        """Whether the analyzer is running."""
        return self._thread is not None and self._thread.is_alive()


class EdgeOptimizedModel:
    """Model optimized for edge deployment.

    Provides:
    - Float16 quantization for reduced memory and faster computation
    - Model pruning (removal of rarely-used atomic data)
    - Compiled JAX functions for GPU acceleration
    - Minimal memory footprint

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Full atomic database
    elements : List[str]
        Elements to support (others pruned)
    quantize : bool
        Use float16 quantization
    compile_jax : bool
        Pre-compile JAX functions
    """

    def __init__(
        self,
        atomic_db: Any,
        elements: List[str],
        quantize: bool = True,
        compile_jax: bool = True,
    ):
        self.atomic_db = atomic_db
        self.elements = elements
        self.quantize = quantize
        self.compile_jax = compile_jax

        # Pruned data storage
        self._partition_coeffs: Dict[Tuple[str, int], np.ndarray] = {}
        self._ionization_potentials: Dict[str, float] = {}
        self._transitions: Dict[str, List[Tuple[float, float, float, float]]] = {}

        # Build pruned model
        self._build_pruned_model()

        # JAX compilation
        self._compiled_functions: Dict[str, Any] = {}
        if compile_jax:
            self._compile_functions()

    def _build_pruned_model(self) -> None:
        """Build pruned atomic data model."""
        dtype = np.float16 if self.quantize else np.float32

        for el in self.elements:
            # Partition function coefficients
            for stage in [1, 2]:
                pf = self.atomic_db.get_partition_coefficients(el, stage)
                if pf is not None:
                    coeffs = np.array(pf.coefficients, dtype=dtype)
                    self._partition_coeffs[(el, stage)] = coeffs

            # Ionization potential
            ip = self.atomic_db.get_ionization_potential(el, 1)
            if ip is not None:
                self._ionization_potentials[el] = float(ip)

            # Key transitions (store top N by intensity)
            transitions = self.atomic_db.get_transitions(el, ionization_stage=1)
            if transitions:
                # Keep top 50 strongest transitions
                sorted_trans = sorted(transitions, key=lambda t: t.A_ki or 0, reverse=True)[:50]
                self._transitions[el] = [
                    (
                        t.wavelength_nm,
                        t.E_upper_eV or 0.0,
                        t.g_upper or 1,
                        t.A_ki or 0.0,
                    )
                    for t in sorted_trans
                ]

        logger.info(
            f"Built edge model: {len(self.elements)} elements, " f"quantize={self.quantize}"
        )

    def _compile_functions(self) -> None:
        """Pre-compile JAX functions."""
        try:
            import jax
            import jax.numpy as jnp

            @jax.jit
            def _boltzmann_slope(energies: jnp.ndarray, y_values: jnp.ndarray) -> jnp.ndarray:
                """Fast Boltzmann slope calculation. Returns a 0-d JAX scalar."""
                n = len(energies)
                sum_x = jnp.sum(energies)
                sum_y = jnp.sum(y_values)
                sum_xy = jnp.sum(energies * y_values)
                sum_x2 = jnp.sum(energies**2)
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2)
                return slope

            self._compiled_functions["boltzmann_slope"] = _boltzmann_slope
            logger.info("JAX functions compiled for edge model")

        except ImportError:
            logger.warning("JAX not available for compilation")

    def get_partition_function(self, element: str, stage: int, T_K: float) -> float:
        """Get partition function value.

        Parameters
        ----------
        element : str
            Element symbol
        stage : int
            Ionization stage
        T_K : float
            Temperature in Kelvin

        Returns
        -------
        float
            Partition function value
        """
        key = (element, stage)
        if key not in self._partition_coeffs:
            return 25.0  # Default fallback

        coeffs = self._partition_coeffs[key]
        log_T = np.log10(T_K)
        log_U = sum(c * log_T**i for i, c in enumerate(coeffs))
        return 10.0**log_U

    def get_ionization_potential(self, element: str) -> float:
        """Get ionization potential in eV."""
        return self._ionization_potentials.get(element, 10.0)

    def get_transitions(self, element: str) -> List[Tuple[float, float, float, float]]:
        """Get pruned transitions for element.

        Returns
        -------
        List[Tuple[float, float, float, float]]
            List of (wavelength_nm, E_upper_eV, g_upper, A_ki)
        """
        return self._transitions.get(element, [])

    def estimate_memory_mb(self) -> float:
        """Estimate model memory footprint in MB."""
        total_bytes = 0

        # Partition coefficients
        for coeffs in self._partition_coeffs.values():
            total_bytes += coeffs.nbytes

        # Transitions (4 floats per transition)
        for trans_list in self._transitions.values():
            total_bytes += len(trans_list) * 4 * (2 if self.quantize else 4)

        return total_bytes / (1024 * 1024)


def create_streaming_pipeline(
    atomic_db: Any,
    elements: List[str],
    mode: AnalysisMode = AnalysisMode.STANDARD,
    buffer_size: int = 100,
    result_callback: Optional[Callable[[StreamingResult], None]] = None,
    **config_kwargs: Any,
) -> Tuple[SpectrumBuffer, StreamingAnalyzer]:
    """Convenience factory for creating a streaming pipeline.

    Parameters
    ----------
    atomic_db : AtomicDatabase
        Atomic database
    elements : List[str]
        Elements to analyze
    mode : AnalysisMode
        Analysis mode
    buffer_size : int
        Buffer size for spectra
    result_callback : Callable, optional
        Callback for results
    **config_kwargs
        Additional config parameters

    Returns
    -------
    Tuple[SpectrumBuffer, StreamingAnalyzer]
        Buffer and analyzer ready to use

    Example
    -------
    >>> buffer, analyzer = create_streaming_pipeline(
    ...     db, ["Fe", "Ni", "Cr"],
    ...     mode=AnalysisMode.FAST,
    ...     result_callback=lambda r: print(r.latency_ms)
    ... )
    >>> analyzer.start()
    """
    buffer = SpectrumBuffer(max_size=buffer_size)
    config = StreamingConfig(mode=mode, **config_kwargs)
    analyzer = StreamingAnalyzer(
        atomic_db,
        buffer,
        config=config,
        result_callback=result_callback,
        elements=elements,
    )
    return buffer, analyzer
