"""
Tests for the real-time streaming analysis pipeline.

These tests validate:
1. SpectrumBuffer thread-safety and ring buffer semantics
2. StreamingAnalyzer processing and latency monitoring
3. EdgeOptimizedModel quantization and pruning
4. Factory function for pipeline creation
"""

import pytest
import numpy as np
import threading
import time
from unittest.mock import Mock

from cflibs.inversion.streaming import (
    AnalysisMode,
    StreamingConfig,
    SpectrumPacket,
    StreamingResult,
    SpectrumBuffer,
    LatencyMonitor,
    FastAnalyzer,
    StandardAnalyzer,
    StreamingAnalyzer,
    EdgeOptimizedModel,
    create_streaming_pipeline,
)
from cflibs.inversion.solver import CFLIBSResult
from cflibs.inversion.boltzmann import LineObservation

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_atomic_db():
    """Create a mock atomic database."""
    db = Mock()

    # Mock partition coefficients
    pf_mock = Mock()
    pf_mock.coefficients = [0.5, 0.1, -0.01]
    db.get_partition_coefficients.return_value = pf_mock

    # Mock ionization potential
    db.get_ionization_potential.return_value = 7.9  # Fe I

    # Mock transitions
    trans = Mock()
    trans.wavelength_nm = 385.991
    trans.E_upper_eV = 4.15
    trans.g_upper = 7
    trans.A_ki = 1.0e8
    db.get_transitions.return_value = [trans] * 10

    return db


@pytest.fixture
def sample_spectrum():
    """Create a sample spectrum for testing."""
    wavelength = np.linspace(200, 800, 1000)
    # Simple Gaussian emission
    intensity = 1000 * np.exp(-((wavelength - 400) ** 2) / 100)
    intensity += np.random.normal(0, 10, len(wavelength))
    intensity = np.maximum(intensity, 1.0)
    return wavelength, intensity


@pytest.fixture
def sample_line_observations():
    """Create sample line observations."""
    return [
        LineObservation(
            wavelength_nm=385.991,
            intensity=1000.0,
            intensity_uncertainty=50.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.15,
            g_k=7,
            A_ki=1.0e8,
        ),
        LineObservation(
            wavelength_nm=404.581,
            intensity=800.0,
            intensity_uncertainty=40.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.55,
            g_k=9,
            A_ki=8.0e7,
        ),
        LineObservation(
            wavelength_nm=438.354,
            intensity=600.0,
            intensity_uncertainty=30.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=4.96,
            g_k=11,
            A_ki=5.0e7,
        ),
        LineObservation(
            wavelength_nm=516.749,
            intensity=400.0,
            intensity_uncertainty=20.0,
            element="Fe",
            ionization_stage=1,
            E_k_ev=5.28,
            g_k=5,
            A_ki=3.0e7,
        ),
    ]


# =============================================================================
# Test StreamingConfig
# =============================================================================


class TestStreamingConfig:
    """Tests for StreamingConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = StreamingConfig()
        assert config.mode == AnalysisMode.STANDARD
        assert config.max_latency_ms == 500.0
        assert config.batch_size == 10
        assert config.downsample_factor == 1
        assert config.use_jit is True
        assert config.cache_atomic_data is True
        assert config.quantize is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = StreamingConfig(
            mode=AnalysisMode.FAST,
            max_latency_ms=100.0,
            batch_size=5,
            quantize=True,
        )
        assert config.mode == AnalysisMode.FAST
        assert config.max_latency_ms == 100.0
        assert config.batch_size == 5
        assert config.quantize is True


# =============================================================================
# Test SpectrumPacket
# =============================================================================


class TestSpectrumPacket:
    """Tests for SpectrumPacket dataclass."""

    def test_packet_creation(self, sample_spectrum):
        """Test basic packet creation."""
        wl, intensity = sample_spectrum
        packet = SpectrumPacket(
            wavelength=wl,
            intensity=intensity,
            timestamp=time.time(),
            metadata={"shot": 1, "position": (0.0, 0.0)},
            sequence_id=42,
        )

        assert len(packet.wavelength) == len(wl)
        assert len(packet.intensity) == len(intensity)
        assert packet.sequence_id == 42
        assert packet.metadata["shot"] == 1

    def test_packet_with_defaults(self, sample_spectrum):
        """Test packet with default metadata."""
        wl, intensity = sample_spectrum
        packet = SpectrumPacket(
            wavelength=wl,
            intensity=intensity,
            timestamp=1.0,
        )

        assert packet.metadata == {}
        assert packet.sequence_id == 0


# =============================================================================
# Test SpectrumBuffer
# =============================================================================


class TestSpectrumBuffer:
    """Tests for SpectrumBuffer thread-safe ring buffer."""

    def test_basic_push_pop(self, sample_spectrum):
        """Test basic push and pop operations."""
        buffer = SpectrumBuffer(max_size=10)
        wl, intensity = sample_spectrum

        # Push
        result = buffer.push(wl, intensity, timestamp=1.0)
        assert result is True
        assert buffer.size == 1

        # Pop
        packet = buffer.pop(timeout=1.0)
        assert packet is not None
        assert len(packet.wavelength) == len(wl)
        assert buffer.size == 0

    def test_ring_buffer_semantics(self, sample_spectrum):
        """Test that old items are dropped when buffer is full."""
        buffer = SpectrumBuffer(max_size=5, drop_old=True)
        wl, intensity = sample_spectrum

        # Push more than max_size
        for i in range(10):
            buffer.push(wl, intensity, metadata={"id": i})

        # Should have exactly max_size items
        assert buffer.size == 5
        assert buffer.dropped_count == 5

        # Oldest items should be dropped (ids 0-4)
        packet = buffer.pop()
        assert packet.metadata["id"] == 5  # First remaining

    def test_blocking_mode(self, sample_spectrum):
        """Test blocking mode when buffer is full."""
        buffer = SpectrumBuffer(max_size=2, drop_old=False)
        wl, intensity = sample_spectrum

        # Fill buffer
        buffer.push(wl, intensity)
        buffer.push(wl, intensity)

        # Try to push with timeout - should fail
        result = buffer.push(wl, intensity, timeout=0.1)
        assert result is False
        assert buffer.dropped_count == 0

    def test_pop_timeout(self):
        """Test pop timeout on empty buffer."""
        buffer = SpectrumBuffer(max_size=10)

        start = time.time()
        packet = buffer.pop(timeout=0.1)
        elapsed = time.time() - start

        assert packet is None
        assert elapsed >= 0.1

    def test_pop_batch(self, sample_spectrum):
        """Test batch pop operation."""
        buffer = SpectrumBuffer(max_size=20)
        wl, intensity = sample_spectrum

        # Push several spectra
        for i in range(15):
            buffer.push(wl, intensity, metadata={"id": i})

        # Pop batch
        batch = buffer.pop_batch(max_count=10, timeout=1.0)
        assert len(batch) == 10
        assert buffer.size == 5

    def test_pop_batch_partial(self, sample_spectrum):
        """Test batch pop with fewer items than requested."""
        buffer = SpectrumBuffer(max_size=20)
        wl, intensity = sample_spectrum

        buffer.push(wl, intensity)
        buffer.push(wl, intensity)

        batch = buffer.pop_batch(max_count=10, timeout=0.1)
        assert len(batch) == 2

    def test_close(self, sample_spectrum):
        """Test buffer close operation."""
        buffer = SpectrumBuffer(max_size=10)
        wl, intensity = sample_spectrum

        buffer.push(wl, intensity)
        buffer.close()

        # Push should fail after close
        result = buffer.push(wl, intensity)
        assert result is False
        assert buffer.is_closed

    def test_clear(self, sample_spectrum):
        """Test clear operation."""
        buffer = SpectrumBuffer(max_size=10)
        wl, intensity = sample_spectrum

        for _ in range(5):
            buffer.push(wl, intensity)

        cleared = buffer.clear()
        assert cleared == 5
        assert buffer.size == 0

    def test_thread_safety(self, sample_spectrum):
        """Test thread-safe concurrent access."""
        buffer = SpectrumBuffer(max_size=100)
        wl, intensity = sample_spectrum
        n_producers = 3
        n_items_per_producer = 50
        results = []

        def producer(producer_id):
            for i in range(n_items_per_producer):
                buffer.push(wl, intensity, metadata={"producer": producer_id, "item": i})

        def consumer():
            while True:
                packet = buffer.pop(timeout=0.5)
                if packet is None:
                    break
                results.append(packet)

        # Start producers
        producer_threads = [
            threading.Thread(target=producer, args=(i,)) for i in range(n_producers)
        ]
        consumer_thread = threading.Thread(target=consumer)

        for t in producer_threads:
            t.start()
        consumer_thread.start()

        for t in producer_threads:
            t.join()

        # Give consumer time to finish
        time.sleep(0.5)
        buffer.close()
        consumer_thread.join(timeout=2.0)

        # Should have received all non-dropped items
        total_sent = n_producers * n_items_per_producer
        total_received = len(results) + buffer.dropped_count
        assert total_received == total_sent


# =============================================================================
# Test LatencyMonitor
# =============================================================================


class TestLatencyMonitor:
    """Tests for LatencyMonitor statistics tracking."""

    def test_empty_stats(self):
        """Test stats on empty monitor."""
        monitor = LatencyMonitor(target_ms=100.0)
        stats = monitor.get_stats()

        assert stats.n_samples == 0
        assert stats.mean_ms == 0.0
        assert stats.target_met_fraction == 1.0

    def test_record_and_stats(self):
        """Test recording latencies and computing stats."""
        monitor = LatencyMonitor(window_size=100, target_ms=100.0)

        # Record some latencies
        for i in range(50):
            monitor.record(50.0 + i * 2)  # 50-148ms

        stats = monitor.get_stats()

        assert stats.n_samples == 50
        assert stats.min_ms == 50.0
        assert stats.max_ms == 148.0
        np.testing.assert_allclose(stats.mean_ms, 99.0, atol=1.0)

    def test_target_met_fraction(self):
        """Test target met fraction calculation."""
        monitor = LatencyMonitor(target_ms=100.0)

        # 80 below target, 20 above
        for _ in range(80):
            monitor.record(50.0)
        for _ in range(20):
            monitor.record(150.0)

        stats = monitor.get_stats()
        np.testing.assert_allclose(stats.target_met_fraction, 0.8, atol=0.01)

    def test_percentiles(self):
        """Test percentile calculations."""
        monitor = LatencyMonitor(window_size=100)

        # Record 100 sequential values
        for i in range(100):
            monitor.record(float(i))

        stats = monitor.get_stats()

        np.testing.assert_allclose(stats.p50_ms, 49.5, atol=1.0)
        np.testing.assert_allclose(stats.p95_ms, 94.5, atol=1.0)
        np.testing.assert_allclose(stats.p99_ms, 98.5, atol=1.0)

    def test_reset(self):
        """Test reset operation."""
        monitor = LatencyMonitor()

        for _ in range(10):
            monitor.record(100.0)

        monitor.reset()
        stats = monitor.get_stats()
        assert stats.n_samples == 0

    def test_window_size(self):
        """Test that window size is respected."""
        monitor = LatencyMonitor(window_size=10)

        # Record more than window size
        for i in range(20):
            monitor.record(float(i))

        stats = monitor.get_stats()
        assert stats.n_samples == 10
        assert stats.min_ms == 10.0  # Oldest values dropped


# =============================================================================
# Test FastAnalyzer
# =============================================================================


class TestFastAnalyzer:
    """Tests for FastAnalyzer simplified physics."""

    def test_analyze_without_lines(self, mock_atomic_db, sample_spectrum):
        """Test analysis when no line observations provided."""
        analyzer = FastAnalyzer(mock_atomic_db, ["Fe", "Cu"])
        wl, intensity = sample_spectrum

        result = analyzer.analyze_spectrum(wl, intensity)

        assert result.temperature_K > 0
        assert result.converged
        assert result.quality_metrics.get("mode") == "fast"
        assert sum(result.concentrations.values()) > 0.99

    def test_analyze_with_lines(self, mock_atomic_db, sample_line_observations):
        """Test analysis with line observations."""
        analyzer = FastAnalyzer(mock_atomic_db, ["Fe"])
        wl = np.linspace(200, 800, 1000)
        intensity = np.ones_like(wl) * 100

        result = analyzer.analyze_spectrum(
            wl, intensity, line_observations=sample_line_observations
        )

        assert result.temperature_K > 0
        assert "Fe" in result.concentrations
        assert result.quality_metrics.get("n_lines", 0) > 0

    def test_insufficient_lines(self, mock_atomic_db):
        """Test handling of insufficient lines."""
        analyzer = FastAnalyzer(mock_atomic_db, ["Fe"])
        wl = np.linspace(200, 800, 1000)
        intensity = np.ones_like(wl)

        # Only 2 lines (need at least 3)
        observations = [
            LineObservation(
                wavelength_nm=400.0,
                intensity=100.0,
                intensity_uncertainty=10.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=4.0,
                g_k=5,
                A_ki=1e8,
            ),
            LineObservation(
                wavelength_nm=500.0,
                intensity=80.0,
                intensity_uncertainty=8.0,
                element="Fe",
                ionization_stage=1,
                E_k_ev=4.5,
                g_k=7,
                A_ki=8e7,
            ),
        ]

        result = analyzer.analyze_spectrum(wl, intensity, line_observations=observations)

        assert not result.converged or result.quality_metrics.get("warning") == "insufficient_lines"

    def test_batch_analyze(self, mock_atomic_db, sample_spectrum):
        """Test batch analysis."""
        analyzer = FastAnalyzer(mock_atomic_db, ["Fe", "Cu"])
        wl, intensity = sample_spectrum

        results = analyzer.analyze_batch(
            [wl, wl, wl],
            [intensity, intensity, intensity],
        )

        assert len(results) == 3
        assert all(r.temperature_K > 0 for r in results)

    def test_downsampling(self, mock_atomic_db, sample_spectrum):
        """Test spectral downsampling."""
        config = StreamingConfig(mode=AnalysisMode.FAST, downsample_factor=2)
        analyzer = FastAnalyzer(mock_atomic_db, ["Fe"], config=config)
        wl, intensity = sample_spectrum

        # Should work with downsampling
        result = analyzer.analyze_spectrum(wl, intensity)
        assert result.temperature_K > 0


# =============================================================================
# Test StandardAnalyzer
# =============================================================================


class TestStandardAnalyzer:
    """Tests for StandardAnalyzer full CF-LIBS."""

    def test_analyze_without_lines(self, mock_atomic_db, sample_spectrum):
        """Test analysis when no line observations provided."""
        analyzer = StandardAnalyzer(mock_atomic_db)
        wl, intensity = sample_spectrum

        result = analyzer.analyze_spectrum(wl, intensity)

        # Should return placeholder when no lines
        assert not result.converged or result.quality_metrics.get("warning") == "no_lines"

    def test_reduced_iterations_config(self, mock_atomic_db):
        """Test that reduced iterations are used for low latency."""
        config = StreamingConfig(mode=AnalysisMode.STANDARD, max_latency_ms=200.0)
        analyzer = StandardAnalyzer(mock_atomic_db, config=config)

        # Internal solver should have reduced max_iterations
        assert analyzer._solver.max_iterations <= 10


# =============================================================================
# Test StreamingAnalyzer
# =============================================================================


class TestStreamingAnalyzer:
    """Tests for StreamingAnalyzer main class."""

    def test_start_stop(self, mock_atomic_db):
        """Test starting and stopping the analyzer."""
        buffer = SpectrumBuffer(max_size=10)
        analyzer = StreamingAnalyzer(mock_atomic_db, buffer, elements=["Fe"])

        assert not analyzer.is_running
        analyzer.start()
        assert analyzer.is_running

        time.sleep(0.1)
        analyzer.stop()
        assert not analyzer.is_running

    def test_process_spectrum(self, mock_atomic_db, sample_spectrum):
        """Test processing a spectrum through the pipeline."""
        buffer = SpectrumBuffer(max_size=10)
        results_received = []

        def callback(result):
            results_received.append(result)

        analyzer = StreamingAnalyzer(
            mock_atomic_db,
            buffer,
            config=StreamingConfig(mode=AnalysisMode.FAST),
            result_callback=callback,
            elements=["Fe"],
        )

        analyzer.start()

        wl, intensity = sample_spectrum
        buffer.push(wl, intensity, timestamp=time.time())

        # Wait for processing
        time.sleep(0.5)
        analyzer.stop()

        assert analyzer.processed_count >= 1
        assert len(results_received) >= 1
        assert results_received[0].latency_ms > 0

    def test_latency_monitoring(self, mock_atomic_db, sample_spectrum):
        """Test that latency is properly monitored."""
        buffer = SpectrumBuffer(max_size=10)
        config = StreamingConfig(mode=AnalysisMode.FAST, max_latency_ms=1000.0)
        analyzer = StreamingAnalyzer(mock_atomic_db, buffer, config=config, elements=["Fe"])

        analyzer.start()

        wl, intensity = sample_spectrum
        for _ in range(5):
            buffer.push(wl, intensity)
            time.sleep(0.05)

        time.sleep(0.3)
        analyzer.stop()

        stats = analyzer.latency_monitor.get_stats()
        assert stats.n_samples >= 1

    def test_get_results(self, mock_atomic_db, sample_spectrum):
        """Test retrieving buffered results."""
        buffer = SpectrumBuffer(max_size=10)
        analyzer = StreamingAnalyzer(
            mock_atomic_db,
            buffer,
            config=StreamingConfig(mode=AnalysisMode.FAST),
            elements=["Fe"],
        )

        analyzer.start()

        wl, intensity = sample_spectrum
        for _ in range(3):
            buffer.push(wl, intensity)

        time.sleep(0.3)
        analyzer.stop()

        results = analyzer.get_results()
        assert len(results) >= 1

    def test_batch_mode(self, mock_atomic_db, sample_spectrum):
        """Test batch processing mode."""
        buffer = SpectrumBuffer(max_size=20)
        config = StreamingConfig(mode=AnalysisMode.BATCH, batch_size=3)
        analyzer = StreamingAnalyzer(mock_atomic_db, buffer, config=config, elements=["Fe"])

        analyzer.start()

        wl, intensity = sample_spectrum
        for _ in range(6):
            buffer.push(wl, intensity)

        time.sleep(0.5)
        analyzer.stop()

        assert analyzer.processed_count >= 3

    def test_quality_assessment(self, mock_atomic_db, sample_spectrum):
        """Test quality flag assignment."""
        buffer = SpectrumBuffer(max_size=10)
        results_received = []

        def callback(result):
            results_received.append(result)

        analyzer = StreamingAnalyzer(
            mock_atomic_db,
            buffer,
            config=StreamingConfig(mode=AnalysisMode.FAST),
            result_callback=callback,
            elements=["Fe"],
        )

        analyzer.start()

        wl, intensity = sample_spectrum
        buffer.push(wl, intensity)

        time.sleep(0.3)
        analyzer.stop()

        if results_received:
            assert results_received[0].quality_flag in ["good", "warning", "poor"]


# =============================================================================
# Test EdgeOptimizedModel
# =============================================================================


class TestEdgeOptimizedModel:
    """Tests for EdgeOptimizedModel edge deployment optimizations."""

    def test_model_creation(self, mock_atomic_db):
        """Test basic model creation."""
        model = EdgeOptimizedModel(
            mock_atomic_db,
            elements=["Fe", "Cu"],
            quantize=True,
            compile_jax=False,  # Skip JAX for testing
        )

        assert "Fe" in model.elements
        assert "Cu" in model.elements

    def test_quantization(self, mock_atomic_db):
        """Test float16 quantization."""
        model = EdgeOptimizedModel(
            mock_atomic_db,
            elements=["Fe"],
            quantize=True,
            compile_jax=False,
        )

        # Partition coefficients should be float16
        key = ("Fe", 1)
        if key in model._partition_coeffs:
            assert model._partition_coeffs[key].dtype == np.float16

    def test_no_quantization(self, mock_atomic_db):
        """Test without quantization (float32)."""
        model = EdgeOptimizedModel(
            mock_atomic_db,
            elements=["Fe"],
            quantize=False,
            compile_jax=False,
        )

        key = ("Fe", 1)
        if key in model._partition_coeffs:
            assert model._partition_coeffs[key].dtype == np.float32

    def test_get_partition_function(self, mock_atomic_db):
        """Test partition function retrieval."""
        model = EdgeOptimizedModel(mock_atomic_db, elements=["Fe"], compile_jax=False)

        pf = model.get_partition_function("Fe", 1, 10000.0)
        assert pf > 0

    def test_get_partition_function_missing(self, mock_atomic_db):
        """Test partition function for missing element."""
        model = EdgeOptimizedModel(mock_atomic_db, elements=["Fe"], compile_jax=False)

        # Should return default
        pf = model.get_partition_function("Xe", 1, 10000.0)
        assert pf == 25.0

    def test_get_ionization_potential(self, mock_atomic_db):
        """Test ionization potential retrieval."""
        model = EdgeOptimizedModel(mock_atomic_db, elements=["Fe"], compile_jax=False)

        ip = model.get_ionization_potential("Fe")
        assert ip > 0

    def test_get_transitions(self, mock_atomic_db):
        """Test pruned transitions retrieval."""
        model = EdgeOptimizedModel(mock_atomic_db, elements=["Fe"], compile_jax=False)

        transitions = model.get_transitions("Fe")
        assert len(transitions) <= 50  # Max 50 per element
        if transitions:
            wl, E, g, A = transitions[0]
            assert wl > 0
            assert A > 0

    def test_memory_estimation(self, mock_atomic_db):
        """Test memory footprint estimation."""
        model = EdgeOptimizedModel(mock_atomic_db, elements=["Fe", "Cu"], compile_jax=False)

        mem_mb = model.estimate_memory_mb()
        assert mem_mb >= 0
        assert mem_mb < 100  # Should be very small

    @pytest.mark.requires_jax
    def test_jax_compilation(self, mock_atomic_db):
        """Test JAX function compilation."""
        pytest.importorskip("jax")

        model = EdgeOptimizedModel(mock_atomic_db, elements=["Fe"], compile_jax=True)

        assert "boltzmann_slope" in model._compiled_functions


# =============================================================================
# Test Factory Function
# =============================================================================


class TestCreateStreamingPipeline:
    """Tests for create_streaming_pipeline factory function."""

    def test_basic_creation(self, mock_atomic_db):
        """Test basic pipeline creation."""
        buffer, analyzer = create_streaming_pipeline(
            mock_atomic_db,
            elements=["Fe", "Cu"],
            mode=AnalysisMode.FAST,
        )

        assert buffer.max_size == 100  # Default
        assert analyzer.config.mode == AnalysisMode.FAST
        assert not analyzer.is_running

    def test_custom_parameters(self, mock_atomic_db):
        """Test pipeline with custom parameters."""
        results = []

        def callback(r):
            results.append(r)

        buffer, analyzer = create_streaming_pipeline(
            mock_atomic_db,
            elements=["Fe"],
            mode=AnalysisMode.STANDARD,
            buffer_size=50,
            result_callback=callback,
            max_latency_ms=200.0,
        )

        assert buffer.max_size == 50
        assert analyzer.config.max_latency_ms == 200.0
        assert analyzer.result_callback is callback


# =============================================================================
# Test StreamingResult
# =============================================================================


class TestStreamingResult:
    """Tests for StreamingResult dataclass."""

    def test_result_creation(self):
        """Test basic result creation."""
        cflibs_result = CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=500.0,
            electron_density_cm3=1e17,
            concentrations={"Fe": 0.7, "Cu": 0.3},
            concentration_uncertainties={"Fe": 0.05, "Cu": 0.05},
            iterations=5,
            converged=True,
        )

        result = StreamingResult(
            sequence_id=42,
            timestamp=1000.0,
            analysis_timestamp=1000.1,
            latency_ms=100.0,
            result=cflibs_result,
            quality_flag="good",
            warnings=[],
            metadata={"shot": 1},
        )

        assert result.sequence_id == 42
        assert result.latency_ms == 100.0
        assert result.result.temperature_K == 10000.0
        assert result.quality_flag == "good"

    def test_result_with_warnings(self):
        """Test result with warnings."""
        cflibs_result = CFLIBSResult(
            temperature_K=10000.0,
            temperature_uncertainty_K=500.0,
            electron_density_cm3=1e17,
            concentrations={},
            concentration_uncertainties={},
            iterations=1,
            converged=False,
        )

        result = StreamingResult(
            sequence_id=1,
            timestamp=1.0,
            analysis_timestamp=1.5,
            latency_ms=500.0,
            result=cflibs_result,
            quality_flag="poor",
            warnings=["Analysis did not converge"],
        )

        assert result.quality_flag == "poor"
        assert len(result.warnings) == 1


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.integration
class TestStreamingIntegration:
    """Integration tests for the full streaming pipeline."""

    def test_end_to_end_pipeline(self, mock_atomic_db, sample_spectrum):
        """Test complete end-to-end pipeline operation."""
        buffer, analyzer = create_streaming_pipeline(
            mock_atomic_db,
            elements=["Fe", "Cu"],
            mode=AnalysisMode.FAST,
            buffer_size=20,
        )

        results = []
        analyzer.result_callback = lambda r: results.append(r)

        analyzer.start()

        wl, intensity = sample_spectrum
        for i in range(10):
            buffer.push(wl, intensity, metadata={"shot": i})
            time.sleep(0.02)

        time.sleep(0.5)
        analyzer.stop()

        # Verify results
        assert len(results) >= 5
        assert analyzer.processed_count >= 5
        assert analyzer.error_count == 0

        # Check latency stats
        stats = analyzer.latency_monitor.get_stats()
        assert stats.n_samples >= 5
        assert stats.mean_ms > 0

    def test_high_throughput(self, mock_atomic_db, sample_spectrum):
        """Test high-throughput scenario."""
        buffer = SpectrumBuffer(max_size=200, drop_old=True)
        config = StreamingConfig(mode=AnalysisMode.FAST, max_latency_ms=100.0)
        analyzer = StreamingAnalyzer(mock_atomic_db, buffer, config=config, elements=["Fe"])

        analyzer.start()

        wl, intensity = sample_spectrum
        # Push rapidly
        for _ in range(100):
            buffer.push(wl, intensity)

        time.sleep(1.0)
        analyzer.stop()

        # Should have processed many, possibly dropping some
        total = analyzer.processed_count + buffer.dropped_count
        assert total >= 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
