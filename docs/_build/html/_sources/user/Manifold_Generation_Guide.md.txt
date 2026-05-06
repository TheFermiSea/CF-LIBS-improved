# Manifold Generation Guide

## Overview

The **spectral manifold** is a pre-computed lookup table of synthetic LIBS spectra that enables ultra-fast inference without solving physics equations at runtime. This is the foundation of the high-throughput CF-LIBS architecture.

## What is a Manifold?

A manifold is a high-dimensional array where:
- **Dimensions**: Temperature × Electron Density × Element Concentrations
- **Content**: Pre-computed spectral intensities at specific wavelength pixels
- **Purpose**: Fast lookup and similarity search for measured spectra

Instead of solving Saha-Boltzmann equations for each measurement, the system finds the nearest matching pre-computed spectrum in the manifold.

## Architecture

### Offline Generation (This Module)

- **Hardware**: GPU (NVIDIA V100 or similar)
- **Software**: JAX for GPU acceleration
- **Output**: HDF5 file (~500 GB - 2 TB)
- **Time**: Hours to days (one-time cost)

### Online Inference (Phase 3)

- **Hardware**: CPU with large RAM
- **Software**: Fast similarity search
- **Latency**: <1 ms per query
- **Throughput**: kHz rates

## Usage

### Basic Generation

```python
from cflibs.manifold import ManifoldGenerator, ManifoldConfig

# Create configuration
config = ManifoldConfig(
    db_path="libs_production.db",
    output_path="spectral_manifold.h5",
    elements=["Ti", "Al", "V", "Fe"],
    wavelength_range=(250.0, 550.0),
    temperature_range=(0.5, 2.0),
    temperature_steps=50,
    density_range=(1e16, 1e19),
    density_steps=20,
    concentration_steps=20
)

# Generate manifold
generator = ManifoldGenerator(config)
generator.generate_manifold()
```

### Using Configuration File

```bash
# Generate from YAML config
cflibs generate-manifold examples/manifold_config_example.yaml --progress
```

### Loading and Querying

```python
from cflibs.manifold import ManifoldLoader
import numpy as np

# Load manifold
loader = ManifoldLoader("spectral_manifold.h5")

# Find nearest match for measured spectrum
measured_spectrum = np.load("measured_spectrum.npy")
index, similarity, params = loader.find_nearest_spectrum(measured_spectrum)

print(f"Best match: similarity={similarity:.4f}")
print(f"Temperature: {params['T_eV']:.3f} eV")
print(f"Density: {params['n_e_cm3']:.2e} cm^-3")
print(f"Composition: Ti={params['Ti']:.1%}, Al={params['Al']:.1%}, V={params['V']:.1%}")
```

## Configuration

### Parameter Grid

The manifold covers a multi-dimensional parameter space:

#### Temperature Grid
- **Range**: 0.5 - 2.0 eV (typical for ultrafast LIBS)
- **Steps**: 50 points (linear spacing)
- **Total**: 50 temperature values

#### Electron Density Grid
- **Range**: 10¹⁶ - 10¹⁹ cm⁻³
- **Steps**: 20 points (logarithmic spacing)
- **Total**: 20 density values

#### Concentration Grid
- **Resolution**: 20 steps per element
- **Method**: Simplex (concentrations sum to 1.0)
- **Example**: For Ti-Al-V system:
  - Ti: 0.88 - 1.0 (varying)
  - Al: 0.0 - 0.12
  - V: 0.0 - 0.12
  - Fe: 0.002 (fixed trace)

#### Total Manifold Size

For Ti-Al-V-Fe system with default settings:
- Temperature: 50 points
- Density: 20 points
- Concentrations: ~400 combinations (20×20)
- **Total**: 50 × 20 × 400 = **400,000 spectra**

Each spectrum has 4096 wavelength points, so total data:
- **Size**: ~6.5 GB (uncompressed)
- **With compression**: ~2-3 GB

## Physics Model

### Time-Integrated Emission

The manifold uses **time-integrated** spectra to model ultrafast plasma cooling:

$$
I_{\text{synth}}(\lambda) = \int_{t_{gate}}^{t_{end}} \epsilon(\lambda, T_e(t), n_e(t)) \, dt
$$

Where:
- $T_e(t) = T_{max} \cdot (1 + t/t_0)^{-0.5}$ (power-law cooling)
- $n_e(t) = n_{e,max} \cdot (1 + t/t_0)^{-1.0}$ (density decay)

### Integration Parameters

- **Gate Delay**: 300 ns (typical ICCD delay)
- **Gate Width**: 5 μs (integration time)
- **Time Steps**: 20 (integration resolution)

## Performance

### Generation Speed

On NVIDIA V100 GPU:
- **~10,000 spectra/second** (with JAX)
- **400,000 spectra**: ~40 seconds
- **Larger grids**: Scale linearly

### Memory Requirements

- **GPU VRAM**: ~8-16 GB (depending on batch size)
- **System RAM**: ~32 GB recommended
- **Disk**: 2-10 GB per manifold

### Optimization Tips

1. **Reduce grid resolution** for faster generation
2. **Use smaller wavelength range** if only specific region needed
3. **Adjust batch size** based on GPU memory
4. **Use compression** in HDF5 (enabled by default)

## File Format

The manifold is stored as HDF5 with:

### Datasets

- **`spectra`**: (N_samples, N_pixels) - Spectral intensities
- **`params`**: (N_samples, N_elements + 2) - Parameters [T, ne, C1, C2, ...]
- **`wavelength`**: (N_pixels,) - Wavelength grid

### Attributes

- `elements`: List of element symbols
- `wavelength_range`: [min, max] wavelength
- `temperature_range`: [min, max] temperature
- `density_range`: [min, max] density

## Similarity Search Methods

### Cosine Similarity (Default)

$$
\text{similarity} = \frac{\mathbf{s}_1 \cdot \mathbf{s}_2}{|\mathbf{s}_1| |\mathbf{s}_2|}
$$

- Fast and robust to scaling
- Good for normalized spectra

### Euclidean Distance

$$
\text{distance} = |\mathbf{s}_1 - \mathbf{s}_2|
$$

- Sensitive to absolute intensity
- Useful when intensity calibration is accurate

### Correlation

$$
\text{correlation} = \frac{\text{Cov}(\mathbf{s}_1, \mathbf{s}_2)}{\sigma_1 \sigma_2}
$$

- Robust to baseline shifts
- Good for noisy data

## Integration with Inversion (Phase 3)

The manifold enables fast inference:

```python
# Phase 3: Fast inference using manifold
from cflibs.manifold import ManifoldLoader
from cflibs.inversion import ManifoldInversion

loader = ManifoldLoader("spectral_manifold.h5")
inverter = ManifoldInversion(loader)

# Fast inference (<1 ms)
result = inverter.invert(measured_spectrum)
```

## Limitations

### Current Implementation (Phase 1)

- **Gaussian broadening only** (Voigt in Phase 2)
- **Simplified partition functions** (interpolation tables in Phase 2)
- **Fixed cooling law** (configurable in Phase 2)
- **Single-zone plasma** (multi-zone in Phase 2)

### Grid Resolution

- **Coarse grids**: Faster generation, lower accuracy
- **Fine grids**: Slower generation, higher accuracy
- **Trade-off**: Balance between generation time and inference accuracy

## Troubleshooting

### "JAX not available"

**Solution**: Install JAX:
```bash
pip install jax jaxlib
# For GPU support:
pip install jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### "Out of memory"

**Solutions**:
1. Reduce `batch_size` in config
2. Reduce grid resolution
3. Use smaller wavelength range
4. Process in chunks

### "Generation too slow"

**Solutions**:
1. Use GPU (JAX automatically uses GPU if available)
2. Reduce grid resolution
3. Reduce wavelength range
4. Increase batch size (if memory allows)

## Future Enhancements

- **Voigt profiles**: More accurate line broadening
- **Stark broadening**: Electron density dependent
- **Multi-zone plasmas**: Spatial variation
- **Adaptive grids**: Higher resolution where needed
- **Incremental updates**: Add new parameter combinations
- **Distributed generation**: Multi-GPU support

## See Also

- [API Reference](../reference/API_Reference.md) — `ManifoldGenerator` API
- `docs/archive/legacy/HIGH_THROUGHPUT_FRAMEWORK.md` — historical architecture notes (superseded)
