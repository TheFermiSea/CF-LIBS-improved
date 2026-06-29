# Echellogram Processing Guide

## Overview

Echelle spectrometers (such as the Andor Mechelle 5000) produce 2D spectral images called **echellograms** that must be "unwrapped" into 1D spectra for analysis. This guide explains the extraction algorithm and how to use the `EchelleExtractor` class in CF-LIBS.

## The Problem

An echellogram is a 2D image where:
- **X-axis**: High-resolution wavelength (fine dispersion within each order)
- **Y-axis**: Spectral order (coarse wavelength separation)

Multiple spectral orders are stacked vertically, with each order covering a different wavelength range. The orders are curved due to the echelle grating geometry.

## The Algorithm

CF-LIBS implements a standard **"Trace and Sum"** extraction algorithm with the following steps:

### 1. Order Tracing

Each spectral order $m$ has a center trace defined by a polynomial:

$$y_m(x) = a_2 x^2 + a_1 x + a_0$$

Where:
- $x$ is the pixel column (0 to width-1)
- $y_m(x)$ is the row position of the order center
- $a_0, a_1, a_2$ are polynomial coefficients from calibration

### 2. Flux Extraction

For each pixel column $x$, we extract flux by summing intensity in a window around the trace:

$$F_m(x) = \sum_{j=-\delta}^{\delta} I(x, y_m(x) + j) - \text{background}$$

Where:
- $\delta$ is the extraction window (typically 5 pixels)
- $I(x, y)$ is the image intensity at pixel $(x, y)$
- Background is estimated from pixels outside the extraction window

### 3. Wavelength Mapping

Each pixel position $x$ in order $m$ is mapped to wavelength using a polynomial calibration:

$$\lambda_m(x) = \text{Poly}_m(x)$$

The polynomial coefficients are determined during instrument calibration using reference lines (e.g., from a calibration lamp).

### 4. Order Merging

The 1D spectra from all orders are:
1. Interpolated onto a common linear wavelength grid (typically 0.05 nm step)
2. Merged in overlapping regions using one of:
   - **Weighted average**: Average weighted by number of contributing orders
   - **Simple average**: Simple arithmetic mean
   - **Maximum**: Take maximum value (useful for avoiding order overlap artifacts)

## Usage

### Basic Usage

```python
from cflibs.instrument.echelle import EchelleExtractor
import numpy as np

# Load calibration
extractor = EchelleExtractor(calibration_file='calibration.json')

# Load 2D image (e.g., from FITS, TIFF, or NumPy array)
image_2d = np.load('echellogram.npy')  # Shape: (height, width)

# Extract 1D spectrum
wavelength, intensity = extractor.extract_spectrum(image_2d)

# Save result
from cflibs.io import save_spectrum
save_spectrum('extracted_spectrum.csv', wavelength, intensity)
```

### Calibration File Format

The calibration file is a JSON file with the following structure:

```json
{
    "order_50": {
        "y_coeffs": [0.0001, 0.1, 500],
        "wl_coeffs": [0.05, 300.0]
    },
    "order_49": {
        "y_coeffs": [0.0001, 0.1, 700],
        "wl_coeffs": [0.05, 350.0]
    },
    ...
}
```

Where:
- `order_N`: Order identifier (typically the echelle order number)
- `y_coeffs`: Polynomial coefficients for trace position $y(x)$
  - Format: `[a2, a1, a0]` for $y = a_2 x^2 + a_1 x + a_0$
- `wl_coeffs`: Polynomial coefficients for wavelength $\lambda(x)$
  - Format: `[k1, k0]` for linear: $\lambda = k_1 x + k_0$
  - Or higher order: `[kn, ..., k1, k0]` for $\lambda = k_n x^n + ... + k_1 x + k_0$

**Note**: Polynomial coefficients are in order `[highest_order, ..., lowest_order]` as expected by `numpy.polyval`.

### Advanced Options

```python
# Custom extraction window
extractor = EchelleExtractor(
    calibration_file='calibration.json',
    extraction_window=7  # Use ±7 pixels instead of default ±5
)

# Custom merge method and wavelength step
wavelength, intensity = extractor.extract_spectrum(
    image_2d,
    wavelength_step_nm=0.01,  # Higher resolution output
    merge_method='max',  # Use maximum in overlaps
    min_valid_pixels=20  # Require at least 20 valid pixels per order
)
```

### Extracting Individual Orders

```python
# Extract a single order
wl_order, flux_order = extractor.extract_order(
    image_2d,
    order_name='order_50',
    background_subtract=True
)
```

### Creating Mock Calibration (Testing)

For testing without real calibration data:

```python
extractor = EchelleExtractor()
extractor.create_mock_calibration(
    width=2048,
    num_orders=3,
    wavelength_range=(300.0, 400.0)
)

# Now you can use the extractor with synthetic data
wavelength, intensity = extractor.extract_spectrum(mock_image)
```

## Calibration Procedure

To create a calibration file for your instrument:

1. **Acquire calibration lamp spectrum**: Use a reference lamp (e.g., Hg-Ar, Th-Ar) with known emission lines
2. **Identify orders**: Locate spectral orders in the 2D image
3. **Trace fitting**: Fit polynomial curves to order centers
4. **Wavelength calibration**: Match known reference lines to pixel positions in each order
5. **Save calibration**: Export polynomial coefficients to JSON format

Many echelle spectrometers come with calibration software that can export these coefficients. Alternatively, you can use tools like `pyspeckit` or custom scripts to perform the calibration.

## Integration with CF-LIBS Workflow

The extracted spectrum can be directly used in CF-LIBS analysis:

```python
from cflibs.instrument.echelle import EchelleExtractor
from cflibs.io import save_spectrum
from cflibs.inversion import LTEInversion  # Phase 3

# Extract from echellogram
extractor = EchelleExtractor('calibration.json')
wavelength, intensity = extractor.extract_spectrum(image_2d)

# Save for analysis
save_spectrum('extracted_spectrum.csv', wavelength, intensity)

# Use in inversion (Phase 3)
# inv = LTEInversion(...)
# result = inv.fit(wavelength, intensity)
```

## Performance Considerations

- **Extraction window**: Larger windows (e.g., ±7 pixels) capture more flux but may include more background
- **Wavelength step**: Smaller steps (e.g., 0.01 nm) provide higher resolution but larger file sizes
- **Merge method**: `max` is fastest, `weighted_average` provides best signal-to-noise in overlaps
- **Memory**: Large images (>2048×2048) may require chunked processing (not yet implemented)

## Limitations

- Currently implements simple summation extraction (not optimal extraction with profile weighting)
- Background subtraction is basic (median of nearby pixels)
- No automatic order detection (requires calibration file)
- No handling of cosmic rays or bad pixels (pre-processing required)

## Future Enhancements

Planned improvements for Phase 2+:
- Optimal extraction with profile weighting
- Automatic order detection
- Cosmic ray rejection
- Bad pixel masking
- Support for variable extraction windows per order
- Blaze function correction

---

For more information, see the `EchelleExtractor` class documentation in `cflibs.instrument.echelle`.

