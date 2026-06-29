# Atomic Database Generation Guide

## Overview

CF-LIBS requires an atomic database (`libs_production.db`) containing spectral line data, energy levels, and ionization potentials from NIST. This guide explains how to generate this database.

## Quick Start

### Method 1: Using the CLI Command

```bash
cflibs generate-db
```

This will generate `libs_production.db` in the current directory.

### Method 2: Using the Script Directly

```bash
python datagen_v2.py
```

## Requirements

### Python Dependencies

```bash
pip install requests-cache pandas
```

### ASDCache Library

The database generator requires the `ASDCache` library to fetch data from NIST. You can:

1. **Install from source**: Clone the ASDCache repository
2. **Use the fallback**: The script includes a fallback mode, but it won't fetch data

### Internet Access

The script fetches data from NIST Atomic Spectra Database (ASD), so internet access is required. Data is cached locally to speed up subsequent runs.

## Database Generation Process

### What Gets Generated

The database contains three main tables:

1. **`lines`**: Spectral line data
   - Wavelength, Einstein A coefficients
   - Upper and lower level energies
   - Statistical weights
   - Relative intensities

2. **`energy_levels`**: Energy level data
   - Level energies
   - Statistical weights
   - Used for partition function calculations

3. **`species_physics`**: Ionization potentials
   - Ionization energies for each species
   - Required for Saha equation calculations

### Filtering Options

The `datagen_v2.py` script includes filtering optimized for ultrafast LIBS:

- **MAX_IONIZATION_STAGE = 2**: Only neutral (I) and singly ionized (II) species
- **MAX_UPPER_ENERGY_EV = 12.0**: Exclude high-energy levels unlikely to be populated
- **MIN_RELATIVE_INTENSITY = 50**: Exclude very weak lines

You can modify these constants in `datagen_v2.py` if needed.

### Elements Included

By default, the script generates data for all stable elements (H through U), excluding:
- Short-lived radioactives (Tc, Pm)
- Heavy actinides beyond U

## Usage Examples

### Generate Full Database

```bash
# This will take several hours
python datagen_v2.py
```

### Generate for Specific Elements

Modify `datagen_v2.py` to change the `ALL_ELEMENTS` list:

```python
ALL_ELEMENTS = ["Ti", "Al", "V", "Fe"]  # Only these elements
```

### Custom Database Path

Modify `DB_NAME` in `datagen_v2.py`:

```python
DB_NAME = "my_custom_database.db"
```

## Database Structure

### Tables Schema

#### `lines` Table

```sql
CREATE TABLE lines (
    id INTEGER PRIMARY KEY,
    element TEXT,
    sp_num INTEGER,           -- Ionization stage (1=neutral, 2=singly ionized)
    wavelength_nm REAL,
    aki REAL,                 -- Einstein A coefficient (s^-1)
    ei_ev REAL,               -- Lower level energy (eV)
    ek_ev REAL,               -- Upper level energy (eV)
    gi INTEGER,                -- Lower level statistical weight
    gk INTEGER,                -- Upper level statistical weight
    rel_int REAL,              -- Relative intensity
    UNIQUE(element, sp_num, wavelength_nm, ek_ev)
)
```

#### `energy_levels` Table

```sql
CREATE TABLE energy_levels (
    element TEXT,
    sp_num INTEGER,
    g_level INTEGER,          -- Statistical weight
    energy_ev REAL            -- Energy above ground state (eV)
)
```

#### `species_physics` Table

```sql
CREATE TABLE species_physics (
    element TEXT,
    sp_num INTEGER,
    ip_ev REAL,              -- Ionization potential (eV)
    PRIMARY KEY (element, sp_num)
)
```

## Troubleshooting

### "ASDCache not found"

**Problem**: The ASDCache library is not available.

**Solutions**:
1. Install ASDCache from source
2. The script will use a fallback mode, but won't fetch data
3. Use a pre-generated database if available

### "Database generation takes too long"

**Problem**: Generating for all elements takes hours.

**Solutions**:
1. Generate only for elements you need (modify `ALL_ELEMENTS`)
2. Use a pre-generated database
3. Run in background: `nohup python datagen_v2.py &`

### "Network errors"

**Problem**: Cannot fetch data from NIST.

**Solutions**:
1. Check internet connection
2. NIST servers may be temporarily unavailable
3. The script uses caching - retry later
4. Check firewall/proxy settings

### "Database file is too large"

**Problem**: The database file is very large (>100 MB).

**Solutions**:
1. This is normal - atomic data is extensive
2. Use filtering options to reduce size
3. Generate only for needed elements

## Pre-Generated Databases

If available, pre-generated databases may be provided:
- Check the releases page
- Contact maintainers
- Generate your own using the scripts

## Updating the Database

To update the database with new NIST data:

1. Delete the old database (or rename it)
2. Delete cache directories (`nist_*_cache/`)
3. Run `datagen_v2.py` again

## Integration with CF-LIBS

Once generated, use the database in your code:

```python
from cflibs.atomic import AtomicDatabase

# Load database
atomic_db = AtomicDatabase("libs_production.db")

# Use in spectrum model
from cflibs.radiation import SpectrumModel
model = SpectrumModel(plasma, atomic_db, instrument, ...)
```

Or specify in config file:

```yaml
atomic_database: libs_production.db
```

## Performance Notes

- **First run**: Very slow (hours) - fetches all data from NIST
- **Subsequent runs**: Fast - uses cached data
- **Cache location**: `nist_levels_cache/` and `nist_ie_cache/` directories
- **Cache size**: Can be several GB

## Future Improvements

Planned enhancements:
- CLI command integration (in progress)
- Progress bars and better status reporting
- Parallel fetching for faster generation
- Incremental updates (add new elements without regenerating)
- Database versioning and validation

## See Also

- [User Guide](User_Guide.md) - How to use the database
- [API Reference](API_Reference.md) - Database API documentation
- `datagen_v2.py` - Source code for database generation

