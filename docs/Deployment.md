# CF-LIBS Deployment Guide

## Physics-Only Constraint

**HARD CONSTRAINT:** The shipped CF-LIBS algorithm must not import or use `sklearn`, `torch`, `tensorflow`, `keras`, `flax`, `equinox`, `transformers`, `jax.nn`, or `jax.experimental.stax`. Machine learning is allowed only in:
- `cflibs/evolution/` — LLM-driven algorithm optimization (tooling only)
- `cflibs/experimental/ml/` — Quarantined ML modules (deletion candidates)

**Enforcement:**
1. **Ruff TID251 static rule** — `pyproject.toml` bans these APIs from the shipped codebase
2. **AST blocklist scanner** — `cflibs/evolution/evaluator.py` rejects evolved code that violates the ban

See CLAUDE.md for full specifications.

## Environment Setup

CF-LIBS uses `uv` for fast, reliable Python environment management.

### Local Development (Apple Silicon)

For development on MacBooks with M1/M2/M3 chips:

```bash
# Create virtual environment with uv
uv venv --python 3.12

# Install with local development dependencies (includes JAX + Metal)
uv pip install -e ".[local]"

# Note: JAX Metal backend is experimental. Tests run with CPU backend:
JAX_PLATFORMS=cpu pytest tests/
```

**Current limitations:**
- JAX Metal (Apple GPU) backend is experimental in JAX 0.8.x
- Some operations may not be supported on Metal
- CPU backend works reliably for all operations

### Cluster Deployment (NVIDIA CUDA)

**Target cluster specifications:**
- 3 nodes
- Dual Intel Xeon Gold 20-core CPUs per node (40 cores/node, 120 total)
- ~380GB RAM per node (~1.1TB total)
- NVIDIA Tesla V100S GPU per node (3 GPUs total, 32GB HBM2 each)
- InfiniBand interconnect

#### Installation

```bash
# Create environment
uv venv --python 3.12

# Install with CUDA support
uv pip install -e ".[cluster]"
```

This installs:
- `jax[cuda12]` - JAX with CUDA 12 support for V100S GPUs
- `h5py` - HDF5 for manifold storage
- `mpi4py` - MPI for multi-node parallelism

#### JAX Configuration for Multi-GPU

```python
import os

# Force CUDA backend
os.environ['JAX_PLATFORMS'] = 'cuda'

# Set visible devices for multi-GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'  # All 3 V100S GPUs

import jax
print(jax.devices())  # Should show 3 CudaDevice
```

#### Multi-Node with MPI

For manifold generation across all 3 nodes:

```bash
# Run on 3 nodes, 1 GPU each
mpirun -np 3 --hostfile hosts.txt python generate_manifold.py

# Or with SLURM
srun -N 3 --gpus-per-node=1 python generate_manifold.py
```

Example `hosts.txt`:
```
node1 slots=1
node2 slots=1
node3 slots=1
```

#### Memory Considerations

Each V100S has 32GB HBM2. For large manifolds:

```python
# Estimate memory per spectrum point
# Typical: ~4KB per spectrum (1024 pixels × float32)
# 1M spectra = ~4GB

# Recommended batch sizes for V100S:
config = ManifoldConfig(
    batch_size=10000,  # Process 10k spectra per GPU batch
    ...
)
```

#### Performance Expectations

Based on V100S specifications:
- ~15.7 TFLOPS FP32
- ~125 TFLOPS Tensor Core (FP16)
- ~900 GB/s HBM2 bandwidth

Estimated manifold generation rates:
- Single V100S: ~50,000-100,000 spectra/second
- 3× V100S cluster: ~150,000-300,000 spectra/second

## Installation Options Summary

| Extra | Use Case | Includes |
|-------|----------|----------|
| `jax-cpu` | Testing, CI/CD | JAX CPU only |
| `jax-metal` | Apple Silicon dev | JAX + Metal |
| `jax-cuda` | NVIDIA GPUs | JAX + CUDA 12 |
| `hdf5` | Manifold storage | h5py |
| `dev` | Development | pytest, black, mypy, ruff |
| `local` | Full local dev (Mac) | All above + Metal |
| `cluster` | Production cluster | JAX CUDA + h5py + MPI |

## Verification

After installation, verify JAX backend:

```python
import jax
print(f"JAX version: {jax.__version__}")
print(f"Devices: {jax.devices()}")
print(f"Backend: {jax.default_backend()}")
```

Expected output on cluster:
```
JAX version: 0.4.x
Devices: [CudaDevice(id=0), CudaDevice(id=1), CudaDevice(id=2)]
Backend: cuda
```

## Troubleshooting

### JAX Metal not working (Apple Silicon)
```bash
# Fall back to CPU
export JAX_PLATFORMS=cpu
```

### CUDA not detected
```bash
# Check CUDA installation
nvidia-smi

# Verify JAX CUDA
python -c "import jax; print(jax.devices())"
```

### MPI errors on cluster
```bash
# Verify MPI installation
mpirun --version

# Test MPI communication
mpirun -np 3 hostname
```
