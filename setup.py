"""
Setup script for CF-LIBS package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text() if readme_file.exists() else ""

# Read version from package
version_file = Path(__file__).parent / "cflibs" / "__init__.py"
version = "0.1.0"
if version_file.exists():
    for line in version_file.read_text().splitlines():
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break

setup(
    name="cflibs",
    version=version,
    author="TheFermiSea",
    description="Production-grade computational framework for laser-induced breakdown spectroscopy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheFermiSea/CF-LIBS",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "pandas>=1.3.0",
        "pyyaml>=5.4.0",
    ],
    extras_require={
        "jax": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
        ],
        "hdf5": [
            "h5py>=3.0.0",
        ],
        "widgets": [
            "ipywidgets>=8.0.0",
            "plotly>=5.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "ruff>=0.0.200",
        ],
        "all": [
            "jax>=0.4.0",
            "jaxlib>=0.4.0",
            "h5py>=3.0.0",
            "ipywidgets>=8.0.0",
            "plotly>=5.0.0",
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
            "ruff>=0.0.200",
        ],
    },
    entry_points={
        "console_scripts": [
            "cflibs=cflibs.cli.main:main",
        ],
    },
)
