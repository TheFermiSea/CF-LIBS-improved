"""Compatibility shim for legacy setuptools entry points.

Packaging metadata lives in ``pyproject.toml``. Keep this file minimal so
editable installs via older tooling do not drift from the canonical config.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
