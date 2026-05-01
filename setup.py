"""Compatibility shim for legacy setuptools entry points.

Project metadata now lives in ``pyproject.toml`` and BioNetGen bundle vendoring
is an explicit release step handled by ``scripts/vendor_bionetgen_assets.py``.
"""

from setuptools import setup


if __name__ == "__main__":
    setup()
