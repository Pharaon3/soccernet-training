"""Shim so `pip install -e .` works on older pip that does not fully support pyproject-only projects."""

from setuptools import setup

setup()
