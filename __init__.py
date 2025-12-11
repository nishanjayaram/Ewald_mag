# ewald_spin_ice/__init__.py

"""
Ewald-based stray-field solver for spin-ice modes.

This package exposes:
- ewald_spin_ice.core: main numerical machinery (no hard-coded paths)
- ewald_spin_ice.cli: example/production CLI driver for batch runs
"""

from .core import (
    GeometryConfig,
    EwaldConfig,
    SpinIceEwaldSolver,
    enumerate_ktriples_first_N,
    enumerate_ktriples_first_N_sc,
    enumerate_ktriples_band,
    enumerate_ktriples_nlist,
    make_tasks,
)

__all__ = [
    "GeometryConfig",
    "EwaldConfig",
    "SpinIceEwaldSolver",
    "enumerate_ktriples_first_N",
    "enumerate_ktriples_first_N_sc",
    "enumerate_ktriples_band",
    "enumerate_ktriples_nlist",
    "make_tasks",
]
