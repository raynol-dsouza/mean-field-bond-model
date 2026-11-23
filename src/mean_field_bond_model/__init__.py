"""Site-symmetry-aware bond shells and the mean-field bond model."""

import importlib

from . import example_data, hessian, md
from .covariances import BondPairCovarianceCalculator
from .mean_field import (
    MeanFieldBondModel,
    build_bonding_potentials_by_bond_type,
    generate_bond_grids,
)
from .morse import (
    MorsePotential,
    build_bonding_potentials,
    load_morse_params,
    save_morse_params,
)
from .symmetry import NeighborBondSymmetryAnalysis

__all__ = [
    "example_data",
    "hessian",
    "md",
    "BondPairCovarianceCalculator",
    "MeanFieldBondModel",
    "MorsePotential",
    "NeighborBondSymmetryAnalysis",
    "build_bonding_potentials",
    "build_bonding_potentials_by_bond_type",
    "generate_bond_grids",
    "load_morse_params",
    "save_morse_params",
]


def __getattr__(name):
    """Load plot and plot_data on first use, as they need matplotlib."""
    if name in ("plot", "plot_data"):
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
