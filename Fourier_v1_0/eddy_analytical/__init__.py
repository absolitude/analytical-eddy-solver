"""Analytical eddy-current solver for thin conducting coatings."""
__version__ = "0.2.4+grid"
from .physics import MU0, Wire
from .solvers import (
    CircularResult, CircularSolver, calculate_circular_case,
    EllipticalResult, EllipticalSolver, calculate_elliptical_case,
    FlatPipeResult, FlatPipeSolver, calculate_flat_pipe_case,
)

__all__ = [
    "__version__", "MU0", "Wire",
    "CircularResult", "CircularSolver", "calculate_circular_case",
    "EllipticalResult", "EllipticalSolver", "calculate_elliptical_case",
    "FlatPipeResult", "FlatPipeSolver", "calculate_flat_pipe_case",
]
