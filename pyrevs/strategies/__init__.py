"""Set of sampling strategies for pyREVS."""

from .ams import AMS
from .montecarlo import MonteCarlo

__all__ = ["AMS", "MonteCarlo"]
