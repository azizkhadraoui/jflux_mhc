"""jflux - JAX implementation of Flux diffusion model."""

from jflux.mhc import (
    HyperConnection,
    HyperConnectionManager,
    SimpleHyperConnection,
)

__all__ = [
    "SimpleHyperConnection",
    "HyperConnection", 
    "HyperConnectionManager",
]
