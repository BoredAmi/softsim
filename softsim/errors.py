"""Exception hierarchy. Everything softsim raises on purpose derives from SoftSimError."""

from __future__ import annotations


class SoftSimError(Exception):
    """Base class for all softsim errors."""


class ConfigError(SoftSimError):
    """A configuration file or value is invalid."""


class UnsupportedFeatureError(SoftSimError):
    """A scene requests features the selected backend cannot simulate faithfully.

    Attributes:
        backend: Name of the backend that rejected the scene.
        missing: Human-readable descriptions of each unsupported requirement.
    """

    def __init__(self, backend: str, missing: list[str]):
        self.backend = backend
        self.missing = missing
        lines = "\n".join(f"  - {m}" for m in missing)
        super().__init__(f"backend '{backend}' cannot simulate this scene:\n{lines}")


class StabilityError(SoftSimError):
    """Simulation settings are known to be unstable or the simulation diverged."""
