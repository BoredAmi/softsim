"""The 3D reference solver as a backend (slow, locking-free); see :class:`FemBackend`."""

from softsim.backends.fem.backend import FemBackend

__all__ = ["FemBackend"]
