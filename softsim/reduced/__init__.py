"""Fast fingers: a chain of links whose joints follow tables taken from the reference solver (docs/gripper.md)."""

from softsim.reduced.finger import ReducedFinger, ReducedFingerConfig, extract_tables
from softsim.reduced.mjcf import ReducedGripper

__all__ = ["ReducedFinger", "ReducedFingerConfig", "ReducedGripper", "extract_tables"]
