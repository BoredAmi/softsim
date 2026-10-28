"""Soft and rigid body geometry primitives."""

from softsim.geometry.rigid import (
    Part,
    Plate,
    RigidBodyDef,
    RigidCylinder,
    RigidShape,
    WeightWithKnob,
    load_rigid_body,
    rigid_presets,
)
from softsim.geometry.soft import MeshSettings, SoftBox, SoftCylinder, SoftShape, check_resolution

__all__ = [
    "MeshSettings",
    "Part",
    "Plate",
    "RigidBodyDef",
    "RigidCylinder",
    "RigidShape",
    "SoftBox",
    "SoftCylinder",
    "SoftShape",
    "WeightWithKnob",
    "check_resolution",
    "load_rigid_body",
    "rigid_presets",
]
