"""Soft-body primitive shapes. Each knows how to mesh itself.

Convention: the body frame origin is the centre of the base, +z up, so a body
"resting on the ground" has its pose at z = 0.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import Field

from softsim.config.schema import StrictModel
from softsim.config.units import Length
from softsim.errors import ConfigError
from softsim.meshing import TetMesh, gmsh_box, gmsh_cylinder, structured_box, structured_cylinder

PositiveLength = Annotated[Length, Field(gt=0)]


class MeshSettings(StrictModel):
    """Mesh resolution. ``size`` is the target tetrahedron edge length."""

    size: PositiveLength = 0.0035
    method: Literal["structured", "gmsh"] = "structured"


class SoftCylinder(StrictModel):
    """Solid cylinder, axis along +z."""

    type: Literal["cylinder"] = "cylinder"
    diameter: PositiveLength
    height: PositiveLength

    @property
    def radius(self) -> float:
        return self.diameter / 2

    @property
    def top_height(self) -> float:
        return self.height

    @property
    def min_feature(self) -> float:
        return min(self.height, self.diameter)

    def volume(self) -> float:
        return math.pi * self.radius**2 * self.height

    def mesh(self, settings: MeshSettings) -> TetMesh:
        if settings.method == "gmsh":
            return gmsh_cylinder(self.radius, self.height, settings.size)
        return structured_cylinder(self.radius, self.height, settings.size)


class SoftBox(StrictModel):
    """Solid box (beams and plates are boxes); ``size`` are full extents x, y, z."""

    type: Literal["box"] = "box"
    size: tuple[PositiveLength, PositiveLength, PositiveLength]

    @property
    def top_height(self) -> float:
        return self.size[2]

    @property
    def min_feature(self) -> float:
        return min(self.size)

    def volume(self) -> float:
        return self.size[0] * self.size[1] * self.size[2]

    def mesh(self, settings: MeshSettings) -> TetMesh:
        if settings.method == "gmsh":
            return gmsh_box(self.size, settings.size)
        return structured_box(self.size, settings.size)


SoftShape = Annotated[Union[SoftCylinder, SoftBox], Field(discriminator="type")]


def check_resolution(shape: SoftCylinder | SoftBox, settings: MeshSettings) -> None:
    """Reject meshes so coarse they cannot represent the shape at all."""
    if settings.size > shape.min_feature:
        raise ConfigError(
            f"mesh size {settings.size * 1e3:.2f} mm exceeds the smallest body dimension "
            f"{shape.min_feature * 1e3:.2f} mm"
        )
