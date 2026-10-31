"""Soft-body primitive shapes. Each knows how to mesh itself and names its surface regions.

Convention: the body frame origin is the centre of the base, +z up, so a body
"resting on the ground" has its pose at z = 0.

Regions are sets of boundary triangles (outward normals), resolved on the mesh
from the shape's own geometry: ``base``, ``top``, ``side`` and, for hollow
shapes, ``cavity`` (the closed inner surface, where a pressure actuator acts)
and ``outer``.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import Field

from softsim.config.schema import StrictModel
from softsim.config.units import Length
from softsim.errors import ConfigError
from softsim.meshing import (
    TetMesh,
    gmsh_box,
    gmsh_closed_tube,
    gmsh_cylinder,
    gmsh_hollow_sphere,
    structured_box,
    structured_cylinder,
)

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

    def regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        tol = _tol(mesh)
        return _resolve(mesh, {"base": lambda p: p[:, 2] < tol, "top": lambda p: p[:, 2] > self.height - tol,
                               "side": lambda p: np.hypot(p[:, 0], p[:, 1]) > self.radius - tol})


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

    def regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        tol = _tol(mesh)
        hx, hy = self.size[0] / 2, self.size[1] / 2
        return _resolve(mesh, {"base": lambda p: p[:, 2] < tol, "top": lambda p: p[:, 2] > self.size[2] - tol,
                               "side": lambda p: (np.abs(p[:, 0]) > hx - tol) | (np.abs(p[:, 1]) > hy - tol)})


class SoftHollowSphere(StrictModel):
    """Thick spherical shell around a closed spherical cavity; rests on its bottom pole (the
    body origin), centre at z = outer radius. Meshed with gmsh."""

    type: Literal["hollow_sphere"] = "hollow_sphere"
    outer_diameter: PositiveLength
    inner_diameter: PositiveLength

    @property
    def top_height(self) -> float:
        return self.outer_diameter

    @property
    def min_feature(self) -> float:
        return (self.outer_diameter - self.inner_diameter) / 2

    def volume(self) -> float:
        return math.pi / 6 * (self.outer_diameter**3 - self.inner_diameter**3)

    def mesh(self, settings: MeshSettings) -> TetMesh:
        _check_hollow(self.inner_diameter, self.outer_diameter)
        m = gmsh_hollow_sphere(self.inner_diameter / 2, self.outer_diameter / 2, settings.size)
        return TetMesh(m.points + [0.0, 0.0, self.outer_diameter / 2], m.tets)

    def regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        tol = _tol(mesh)
        c = np.array([0.0, 0.0, self.outer_diameter / 2])
        r = lambda p: np.linalg.norm(p - c, axis=1)  # noqa: E731
        return _resolve(mesh, {"cavity": lambda p: r(p) < self.inner_diameter / 2 + tol,
                               "outer": lambda p: r(p) > self.outer_diameter / 2 - tol})


class SoftClosedTube(StrictModel):
    """Cylinder around a closed cylindrical cavity: axis +z, base at the origin, end walls of ``cap``.
    Centred (default), it is a balloon actuator that extends; with the cavity moved off-axis by
    ``cavity_offset`` (x, y), the thin side stretches more and the tube bends away from it (a
    single-material bending actuator). Meshed with gmsh."""

    type: Literal["closed_tube"] = "closed_tube"
    outer_diameter: PositiveLength
    inner_diameter: PositiveLength
    height: PositiveLength
    cap: PositiveLength
    cavity_offset: tuple[Length, Length] = (0.0, 0.0)

    @property
    def top_height(self) -> float:
        return self.height

    @property
    def min_feature(self) -> float:
        thinnest = (self.outer_diameter - self.inner_diameter) / 2 - math.hypot(*self.cavity_offset)
        return min(thinnest, self.cap)

    def volume(self) -> float:
        return math.pi / 4 * (self.outer_diameter**2 * self.height
                              - self.inner_diameter**2 * (self.height - 2 * self.cap))

    def mesh(self, settings: MeshSettings) -> TetMesh:
        _check_hollow(self.inner_diameter, self.outer_diameter)
        return gmsh_closed_tube(self.inner_diameter / 2, self.outer_diameter / 2, self.height, self.cap,
                                settings.size, tuple(self.cavity_offset))

    def regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        tol = _tol(mesh)
        ri, ro, h, t = self.inner_diameter / 2, self.outer_diameter / 2, self.height, self.cap
        ox, oy = self.cavity_offset
        rad = lambda p: np.hypot(p[:, 0], p[:, 1])  # noqa: E731
        inside = lambda p: ((np.hypot(p[:, 0] - ox, p[:, 1] - oy) < ri + tol) & (p[:, 2] > t - tol)  # noqa: E731
                            & (p[:, 2] < h - t + tol))
        return _resolve(mesh, {"cavity": inside, "base": lambda p: p[:, 2] < tol,
                               "top": lambda p: p[:, 2] > h - tol, "side": lambda p: rad(p) > ro - tol})


SoftShape = Annotated[Union[SoftCylinder, SoftBox, SoftHollowSphere, SoftClosedTube], Field(discriminator="type")]


def _tol(mesh: TetMesh) -> float:
    return 1e-6 * float(np.ptp(mesh.points, axis=0).max())


def _resolve(mesh: TetMesh, predicates: dict) -> dict[str, np.ndarray]:
    return {name: mesh.surface_where(pred) for name, pred in predicates.items()}


def _check_hollow(inner: float, outer: float) -> None:
    if inner >= outer:
        raise ConfigError(f"inner diameter {inner} m must be smaller than the outer diameter {outer} m")


def check_resolution(shape, settings: MeshSettings) -> None:
    """Reject meshes so coarse they cannot represent the shape at all."""
    if settings.size > shape.min_feature:
        raise ConfigError(
            f"mesh size {settings.size * 1e3:.2f} mm exceeds the smallest body dimension "
            f"{shape.min_feature * 1e3:.2f} mm"
        )
