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
from pathlib import Path
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import Field, PrivateAttr, model_validator

from softsim.config.schema import StrictModel
from softsim.config.units import Length
from softsim.errors import ConfigError
from softsim.meshing import (
    TetMesh,
    gmsh_box,
    gmsh_cable_finger,
    gmsh_closed_tube,
    gmsh_box_union,
    gmsh_cylinder,
    gmsh_hollow_sphere,
    gmsh_pneunet,
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

    def export(self, path) -> None:
        """Write the shape as CAD (``.step``, ``.brep``...; metres, body frame)."""
        gmsh_cylinder(self.radius, self.height, 1.0, export=path)

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

    def export(self, path) -> None:
        """Write the shape as CAD (``.step``, ``.brep``...; metres, body frame)."""
        gmsh_box(self.size, 1.0, export=path)

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

    def export(self, path) -> None:
        """Write the shape as CAD (``.step``, ``.brep``...; metres, body frame)."""
        _check_hollow(self.inner_diameter, self.outer_diameter)
        gmsh_closed_tube(self.inner_diameter / 2, self.outer_diameter / 2, self.height, self.cap, 1.0,
                         tuple(self.cavity_offset), export=path)

    def regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        tol = _tol(mesh)
        ri, ro, h, t = self.inner_diameter / 2, self.outer_diameter / 2, self.height, self.cap
        ox, oy = self.cavity_offset
        rad = lambda p: np.hypot(p[:, 0], p[:, 1])  # noqa: E731
        inside = lambda p: ((np.hypot(p[:, 0] - ox, p[:, 1] - oy) < ri + tol) & (p[:, 2] > t - tol)  # noqa: E731
                            & (p[:, 2] < h - t + tol))
        return _resolve(mesh, {"cavity": inside, "base": lambda p: p[:, 2] < tol,
                               "top": lambda p: p[:, 2] > h - tol, "side": lambda p: rad(p) > ro - tol})


class SoftPneuNet(StrictModel):
    """PneuNet bending actuator (finger): a block along +x, base face at x = 0, ``chambers`` closed
    chambers in its upper part, joined by a channel just above the solid base slab, so they form one
    cavity. Pressure expands the chambers' thin walls; the base slab (and a stiffer ``layer`` at its
    bottom, given as a material zone on the ``layer`` volume region) resists stretching, so the finger
    curls towards -z. Meshed with gmsh; with ``layer`` > 0 the mesh conforms to z = layer.

    Regions: ``cavity``, ``base`` (x = 0), ``tip`` (x = length), ``bottom``, ``top``, ``side``.
    Volume regions: ``layer`` (z < layer), ``body`` (the rest)."""

    type: Literal["pneunet"] = "pneunet"
    chambers: int = Field(ge=1)
    chamber_length: PositiveLength           # along x
    wall: PositiveLength                     # between chambers
    end_wall: PositiveLength                 # at both ends
    width: PositiveLength                    # y
    side_wall: PositiveLength
    height: PositiveLength                   # z
    base: PositiveLength                     # solid slab below the chambers
    top_wall: PositiveLength
    channel_width: PositiveLength
    channel_height: PositiveLength           # from the slab up
    layer: Length = 0.0                      # strain-limiting layer thickness (in the slab), 0 = none

    @model_validator(mode="after")
    def _consistent(self) -> "SoftPneuNet":
        if not (self.layer < self.base and self.base + self.top_wall < self.height
                and self.channel_height < self.height - self.base - self.top_wall
                and self.channel_width < self.width - 2 * self.side_wall and 2 * self.side_wall < self.width):
            raise ValueError("pneunet: need layer < base, base + top_wall < height, a channel lower than the "
                             "chambers and narrower than them, and side walls thinner than half the width")
        return self

    @property
    def length(self) -> float:
        return 2 * self.end_wall + self.chambers * self.chamber_length + (self.chambers - 1) * self.wall

    @property
    def top_height(self) -> float:
        return self.height

    @property
    def min_feature(self) -> float:
        return min(self.wall, self.end_wall, self.side_wall, self.top_wall, self.base - self.layer,
                   *([self.layer] if self.layer > 0 else []))

    def _chambers(self) -> list[tuple[float, float]]:
        pitch = self.chamber_length + self.wall
        return [(self.end_wall + i * pitch, self.end_wall + i * pitch + self.chamber_length)
                for i in range(self.chambers)]

    def volume(self) -> float:
        cy = self.width / 2 - self.side_wall
        ch = self.height - self.base - self.top_wall
        n_gaps = self.chambers - 1
        return (self.length * self.width * self.height - self.chambers * self.chamber_length * 2 * cy * ch
                - n_gaps * self.wall * self.channel_width * self.channel_height)

    def mesh(self, settings: MeshSettings) -> TetMesh:
        return self._gmsh(settings.size)

    def export(self, path) -> None:
        """Write the shape as CAD (``.step``, ``.brep``...; metres, body frame); with a ``layer``, as two
        solids (the layer and the rest)."""
        self._gmsh(1.0, export=path)

    def _gmsh(self, size: float, export=None):
        chambers = self._chambers()
        return gmsh_pneunet(self.length, self.width, self.height, chambers, self.width / 2 - self.side_wall,
                            (self.base, self.height - self.top_wall),
                            (chambers[0][0], chambers[-1][1], self.channel_width / 2, self.base + self.channel_height),
                            self.layer, size, export=export)

    def regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        tol = _tol(mesh)
        length, hw, h = self.length, self.width / 2, self.height
        inside = lambda p: ((p[:, 0] > tol) & (p[:, 0] < length - tol) & (np.abs(p[:, 1]) < hw - tol)  # noqa: E731
                            & (p[:, 2] > tol) & (p[:, 2] < h - tol))
        return _resolve(mesh, {"cavity": inside, "base": lambda p: p[:, 0] < tol,
                               "tip": lambda p: p[:, 0] > length - tol, "bottom": lambda p: p[:, 2] < tol,
                               "top": lambda p: p[:, 2] > h - tol, "side": lambda p: np.abs(p[:, 1]) > hw - tol})

    def volume_regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        below = mesh.points[mesh.tets].mean(axis=1)[:, 2] < self.layer
        return {"layer": np.nonzero(below)[0], "body": np.nonzero(~below)[0]}


class SoftRibbedFinger(StrictModel):
    """Cast pneumatic finger with a ribbed roof (Wang et al., Actuators 15(9), 488, 2026: the geometry of
    its Abaqus mesh). A block along +x, base face at x = 0, width along y, height along z, built from
    boxes, so the shape is exact:

    - one cavity: a manifold ``manifold_height`` high above a ``floor`` slab, ``wall`` in from the
      two ends and ``side_wall`` in from the sides, with a pocket ``chamber_height`` higher over each
      of the ``chambers`` chambers (``chamber_length`` long, ``chamber_spacing`` apart);
    - a bump over every chamber (its pocket plus ``wall`` on both sides, ``top_wall`` over the pocket) and
      a ``roof`` over the manifold between them;
    - ``teeth`` under the floor, ``tooth_height`` high and ``tooth_width`` wide every ``tooth_pitch``
      (``tooth_height`` 0 omits them: a plain floor, much cheaper to mesh);
    - a ``tip_length`` slab, ``tip_thickness`` thick on top of the teeth's level, past the last wall: the
      distal reserve of the cast, with no cavity.

    The pressurised roof stretches while the floor resists, so the finger curls towards -z. The bottom
    of the floor slab is z = ``tooth_height``; the body's lowest point is z = 0.

    Regions: ``cavity``, ``base`` (x = 0), ``tip`` (x = length), ``bottom``, ``top``, ``side``."""

    type: Literal["ribbed_finger"] = "ribbed_finger"
    chambers: int = Field(default=7, ge=1)
    chamber_length: PositiveLength = 0.005       # along x
    chamber_spacing: PositiveLength = 0.009      # between chambers (the manifold's roof span)
    wall: PositiveLength = 0.002                 # at both ends and beside every chamber (x)
    side_wall: PositiveLength = 0.002            # y
    width: PositiveLength = 0.024                # y
    floor: PositiveLength = 0.002                # the constraint slab under the manifold
    manifold_height: PositiveLength = 0.005
    chamber_height: PositiveLength = 0.005       # pocket height above the manifold
    roof: PositiveLength = 0.003                 # over the manifold between the chambers
    top_wall: PositiveLength = 0.003             # over the chambers
    tooth_height: Length = 0.001
    tooth_width: PositiveLength = 0.003
    tooth_pitch: PositiveLength = 0.005
    tip_length: PositiveLength = 0.010
    tip_thickness: PositiveLength = 0.003

    @model_validator(mode="after")
    def _consistent(self) -> "SoftRibbedFinger":
        if 2 * self.side_wall >= self.width:
            raise ValueError("ribbed_finger: side walls thinner than half the width")
        if self.tooth_height > 0 and self.tooth_width >= self.tooth_pitch:
            raise ValueError("ribbed_finger: tooth_width must be below tooth_pitch")
        return self

    @property
    def body_length(self) -> float:
        """The cast part, up to the tip slab."""
        return 2 * self.wall + self.chambers * self.chamber_length + (self.chambers - 1) * self.chamber_spacing

    @property
    def length(self) -> float:
        return self.body_length + self.tip_length

    @property
    def top_height(self) -> float:
        return self.tooth_height + self.floor + self.manifold_height + self.chamber_height + self.top_wall

    @property
    def min_feature(self) -> float:
        return min(self.wall, self.side_wall, self.floor, self.roof, self.top_wall, self.tip_thickness)

    def _pockets(self) -> list[tuple[float, float]]:
        pitch = self.chamber_length + self.chamber_spacing
        return [(self.wall + i * pitch, self.wall + i * pitch + self.chamber_length) for i in range(self.chambers)]

    @property
    def n_teeth(self) -> int:
        return int((self.body_length - self.tooth_width) / self.tooth_pitch + 1e-9) + 1 if self.tooth_height > 0 else 0

    def _levels(self) -> tuple[float, float, float, float, float]:
        """z of the floor's bottom, the manifold's floor and roof, the pocket's roof, the bump's top."""
        z0 = self.tooth_height
        z1 = z0 + self.floor
        z2 = z1 + self.manifold_height
        z3 = z2 + self.chamber_height
        return z0, z1, z2, z3, z3 + self.top_wall

    def _boxes(self) -> tuple[list, list]:
        hw = self.width / 2
        z0, z1, z2, z3, z4 = self._levels()
        valley_top = z2 + self.roof
        solids = [(0.0, -hw, z0, self.body_length, hw, valley_top),
                  (self.body_length, -hw, z0, self.length, hw, z0 + self.tip_thickness)]
        solids += [(a - self.wall, -hw, valley_top, b + self.wall, hw, z4) for a, b in self._pockets()]
        solids += [(i * self.tooth_pitch, -hw, 0.0, i * self.tooth_pitch + self.tooth_width, hw, z0)
                   for i in range(self.n_teeth)]
        cy = hw - self.side_wall
        cavities = [(self.wall, -cy, z1, self.body_length - self.wall, cy, z2)]
        cavities += [(a, -cy, z2, b, cy, z3) for a, b in self._pockets()]
        return solids, cavities

    def volume(self) -> float:
        z0, _, z2, _, z4 = self._levels()
        valley_top = z2 + self.roof
        cy = self.width - 2 * self.side_wall
        body = (self.body_length * self.width * (valley_top - z0) + self.tip_length * self.width * self.tip_thickness
                + self.chambers * (self.chamber_length + 2 * self.wall) * self.width * (z4 - valley_top)
                + self.n_teeth * self.tooth_width * self.width * z0)
        cavity = ((self.body_length - 2 * self.wall) * cy * self.manifold_height
                  + self.chambers * self.chamber_length * cy * self.chamber_height)
        return body - cavity

    def mesh(self, settings: MeshSettings) -> TetMesh:
        return self._gmsh(settings.size)

    def export(self, path) -> None:
        """Write the shape as CAD (``.step``, ``.brep``...; metres, body frame)."""
        self._gmsh(1.0, export=path)

    def _gmsh(self, size: float, export=None):
        solids, cavities = self._boxes()
        return gmsh_box_union(solids, cavities, size, export=export)

    def regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        tol = _tol(mesh)
        _, cavities = self._boxes()

        def cavity(p):  # the cavity's walls are at least ``wall`` from every outer face
            inside = np.zeros(len(p), bool)
            for x0, y0, z0, x1, y1, z1 in cavities:
                inside |= ((p[:, 0] > x0 - tol) & (p[:, 0] < x1 + tol) & (p[:, 1] > y0 - tol) & (p[:, 1] < y1 + tol)
                           & (p[:, 2] > z0 - tol) & (p[:, 2] < z1 + tol))
            return inside

        hw = self.width / 2
        return _resolve(mesh, {"cavity": cavity, "base": lambda p: p[:, 0] < tol,
                               "tip": lambda p: p[:, 0] > self.length - tol, "bottom": lambda p: p[:, 2] < self.tooth_height + tol,
                               "top": lambda p: p[:, 2] > self.top_height - tol,
                               "side": lambda p: np.abs(p[:, 1]) > hw - tol})


class SoftCableFinger(StrictModel):
    """Tendon-driven finger: a block (``width`` x ``depth`` x ``length``, base at z = 0, centred on the z axis)
    cut into ``segments`` links by ``segments - 1`` transverse slots, ``slot_width`` wide (along z) and
    ``slot_depth`` deep (along -x) in its +x face. The tendon runs along z at x = ``width / 2 -
    tendon_offset``, through the web the slots leave (``tendon_offset > slot_depth``), and pulls the +x
    side shut: the finger curls towards +x, the slots closing. ``cable: path: geometry`` takes the tendon's
    path (:meth:`tendon_path`) from here; add ``anchor: top`` to spread its end force over the tip face.
    With one segment the finger is a plain beam. Meshed with gmsh.

    Regions: ``base`` (z = 0), ``top`` (z = length), ``side`` (|y| = depth / 2), ``cable_side`` (the +x
    face and the slots' walls)."""

    type: Literal["cable_finger"] = "cable_finger"
    length: PositiveLength
    width: PositiveLength                     # x: the bending direction
    depth: PositiveLength                     # y
    segments: int = Field(default=1, ge=1)
    slot_width: PositiveLength | None = None  # required with more than one segment
    slot_depth: PositiveLength | None = None
    tendon_offset: PositiveLength             # from the +x face to the tendon

    @model_validator(mode="after")
    def _consistent(self) -> "SoftCableFinger":
        if self.segments > 1:
            if self.slot_width is None or self.slot_depth is None:
                raise ValueError("cable_finger: more than one segment needs slot_width and slot_depth")
            if self.slot_width >= self.length / self.segments:
                raise ValueError("cable_finger: slot_width must be below the segment pitch length / segments")
            if self.slot_depth >= self.tendon_offset:
                raise ValueError("cable_finger: the tendon must run through the web: tendon_offset > slot_depth")
        if self.tendon_offset >= self.width / 2:
            raise ValueError("cable_finger: the tendon must lie on the +x side of the axis (tendon_offset < width / 2)")
        return self

    @property
    def top_height(self) -> float:
        return self.length

    @property
    def tendon_x(self) -> float:
        return self.width / 2 - self.tendon_offset

    def tendon_path(self) -> list[tuple[float, float, float]]:
        """Corner points of the tendon from the base to the tip, body frame."""
        return [(self.tendon_x, 0.0, 0.0), (self.tendon_x, 0.0, self.length)]

    def slots(self) -> list[tuple[float, float]]:
        """The slots' extents (z0, z1), centred on the segment boundaries."""
        if self.segments == 1:
            return []
        pitch = self.length / self.segments
        return [(i * pitch - self.slot_width / 2, i * pitch + self.slot_width / 2) for i in range(1, self.segments)]

    @property
    def min_feature(self) -> float:
        return min(self.width - (self.slot_depth or 0.0), self.depth, *([self.slot_width] if self.segments > 1 else []))

    def volume(self) -> float:
        return self.length * self.width * self.depth - len(self.slots()) * (self.slot_width or 0.0) * (
            self.slot_depth or 0.0) * self.depth

    def mesh(self, settings: MeshSettings) -> TetMesh:
        return self._gmsh(settings.size)

    def export(self, path) -> None:
        """Write the shape as CAD (``.step``, ``.brep``...; metres, body frame)."""
        self._gmsh(1.0, export=path)

    def _gmsh(self, size: float, export=None):
        return gmsh_cable_finger(self.width, self.depth, self.length, self.slots(), self.slot_depth or 0.0, size,
                                 export=export)

    def regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        tol = _tol(mesh)
        hx, hy = self.width / 2, self.depth / 2
        depth = self.slot_depth or 0.0
        slots = self.slots()

        def cable_side(p):  # the +x face, or a slot's floor (x = hx - depth) / flank (a slot's z-limits)
            on_face = p[:, 0] > hx - tol
            on_floor = np.zeros(len(p), bool)
            for z0, z1 in slots:
                on_floor |= (np.abs(p[:, 0] - (hx - depth)) < tol) & (p[:, 2] > z0 - tol) & (p[:, 2] < z1 + tol)
                on_floor |= ((np.abs(p[:, 2] - z0) < tol) | (np.abs(p[:, 2] - z1) < tol)) & (p[:, 0] > hx - depth - tol)
            return on_face | on_floor

        return _resolve(mesh, {"base": lambda p: p[:, 2] < tol, "top": lambda p: p[:, 2] > self.length - tol,
                               "side": lambda p: np.abs(p[:, 1]) > hy - tol, "cable_side": cable_side})


Box = tuple[tuple[Length, Length, Length], tuple[Length, Length, Length]]


class SoftMeshFile(StrictModel):
    """A body from a file (softsim.meshing.io): CAD (STEP, IGES, BREP) or a closed STL or OBJ surface
    meshed at ``mesh.size`` (its own triangles are kept with ``remesh: false``), a volume mesh (gmsh
    ``.msh``, ``.vtk``, Medit ``.mesh``; used as it is) or a gmsh ``.geo`` script. ``path`` is
    relative to the config file. Files without units are multiplied by ``scale`` (STL from CAD:
    usually 1e-3, millimetres); STEP and IGES carry theirs.

    Placement, in order: ``rotate`` (degrees about x, then y, then z), ``center: base`` (the
    bounding box's bottom centre to the origin, the convention of the other shapes), then
    ``translate``.

    Regions: the file's 2D physical groups (boundary triangles only), ``surface`` (the whole
    boundary), ``cavity`` (``cavity_1``, ``cavity_2``... if several, largest first) for every closed
    inner shell, and ``box_regions`` given here as boxes (boundary triangles with all three vertices
    inside, after placement). Volume regions: the file's 3D physical groups (else ``volume_<tag>`` for each of several
    volumes) and ``body`` (all)."""

    type: Literal["mesh_file"] = "mesh_file"
    path: str
    scale: Annotated[float, Field(gt=0)] = 1.0
    remesh: bool = True                        # STL, OBJ: remesh at mesh.size (false: keep its triangles)
    feature_angle: Annotated[float, Field(gt=0, lt=180)] = 40.0  # STL: edges sharper than this stay
    rotate: tuple[float, float, float] = (0.0, 0.0, 0.0)
    center: Literal["none", "base"] = "none"
    translate: tuple[Length, Length, Length] = (0.0, 0.0, 0.0)
    box_regions: dict[str, Box] = {}

    _cache: dict = PrivateAttr(default_factory=dict)
    _base_dir: Path | None = PrivateAttr(default=None)

    def bind(self, base_dir) -> None:
        """Resolve a relative ``path`` against ``base_dir`` (the config file's folder)."""
        self._base_dir = Path(base_dir)

    @property
    def file(self) -> Path:
        path = Path(self.path).expanduser()
        return path if path.is_absolute() or self._base_dir is None else (self._base_dir / path).resolve()

    def _load(self, settings: MeshSettings):
        from softsim.meshing.io import load_mesh_file

        key = (settings.size, settings.method)
        if key not in self._cache:
            f = load_mesh_file(self.file, settings.size, self.scale,
                               remesh=self.remesh and self.file.suffix.lower() in (".stl", ".obj"),
                               feature_angle=self.feature_angle)
            x = f.mesh.points @ _rotation(self.rotate).T
            if self.center == "base":
                lo, hi = x.min(axis=0), x.max(axis=0)
                x = x - [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]
            mesh = TetMesh(x + np.asarray(self.translate), f.mesh.tets)
            self._cache.clear()
            self._cache[key] = (mesh, f.surfaces, f.volumes)
            self._cache["last"] = mesh
        return self._cache[key]

    @property
    def _mesh(self) -> TetMesh:
        if "last" not in self._cache:
            self._load(MeshSettings(method="gmsh"))
        return self._cache["last"]

    @property
    def top_height(self) -> float:
        return float(self._mesh.points[:, 2].max())

    @property
    def min_feature(self) -> float:
        return math.inf  # unknown before meshing; the mesh quality check covers the result

    def volume(self) -> float:
        return self._mesh.volume()

    def mesh(self, settings: MeshSettings) -> TetMesh:
        return self._load(settings)[0]

    def _files(self, mesh: TetMesh):
        for key, value in self._cache.items():
            if key != "last" and value[0] is mesh:
                return value
        raise ConfigError("regions of a mesh_file shape: pass the mesh its mesh() returned")

    def regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        from softsim.meshing.io import surface_cavities

        _, surfaces, _ = self._files(mesh)
        boundary = mesh.surface_triangles()
        keys = {tuple(k): i for i, k in enumerate(np.sort(boundary, axis=1))}
        out = {}
        for name, tris in surfaces.items():
            rows = [keys[k] for k in map(tuple, np.sort(tris, axis=1)) if k in keys]
            if rows:
                out[name] = boundary[rows]
        out["surface"] = boundary
        cavities = surface_cavities(mesh)
        for i, c in enumerate(cavities):
            out.setdefault("cavity" if len(cavities) == 1 else f"cavity_{i + 1}", c)
        tol = _tol(mesh)
        for name, (lo, hi) in self.box_regions.items():
            lo, hi = np.asarray(lo) - tol, np.asarray(hi) + tol
            out[name] = mesh.surface_where(lambda p: np.all((p >= lo) & (p <= hi), axis=1))
            if not len(out[name]):
                raise ConfigError(f"mesh_file region {name!r}: no boundary triangle inside its box")
        return out

    def volume_regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        _, _, volumes = self._files(mesh)
        return {**volumes, "body": np.arange(mesh.n_tets)}


def _rotation(degrees: tuple[float, float, float]) -> np.ndarray:
    """Rotation about x, then y, then z (fixed axes), angles in degrees."""
    r = np.eye(3)
    for axis, a in enumerate(np.radians(degrees)):
        c, s = math.cos(a), math.sin(a)
        i, j = [k for k in range(3) if k != axis]
        m = np.eye(3)
        m[i, i], m[i, j], m[j, i], m[j, j] = c, -s, s, c
        r = m @ r
    return r


def bind_files(model, base_dir) -> None:
    """Bind every file-backed shape inside the config ``model`` to ``base_dir``."""
    if hasattr(model, "bind"):
        model.bind(base_dir)
    elif isinstance(model, StrictModel):
        for name in type(model).model_fields:
            bind_files(getattr(model, name), base_dir)
    elif isinstance(model, (list, tuple)):
        for item in model:
            bind_files(item, base_dir)
    elif isinstance(model, dict):
        for item in model.values():
            bind_files(item, base_dir)


class SoftGripper(StrictModel):
    """Two opposed ribbed fingers (:class:`SoftRibbedFinger`) hanging from a palm at z = ``height``, meshed as one
    soft body with two components: the reference solver takes one body, and one air line feeds both cavities.

    The right finger's axis runs down from (``gap`` / 2 + its thickness, 0, ``height``); its toothed face is the
    plane x = ``gap`` / 2 and it curls towards x = 0. The left finger is the right one turned 180 degrees about
    z, a proper rotation, so tetrahedra stay positive and both fingers keep y as their width axis (``plane_strain:
    y`` holds for the pair). Fingers bend towards each other but do not collide with each other.

    Regions: ``cavity`` (both, closed), ``base``, ``base_right``, ``base_left``, ``tips``, ``pads`` (the toothed
    faces), ``backs``, ``side``. Moving ``base_right`` and ``base_left`` as separate prescribed regions gives each
    finger's reaction: its x component is that finger's squeeze on the object."""

    type: Literal["gripper"] = "gripper"
    finger: SoftRibbedFinger = SoftRibbedFinger()
    gap: PositiveLength = 0.05       # between the toothed faces at rest
    height: PositiveLength = 0.12    # of the palm above the table (inflated fingers lengthen: leave room)

    @model_validator(mode="after")
    def _clears_the_table(self) -> "SoftGripper":
        if self.height <= self.finger.length:
            raise ValueError(f"gripper: height {self.height} m puts the {self.finger.length} m fingers through the table")
        return self

    @property
    def min_feature(self) -> float:
        return self.finger.min_feature

    @property
    def top_height(self) -> float:
        return self.height

    def volume(self) -> float:
        return 2 * self.finger.volume()

    def mesh(self, settings: MeshSettings) -> TetMesh:
        one = self.finger.mesh(settings)
        x, y, z = one.points.T
        right = np.column_stack([self.gap / 2 + z, y, self.height - x])  # finger x -> -z, finger z -> +x: a rotation
        left = right * (-1.0, -1.0, 1.0)                                # 180 degrees about z
        return TetMesh(np.vstack([right, left]), np.vstack([one.tets, one.tets + one.n_points]))

    def regions(self, mesh: TetMesh) -> dict[str, np.ndarray]:
        tol, f = _tol(mesh), self.finger
        _, cavities = f._boxes()

        def local(p):  # finger coordinates of either finger (they are symmetric in y)
            return self.height - p[:, 2], np.abs(p[:, 1]), np.abs(p[:, 0]) - self.gap / 2

        def cavity(p):
            x, y, z = local(p)
            inside = np.zeros(len(p), bool)
            for x0, _, z0, x1, y1, z1 in cavities:
                inside |= (x > x0 - tol) & (x < x1 + tol) & (y < y1 + tol) & (z > z0 - tol) & (z < z1 + tol)
            return inside

        return _resolve(mesh, {"cavity": cavity, "base": lambda p: local(p)[0] < tol,
                               "base_right": lambda p: (local(p)[0] < tol) & (p[:, 0] > 0),
                               "base_left": lambda p: (local(p)[0] < tol) & (p[:, 0] < 0),
                               "tips": lambda p: local(p)[0] > f.length - tol,
                               "pads": lambda p: local(p)[2] < f.tooth_height + tol,
                               "backs": lambda p: local(p)[2] > f.top_height - tol,
                               "side": lambda p: local(p)[1] > f.width / 2 - tol})


SoftShape = Annotated[Union[SoftCylinder, SoftBox, SoftHollowSphere, SoftClosedTube, SoftPneuNet, SoftRibbedFinger, SoftCableFinger,
                            SoftGripper, SoftMeshFile],
                      Field(discriminator="type")]


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
