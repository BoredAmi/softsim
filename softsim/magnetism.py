"""The applied magnetic field of a scene (``magnetic_field:`` in a scenario config): a uniform part and permanent magnets.

    magnetic_field:
      uniform: [0, 0, 20 kA/m]                          # H, constant; or a schedule, linear between waypoints:
      # uniform: [[0 s, [0, 0, 0]], [0.5 s, [0, 0, 20 kA/m]]]
      magnets:
        - name: ring
          remanence: 1.23 T                             # Br; magnetised along ``axis``
          diameter: 20 mm
          hole: 4.2 mm                                  # inner diameter (0: a disc)
          thickness: 5 mm
          position: [0, 0, -12 mm]                      # centre at rest [m], world frame
          axis: [0, 1, 0]
          path: [[0 s, [0, 0, 0]], [0.2 s, [0, 0, 12 mm]]]  # optional: displacement from ``position``, minimum jerk
          solid: true                                   # optional: the soft body touches it (reference backend)
          edge: 0.2 mm                                  # optional: its edges' chamfer (default 0.2 mm), see below

A magnet is an axially magnetised cylinder (a ring is a disc minus its hole), its field exact outside it
(softsim.fem.magnetic.cylinder_field), magnetised uniformly at ``Br / mu0``: rare-earth magnets' recoil permeability
is about 1.05, so the field of a real one is within a few percent of that. The field is applied, not solved for: bodies
of ``magneto_active`` material take it up (softsim.materials.models.MagnetoActive) and feel its torque and pull; they do
not change it, and the magnets feel no force back. A ``solid`` magnet is also an obstacle: its outer cylinder (the hole
filled in) touches the soft body with the body's friction, fixed or moving along its path, like a rigid body that cannot
be pushed.

A sharp-edged magnet's field is infinite on the rims of its end faces (logarithmically), and a soft body drawn onto a
magnet goes there: Newton then fails. Real magnets have chamfers, and ``edge`` gives the field and a solid magnet's
contact surface one: the field near those rims is softened, finite on them, and changes little farther away (for
0.2 mm: 4.8% at 0.5 mm from a rim, 1.7% at 1 mm, 0.6% at 2 mm, 0.4% mid-way along the side; tests/test_magnet_field.py),
and the contact rounds the edges by ``edge``. The default 0.2 mm is the small end of a
sintered magnet's chamfer; on Bernat et al.'s rolling strip it lowers the torque by 3.3% (0.1 mm: 1.0%).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Union

import numpy as np
from pydantic import Field, field_validator, model_validator

from softsim.config.schema import StrictModel
from softsim.config.units import FieldStrength, FluxDensity, Length, Time
from softsim.scene import MountMotion

MU0 = 4e-7 * np.pi  # H/m
FIELD_SIZE = 9  # uniform H (3) [A/m], dipole moment (3) [A m^2], dipole position (3) [m]
CYLINDER_SIZE = 10  # per cylinder after those: centre (3) [m], unit axis (3), magnetisation along it [A/m], radius,
#                    half-length, edge softening [m] (softsim.fem.magnetic.cylinder_field)
EDGE = 2e-4  # m: a magnet's default edge (chamfer), see PermanentMagnetConfig.edge

Vector = tuple[float, float, float]
FieldVector = tuple[FieldStrength, FieldStrength, FieldStrength]
Position = tuple[Length, Length, Length]


def field_parameters(uniform=(0.0, 0.0, 0.0), dipole=(0.0, 0.0, 0.0), dipole_at=(0.0, 0.0, 0.0), cylinders=()) -> np.ndarray:
    """The parameter vector softsim.fem.magnetic.field reads: ``cylinders`` are ``(centre, axis, magnetisation, radius,
    length[, edge])``, in m and A/m (a magnet of remanence ``Br`` has ``Br / mu0``); the axis need not be unit length, and a
    negative magnetisation with the hole's radius and the ring's length cuts the hole out of a ring magnet (:func:`ring`).
    ``edge`` (default 0: the exact, sharp-edged field) softens the field within about that distance of the end faces' rims,
    as a chamfer does (softsim.fem.magnetic.cylinder_field)."""
    q = [np.asarray(uniform, float), np.asarray(dipole, float), np.asarray(dipole_at, float)]
    for centre, axis, m, radius, length, *edge in cylinders:
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        q += [np.asarray(centre, float), axis, np.array([m, radius, 0.5 * length, edge[0] if edge else 0.0], float)]
    return np.concatenate(q)


def ring(centre, axis, remanence: float, outer: float, inner: float, length: float, edge: float = 0.0) -> list[tuple]:
    """An axially magnetised ring magnet [T, m] as :func:`field_parameters`' cylinders: the full disc and its hole, reversed;
    ``edge`` softens both cylinders' edges."""
    m = remanence / MU0
    return [(centre, axis, m, outer, length, edge)] + ([(centre, axis, -m, inner, length, edge)] if inner > 0 else [])


@dataclass(frozen=True)
class PermanentMagnet:
    """An axially magnetised ring or disc (``inner_radius`` 0), centred at ``position`` [m] at rest, moved by ``path``."""

    name: str
    remanence: float  # T
    outer_radius: float  # m
    inner_radius: float  # m
    length: float  # m, along the axis
    position: Vector
    axis: Vector
    path: MountMotion | None = None
    solid: bool = False
    edge: float = EDGE  # m

    def centre(self, t: float) -> np.ndarray:
        c = np.asarray(self.position, float)
        return c + self.path.position(t) if self.path is not None else c

    def cylinders(self, t: float) -> list[tuple]:
        return ring(self.centre(t), self.axis, self.remanence, self.outer_radius, self.inner_radius, self.length, self.edge)


@dataclass(frozen=True)
class MagneticField:
    """The scene's applied field: uniform ``H`` [A/m] linear between ``(times, values)`` (held outside them), plus magnets."""

    times: tuple[float, ...] = (0.0,)
    values: tuple[Vector, ...] = ((0.0, 0.0, 0.0),)
    magnets: tuple[PermanentMagnet, ...] = ()

    def uniform(self, t: float) -> np.ndarray:
        v = np.asarray(self.values, float)
        return np.array([np.interp(t, self.times, v[:, k]) for k in range(3)])

    def parameters(self, t: float) -> np.ndarray:
        """softsim.fem.magnetic.field's parameters at time ``t``."""
        return field_parameters(uniform=self.uniform(t), cylinders=[c for m in self.magnets for c in m.cylinders(t)])

    @property
    def active(self) -> bool:
        return bool(self.magnets) or bool(np.any(np.asarray(self.values) != 0))


class PermanentMagnetConfig(StrictModel):
    """See the module docstring."""

    name: str = "magnet"
    remanence: Annotated[FluxDensity, Field(gt=0)]
    diameter: Annotated[Length, Field(gt=0)]
    hole: Annotated[Length, Field(ge=0)] = 0.0
    thickness: Annotated[Length, Field(gt=0)]
    position: Position = (0.0, 0.0, 0.0)
    axis: Vector = (0.0, 0.0, 1.0)
    path: list[tuple[Annotated[Time, Field(ge=0)], Position]] = []
    solid: bool = False
    edge: Annotated[Length, Field(ge=0)] = EDGE

    @field_validator("axis")
    @classmethod
    def _nonzero(cls, v: Vector) -> Vector:
        if not np.linalg.norm(v) > 0:
            raise ValueError("axis must be non-zero")
        return v

    @field_validator("hole")
    @classmethod
    def _hole(cls, v: float, info) -> float:
        if "diameter" in info.data and v >= info.data["diameter"]:
            raise ValueError("hole must be smaller than diameter")
        return v

    def build(self) -> PermanentMagnet:
        path = MountMotion(tuple(t for t, _ in self.path), tuple(tuple(p) for _, p in self.path)) if self.path else None
        return PermanentMagnet(self.name, self.remanence, 0.5 * self.diameter, 0.5 * self.hole, self.thickness,
                               tuple(self.position), tuple(self.axis), path, self.solid, self.edge)


class MagneticFieldConfig(StrictModel):
    """See the module docstring."""

    uniform: Union[FieldVector, list[tuple[Annotated[Time, Field(ge=0)], FieldVector]]] = (0.0, 0.0, 0.0)
    magnets: list[PermanentMagnetConfig] = []

    @field_validator("uniform")
    @classmethod
    def _increasing(cls, v):
        if isinstance(v, list):
            if not v:
                raise ValueError("a uniform-field schedule needs at least one waypoint")
            if np.any(np.diff([t for t, _ in v]) <= 0):
                raise ValueError("uniform-field waypoints must be at increasing times")
        return v

    @model_validator(mode="after")
    def _unique_names(self) -> "MagneticFieldConfig":
        names = [m.name for m in self.magnets]
        if len(set(names)) != len(names):
            raise ValueError(f"magnet names must be unique, got {names}")
        return self

    def build(self) -> MagneticField:
        if isinstance(self.uniform, list):
            times, values = tuple(t for t, _ in self.uniform), tuple(tuple(h) for _, h in self.uniform)
        else:
            times, values = (0.0,), (tuple(self.uniform),)
        return MagneticField(times, values, tuple(m.build() for m in self.magnets))
