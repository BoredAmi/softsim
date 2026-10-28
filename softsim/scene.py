"""Backend-neutral scene description.

Scenarios turn their YAML config into a :class:`Scene`; backends turn a Scene
into a runnable model. Nothing here is specific to MuJoCo or SOFA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Annotated, Literal, Union

import numpy as np
from pydantic import Field

from softsim.capabilities import Capability
from softsim.config.schema import StrictModel
from softsim.config.units import Frequency, Length, Time
from softsim.geometry.rigid import Part
from softsim.materials.models import Friction, Material
from softsim.meshing import TetMesh


class SimulationSettings(StrictModel):
    """Integrator and run settings (SI units).

    ``integrator`` and ``solver`` are passed to the backend; backends reject
    values that are invalid for the scene (e.g. MuJoCo flex elasticity needs
    the ``discrete`` integrator).
    """

    timestep: Annotated[Time, Field(gt=0)] = 1e-4
    duration: Annotated[Time, Field(gt=0)] = 0.1
    output_rate: Annotated[Frequency, Field(gt=0)] = 3000.0
    integrator: str = "discrete"
    solver: str = "CG"
    iterations: Annotated[int, Field(ge=1)] = 100
    tolerance: Annotated[float, Field(gt=0)] = 1e-8
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)

    @property
    def output_every(self) -> int:
        """Number of physics steps between output samples (at least 1)."""
        return max(1, round(1.0 / (self.output_rate * self.timestep)))


class ContactSettings(StrictModel):
    """Contact compliance, shared by all pairs unless overridden.

    ``timeconst`` / ``dampratio`` follow MuJoCo's solref convention (other
    backends map them to their own penalty/compliance parameters). ``"auto"``
    uses twice the timestep, the stiffest contact that stays accurate.
    ``soft_radius`` is the collision skin thickness around soft-body elements.
    """

    timeconst: Union[Annotated[Time, Field(gt=0)], Literal["auto"]] = "auto"
    dampratio: Annotated[float, Field(gt=0)] = 1.0
    solimp: tuple[float, float, float] = (0.95, 0.99, 0.0002)
    soft_radius: Annotated[Length, Field(gt=0)] = 1e-4
    active_layers: Annotated[int, Field(ge=1)] = 1

    def resolved_timeconst(self, timestep: float) -> float:
        return 2.0 * timestep if self.timeconst == "auto" else float(self.timeconst)


class GroundSettings(StrictModel):
    """Flat rigid ground at z = 0."""

    enabled: bool = True
    friction: Friction = Friction(sliding=0.8)


@dataclass(frozen=True)
class Pose:
    """Position [m] and unit quaternion (w, x, y, z) in the world frame."""

    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def from_tilt(position: tuple[float, float, float], tilt_x: float = 0.0, tilt_y: float = 0.0) -> "Pose":
        """Pose rotated by ``tilt_x`` about world x, then ``tilt_y`` about world y (radians)."""
        qx = np.array([math.cos(tilt_x / 2), math.sin(tilt_x / 2), 0.0, 0.0])
        qy = np.array([math.cos(tilt_y / 2), 0.0, math.sin(tilt_y / 2), 0.0])
        q = quat_mul(qy, qx)
        return Pose(tuple(position), tuple(q / np.linalg.norm(q)))


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of quaternions (w, x, y, z)."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


@dataclass(frozen=True)
class SoftBodySpec:
    """A deformable body.

    Attributes:
        name: Unique name (used for topics, logs, contact reports).
        mesh: Tetrahedral mesh in the body frame.
        material: Constitutive + damping + friction description.
        pose: Body frame pose in the world.
        mass: Total mass override [kg]; if None, density * mesh volume.
        pinned: Indices of vertices fixed to the world (bonded boundary).
        self_collision: Whether elements may collide with each other.
        rgba: Display colour.
    """

    name: str
    mesh: TetMesh
    material: Material
    pose: Pose = Pose()
    mass: float | None = None
    pinned: tuple[int, ...] = ()
    self_collision: bool = False
    rgba: tuple[float, float, float, float] = (0.85, 0.92, 0.85, 1.0)

    @property
    def total_mass(self) -> float:
        return self.mass if self.mass is not None else self.material.density * self.mesh.volume()

    @property
    def effective_density(self) -> float:
        return self.total_mass / self.mesh.volume()

    @property
    def requires(self) -> frozenset[Capability]:
        req = set(self.material.requires) | {Capability.TET_MESH}
        if self.pinned:
            req.add(Capability.PINNED_VERTICES)
        if self.self_collision:
            req.add(Capability.SELF_COLLISION)
        return frozenset(req)


@dataclass(frozen=True)
class RigidBodySpec:
    """A free rigid body made of primitive parts.

    Velocities are in the world frame; angular velocity in rad/s.
    """

    name: str
    parts: tuple[Part, ...]
    mass: float
    pose: Pose = Pose()
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: Friction = Friction()
    rgba: tuple[float, float, float, float] = (0.75, 0.6, 0.2, 1.0)

    @property
    def requires(self) -> frozenset[Capability]:
        return frozenset({Capability.RIGID_BODIES}) | self.friction.requires


@dataclass(frozen=True)
class Scene:
    """Everything a backend needs to build a model."""

    soft_bodies: tuple[SoftBodySpec, ...]
    rigid_bodies: tuple[RigidBodySpec, ...] = ()
    simulation: SimulationSettings = SimulationSettings()
    contact: ContactSettings = ContactSettings()
    ground: GroundSettings = GroundSettings()
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def requires(self) -> frozenset[Capability]:
        req: set[Capability] = set()
        for body in (*self.soft_bodies, *self.rigid_bodies):
            req |= body.requires
        if self.ground.enabled:
            req |= self.ground.friction.requires
        return frozenset(req)

    def soft(self, name: str) -> SoftBodySpec:
        for body in self.soft_bodies:
            if body.name == name:
                return body
        raise KeyError(f"no soft body named {name!r}; have {[b.name for b in self.soft_bodies]}")

    def with_material(self, body_name: str, material: Material) -> "Scene":
        """Copy of the scene with one soft body's material replaced."""
        self.soft(body_name)
        bodies = tuple(replace(b, material=material) if b.name == body_name else b for b in self.soft_bodies)
        return replace(self, soft_bodies=bodies)
