"""Backend-neutral scene description.

Scenarios turn their YAML config into a :class:`Scene`; backends turn a Scene
into a runnable model. Nothing here is specific to MuJoCo or SOFA.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Annotated, Literal, Optional, Union

import numpy as np
from pydantic import Field

from softsim.actuation import Actuator, CableActuator, ForceActuator, PressureActuator, cable_rows
from softsim.capabilities import Capability
from softsim.config.schema import StrictModel
from softsim.config.units import Frequency, Length, Time, Velocity
from softsim.geometry.rigid import Part
from softsim.materials.models import Adhesion, Friction, Material
from softsim.meshing import TetMesh

if TYPE_CHECKING:  # softsim.magnetism builds on MountMotion, defined here
    from softsim.magnetism import MagneticField


GROUND_NAME = "ground"  # name of the ground in contact reports, in every backend


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
    ``cone`` and ``impratio`` are MuJoCo's friction cone and frictional-to-normal
    impedance ratio (other backends ignore them): an elliptic cone with
    ``impratio`` 10-100 lets soft bodies slip less under sticking friction, which
    otherwise costs impact energy (docs/backends/mujoco.md). ``slip_speed`` is the
    reference's (``fem``) Coulomb smoothing as a slip speed: below it friction acts as a
    spring, so a stuck contact creeps at a fraction of ``slip_speed`` (``1 - sqrt(1 - r)`` at
    a fraction ``r`` of its limit), whatever the step. ``null`` restores the old smoothing,
    1 um of slip per step, whose creep scales with 1 / dt (docs/decisions.md, entry 6).
    """

    timeconst: Union[Annotated[Time, Field(gt=0)], Literal["auto"]] = "auto"
    dampratio: Annotated[float, Field(gt=0)] = 1.0
    solimp: tuple[float, float, float] = (0.95, 0.99, 0.0002)
    soft_radius: Annotated[Length, Field(gt=0)] = 1e-4
    active_layers: Annotated[int, Field(ge=1)] = 1
    cone: Literal["pyramidal", "elliptic"] = "pyramidal"
    impratio: Annotated[float, Field(ge=1)] = 1.0
    slip_speed: Optional[Annotated[Velocity, Field(gt=0)]] = 1e-5

    def resolved_timeconst(self, timestep: float) -> float:
        return 2.0 * timestep if self.timeconst == "auto" else float(self.timeconst)


class GroundSettings(StrictModel):
    """Flat rigid ground at z = 0, with optional adhesion of soft bodies to it."""

    enabled: bool = True
    friction: Friction = Friction(sliding=0.8)
    adhesion: Adhesion | None = None


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
class MaterialZone:
    """Elements ``elements`` (tetrahedron indices) of a soft body made of ``material`` instead of the
    body's own (e.g. a strain-limiting layer, or a heavier insert). It may have its own elastic law and
    density; it must have the body's damping, so the viscous history stays that of one material."""

    name: str
    elements: np.ndarray
    material: Material


@dataclass(frozen=True)
class PrescribedMotion:
    """Displacement control of a surface region, e.g. a grip in a tension test: every vertex of
    ``region`` translates by ``path.position(t)`` from rest along the ``axes`` it holds (x, y, z; the
    others are free, a roller). ``path`` is a minimum-jerk waypoint path like a moving mount's; with a
    moving mount it is relative to the mount. Backends report the force the region's support exerts on
    the body, ``SoftState.reactions[region]`` [N] (inertia of the region's own mass included)."""

    region: str
    vertices: np.ndarray
    axes: tuple[bool, bool, bool]
    path: MountMotion

    def __post_init__(self) -> None:
        if not any(self.axes):
            raise ValueError(f"prescribed region {self.region!r} holds no axis (all null): drop it or fix one")


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
        regions: Named surface regions: boundary triangles (K, 3) of mesh vertex
            indices, outward normals (see softsim.geometry.soft).
        zones: Groups of elements with their own material (multi-material bodies);
            the other elements are made of ``material``.
        prescribed: Regions moved along prescribed paths (displacement control); their
            vertices are neither pinned nor in another prescribed region.
        plane_strain: Body axis (0, 1, 2) along which no vertex moves (plane strain: a wide strip, as in
            a 2D-like Abaqus model with ``U3 = 0``), or None.
    """

    name: str
    mesh: TetMesh
    material: Material
    pose: Pose = Pose()
    mass: float | None = None
    pinned: tuple[int, ...] = ()
    self_collision: bool = False
    rgba: tuple[float, float, float, float] = (0.85, 0.92, 0.85, 1.0)
    regions: dict[str, np.ndarray] = field(default_factory=dict)
    zones: tuple[MaterialZone, ...] = ()
    prescribed: tuple[PrescribedMotion, ...] = ()
    plane_strain: int | None = None

    def __post_init__(self) -> None:
        held = np.zeros(self.mesh.n_points, bool)
        held[list(self.pinned)] = True
        for p in self.prescribed:
            if held[p.vertices].any():
                raise ValueError(f"soft body {self.name!r}: prescribed region {p.region!r} shares vertices with "
                                 "a fixed or another prescribed region")
            held[p.vertices] = True
        seen = np.zeros(len(self.mesh.tets), bool)
        for z in self.zones:
            if len(z.elements) == 0:
                raise ValueError(f"soft body {self.name!r}: zone {z.name!r} has no elements")
            if seen[z.elements].any():
                raise ValueError(f"soft body {self.name!r}: zone {z.name!r} overlaps another zone")
            seen[z.elements] = True

    def lowered(self) -> "SoftBodySpec":
        """The body with ``plane_strain`` expressed as a prescribed motion, for backends that have displacement
        control but no plane-strain option: every vertex that is not already held keeps its coordinate along the
        axis (a zero path; the region ``plane_strain`` holds only that axis, so the others move freely)."""
        if self.plane_strain is None:
            return self
        held = np.zeros(self.mesh.n_points, bool)
        held[list(self.pinned)] = True
        for p in self.prescribed:
            held[p.vertices] = True
        axes = tuple(k == self.plane_strain for k in range(3))
        hold = PrescribedMotion("plane_strain", np.nonzero(~held)[0], axes,
                                MountMotion((0.0, 1.0), ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))))
        return replace(self, plane_strain=None, prescribed=(*self.prescribed, hold))

    def zone_damping_differs(self) -> list[str]:
        """Names of the zones whose damping is not the body's (a backend that has one damping per body refuses
        them)."""
        return [z.name for z in self.zones if z.material.damping != self.material.damping]

    def material_groups(self) -> list[tuple[Material, np.ndarray]]:
        """(material, element indices) for the body's own material, then each zone."""
        rest = np.ones(len(self.mesh.tets), bool)
        for z in self.zones:
            rest[z.elements] = False
        return [(self.material, np.nonzero(rest)[0]), *((z.material, np.asarray(z.elements)) for z in self.zones)]

    def region(self, name: str) -> np.ndarray:
        if name not in self.regions:
            raise KeyError(f"soft body {self.name!r} has no region {name!r}; have {sorted(self.regions)}")
        return self.regions[name]

    def cable_rows(self, cable: CableActuator) -> tuple[np.ndarray, np.ndarray]:
        """``cable``'s via points as (nodes, weights) on this body's mesh (see :func:`softsim.actuation.cable_rows`);
        ``ValueError`` for a via point outside the body or a bad anchor."""
        tris = self.region(cable.anchor) if cable.anchor else None
        return cable_rows(self.mesh.points, self.mesh.tets, np.array(cable.via), tris)

    def _material_densities(self) -> np.ndarray:
        rho = np.full(len(self.mesh.tets), float(self.material.density))
        for z in self.zones:
            rho[z.elements] = z.material.density
        return rho

    def element_densities(self) -> np.ndarray:
        """Density of every tetrahedron [kg/m^3]: its material's, all scaled to ``mass`` if given."""
        rho = self._material_densities()
        return rho * (self.mass / self.material_mass) if self.mass is not None else rho

    def vertex_masses(self) -> np.ndarray:
        """Lumped vertex masses [kg]: a quarter of each tetrahedron's mass to each of its vertices."""
        m = np.zeros(self.mesh.n_points)
        np.add.at(m, self.mesh.tets.ravel(), np.repeat(self.element_densities() * self.mesh.tet_volumes() / 4, 4))
        return m

    @property
    def uniform_density(self) -> bool:
        return all(math.isclose(z.material.density, self.material.density, rel_tol=1e-12) for z in self.zones)

    @property
    def material_mass(self) -> float:
        """Mass from the materials' densities (ignores the ``mass`` override)."""
        return float(self._material_densities() @ self.mesh.tet_volumes())

    @property
    def total_mass(self) -> float:
        return self.mass if self.mass is not None else self.material_mass

    @property
    def effective_density(self) -> float:
        """Mean density (total mass / volume)."""
        return self.total_mass / self.mesh.volume()

    @property
    def requires(self) -> frozenset[Capability]:
        req = set(self.material.requires) | {Capability.TET_MESH}
        for z in self.zones:
            req |= z.material.requires | {Capability.MULTI_MATERIAL}
        if self.pinned:
            req.add(Capability.PINNED_VERTICES)
        if self.prescribed:
            req.add(Capability.PRESCRIBED_MOTION)
        if self.self_collision:
            req.add(Capability.SELF_COLLISION)
        if self.plane_strain is not None:
            req.add(Capability.PLANE_STRAIN)
        return frozenset(req)


@dataclass(frozen=True)
class RigidBodySpec:
    """A rigid body made of primitive parts.

    Velocities are in the world frame; angular velocity in rad/s. ``mount``
    is ``"free"`` (6 DOF) or ``"slider_z"``: vertical translation only, held
    by a spring (``mount_stiffness`` [N/m], ``mount_damping`` [N s/m]) whose
    rest position is the initial pose, e.g. a compliant support. ``floor`` [m]: the lowest height of the centre of
    mass, where the body stands on the table. The reference needs it, having no contact between a rigid body and the
    ground; the other backends collide the body with the ground and ignore it.
    """

    name: str
    parts: tuple[Part, ...]
    mass: float
    pose: Pose = Pose()
    linear_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: Friction = Friction()
    rgba: tuple[float, float, float, float] = (0.75, 0.6, 0.2, 1.0)
    mount: Literal["free", "slider_z"] = "free"
    mount_stiffness: float = 0.0
    mount_damping: float = 0.0
    floor: float | None = None

    @property
    def requires(self) -> frozenset[Capability]:
        req = frozenset({Capability.RIGID_BODIES}) | self.friction.requires
        return req | {Capability.SPRING_MOUNT} if self.mount != "free" else req


@dataclass(frozen=True)
class MountMotion:
    """Prescribed translation of the soft body's fixed (pinned) vertices: a moving mount, e.g. a finger
    on a robot arm. ``positions[i]`` [m] is the mount's displacement from rest at ``times[i]`` [s] (the
    first is zero). Between waypoints it moves with minimum jerk (quintic): at rest at every waypoint by
    default (continuous acceleration, no overshoot), or passing waypoint ``i`` at speed ``velocities[i]``
    [m/s] (quintic Hermite, zero acceleration at the waypoints: two waypoints ``v T`` apart that both carry
    ``v`` are a constant speed exactly). Before the first waypoint it is at rest (its speed must be zero);
    after the last it keeps the last waypoint's speed (held, by default).

    Backends simulate in the mount's frame, which is exact for a translation: the pinned vertices stay
    put and every element feels the inertial body force ``-rho a(t)``, i.e. gravity ``g - a(t)``
    (:meth:`Backend.mount_gravity`); reported positions and velocities add the mount's back."""

    times: tuple[float, ...]
    positions: tuple[tuple[float, float, float], ...]
    velocities: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.times, float)
        if len(t) < 2 or np.any(np.diff(t) <= 0) or len(self.positions) != len(t):
            raise ValueError("mount motion needs two or more waypoints at increasing times")
        if np.any(np.asarray(self.positions[0], float) != 0):
            raise ValueError("mount motion: the first waypoint is the rest position, displacement [0, 0, 0]")
        if self.velocities is not None:
            if len(self.velocities) != len(t):
                raise ValueError("mount motion: one speed per waypoint")
            if np.any(np.asarray(self.velocities[0], float) != 0):
                raise ValueError("mount motion: the first waypoint is at rest (speed 0)")

    def _speed(self, i: int) -> np.ndarray:
        return np.zeros(3) if self.velocities is None else np.asarray(self.velocities[i], float)

    def _eval(self, t: float, nu: int) -> np.ndarray:
        """``nu``-th time derivative of the displacement at ``t``."""
        times = self.times
        if t <= times[0]:
            return np.zeros(3)
        if t >= times[-1]:
            v = self._speed(-1)
            return (np.asarray(self.positions[-1], float) + v * (t - times[-1]), v, np.zeros(3))[nu]
        i = int(np.searchsorted(times, t, side="right")) - 1
        span = times[i + 1] - times[i]
        u = (t - times[i]) / span
        p0, p1 = np.asarray(self.positions[i], float), np.asarray(self.positions[i + 1], float)
        v0, v1 = self._speed(i) * span, self._speed(i + 1) * span  # per unit u
        # quintic Hermite basis with zero end accelerations: value, d/du, d2/du2
        h0 = (1 - 10 * u**3 + 15 * u**4 - 6 * u**5, -30 * u**2 + 60 * u**3 - 30 * u**4, -60 * u + 180 * u**2 - 120 * u**3)
        h1 = (u - 6 * u**3 + 8 * u**4 - 3 * u**5, 1 - 18 * u**2 + 32 * u**3 - 15 * u**4, -36 * u + 96 * u**2 - 60 * u**3)
        h4 = (-4 * u**3 + 7 * u**4 - 3 * u**5, -12 * u**2 + 28 * u**3 - 15 * u**4, -24 * u + 84 * u**2 - 60 * u**3)
        h5 = (10 * u**3 - 15 * u**4 + 6 * u**5, 30 * u**2 - 60 * u**3 + 30 * u**4, 60 * u - 180 * u**2 + 120 * u**3)
        return (h0[nu] * p0 + h1[nu] * v0 + h4[nu] * v1 + h5[nu] * p1) / span**nu

    def position(self, t: float) -> np.ndarray:
        return self._eval(t, 0)

    def velocity(self, t: float) -> np.ndarray:
        return self._eval(t, 1)

    def acceleration(self, t: float) -> np.ndarray:
        return self._eval(t, 2)


@dataclass(frozen=True)
class MountSpin:
    """Prescribed rotation of the mount about a fixed axis, e.g. a finger on a spinning arm: the fixed
    vertices turn about ``axis`` (a unit vector) through the point ``center`` [m] (rest frame) by
    ``angles[i]`` [rad] at ``times[i]`` [s] (the first is zero), minimum jerk between waypoints, at rest at
    each or turning at ``rates[i]`` [rad/s] (a constant rate between two waypoints that carry it; after the
    last the last rate is kept), like :class:`MountMotion`, whose translation it composes with:
    ``x_world = R (x - c) + c + p``.

    In the mount's frame a point feels, per unit mass, gravity ``R^T (g - p'')`` and the fictitious
    accelerations ``-alpha x r - omega x (omega x r) - 2 omega x v`` (Euler, centrifugal, Coriolis), with
    ``r = x - c`` (:class:`MountFrame`). The reference runs in the world instead, with the fixed
    vertices driven along the rotation, so it has none of them."""

    axis: tuple[float, float, float]
    center: tuple[float, float, float]
    times: tuple[float, ...]
    angles: tuple[float, ...]
    rates: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if abs(np.linalg.norm(np.asarray(self.axis, float)) - 1.0) > 1e-9:
            raise ValueError("mount rotation: the axis must be a unit vector")
        if self.rates is not None and len(self.rates) != len(self.times):
            raise ValueError("mount rotation: one rate per waypoint")
        speeds = None if self.rates is None else tuple((w, 0.0, 0.0) for w in self.rates)
        object.__setattr__(self, "_path", MountMotion(tuple(self.times), tuple((a, 0.0, 0.0) for a in self.angles),
                                                      speeds))

    def angle(self, t: float) -> float:
        return float(self._path.position(t)[0])

    def rate(self, t: float) -> float:
        return float(self._path.velocity(t)[0])

    def acceleration(self, t: float) -> float:
        return float(self._path.acceleration(t)[0])

    def rotation(self, t: float) -> np.ndarray:
        """Rotation matrix mount -> world at ``t`` (Rodrigues)."""
        n, a = np.asarray(self.axis, float), self.angle(t)
        k = np.array([[0.0, -n[2], n[1]], [n[2], 0.0, -n[0]], [-n[1], n[0], 0.0]])
        return np.eye(3) + math.sin(a) * k + (1.0 - math.cos(a)) * k @ k


@dataclass(frozen=True)
class MountFrame:
    """The mount's frame at one instant: rotation ``R`` (mount -> world) about ``center``, origin
    translation ``offset`` ``p`` with its velocity and acceleration, and the angular velocity ``omega`` and
    acceleration ``alpha`` [rad/s, rad/s^2] in mount-frame components."""

    rotation: np.ndarray
    center: np.ndarray
    offset: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    omega: np.ndarray
    alpha: np.ndarray

    def to_world(self, x: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """World positions and velocities of points at ``x``, ``v`` (N, 3) in the mount frame."""
        r = x - self.center
        return r @ self.rotation.T + self.center + self.offset, (v + np.cross(self.omega, r)) @ self.rotation.T + self.velocity

    def gravity(self, g: np.ndarray) -> np.ndarray:
        """Gravity in the mount frame: ``R^T (g - p'')`` (the translation's inertial force included)."""
        return self.rotation.T @ (np.asarray(g, float) - self.acceleration)

    def fictitious(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Acceleration (N, 3) the rotation adds in the mount frame: ``-alpha x r - omega x (omega x r)
        - 2 omega x v``."""
        r = x - self.center
        return -np.cross(self.alpha, r) - np.cross(self.omega, np.cross(self.omega, r)) - 2.0 * np.cross(self.omega, v)


def mount_frame(mount: MountMotion | None, spin: MountSpin | None, t: float) -> MountFrame:
    """The frame of a translating and/or rotating mount at ``t``."""
    zero = np.zeros(3)
    p, v, a = (mount.position(t), mount.velocity(t), mount.acceleration(t)) if mount is not None else (zero, zero, zero)
    if spin is None:
        return MountFrame(np.eye(3), zero, p, v, a, zero, zero)
    n = np.asarray(spin.axis, float)
    return MountFrame(spin.rotation(t), np.asarray(spin.center, float), p, v, a, n * spin.rate(t),
                      n * spin.acceleration(t))


@dataclass(frozen=True)
class Scene:
    """Everything a backend needs to build a model."""

    soft_bodies: tuple[SoftBodySpec, ...]
    rigid_bodies: tuple[RigidBodySpec, ...] = ()
    simulation: SimulationSettings = SimulationSettings()
    contact: ContactSettings = ContactSettings()
    ground: GroundSettings = GroundSettings()
    metadata: dict[str, object] = field(default_factory=dict)
    actuators: tuple[Actuator, ...] = ()
    mount: MountMotion | None = None
    spin: MountSpin | None = None
    magnetic_field: "MagneticField | None" = None  # the applied field (softsim.magnetism), world frame

    def __post_init__(self) -> None:
        if self.mount is not None or self.spin is not None:
            if self.rigid_bodies or self.ground.enabled or len(self.soft_bodies) != 1:
                raise ValueError("a moving mount is simulated in the mount's frame: it needs one soft body and no "
                                 "ground or rigid bodies (they would have to move with it)")
            if not self.soft_bodies[0].pinned and not self.soft_bodies[0].prescribed:
                raise ValueError("a moving mount moves the body's fixed vertices: fix a region (body.fixed)")
        names = [a.name for a in self.actuators]
        if len(set(names)) != len(names):
            raise ValueError(f"actuator names must be unique, got {names}")
        for a in self.actuators:
            body = self.soft(a.body)
            if isinstance(a, (PressureActuator, ForceActuator)):
                body.region(a.region)
            elif isinstance(a, CableActuator):
                try:
                    body.cable_rows(a)
                except (ValueError, KeyError) as exc:
                    raise ValueError(f"actuator {a.name!r}: {exc}") from None

    def frame(self, t: float) -> MountFrame:
        """The mount's frame at ``t`` (the identity without a moving or rotating mount)."""
        return mount_frame(self.mount, self.spin, t)

    def actuator(self, name: str) -> Actuator:
        for a in self.actuators:
            if a.name == name:
                return a
        raise KeyError(f"no actuator named {name!r}; have {[a.name for a in self.actuators]}")

    @property
    def requires(self) -> frozenset[Capability]:
        req: set[Capability] = set()
        for body in (*self.soft_bodies, *self.rigid_bodies, *self.actuators):
            req |= body.requires
        if self.mount is not None:
            req.add(Capability.MOVING_MOUNT)
        if self.spin is not None:
            req.add(Capability.ROTATING_MOUNT)
        if self.magnetic_field is not None and self.magnetic_field.active:
            req.add(Capability.MAGNETO_ACTIVE)
        if self.ground.enabled:
            req |= self.ground.friction.requires
            if self.ground.adhesion is not None and self.soft_bodies:
                req |= self.ground.adhesion.requires
        return frozenset(req)

    def soft(self, name: str) -> SoftBodySpec:
        for body in self.soft_bodies:
            if body.name == name:
                return body
        raise KeyError(f"no soft body named {name!r}; have {[b.name for b in self.soft_bodies]}")

    def lowered(self) -> "Scene":
        """The scene with every body's ``plane_strain`` lowered to a prescribed motion (:meth:`SoftBodySpec.lowered`)."""
        return replace(self, soft_bodies=tuple(b.lowered() for b in self.soft_bodies))

    def with_material(self, body_name: str, material: Material) -> "Scene":
        """Copy of the scene with one soft body's material replaced."""
        self.soft(body_name)
        bodies = tuple(replace(b, material=material) if b.name == body_name else b for b in self.soft_bodies)
        return replace(self, soft_bodies=bodies)


def rotating_mount_problems(scene: Scene) -> list[str]:
    """What a rotating mount cannot be combined with (every backend refuses these): a region force's
    direction is a world direction, so it would have to turn with the mount's frame, and a prescribed
    region's path is a translation along the mount frame's axes."""
    if scene.spin is None:
        return []
    out = [f"actuator {a.name!r}: a region force with a rotating mount (its direction would turn with the frame)"
           for a in scene.actuators if isinstance(a, ForceActuator)]
    out += [f"{b.name}: prescribed region {p.region!r} with a rotating mount" for b in scene.soft_bodies for p in b.prescribed]
    return out
