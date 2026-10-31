"""The 3D reference FEM (:mod:`softsim.fem.solid3d`) behind the backend interface.

Slow (seconds per step on a laptop) and exact where the general backends are
not: quadratic, locking-free tetrahedra, the material laws without
approximation, Prony viscoelasticity in the implicit solve, node-based penalty
contact against rigid shapes with rounded edges. Use it to check MuJoCo/SOFA
runs and to fit physical material cards.

Mapping:
    * one soft body: quadratic tetrahedra built from its linear mesh (edge midpoints);
      ``stable_neo_hookean`` runs as the split Neo-Hookean of the reference (same
      small-strain response; see softsim.fem.materials);
    * ``base: bonded``: the pinned vertices and the mid-edge nodes between them;
    * rigid bodies: ``free`` (6 DOF) or ``slider_z`` (spring and damper), their parts
      as cylinders/boxes with 0.5 mm rounded edges;
    * contact: penalty 2e11 Pa/m on surface nodes; friction coefficients combine by
      maximum, as in MuJoCo: 0 = frictionless, >= 1 = sticking (for a tacky silicone the
      reference's static punch is identical at mu = 1 and sticking, docs/reference_fem.md,
      and sticking keeps Newton fast), otherwise Coulomb regularised over 1 um of slip per
      step; adhesion from the material (soft-rigid) and ``ground.adhesion``.
    * pressure actuators: the potential ``-p V`` of the volume the region encloses, on the
      quadratic faces (exact follower load, softsim.actuation), at the end-of-step value of the
      control; the region must be closed (or bounded only by pinned vertices).
Refused: several soft bodies, linear, StVK, Rayleigh/Kelvin-Voigt damping (use
Prony), self-collision, torsional/rolling friction, open pressure regions.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from softsim.actuation import open_edges
from softsim.backends.base import Backend, ContactWrench, RigidState, SimState, SoftState
from softsim.capabilities import Capability
from softsim.errors import StabilityError
from softsim.geometry.rigid import rigid_mass_properties
from softsim.scene import GROUND_NAME, Scene

CONTACT_STIFFNESS = 2e11  # Pa/m: penetration ~ p / k, 50 nm at 10 kPa
EDGE_RADIUS = 5e-4
STICK_MU = 1.0            # friction coefficients from here on stick
SLIP_TOLERANCE = 1e-6     # m per step: Coulomb regularisation (C1 stick zone)


def _matrix_to_quat(r: np.ndarray) -> np.ndarray:
    w = np.sqrt(max(1.0 + np.trace(r), 1e-12)) / 2
    q = np.array([w, (r[2, 1] - r[1, 2]) / (4 * w), (r[0, 2] - r[2, 0]) / (4 * w), (r[1, 0] - r[0, 1]) / (4 * w)])
    if 1.0 + np.trace(r) < 1e-6:  # near 180 degrees: fall back to a robust eigen-decomposition
        k = np.array([[r[0, 0] - r[1, 1] - r[2, 2], r[1, 0] + r[0, 1], r[2, 0] + r[0, 2], r[1, 2] - r[2, 1]],
                      [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], r[2, 1] + r[1, 2], r[2, 0] - r[0, 2]],
                      [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], r[0, 1] - r[1, 0]],
                      [r[1, 2] - r[2, 1], r[2, 0] - r[0, 2], r[0, 1] - r[1, 0], r[0, 0] + r[1, 1] + r[2, 2]]]) / 3
        w_, v = np.linalg.eigh(k)
        x, y, z, w0 = v[:, -1]
        q = np.array([w0, x, y, z])
    return q / np.linalg.norm(q)


class FemBackend(Backend):
    """3D quadratic-tetrahedron reference solver; see module docstring."""

    name: ClassVar[str] = "fem"
    locking_free: ClassVar[bool] = True
    capabilities: ClassVar[frozenset[Capability]] = frozenset({
        Capability.NEO_HOOKEAN,
        Capability.STABLE_NEO_HOOKEAN,
        Capability.MOONEY_RIVLIN,
        Capability.YEOH,
        Capability.GENT,
        Capability.ARRUDA_BOYCE,
        Capability.OGDEN,
        Capability.VISCOELASTIC_PRONY,
        Capability.TET_MESH,
        Capability.PINNED_VERTICES,
        Capability.RIGID_BODIES,
        Capability.SPRING_MOUNT,
        Capability.CONTACT_FRICTION,
        Capability.CONTACT_FORCES,
        Capability.CONTACT_ADHESION,
        Capability.PRESSURE_ACTUATION,
    })

    def __init__(self) -> None:
        self._scene: Scene | None = None
        self.solver = None

    @classmethod
    def extra_checks(cls, scene: Scene) -> list[str]:
        out = []
        if len(scene.soft_bodies) != 1:
            out.append(f"exactly one soft body (got {len(scene.soft_bodies)})")
        if any(b.self_collision for b in scene.soft_bodies):
            out.append("soft-body self-collision")
        for a in scene.actuators:
            body = scene.soft(a.body)
            loose = set(open_edges(body.region(a.region)).ravel().tolist()) - set(body.pinned)
            if loose:
                out.append(f"actuator {a.name!r}: region {a.region!r} is open ({len(loose)} unpinned boundary "
                           "vertices); the reference applies pressure exactly only to closed regions")
        return out

    # -- build ---------------------------------------------------------------------------------

    def build(self, scene: Scene) -> None:
        from softsim.fem.axisym import ContactSettings
        from softsim.fem.materials import from_material
        from softsim.fem.solid3d import Problem3D, QuadraticTetMesh, RigidBody3D, Solid3DSolver, quat_to_matrix

        self.check(scene)
        self._scene = scene
        self._clear_controls()
        body = scene.soft_bodies[0]
        rot = quat_to_matrix(body.pose.quaternion)
        points = body.mesh.points @ rot.T + np.asarray(body.pose.position, float)
        mesh = QuadraticTetMesh.from_linear(points, body.mesh.tets)
        law, kappa, prony = from_material(body.material)
        mat = body.material

        def contact(mu: float, adhesion) -> ContactSettings:
            return ContactSettings(stiffness=CONTACT_STIFFNESS, stick=mu >= STICK_MU,
                                   friction=0.0 if mu >= STICK_MU else mu, slip_tolerance=SLIP_TOLERANCE,
                                   adhesion_strength=adhesion.strength if adhesion else 0.0,
                                   adhesion_range=adhesion.range if adhesion else 3e-4)

        rigid = []
        if scene.ground.enabled:
            rigid.append(RigidBody3D.ground(contact(max(mat.friction.sliding, scene.ground.friction.sliding),
                                                    scene.ground.adhesion)))
        for rb in scene.rigid_bodies:
            com, inertia = rigid_mass_properties(rb)
            r = quat_to_matrix(rb.pose.quaternion)
            omega = np.asarray(rb.angular_velocity, float)
            v_com = np.asarray(rb.linear_velocity, float) + np.cross(omega, r @ com)  # given: frame-origin velocity
            rigid.append(RigidBody3D(
                rb.name, parts=tuple(rb.parts), mass=rb.mass, inertia=tuple(map(tuple, inertia)), com=tuple(com),
                position=tuple(rb.pose.position), quaternion=tuple(rb.pose.quaternion), velocity=tuple(v_com),
                angular_velocity=tuple(omega), mount=rb.mount, stiffness=rb.mount_stiffness,
                damping=rb.mount_damping, edge_radius=EDGE_RADIUS,
                contact=contact(max(mat.friction.sliding, rb.friction.sliding), mat.adhesion)))
        pins = set(int(i) for i in body.pinned)
        pinned = sorted(pins | {int(t[4 + k]) for t in mesh.tets for k, (i, j) in enumerate(
            ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))) if int(t[i]) in pins and int(t[j]) in pins})
        self._problem = Problem3D(mesh, law, kappa, body.effective_density, prony=prony,
                                  gravity=tuple(scene.simulation.gravity), rigid=tuple(rigid),
                                  pinned=np.array(pinned, int),
                                  pressure_faces=tuple(mesh.faces_of(body.region(a.region)) for a in scene.actuators))
        self._names = [GROUND_NAME] * scene.ground.enabled + [rb.name for rb in scene.rigid_bodies]
        self._solver_cls = Solid3DSolver
        self.solver = Solid3DSolver(self._problem)
        self._prev_t = 0.0

    def reset(self) -> None:
        self.solver = self._solver_cls(self._problem)

    # -- stepping ------------------------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        dt = self._scene.simulation.timestep
        for _ in range(n):
            for k, a in enumerate(self._scene.actuators):
                self.solver.pressures[k] = self.control(a.name, self.solver.t + dt)
            if not self.solver.advance(dt):
                raise StabilityError(f"fem: Newton did not converge at t = {self.solver.t:.6f} s even with "
                                     f"{dt / 16:.2e} s substeps")

    @property
    def time(self) -> float:
        return self.solver.t if self.solver is not None else 0.0

    @property
    def scene(self) -> Scene:
        return self._scene

    # -- state ---------------------------------------------------------------------------------

    def state(self) -> SimState:
        s = self.solver
        body = self._scene.soft_bodies[0]
        nc = self._problem.mesh.n_corner
        x = s.current_nodes()[:nc]
        v = s.v[:3 * s.n].reshape(s.n, 3)[:nc]
        soft = {body.name: SoftState(x.copy(), v.copy())}
        rigid, contacts = {}, []
        for i, (rb, name) in enumerate(zip(self._problem.rigid, self._names)):
            force = s.contact_force(i)
            n_active = int(np.count_nonzero(s._bond[i])) if np.any(s._bond[i]) else 0
            if np.any(force != 0):
                contacts.append(ContactWrench(body.name, name, force, np.zeros(3), np.zeros(3), max(n_active, 1)))
            if rb.mount == "fixed":
                continue
            c, r = s.rigid_pose(i)
            lin, ang = s.rigid_velocity(i)
            com_world = r @ np.asarray(rb.com, float)
            rigid[name] = RigidState(position=c - com_world, quaternion=_matrix_to_quat(r),
                                     linear_velocity=lin - np.cross(ang, com_world), angular_velocity=ang, com=c)
        return SimState(time=s.t, rigid=rigid, soft=soft, contacts=contacts, controls=self.controls())

    def is_finite(self) -> bool:
        return self.solver is not None and bool(np.all(np.isfinite(self.solver.y)))

    def close(self) -> None:
        self.solver = None
