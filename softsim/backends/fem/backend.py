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
    * rotating mount: the world frame, the pinned nodes driven along the rotation (no fictitious forces),
      integrated with the trapezoidal rule (the default's numerical damping dissipates the rotation);
    * prescribed regions: their held dofs (vertices and the mid-edge nodes between them) follow the path
      exactly at each step's end; the reaction is the step's converged residual at those dofs;
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
    * material zones: each zone's elements with their own law (any law the reference has);
    * cable actuators: the potential ``T L`` of the cable's length through its via points,
      interpolated with the quadratic shape functions, at the end-of-step tension.
Refused: several soft bodies, linear, StVK, Rayleigh/Kelvin-Voigt damping (use
Prony), self-collision, torsional/rolling friction, open pressure regions.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from softsim.actuation import (CableActuator, ForceActuator, PressureActuator, anchor_weights, embed, open_edges,
                               pad_rows)
from softsim.backends.base import Backend, ContactWrench, RigidState, SimState, SoftState
from softsim.capabilities import Capability
from softsim.errors import StabilityError
from softsim.geometry.rigid import rigid_mass_properties
from softsim.scene import GROUND_NAME, Scene, rotating_mount_problems

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
    implicit_actuation: ClassVar[bool] = True
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
        Capability.CABLE_ACTUATION,
        Capability.REGION_FORCE,
        Capability.MOVING_MOUNT,
        Capability.ROTATING_MOUNT,
        Capability.PRESCRIBED_MOTION,
        Capability.MULTI_MATERIAL,
        Capability.PLANE_STRAIN,
    })

    def __init__(self) -> None:
        self._scene: Scene | None = None
        self.solver = None
        self._prescribed: list = []

    @classmethod
    def extra_checks(cls, scene: Scene) -> list[str]:
        out = rotating_mount_problems(scene)
        if len(scene.soft_bodies) != 1:
            out.append(f"exactly one soft body (got {len(scene.soft_bodies)})")
        if any(b.self_collision for b in scene.soft_bodies):
            out.append("soft-body self-collision")
        if any(b.plane_strain is not None and np.abs(np.asarray(b.pose.quaternion) - (1, 0, 0, 0)).max() > 1e-12
               for b in scene.soft_bodies):
            out.append("plane strain on a rotated body (the axis is a body axis; the reference holds world dofs)")
        for a in scene.actuators:
            if not isinstance(a, PressureActuator):
                continue
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
        from softsim.fem.solid3d import (Problem3D, QuadraticTetMesh, RigidBody3D, Solid3DSolver, quat_to_matrix,
                                         tet10_shape)

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
        def with_midpoints(corners) -> list[int]:
            """``corners`` and the mid-edge nodes between two of them (they move with them)."""
            c = set(int(i) for i in corners)
            return sorted(c | {int(t[4 + k]) for t in mesh.tets for k, (i, j) in enumerate(
                ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))) if int(t[i]) in c and int(t[j]) in c})

        pinned = with_midpoints(body.pinned)
        # prescribed regions: (region, path, dofs per held axis); the dofs join the fixed ones
        self._prescribed = [(p.region, p.path, [3 * np.array(with_midpoints(p.vertices)) + k
                                                if p.axes[k] else np.zeros(0, int) for k in range(3)])
                            for p in body.prescribed]
        held = np.concatenate([d for *_, dofs in self._prescribed for d in dofs] or [np.zeros(0, int)])
        if body.plane_strain is not None:  # every node (corner and mid-edge) keeps its coordinate along the axis
            held = np.union1d(held, 3 * np.arange(len(mesh.nodes)) + body.plane_strain)
        self._pressure = [a for a in scene.actuators if isinstance(a, PressureActuator)]
        self._cables = [a for a in scene.actuators if isinstance(a, CableActuator)]
        self._forces = [a for a in scene.actuators if isinstance(a, ForceActuator)]
        cables = []
        for a in self._cables:
            via = np.array(a.via) @ rot.T + np.asarray(body.pose.position, float)
            n = len(via) - bool(a.anchor)
            elems, bary = embed(mesh.nodes, mesh.tets[:, :4], via[:n])
            rows = list(zip(mesh.tets[elems], tet10_shape(bary[:, 1:])[0]))
            if a.anchor:  # the last via point is a traction over a region's corner nodes
                rows.append(anchor_weights(mesh.nodes, mesh.faces_of(body.region(a.anchor)), via[-1]))
            cables.append(pad_rows(rows))
        self._problem = Problem3D(mesh, law, kappa,
                                  body.effective_density if body.uniform_density else body.element_densities(),
                                  prony=prony,
                                  gravity=tuple(scene.simulation.gravity), rigid=tuple(rigid),
                                  pinned=np.array(pinned, int), fixed_dofs=held.astype(int),
                                  pressure_faces=tuple(mesh.faces_of(body.region(a.region)) for a in self._pressure),
                                  cables=tuple(cables),
                                  cable_stiffness=tuple(a.stiffness or 0.0 for a in self._cables),
                                  force_loads=tuple((mesh.faces_of(body.region(a.region)), np.array(a.direction))
                                                    for a in self._forces),
                                  zones=tuple((elements, *from_material(m)[:2]) for m, elements in
                                              body.material_groups()[1:]),
                                  zone_pronys=tuple(from_material(m)[2] for m, _ in body.material_groups()[1:]))
        self._names = [GROUND_NAME] * scene.ground.enabled + [rb.name for rb in scene.rigid_bodies]
        self._solver_cls = Solid3DSolver
        self.solver = self.attach(self._new_solver())
        self._prev_t = 0.0

    def attach(self, solver):
        """Give ``solver`` (one built on this backend's problem, e.g. a static one) its prescribed motion:
        the prescribed regions' paths (held dofs' values at each step's end; pinned dofs stay at rest, in the
        mount's frame), or with a rotating mount the world-frame motion of the pinned nodes."""
        if self._scene.spin is not None:
            return self._attach_spin(solver)
        if not self._prescribed:
            return solver
        fixed = np.nonzero(solver.fixed)[0]
        slots = [(path, [np.searchsorted(fixed, d) for d in dofs]) for _, path, dofs in self._prescribed]

        def motion(t: float) -> np.ndarray:
            values = np.zeros(len(fixed))
            for path, idx in slots:
                x = path.position(t)
                for k in range(3):
                    values[idx[k]] = x[k]
            return values

        solver.fixed_motion = motion
        return solver

    def _attach_spin(self, solver):
        """A rotating mount, in the **world** frame: the pinned nodes (and mid-edge nodes between them)
        follow ``x = R (X - c) + c + p`` exactly, and every element feels plain gravity: no fictitious
        forces, so this is the reference the other backends' mount-frame runs are checked against."""
        fixed = np.nonzero(solver.fixed)[0]
        nodes, axis = fixed // 3, fixed % 3
        rest = self._problem.mesh.nodes[nodes]
        scene = self._scene

        def motion(t: float) -> np.ndarray:
            frame = scene.frame(t)
            moved = (rest - frame.center) @ frame.rotation.T + frame.center + frame.offset
            return (moved - rest)[np.arange(len(fixed)), axis]

        solver.fixed_motion = motion
        return solver

    def _new_solver(self):
        """A solver for the problem: the default Newmark scheme has numerical damping 0.25 (first order),
        which dissipates a rotation's centripetal balance: a body turning 0.3 rad per step stretched 32%
        less under its centrifugal load, with 4.6 mm of lateral deflection. A rotating mount therefore uses
        the trapezoidal rule (no numerical damping, second order: 0.12 mm at half the step)."""
        return self._solver_cls(self._problem, numerical_damping=0.0 if self._scene.spin is not None else 0.25)

    def reset(self) -> None:
        self.solver = self.attach(self._new_solver())

    # -- stepping ------------------------------------------------------------------------------

    def step(self, n: int = 1) -> None:
        dt = self._scene.simulation.timestep
        for _ in range(n):
            for k, a in enumerate(self._pressure):
                self.solver.pressures[k] = self.control(a.name, self.solver.t + dt)
            for k, a in enumerate(self._cables):
                value = self.control(a.name, self.solver.t + dt)
                if a.stroke_controlled:
                    self.solver.free_lengths[k] = self.solver.rest_lengths[k] - value
                else:
                    self.solver.tensions[k] = value
            for k, a in enumerate(self._forces):
                self.solver.forces[k] = self.control(a.name, self.solver.t + dt)
            if self._scene.mount is not None and self._scene.spin is None:  # in the mount's frame (a rotating
                # mount is simulated in the world instead, see _attach_spin); Newmark balances forces at the step's end
                self.solver.gravity = self.mount_gravity(self.solver.t + dt)
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
        reactions = {region: np.array([s.residual[d].sum() for d in dofs]) for region, _, dofs in self._prescribed}
        soft = {body.name: SoftState(x.copy(), v.copy(), reactions)}
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
        state = SimState(time=s.t, rigid=rigid, soft=soft, contacts=contacts, controls=self.controls())
        return state if self._scene.spin is not None else self._to_world(state)  # a spin runs in the world

    def is_finite(self) -> bool:
        return self.solver is not None and bool(np.all(np.isfinite(self.solver.y)))

    def close(self) -> None:
        self.solver = None
