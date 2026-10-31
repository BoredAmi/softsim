"""SOFA (https://www.sofa-framework.org) FEM backend.

Scene layout (built with SofaPython3, verified against SOFA v26.06):

    root: FreeMotionAnimationLoop, BlockGaussSeidelConstraintSolver,
          collision pipeline (MinProximityIntersection, frictional Lagrange contact)
    ground: static triangle pair at z = 0
    <soft body>: EulerImplicitSolver (Rayleigh alpha/beta), SparseLDLSolver,
                 tetrahedral topology, MeshMatrixMass (lumped), FEM force field,
                 surface collision submesh (triangles + points, IdentityMapping)
    <rigid body>: Rigid3d DOF at the centre of mass, UniformMass with inertia,
                  triangulated parts (RigidMapping)

Notes and limits (see docs/backends/sofa.md):
    * contacts are hard Lagrange constraints; ``contact.timeconst/dampratio/
      solimp`` (MuJoCo compliance) are not used. One friction coefficient
      applies to all pairs (the largest sliding friction in the scene).
    * ``simulation.integrator`` / ``solver`` are MuJoCo settings and ignored:
      SOFA uses implicit Euler + sparse LDL + Gauss-Seidel constraint solving.
    * contact wrenches are reconstructed from momentum balance per body (net
      force only; no centre of pressure, no torque); they include adhesion.
    * adhesion (``softsim.contact.adhesion``) is applied through
      ``ConstantForceField``s updated before every step, like on MuJoCo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np

from softsim.backends.base import Backend, ContactWrench, RigidState, SimState, SoftState
from softsim.capabilities import Capability
from softsim.contact.adhesion import RigidFrame, adhesion_fields
from softsim.errors import SoftSimError
from softsim.geometry.rigid import Part
from softsim.materials.models import (
    KelvinVoigt,
    LinearElastic,
    MooneyRivlin,
    NeoHookean,
    Ogden,
    RayleighDamping,
    StableNeoHookean,
    StVenantKirchhoff,
)
from softsim.scene import RigidBodySpec, Scene, SoftBodySpec

GROUND = "ground"
GROUND_GROUP = 1  # collision group of the ground (models in the same group never collide)
PLUGINS = [
    "Sofa.Component.AnimationLoop", "Sofa.Component.Collision.Detection.Algorithm",
    "Sofa.Component.Collision.Detection.Intersection", "Sofa.Component.Collision.Geometry",
    "Sofa.Component.Collision.Response.Contact", "Sofa.Component.Constraint.Lagrangian.Correction",
    "Sofa.Component.Constraint.Lagrangian.Solver", "Sofa.Component.Constraint.Projective",
    "Sofa.Component.LinearSolver.Direct", "Sofa.Component.LinearSolver.Iterative",
    "Sofa.Component.Mapping.Linear", "Sofa.Component.Mapping.NonLinear", "Sofa.Component.Mass",
    "Sofa.Component.ODESolver.Backward", "Sofa.Component.SolidMechanics.FEM.Elastic",
    "Sofa.Component.SolidMechanics.FEM.HyperElastic", "Sofa.Component.StateContainer",
    "Sofa.Component.Topology.Container.Constant", "Sofa.Component.Topology.Container.Dynamic",
    "Sofa.Component.MechanicalLoad",
]
# Blow-up guard: an exploding FEM state floods the constraint solver with contacts
# and SOFA segfaults before any NaN appears, so stepping stops above this speed.
MAX_SPEED = float(os.environ.get("SOFTSIM_SOFA_MAX_SPEED", 100.0))  # m/s


def import_sofa():
    """Import SofaPython3, using ``$SOFA_ROOT`` to locate it if needed."""
    try:
        import Sofa  # noqa: F401
    except ImportError:
        root = os.environ.get("SOFA_ROOT")
        if not root:
            raise SoftSimError("SOFA backend: SofaPython3 not importable and SOFA_ROOT is not set "
                               "(see docs/backends/sofa.md)") from None
        site = Path(root) / "plugins" / "SofaPython3" / "lib" / "python3" / "site-packages"
        if not site.is_dir():
            raise SoftSimError(f"SOFA backend: {site} not found; is SOFA_ROOT={root} a SOFA binary release?") from None
        sys.path.insert(0, str(site))
    import Sofa
    import Sofa.Core
    import Sofa.Simulation

    return Sofa


def lame(youngs: float, poisson: float) -> tuple[float, float]:
    """(mu, lambda) from Young's modulus and Poisson's ratio."""
    return youngs / (2 * (1 + poisson)), youngs * poisson / ((1 + poisson) * (1 - 2 * poisson))


def force_field_args(body: SoftBodySpec) -> tuple[str, dict]:
    """SOFA force field component and arguments for a soft body's elastic model.

    SOFA v26.06 hyperelastic ``ParameterSet`` conventions (from the sources):
    NeoHookean / StableNeoHookean / StVenantKirchhoff: [mu, lambda]; MooneyRivlin: [C10, C01, k0]; Ogden (one term):
    W = mu1/alpha^2 (J^(-alpha/3) sum lambda_i^alpha - 3) + k0/2 ln(J)^2 -> [mu1, alpha, k0],
    which equals softsim's Ogden term (mu/alpha) with mu1 = mu * alpha.
    """
    e = body.material.elastic
    if isinstance(e, LinearElastic):
        return "TetrahedronFEMForceField", {"method": "small", "youngModulus": e.youngs_modulus,
                                            "poissonRatio": e.poisson_ratio}
    if isinstance(e, (NeoHookean, StableNeoHookean, StVenantKirchhoff)):
        name = {NeoHookean: "NeoHookean", StableNeoHookean: "StableNeoHookean",
                StVenantKirchhoff: "StVenantKirchhoff"}[type(e)]
        return "TetrahedronHyperelasticityFEMForceField", {"materialName": name,
                                                           "ParameterSet": list(lame(e.youngs_modulus, e.poisson_ratio))}
    if isinstance(e, MooneyRivlin):
        # SOFA's MooneyRivlin energy function contains an integer division (pow(J, -2/3) -> J^0),
        # but its forces were verified: zero stress at rest and the standard small-strain
        # stiffness mu = 2 (C10 + C01) (tests/test_sofa_backend.py).
        return "TetrahedronHyperelasticityFEMForceField", {
            "materialName": "MooneyRivlin", "ParameterSet": [e.c10, e.c01, e.bulk_modulus]}
    if isinstance(e, Ogden):
        return "TetrahedronHyperelasticityFEMForceField", {
            "materialName": "Ogden", "ParameterSet": [e.mu[0] * e.alpha[0], e.alpha[0], e.bulk_modulus]}
    raise TypeError(f"elastic model {e.model!r} is not supported by SOFA backend")


def rayleigh(body: SoftBodySpec) -> tuple[float, float]:
    """(rayleighMass alpha [1/s], rayleighStiffness beta [s])."""
    d = body.material.damping
    if d is None:
        return 0.0, 0.0
    if isinstance(d, RayleighDamping):
        return d.alpha, d.beta
    if isinstance(d, KelvinVoigt):
        return 0.0, d.viscosity / body.material.elastic.initial_youngs_modulus()
    raise TypeError(f"damping model {d.model!r} is not supported by SOFA backend")


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def part_mesh(part: Part, segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Closed outward-oriented triangle mesh of a primitive part (body frame)."""
    cx, cy, cz = part.center
    if part.kind == "cylinder":
        r, hh = part.size[0], part.size[1]
        th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        ring = np.c_[r * np.cos(th), r * np.sin(th)]
        pts = np.vstack([np.c_[ring, np.full(segments, -hh)], np.c_[ring, np.full(segments, hh)],
                         [[0, 0, -hh], [0, 0, hh]]])
        n, cb, ct = segments, 2 * segments, 2 * segments + 1
        tris = []
        for i in range(n):
            j = (i + 1) % n
            tris += [(i, j, n + j), (i, n + j, n + i), (cb, j, i), (ct, n + i, n + j)]
        return pts + [cx, cy, cz], np.array(tris)
    sx, sy, sz = part.size
    pts = np.array([[x, y, z] for z in (-sz, sz) for y in (-sy, sy) for x in (-sx, sx)]) + [cx, cy, cz]
    tris = np.array([(0, 2, 1), (1, 2, 3), (4, 5, 6), (5, 7, 6), (0, 1, 4), (1, 5, 4),
                     (2, 6, 3), (3, 6, 7), (0, 4, 2), (2, 4, 6), (1, 3, 5), (3, 7, 5)])
    return pts, tris


def rigid_mass_properties(body: RigidBodySpec) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass (body frame) and inertia tensor about it [kg m^2]; mass split by volume."""
    vols = np.array([p.volume() for p in body.parts])
    masses = body.mass * vols / vols.sum()
    centers = np.array([p.center for p in body.parts], float)
    com = (masses[:, None] * centers).sum(axis=0) / body.mass
    inertia = np.zeros((3, 3))
    for part, m, c in zip(body.parts, masses, centers):
        if part.kind == "cylinder":
            r, h = part.size[0], 2 * part.size[1]
            local = np.diag([m * (3 * r**2 + h**2) / 12] * 2 + [m * r**2 / 2])
        else:
            a, b, cc = (2 * s for s in part.size)
            local = np.diag([m * (b**2 + cc**2) / 12, m * (a**2 + cc**2) / 12, m * (a**2 + b**2) / 12])
        d = c - com
        inertia += local + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    return com, inertia


class SofaBackend(Backend):
    """SOFA FEM backend (hyperelastic tetrahedra, Lagrange-multiplier frictional contact)."""

    name: ClassVar[str] = "sofa"
    capabilities: ClassVar[frozenset[Capability]] = frozenset({
        Capability.LINEAR_ELASTIC,
        Capability.STVK,
        Capability.NEO_HOOKEAN,
        Capability.STABLE_NEO_HOOKEAN,
        Capability.MOONEY_RIVLIN,
        Capability.OGDEN,
        Capability.DAMPING_STIFFNESS_PROPORTIONAL,
        Capability.DAMPING_MASS_PROPORTIONAL,
        Capability.TET_MESH,
        Capability.PINNED_VERTICES,
        Capability.RIGID_BODIES,
        Capability.CONTACT_FRICTION,
        Capability.CONTACT_FORCES,
        Capability.CONTACT_ADHESION,
    })

    def __init__(self) -> None:
        self._scene: Scene | None = None
        self.root = None
        self._diverged = False
        self._adhesion = {}

    @classmethod
    def extra_checks(cls, scene: Scene) -> list[str]:
        problems = []
        for body in scene.soft_bodies:
            e = body.material.elastic
            if isinstance(e, Ogden) and len(e.mu) > 1:
                problems.append(f"{body.name}: SOFA's Ogden material has a single term ({len(e.mu)} given)")
        return problems

    # -- build --------------------------------------------------------------

    def build(self, scene: Scene) -> None:
        self.check(scene)
        Sofa = import_sofa()
        self.close()
        self._scene = scene
        sim, contact = scene.simulation, scene.contact
        root = Sofa.Core.Node("root")
        root.addObject("RequiredPlugin", pluginName=PLUGINS)
        root.gravity = list(sim.gravity)
        root.dt = sim.timestep
        root.addObject("FreeMotionAnimationLoop")
        self._solver = root.addObject("BlockGaussSeidelConstraintSolver",
                                      maxIterations=int(os.environ.get("SOFTSIM_SOFA_CS_ITERS", max(sim.iterations, 50))),
                                      tolerance=float(os.environ.get("SOFTSIM_SOFA_CS_TOL", 1e-6)))
        root.addObject("CollisionPipeline")
        root.addObject("BruteForceBroadPhase")
        root.addObject("BVHNarrowPhase")
        alarm = float(os.environ.get("SOFTSIM_SOFA_ALARM", max(4 * contact.soft_radius, 4e-4)))
        root.addObject("MinProximityIntersection", alarmDistance=alarm, contactDistance=contact.soft_radius)
        frictions = [b.material.friction.sliding for b in scene.soft_bodies] + \
                    [b.friction.sliding for b in scene.rigid_bodies]
        if scene.ground.enabled:
            frictions.append(scene.ground.friction.sliding)
        root.addObject("CollisionResponse", response="FrictionContactConstraint",
                       responseParams=f"mu={max(frictions, default=0.0)}")

        if scene.ground.enabled:
            g = root.addChild(GROUND)
            # one large triangle: a shared edge under the specimen (e.g. a square's diagonal through
            # the origin) lets vertices that project exactly onto it slip through point-triangle contact
            gp = [[-10, -10, 0], [20, -10, 0], [-10, 20, 0]]
            g.addObject("MeshTopology", position=gp, triangles=[[0, 1, 2]])
            g.addObject("MechanicalObject", position=gp)
            g.addObject("TriangleCollisionModel", moving=False, simulated=False, group=GROUND_GROUP)

        self._adhesion = {f.soft_name: f for f in adhesion_fields(scene)}
        self._soft, self._rigid = {}, {}
        for body in scene.soft_bodies:
            self._soft[body.name] = self._add_soft(root, body, contact.soft_radius)
        for body in scene.rigid_bodies:
            self._rigid[body.name] = self._add_rigid(root, body)
        Sofa.Simulation.init(root)
        self.root = root
        self._sofa = Sofa
        self._prev = None  # no step taken yet: no contact forces to report
        self._diverged = False

    def _add_soft(self, root, body: SoftBodySpec, lift: float) -> dict:
        rot = _quat_to_mat(body.pose.quaternion)
        offset = np.asarray(body.pose.position) + [0.0, 0.0, lift]
        pts = body.mesh.points @ rot.T + offset
        node = root.addChild(body.name)
        alpha, beta = rayleigh(body)
        node.addObject("EulerImplicitSolver", rayleighStiffness=beta, rayleighMass=alpha)
        node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixMat3x3d")
        node.addObject("TetrahedronSetTopologyContainer", position=pts.tolist(), tetrahedra=body.mesh.tets.tolist())
        dofs = node.addObject("MechanicalObject", template="Vec3d", position=pts.tolist())
        node.addObject("MeshMatrixMass", massDensity=body.effective_density, lumping=True)
        ff, args = force_field_args(body)
        node.addObject(ff, **args)
        if body.pinned:
            node.addObject("FixedProjectiveConstraint", indices=list(body.pinned))
        node.addObject("LinearSolverConstraintCorrection")
        adhesion = None
        if body.name in self._adhesion:
            vertices = self._adhesion[body.name].vertices
            adhesion = node.addObject("ConstantForceField", name="adhesion", indices=vertices.tolist(),
                                      forces=np.zeros((len(vertices), 3)).tolist())
        coll = node.addChild("collision")
        coll.addObject("MeshTopology", position=pts.tolist(), triangles=body.mesh.surface_triangles().tolist())
        coll.addObject("MechanicalObject", position=pts.tolist())
        # pinned bodies don't collide with the ground: contact constraints on fixed vertices
        # (zero compliance) make the constraint solve singular (NaN); the pins hold them anyway
        groups = {"group": GROUND_GROUP} if body.pinned else {}
        coll.addObject("TriangleCollisionModel", **groups)
        coll.addObject("PointCollisionModel", **groups)  # line-line tests omitted: ~3x faster, same result here
        coll.addObject("IdentityMapping")
        masses = body.mesh.lumped_masses(body.effective_density)
        return {"dofs": dofs, "masses": masses, "offset": offset, "rot": rot, "adhesion": adhesion}

    def _add_rigid(self, root, body: RigidBodySpec) -> dict:
        com_local, inertia = rigid_mass_properties(body)
        rot = _quat_to_mat(body.pose.quaternion)
        com_world = np.asarray(body.pose.position) + rot @ com_local
        w, x, y, z = body.pose.quaternion
        node = root.addChild(body.name)
        node.addObject("EulerImplicitSolver")
        # direct solve: CG hits its denominator threshold (NaN) for light bodies with small inertia
        node.addObject("SparseLDLSolver", template="CompressedRowSparseMatrixd")
        dofs = node.addObject("MechanicalObject", template="Rigid3d",
                              position=[[*com_world, x, y, z, w]],
                              velocity=[[*body.linear_velocity, *body.angular_velocity]])
        volume = sum(p.volume() for p in body.parts)
        node.addObject("UniformMass", vertexMass=[body.mass, volume, *(inertia / body.mass).ravel()])
        node.addObject("UncoupledConstraintCorrection")
        adhesion = (node.addObject("ConstantForceField", name="adhesion", template="Rigid3d", indices=[0],
                                   forces=[[0.0] * 6]) if self._adhesion else None)
        coll = node.addChild("collision")
        meshes = [part_mesh(p) for p in body.parts]
        pts, tris, n = [], [], 0
        for p_, t_ in meshes:
            pts.append(p_ - com_local)
            tris.append(t_ + n)
            n += len(p_)
        pts, tris = np.vstack(pts), np.vstack(tris)
        coll.addObject("MeshTopology", position=pts.tolist(), triangles=tris.tolist())
        coll.addObject("MechanicalObject", position=pts.tolist())
        coll.addObject("TriangleCollisionModel")
        coll.addObject("LineCollisionModel")
        coll.addObject("PointCollisionModel")
        coll.addObject("RigidMapping")
        return {"dofs": dofs, "com_local": com_local, "mass": body.mass, "adhesion": adhesion}

    # -- runtime --------------------------------------------------------------

    def _velocities(self) -> dict[str, np.ndarray]:
        out = {n: np.array(d["dofs"].velocity.value, float) for n, d in self._soft.items()}
        out.update({n: np.array(d["dofs"].velocity.value[0], float) for n, d in self._rigid.items()})
        return out

    @staticmethod
    def _blown_up(velocities: dict[str, np.ndarray]) -> bool:
        """Non-finite or faster than MAX_SPEED (linear velocity; rigid angular rates excluded)."""
        return any(not np.isfinite(v).all() or np.abs(v[..., :3]).max(initial=0.0) > MAX_SPEED
                   for v in velocities.values())

    def step(self, n: int = 1) -> None:
        dt = self.scene.simulation.timestep
        for _ in range(n):
            if self._diverged:
                return
            self._prev = self._velocities()
            if self._blown_up(self._prev):
                self._diverged = True
                return
            if self._adhesion:
                self._apply_adhesion()
            self._sofa.Simulation.animate(self.root, dt)

    def _apply_adhesion(self) -> None:
        """Recompute adhesive forces from the current state and load them into the force fields."""
        frames = {}
        for name, d in self._rigid.items():
            p = np.asarray(d["dofs"].position.value[0], float)
            rot = _quat_to_mat(np.array([p[6], p[3], p[4], p[5]]))
            frames[name] = RigidFrame(p[:3] - rot @ d["com_local"], rot, p[:3].copy())
        totals = {name: np.zeros(6) for name in self._rigid}
        for name, field in self._adhesion.items():
            d = self._soft[name]
            positions = np.asarray(d["dofs"].position.value, float)[field.vertices]
            result = field.compute(positions, frames)
            d["adhesion"].forces.value = result.vertex_forces.tolist()
            for other, (force, torque, _point) in result.wrenches.items():
                if other in totals:
                    totals[other] += np.r_[force, torque]
        for name, d in self._rigid.items():
            if d["adhesion"] is not None:
                d["adhesion"].forces.value = [totals[name].tolist()]

    def reset(self) -> None:
        for field in self._adhesion.values():
            field.reset()
        self._sofa.Simulation.reset(self.root)
        self._prev = None
        self._diverged = False

    @property
    def time(self) -> float:
        return float(self.root.time.value) if self.root is not None else 0.0

    @property
    def scene(self) -> Scene:
        if self._scene is None:
            raise SoftSimError("backend has no scene; call build() first")
        return self._scene

    def is_finite(self) -> bool:
        if self._diverged or self._blown_up(self._velocities()):
            return False
        arrays = [np.asarray(d["dofs"].position.value) for d in (*self._soft.values(), *self._rigid.values())]
        return all(np.isfinite(a).all() for a in arrays)

    def state(self) -> SimState:
        rigid = {}
        for name, d in self._rigid.items():
            p = np.array(d["dofs"].position.value[0], float)
            v = np.array(d["dofs"].velocity.value[0], float)
            q = np.array([p[6], p[3], p[4], p[5]])  # SOFA (x, y, z, w) -> (w, x, y, z)
            rot = _quat_to_mat(q)
            origin = p[:3] - rot @ d["com_local"]
            omega = v[3:]
            rigid[name] = RigidState(origin, q, v[:3] + np.cross(omega, origin - p[:3]), omega, p[:3].copy())
        soft = {}
        for name, d in self._soft.items():
            pos = np.array(d["dofs"].position.value, float) - [0.0, 0.0, d["offset"][2] - self.scene.soft(name).pose.position[2]]
            soft[name] = SoftState(pos, np.array(d["dofs"].velocity.value, float))
        return SimState(self.time, rigid, soft, self._contacts())

    def _contacts(self) -> list[ContactWrench]:
        """Net contact forces from momentum balance (implicit Euler: m dv/dt = f_ext + f_contact)."""
        if self._prev is None:
            return []
        dt = self.scene.simulation.timestep
        g = np.asarray(self.scene.simulation.gravity)
        now = self._velocities()
        out = []
        on_rigid: dict[str, np.ndarray] = {}
        for name, d in self._rigid.items():
            f = d["mass"] * (now[name][:3] - self._prev[name][:3]) / dt - d["mass"] * g
            on_rigid[name] = f
        for name, d in self._soft.items():
            m = d["masses"]
            total = (m[:, None] * (now[name] - self._prev[name])).sum(axis=0) / dt - m.sum() * g
            com = (m[:, None] * np.asarray(d["dofs"].position.value)).sum(axis=0) / m.sum()
            from_rigid = np.zeros(3)
            for rname, f in on_rigid.items():  # single-specimen scenes: rigid contacts are with this body
                if np.linalg.norm(f) > 0:
                    pt = np.asarray(self._rigid[rname]["dofs"].position.value[0][:3])
                    out.append(ContactWrench(name, rname, f, np.zeros(3), pt, 0))
                    from_rigid += -f
            if self.scene.ground.enabled:
                out.append(ContactWrench(GROUND, name, total - from_rigid, np.zeros(3), com * [1, 1, 0], 0))
        return out

    def export_model(self) -> str | None:
        if self.root is None:
            return None
        lines = ["# SOFA scene generated by softsim (node / component tree)"]

        def walk(node, depth):
            lines.append("  " * depth + f"[{node.name.value}]")
            for obj in node.objects:
                lines.append("  " * (depth + 1) + f"{obj.getClassName()} name={obj.name.value}")
            for child in node.children:
                walk(child, depth + 1)

        walk(self.root, 0)
        return "\n".join(lines)

    def close(self) -> None:
        if self.root is not None:
            self._sofa.Simulation.unload(self.root)
            self.root = None
