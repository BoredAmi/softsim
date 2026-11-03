"""MuJoCo flex backend runtime."""

from __future__ import annotations

import dataclasses
import warnings
from typing import ClassVar

import mujoco
import numpy as np

from softsim.actuation import (CableActuator, ForceActuator, PressureActuator, cable_forces, cable_length,
                               pressure_forces, region_force_weights)
from softsim.backends.base import Backend, ContactWrench, RigidState, SimState, SoftState
from softsim.backends.mujoco.builder import GROUND, build_spec, prescribed_equality, scene_to_mjcf
from softsim.capabilities import Capability
from softsim.contact.adhesion import AdhesionField, RigidFrame, adhesion_fields
from softsim.errors import SoftSimError
from softsim.materials.models import StableNeoHookean
from softsim.materials.viscoelastic import PronyOverstress, native_scene, prony_of
from softsim.scene import Scene, rotating_mount_problems

_ELASTIC_SOLVERS = {"CG", "Newton"}


class MujocoBackend(Backend):
    """MuJoCo (>= 3.17) volumetric flex backend.

    Physics notes (see docs/backends/mujoco.md):
        * 3D flex elements are linear tetrahedra; nearly incompressible
          materials lock, so results depend on mesh size.
        * ``stvk`` is MuJoCo's default flex model; ``stable_neo_hookean`` uses the
          experimental Stable Neo-Hookean model (MuJoCo 3.15+). Classic ``neo_hookean``
          (ln J form) is not available.
        * Native damping is stiffness-proportional (Rayleigh beta) only. Prony
          viscoelasticity: the flex carries the long-term law and the Maxwell-branch
          overstresses (``softsim.materials.viscoelastic``) are explicit vertex forces.
        * Adhesion, the Prony overstress and pressure and cable actuators are applied as explicit
          external forces (``xfrc_applied``) on vertices and rigid bodies, recomputed
          every step.
        * Prescribed regions: a joint equality per held axis of each vertex (timeconst 2 steps,
          path velocity fed forward); the reaction is their constraint force. Only all three
          axes may be held on a rotated body (vertices slide along the body's axes).
    """

    name: ClassVar[str] = "mujoco"
    capabilities: ClassVar[frozenset[Capability]] = frozenset({
        Capability.STVK,
        Capability.STABLE_NEO_HOOKEAN,
        Capability.DAMPING_STIFFNESS_PROPORTIONAL,
        Capability.TET_MESH,
        Capability.PINNED_VERTICES,
        Capability.RIGID_BODIES,
        Capability.SPRING_MOUNT,
        Capability.CONTACT_FRICTION,
        Capability.TORSIONAL_ROLLING_FRICTION,
        Capability.SELF_COLLISION,
        Capability.CONTACT_FORCES,
        Capability.CONTACT_ADHESION,
        Capability.VISCOELASTIC_PRONY,
        Capability.PRESSURE_ACTUATION,
        Capability.CABLE_ACTUATION,
        Capability.REGION_FORCE,
        Capability.MOVING_MOUNT,
        Capability.ROTATING_MOUNT,
        Capability.PRESCRIBED_MOTION,
        Capability.MULTI_MATERIAL,
    })

    def __init__(self) -> None:
        self._scene: Scene | None = None
        self._pressure: list = []
        self._cables: list = []
        self._forces: list = []
        self.spec: mujoco.MjSpec | None = None
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self._adhesion: list[AdhesionField] = []
        self._prony: list[tuple[PronyOverstress, np.ndarray, np.ndarray, np.ndarray]] = []
        self._adhesion_last: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    @classmethod
    def extra_checks(cls, scene: Scene) -> list[str]:
        problems = []
        sim = scene.simulation
        if scene.soft_bodies and sim.integrator != "discrete":
            problems.append(
                f"integrator '{sim.integrator}': MuJoCo >= 3.13 integrates flex elasticity only "
                "with integrator 'discrete'")
        if scene.soft_bodies and sim.solver not in _ELASTIC_SOLVERS:
            problems.append(f"solver '{sim.solver}': flex elasticity needs CG or Newton")
        problems += rotating_mount_problems(scene)
        for body in scene.soft_bodies:
            if body.material.elastic.initial_poisson_ratio() < 0:
                problems.append(f"{body.name}: negative Poisson's ratio (MuJoCo accepts [0, 0.5))")
            for z in body.zones:
                if type(z.material.elastic) is not type(body.material.elastic):
                    problems.append(f"{body.name}: material zone {z.name!r} needs the body's elastic model "
                                    f"({body.material.elastic.model}; one flex has one model)")
                if prony_of(body.material) is not None:
                    problems.append(f"{body.name}: material zones with Prony viscoelasticity")
            if body.zone_damping_differs():
                problems.append(f"{body.name}: zones {body.zone_damping_differs()} with their own damping (one flex "
                                "has one damping coefficient; SOFA has one per material)")
            rotated = not np.allclose(body.pose.quaternion, (1.0, 0.0, 0.0, 0.0))
            for p in body.prescribed:
                if rotated and not all(p.axes):
                    problems.append(f"{body.name}: prescribed region {p.region!r} holds only some world axes of a "
                                    "rotated body (a vertex slides along the body's axes); hold all three or "
                                    "keep the body unrotated")
        return problems

    # -- lifecycle ---------------------------------------------------------

    def build(self, scene: Scene) -> None:
        self.check(scene)
        self._scene = scene
        self._clear_controls()
        native = native_scene(scene)  # Prony cards: the long-term law; overstresses are added each step
        self.mjcf = scene_to_mjcf(native)
        self.spec = build_spec(native, self.mjcf)
        neo = [b.name for b in scene.soft_bodies if isinstance(b.material.elastic, StableNeoHookean)]
        self.spec_note = (
            "Generated by softsim. Applied via MjSpec after parsing (not expressible in MJCF): "
            f"lumped vertex masses for {[b.name for b in scene.soft_bodies]}"
            + (f"; Stable Neo-Hookean elasticity (elastic3d=1) for {neo}" if neo else "")
        )
        self.model = self._compile(self.spec)
        self._apply_zones(native)
        self.data = mujoco.MjData(self.model)
        self._kin = mujoco.MjData(self.model)  # scratch: kinematics at the current state (``_current``)
        self._index()
        self.reset()

    @staticmethod
    def _compile(spec) -> "mujoco.MjModel":
        try:
            with warnings.catch_warnings():
                # MuJoCo counts only flex damping as "passive forces" here; the 3D elasticity
                # (elastic3d, set on the spec) is active all the same
                warnings.filterwarnings("ignore", message=".*has no equality constraints or passive forces")
                return spec.compile()
        except ValueError as exc:
            raise SoftSimError(f"MuJoCo rejected the generated model: {exc}") from None

    @staticmethod
    def _stiffness_rows(m: "mujoco.MjModel", flex: str) -> np.ndarray:
        """View of a flex's per-element precomputed elasticity data (elements, values per element)."""
        f = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, flex)
        n = m.flex_elemnum[f]
        end = m.flex_stiffnessadr[f + 1] if f + 1 < m.nflex else len(m.flex_stiffness)
        return m.flex_stiffness[m.flex_stiffnessadr[f]:end].reshape(n, -1)

    def _apply_zones(self, native: Scene) -> None:
        """Material zones. MuJoCo precomputes each element's elasticity data from the flex's material
        (24 values: the material constants times element geometry, and geometry alone), so the zone
        elements' rows are copied from a model compiled with the zone's material on the whole body."""
        for body in native.soft_bodies:
            if not body.zones:
                continue
            rows = self._stiffness_rows(self.model, body.name)
            for z in body.zones:
                probe = native.with_material(body.name, z.material)
                probe = dataclasses.replace(probe, soft_bodies=tuple(
                    dataclasses.replace(b, zones=()) if b.name == body.name else b for b in probe.soft_bodies))
                rows[np.asarray(z.elements)] = self._stiffness_rows(
                    self._compile(build_spec(probe, scene_to_mjcf(probe))), body.name)[np.asarray(z.elements)]

    def _index(self) -> None:
        m = self.model
        self._rigid_ids = {b.name: m.body(b.name).id for b in self._scene.rigid_bodies}
        self._free_dof = {b.name: m.jnt_dofadr[m.joint(f"{b.name}/free").id]
                          for b in self._scene.rigid_bodies if b.mount == "free"}
        self._flex = {}
        for body in self._scene.soft_bodies:
            fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_FLEX, body.name)
            adr, num = m.flex_vertadr[fid], m.flex_vertnum[fid]
            bodies = m.flex_vertbodyid[adr:adr + num]
            # each free vertex body carries 3 slide joints along its local x, y, z
            dofs = np.full((num, 3), -1)
            for i, bid in enumerate(bodies):
                if m.body_dofnum[bid] == 3:
                    dofs[i] = m.body_dofadr[bid] + np.arange(3)
            self._flex[body.name] = (adr, num, bodies, dofs)
        self._owner_geom = {}
        for gid in range(m.ngeom):
            bid = m.geom_bodyid[gid]
            self._owner_geom[gid] = GROUND if bid == 0 else m.body(bid).name
        self._owner_flex = {i: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_FLEX, i) for i in range(m.nflex)}
        self._prony = []
        for body in self._scene.soft_bodies:
            if prony_of(body.material) is not None:
                adr, num, bodies, dofs = self._flex[body.name]
                overstress = PronyOverstress(body.material, body.mesh.points, body.mesh.tets)
                self._prony.append((overstress, adr + np.arange(num), bodies, dofs[:, 0] >= 0))
        self._pressure, self._cables, self._forces = [], [], []
        for a in self._scene.actuators:
            adr, num, bodies, dofs = self._flex[a.body]
            soft = self._scene.soft(a.body)
            if isinstance(a, PressureActuator):
                self._pressure.append((a.name, soft.region(a.region), adr + np.arange(num), bodies, dofs[:, 0] >= 0))
            elif isinstance(a, CableActuator):
                nodes, bary = soft.cable_rows(a)
                self._cables.append((a, nodes, bary, adr + np.arange(num), bodies, dofs[:, 0] >= 0,
                                     cable_length(soft.mesh.points, nodes, bary)))
            elif isinstance(a, ForceActuator):
                share = region_force_weights(soft.mesh.points, soft.region(a.region))[:, None] * np.array(a.direction)
                self._forces.append((a.name, share, bodies, dofs[:, 0] >= 0))
        # (body, region, path, equality ids, their axes, a vertex body: its frame is every vertex's axes)
        self._prescribed = []
        for body in self._scene.soft_bodies:
            for p in body.prescribed:
                pairs = [(int(i), k) for i in p.vertices for k in range(3) if p.axes[k]]
                ids = np.array([m.equality(prescribed_equality(body.name, p.region, i, k)).id for i, k in pairs])
                self._prescribed.append((body.name, p.region, p.path, ids, np.array([k for _, k in pairs]),
                                         m.body(f"{body.name}_{int(p.vertices[0])}").id))
        self._masses = {b.name: b.vertex_masses() for b in self._scene.soft_bodies}
        self._adhesion = adhesion_fields(self._scene)
        self._adhesion_bodies = {}
        for f in self._adhesion:
            adr, num, bodies, dofs = self._flex[f.soft_name]
            movable = dofs[f.vertices, 0] >= 0  # pinned vertices take no external force
            self._adhesion_bodies[f.soft_name] = (adr + f.vertices, bodies[f.vertices], movable)

    def reset(self) -> None:
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for body in self._scene.rigid_bodies:
            if body.name not in self._free_dof:  # mounted bodies start at rest
                continue
            dof = self._free_dof[body.name]
            d.qvel[dof:dof + 3] = body.linear_velocity
            rot = np.zeros(9)
            mujoco.mju_quat2Mat(rot, np.array(body.pose.quaternion, dtype=float))
            d.qvel[dof + 3:dof + 6] = rot.reshape(3, 3).T @ np.array(body.angular_velocity)  # local frame
        for f in self._adhesion:
            f.reset()
        for overstress, *_ in self._prony:
            overstress.reset()
        self._adhesion_last = {}
        mujoco.mj_forward(m, d)

    def step(self, n: int = 1) -> None:
        mount = self._scene.mount is not None or self._scene.spin is not None
        if not (self._adhesion or self._prony or self._pressure or self._cables or self._forces or mount
                or self._prescribed):
            mujoco.mj_step(self.model, self.data, nstep=n)
            return
        dt = self.model.opt.timestep
        for _ in range(n):
            if mount:  # in the mount's frame
                self.model.opt.gravity[:] = self.mount_gravity(self.data.time + 0.5 * dt)
            self._drive_prescribed(self.data.time + dt)  # before mj_step1: it builds the constraints
            # mj_step computes positions before it integrates, so after it ``data``'s positions are
            # a step old: split it, and compute the state-dependent forces from the current positions
            mujoco.mj_step1(self.model, self.data)
            self._apply_external()
            mujoco.mj_step2(self.model, self.data)

    def _drive_prescribed(self, t: float) -> None:
        """Targets of the prescribed regions' joint equalities for a step ending at ``t``: the path plus
        its velocity times 2 timeconst, which cancels the lag of MuJoCo's critically damped constraint
        (reference acceleration -2 v / timeconst - error / timeconst^2: a target moving at v trails by
        2 timeconst v). Measured: 0.75 um lag -> 0.006 um on a 200 um minimum-jerk pull in 0.2 s."""
        for _, _, path, ids, axes, bid in self._prescribed:
            lead = 2.0 * self.model.eq_solref[ids[0], 0]
            local = self.data.xmat[bid].reshape(3, 3).T @ (path.position(t) + lead * path.velocity(t))
            self.model.eq_data[ids, 0] = local[axes]

    def _reactions(self) -> dict[str, dict[str, np.ndarray]]:
        """Force each prescribed region's support exerted on the body in the last step: the sum of its
        joint equalities' constraint forces, per body axis, rotated to the world."""
        d = self.data
        out: dict[str, dict[str, np.ndarray]] = {}
        if not self._prescribed:
            return out
        rows = d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        per_eq = np.bincount(d.efc_id[rows], weights=d.efc_force[rows], minlength=self.model.neq)
        for body, region, _, ids, axes, bid in self._prescribed:
            local = np.bincount(axes, weights=per_eq[ids], minlength=3)
            out.setdefault(body, {})[region] = d.xmat[bid].reshape(3, 3) @ local
        return out

    def _apply_external(self) -> None:
        """Recompute the explicit forces (adhesion, Prony overstress, actuators) from the current state
        into ``xfrc_applied``."""
        d = self.data
        d.xfrc_applied[:] = 0.0
        dt = self.model.opt.timestep
        for overstress, rows, bodies, movable in self._prony:
            f = overstress.forces(d.flexvert_xpos[rows], dt)
            np.add.at(d.xfrc_applied, (bodies[movable], slice(0, 3)), f[movable])
        for name, tris, rows, bodies, movable in self._pressure:
            f = pressure_forces(d.flexvert_xpos[rows], tris, self.control(name))
            np.add.at(d.xfrc_applied, (bodies[movable], slice(0, 3)), f[movable])
        for a, nodes, weights, rows, bodies, movable, rest in self._cables:
            x = d.flexvert_xpos[rows]
            tension = a.tension_at(cable_length(x, nodes, weights) if a.stroke_controlled else 0.0,
                                   self.control(a.name), rest)
            f = cable_forces(x, nodes, weights, tension)
            np.add.at(d.xfrc_applied, (bodies[movable], slice(0, 3)), f[movable])
        for name, share, bodies, movable in self._forces:
            np.add.at(d.xfrc_applied, (bodies[movable], slice(0, 3)), self.control(name) * share[movable])
        if self._scene.spin is not None:
            self._apply_fictitious(d.time + 0.5 * dt)
        self._apply_adhesion()

    def _apply_fictitious(self, t: float) -> None:
        """A rotating mount's Euler, centrifugal and Coriolis forces, ``m`` times
        ``MountFrame.fictitious`` at the current positions and velocities, with the frame at ``t`` (the
        step's middle, like gravity); explicit, like the other loads."""
        d = self.data
        frame = self._scene.frame(t)
        lift = np.array([0.0, 0.0, self._scene.contact.soft_radius])
        for name, (adr, num, bodies, dofs) in self._flex.items():
            free = dofs[:, 0] >= 0
            v = np.zeros((num, 3))
            v[free] = np.einsum("nij,nj->ni", d.xmat[bodies[free]].reshape(-1, 3, 3), d.qvel[dofs[free]])
            f = self._masses[name][:, None] * frame.fictitious(d.flexvert_xpos[adr:adr + num] - lift, v)
            np.add.at(d.xfrc_applied, (bodies[free], slice(0, 3)), f[free])

    def _apply_adhesion(self) -> None:
        """Add adhesive forces from the current state to ``xfrc_applied``."""
        d = self.data
        self._adhesion_last = {}
        if not self._adhesion:
            return
        frames = {name: RigidFrame(d.xpos[bid].copy(), d.xmat[bid].reshape(3, 3).copy(), d.xipos[bid].copy())
                  for name, bid in self._rigid_ids.items()}
        for f in self._adhesion:
            xpos_rows, bodies, movable = self._adhesion_bodies[f.soft_name]
            result = f.compute(d.flexvert_xpos[xpos_rows], frames)
            np.add.at(d.xfrc_applied, (bodies[movable], slice(0, 3)), result.vertex_forces[movable])
            for name, (force, torque, point) in result.wrenches.items():
                if name in self._rigid_ids:
                    d.xfrc_applied[self._rigid_ids[name], :3] += force
                    d.xfrc_applied[self._rigid_ids[name], 3:] += torque
                self._adhesion_last[(f.soft_name, name)] = (force, torque, point)

    @property
    def time(self) -> float:
        return float(self.data.time)

    @property
    def scene(self) -> Scene:
        if self._scene is None:
            raise SoftSimError("backend has no scene; call build() first")
        return self._scene

    _DIVERGENCE_WARNINGS = (mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADQPOS,
                            mujoco.mjtWarning.mjWARN_BADQVEL)

    def is_finite(self) -> bool:
        # MuJoCo resets the state itself when it detects NaN/inf, so the warning
        # counters are the reliable divergence signal, not the (reset) state.
        d = self.data
        if any(d.warning[w].number for w in self._DIVERGENCE_WARNINGS):
            return False
        return bool(np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all())

    def export_model(self) -> str:
        """Compiled spec as MJCF, or (when it holds spec-only settings) the
        generated MJCF annotated with what was applied on top of it."""
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*has no equality constraints or passive forces")
                return self.spec.to_xml()
        except mujoco.FatalError:
            return f"<!-- {self.spec_note} -->\n{self.mjcf}"

    def diagnostics(self) -> dict[str, int]:
        """Counts of MuJoCo warnings raised since the last reset."""
        return {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(self.data.warning) if w.number}

    # -- state -------------------------------------------------------------

    def _current(self) -> "mujoco.MjData":
        """Positions and velocities at the current state (``qpos``, ``qvel``, ``time``). ``mj_step``
        computes them before it integrates, so after it ``data``'s (``flexvert_xpos``, ``xpos``,
        ``cvel``) are a step old (measured: 9.5 um on a vertex moving 94 mm/s at 0.1 ms). They are
        recomputed on a scratch copy; ``data`` keeps the step's contacts and constraint forces."""
        m, k = self.model, self._kin
        k.qpos[:], k.qvel[:], k.time = self.data.qpos, self.data.qvel, self.data.time
        mujoco.mj_kinematics(m, k)
        mujoco.mj_comPos(m, k)
        mujoco.mj_comVel(m, k)
        mujoco.mj_flex(m, k)
        return k

    def state(self) -> SimState:
        m, d = self.model, self._current()
        rigid = {}
        vel = np.zeros(6)
        for name, bid in self._rigid_ids.items():
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, bid, vel, 0)
            rigid[name] = RigidState(d.xpos[bid].copy(), d.xquat[bid].copy(), vel[3:].copy(), vel[:3].copy(),
                                     d.xipos[bid].copy())
        soft = {}
        reactions = self._reactions()
        for name, (adr, num, bodies, dofs) in self._flex.items():
            radius_offset = np.array([0.0, 0.0, self._scene.contact.soft_radius])
            vertices = d.flexvert_xpos[adr:adr + num] - radius_offset
            free = dofs[:, 0] >= 0
            velocities = np.zeros((num, 3))
            rot = d.xmat[bodies[free]].reshape(-1, 3, 3)
            velocities[free] = np.einsum("nij,nj->ni", rot, d.qvel[dofs[free]])
            soft[name] = SoftState(vertices.copy(), velocities, reactions.get(name, {}))
        return self._to_world(SimState(float(d.time), rigid, soft, self._contacts(), self.controls()))

    def _owner(self, geom: int, flex: int) -> str:
        # geom/flex ids are -1 when unused: never index with them
        return self._owner_flex[flex] if geom < 0 else self._owner_geom[geom]

    def _contacts(self) -> list[ContactWrench]:
        m, d = self.model, self.data
        acc: dict[tuple[str, str], list] = {}
        local = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            a = self._owner(c.geom[0], c.flex[0])
            b = self._owner(c.geom[1], c.flex[1])
            if a == b:
                continue
            mujoco.mj_contactForce(m, d, i, local)
            frame = c.frame.reshape(3, 3)
            force = frame.T @ local[:3]  # MuJoCo: force exerted by geom1 on geom2
            if (b, a) in acc:
                a, b, force = b, a, -force
            entry = acc.setdefault((a, b), [np.zeros(3), np.zeros(3), np.zeros(3), 0.0, 0])
            entry[0] += force
            entry[1] += np.cross(c.pos, force)
            entry[2] += abs(local[0]) * c.pos
            entry[3] += abs(local[0])
            entry[4] += 1
        out = []
        for (a, b), (force, moment, weighted, normal_sum, n) in acc.items():
            point = weighted / normal_sum if normal_sum > 0 else np.zeros(3)
            out.append(ContactWrench(a, b, force, moment - np.cross(point, force), point, n))
        return self._with_adhesion(out)

    def _with_adhesion(self, contacts: list[ContactWrench]) -> list[ContactWrench]:
        """Add the adhesive forces of the last step to the (soft, counterpart) contact wrenches."""
        if not self._adhesion_last:
            return contacts
        out = list(contacts)
        for (soft, other), (force, _torque, point) in self._adhesion_last.items():
            if not np.any(force):
                continue
            for i, c in enumerate(out):
                if {c.body_a, c.body_b} == {soft, other}:
                    sign = 1.0 if (c.body_a, c.body_b) == (soft, other) else -1.0
                    total = c.force + sign * force
                    torque = c.torque + sign * np.cross(point - c.point, force)
                    out[i] = ContactWrench(c.body_a, c.body_b, total, torque, c.point, c.n_contacts)
                    break
            else:
                out.append(ContactWrench(soft, other, force.copy(), np.zeros(3), point.copy(), 0))
        return out
