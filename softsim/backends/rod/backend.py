"""Rod backend: a slender soft body run as a chain of beams on SOFA BeamAdapter.

    softsim run scenarios/actuation/cable_arm.yaml --backend rod

The body (a column along z, clamped at its z-min face) is reduced to beam frames by
:func:`softsim.backends.rod.reduce.reduce_body`: per-beam extensional and bending stiffnesses from the
tetrahedra (material zones included), carried to BeamAdapter as an equivalent rectangle and modulus per beam.
The mesh follows the frames through the vertex map, so every region metric of the scenario works unchanged.

* Geometrically nonlinear (finite rotations of the frames), materially linear: each element contributes its
  small-strain modulus, so strains in the section must stay small (a beam bending to a tip angle of a radian
  is fine; a squashed or inflated body is not a rod).
* Cables and region forces use the actuation functions of the 3D backends on the *deformed vertices* and are
  carried to the frames by the transpose of the vertex map (virtual work), explicitly once per step.
* Damping: stiffness-proportional (Rayleigh beta) only; the solver's mass-proportional alpha does not act on
  BeamAdapter's mass (measured, tests/test_rod_backend.py), so it is refused.
* Refused: pressure, contact and ground, rigid bodies, moving and rotating mounts, prescribed motion, Prony,
  zones with their own damping, more than one soft body, bodies that are not slender columns.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np

from softsim.actuation import (CableActuator, ForceActuator, cable_forces, cable_length, region_force_weights)
from softsim.backends.base import Backend, SimState, SoftState
from softsim.backends.rod.reduce import RodModel, reduce_body, slender_problems
from softsim.backends.sofa.backend import import_sofa, rayleigh
from softsim.capabilities import Capability
from softsim.errors import SoftSimError
from softsim.scene import Scene

PLUGINS = ("Sofa.Component.StateContainer", "Sofa.Component.Mass", "Sofa.Component.ODESolver.Backward",
           "Sofa.Component.LinearSolver.Direct", "Sofa.Component.Constraint.Projective",
           "Sofa.Component.Topology.Container.Constant", "Sofa.Component.MechanicalLoad",
           "Sofa.Component.AnimationLoop", "BeamAdapter")


class RodBackend(Backend):
    """See module docstring."""

    name: ClassVar[str] = "rod"
    capabilities: ClassVar[frozenset[Capability]] = frozenset({
        Capability.LINEAR_ELASTIC, Capability.STVK, Capability.NEO_HOOKEAN, Capability.STABLE_NEO_HOOKEAN,
        Capability.MOONEY_RIVLIN, Capability.OGDEN, Capability.YEOH, Capability.GENT, Capability.ARRUDA_BOYCE,
        Capability.DAMPING_STIFFNESS_PROPORTIONAL,  # mass-proportional (alpha) has no effect on BeamAdapter's mass (measured): refused
        Capability.TET_MESH, Capability.PINNED_VERTICES, Capability.CABLE_ACTUATION, Capability.REGION_FORCE,
        Capability.MULTI_MATERIAL,
        Capability.CONTACT_FRICTION,  # a material's friction coefficient is harmless: nothing is touched (ground refused below)
    })

    def __init__(self) -> None:
        self._scene: Scene | None = None
        self.root = None
        self.model: RodModel | None = None

    @classmethod
    def extra_checks(cls, scene: Scene) -> list[str]:
        problems = []
        if scene.ground.enabled:
            problems.append("a ground plane (the rod backend has no contact)")
        if len(scene.soft_bodies) != 1:
            problems.append(f"{len(scene.soft_bodies)} soft bodies (the rod backend runs exactly one)")
        for body in scene.soft_bodies:
            problems += slender_problems(body)
            if body.zone_damping_differs():
                problems.append(f"{body.name}: zones {body.zone_damping_differs()} with their own damping (the "
                                "Rayleigh damping belongs to the solver, one per body)")
        return problems

    # -- build ------------------------------------------------------------------------------------------

    def build(self, scene: Scene) -> None:
        self.check(scene)
        Sofa = import_sofa()
        self.close()
        self._scene, self._sofa = scene, Sofa
        self._clear_controls()
        body = scene.soft_bodies[0]
        self.model = model = reduce_body(body)
        alpha, beta = rayleigh(body)
        root = Sofa.Core.Node("root")
        for plugin in PLUGINS:
            root.addObject("RequiredPlugin", pluginName=plugin)
        root.gravity = model.to_internal(np.asarray(scene.simulation.gravity, float)).tolist()
        root.dt = scene.simulation.timestep
        root.addObject("DefaultAnimationLoop")
        root.addObject("EulerImplicitSolver", rayleighStiffness=beta, rayleighMass=alpha)
        root.addObject("BTDLinearSolver")
        n = model.n_beams
        rest = model.rest_frames()
        root.addObject("MechanicalObject", name="frames", template="Rigid3d", position=rest.tolist())
        root.addObject("MeshTopology", name="topology", edges=[[i, i + 1] for i in range(n)])
        ly, lz, young = model.equivalent_section()
        poisson = float(body.material.elastic.initial_poisson_ratio())
        root.addObject("BeamInterpolation", name="interpolation", crossSectionShape="rectangular",
                       lengthY=ly.tolist(), lengthZ=lz.tolist(), defaultYoungModulus=young.tolist(),
                       defaultPoissonRatio=[poisson] * n, dofsAndBeamsAligned=True)
        root.addObject("AdaptiveBeamForceFieldAndMass", name="field", interpolation="@interpolation",
                       massDensity=(model.mass_per_length / (ly * lz)).tolist(), computeMass=True)
        root.addObject("FixedProjectiveConstraint", indices=[0])
        self._load = root.addObject("ConstantForceField", name="actuation", indices=list(range(n + 1)),
                                    forces=np.zeros((n + 1, 6)).tolist(), showArrowSize=0)
        self._loads = self._make_loads(scene, body)
        Sofa.Simulation.init(root)
        self.root = root

    def _make_loads(self, scene: Scene, body) -> list:
        """(actuator name, force function of the body-frame vertex positions and the actuator's value)."""
        loads = []
        for a in scene.actuators:
            if a.body != body.name:
                continue
            if isinstance(a, CableActuator):
                nodes, bary = body.cable_rows(a)
                rest = cable_length(body.mesh.points, nodes, bary)
                loads.append((a.name, lambda x, v, a=a, n=nodes, w=bary, rest=rest: cable_forces(
                    x, n, w, a.tension_at(cable_length(x, n, w) if a.stroke_controlled else 0.0, v, rest))))
            elif isinstance(a, ForceActuator):
                share = region_force_weights(body.mesh.points, body.region(a.region))[:, None] * np.array(a.direction)
                loads.append((a.name, lambda x, v, share=share: v * share))
        return loads

    # -- stepping -----------------------------------------------------------------------------------------

    def _frames(self) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(self.root.frames.position.value, float)
        return p[:, :3], p[:, 3:]

    def _vertices(self) -> np.ndarray:
        pos, quat = self._frames()
        return self.model.vertex_positions(pos, quat)

    def step(self, n: int = 1) -> None:
        dt = self.scene.simulation.timestep
        body = self.scene.soft_bodies[0]
        origin = np.asarray(body.pose.position)
        for _ in range(n):
            if self._loads:
                pos, quat = self._frames()
                x = self.model.vertex_positions(pos, quat) - origin
                f = sum(load(x, self.control(name)) for name, load in self._loads)
                self._load.forces.value = self.model.frame_wrenches(pos, quat, f).tolist()
            self._sofa.Simulation.animate(self.root, dt)

    def reset(self) -> None:
        self._sofa.Simulation.reset(self.root)

    @property
    def time(self) -> float:
        return float(self.root.time.value) if self.root is not None else 0.0

    @property
    def scene(self) -> Scene:
        if self._scene is None:
            raise SoftSimError("backend has no scene; call build() first")
        return self._scene

    def is_finite(self) -> bool:
        return bool(np.isfinite(np.asarray(self.root.frames.position.value)).all())

    def state(self) -> SimState:
        pos, quat = self._frames()
        v = np.asarray(self.root.frames.velocity.value, float)
        velocities = self.model.vertex_velocities(pos, quat, v[:, :3], v[:, 3:])
        soft = {self.scene.soft_bodies[0].name: SoftState(self.model.vertex_positions(pos, quat), velocities)}
        return SimState(self.time, {}, soft, [], self.controls())

    def close(self) -> None:
        if self.root is not None:
            self._sofa.Simulation.unload(self.root)
            self.root = None
