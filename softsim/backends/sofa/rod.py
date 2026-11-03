"""Rod (beam) cantilever on SOFA BeamAdapter: the slender-body model that the 3D backends are cross-checked against.

A rod is a chain of rigid frames joined by Euler-Bernoulli beam elements (``AdaptiveBeamForceFieldAndMass``): one
frame per node carries position and orientation, the section enters only through its area and second moments, so a
slender body costs a handful of degrees of freedom instead of thousands of tetrahedra. This module is the first
step of the rod backend (docs/decisions.md, entry 4): the cantilever under a tip load, checked against the
Euler-Bernoulli solution u = F L^3 / (3 E I) in tests/test_rod_sofa.py. It does not yet read a ``Scene``.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from softsim.backends.sofa.backend import import_sofa

_PLUGINS = ("Sofa.Component.StateContainer", "Sofa.Component.Mass", "Sofa.Component.ODESolver.Backward",
            "Sofa.Component.LinearSolver.Direct", "Sofa.Component.Constraint.Projective",
            "Sofa.Component.Topology.Container.Constant", "Sofa.Component.MechanicalLoad",
            "Sofa.Component.AnimationLoop", "BeamAdapter")


class RodCantileverConfig(BaseModel):
    """A straight rectangular rod along x, clamped at x = 0, loaded by a transverse tip force along -z."""

    model_config = ConfigDict(extra="forbid")

    length: float = Field(gt=0, description="Length of the rod [m].")
    width: float = Field(gt=0, description="Section size along y [m].")
    height: float = Field(gt=0, description="Section size along z, the bending direction [m].")
    youngs: float = Field(gt=0, description="Young's modulus [Pa].")
    poisson: float = Field(0.45, gt=-1, lt=0.5, description="Poisson's ratio [-].")
    density: float = Field(1000.0, gt=0, description="Mass density [kg/m^3].")
    tip_force: float = Field(description="Force at the free end along -z [N].")
    segments: int = Field(10, ge=1, description="Number of beam elements.")
    dt: float = Field(0.05, gt=0, description="Time step [s].")
    steps: int = Field(400, ge=1, description="Steps run; the damping (``rayleigh_mass``) settles the rod.")
    rayleigh_mass: float = Field(5.0, ge=0, description="Mass-proportional damping [1/s] that kills the oscillation.")


def euler_bernoulli_tip(cfg: RodCantileverConfig) -> float:
    """Tip deflection along -z [m, negative] of the Euler-Bernoulli cantilever, F L^3 / (3 E I)."""
    inertia = cfg.width * cfg.height**3 / 12
    return -cfg.tip_force * cfg.length**3 / (3 * cfg.youngs * inertia)


def rod_cantilever_tip(cfg: RodCantileverConfig) -> np.ndarray:
    """Final frame of the tip node: (x, y, z, qx, qy, qz, qw) after ``cfg.steps`` steps of BeamAdapter."""
    Sofa = import_sofa()
    root = Sofa.Core.Node("root")
    for plugin in _PLUGINS:
        root.addObject("RequiredPlugin", pluginName=plugin)
    root.addObject("DefaultAnimationLoop")
    root.gravity = [0, 0, 0]
    root.addObject("EulerImplicitSolver", rayleighStiffness=0.0, rayleighMass=cfg.rayleigh_mass)
    root.addObject("BTDLinearSolver")
    n = cfg.segments + 1
    frames = [[cfg.length * i / cfg.segments, 0, 0, 0, 0, 0, 1] for i in range(n)]
    root.addObject("MechanicalObject", name="frames", template="Rigid3d", position=frames)
    root.addObject("MeshTopology", name="topology", edges=[[i, i + 1] for i in range(cfg.segments)])
    root.addObject("BeamInterpolation", name="interpolation", crossSectionShape="rectangular", lengthY=cfg.width,
                   lengthZ=cfg.height, defaultYoungModulus=cfg.youngs, defaultPoissonRatio=cfg.poisson,
                   dofsAndBeamsAligned=True)
    root.addObject("AdaptiveBeamForceFieldAndMass", name="field", interpolation="@interpolation",
                   massDensity=cfg.density, computeMass=True)
    root.addObject("FixedProjectiveConstraint", indices=[0])
    root.addObject("ConstantForceField", indices=[n - 1], forces=[[0, 0, -cfg.tip_force, 0, 0, 0]], showArrowSize=0)
    Sofa.Simulation.init(root)
    for _ in range(cfg.steps):
        Sofa.Simulation.animate(root, float(cfg.dt))
    return np.array(root.frames.position.value[-1], dtype=float)
