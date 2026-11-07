"""Grasp: a soft gripper closes on a rigid object on a table, lifts it and holds it (M9).

The body, its actuators and the moved regions are those of the actuation scenario (the palm is moved through
``body.prescribed``); this scenario adds the object, free on the table.

Outputs per sample, on top of the actuation scenario's: ``object_x/y/z`` [m], ``object_lift`` [m] (rise of the
object) and, with ``base_right`` and ``base_left`` prescribed, ``grip_force`` [N], the mean squeeze of the two
fingers on the object (at equilibrium a finger's x reaction at the palm is the x force the object exerts on it;
a net contact force cannot say this: one body, the two fingers cancel). ``palm_lift`` [m] is the palm's rise.

Summary: ``held`` (1 if the object rose by at least ``held_fraction`` of the palm's rise and stayed there),
``slip`` [m] (palm rise minus object rise at the end), ``peak_grip_force`` [N].
"""

from __future__ import annotations

from dataclasses import replace
from typing import Annotated, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from softsim.backends.base import SimState
from softsim.config.schema import StrictModel
from softsim.config.units import Length, Mass
from softsim.geometry.rigid import Part
from softsim.materials.models import Friction
from softsim.scenarios.base import ScenarioConfig
from softsim.scenarios.inflation import InflationConfig, InflationScenario
from softsim.scene import Pose, RigidBodySpec, Scene

PositiveLength = Annotated[Length, Field(gt=0)]


class GraspObjectConfig(StrictModel):
    """The object: a ``box`` (``size`` = half extents) or a ``cylinder`` with a vertical axis (``size`` = radius,
    half height), centred above the origin, starting ``clearance`` above the table it stands on."""

    name: str = "object"
    shape: Literal["box", "cylinder"] = "box"
    size: tuple[PositiveLength, PositiveLength, PositiveLength] = (0.016, 0.012, 0.03)
    mass: Annotated[Mass, Field(gt=0)] = 0.05
    clearance: Annotated[Length, Field(ge=0)] = 0.0002
    friction: Friction = Friction(sliding=0.8)


class GraspConfig(InflationConfig):
    scenario: Literal["grasp"] = "grasp"
    object: GraspObjectConfig = GraspObjectConfig()
    held_fraction: Annotated[float, Field(gt=0, le=1)] = 0.8


class GraspScenario(InflationScenario):
    """A gripper, an object, a table; see module docstring."""

    name: ClassVar[str] = "grasp"
    config_model: ClassVar[type[ScenarioConfig]] = GraspConfig

    config: GraspConfig

    def build_scene(self) -> Scene:
        scene = super().build_scene()
        o = self.config.object
        half_height = o.size[2] if o.shape == "box" else o.size[1]
        size = o.size if o.shape == "box" else (o.size[0], o.size[1], 0.0)
        self._z0 = half_height + o.clearance
        rigid = RigidBodySpec(name=o.name, parts=(Part(o.shape, tuple(size), (0.0, 0.0, 0.0)),), mass=o.mass,
                              pose=Pose((0.0, 0.0, self._z0)), friction=o.friction, floor=half_height)
        return replace(scene, rigid_bodies=(rigid,))

    def metrics(self, state: SimState) -> dict[str, float]:
        out = super().metrics(state)
        p = state.rigid[self.config.object.name].position
        out.update(object_x=float(p[0]), object_y=float(p[1]), object_z=float(p[2]), object_lift=float(p[2] - self._z0))
        if "base_right_reaction_x" in out and "base_left_reaction_x" in out:
            out["grip_force"] = 0.5 * (abs(out["base_right_reaction_x"]) + abs(out["base_left_reaction_x"]))
        rises = [out[f"{r}_displacement_z"] for r in ("base", "base_right", "base_left") if f"{r}_displacement_z" in out]
        out["palm_lift"] = float(np.mean(rises)) if rises else 0.0
        return out

    def summarize(self, ts: pd.DataFrame) -> dict[str, float]:
        out = super().summarize(ts)
        palm, lift = float(ts["palm_lift"].iloc[-1]), float(ts["object_lift"].iloc[-1])
        out["final_palm_lift"], out["final_object_lift"], out["slip"] = palm, lift, palm - lift
        out["held"] = float(palm > 0 and lift >= self.config.held_fraction * palm)
        if "grip_force" in ts:
            out["peak_grip_force"] = float(ts["grip_force"].max())
        return out
