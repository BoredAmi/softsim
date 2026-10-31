"""Inflation: a soft body with a cavity, driven by pressure actuators.

The first actuation scenario (M6): a hollow sphere or a closed tube (balloon
actuator), optionally bonded to the ground through named regions (``fixed``),
with one or more pressure actuators on its regions following a schedule or a
controller (``Backend.set_control``).

Outputs per sample: each actuator's pressure ``<name>_pressure`` [Pa] and the
volume its region encloses ``<name>_volume`` [m^3] (from the backend's surface
vertices), and the body's extent (``max_z``, ``min_z``, ``max_radius`` about the
body's axis). Shapes with a ``top`` region also give the tip: the top face's
area-weighted centroid (``tip_x``, ``tip_y``, ``tip_z``) and its tilt from +z
(``bend_angle`` [rad], from the area-weighted normal), i.e. how far a bending
actuator curls. The summary gives the final and peak values and the volume
change of each cavity.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from softsim.actuation import PressureActuator, PressureActuatorConfig, enclosed_volume
from softsim.backends.base import SimState
from softsim.config.schema import StrictModel
from softsim.config.units import Length, Mass
from softsim.errors import ConfigError
from softsim.geometry import MeshSettings, SoftShape, check_resolution
from softsim.scenarios.base import Scenario, ScenarioConfig
from softsim.scene import Pose, Scene, SoftBodySpec


class InflatedBodyConfig(StrictModel):
    name: str = "body"
    geometry: SoftShape
    material: str | dict[str, Any]
    mass: Annotated[Mass, Field(gt=0)] | None = None
    mesh: MeshSettings = MeshSettings(method="gmsh")
    position: tuple[Length, Length, Length] = (0.0, 0.0, 0.0)
    fixed: list[str] = []  # regions whose vertices are pinned (e.g. ["base"])


class InflationConfig(ScenarioConfig):
    scenario: Literal["inflation"] = "inflation"
    body: InflatedBodyConfig
    actuators: list[PressureActuatorConfig] = [PressureActuatorConfig()]


class InflationScenario(Scenario):
    """Pressurised soft body; see module docstring."""

    name: ClassVar[str] = "inflation"
    config_model: ClassVar[type[ScenarioConfig]] = InflationConfig

    config: InflationConfig

    def build_scene(self) -> Scene:
        cfg = self.config.body
        check_resolution(cfg.geometry, cfg.mesh)
        mesh = cfg.geometry.mesh(cfg.mesh)
        regions = cfg.geometry.regions(mesh)
        for name in cfg.fixed:
            if name not in regions:
                raise ConfigError(f"body.fixed: no region {name!r}; {type(cfg.geometry).__name__} has {sorted(regions)}")
        pinned = sorted({int(i) for name in cfg.fixed for i in np.unique(regions[name])})
        body = SoftBodySpec(name=cfg.name, mesh=mesh, material=self.load_material(cfg.material),
                            pose=Pose(tuple(cfg.position)), mass=cfg.mass, pinned=tuple(pinned), regions=regions)
        actuators = tuple(PressureActuator.from_config(a, cfg.name) for a in self.config.actuators)
        for a in actuators:
            if a.region not in regions:
                raise ConfigError(f"actuator {a.name!r}: no region {a.region!r}; have {sorted(regions)}")
        self._regions = regions
        self._axis = np.asarray(cfg.position[:2], float)
        return Scene(soft_bodies=(body,), simulation=self.config.simulation, contact=self.config.contact,
                     ground=self.config.ground, metadata={"name": self.config.name or "inflation"},
                     actuators=actuators)

    def metrics(self, state: SimState) -> dict[str, float]:
        cfg = self.config
        x = state.soft[cfg.body.name].vertices
        out = {"t": state.time}
        for a in cfg.actuators:
            out[f"{a.name}_pressure"] = float(state.controls.get(a.name, np.nan))
            out[f"{a.name}_volume"] = enclosed_volume(x, self._regions[a.region])
        if "top" in self._regions:
            tri = x[self._regions["top"]]
            area_n = 0.5 * np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
            area = np.linalg.norm(area_n, axis=1)
            tip = (tri.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum()
            n = area_n.sum(axis=0)
            out.update(tip_x=float(tip[0]), tip_y=float(tip[1]), tip_z=float(tip[2]),
                       bend_angle=float(np.arctan2(np.hypot(n[0], n[1]), n[2])))
        out["max_z"] = float(x[:, 2].max())
        out["min_z"] = float(x[:, 2].min())
        out["max_radius"] = float(np.hypot(*(x[:, :2] - self._axis).T).max())
        return out

    def summarize(self, ts: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for a in self.config.actuators:
            v = ts[f"{a.name}_volume"].to_numpy()
            out[f"{a.name}_final_pressure"] = float(ts[f"{a.name}_pressure"].iloc[-1])
            out[f"{a.name}_final_volume"] = float(v[-1])
            out[f"{a.name}_volume_ratio"] = float(v[-1] / v[0])
        if "bend_angle" in ts:
            out["final_bend_angle"] = float(ts["bend_angle"].iloc[-1])
            out["final_tip_x"] = float(ts["tip_x"].iloc[-1])
        out["final_max_z"] = float(ts["max_z"].iloc[-1])
        out["peak_max_radius"] = float(ts["max_radius"].max())
        return out

    def material_key(self, body: str) -> str:
        if body != self.config.body.name:
            return super().material_key(body)
        return "body.material"
