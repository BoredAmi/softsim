"""Actuation (``scenario: actuation``, or ``inflation``): one soft body driven by actuators.

The actuation scenario (M6): a soft body, e.g. a hollow sphere or a closed tube
(balloon actuator) or a beam, optionally bonded to the ground through named
regions (``fixed``), with pressure actuators on its regions and cables routed
through it (``type: cable``), each following a schedule or a controller
(``Backend.set_control``). An actuator without ``type`` is a pressure actuator.
``body.zones`` give parts of the body other materials (e.g. a strain-limiting layer).

Outputs per sample: each pressure actuator's pressure ``<name>_pressure`` [Pa]
and the volume its region encloses ``<name>_volume`` [m^3] (from the backend's
surface vertices); each cable's tension ``<name>_tension`` [N] and length
through its via points ``<name>_length`` [m]; and the body's extent (``max_z``,
``min_z``, ``max_radius`` about the body's axis). Shapes with a ``tip`` (else
``top``) region also give the tip: that face's area-weighted centroid (``tip_x``,
``tip_y``, ``tip_z``) and its rotation from its rest orientation (``bend_angle``
[rad], from the area-weighted normal), i.e. how far a bending actuator curls. The summary gives the final and peak values, the volume
change of each cavity and the stroke (shortening) of each cable.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Union

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from softsim.actuation import (CableActuator, CableActuatorConfig, PressureActuator, PressureActuatorConfig,
                               cable_length, embed, enclosed_volume)
from softsim.backends.base import SimState
from softsim.config.schema import StrictModel
from softsim.config.units import Length, Mass
from softsim.errors import ConfigError
from softsim.geometry import MeshSettings, SoftShape, check_resolution
from softsim.scenarios.base import Scenario, ScenarioConfig
from softsim.scene import MaterialZone, Pose, Scene, SoftBodySpec


class MaterialZoneConfig(StrictModel):
    """Elements of the body made of another material, e.g. a strain-limiting layer: a volume region of
    the shape (``region: layer``, exact where the mesh conforms to it) or those whose centroid lies in
    ``box`` (``[[x0, y0, z0], [x1, y1, z1]]``, body frame)."""

    name: str
    material: str | dict[str, Any]
    region: str | None = None
    box: tuple[tuple[Length, Length, Length], tuple[Length, Length, Length]] | None = None

    @model_validator(mode="after")
    def _one_selector(self) -> "MaterialZoneConfig":
        if (self.region is None) == (self.box is None):
            raise ValueError(f"zone {self.name!r}: give exactly one of region and box")
        return self


class InflatedBodyConfig(StrictModel):
    name: str = "body"
    geometry: SoftShape
    material: str | dict[str, Any]
    mass: Annotated[Mass, Field(gt=0)] | None = None
    mesh: MeshSettings = MeshSettings(method="gmsh")
    position: tuple[Length, Length, Length] = (0.0, 0.0, 0.0)
    fixed: list[str] = []  # regions whose vertices are pinned (e.g. ["base"])
    zones: list[MaterialZoneConfig] = []  # multi-material bodies


class InflationConfig(ScenarioConfig):
    scenario: Literal["inflation", "actuation"] = "inflation"
    body: InflatedBodyConfig
    actuators: list[Annotated[Union[PressureActuatorConfig, CableActuatorConfig], Field(discriminator="type")]] = [
        PressureActuatorConfig()]

    @model_validator(mode="before")
    @classmethod
    def _pressure_by_default(cls, data: Any) -> Any:
        if isinstance(data, dict) and isinstance(data.get("actuators"), list):
            data = {**data, "actuators": [{"type": "pressure", **a} if isinstance(a, dict) and "type" not in a else a
                                          for a in data["actuators"]]}
        return data


def _face_centre_normal(x: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Area-weighted centroid and unit mean normal of the triangles ``tris`` at positions ``x``."""
    tri = x[tris]
    area_n = 0.5 * np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(area_n, axis=1)
    n = area_n.sum(axis=0)
    return (tri.mean(axis=1) * area[:, None]).sum(axis=0) / area.sum(), n / np.linalg.norm(n)


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
        centroids = mesh.points[mesh.tets].mean(axis=1)
        zones = []
        volumes = cfg.geometry.volume_regions(mesh) if hasattr(cfg.geometry, "volume_regions") else {}
        for z in cfg.zones:
            if z.region is not None:
                if z.region not in volumes:
                    raise ConfigError(f"body.zones: {type(cfg.geometry).__name__} has no volume region "
                                      f"{z.region!r}; have {sorted(volumes)}")
                elements = volumes[z.region]
            else:
                lo, hi = np.array(z.box[0]), np.array(z.box[1])
                elements = np.nonzero(np.all((centroids >= lo) & (centroids <= hi), axis=1))[0]
            zones.append(MaterialZone(z.name, elements, self.load_material(z.material)))
        try:
            body = SoftBodySpec(name=cfg.name, mesh=mesh, material=self.load_material(cfg.material),
                                pose=Pose(tuple(cfg.position)), mass=cfg.mass, pinned=tuple(pinned), regions=regions,
                                zones=tuple(zones))
        except ValueError as exc:
            raise ConfigError(f"body.zones: {exc}") from None
        actuators = tuple(PressureActuator.from_config(a, cfg.name) if isinstance(a, PressureActuatorConfig)
                          else CableActuator.from_config(a, cfg.name, cfg.mesh.size) for a in self.config.actuators)
        self._cables = {}
        for a in actuators:
            if isinstance(a, PressureActuator) and a.region not in regions:
                raise ConfigError(f"actuator {a.name!r}: no region {a.region!r}; have {sorted(regions)}")
            if isinstance(a, CableActuator):
                try:
                    elems, bary = embed(mesh.points, mesh.tets, np.array(a.via))
                except ValueError as exc:
                    raise ConfigError(f"actuator {a.name!r}: {exc}") from None
                self._cables[a.name] = (mesh.tets[elems], bary)
        self._regions = regions
        self._tip = next((r for r in ("tip", "top") if r in regions), None)
        if self._tip is not None:
            self._tip_normal = _face_centre_normal(mesh.points, regions[self._tip])[1]
        self._axis = np.asarray(cfg.position[:2], float)
        return Scene(soft_bodies=(body,), simulation=self.config.simulation, contact=self.config.contact,
                     ground=self.config.ground, metadata={"name": self.config.name or self.config.scenario},
                     actuators=actuators)

    def metrics(self, state: SimState) -> dict[str, float]:
        cfg = self.config
        x = state.soft[cfg.body.name].vertices
        out = {"t": state.time}
        for a in cfg.actuators:
            if isinstance(a, CableActuatorConfig):
                out[f"{a.name}_tension"] = float(state.controls.get(a.name, np.nan))
                out[f"{a.name}_length"] = cable_length(x, *self._cables[a.name])
                continue
            out[f"{a.name}_pressure"] = float(state.controls.get(a.name, np.nan))
            out[f"{a.name}_volume"] = enclosed_volume(x, self._regions[a.region])
        if self._tip is not None:
            tip, n = _face_centre_normal(x, self._regions[self._tip])
            out.update(tip_x=float(tip[0]), tip_y=float(tip[1]), tip_z=float(tip[2]),
                       bend_angle=float(np.arctan2(np.linalg.norm(np.cross(n, self._tip_normal)), n @ self._tip_normal)))
        out["max_z"] = float(x[:, 2].max())
        out["min_z"] = float(x[:, 2].min())
        out["max_radius"] = float(np.hypot(*(x[:, :2] - self._axis).T).max())
        return out

    def summarize(self, ts: pd.DataFrame) -> dict[str, float]:
        out: dict[str, float] = {}
        for a in self.config.actuators:
            if isinstance(a, CableActuatorConfig):
                length = ts[f"{a.name}_length"].to_numpy()
                out[f"{a.name}_final_tension"] = float(ts[f"{a.name}_tension"].iloc[-1])
                out[f"{a.name}_stroke"] = float(length[0] - length[-1])
                continue
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
