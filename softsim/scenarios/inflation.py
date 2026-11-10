"""Actuation (``scenario: actuation``, or ``inflation``): one soft body driven by actuators.

The actuation scenario (M6): a soft body, e.g. a hollow sphere or a closed tube
(balloon actuator) or a beam, optionally bonded to the ground through named
regions (``fixed``), with pressure actuators on its regions and cables routed
through it (``type: cable``), each following a schedule or a controller
(``Backend.set_control``). An actuator without ``type`` is a pressure actuator.
``body.zones`` give parts of the body other materials (e.g. a strain-limiting layer);
``body.mount_path`` moves the fixed regions along a prescribed translation (a moving mount,
softsim.scene.MountMotion); ``body.prescribed`` moves single regions along their own paths,
per axis (displacement control, softsim.scene.PrescribedMotion), and reports their reactions.

Outputs per sample: each pressure actuator's pressure ``<name>_pressure`` [Pa]
and the volume its region encloses ``<name>_volume`` [m^3] (from the backend's
surface vertices); each cable's tension ``<name>_tension`` [N] and length
through its via points ``<name>_length`` [m] (a stroke-controlled cable also gives its
command ``<name>_command`` [m]; its tension follows from its length); each region force its
force ``<name>_force`` [N] and its region's displacement along it ``<name>_displacement`` [m]
(area-weighted centroid); each prescribed region's reaction ``<region>_reaction_<axis>`` [N] and
centroid motion ``<region>_displacement_<axis>`` [m] along its held axes; and the body's extent (``max_z``,
``min_z``, ``max_radius`` about the body's axis). Shapes with a ``tip`` (else
``top``) region also give the tip: that face's area-weighted centroid (``tip_x``,
``tip_y``, ``tip_z``) and its rotation from its rest orientation (``bend_angle``
[rad], from the area-weighted normal), i.e. how far a bending actuator curls, and, with a ``base`` region,
the angle between the root-to-tip line and its rest direction (``chord_angle`` [rad], what a camera
measures on a finger photographed from the side). The summary gives the final and peak values, the volume
change of each cavity and the stroke (shortening) of each cable.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal, Union

import numpy as np
import pandas as pd
from pydantic import Field, model_validator

from softsim.actuation import (CableActuator, CableActuatorConfig, ForceActuator, ForceActuatorConfig,
                               PressureActuator, PressureActuatorConfig,
                               cable_length, cable_rows, enclosed_volume)
from softsim.backends.base import SimState
from softsim.config.schema import StrictModel
from softsim.config.units import Angle, Length, Mass, Time
from softsim.errors import ConfigError
from softsim.geometry import MeshSettings, SoftShape, check_resolution
from softsim.scenarios.base import Scenario, ScenarioConfig
from softsim.scene import MaterialZone, MountMotion, MountSpin, Pose, PrescribedMotion, Scene, SoftBodySpec


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


class PrescribedRegionConfig(StrictModel):
    """Displacement control of a region (softsim.scene.PrescribedMotion), e.g. the grip of a tension
    test: ``path: [[time, [dx, dy, dz]], ...]`` from rest (the first waypoint is zero), minimum jerk
    between waypoints, held after the last. An axis that is ``null`` in every waypoint is free (a
    roller: ``[null, null, 0]`` holds a face in z only). Outputs ``<region>_reaction_<axis>`` [N],
    the force the support exerts on the body, and ``<region>_displacement_<axis>`` [m], the region's
    area-weighted centroid motion, for each held axis."""

    region: str
    path: list[tuple[Annotated[Time, Field(ge=0)], tuple[Length | None, Length | None, Length | None]]]

    @model_validator(mode="after")
    def _same_free_axes(self) -> "PrescribedRegionConfig":
        held = {tuple(v is not None for v in p) for _, p in self.path}
        if len(held) > 1:
            raise ValueError(f"prescribed region {self.region!r}: an axis is null in some waypoints and not "
                             "in others; an axis is free (null) or held for the whole path")
        return self

    @property
    def axes(self) -> tuple[bool, bool, bool]:
        return tuple(v is not None for v in self.path[0][1])


class MountRotationConfig(StrictModel):
    """The fixed regions turn about ``axis`` through ``center`` (softsim.scene.MountSpin): ``path`` is
    ``[[time, angle], ...]`` from rest (the first angle is zero), minimum jerk between waypoints, held
    after the last. Composes with ``mount_path``. Region forces and prescribed regions are refused (their
    directions would turn with the mount's frame). Outputs ``mount_angle`` [rad]."""

    axis: tuple[float, float, float]
    center: tuple[Length, Length, Length] = (0.0, 0.0, 0.0)
    path: list[tuple[Annotated[Time, Field(ge=0)], Angle]]


class InflatedBodyConfig(StrictModel):
    name: str = "body"
    geometry: SoftShape
    material: str | dict[str, Any]
    mass: Annotated[Mass, Field(gt=0)] | None = None
    mesh: MeshSettings = MeshSettings(method="gmsh")
    position: tuple[Length, Length, Length] = (0.0, 0.0, 0.0)
    fixed: list[str] = []  # regions whose vertices are pinned (e.g. ["base"])
    plane_strain: Literal["x", "y", "z"] | None = None  # body axis no vertex moves along (a wide strip, U3 = 0)
    zones: list[MaterialZoneConfig] = []  # multi-material bodies
    # moving mount: the fixed regions follow this translation, [[time, [dx, dy, dz]], ...] from rest
    mount_path: list[tuple[Annotated[Time, Field(ge=0)], tuple[Length, Length, Length]]] = []
    prescribed: list[PrescribedRegionConfig] = []  # regions moved along paths (displacement control)
    mount_rotation: MountRotationConfig | None = None  # the fixed regions turn about an axis


class InflationConfig(ScenarioConfig):
    scenario: Literal["inflation", "actuation"] = "inflation"
    body: InflatedBodyConfig
    actuators: list[Annotated[Union[PressureActuatorConfig, CableActuatorConfig, ForceActuatorConfig],
                              Field(discriminator="type")]] = [
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
        prescribed = []
        for p in cfg.prescribed:
            if p.region not in regions:
                raise ConfigError(f"body.prescribed: no region {p.region!r}; {type(cfg.geometry).__name__} has "
                                  f"{sorted(regions)}")
            try:
                path = MountMotion(tuple(t for t, _ in p.path),
                                   tuple(tuple(0.0 if v is None else v for v in x) for _, x in p.path))
                prescribed.append(PrescribedMotion(p.region, np.unique(regions[p.region]), p.axes, path))
            except ValueError as exc:
                raise ConfigError(f"body.prescribed ({p.region}): {exc}") from None
        try:
            body = SoftBodySpec(name=cfg.name, mesh=mesh, material=self.load_material(cfg.material),
                                pose=Pose(tuple(cfg.position)), mass=cfg.mass, pinned=tuple(pinned), regions=regions,
                                zones=tuple(zones), prescribed=tuple(prescribed),
                                plane_strain=None if cfg.plane_strain is None else "xyz".index(cfg.plane_strain))
        except ValueError as exc:
            raise ConfigError(f"body: {exc}") from None
        configs = []
        for a in self.config.actuators:  # ``path: geometry``: the shape's own tendon path
            if isinstance(a, CableActuatorConfig) and a.path == "geometry" and hasattr(cfg.geometry, "tendon_path"):
                a = a.model_copy(update={"path": cfg.geometry.tendon_path()})
            configs.append(a)
        try:
            actuators = tuple(PressureActuator.from_config(a, cfg.name) if isinstance(a, PressureActuatorConfig)
                              else ForceActuator.from_config(a, cfg.name) if isinstance(a, ForceActuatorConfig)
                              else CableActuator.from_config(a, cfg.name, cfg.mesh.size) for a in configs)
        except ValueError as exc:
            raise ConfigError(str(exc)) from None
        self._cables, self._loaded = {}, {}
        for a in actuators:
            if isinstance(a, (PressureActuator, ForceActuator)) and a.region not in regions:
                raise ConfigError(f"actuator {a.name!r}: no region {a.region!r}; have {sorted(regions)}")
            if isinstance(a, ForceActuator):
                self._loaded[a.name] = (_face_centre_normal(mesh.points, regions[a.region])[0], np.array(a.direction))
            if isinstance(a, CableActuator):
                if a.anchor and a.anchor not in regions:
                    raise ConfigError(f"actuator {a.name!r}: no anchor region {a.anchor!r}; have {sorted(regions)}")
                try:
                    nodes, bary = cable_rows(mesh.points, mesh.tets, np.array(a.via),
                                             regions[a.anchor] if a.anchor else None)
                except ValueError as exc:
                    raise ConfigError(f"actuator {a.name!r}: {exc}") from None
                self._cables[a.name] = (nodes, bary, a, cable_length(mesh.points, nodes, bary))
        self._regions = regions
        self._prescribed = {p.region: (_face_centre_normal(mesh.points, regions[p.region])[0], p.axes)
                            for p in prescribed}
        self._tip = next((r for r in ("tip", "top") if r in regions), None)
        if self._tip is not None:
            self._tip_normal = _face_centre_normal(mesh.points, regions[self._tip])[1]
            self._chord0 = (_face_centre_normal(mesh.points, regions[self._tip])[0]
                            - _face_centre_normal(mesh.points, regions["base"])[0]) if "base" in regions else None
        self._axis = np.asarray(cfg.position[:2], float)
        self._spin = None
        rotation = cfg.mount_rotation
        try:
            mount = MountMotion(tuple(t for t, _ in cfg.mount_path),
                                tuple(tuple(p) for _, p in cfg.mount_path)) if cfg.mount_path else None
            axis = np.asarray(rotation.axis, float) if rotation else None
            if rotation and not np.linalg.norm(axis) > 0:
                raise ValueError("mount_rotation: the axis must be non-zero")
            spin = (MountSpin(tuple(axis / np.linalg.norm(axis)), tuple(rotation.center),
                              tuple(t for t, _ in rotation.path), tuple(a for _, a in rotation.path))
                    if rotation else None)
            self._spin = spin
            return Scene(soft_bodies=(body,), simulation=self.config.simulation, contact=self.config.contact,
                         ground=self.config.ground, metadata={"name": self.config.name or self.config.scenario},
                         actuators=actuators, mount=mount, spin=spin, magnetic_field=self.magnetic_field())
        except ValueError as exc:
            if not (cfg.mount_path or rotation):
                raise
            raise ConfigError(f"body.mount_path / mount_rotation: {exc}") from None

    def metrics(self, state: SimState) -> dict[str, float]:
        cfg = self.config
        x = state.soft[cfg.body.name].vertices
        out = {"t": state.time}
        for a in cfg.actuators:
            if isinstance(a, CableActuatorConfig):
                nodes, bary, cable, rest = self._cables[a.name]
                length = cable_length(x, nodes, bary)
                value = float(state.controls.get(a.name, np.nan))
                if cable.stroke_controlled:
                    out[f"{a.name}_command"] = value
                out[f"{a.name}_tension"] = cable.tension_at(length, value, rest)
                out[f"{a.name}_length"] = length
                continue
            if isinstance(a, ForceActuatorConfig):
                centre0, direction = self._loaded[a.name]
                out[f"{a.name}_force"] = float(state.controls.get(a.name, np.nan))
                out[f"{a.name}_displacement"] = float(
                    (_face_centre_normal(x, self._regions[a.region])[0] - centre0) @ direction)
                continue
            out[f"{a.name}_pressure"] = float(state.controls.get(a.name, np.nan))
            out[f"{a.name}_volume"] = enclosed_volume(x, self._regions[a.region])
        reactions = state.soft[cfg.body.name].reactions if self._prescribed else {}
        for region, (centre0, axes) in self._prescribed.items():
            moved = _face_centre_normal(x, self._regions[region])[0] - centre0
            force = reactions.get(region, np.full(3, np.nan))
            for k, axis in enumerate("xyz"):
                if axes[k]:
                    out[f"{region}_reaction_{axis}"] = float(force[k])
                    out[f"{region}_displacement_{axis}"] = float(moved[k])
        if self._spin is not None:
            out["mount_angle"] = self._spin.angle(state.time)
        if self._tip is not None:
            tip, n = _face_centre_normal(x, self._regions[self._tip])
            out.update(tip_x=float(tip[0]), tip_y=float(tip[1]), tip_z=float(tip[2]),
                       bend_angle=float(np.arctan2(np.linalg.norm(np.cross(n, self._tip_normal)), n @ self._tip_normal)))
            if self._chord0 is not None:  # the angle a camera sees: the root-to-tip line against its rest direction
                chord = tip - _face_centre_normal(x, self._regions["base"])[0]
                out["chord_angle"] = float(np.arctan2(np.linalg.norm(np.cross(self._chord0, chord)), self._chord0 @ chord))
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
            if isinstance(a, ForceActuatorConfig):
                out[f"{a.name}_final_force"] = float(ts[f"{a.name}_force"].iloc[-1])
                out[f"{a.name}_final_displacement"] = float(ts[f"{a.name}_displacement"].iloc[-1])
                continue
            v = ts[f"{a.name}_volume"].to_numpy()
            out[f"{a.name}_final_pressure"] = float(ts[f"{a.name}_pressure"].iloc[-1])
            out[f"{a.name}_final_volume"] = float(v[-1])
            out[f"{a.name}_volume_ratio"] = float(v[-1] / v[0])
        for region, (_, axes) in self._prescribed.items():
            for k, axis in enumerate("xyz"):
                if axes[k]:
                    out[f"{region}_final_reaction_{axis}"] = float(ts[f"{region}_reaction_{axis}"].iloc[-1])
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
