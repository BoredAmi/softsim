"""Drop test: a rigid weight falls onto a soft specimen resting on a table.

Reference experiment: Czopek, Superczynska, Marcinkowska, Gajewski, Bernat,
"Records of falling mass on Ecoflex 00-10 cylinder", Zenodo,
doi:10.5281/zenodo.23185808 (CC-BY-4.0).

Simulation starts with the weight's lowest point ``drop.gap`` above the
specimen, moving at the speed it would have after free fall from
``drop.height`` (air drag neglected), so the long free fall is not simulated.

Sign conventions: ``indentation`` = height of the specimen's undeformed contact
surface (top face plus the backend's collision skin, ``contact.soft_radius``)
minus the height of the weight's bottom centre, positive = compressing.
``compression`` is the downward displacement of the specimen's top-centre
vertex (no skin). Forces are z components in newtons, positive up.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, ClassVar, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from softsim.backends.base import SimState
from softsim.config.schema import StrictModel
from softsim.config.units import Angle, Length, Mass, Velocity
from softsim.errors import ConfigError
from softsim.geometry import MeshSettings, SoftShape, check_resolution, load_rigid_body
from softsim.materials.models import Friction
from softsim.scenarios.base import Scenario, ScenarioConfig
from softsim.scene import Pose, RigidBodySpec, Scene, SoftBodySpec

GRAVITY_FALLBACK = 9.81


class SpecimenConfig(StrictModel):
    name: str = "specimen"
    geometry: SoftShape
    material: str | dict[str, Any]
    mass: Annotated[Mass, Field(gt=0)] | None = None
    mesh: MeshSettings = MeshSettings()
    base: Literal["free", "bonded"] = "free"
    self_collision: bool = False


class WeightConfig(StrictModel):
    name: str = "weight"
    body: str | dict[str, Any]
    friction: Friction = Friction(sliding=0.5)


class DropConfig(StrictModel):
    height: Annotated[Length, Field(ge=0)] = 0.38
    initial_speed: Annotated[Velocity, Field(ge=0)] | None = None
    gap: Annotated[Length, Field(gt=0)] = 0.0005
    offset: tuple[Length, Length] = (0.0, 0.0)
    tilt: tuple[Angle, Angle] = (0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


class DropTestConfig(ScenarioConfig):
    scenario: Literal["drop_test"] = "drop_test"
    specimen: SpecimenConfig
    weight: WeightConfig
    drop: DropConfig = DropConfig()


class DropTestScenario(Scenario):
    """Weight dropped on a soft specimen; see module docstring."""

    name: ClassVar[str] = "drop_test"
    config_model: ClassVar[type[ScenarioConfig]] = DropTestConfig
    actions: ClassVar[dict[str, str]] = {
        "drop": "re-drop with optional preset (str), mass [kg], height [m], offset [m, m], tilt [rad, rad]",
    }

    config: DropTestConfig

    # -- scene -------------------------------------------------------------

    @property
    def gravity(self) -> float:
        g = -self.config.simulation.gravity[2]
        return g if g > 0 else GRAVITY_FALLBACK

    def impact_speed(self) -> float:
        drop = self.config.drop
        if drop.initial_speed is not None:
            return math.sqrt(drop.initial_speed**2 + 2 * self.gravity * drop.gap)
        return math.sqrt(2 * self.gravity * drop.height)

    def release_speed(self) -> float:
        drop = self.config.drop
        if drop.initial_speed is not None:
            return drop.initial_speed
        if drop.height < drop.gap:
            raise ConfigError(f"drop.height {drop.height} m is smaller than drop.gap {drop.gap} m")
        return math.sqrt(2 * self.gravity * (drop.height - drop.gap))

    def build_scene(self) -> Scene:
        cfg = self.config
        spec_cfg = cfg.specimen
        check_resolution(spec_cfg.geometry, spec_cfg.mesh)
        mesh = spec_cfg.geometry.mesh(spec_cfg.mesh)
        pinned = tuple(mesh.vertices_where(lambda p: p[:, 2] < 1e-9).tolist()) if spec_cfg.base == "bonded" else ()
        specimen = SoftBodySpec(
            name=spec_cfg.name,
            mesh=mesh,
            material=self.load_material(spec_cfg.material),
            mass=spec_cfg.mass,
            pinned=pinned,
            self_collision=spec_cfg.self_collision,
        )
        self._top = spec_cfg.geometry.top_height
        self._top_center = int(np.argmin(np.hypot(mesh.points[:, 0], mesh.points[:, 1])
                                         + 1e3 * (self._top - mesh.points[:, 2])))

        body = load_rigid_body(cfg.weight.body)
        self._weight_def = body
        drop = cfg.drop
        pose = Pose.from_tilt((0.0, 0.0, 0.0), drop.tilt[0], drop.tilt[1])
        rot = _quat_to_mat(pose.quaternion)
        drop_below_origin = body.shape.bottom_radius * math.hypot(rot[2, 0], rot[2, 1])
        z0 = self._top + drop.gap + drop_below_origin
        pose = Pose((drop.offset[0], drop.offset[1], z0), pose.quaternion)
        weight = RigidBodySpec(
            name=cfg.weight.name,
            parts=tuple(body.shape.parts()),
            mass=body.mass,
            pose=pose,
            linear_velocity=(0.0, 0.0, -self.release_speed()),
            angular_velocity=drop.angular_velocity,
            friction=cfg.weight.friction,
        )
        return Scene(
            soft_bodies=(specimen,),
            rigid_bodies=(weight,),
            simulation=cfg.simulation,
            contact=cfg.contact,
            ground=cfg.ground,
            metadata={"name": cfg.name or "drop_test", "specimen_top": self._top},
        )

    # -- metrics -----------------------------------------------------------

    def metrics(self, state: SimState) -> dict[str, float]:
        cfg = self.config
        w = state.rigid[cfg.weight.name]
        verts = state.soft[cfg.specimen.name].vertices
        center_xy = verts[:, :2].mean(axis=0)
        radial = np.hypot(verts[:, 0] - center_xy[0], verts[:, 1] - center_xy[1])
        contact = state.contact_between(cfg.specimen.name, cfg.weight.name)
        ground = state.contact_between("ground", cfg.specimen.name)
        return {
            "t": state.time,
            "weight_z": float(w.position[2]),
            "weight_vz": float(w.linear_velocity[2]),
            "weight_com_x": float(w.com[0]) if w.com is not None else float("nan"),
            "weight_com_z": float(w.com[2]) if w.com is not None else float("nan"),
            "indentation": float(self._top + cfg.contact.soft_radius - w.position[2]),
            "top_center_z": float(verts[self._top_center, 2]),
            "compression": float(self._top - verts[self._top_center, 2]),
            "specimen_max_z": float(verts[:, 2].max()),
            "specimen_min_z": float(verts[:, 2].min()),
            "bulge_diameter": float(2 * radial.max()),
            "contact_force_z": float(contact.force[2]) if contact else 0.0,
            "ground_force_z": float(ground.force[2]) if ground else 0.0,
            "weight_tilt": float(_tilt_angle(w.quaternion)),
        }

    def summarize(self, ts: pd.DataFrame) -> dict[str, float]:
        """Impact summary. Contact = weight/specimen force above 1% of the weight's gravity load."""
        mg = self._weight_def.mass * self.gravity
        in_contact = ts["contact_force_z"].to_numpy() > 0.01 * mg
        out: dict[str, float] = {"impact_speed": self.impact_speed()}
        if not in_contact.any():
            out["contact"] = 0.0
            return out
        i0 = int(np.argmax(in_contact))
        after = np.nonzero(~in_contact[i0:])[0]
        i1 = i0 + int(after[0]) if len(after) else len(ts) - 1
        first = ts.iloc[i0:i1 + 1]
        k = int(first["indentation"].to_numpy().argmax())
        n_tail = max(1, len(ts) // 10)
        final = float(ts["indentation"].iloc[-n_tail:].mean())
        peak = float(first["indentation"].iloc[k])
        out.update({
            "contact": 1.0,
            "t_contact": float(ts["t"].iloc[i0]),
            "peak_indentation": peak,
            "t_to_peak": float(first["t"].iloc[k] - ts["t"].iloc[i0]),
            "peak_contact_force": float(first["contact_force_z"].max()),
            "peak_ground_force": float(ts["ground_force_z"].max()),
            "max_bulge_diameter": float(ts["bulge_diameter"].max()),
            "first_contact_duration": float(ts["t"].iloc[i1] - ts["t"].iloc[i0]) if len(after) else float("nan"),
            "rebound_speed": float(ts["weight_vz"].iloc[i1]) if len(after) else float("nan"),
            "final_indentation": final,
            "specimen_lift_off": float(ts["specimen_min_z"].max()),
        })
        out["restitution"] = out["rebound_speed"] / out["impact_speed"]
        settled = np.abs(ts["indentation"].to_numpy() - final) <= 0.05 * max(peak, 1e-9)
        unsettled = np.nonzero(~settled)[0]
        out["settling_time"] = (float(ts["t"].iloc[unsettled[-1] + 1] - ts["t"].iloc[i0])
                                if len(unsettled) and unsettled[-1] + 1 < len(ts) else float("nan"))
        return out

    def estimates(self) -> dict[str, float]:
        """Crude linear-spring impact estimate (contact area = smaller of the two faces)."""
        cfg = self.config
        body = load_rigid_body(cfg.weight.body)
        material = self.load_material(cfg.specimen.material)
        geom = cfg.specimen.geometry
        r_contact = body.shape.bottom_radius
        if hasattr(geom, "radius"):
            r_contact = min(r_contact, geom.radius)
        height = geom.top_height
        k = material.elastic.initial_youngs_modulus() * math.pi * r_contact**2 / height
        v = self.impact_speed()
        indentation = min(v * math.sqrt(body.mass / k), height)
        return {
            "impact_speed": v,
            "expected_indentation": indentation,
            "expected_strain": indentation / height,
            "impact_duration": math.pi * math.sqrt(body.mass / k),
            "smallest_contact_radius": r_contact,
        }

    # -- actions -----------------------------------------------------------

    def material_key(self, body: str) -> str:
        if body != self.config.specimen.name:
            return super().material_key(body)
        return "specimen.material"

    def apply_action(self, action: str, raw_config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        if action != "drop":
            return super().apply_action(action, raw_config, **kwargs)
        cfg = {**raw_config}
        weight = dict(cfg.get("weight", {}))
        body = weight.get("body")
        preset = kwargs.pop("preset", None)
        if preset:
            body = preset
        mass = kwargs.pop("mass", None)
        if mass:
            body = {"preset": body, "mass": float(mass)} if isinstance(body, str) else {**body, "mass": float(mass)}
        weight["body"] = body
        cfg["weight"] = weight
        drop = dict(cfg.get("drop", {}))
        for key in ("height", "offset", "tilt"):
            value = kwargs.pop(key, None)
            if value is not None:
                drop[key] = value
        cfg["drop"] = drop
        if kwargs:
            raise ConfigError(f"unknown drop action arguments: {sorted(kwargs)}")
        return cfg


def _quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def _tilt_angle(q) -> float:
    """Angle between the body z axis and world z [rad]."""
    return math.acos(max(-1.0, min(1.0, _quat_to_mat(q)[2, 2])))
