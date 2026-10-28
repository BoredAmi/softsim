"""Rigid primitive shapes and the rigid-body preset database.

Convention: the body frame origin is the centre of the bottom face, +z up.
Shapes report their parts as simple primitives (cylinders/boxes with offsets)
so every backend can build them; mass is distributed over parts by volume.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping, Union

import yaml
from pydantic import Field

from softsim.config.schema import StrictModel, validate
from softsim.config.units import Length
from softsim.errors import ConfigError

PositiveLength = Annotated[Length, Field(gt=0)]
PRESET_FILE = Path(__file__).parent / "presets" / "rigid_bodies.yaml"


@dataclass(frozen=True)
class Part:
    """A primitive piece of a rigid shape.

    ``kind`` is ``"cylinder"`` (``size`` = radius, half-height, unused) or
    ``"box"`` (``size`` = half extents); ``center`` is in the body frame.
    """

    kind: str
    size: tuple[float, float, float]
    center: tuple[float, float, float]

    def volume(self) -> float:
        if self.kind == "cylinder":
            return math.pi * self.size[0] ** 2 * 2 * self.size[1]
        return 8 * self.size[0] * self.size[1] * self.size[2]


class RigidCylinder(StrictModel):
    type: Literal["cylinder"] = "cylinder"
    diameter: PositiveLength
    height: PositiveLength

    @property
    def bottom_radius(self) -> float:
        return self.diameter / 2

    def parts(self) -> list[Part]:
        return [Part("cylinder", (self.diameter / 2, self.height / 2, 0), (0, 0, self.height / 2))]


class WeightWithKnob(StrictModel):
    """Calibration-style weight: a cylinder with a smaller cylindrical knob on top."""

    type: Literal["weight_with_knob"] = "weight_with_knob"
    diameter: PositiveLength
    height: PositiveLength
    knob_diameter: PositiveLength
    knob_height: PositiveLength

    @property
    def bottom_radius(self) -> float:
        return self.diameter / 2

    def parts(self) -> list[Part]:
        h, kh = self.height, self.knob_height
        return [
            Part("cylinder", (self.diameter / 2, h / 2, 0), (0, 0, h / 2)),
            Part("cylinder", (self.knob_diameter / 2, kh / 2, 0), (0, 0, h + kh / 2)),
        ]


class Plate(StrictModel):
    """Rectangular plate / box; ``size`` are full extents x, y, z."""

    type: Literal["plate"] = "plate"
    size: tuple[PositiveLength, PositiveLength, PositiveLength]

    @property
    def bottom_radius(self) -> float:
        return min(self.size[0], self.size[1]) / 2

    def parts(self) -> list[Part]:
        sx, sy, sz = self.size
        return [Part("box", (sx / 2, sy / 2, sz / 2), (0, 0, sz / 2))]


RigidShape = Annotated[Union[RigidCylinder, WeightWithKnob, Plate], Field(discriminator="type")]


class RigidBodyDef(StrictModel):
    """A rigid body definition (shape + mass), as stored in presets."""

    description: str = ""
    source: str = ""
    mass: Annotated[float, Field(gt=0)]
    shape: RigidShape


def _with_units(data: Mapping[str, Any]) -> dict[str, Any]:
    from softsim.config.units import parse_quantity

    out = dict(data)
    out["mass"] = parse_quantity(out["mass"], "mass")
    return out


def rigid_presets() -> dict[str, dict[str, Any]]:
    """Raw preset mappings by name."""
    return yaml.safe_load(PRESET_FILE.read_text()) or {}


def load_rigid_body(ref: str | Mapping[str, Any]) -> RigidBodyDef:
    """Resolve a rigid body reference: preset name, inline definition, or
    ``{preset: name, ...overrides}``. ``mass`` accepts unit strings."""
    presets = rigid_presets()
    if isinstance(ref, str):
        if ref not in presets:
            raise ConfigError(f"unknown rigid body preset {ref!r}; available: {', '.join(sorted(presets))}")
        data = presets[ref]
    elif isinstance(ref, Mapping):
        data = dict(ref)
        name = data.pop("preset", None)
        if name is not None:
            from softsim.config.loader import deep_merge

            if name not in presets:
                raise ConfigError(f"unknown rigid body preset {name!r}; available: {', '.join(sorted(presets))}")
            data = deep_merge(presets[name], data)
    else:
        raise ConfigError(f"rigid body must be a preset name or mapping, got {type(ref).__name__}")
    if "mass" not in data:
        raise ConfigError(f"rigid body {ref!r} has no mass")
    return validate(RigidBodyDef, _with_units(data), where=f"rigid body {ref!r:.60}")
