"""Unit handling. Internally everything is SI; YAML may use unit strings.

A config value can be a plain number (taken as SI) or a string such as
``"17 mm"``, ``"19.49 g"``, ``"55 kPa"`` or ``"0.1 ms"``. Strings are converted
to SI when the config is validated, and the unit's dimension must match the
field (a length field rejects ``"5 g"``).

Pydantic field types are provided for each dimension, e.g.::

    class Cylinder(BaseModel):
        radius: Length
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Annotated

from pydantic import BeforeValidator

# unit -> (factor to SI, dimension)
_UNITS: dict[str, tuple[float, str]] = {
    "m": (1.0, "length"), "cm": (1e-2, "length"), "mm": (1e-3, "length"), "um": (1e-6, "length"),
    "kg": (1.0, "mass"), "g": (1e-3, "mass"), "mg": (1e-6, "mass"),
    "s": (1.0, "time"), "ms": (1e-3, "time"), "us": (1e-6, "time"),
    "Pa": (1.0, "pressure"), "kPa": (1e3, "pressure"), "MPa": (1e6, "pressure"),
    "psi": (6894.757293168, "pressure"),
    "kg/m^3": (1.0, "density"), "kg/m3": (1.0, "density"), "g/cm^3": (1e3, "density"),
    "g/cm3": (1e3, "density"),
    "m/s": (1.0, "velocity"), "mm/s": (1e-3, "velocity"),
    "N": (1.0, "force"), "mN": (1e-3, "force"),
    "Hz": (1.0, "frequency"), "kHz": (1e3, "frequency"),
    "rad": (1.0, "angle"), "deg": (0.017453292519943295, "angle"),
    "rad/s": (1.0, "angular_velocity"), "deg/s": (0.017453292519943295, "angular_velocity"),
    "N/m": (1.0, "stiffness"), "N/mm": (1e3, "stiffness"),
    "N*s/m": (1.0, "viscous_damping"), "Ns/m": (1.0, "viscous_damping"),
    "1/s": (1.0, "rate"),
    "J/m^2": (1.0, "energy_per_area"), "J/m2": (1.0, "energy_per_area"), "mJ/m^2": (1e-3, "energy_per_area"),
    "Pa*s": (1.0, "viscosity"), "Pas": (1.0, "viscosity"), "kPa*s": (1e3, "viscosity"),
    "K": (1.0, "temperature"), "degC": (1.0, "temperature"),
    "T": (1.0, "flux_density"), "mT": (1e-3, "flux_density"),
    "A/m": (1.0, "field_strength"), "kA/m": (1e3, "field_strength"),
}
# affine units: SI value = number * factor + offset
_OFFSETS: dict[str, float] = {"degC": 273.15}

# SI unit symbol per dimension (what values are stored and printed in)
SI_UNITS: dict[str, str] = {
    "length": "m", "mass": "kg", "time": "s", "pressure": "Pa", "density": "kg/m^3", "velocity": "m/s",
    "force": "N", "frequency": "Hz", "angle": "rad", "angular_velocity": "rad/s", "stiffness": "N/m", "viscous_damping": "N*s/m",
    "rate": "1/s", "energy_per_area": "J/m^2", "viscosity": "Pa*s", "temperature": "K",
    "flux_density": "T", "field_strength": "A/m",
}

_QUANTITY = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z/^*0-9]+)?\s*$")


def parse_quantity(value: object, dimension: str) -> float:
    """Convert ``value`` to an SI float of the given dimension.

    Args:
        value: A number (already SI) or a string ``"<number> <unit>"``.
        dimension: Expected dimension, e.g. ``"length"``.

    Raises:
        ValueError: The string cannot be parsed or has the wrong dimension.
    """
    if isinstance(value, bool):
        raise ValueError(f"expected a {dimension}, got a boolean")
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"expected a {dimension} (number or '<value> <unit>'), got {value!r}")
    match = _QUANTITY.match(value)
    if not match:
        raise ValueError(f"cannot parse quantity {value!r}")
    number, unit = float(match.group(1)), match.group(2)
    if unit is None:
        return number
    if unit not in _UNITS:
        raise ValueError(f"unknown unit {unit!r} in {value!r}; known: {', '.join(_UNITS)}")
    factor, dim = _UNITS[unit]
    if dim != dimension:
        raise ValueError(f"{value!r} is a {dim}, expected a {dimension}")
    return number * factor + _OFFSETS.get(unit, 0.0)


@dataclass(frozen=True)
class Unit:
    """Marker in a field's ``Annotated`` metadata: the field is a quantity of ``dimension``
    (stored in SI). Lets tools such as ``softsim params`` print units."""

    dimension: str

    @property
    def symbol(self) -> str:
        return SI_UNITS[self.dimension]


def _quantity(dimension: str):
    return Annotated[float, BeforeValidator(lambda v: parse_quantity(v, dimension)), Unit(dimension)]


Length = _quantity("length")
Mass = _quantity("mass")
Time = _quantity("time")
Pressure = _quantity("pressure")
Density = _quantity("density")
Velocity = _quantity("velocity")
Force = _quantity("force")
Frequency = _quantity("frequency")
Angle = _quantity("angle")
AngularVelocity = _quantity("angular_velocity")
Stiffness = _quantity("stiffness")
ViscousDamping = _quantity("viscous_damping")
Rate = _quantity("rate")  # 1/s: strain rates, Rayleigh alpha
StrainRate = Rate
EnergyPerArea = _quantity("energy_per_area")
Viscosity = _quantity("viscosity")
Temperature = _quantity("temperature")
FluxDensity = _quantity("flux_density")  # T: a magnet's remanence
FieldStrength = _quantity("field_strength")  # A/m: an applied field H
