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
}

_QUANTITY = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*([A-Za-z/^0-9]+)?\s*$")


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
    return number * factor


def _field(dimension: str):
    return BeforeValidator(lambda v: parse_quantity(v, dimension))


Length = Annotated[float, _field("length")]
Mass = Annotated[float, _field("mass")]
Time = Annotated[float, _field("time")]
Pressure = Annotated[float, _field("pressure")]
Density = Annotated[float, _field("density")]
Velocity = Annotated[float, _field("velocity")]
Force = Annotated[float, _field("force")]
Frequency = Annotated[float, _field("frequency")]
Angle = Annotated[float, _field("angle")]
