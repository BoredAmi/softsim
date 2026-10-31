"""Actuators: inputs that drive soft bodies (pressure now; cables and others later).

A :class:`PressureActuator` loads a named surface region of a soft body (e.g. a
hollow shape's ``cavity``) with a gauge pressure ``p``: the traction ``-p n``,
``n`` the body's outward normal, following the deforming surface. Its value is
a piecewise-linear schedule in time, which a controller can override at run
time (``Backend.set_control``).

Backends apply it as follows:
    * fem (reference): the potential ``-p V``, with ``V`` the volume the region
      encloses (exact consistent follower load on the quadratic faces, symmetric
      tangent). Exact for closed regions and for regions closed by symmetry
      planes through the origin; :func:`open_edges` finds regions that are not.
    * MuJoCo, SOFA: :func:`pressure_forces` on the linear surface triangles,
      recomputed every step from the current positions (explicit).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import Field

from softsim.capabilities import Capability
from softsim.config.schema import StrictModel
from softsim.config.units import Pressure, Time


class PressureActuatorConfig(StrictModel):
    """YAML form of a pressure actuator. ``schedule`` is ``[[time, pressure], ...]``, linear in
    between and held outside; without it the pressure is constant."""

    name: str = "pressure"
    type: Literal["pressure"] = "pressure"
    region: str = "cavity"
    pressure: Pressure = 0.0
    schedule: list[tuple[Annotated[Time, Field(ge=0)], Pressure]] = []


@dataclass(frozen=True)
class PressureActuator:
    """Gauge pressure [Pa] on region ``region`` of soft body ``body``; see module docstring."""

    name: str
    body: str
    region: str
    pressure: float = 0.0
    schedule: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_config(cls, cfg: PressureActuatorConfig, body: str) -> "PressureActuator":
        return cls(cfg.name, body, cfg.region, cfg.pressure, tuple(map(tuple, cfg.schedule)))

    def value(self, t: float) -> float:
        if not self.schedule:
            return self.pressure
        times, values = zip(*self.schedule)
        return float(np.interp(t, times, values))

    @property
    def requires(self) -> frozenset[Capability]:
        return frozenset({Capability.PRESSURE_ACTUATION})


def pressure_forces(points: np.ndarray, tris: np.ndarray, pressure: float) -> np.ndarray:
    """Nodal forces (N, 3) of ``pressure`` on triangles ``tris`` (outward normals) at ``points``:
    ``-p A n / 3`` to each corner of each triangle."""
    p = points[tris]
    share = -pressure / 6.0 * np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    out = np.zeros_like(points, dtype=float)
    np.add.at(out, tris.ravel(), np.repeat(share, 3, axis=0))
    return out


def enclosed_volume(points: np.ndarray, tris: np.ndarray) -> float:
    """Volume enclosed by triangles ``tris`` whose normals point out of the solid, i.e. into the
    enclosed space (a cavity's volume, positive)."""
    p = points[tris]
    return -float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum()) / 6.0


def open_edges(tris: np.ndarray) -> np.ndarray:
    """Edges (K, 2) used by only one triangle of ``tris``: the region's boundary (empty if closed)."""
    e = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    key, counts = np.unique(e, axis=0, return_counts=True)
    return key[counts == 1]
