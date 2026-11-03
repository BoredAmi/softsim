"""Actuators: inputs that drive soft bodies (pressure, cables and region forces).

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

A :class:`CableActuator` is a tendon of tension ``T`` routed through the body
along a polyline of via points that are embedded in its elements (frictionless
guides that move with the material). Its first point is where it leaves the
body towards the motor, which is mounted on the body there: the cable is
internal, so the net force on the body is zero. Its load is the potential
``T L(x)``, ``L`` the cable's current length through the via points (follower,
symmetric positive semi-definite tangent). The reference interpolates the via
points with its quadratic shape functions; MuJoCo and SOFA with the linear ones
(:func:`cable_forces`, explicit). The cable stays taut: pulling with ``T > 0``.

A cable with ``control: stroke`` is driven by length instead, like a servo
reeling it in: its value is the stroke ``s`` [m] and the cable is a tendon of
axial ``stiffness`` k, slack when shorter than its free length ``L0 - s`` (``L0``
its length at rest): ``T = k max(L - L0 + s, 0)`` (:meth:`CableActuator.tension_at`).
With a stiff tendon the stroke sets the shortening; the tension follows from
the body. The reference adds ``k/2 (L - L0 + s)^2`` to its potential (implicit);
MuJoCo and SOFA compute ``T`` from the current length every step (explicit).

A :class:`ForceActuator` is a total force ``F`` [N] along a fixed world
``direction`` spread over a named surface region by area (a dead load: e.g. a
payload on a fingertip, or a probe pressing on it). The reference applies the
consistent load of a uniform traction on its quadratic faces (all of it on the
mid-edge nodes); MuJoCo and SOFA give each triangle's vertices a third of its
share (:func:`region_force_weights`). It depends on no position, so it has no
stability limit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from softsim.capabilities import Capability
from softsim.config.schema import StrictModel
from softsim.config.units import Force, Length, Pressure, Stiffness, Time, parse_quantity


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


class CableActuatorConfig(StrictModel):
    """YAML form of a cable actuator: ``path`` is ``[[x, y, z], ...]`` in the body's frame (its mesh
    coordinates), from the exit towards the motor to the anchor; ``schedule`` is
    ``[[time, tension], ...]``, or with ``control: stroke`` ``[[time, stroke], ...]`` (lengths; the
    cable then needs its tendon ``stiffness``). Via points are placed along the path at most
    ``spacing`` apart (default: the body's mesh size). ``path: geometry`` takes the path the body's shape
    defines (a ``cable_finger``'s tendon). With ``anchor: <region>`` the last path point,
    which must lie on that boundary region (a face), is not a point load: the cable's end force is
    spread over the region as the linear traction with the same resultant and moment
    (:func:`anchor_weights`), so the face does not dent under it."""

    name: str = "cable"
    type: Literal["cable"] = "cable"
    path: Annotated[list[tuple[Length, Length, Length]], Field(min_length=2)] | Literal["geometry"]
    anchor: str | None = None
    control: Literal["tension", "stroke"] = "tension"
    tension: Annotated[Force, Field(ge=0)] = 0.0
    stroke: Length = 0.0
    stiffness: Annotated[Stiffness, Field(gt=0)] | None = None
    schedule: list[tuple[Annotated[Time, Field(ge=0)], float]] = []
    spacing: Annotated[Length, Field(gt=0)] | None = None

    @model_validator(mode="before")
    @classmethod
    def _schedule_units(cls, data):
        """Schedule values are forces, or lengths for a stroke-controlled cable."""
        if isinstance(data, dict) and isinstance(data.get("schedule"), list):
            dim = "length" if data.get("control") == "stroke" else "force"
            rows = []
            for row in data["schedule"]:
                if isinstance(row, (list, tuple)) and len(row) == 2:
                    row = (row[0], parse_quantity(row[1], dim))
                rows.append(row)
            data = {**data, "schedule": rows}
        return data

    @model_validator(mode="after")
    def _consistent(self) -> "CableActuatorConfig":
        if self.control == "stroke" and self.stiffness is None:
            raise ValueError(f"cable {self.name!r}: control: stroke needs the tendon's stiffness (N/m)")
        if self.control == "tension" and any(v < 0 for _, v in self.schedule):
            raise ValueError(f"cable {self.name!r}: a cable can only pull, its tensions must be >= 0")
        return self


@dataclass(frozen=True)
class CableActuator:
    """Cable of tension [N] through ``via`` points (body frame) of soft body ``body``; see module
    docstring."""

    name: str
    body: str
    via: tuple[tuple[float, float, float], ...]
    tension: float = 0.0            # the constant value: a tension, or the stroke if stroke-controlled
    schedule: tuple[tuple[float, float], ...] = ()
    stiffness: float | None = None  # set: stroke-controlled tendon of this stiffness [N/m]
    anchor: str | None = None       # set: region the last via point's load is spread over

    @classmethod
    def from_config(cls, cfg: CableActuatorConfig, body: str, spacing: float) -> "CableActuator":
        stroke = cfg.control == "stroke"
        if cfg.path == "geometry":
            raise ValueError(f"cable {cfg.name!r}: path: geometry needs a shape that defines the tendon "
                             "(geometry type cable_finger)")
        return cls(cfg.name, body, tuple(map(tuple, resample(np.array(cfg.path, float), cfg.spacing or spacing))),
                   cfg.stroke if stroke else cfg.tension, tuple(map(tuple, cfg.schedule)),
                   cfg.stiffness if stroke else None, cfg.anchor)

    @property
    def stroke_controlled(self) -> bool:
        return self.stiffness is not None

    def tension_at(self, length: float, value: float, rest_length: float) -> float:
        """Tension [N] at current ``length`` for control ``value`` (a tension, or a stroke)."""
        if self.stiffness is None:
            return value
        return self.stiffness * max(length - rest_length + value, 0.0)

    def value(self, t: float) -> float:
        if not self.schedule:
            return self.tension
        times, values = zip(*self.schedule)
        return float(np.interp(t, times, values))

    @property
    def requires(self) -> frozenset[Capability]:
        return frozenset({Capability.CABLE_ACTUATION})


class ForceActuatorConfig(StrictModel):
    """YAML form of a region force: ``force`` [N] (or ``schedule: [[time, force], ...]``) along the
    world ``direction`` (normalised), spread over ``region`` by area. A negative force pushes the
    other way."""

    name: str = "force"
    type: Literal["force"] = "force"
    region: str
    direction: tuple[float, float, float]
    force: Force = 0.0
    schedule: list[tuple[Annotated[Time, Field(ge=0)], Force]] = []

    @model_validator(mode="after")
    def _direction(self) -> "ForceActuatorConfig":
        if not np.linalg.norm(self.direction) > 0:
            raise ValueError(f"force {self.name!r}: direction must be a non-zero vector")
        return self


@dataclass(frozen=True)
class ForceActuator:
    """Total force [N] along unit ``direction`` (world) on region ``region`` of soft body ``body``,
    spread by area; see module docstring."""

    name: str
    body: str
    region: str
    direction: tuple[float, float, float]
    force: float = 0.0
    schedule: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_config(cls, cfg: ForceActuatorConfig, body: str) -> "ForceActuator":
        d = np.asarray(cfg.direction, float)
        return cls(cfg.name, body, cfg.region, tuple((d / np.linalg.norm(d)).tolist()), cfg.force,
                   tuple(map(tuple, cfg.schedule)))

    def value(self, t: float) -> float:
        if not self.schedule:
            return self.force
        times, values = zip(*self.schedule)
        return float(np.interp(t, times, values))

    @property
    def requires(self) -> frozenset[Capability]:
        return frozenset({Capability.REGION_FORCE})


Actuator = PressureActuator | CableActuator | ForceActuator
ActuatorConfig = PressureActuatorConfig | CableActuatorConfig | ForceActuatorConfig


def region_force_weights(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Share of a region force on every vertex (N,), summing to 1: a third of each triangle's area
    to each of its vertices, over the region's area."""
    tri = points[tris]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    w = np.zeros(len(points))
    np.add.at(w, tris.ravel(), np.repeat(area / 3, 3))
    return w / area.sum()


def resample(path: np.ndarray, spacing: float) -> np.ndarray:
    """``path``'s corners plus equally spaced points on each leg, at most ``spacing`` apart."""
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / spacing - 1e-9)), 1)
        out.extend(a + (b - a) * (k / n) for k in range(1, n + 1))
    return np.array(out)


def embed(points: np.ndarray, tets: np.ndarray, via: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Containing tetrahedron (K,) and barycentric coordinates (K, 4) of each via point; on a shared
    face or edge, any containing element. Raises ``ValueError`` for points outside the mesh."""
    x = points[tets]                                               # (E, 4, 3)
    inv = np.linalg.inv(np.swapaxes(x[:, 1:] - x[:, :1], 1, 2))    # (E, 3, 3)
    elems, bary = [], []
    for k, v in enumerate(np.asarray(via, float)):
        lam = np.einsum("eij,ej->ei", inv, v - x[:, 0])
        lam = np.concatenate([1.0 - lam.sum(axis=1, keepdims=True), lam], axis=1)
        e = int(np.argmax(lam.min(axis=1)))
        if lam[e].min() < -tol:
            raise ValueError(f"cable via point {k} at {np.round(v * 1e3, 3).tolist()} mm is outside the body")
        w = np.clip(lam[e], 0.0, None)
        elems.append(e)
        bary.append(w / w.sum())
    return np.array(elems), np.array(bary)


# degree-4 rule on the unit-area triangle (Dunavant, 6 points): barycentric points and weights (sum 1)
_TA, _TB = 0.445948490915965, 0.091576213509771
_TRI_Q = np.array([[_TA, _TA, 1 - 2 * _TA], [_TA, 1 - 2 * _TA, _TA], [1 - 2 * _TA, _TA, _TA],
                   [_TB, _TB, 1 - 2 * _TB], [_TB, 1 - 2 * _TB, _TB], [1 - 2 * _TB, _TB, _TB]])
_TRI_W = np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def anchor_weights(points: np.ndarray, faces: np.ndarray, point: np.ndarray,
                   tol: float = 1e-6) -> tuple[np.ndarray, np.ndarray]:
    """Nodes (M,) of the region ``faces`` and their weights (M,): the linear traction over the region with
    unit resultant whose centre of action is ``point``. ``faces`` are triangles (K, 3) of a linear mesh, or
    (K, 6) quadratic ones (corners, then the mid-edge nodes of edges 01, 12, 20) with their consistent
    nodal loads. The traction is ``(1 + g . (x - c)) / A`` with ``c`` the area centroid and ``g`` fixed by
    ``sum w_j x_j = point``: a force ``F`` on the cable's end then loads the region with the resultant ``F``
    and the moment ``F`` about ``point`` would have (statically equivalent, exact for a beam end by
    St-Venant). Weights go negative on the far side when ``point`` is outside the region's core (a cable
    3 mm off the axis of a 10 mm square section: tension there, as beam theory has it). Raises ``ValueError``
    when ``point`` is off the region (outside its bounding box, or by more than ``tol`` times its size from
    the plane the traction can reach)."""
    faces = np.asarray(faces, int)
    corner = faces[:, :3]
    tri = points[corner]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    q = _TRI_Q
    shape = q if faces.shape[1] == 3 else np.column_stack(
        [q * (2 * q - 1), 4 * q[:, 0] * q[:, 1], 4 * q[:, 1] * q[:, 2], 4 * q[:, 2] * q[:, 0]])
    a = area[:, None] * _TRI_W[None, :] / area.sum()                      # (K, Q) share of the region's area
    x = np.einsum("qc,kci->kqi", q, tri)                                  # quadrature points (K, Q, 3)
    c = np.einsum("kq,kqi->i", a, x)
    d = x - c
    moment = np.einsum("kq,kqi,kqj->ij", a, d, d)
    nodes, inverse = np.unique(faces, return_inverse=True)
    inverse = inverse.reshape(faces.shape)
    w0 = np.zeros(len(nodes))
    w1 = np.zeros((len(nodes), 3))
    for j in range(faces.shape[1]):
        np.add.at(w0, inverse[:, j], a @ shape[:, j])
        np.add.at(w1, inverse[:, j], np.einsum("kq,q,kqi->ki", a, shape[:, j], d))
    g = np.linalg.lstsq(moment, np.asarray(point, float) - c, rcond=1e-9)[0]
    w = w0 + w1 @ g
    xn = points[nodes]
    size = float(np.linalg.norm(xn.max(axis=0) - xn.min(axis=0)))
    outside = np.maximum(np.maximum(xn.min(axis=0) - point, point - xn.max(axis=0)), 0.0)  # beyond the bounding box
    miss = max(float(np.linalg.norm(w @ xn - point)), float(np.linalg.norm(outside)))
    if miss > max(tol * size, 1e-9):
        raise ValueError(f"cable anchor point {np.round(np.asarray(point) * 1e3, 3).tolist()} mm is "
                         f"{miss * 1e3:.3g} mm off the anchor region")
    return nodes, w


def pad_rows(rows: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Via-point rows ``(nodes, weights)`` of different widths as arrays (K, W): short rows are filled
    with their own first node at weight 0 (no force, and no new coupling)."""
    width = max(len(n) for n, _ in rows)
    nodes = np.array([np.concatenate([n, np.full(width - len(n), n[0])]) for n, _ in rows], int)
    weights = np.array([np.concatenate([w, np.zeros(width - len(w))]) for _, w in rows], float)
    return nodes, weights


def cable_rows(points: np.ndarray, tets: np.ndarray, via: np.ndarray,
               anchor_tris: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Via points of a cable on the linear mesh as ``(nodes, weights)``, (K, W) each: via point ``k`` sits at
    ``sum_j weights[k, j] points[nodes[k, j]]``. Interior points are embedded in their tetrahedron; with
    ``anchor_tris`` the last one is the traction of :func:`anchor_weights` over those triangles."""
    via = np.asarray(via, float)
    n = len(via) - (anchor_tris is not None)
    elems, bary = embed(points, tets, via[:n])
    rows = list(zip(tets[elems], bary))
    if anchor_tris is not None:
        rows.append(anchor_weights(points, anchor_tris, via[-1]))
    return pad_rows(rows)


def cable_length(points: np.ndarray, nodes: np.ndarray, weights: np.ndarray) -> float:
    """Length of the polyline through the via points ``sum_j weights[k, j] points[nodes[k, j]]``."""
    p = np.einsum("kj,kji->ki", weights, points[nodes])
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def cable_forces(points: np.ndarray, nodes: np.ndarray, weights: np.ndarray, tension: float) -> np.ndarray:
    """Nodal forces (N, 3) of a cable of ``tension`` through the via points (see
    :func:`cable_length`): ``-T dL/dx``, each via point's force spread by its weights."""
    p = np.einsum("kj,kji->ki", weights, points[nodes])
    s = np.diff(p, axis=0)
    u = s / np.linalg.norm(s, axis=1, keepdims=True)
    f = tension * (np.vstack([u, np.zeros(3)]) - np.vstack([np.zeros(3), u]))  # pulls each via point along the cable
    out = np.zeros_like(points, dtype=float)
    np.add.at(out, nodes.ravel(), (weights[:, :, None] * f[:, None, :]).reshape(-1, 3))
    return out


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
