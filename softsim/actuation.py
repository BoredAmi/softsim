"""Actuators: inputs that drive soft bodies (pressure, cables, region forces and muscles).

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

A :class:`MuscleActuator` is an active stress ``s`` [Pa] in the elements of a volume region (a zone's
elements, or those whose centroid lies in a box): an *effective* model (``kind: effective``: no real
device), a contracting muscle for evolved soft creatures. The total Cauchy stress of an element is the
material's plus the active ``sigma_a``, and ``s > 0`` **contracts**: along the fibre ``sigma_a = +s m m^T``
(tension in the fibre, which the surroundings balance by compressing the passive material, so a free
bar shortens), ``m = F m0 / |F m0|`` the fibre direction in the current configuration (it follows the
material; ``m0`` is given in the body frame), or isotropic ``sigma_a = +s I`` (the element shrinks).
The nodal loads are the external-equivalent forces of the internal virtual work ``int sigma_a : grad(du) dv``:
``f_a = -int sigma_a grad(N_a) dv`` over the current volume (:func:`muscle_forces`: for a fibre on a linear tet
``f_a = -s v (m . grad N_a) m``). They sum to zero force and zero moment. They are follower loads that
are not the gradient of a potential: MuJoCo and the reference apply them from the current positions (the
reference at the Newton iterate, without a tangent: the active stress is a fraction of the passive
stiffness, so the iteration converges linearly). SOFA refuses (``Capability.MUSCLE_ACTUATION``).
``s(t)`` is a constant, a piecewise-linear schedule, or an oscillator
``(offset + amplitude sin(2 pi f (t - start) + phase)) e(t)`` for ``t >= start``, 0 before, with the
smoothstep envelope ``e`` rising over ``ramp`` (default: 0 if ``start`` is 0, else one period 1/f, so a
delayed muscle starts smoothly).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, model_validator

from softsim.capabilities import Capability
from softsim.config.schema import StrictModel
from softsim.config.units import Angle, Force, Frequency, Length, Pressure, Stiffness, Time, parse_quantity


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


class OscillatorConfig(StrictModel):
    """Sinusoidal muscle stress ``offset + amplitude sin(2 pi frequency (t - start) + phase)`` for
    ``t >= start`` and 0 before, times a smoothstep envelope rising over ``ramp`` (default: none if
    ``start`` is 0, else one period)."""

    frequency: Annotated[Frequency, Field(gt=0)]
    amplitude: Pressure
    phase: Angle = 0.0
    offset: Pressure = 0.0
    start: Annotated[Time, Field(ge=0)] = 0.0
    ramp: Annotated[Time, Field(ge=0)] | None = None


class MuscleActuatorConfig(StrictModel):
    """YAML form of a muscle: the volume ``region`` of the body (or ``box`` ``[[x0, y0, z0], [x1, y1, z1]]``,
    the elements whose centroid lies in it; exactly one), the fibre ``direction`` in the body frame
    (normalised) or ``isotropic``, and the stress ``s`` [Pa] (``s > 0`` contracts) as a constant
    ``stress``, a piecewise-linear ``schedule: [[time, stress], ...]`` (held outside) or an ``oscillator``."""

    name: str = "muscle"
    type: Literal["muscle"] = "muscle"
    region: str | None = None
    box: tuple[tuple[Length, Length, Length], tuple[Length, Length, Length]] | None = None
    direction: tuple[float, float, float] | Literal["isotropic"] = "isotropic"
    stress: Pressure = 0.0
    schedule: list[tuple[Annotated[Time, Field(ge=0)], Pressure]] = []
    oscillator: OscillatorConfig | None = None

    @model_validator(mode="after")
    def _consistent(self) -> "MuscleActuatorConfig":
        if (self.region is None) == (self.box is None):
            raise ValueError(f"muscle {self.name!r}: give exactly one of region and box")
        if self.direction != "isotropic" and not np.linalg.norm(self.direction) > 0:
            raise ValueError(f"muscle {self.name!r}: direction must be a non-zero vector or 'isotropic'")
        if sum([bool(self.stress), bool(self.schedule), self.oscillator is not None]) > 1:
            raise ValueError(f"muscle {self.name!r}: give one of stress, schedule and oscillator")
        return self


@dataclass(frozen=True)
class MuscleActuator:
    """Active stress [Pa] in ``elements`` of soft body ``body`` along the unit body-frame ``direction`` (None:
    isotropic); ``region`` is the selector's name (or ``box``) for messages. ``oscillator`` is
    ``(frequency, amplitude, phase, offset, start, ramp)``; see module docstring."""

    name: str
    body: str
    region: str
    direction: tuple[float, float, float] | None = None
    stress: float = 0.0
    schedule: tuple[tuple[float, float], ...] = ()
    oscillator: tuple[float, float, float, float, float, float] | None = None
    elements: np.ndarray = field(default_factory=lambda: np.zeros(0, int), compare=False, repr=False)

    @classmethod
    def from_config(cls, cfg: MuscleActuatorConfig, body: str, elements: np.ndarray) -> "MuscleActuator":
        d = None
        if cfg.direction != "isotropic":
            v = np.asarray(cfg.direction, float)
            d = tuple((v / np.linalg.norm(v)).tolist())
        o = cfg.oscillator
        osc = None if o is None else (o.frequency, o.amplitude, o.phase, o.offset, o.start,
                                      (0.0 if o.start == 0 else 1.0 / o.frequency) if o.ramp is None else o.ramp)
        return cls(cfg.name, body, cfg.region or "box", d, cfg.stress, tuple(map(tuple, cfg.schedule)), osc,
                   np.asarray(elements, int))

    def value(self, t: float) -> float:
        """The stress s(t) [Pa]."""
        if self.oscillator is not None:
            f, amp, phase, offset, start, ramp = self.oscillator
            if t < start:
                return 0.0
            r = min((t - start) / ramp, 1.0) if ramp > 0 else 1.0
            return float((3 * r * r - 2 * r ** 3) * (offset + amp * np.sin(2 * np.pi * f * (t - start) + phase)))
        if self.schedule:
            times, values = zip(*self.schedule)
            return float(np.interp(t, times, values))
        return self.stress

    @property
    def requires(self) -> frozenset[Capability]:
        return frozenset({Capability.MUSCLE_ACTUATION})


Actuator = PressureActuator | CableActuator | ForceActuator | MuscleActuator
ActuatorConfig = (PressureActuatorConfig | CableActuatorConfig | ForceActuatorConfig
                  | MuscleActuatorConfig)


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


def muscle_rest(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Reference gradients of the shape functions of linear tets at rest (E, 4, 3), and nothing else is needed
    per element: ``grad_X N_a`` (``muscle_forces`` takes the current volume from the positions)."""
    x = points[tets]
    inv = np.linalg.inv(np.swapaxes(x[:, 1:] - x[:, :1], 1, 2))      # (E, 3, 3): dX -> barycentric
    g = np.concatenate([-inv.sum(axis=1, keepdims=True), inv], axis=1)  # (E, 4, 3): rows grad_X lambda_a
    return g


def active_stress_loads(xe: np.ndarray, dndx: np.ndarray, wv: np.ndarray, s: float,
                        m0: np.ndarray | None) -> np.ndarray:
    """Element nodal loads (E, n, 3) of an active stress ``s`` (``s > 0`` contracts) from quadrature:
    ``xe`` current positions (E, n, 3), ``dndx`` reference gradients (E, Q, n, 3), ``wv`` reference weights
    (E, Q) (rest volume), ``m0`` unit fibre (3,) or (E, 3) in the reference frame, None for isotropic.
    The Cauchy stress is ``sigma_a = s m m^T`` (``m = F m0 / |F m0|``) or ``s I``, the load
    ``f_a = -int sigma_a grad_x N_a dv = -wv P grad_X N_a`` with ``P = J sigma_a F^-T``."""
    f = np.einsum("eni,eqnj->eqij", xe, dndx)                       # (E, Q, 3, 3), F = dx/dX
    j = np.linalg.det(f)
    finv_t = np.swapaxes(np.linalg.inv(f), -1, -2)
    if m0 is None:
        sigma = s * np.eye(3) * np.ones((len(f), f.shape[1], 1, 1))
    else:
        fm = np.einsum("eqij,ej->eqi", f, np.broadcast_to(m0, (len(f), 3)))
        m = fm / np.linalg.norm(fm, axis=-1, keepdims=True)
        sigma = s * m[..., :, None] * m[..., None, :]
    p = j[..., None, None] * sigma @ finv_t
    return -np.einsum("eq,eqij,eqnj->eni", wv, p, dndx)


def muscle_forces(x: np.ndarray, tets: np.ndarray, s: float, m0: np.ndarray | None, rest: tuple) -> np.ndarray:
    """Nodal forces (N, 3) of the active stress ``s`` [Pa] on linear ``tets`` at the current positions ``x``:
    ``f_a = -v sigma_a grad N_a`` with ``sigma_a = s m m^T`` (``m = F m0 / |F m0|``; ``m0`` the unit
    fibre in the rest frame, (3,) or (E, 3)) or ``s I`` (``m0`` None), ``s > 0`` contracting (a free
    fibre shortens). ``rest = muscle_data(points, tets)`` (computed once). Forces and moments sum to zero."""
    dndx, vol0 = rest
    out = np.zeros_like(x, dtype=float)
    f = active_stress_loads(x[tets], dndx[:, None], vol0[:, None], s, m0)
    np.add.at(out, tets.ravel(), f.reshape(-1, 3))
    return out


def muscle_data(points: np.ndarray, tets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rest data of :func:`muscle_forces`: reference shape-function gradients (E, 4, 3) and rest volumes (E,)."""
    x = points[tets]
    vol0 = np.linalg.det(x[:, 1:] - x[:, :1]) / 6.0
    return muscle_rest(points, tets), vol0


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
