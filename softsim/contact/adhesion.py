"""Cohesive adhesion between soft-body surfaces and rigid bodies / the ground.

Backend-neutral: given current vertex positions and rigid-body frames, returns
the adhesive forces on soft vertices and the reaction wrenches on rigid
bodies. A backend applies them as external forces every physics step (see
:class:`softsim.materials.Adhesion` for the traction law and bond rules).

Geometry: rigid bodies are unions of primitive parts (cylinders, boxes) with
analytic signed distances; the ground is the plane z = 0. Gaps are measured
from the soft body's collision skin (``contact.soft_radius``) so that zero gap
is exactly where the backend's contact starts. Each surface vertex adheres with
its share of the surface area *facing* the counterpart (area-weighted vertex
normal, current configuration, projected on the counterpart's surface normal),
so side walls next to a bonded face do not stick.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from softsim.geometry.rigid import Part
from softsim.materials.models import Adhesion
from softsim.meshing import TetMesh
from softsim.scene import GROUND_NAME, Scene


def traction(gap: np.ndarray, adhesion: Adhesion) -> np.ndarray:
    """Attractive normal traction [Pa] for opening ``gap`` >= 0 (zero for gap <= 0)."""
    x = np.maximum(gap, 0.0) / adhesion.range
    return adhesion.strength * x * np.exp(1.0 - x)


def _cylinder_sdf(p: np.ndarray, radius: float, half_height: float) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance and outward unit normal of a z-aligned cylinder centred at 0."""
    rho = np.hypot(p[:, 0], p[:, 1])
    qr, qz = rho - radius, np.abs(p[:, 2]) - half_height
    outside = np.hypot(np.maximum(qr, 0), np.maximum(qz, 0))
    dist = np.minimum(np.maximum(qr, qz), 0.0) + outside
    # gradient in (rho, |z|): towards the nearest feature
    gr = np.where(outside > 0, np.maximum(qr, 0) / np.maximum(outside, 1e-300), (qr > qz).astype(float))
    gz = np.where(outside > 0, np.maximum(qz, 0) / np.maximum(outside, 1e-300), (qr <= qz).astype(float))
    radial = np.where(rho[:, None] > 1e-12, p[:, :2] / np.maximum(rho, 1e-300)[:, None], 0.0)
    normal = np.column_stack([gr[:, None] * radial, gz * np.sign(p[:, 2] + 1e-300)])
    return dist, normal


def _box_sdf(p: np.ndarray, half: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance and outward unit normal of an axis-aligned box centred at 0."""
    q = np.abs(p) - half
    outside_vec = np.maximum(q, 0.0)
    outside = np.linalg.norm(outside_vec, axis=1)
    inside = np.minimum(q.max(axis=1), 0.0)
    dist = inside + outside
    face = np.zeros_like(q)
    face[np.arange(len(q)), q.argmax(axis=1)] = 1.0
    grad = np.where((outside > 0)[:, None], outside_vec / np.maximum(outside, 1e-300)[:, None], face)
    return dist, grad * np.sign(p + 1e-300)


def parts_sdf(points: np.ndarray, parts: tuple[Part, ...]) -> tuple[np.ndarray, np.ndarray]:
    """Signed distance and outward normal to the union of ``parts`` (body-frame points)."""
    best_d = np.full(len(points), np.inf)
    best_n = np.zeros_like(points)
    for part in parts:
        p = points - np.asarray(part.center)
        if part.kind == "cylinder":
            d, n = _cylinder_sdf(p, part.size[0], part.size[1])
        elif part.kind == "box":
            d, n = _box_sdf(p, np.asarray(part.size))
        else:
            raise ValueError(f"no signed distance for part kind {part.kind!r}")
        closer = d < best_d
        best_d = np.where(closer, d, best_d)
        best_n[closer] = n[closer]
    return best_d, best_n


def surface_areas(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray]:
    """Surface vertex indices and their tributary (reference) areas [m^2]."""
    idx, tris = surface_topology(mesh)
    p = mesh.points[idx]
    area = 0.5 * np.linalg.norm(np.cross(p[tris[:, 1]] - p[tris[:, 0]], p[tris[:, 2]] - p[tris[:, 0]]), axis=1)
    acc = np.zeros(len(idx))
    np.add.at(acc, tris.ravel(), np.repeat(area / 3.0, 3))
    return idx, acc


def surface_topology(mesh: TetMesh) -> tuple[np.ndarray, np.ndarray]:
    """Surface vertex indices and the outward surface triangles in that local numbering."""
    tris = mesh.surface_triangles()
    idx, local = np.unique(tris, return_inverse=True)
    return idx, local.reshape(tris.shape)


def area_vectors(points: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Per-vertex outward area vectors (sum of 1/3 of each adjacent triangle's area x normal)."""
    cross = 0.5 * np.cross(points[tris[:, 1]] - points[tris[:, 0]], points[tris[:, 2]] - points[tris[:, 0]])
    vec = np.zeros_like(points)
    for k in range(3):
        np.add.at(vec, tris[:, k], cross / 3.0)
    return vec


@dataclass
class _Counterpart:
    name: str
    adhesion: Adhesion
    parts: tuple[Part, ...] | None  # None: the ground plane
    bonded: np.ndarray


@dataclass(frozen=True)
class RigidFrame:
    """World pose of a rigid body: frame origin, rotation (body -> world), centre of mass."""

    origin: np.ndarray
    rotation: np.ndarray
    com: np.ndarray


@dataclass(frozen=True)
class AdhesionResult:
    vertex_forces: np.ndarray                   # (n_surface, 3), on the soft surface vertices
    wrenches: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]  # name -> force, torque about COM, point


class AdhesionField:
    """Adhesive bonds of one soft body with every counterpart it can stick to."""

    def __init__(self, scene: Scene, soft_name: str):
        body = scene.soft(soft_name)
        self.soft_name = soft_name
        self.vertices, self.triangles = surface_topology(body.mesh)
        self.skin = scene.contact.soft_radius
        n = len(self.vertices)
        self.counterparts: list[_Counterpart] = []
        if body.material.adhesion is not None:
            for rigid in scene.rigid_bodies:
                self.counterparts.append(_Counterpart(rigid.name, body.material.adhesion, rigid.parts,
                                                      np.zeros(n, bool)))
        if scene.ground.enabled and scene.ground.adhesion is not None:
            self.counterparts.append(_Counterpart(GROUND_NAME, scene.ground.adhesion, None, np.zeros(n, bool)))

    @property
    def active(self) -> bool:
        return bool(self.counterparts)

    def reset(self) -> None:
        for c in self.counterparts:
            c.bonded[:] = False

    def bonded(self, name: str) -> np.ndarray:
        """Bond flags of the surface vertices with counterpart ``name``."""
        return next(c for c in self.counterparts if c.name == name).bonded

    def compute(self, positions: np.ndarray, frames: dict[str, RigidFrame]) -> AdhesionResult:
        """Update bonds and return forces for surface-vertex ``positions`` (world, skin centres)."""
        forces = np.zeros((len(self.vertices), 3))
        wrenches = {}
        vectors = area_vectors(positions, self.triangles)
        for c in self.counterparts:
            if c.parts is None:
                dist, normal = positions[:, 2].copy(), np.tile([0.0, 0.0, 1.0], (len(positions), 1))
                frame = None
            else:
                frame = frames[c.name]
                local = (positions - frame.origin) @ frame.rotation
                dist, normal_local = parts_sdf(local, c.parts)
                normal = normal_local @ frame.rotation.T
            gap = dist - self.skin
            c.bonded |= gap <= Adhesion.BOND_GAPS * c.adhesion.range
            c.bonded &= gap < Adhesion.BREAK_GAPS * c.adhesion.range
            facing = np.maximum(-np.einsum("ij,ij->i", vectors, normal), 0.0)  # area facing the counterpart
            pull = np.where(c.bonded, traction(gap, c.adhesion) * facing, 0.0)
            f = pull[:, None] * normal  # force on the counterpart (towards the soft vertex)
            forces -= f
            if frame is not None:
                total = f.sum(axis=0)
                points = positions - dist[:, None] * normal  # closest points on the rigid surface
                torque = np.cross(points - frame.com, f).sum(axis=0)
                weight = pull.sum()
                point = (pull[:, None] * points).sum(axis=0) / weight if weight > 0 else frame.com.copy()
                wrenches[c.name] = (total, torque, point)
            else:
                total = f.sum(axis=0)
                weight = pull.sum()
                point = (pull[:, None] * positions).sum(axis=0) / weight if weight > 0 else np.zeros(3)
                wrenches[c.name] = (total, np.zeros(3), point)
        return AdhesionResult(forces, wrenches)


def adhesion_fields(scene: Scene) -> list[AdhesionField]:
    """Active adhesion fields of a scene (one per soft body that sticks to anything)."""
    fields = [AdhesionField(scene, b.name) for b in scene.soft_bodies]
    return [f for f in fields if f.active]
