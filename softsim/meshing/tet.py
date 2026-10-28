"""Volumetric tetrahedral meshes for soft bodies.

Two generators:
    * ``structured`` (default, numpy only): deterministic and symmetric, has a
      vertex on the axis of cylinders (handy for measuring top-centre motion).
      Cylinders are built from concentric rings, extruded into prisms and split
      into 3 tetrahedra with a global-index rule that keeps the mesh conforming.
    * ``gmsh`` (optional dependency): unstructured Delaunay meshes, needed for
      arbitrary CAD geometry later.

All meshes put the body's base centre at the origin, with +z up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from softsim.errors import ConfigError


@dataclass(frozen=True)
class TetMesh:
    """Tetrahedral mesh: ``points`` (N, 3) in metres, ``tets`` (M, 4) vertex indices.

    Tetrahedra are oriented with positive signed volume.
    """

    points: np.ndarray
    tets: np.ndarray
    _surface: list = field(default_factory=list, repr=False, compare=False)

    @property
    def n_points(self) -> int:
        return len(self.points)

    @property
    def n_tets(self) -> int:
        return len(self.tets)

    def tet_volumes(self) -> np.ndarray:
        """Signed volume of every tetrahedron."""
        p = self.points[self.tets]
        return np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0]) / 6.0

    def volume(self) -> float:
        return float(self.tet_volumes().sum())

    def lumped_masses(self, density: float) -> np.ndarray:
        """Per-vertex masses from distributing each tet's mass equally to its 4 vertices."""
        m = np.zeros(self.n_points)
        np.add.at(m, self.tets.ravel(), np.repeat(self.tet_volumes() * density / 4.0, 4))
        return m

    def edge_lengths(self) -> np.ndarray:
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        p = self.points
        return np.concatenate([np.linalg.norm(p[self.tets[:, a]] - p[self.tets[:, b]], axis=1) for a, b in pairs])

    def quality(self) -> dict[str, float]:
        """Basic quality stats; ``min_quality`` is the normalised volume/edge^3 ratio (1 = regular tet)."""
        vol = self.tet_volumes()
        p = self.points[self.tets]
        pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        rms = np.sqrt(np.mean([np.sum((p[:, a] - p[:, b]) ** 2, axis=1) for a, b in pairs], axis=0))
        q = 6.0 * math.sqrt(2.0) * vol / rms**3
        edges = self.edge_lengths()
        return {
            "n_points": self.n_points,
            "n_tets": self.n_tets,
            "min_quality": float(q.min()),
            "mean_quality": float(q.mean()),
            "min_edge": float(edges.min()),
            "max_edge": float(edges.max()),
            "mean_edge": float(edges.mean()),
        }

    def surface_triangles(self) -> np.ndarray:
        """Boundary triangles (K, 3) with outward normals."""
        if self._surface:
            return self._surface[0]
        t = self.tets
        faces = np.concatenate([t[:, [0, 2, 1]], t[:, [0, 1, 3]], t[:, [0, 3, 2]], t[:, [1, 2, 3]]])
        key = np.sort(faces, axis=1)
        _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
        tris = faces[counts[inverse.ravel()] == 1]
        self._surface.append(tris)
        return tris

    def vertices_where(self, predicate) -> np.ndarray:
        """Indices of points for which ``predicate(points)`` (vectorised) is true."""
        return np.nonzero(predicate(self.points))[0]


def _orient(points: np.ndarray, tets: np.ndarray) -> np.ndarray:
    """Return ``tets`` with vertex order fixed so every signed volume is positive."""
    p = points[tets]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    if np.any(np.abs(vol) < 1e-18):
        raise ConfigError("mesh contains degenerate (zero-volume) tetrahedra")
    tets = tets.copy()
    neg = vol < 0
    tets[neg, 2], tets[neg, 3] = tets[neg, 3].copy(), tets[neg, 2].copy()
    return tets


def _ring_disk(radius: float, size: float) -> tuple[np.ndarray, np.ndarray]:
    """2D disk triangulation from concentric rings; ring k has ~2*pi*r_k/size points."""
    n_rings = max(1, round(radius / size))
    pts = [np.zeros((1, 2))]
    counts = [1]
    for k in range(1, n_rings + 1):
        r = radius * k / n_rings
        n = max(6, counts[-1], round(2 * math.pi * r / size))
        theta = 2 * math.pi * np.arange(n) / n
        pts.append(np.column_stack([r * np.cos(theta), r * np.sin(theta)]))
        counts.append(n)
    offsets = np.concatenate([[0], np.cumsum(counts)])
    tris = []
    for k in range(1, len(counts)):
        a0, na = offsets[k - 1], counts[k - 1]
        b0, nb = offsets[k], counts[k]
        if na == 1:
            tris += [(a0, b0 + j, b0 + (j + 1) % nb) for j in range(nb)]
            continue
        i = j = 0
        while i < na or j < nb:  # zipper walk in angle order
            next_a = (i + 1) / na if i < na else math.inf
            next_b = (j + 1) / nb if j < nb else math.inf
            if next_a <= next_b:
                tris.append((a0 + i % na, b0 + j % nb, a0 + (i + 1) % na))
                i += 1
            else:
                tris.append((a0 + i % na, b0 + j % nb, b0 + (j + 1) % nb))
                j += 1
    return np.concatenate(pts), np.array(tris)


def _extrude(disk_pts: np.ndarray, tris: np.ndarray, height: float, n_layers: int) -> TetMesh:
    """Extrude a 2D triangulation into prisms, each split into 3 conforming tets."""
    nd = len(disk_pts)
    z = np.linspace(0.0, height, n_layers + 1)
    points = np.concatenate([np.column_stack([disk_pts, np.full(nd, zi)]) for zi in z])
    s = np.sort(tris, axis=1)  # global-index ordering => neighbouring prisms pick the same diagonals
    tets = []
    for layer in range(n_layers):
        b, t = s + layer * nd, s + (layer + 1) * nd
        tets += [
            np.column_stack([b[:, 0], b[:, 1], b[:, 2], t[:, 0]]),
            np.column_stack([b[:, 1], b[:, 2], t[:, 0], t[:, 1]]),
            np.column_stack([b[:, 2], t[:, 0], t[:, 1], t[:, 2]]),
        ]
    tets = np.concatenate(tets)
    return TetMesh(points, _orient(points, tets))


def structured_cylinder(radius: float, height: float, size: float) -> TetMesh:
    """Cylinder mesh with target edge length ``size``; base centre at origin, axis +z."""
    if min(radius, height, size) <= 0:
        raise ConfigError("cylinder radius, height and mesh size must be positive")
    disk, tris = _ring_disk(radius, size)
    return _extrude(disk, tris, height, max(2, round(height / size)))


def structured_box(size_xyz: tuple[float, float, float], size: float) -> TetMesh:
    """Box mesh (full extents ``size_xyz``), base centre at origin; hexes split into 6 Kuhn tets."""
    sx, sy, sz = size_xyz
    if min(sx, sy, sz, size) <= 0:
        raise ConfigError("box extents and mesh size must be positive")
    n = [max(1, round(e / size)) for e in (sx, sy, sz)]
    axes = [np.linspace(-sx / 2, sx / 2, n[0] + 1), np.linspace(-sy / 2, sy / 2, n[1] + 1),
            np.linspace(0.0, sz, n[2] + 1)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    idx = np.arange(len(points)).reshape(n[0] + 1, n[1] + 1, n[2] + 1)
    base = idx[:-1, :-1, :-1].ravel()
    step = np.array([idx[1, 0, 0] - idx[0, 0, 0], idx[0, 1, 0] - idx[0, 0, 0], idx[0, 0, 1] - idx[0, 0, 0]])
    tets = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        v1 = base + step[perm[0]]
        v2 = v1 + step[perm[1]]
        tets.append(np.column_stack([base, v1, v2, base + step.sum()]))
    tets = np.concatenate(tets)
    return TetMesh(points, _orient(points, tets))


def _gmsh_mesh(build, size: float) -> TetMesh:
    try:
        import gmsh
    except ImportError:
        raise ConfigError("mesh method 'gmsh' needs the gmsh package (pip install softsim[mesh])") from None
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        build(gmsh.model.occ)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", size)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size)
        gmsh.model.mesh.generate(3)
        tags, xyz, _ = gmsh.model.mesh.getNodes()
        _, _, node_tags = gmsh.model.mesh.getElements(3, -1)
    finally:
        gmsh.finalize()
    index = {t: i for i, t in enumerate(tags)}
    tets = np.array([index[t] for t in node_tags[0]]).reshape(-1, 4)
    used = np.unique(tets)
    remap = np.full(len(tags), -1)
    remap[used] = np.arange(len(used))
    points = xyz.reshape(-1, 3)[used]
    tets = remap[tets]
    return TetMesh(points, _orient(points, tets))


def gmsh_cylinder(radius: float, height: float, size: float) -> TetMesh:
    """Unstructured cylinder mesh via gmsh."""
    return _gmsh_mesh(lambda occ: occ.addCylinder(0, 0, 0, 0, 0, height, radius), size)


def gmsh_box(size_xyz: tuple[float, float, float], size: float) -> TetMesh:
    """Unstructured box mesh via gmsh."""
    sx, sy, sz = size_xyz
    return _gmsh_mesh(lambda occ: occ.addBox(-sx / 2, -sy / 2, 0, sx, sy, sz), size)
