"""Signed gap between two surface regions of a body: how far apart two faces are, or how deep one has gone into
the other (negative). The self-collision check (``body.gaps`` of the actuation scenario) measures it between the
two walls of a slot that the body closes on itself."""

from __future__ import annotations

import numpy as np


def closest_on_triangles(p: np.ndarray, tri: np.ndarray) -> np.ndarray:
    """Closest point to each point ``p`` (M, 3) on each triangle ``tri`` (K, 3, 3): (M, K, 3).

    Ericson, Real-Time Collision Detection (2005) 5.1.5, by Voronoi region of the triangle, vectorised."""
    a, b, c = (tri[None, :, k] for k in range(3))
    p = p[:, None]
    ab, ac, ap = b - a, c - a, p - a
    bp, cp = p - b, p - c

    def dot(u, v):
        return np.einsum("...i,...i->...", u, v)

    d1, d2 = dot(ab, ap), dot(ac, ap)
    d3, d4 = dot(ab, bp), dot(ac, bp)
    d5, d6 = dot(ab, cp), dot(ac, cp)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v, w = vb / denom, vc / denom  # inside the face
        out = a + ab * v[..., None] + ac * w[..., None]
        t = d1 / (d1 - d3)  # edge ab
        edge_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
        out = np.where(edge_ab[..., None], a + ab * t[..., None], out)
        t = d2 / (d2 - d6)  # edge ac
        edge_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
        out = np.where(edge_ac[..., None], a + ac * t[..., None], out)
        t = (d4 - d3) / ((d4 - d3) + (d5 - d6))  # edge bc
        edge_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
        out = np.where(edge_bc[..., None], b + (c - b) * t[..., None], out)
    out = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, out)  # vertex regions last: they win
    out = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, out)
    out = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, out)
    return out


def signed_distances(points: np.ndarray, x: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Signed distance [m] of each point (M, 3) from the surface ``tris`` (K, 3; outward normals) at positions
    ``x``: the distance to the nearest triangle, negative on the inner side of that triangle's plane (inside the
    body the surface bounds)."""
    tri = x[tris]
    c = closest_on_triangles(points, tri)
    d = points[:, None] - c
    dist = np.linalg.norm(d, axis=2)
    k = dist.argmin(axis=1)
    rows = np.arange(len(points))
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[k]
    side = np.einsum("ij,ij->i", d[rows, k], n)
    return np.where(side < 0, -dist[rows, k], dist[rows, k])


def region_gap(x: np.ndarray, tris_a: np.ndarray, tris_b: np.ndarray) -> float:
    """Signed gap [m] between surface regions ``tris_a`` and ``tris_b`` (outward normals) at positions ``x``: the
    smallest signed distance of either region's vertices from the other region's surface. Positive: the faces are
    apart by that much; negative: a vertex of one has gone that deep behind the other (interpenetration)."""
    va, vb = np.unique(tris_a), np.unique(tris_b)
    return float(min(signed_distances(x[vb], x, tris_a).min(), signed_distances(x[va], x, tris_b).min()))
