"""3D finite-strain FEM with quadratic tetrahedra: the reference solver for any geometry.

The formulation of the axisymmetric solver (:mod:`softsim.fem.axisym`), in 3D:

* **Elements:** 10-node quadratic tetrahedra (P2) with one dilatation per
  element (mean dilatation, P2/P0): no volumetric locking up to
  incompressibility. Built from any linear tetrahedral mesh by adding edge
  midpoints (straight-sided), so mesh convergence also converges the geometry.
* **Materials:** the isochoric laws of :mod:`softsim.fem.materials`
  (Neo-Hookean, Mooney-Rivlin, Yeoh, Gent, Arruda-Boyce, and Ogden through
  matrix functions, smooth where principal stretches coincide) + Simo-Taylor
  volumetric energy, Prony series on the isochoric part.
* **Time integration:** Newmark (gamma = 1/2 + a, beta = (1 + a)^2 / 4) on the
  incremental potential, or quasi-static. Newton with exact JAX tangents, energy
  line search, PSD-projected fallback and step halving. A folded quadrature
  point has infinite energy.
* **Rigid bodies:** ``free`` (centre-of-mass displacement and an incremental
  rotation vector per step; the gyroscopic term is neglected, which is exact
  for axisymmetric bodies spinning about their axis and of second order in the
  step otherwise), ``slider_z`` (vertical, on a spring and damper),
  ``fixed`` or ``prescribed`` (a displacement function of time). Shapes are
  unions of cylinders and boxes with rounded edges (exact signed distances); a
  body without parts is the ground half-space z <= 0.
* **Contact:** penalty on every surface node with its lumped area. As in the
  axisymmetric solver it is node-based: a uniform pressure on a quadratic face
  gives its corner nodes zero load, so quadrature-point contact would let
  rigid edges pierce them. Frictionless, sticking or Coulomb (lagged normal
  pressure, smoothed), with optional cohesive adhesion
  (:class:`softsim.fem.axisym.ContactSettings`).

Units are SI. Node displacements ``u`` (N, 3); the first ``n_corner`` nodes are
the vertices of the linear mesh the quadratic one was built from.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from softsim.fem import jax_setup
from softsim.fem.axisym import ContactSettings
from softsim.fem.materials import (IsochoricLaw, InvariantLaw, OgdenGeneralIso, OgdenIso, Prony, det3,
                                   volumetric_energy)
from softsim.geometry.rigid import Part

jax, jnp = jax_setup()

# -- reference element --------------------------------------------------------------------------

EDGES = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))  # mid-edge nodes 4..9 of the 10-node tetrahedron
_FACES = ((1, 2, 3), (0, 2, 3), (0, 1, 3), (0, 1, 2))   # face opposite each vertex


def tet_quadrature(n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Collapsed (Duffy) Gauss-Jacobi rule on the reference tetrahedron: n^3 points, positive weights
    (summing to 1/6), exact for polynomials of degree 2n - 1."""
    from scipy.special import roots_jacobi

    def rule(alpha):
        x, w = roots_jacobi(n, alpha, 0.0)
        return (1 + x) / 2, w / 2 ** (alpha + 1)

    (a, wa), (b, wb), (c, wc) = rule(2.0), rule(1.0), rule(0.0)
    pts, wts = [], []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                xi1 = a[i]
                xi2 = b[j] * (1 - a[i])
                xi3 = c[k] * (1 - a[i]) * (1 - b[j])
                pts.append((xi1, xi2, xi3))
                wts.append(wa[i] * wb[j] * wc[k])
    return np.array(pts), np.array(wts)


def tet10_shape(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Shape functions (Q, 10) and their derivatives (Q, 10, 3) at reference points ``xi`` (Q, 3)."""
    xi = np.atleast_2d(xi)
    lam = np.column_stack([1 - xi.sum(axis=1), xi])                       # barycentric L0..L3
    dlam = np.array([[-1.0, -1.0, -1.0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])  # dL_i / dxi
    n = np.zeros((len(xi), 10))
    dn = np.zeros((len(xi), 10, 3))
    for i in range(4):
        n[:, i] = lam[:, i] * (2 * lam[:, i] - 1)
        dn[:, i] = (4 * lam[:, i] - 1)[:, None] * dlam[i]
    for e, (i, j) in enumerate(EDGES):
        n[:, 4 + e] = 4 * lam[:, i] * lam[:, j]
        dn[:, 4 + e] = 4 * (lam[:, j, None] * dlam[i] + lam[:, i, None] * dlam[j])
    return n, dn


_XI, _W = tet_quadrature(2)          # stiffness: 8 points, exact to degree 3 (P2 needs degree 2)
_N, _DN = tet10_shape(_XI)
# 4-point rule (degree 2, as Abaqus C3D10): exact for small-strain P2 stiffness on straight tets
_A4, _B4 = 0.5854101966249685, 0.1381966011250105
_XI4 = np.array([[_B4, _B4, _B4], [_A4, _B4, _B4], [_B4, _A4, _B4], [_B4, _B4, _A4]])
_W4 = np.full(4, 1.0 / 24.0)
_STIFFNESS_RULES = {8: (_W, _DN), 4: (_W4, tet10_shape(_XI4)[1])}
ELEMENT_CHUNK = 25_000  # elements per kernel launch (see ``Solid3DSolver._chunked``)
_XI_M, _W_M = tet_quadrature(3)      # mass: 27 points, exact to degree 5 (P2 x P2 is degree 4)
_N_M, _DN_M = tet10_shape(_XI_M)

# degree-4 rule on the reference triangle (Dunavant, 6 points; weights sum to its area 1/2): exact for
# x . (x_xi x x_eta) on 6-node faces, so the enclosed volume and its derivatives are exact
_TRI_A, _TRI_B = 0.445948490915965, 0.091576213509771
_TRI_XI = np.array([[_TRI_A, _TRI_A], [_TRI_A, 1 - 2 * _TRI_A], [1 - 2 * _TRI_A, _TRI_A],
                    [_TRI_B, _TRI_B], [_TRI_B, 1 - 2 * _TRI_B], [1 - 2 * _TRI_B, _TRI_B]])
_TRI_W = 0.5 * np.array([0.223381589678011] * 3 + [0.109951743655322] * 3)


def tri6_shape(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """6-node triangle (corners a, b, c, then mid-edges ab, bc, ca) shape functions (Q, 6) and their
    (xi, eta) derivatives (Q, 6, 2), with xi along a -> b and eta along a -> c."""
    x, y = xi[:, 0], xi[:, 1]
    l1, l2, l3 = 1 - x - y, x, y
    n = np.stack([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1), 4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1], 1)
    d1 = np.stack([-(4 * l1 - 1), 4 * l2 - 1, 0 * x, 4 * (l1 - l2), 4 * l3, -4 * l3], 1)
    d2 = np.stack([-(4 * l1 - 1), 0 * x, 4 * l3 - 1, -4 * l2, 4 * l2, 4 * (l1 - l3)], 1)
    return n, np.stack([d1, d2], -1)


_TRI_N, _TRI_DN = tri6_shape(_TRI_XI)


@dataclass
class QuadraticTetMesh:
    """10-node tetrahedral mesh: ``nodes`` (N, 3), ``tets`` (E, 10), boundary faces ``faces`` (F, 6:
    three corners, then the mid-edge nodes of edges 01, 12, 20); the first ``n_corner`` nodes are the
    linear mesh's vertices."""

    nodes: np.ndarray
    tets: np.ndarray
    faces: np.ndarray
    n_corner: int

    @classmethod
    def from_linear(cls, points: np.ndarray, tets: np.ndarray) -> "QuadraticTetMesh":
        points = np.asarray(points, float)
        tets = np.array(tets, int)
        d = points[tets[:, 1:]] - points[tets[:, [0]]]
        flip = np.linalg.det(d) < 0
        tets[flip, 1], tets[flip, 2] = tets[flip, 2].copy(), tets[flip, 1].copy()  # positive orientation
        mid: dict[tuple[int, int], int] = {}
        nodes = [p for p in points]

        def midpoint(i: int, j: int) -> int:
            key = (min(i, j), max(i, j))
            if key not in mid:
                mid[key] = len(nodes)
                nodes.append(0.5 * (points[i] + points[j]))
            return mid[key]

        t10 = np.array([[*t, *(midpoint(t[i], t[j]) for i, j in EDGES)] for t in tets])
        counts: dict[tuple[int, ...], list] = {}
        for t in tets:
            for opposite, f in enumerate(_FACES):
                corners = tuple(int(t[k]) for k in f)
                counts.setdefault(tuple(sorted(corners)), []).append((corners, int(t[opposite])))
        faces = []
        for occ in counts.values():
            if len(occ) != 1:
                continue
            (a, b, c), o = occ[0]
            if np.dot(np.cross(points[b] - points[a], points[c] - points[a]), points[a] - points[o]) < 0:
                b, c = c, b  # outward normal (b - a) x (c - a)
            faces.append([a, b, c, mid[(min(a, b), max(a, b))], mid[(min(b, c), max(b, c))],
                          mid[(min(a, c), max(a, c))]])
        return cls(np.array(nodes), t10, np.array(faces, int), len(points))

    def faces_of(self, tris: np.ndarray) -> np.ndarray:
        """Rows of ``faces`` (outward) for the linear boundary triangles ``tris`` (corner indices)."""
        index = {tuple(sorted(f[:3])): i for i, f in enumerate(self.faces.tolist())}
        try:
            return self.faces[[index[tuple(sorted(t))] for t in np.asarray(tris).tolist()]]
        except KeyError as exc:
            raise ValueError(f"triangle {exc.args[0]} is not a boundary face of the mesh") from None

    def surface_nodes_and_areas(self) -> tuple[np.ndarray, np.ndarray]:
        """Boundary nodes and their lumped areas: each quadratic face is split into 4 flat triangles
        through its mid-edge nodes, whose areas go a third to each of their nodes (all positive)."""
        f = self.faces
        subs = np.concatenate([f[:, [0, 3, 5]], f[:, [1, 4, 3]], f[:, [2, 5, 4]], f[:, [3, 4, 5]]])
        p = self.nodes[subs]
        area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        acc = np.zeros(len(self.nodes))
        np.add.at(acc, subs.ravel(), np.repeat(area / 3.0, 3))
        idx = np.nonzero(acc > 0)[0]
        return idx, acc[idx]


_EPS = np.zeros((3, 3, 3))
_EPS[0, 1, 2] = _EPS[1, 2, 0] = _EPS[2, 0, 1] = 1.0
_EPS[0, 2, 1] = _EPS[2, 1, 0] = _EPS[1, 0, 2] = -1.0
_II = np.einsum("ik,JL->iJkL", np.eye(3), np.eye(3))  # d F_iJ / d F_kL


def cof3(f):
    """dJ/dF = cof F = J F^-T: its columns are cross products of F's columns."""
    c0, c1, c2 = f[:, 0], f[:, 1], f[:, 2]
    return jnp.stack([jnp.cross(c1, c2), jnp.cross(c2, c0), jnp.cross(c0, c1)], axis=1)


def d2det3(f):
    """d^2 J / dF_iJ dF_kL = eps_ikm eps_JLN F_mN (no division by J)."""
    return jnp.einsum("ikm,JLN,mN->iJkL", _EPS, _EPS, f)


def invariant_tangent(law, f, bq, c_iso):
    """First Piola stress and dP/dF of c_iso psi(I1_bar, I2_bar) + 1/2 bq : C in closed form; only the
    scalar psi is differentiated (~10x faster than ``jax.hessian`` of the 3x3 energy)."""
    c = f.T @ f
    i1 = jnp.trace(c)
    j = det3(f)
    dj, d2j = cof3(f), d2det3(f)

    def isochoric(p, inv, d_inv, d2_inv):  # J^p I and its first two F-derivatives
        jp, jp1 = j ** p, p * j ** (p - 1.0)
        g = jp * d_inv + jp1 * inv * dj
        h = (jp * d2_inv + jp1 * (jnp.einsum("iJ,kL->iJkL", d_inv, dj) + jnp.einsum("iJ,kL->iJkL", dj, d_inv))
             + p * (p - 1.0) * j ** (p - 2.0) * inv * jnp.einsum("iJ,kL->iJkL", dj, dj) + jp1 * inv * d2j)
        return jp * inv, g, h

    ib1, g1, h1 = isochoric(-2.0 / 3.0, i1, 2.0 * f, 2.0 * _II)
    if law.uses_i2:
        i2 = 0.5 * (i1 * i1 - jnp.sum(c * c))
        b = f @ f.T
        d2_i2 = 2.0 * (2.0 * jnp.einsum("iJ,kL->iJkL", f, f) + i1 * _II - jnp.einsum("ik,JL->iJkL", jnp.eye(3), c)
                       - jnp.einsum("iL,kJ->iJkL", f, f) - jnp.einsum("ik,JL->iJkL", b, jnp.eye(3)))
        ib2, g2, h2 = isochoric(-4.0 / 3.0, i2, 2.0 * (i1 * f - f @ c), d2_i2)
    else:
        ib2, g2, h2 = 3.0, jnp.zeros((3, 3)), jnp.zeros((3, 3, 3, 3))
    x = jnp.stack([ib1, jnp.asarray(ib2, ib1.dtype)])
    d = jax.grad(lambda x: law.invariant_energy(x[0], x[1]))(x)
    dd = jax.hessian(lambda x: law.invariant_energy(x[0], x[1]))(x)
    bs = 0.5 * (bq + bq.T)
    pk = c_iso * (d[0] * g1 + d[1] * g2) + f @ bs
    a4 = c_iso * (dd[0, 0] * jnp.einsum("iJ,kL->iJkL", g1, g1) + dd[1, 1] * jnp.einsum("iJ,kL->iJkL", g2, g2)
                  + dd[0, 1] * (jnp.einsum("iJ,kL->iJkL", g1, g2) + jnp.einsum("iJ,kL->iJkL", g2, g1))
                  + d[0] * h1 + d[1] * h2) + jnp.einsum("ik,JL->iJkL", jnp.eye(3), bs)
    return pk, a4


class _UpperPattern:
    """Fixed sparsity pattern of the Hessian's free-free upper triangle (CSR, sorted).

    Built once from every dof group that can ever couple (elements, contact
    candidates, rigid blocks), so each Newton iteration only scatters values
    (``bincount``) instead of building, sorting and slicing a new sparse matrix,
    and PARDISO's symbolic analysis (reordering) is done once per solver.
    """

    def __init__(self, ndof: int, free: np.ndarray, groups: list[np.ndarray]):
        self.nf = len(free)
        self._fidx = np.full(ndof, -1, np.int64)
        self._fidx[free] = np.arange(self.nf)
        keys = [np.arange(self.nf, dtype=np.int64) * (self.nf + 1)]  # the diagonal
        for g in groups:
            r, c = np.repeat(g, g.shape[1], axis=1).ravel(), np.tile(g, (1, g.shape[1])).ravel()
            keys.append(self._keys(r, c)[1])
        self.keys = np.unique(np.concatenate(keys))
        rows = self.keys // self.nf
        self.indices = (self.keys % self.nf).astype(np.int32)
        self.indptr = np.searchsorted(rows, np.arange(self.nf + 1)).astype(np.int32)
        self.diag = np.searchsorted(self.keys, np.arange(self.nf, dtype=np.int64) * (self.nf + 1))
        self._solver = None

    @property
    def nnz(self) -> int:
        return len(self.keys)

    def _keys(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        fr, fc = self._fidx[rows], self._fidx[cols]
        sel = np.nonzero((fr >= 0) & (fc >= 0) & (fr <= fc))[0]
        return sel, fr[sel] * self.nf + fc[sel]

    def locate(self, rows: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For global (row, col) entries: which ones are kept (free-free upper) and their data positions."""
        sel, keys = self._keys(np.ravel(rows), np.ravel(cols))
        pos = np.searchsorted(self.keys, keys)
        if len(pos) and (pos.max() >= self.nnz or np.any(self.keys[pos] != keys)):
            raise ValueError("Hessian entry outside the precomputed pattern")
        return sel, pos

    def positions(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Data position of every (row, col) entry, ``nnz`` (a dropped bin) where it is not kept; int32."""
        sel, pos = self.locate(rows, cols)
        out = np.full(np.size(rows), self.nnz, np.int32)
        out[sel] = pos
        return out

    def scatter(self, pos: np.ndarray, vals: np.ndarray) -> np.ndarray:
        return np.bincount(pos, weights=vals, minlength=self.nnz + 1)[:self.nnz]

    def matrix(self, data: np.ndarray) -> sp.csr_matrix:
        m = sp.csr_matrix((data, self.indices, self.indptr), shape=(self.nf, self.nf))
        m.has_sorted_indices = True
        return m

    def __del__(self):
        if self._solver:
            try:
                self._solver.free_memory(everything=True)
            except Exception:  # noqa: BLE001 - interpreter shutdown
                pass

    def solve(self, data: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Solve with the symmetric matrix whose upper triangle is ``data``: MKL PARDISO (symmetric
        indefinite, analysis reused) when ``pypardiso`` is installed, else SuperLU; NaN where both fail."""
        u = self.matrix(data)
        if self._solver is None:
            try:
                import pypardiso
                self._solver = pypardiso.PyPardisoSolver(mtype=-2)  # real symmetric indefinite
                self._solver.set_phase(11)
                self._solver._call_pardiso(u, np.zeros_like(b))
            except ImportError:
                self._solver = False
            except Exception:  # noqa: BLE001 - analysis failed: SuperLU from now on
                self._solver = False
        if self._solver:
            try:
                self._solver.set_phase(23)  # numerical factorisation + solve, symbolic analysis kept
                x = self._solver._call_pardiso(u, np.asarray(b, float))
                if np.all(np.isfinite(x)):
                    return x
            except Exception:  # noqa: BLE001 - singular or numerically broken: fall back
                pass
        full = (u + sp.triu(u, k=1).T).tocsc()
        try:
            return spla.splu(full, permc_spec="MMD_AT_PLUS_A", diag_pivot_thresh=0.0,
                             options={"SymmetricMode": True}).solve(b)
        except RuntimeError:
            try:
                return spla.splu(full).solve(b)
            except RuntimeError:
                return np.full_like(b, np.nan)


# -- rigid bodies -------------------------------------------------------------------------------

@dataclass(frozen=True)
class RigidBody3D:
    """A rigid body for the 3D solver (see module docstring for ``mount``).

    ``parts`` are in the body frame (origin at the bottom centre, as in
    :mod:`softsim.geometry.rigid`); no parts = the ground half-space z <= 0.
    ``position`` / ``quaternion`` (w, x, y, z) place the body frame; ``velocity``
    is the centre of mass's, ``angular_velocity`` in the world frame.
    ``motion(t)`` is the displacement of a ``prescribed`` body.
    """

    name: str
    parts: tuple[Part, ...] = ()
    mass: float = 0.0
    inertia: tuple[tuple[float, ...], ...] = ((0, 0, 0), (0, 0, 0), (0, 0, 0))  # about the COM, body axes
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)                           # body frame
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    quaternion: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)
    mount: str = "free"
    stiffness: float = 0.0
    damping: float = 0.0
    motion: Callable[[float], np.ndarray] | None = None
    edge_radius: float = 2e-4
    contact: ContactSettings = field(default_factory=ContactSettings)

    def __post_init__(self) -> None:
        if self.mount not in ("free", "slider_z", "fixed", "prescribed"):
            raise ValueError(f"unknown mount {self.mount!r}")
        if not self.parts and self.mount != "fixed":
            raise ValueError(f"{self.name}: a body without parts is the ground half-space and must be fixed")
        if self.mount in ("free", "slider_z") and self.mass <= 0:
            raise ValueError(f"{self.name}: a {self.mount} body needs a positive mass")
        if self.mount == "prescribed" and self.motion is None:
            raise ValueError(f"{self.name}: a prescribed body needs motion(t)")

    @classmethod
    def ground(cls, contact: ContactSettings | None = None) -> "RigidBody3D":
        """The half-space z <= 0."""
        return cls("ground", mount="fixed", contact=contact or ContactSettings())

    @property
    def n_dofs(self) -> int:
        return {"free": 6, "slider_z": 1, "fixed": 0, "prescribed": 0}[self.mount]


def quat_to_matrix(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def _expm_so3(theta):
    """Rotation matrix of a rotation vector (Rodrigues), smooth through zero."""
    a2 = jnp.dot(theta, theta)
    a = jnp.sqrt(a2 + 1e-30)
    small = a2 < 1e-8
    s = jnp.where(small, 1.0 - a2 / 6.0, jnp.sin(a) / a)
    c = jnp.where(small, 0.5 - a2 / 24.0, (1.0 - jnp.cos(a)) / (a * a))
    k = jnp.array([[0.0, -theta[2], theta[1]], [theta[2], 0.0, -theta[0]], [-theta[1], theta[0], 0.0]])
    return jnp.eye(3) + s * k + c * (k @ k)


def expm_so3(theta) -> np.ndarray:
    return np.asarray(_expm_so3(jnp.asarray(theta, dtype=float)))


def _sdf_function(body: RigidBody3D):
    """Signed distance in the body's centre-of-mass frame (body axes), JAX-traceable."""
    if not body.parts:
        return lambda p: p[2]
    com = np.asarray(body.com, float)
    eps = 1e-30
    pieces = []
    for part in body.parts:
        c = np.asarray(part.center, float) - com
        if part.kind == "cylinder":
            r, hh = part.size[0], part.size[1]
            rho = min(body.edge_radius, 0.5 * r, hh)

            def cyl(p, c=c, r=r, hh=hh, rho=rho):
                q = p - c
                dr = jnp.sqrt(q[0] ** 2 + q[1] ** 2 + eps) - (r - rho)
                dz = jnp.sqrt(q[2] ** 2 + eps) - (hh - rho)
                outside = jnp.sqrt(jnp.maximum(dr, 0.0) ** 2 + jnp.maximum(dz, 0.0) ** 2 + eps)
                return outside + jnp.minimum(jnp.maximum(dr, dz), 0.0) - rho
            pieces.append(cyl)
        else:
            half = np.asarray(part.size, float)
            rho = min(body.edge_radius, *half)

            def box(p, c=c, half=half, rho=rho):
                q = jnp.sqrt((p - c) ** 2 + eps) - (half - rho)
                outside = jnp.sqrt(jnp.sum(jnp.maximum(q, 0.0) ** 2) + eps)
                return outside + jnp.minimum(jnp.max(q), 0.0) - rho
            pieces.append(box)

    def sdf(p):
        d = pieces[0](p)
        for f in pieces[1:]:
            d = jnp.minimum(d, f(p))
        return d
    return sdf


# -- problem ------------------------------------------------------------------------------------

@dataclass
class Problem3D:
    """One soft body, rigid bodies (the ground is a rigid body without parts) and loads.

    Pressure loads: ``pressure_faces[k]`` carries the gauge pressure ``Solver.pressures[k]`` (set it
    before each step) as the potential ``p S(x)``, ``S = 1/3 int x . n dA`` over the faces = minus the
    volume they enclose: the exact follower load ``-p n`` for closed regions, and for regions closed
    by symmetry planes through the origin (where ``x . n = 0``).

    Cable loads: ``cables[k] = (nodes, weights)``, (K, W) each: via point ``i`` sits at
    ``sum_j weights[i, j] x[nodes[i, j]]`` (the quadratic shape functions of its element, W = 10; or for a
    cable anchored over a region its last row spreads over the region's corner nodes, W larger, short rows
    padded with zero weights); the cable carries the tension ``Solver.tensions[k]`` as the potential
    ``T L(x)``, ``L`` the length of the polyline through the via points. A cable with
    ``cable_stiffness[k]`` = k > 0 is stroke-controlled instead: a tendon of stiffness k and free
    length ``Solver.free_lengths[k]`` (its rest length minus the motor's stroke), potential
    ``k/2 (L - L_free)^2`` while taut (``L > L_free``), tension ``k (L - L_free)``."""

    mesh: QuadraticTetMesh
    law: IsochoricLaw
    bulk_modulus: float
    density: float | np.ndarray                 # [kg/m^3], or one per element (material zones)
    prony: Prony | None = None
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    rigid: tuple[RigidBody3D, ...] = ()
    pinned: np.ndarray = field(default_factory=lambda: np.zeros(0, int))   # node indices fixed in place
    fixed_dofs: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # single dofs (3 * node + axis)
    pressure_faces: tuple[np.ndarray, ...] = ()  # per pressure load: outward faces (K, 6); ``Solid3DSolver.pressures``
    cables: tuple[tuple[np.ndarray, np.ndarray], ...] = ()  # per cable: via-point (nodes, weights); ``.tensions``
    cable_stiffness: tuple[float, ...] = ()  # per cable: 0 = tension-controlled, k > 0 = stroke-controlled
    force_loads: tuple[tuple[np.ndarray, np.ndarray], ...] = ()  # per region force: (faces (K, 6), direction (3,));
    #   ``Solid3DSolver.forces`` [N]: uniform traction, consistent (mid-edge nodes, a third of the face area each)
    zones: tuple[tuple[np.ndarray, IsochoricLaw, float], ...] = ()  # (elements, law, bulk modulus) replacing ``law``
    zone_pronys: tuple[Prony | None, ...] = ()  # each zone's own series (None: undamped); empty: the body's ``prony``


class Solid3DSolver:
    """Newton solver for a :class:`Problem3D`; ``advance(dt)`` takes one implicit step (halving if needed)."""

    def __init__(self, problem: Problem3D, dynamic: bool = True, numerical_damping: float = 0.25,
                 viscous: bool | None = None, quadrature: int = 8):
        """``dynamic=False`` solves quasi-statically (no inertia); the Prony history is then used only
        with ``viscous=True`` (quasi-static viscoelasticity, e.g. relaxation tests), otherwise the
        material is fully relaxed. By default ``viscous = dynamic``. ``quadrature``: stiffness points
        per element, 8 (degree 3) or 4 (degree 2, about twice as fast)."""
        if quadrature not in _STIFFNESS_RULES:
            raise ValueError(f"quadrature must be one of {sorted(_STIFFNESS_RULES)}, got {quadrature}")
        w_q, dn_q = _STIFFNESS_RULES[quadrature]
        def general(law):  # the axisymmetric Ogden form assumes a block-diagonal C
            return OgdenGeneralIso(law.mu, law.alpha) if isinstance(law, OgdenIso) else law

        problem = Problem3D(**{**problem.__dict__, "law": general(problem.law),
                               "zones": tuple((np.asarray(e, int), general(law), k) for e, law, k in problem.zones)})
        self.p = problem
        self.dynamic = dynamic
        self.viscous = dynamic if viscous is None else viscous
        self.gamma = 0.5 + numerical_damping
        self.beta = 0.25 * (1.0 + numerical_damping) ** 2
        m = problem.mesh
        self.n = len(m.nodes)
        x = m.nodes[m.tets]                                                     # (E, 10, 3)
        jac = np.einsum("eni,qnj->eqij", x, dn_q)                               # (E, Q, 3, 3)
        det = np.linalg.det(jac)
        if np.any(det <= 0):
            raise ValueError("mesh has inverted or degenerate tetrahedra")
        self._dndx = np.einsum("qnk,eqkj->eqnj", dn_q, np.linalg.inv(jac))     # (E, Q, 10, 3)
        self._wv = det * w_q                                                    # (E, Q)
        self.volume = float(self._wv.sum())

        det_m = np.linalg.det(np.einsum("eni,qnj->eqij", x, _DN_M))
        nn = np.einsum("qa,qb->qab", _N_M, _N_M)
        rho = np.broadcast_to(np.asarray(problem.density, float), (len(m.tets),))
        me = rho[:, None, None] * np.einsum("eq,qab->eab", det_m * _W_M, nn)
        rows = np.repeat(m.tets, 10, axis=1).ravel()
        cols = np.tile(m.tets, (1, 10)).ravel()
        mass = sp.csr_matrix((me.ravel(), (rows, cols)), shape=(self.n, self.n))
        self.mass_matrix = sp.kron(mass, sp.eye(3), format="csr")
        self._node_mass = np.asarray(mass.sum(axis=1)).ravel()                # consistent nodal masses
        self.total_mass = float(rho @ self._wv.sum(axis=1))

        # dof layout: 3 per node, then each rigid body's dofs
        self._rdof = []
        off = 3 * self.n
        for body in problem.rigid:
            self._rdof.append(np.arange(off, off + body.n_dofs))
            off += body.n_dofs
        self.ndof = off
        fixed = np.zeros(self.ndof, bool)
        pinned = np.asarray(problem.pinned, int)
        fixed[(3 * pinned[:, None] + np.arange(3)).ravel()] = True
        fixed[np.asarray(problem.fixed_dofs, int)] = True
        self.fixed = fixed
        self.free = np.nonzero(~fixed)[0]
        self._edofs = (3 * m.tets[:, :, None] + np.arange(3)).reshape(len(m.tets), 30)

        # contact nodes (all surface nodes, against every rigid body)
        self._snodes, self._sarea = m.surface_nodes_and_areas()
        self._sdofs = 3 * self._snodes[:, None] + np.arange(3)
        nb = len(problem.rigid)
        k = len(self._snodes)
        self._bond = [np.zeros(k, bool) for _ in range(nb)]
        self._anchor = [np.zeros((k, 3)) for _ in range(nb)]
        self._lam = [np.zeros(k) for _ in range(nb)]
        groups = [self._edofs]
        self.pressures = np.zeros(len(problem.pressure_faces))
        self.forces = np.zeros(len(problem.force_loads))
        self.gravity = np.asarray(problem.gravity, float)  # changeable between steps (a moving frame)
        self.fixed_motion = None  # optional t -> values of the fixed dofs (``self.fixed``): prescribed motion
        # gradient of the last accepted step's incremental potential, all dofs: zero at free dofs, and at
        # held ones the force their support exerts on the body (internal + inertial - external) [N]
        self.residual = np.zeros(self.ndof)
        self._force_vecs = []  # per region force: its nodal loads for 1 N total
        for faces, direction in problem.force_loads:
            c = m.nodes[faces[:, :3]]
            area = 0.5 * np.linalg.norm(np.cross(c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]), axis=1)
            share = np.zeros(self.n)
            np.add.at(share, faces[:, 3:].ravel(), np.repeat(area / 3, 3))
            self._force_vecs.append((share[:, None] / area.sum() * np.asarray(direction, float)).ravel())
        self._pdofs = [(3 * np.asarray(f, int)[:, :, None] + np.arange(3)).reshape(len(f), 18)
                       for f in problem.pressure_faces]
        self.tensions = np.zeros(len(problem.cables))
        self._cables = []  # per cable: segment dofs (S, 6W), signed segment weights (S, 2W), rest positions
        for nodes, weights in problem.cables:
            seg_nodes = np.concatenate([nodes[:-1], nodes[1:]], axis=1)                 # (S, 2W)
            seg_dofs = (3 * seg_nodes[:, :, None] + np.arange(3)).reshape(len(seg_nodes), -1)
            self._cables.append((seg_dofs, np.concatenate([-weights[:-1], weights[1:]], axis=1),
                                 m.nodes[seg_nodes]))
            groups.append(seg_dofs)
        self.stiffness = np.zeros(len(problem.cables))
        self.stiffness[:len(problem.cable_stiffness)] = problem.cable_stiffness
        self._cable_dofs = {}  # stroke-controlled cables: every dof of the cable (the k dL dL^T term couples them)
        for k in np.nonzero(self.stiffness)[0]:
            dofs = np.unique(self._cables[k][0])
            self._cable_dofs[k] = dofs
            groups.append(dofs[None, :])
        for i in range(nb):
            rigid = np.broadcast_to(self._rdof[i], (k, len(self._rdof[i])))
            groups.append(np.concatenate([self._sdofs, rigid], axis=1))
            if len(self._rdof[i]):
                groups.append(self._rdof[i][None, :])
        self._pattern = _UpperPattern(self.ndof, self.free, groups)
        self._epos = self._pattern.positions(np.repeat(self._edofs, 30, axis=1), np.tile(self._edofs, (1, 30)))
        self._ppos = [self._pattern.positions(np.repeat(d, 18, axis=1), np.tile(d, (1, 18))) for d in self._pdofs]
        self._cpos = [self._pattern.positions(np.repeat(d, d.shape[1], axis=1), np.tile(d, (1, d.shape[1])))
                      for d, *_ in self._cables]
        self._cdense = {k: self._pattern.positions(np.repeat(d[None, :], len(d), axis=1),
                                                   np.tile(d[None, :], (1, len(d))))
                        for k, d in self._cable_dofs.items()}
        self.rest_lengths = np.array([self.cable_length(k, np.zeros(self.ndof)) for k in range(len(self._cables))])
        self.free_lengths = self.rest_lengths.copy()  # stroke-controlled cables: rest length - stroke
        mass = self.mass_matrix.tocoo()
        msel, self._mpos = self._pattern.locate(mass.row, mass.col)
        self._mval = mass.data[msel]

        # rigid state: COM position, rotation, velocities/accelerations
        self._c0 = []
        self.R, self.omega, self.alpha = [], [], []
        for body in problem.rigid:
            r0 = quat_to_matrix(body.quaternion)
            self._c0.append(np.asarray(body.position, float) + r0 @ np.asarray(body.com, float))
            self.R.append(r0)
            self.omega.append(np.asarray(body.angular_velocity, float))
            self.alpha.append(np.zeros(3))
        self._inertia = [np.asarray(b.inertia, float) for b in problem.rigid]

        # viscous memory per material group (its elements, its Prony series): terms q_i and the last stress
        self._visc = self._viscous_groups()
        self._q = [np.zeros((len(pr.g) if pr else 0, len(idx), *self._wv.shape[1:], 3, 3)) for idx, pr in self._visc]
        self._s_prev = [np.zeros((len(idx), *self._wv.shape[1:], 3, 3)) for idx, _ in self._visc]

        self.y = np.zeros(self.ndof)    # u, then rigid dofs (free: d, theta (theta = 0 between steps); slider: z)
        self.v = np.zeros(self.ndof)
        self.a = np.zeros(self.ndof)
        for body, dofs in zip(problem.rigid, self._rdof):
            if body.mount == "free":
                self.v[dofs[:3]] = body.velocity
                self.v[dofs[3:]] = body.angular_velocity
                self.a[dofs[:3]] = problem.gravity  # bodies start in free flight (consistent initial acceleration)
            elif body.mount == "slider_z":
                self.v[dofs] = body.velocity[2]
        self.t = 0.0
        self._build_kernels()

    # -- kernels (JAX) -----------------------------------------------------------------------------

    def _viscous_groups(self) -> list[tuple[np.ndarray, Prony | None]]:
        """(elements, Prony series) per material group, in the order of the element kernels: the body's own
        elements, then each zone (its own series, or the body's when the problem gives none)."""
        p = self.p
        rest = np.ones(len(p.mesh.tets), bool)
        for elements, _, _ in p.zones:
            rest[elements] = False
        series = p.zone_pronys or (p.prony,) * len(p.zones)
        groups = [(np.nonzero(rest)[0], p.prony), *((np.asarray(e, int), s) for (e, _, _), s in zip(p.zones, series))]
        return [g for g in groups if len(g[0])]

    def _build_kernels(self) -> None:
        """Element kernels per material group (the body's law, then each zone's), behind dispatchers
        that keep the single-material call signatures."""
        p = self.p
        rest = np.ones(len(p.mesh.tets), bool)
        for elements, _, _ in p.zones:
            rest[elements] = False
        groups = [(slice(None) if rest.all() else np.nonzero(rest)[0], p.law, p.bulk_modulus),
                  *((e, law, kappa) for e, law, kappa in p.zones)]
        kernels = [(idx, self._chunked(self._group_kernels(law, kappa))) for idx, law, kappa in groups
                   if np.size(rest[idx])]
        n_elem = len(p.mesh.tets)

        def elem_grad_hess(u, dndx, wv, c, b):
            if len(kernels) == 1:
                return kernels[0][1][1](u, dndx, wv, c, b)
            ge, he = np.zeros((n_elem, 30)), np.zeros((n_elem, 30, 30))
            for j, (idx, k) in enumerate(kernels):
                g, h = k[1](u[idx], dndx[idx], wv[idx], c[j] if isinstance(c, tuple) else c, b[idx])
                ge[idx], he[idx] = g, h
            return ge, he

        def elem_energy(u, c, b):
            return sum(k[0](u[idx], self._dndx_j[idx], self._wv_j[idx], c[j] if isinstance(c, tuple) else c, b[idx])
                       for j, (idx, k) in enumerate(kernels))

        def iso_stress(u, dndx):
            if len(kernels) == 1:
                return kernels[0][1][2](u, dndx)
            out = np.zeros((*self._wv.shape, 3, 3))
            for idx, k in kernels:
                out[idx] = k[2](u[idx], dndx[idx])
            return out

        self._dndx_j, self._wv_j = jnp.asarray(self._dndx), jnp.asarray(self._wv)
        self._elem_grad_hess, self._elem_energy, self._iso_stress = elem_grad_hess, elem_energy, iso_stress
        self._build_load_kernels()

    @staticmethod
    def _chunked(kernels):
        """The three kernels of :meth:`_group_kernels` evaluated at most ``ELEMENT_CHUNK`` elements at a time
        (env ``SOFTSIM_FEM_CHUNK``, default 25 000): the vmapped Hessians are 7 kB an element, so one
        launch over 100 000 elements asks the GPU for 4.8 GB and a 12 GB card runs out at about 0.8 mm on
        the PneuNet finger. Within the limit the call is unchanged."""
        energy, grad_hess, stress = kernels
        size = int(os.environ.get("SOFTSIM_FEM_CHUNK", ELEMENT_CHUNK))

        def pieces(n):
            return [slice(s, min(s + size, n)) for s in range(0, n, size)]

        def chunked_energy(u, dndx, wv, c, b):
            if len(u) <= size:
                return energy(u, dndx, wv, c, b)
            return sum(float(energy(u[s], dndx[s], wv[s], c, b[s])) for s in pieces(len(u)))

        def chunked_grad_hess(u, dndx, wv, c, b):
            if len(u) <= size:
                return grad_hess(u, dndx, wv, c, b)
            ge, he = np.empty((len(u), 30)), np.empty((len(u), 30, 30))
            for s in pieces(len(u)):
                ge[s], he[s] = (np.asarray(a) for a in grad_hess(u[s], dndx[s], wv[s], c, b[s]))
            return ge, he

        def chunked_stress(u, dndx):
            if len(u) <= size:
                return stress(u, dndx)
            return np.concatenate([np.asarray(stress(u[s], dndx[s])) for s in pieces(len(u))])

        return chunked_energy, chunked_grad_hess, chunked_stress

    @staticmethod
    def _group_kernels(law, kappa):
        """(energy(u, dndx, wv, c, b) summed, grad_hess per element, iso_stress per quadrature point) for one law."""

        def element_energy(ue, dndx_e, wv_e, c_iso, b_e):
            ue = ue.reshape(10, 3)

            def at_qp(dn, w, bq):
                f = jnp.eye(3) + ue.T @ dn
                c = f.T @ f
                j = det3(f)
                return w * (c_iso * law.energy(c) + 0.5 * jnp.sum(bq * c)), w * j, j > 0.0

            e_iso, wj, ok = jax.vmap(at_qp)(dndx_e, wv_e, b_e)
            ve = jnp.sum(wv_e)
            return jnp.sum(e_iso) + ve * volumetric_energy(jnp.sum(wj) / ve, kappa) + jnp.where(jnp.all(ok), 0.0, jnp.inf)

        def qp_tangent(f, bq, c_iso):
            if isinstance(law, InvariantLaw):
                pk, a4 = invariant_tangent(law, f, bq, c_iso)
            else:
                def psi(ff):
                    cc = ff.T @ ff
                    return c_iso * law.energy(cc) + 0.5 * jnp.sum(bq * cc)
                pk, a4 = jax.grad(psi)(f), jax.hessian(psi)(f)
            return pk, a4, cof3(f), d2det3(f), det3(f)

        def element_grad_hess(ue, dndx_e, wv_e, c_iso, b_e):
            """Gradient and Hessian from the 3x3 material tangents at the quadrature points (much
            cheaper than differentiating the whole element twice): iso part sum_q w G^T A G, volume part
            U''(J_mean) / V g g^T + U'(J_mean) sum_q w G^T d2J G."""
            f = jnp.eye(3)[None] + jnp.einsum("ni,qnj->qij", ue.reshape(10, 3), dndx_e)
            pk, a4, dj, d2j, j = jax.vmap(qp_tangent, in_axes=(0, 0, None))(f, b_e, c_iso)
            ve = jnp.sum(wv_e)
            jm = jnp.sum(wv_e * j) / ve
            du = jax.grad(volumetric_energy)(jm, kappa)
            d2u = jax.grad(jax.grad(volumetric_energy))(jm, kappa)
            g_iso = jnp.einsum("q,qiJ,qnJ->ni", wv_e, pk, dndx_e).reshape(30)
            g_j = jnp.einsum("q,qiJ,qnJ->ni", wv_e, dj, dndx_e).reshape(30)
            h = jnp.einsum("q,qiJkL,qnJ,qmL->nimk", wv_e, a4 + du * d2j, dndx_e, dndx_e).reshape(30, 30)
            return g_iso + du * g_j, h + d2u / ve * jnp.outer(g_j, g_j)

        def iso_stress(ue, dn):
            f = jnp.eye(3) + ue.reshape(10, 3).T @ dn
            return 2.0 * jax.grad(law.energy)(f.T @ f)

        return (jax.jit(lambda u, dndx, wv, c, b: jnp.sum(jax.vmap(element_energy, in_axes=(0, 0, 0, None, 0))(
                    u, dndx, wv, c, b))),
                jax.jit(jax.vmap(element_grad_hess, in_axes=(0, 0, 0, None, 0))),
                jax.jit(jax.vmap(jax.vmap(iso_stress, in_axes=(None, 0)), in_axes=(0, 0))))

    def _build_load_kernels(self) -> None:
        p = self.p
        n_tri, dn_tri, w_tri = jnp.asarray(_TRI_N), jnp.asarray(_TRI_DN), jnp.asarray(_TRI_W)

        def face_s(xf):  # 1/3 int x . n dA over one 6-node face, xf (18,)
            xf = xf.reshape(6, 3)
            x = n_tri @ xf
            xa = jnp.einsum("qn,ni->qi", dn_tri[:, :, 0], xf)
            xb = jnp.einsum("qn,ni->qi", dn_tri[:, :, 1], xf)
            return jnp.sum(w_tri * jnp.einsum("qi,qi->q", x, jnp.cross(xa, xb))) / 3.0

        self._face_s = jax.jit(lambda xf: jnp.sum(jax.vmap(face_s)(xf)))
        self._face_gh = jax.jit(jax.vmap(lambda xf: (jax.grad(face_s)(xf), jax.hessian(face_s)(xf))))
        self._face_nodes0 = [jnp.asarray(p.mesh.nodes[np.asarray(f, int)].reshape(len(f), 18))
                             for f in p.pressure_faces]

        x0 = jnp.asarray(p.mesh.nodes[self._snodes])
        area = jnp.asarray(self._sarea)
        self._contact = []
        for body in p.rigid:
            self._contact.append(self._contact_kernels(body, x0, area))

    def _contact_kernels(self, body: RigidBody3D, x0, area):
        sdf = _sdf_function(body)
        grad_sdf = jax.grad(sdf)
        cs = body.contact
        mount = body.mount

        def pose(r, c_ref, r_ref):
            if mount == "free":
                return c_ref + r[:3], _expm_so3(r[3:]) @ r_ref
            if mount == "slider_z":
                return c_ref + jnp.array([0.0, 0.0, r[0]]), r_ref
            return c_ref, r_ref

        def local(u, r, x0i, c_ref, r_ref):
            c, rot = pose(r, c_ref, r_ref)
            return rot.T @ (x0i + u - c)

        def coh(gap):
            gp = jnp.maximum(gap, 0.0) / cs.adhesion_range
            return math.e * cs.adhesion_strength * cs.adhesion_range * (1.0 - (1.0 + gp) * jnp.exp(-gp))

        def f0(x, eps):
            return jnp.where(x < eps, -x**3 / (3 * eps**2) + x**2 / eps + eps / 3, x)

        def node_energy(z, x0i, a, bond, anchor, lam, c_ref, r_ref):
            u, r = z[:3], z[3:]
            pl = local(u, r, x0i, c_ref, r_ref)
            gap = sdf(pl)
            e = 0.5 * cs.stiffness * jnp.minimum(gap, 0.0) ** 2
            if cs.stick or cs.friction > 0:
                n = grad_sdf(pl)
                n = n / jnp.sqrt(jnp.dot(n, n) + 1e-30)
                d = pl - anchor
                slip = d - n * jnp.dot(n, d)
                if cs.stick:
                    e = e + jnp.where(bond, 0.5 * cs.stiffness * jnp.dot(slip, slip), 0.0)
                else:
                    e = e + cs.friction * lam * f0(jnp.sqrt(jnp.dot(slip, slip) + 1e-30), cs.slip_tolerance)
            if cs.adhesion:
                e = e + jnp.where(bond, coh(gap), 0.0)
            return a * e

        args = (0, 0, 0, 0, 0, 0, None, None)
        gh = jax.jit(jax.vmap(lambda *a: (jax.grad(node_energy)(*a), jax.hessian(node_energy)(*a)), in_axes=args))
        total = jax.jit(lambda z, b, an, lm, cr, rr: jnp.sum(jax.vmap(node_energy, in_axes=args)(
            z, x0, area, b, an, lm, cr, rr)))
        points = jax.jit(jax.vmap(local, in_axes=(0, None, 0, None, None)))
        gaps = jax.jit(jax.vmap(sdf))
        return {"grad_hess": gh, "energy": total, "local": points, "gap": gaps, "n": body.n_dofs}

    # -- assembly ----------------------------------------------------------------------------------

    def _rigid_refs(self, t_new: float):
        refs = []
        for i, body in enumerate(self.p.rigid):
            c_ref = self._c0[i].copy()
            if body.mount == "prescribed":
                c_ref = c_ref + np.asarray(body.motion(t_new), float)
            refs.append((jnp.asarray(c_ref), jnp.asarray(self.R[i])))
        return refs

    def _contact_z(self, y: np.ndarray, i: int) -> np.ndarray:
        u = y[self._sdofs]
        r = y[self._rdof[i]]
        return np.concatenate([u, np.broadcast_to(r, (len(u), len(r)))], axis=1)

    def _viscous_terms(self, dt: float):
        """(c, b) for the element kernels: the isochoric weight ``c`` (a float, or a tuple with one per material
        group when there are zones) and the viscous stress history ``b`` (E, Q, 3, 3)."""
        cs = []
        b = np.zeros((*self._wv.shape, 3, 3))
        for (idx, pr), q, s_prev in zip(self._visc, self._q, self._s_prev):
            if pr is None or not self.viscous:
                cs.append(1.0 if pr is None else 1.0 - sum(pr.g))
                continue
            c = 1.0
            for i, (g, tau) in enumerate(zip(pr.g, pr.tau)):
                x = dt / tau
                a = -math.expm1(-x) / x
                c -= g * (1.0 - a)
                b[idx] += math.exp(-x) * q[i] - g * a * s_prev
            cs.append(c)
        return (cs[0] if len(cs) == 1 else tuple(cs)), b

    def _context(self, dt: float) -> dict:
        c, b = self._viscous_terms(dt)
        beta = self.beta
        y_tilde = self.y + dt * self.v + dt * dt * (0.5 - beta) * self.a
        # rigid inertia (world frame) and loads
        rows, cols, vals = [], [], []
        f_ext = np.zeros(self.ndof)
        g = np.asarray(self.gravity, float)
        f_ext[:3 * self.n] = np.repeat(self._node_mass, 3) * np.tile(g, self.n)
        for force, vec in zip(self.forces, self._force_vecs):
            f_ext[:3 * self.n] += force * vec
        spring = np.zeros(self.ndof)
        damper = np.zeros(self.ndof)
        for i, body in enumerate(self.p.rigid):
            d = self._rdof[i]
            if body.mount == "free":
                block = np.zeros((6, 6))
                block[:3, :3] = body.mass * np.eye(3)
                block[3:, 3:] = self.R[i] @ self._inertia[i] @ self.R[i].T
                f_ext[d[:3]] = body.mass * g
            elif body.mount == "slider_z":
                block = np.array([[body.mass]])
                f_ext[d] = body.mass * g[2]
                spring[d] = body.stiffness
                damper[d] = body.damping
            else:
                continue
            rr, cc = np.meshgrid(d, d, indexing="ij")
            rows.append(rr.ravel())
            cols.append(cc.ravel())
            vals.append(block.ravel())
        inertia = self.mass_matrix if not rows else sp.block_diag([self.mass_matrix, sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows) - 3 * self.n, np.concatenate(cols) - 3 * self.n)),
            shape=(self.ndof - 3 * self.n,) * 2)], format="csr")
        # the quadratic terms' Hessian (constant over the step) in the pattern's upper triangle
        pat = self._pattern
        h_quad = np.zeros(pat.nnz)
        h_quad[pat.diag] += (spring + damper / dt)[self.free]
        if self.dynamic:
            h_quad += pat.scatter(self._mpos, self._mval / (beta * dt * dt))
            if rows:
                sel, pos = pat.locate(np.concatenate(rows), np.concatenate(cols))
                h_quad += pat.scatter(pos, np.concatenate(vals)[sel] / (beta * dt * dt))
        return {"c": c, "b": jnp.asarray(b), "beta_dt2": beta * dt * dt, "dt": dt, "y_tilde": y_tilde,
                "y_n": self.y.copy(), "inertia": inertia, "f_ext": f_ext, "spring": spring, "damper": damper,
                "h_quad": h_quad, "refs": self._rigid_refs(self.t + dt)}

    def _quadratic(self, y: np.ndarray, ctx) -> tuple[float, np.ndarray]:
        """Gravity, inertia, springs and dampers: energy and gradient (Hessian: ``ctx["h_quad"]``)."""
        e = -float(ctx["f_ext"] @ y)
        grad = -ctx["f_ext"].copy()
        e += 0.5 * float(np.sum(ctx["spring"] * y * y))
        dy = y - ctx["y_n"]
        e += 0.5 * float(np.sum(ctx["damper"] * dy * dy)) / ctx["dt"]
        grad += ctx["spring"] * y + ctx["damper"] * dy / ctx["dt"]
        if self.dynamic:
            d = y - ctx["y_tilde"]
            md = ctx["inertia"] @ d
            e += 0.5 * float(d @ md) / ctx["beta_dt2"]
            grad += md / ctx["beta_dt2"]
        return e, grad

    def energy(self, y: np.ndarray, ctx) -> float:
        u = y[:3 * self.n].reshape(self.n, 3)
        e = float(self._elem_energy(jnp.asarray(u[self.p.mesh.tets].reshape(-1, 30)), ctx["c"], ctx["b"]))
        e += self._quadratic(y, ctx)[0]
        for k, pressure in enumerate(self.pressures):
            if pressure:
                e += pressure * float(self._face_s(self._face_x(y, k)))
        for k, tension in enumerate(self.tensions):
            if self.stiffness[k]:
                e += 0.5 * self.stiffness[k] * max(self.cable_length(k, y) - self.free_lengths[k], 0.0) ** 2
            elif tension:
                e += tension * self.cable_length(k, y)
        for i, k in enumerate(self._contact):
            cr, rr = ctx["refs"][i]
            e += float(k["energy"](jnp.asarray(self._contact_z(y, i)), jnp.asarray(self._bond[i]),
                                   jnp.asarray(self._anchor[i]), jnp.asarray(self._lam[i]), cr, rr))
        return e

    def _face_x(self, y: np.ndarray, k: int):
        """Current positions of pressure load ``k``'s face nodes, (K, 18)."""
        return self._face_nodes0[k] + jnp.asarray(y[self._pdofs[k]])

    def _segments(self, k: int, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cable ``k``'s segment vectors (S, 3) and lengths (S,)."""
        dofs, w, x0 = self._cables[k]
        s = np.einsum("sj,sji->si", w, x0 + y[dofs].reshape(x0.shape))
        return s, np.linalg.norm(s, axis=1)

    def cable_length(self, k: int, y: np.ndarray | None = None) -> float:
        """Current length of cable ``k`` through its via points."""
        return float(self._segments(k, self.y if y is None else y)[1].sum())

    def cable_tension(self, k: int, y: np.ndarray | None = None) -> float:
        """Tension [N] of cable ``k``: its set tension, or for a stroke-controlled one ``k (L - L_free)+``."""
        if not self.stiffness[k]:
            return float(self.tensions[k])
        return float(self.stiffness[k] * max(self.cable_length(k, y) - self.free_lengths[k], 0.0))

    def enclosed_volume(self, k: int) -> float:
        """Volume enclosed by pressure load ``k``'s faces (the cavity), current configuration."""
        return -float(self._face_s(self._face_x(self.y, k)))

    @staticmethod
    def _project_psd(h: np.ndarray) -> np.ndarray:
        w, v = np.linalg.eigh(0.5 * (h + np.swapaxes(h, -1, -2)))
        return np.einsum("...ij,...j,...kj->...ik", v, np.maximum(w, 0.0), v)

    def _grad_hess(self, y: np.ndarray, ctx, project: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Gradient (all dofs) and the Hessian's free-free upper triangle as ``self._pattern`` data."""
        u = y[:3 * self.n].reshape(self.n, 3)
        ge, he = self._elem_grad_hess(jnp.asarray(u[self.p.mesh.tets].reshape(-1, 30)), self._dndx, self._wv,
                                      ctx["c"], ctx["b"])
        ge, he = np.asarray(ge), np.asarray(he)
        if project:
            he = self._project_psd(he)
        pat = self._pattern
        grad = np.bincount(self._edofs.ravel(), weights=ge.ravel(), minlength=self.ndof)
        hess = pat.scatter(self._epos, he.reshape(-1)) + ctx["h_quad"]
        for k, pressure in enumerate(self.pressures):
            if not pressure:
                continue
            gf, hf = (np.asarray(a) * pressure for a in self._face_gh(self._face_x(y, k)))
            if project:
                hf = self._project_psd(hf)
            grad += np.bincount(self._pdofs[k].ravel(), weights=gf.ravel(), minlength=self.ndof)
            hess += pat.scatter(self._ppos[k], hf.reshape(-1))
        for k in range(len(self._cables)):
            tension = self.cable_tension(k, y)
            if not tension:
                continue
            dofs, w, _ = self._cables[k]
            s, n = self._segments(k, y)
            u = s / n[:, None]
            gl = w[:, :, None] * u[:, None, :]                                           # dL/dx, (S, 20, 3)
            proj = (np.eye(3) - u[:, :, None] * u[:, None, :]) / n[:, None, None]       # PSD
            hc = tension * np.einsum("sa,sb,sij->saibj", w, w, proj)
            g = np.bincount(dofs.ravel(), weights=gl.ravel(), minlength=self.ndof)
            grad += tension * g
            hess += pat.scatter(self._cpos[k], hc.reshape(-1))
            if self.stiffness[k]:  # taut tendon: + k dL dL^T over the whole cable
                gd = g[self._cable_dofs[k]]
                hess += pat.scatter(self._cdense[k], self.stiffness[k] * np.outer(gd, gd).ravel())
        x0 = jnp.asarray(self.p.mesh.nodes[self._snodes])
        area = jnp.asarray(self._sarea)
        for i, k in enumerate(self._contact):
            cr, rr = ctx["refs"][i]
            gb, hb = k["grad_hess"](jnp.asarray(self._contact_z(y, i)), x0, area, jnp.asarray(self._bond[i]),
                                    jnp.asarray(self._anchor[i]), jnp.asarray(self._lam[i]), cr, rr)
            gb, hb = np.asarray(gb), np.asarray(hb)
            active = np.any(gb != 0, axis=1) | np.any(hb != 0, axis=(1, 2))
            if not active.any():
                continue
            gb, hb = gb[active], hb[active]
            if project:
                hb = self._project_psd(hb)
            dofs = np.concatenate([self._sdofs[active], np.broadcast_to(self._rdof[i], (int(active.sum()), k["n"]))],
                                  axis=1)
            np.add.at(grad, dofs, gb)
            m = dofs.shape[1]
            sel, pos = pat.locate(np.repeat(dofs, m, axis=1), np.tile(dofs, (1, m)))
            hess += pat.scatter(pos, hb.reshape(-1)[sel])
        return grad + self._quadratic(y, ctx)[1], hess

    # -- stepping ----------------------------------------------------------------------------------

    def solve_step(self, dt: float, tol: float = 1e-8, max_iter: int = 60) -> bool:
        """One implicit step; returns False (state unchanged) if Newton does not converge."""
        ctx = self._context(dt)
        y = self.y + dt * self.v + 0.5 * dt * dt * self.a if self.dynamic else self.y.copy()
        y[self.fixed] = self.y[self.fixed] if self.fixed_motion is None else self.fixed_motion(self.t + dt)
        scale = max(float(np.ptp(self.p.mesh.nodes, axis=0).max()), 1e-3)
        e_old = self.energy(y, ctx)
        if not np.isfinite(e_old):
            y = self.y.copy()
            e_old = self.energy(y, ctx)
        for _ in range(max_iter):
            converged = False
            for project in (False, True):
                grad, hess = self._grad_hess(y, ctx, project)
                g_free = grad[self.free]
                step = -self._pattern.solve(hess, g_free)
                if np.all(np.isfinite(step)) and np.max(np.abs(step), initial=0.0) < tol * scale:
                    converged = True
                    break
                slope = float(g_free @ step)
                if np.all(np.isfinite(step)) and slope < 0:
                    found = self._line_search(y, step, slope, e_old, ctx)
                    if found is not None:
                        break
            else:
                return False
            if converged:
                break
            _, y, e_old = found
            if np.max(np.abs(step)) < tol * scale:
                break
        else:
            return False
        self.residual = grad  # at the held dofs: the supports' forces on the body (within tol of the end state)
        self._accept(y, dt, ctx)
        return True

    def advance(self, dt: float, depth: int = 4) -> bool:
        """Advance by ``dt``, splitting into halves (recursively, ``depth`` levels) where a step fails."""
        if self.solve_step(dt):
            return True
        if depth == 0:
            return False
        return self.advance(dt / 2, depth - 1) and self.advance(dt / 2, depth - 1)

    def _line_search(self, y, step, slope, e_old, ctx):
        if -slope < 1e-10 * abs(e_old):
            trial = y.copy()
            trial[self.free] += step
            return 1.0, trial, self.energy(trial, ctx)
        alpha = 1.0
        for _ in range(30):
            trial = y.copy()
            trial[self.free] += alpha * step
            e_new = self.energy(trial, ctx)
            if np.isfinite(e_new) and e_new <= e_old + 1e-4 * alpha * slope + 1e-15 * abs(e_old):
                return alpha, trial, e_new
            alpha *= 0.5
        return None

    def _accept(self, y: np.ndarray, dt: float, ctx) -> None:
        if self.dynamic:
            a_new = (y - ctx["y_tilde"]) / (self.beta * dt * dt)
            v_new = self.v + dt * ((1 - self.gamma) * self.a + self.gamma * a_new)
        else:
            a_new = np.zeros_like(y)
            v_new = (y - self.y) / dt
        for i, body in enumerate(self.p.rigid):
            if body.mount == "free":
                rot = self._rdof[i][3:]
                self.R[i] = expm_so3(y[rot]) @ self.R[i]
                y[rot] = 0.0  # rotation dofs are increments per step
        self.y, self.v, self.a = y, v_new, a_new
        self.t += dt
        self._update_history(dt)
        self._update_contact_state(ctx)

    def _update_history(self, dt: float) -> None:
        if all(pr is None for _, pr in self._visc):
            return
        u = self.y[:3 * self.n].reshape(self.n, 3)
        s_all = np.asarray(self._iso_stress(jnp.asarray(u[self.p.mesh.tets].reshape(-1, 30)), self._dndx))
        for k, (idx, pr) in enumerate(self._visc):
            if pr is None:
                continue
            s_new = s_all[idx]
            if self.viscous:
                for i, (g, tau) in enumerate(zip(pr.g, pr.tau)):
                    x = dt / tau
                    self._q[k][i] = math.exp(-x) * self._q[k][i] + g * (-math.expm1(-x) / x) * (s_new - self._s_prev[k])
            self._s_prev[k] = s_new

    def _update_contact_state(self, ctx) -> None:
        refs = self._rigid_refs(self.t)
        for i, (body, k) in enumerate(zip(self.p.rigid, self._contact)):
            cs = body.contact
            z = self._contact_z(self.y, i)
            cr, rr = refs[i]
            pts = np.asarray(k["local"](jnp.asarray(z[:, :3]), jnp.asarray(z[0, 3:]) if k["n"] else jnp.zeros(0),
                                        jnp.asarray(self.p.mesh.nodes[self._snodes]), cr, rr))
            gaps = np.asarray(k["gap"](jnp.asarray(pts)))
            bond, anchor, lam = self._bond[i], self._anchor[i], self._lam[i]
            if cs.friction > 0 and not cs.stick:
                anchor[:] = pts
                lam[:] = cs.stiffness * np.maximum(-gaps, 0.0)
            new = (gaps <= 0.0) & ~bond
            if cs.stick:
                anchor[new] = pts[new]
            bond |= gaps <= 0.0
            limit = 6.0 * cs.adhesion_range if cs.adhesion else 0.0
            bond &= gaps <= limit

    # -- outputs -----------------------------------------------------------------------------------

    @property
    def u(self) -> np.ndarray:
        return self.y[:3 * self.n].reshape(self.n, 3)

    def current_nodes(self) -> np.ndarray:
        return self.p.mesh.nodes + self.u

    def rigid_pose(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """(centre of mass position, rotation matrix) of rigid body ``i``."""
        body = self.p.rigid[i]
        c = self._c0[i].copy()
        if body.mount == "free":
            c += self.y[self._rdof[i][:3]]
        elif body.mount == "slider_z":
            c[2] += self.y[self._rdof[i][0]]
        elif body.mount == "prescribed":
            c += np.asarray(body.motion(self.t), float)
        return c, self.R[i]

    def rigid_velocity(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        body = self.p.rigid[i]
        d = self._rdof[i]
        if body.mount == "free":
            return self.v[d[:3]].copy(), self.v[d[3:]].copy()
        if body.mount == "slider_z":
            return np.array([0.0, 0.0, self.v[d[0]]]), np.zeros(3)
        return np.zeros(3), np.zeros(3)

    def contact_force(self, i: int) -> np.ndarray:
        """Force [N] the soft body exerts on rigid body ``i`` through contact and adhesion (world frame)."""
        k = self._contact[i]
        z = self._contact_z(self.y, i)
        cr, rr = self._rigid_refs(self.t)[i]
        gb, _ = k["grad_hess"](jnp.asarray(z), jnp.asarray(self.p.mesh.nodes[self._snodes]),
                               jnp.asarray(self._sarea), jnp.asarray(self._bond[i]), jnp.asarray(self._anchor[i]),
                               jnp.asarray(self._lam[i]), cr, rr)
        return np.asarray(gb)[:, :3].sum(axis=0)  # dE/du summed = force of the body on the soft nodes, reversed

    def internal_forces(self) -> np.ndarray:
        """Nodal internal forces (N, 3) of the current stress, viscous memory included."""
        cs = tuple(1.0 - sum(pr.g) if pr is not None else 1.0 for _, pr in self._visc)
        c = cs[0] if len(cs) == 1 else cs
        b = np.zeros((*self._wv.shape, 3, 3))
        if self.viscous:
            for (idx, pr), q in zip(self._visc, self._q):
                if pr is not None:
                    b[idx] = q.sum(axis=0)
        ge, _ = self._elem_grad_hess(jnp.asarray(self.u[self.p.mesh.tets].reshape(-1, 30)), self._dndx, self._wv,
                                     c, jnp.asarray(b))
        f = np.zeros(3 * self.n)
        np.add.at(f, self._edofs, np.asarray(ge))
        return f.reshape(self.n, 3)

    def strain_energy(self) -> float:
        return float(self._elem_energy(jnp.asarray(self.u[self.p.mesh.tets].reshape(-1, 30)), 1.0,
                                       jnp.zeros((*self._wv.shape, 3, 3))))

    def kinetic_energy(self) -> float:
        v = self.v[:3 * self.n]
        e = 0.5 * float(v @ (self.mass_matrix @ v))
        for i, body in enumerate(self.p.rigid):
            lin, ang = self.rigid_velocity(i)
            if body.mount in ("free", "slider_z"):
                e += 0.5 * body.mass * float(lin @ lin)
            if body.mount == "free":
                e += 0.5 * float(ang @ (self.R[i] @ self._inertia[i] @ self.R[i].T @ ang))
        return e
