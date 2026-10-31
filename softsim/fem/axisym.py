"""Axisymmetric finite-strain FEM: the reference solver for soft-material tests.

Why a second solver: the general backends (MuJoCo flex, SOFA) use linear
tetrahedra, which lock for nearly incompressible materials, so their results
depend on the mesh and fitted moduli are "effective" values (see
docs/backends/mujoco.md). For axisymmetric problems (cylinders under a centred
punch, uniaxial tests) this solver gives mesh-converged answers to compare them
with:

* **Elements:** 6-node quadratic triangles in the r-z plane (P2) with one
  dilatation per element (mean dilatation, P2/P0), which is free of
  volumetric locking up to incompressibility.
* **Materials:** isochoric/volumetric split hyperelasticity (Neo-Hookean,
  Mooney-Rivlin, Yeoh, Ogden) plus Prony-series viscoelasticity on the
  isochoric part (``softsim.fem.materials``).
* **Time integration:** quasi-static or Newmark (average acceleration, no
  numerical damping). Each step minimises an incremental potential, with
  energy-based line search; exact tangents come from JAX. Where the tangent
  is indefinite (finite strain far from equilibrium), Newton falls back to
  element Hessians projected onto their positive part, which guarantees a
  descent direction.
* **Contact:** a rigid flat punch with a rounded edge (one vertical degree of
  freedom, a free-falling weight) and the ground plane, both by penalty on the
  boundary nodes with their tributary (annular) areas, frictionless or
  sticking, with optional cohesive adhesion (the same law as
  ``softsim.materials.Adhesion``). Node-based, because with quadratic edges in
  axisymmetry a uniform pressure gives the node on the axis exactly zero load
  (the integral of its shape function times r vanishes), so quadrature-point
  contact lets the punch pierce the axis.

Units are SI. ``u`` is the nodal displacement (radial, axial); the axis r = 0
has u_r = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from softsim.fem import jax_setup
from softsim.fem.materials import IsochoricLaw, Prony

jax, jnp = jax_setup()

# -- reference element ------------------------------------------------------------------------

# 7-point Dunavant rule (degree 5) on the reference triangle (area 1/2)
_A, _B, _WAB = 0.059715871789770, 0.470142064105115, 0.132394152788506
_C, _D, _WCD = 0.797426985353087, 0.101286507323456, 0.125939180544827
_BARY = np.array([[1 / 3, 1 / 3, 1 / 3], [_A, _B, _B], [_B, _A, _B], [_B, _B, _A],
                  [_C, _D, _D], [_D, _C, _D], [_D, _D, _C]])
_W_TRI = 0.5 * np.array([0.225, _WAB, _WAB, _WAB, _WCD, _WCD, _WCD])


def _tri_shape(l1, l2, l3):
    """P2 shape functions (corners 0-2, then mid-edges 01, 12, 20) and their (xi, eta) derivatives."""
    n = np.array([l1 * (2 * l1 - 1), l2 * (2 * l2 - 1), l3 * (2 * l3 - 1), 4 * l1 * l2, 4 * l2 * l3, 4 * l3 * l1])
    # xi = l2, eta = l3, l1 = 1 - xi - eta
    dxi = np.array([-(4 * l1 - 1), 4 * l2 - 1, 0.0, 4 * (l1 - l2), 4 * l3, -4 * l3])
    deta = np.array([-(4 * l1 - 1), 0.0, 4 * l3 - 1, -4 * l2, 4 * l2, 4 * (l1 - l3)])
    return n, np.stack([dxi, deta], axis=1)


_N_TRI, _DN_TRI = (np.array(a) for a in zip(*[_tri_shape(*b) for b in _BARY]))  # (7, 6), (7, 6, 2)


def _tributary_areas(nodes: np.ndarray, edges: np.ndarray) -> dict[int, float]:
    """Surface area of revolution owned by each boundary node: the band between the midpoints
    to its neighbours along the (end, end, mid) edges (exact for straight sub-segments)."""
    area: dict[int, float] = {}
    for a, b, m in edges:
        for p, q in ((a, m), (m, b)):
            rp, rq = nodes[p, 0], nodes[q, 0]
            half = 0.5 * np.linalg.norm(nodes[q] - nodes[p])
            area[p] = area.get(p, 0.0) + 2 * np.pi * (3 * rp + rq) / 4 * half
            area[q] = area.get(q, 0.0) + 2 * np.pi * (rp + 3 * rq) / 4 * half
    return area


# -- mesh --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class AxisymMesh:
    """P2 triangle mesh of the r-z half-section of a body of revolution.

    ``edges`` hold boundary edges as (end, end, mid) node triples, grouped by name
    (``top``, ``bottom``, ``side``).
    """

    nodes: np.ndarray                      # (N, 2) reference (R, Z)
    triangles: np.ndarray                  # (E, 6)
    edges: dict[str, np.ndarray]           # name -> (K, 3)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def volume(self) -> float:
        """Volume of revolution, by quadrature (exact for straight-sided P2 elements)."""
        x = self.nodes[self.triangles]
        jac = np.einsum("eni,qnj->eqij", x, _DN_TRI)
        det = np.abs(np.linalg.det(jac))
        r = np.einsum("qn,en->eq", _N_TRI, x[..., 0])
        return float(np.sum(det * _W_TRI * 2 * np.pi * r))


def cylinder_mesh(radius: float, height: float, nr: int, nz: int, grading: float = 1.0) -> AxisymMesh:
    """Structured P2 mesh of a cylinder's half-section [0, radius] x [0, height].

    ``grading`` > 1 concentrates rows towards the top (where a punch acts).
    """
    gi = np.linspace(0.0, 1.0, 2 * nr + 1)
    gj = np.linspace(0.0, 1.0, 2 * nz + 1)
    zs = height * (1 - (1 - gj) ** grading) if grading != 1.0 else height * gj
    # keep mid-side nodes exactly halfway between their corners (straight edges)
    zs[1::2] = 0.5 * (zs[:-1:2] + zs[2::2])
    rr, zz = np.meshgrid(radius * gi, zs, indexing="ij")
    nodes = np.column_stack([rr.ravel(), zz.ravel()])
    nid = np.arange(len(nodes)).reshape(2 * nr + 1, 2 * nz + 1)
    tris = []
    for i in range(nr):
        for j in range(nz):
            a, b, c, d = nid[2 * i, 2 * j], nid[2 * i + 2, 2 * j], nid[2 * i + 2, 2 * j + 2], nid[2 * i, 2 * j + 2]
            mab, mbc, mcd, mda = nid[2 * i + 1, 2 * j], nid[2 * i + 2, 2 * j + 1], nid[2 * i + 1, 2 * j + 2], nid[2 * i, 2 * j + 1]
            mac = nid[2 * i + 1, 2 * j + 1]
            tris.append([a, b, c, mab, mbc, mac])
            tris.append([a, c, d, mac, mcd, mda])
    top = [[nid[2 * i, 2 * nz], nid[2 * i + 2, 2 * nz], nid[2 * i + 1, 2 * nz]] for i in range(nr)]
    bottom = [[nid[2 * i, 0], nid[2 * i + 2, 0], nid[2 * i + 1, 0]] for i in range(nr)]
    side = [[nid[2 * nr, 2 * j], nid[2 * nr, 2 * j + 2], nid[2 * nr, 2 * j + 1]] for j in range(nz)]
    return AxisymMesh(nodes, np.array(tris), {"top": np.array(top), "bottom": np.array(bottom), "side": np.array(side)})


# -- problem definition -----------------------------------------------------------------------

@dataclass(frozen=True)
class Punch:
    """Rigid flat punch (axisymmetric weight) above the body, moving vertically.

    The punch occupies r <= ``radius`` above its bottom face at height ``z``;
    its edge is rounded with ``edge_radius``. ``mass`` and the initial state
    (``z0`` of the bottom face, ``v0`` vertical velocity) define its dynamics.
    ``com_height`` is the centre of mass above the bottom face (reporting only).
    """

    radius: float
    mass: float
    z0: float
    v0: float = 0.0
    edge_radius: float = 2e-4
    com_height: float = 0.0


@dataclass(frozen=True)
class ContactSettings:
    """Penalty contact on boundary nodes.

    ``stiffness`` [Pa/m] relates penetration to contact pressure. Tangentially:
    frictionless by default; ``stick`` = no slip (penalty on the slip since
    first contact); ``friction`` > 0 = Coulomb friction with that coefficient,
    as an incremental potential mu p_n f0(|slip in the step|) with the normal
    pressure p_n of the previous step (lagged) and f0 a C1-smoothed |x| that
    sticks below ``slip_tolerance`` per step. ``adhesion_strength`` /
    ``adhesion_range`` (0 = off) give the exponential cohesive law of
    :class:`softsim.materials.Adhesion`.
    """

    stiffness: float = 2e11
    stick: bool = False
    adhesion_strength: float = 0.0
    adhesion_range: float = 3e-4
    friction: float = 0.0
    slip_tolerance: float = 1e-7

    @property
    def adhesion(self) -> bool:
        return self.adhesion_strength > 0


@dataclass
class Problem:
    """Body + boundary conditions + loads. ``base`` is ``"free"`` (ground contact),
    ``"slide"`` (bottom u_z = 0, free radially) or ``"bonded"`` (bottom fixed)."""

    mesh: AxisymMesh
    law: IsochoricLaw
    bulk_modulus: float
    density: float
    prony: Prony | None = None
    gravity: float = 9.81
    base: str = "free"
    punch: Punch | None = None
    punch_contact: ContactSettings = field(default_factory=ContactSettings)
    ground_contact: ContactSettings = field(default_factory=ContactSettings)
    top_displacement: Callable[[float], float] | None = None  # prescribed u_z(t) of the top face
    top_bonded: bool = False  # with top_displacement: also hold the top face radially (no slip)


# -- the solver ---------------------------------------------------------------------------------

def _u_volumetric(j, kappa):
    """Simo-Taylor volumetric energy kappa/4 (J^2 - 1 - 2 ln J): convex, U''(1) = kappa."""
    return 0.25 * kappa * (j * j - 1.0 - 2.0 * jnp.log(j))


def _right_cauchy_green(f2, hoop):
    c2 = f2.T @ f2
    return jnp.array([[c2[0, 0], c2[0, 1], 0.0], [c2[1, 0], c2[1, 1], 0.0], [0.0, 0.0, hoop * hoop]])


class AxisymSolver:
    """Newton solver for a :class:`Problem`; see module docstring.

    State: ``u`` (N, 2), ``v``, ``a`` (velocities, accelerations), punch ``zp``,
    ``vp``, ``ap``, time ``t``. ``step(dt)`` advances one implicit step (with
    automatic step halving if Newton fails).
    """

    def __init__(self, problem: Problem, dynamic: bool = True, numerical_damping: float = 0.0):
        self.p = problem
        self.dynamic = dynamic
        # Newmark gamma = 1/2 + a, beta = (1 + a)^2 / 4: a = 0 is the energy-conserving trapezoidal
        # rule; a small a damps only the unresolved high-frequency content (e.g. contact chatter)
        self.gamma = 0.5 + numerical_damping
        self.beta = 0.25 * (1.0 + numerical_damping) ** 2
        m = problem.mesh
        self.n = m.n_nodes
        x = m.nodes[m.triangles]                                             # (E, 6, 2)
        jac = np.einsum("eni,qnj->eqij", x, _DN_TRI)                          # (E, 7, 2, 2)
        self._det = np.linalg.det(jac)
        if np.any(self._det <= 0):
            raise ValueError("mesh has inverted or degenerate triangles")
        self._dndx = np.einsum("qnk,eqkj->eqnj", _DN_TRI, np.linalg.inv(jac))  # (E, 7, 6, 2)
        self._r0 = np.einsum("qn,en->eq", _N_TRI, x[..., 0])
        self._wv = self._det * _W_TRI * 2 * np.pi * self._r0                   # (E, 7) reference volume weights
        self._x = x
        self.volume = float(self._wv.sum())

        # contact / adhesion act on boundary nodes with their tributary areas (see module docstring);
        # stored as one-node "entities" (K, 1) so the energy kernels stay generic
        # the side surface can touch both: the punch when it is wider than the body, the ground when
        # a strongly compressed body bulges and its side rolls down onto the table
        top_area = _tributary_areas(m.nodes, np.concatenate([m.edges["top"], m.edges["side"]]))
        bot_area = _tributary_areas(m.nodes, np.concatenate([m.edges["bottom"], m.edges["side"]]))
        self._top_contact = np.array(sorted(top_area))[:, None]
        self._top_w = np.array([top_area[i] for i in self._top_contact[:, 0]])[:, None]
        self._bot_contact = np.array(sorted(bot_area))[:, None]
        self._bot_w = np.array([bot_area[i] for i in self._bot_contact[:, 0]])[:, None]

        # consistent mass matrix (scalar, per node pair)
        nn = np.einsum("qa,qb->qab", _N_TRI, _N_TRI)
        me = problem.density * np.einsum("eq,qab->eab", self._wv, nn)
        rows = np.repeat(m.triangles, 6, axis=1).ravel()
        cols = np.tile(m.triangles, (1, 6)).ravel()
        mass = sp.csr_matrix((me.ravel(), (rows, cols)), shape=(self.n, self.n))
        self.mass_matrix = sp.kron(mass, sp.eye(2), format="csr")              # dof order: node-major (r, z)
        self.total_mass = float(problem.density * self.volume)
        self._mass_z = np.asarray(mass.sum(axis=1)).ravel()  # nodal gravity loads / g (rows of M sum to int rho N)

        # degrees of freedom: 2 per node + punch
        self.ndof = 2 * self.n + (1 if problem.punch else 0)
        fixed = np.zeros(self.ndof, bool)
        on_axis = np.isclose(m.nodes[:, 0], 0.0)
        fixed[2 * np.nonzero(on_axis)[0]] = True
        bottom = np.unique(m.edges["bottom"])
        if problem.base == "slide":
            fixed[2 * bottom + 1] = True
        elif problem.base == "bonded":
            fixed[2 * bottom] = fixed[2 * bottom + 1] = True
        elif problem.base != "free":
            raise ValueError(f"unknown base {problem.base!r}")
        self._top_nodes = np.unique(m.edges["top"])
        if problem.top_displacement is not None:
            fixed[2 * self._top_nodes + 1] = True
            if problem.top_bonded:
                fixed[2 * self._top_nodes] = True
        self.fixed = fixed
        self.free = np.nonzero(~fixed)[0]

        # sparse pattern: elements (12 dofs), punch-side contact nodes (2 + punch), ground-side contact nodes (2)
        def dofs_of(nodes):
            return np.stack([2 * nodes, 2 * nodes + 1], axis=-1).reshape(len(nodes), -1)

        self._edofs = dofs_of(m.triangles)
        self._tdofs = dofs_of(self._top_contact)
        if problem.punch:
            self._tdofs = np.concatenate([self._tdofs, np.full((len(self._tdofs), 1), 2 * self.n)], axis=1)
        self._bdofs = dofs_of(self._bot_contact)

        # material history (per element quadrature point): Prony memory q_i and last isochoric stress
        n_terms = len(problem.prony.g) if problem.prony else 0
        self._q = np.zeros((n_terms, *self._wv.shape, 3, 3))
        self._s_prev = np.zeros((*self._wv.shape, 3, 3))
        # contact state per contact node
        self._bond_top = np.zeros(self._top_w.shape, bool)
        self._bond_bot = np.zeros(self._bot_w.shape, bool)
        self._anchor_top = self.p.mesh.nodes[self._top_contact, 0].copy()  # radial positions (slip reference)
        self._anchor_bot = self.p.mesh.nodes[self._bot_contact, 0].copy()
        self._lam_top = np.zeros(self._top_w.shape)  # lagged normal contact pressure (Coulomb friction)
        self._lam_bot = np.zeros(self._bot_w.shape)

        self.u = np.zeros((self.n, 2))
        self.v = np.zeros((self.n, 2))
        self.a = np.zeros((self.n, 2))
        pu = problem.punch
        self.zp, self.vp, self.ap = (pu.z0, pu.v0, 0.0) if pu else (0.0, 0.0, 0.0)
        self.t = 0.0
        self._build_kernels()

    # -- energies (JAX) -------------------------------------------------------------------------

    def _build_kernels(self) -> None:
        p = self.p
        law, kappa = p.law, p.bulk_modulus
        dndx = jnp.asarray(self._dndx)
        nq = jnp.asarray(_N_TRI)
        wv = jnp.asarray(self._wv)
        r0 = jnp.asarray(self._r0)

        def kinematics(ue, dn, n, r0q):
            f2 = jnp.eye(2) + ue.T @ dn                             # rows (r, z), columns (R, Z)
            hoop = 1.0 + (n @ ue[:, 0]) / r0q
            # J = det(F_rz) * hoop: both factors must be positive (a point pushed across the axis has
            # hoop < 0 and can hide an r-z fold, det < 0, in a positive product)
            ok = (jnp.linalg.det(f2) > 0.0) & (hoop > 0.0)
            return _right_cauchy_green(f2, hoop), jnp.linalg.det(f2) * hoop, ok

        def element_energy(ue, dndx_e, wv_e, r0_e, c_iso, b_e):
            """Incremental potential of one element: c_iso W_iso + 1/2 B:C per quadrature
            point, plus the volumetric energy of the element's mean dilatation."""
            ue = ue.reshape(6, 2)

            def at_qp(dn, n, w, r0q, bq):
                c, j, ok = kinematics(ue, dn, n, r0q)
                return w * (c_iso * law.energy(c) + 0.5 * jnp.sum(bq * c)), w * j, ok

            e_iso, wj, ok = jax.vmap(at_qp)(dndx_e, nq, wv_e, r0_e, b_e)
            ve = jnp.sum(wv_e)
            # the isochoric energy only sees C = F^T F and the volumetric one the element's mean J, so
            # neither notices a fold at a single quadrature point: infinite energy makes the line
            # search reject any step that inverts one
            folded = jnp.where(jnp.all(ok), 0.0, jnp.inf)
            return jnp.sum(e_iso) + ve * _u_volumetric(jnp.sum(wj) / ve, kappa) + folded

        self._elem_grad_hess = jax.jit(jax.vmap(
            lambda ue, dn, w, r, c, b: (jax.grad(element_energy)(ue, dn, w, r, c, b),
                                        jax.hessian(element_energy)(ue, dn, w, r, c, b)),
            in_axes=(0, 0, 0, 0, None, 0)))
        self._elem_energy = jax.jit(lambda u, c, b: jnp.sum(jax.vmap(
            element_energy, in_axes=(0, 0, 0, 0, None, 0))(u, dndx, wv, r0, c, b)))

        def iso_stress(ue, dn, n, r0q):
            c, _, _ = kinematics(ue.reshape(6, 2), dn, n, r0q)
            return 2.0 * jax.grad(law.energy)(c)  # isochoric second Piola-Kirchhoff stress

        self._iso_stress = jax.jit(jax.vmap(jax.vmap(iso_stress, in_axes=(None, 0, 0, 0)), in_axes=(0, 0, None, 0)))

        # boundary terms: punch/top-side and ground/bottom
        n_e = jnp.ones((1, 1))  # one contact point per entity: the node itself
        pc, gc = p.punch_contact, p.ground_contact
        punch = p.punch

        def coh(gap, strength, rng):
            gp = jnp.maximum(gap, 0.0) / rng
            return math.e * strength * rng * (1.0 - (1.0 + gp) * jnp.exp(-gp))

        def punch_gap(x):  # signed distance from the rounded punch (r <= a, z >= zp)
            rho = punch.edge_radius
            qx = x[0] - (punch.radius - rho)
            qy = (x[2] + rho) - x[1]  # x[2] carries zp
            outside = jnp.sqrt(jnp.maximum(qx, 0.0) ** 2 + jnp.maximum(qy, 0.0) ** 2 + 1e-30)
            return outside + jnp.minimum(jnp.maximum(qx, qy), 0.0) - rho

        def f0(x, eps):  # C1 smoothing of |x| (IPC): x^2/eps - x^3/(3 eps^2) + eps/3 below eps
            return jnp.where(x < eps, -x**3 / (3 * eps**2) + x**2 / eps + eps / 3, x)

        def tangential(cs, xr, bond, anchor, lam):
            if cs.stick:
                return 0.5 * cs.stiffness * jnp.where(bond, (xr - anchor) ** 2, 0.0)
            if cs.friction > 0:
                return cs.friction * lam * f0(jnp.abs(xr - anchor), cs.slip_tolerance)
            return 0.0

        def top_energy(ued, x0, w, bond, anchor, lam):
            # ued: node (ur, uz) + zp
            u3 = ued[:-1].reshape(-1, 2)
            zp = ued[-1]
            xq = n_e @ (x0 + u3)                                   # (3, 2) current points
            gaps = jax.vmap(lambda xx: punch_gap(jnp.array([xx[0], xx[1], zp])))(xq)
            e = 0.5 * pc.stiffness * jnp.minimum(gaps, 0.0) ** 2 + tangential(pc, xq[:, 0], bond, anchor, lam)
            if pc.adhesion:
                e = e + jnp.where(bond, coh(gaps, pc.adhesion_strength, pc.adhesion_range), 0.0)
            return jnp.sum(w * e)

        def bottom_energy(ued, x0, w, bond, anchor, lam):
            u3 = ued.reshape(-1, 2)
            xq = n_e @ (x0 + u3)
            gaps = xq[:, 1]
            e = 0.5 * gc.stiffness * jnp.minimum(gaps, 0.0) ** 2 + tangential(gc, xq[:, 0], bond, anchor, lam)
            if gc.adhesion:
                e = e + jnp.where(bond, coh(gaps, gc.adhesion_strength, gc.adhesion_range), 0.0)
            return jnp.sum(w * e)

        x_top = jnp.asarray(self.p.mesh.nodes[self._top_contact])
        x_bot = jnp.asarray(self.p.mesh.nodes[self._bot_contact])
        w_top, w_bot = jnp.asarray(self._top_w), jnp.asarray(self._bot_w)
        gh = lambda f: jax.jit(jax.vmap(lambda *a: (jax.grad(f)(*a), jax.hessian(f)(*a))))  # noqa: E731
        self._top_grad_hess = gh(top_energy) if punch else None
        self._bot_grad_hess = gh(bottom_energy) if p.base == "free" else None
        self._top_energy = jax.jit(lambda ud, b, a, lm: jnp.sum(jax.vmap(top_energy)(ud, x_top, w_top, b, a, lm))) \
            if punch else None
        self._bot_energy = jax.jit(lambda ud, b, a, lm: jnp.sum(jax.vmap(bottom_energy)(ud, x_bot, w_bot, b, a, lm))) \
            if p.base == "free" else None
        self._top_points = jax.jit(lambda ud: jax.vmap(lambda u6, x0: n_e @ (x0 + u6[:-1].reshape(-1, 2)))(ud, x_top))
        self._bot_points = jax.jit(lambda ud: jax.vmap(lambda u6, x0: n_e @ (x0 + u6.reshape(-1, 2)))(ud, x_bot))
        self._gap_fn = jax.jit(jax.vmap(jax.vmap(lambda xx, zp: punch_gap(jnp.array([xx[0], xx[1], zp])),
                                                 in_axes=(0, None)), in_axes=(0, None))) if punch else None

    # -- assembly -------------------------------------------------------------------------------

    def _vector(self, u: np.ndarray, zp: float) -> np.ndarray:
        vec = u.ravel()
        return np.append(vec, zp) if self.p.punch else vec

    def _viscous_terms(self, dt: float):
        """Coefficient of the isochoric energy and history tensors B for a step of length dt."""
        pr = self.p.prony
        if pr is None or not self.dynamic:
            c = 1.0 if pr is None else 1.0 - sum(pr.g)  # quasi-static: fully relaxed
            return c, np.zeros((*self._wv.shape, 3, 3))
        c = 1.0
        b = np.zeros((*self._wv.shape, 3, 3))
        for i, (g, tau) in enumerate(zip(pr.g, pr.tau)):
            x = dt / tau
            a = -math.expm1(-x) / x
            c -= g * (1.0 - a)
            b += math.exp(-x) * self._q[i] - g * a * self._s_prev
        return c, b

    def energy(self, y: np.ndarray, ctx) -> float:
        """Incremental potential of the step at the trial state ``y``."""
        u = y[:2 * self.n].reshape(self.n, 2)
        e = float(self._elem_energy(jnp.asarray(u[self.p.mesh.triangles].reshape(-1, 12)), ctx["c"], ctx["b"]))
        e += self._external(y, ctx)
        if self._top_energy is not None:
            e += float(self._top_energy(jnp.asarray(y[self._tdofs]), ctx["bond_top"], ctx["anchor_top"], ctx["lam_top"]))
        if self._bot_energy is not None:
            e += float(self._bot_energy(jnp.asarray(y[self._bdofs]), ctx["bond_bot"], ctx["anchor_bot"], ctx["lam_bot"]))
        return e

    def _external(self, y: np.ndarray, ctx) -> float:
        """Gravity and inertia terms (quadratic: also used analytically in the gradient/Hessian)."""
        g = self.p.gravity
        uz = y[1:2 * self.n:2]
        e = g * float(ctx["mass_z"] @ uz)
        if self.p.punch:
            e += self.p.punch.mass * g * y[-1]
        if self.dynamic:
            d = y - ctx["y_tilde"]
            e += 0.5 * float(d[:2 * self.n] @ (self.mass_matrix @ d[:2 * self.n])) / ctx["beta_dt2"]
            if self.p.punch:
                e += 0.5 * self.p.punch.mass * d[-1] ** 2 / ctx["beta_dt2"]
        return e

    @staticmethod
    def _project_psd(h: np.ndarray) -> np.ndarray:
        """Clip negative eigenvalues of a batch of symmetric matrices to zero."""
        w, v = np.linalg.eigh(0.5 * (h + np.swapaxes(h, -1, -2)))
        return np.einsum("...ij,...j,...kj->...ik", v, np.maximum(w, 0.0), v)

    def _grad_hess(self, y: np.ndarray, ctx, project: bool = False) -> tuple[np.ndarray, sp.csr_matrix]:
        n_all = self.ndof
        u = y[:2 * self.n].reshape(self.n, 2)
        tri = self.p.mesh.triangles
        ge, he = self._elem_grad_hess(jnp.asarray(u[tri].reshape(-1, 12)), self._dndx, self._wv,
                                      self._r0, ctx["c"], ctx["b"])
        ge, he = np.asarray(ge), np.asarray(he)
        if project:
            he = self._project_psd(he)
        grad = np.zeros(n_all)
        np.add.at(grad, self._edofs, ge)
        rows, cols, vals = [np.repeat(self._edofs, 12, axis=1).ravel()], [np.tile(self._edofs, (1, 12)).ravel()], [he.ravel()]
        for fn, dofs, bond, anchor, lam in ((self._top_grad_hess, self._tdofs, "bond_top", "anchor_top", "lam_top"),
                                            (self._bot_grad_hess, self._bdofs, "bond_bot", "anchor_bot", "lam_bot")):
            if fn is None:
                continue
            x0 = self.p.mesh.nodes[self._top_contact if dofs is self._tdofs else self._bot_contact]
            w = self._top_w if dofs is self._tdofs else self._bot_w
            gb, hb = fn(jnp.asarray(y[dofs]), jnp.asarray(x0), jnp.asarray(w), ctx[bond], ctx[anchor], ctx[lam])
            gb, hb = np.asarray(gb), np.asarray(hb)
            if project:
                hb = self._project_psd(hb)
            np.add.at(grad, dofs, gb)
            k = dofs.shape[1]
            rows.append(np.repeat(dofs, k, axis=1).ravel())
            cols.append(np.tile(dofs, (1, k)).ravel())
            vals.append(hb.ravel())
        hess = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n_all, n_all))
        # gravity and inertia
        g = self.p.gravity
        grad[1:2 * self.n:2] += g * ctx["mass_z"]
        if self.p.punch:
            grad[-1] += self.p.punch.mass * g
        if self.dynamic:
            d = y - ctx["y_tilde"]
            grad[:2 * self.n] += (self.mass_matrix @ d[:2 * self.n]) / ctx["beta_dt2"]
            if self.p.punch:
                grad[-1] += self.p.punch.mass * d[-1] / ctx["beta_dt2"]
            extra = sp.block_diag([self.mass_matrix / ctx["beta_dt2"]] +
                                  ([sp.csr_matrix([[self.p.punch.mass / ctx["beta_dt2"]]])] if self.p.punch else []))
            hess = hess + extra
        return grad, hess.tocsc()

    # -- stepping -------------------------------------------------------------------------------

    def _context(self, dt: float) -> dict:
        c, b = self._viscous_terms(dt)
        beta = self.beta
        y_n = self._vector(self.u, self.zp)
        v_n = self._vector(self.v, self.vp)
        a_n = self._vector(self.a, self.ap)
        ctx = {"c": c, "b": jnp.asarray(b), "beta_dt2": beta * dt * dt, "mass_z": self._mass_z,
               "y_tilde": y_n + dt * v_n + dt * dt * (0.5 - beta) * a_n,
               "bond_top": jnp.asarray(self._bond_top), "anchor_top": jnp.asarray(self._anchor_top),
               "bond_bot": jnp.asarray(self._bond_bot), "anchor_bot": jnp.asarray(self._anchor_bot),
               "lam_top": jnp.asarray(self._lam_top), "lam_bot": jnp.asarray(self._lam_bot)}
        return ctx

    def solve_step(self, dt: float, tol: float = 1e-8, max_iter: int = 60) -> bool:
        """One implicit step; returns False (state unchanged) if Newton does not converge."""
        ctx = self._context(dt)
        y_n = self._vector(self.u, self.zp)
        y = y_n + dt * self._vector(self.v, self.vp) if self.dynamic else y_n.copy()  # predictor
        y[self.fixed] = y_n[self.fixed]  # axis, base: unchanged
        if self.p.top_displacement is not None:
            # consistent predictor: the linearised response of the free dofs to the prescribed
            # increment (moving only the top row would distort it and send Newton far off)
            prescribed = y.copy()
            prescribed[2 * self._top_nodes + 1] = self.p.top_displacement(self.t + dt)
            dp = (prescribed - y) * self.fixed
            _, hess = self._grad_hess(y, ctx, project=True)
            rhs = -(hess @ dp)[self.free]
            y = prescribed
            y[self.free] += spla.splu(hess[self.free][:, self.free]).solve(rhs)
        scale = max(self.p.mesh.nodes[:, 1].max(), 1e-3)
        e_old = self.energy(y, ctx)
        if not np.isfinite(e_old):  # the extrapolated predictor inverted an element: start from the last state
            y[self.free] = y_n[self.free]
            e_old = self.energy(y, ctx)
        for _ in range(max_iter):
            converged = False
            for project in (False, True):
                grad, hess = self._grad_hess(y, ctx, project)
                g_free = grad[self.free]
                try:
                    step = -spla.splu(hess[self.free][:, self.free]).solve(g_free)
                except RuntimeError:
                    step = np.full_like(g_free, np.nan)
                if np.all(np.isfinite(step)) and np.max(np.abs(step), initial=0.0) < tol * scale:
                    converged = True  # already at the minimum (e.g. free flight: the predictor is exact)
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
            alpha, y, e_old = found
            if np.max(np.abs(step)) < tol * scale:
                break
        else:
            return False
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
        """Backtracking (Armijo) on the incremental potential; None if no decrease is found.

        When the predicted decrease is below the energy's round-off (element terms can be far
        larger than the total), the full step is taken."""
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
        beta, gamma = self.beta, self.gamma
        y_n = self._vector(self.u, self.zp)
        if self.dynamic:
            a_new = (y - ctx["y_tilde"]) / (beta * dt * dt)
            v_new = self._vector(self.v, self.vp) + dt * ((1 - gamma) * self._vector(self.a, self.ap) + gamma * a_new)
        else:
            a_new = np.zeros_like(y)
            v_new = (y - y_n) / dt
        self.u = y[:2 * self.n].reshape(self.n, 2)
        self.v = v_new[:2 * self.n].reshape(self.n, 2)
        self.a = a_new[:2 * self.n].reshape(self.n, 2)
        if self.p.punch:
            self.zp, self.vp, self.ap = float(y[-1]), float(v_new[-1]), float(a_new[-1])
        self.t += dt
        self._update_history(dt)
        self._update_contact_state()

    def _update_history(self, dt: float) -> None:
        pr = self.p.prony
        if pr is None:
            return
        s_new = np.asarray(self._iso_stress(jnp.asarray(self.u[self.p.mesh.triangles].reshape(-1, 12)),
                                            self._dndx, _N_TRI, self._r0))
        if self.dynamic:
            for i, (g, tau) in enumerate(zip(pr.g, pr.tau)):
                x = dt / tau
                self._q[i] = math.exp(-x) * self._q[i] + g * (-math.expm1(-x) / x) * (s_new - self._s_prev)
        self._s_prev = s_new

    def _update_contact_state(self) -> None:
        pc, gc = self.p.punch_contact, self.p.ground_contact
        if self.p.punch:
            pts = np.asarray(self._top_points(jnp.asarray(self._vector(self.u, self.zp)[self._tdofs])))
            gaps = np.asarray(self._gap_fn(jnp.asarray(pts), self.zp))
            self._update_bonds(gaps, pts, pc, self._bond_top, self._anchor_top, self._lam_top)
        if self.p.base == "free":
            pts = np.asarray(self._bot_points(jnp.asarray(self.u.ravel()[self._bdofs])))
            self._update_bonds(pts[..., 1], pts, gc, self._bond_bot, self._anchor_bot, self._lam_bot)

    @staticmethod
    def _update_bonds(gaps, pts, cs: ContactSettings, bond, anchor, lam) -> None:
        if cs.friction > 0 and not cs.stick:  # Coulomb: slip is measured per step, at the pressure of this state
            anchor[:] = pts[..., 0]
            lam[:] = cs.stiffness * np.maximum(-gaps, 0.0)
        new = (gaps <= 0.0) & ~bond
        if cs.stick:
            anchor[new] = pts[..., 0][new]
        bond |= gaps <= 0.0
        # bonds break beyond the adhesive range (or as soon as contact opens, without adhesion)
        limit = 6.0 * cs.adhesion_range if cs.adhesion else 0.0
        bond &= gaps <= limit

    # -- outputs -------------------------------------------------------------------------------

    def punch_force(self) -> float:
        """Vertical force of the body on the punch [N] (contact + adhesion; + = pushing up)."""
        if not self.p.punch:
            return 0.0
        y = self._vector(self.u, self.zp)
        gb, _ = self._top_grad_hess(jnp.asarray(y[self._tdofs]), jnp.asarray(self.p.mesh.nodes[self._top_contact]),
                                    jnp.asarray(self._top_w), jnp.asarray(self._bond_top),
                                    jnp.asarray(self._anchor_top), jnp.asarray(self._lam_top))
        return -float(np.asarray(gb)[:, -1].sum())

    def internal_forces(self) -> np.ndarray:
        """Nodal internal forces (N, 2) of the *current* stress, viscous memory included."""
        pr = self.p.prony
        c = 1.0 - sum(pr.g) if pr is not None else 1.0       # long-term part of the isochoric stress ...
        b = self._q.sum(axis=0) if pr is not None else np.zeros((*self._wv.shape, 3, 3))  # ... plus memory
        if pr is not None and not self.dynamic:
            b = np.zeros_like(b)                              # quasi-static: fully relaxed
        ge, _ = self._elem_grad_hess(jnp.asarray(self.u[self.p.mesh.triangles].reshape(-1, 12)), self._dndx,
                                     self._wv, self._r0, c, jnp.asarray(b))
        f = np.zeros(2 * self.n)
        np.add.at(f, self._edofs, np.asarray(ge))
        return f.reshape(self.n, 2)

    def top_reaction(self) -> float:
        """Vertical force the body exerts on its top-face support [N] (+ = pushing up).

        From the equation of motion of the prescribed dofs: reaction on the body
        R = M a + f_int - f_gravity; the body pushes on the support with -R."""
        top = self._top_nodes
        r = self.internal_forces()[top, 1] + self.p.gravity * self._mass_z[top]
        if self.dynamic:
            r = r + (self.mass_matrix @ self.a.ravel())[2 * top + 1]
        return -float(r.sum())

    def strain_energy(self) -> float:
        c, b = 1.0, jnp.zeros((*self._wv.shape, 3, 3))
        return float(self._elem_energy(jnp.asarray(self.u[self.p.mesh.triangles].reshape(-1, 12)), c, b))

    def kinetic_energy(self) -> float:
        v = self.v.ravel()
        e = 0.5 * float(v @ (self.mass_matrix @ v))
        if self.p.punch:
            e += 0.5 * self.p.punch.mass * self.vp**2
        return e

    def current_nodes(self) -> np.ndarray:
        return self.p.mesh.nodes + self.u
