"""A soft-magnetic elastomer in an applied field: silicone filled with iron powder (MRE), as in Bernat et al.'s gripper.

The filler has no magnetisation of its own: the field induces it. A slender strip magnetises more easily along its length
than through its thickness (its demagnetising factors differ), so a strip across a field feels a torque that turns it along
the field, and in a field gradient it is pulled towards the stronger field. Both come from one energy per unit reference volume,

    w = -mu0 / 2 (chi_a (a.H)^2 + chi_n (n.H)^2 + chi_b (H.H - (a.H)^2 - (n.H)^2)),

with ``a = F a0 / |F a0|`` the strip's material axis as it deforms, ``n = F^-T n0 / |F^-T n0|`` its normal, and ``chi_i`` the
effective susceptibilities along them, ``chi_i = chi / (1 + N_i chi)`` for a composite ``chi`` and demagnetising factors ``N_i``.
``H(x)`` [A/m] is the applied field at the deformed position: a uniform part, a point dipole and any number of permanent
magnets, each an axially magnetised cylinder in closed form (a ring is a cylinder minus its hole; :func:`field`).

**What this leaves out.** Each piece of the body feels the applied field through the shape of the whole, a reduced model: the
field is not solved for, so ``N_i`` stay at their reference values as the strip bends, nothing magnetises anything else
(no finger-to-finger force, no pull on the magnet), and the response is linear, without saturation. It is meant for
fields well below the filler's saturation. Exact derivatives come from JAX.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from softsim.fem import jax_setup
from softsim.magnetism import CYLINDER_SIZE, FIELD_SIZE, MU0, field_parameters, ring  # noqa: F401  (re-exported)
from softsim.materials.models import composite_susceptibility  # noqa: F401  (re-exported: the composite's chi)

jax, jnp = jax_setup()  # float64: the field's elliptic integrals and Newton's Hessian need it

CEL_STEPS = 24  # Bulirsch's iteration converges quadratically: 24 steps settle even next to a magnet's rim (k ~ 1e-6)


@dataclass(frozen=True)
class SoftMagnet:
    """Effective susceptibilities along ``along`` (a0), ``normal`` (n0, perpendicular to it) and across both."""

    chi_along: float
    chi_normal: float
    chi_across: float
    along: tuple[float, float, float] = (1.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def __post_init__(self):
        if abs(float(np.dot(self.along, self.normal))) > 1e-12:
            raise ValueError("along and normal must be perpendicular")


def prism_demagnetizing_factor(a: float, b: float, c: float) -> float:
    """Demagnetising factor along the side ``c`` of a rectangular block of sides ``a, b, c``, exact (Aharoni, J. Appl. Phys.
    83, 3432 (1998), eq. 1, magnetometric): the three factors of a block sum to 1, a cube's are 1/3."""
    a, b, c = 0.5 * a, 0.5 * b, 0.5 * c
    r = np.sqrt(a * a + b * b + c * c)
    ab, bc, ac = np.hypot(a, b), np.hypot(b, c), np.hypot(a, c)
    out = ((b * b - c * c) / (2 * b * c) * np.log((r - a) / (r + a)) + (a * a - c * c) / (2 * a * c) * np.log((r - b) / (r + b))
           + b / (2 * c) * np.log((ab + a) / (ab - a)) + a / (2 * c) * np.log((ab + b) / (ab - b))
           + c / (2 * a) * np.log((bc - b) / (bc + b)) + c / (2 * b) * np.log((ac - a) / (ac + a))
           + 2 * np.arctan(a * b / (c * r)) + (a ** 3 + b ** 3 - 2 * c ** 3) / (3 * a * b * c)
           + (a * a + b * b - 2 * c * c) / (3 * a * b * c) * r + c / (a * b) * (ac + bc)
           - (ab ** 3 + bc ** 3 + ac ** 3) / (3 * a * b * c))
    return float(out / np.pi)


def strip_magnet(chi: float, length: float, normal: float, across: float, along=(1.0, 0.0, 0.0),
                 normal_axis=(0.0, 0.0, 1.0)) -> SoftMagnet:
    """A block of composite susceptibility ``chi`` and sides ``length`` (along ``along``), ``normal`` (along
    ``normal_axis``) and ``across``: each axis's effective susceptibility ``chi / (1 + N chi)`` with the block's own
    demagnetising factor (:func:`prism_demagnetizing_factor`)."""
    n_a = prism_demagnetizing_factor(normal, across, length)
    n_n = prism_demagnetizing_factor(length, across, normal)
    n_b = prism_demagnetizing_factor(length, normal, across)
    return SoftMagnet(chi / (1 + n_a * chi), chi / (1 + n_n * chi), chi / (1 + n_b * chi), tuple(along), tuple(normal_axis))


def cel(kc, p, c, s):
    """Bulirsch's general complete elliptic integral, ``int_0^pi/2 (c cos^2 + s sin^2) / ((cos^2 + p sin^2)
    sqrt(cos^2 + kc^2 sin^2))``, for ``p > 0``: a fixed number of steps of his arithmetic-geometric iteration, so JAX
    can differentiate it (Derby and Olbert, Am. J. Phys. 78, 229 (2010), whose form this follows)."""
    k = jnp.abs(kc)
    pp = jnp.sqrt(p)
    ss = s / pp
    em = jnp.ones_like(k)
    f = c
    cc = c + ss / pp
    g = k / pp
    ss = 2.0 * (ss + f * g)
    pp = g + pp
    g = em
    em = k + em
    kk = k

    def body(_, st):
        k, kk, cc, ss, pp, g, em = st
        k = 2.0 * jnp.sqrt(kk)
        kk = k * em
        f = cc
        cc = cc + ss / pp
        g = kk / pp
        ss = 2.0 * (ss + f * g)
        pp = g + pp
        g = em
        em = k + em
        return k, kk, cc, ss, pp, g, em

    k, kk, cc, ss, pp, g, em = jax.lax.fori_loop(0, CEL_STEPS, body, (k, kk, cc, ss, pp, g, em))
    return 0.5 * jnp.pi * (ss + cc * em) / (em * (em + pp))


def cylinder_field(x, centre, axis, m, a, b, edge=0.0):
    """H [A/m] of an axially magnetised cylinder (magnetisation ``m`` [A/m] along the unit ``axis``, radius ``a``,
    half-length ``b``), exact: Derby and Olbert's closed form gives B = mu0 m / pi times elliptic integrals of the point's
    radius and its height above each end face, and H = B / mu0 outside, B / mu0 - M inside. The inside matters although
    nothing should be there: contact holds nodes out, not quadrature points, and B / mu0 just inside the rim is twice H
    (the magnet's surface current), a false well that pulled a strip through the magnet's skin. With H, the field along
    the rim is continuous across it, as it must be.

    The elliptic modulus ``kc`` is zero only on the rims of the end faces, where the field is infinite (logarithmically);
    near them ``kc`` ~ distance / 2a. ``edge`` > 0 replaces ``kc^2`` by ``kc^2 + (edge / 2a)^2``: the field as if no point
    came closer than about ``edge`` to a rim, as a chamfer makes it: finite there, and for 0.2 mm 1.7% off 1 mm away, 0.6%
    at 2 mm."""
    r = x - centre
    z = jnp.dot(r, axis)
    radial = r - z * axis
    rho2 = jnp.dot(radial, radial)
    rho = jnp.sqrt(rho2 + 1e-30)  # on the axis the radial part vanishes: keep its derivative finite
    gamma = (a - rho) / (a + rho)
    hr, hz = 0.0, 0.0
    for sign, zz in ((1.0, z + b), (-1.0, z - b)):
        root = jnp.sqrt(zz * zz + (a + rho) ** 2)
        alpha, beta = a / root, zz / root
        kc = jnp.sqrt((zz * zz + (a - rho) ** 2) / (zz * zz + (a + rho) ** 2) + (0.5 * edge / a) ** 2)
        hr += sign * alpha * cel(kc, 1.0, 1.0, -1.0)
        hz += sign * beta * cel(kc, gamma * gamma + 1e-30, 1.0, gamma)
    scale = m / jnp.pi  # B0 / mu0
    hz = scale * a / (a + rho) * hz - jnp.where((rho < a) & (jnp.abs(z) < b), m, 0.0)
    hr = scale * hr
    return hz * axis + hr * radial / rho


def field(x, q):
    """Applied field [A/m] at ``x`` for parameters ``q`` (:func:`field_parameters`): uniform, plus a point dipole's
    ``(3 r (m.r) / r^2 - m) / 4 pi r^3`` if its moment is non-zero (a body at the dipole itself has no finite field, so keep
    it away), plus each cylinder magnet's :func:`cylinder_field`."""
    h0, m, p = q[:3], q[3:6], q[6:9]
    r = x - p
    r2 = jnp.dot(r, r)
    safe = jnp.where(jnp.any(m != 0), r2, 1.0)
    h = h0 + (3.0 * r * jnp.dot(m, r) / safe - m) / (4.0 * jnp.pi * safe ** 1.5)
    for i in range((q.shape[0] - FIELD_SIZE) // CYLINDER_SIZE):
        c = q[FIELD_SIZE + CYLINDER_SIZE * i:FIELD_SIZE + CYLINDER_SIZE * (i + 1)]
        h = h + cylinder_field(x, c[:3], c[3:6], c[6], c[7], c[8], c[9])
    return h


def density(f, x, q, material: SoftMagnet):
    """``w`` [J/m^3] of the module docstring, for deformation gradient ``f`` at deformed position ``x``."""
    return density_in(f, field(x, q), material)


def density_in(f, h, material: SoftMagnet):
    """``w`` [J/m^3] for deformation gradient ``f`` in the field ``h`` [A/m]."""
    a, n = f @ jnp.asarray(material.along), jnp.linalg.inv(f).T @ jnp.asarray(material.normal)
    a, n = a / jnp.linalg.norm(a), n / jnp.linalg.norm(n)
    ha, hn = jnp.dot(a, h), jnp.dot(n, h)
    across = jnp.dot(h, h) - ha ** 2 - hn ** 2
    return -0.5 * MU0 * (material.chi_along * ha ** 2 + material.chi_normal * hn ** 2 + material.chi_across * across)


def field_jets(x, q):
    """The field at points ``x`` (..., 3) with its first and second derivatives in space, forward mode: (..., 3),
    (..., 3, 3) and (..., 3, 3, 3), ``[i, j, k] = d2 H_i / dx_j dx_k``."""
    def jets(y):
        h = lambda z: field(z, q)  # noqa: E731
        return h(y), jax.jacfwd(h)(y), jax.jacfwd(jax.jacfwd(h))(y)
    flat = x.reshape(-1, 3)
    h, j, k = jax.vmap(jets)(flat)
    return h.reshape(x.shape), j.reshape(*x.shape, 3), k.reshape(*x.shape, 3, 3)


def element_kernels(material: SoftMagnet, shapes: np.ndarray):
    """(energy summed over elements, per-element gradient and Hessian): each takes displacements (E, 30), rest positions
    (E, 30), shape gradients (E, Q, 10, 3), weights (E, Q) and the field's parameters. ``shapes``: (Q, 10) at the points.

    The energy reads the field itself. The gradient and Hessian read, at each quadrature point, the field's second-order
    expansion about where the point is now (:func:`field_jets`): its value and first two derivatives there are the
    field's, so both are exact, and differentiating it costs a few products instead of the magnet's elliptic integrals
    taken twice in reverse (which was ~4 MB and most of a step per element)."""
    n_q = jnp.asarray(shapes)

    def element(ue, x0, dn, w, q):
        ue, x0 = ue.reshape(10, 3), x0.reshape(10, 3)

        def at_qp(n, d, wq):
            return wq * density(jnp.eye(3) + ue.T @ d, n @ (x0 + ue), q, material)

        return jnp.sum(jax.vmap(at_qp)(n_q, dn, w))

    def expanded(ue, x0, dn, w, hq, jq, kq):
        """The element's energy with the field replaced by its expansion about the quadrature points' current places."""
        ue, x0 = ue.reshape(10, 3), x0.reshape(10, 3)
        here = jax.lax.stop_gradient(n_q @ (x0 + ue))

        def at_qp(n, d, wq, at, h, j, k):
            s = n @ (x0 + ue) - at
            return wq * density_in(jnp.eye(3) + ue.T @ d, h + j @ s + 0.5 * jnp.einsum("ijk,j,k->i", k, s, s), material)

        return jnp.sum(jax.vmap(at_qp)(n_q, dn, w, here, hq, jq, kq))

    def grad_hess(ue, x0, dn, w, q):
        x = jnp.einsum("qn,enc->eqc", n_q, (x0 + ue).reshape(-1, 10, 3))
        hq, jq, kq = field_jets(x, q)
        return jax.vmap(lambda *a: (jax.grad(expanded)(*a), jax.hessian(expanded)(*a)))(ue, x0, dn, w, hq, jq, kq)

    all_elements = (0, 0, 0, 0, None)
    energy = jax.jit(lambda *a: jnp.sum(jax.vmap(element, in_axes=all_elements)(*a)))
    return energy, jax.jit(grad_hess)
