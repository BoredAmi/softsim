"""A soft-magnetic elastomer in an applied field: silicone filled with iron powder (MRE), as in Bernat et al.'s gripper.

The filler has no magnetisation of its own: the field induces it. A slender strip magnetises more easily along its length
than through its thickness (its demagnetising factors differ), so a strip across a field feels a torque that turns it along
the field, and in a field gradient it is pulled towards the stronger field. Both come from one energy per unit reference volume,

    w = -mu0 / 2 (chi_a (a.H)^2 + chi_n (n.H)^2 + chi_b (H.H - (a.H)^2 - (n.H)^2)),

with ``a = F a0 / |F a0|`` the strip's material axis as it deforms, ``n = F^-T n0 / |F^-T n0|`` its normal, and ``chi_i`` the
effective susceptibilities along them, ``chi_i = chi / (1 + N_i chi)`` for a composite ``chi`` and demagnetising factors ``N_i``.
``H(x)`` [A/m] is the applied field at the deformed position, a uniform part plus a point dipole (:func:`field`).

**What this leaves out.** Each piece of the body feels the applied field through the shape of the whole, a reduced model: the
field is not solved for, so ``N_i`` stay at their reference values as the strip bends, nothing magnetises anything else
(no finger-to-finger force, no pull on the magnet), and the response is linear, without saturation. It is meant for
fields well below the filler's saturation. Exact derivatives come from JAX.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

MU0 = 4e-7 * np.pi  # H/m
FIELD_SIZE = 9  # uniform H (3) [A/m], dipole moment (3) [A m^2], dipole position (3) [m]


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


def field(x, q):
    """Applied field [A/m] at ``x`` for parameters ``q``: uniform, plus a point dipole's ``(3 r (m.r) / r^2 - m) / 4 pi r^3``
    if its moment is non-zero (a body at the dipole itself has no finite field, so keep it away)."""
    h0, m, p = q[:3], q[3:6], q[6:9]
    r = x - p
    r2 = jnp.dot(r, r)
    safe = jnp.where(jnp.any(m != 0), r2, 1.0)
    return h0 + (3.0 * r * jnp.dot(m, r) / safe - m) / (4.0 * jnp.pi * safe ** 1.5)


def density(f, x, q, material: SoftMagnet):
    """``w`` [J/m^3] of the module docstring, for deformation gradient ``f`` at deformed position ``x``."""
    a, n = f @ jnp.asarray(material.along), jnp.linalg.inv(f).T @ jnp.asarray(material.normal)
    a, n = a / jnp.linalg.norm(a), n / jnp.linalg.norm(n)
    h = field(x, q)
    ha, hn = jnp.dot(a, h), jnp.dot(n, h)
    across = jnp.dot(h, h) - ha ** 2 - hn ** 2
    return -0.5 * MU0 * (material.chi_along * ha ** 2 + material.chi_normal * hn ** 2 + material.chi_across * across)


def element_kernels(material: SoftMagnet, shapes: np.ndarray):
    """(energy summed over elements, per-element gradient and Hessian): each takes displacements (E, 30), rest positions
    (E, 30), shape gradients (E, Q, 10, 3), weights (E, Q) and the field's parameters. ``shapes``: (Q, 10) at the points."""
    n_q = jnp.asarray(shapes)

    def element(ue, x0, dn, w, q):
        ue, x0 = ue.reshape(10, 3), x0.reshape(10, 3)

        def at_qp(n, d, wq):
            return wq * density(jnp.eye(3) + ue.T @ d, n @ (x0 + ue), q, material)

        return jnp.sum(jax.vmap(at_qp)(n_q, dn, w))

    all_elements = (0, 0, 0, 0, None)
    energy = jax.jit(lambda *a: jnp.sum(jax.vmap(element, in_axes=all_elements)(*a)))
    grad_hess = jax.jit(jax.vmap(lambda *a: (jax.grad(element)(*a), jax.hessian(element)(*a)), in_axes=all_elements))
    return energy, grad_hess
