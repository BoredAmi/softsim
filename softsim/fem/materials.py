"""Hyperelastic and viscoelastic laws for the reference FEM (JAX-traceable).

Every law is an isochoric strain energy W_iso(C) written in terms of
C_bar = J^(-2/3) C (J = sqrt(det C)), so it carries no volume change; the
volumetric part is handled by the solver (mean dilatation, bulk modulus).
:func:`from_material` maps a softsim :class:`~softsim.materials.Material`:

* ``neo_hookean`` and ``stable_neo_hookean`` -> split Neo-Hookean
  mu/2 (I1_bar - 3) with mu = E / (2 (1 + nu)), kappa = E / (3 (1 - 2 nu)).
  Identical at small strain; at large strain the classic ln J form and the
  Stable Neo-Hookean form differ from it only through their volumetric
  terms, which are small for nearly incompressible solids.
* ``mooney_rivlin``, ``yeoh``, ``ogden`` -> the same isochoric forms with their
  ``bulk_modulus``.
* ``prony`` damping -> :class:`Prony` on the isochoric part (instantaneous
  moduli are the given elastic constants, Abaqus convention).
"""

from __future__ import annotations

from dataclasses import dataclass

from softsim.errors import UnsupportedFeatureError
from softsim.fem import jax_setup
from softsim.materials.models import (
    Material,
    MooneyRivlin,
    NeoHookean,
    Ogden,
    PronySeries,
    StableNeoHookean,
    Yeoh,
)

jax, jnp = jax_setup()


def _isochoric(c):
    """C_bar invariants: (I1_bar, I2_bar, C_bar)."""
    j23 = jnp.linalg.det(c) ** (-1.0 / 3.0)
    cb = j23 * c
    i1 = jnp.trace(cb)
    i2 = 0.5 * (i1 * i1 - jnp.trace(cb @ cb))
    return i1, i2, cb


class IsochoricLaw:
    """Isochoric strain energy per reference volume, ``energy(C)`` with C the 3x3 right Cauchy-Green tensor."""

    shear_modulus: float

    def energy(self, c):  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class NeoHookeanIso(IsochoricLaw):
    shear_modulus: float

    def energy(self, c):
        i1, _, _ = _isochoric(c)
        return 0.5 * self.shear_modulus * (i1 - 3.0)


@dataclass(frozen=True)
class MooneyRivlinIso(IsochoricLaw):
    c10: float
    c01: float

    @property
    def shear_modulus(self) -> float:
        return 2.0 * (self.c10 + self.c01)

    def energy(self, c):
        i1, i2, _ = _isochoric(c)
        return self.c10 * (i1 - 3.0) + self.c01 * (i2 - 3.0)


@dataclass(frozen=True)
class YeohIso(IsochoricLaw):
    c10: float
    c20: float = 0.0
    c30: float = 0.0

    @property
    def shear_modulus(self) -> float:
        return 2.0 * self.c10

    def energy(self, c):
        i1, _, _ = _isochoric(c)
        x = i1 - 3.0
        return self.c10 * x + self.c20 * x * x + self.c30 * x * x * x


@dataclass(frozen=True)
class OgdenIso(IsochoricLaw):
    """sum_k mu_k / alpha_k (lambda1^a + lambda2^a + lambda3^a - 3) of the isochoric stretches.

    Principal stretches come from the eigenvalues of C_bar; a tiny shift keeps
    their derivatives finite when two stretches coincide (e.g. at rest).
    """

    mu: tuple[float, ...]
    alpha: tuple[float, ...]

    @property
    def shear_modulus(self) -> float:
        return 0.5 * sum(m * a for m, a in zip(self.mu, self.alpha))

    def energy(self, c):
        _, _, cb = _isochoric(c)
        # axisymmetric C: block diag(2x2 in r-z, hoop); eigenvalues of the 2x2 block in closed form
        a, b, d = cb[0, 0], cb[0, 1], cb[1, 1]
        mean = 0.5 * (a + d)
        rad = jnp.sqrt(0.25 * (a - d) ** 2 + b * b + 1e-24)
        lam2 = jnp.stack([mean + rad, mean - rad, cb[2, 2]])  # squared stretches
        return sum(m / al * (jnp.sum(lam2 ** (0.5 * al)) - 3.0) for m, al in zip(self.mu, self.alpha))


@dataclass(frozen=True)
class Prony:
    """Isochoric relaxation G(t) = G0 (1 - sum g_i (1 - exp(-t / tau_i)))."""

    g: tuple[float, ...]
    tau: tuple[float, ...]


def from_material(material: Material) -> tuple[IsochoricLaw, float, Prony | None]:
    """(isochoric law, bulk modulus, Prony series or None) for a softsim material."""
    e = material.elastic
    if isinstance(e, (NeoHookean, StableNeoHookean)):
        mu = e.youngs_modulus / (2.0 * (1.0 + e.poisson_ratio))
        kappa = e.youngs_modulus / (3.0 * (1.0 - 2.0 * e.poisson_ratio))
        law: IsochoricLaw = NeoHookeanIso(mu)
    elif isinstance(e, MooneyRivlin):
        law, kappa = MooneyRivlinIso(e.c10, e.c01), e.bulk_modulus
    elif isinstance(e, Yeoh):
        law, kappa = YeohIso(e.c10, e.c20, e.c30), e.bulk_modulus
    elif isinstance(e, Ogden):
        law, kappa = OgdenIso(tuple(e.mu), tuple(e.alpha)), e.bulk_modulus
    else:
        raise UnsupportedFeatureError(f"reference FEM has no {e.model!r} law (hyperelastic laws only)")
    damping = material.damping
    prony = None
    if isinstance(damping, PronySeries):
        prony = Prony(tuple(damping.g), tuple(damping.tau))
    elif damping is not None:
        raise UnsupportedFeatureError(f"reference FEM models dissipation as a Prony series, not {damping.model!r}")
    return law, kappa, prony
