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
* ``mooney_rivlin``, ``yeoh``, ``ogden``, ``gent``, ``arruda_boyce`` -> the same
  isochoric forms with their ``bulk_modulus``.
* ``prony`` damping -> :class:`Prony` on the isochoric part (instantaneous
  moduli are the given elastic constants, Abaqus convention).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from softsim.errors import UnsupportedFeatureError
from softsim.fem import jax_setup
from softsim.materials.models import (
    ARRUDA_BOYCE_C,
    ArrudaBoyce,
    Gent,
    MagnetoActive,
    Material,
    MooneyRivlin,
    NeoHookean,
    Ogden,
    PronySeries,
    StableNeoHookean,
    Yeoh,
    arruda_boyce_shear_factor,
)

jax, jnp = jax_setup()


def det3(m):
    """Determinant of (batched) 3x3 matrices by cofactors: much cheaper to differentiate in JAX than
    ``jnp.linalg.det``, which goes through an LU factorisation."""
    return (m[..., 0, 0] * (m[..., 1, 1] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 1])
            - m[..., 0, 1] * (m[..., 1, 0] * m[..., 2, 2] - m[..., 1, 2] * m[..., 2, 0])
            + m[..., 0, 2] * (m[..., 1, 0] * m[..., 2, 1] - m[..., 1, 1] * m[..., 2, 0]))


def _isochoric(c):
    """C_bar invariants: (I1_bar, I2_bar, C_bar)."""
    j23 = det3(c) ** (-1.0 / 3.0)
    cb = j23 * c
    i1 = jnp.trace(cb)
    i2 = 0.5 * (i1 * i1 - jnp.trace(cb @ cb))
    return i1, i2, cb


def volumetric_energy(j, kappa):
    """Simo-Taylor volumetric energy kappa/4 (J^2 - 1 - 2 ln J): convex, U''(1) = kappa."""
    return 0.25 * kappa * (j * j - 1.0 - 2.0 * jnp.log(j))


class IsochoricLaw:
    """Isochoric strain energy per reference volume, ``energy(C)`` with C the 3x3 right Cauchy-Green tensor."""

    shear_modulus: float

    def energy(self, c):  # pragma: no cover - interface
        raise NotImplementedError


class InvariantLaw(IsochoricLaw):
    """A law psi(I1_bar, I2_bar): the 3D solver then builds its tangent in closed form and only
    differentiates this scalar function (``uses_i2 = False`` skips the I2 terms)."""

    uses_i2: ClassVar[bool] = False

    def invariant_energy(self, i1, i2):  # pragma: no cover - interface
        raise NotImplementedError

    def energy(self, c):
        i1, i2, _ = _isochoric(c)
        return self.invariant_energy(i1, i2)


@dataclass(frozen=True)
class NeoHookeanIso(InvariantLaw):
    shear_modulus: float

    def invariant_energy(self, i1, i2):
        return 0.5 * self.shear_modulus * (i1 - 3.0)


@dataclass(frozen=True)
class MooneyRivlinIso(InvariantLaw):
    c10: float
    c01: float

    @property
    def shear_modulus(self) -> float:
        return 2.0 * (self.c10 + self.c01)

    uses_i2: ClassVar[bool] = True

    def invariant_energy(self, i1, i2):
        return self.c10 * (i1 - 3.0) + self.c01 * (i2 - 3.0)


@dataclass(frozen=True)
class YeohIso(InvariantLaw):
    c10: float
    c20: float = 0.0
    c30: float = 0.0

    @property
    def shear_modulus(self) -> float:
        return 2.0 * self.c10

    def invariant_energy(self, i1, i2):
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


def _sqrtm_spd(x, iters: int = 14):
    """Square root of a symmetric positive definite 3x3 matrix (Denman-Beavers iteration: products and
    inverses only, so it differentiates cleanly through repeated eigenvalues)."""
    def body(_, yz):
        y, z = yz
        return 0.5 * (y + jnp.linalg.inv(z)), 0.5 * (z + jnp.linalg.inv(y))

    return jax.lax.fori_loop(0, iters, body, (x, jnp.eye(3)))[0]


def _logm_spd(x, roots: int = 6, terms: int = 10):
    """log of an SPD matrix: 2^roots log(x^(1/2^roots)), with the series of log(I + e) for the root."""
    x = jax.lax.fori_loop(0, roots, lambda _, y: _sqrtm_spd(y), x)
    e = x - jnp.eye(3)
    out, power = jnp.zeros((3, 3)), jnp.eye(3)
    for k in range(1, terms + 1):
        power = power @ e
        out = out + ((-1) ** (k + 1) / k) * power
    return (2.0 ** roots) * out


def _expm_sym(m, squarings: int = 6, terms: int = 12):
    m = m / 2.0 ** squarings
    out, power = jnp.eye(3), jnp.eye(3)
    for k in range(1, terms + 1):
        power = power @ m / k
        out = out + power
    return jax.lax.fori_loop(0, squarings, lambda _, y: y @ y, out)


def trace_power(c, p: float):
    """tr(C^p) of an SPD matrix through matrix functions (smooth where eigenvalues coincide)."""
    return jnp.trace(_expm_sym(p * _logm_spd(c)))


@dataclass(frozen=True)
class OgdenGeneralIso(IsochoricLaw):
    """The Ogden law for a general (3D) C: sum_k mu_k / alpha_k (tr(C_bar^(alpha_k / 2)) - 3).

    Same energy as :class:`OgdenIso`, without its axisymmetric block assumption.
    """

    mu: tuple[float, ...]
    alpha: tuple[float, ...]

    @property
    def shear_modulus(self) -> float:
        return 0.5 * sum(m * a for m, a in zip(self.mu, self.alpha))

    def energy(self, c):
        _, _, cb = _isochoric(c)
        log_cb = _logm_spd(cb)
        return sum(m / al * (jnp.trace(_expm_sym(0.5 * al * log_cb)) - 3.0) for m, al in zip(self.mu, self.alpha))


@dataclass(frozen=True)
class GentIso(InvariantLaw):
    """-(mu J_m / 2) ln(1 - (I1_bar - 3) / J_m); infinite beyond the extensibility limit."""

    mu: float
    jm: float

    @property
    def shear_modulus(self) -> float:
        return self.mu

    def invariant_energy(self, i1, i2):
        x = 1.0 - (i1 - 3.0) / self.jm
        return jnp.where(x > 0, -0.5 * self.mu * self.jm * jnp.log(jnp.where(x > 0, x, 1.0)), jnp.inf)


@dataclass(frozen=True)
class ArrudaBoyceIso(InvariantLaw):
    """mu sum_i C_i / lambda_m^(2i-2) (I1_bar^i - 3^i), five-term series."""

    mu: float
    lambda_m: float

    @property
    def shear_modulus(self) -> float:
        return self.mu * arruda_boyce_shear_factor(self.lambda_m)

    def invariant_energy(self, i1, i2):
        return self.mu * sum(ci / self.lambda_m ** (2 * k - 2) * (i1**k - 3.0**k)
                             for k, ci in enumerate(ARRUDA_BOYCE_C, 1))


@dataclass(frozen=True)
class Prony:
    """Isochoric relaxation G(t) = G0 (1 - sum g_i (1 - exp(-t / tau_i)))."""

    g: tuple[float, ...]
    tau: tuple[float, ...]


def from_material(material: Material) -> tuple[IsochoricLaw, float, Prony | None]:
    """(isochoric law, bulk modulus, Prony series or None) for a softsim material; a magneto-active one's are its
    matrix's (its magnetic part is softsim.fem.magnetic's, see :func:`softsim.backends.fem.backend`)."""
    e = material.elastic
    if isinstance(e, MagnetoActive):
        e = e.matrix
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
    elif isinstance(e, Gent):
        law, kappa = GentIso(e.mu, e.jm), e.bulk_modulus
    elif isinstance(e, ArrudaBoyce):
        law, kappa = ArrudaBoyceIso(e.mu, e.lambda_m), e.bulk_modulus
    else:
        raise UnsupportedFeatureError("reference FEM", [f"{e.model!r} elasticity (hyperelastic laws only)"])
    damping = material.damping
    prony = None
    if isinstance(damping, PronySeries):
        prony = Prony(tuple(damping.g), tuple(damping.tau))
    elif damping is not None:
        raise UnsupportedFeatureError("reference FEM", [f"{damping.model!r} damping (dissipation is a Prony series)"])
    return law, kappa, prony


# Laws are pytrees: their parameters can be traced, so jitted code (homogeneous tests, fits)
# compiles once per law type instead of once per parameter value.
for _cls, _fields in ((NeoHookeanIso, ["shear_modulus"]), (MooneyRivlinIso, ["c10", "c01"]),
                      (YeohIso, ["c10", "c20", "c30"]), (OgdenIso, ["mu", "alpha"]),
                      (OgdenGeneralIso, ["mu", "alpha"]), (GentIso, ["mu", "jm"]),
                      (ArrudaBoyceIso, ["mu", "lambda_m"])):
    jax.tree_util.register_dataclass(_cls, data_fields=_fields, meta_fields=[])
