"""Material models as validated, backend-neutral data.

A :class:`Material` combines density, an elastic constitutive model, optional
damping, surface friction and optional surface adhesion (tack). Each model declares the capabilities it needs
(``requires``) so backends can reject what they cannot simulate.

A material is also a *card*: ``kind`` says whether its numbers describe the
physical material (``physical``, backend independent) or are effective values
of one backend, mesh and contact model (``effective``, e.g. the output of a
drop-test calibration); ``validity``, ``uncertainty`` and ``references`` say
where the numbers hold, how well they are known and where they come from.

All elastic models expose small-strain equivalents (``initial_youngs_modulus``,
``initial_poisson_ratio``, ``initial_shear_modulus``, ``initial_bulk_modulus``).
These are used by stability checks and analytic tests, never as a silent
substitute for the requested model. Isotropic E/nu models accept any two of
``youngs_modulus``, ``poisson_ratio``, ``shear_modulus``, ``bulk_modulus``, and
the hyperelastic families ``poisson_ratio`` instead of ``bulk_modulus``; the
card stores the canonical pair.
"""

from __future__ import annotations

import math
from typing import Annotated, Any, ClassVar, Iterable, Literal, Union

import numpy as np
from pydantic import Field, model_validator

from softsim.capabilities import Capability
from softsim.config.schema import StrictModel, validate
from softsim.config.units import (
    Density, FieldStrength, Length, Pressure, Rate, StrainRate, Temperature, Time, Viscosity, parse_quantity,
)

PoissonRatio = Annotated[float, Field(gt=-1.0, lt=0.5)]
NonNegative = Annotated[float, Field(ge=0.0)]


_ISOTROPIC_CONSTANTS = ("youngs_modulus", "poisson_ratio", "shear_modulus", "bulk_modulus")


def isotropic_e_nu(given: dict[str, float]) -> tuple[float, float]:
    """(E, nu) from any two of youngs_modulus, poisson_ratio, shear_modulus, bulk_modulus (SI)."""
    e, nu, mu, k = (given.get(n) for n in _ISOTROPIC_CONSTANTS)
    if e is not None and nu is not None:
        return e, nu
    if mu is not None and nu is not None:
        return 2.0 * mu * (1.0 + nu), nu
    if k is not None and nu is not None:
        return 3.0 * k * (1.0 - 2.0 * nu), nu
    if e is not None and mu is not None:
        return e, e / (2.0 * mu) - 1.0
    if e is not None and k is not None:
        return e, (3.0 * k - e) / (6.0 * k)
    if mu is not None and k is not None:
        return 9.0 * k * mu / (3.0 * k + mu), (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))
    raise ValueError(f"give exactly two of {', '.join(_ISOTROPIC_CONSTANTS)}")


def _elastic_constant(name: str, raw: Any) -> float:
    return float(raw) if name == "poisson_ratio" else parse_quantity(raw, "pressure")


class _Elastic(StrictModel):
    """Small-strain equivalents shared by all elastic models."""

    def initial_shear_modulus(self) -> float:
        raise NotImplementedError

    def initial_bulk_modulus(self) -> float:
        raise NotImplementedError

    def initial_youngs_modulus(self) -> float:
        mu, k = self.initial_shear_modulus(), self.initial_bulk_modulus()
        return 9.0 * k * mu / (3.0 * k + mu)

    def initial_poisson_ratio(self) -> float:
        mu, k = self.initial_shear_modulus(), self.initial_bulk_modulus()
        return (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))


class _IsotropicEnu(_Elastic):
    """Models parametrised by Young's modulus and Poisson's ratio (or any two isotropic constants)."""

    youngs_modulus: Annotated[Pressure, Field(gt=0)]
    poisson_ratio: PoissonRatio

    @model_validator(mode="before")
    @classmethod
    def _from_any_two(cls, data: Any) -> Any:
        if not isinstance(data, dict) or not {"shear_modulus", "bulk_modulus"} & data.keys():
            return data
        data = dict(data)
        given = {n: _elastic_constant(n, data.pop(n)) for n in _ISOTROPIC_CONSTANTS if n in data}
        if len(given) != 2:
            raise ValueError(f"give exactly two of {', '.join(_ISOTROPIC_CONSTANTS)}, got {', '.join(given)}")
        data["youngs_modulus"], data["poisson_ratio"] = isotropic_e_nu(given)
        return data

    def initial_youngs_modulus(self) -> float:
        return self.youngs_modulus

    def initial_poisson_ratio(self) -> float:
        return self.poisson_ratio

    def initial_shear_modulus(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))

    def initial_bulk_modulus(self) -> float:
        return self.youngs_modulus / (3.0 * (1.0 - 2.0 * self.poisson_ratio))


class LinearElastic(_IsotropicEnu):
    """Small-strain linear elasticity (Hooke). Only valid for strains of a few percent."""

    model: Literal["linear"] = "linear"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.LINEAR_ELASTIC})


class StVenantKirchhoff(_IsotropicEnu):
    """Green-Lagrange strain with linear stress-strain law.

    Handles large rotations but softens and can collapse under compression
    beyond roughly 30-40 % strain.
    """

    model: Literal["stvk"] = "stvk"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.STVK})


class NeoHookean(_IsotropicEnu):
    """Compressible Neo-Hookean, parametrised by its small-strain E and nu.

    mu = E / (2 (1 + nu)), lambda = E nu / ((1 + nu)(1 - 2 nu)).
    """

    model: Literal["neo_hookean"] = "neo_hookean"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.NEO_HOOKEAN})

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))


class StableNeoHookean(_IsotropicEnu):
    """Stable Neo-Hookean (Smith, de Goes, Kim 2018), parametrised by its small-strain E and nu.

    Same linearisation as :class:`NeoHookean` but a different energy at large
    strain; unlike the classic ln(J) form it stays well defined for inverted
    elements, which matters for violent impacts. MuJoCo's volumetric flex
    "Neo-Hookean" option is this model.
    """

    model: Literal["stable_neo_hookean"] = "stable_neo_hookean"
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.STABLE_NEO_HOOKEAN})

    @property
    def shear_modulus(self) -> float:
        return self.youngs_modulus / (2.0 * (1.0 + self.poisson_ratio))


class _BulkHyperelastic(_Elastic):
    """Hyperelastic models with an explicit bulk modulus for the volumetric part.

    ``poisson_ratio`` may be given instead of ``bulk_modulus``; it is converted
    with the law's initial shear modulus (K = 2 mu0 (1 + nu) / (3 (1 - 2 nu))).
    """

    bulk_modulus: Annotated[Pressure, Field(gt=0)]

    @model_validator(mode="before")
    @classmethod
    def _bulk_from_poisson(cls, data: Any) -> Any:
        if not isinstance(data, dict) or "poisson_ratio" not in data:
            return data
        data = dict(data)
        nu = float(data.pop("poisson_ratio"))
        if "bulk_modulus" in data:
            raise ValueError("give bulk_modulus or poisson_ratio, not both")
        if not -1.0 < nu < 0.5:
            raise ValueError(f"poisson_ratio must be in (-1, 0.5), got {nu}")
        try:
            mu = cls._raw_shear_modulus(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"poisson_ratio needs the law's shear parameters to give bulk_modulus ({exc})") from None
        data["bulk_modulus"] = 2.0 * mu * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))
        return data

    @classmethod
    def _raw_shear_modulus(cls, data: dict[str, Any]) -> float:
        """Initial shear modulus from unvalidated parameters (for the poisson_ratio input)."""
        raise NotImplementedError

    def initial_bulk_modulus(self) -> float:
        return self.bulk_modulus


class MooneyRivlin(_BulkHyperelastic):
    """Two-term Mooney-Rivlin: W = C10 (I1 - 3) + C01 (I2 - 3) + volumetric."""

    model: Literal["mooney_rivlin"] = "mooney_rivlin"
    c10: Pressure
    c01: Pressure
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.MOONEY_RIVLIN})

    def initial_shear_modulus(self) -> float:
        return 2.0 * (self.c10 + self.c01)

    @classmethod
    def _raw_shear_modulus(cls, data: dict[str, Any]) -> float:
        return 2.0 * (_p(data["c10"]) + _p(data["c01"]))

    @model_validator(mode="after")
    def _positive_shear(self) -> "MooneyRivlin":
        if self.c10 + self.c01 <= 0:
            raise ValueError("c10 + c01 must be positive")
        return self


class Yeoh(_BulkHyperelastic):
    """Yeoh: W = C10 (I1 - 3) + C20 (I1 - 3)^2 + C30 (I1 - 3)^3 + volumetric.

    Usually the best simple fit for Ecoflex-type silicones at large strain.
    """

    model: Literal["yeoh"] = "yeoh"
    c10: Annotated[Pressure, Field(gt=0)]
    c20: Pressure = 0.0
    c30: Pressure = 0.0
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.YEOH})

    def initial_shear_modulus(self) -> float:
        return 2.0 * self.c10

    @classmethod
    def _raw_shear_modulus(cls, data: dict[str, Any]) -> float:
        return 2.0 * _p(data["c10"])


class Ogden(_BulkHyperelastic):
    """N-term Ogden model with shear moduli ``mu`` and exponents ``alpha``."""

    model: Literal["ogden"] = "ogden"
    mu: list[Pressure] = Field(min_length=1)
    alpha: list[float] = Field(min_length=1)
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.OGDEN})

    def initial_shear_modulus(self) -> float:
        return 0.5 * sum(m * a for m, a in zip(self.mu, self.alpha))

    @classmethod
    def _raw_shear_modulus(cls, data: dict[str, Any]) -> float:
        return 0.5 * sum(_p(m) * float(a) for m, a in zip(data["mu"], data["alpha"]))

    @model_validator(mode="after")
    def _matching_terms(self) -> "Ogden":
        if len(self.mu) != len(self.alpha):
            raise ValueError("mu and alpha must have the same length")
        if self.initial_shear_modulus() <= 0:
            raise ValueError("initial shear modulus 0.5*sum(mu*alpha) must be positive")
        return self


class Gent(_BulkHyperelastic):
    """Gent: W = -(mu J_m / 2) ln(1 - (I1 - 3) / J_m) + volumetric.

    Neo-Hookean with shear modulus ``mu`` at small strain, stiffening without
    bound as I1 - 3 approaches the limiting extensibility ``jm`` (finite chain
    length). A two-parameter alternative to Yeoh for silicones that stiffen at
    large stretch.
    """

    model: Literal["gent"] = "gent"
    mu: Annotated[Pressure, Field(gt=0)]
    jm: Annotated[float, Field(gt=0)]
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.GENT})

    def initial_shear_modulus(self) -> float:
        return self.mu

    @classmethod
    def _raw_shear_modulus(cls, data: dict[str, Any]) -> float:
        return _p(data["mu"])


ARRUDA_BOYCE_C = (1 / 2, 1 / 20, 11 / 1050, 19 / 7000, 519 / 673750)


def arruda_boyce_shear_factor(lambda_m: float) -> float:
    """Initial shear modulus over ``mu`` for the five-term Arruda-Boyce series."""
    return sum(2.0 * i * c * 3.0 ** (i - 1) / lambda_m ** (2 * i - 2) for i, c in enumerate(ARRUDA_BOYCE_C, 1))


class ArrudaBoyce(_BulkHyperelastic):
    """Arruda-Boyce eight-chain model, five-term series (the Abaqus form)::

        W = mu * sum_i C_i / lambda_m^(2i-2) * (I1^i - 3^i) + volumetric,
        C = 1/2, 1/20, 11/1050, 19/7000, 519/673750

    ``lambda_m`` is the locking (chain) stretch. The initial shear modulus is
    ``mu`` times 1 + 3/(5 lambda_m^2) + ... (``initial_shear_modulus``), close
    to ``mu`` for large ``lambda_m``.
    """

    model: Literal["arruda_boyce"] = "arruda_boyce"
    mu: Annotated[Pressure, Field(gt=0)]
    lambda_m: Annotated[float, Field(gt=1)]
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.ARRUDA_BOYCE})

    def initial_shear_modulus(self) -> float:
        return self.mu * arruda_boyce_shear_factor(self.lambda_m)

    @classmethod
    def _raw_shear_modulus(cls, data: dict[str, Any]) -> float:
        return _p(data["mu"]) * arruda_boyce_shear_factor(float(data["lambda_m"]))


def _p(raw: Any) -> float:
    return parse_quantity(raw, "pressure")


HyperelasticMatrix = Annotated[
    Union[NeoHookean, StableNeoHookean, MooneyRivlin, Yeoh, Ogden, Gent, ArrudaBoyce], Field(discriminator="model")
]


def composite_susceptibility(volume_fraction: float) -> float:
    """Susceptibility of an elastomer filled with a volume fraction of iron particles, Maxwell Garnett's mixing rule for
    spheres far more permeable than the matrix: ``3 phi / (1 - phi)``. A dilute estimate, meant for the 10-30% of
    iron-powder MRE; chains of particles (an MRE cured in a field) are more permeable along them."""
    return 3.0 * volume_fraction / (1.0 - volume_fraction)


class MagnetoActive(StrictModel):
    """Soft-magnetic elastomer (MRE): a hyperelastic matrix filled with iron particles that the applied field magnetises.

    The magnetic energy per unit reference volume is ``-mu0 / 2 (chi_a (a.H)^2 + chi_n (n.H)^2 + chi_b (...)^2)`` with the
    material axes ``along`` and ``normal`` (body frame) carried by the deformation (softsim.fem.magnetic). Each axis's
    effective susceptibility is ``chi / (1 + N chi)``: ``chi`` is the composite's (``susceptibility``, or Maxwell Garnett's
    ``3 phi / (1 - phi)`` from ``particle_volume_fraction``), ``N`` the body's demagnetising factor along the axis, by
    default that of the rectangular block bounding the body at rest along ``along``, ``normal`` and across (exact,
    Aharoni 1998: right for strips and plates, approximate for other shapes); or give ``demagnetization`` as
    ``[N_along, N_normal, N_across]``. With ``saturation_magnetization`` M_s > 0 the induced magnetisation saturates,
    ``M = M_s tanh(chi H / M_s)`` along the easiest axis (softsim.fem.magnetic); 0 (the default) keeps the linear law,
    right while ``chi H`` is well below the composite's saturation. ``remanent_magnetization`` [A/m, body frame] is a
    fixed magnetisation from hard-magnetic particles (NdFeB), carried by the deformation: energy ``-mu0 (F M_r) . H``
    (Zhao et al. 2019); a purely hard-magnetic composite has ``susceptibility: 0``. The field comes from the scene's
    ``magnetic_field`` (softsim.magnetism).
    """

    model: Literal["magneto_active"] = "magneto_active"
    matrix: HyperelasticMatrix
    susceptibility: Annotated[float, Field(ge=0)] | None = None
    particle_volume_fraction: Annotated[float, Field(ge=0, lt=1)] | None = None
    saturation_magnetization: Annotated[FieldStrength, Field(ge=0)] = 0.0  # A/m (units allowed: "400 kA/m")
    remanent_magnetization: tuple[FieldStrength, FieldStrength, FieldStrength] = (0.0, 0.0, 0.0)  # A/m, body frame
    demagnetization: Literal["block"] | tuple[Annotated[float, Field(ge=0, le=1)], Annotated[float, Field(ge=0, le=1)],
                                              Annotated[float, Field(ge=0, le=1)]] = "block"
    along: tuple[float, float, float] = (1.0, 0.0, 0.0)
    normal: tuple[float, float, float] = (0.0, 0.0, 1.0)
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.MAGNETO_ACTIVE})

    @model_validator(mode="after")
    def _check(self) -> "MagnetoActive":
        if (self.susceptibility is None) == (self.particle_volume_fraction is None):
            raise ValueError("give exactly one of susceptibility and particle_volume_fraction")
        a, n = np.asarray(self.along, float), np.asarray(self.normal, float)
        if np.linalg.norm(a) == 0 or np.linalg.norm(n) == 0:
            raise ValueError("along and normal must be non-zero")
        if abs(a @ n) > 1e-9 * np.linalg.norm(a) * np.linalg.norm(n):
            raise ValueError("along and normal must be perpendicular")
        return self

    @property
    def chi(self) -> float:
        """The composite's susceptibility (before demagnetisation)."""
        if self.susceptibility is not None:
            return self.susceptibility
        return composite_susceptibility(self.particle_volume_fraction)

    def initial_youngs_modulus(self) -> float:
        return self.matrix.initial_youngs_modulus()

    def initial_poisson_ratio(self) -> float:
        return self.matrix.initial_poisson_ratio()

    def initial_shear_modulus(self) -> float:
        return self.matrix.initial_shear_modulus()

    def initial_bulk_modulus(self) -> float:
        return self.matrix.initial_bulk_modulus()


ElasticModel = Annotated[
    Union[LinearElastic, StVenantKirchhoff, NeoHookean, StableNeoHookean, MooneyRivlin, Yeoh, Ogden, Gent,
          ArrudaBoyce, MagnetoActive],
    Field(discriminator="model"),
]


class RayleighDamping(StrictModel):
    """Rayleigh damping C = alpha M + beta K.

    ``beta`` (seconds) damps high-frequency/strain-rate motion, ``alpha`` (1/s)
    damps rigid-like motion. Note this is a crude stand-in for silicone
    viscoelasticity: the loss factor grows linearly with frequency.
    """

    model: Literal["rayleigh"] = "rayleigh"
    alpha: Annotated[Rate, Field(ge=0)] = 0.0
    beta: Annotated[Time, Field(ge=0)] = 0.0

    @property
    def requires(self) -> frozenset[Capability]:
        req = set()
        if self.alpha > 0:
            req.add(Capability.DAMPING_MASS_PROPORTIONAL)
        if self.beta > 0:
            req.add(Capability.DAMPING_STIFFNESS_PROPORTIONAL)
        return frozenset(req)


class KelvinVoigt(StrictModel):
    """Kelvin-Voigt solid: sigma = sigma_elastic + eta * d(epsilon)/dt.

    Equivalent to stiffness-proportional damping with beta = eta / E, which is
    how backends map it.
    """

    model: Literal["kelvin_voigt"] = "kelvin_voigt"
    viscosity: Annotated[Viscosity, Field(ge=0)]
    requires: ClassVar[frozenset[Capability]] = frozenset(
        {Capability.DAMPING_STIFFNESS_PROPORTIONAL}
    )


class PronySeries(StrictModel):
    """Generalised Maxwell (Prony series) relaxation: G(t) = G0 (1 - sum g_i (1 - exp(-t/tau_i)))."""

    model: Literal["prony"] = "prony"
    g: list[Annotated[float, Field(gt=0, lt=1)]] = Field(min_length=1)
    tau: list[Annotated[Time, Field(gt=0)]] = Field(min_length=1)
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.VISCOELASTIC_PRONY})

    @model_validator(mode="after")
    def _matching_terms(self) -> "PronySeries":
        if len(self.g) != len(self.tau):
            raise ValueError("g and tau must have the same length")
        if sum(self.g) >= 1:
            raise ValueError("sum(g) must be < 1")
        return self


DampingModel = Annotated[
    Union[RayleighDamping, KelvinVoigt, PronySeries], Field(discriminator="model")
]


class Friction(StrictModel):
    """Coulomb friction coefficients (sliding [-], torsional [m], rolling [m])."""

    sliding: NonNegative = 0.8
    torsional: NonNegative = 0.0
    rolling: NonNegative = 0.0

    @property
    def requires(self) -> frozenset[Capability]:
        req = {Capability.CONTACT_FRICTION} if self.sliding > 0 else set()
        if self.torsional > 0 or self.rolling > 0:
            req.add(Capability.TORSIONAL_ROLLING_FRICTION)
        return frozenset(req)


class Adhesion(StrictModel):
    """Normal adhesion (tack) between a soft surface and what it touches.

    Exponential cohesive law per unit (reference) surface area, for a gap
    ``g >= 0`` opened after the surface has touched the counterpart::

        t(g) = strength * (g / range) * exp(1 - g / range)

    ``strength`` is the peak tensile traction (reached at ``g = range``); the
    work of separation is ``e * strength * range`` per area. A bond forms when
    a surface point comes into contact (gap below ``BOND_GAPS * range``, a
    round-off tolerance) and breaks for good once the gap exceeds
    ``BREAK_GAPS * range`` (where the traction has dropped to 4% of the peak),
    until the next contact. Elastic while bonded, so it adds no damping; the
    energy still stored at breakage is the only loss. Rate-dependent
    (viscoelastic) tack is not modelled.
    """

    BOND_GAPS: ClassVar[float] = 0.01
    BREAK_GAPS: ClassVar[float] = 6.0

    strength: Annotated[Pressure, Field(gt=0)]
    range: Annotated[Length, Field(gt=0)]

    @property
    def work_of_separation(self) -> float:
        """Energy per unit area to separate a bond completely [J/m^2]."""
        return math.e * self.strength * self.range

    @property
    def max_stiffness(self) -> float:
        """Largest slope dt/dg of the traction law (at g = 0) [Pa/m]."""
        return math.e * self.strength / self.range

    @property
    def requires(self) -> frozenset[Capability]:
        return frozenset({Capability.CONTACT_ADHESION})


class Validity(StrictModel):
    """Where a card's numbers were measured or fitted; runs outside are warned about.

    ``strain`` is the nominal uniaxial strain range of the data (negative =
    compression), ``strain_rate`` its range of strain rates, ``temperature``
    the test temperature.
    """

    strain: tuple[float, float] | None = None
    strain_rate: tuple[Annotated[StrainRate, Field(ge=0)], Annotated[StrainRate, Field(ge=0)]] | None = None
    temperature: Annotated[Temperature, Field(gt=0)] | None = None

    @model_validator(mode="after")
    def _ordered(self) -> "Validity":
        for name in ("strain", "strain_rate"):
            lo_hi = getattr(self, name)
            if lo_hi is not None and lo_hi[0] > lo_hi[1]:
                raise ValueError(f"{name} range must be [low, high], got {list(lo_hi)}")
        return self


class ParameterUncertainty(StrictModel):
    """How well one parameter is known: ``rel`` (fraction) or ``abs`` (in the parameter's unit).

    For ``normal`` and ``lognormal`` it is one standard deviation, for
    ``uniform`` the half-width of the interval.
    """

    rel: Annotated[float, Field(gt=0)] | None = None
    abs: Annotated[float, Field(gt=0)] | str | None = None
    dist: Literal["normal", "lognormal", "uniform"] = "normal"

    @model_validator(mode="after")
    def _one_of(self) -> "ParameterUncertainty":
        if (self.rel is None) == (self.abs is None):
            raise ValueError("give exactly one of rel, abs")
        return self


class Reference(StrictModel):
    """A source for a card's numbers."""

    doi: str | None = None
    url: str | None = None
    note: str = ""


class Material(StrictModel):
    """A complete soft material description (a material card).

    Attributes:
        name: Identifier (preset name for presets).
        description: Free text.
        source: Where the numbers come from (datasheet, fit, paper); be honest.
        kind: ``physical`` (meant to describe the material, independent of any
            solver) or ``effective`` (values that reproduce data with one backend,
            mesh and contact model only; ``provenance`` must name backend and mesh).
        density: Mass density [kg/m^3].
        elastic: Constitutive model.
        damping: Optional dissipation model.
        friction: Surface friction against other bodies.
        adhesion: Optional tack against rigid bodies (the ground has its own setting).
        validity: Strain, strain-rate and temperature range of the underlying data.
        uncertainty: Parameter (dotted key, as in overrides) -> uncertainty, used by
            ``softsim.materials.sample`` and sweeps.
        references: DOIs/URLs of the data.
        provenance: Optional metadata, e.g. the mesh/backend a fit is tied to.
    """

    name: str = "custom"
    description: str = ""
    source: str = "unspecified"
    kind: Literal["physical", "effective"] = "physical"
    density: Annotated[Density, Field(gt=0)]
    elastic: ElasticModel
    damping: DampingModel | None = None
    friction: Friction = Friction()
    adhesion: Adhesion | None = None
    validity: Validity | None = None
    uncertainty: dict[str, ParameterUncertainty] = Field(default_factory=dict)
    references: list[Reference] = Field(default_factory=list)
    provenance: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _card(self) -> "Material":
        if isinstance(self.damping, PronySeries) and isinstance(self.elastic, (LinearElastic, StVenantKirchhoff)):
            raise ValueError(f"Prony viscoelasticity relaxes the isochoric (shear) part of a hyperelastic law; "
                             f"{self.elastic.model!r} has no such split (use e.g. neo_hookean)")
        if self.kind == "effective":
            missing = [k for k in ("backend", "mesh_size") if not self.provenance.get(k)]
            if missing:
                raise ValueError(f"an effective material must name {' and '.join(missing)} in provenance "
                                 "(its values only hold there)")
        if self.uncertainty:
            from softsim.config.params import find_param

            for key, u in self.uncertainty.items():
                try:
                    param = find_param(self, key)
                except KeyError:
                    raise ValueError(f"uncertainty key {key!r} is not a parameter of this material") from None
                if not isinstance(param.value, (int, float)) or isinstance(param.value, bool):
                    raise ValueError(f"uncertainty key {key!r} is not numeric ({param.value!r})")
                if isinstance(u.abs, str):
                    if param.unit is None:
                        raise ValueError(f"uncertainty of {key!r}: {u.abs!r} has a unit but the parameter has none")
                    object.__setattr__(u, "abs", parse_quantity(u.abs, param.unit.dimension))
        return self

    @property
    def requires(self) -> frozenset[Capability]:
        req = set(self.elastic.requires) | set(self.friction.requires)
        if self.damping is not None:
            req |= self.damping.requires
        if self.adhesion is not None:
            req |= self.adhesion.requires
        return frozenset(req)

    def with_overrides(self, overrides: Iterable[str]) -> "Material":
        """A copy with dotted ``key=value`` overrides applied, e.g. ``["elastic.c10=12 kPa",
        "damping.tau[1]=20 ms", "elastic.model=yeoh"]`` (switching a model keeps only the
        fields the new model understands, as for presets)."""
        from softsim.materials.library import apply_material_overrides

        data = self.model_dump(mode="json", exclude_none=True)
        return validate(Material, apply_material_overrides(data, overrides), where=f"material {self.name!r}")
