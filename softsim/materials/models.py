"""Material models as validated, backend-neutral data.

A :class:`Material` combines density, an elastic constitutive model, optional
damping, surface friction and optional surface adhesion (tack). Each model declares the capabilities it needs
(``requires``) so backends can reject what they cannot simulate.

All elastic models expose small-strain equivalents (``initial_youngs_modulus``,
``initial_poisson_ratio``). These are used by stability checks and analytic
tests, never as a silent substitute for the requested model.
"""

from __future__ import annotations

import math
from typing import Annotated, ClassVar, Literal, Union

from pydantic import Field, model_validator

from softsim.capabilities import Capability
from softsim.config.schema import StrictModel
from softsim.config.units import Density, Length, Pressure, Time

PoissonRatio = Annotated[float, Field(gt=-1.0, lt=0.5)]
NonNegative = Annotated[float, Field(ge=0.0)]


class _IsotropicEnu(StrictModel):
    """Models parametrised by Young's modulus and Poisson's ratio."""

    youngs_modulus: Annotated[Pressure, Field(gt=0)]
    poisson_ratio: PoissonRatio

    def initial_youngs_modulus(self) -> float:
        return self.youngs_modulus

    def initial_poisson_ratio(self) -> float:
        return self.poisson_ratio


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


class _BulkHyperelastic(StrictModel):
    """Hyperelastic models with an explicit bulk modulus for the volumetric part."""

    bulk_modulus: Annotated[Pressure, Field(gt=0)]

    def _shear_modulus(self) -> float:
        raise NotImplementedError

    def initial_youngs_modulus(self) -> float:
        mu, k = self._shear_modulus(), self.bulk_modulus
        return 9.0 * k * mu / (3.0 * k + mu)

    def initial_poisson_ratio(self) -> float:
        mu, k = self._shear_modulus(), self.bulk_modulus
        return (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))


class MooneyRivlin(_BulkHyperelastic):
    """Two-term Mooney-Rivlin: W = C10 (I1 - 3) + C01 (I2 - 3) + volumetric."""

    model: Literal["mooney_rivlin"] = "mooney_rivlin"
    c10: Pressure
    c01: Pressure
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.MOONEY_RIVLIN})

    def _shear_modulus(self) -> float:
        return 2.0 * (self.c10 + self.c01)

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

    def _shear_modulus(self) -> float:
        return 2.0 * self.c10


class Ogden(_BulkHyperelastic):
    """N-term Ogden model with shear moduli ``mu`` and exponents ``alpha``."""

    model: Literal["ogden"] = "ogden"
    mu: list[Pressure] = Field(min_length=1)
    alpha: list[float] = Field(min_length=1)
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.OGDEN})

    def _shear_modulus(self) -> float:
        return 0.5 * sum(m * a for m, a in zip(self.mu, self.alpha))

    @model_validator(mode="after")
    def _matching_terms(self) -> "Ogden":
        if len(self.mu) != len(self.alpha):
            raise ValueError("mu and alpha must have the same length")
        if self._shear_modulus() <= 0:
            raise ValueError("initial shear modulus 0.5*sum(mu*alpha) must be positive")
        return self


HyperelasticMatrix = Annotated[
    Union[NeoHookean, StableNeoHookean, MooneyRivlin, Yeoh, Ogden], Field(discriminator="model")
]


class MagnetoActive(StrictModel):
    """Placeholder for magnetorheological elastomers (MRE).

    Holds the passive matrix model and particle data so presets can already be
    written; no backend implements the magneto-mechanical coupling yet, so any
    scene using it is rejected rather than silently simulated as passive.
    """

    model: Literal["magneto_active"] = "magneto_active"
    matrix: HyperelasticMatrix
    particle_volume_fraction: Annotated[float, Field(ge=0, lt=1)]
    saturation_magnetization: Annotated[float, Field(ge=0)] = 0.0  # A/m
    requires: ClassVar[frozenset[Capability]] = frozenset({Capability.MAGNETO_ACTIVE})

    def initial_youngs_modulus(self) -> float:
        return self.matrix.initial_youngs_modulus()

    def initial_poisson_ratio(self) -> float:
        return self.matrix.initial_poisson_ratio()


ElasticModel = Annotated[
    Union[LinearElastic, StVenantKirchhoff, NeoHookean, StableNeoHookean, MooneyRivlin, Yeoh, Ogden,
          MagnetoActive],
    Field(discriminator="model"),
]


class RayleighDamping(StrictModel):
    """Rayleigh damping C = alpha M + beta K.

    ``beta`` (seconds) damps high-frequency/strain-rate motion, ``alpha`` (1/s)
    damps rigid-like motion. Note this is a crude stand-in for silicone
    viscoelasticity: the loss factor grows linearly with frequency.
    """

    model: Literal["rayleigh"] = "rayleigh"
    alpha: Annotated[float, Field(ge=0)] = 0.0
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
    viscosity: Annotated[float, Field(ge=0)]  # Pa*s
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


class Material(StrictModel):
    """A complete soft material description.

    Attributes:
        name: Identifier (preset name for presets).
        description: Free text.
        source: Where the numbers come from (datasheet, fit, paper); be honest.
        density: Mass density [kg/m^3].
        elastic: Constitutive model.
        damping: Optional dissipation model.
        friction: Surface friction against other bodies.
        adhesion: Optional tack against rigid bodies (the ground has its own setting).
        provenance: Optional metadata, e.g. the mesh/backend a fit is tied to.
    """

    name: str = "custom"
    description: str = ""
    source: str = "unspecified"
    density: Annotated[Density, Field(gt=0)]
    elastic: ElasticModel
    damping: DampingModel | None = None
    friction: Friction = Friction()
    adhesion: Adhesion | None = None
    provenance: dict[str, object] = Field(default_factory=dict)

    @property
    def requires(self) -> frozenset[Capability]:
        req = set(self.elastic.requires) | set(self.friction.requires)
        if self.damping is not None:
            req |= self.damping.requires
        if self.adhesion is not None:
            req |= self.adhesion.requires
        return frozenset(req)
