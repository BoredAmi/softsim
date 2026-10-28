"""Capability vocabulary shared by scene descriptions and backends.

Every scene element declares the capabilities it needs (``requires``); every
backend declares what it supports. ``Backend.check`` compares the two and
refuses the scene with a full list of what is missing, so an unsupported
combination never runs silently with different physics.
"""

from __future__ import annotations

from enum import Enum


class Capability(str, Enum):
    """A physics feature a scene may require. Values are human-readable."""

    # Elastic constitutive models (3D solids).
    LINEAR_ELASTIC = "small-strain linear elasticity"
    STVK = "Saint Venant-Kirchhoff elasticity"
    NEO_HOOKEAN = "Neo-Hookean hyperelasticity"
    MOONEY_RIVLIN = "Mooney-Rivlin hyperelasticity"
    YEOH = "Yeoh hyperelasticity"
    OGDEN = "Ogden hyperelasticity"
    MAGNETO_ACTIVE = "magneto-active (MRE) coupling"

    # Dissipation.
    DAMPING_STIFFNESS_PROPORTIONAL = "stiffness-proportional (Rayleigh beta) damping"
    DAMPING_MASS_PROPORTIONAL = "mass-proportional (Rayleigh alpha) damping"
    VISCOELASTIC_PRONY = "Prony-series viscoelasticity"

    # Geometry / boundary conditions / interaction.
    TET_MESH = "volumetric tetrahedral soft bodies"
    PINNED_VERTICES = "pinned (bonded) soft-body vertices"
    RIGID_BODIES = "free rigid bodies"
    CONTACT_FRICTION = "frictional contact"
    TORSIONAL_ROLLING_FRICTION = "torsional/rolling friction"
    SELF_COLLISION = "soft-body self-collision"
    CONTACT_FORCES = "contact force readout"
