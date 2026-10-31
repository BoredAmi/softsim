"""Closed-form solutions used to verify the solvers (tests, docs/actuation.md)."""

from __future__ import annotations

import numpy as np


def hollow_sphere_pressure(mu: float, inner: float, outer: float, stretch: np.ndarray) -> np.ndarray:
    """Internal pressure [Pa] that holds an incompressible Neo-Hookean thick sphere (shear modulus
    ``mu``, undeformed radii ``inner`` < ``outer``) at inner hoop stretch ``stretch``.

    Exact at finite strain: ``p = 2 mu [(1/lb + 1/(4 lb^4)) - (1/la + 1/(4 la^4))]`` with the outer
    stretch from volume conservation, ``lb^3 = 1 + (inner/outer)^3 (la^3 - 1)``. It has a maximum (the
    limit pressure, :func:`hollow_sphere_limit`) beyond which no static inflated state exists.
    """
    la = np.asarray(stretch, float)
    lb = np.cbrt(1.0 + (inner / outer) ** 3 * (la**3 - 1.0))
    return 2.0 * mu * ((1.0 / lb + 0.25 / lb**4) - (1.0 / la + 0.25 / la**4))


def hollow_sphere_limit(mu: float, inner: float, outer: float) -> tuple[float, float]:
    """(limit pressure [Pa], inner stretch at it) of :func:`hollow_sphere_pressure`."""
    la = np.linspace(1.0, 8.0, 70001)
    p = hollow_sphere_pressure(mu, inner, outer, la)
    i = int(p.argmax())
    return float(p[i]), float(la[i])


def hollow_sphere_linear(youngs: float, poisson: float, inner: float, outer: float, pressure: float,
                         r: float) -> float:
    """Radial displacement [m] at radius ``r`` of a linear elastic thick sphere under internal
    ``pressure`` (Lame): ``u = p a^3 / (E (b^3 - a^3)) [(1 - 2 nu) r + (1 + nu) b^3 / (2 r^2)]``."""
    a3, b3 = inner**3, outer**3
    return pressure * a3 / (youngs * (b3 - a3)) * ((1 - 2 * poisson) * r + (1 + poisson) * b3 / (2 * r * r))


def composite_section(youngs: list[float], edges: list[float], width: float) -> tuple[float, float, float]:
    """(neutral axis x [m], bending stiffness EI [N m^2], axial stiffness EA [N]) of a rectangular
    section of ``width`` made of layers: layer ``i`` spans ``edges[i]`` to ``edges[i + 1]`` along x with
    Young's modulus ``youngs[i]`` (Euler-Bernoulli composite beam, plane sections)."""
    e = np.asarray(youngs, float)
    lo, hi = np.asarray(edges[:-1], float), np.asarray(edges[1:], float)
    area, centre = (hi - lo) * width, 0.5 * (lo + hi)
    ea = float(np.sum(e * area))
    x_na = float(np.sum(e * area * centre) / ea)
    ei = float(np.sum(e * (width * (hi - lo) ** 3 / 12 + area * (centre - x_na) ** 2)))
    return x_na, ei, ea


def cable_beam_tip_angle(tension: float, cable_x: float, length: float, youngs: list[float], edges: list[float],
                         width: float) -> float:
    """Tip rotation [rad] of a clamped (composite) beam bent by a cable at ``cable_x`` anchored at
    ``length``: a pure moment ``T (cable_x - x_na)``, so ``theta = T (cable_x - x_na) L / (E_t I)``, exact
    at any rotation in Euler-Bernoulli. ``E_t = E (1 + 2 T / EA)`` is the Neo-Hookean tangent under the
    cable's axial pre-compression."""
    x_na, ei, ea = composite_section(youngs, edges, width)
    return tension * (cable_x - x_na) * length / (ei * (1 + 2 * tension / ea))
