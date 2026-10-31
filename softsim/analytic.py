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
