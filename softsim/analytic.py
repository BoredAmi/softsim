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


def cable_beam_stroke(stroke: float, stiffness: float, cable_x: float, length: float, youngs: list[float],
                      edges: list[float], width: float) -> tuple[float, float]:
    """(tension [N], tip rotation [rad]) of the clamped (composite) beam of :func:`cable_beam_tip_angle`
    driven by a stroke-controlled cable (tendon ``stiffness`` [N/m], motor stroke ``stroke`` [m]). The
    cable's shortening is the tendon's stretch plus the beam's: bending ``L a theta / L`` and axial
    ``T L / EA`` along the cable, so ``stroke = T / k + L (a^2 T / (E_t I) + T / EA)`` with arm ``a``;
    solved for ``T`` with the Neo-Hookean tangent ``E_t`` of :func:`cable_beam_tip_angle`."""
    x_na, ei, ea = composite_section(youngs, edges, width)
    arm = cable_x - x_na
    tension = 0.0
    for _ in range(100):
        new = stroke / (1 / stiffness + length * (arm**2 / (ei * (1 + 2 * tension / ea)) + 1 / ea))
        if abs(new - tension) < 1e-12 * max(new, 1e-12):
            break
        tension = new
    return tension, cable_beam_tip_angle(tension, cable_x, length, youngs, edges, width)


def cantilever_tip_deflection(force: float, length: float, youngs: float, poisson: float, width: float,
                              height: float) -> float:
    """Tip deflection [m] of a clamped rectangular beam (``width`` along the load) under an end force
    (Timoshenko, linear): bending ``F L^3 / (3 E I)`` plus shear ``F L / (k G A)``, ``k = 10 (1 + nu) /
    (12 + 11 nu)`` (Cowper's rectangle)."""
    inertia = height * width**3 / 12
    shear = youngs / (2 * (1 + poisson))
    k = 10 * (1 + poisson) / (12 + 11 * poisson)
    return force * length**3 / (3 * youngs * inertia) + force * length / (k * shear * width * height)


def uniaxial_force(material: dict, stretch: float, area: float) -> float:
    """Pull [N] on a bar of cross-section ``area`` [m^2] held in homogeneous uniaxial stress at ``stretch``
    (the nominal stress of :func:`softsim.materials.homogeneous.curve`, exact at finite strain; needs the
    fem extra). ``material`` is a card mapping or preset name."""
    from softsim.materials import load_material
    from softsim.materials.homogeneous import curve

    return float(curve(load_material(material), "uniaxial", [stretch]).stress.iloc[0] * area)
