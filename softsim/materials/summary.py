"""Derived quantities of a material card (``softsim material show``)."""

from __future__ import annotations

import math
from typing import Any

from softsim.materials.models import PronySeries, RayleighDamping, KelvinVoigt, Material


def material_summary(m: Material, mesh_sizes: tuple[float, ...] = (0.002, 0.005)) -> dict[str, Any]:
    """Small-strain constants, wave speeds, explicit time-step estimates, rate and surface data.

    Values are SI. ``dt_explicit[h]`` is the P-wave transit time across an element
    of size ``h``, the order of the largest stable explicit step (implicit solvers
    may take more, but resolve less).
    """
    e = m.elastic
    mu, k = e.initial_shear_modulus(), e.initial_bulk_modulus()
    rho = m.density
    c_s = math.sqrt(mu / rho)
    c_p = math.sqrt((k + 4.0 * mu / 3.0) / rho)
    out: dict[str, Any] = {
        "kind": m.kind,
        "law": e.model,
        "youngs_modulus_0": e.initial_youngs_modulus(),
        "shear_modulus_0": mu,
        "bulk_modulus_0": k,
        "poisson_ratio_0": e.initial_poisson_ratio(),
        "density": rho,
        "shear_wave_speed": c_s,
        "p_wave_speed": c_p,
        "dt_explicit": {h: h / c_p for h in mesh_sizes},
    }
    d = m.damping
    if isinstance(d, PronySeries):
        g_inf = 1.0 - sum(d.g)
        out["prony"] = {"long_term_shear_modulus": mu * g_inf, "relaxed_fraction": sum(d.g),
                        "terms": list(zip(d.g, d.tau))}
    elif isinstance(d, RayleighDamping):
        out["rayleigh"] = {"alpha": d.alpha, "beta": d.beta,
                           "note": "numerical damping, not a material model (loss grows with frequency)"}
    elif isinstance(d, KelvinVoigt):
        out["kelvin_voigt"] = {"viscosity": d.viscosity, "beta_equivalent": d.viscosity / e.initial_youngs_modulus()}
    if m.adhesion is not None:
        out["adhesion"] = {"strength": m.adhesion.strength, "range": m.adhesion.range,
                           "work_of_separation": m.adhesion.work_of_separation}
    out["friction"] = m.friction.sliding
    if m.validity is not None:
        out["validity"] = m.validity.model_dump(exclude_none=True)
    if m.uncertainty:
        out["uncertainty"] = {k_: u.model_dump(exclude_none=True) for k_, u in m.uncertainty.items()}
    if m.kind == "effective":
        out["effective_for"] = {k_: m.provenance.get(k_) for k_ in ("backend", "mesh_size", "element", "timestep")
                                if m.provenance.get(k_) is not None}
    return out
