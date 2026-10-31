"""Physics sanity checks run before every simulation.

Checks use the scene plus optional scenario ``estimates`` (rough, closed-form
predictions such as impact speed or expected strain). Findings are advisory
(``info``/``warning``) except ``error``, which the runner refuses to execute.

Recognised estimate keys:
    impact_speed [m/s], expected_indentation [m], expected_strain [-],
    impact_duration [s], smallest_contact_radius [m].
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import numpy as np

from softsim.materials.models import LinearElastic, StVenantKirchhoff
from softsim.scene import Scene

Level = Literal["info", "warning", "error"]


@dataclass(frozen=True)
class Finding:
    level: Level
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.code}: {self.message}"


def p_wave_speed(youngs: float, poisson: float, density: float) -> float:
    """Dilatational wave speed sqrt((lambda + 2 mu) / rho)."""
    return math.sqrt(youngs * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson) * density))


def check_scene(scene: Scene, estimates: Mapping[str, float] | None = None) -> list[Finding]:
    """Return all findings for ``scene``."""
    est = dict(estimates or {})
    sim, contact = scene.simulation, scene.contact
    dt = sim.timestep
    tc = contact.resolved_timeconst(dt)
    out: list[Finding] = []

    if tc * contact.dampratio < dt:
        out.append(Finding("warning", "contact-stiffer-than-step",
                           f"contact timeconst*dampratio {tc * contact.dampratio:.2e} s < timestep {dt:.2e} s; "
                           "the backend cannot resolve it (MuJoCo replaces it by the stiffest resolvable "
                           "zero-restitution contact)"))
    if contact.timeconst != "auto" and tc > 10 * dt:
        out.append(Finding("info", "soft-contact",
                           f"contact timeconst {tc:.1e} s is much larger than the timestep; contacts will be "
                           "compliant (penetration ~ v * timeconst)"))

    steps_per_output = 1.0 / (sim.output_rate * dt)
    if abs(steps_per_output - round(steps_per_output)) > 1e-6 and steps_per_output > 1:
        out.append(Finding("info", "output-rate-rounded",
                           f"output every {sim.output_every} steps = {1 / (sim.output_every * dt):.1f} Hz "
                           f"(requested {sim.output_rate:g} Hz)"))
    if steps_per_output < 1:
        out.append(Finding("warning", "output-faster-than-step",
                           f"output rate {sim.output_rate:g} Hz exceeds the physics rate {1 / dt:g} Hz"))

    v = est.get("impact_speed")
    if v is not None:
        pen = v * tc
        indent = est.get("expected_indentation")
        if indent and pen > 0.1 * indent:
            out.append(Finding("warning", "contact-penetration",
                               f"contact compliance allows ~{pen * 1e3:.2f} mm penetration at {v:.2f} m/s "
                               f"(>{10}% of the expected {indent * 1e3:.1f} mm indentation); lower the timestep "
                               "(timeconst auto = 2*dt)"))
    duration = est.get("impact_duration")
    if duration is not None and dt > duration / 20:
        out.append(Finding("warning", "timestep-vs-impact",
                           f"timestep {dt:.1e} s resolves the ~{duration * 1e3:.1f} ms impact with fewer than "
                           "20 steps; expect timestep-dependent results (<= 2e-4 s was converged for the "
                           "drop test)"))

    for body in scene.soft_bodies:
        elastic = body.material.elastic
        nu = elastic.initial_poisson_ratio()
        quality = body.mesh.quality()
        h = quality["mean_edge"]
        span = body.mesh.points.max(axis=0) - body.mesh.points.min(axis=0)
        if span.min() / h < 4:
            out.append(Finding("warning", "coarse-mesh",
                               f"{body.name}: only ~{span.min() / h:.1f} elements across the thinnest dimension "
                               f"(mean edge {h * 1e3:.2f} mm); bending and indentation will be too stiff"))
        r_contact = est.get("smallest_contact_radius")
        if r_contact is not None and r_contact / h < 3:
            out.append(Finding("warning", "coarse-contact",
                               f"{body.name}: contact radius {r_contact * 1e3:.1f} mm spans < 3 elements"))
        if quality["min_quality"] < 0.1:
            out.append(Finding("warning", "poor-elements",
                               f"{body.name}: worst tetrahedron quality {quality['min_quality']:.2f} (< 0.1)"))
        if nu >= 0.48:
            out.append(Finding("warning", "volumetric-locking",
                               f"{body.name}: Poisson's ratio {nu:.3f} with linear tetrahedra locks; the body "
                               "will be too stiff and mesh-dependent. 0.45 is a common compromise"))
        elif nu >= 0.45:
            out.append(Finding("info", "volumetric-locking",
                               f"{body.name}: Poisson's ratio {nu:.2f}: linear tetrahedra are somewhat stiff; "
                               "fitted moduli are tied to the mesh resolution"))
        strain = est.get("expected_strain")
        if strain is not None:
            if isinstance(elastic, LinearElastic) and strain > 0.05:
                out.append(Finding("warning", "linear-large-strain",
                                   f"{body.name}: linear elasticity at ~{strain:.0%} strain (valid only to a few %)"))
            if isinstance(elastic, StVenantKirchhoff) and strain > 0.3:
                out.append(Finding("warning", "stvk-compression",
                                   f"{body.name}: StVK at ~{strain:.0%} compressive strain softens and can "
                                   "collapse; use stable_neo_hookean"))
            if strain >= 0.6:
                out.append(Finding("warning", "extreme-compression",
                                   f"{body.name}: crude estimate predicts ~{min(strain, 1):.0%} compression; "
                                   "results depend on the large-strain law, compressibility (silicone is nearly "
                                   "incompressible) and rate effects; validate against data at this strain"))
        density = body.material.density
        if body.mass is not None and abs(body.effective_density / density - 1) > 0.05:
            out.append(Finding("info", "density-mismatch",
                               f"{body.name}: mass {body.mass * 1e3:.2f} g implies "
                               f"{body.effective_density:.0f} kg/m^3 vs material {density:.0f} kg/m^3"))
        if sim.integrator not in ("discrete", "implicit", "implicitfast"):
            c = p_wave_speed(elastic.initial_youngs_modulus(), nu, body.effective_density)
            limit = quality["min_edge"] / c
            if dt > limit:
                out.append(Finding("error", "explicit-unstable",
                                   f"{body.name}: explicit integrator with dt {dt:.1e} s > element wave transit "
                                   f"time {limit:.1e} s"))
        out += _adhesion_findings(scene, body, dt)
    return out


def _adhesion_findings(scene: Scene, body, dt: float) -> list[Finding]:
    """Adhesive forces are explicit: their stiffness per surface vertex must be resolvable at ``dt``."""
    laws = [a for a in (body.material.adhesion, scene.ground.adhesion if scene.ground.enabled else None) if a]
    if not laws:
        return []
    from softsim.contact.adhesion import surface_areas

    idx, area = surface_areas(body.mesh)
    mass = body.mesh.lumped_masses(body.effective_density)[idx]
    omega_dt = dt * math.sqrt(max(a.max_stiffness for a in laws) * float(np.max(area / mass)))
    if omega_dt > 1.0:
        return [Finding("error", "adhesion-unstable",
                        f"{body.name}: adhesion stiffness e*strength/range is too high for dt {dt:.1e} s "
                        f"(omega*dt = {omega_dt:.2f} > 1 on the lightest surface vertex); raise the range or "
                        "lower the timestep")]
    if omega_dt > 0.3:
        return [Finding("warning", "adhesion-stiff",
                        f"{body.name}: adhesion resolved with only omega*dt = {omega_dt:.2f}; results may depend "
                        "on the timestep")]
    return []
