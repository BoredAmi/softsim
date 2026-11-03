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


def check_scene(scene: Scene, estimates: Mapping[str, float] | None = None,
                locking_free: bool = False, implicit_actuation: bool = False) -> list[Finding]:
    """Return all findings for ``scene``; ``locking_free``: the backend's elements don't lock (no
    volumetric-locking findings); ``implicit_actuation``: it applies actuator loads implicitly (no
    explicit-stability findings for them)."""
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
        if locking_free:
            pass
        elif nu >= 0.48:
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
        if body.mass is not None and abs(body.mass / body.material_mass - 1) > 0.05:
            out.append(Finding("info", "density-mismatch",
                               f"{body.name}: mass {body.mass * 1e3:.2f} g vs {body.material_mass * 1e3:.2f} g from "
                               f"the material densities: every density is scaled by "
                               f"{body.mass / body.material_mass:.3g}"))
        if sim.integrator not in ("discrete", "implicit", "implicitfast"):
            scale = body.total_mass / body.material_mass
            limit = min(quality["min_edge"] / p_wave_speed(m.elastic.initial_youngs_modulus(),
                                                           m.elastic.initial_poisson_ratio(), m.density * scale)
                        for m, _ in body.material_groups())
            if dt > limit:
                out.append(Finding("error", "explicit-unstable",
                                   f"{body.name}: explicit integrator with dt {dt:.1e} s > element wave transit "
                                   f"time {limit:.1e} s"))
        out += _adhesion_findings(scene, body, dt)
        if not implicit_actuation:
            out += _cable_findings(scene, body, dt)
        out += _prony_findings(body, dt)
        out += _spin_findings(scene, body, dt)
        validity = body.material.validity
        strain = est.get("expected_strain")  # compressive, positive
        if validity is not None and validity.strain is not None and strain is not None and -strain < validity.strain[0]:
            out.append(Finding("warning", "outside-validity",
                               f"{body.name}: expected compression ~{100 * strain:.0f}% but material "
                               f"{body.material.name!r} was measured over strain {validity.strain[0]:g} to "
                               f"{validity.strain[1]:g}; its law is extrapolated here"))
        if body.material.damping is None and est.get("impact_speed", 0.0) > 0:
            out.append(Finding("info", "no-rate-data",
                               f"{body.name}: material {body.material.name!r} has no damping (no rate data), so "
                               "impacts are undamped and rebound too high; give it a Prony series if one is known"))
    return out


def _spin_findings(scene: Scene, body, dt: float) -> list[Finding]:
    """A rotating mount: the rotation per step, and the centrifugal strain it will cause."""
    spin = scene.spin
    if spin is None:
        return []
    out = []
    # minimum jerk peaks at 1.875 x the mean rate of its segment
    peak = max(1.875 * abs(b - a) / (t1 - t0) for a, b, t0, t1 in zip(spin.angles, spin.angles[1:], spin.times, spin.times[1:]))
    if peak * dt > 0.15:
        out.append(Finding("warning", "spin-step",
                           f"{body.name}: the mount turns {peak * dt:.2f} rad per step at its peak rate "
                           f"({peak:.0f} rad/s x {dt:.1e} s). The reference's trapezoidal rule develops a growing "
                           "whirl at large rotations per step (a 6 mm lateral deflection at 0.23 rad on a 60 mm beam, "
                           "0.5 mm at 0.11); MuJoCo and SOFA apply the Coriolis and centrifugal forces explicitly. "
                           "Tested up to 0.11: keep it below ~0.15"))
    n, c = np.asarray(spin.axis, float), np.asarray(spin.center, float)
    d = body.mesh.points - c
    radius = np.linalg.norm(d - np.outer(d @ n, n), axis=1).max()
    strain = body.material.density * peak**2 * radius**2 / (3 * body.material.elastic.initial_youngs_modulus())
    if strain > 0.2:
        out.append(Finding("warning", "spin-centrifugal",
                           f"{body.name}: at {peak:.0f} rad/s its farthest point, {radius * 1e3:.0f} mm from the axis, "
                           f"is pulled out with a strain of about {strain:.0%} (rho w^2 r^2 / 3E): the body stretches "
                           "a lot or is unstable (a soft beam spun fast enough has no equilibrium)"))
    return out


def _prony_findings(body, dt: float) -> list[Finding]:
    """The Maxwell-branch overstress is explicit: its shear stiffness must be resolvable at ``dt``."""
    from softsim.materials.viscoelastic import max_stable_timestep, prony_of

    prony = prony_of(body.material)
    if prony is None:
        return []
    out = []
    dt_max = max_stable_timestep(body.material, body.effective_density, body.mesh.quality()["min_edge"])
    if dt > dt_max:
        out.append(Finding("error", "prony-unstable",
                           f"{body.name}: explicit Prony overstress (sum g * mu0) needs dt <= {dt_max:.2e} s on the "
                           f"smallest element, got {dt:.2e} s; lower the timestep or coarsen the mesh"))
    elif dt > 0.3 * dt_max:
        out.append(Finding("warning", "prony-stiff",
                           f"{body.name}: Prony overstress resolved with omega*dt = {dt / dt_max:.2f}; results may "
                           "depend on the timestep"))
    if min(prony.tau) < 2 * dt:
        out.append(Finding("info", "prony-fast-term",
                           f"{body.name}: Prony time {min(prony.tau):.1e} s is shorter than two steps; that term "
                           "relaxes within a step (it is integrated exactly, but its transient is not resolved)"))
    return out


def _cable_findings(scene: Scene, body, dt: float) -> list[Finding]:
    """MuJoCo and SOFA apply cable forces explicitly: the cable's transverse (string) stiffness at a
    via point, ``T (1/l1 + 1/l2)``, must be resolvable at ``dt`` on the vertices it acts on. Both
    integrate the elasticity implicitly, which adds about ``dt^2 k`` to each vertex's mass in the
    step (``k ~ E h``, a lower bound on a vertex's elastic stiffness). Measured: the 1 mm cable
    cantilever on MuJoCo at 0.1 ms tracks the ramp up to omega dt ~ 1.3, then rings (self-excited,
    bounded) instead of settling. Once ``dt^2 k`` dominates the mass, omega dt tends to
    ``sqrt(k_cable / k)``: a smaller timestep helps little, a larger via-point spacing does."""
    from softsim.actuation import CableActuator

    out = []
    k_elastic = body.material.elastic.initial_youngs_modulus() * body.mesh.quality()["mean_edge"]
    inv_mass = 1.0 / (body.vertex_masses() + dt * dt * k_elastic)
    inv_mass[list(body.pinned)] = 0.0
    for a in scene.actuators:
        if not isinstance(a, CableActuator) or a.body != body.name:
            continue
        values = [a.tension, *(v for _, v in a.schedule)]
        # a stroke-controlled tendon's tension is bounded by k times its largest stroke
        tension = a.stiffness * max(max(values), 0.0) if a.stroke_controlled else max(values)
        via = np.array(a.via)
        seg = np.diff(via, axis=0)
        inv_len = 1.0 / np.linalg.norm(seg, axis=1)
        stiffness = tension * (np.append(inv_len, 0.0) + np.insert(inv_len, 0, 0.0))
        nodes, w = body.cable_rows(a)
        lam = float(np.max(stiffness * np.sum(w**2 * inv_mass[nodes], axis=1)))
        if a.stroke_controlled:
            # the taut tendon's axial term k dL dL^T: largest eigenvalue k sum_v |dL/dx_v|^2 / m_v
            u = seg * inv_len[:, None]
            g_via = np.zeros_like(via)
            g_via[:-1] -= u
            g_via[1:] += u
            g = np.zeros_like(body.mesh.points)
            np.add.at(g, nodes.ravel(), (w[:, :, None] * g_via[:, None, :]).reshape(-1, 3))
            lam = max(lam, a.stiffness * float(np.sum(np.sum(g**2, axis=1) * inv_mass)))
        omega_dt = dt * math.sqrt(lam)
        if omega_dt > 1.0:
            what = (f"tendon stiffness {a.stiffness:.3g} N/m" if a.stroke_controlled else
                    f"cable tension {tension:.3g} N over {1e3 / inv_len.max():.2f} mm segments")
            fix = ("lower the tendon stiffness or the timestep" if a.stroke_controlled else
                   "use a larger via-point spacing (a smaller timestep helps little)")
            out.append(Finding("error", "cable-unstable",
                               f"actuator {a.name!r}: {what} is too stiff for explicit forces at dt {dt:.1e} s "
                               f"(omega*dt = {omega_dt:.2f} > 1); {fix}"))
        elif omega_dt > 0.5:
            out.append(Finding("warning", "cable-stiff",
                               f"actuator {a.name!r}: explicit cable forces resolved with only omega*dt = "
                               f"{omega_dt:.2f} at the peak tension; results may depend on the timestep"))
    return out


def _adhesion_findings(scene: Scene, body, dt: float) -> list[Finding]:
    """Adhesive forces are explicit: their stiffness per surface vertex must be resolvable at ``dt``."""
    laws = [a for a in (body.material.adhesion, scene.ground.adhesion if scene.ground.enabled else None) if a]
    if not laws:
        return []
    from softsim.contact.adhesion import surface_areas

    idx, area = surface_areas(body.mesh)
    mass = body.vertex_masses()[idx]
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
