"""Centred drop test on the axisymmetric reference solver.

Builds the problem from the same drop-test config as the general backends
(specimen cylinder, material, weight, measured impact speed) and returns a time
series with the drop test's column names, so ``softsim.experiments.compare``
applies unchanged. Axisymmetry means a centred, untilted, non-spinning impact:
``drop.offset``, ``drop.tilt`` and ``drop.angular_velocity`` are refused unless
``ignore_asymmetry`` is set (then they are dropped, and the result says so).

Not modelled here: the compliant support (the specimen rests on a rigid
table), and Rayleigh damping (use a Prony series).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError
from softsim.fem.axisym import AxisymSolver, ContactSettings, Problem, Punch, cylinder_mesh
from softsim.fem.materials import from_material
from softsim.geometry import load_rigid_body
from softsim.materials import load_material
from softsim.scenarios.drop_test import DropTestConfig


@dataclass(frozen=True)
class FemSettings:
    """Discretisation of the reference drop simulation.

    ``cells_r`` x ``cells_z`` P2 cells over the specimen's half-section (rows graded towards
    the top by ``grading``); ``timestep`` is the largest step (failed steps are halved).
    """

    cells_r: int = 16
    cells_z: int = 10
    grading: float = 1.5
    timestep: float = 2e-5
    numerical_damping: float = 0.25    # Newmark gamma = 0.75: stabilises penalty contact (lower values gained energy)
    contact_stiffness: float = 2e11
    friction: str | float = "stick"    # "stick" (no slip once touching), "frictionless", or a Coulomb coefficient
    punch_edge_radius: float = 5e-4
    max_bulk_ratio: float = 1e3        # caps kappa/mu: beyond ~1e3 the material is incompressible in practice


def build_problem(raw: dict[str, Any], base_dir: Path, fem: FemSettings,
                  ignore_asymmetry: bool = False) -> tuple[Problem, dict[str, Any]]:
    cfg = validate(DropTestConfig, raw)
    drop = cfg.drop
    asym = {k: v for k, v in (("offset", drop.offset), ("tilt", drop.tilt), ("angular_velocity", drop.angular_velocity))
            if any(abs(x) > 1e-12 for x in v)}
    if asym and not ignore_asymmetry:
        raise ConfigError(f"axisymmetric reference: the impact must be centred and untilted, got {asym} "
                          "(pass ignore_asymmetry to drop them)")
    if cfg.support.enabled:
        raise ConfigError("axisymmetric reference: the compliant support is not modelled")
    geom = cfg.specimen.geometry
    if geom.type != "cylinder":
        raise ConfigError("axisymmetric reference: the specimen must be a cylinder")
    material = load_material(cfg.specimen.material, base_dir=base_dir)
    law, kappa, prony = from_material(material)
    kappa = min(kappa, fem.max_bulk_ratio * law.shear_modulus)
    radius, height = geom.radius, geom.height
    mesh = cylinder_mesh(radius, height, fem.cells_r, fem.cells_z, fem.grading)
    density = cfg.specimen.mass / mesh.volume() if cfg.specimen.mass else material.density
    body = load_rigid_body(cfg.weight.body)
    g = -cfg.simulation.gravity[2]
    v_release = drop.initial_speed if drop.initial_speed is not None else \
        math.sqrt(max(2 * g * (drop.height - drop.gap), 0.0))
    com = sum(p.volume() * p.center[2] for p in body.shape.parts()) / sum(p.volume() for p in body.shape.parts())
    punch = Punch(radius=body.shape.bottom_radius, mass=body.mass, z0=height + drop.gap, v0=-v_release,
                  edge_radius=fem.punch_edge_radius, com_height=com)
    stick = fem.friction == "stick"
    mu = float(fem.friction) if not isinstance(fem.friction, str) else 0.0
    if isinstance(fem.friction, str) and fem.friction not in ("stick", "frictionless"):
        raise ConfigError(f"friction must be 'stick', 'frictionless' or a coefficient, got {fem.friction!r}")
    adhesion = material.adhesion
    punch_contact = ContactSettings(fem.contact_stiffness, stick, adhesion.strength if adhesion else 0.0,
                                    adhesion.range if adhesion else 3e-4, friction=mu)
    ground = cfg.ground.adhesion
    ground_contact = ContactSettings(fem.contact_stiffness, stick, ground.strength if ground else 0.0,
                                     ground.range if ground else 3e-4, friction=mu)
    problem = Problem(mesh, law, kappa, density, prony, gravity=g, base="free", punch=punch,
                      punch_contact=punch_contact, ground_contact=ground_contact)
    info = {"ignored_asymmetry": asym, "kappa_over_mu": kappa / law.shear_modulus, "material": material.name,
            "nodes": mesh.n_nodes, "elements": len(mesh.triangles)}
    return problem, info


def run_drop(config: str | Path | dict, overrides: list[str] = (), fem: FemSettings = FemSettings(),
             duration: float | None = None, base_dir: Path | None = None, ignore_asymmetry: bool = False,
             progress: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Simulate a centred drop; returns (time series, info)."""
    if isinstance(config, (str, Path)):
        base_dir = Path(config).parent
        raw = load_yaml(config)
    else:
        raw = dict(config)
    raw = apply_overrides(raw, list(overrides))
    problem, info = build_problem(raw, base_dir or Path.cwd(), fem, ignore_asymmetry)
    cfg = validate(DropTestConfig, raw)
    solver = AxisymSolver(problem, dynamic=True, numerical_damping=fem.numerical_damping)
    t_end = duration if duration is not None else cfg.simulation.duration
    sample = 1.0 / cfg.simulation.output_rate
    height = problem.mesh.nodes[:, 1].max()
    axis_top = int(np.argmin(np.hypot(problem.mesh.nodes[:, 0], problem.mesh.nodes[:, 1] - height)))
    rows, next_out, start = [], 0.0, time.perf_counter()
    while solver.t < t_end - 1e-12:
        if solver.t >= next_out - 1e-12:
            rows.append(_sample(solver, problem, axis_top))
            next_out += sample
        dt = min(fem.timestep, next_out - solver.t) if next_out > solver.t else fem.timestep
        if not solver.advance(dt, depth=4):
            raise RuntimeError(f"reference FEM did not converge at t = {solver.t * 1e3:.3f} ms")
        if progress and len(rows) % 20 == 0:
            print(f"  t = {solver.t * 1e3:.2f} ms  ({time.perf_counter() - start:.0f} s)", flush=True)
    rows.append(_sample(solver, problem, axis_top))
    info["wall_time"] = time.perf_counter() - start
    return pd.DataFrame(rows), info


def _sample(solver: AxisymSolver, problem: Problem, axis_top: int) -> dict[str, float]:
    x = solver.current_nodes()
    punch = problem.punch
    return {
        "t": solver.t,
        "weight_z": solver.zp,
        "weight_vz": solver.vp,
        "weight_com_x": 0.0,
        "weight_com_z": solver.zp + punch.com_height,
        "top_center_z": float(x[axis_top, 1]),
        "specimen_max_z": float(x[:, 1].max()),
        "specimen_min_z": float(x[:, 1].min()),
        "bulge_diameter": float(2 * x[:, 0].max()),
        "contact_force_z": solver.punch_force(),
    }
