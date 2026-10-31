"""Scene -> MuJoCo model.

The model skeleton is generated as MJCF (readable, saved for inspection) and
then post-processed through ``mujoco.MjSpec`` for things MJCF cannot express:
    * Stable Neo-Hookean flex elasticity (``MjsFlex.elastic3d = 1``, spec-only in 3.17),
    * mass-consistent lumped vertex masses (flexcomp spreads mass uniformly
      over vertices, which is wrong for non-uniform meshes).

Verified against MuJoCo 3.17.0; the flex API changes between releases, so the
supported version range is pinned in pyproject.toml.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from softsim.materials.models import (
    KelvinVoigt,
    RayleighDamping,
    StableNeoHookean,
    StVenantKirchhoff,
)
from softsim.materials.models import Friction
from softsim.scene import RigidBodySpec, Scene, SoftBodySpec

GROUND = "ground"


def _fmt(values) -> str:
    return " ".join(f"{float(v):.9g}" for v in np.ravel(values))


def _condim(friction: Friction) -> int:
    if friction.sliding == 0:
        return 1
    if friction.rolling > 0:
        return 6
    if friction.torsional > 0:
        return 4
    return 3


def _friction_attr(friction: Friction) -> str:
    return _fmt([friction.sliding, friction.torsional, friction.rolling])


def rayleigh_beta(body: SoftBodySpec) -> float:
    """Stiffness-proportional damping coefficient (seconds) for MuJoCo's flex damping."""
    damping = body.material.damping
    if damping is None:
        return 0.0
    if isinstance(damping, RayleighDamping):
        return damping.beta
    if isinstance(damping, KelvinVoigt):
        return damping.viscosity / body.material.elastic.initial_youngs_modulus()
    raise TypeError(f"damping model {damping.model!r} is not supported by MuJoCo")


def _soft_body_xml(parent: ET.Element, body: SoftBodySpec, scene: Scene) -> None:
    elastic = body.material.elastic
    if not isinstance(elastic, (StVenantKirchhoff, StableNeoHookean)):
        raise TypeError(f"elastic model {elastic.model!r} is not supported by MuJoCo")
    contact = scene.contact
    timeconst = contact.resolved_timeconst(scene.simulation.timestep)
    pos = np.array(body.pose.position) + [0.0, 0.0, contact.soft_radius]  # skin rests on the ground
    flex = ET.SubElement(parent, "flexcomp", {
        "name": body.name,
        "type": "direct",
        "dim": "3",
        "point": _fmt(body.mesh.points),
        "element": " ".join(map(str, body.mesh.tets.ravel())),
        "mass": f"{body.total_mass:.9g}",
        "radius": f"{contact.soft_radius:.9g}",
        "pos": _fmt(pos),
        "quat": _fmt(body.pose.quaternion),
        "rgba": _fmt(body.rgba),
    })
    ET.SubElement(flex, "elasticity", {
        "young": f"{elastic.youngs_modulus:.9g}",
        "poisson": f"{elastic.poisson_ratio:.9g}",
        "damping": f"{rayleigh_beta(body):.9g}",
    })
    ET.SubElement(flex, "contact", {
        "selfcollide": "auto" if body.self_collision else "none",
        "activelayers": str(contact.active_layers),
        "condim": str(_condim(body.material.friction)),
        "friction": _friction_attr(body.material.friction),
        "solref": _fmt([timeconst, contact.dampratio]),
        "solimp": _fmt(contact.solimp),
    })
    ET.SubElement(flex, "edge", {"equality": "false"})
    if body.pinned:
        ET.SubElement(flex, "pin", {"id": " ".join(map(str, body.pinned))})


def _rigid_body_xml(parent: ET.Element, body: RigidBodySpec) -> None:
    el = ET.SubElement(parent, "body", {
        "name": body.name, "pos": _fmt(body.pose.position), "quat": _fmt(body.pose.quaternion)})
    if body.mount == "slider_z":
        ET.SubElement(el, "joint", {
            "name": f"{body.name}/slide", "type": "slide", "axis": "0 0 1", "springref": "0",
            "stiffness": f"{body.mount_stiffness:.9g}", "damping": f"{body.mount_damping:.9g}"})
    else:
        ET.SubElement(el, "freejoint", {"name": f"{body.name}/free"})
    volumes = np.array([p.volume() for p in body.parts])
    for i, (part, vol) in enumerate(zip(body.parts, volumes)):
        size = part.size[:2] if part.kind == "cylinder" else part.size
        ET.SubElement(el, "geom", {
            "name": f"{body.name}/{i}",
            "type": part.kind,
            "size": _fmt(size),
            "pos": _fmt(part.center),
            "mass": f"{body.mass * vol / volumes.sum():.9g}",
            "condim": str(_condim(body.friction)),
            "friction": _friction_attr(body.friction),
            "rgba": _fmt(body.rgba),
        })


def scene_to_mjcf(scene: Scene) -> str:
    """MJCF text for ``scene`` (before spec post-processing)."""
    sim, contact = scene.simulation, scene.contact
    root = ET.Element("mujoco", {"model": str(scene.metadata.get("name", "softsim"))})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "option", {
        "timestep": f"{sim.timestep:.9g}",
        "integrator": sim.integrator,
        "solver": sim.solver,
        "iterations": str(sim.iterations),
        "tolerance": f"{sim.tolerance:.9g}",
        "gravity": _fmt(sim.gravity),
    })
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "geom", {
        "solref": _fmt([contact.resolved_timeconst(sim.timestep), contact.dampratio]),
        "solimp": _fmt(contact.solimp),
    })
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", {"offwidth": "1280", "offheight": "960"})
    ET.SubElement(visual, "map", {"znear": "0.001"})
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", {"pos": "0 -0.3 0.5", "dir": "0 0.5 -1"})
    if scene.ground.enabled:
        ET.SubElement(world, "geom", {
            "name": GROUND, "type": "plane", "size": "1 1 0.01", "rgba": "0.2 0.5 0.3 1",
            "condim": str(_condim(scene.ground.friction)),
            "friction": _friction_attr(scene.ground.friction),
        })
    for body in scene.soft_bodies:
        _soft_body_xml(world, body, scene)
    for body in scene.rigid_bodies:
        _rigid_body_xml(world, body)
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")


def build_spec(scene: Scene, mjcf: str | None = None) -> mujoco.MjSpec:
    """MjSpec for ``scene`` with spec-only settings applied (``mjcf`` if already generated)."""
    spec = mujoco.MjSpec.from_string(mjcf or scene_to_mjcf(scene))
    flexes = {f.name: f for f in spec.flexes}
    for body in scene.soft_bodies:
        if isinstance(body.material.elastic, StableNeoHookean):
            flexes[body.name].elastic3d = 1
        masses = body.mesh.lumped_masses(body.effective_density)
        pinned = set(body.pinned)
        for i, mass in enumerate(masses):
            if i in pinned:
                continue
            spec.body(f"{body.name}_{i}").mass = float(mass)
    return spec
