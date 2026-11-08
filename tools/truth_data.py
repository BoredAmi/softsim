"""The Truth tab's data: one finite-element grasp and the fast finger on the same schedule, frame for frame.

A grasp is plane strain, so the page draws the y = +half-width face in the x-z plane. The pad bulges further mid-width,
where the cavity is, so contact is often not on that face: touching nodes are kept from the whole surface.
Positions are int16 at ``QUANTUM``. The fast finger runs :data:`SCENARIO`'s pressure schedule and follows the palm the
finite-element run recorded, so the two share a clock. Chapters are found in the run, not written in by hand.
"""

from __future__ import annotations

import base64
from pathlib import Path

import mujoco
import numpy as np
import yaml

from plot_fe_grasp import FRICTION, HALF_Z, OFF_TABLE, touching
from softsim.actuation import PressureActuatorConfig
from softsim.reduced import ReducedFinger, ReducedGripper
from softsim.reduced.mjcf import OBJECTS
from softsim.scenarios.grasp import is_held

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "scenarios" / "grasp" / "box.yaml"
QUANTUM = 1e-5  # m
CLOSED = 0.7  # s: the fingers have closed and the palm starts up
MASS = 0.01  # kg: the box in the run the page shows


def pressure_schedule(path: Path = SCENARIO) -> tuple[np.ndarray, np.ndarray]:
    """(times [s], pressures [Pa]) of the scenario's cavity, linear between them."""
    cfg = PressureActuatorConfig.model_validate(yaml.safe_load(path.read_text())["actuators"][0])
    return tuple(np.array(cfg.schedule).T)


def face_nodes(rest: np.ndarray, tris: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(nodes, triangles renumbered into them) of the mesh's y = max face."""
    y = rest[:, 1]
    face = tris[np.all(np.isclose(y[tris], y.max()), axis=1)]
    nodes = np.unique(face)
    return nodes, np.searchsorted(nodes, face)


def fast_run(f: ReducedFinger, t: np.ndarray, palm: np.ndarray) -> dict:
    """The fast finger at times ``t`` [s], its palm asked to follow ``palm`` [m]. ``rects`` (F, G, 4, 2): each link's box
    in the x-z plane; ``box`` (F, 2), ``rise``, ``palm`` [m]; ``grip`` [N]."""
    g = ReducedGripper(f, fingers=2, shape="box", size=OBJECTS["box"], mass=MASS, friction=FRICTION)
    m, d, dt = g.model, g.data, g.model.opt.timestep
    links = np.flatnonzero((m.geom_type == mujoco.mjtGeom.mjGEOM_BOX) & (m.geom_contype != 0) & ~np.isin(np.arange(m.ngeom), sorted(g._geoms)))
    corners = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], float)
    times, pressures = pressure_schedule()
    mujoco.mj_kinematics(m, d)  # nothing is placed until the first step
    now, z0 = 0.0, g.object_height
    out = dict(rects=[], box=[], rise=[], palm=[], grip=[])
    for frame in t:
        while now < frame - dt / 2:
            now += dt
            g.step(float(np.interp(now, times, pressures)), float(np.interp(now, t, palm)))
        mujoco.mj_kinematics(m, d)  # xpos trails qpos by a step
        pts = np.stack([d.geom_xpos[k] + (corners * m.geom_size[k]) @ d.geom_xmat[k].reshape(3, 3).T for k in links])
        out["rects"].append(pts[:, :, [0, 2]])
        out["box"].append(d.xpos[g._object, [0, 2]].copy())
        out["rise"].append(g.object_height - z0)
        out["palm"].append(g.palm_rise)
        out["grip"].append(g.grip_force())
    return {k: np.array(v) for k, v in out.items()} | dict(held=is_held(t, np.array(out["rise"])))


def chapters(d: dict, settled_height: np.ndarray) -> list[dict]:
    """The moments worth stopping on: closed, tips clear of the box, off the table, held."""
    t, x, box = d["t"], d["x"], d["object"]
    closed = int(np.argmax(t >= CLOSED))
    tips = touching(x[closed], box[closed])
    peel = next((i for i in range(closed, len(t)) if not np.intersect1d(tips, touching(x[i], box[i])).size), None)
    off = next((i for i in range(closed, len(t)) if settled_height[i] > OFF_TABLE), None)
    marks = [("Closed", closed), ("Tips let go", peel), ("Off the table", off), ("End", len(t) - 1)]
    return [dict(label=label, frame=int(i)) for label, i in marks if i is not None]


def pack(d: dict, f: ReducedFinger, label: str) -> dict:
    """What the page embeds, from the arrays tools/run_fe_grasp.py wrote."""
    nodes, tris = face_nodes(d["rest"], d["tris"])
    x = d["x"][:, nodes][:, :, [0, 2]]
    q = np.round(x / QUANTUM).astype("<i2")
    height = d["object"][:, 2] - HALF_Z
    rise = height - height[int(np.argmax(d["t"] >= 0.1))]  # it starts on a small clearance and lands
    fast = fast_run(f, d["t"], d["palm"])
    r = lambda a, n=6: np.round(a, n).tolist()
    return dict(label=label, t=r(d["t"], 4), quantum=QUANTUM, nodes=len(nodes), xz=base64.b64encode(q.tobytes()).decode(), tris=tris.ravel().tolist(),
                touch=[r(d["x"][i][touching(d["x"][i], d["object"][i])][:, [0, 2]], 5) for i in range(len(d["t"]))],
                box=r(d["object"][:, [0, 2]]), rise=r(rise), palm=r(d["palm"]), grip=r(d["grip"], 4), held=is_held(d["t"], rise),
                half=[0.016, HALF_Z], chapters=chapters(d, rise),
                fast=dict(rects=r(fast["rects"]), box=r(fast["box"]), rise=r(fast["rise"]), palm=r(fast["palm"]), grip=r(fast["grip"], 4), held=fast["held"]))
