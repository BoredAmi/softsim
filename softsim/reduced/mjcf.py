"""The reduced gripper as a MuJoCo model: fingers (:class:`~softsim.reduced.finger.ReducedFinger`) around a palm on a
vertical slide, an object on a table. Runs tens of times faster than real time, in Python and in the browser.

Every step the caller gives a pressure: each hinge gets the torque ``stiffness(p) (angle(p) - q)`` and each slide a
stiff pull to ``length(p)``. Contact is between fingers and object, object and table; fingers ignore each other.
"""

from __future__ import annotations

import numpy as np
import mujoco

from softsim.reduced.finger import ReducedFinger

FINGER, WORLD = 1, 2  # contype/conaffinity bits: a finger never meets a finger
OBJECTS = {"box": (0.016, 0.012, 0.03), "cylinder": (0.02, 0.03), "bottle": (0.02, 0.03)}  # half sizes [m], as gripper_xml
SLIDE_KP, SLIDE_KV = 400.0, 4.0


def _link(f: ReducedFinger, tag: str, i: int, mu: float) -> str:
    n = f.links
    if i == n:
        return ""
    l0 = float(f.lengths[0, i])
    mass = f.mass / n
    # pad face at -offset; the back of the box is ``thickness`` from it; boxes overlap so stretching leaves no gap
    box = (0.6 * l0, f.width / 2, f.thickness / 2)
    z = f.thickness / 2 - f.offset
    slide = (f'<joint name="{tag}s{i}" type="slide" axis="1 0 0" armature="1e-4" damping="0.5"/>' if i else "")
    pos = float(f.lengths[0, i - 1]) if i else 0.0
    return (f'<body name="{tag}l{i}" pos="{pos:.6f} 0 0">{slide}'
            f'<joint name="{tag}h{i}" type="hinge" axis="0 1 0" armature="1e-5" damping="2e-3"/>'
            f'<geom type="box" size="{box[0]:.5f} {box[1]:.5f} {box[2]:.5f}" pos="{l0 / 2:.5f} 0 {z:.5f}" mass="{mass:.6f}" '
            f'contype="{FINGER}" conaffinity="{WORLD}" friction="{mu} 0.005 0.0001" solref="0.004 1"/>'
            f'{_link(f, tag, i + 1, mu)}</body>')


def gripper_xml(f: ReducedFinger, fingers: int = 2, gap: float = 0.05, height: float = 0.14, shape: str = "box",
                size=(0.016, 0.012, 0.03), mass: float = 0.05, friction: float = 0.8, gravity: float = 9.81,
                timestep: float = 0.001) -> str:
    """MJCF of the gripper. Fingers sit on a circle of diameter ``gap`` (their pad faces); object as in
    :class:`softsim.scenarios.grasp.GraspObjectConfig` (or ``shape="bottle"``: radius and half height of a cylinder
    with a neck), resting on the table."""
    r = gap / 2 + f.offset
    body = ""
    for k in range(fingers):
        a = 2 * np.pi * k / fingers
        body += (f'<body pos="{r * np.cos(a):.6f} {r * np.sin(a):.6f} 0" euler="0 0 {np.degrees(a):.4f}">'
                 f'<body euler="0 90 0">{_link(f, f"f{k}", 0, friction)}</body></body>')
    half = size[2] if shape == "box" else size[1]
    surface = (f'friction="{friction} 0.005 0.0001" contype="{WORLD}" conaffinity="{WORLD | FINGER}" '
               'rgba="0.85 0.55 0.2 1"')
    if shape == "box":
        solid = f'<geom name="object" type="box" size="{size[0]} {size[1]} {size[2]}" mass="{mass}" {surface}/>'
    elif shape == "cylinder":
        solid = f'<geom name="object" type="cylinder" size="{size[0]} {size[1]}" mass="{mass}" {surface}/>'
    elif shape == "bottle":  # a body and a neck on top of it, four fifths of the mass in the body
        solid = (f'<geom name="object" type="cylinder" size="{size[0]} {size[1]}" mass="{0.8 * mass}" {surface}/>'
                 f'<geom name="neck" type="cylinder" size="{0.45 * size[0]} {0.4 * size[1]}" pos="0 0 {1.4 * size[1]}" '
                 f'mass="{0.2 * mass}" {surface}/>')
    else:
        raise ValueError(f"unknown object shape {shape!r}")
    motors = "".join(f'<motor joint="f{k}h{i}"/>' for k in range(fingers) for i in range(f.links))
    slides = "".join(f'<position joint="f{k}s{i}" kp="{SLIDE_KP}" kv="{SLIDE_KV}"/>' for k in range(fingers)
                     for i in range(1, f.links))
    return f'''<mujoco model="reduced gripper">
<option timestep="{timestep}" gravity="0 0 {-gravity}" cone="elliptic" impratio="10" integrator="implicitfast"/>
<worldbody>
<light pos="0 0 1"/>
<geom name="table" type="plane" size="1 1 0.1" friction="{friction} 0.005 0.0001" contype="{WORLD}" conaffinity="{WORLD | FINGER}"/>
<body name="palm" pos="0 0 {height}"><joint name="lift" type="slide" axis="0 0 1"/><inertial pos="0 0 0" mass="0.3" diaginertia="1e-3 1e-3 1e-3"/>{body}</body>
<body name="object" pos="0 0 {half + 1e-4}"><freejoint name="object"/>{solid}</body>
</worldbody>
<actuator>{motors}{slides}<position joint="lift" kp="3000" kv="150"/></actuator>
</mujoco>'''


class ReducedGripper:
    """A running reduced gripper. ``step(pressure, lift)``: one timestep at pressure [Pa] and palm height change [m]."""

    def __init__(self, finger: ReducedFinger, fingers: int = 2, **kwargs):
        self.finger, self.fingers = finger, fingers
        self.xml = gripper_xml(finger, fingers, **kwargs)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m, n = self.model, finger.links
        name = lambda k, i, c: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"f{k}{c}{i}")
        self._hinge = np.array([[m.jnt_qposadr[name(k, i, "h")] for i in range(n)] for k in range(fingers)])
        self._slide = np.array([[m.jnt_qposadr[name(k, i, "s")] for i in range(1, n)] for k in range(fingers)])
        self._object = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "object")
        self._geoms = set(np.nonzero(m.geom_bodyid == self._object)[0].tolist())
        self._nh = fingers * n
        self._ns = fingers * (n - 1)
        self._last_length = float(finger.lengths[0, -1])

    def step(self, pressure: float, lift: float = 0.0) -> None:
        d, f = self.data, self.finger
        angles, lengths, k = f.at(pressure)
        self._last_length = float(lengths[-1])
        # the finger frame's z is outward and curling is towards -z: a joint angle a is a hinge angle -a about +y
        q = d.qpos[self._hinge]
        d.ctrl[:self._nh] = (k * (-angles[None, :] - q)).ravel()
        stretch = lengths[:-1] - f.lengths[0, :-1]
        d.ctrl[self._nh:self._nh + self._ns] = np.tile(stretch, self.fingers)
        d.ctrl[-1] = lift
        mujoco.mj_step(self.model, d)

    @property
    def object_height(self) -> float:
        return float(self.data.xpos[self._object, 2])

    @property
    def palm_rise(self) -> float:
        """The palm's actual rise [m]: its position actuator lags the height ``step`` asks for."""
        return float(self.data.qpos[self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "lift")]])

    def layout(self) -> dict:
        """What a client of the model needs to drive it without its own lookups: the page uses this, not a copy."""
        m, n = self.model, self.finger.links
        return dict(hinge=self._hinge.tolist(), links=[[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"f{k}l{i}")
                                                        for i in range(n)] for k in range(self.fingers)],
                    object=int(self._object), objectGeoms=sorted(self._geoms), nh=self._nh, ns=self._ns,
                    objectQpos=int(m.jnt_qposadr[m.body_jntadr[self._object]]))

    def tip(self, k: int = 0) -> np.ndarray:
        """World position of finger ``k``'s tip on the layer."""
        m, d, n = self.model, self.data, self.finger.links
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"f{k}l{n - 1}")
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ np.array([self._last_length, 0, 0])

    def grip_force(self) -> float:
        """Mean normal force [N] a finger presses on the object with (contacts on the object, not from the table)."""
        m, d, total = self.model, self.data, 0.0
        force = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            if self._geoms & {c.geom1, c.geom2} and 0 not in (c.geom1, c.geom2):
                mujoco.mj_contactForce(m, d, i, force)
                total += force[0]
        return total / self.fingers


def run_grasp(f: ReducedFinger, pressure: float, mass: float, friction: float = 0.8, shape: str = "box", fingers: int = 2,
              lift: float = 0.04, held_fraction: float = 0.8, **kwargs) -> dict:
    """One scripted grasp: ramp to ``pressure`` [Pa] over 0.6 s, settle to 1.5 s, lift by ``lift`` [m] over 0.5 s, hold
    to 3.5 s. ``held``: the object ended at least ``held_fraction`` of the lift above where it started.
    ``grip``: :meth:`ReducedGripper.grip_force` [N] just before the lift."""
    g = ReducedGripper(f, fingers=fingers, shape=shape, size=OBJECTS[shape], mass=mass, friction=friction, **kwargs)
    for i in range(1500):
        g.step(pressure * min(i / 600, 1))
    z0, grip = g.object_height, g.grip_force()
    for i in range(2000):
        g.step(pressure, lift * min(i / 500, 1))
    rise = g.object_height - z0
    return dict(held=bool(rise > held_fraction * lift), rise=rise, grip=grip)
