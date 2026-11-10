"""Backend interface and simulation state types.

A backend owns one compiled model at a time. Lifecycle::

    backend = MujocoBackend()
    backend.check(scene)      # raises UnsupportedFeatureError listing every gap
    backend.build(scene)      # compile; state is at the scene's initial condition
    backend.step(n)           # advance n physics steps
    state = backend.state()   # SimState snapshot (copies, safe to keep)
    backend.reset()           # back to the initial condition

Actuators follow their schedules unless a controller sets them:
``backend.set_control("pressure", 20e3)`` holds that value from the next step on
(until ``release_control`` or a new ``build``); ``backend.control(name)`` is the
value in effect.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import ClassVar

import numpy as np

from softsim.actuation import CableActuator
from softsim.capabilities import Capability
from softsim.errors import UnsupportedFeatureError
from softsim.materials.models import Material
from softsim.scene import Scene


@dataclass(frozen=True)
class RigidState:
    """Pose and twist of a rigid body frame (world frame, SI); ``com`` is the centre of mass."""

    position: np.ndarray
    quaternion: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray
    com: np.ndarray | None = None


@dataclass(frozen=True)
class SoftState:
    """Current vertex positions/velocities of a soft body, indexed like its mesh points.

    ``reactions`` maps each prescribed region (``SoftBodySpec.prescribed``) to the force its support
    exerts on the body [N, world axes; zero along free axes]: e.g. positive along a pull."""

    vertices: np.ndarray
    velocities: np.ndarray
    reactions: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class ContactWrench:
    """Net contact interaction between two named bodies.

    ``force`` is the total force exerted *by* ``body_a`` *on* ``body_b`` (world
    frame). ``point`` is the normal-force-weighted centre of pressure and
    ``torque`` the remaining moment about it. ``ground`` is the name used for
    the static ground.
    """

    body_a: str
    body_b: str
    force: np.ndarray
    torque: np.ndarray
    point: np.ndarray
    n_contacts: int

    def on(self, body: str) -> np.ndarray:
        """Force acting on ``body`` (sign-flipped if ``body`` is ``body_a``)."""
        if body == self.body_b:
            return self.force
        if body == self.body_a:
            return -self.force
        raise KeyError(body)


@dataclass(frozen=True)
class SimState:
    """Snapshot of the whole simulation."""

    time: float
    rigid: dict[str, RigidState] = field(default_factory=dict)
    soft: dict[str, SoftState] = field(default_factory=dict)
    contacts: list[ContactWrench] = field(default_factory=list)
    controls: dict[str, float] = field(default_factory=dict)  # actuator values in effect

    def contact_between(self, a: str, b: str) -> ContactWrench | None:
        """Aggregated contact between ``a`` and ``b``, oriented as force by ``a`` on ``b``."""
        for c in self.contacts:
            if (c.body_a, c.body_b) == (a, b):
                return c
            if (c.body_a, c.body_b) == (b, a):
                return ContactWrench(a, b, -c.force, -c.torque, c.point, c.n_contacts)
        return None

    def net_contact_force(self, body: str) -> np.ndarray:
        """Sum of all contact forces acting on ``body``."""
        total = np.zeros(3)
        for c in self.contacts:
            if body in (c.body_a, c.body_b):
                total += c.on(body)
        return total


ELASTIC_LAWS = frozenset({Capability.LINEAR_ELASTIC, Capability.STVK, Capability.NEO_HOOKEAN,
                          Capability.STABLE_NEO_HOOKEAN, Capability.MOONEY_RIVLIN, Capability.YEOH,
                          Capability.OGDEN, Capability.GENT, Capability.ARRUDA_BOYCE})


# ``softsim material project --to`` target for each law capability
PROJECTABLE = {Capability.STABLE_NEO_HOOKEAN: "neo_hookean", Capability.NEO_HOOKEAN: "neo_hookean",
               Capability.MOONEY_RIVLIN: "mooney_rivlin", Capability.YEOH: "yeoh", Capability.OGDEN: "ogden1",
               Capability.GENT: "gent", Capability.ARRUDA_BOYCE: "arruda_boyce"}


class Backend(ABC):
    """Physics backend interface."""

    name: ClassVar[str]
    capabilities: ClassVar[frozenset[Capability]]
    locking_free: ClassVar[bool] = False  # elements that do not lock near incompressibility (no locking warning)
    implicit_actuation: ClassVar[bool] = False  # actuator loads in the implicit solve (no explicit-stability limit)
    implicit_prony: ClassVar[bool] = False  # Prony overstress in the implicit solve (no explicit-stability limit)

    @classmethod
    def unsupported(cls, scene: Scene) -> list[str]:
        """Descriptions of everything in ``scene`` this backend cannot simulate."""
        missing = scene.requires - cls.capabilities
        out = sorted(c.value for c in missing)
        if missing & ELASTIC_LAWS:
            laws = sorted({PROJECTABLE[c] for c in cls.capabilities if c in PROJECTABLE})
            out.append("(fit a law this backend has to the card: `softsim material project <card> --to "
                       f"{'|'.join(laws)}`)")
        return out + cls.extra_checks(scene)

    @classmethod
    def extra_checks(cls, scene: Scene) -> list[str]:
        """Backend-specific rejections beyond the capability set (e.g. settings)."""
        return []

    @classmethod
    def check(cls, scene: Scene) -> None:
        """Raise UnsupportedFeatureError if the scene cannot be simulated faithfully."""
        missing = cls.unsupported(scene)
        if missing:
            raise UnsupportedFeatureError(cls.name, missing)

    @abstractmethod
    def build(self, scene: Scene) -> None:
        """Compile ``scene`` and set the initial condition."""

    @abstractmethod
    def step(self, n: int = 1) -> None:
        """Advance ``n`` physics steps."""

    @abstractmethod
    def state(self) -> SimState:
        """Snapshot of the current state."""

    @abstractmethod
    def reset(self) -> None:
        """Return to the initial condition of the current scene."""

    @property
    @abstractmethod
    def time(self) -> float:
        """Current simulation time [s]."""

    @property
    @abstractmethod
    def scene(self) -> Scene:
        """The scene currently built."""

    def is_finite(self) -> bool:
        """False once the state contains NaN/inf (divergence)."""
        s = self.state()
        arrays = [r.position for r in s.rigid.values()] + [v.vertices for v in s.soft.values()]
        return all(np.isfinite(a).all() for a in arrays)

    def mount_gravity(self, t: float) -> np.ndarray:
        """Gravity in the mount's frame at ``t``: ``R^T (g - p'')`` (the scene's gravity without a moving or
        rotating mount)."""
        return self.scene.frame(t).gravity(np.asarray(self.scene.simulation.gravity, float))

    def _to_world(self, state: SimState) -> SimState:
        """``state`` with the mount's motion added to the soft bodies' (rotation and translation)."""
        scene = self.scene
        if scene.mount is None and scene.spin is None:
            return state
        frame = scene.frame(state.time)
        soft = {}
        for n, s in state.soft.items():
            x, v = frame.to_world(s.vertices, s.velocities)
            soft[n] = SoftState(x, v, s.reactions)
        return replace(state, soft=soft)

    def set_control(self, name: str, value: float) -> None:
        """Override actuator ``name`` with ``value`` (SI: a pressure, a tension, or a stroke-controlled
        cable's stroke) from the next step on."""
        actuator = self.scene.actuator(name)
        if isinstance(actuator, CableActuator) and not actuator.stroke_controlled and value < 0:
            raise ValueError(f"actuator {name!r}: a cable can only pull, tension must be >= 0 (got {value})")
        self.__dict__.setdefault("_control_overrides", {})[name] = float(value)

    def release_control(self, name: str) -> None:
        """Return actuator ``name`` to its schedule."""
        self.__dict__.get("_control_overrides", {}).pop(name, None)

    def control(self, name: str, t: float | None = None) -> float:
        """Value of actuator ``name`` at time ``t`` (default: now): the override, else its schedule."""
        overrides = self.__dict__.get("_control_overrides", {})
        if name in overrides:
            return overrides[name]
        return self.scene.actuator(name).value(self.time if t is None else t)

    def controls(self) -> dict[str, float]:
        """Every actuator's current value."""
        return {a.name: self.control(a.name) for a in self.scene.actuators}

    def _clear_controls(self) -> None:
        self.__dict__.pop("_control_overrides", None)

    def checkpoint(self) -> dict[str, np.ndarray]:
        """The running state as arrays, for :meth:`restore` on a backend built from the same scene."""
        raise NotImplementedError(f"the {self.name} backend has no checkpoints")

    def restore(self, state: dict[str, np.ndarray]) -> None:
        """Go on from a :meth:`checkpoint`."""
        raise NotImplementedError(f"the {self.name} backend has no checkpoints")

    def set_material(self, body_name: str, material: Material) -> None:
        """Replace a soft body's material; the model is rebuilt and reset."""
        new_scene = self.scene.with_material(body_name, material)
        self.check(new_scene)
        self.build(new_scene)

    def export_model(self) -> str | None:
        """Backend-native model text (e.g. MJCF) for inspection, if available."""
        return None

    def close(self) -> None:
        """Release resources."""
