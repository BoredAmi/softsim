"""Backend interface and simulation state types.

A backend owns one compiled model at a time. Lifecycle::

    backend = MujocoBackend()
    backend.check(scene)      # raises UnsupportedFeatureError listing every gap
    backend.build(scene)      # compile; state is at the scene's initial condition
    backend.step(n)           # advance n physics steps
    state = backend.state()   # SimState snapshot (copies, safe to keep)
    backend.reset()           # back to the initial condition
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from softsim.capabilities import Capability
from softsim.errors import UnsupportedFeatureError
from softsim.materials.models import Material
from softsim.scene import Scene


@dataclass(frozen=True)
class RigidState:
    """Pose and twist of a rigid body frame (world frame, SI)."""

    position: np.ndarray
    quaternion: np.ndarray
    linear_velocity: np.ndarray
    angular_velocity: np.ndarray


@dataclass(frozen=True)
class SoftState:
    """Current vertex positions/velocities of a soft body, indexed like its mesh points."""

    vertices: np.ndarray
    velocities: np.ndarray


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


class Backend(ABC):
    """Physics backend interface."""

    name: ClassVar[str]
    capabilities: ClassVar[frozenset[Capability]]

    @classmethod
    def unsupported(cls, scene: Scene) -> list[str]:
        """Descriptions of everything in ``scene`` this backend cannot simulate."""
        return sorted(c.value for c in scene.requires - cls.capabilities) + cls.extra_checks(scene)

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
