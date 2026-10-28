"""SOFA backend (stub, planned for milestone M4).

SOFA (https://www.sofa-framework.org) with the SoftRobots plugin provides
proper FEM: corotational linear and hyperelastic tetrahedral force fields
(Neo-Hookean, Mooney-Rivlin, Ogden, ...), implicit Euler and Lagrange-multiplier
contact. The capability set below is what the planned implementation will
support; it lets ``softsim check --backend sofa`` report gaps today. Items
still to be verified against SOFA's actual force fields are listed in
``UNVERIFIED``.
"""

from __future__ import annotations

from typing import ClassVar

from softsim.backends.base import Backend, SimState
from softsim.capabilities import Capability
from softsim.errors import SoftSimError
from softsim.scene import Scene

UNVERIFIED = frozenset({Capability.YEOH, Capability.VISCOELASTIC_PRONY})


class SofaBackend(Backend):
    """Placeholder: declares capabilities, refuses to build."""

    name: ClassVar[str] = "sofa"
    capabilities: ClassVar[frozenset[Capability]] = frozenset({
        Capability.LINEAR_ELASTIC,
        Capability.STVK,
        Capability.NEO_HOOKEAN,
        Capability.MOONEY_RIVLIN,
        Capability.OGDEN,
        Capability.DAMPING_STIFFNESS_PROPORTIONAL,
        Capability.DAMPING_MASS_PROPORTIONAL,
        Capability.TET_MESH,
        Capability.PINNED_VERTICES,
        Capability.RIGID_BODIES,
        Capability.CONTACT_FRICTION,
        Capability.SELF_COLLISION,
        Capability.CONTACT_FORCES,
    })

    def build(self, scene: Scene) -> None:
        self.check(scene)
        raise SoftSimError("the SOFA backend is not implemented yet (planned for milestone M4)")

    def _unbuilt(self):
        raise SoftSimError("SOFA backend has no model; build() is not implemented yet")

    def step(self, n: int = 1) -> None:
        self._unbuilt()

    def state(self) -> SimState:
        self._unbuilt()

    def reset(self) -> None:
        self._unbuilt()

    @property
    def time(self) -> float:
        self._unbuilt()

    @property
    def scene(self) -> Scene:
        self._unbuilt()
