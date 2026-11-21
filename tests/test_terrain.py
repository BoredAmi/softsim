"""Terrain on the ground (MuJoCo): static obstacle boxes and a compliant (sand-like) ground."""

import numpy as np
import pytest

from softsim.backends import get_backend
from softsim.backends.mujoco.builder import scene_to_mjcf
from softsim.config.schema import validate
from softsim.errors import SoftSimError
from softsim.materials import Material
from softsim.meshing import structured_box
from softsim.scene import GroundBox, GroundCompliance, GroundSettings, Scene, SimulationSettings, SoftBodySpec

STIFF = {"model": "stable_neo_hookean", "youngs_modulus": 1e6, "poisson_ratio": 0.3}


def _scene(ground: GroundSettings, lift: float = 0.0, duration: float = 0.3) -> Scene:
    mesh = structured_box((0.02, 0.02, 0.01), 0.005)
    mesh.points[:, 2] += lift
    material = validate(Material, {"density": 1000, "elastic": STIFF, "damping": {"model": "rayleigh", "beta": 2e-3}})
    sim = SimulationSettings(timestep=1e-4, duration=duration, output_rate=1000)
    return Scene((SoftBodySpec("block", mesh, material),), (), simulation=sim, ground=ground)


def _bottom_after(scene: Scene) -> float:
    backend = get_backend("mujoco")()
    backend.build(scene)
    backend.step(int(round(scene.simulation.duration / scene.simulation.timestep)))
    assert backend.is_finite()
    return float(backend.state().soft["block"].vertices[:, 2].min())


def test_mjcf_has_obstacles_and_compliance():
    ground = GroundSettings(obstacles=[GroundBox(center=(0, 0, 0.002), half_size=(0.01, 0.01, 0.002))],
                            compliance=GroundCompliance(timeconst=0.02))
    xml = scene_to_mjcf(_scene(ground))
    assert 'name="ground/obs0"' in xml and 'type="box"' in xml
    assert xml.count('priority="1"') == 2 and 'solref="0.02 1"' in xml


def test_block_rests_on_an_obstacle():
    """A block dropped on a 4 mm high box under its centre rests on the box, not on the ground."""
    box = GroundBox(center=(0, 0, 0.002), half_size=(0.015, 0.015, 0.002))
    bottom = _bottom_after(_scene(GroundSettings(obstacles=[box]), lift=0.005))
    assert bottom == pytest.approx(0.004, abs=0.5e-3)


def test_compliant_ground_sinks_deeper():
    hard = _bottom_after(_scene(GroundSettings(), lift=0.001))
    soft = _bottom_after(_scene(GroundSettings(compliance=GroundCompliance(timeconst=0.02)), lift=0.001))
    assert hard > -0.5e-3
    assert soft < hard - 1e-3


@pytest.mark.parametrize("ground", [
    GroundSettings(obstacles=[GroundBox(center=(0, 0, 0.002), half_size=(0.01, 0.01, 0.002))]),
    GroundSettings(compliance=GroundCompliance()),
])
def test_fem_refuses_terrain(ground):
    with pytest.raises(SoftSimError):
        get_backend("fem").check(_scene(ground))


def test_scene_requires():
    from softsim.capabilities import Capability

    assert Capability.GROUND_OBSTACLES not in _scene(GroundSettings()).requires
    req = _scene(GroundSettings(obstacles=[GroundBox(center=(0, 0, 0), half_size=(1e-3, 1e-3, 1e-3))],
                                compliance=GroundCompliance())).requires
    assert {Capability.GROUND_OBSTACLES, Capability.GROUND_COMPLIANCE} <= req
    assert np.isfinite(GroundCompliance().timeconst)
