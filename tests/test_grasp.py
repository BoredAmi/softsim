"""The gripper's geometry and the grasp scenario's bookkeeping (M9). The grasp physics runs are slow and live in
docs/gripper.md; here: the two fingers are the validated finger placed correctly, and held/slip mean what they say."""

import numpy as np
import pandas as pd
import pytest

from softsim.geometry import SoftGripper, SoftRibbedFinger
from softsim.geometry.soft import MeshSettings
from softsim.runner import Simulation
from softsim.scenarios.grasp import GraspConfig, GraspScenario, is_held

FINGER = SoftRibbedFinger(tooth_height=0)  # teeth need a finer, slower mesh; the placement is the same
SETTINGS = MeshSettings(size=0.003, method="gmsh")


@pytest.fixture(scope="module")
def gripper():
    g = SoftGripper(finger=FINGER.model_copy(update={"wall": 0.003, "side_wall": 0.003, "floor": 0.003}))
    m = g.mesh(SETTINGS)
    return g, m, g.regions(m)


def test_two_fingers_are_exactly_two_of_the_finger(gripper):
    g, m, _ = gripper
    one = g.finger.mesh(SETTINGS)
    assert m.n_points == 2 * one.n_points and m.n_tets == 2 * one.n_tets
    assert m.volume() == pytest.approx(2 * one.volume(), rel=1e-12)
    assert m.volume() == pytest.approx(g.volume(), rel=1e-9)


def test_fingers_are_rotated_not_mirrored(gripper):
    """A mirror image would turn every tetrahedron inside out."""
    assert gripper[1].tet_volumes().min() > 0


def test_fingers_hang_from_the_palm_with_toothed_faces_inward(gripper):
    g, m, r = gripper
    assert m.points[:, 2].max() == pytest.approx(g.height)
    assert m.points[:, 2].min() == pytest.approx(g.height - g.finger.length)
    pads = m.points[np.unique(r["pads"])]
    assert np.abs(pads[:, 0]).min() == pytest.approx(g.gap / 2)  # nothing lies inside the gap at rest
    assert np.abs(m.points[:, 0]).min() == pytest.approx(g.gap / 2)
    assert np.abs(m.points[np.unique(r["backs"])][:, 0]).min() > g.gap / 2 + g.finger.floor


def test_bases_split_by_side_and_cavity_is_closed(gripper):
    from softsim.actuation import open_edges
    _, m, r = gripper
    assert (m.points[np.unique(r["base_right"])][:, 0] > 0).all()
    assert (m.points[np.unique(r["base_left"])][:, 0] < 0).all()
    assert len(r["base_right"]) + len(r["base_left"]) == len(r["base"])
    assert len(open_edges(r["cavity"])) == 0


def test_a_gripper_that_would_hit_the_table_is_refused():
    with pytest.raises(ValueError, match="through the table"):
        SoftGripper(height=0.05)


def _summary(palm, lift):
    s = GraspScenario.__new__(GraspScenario)
    s.config = GraspConfig.model_construct(actuators=[], held_clearance=0.01, held_sink=1e-3, _prescribed={})
    s._prescribed = {}
    ts = pd.DataFrame({"t": [0.0, 1.0], "palm_lift": [0.0, palm], "object_lift": [0.0, lift], "max_z": [0.1, 0.1], "max_radius": [0.1, 0.1]})
    return s.summarize(ts)


@pytest.mark.parametrize("palm, lift, held, slip", [(0.03, 0.029, 1.0, 0.001), (0.03, 0.0, 0.0, 0.03),
                                                    (0.03, 0.017, 1.0, 0.013), (0.03, 0.009, 0.0, 0.021), (0.0, 0.0, 0.0, 0.0)])
def test_held_and_slip(palm, lift, held, slip):
    out = _summary(palm, lift)
    assert out["held"] == held and out["slip"] == pytest.approx(slip)


def test_object_stands_on_the_table():
    """The object's centre of mass stops at half its height, so its base stops on the table, whatever the backend."""
    sim = Simulation("scenarios/grasp/box.yaml", ["body.mesh.size=2mm", "object.clearance=1mm"])
    box = sim.scene.rigid_bodies[0]
    sim.close()
    assert box.floor == pytest.approx(0.03)
    assert box.pose.position[2] == pytest.approx(0.031)


def test_held_means_off_the_table_and_staying_there():
    """Up and steady is held however much of the lift the soft fingers lost; an object sliding out is not, even while
    it is still high; one that came back to the table is not."""
    t = np.linspace(0, 1, 101)
    rise = 0.017 * np.clip((t - 0.5) / 0.2, 0, 1)
    assert is_held(t, rise)
    assert not is_held(t, rise - 0.03 * np.clip(t - 0.8, 0, None))       # 6 mm lost in the last 0.2 s, still 11 mm up
    assert not is_held(t, np.where(t < 0.5, 0.03, 0.0))                  # steady again, on the table
    assert is_held(t, rise - 0.002 * np.clip(t - 0.75, 0, None))         # settling by 0.4 mm is not sliding out
