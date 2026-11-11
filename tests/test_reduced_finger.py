"""The fast finger: tables from a known shape, and the MuJoCo chain obeying them (docs/gripper.md). No reference run
is needed here: the frames are built from an exact arc whose curvature and stretch grow with pressure."""

from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")


from softsim.reduced import ReducedFinger, ReducedFingerConfig, ReducedGripper, extract_tables
from softsim.reduced.finger import fit_stiffness, pad_load, tip_load
from softsim.reduced.mjcf import run_grasp

LENGTH = 0.1
PRESSURES = np.linspace(0, 40e3, 81)


def _arc_frames(curl=2.0, stretch=0.3):
    """A strip of vertices bending into an arc of total angle ``curl`` p/pmax towards -z, stretching by ``stretch``
    p/pmax. Rest vertices: stations along x in the layer, two rows across y."""
    xs = np.linspace(0, LENGTH, 401)
    rest = np.array([(x, y, 0.002) for x in xs for y in (-0.01, 0.01)])
    frames = []
    for p in PRESSURES / PRESSURES.max():
        s = rest[:, 0] * (1 + stretch * p)
        k = curl * p / (LENGTH * (1 + stretch * p))
        th = k * s
        x = np.sin(th) / k if k > 0 else s
        z = (np.cos(th) - 1) / k if k > 0 else 0 * s
        frames.append(np.column_stack([x, rest[:, 1], rest[:, 2] + z]))
    return rest, np.array(frames)


@pytest.fixture(scope="module")
def finger():
    rest, x = _arc_frames()
    p, a, l = extract_tables(rest, x, PRESSURES, LENGTH, ReducedFingerConfig(station_width=0.001, layer=(0.001, 0.003)))
    return ReducedFinger(p, a, l, np.full(len(p), 0.05)), rest, x


def test_tables_recover_the_arc(finger):
    f, *_ = finger
    assert f.angles[-1].sum() == pytest.approx(-2.0 * (1 - 0.5 / 8), rel=0.03)  # first and last half links: chords
    chord_over_arc = np.sin(2.0 / 16) / (2.0 / 16)  # 8 straight links on an arc of 2 rad
    assert f.lengths[-1].sum() / f.lengths[0].sum() == pytest.approx(1.3 * chord_over_arc, rel=1e-3)


def test_centreline_reproduces_the_frames(finger):
    f, rest, x = finger
    for k in (20, 50, 80):
        tip = x[k][np.argmax(rest[:, 0]), [0, 2]] - (0.0, 0.002)
        assert np.linalg.norm(f.centreline(PRESSURES[k])[-1] - tip) < 3e-4


def test_stiffness_does_not_change_the_unloaded_shape(finger):
    f, *_ = finger
    tips = []
    for k in (0.02, 0.2):
        g = ReducedGripper(ReducedFinger(f.pressures, f.angles, f.lengths, np.full(len(f.pressures), k)), gravity=0.0, height=0.5)
        for i in range(2000):
            g.step(20e3 * min(i / 1000, 1))
        tips.append(g.tip(0))
    assert np.linalg.norm(tips[0] - tips[1]) < 5e-4


def test_chain_follows_its_table_and_curls_inwards(finger):
    f, *_ = finger
    g = ReducedGripper(f, gravity=0.0, height=0.5)
    for i in range(2500):
        g.step(30e3 * min(i / 1500, 1))
    c = f.centreline(30e3)[-1]
    want = np.array([0.025 + f.offset + c[1], 0.0, 0.5 - c[0]])
    assert np.linalg.norm(g.tip(0) - want) < 2e-4
    assert g.tip(0)[0] < 0.025  # towards the centre


def test_tables_survive_json(finger, tmp_path):
    f, *_ = finger
    f.to_json(tmp_path / "f.json")
    r = ReducedFinger.from_json(tmp_path / "f.json")
    assert np.array_equal(r.angles, f.angles) and np.array_equal(r.stiffness, f.stiffness) and r.offset == f.offset


def test_a_stiffer_finger_gives_less_under_a_tip_load(finger):
    """Stiffness is what the probes fit: doubling it must about halve the deflection under a load."""
    f, *_ = finger
    moved = []
    for k in (0.05, 0.1):
        g = ReducedGripper(ReducedFinger(f.pressures, f.angles, f.lengths, np.full(len(f.pressures), k)), gravity=0.0, height=0.5)
        for i in range(1500):
            g.step(10e3)
        before = g.tip(0).copy()
        b = g.model.body("f0l7").id
        for i in range(2500):
            g.data.xfrc_applied[b, 0] = 0.1
            g.step(10e3)
        moved.append(np.linalg.norm(g.tip(0) - before))
    assert moved[0] / moved[1] == pytest.approx(2.0, rel=0.15)


def _straight(k=0.05, n=8, length=0.1):
    p = np.array([0.0, 40e3])
    return ReducedFinger(p, np.zeros((2, n)), np.full((2, n), length / n), np.full(2, k))


def test_small_tip_load_matches_the_hinged_cantilever():
    """Each hinge turns by F (L - x_i) / k and lifts the tip by that times (L - x_i): the sum, by hand."""
    f, force = _straight(), 1e-3
    x = np.arange(8) * 0.1 / 8
    expected = force * ((0.1 - x) ** 2).sum() / 0.05
    assert f.loaded(0.0, tip_load(8, force))[-1, 1] == pytest.approx(expected, rel=2e-2)


def test_spread_load_bends_less_than_the_same_load_at_the_tip():
    f = _straight()
    assert 0 < f.loaded(0.0, pad_load(8, 0.01))[-1, 1] < 0.5 * f.loaded(0.0, tip_load(8, 0.01))[-1, 1]


def test_stiffness_fit_recovers_a_known_stiffness(finger):
    """Synthetic probes made by a finger whose stiffness doubles with pressure: the fit returns that law."""
    f, *_ = finger
    law = lambda p: 0.04 * (1 + p / 40e3)
    probes = []
    for p in (5e3, 20e3, 35e3):
        forces = [0.05, 0.1]
        probes.append(dict(pressure=p, forces=forces, free=f.centreline(p).tolist(),
                           tip=[f.loaded(p, tip_load(f.links, F), law(p)).tolist() for F in forces]))
    table = fit_stiffness(f, probes)
    for p in (5e3, 20e3, 35e3):
        assert np.interp(p, f.pressures, table) == pytest.approx(law(p), rel=1e-3)


def test_mujoco_chain_agrees_with_the_static_balance(finger):
    """The same load on the running chain and in :meth:`ReducedFinger.loaded`: one model, two solvers."""
    f, *_ = finger
    g = ReducedGripper(f, fingers=1, gravity=0.0, height=0.5, gap=0.05)
    m, d, force, p = g.model, g.data, 0.05, 20e3
    last = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"f0l{f.links - 1}")
    for i in range(6000):
        d.qfrc_applied[:] = 0
        mujoco.mj_applyFT(m, d, np.array([force, 0, 0]), np.zeros(3), g.tip(0), last, d.qfrc_applied)
        g.step(p * min(i / 2000, 1))
    want = f.loaded(p, tip_load(f.links, force))[-1]
    got = g.tip(0)
    r = 0.05 / 2 + f.offset  # finger frame x runs down the world, z out along world x
    assert np.linalg.norm([got[0] - r - want[1], 0.5 - got[2] - want[0]]) < 2e-3


def test_friction_sets_how_much_the_gripper_can_lift():
    """Two fingers pressing with N each carry at most 2 mu N: a mass well under that rises, one well over stays. The
    finger curls gently here, as the real one does at this pressure: one that curls right under the object hooks it."""
    rest, x = _arc_frames(curl=1.0)
    p, a, l = extract_tables(rest, x, PRESSURES, LENGTH, ReducedFingerConfig(station_width=0.001, layer=(0.001, 0.003)))
    f = ReducedFinger(p, a, l, np.full(len(p), 0.05))
    grip = run_grasp(f, 30e3, 0.002, friction=0.8)["grip"]
    limit = 2 * 0.8 * grip / 9.81  # kg
    assert grip > 0
    assert run_grasp(f, 30e3, 0.6 * limit, friction=0.8)["held"]
    assert not run_grasp(f, 30e3, 1.5 * limit, friction=0.8)["held"]
    assert not run_grasp(f, 30e3, 0.6 * limit, friction=0.3)["held"]  # the same mass with less friction


def test_no_pressure_no_grasp(finger):
    f, *_ = finger
    out = run_grasp(f, 0.0, 0.01)
    assert out["grip"] == 0 and not out["held"]


def test_held_means_it_lasts():
    """A 10 g cylinder at 30 kPa is still up 2 s after the lift and back on the table by 20 s: that is a drop."""
    f = ReducedFinger.from_json(Path(__file__).resolve().parents[1] / "softsim" / "reduced" / "wang_finger.json")
    assert run_grasp(f, 30e3, 0.01, shape="cylinder", end=3.5)["held"]
    assert not run_grasp(f, 30e3, 0.01, shape="cylinder")["held"]


def test_palm_rise_is_where_the_palm_is_not_where_it_was_asked_to_go():
    f = ReducedFinger.from_json(Path(__file__).resolve().parents[1] / "softsim" / "reduced" / "wang_finger.json")
    g = ReducedGripper(f, shape="box", size=(0.016, 0.012, 0.03), mass=0.01, friction=0.8)
    palm = mujoco.mj_name2id(g.model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    mujoco.mj_kinematics(g.model, g.data)
    start = g.data.xpos[palm, 2] - g.palm_rise
    for i in range(300):
        g.step(0.0, 0.03 * i / 300)
    mujoco.mj_kinematics(g.model, g.data)  # xpos is from before the step's integration
    assert g.data.xpos[palm, 2] - start == pytest.approx(g.palm_rise, abs=1e-9)
    assert 0 < g.palm_rise < 0.03 * 299 / 300
