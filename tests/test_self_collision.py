"""Soft-body self-collision (roadmap M9): a two-link cable finger pulled until its slot closes on itself.

Checked against geometry, not another solver: the signed gap between the slot's two walls
(``softsim.contact.gap.region_gap``; negative = one wall has gone into the other). The slot is 2 mm wide, the cable
pull closes it fully, so without self-collision the walls must pass through each other (the test can fail) and with
it they must not. fem and rod have no self-contact and must refuse."""

from pathlib import Path

import numpy as np
import pytest

from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.contact.gap import region_gap, signed_distances
from softsim.errors import UnsupportedFeatureError
from softsim.runner import Simulation
from softsim.scenarios.inflation import InflationConfig, InflationScenario

FINGER = Path(__file__).parents[1] / "scenarios" / "actuation" / "self_collision_finger.yaml"
TOL = 1e-4  # [m] allowed overlap with self-collision on (the contact skin is 0.1 mm; measured: faces stay >= +0.1 mm apart)
PASS_THROUGH = -5e-4  # [m] without it the walls overlap by more than this (measured -1.3 to -1.4 mm on both backends)


def _min_gap(backend, self_collision, extra=()):
    sim = Simulation(FINGER, ["output.save_model=false", f"body.self_collision={str(self_collision).lower()}", *extra],
                     backend=backend)
    try:
        ts = sim.run().timeseries
    finally:
        sim.close()
    return float(ts.slot_gap.min())


def test_region_gap_is_the_signed_distance():
    """Two parallel unit squares 0.3 apart (outward normals facing each other): gap +0.3; slid through: -0.1."""
    def sheet(z, up):
        p = [[0, 0, z], [1, 0, z], [1, 1, z], [0, 1, z]]
        return p, ([[0, 1, 2], [0, 2, 3]] if up else [[0, 2, 1], [0, 3, 2]])
    lo, t_lo = sheet(0.0, True)       # normal +z
    hi, t_hi = sheet(0.3, False)      # normal -z
    x = np.array(lo + hi, float)
    tris_lo, tris_hi = np.array(t_lo), np.array(t_hi) + 4
    assert region_gap(x, tris_lo, tris_hi) == pytest.approx(0.3)
    x[4:, 2] = -0.1  # the upper sheet now below the lower one
    assert region_gap(x, tris_lo, tris_hi) == pytest.approx(-0.1)
    assert signed_distances(np.array([[0.5, 0.5, 0.2]]), x, tris_lo)[0] == pytest.approx(0.2)


def test_config_default_off_and_strict():
    raw = load_yaml(FINGER)
    assert raw["body"]["self_collision"] is True
    off = InflationScenario(validate(InflationConfig, apply_overrides(raw, ["body.self_collision=false"])), FINGER.parent)
    assert off.build_scene().soft_bodies[0].self_collision is False
    del raw["body"]["self_collision"]
    default = InflationScenario(validate(InflationConfig, raw), FINGER.parent)
    assert default.build_scene().soft_bodies[0].self_collision is False
    with pytest.raises(Exception, match="Extra inputs"):
        validate(InflationConfig, apply_overrides(raw, ["body.self_collide=true"]))


@pytest.mark.parametrize("backend", ["fem", "rod"])
def test_backends_without_self_contact_refuse(backend):
    with pytest.raises(UnsupportedFeatureError, match="self-collision"):
        Simulation(FINGER, ["output.save_model=false"], backend=backend)


def test_mujoco_walls_do_not_pass_through():
    pytest.importorskip("mujoco")
    on, off = _min_gap("mujoco", True), _min_gap("mujoco", False)
    assert on > -TOL, on
    assert off < PASS_THROUGH, off


@pytest.mark.slow
def test_sofa_walls_do_not_pass_through():
    on, off = _min_gap("sofa", True, ["simulation.timestep=0.5ms"]), _min_gap("sofa", False, ["simulation.timestep=0.5ms"])
    assert on > -TOL, on
    assert off < PASS_THROUGH, off
