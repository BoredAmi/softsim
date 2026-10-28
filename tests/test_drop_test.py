"""Headless drop-test runs: sanity, no penetration, stability, convergence, actions, analytic check."""

import math

import numpy as np
import pytest

pytest.importorskip("mujoco")

from softsim.config.loader import apply_overrides, load_yaml  # noqa: E402
from softsim.errors import StabilityError  # noqa: E402
from softsim.runner import Simulation, run_config  # noqa: E402

from conftest import DROP_DIR  # noqa: E402


def test_short_headless_run(drop_config, drop_base_dir, tmp_path):
    result = run_config(drop_config, base_dir=drop_base_dir, out_dir=tmp_path / "run")
    ts, s = result.timeseries, result.summary
    assert np.isfinite(ts.to_numpy()).all()
    assert s["contact"] == 1.0
    assert s["impact_speed"] == pytest.approx(math.sqrt(2 * 9.81 * 0.38))
    # first impact: indentation of a few mm, peak within a few ms, weight rebounds
    assert 0.5e-3 < s["peak_indentation"] < 8e-3
    assert 0.5e-3 < s["t_to_peak"] < 8e-3
    assert 0 < s["rebound_speed"] < s["impact_speed"]
    assert s["max_bulge_diameter"] > 0.040
    for name in ("config.yaml", "summary.yaml", "timeseries.parquet"):
        assert (tmp_path / "run" / name).exists()


def test_no_penetration(drop_config, drop_base_dir):
    ts = run_config(drop_config, base_dir=drop_base_dir).timeseries
    tol = 2.73 * 0.0002 + 0.0002  # contact compliance v*timeconst + skin radius
    # weight bottom never sinks below the top-centre surface of the specimen it presses on
    assert (ts["weight_z"] >= ts["top_center_z"] - tol).all()
    # specimen never sinks into the ground, weight never reaches the ground
    assert ts["specimen_min_z"].min() > -tol
    assert ts["weight_z"].min() > 0.005


def test_divergence_is_reported(drop_config, drop_base_dir):
    sim = Simulation(drop_config, base_dir=drop_base_dir)
    sim.backend.data.qvel[:] = np.nan
    with pytest.raises(StabilityError, match="diverged"):
        sim.step()


def test_explicit_integrator_refused(drop_config, drop_base_dir):
    from softsim.errors import UnsupportedFeatureError

    with pytest.raises((StabilityError, UnsupportedFeatureError)):
        Simulation(apply_overrides(drop_config, ["simulation.integrator=Euler"]), base_dir=drop_base_dir)


@pytest.mark.slow
def test_timestep_convergence(drop_config, drop_base_dir):
    """At a fixed contact timeconst, peak indentation at dt = 0.2 ms and 0.1 ms agree within 3%.

    (With timeconst 'auto' = 2 dt the contact model itself changes with dt.)
    """
    peaks = []
    for dt in (0.0002, 0.0001):
        r = run_config(apply_overrides(drop_config, [f"simulation.timestep={dt}", "simulation.duration=0.008",
                                                     "contact.timeconst=0.0004"]),
                       base_dir=drop_base_dir)
        peaks.append(r.summary["peak_indentation"])
    assert peaks[0] == pytest.approx(peaks[1], rel=0.03)


@pytest.mark.slow
def test_stable_heavy_drop(drop_base_dir):
    """500 g (1.86 J into a 17 mm cylinder) stays finite and does not crush through with Neo-Hookean."""
    raw = apply_overrides(load_yaml(DROP_DIR / "cases" / "w500.yaml"), [
        "specimen.mesh.size=0.005", "simulation.duration=0.012", "simulation.timestep=0.0002",
        "output.save_model=false"])
    s = run_config(raw, base_dir=drop_base_dir).summary
    assert 0 < s["peak_indentation"] < 0.017


def test_drop_action_and_set_material(drop_config, drop_base_dir):
    sim = Simulation(apply_overrides(drop_config, ["simulation.duration=0.006"]), base_dir=drop_base_dir)
    base = sim.run().summary["peak_indentation"]
    sim.set_material("specimen", {"preset": "ecoflex_00_10", "elastic": {"youngs_modulus": "40 kPa"}})
    assert sim.time == 0.0
    soft = sim.run().summary["peak_indentation"]
    assert soft > base * 1.2
    assert sim.raw_config["specimen"]["material"]["elastic"]["youngs_modulus"] == 40e3
    sim.action("drop", preset="czopek2026_w050", height=0.2)
    assert sim.scene.rigid_bodies[0].mass == pytest.approx(0.050)
    assert sim.scenario.impact_speed() == pytest.approx(math.sqrt(2 * 9.81 * 0.2))
    sim.action("drop", mass=0.03)
    assert sim.scene.rigid_bodies[0].mass == pytest.approx(0.030)
    sim.close()


@pytest.mark.parametrize("model", ["neo_hookean", "stvk"])
def test_static_uniaxial_compression(model, drop_base_dir):
    """Analytic check delta = m g H / (E A) at 3% strain.

    Frictionless rigid disc wider than the cylinder on a frictionless table gives
    homogeneous uniaxial stress, which linear tetrahedra represent exactly. The
    disc is checked to stay centred: with zero friction it eventually drifts off.
    """
    E, H, R, g, strain = 50e3, 0.017, 0.02, 9.81, 0.03
    mass = strain * E * math.pi * R**2 / g
    raw = load_yaml(DROP_DIR / "scenario.yaml")
    raw["specimen"] = {**raw["specimen"], "mesh": {"size": 0.006}, "material": {
        "density": 1000, "elastic": {"model": model, "youngs_modulus": E, "poisson_ratio": 0.3},
        "damping": {"model": "rayleigh", "beta": 0.01}, "friction": {"sliding": 0}}}
    raw["specimen"].pop("mass")
    raw["weight"] = {"name": "weight", "friction": {"sliding": 0},
                     "body": {"mass": mass, "shape": {"type": "cylinder", "diameter": 0.044, "height": 0.005}}}
    raw["ground"] = {"friction": {"sliding": 0}}
    raw = apply_overrides(raw, ["drop.initial_speed=0", "drop.gap=0.0001", "simulation.timestep=0.0002"])
    sim = Simulation(raw, base_dir=drop_base_dir)
    points = sim.scene.soft_bodies[0].mesh.points
    top, bottom = np.isclose(points[:, 2], H), np.isclose(points[:, 2], 0.0)

    def compression():
        v = sim.state().soft["specimen"].vertices
        return H - (v[top, 2].mean() - v[bottom, 2].mean())

    sim.step(750)   # 0.15 s
    early = compression()
    sim.step(250)   # 0.20 s
    late = compression()
    sim.close()
    assert np.linalg.norm(sim.state().rigid["weight"].position[:2]) < 0.5e-3
    assert late == pytest.approx(early, rel=0.02), "not settled"
    assert late == pytest.approx(strain * H, rel=0.05)
