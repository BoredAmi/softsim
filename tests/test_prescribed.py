"""Prescribed region motion (displacement control): the config, tracking, the reaction forces against
the homogeneous solution and against force control on every backend, and the momentum balance of a
fast pull (the reaction includes the region's inertia)."""

from pathlib import Path

import numpy as np
import pytest

from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError, SoftSimError
from softsim.materials import load_material
from softsim.scenarios.inflation import InflationConfig, InflationScenario
from softsim.scene import MountMotion

TENSION = Path(__file__).parents[1] / "scenarios" / "actuation" / "tension_test.yaml"
STEP = {"mujoco": [], "sofa": ["simulation.timestep=1ms"], "fem": ["simulation.timestep=5ms"]}
AREA, STROKE = 1e-4, 0.002  # the bar's cross-section [m^2] and the top's pull [m]
# a top layer of 20 000 kg/m^3: the held region's own inertia is then a third of the reaction (a
# light bar's is 2%, below SOFA's error, so a reaction without it would pass)
HEAVY_TOP = ("body.zones=[{name: weight, material: {density: 20000, elastic: {model: stable_neo_hookean, "
             "youngs_modulus: 100 kPa, poisson_ratio: 0.45}}, box: [[-1 cm, -1 cm, 17.6 mm], [1 cm, 1 cm, 21 mm]]}]")
BASE_ONLY = "body.prescribed=[{region: base, path: [[0 ms, [null, null, 0]], [300 ms, [null, null, 0]]]}]"


def _scenario(overrides=()):
    raw = apply_overrides(load_yaml(TENSION), list(overrides))
    return InflationScenario(validate(InflationConfig, raw), TENSION.parent)


def _run(backend, overrides=()):
    """Run the tension scenario; skip if the backend is not installed."""
    from softsim.runner import run_config

    if backend == "fem":
        pytest.importorskip("jax")
    try:
        return run_config(TENSION, [f"backend={backend}", *STEP[backend], *overrides, "output.save_model=false"])
    except (ImportError, SoftSimError) as exc:
        if "SOFA" in str(exc) or "Sofa" in str(exc):
            pytest.skip(str(exc))
        raise


def test_prescribed_config():
    scene = _scenario().build_scene()
    body = scene.soft_bodies[0]
    base, top = body.prescribed
    assert (base.region, base.axes, top.axes) == ("base", (False, False, True), (False, False, True))
    assert np.all(body.mesh.points[top.vertices, 2] == pytest.approx(0.02))
    np.testing.assert_allclose(top.path.position(0.1), [0, 0, 0.001])  # minimum jerk: half way at half time
    assert top.path.velocity(0.25) @ [0, 0, 1] == 0.0  # held after the last waypoint
    assert not body.pinned
    bad = {
        "an axis is null in some waypoints": ["body.prescribed[1].path=[[0 ms, [null, null, 0]], [1 s, [0, 0, 1 mm]]]"],
        "shares vertices": ["body.fixed=[base]"],
        "no region 'lid'": ["body.prescribed[1].region=lid"],
        "rest position": ["body.prescribed[1].path=[[0 ms, [null, null, 1 mm]], [1 s, [null, null, 2 mm]]]"],
        "holds no axis": ["body.prescribed[1].path=[[0 ms, [null, null, null]], [1 s, [null, null, null]]]"],
    }
    for message, overrides in bad.items():
        with pytest.raises(Exception, match=message):
            _scenario(overrides).build_scene()
    with pytest.raises(ConfigError, match="no region 'lid'"):
        _scenario(["body.prescribed[1].region=lid"]).build_scene()


def test_mujoco_refuses_partial_axes_on_a_rotated_body():
    """A MuJoCo vertex slides along its body's axes: a roller in world z on a turned body is refused."""
    import dataclasses

    from softsim.backends.mujoco import MujocoBackend
    from softsim.scene import Pose

    scene = _scenario().build_scene()
    turned = Pose((0.0, 0.0, 0.0), (0.7071068, 0.7071068, 0.0, 0.0))
    scene = dataclasses.replace(scene, soft_bodies=(dataclasses.replace(scene.soft_bodies[0], pose=turned),))
    assert any("holds only some world axes" in m for m in MujocoBackend.unsupported(scene))
    assert not MujocoBackend.unsupported(_scenario().build_scene())


@pytest.mark.parametrize("backend", ["mujoco", "sofa", "fem"])
def test_tension_reactions_tracking_and_force_control(backend):
    """10% roller tension: the top follows its path (MuJoCo's joint equalities to 1e-4 of the stroke
    with the velocity feed-forward, SOFA and the reference exactly); at the hold the base reaction
    balances the top's, the reference's equals the homogeneous solution, and on every backend a force
    control run with that force on the top face stretches the bar by the prescribed 2 mm (the
    reaction is the force the motion needs)."""
    ts = _run(backend).timeseries
    t = ts.t.to_numpy()
    path = MountMotion((0.0, 0.2), ((0, 0, 0), (0, 0, STROKE)))
    target = np.array([path.position(x)[2] for x in t])
    error = np.abs(ts.top_displacement_z.to_numpy() - target).max()
    assert error < (2e-4 * STROKE if backend == "mujoco" else 1e-8), error
    hold = ts[t >= 0.2]
    pull = hold.top_reaction_z.mean()
    assert hold.base_reaction_z.mean() == pytest.approx(-pull, rel=1e-4)
    card = load_material({"density": 1070, "elastic": {"model": "neo_hookean", "youngs_modulus": 100e3,
                                                       "poisson_ratio": 0.45}})
    from softsim.materials.homogeneous import curve

    exact = curve(card, "uniaxial", [1.1]).stress.iloc[0] * AREA
    if backend == "fem":
        assert pull == pytest.approx(exact, rel=1e-5)
    else:  # their stable Neo-Hookean (Smith et al.) stiffens 0.9% over the split Neo-Hookean at 10%
        assert pull == pytest.approx(exact, rel=0.015)

    times = np.linspace(0, 0.2, 21)
    schedule = ", ".join(f"[{1e3 * x:.1f} ms, {pull * path.position(x)[2] / STROKE:.9g}]" for x in times)
    forced = _run(backend, [BASE_ONLY, f"actuators=[{{name: pull, type: force, region: top, direction: [0, 0, 1], "
                                       f"schedule: [{schedule}, [300 ms, {pull:.9g}]]}}]"]).timeseries
    stretch = forced[forced.t >= 0.2].tip_z.mean() - 0.02  # the top face's centroid
    assert stretch == pytest.approx(STROKE, rel=2e-3)


def test_reference_tension_matches_homogeneous_at_large_strain():
    """30% stretch on the reference: the pull per area is the homogeneous nominal stress and every
    vertex contracts sideways by the homogeneous lateral stretch (the exact solution)."""
    pytest.importorskip("jax")
    from softsim.backends.fem import FemBackend
    from softsim.materials.homogeneous import curve

    scene = _scenario(["body.prescribed[1].path=[[0 ms, [null, null, 0]], [200 ms, [null, null, 6 mm]]]",
                       "simulation.timestep=5ms"]).build_scene()
    backend = FemBackend()
    backend.build(scene)
    backend.step(60)
    state = backend.state()
    exact = curve(load_material({"density": 1070, "elastic": {"model": "neo_hookean", "youngs_modulus": 100e3,
                                                             "poisson_ratio": 0.45}}), "uniaxial", [1.3])
    assert state.soft["bar"].reactions["top"][2] == pytest.approx(exact.stress.iloc[0] * AREA, rel=1e-5)
    x0 = scene.soft_bodies[0].mesh.points
    x = state.soft["bar"].vertices
    np.testing.assert_allclose(x[:, :2], x0[:, :2] * exact.lateral.iloc[0], atol=1e-8)
    np.testing.assert_allclose(x[:, 2], x0[:, 2] * 1.3, atol=1e-8)


@pytest.mark.parametrize("backend", ["mujoco", "sofa", "fem"])
def test_fast_pull_momentum_balance(backend):
    """Only the top face is held, pulled 1 mm in 20 ms (about 2.5 axial periods) under gravity, with a
    heavy top layer so that the held region's own inertia is a third of the reaction (a reaction
    without it fails on SOFA by 41%). Reaction + weight = rate of change of the body's momentum:
    * reference: its end-of-step residual balances sum(M a) of the step's end exactly (Newmark);
      measured 1.4e-6 of the peak reaction (Newton's tolerance);
    * MuJoCo: the constraint forces a step applied; its discrete integrator's velocity update differs
      from dt * forces / mass by 1.6e-4 of the peak here;
    * SOFA: one step back, and approximate in fast motion: its implicit Euler balances the forces at
      the step's end, which SofaPython3 does not expose; measured 0.9% of the peak here (exact at
      rest and in quasi-static loading, test above)."""
    from softsim.backends import get_backend
    from softsim.errors import SoftSimError

    dt = {"mujoco": 1e-4, "sofa": 1e-4, "fem": 2e-4}[backend]
    sc = _scenario(["body.prescribed=[{region: top, path: [[0 ms, [null, null, 0]], [20 ms, [null, null, 1 mm]]]}]",
                    HEAVY_TOP, "simulation.gravity=[0, 0, -9.81]", f"simulation.timestep={dt}",
                    "simulation.duration=0.03"])
    if backend == "fem":
        pytest.importorskip("jax")
    try:
        b = get_backend(backend)()
        b.build(sc.build_scene())
    except (ImportError, SoftSimError) as exc:
        pytest.skip(str(exc))
    body = b.scene.soft_bodies[0]
    weight = body.total_mass * -9.81
    m = body.vertex_masses()
    p, r, ma = [float(m @ b.state().soft["bar"].velocities[:, 2])], [], []
    for _ in range(int(round(0.03 / dt))):
        b.step(1)
        state = b.state()
        p.append(float(m @ state.soft["bar"].velocities[:, 2]))
        r.append(state.soft["bar"].reactions["top"][2])
        if backend == "fem":
            s = b.solver
            ma.append(float((s.mass_matrix @ s.a[:3 * s.n])[2::3].sum()))
    rate = np.diff(p) / dt - weight     # [k]: over step k (t_k -> t_k+1)
    r = np.array(r)                     # [k]: reported after step k
    scale = np.abs(r).max()
    inertia = np.abs(np.diff(p) / dt).max()
    print(f"{backend}: peak reaction {scale:.4f} N, peak momentum rate {inertia:.4f} N, weight {weight:.4f} N")
    assert inertia > 0.5 * scale  # the pull is dynamic: inertia, not weight, carries the reaction
    if backend == "fem":
        gap = np.abs(r - (np.array(ma) - weight)).max()
        assert gap < 1e-5 * scale, gap / scale  # Newton tolerance: the residual is the last iterate's (1.6e-6)
    elif backend == "mujoco":
        gap = np.abs(r - rate).max()
        assert gap < 3e-4 * scale, gap / scale
    else:
        gap = np.abs(r[1:] - rate[:-1]).max()  # one step back
        assert gap < 0.02 * scale, gap / scale
