"""Muscle actuators (active stress): the nodal loads, the free bar against the homogeneous closed form, backends."""

from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import brentq

from softsim.actuation import MuscleActuator, active_stress_loads, MuscleActuatorConfig, muscle_data, muscle_forces
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError, UnsupportedFeatureError
from softsim.geometry import MeshSettings, SoftBox
from softsim.materials import homogeneous
from softsim.materials.models import Material
from softsim.runner import Simulation
from softsim.scenarios.inflation import InflationConfig, InflationScenario

E, G0 = 100e3, 1000.0  # Pa; kg/m^3


# -- the loads ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("m0", [np.array([1.0, 0.0, 0.0]), np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8]),
                                None])
def test_forces_are_self_equilibrated(m0):
    """No net force or moment on random deformed tets, for a fibre and for the isotropic stress."""
    mesh = SoftBox(size=(0.01, 0.01, 0.03)).mesh(MeshSettings(size=0.0025))
    rng = np.random.default_rng(0)
    x = mesh.points + rng.normal(size=mesh.points.shape) * 5e-4
    f = muscle_forces(x, mesh.tets, 7e3, m0, muscle_data(mesh.points, mesh.tets))
    assert np.abs(f).max() > 0.0
    np.testing.assert_allclose(f.sum(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.cross(x, f).sum(axis=0), 0.0, atol=1e-12)


def test_undeformed_patch_is_the_boundary_traction():
    """Fibre along x, uniform s: nodes inside the block get no force; the two x faces carry the traction s (sum
    s A), pulling them in (contraction); no y or z force anywhere."""
    mesh = SoftBox(size=(0.02, 0.02, 0.02)).mesh(MeshSettings(size=0.005))
    p, s = mesh.points, 5e3
    f = muscle_forces(p, mesh.tets, s, np.array([1.0, 0, 0]), muscle_data(p, mesh.tets))
    lo, hi = np.isclose(p[:, 0], p[:, 0].min()), np.isclose(p[:, 0], p[:, 0].max())
    assert np.abs(f[~(lo | hi)]).max() < 1e-12 and np.abs(f[:, 1:]).max() < 1e-12
    assert f[lo, 0].sum() == pytest.approx(s * 0.02 * 0.02) and f[hi, 0].sum() == pytest.approx(-s * 0.02 * 0.02)
    assert f[lo, 0].min() >= 0 and f[hi, 0].max() <= 0
    # isotropic: the traction s on every face, inwards
    g = muscle_forces(p, mesh.tets, s, None, muscle_data(p, mesh.tets))
    inner = np.all((p > p.min(axis=0) + 1e-9) & (p < p.max(axis=0) - 1e-9), axis=1)
    assert np.abs(g[inner]).max() < 1e-12
    for axis in range(3):
        face = np.isclose(p[:, axis], p[:, axis].min())
        assert g[face, axis].sum() == pytest.approx(s * 0.02 * 0.02)


def test_fibre_follows_the_material():
    """After a rigid rotation the fibre (and so the load) turns with the body."""
    mesh = SoftBox(size=(0.02, 0.01, 0.01)).mesh(MeshSettings(size=0.005))
    r = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # 90 degrees about z
    rest = muscle_data(mesh.points, mesh.tets)
    f0 = muscle_forces(mesh.points, mesh.tets, 3e3, np.array([1.0, 0, 0]), rest)
    f1 = muscle_forces(mesh.points @ r.T, mesh.tets, 3e3, np.array([1.0, 0, 0]), rest)
    np.testing.assert_allclose(f1, f0 @ r.T, atol=1e-12)


def test_value_and_config():
    cfg = validate(MuscleActuatorConfig, {"name": "m", "box": [["0 mm", "0 mm", "0 mm"], ["1 mm", "1 mm", "1 mm"]],
                                          "direction": [2, 0, 0], "schedule": [[0, 0], ["0.1 s", "5 kPa"]]})
    a = MuscleActuator.from_config(cfg, "b", np.arange(3))
    assert a.direction == (1.0, 0.0, 0.0) and a.value(0.05) == pytest.approx(2500.0) and a.value(5.0) == 5e3
    osc = validate(MuscleActuatorConfig, {"region": "r", "oscillator": {
        "frequency": "10 Hz", "amplitude": "4 kPa", "offset": "1 kPa", "phase": "90 deg", "start": "0.2 s"}})
    m = MuscleActuator.from_config(osc, "b", np.arange(3))
    assert m.direction is None and m.value(0.1) == 0.0 and m.value(0.2) == 0.0      # silent, starts at zero
    assert m.value(0.3) == pytest.approx(5e3)                                       # one period of ramp, then full
    assert m.value(0.325) == pytest.approx(1e3 + 4e3 * np.sin(np.pi / 2 + np.pi / 2))
    plain = MuscleActuator.from_config(validate(MuscleActuatorConfig, {"region": "r", "oscillator": {
        "frequency": 5, "amplitude": 2e3}}), "b", np.arange(3))
    assert plain.value(0.05) == pytest.approx(2e3)  # start 0: no envelope
    for bad in ({}, {"region": "r", "box": [[0, 0, 0], [1, 1, 1]]}, {"region": "r", "stress": 1.0, "schedule": [[0, 1]]},
                {"region": "r", "direction": [0, 0, 0]}):
        with pytest.raises(ConfigError):
            validate(MuscleActuatorConfig, bad)


# -- a free bar against the homogeneous closed form --------------------------------------------------------------------

def _config(backend, size, nu, direction, stress, dt, duration, beta="2 ms", actuator=None, geometry=("40 mm", "10 mm", "10 mm")):
    damping = {"damping": {"model": "rayleigh", "alpha": 0.0, "beta": beta}} if backend == "mujoco" else {}
    return {"scenario": "actuation", "backend": backend,
            "simulation": {"timestep": dt, "duration": duration, "output_rate": "100 Hz", "gravity": [0, 0, 0]},
            "ground": {"enabled": False},
            "body": {"name": "bar", "geometry": {"type": "box", "size": list(geometry)},
                     "material": {"density": f"{G0} kg/m^3", "elastic": {
                         "model": "stable_neo_hookean", "youngs_modulus": f"{E} Pa", "poisson_ratio": nu}, **damping},
                     "mesh": {"size": size, "method": "structured"}},
            "actuators": [actuator or {"name": "m", "type": "muscle", "direction": direction, "stress": stress,
                                       "box": [["-1 m", "-1 m", "-1 m"], ["1 m", "1 m", "1 m"]]}]}


def _closed_form(nu, s, mode):
    """Stretch (uniaxial) or volume ratio (volumetric) at which the passive Cauchy stress is -s: the active
    stress ``+s m m^T`` (or ``s I``) is balanced by the free bar's compression."""
    m = validate(Material, {"density": G0, "elastic": {"model": "stable_neo_hookean", "youngs_modulus": E,
                                                         "poisson_ratio": nu}})
    return brentq(lambda x: float(homogeneous.curve(m, mode, [x])["cauchy"].iloc[0]) + s, 0.5, 1.0)


def _length(x, x0):
    """Distance between the centroids of the two end faces, over its rest value."""
    lo, hi = np.isclose(x0[:, 0], x0[:, 0].min()), np.isclose(x0[:, 0], x0[:, 0].max())
    rest = np.linalg.norm(x0[hi].mean(axis=0) - x0[lo].mean(axis=0))
    return float(np.linalg.norm(x[hi].mean(axis=0) - x[lo].mean(axis=0)) / rest)


def _volume(x, tets):
    p = x[tets]
    return float(np.linalg.det(p[:, 1:] - p[:, :1]).sum() / 6.0)


def _settle(raw, seconds, probe):
    sim = Simulation(raw)
    try:
        x0 = sim.state().soft["bar"].vertices.copy()
        tets = sim.scene.soft_bodies[0].mesh.tets
        sim.backend.step(int(round(seconds / sim.scene.simulation.timestep)))
        x = sim.state().soft["bar"].vertices
        return probe(x, x0, tets)
    finally:
        sim.close()


@pytest.mark.parametrize("size,tolerance", [("10 mm", 0.005), ("5 mm", 0.005)])
def test_free_bar_mujoco(size, tolerance):
    """MuJoCo: a free 40 x 10 x 10 mm bar, the whole body a muscle along x with s = 10 kPa (E = 100 kPa, nu = 0.45)
    ends at the uniaxial stretch with Cauchy stress -s (0.9016), to 0.1% at both meshes (the field is
    uniform, so linear tets represent it at any size)."""
    target = _closed_form(0.45, 1e4, "uniaxial")
    got = _settle(_config("mujoco", size, 0.45, [1, 0, 0], "10 kPa", "0.1 ms", 0.5), 0.5,
                  lambda x, x0, t: _length(x, x0))
    print(f"mujoco {size}: bar length {got:.5f} vs closed form {target:.5f} ({got / target - 1:+.3%})")
    assert got < 1.0 and abs(got / target - 1) < tolerance


def test_free_bar_mujoco_isotropic():
    """Isotropic s = 10 kPa on a compressible bar (nu = 0.3): the volume follows the hydrostatic closed form."""
    target = _closed_form(0.3, 1e4, "volumetric")
    got = _settle(_config("mujoco", "5 mm", 0.3, "isotropic", "10 kPa", "0.1 ms", 0.5), 0.5,
                  lambda x, x0, t: _volume(x, t) / _volume(x0, t))
    print(f"mujoco isotropic: J {got:.5f} vs closed form {target:.5f} ({got / target - 1:+.3%})")
    assert got < 1.0 and abs(got / target - 1) < 0.03


def test_free_bar_fem():
    """The reference at 5 mm (quadratic tets) lands on the closed forms within 1%."""
    pytest.importorskip("jax")
    target = _closed_form(0.45, 1e4, "uniaxial")
    got = _settle(_config("fem", "5 mm", 0.45, [1, 0, 0], "10 kPa", "5 ms", 0.3), 0.3, lambda x, x0, t: _length(x, x0))
    print(f"fem: bar length {got:.5f} vs closed form {target:.5f} ({got / target - 1:+.3%})")
    assert abs(got / target - 1) < 0.01
    target = _closed_form(0.3, 1e4, "volumetric")
    got = _settle(_config("fem", "10 mm", 0.3, "isotropic", "10 kPa", "5 ms", 0.3), 0.3,
                  lambda x, x0, t: _volume(x, t) / _volume(x0, t))
    print(f"fem isotropic: J {got:.5f} vs closed form {target:.5f} ({got / target - 1:+.3%})")
    assert abs(got / target - 1) < 0.01


def test_fem_tangent_matches_the_loads():
    """The reference's muscle tangent (jax) is the Jacobian of its muscle loads (finite differences)."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    sim = Simulation(_config("fem", "10 mm", 0.45, [0.6, 0.8, 0], "10 kPa", "5 ms", 0.01))
    try:
        solver = sim.backend.solver
        solver.muscle_stress[0] = 1.0
        elements = solver.p.muscles[0][0]
        tets = solver.p.mesh.tets
        rng = np.random.default_rng(3)
        y = np.zeros(solver.ndof)
        y[:3 * solver.n] = rng.normal(size=3 * solver.n) * 5e-4
        jac = np.asarray(solver._muscle_jac[0](
            jnp.asarray(y[:3 * solver.n].reshape(-1, 3)[tets[elements]].reshape(-1, 30)),
            jnp.asarray(solver.p.mesh.nodes[tets[elements]]), jnp.asarray(solver._dndx[elements]),
            jnp.asarray(solver._wv[elements])))
        e = 1
        assert jac.shape == (len(elements), 30, 30)
        ue = y[:3 * solver.n].reshape(-1, 3)[tets[elements[e]]]
        x0 = solver.p.mesh.nodes[tets[elements[e]]]

        def own(u):
            return active_stress_loads((x0 + u.reshape(10, 3))[None], solver._dndx[elements[e]][None],
                                       solver._wv[elements[e]][None], 1.0, np.array([0.6, 0.8, 0]))[0].reshape(30)

        h = 1e-7
        fd = np.column_stack([(own(ue.reshape(-1) + h * np.eye(30)[k]) - own(ue.reshape(-1) - h * np.eye(30)[k])) / (2 * h)
                              for k in range(30)])
        np.testing.assert_allclose(jac[e], fd, rtol=1e-5, atol=1e-6 * np.abs(fd).max())
    finally:
        sim.close()


def test_oscillator_reaches_the_body():
    """MuJoCo: a 40 Hz oscillating stress makes the bar length oscillate at 40 Hz (FFT peak), amplitude growing
    from rest after the start."""
    f = 40.0
    raw = _config("mujoco", "5 mm", 0.45, None, 0, "0.1 ms", 0.25, actuator={
        "name": "m", "type": "muscle", "direction": [1, 0, 0], "box": [["-1 m", "-1 m", "-1 m"], ["1 m", "1 m", "1 m"]],
        "oscillator": {"frequency": f"{f} Hz", "amplitude": "5 kPa", "start": "25 ms"}})
    sim = Simulation(raw)
    try:
        x0 = sim.state().soft["bar"].vertices.copy()
        n = 10  # steps per sample: 1 kHz
        lengths = []
        for _ in range(250):
            sim.backend.step(n)
            lengths.append(_length(sim.state().soft["bar"].vertices, x0))
        lengths = np.array(lengths)
        assert sim.backend.control("m", 0.0) == 0.0
        assert abs(lengths[:20] - 1).max() < 1e-3                       # nothing before the start (25 ms)
        tail = lengths[100:] - lengths[100:].mean()                    # 150 ms = 6 periods
        spectrum = np.abs(np.fft.rfft(tail * np.hanning(len(tail))))
        peak = np.fft.rfftfreq(len(tail), 1e-3)[spectrum.argmax()]
        assert abs(peak - f) <= 2.5
        assert np.ptp(tail) > 5e-3                                      # an actual oscillation, > 0.5% of the length
    finally:
        sim.close()


def test_set_control_overrides():
    sim = Simulation(_config("mujoco", "10 mm", 0.45, [1, 0, 0], "0 Pa", "0.2 ms", 0.01))
    try:
        sim.backend.set_control("m", 8e3)
        sim.backend.step(200)
        assert sim.state().controls["m"] == 8e3 and sim.metrics()["m_stress"] == 8e3
    finally:
        sim.close()


# -- scenario wiring and refusals ------------------------------------------------------------------------------------

def test_sofa_refuses():
    from softsim.backends.sofa import SofaBackend

    scene = InflationScenario(validate(InflationConfig, _config("mujoco", "10 mm", 0.45, [1, 0, 0], "5 kPa", "0.1 ms", 0.1)),
                              Path(".")).build_scene()
    with pytest.raises(UnsupportedFeatureError, match="muscle actuators"):
        SofaBackend.check(scene)


def test_regions():
    pytest.importorskip("gmsh")
    path = Path(__file__).parents[1] / "scenarios" / "actuation" / "pneunet.yaml"
    raw = apply_overrides(load_yaml(path), ["actuators=[{type: muscle, name: m, region: layer, stress: 1 kPa}]"])
    scene = InflationScenario(validate(InflationConfig, raw), path.parent).build_scene()
    body = scene.soft_bodies[0]
    (a,) = scene.actuators
    assert 0 < len(a.elements) < len(body.mesh.tets)
    bad = apply_overrides(load_yaml(path), ["actuators=[{type: muscle, name: m, region: nonsense}]"])
    with pytest.raises(ConfigError, match="volume region"):
        InflationScenario(validate(InflationConfig, bad), path.parent).build_scene()
    far = _config("mujoco", "10 mm", 0.45, None, 0, "0.1 ms", 0.1,
                  actuator={"name": "m", "type": "muscle", "box": [["1 m", "1 m", "1 m"], ["2 m", "2 m", "2 m"]]})
    with pytest.raises(ConfigError, match="no element"):
        InflationScenario(validate(InflationConfig, far), Path(".")).build_scene()
