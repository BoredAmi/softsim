"""Contact adhesion and spring-mounted bodies: traction law, geometry, analytic equilibria on MuJoCo."""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from softsim.backends import get_backend
from softsim.backends.sofa import SofaBackend
from softsim.config.schema import validate
from softsim.errors import SoftSimError
from softsim.contact.adhesion import AdhesionField, _box_sdf, _cylinder_sdf, area_vectors, traction
from softsim.geometry.rigid import Part
from softsim.materials import Adhesion, Material
from softsim.meshing import structured_box
from softsim.scene import GroundSettings, Pose, RigidBodySpec, Scene, SimulationSettings, SoftBodySpec

STIFF = {"model": "stable_neo_hookean", "youngs_modulus": 1e6, "poisson_ratio": 0.3}


def _material(**extra) -> Material:
    return validate(Material, {"density": 1000, "elastic": STIFF, "damping": {"model": "rayleigh", "beta": 2e-3},
                               **extra})


def _box_scene(gravity, ground_adhesion=None, material=None, rigid=(), pinned=False, duration=0.3):
    mesh = structured_box((0.02, 0.02, 0.01), 0.005)
    pins = tuple(np.nonzero(mesh.points[:, 2] < 1e-9)[0].tolist()) if pinned else ()
    body = SoftBodySpec("block", mesh, material or _material(), pinned=pins)
    sim = SimulationSettings(timestep=1e-4, duration=duration, output_rate=1000, gravity=gravity)
    return Scene((body,), tuple(rigid), simulation=sim, ground=GroundSettings(adhesion=ground_adhesion))


def _run(scene, duration):
    backend = get_backend("mujoco")()
    backend.build(scene)
    backend.step(int(round(duration / scene.simulation.timestep)))
    assert backend.is_finite()
    return backend


def _backend(name):
    if name == "sofa":
        try:
            from softsim.backends.sofa.backend import import_sofa

            import_sofa()
        except SoftSimError as exc:
            pytest.skip(str(exc))
    return get_backend(name)()


def _set_gravity_z(backend, gz: float) -> None:
    if backend.name == "mujoco":
        backend.model.opt.gravity[2] = gz
    else:
        backend.root.gravity.value = [0.0, 0.0, gz]


def _settle_under_ramped_gravity(scene, measure, backend_name="mujoco", ramp=0.5, hold=0.1, average=0.05):
    """Ramp gravity from -9.81 to the scene's value (z) slowly enough to stay quasi-static, then
    hold and average ``measure(state)``: adhesive bonds are undamped springs, so a sudden load
    would leave the body oscillating about its equilibrium."""
    backend = _backend(backend_name)
    backend.build(scene)
    target, dt = scene.simulation.gravity[2], scene.simulation.timestep
    chunk = 10
    n_ramp, n_hold, n_avg = (int(round(x / (chunk * dt))) for x in (ramp, hold, average))
    values = []
    for i in range(n_ramp + n_hold):
        _set_gravity_z(backend, -9.81 + (target + 9.81) * min(1.0, (i + 1) / n_ramp))
        backend.step(chunk)
        if i >= n_ramp + n_hold - n_avg:
            values.append(measure(backend.state()))
    assert backend.is_finite()
    return np.mean(values, axis=0), backend


def _gap_at(fraction: float) -> float:
    """Opening (in units of range) on the rising branch where traction = fraction * strength."""
    return brentq(lambda x: x * math.exp(1 - x) - fraction, 1e-9, 1.0)


def test_traction_law():
    a = Adhesion(strength="2 kPa", range="0.5 mm")
    g = np.linspace(0, 40 * a.range, 400001)
    assert traction(np.array([a.range]), a)[0] == pytest.approx(a.strength)
    assert traction(np.array([-1e-3, 0.0]), a).tolist() == [0.0, 0.0]  # no attraction in compression
    assert np.trapz(traction(g, a), g) == pytest.approx(a.work_of_separation, rel=1e-6)
    assert a.work_of_separation == pytest.approx(math.e * 2e3 * 0.5e-3)


def test_signed_distances():
    d, n = _cylinder_sdf(np.array([[0, 0, 2.0], [3, 0, 0], [0, 0, 0.5], [2.5, 0, 1.5]]), 1.0, 1.0)
    assert d == pytest.approx([1.0, 2.0, -0.5, math.hypot(1.5, 0.5)])
    assert n[3] == pytest.approx([1.5, 0, 0.5] / np.hypot(1.5, 0.5))
    d, n = _box_sdf(np.array([[2.0, 0, 0], [0, 0, 0.9]]), np.ones(3))
    assert d == pytest.approx([1.0, -0.1]) and n[1] == pytest.approx([0, 0, 1])


def test_only_the_facing_area_sticks():
    scene = _box_scene((0, 0, -9.81), ground_adhesion=Adhesion(strength=100.0, range=1e-3))
    field = AdhesionField(scene, "block")
    pts = scene.soft("block").mesh.points[field.vertices].copy()
    pts[:, 2] += scene.contact.soft_radius + 0.5e-3  # uniform 0.5 mm opening, all bottom vertices bonded
    field.counterparts[0].bonded[pts[:, 2] < 0.7e-3] = True
    result = field.compute(pts, {})
    total = result.wrenches["ground"][0]
    expected = 0.02 * 0.02 * traction(np.array([0.5e-3]), scene.ground.adhesion)[0]  # bottom face only, no side walls
    assert total == pytest.approx([0, 0, expected], rel=1e-9)
    assert result.vertex_forces.sum(axis=0) == pytest.approx(-total)


@pytest.mark.parametrize("backend_name", ["mujoco", "sofa"])
@pytest.mark.parametrize("load_fraction", [0.5, 1.4])
def test_block_hangs_from_the_ground_under_reversed_gravity(load_fraction, backend_name):
    """Gravity pulls a 4 g block off the ground; adhesion holds it at the analytic gap or lets go."""
    adhesion = Adhesion(strength=500.0, range=1e-3)
    area, mass = 0.02 * 0.02, 1000 * 0.02 * 0.02 * 0.01
    g_up = load_fraction * adhesion.strength * area / mass
    scene = _box_scene((0, 0, g_up), ground_adhesion=adhesion)
    is_bottom = scene.soft("block").mesh.points[:, 2] < 1e-9
    bottom, _ = _settle_under_ramped_gravity(scene, lambda s: s.soft["block"].vertices[is_bottom, 2], backend_name)
    if load_fraction < 1:
        assert bottom.mean() == pytest.approx(_gap_at(load_fraction) * adhesion.range, rel=0.02)
        assert np.ptp(bottom) < 0.05 * adhesion.range  # stiff block: uniform opening
    else:
        assert bottom.min() > 50 * adhesion.range  # torn off and gone


@pytest.mark.parametrize("backend_name", ["mujoco", "sofa"])
def test_weight_hangs_from_a_soft_block(backend_name):
    """A rigid disc stuck on the block's top under reversed gravity: analytic gap and reported force."""
    adhesion = Adhesion(strength=1000.0, range=1e-3)
    radius, mass = 0.006, 0.004
    area = math.pi * radius**2  # disc fully inside the 20 x 20 mm top face
    g_up = 0.5 * adhesion.strength * area / mass
    skin = 1e-4
    # starts 0.5 mm above the block, lands under normal gravity and bonds; then gravity reverses
    disc = RigidBodySpec("disc", (Part("cylinder", (radius, 0.002, 0), (0, 0, 0.002)),), mass,
                         pose=Pose((0.0, 0.0, 0.01 + 2 * skin + 5e-4)))
    scene = _box_scene((0, 0, g_up), material=_material(adhesion={"strength": 1000.0, "range": 1e-3}),
                       rigid=(disc,), pinned=True)
    p0 = scene.soft("block").mesh.points
    under = (p0[:, 2] > 0.01 - 1e-9) & (np.hypot(p0[:, 0], p0[:, 1]) < radius - 1e-3)

    def measure(state):
        gap = state.rigid["disc"].position[2] - state.soft["block"].vertices[under, 2].mean() - 2 * skin
        return gap, state.contact_between("block", "disc").force[2]

    (gap, force_z), backend = _settle_under_ramped_gravity(scene, measure, backend_name)
    # the 5 mm mesh lumps the disc onto 5 vertices whose tributary area exceeds the disc's by ~10%;
    # the equilibrium opening follows from the area that is actually bonded
    field = AdhesionField(scene, "block")
    v = backend.state().soft["block"].vertices[field.vertices]
    bonded = np.hypot(v[:, 0], v[:, 1]) < radius + 1e-4
    bonded &= scene.soft("block").mesh.points[field.vertices, 2] > 0.01 - 1e-9
    area_bonded = area_vectors(v, field.triangles)[bonded, 2].sum()
    assert area_bonded == pytest.approx(area, rel=0.15)
    assert gap == pytest.approx(_gap_at(mass * g_up / (adhesion.strength * area_bonded)) * adhesion.range, rel=0.02)
    assert force_z == pytest.approx(-mass * g_up, rel=0.02)  # the block pulls the disc down


def test_spring_mounted_plate_sags_by_mg_over_k():
    plate = RigidBodySpec("plate", (Part("box", (0.02, 0.02, 0.001), (0, 0, 0.001)),), 0.01,
                          pose=Pose((0.0, 0.0, 0.05)), mount="slider_z", mount_stiffness=20.0, mount_damping=0.5)
    sim = SimulationSettings(timestep=1e-4, duration=1.0, gravity=(0, 0, -9.81))
    backend = get_backend("mujoco")()
    backend.build(Scene((), (plate,), simulation=sim))
    backend.step(15000)
    state = backend.state().rigid["plate"]
    assert state.position[2] == pytest.approx(0.05 - 0.01 * 9.81 / 20.0, rel=1e-4)
    assert state.position[:2] == pytest.approx([0, 0]) and abs(state.linear_velocity[2]) < 1e-4


def test_sofa_refuses_spring_mounts():
    plate = RigidBodySpec("plate", (Part("box", (0.02, 0.02, 0.001), (0, 0, 0.001)),), 0.01, mount="slider_z")
    scene = _box_scene((0, 0, -9.81), ground_adhesion=Adhesion(strength=100.0, range=1e-3), rigid=(plate,))
    assert SofaBackend.unsupported(scene) == ["rigid bodies on a vertical spring-damper slider"]
