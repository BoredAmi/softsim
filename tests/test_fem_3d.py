"""3D quadratic-tetrahedron reference solver (softsim.fem.solid3d) and the fem backend."""

import dataclasses
import math

import numpy as np
import pytest

pytest.importorskip("jax")

from softsim.fem.materials import (  # noqa: E402
    ArrudaBoyceIso,
    GentIso,
    MooneyRivlinIso,
    NeoHookeanIso,
    YeohIso,
    from_material,
)
from softsim.fem.solid3d import (  # noqa: E402
    Problem3D,
    QuadraticTetMesh,
    RigidBody3D,
    Solid3DSolver,
    _sdf_function,
    expm_so3,
    tet10_shape,
    tet_quadrature,
)
from softsim.geometry.rigid import Part  # noqa: E402
from softsim.materials import load_material  # noqa: E402
from softsim.materials.homogeneous import curve, history  # noqa: E402
from softsim.meshing import structured_box  # noqa: E402


def test_quadrature_and_shape_functions():
    xi, w = tet_quadrature(3)
    assert w.sum() == pytest.approx(1 / 6) and np.all(w > 0)
    # exact for degree 5: integral of x^a y^b z^c over the unit tetrahedron = a! b! c! / (a + b + c + 3)!
    for a, b, c in [(2, 1, 1), (5, 0, 0), (1, 2, 2), (0, 0, 3)]:
        exact = math.factorial(a) * math.factorial(b) * math.factorial(c) / math.factorial(a + b + c + 3)
        assert np.sum(w * xi[:, 0] ** a * xi[:, 1] ** b * xi[:, 2] ** c) == pytest.approx(exact, rel=1e-12)
    nodes = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [.5, 0, 0], [.5, .5, 0], [0, .5, 0], [0, 0, .5],
                      [.5, 0, .5], [0, .5, .5]])
    n, dn = tet10_shape(nodes)
    np.testing.assert_allclose(n, np.eye(10), atol=1e-14)        # Kronecker property
    np.testing.assert_allclose(dn.sum(axis=1), 0.0, atol=1e-13)   # partition of unity


def test_quadratic_mesh_geometry():
    lin = structured_box((0.02, 0.02, 0.01), 0.005)
    m = QuadraticTetMesh.from_linear(lin.points, lin.tets)
    s = Solid3DSolver(Problem3D(m, NeoHookeanIso(1e4), 1e6, 1000.0, rigid=(RigidBody3D.ground(),)), dynamic=False)
    assert s.volume == pytest.approx(0.02 * 0.02 * 0.01, rel=1e-12)
    assert s.mass_matrix.sum() == pytest.approx(3 * 1000.0 * s.volume, rel=1e-12)
    idx, area = m.surface_nodes_and_areas()
    assert area.sum() == pytest.approx(2 * 0.02**2 + 4 * 0.02 * 0.01, rel=1e-12) and np.all(area > 0)


def test_signed_distances_and_rotations():
    body = RigidBody3D("w", parts=(Part("cylinder", (0.01, 0.005, 0), (0, 0, 0.005)),
                                   Part("box", (0.002, 0.002, 0.003), (0, 0, 0.013))),
                       mass=1.0, com=(0, 0, 0.005), edge_radius=1e-4)
    sdf = _sdf_function(body)
    # COM frame: cylinder centre at 0, box centre at z = 8 mm
    assert float(sdf(np.array([0.0, 0.0, -0.007]))) == pytest.approx(0.002)        # below the bottom face
    assert float(sdf(np.array([0.013, 0.0, 0.0]))) == pytest.approx(0.003)         # beside the side
    assert float(sdf(np.array([0.0, 0.0, 0.0]))) == pytest.approx(-0.005)          # centre: nearest face 5 mm
    assert float(sdf(np.array([0.0, 0.0, 0.015]))) == pytest.approx(0.004)         # above the box (top at 11 mm)
    r = expm_so3([0.0, 0.0, math.pi / 2])
    np.testing.assert_allclose(r @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(expm_so3([1e-9, 0, 0]) @ expm_so3([1e-9, 0, 0]).T, np.eye(3), atol=1e-14)


def _box_problem(material, h=0.01, density=1000.0, rigid=(), fixed_extra=None):
    lin = structured_box((0.02, 0.02, 0.01), h)
    m = QuadraticTetMesh.from_linear(lin.points, lin.tets)
    x, y = m.nodes[:, 0], m.nodes[:, 1]
    fixed = np.r_[3 * np.nonzero(np.isclose(x, 0))[0], 3 * np.nonzero(np.isclose(y, 0))[0] + 1]  # symmetry
    law, kappa, prony = from_material(material)
    return Problem3D(m, law, kappa, density, prony=prony, gravity=(0, 0, 0), fixed_dofs=fixed, rigid=rigid)


def _plate(motion):
    return RigidBody3D("plate", parts=(Part("box", (0.02, 0.02, 0.001), (0, 0, 0.001)),), mount="prescribed",
                       position=(0, 0, 0.01), motion=motion, edge_radius=1e-4)


@pytest.mark.parametrize("elastic", [
    {"model": "yeoh", "c10": 10e3, "c20": 500.0, "poisson_ratio": 0.499},
    pytest.param({"model": "ogden", "mu": [-8e3, 300.0], "alpha": [-3.0, 4.0], "poisson_ratio": 0.499},
                 marks=pytest.mark.slow),  # matrix-function Ogden: minutes of JAX compilation
])
@pytest.mark.parametrize("quadrature", [8, 4])
def test_frictionless_compression_matches_homogeneous(elastic, quadrature):
    card = load_material({"density": 1000, "elastic": elastic})
    delta = 0.003
    prob = _box_problem(card, rigid=(RigidBody3D.ground(), _plate(lambda t: np.array([0, 0, -delta * min(t, 1)]))))
    s = Solid3DSolver(prob, dynamic=False, quadrature=quadrature)
    for _ in range(5):
        assert s.advance(0.2)
    exact = curve(card, "uniaxial", [1 - delta / 0.01])
    assert s.contact_force(1)[2] == pytest.approx(-exact.stress.iloc[0] * 0.02**2, rel=1e-3)
    assert s.current_nodes()[:, 0].max() / 0.01 == pytest.approx(exact.lateral.iloc[0], rel=1e-4)


def test_prony_relaxation_matches_homogeneous():
    """Quasi-static ramp-and-hold compression: the plate force follows the homogeneous viscoelastic
    history (same update), so the 3D Prony implementation is exact."""
    card = load_material({"density": 1000, "elastic": {"model": "neo_hookean", "shear_modulus": 10e3,
                                                       "bulk_modulus": 10e6},
                          "damping": {"model": "prony", "g": [0.6], "tau": [0.02]}})
    ramp, delta = 0.002, 0.002
    motion = lambda t: np.array([0, 0, -delta * min(t / ramp, 1.0)])  # noqa: E731
    prob = _box_problem(card, rigid=(RigidBody3D.ground(), _plate(motion)))
    s = Solid3DSolver(prob, dynamic=False, viscous=True)
    t = np.r_[np.linspace(0, ramp, 11)[1:], ramp + np.geomspace(1e-3, 0.1, 12)]
    forces = []
    for t_next in t:
        assert s.advance(t_next - s.t)
        forces.append(s.contact_force(1)[2])
    stretch = 1.0 + np.array([motion(x)[2] for x in t]) / 0.01
    h = history(card, "uniaxial", np.r_[0.0, t], np.r_[1.0, stretch])
    np.testing.assert_allclose(forces, -h.stress.to_numpy()[1:] * 0.02**2, rtol=2e-3)


@pytest.mark.parametrize("zone_card", [
    {"model": "prony", "g": [0.3], "tau": [0.005]},   # a different series
    None,                                              # an undamped zone (a stiff cap): no relaxation at all
])
def test_zone_prony_follows_its_own_series(zone_card):
    """Two columns side by side under the same uniaxial compression keep the homogeneous state (the elastic
    cards are equal and the material is near-incompressible, so the lateral stretch does not depend on the
    relaxing shear modulus): the plate force is the area-weighted average of what each column's own series
    gives, whose exact history is ``history`` (measured: within 0.1%). A global series would put the body's
    relaxation on both columns."""
    elastic = {"model": "neo_hookean", "shear_modulus": 10e3, "bulk_modulus": 10e6}
    body_card = load_material({"density": 1000, "elastic": elastic,
                               "damping": {"model": "prony", "g": [0.6], "tau": [0.02]}})
    zone = load_material({"density": 1000, "elastic": elastic, **({"damping": zone_card} if zone_card else {})})
    ramp, delta = 0.002, 0.002
    motion = lambda t: np.array([0, 0, -delta * min(t / ramp, 1.0)])  # noqa: E731
    prob = _box_problem(body_card, h=0.005, rigid=(RigidBody3D.ground(), _plate(motion)))
    right = np.nonzero(prob.mesh.nodes[prob.mesh.tets[:, :4], 0].mean(axis=1) > 0.005)[0]  # the quadrant's outer half
    assert 0 < len(right) < len(prob.mesh.tets)
    law, kappa, zone_prony = from_material(zone)
    prob = dataclasses.replace(prob, zones=((right, law, kappa),), zone_pronys=(zone_prony,))
    s = Solid3DSolver(prob, dynamic=False, viscous=True)
    t = np.r_[np.linspace(0, ramp, 11)[1:], ramp + np.geomspace(1e-3, 0.1, 12)]
    forces = []
    for t_next in t:
        assert s.advance(t_next - s.t)
        forces.append(s.contact_force(1)[2])
    stretch = 1.0 + np.array([motion(x)[2] for x in t]) / 0.01
    share = len(right) / len(prob.mesh.tets)  # the zone's share of the cross-section (equal-volume tets)
    expected = [w * -history(card, "uniaxial", np.r_[0.0, t], np.r_[1.0, stretch]).stress.to_numpy()[1:] * 0.02**2
                for w, card in ((1 - share, body_card), (share, zone))]
    np.testing.assert_allclose(forces, expected[0] + expected[1], rtol=3e-3)
    # the series differ enough for this to mean something: a global series is off by far more
    glob = -history(body_card, "uniaxial", np.r_[0.0, t], np.r_[1.0, stretch]).stress.to_numpy()[1:] * 0.02**2
    assert abs(glob[-1] - forces[-1]) > 0.08 * glob[-1]


def test_free_flight_and_spin_are_exact():
    body = RigidBody3D("w", parts=(Part("cylinder", (0.01, 0.005, 0), (0, 0, 0.005)),), mass=0.05,
                       inertia=((1e-6, 0, 0), (0, 1e-6, 0), (0, 0, 2e-6)), com=(0, 0, 0.005), position=(0, 0, 0.1),
                       velocity=(0.1, 0, -1.0), angular_velocity=(0, 0, 10.0))
    lin = structured_box((0.02, 0.02, 0.01), 0.01)
    m = QuadraticTetMesh.from_linear(lin.points, lin.tets)
    s = Solid3DSolver(Problem3D(m, NeoHookeanIso(1e4), 1e6, 1000.0, rigid=(RigidBody3D.ground(), body)))
    for _ in range(10):
        assert s.advance(1e-3)
    c, r = s.rigid_pose(1)
    t = 0.01
    np.testing.assert_allclose(c, [0.1 * t, 0, 0.105 - 1.0 * t - 0.5 * 9.81 * t * t], atol=1e-12)
    np.testing.assert_allclose(r, expm_so3([0, 0, 10.0 * t]), atol=1e-12)


def test_fem_backend_refuses_numerical_damping_and_stvk():
    from softsim.backends import get_backend
    from softsim.runner import Simulation

    sim = Simulation("scenarios/drop_test/cases/w050.yaml", ["simulation.duration=0.0001"])
    fem = get_backend("fem")
    missing = fem.unsupported(sim.scene)
    sim.close()
    assert any("Rayleigh beta" in m for m in missing)
    stvk = sim.scene.with_material("specimen", load_material("ecoflex_00_10", overrides=[
        "elastic.model=stvk", "damping=null"]))
    assert "Saint Venant-Kirchhoff elasticity" in fem.unsupported(stvk)


def test_fem_backend_runs_a_drop():
    """A short coarse drop through the normal runner: contact force builds up, the weight decelerates
    and the specimen stays above the ground."""
    from softsim.runner import run_config

    r = run_config("scenarios/drop_test/cases/w050.yaml", [
        "backend=fem", "specimen.material.damping={model: prony, g: [0.5], tau: [2 ms]}",
        "specimen.material.elastic.poisson_ratio=0.49", "specimen.mesh.size=0.008",
        "simulation.timestep=1e-4", "simulation.duration=0.0015", "simulation.output_rate=10000",
        "drop.gap=0.0002", "output.save_model=false"])
    ts = r.timeseries
    assert ts.contact_force_z.abs().max() > 1.0
    v = np.gradient(ts.weight_com_z.to_numpy(), ts.t.to_numpy())
    assert v[-1] > v[1]  # decelerated by the specimen
    assert ts.specimen_min_z.min() > -1e-4


@pytest.mark.parametrize("law", [
    NeoHookeanIso(1e4),
    MooneyRivlinIso(4e3, 1.5e3),
    YeohIso(5e3, 300.0, 20.0),
    GentIso(1e4, 25.0),
    ArrudaBoyceIso(1e4, 3.0),
], ids=lambda law: type(law).__name__)
def test_closed_form_tangent_matches_autodiff(law):
    """The invariant-law fast path (closed-form dP/dF) equals jax.hessian of the 3x3 energy."""
    import jax
    import jax.numpy as jnp

    from softsim.fem.solid3d import cof3, d2det3, invariant_tangent

    rng = np.random.default_rng(1)
    for _ in range(4):
        f = jnp.asarray(np.eye(3) + rng.normal(0, 0.15, (3, 3)))
        bq = rng.normal(0, 500.0, (3, 3))
        bq = jnp.asarray(bq + bq.T)

        def psi(ff):
            cc = ff.T @ ff
            return 0.7 * law.energy(cc) + 0.5 * jnp.sum(bq * cc)

        pk, a4 = invariant_tangent(law, f, bq, 0.7)
        np.testing.assert_allclose(pk, jax.grad(psi)(f), rtol=1e-9, atol=1e-9 * law.shear_modulus)
        np.testing.assert_allclose(a4, jax.hessian(psi)(f), rtol=1e-8, atol=1e-8 * law.shear_modulus)
        np.testing.assert_allclose(cof3(f), jax.grad(jnp.linalg.det)(f), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(d2det3(f), jax.hessian(jnp.linalg.det)(f), rtol=1e-10, atol=1e-12)


_GPU_PROBE = """
import time
from softsim.fem import jax_setup
jax, jnp = jax_setup()
t = time.perf_counter()
x = jnp.ones(125_000_000)  # 1 GB of float64, about a 5 mm drop case's working set
print(jax.devices()[0].platform, float(x.sum()), time.perf_counter() - t, flush=True)
time.sleep(8)  # hold the memory while the other processes allocate theirs
"""


def test_three_processes_share_one_gpu():
    """A calibration runs a parent and a worker per case on one GPU, and each needs about 1 GB.
    With JAX's default preallocation the first process reserves 75% of the card (9 GB of 12):
    measured, the second then died with CUDA_ERROR_OUT_OF_MEMORY and the third waited 11 s for
    the first to exit. With jax_setup's on-demand allocation all three get theirs at once."""
    import os
    import subprocess
    import sys

    import jax

    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("needs a CUDA device")
    env = {k: v for k, v in os.environ.items() if k != "XLA_PYTHON_CLIENT_PREALLOCATE"}  # jax_setup's default
    procs = [subprocess.Popen([sys.executable, "-c", _GPU_PROBE], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True) for _ in range(3)]
    out = [p.communicate(timeout=120)[0].split() for p in procs]
    assert [o[:1] for o in out] == [["gpu"]] * 3, out  # none died or fell back to the CPU
    assert max(float(o[2]) for o in out) < 4.0, out  # and none waited for another to exit


def test_chunked_kernels_equal_the_single_launch(monkeypatch):
    import jax.numpy as jnp

    """Evaluating the element kernels in chunks (so a fine mesh does not ask the GPU for the whole batch at once) gives
    the same energy, gradient/Hessian and stress as one launch, with two material groups (a zone on a third of the elements); a one-element chunk
    is the extreme. Mutant: the last, shorter chunk dropped."""
    card = load_material({"density": 1000, "elastic": {"model": "neo_hookean", "shear_modulus": 10e3,
                                                       "bulk_modulus": 1e6}})

    def evaluate(chunk, h=0.005):
        monkeypatch.setenv("SOFTSIM_FEM_CHUNK", str(chunk))
        prob = _box_problem(card, h=h)
        law, kappa, _ = from_material(card)
        stiff = load_material({"density": 1000, "elastic": {"model": "neo_hookean", "shear_modulus": 40e3,
                                                            "bulk_modulus": 1e6}})
        slaw, skappa, _ = from_material(stiff)
        zone = np.arange(0, len(prob.mesh.tets), 3)
        prob = dataclasses.replace(prob, zones=((zone, slaw, skappa),), zone_pronys=(None,))
        s = Solid3DSolver(prob, dynamic=False)
        rng = np.random.default_rng(1)
        s.y[:3 * s.n] = 1e-4 * rng.standard_normal(3 * s.n)
        u = s.y[:3 * s.n].reshape(s.n, 3)[prob.mesh.tets].reshape(-1, 30)
        ge, he = s._elem_grad_hess(jnp.asarray(u), s._dndx, s._wv, (1.0, 1.0), jnp.zeros((*s._wv.shape, 3, 3)))
        e = s._elem_energy(jnp.asarray(u), (1.0, 1.0), jnp.zeros((*s._wv.shape, 3, 3)))
        sig = s._iso_stress(jnp.asarray(u), s._dndx)
        return np.asarray(ge), np.asarray(he), float(e), np.asarray(sig), len(prob.mesh.tets)

    whole = evaluate(10**9)
    n = whole[4]
    for chunk in (7, n - 1, 1):
        part = evaluate(chunk)
        np.testing.assert_allclose(part[0], whole[0], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(part[1], whole[1], rtol=1e-12, atol=1e-12)
        assert part[2] == pytest.approx(whole[2], rel=1e-12)
        np.testing.assert_allclose(part[3], whole[3], rtol=1e-12, atol=1e-14)
