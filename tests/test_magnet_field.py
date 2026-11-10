"""A permanent magnet's field (softsim.fem.magnetic.cylinder_field) against what doesn't share its formula: the elliptic
integral by quadrature, the magnet as a stack of current loops (Biot-Savart with scipy's own elliptic integrals), the
closed form on its axis, a dipole far away, and Maxwell's equations outside it."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import ellipe, ellipk

from softsim.fem.magnetic import MU0, cel, field, field_parameters, ring

M, A, B = 1e6, 0.01, 0.0025  # A/m, radius, half-length [m]: about Bernat et al.'s ring magnet, without its hole
Z = np.array([0.0, 0.0, 1.0])


def h_at(x, q):
    return np.asarray(field(jnp.asarray(x, float), jnp.asarray(q)))


def loop_b(rho, z, a, current):
    """B [T] (radial, axial) of a circular loop of radius ``a`` at height 0: the textbook elliptic-integral form."""
    m = 4 * a * rho / ((a + rho) ** 2 + z * z)
    k, e = ellipk(m), ellipe(m)
    pre = MU0 * current / (2 * np.pi * np.sqrt((a + rho) ** 2 + z * z))
    bz = pre * (k + (a * a - rho * rho - z * z) / ((a - rho) ** 2 + z * z) * e)
    br = pre * z / rho * (-k + (a * a + rho * rho + z * z) / ((a - rho) ** 2 + z * z) * e) if rho > 0 else 0.0
    return br, bz


@pytest.mark.parametrize("kc, p, c, s", [(0.3, 1, 1, -1), (0.7, 0.2, 1, 0.4), (1e-6, 0.5, 1, 0.7), (0.5, 1e-8, 1, 1e-4)])
def test_the_elliptic_integral_is_its_quadrature(kc, p, c, s):
    """Including next to a magnet's rim (kc -> 0) and its side (p -> 0)."""
    def f(t):
        co, si = np.cos(t) ** 2, np.sin(t) ** 2
        return (c * co + s * si) / ((co + p * si) * np.sqrt(co + kc * kc * si))
    want = quad(f, 0, np.pi / 2, limit=500, epsabs=0, epsrel=1e-12)[0]
    assert float(cel(jnp.asarray(kc), p, c, s)) == pytest.approx(want, rel=1e-9)


def test_a_cylinder_magnet_is_a_stack_of_current_loops():
    """Axial magnetisation M is a surface current M per unit length: 400 Gauss-Legendre loops, at points around the
    magnet, 0.5 mm off its side and 0.1 mm past its rim among them."""
    q = field_parameters(cylinders=[((0, 0, 0), Z, M, A, 2 * B)])
    xs, ws = np.polynomial.legendre.leggauss(400)
    for rho, z in [(0.005, 0.004), (0.0105, 0.0), (0.0105, 0.0026), (0.0101, 0.0024), (0.02, 0.01), (0.012, -0.006)]:
        br = bz = 0.0
        for xi, wi in zip(xs, ws):
            dr, dz = loop_b(rho, z - B * xi, A, M * B * wi)
            br, bz = br + dr, bz + dz
        np.testing.assert_allclose(MU0 * h_at([rho, 0, z], q), [br, 0, bz], rtol=1e-7, atol=1e-9)


def test_on_its_axis_and_far_away():
    """On the axis B = mu0 M / 2 (z+ / sqrt(z+^2 + a^2) - z- / sqrt(z-^2 + a^2)); far away a dipole of moment M pi a^2 L
    (no quadrupole, by symmetry: off by about (a / r)^2)."""
    q = field_parameters(cylinders=[((0, 0, 0), Z, M, A, 2 * B)])
    for z in (0.003, 0.01, -0.02):
        want = 0.5 * M * ((z + B) / np.hypot(z + B, A) - (z - B) / np.hypot(z - B, A))
        np.testing.assert_allclose(h_at([0, 0, z], q), [0, 0, want], rtol=1e-12, atol=1e-9 * M)
    far = field_parameters(dipole=M * np.pi * A * A * 2 * B * Z)
    for x in ([0.4, 0, 0.3], [0, 0.5, 0], [0.1, -0.2, 0.6]):
        np.testing.assert_allclose(h_at(x, q), h_at(x, far), rtol=3e-3, atol=3e-3 * np.abs(h_at(x, far)).max())


def test_outside_the_field_has_no_divergence_and_no_curl():
    """div H = 0 and curl H = 0 away from the magnet, whatever its pose: a ring, tilted and moved off the origin."""
    q = jnp.asarray(field_parameters(cylinders=ring((0.003, -0.002, 0.001), (0.3, 1.0, 0.2), 1.23, A, 0.0021, 2 * B)))
    jac = jax.jacfwd(lambda x: field(x, q))
    rng = np.random.default_rng(1)
    for x in rng.uniform(-0.03, 0.03, (6, 3)):
        if np.linalg.norm(x - (0.003, -0.002, 0.001)) < 0.015:
            x = x * 2.0
        j = np.asarray(jac(jnp.asarray(x)))
        scale = np.abs(j).max()
        assert abs(np.trace(j)) < 1e-9 * scale
        np.testing.assert_allclose(j, j.T, atol=1e-9 * scale)  # curl-free: the Jacobian is symmetric


def test_a_turned_magnet_turns_its_field():
    """The axis is a parameter, not a convention: a magnet along y gives at (y, x) what one along z gives at (z, x)."""
    along_z = field_parameters(cylinders=[((0, 0, 0), Z, M, A, 2 * B)])
    along_y = field_parameters(cylinders=[((0, 0, 0), (0, 1, 0), M, A, 2 * B)])
    hz = h_at([0.012, 0, 0.004], along_z)
    hy = h_at([0.012, 0.004, 0], along_y)
    np.testing.assert_allclose(hy, [hz[0], hz[2], 0], rtol=1e-12, atol=1e-9 * M)


def test_a_ring_is_a_disc_without_its_hole():
    """On the axis of a ring magnet, the disc's closed form minus the hole's: the hole reverses the field near it."""
    q = field_parameters(cylinders=ring((0, 0, 0), Z, 1.23, A, 0.0021, 2 * B))
    for z in (0.003, 0.008):
        def disc(a):
            return 0.5 * 1.23 / MU0 * ((z + B) / np.hypot(z + B, a) - (z - B) / np.hypot(z - B, a))
        np.testing.assert_allclose(h_at([0, 0, z], q)[2], disc(A) - disc(0.0021), rtol=1e-12)


def test_a_strip_rolled_onto_a_ring_magnet_takes_the_energy_of_its_cross_section():
    """The rolling torque of tools/mre_rolling_torque.py, by two routes. The magnet's field is symmetric about its axis,
    so each radian of strip on the rim holds the same energy: T = -(R + t / 2) (integral of w over the cross-section on the
    rim - that over the tail's far end, which leaves the field as the strip rolls on). There the field has no part along
    the strip, so w = -mu0 / 2 (chi_n H_r^2 + chi_b H_y^2). Against the solver's energy of the meshed strip, rolled by the
    tool's map (its F turns the axes), at 0.5 mm."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
    from mre_rolling_torque import LENGTH, R_IN, R_OUT, THICK, WIDTH, rolled

    from softsim.fem.magnetic import strip_magnet
    from softsim.fem.materials import NeoHookeanIso
    from softsim.fem.solid3d import Problem3D, QuadraticTetMesh, Solid3DSolver
    from softsim.meshing import structured_box

    t, chi, theta, step = 0.003, 0.5, np.radians(120), np.radians(10)
    magnet = strip_magnet(chi, LENGTH, t, WIDTH)
    q = field_parameters(cylinders=ring((0, 0, 0), (0, 1, 0), 1.23, R_OUT, R_IN, THICK))
    lin = structured_box((LENGTH, WIDTH, t), 0.0005)
    mesh = QuadraticTetMesh.from_linear(lin.points, lin.tets)
    s = Solid3DSolver(Problem3D(mesh, NeoHookeanIso(1e5), 1e7, 2000.0, gravity=(0, 0, 0), magnetic=magnet), dynamic=False)
    s.magnetic_field = q
    energy = [s.magnetic_energy(rolled(mesh.nodes, a, t).ravel()) for a in (theta - step, theta + step)]
    fe = -(energy[1] - energy[0]) / (2 * step)

    mid = R_OUT + t / 2
    # the field kinks at the magnet's faces (y = +-THICK / 2) and is log-singular at its rim: split there, and grade
    # towards the rim (r = R + t u^2); converged to 1e-6 (twice the points move it by 1e-7)
    zs, wz = np.polynomial.legendre.leggauss(48)
    ys, wy = np.polynomial.legendre.leggauss(64)
    u = 0.5 * (zs + 1)
    radii, wr = R_OUT + t * u * u, wz * t * u
    edges = (-WIDTH / 2, -THICK / 2, THICK / 2, WIDTH / 2)
    across = np.concatenate([0.5 * (b - a) * ys + 0.5 * (a + b) for a, b in zip(edges[:-1], edges[1:])])
    wa = np.concatenate([0.5 * (b - a) * wy for a, b in zip(edges[:-1], edges[1:])])
    r, y = np.meshgrid(radii, across, indexing="ij")
    weight = np.outer(wr, wa)
    top = LENGTH - mid * theta  # the tail's far end above the take-off point
    at = jax.jit(jax.vmap(field, in_axes=(0, None)))
    on_rim = np.asarray(at(jnp.asarray(np.stack([r, y, 0 * r], -1).reshape(-1, 3)), jnp.asarray(q))).reshape(*r.shape, 3)
    at_top = np.asarray(at(jnp.asarray(np.stack([r, y, 0 * r + top], -1).reshape(-1, 3)), jnp.asarray(q))).reshape(*r.shape, 3)
    # the rim's field is the same at every angle: take the plane y-x, where the strip runs along -z and its normal is x
    arc = np.sum(weight * -0.5 * MU0 * (magnet.chi_normal * on_rim[..., 0] ** 2 + magnet.chi_across * on_rim[..., 1] ** 2))
    assert np.abs(on_rim[..., 2]).max() < 1e-9 * np.abs(on_rim).max()  # nothing along the strip on the rim
    end = np.sum(weight * -0.5 * MU0 * (magnet.chi_along * at_top[..., 2] ** 2 + magnet.chi_normal * at_top[..., 0] ** 2
                                        + magnet.chi_across * at_top[..., 1] ** 2))
    assert fe == pytest.approx(-mid * (arc - end), rel=2e-3)  # 2 mm: +0.32%, 1 mm: -0.31%, 0.5 mm: -0.09%


def test_a_blocks_demagnetising_factors():
    """Aharoni's exact factors: a cube's are 1/3; any block's three sum to 1; a thin plate's normal one tends to 1 and a
    long bar's transverse ones to 1/2 (an infinite cylinder's), its axial one to 0."""
    from softsim.fem.magnetic import prism_demagnetizing_factor as n
    assert n(1, 1, 1) == pytest.approx(1 / 3, rel=1e-12)
    for a, b, c in [(60, 15, 3), (1, 2, 3), (5, 0.3, 40)]:
        assert n(b, c, a) + n(c, a, b) + n(a, b, c) == pytest.approx(1, rel=1e-12)
    assert n(1e4, 1e4, 1) == pytest.approx(1, abs=1e-3)  # slowly: 1 - N ~ (t / a) ln(a / t)
    assert n(1000, 1, 1) == pytest.approx(0.5, abs=1e-3) and n(1, 1, 1000) == pytest.approx(0, abs=1e-3)
    assert n(1, 1, 2) < 1 / 3 < n(2, 2, 1)  # the longer side magnetises more easily


def test_the_composites_susceptibility():
    """Maxwell Garnett for very permeable spheres: dilute, chi = 3 phi; at 1 : 1 iron : silicone by weight, 12-14% by
    volume, 0.41-0.47."""
    from softsim.materials.models import composite_susceptibility
    assert composite_susceptibility(1e-6) == pytest.approx(3e-6, rel=1e-5)
    phi = (1 / 7.87) / (1 / 7.87 + 1 / 1.07)
    assert composite_susceptibility(phi) == pytest.approx(0.41, abs=0.005)


def test_across_the_magnets_skin_maxwells_jump_conditions_hold():
    """Through the rim (the side) H along it is continuous and B across it too; through an end face B across it is
    continuous and H across it jumps by M. The closed form gives B: H inside is B / mu0 - M, not B / mu0, whose false
    jump at the rim would be a well pulling material into the magnet."""
    q = field_parameters(cylinders=[((0, 0, 0), Z, M, A, 2 * B)])
    e = 1e-9
    for z in (0.0, 0.0012, -0.002):  # through the side, radially
        out, inn = h_at([A + e, 0, z], q), h_at([A - e, 0, z], q)
        assert out[2] == pytest.approx(inn[2], rel=1e-5, abs=1e-5 * M)  # H along the rim
        assert out[0] == pytest.approx(inn[0], rel=1e-5, abs=1e-5 * M)  # B across = mu0 H across (no M radially)
    for rho in (0.0, 0.004, 0.008):  # through the top face, axially
        out, inn = h_at([rho, 0, B + e], q), h_at([rho, 0, B - e], q)
        assert out[2] == pytest.approx(inn[2] + M, rel=1e-5)  # B_z continuous: H_z jumps by M
        assert out[0] == pytest.approx(inn[0], rel=1e-5, abs=1e-5 * M)  # H along the face


def test_a_softened_edge_is_finite_and_changes_little_away_from_it():
    """``edge`` (a chamfer, 0.2 mm) makes the field finite on the rim of an end face, where the sharp magnet's is infinite,
    and leaves it within 2% from 1 mm away, 0.2% from 5 mm, and 0.5% mid-way along the side."""
    sharp = field_parameters(cylinders=[((0, 0, 0), Z, M, A, 2 * B)])
    soft = field_parameters(cylinders=[((0, 0, 0), Z, M, A, 2 * B, 2e-4)])
    assert np.all(np.isfinite(h_at([A, 0, B], soft))) and np.linalg.norm(h_at([A, 0, B], soft)) < 0.6 * M
    for d, within in ((1e-3, 0.02), (5e-3, 2e-3)):
        for ang in np.linspace(0, np.pi / 2, 7):  # round the rim, outside the magnet
            p = [A + d * np.cos(ang), 0, B + d * np.sin(ang)]
            assert np.linalg.norm(h_at(p, soft) - h_at(p, sharp)) < within * np.linalg.norm(h_at(p, sharp))
    p = [A + 1e-4, 0, 0]
    assert np.linalg.norm(h_at(p, soft) - h_at(p, sharp)) < 5e-3 * np.linalg.norm(h_at(p, sharp))
