"""The soft-magnetic body against what can be worked out by hand: its energy under rotation and shear, the pull of a
dipole on a small piece, the field that buckles a cantilever, and Newton's Hessian against its gradient."""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")
import scipy.sparse as sp

from softsim.fem.magnetic import MU0, SoftMagnet, field_parameters, ring
from softsim.fem.materials import NeoHookeanIso
from softsim.fem.solid3d import Problem3D, QuadraticTetMesh, Solid3DSolver
from softsim.meshing import structured_box

E_MOD, NU = 2e5, 0.3
MU, KAPPA = E_MOD / (2 * (1 + NU)), E_MOD / (3 * (1 - 2 * NU))


def bar(length, width, height, size, magnet, pinned=False):
    lin = structured_box((length, width, height), size)
    mesh = QuadraticTetMesh.from_linear(lin.points, lin.tets)
    clamp = np.flatnonzero(np.isclose(mesh.nodes[:, 0], mesh.nodes[:, 0].min())) if pinned else np.zeros(0, int)
    return Problem3D(mesh, NeoHookeanIso(MU), KAPPA, 1000.0, gravity=(0, 0, 0), magnetic=magnet, pinned=clamp)


def rotated(solver, angle):
    """Displacements that turn the body by ``angle`` about y."""
    c, s = np.cos(angle), np.sin(angle)
    r = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return (solver.p.mesh.nodes @ r.T - solver.p.mesh.nodes).ravel()


def test_a_turned_body_has_the_energy_of_its_two_axes():
    """E(t) = -mu0 V H^2 / 2 (chi_a cos^2 t + chi_n sin^2 t) in a uniform field along x: a torque, and no force."""
    m = SoftMagnet(0.9, 0.2, 0.4)
    s = Solid3DSolver(bar(0.03, 0.01, 0.004, 0.005, m), dynamic=False)
    h = 3e4
    s.magnetic_field[:3] = (h, 0, 0)
    v = 0.03 * 0.01 * 0.004
    for t in (0.0, 0.3, 0.8, np.pi / 2):
        want = -0.5 * MU0 * v * h * h * (0.9 * np.cos(t) ** 2 + 0.2 * np.sin(t) ** 2)
        assert s.magnetic_energy(rotated(s, t)) == pytest.approx(want, rel=1e-9)
    s.magnetic_field[:3] = (0, h, 0)  # along the turning axis: across both, whatever the turn
    for t in (0.0, 0.8):
        assert s.magnetic_energy(rotated(s, t)) == pytest.approx(-0.5 * MU0 * v * h * h * 0.4, rel=1e-9)


def test_a_sheared_body_turns_its_fibres_and_keeps_its_layers():
    """Shear u_z = g x: the fibre along x turns to x + g z, while the layers' normal goes to z - g x (F^-T, not F) and the
    two stay perpendicular. In a field H along x: E = -mu0 V H^2 (chi_a + chi_n g^2) / 2 (1 + g^2). A rotation can't
    tell F from F^-T, so this is the check on the normal."""
    m = SoftMagnet(0.9, 0.2, 0.4)
    s = Solid3DSolver(bar(0.03, 0.01, 0.004, 0.005, m), dynamic=False)
    h, v = 3e4, 0.03 * 0.01 * 0.004
    s.magnetic_field[:3] = (h, 0, 0)
    for g in (0.1, 0.5, 1.0):
        y = np.zeros((s.n, 3))
        y[:, 2] = g * s.p.mesh.nodes[:, 0]
        want = -0.5 * MU0 * v * h * h * (0.9 + 0.2 * g * g) / (1 + g * g)
        assert s.magnetic_energy(y.ravel()) == pytest.approx(want, rel=1e-9)


def test_the_body_is_pulled_up_a_dipoles_gradient():
    """A small isotropic piece on a dipole's axis, r above it: H = 2 m / (4 pi r^3), so F_z = mu0 chi V d(H^2)/dz / 2 =
    -3 mu0 chi V m^2 / (4 pi^2 r^7), by hand, the piece taken as a point: 0.67% off at 15 of its half-widths away (the
    field varies across it, about 7 (a / r)^2), so this one stands 50 away."""
    chi = 0.5
    s = Solid3DSolver(bar(0.002, 0.002, 0.002, 0.001, SoftMagnet(chi, chi, chi)), dynamic=False)
    centre = jnp.asarray(s.p.mesh.nodes.min(axis=0) + 0.001)
    s.magnetic_field[:] = (0, 0, 0, 0, 0, 1.0, *np.asarray(centre + jnp.array([0, 0, -0.1])))
    r = 0.1
    want_z = -3 * MU0 * chi * 0.002 ** 3 / (4 * np.pi ** 2 * r ** 7)  # m = 1 A m^2; towards the dipole, below
    d = 1e-6
    got = []
    for k in range(3):
        y = np.zeros(3 * s.n)
        y[k::3] = d
        got.append(-(s.magnetic_energy(y) - s.magnetic_energy(-y)) / (2 * d))
    assert got[2] == pytest.approx(want_z, rel=1e-3)  # 7 (a / r)^2 = 0.07%
    assert abs(got[0]) < 1e-6 * abs(want_z) and abs(got[1]) < 1e-6 * abs(want_z)  # on the axis: no sideways pull


# a tilted ring magnet above the bar's top face, its rim 0.57 mm from the nearest node of the deformed bar (none inside it)
RING_BESIDE = tuple(field_parameters(uniform=(0, 0, 3e3), cylinders=ring((0.012, -0.004, 0.0158), (0.15, 1, 0), 1.23, 0.01,
                                                                          0.0021, 0.005)))


@pytest.mark.parametrize("q", [(2e4, 0, 1e4, 0, 0, 5.0, 0.015, 0.005, -0.03), RING_BESIDE], ids=["dipole", "ring magnet"])
def test_the_solvers_hessian_is_the_derivative_of_its_gradient(q):
    """The magnetic part of Newton's gradient and Hessian (field on minus field off) against central differences, on a
    sheared, stretched body in a dipole's uneven field, and next to a tilted ring magnet whose rim comes within 0.6 mm of
    it (there the solver differentiates the field's local expansion, not the field: this is what checks that). Without the
    Hessian Newton still converges, only slower, so the solutions alone would not show it missing."""
    s = Solid3DSolver(bar(0.03, 0.01, 0.004, 0.005, SoftMagnet(0.9, 0.2, 0.4), pinned=True), dynamic=False)
    rng = np.random.default_rng(0)
    x = s.p.mesh.nodes
    y = np.zeros(s.ndof)
    y[:3 * s.n] = (0.1 * np.stack([x[:, 2], 0.2 * x[:, 0], x[:, 0]], 1) + 1e-4 * rng.standard_normal((s.n, 3))).ravel()
    ctx = s._context(1e-3)

    def magnetic(y):
        s.magnetic_field = np.array(q, float)
        g_on, h_on = s._grad_hess(y, ctx)
        s.magnetic_field = np.zeros(len(q))
        g_off, h_off = s._grad_hess(y, ctx)
        return (g_on - g_off)[s.free], h_on - h_off

    g, h = magnetic(y)
    upper = s._pattern.matrix(h)
    full = upper + upper.T - sp.diags(upper.diagonal())
    d = np.zeros(s.ndof)
    d[s.free] = rng.standard_normal(len(s.free))
    eps = 1e-7
    fd = (magnetic(y + eps * d)[0] - magnetic(y - eps * d)[0]) / (2 * eps)
    assert np.abs(g).max() > 0
    np.testing.assert_allclose(full @ d[s.free], fd, rtol=0, atol=1e-5 * np.abs(fd).max())
    s.magnetic_field = np.array(q, float)  # and the gradient is the energy's (the line search reads the energy)
    slope = (s.magnetic_energy(y + eps * d) - s.magnetic_energy(y - eps * d)) / (2 * eps)
    assert g @ d[s.free] == pytest.approx(slope, rel=1e-6)


def test_a_cantilever_follows_its_closed_form_up_to_buckling():
    """A field across a strip, chi_a > chi_n, acts like an axial load: E I t'' + K (t + e) = 0, with K = mu0 A (chi_a - chi_n) H^2
    and e the field's small tilt. Straight is unstable at K = E I (pi / 2 L)^2, and below that the tip moves
    e (sin(lL) / l + tan(lL) (1 - cos(lL)) / l - L), l^2 = K / E I. Two elements through the thickness: one is far too soft."""
    length, side, tilt = 0.06, 0.004, np.radians(0.2)
    s = Solid3DSolver(bar(length, side, side, side / 2, SoftMagnet(1.0, 0.1, 0.1), pinned=True), dynamic=False)
    critical = np.sqrt(E_MOD * side ** 2 / 12 * (np.pi / (2 * length)) ** 2 / (MU0 * 0.9))
    nodes = s.p.mesh.nodes
    tip = np.flatnonzero(np.isclose(nodes[:, 0], nodes[:, 0].max()))
    for ratio, tolerance in ((0.3, 0.1), (0.5, 0.1), (0.7, 0.15)):
        h = ratio * critical
        s.magnetic_field[:3] = (h * np.sin(tilt), 0, h * np.cos(tilt))
        assert s.advance(1.0)
        l = ratio * np.pi / (2 * length)
        want = tilt * (np.sin(l * length) / l + np.tan(l * length) * (1 - np.cos(l * length)) / l - length)
        assert s.y[:3 * s.n].reshape(-1, 3)[tip, 2].mean() == pytest.approx(want, rel=tolerance)


def test_the_magnetic_kernels_in_chunks_are_the_same(monkeypatch):
    """Elements a few at a time (a magnet's Hessian is large: big meshes need it) give the same energy, gradient and
    Hessian as all at once."""
    def solver():
        s = Solid3DSolver(bar(0.03, 0.01, 0.004, 0.005, SoftMagnet(0.9, 0.2, 0.4), pinned=True), dynamic=False)
        s.magnetic_field[:] = (2e4, 0, 1e4, 0, 0, 5.0, 0.015, 0.005, -0.03)
        return s
    whole = solver()
    monkeypatch.setenv("SOFTSIM_FEM_MAG_CHUNK", "7")
    parts = solver()
    assert len(whole.p.mesh.tets) > 3 * 7
    y = 1e-5 * np.random.default_rng(2).standard_normal(whole.ndof)  # small: no element turned inside out
    ctx = whole._context(1e-3)
    assert parts.magnetic_energy(y) == pytest.approx(whole.magnetic_energy(y), rel=1e-12)
    for a, b in zip(parts._grad_hess(y, ctx), whole._grad_hess(y, ctx)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max())
