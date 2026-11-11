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


def _magnetisation(material, h):
    """M = -dw/d(mu0 H) [A/m] at rest (F = I)."""
    import jax

    from softsim.fem.magnetic import density_in

    return -np.asarray(jax.grad(lambda hh: density_in(jnp.eye(3), hh, material))(jnp.asarray(h, float))) / MU0


def test_saturation_is_a_tanh_along_the_easy_axis_and_bounded_everywhere():
    """M = M_s tanh(chi H / M_s) along the easiest axis, chi H at small fields (the linear law), never above M_s in
    any direction; M_s = 0 is exactly the linear law."""
    m_s, chi = 4e5, 2.0
    sat = SoftMagnet(chi, 0.5, 0.8, saturation=m_s)
    lin = SoftMagnet(chi, 0.5, 0.8)
    for h in (1.0, 1e3, 5e4, 2e5, 1e6, 1e8):
        assert _magnetisation(sat, (h, 0, 0))[0] == pytest.approx(m_s * np.tanh(chi * h / m_s), rel=1e-9)
    small = (30.0, -20.0, 10.0)
    np.testing.assert_allclose(_magnetisation(sat, small), _magnetisation(lin, small), rtol=1e-6)
    np.testing.assert_allclose(_magnetisation(lin, small), [chi * 30, 0.8 * -20, 0.5 * 10], rtol=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(50):
        h = rng.normal(size=3) * 10 ** rng.uniform(2, 8)
        assert np.linalg.norm(_magnetisation(sat, h)) <= m_s * (1 + 1e-9)
    assert np.all(np.isfinite(_magnetisation(sat, (0.0, 0.0, 0.0))))
    with pytest.raises(ValueError, match="saturation"):
        SoftMagnet(1, 1, 1, saturation=-1)


def test_a_saturated_body_has_the_energy_of_its_tanh_law():
    """A uniform field along the bar's easy axis: E = -mu0 V M_s^2 / chi_a ln cosh(chi_a H / M_s), from below to well
    above saturation."""
    m_s = 2e4
    s = Solid3DSolver(bar(0.03, 0.01, 0.004, 0.005, SoftMagnet(0.9, 0.2, 0.4, saturation=m_s)), dynamic=False)
    v = 0.03 * 0.01 * 0.004
    for h in (1e3, 3e4, 3e5):
        s.magnetic_field[:3] = (h, 0, 0)
        want = -MU0 * v * m_s ** 2 / 0.9 * np.log(np.cosh(0.9 * h / m_s))
        assert s.magnetic_energy(np.zeros(3 * s.n)) == pytest.approx(want, rel=1e-9)


def test_a_saturated_piece_is_pulled_by_its_saturation_not_its_susceptibility():
    """The small piece of the gradient test, in a dipole strong enough to saturate it (chi H / M_s = 3): F_z =
    mu0 V M dH/dz with M = M_s tanh(chi H / M_s), not the linear chi H, which would pull 3.0 times harder."""
    chi, m_s, r = 0.5, 1e3, 0.1
    s = Solid3DSolver(bar(0.002, 0.002, 0.002, 0.001, SoftMagnet(chi, chi, chi, saturation=m_s)), dynamic=False)
    centre = jnp.asarray(s.p.mesh.nodes.min(axis=0) + 0.001)
    h_target = 3 * m_s / chi
    moment = h_target * 4 * np.pi * r ** 3 / 2
    s.magnetic_field[:] = (0, 0, 0, 0, 0, moment, *np.asarray(centre + jnp.array([0, 0, -r])))
    h, dh = 2 * moment / (4 * np.pi * r ** 3), -6 * moment / (4 * np.pi * r ** 4)
    want = MU0 * 0.002 ** 3 * m_s * np.tanh(chi * h / m_s) * dh
    d = 1e-6
    y = np.zeros(3 * s.n)
    y[2::3] = d
    got = -(s.magnetic_energy(y) - s.magnetic_energy(-y)) / (2 * d)
    assert got == pytest.approx(want, rel=2e-3)
    assert abs(MU0 * 0.002 ** 3 * chi * h * dh / got) == pytest.approx(3 / np.tanh(3), rel=2e-3)


def test_a_hard_magnetic_body_turns_and_stretches_with_its_remanence():
    """Remanence M_r along x, uniform H: E = -mu0 V (F M_r) . H. Turned by t about y, F M_r = M_r (cos t, 0, -sin t);
    stretched by s along x, F M_r = s M_r x (F, not its rotation: the magnetised material carries the moment)."""
    m_r, h, v = 8e4, 2e3, 0.03 * 0.01 * 0.004
    s = Solid3DSolver(bar(0.03, 0.01, 0.004, 0.005, SoftMagnet(0.0, 0.0, 0.0, remanence=(m_r, 0.0, 0.0))), dynamic=False)
    for t in (0.0, 0.4, np.pi / 2, 2.5):
        s.magnetic_field[:3] = (h, 0, 0)
        assert s.magnetic_energy(rotated(s, t)) == pytest.approx(-MU0 * v * m_r * h * np.cos(t), abs=1e-12 * MU0 * v * m_r * h)
        s.magnetic_field[:3] = (0, 0, h)
        assert s.magnetic_energy(rotated(s, t)) == pytest.approx(MU0 * v * m_r * h * np.sin(t), abs=1e-12 * MU0 * v * m_r * h)
    s.magnetic_field[:3] = (h, 0, 0)
    for stretch in (1.1, 1.5):
        y = np.zeros((s.n, 3))
        y[:, 0] = (stretch - 1) * s.p.mesh.nodes[:, 0]
        assert s.magnetic_energy(y.ravel()) == pytest.approx(-MU0 * v * m_r * h * stretch, rel=1e-9)


@pytest.mark.slow
def test_a_hard_magnetic_cantilever_bends_under_its_distributed_couple():
    """Magnetised along its length, in a small field across it: the couple mu0 M_r x H, c = mu0 M_r H A per length, gives
    Euler-Bernoulli's tip deflection c L^3 / (3 E I) towards the field (the moment turns to follow it). For a uniform
    remanence the energy -mu0 M_r H du_z/dX integrates to the fibres' end displacements, so the load sits on the end
    face (the same tip deflection as the couple's); the coarse mesh reads it locally soft: +5.5% at 2 mm, +2.1% at
    1.33 mm, +1.1% at 1 mm (second order)."""
    length, side, m_r, h = 0.06, 0.004, 1e5, 30.0
    s = Solid3DSolver(bar(length, side, side, side / 4, SoftMagnet(0.0, 0.0, 0.0, remanence=(m_r, 0.0, 0.0)),
                          pinned=True), dynamic=False)
    s.magnetic_field[:3] = (0, 0, h)
    assert s.advance(1.0)
    nodes = s.p.mesh.nodes
    tip = np.flatnonzero(np.isclose(nodes[:, 0], nodes[:, 0].max()))
    couple = MU0 * m_r * h * side ** 2
    want = couple * length ** 3 / (3 * E_MOD * side ** 4 / 12)
    assert want == pytest.approx(1e-3, rel=0.05)  # about a millimetre: small deflection
    assert s.y[:3 * s.n].reshape(-1, 3)[tip, 2].mean() == pytest.approx(want, rel=0.02)
    s.magnetic_field[:3] = (0, 0, -h)  # the other way
    assert s.advance(1.0)
    assert s.y[:3 * s.n].reshape(-1, 3)[tip, 2].mean() == pytest.approx(-want, rel=0.02)


def zoned_bar(zones, magnet=None, size=0.005):
    """The 30 x 10 x 4 mm bar with magnetic zones ``zones``: (selector on element centroids, SoftMagnet) pairs."""
    p = bar(0.03, 0.01, 0.004, size, magnet)
    centroids = p.mesh.nodes[p.mesh.tets[:, :4]].mean(axis=1)
    groups = tuple((np.flatnonzero(select(centroids)), m) for select, m in zones)
    assert all(len(e) for e, _ in groups)
    return Problem3D(**{**p.__dict__, "magnetic_zones": groups})


def test_opposite_remanence_zones_cancel_and_a_non_magnetic_zone_halves():
    """Magnetic zones, each element with its own magnet. A bar split at mid-length (x = 0, the mesh is centred) into +M_r and -M_r along its length, in
    a uniform field along it: the halves' energies -mu0 V/2 M_r H and +mu0 V/2 M_r H cancel, at rest, turned by any angle
    and stretched. With the second half not magnetic, the energy is the first half's, -mu0 V/2 M_r H cos t: half the whole
    bar's."""
    m_r, h, v = 8e4, 2e3, 0.03 * 0.01 * 0.004
    plus, minus = (SoftMagnet(0.0, 0.0, 0.0, remanence=(s * m_r, 0.0, 0.0)) for s in (1, -1))
    left, right = (lambda c: c[:, 0] < 0.0), (lambda c: c[:, 0] > 0.0)
    opposite = Solid3DSolver(zoned_bar([(left, plus), (right, minus)]), dynamic=False)
    half = Solid3DSolver(zoned_bar([(left, plus)]), dynamic=False)
    whole = Solid3DSolver(bar(0.03, 0.01, 0.004, 0.005, plus), dynamic=False)
    scale = MU0 * v * m_r * h
    for s in (opposite, half, whole):
        s.magnetic_field[:3] = (h, 0, 0)
    assert whole.magnetic_energy(np.zeros(3 * whole.n)) == pytest.approx(-scale, rel=1e-12)
    for t in (0.0, 0.4, 2.5):
        assert abs(opposite.magnetic_energy(rotated(opposite, t))) < 1e-12 * scale
        assert half.magnetic_energy(rotated(half, t)) == pytest.approx(-0.5 * scale * np.cos(t), abs=1e-12 * scale)
        assert half.magnetic_energy(rotated(half, t)) == pytest.approx(0.5 * whole.magnetic_energy(rotated(whole, t)),
                                                                        abs=1e-12 * scale)
    y = np.zeros((opposite.n, 3))  # stretched by 1.3 along x: each half's moment grows with it, still cancelling
    y[:, 0] = 0.3 * opposite.p.mesh.nodes[:, 0]
    assert abs(opposite.magnetic_energy(y.ravel())) < 1e-12 * scale
    assert half.magnetic_energy(y.ravel()) == pytest.approx(-0.5 * scale * 1.3, rel=1e-9)


def test_a_turned_two_zone_body_has_the_sum_of_its_zones_closed_forms():
    """One zone hard (M_r along x), the other soft (chi_a along x, chi_n along z), turned by t about y in a field along x:
    E = -mu0 V1 M_r H cos t - mu0 V2 H^2 / 2 (chi_a cos^2 t + chi_n sin^2 t), each zone's closed form over its own
    volume (here a quarter and three quarters of the bar)."""
    m_r, h, v = 8e4, 3e4, 0.03 * 0.01 * 0.004
    hard, soft = SoftMagnet(0.0, 0.0, 0.0, remanence=(m_r, 0.0, 0.0)), SoftMagnet(0.9, 0.2, 0.4)
    s = Solid3DSolver(zoned_bar([(lambda c: c[:, 0] < -0.0075, hard), (lambda c: c[:, 0] > -0.0075, soft)],
                                size=0.0025), dynamic=False)
    s.magnetic_field[:3] = (h, 0, 0)
    v1, v2 = v / 4, 3 * v / 4
    for t in (0.0, 0.3, 0.8, np.pi / 2, 2.0):
        want = -MU0 * v1 * m_r * h * np.cos(t) - 0.5 * MU0 * v2 * h * h * (0.9 * np.cos(t) ** 2 + 0.2 * np.sin(t) ** 2)
        assert s.magnetic_energy(rotated(s, t)) == pytest.approx(want, rel=1e-9)


def test_zones_assemble_each_elements_own_gradient_and_hessian():
    """Newton's magnetic gradient and Hessian with zones: (1) zones that all carry the body's magnet give exactly the
    single-magnet body's; (2) with two different magnets, the gradient is the energy's slope and the Hessian the
    gradient's derivative (central differences), on a deformed body in a dipole's uneven field."""
    q = (2e4, 0, 1e4, 0, 0, 5.0, 0.015, 0.005, -0.03)
    m = SoftMagnet(0.9, 0.2, 0.4, remanence=(3e4, 0, 1e4))
    other = SoftMagnet(0.3, 0.6, 0.1, along=(0.0, 1.0, 0.0), normal=(1.0, 0.0, 0.0), remanence=(0, -5e4, 0))
    left, right = (lambda c: c[:, 0] < 0.0), (lambda c: c[:, 0] > 0.0)
    rng = np.random.default_rng(4)

    def solver(p):
        clamp = np.flatnonzero(np.isclose(p.mesh.nodes[:, 0], p.mesh.nodes[:, 0].min()))
        s = Solid3DSolver(Problem3D(**{**p.__dict__, "pinned": clamp}), dynamic=False)
        s.magnetic_field[:] = q
        return s

    whole, same = solver(bar(0.03, 0.01, 0.004, 0.005, m)), solver(zoned_bar([(left, m), (right, m)]))
    x = whole.p.mesh.nodes
    y = np.zeros(whole.ndof)
    y[:3 * whole.n] = (0.1 * np.stack([x[:, 2], 0.2 * x[:, 0], x[:, 0]], 1)
                       + 1e-4 * rng.standard_normal((whole.n, 3))).ravel()
    ctx = whole._context(1e-3)
    assert same.magnetic_energy(y) == pytest.approx(whole.magnetic_energy(y), rel=1e-12)
    for a, b in zip(same._grad_hess(y, ctx), whole._grad_hess(y, ctx)):
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max())

    s = solver(zoned_bar([(left, m), (right, other)]))

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
    np.testing.assert_allclose(full @ d[s.free], fd, rtol=0, atol=1e-5 * np.abs(fd).max())
    s.magnetic_field = np.array(q, float)
    slope = (s.magnetic_energy(y + eps * d) - s.magnetic_energy(y - eps * d)) / (2 * eps)
    assert g @ d[s.free] == pytest.approx(slope, rel=1e-6)
    s_whole_other = solver(bar(0.03, 0.01, 0.004, 0.005, other))  # and the zoned body is neither magnet's alone
    assert s.magnetic_energy(y) not in (pytest.approx(whole.magnetic_energy(y)),
                                        pytest.approx(s_whole_other.magnetic_energy(y)))
