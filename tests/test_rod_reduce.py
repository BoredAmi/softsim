"""Reduction of a slender tet body to beam frames (no SOFA): section integrals, vertex map, quaternions."""

import numpy as np
import pytest

from softsim.backends.rod.reduce import P, quat_to_matrix, reduce_body, slerp
from softsim.errors import UnsupportedFeatureError
from softsim.materials.models import Material
from softsim.meshing import structured_box
from softsim.scene import MaterialZone, SoftBodySpec


def _material(youngs):
    return Material.model_validate({"density": 1000.0, "elastic": {"model": "stable_neo_hookean",
                                                                    "youngs_modulus": youngs, "poisson_ratio": 0.3}})


def _body(w=0.012, h=0.008, length=0.06, size=0.002, zones=(), E=1e6):
    mesh = structured_box((w, h, length), size)
    return SoftBodySpec(name="beam", mesh=mesh, material=_material(E), pinned=tuple(np.nonzero(mesh.points[:, 2] < 1e-9)[0]),
                        zones=tuple(zones))


def test_section_integrals_of_a_box_are_exact():
    body = _body()
    rod = reduce_body(body)
    w, h = 0.012, 0.008
    assert rod.ea == pytest.approx(1e6 * w * h, rel=1e-10)
    assert rod.ei_x == pytest.approx(1e6 * h * w**3 / 12, rel=1e-10)  # deflection along x: I = h w^3 / 12
    assert rod.ei_y == pytest.approx(1e6 * w * h**3 / 12, rel=1e-10)
    assert rod.mass_per_length == pytest.approx(1000.0 * w * h, rel=1e-10)


def test_rest_frames_reproduce_the_mesh():
    body = _body()
    rod = reduce_body(body)
    frames = rod.rest_frames()
    pos = rod.vertex_positions(frames[:, :3], frames[:, 3:])
    assert np.abs(pos - body.mesh.points).max() < 1e-12


def test_zones_enter_the_section_stiffness():
    body = _body()
    z = np.nonzero(np.abs(body.mesh.points[body.mesh.tets].mean(axis=1)[:, 2] - 0.03) < 0.0035)[0]
    zoned = _body(zones=[MaterialZone("stiff", z, _material(5e6))])
    soft, stiff = reduce_body(body), reduce_body(zoned)
    assert stiff.ei_x.max() == pytest.approx(5 * soft.ei_x[0], rel=1e-9)
    assert stiff.ei_x.min() == pytest.approx(soft.ei_x[0], rel=1e-9)


def test_layered_section_moves_the_neutral_axis_and_gives_the_parallel_axis_sum():
    # a stiff half (x > 0) and a soft half: EI about the E-weighted centroid = sum E_i (I_i + A_i d_i^2)
    body = _body(w=0.012, h=0.008, size=0.002)
    cx = body.mesh.points[body.mesh.tets].mean(axis=1)[:, 0]
    right = np.nonzero(cx > cx.mean())[0]
    layered = _body(zones=[MaterialZone("right", right, _material(4e6))])
    rod = reduce_body(layered)
    w, h, e1, e2 = 0.006, 0.008, 1e6, 4e6  # each half: width w
    xc = (e1 * w * h * 0.5 * w + e2 * w * h * 1.5 * w) / ((e1 + e2) * w * h)  # centroid measured from x = 0
    exact = sum(e * (h * w**3 / 12 + w * h * (c - xc) ** 2) for e, c in ((e1, 0.5 * w), (e2, 1.5 * w)))
    assert rod.ei_x[0] == pytest.approx(exact, rel=1e-9)
    assert rod.centre[0, 0] == pytest.approx(xc - 0.006, rel=1e-9)  # the mesh spans x = -6 .. 6 mm


def test_equivalent_section_reproduces_all_three_stiffnesses_also_for_a_composite():
    body = _body()
    cx = body.mesh.points[body.mesh.tets].mean(axis=1)[:, 0]
    layered = _body(zones=[MaterialZone("right", np.nonzero(cx > cx.mean())[0], _material(4e6))])
    for b in (body, layered):
        rod = reduce_body(b)
        ly, lz, young = rod.equivalent_section()
        assert young * ly * lz == pytest.approx(rod.ea, rel=1e-10)
        assert young * lz * ly**3 / 12 == pytest.approx(rod.ei_x, rel=1e-10)
        assert young * ly * lz**3 / 12 == pytest.approx(rod.ei_y, rel=1e-10)
    ly, lz, young = reduce_body(body).equivalent_section()
    assert (ly[0], lz[0], young[0]) == pytest.approx((0.012, 0.008, 1e6))


def test_rigid_rotation_of_the_frames_moves_the_mesh_rigidly():
    body = _body()
    rod = reduce_body(body)
    angle = 0.3  # about scene y = internal Z
    q = np.array([0, 0, np.sin(angle / 2), np.cos(angle / 2)])
    frames = rod.rest_frames()
    pos, quat = frames[:, :3].copy(), np.tile(q, (len(frames), 1))
    # rotate every frame position about the origin too, as a rigid motion of the whole rod
    r = quat_to_matrix(q)
    pos = pos @ r.T
    moved = rod.vertex_positions(pos, quat)
    rs = P.T @ r @ P  # the same rotation in scene coordinates
    assert np.abs(moved - body.mesh.points @ rs.T).max() < 1e-12


def test_slerp_between_known_rotations():
    a = np.array([[0, 0, 0, 1.0]])
    b = np.array([[0, 0, np.sin(0.4), np.cos(0.4)]])  # 0.8 rad about z
    mid = slerp(a, b, np.array([0.5]))
    assert mid[0] == pytest.approx([0, 0, np.sin(0.2), np.cos(0.2)])


def test_refuses_stubby_and_free_bodies():
    with pytest.raises(UnsupportedFeatureError, match="longer than wide"):
        reduce_body(_body(length=0.02))
    mesh = structured_box((0.01, 0.01, 0.06), 0.002)
    with pytest.raises(UnsupportedFeatureError, match="no fixed end"):
        reduce_body(SoftBodySpec(name="b", mesh=mesh, material=_material(1e6)))
