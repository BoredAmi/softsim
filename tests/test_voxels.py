"""Voxel bodies: conforming meshes of occupied cubes, regions, labels and validation."""

import numpy as np
import pytest
from pydantic import ValidationError

from softsim.geometry import MeshSettings, SoftVoxels, check_resolution
from softsim.meshing import structured_box, voxel_mesh
from softsim.meshing.tet import _orient

L_SHAPE = [["AB.", "A..", "..."], ["A..", "...", "..."]]  # 4 cells, face-connected


def _old_structured_box(size_xyz, size):
    sx, sy, sz = size_xyz
    n = [max(1, round(e / size)) for e in (sx, sy, sz)]
    axes = [np.linspace(-sx / 2, sx / 2, n[0] + 1), np.linspace(-sy / 2, sy / 2, n[1] + 1),
            np.linspace(0.0, sz, n[2] + 1)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    idx = np.arange(len(points)).reshape(n[0] + 1, n[1] + 1, n[2] + 1)
    base = idx[:-1, :-1, :-1].ravel()
    step = np.array([idx[1, 0, 0] - idx[0, 0, 0], idx[0, 1, 0] - idx[0, 0, 0], idx[0, 0, 1] - idx[0, 0, 0]])
    tets = []
    for perm in [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]:
        v1 = base + step[perm[0]]
        v2 = v1 + step[perm[1]]
        tets.append(np.column_stack([base, v1, v2, base + step.sum()]))
    return points, _orient(points, np.concatenate(tets))


def test_structured_box_unchanged():
    pts, tets = _old_structured_box((0.04, 0.03, 0.02), 0.005)
    m = structured_box((0.04, 0.03, 0.02), 0.005)
    assert np.array_equal(m.points, pts) and np.array_equal(m.tets, tets)


@pytest.mark.parametrize("s", [1, 2])
def test_volume_and_conformity(s):
    shape = SoftVoxels(voxel=0.01, cells=L_SHAPE, subdivide=s)
    mesh = shape.mesh(MeshSettings())
    assert mesh.volume() == pytest.approx(4 * 0.01**3)
    assert shape.volume() == pytest.approx(4 * 0.01**3)
    assert np.all(mesh.tet_volumes() > 0)
    assert len(np.unique(mesh.points.round(9), axis=0)) == mesh.n_points  # no duplicated nodes
    assert len(np.unique(mesh.tets)) == mesh.n_points  # no unused nodes
    t = mesh.tets
    faces = np.sort(np.concatenate([t[:, [0, 1, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 3]], t[:, [1, 2, 3]]]), axis=1)
    _, counts = np.unique(faces, axis=0, return_counts=True)
    assert set(counts) <= {1, 2}
    # conforming: boundary area equals the cubes' exposed area (L shape: 4 cubes, 3 shared faces)
    tri = mesh.points[mesh.surface_triangles()]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
    assert area == pytest.approx((4 * 6 - 2 * 3) * 0.01**2)


def test_frame():
    mesh = SoftVoxels(voxel=0.01, cells=[["X.."]]).mesh(MeshSettings())
    assert mesh.points[:, 2].min() == 0 and mesh.points[:, 2].max() == pytest.approx(0.01)
    assert mesh.points[:, 0].min() == pytest.approx(-0.015) and mesh.points[:, 0].max() == pytest.approx(-0.005)


def test_voxel_mesh_owner():
    occ = np.zeros((2, 1, 2), bool)
    occ[0, 0, 0] = occ[1, 0, 0] = occ[1, 0, 1] = True
    mesh, owner = voxel_mesh(occ, 0.01, 2)
    assert len(owner) == mesh.n_tets and set(owner) == {0, 2, 3}
    assert np.bincount(owner)[[0, 2, 3]].tolist() == [48] * 3


@pytest.mark.parametrize("s", [1, 2])
def test_regions_cover_boundary_once(s):
    shape = SoftVoxels(voxel=0.01, cells=L_SHAPE, subdivide=s)
    mesh = shape.mesh(MeshSettings())
    r = shape.regions(mesh)
    allt = np.sort(np.concatenate([r["base"], r["top"], r["side"]]), axis=1)
    surf = np.sort(mesh.surface_triangles(), axis=1)
    assert len(allt) == len(surf) and len(np.unique(allt, axis=0)) == len(allt)
    assert np.all(mesh.points[r["base"]][:, :, 2] == 0)
    for t in r["top"]:  # all upward
        p = mesh.points[t]
        assert np.cross(p[1] - p[0], p[2] - p[0])[2] > 0
    # L shape: footprint 3 cells, tops: 3 cells (top of the stack, the B cell and A at z=1 is the stack top)
    area = lambda tris: 0.5 * np.linalg.norm(  # noqa: E731
        np.cross(*[(mesh.points[tris][:, i] - mesh.points[tris][:, 0]) for i in (1, 2)]), axis=1).sum()
    assert area(r["base"]) == pytest.approx(3 * 0.01**2)
    assert area(r["top"]) == pytest.approx(3 * 0.01**2)


def test_volume_regions_partition_and_labels():
    shape = SoftVoxels(voxel=0.01, cells=L_SHAPE, subdivide=2)
    mesh = shape.mesh(MeshSettings())
    vr = shape.volume_regions(mesh)
    assert set(vr) == {"A", "B"}
    assert np.array_equal(np.sort(np.concatenate(list(vr.values()))), np.arange(mesh.n_tets))
    vol = mesh.tet_volumes()
    assert vol[vr["A"]].sum() == pytest.approx(3 * 0.01**3) and vol[vr["B"]].sum() == pytest.approx(0.01**3)
    cen = mesh.points[mesh.tets][vr["B"]].mean(axis=1)
    assert np.all((cen[:, 0] > -0.005 + 0 - 0.015) & (cen[:, 0] < 0.015)) and np.all(cen[:, 2] < 0.01)
    # a foreign mesh (e.g. rebuilt) falls back to the centroids and agrees
    other = voxel_mesh(shape.occupancy(), 0.01, 2)[0]
    vr2 = shape.volume_regions(other)
    assert {k: v.tolist() for k, v in vr2.items()} == {k: v.tolist() for k, v in vr.items()}


def test_from_array_roundtrip_and_resolution():
    a = np.full((3, 2, 2), ".", dtype="<U1")
    a[:, 0, 0] = "M"
    a[1, 1, 0] = "P"
    a[1, 1, 1] = "P"
    shape = SoftVoxels.from_array(a, 0.002, subdivide=2)
    assert np.array_equal(shape.to_array(), a)
    check_resolution(shape, MeshSettings(size=0.0035))  # size ignored: no refusal
    assert shape.min_feature == pytest.approx(0.001) and shape.top_height == pytest.approx(0.004)


@pytest.mark.parametrize("cells, msg", [
    ([], "empty"),
    ([["..", ".."]], "no occupied"),
    ([["A.", "..."]], "same size"),
    ([["A."], ["AA", "A."]], "same size"),
    ([["..", ".."], ["A.", ".."]], "bottom layer"),
    ([["A.", ".A"]], "islands"),
    ([["A A"]], "whitespace"),
])
def test_validation_errors(cells, msg):
    with pytest.raises(ValidationError, match=msg):
        SoftVoxels(voxel=0.01, cells=cells)


def test_island_sizes_listed():
    with pytest.raises(ValidationError, match=r"\[2, 1\]"):
        SoftVoxels(voxel=0.01, cells=[["AA.B"]])
