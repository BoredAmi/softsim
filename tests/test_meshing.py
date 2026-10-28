import math

import numpy as np
import pytest

from softsim.meshing import structured_box, structured_cylinder


def _check_mesh(mesh, expected_volume, rel=0.02):
    assert (mesh.tet_volumes() > 0).all()
    assert mesh.volume() == pytest.approx(expected_volume, rel=rel)
    # conforming: every face shared by at most two tets
    t = mesh.tets
    faces = np.sort(np.concatenate([t[:, [0, 1, 2]], t[:, [0, 1, 3]], t[:, [0, 2, 3]], t[:, [1, 2, 3]]]), axis=1)
    _, counts = np.unique(faces, axis=0, return_counts=True)
    assert counts.max() == 2
    # closed surface: every surface edge belongs to exactly two surface triangles
    tris = mesh.surface_triangles()
    edges = np.sort(np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]), axis=1)
    _, ecounts = np.unique(edges, axis=0, return_counts=True)
    assert (ecounts == 2).all()
    # outward normals: divergence theorem recovers the volume
    p = mesh.points[tris]
    vol = np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6
    assert vol == pytest.approx(mesh.volume(), rel=1e-9)
    # every vertex used
    assert len(np.unique(t)) == mesh.n_points


@pytest.mark.parametrize("size", [0.006, 0.0035, 0.0025])
def test_structured_cylinder(size):
    mesh = structured_cylinder(0.02, 0.017, size)
    _check_mesh(mesh, math.pi * 0.02**2 * 0.017)
    assert mesh.points[:, 2].min() == 0 and mesh.points[:, 2].max() == pytest.approx(0.017)
    assert np.hypot(mesh.points[:, 0], mesh.points[:, 1]).max() == pytest.approx(0.02)
    assert np.any(np.all(np.isclose(mesh.points, [0, 0, 0.017]), axis=1)), "vertex at top centre"
    assert mesh.quality()["min_quality"] > 0.3


def test_structured_box():
    mesh = structured_box((0.03, 0.02, 0.01), 0.005)
    _check_mesh(mesh, 0.03 * 0.02 * 0.01, rel=1e-9)


def test_lumped_masses():
    mesh = structured_cylinder(0.02, 0.017, 0.005)
    m = mesh.lumped_masses(1000.0)
    assert m.sum() == pytest.approx(1000.0 * mesh.volume())
    assert (m > 0).all()


def test_gmsh_cylinder():
    pytest.importorskip("gmsh")
    from softsim.meshing import gmsh_cylinder

    mesh = gmsh_cylinder(0.02, 0.017, 0.005)
    _check_mesh(mesh, math.pi * 0.02**2 * 0.017, rel=0.03)
