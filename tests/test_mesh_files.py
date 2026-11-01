"""Bodies from files (type: mesh_file): STEP, STL and gmsh meshes, physical groups as regions and
zones, cavities found by themselves, placement, CAD export, and the PneuNet from its own STEP."""

from pathlib import Path

import numpy as np
import pytest

from softsim.actuation import enclosed_volume, open_edges
from softsim.cli import main
from softsim.config.loader import load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError
from softsim.geometry import MeshSettings, SoftMeshFile, SoftPneuNet
from softsim.scenarios.inflation import InflationConfig, InflationScenario

gmsh = pytest.importorskip("gmsh")

ACTUATION = Path(__file__).parents[1] / "scenarios" / "actuation"
FINGER = dict(chambers=3, chamber_length=0.006, wall=0.002, end_wall=0.004, width=0.016, side_wall=0.002,
              height=0.016, base=0.005, top_wall=0.002, channel_width=0.004, channel_height=0.002, layer=0.0015)
SETTINGS = MeshSettings(size=0.002, method="gmsh")


def _tet_volumes(mesh):
    x = mesh.points[mesh.tets]
    return np.abs(np.linalg.det(x[:, 1:] - x[:, :1])) / 6


def _hollow_cube_stl(path):
    """20 mm cube around a 10 mm cubic cavity, as a millimetre STL."""
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        occ.cut([(3, occ.addBox(0, 0, 0, 20, 20, 20))], [(3, occ.addBox(5, 5, 5, 10, 10, 10))])
        occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", 4)
        gmsh.model.mesh.generate(2)
        gmsh.write(str(path))
    finally:
        gmsh.finalize()


def test_step_round_trip_is_the_same_body(tmp_path):
    """A parametric finger written as STEP (two solids) and read back: same volumes, one bonded body."""
    g = SoftPneuNet(**FINGER)
    g.export(tmp_path / "finger.step")
    f = SoftMeshFile(path=str(tmp_path / "finger.step"), box_regions={"base": [[0, -1, -1], [0, 1, 1]]})
    mesh = f.mesh(SETTINGS)
    assert mesh.volume() == pytest.approx(g.volume(), rel=1e-9)
    regions = f.regions(mesh)
    # the layer and the rest are fragmented into one conforming body: their interface is not a boundary
    assert "cavity" in regions and "cavity_2" not in regions and len(open_edges(regions["cavity"])) == 0
    assert enclosed_volume(mesh.points, regions["cavity"]) == pytest.approx(
        3 * 0.006 * 0.012 * 0.009 + 2 * 0.002 * 0.004 * 0.002, rel=1e-9)
    assert np.all(mesh.points[np.unique(regions["base"]), 0] == pytest.approx(0, abs=1e-12))
    below = mesh.points[mesh.tets].mean(axis=1)[:, 2] < FINGER["layer"]
    assert _tet_volumes(mesh)[below].sum() == pytest.approx(g.length * 0.016 * 0.0015, rel=1e-9)
    # the two solids, by gmsh tag
    volumes = {k: _tet_volumes(mesh)[v].sum() for k, v in f.volume_regions(mesh).items() if k != "body"}
    assert len(volumes) == 2 and min(volumes.values()) == pytest.approx(g.length * 0.016 * 0.0015, rel=1e-9)


@pytest.mark.parametrize("remesh", [True, False])
def test_stl_with_a_cavity(tmp_path, remesh):
    _hollow_cube_stl(tmp_path / "cube.stl")
    f = SoftMeshFile(path=str(tmp_path / "cube.stl"), scale=1e-3, remesh=remesh, center="base")
    mesh = f.mesh(MeshSettings(size=0.003, method="gmsh"))
    assert mesh.volume() == pytest.approx(7e-6, rel=1e-9)
    assert enclosed_volume(mesh.points, f.regions(mesh)["cavity"]) == pytest.approx(1e-6, rel=1e-9)
    np.testing.assert_allclose(mesh.points.min(axis=0), [-0.01, -0.01, 0], atol=1e-12)
    assert f.top_height == pytest.approx(0.02)


def test_obj_and_volume_meshes(tmp_path):
    """OBJ (quads, a cavity) is closed into a volume; VTK and Medit meshes are used as they are."""
    lines = []
    for base, (x0, x1, flip) in ((0, (0, 20, False)), (8, (5, 15, True))):  # outer cube, cavity (inward)
        lines += [f"v {x} {y} {z}" for x in (x0, x1) for y in (x0, x1) for z in (x0, x1)]
        quads = [(1, 3, 4, 2), (5, 6, 8, 7), (1, 2, 6, 5), (3, 7, 8, 4), (1, 5, 7, 3), (2, 4, 8, 6)]
        lines += ["f " + " ".join(str(base + i) for i in (q[::-1] if flip else q)) for q in quads]
    (tmp_path / "cube.obj").write_text("\n".join(lines) + "\n")
    f = SoftMeshFile(path=str(tmp_path / "cube.obj"), scale=1e-3)
    mesh = f.mesh(MeshSettings(size=0.004, method="gmsh"))
    assert mesh.volume() == pytest.approx(7e-6, rel=1e-9)
    assert enclosed_volume(mesh.points, f.regions(mesh)["cavity"]) == pytest.approx(1e-6, rel=1e-9)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.occ.addBox(0, 0, 0, 0.01, 0.02, 0.03)
        gmsh.model.occ.synchronize()
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.004)
        gmsh.model.mesh.generate(3)
        n_tets = sum(len(n) // 4 for n in gmsh.model.mesh.getElements(3, -1)[2])
        for ext in ("vtk", "mesh"):
            gmsh.write(str(tmp_path / f"box.{ext}"))
    finally:
        gmsh.finalize()
    for ext in ("vtk", "mesh"):
        mesh = SoftMeshFile(path=str(tmp_path / f"box.{ext}")).mesh(SETTINGS)
        assert mesh.n_tets == n_tets and mesh.volume() == pytest.approx(6e-6, rel=1e-9)


def test_msh_physical_groups(tmp_path):
    """A gmsh mesh is kept as is; its named groups become regions (2D) and volume regions (3D)."""
    path = tmp_path / "layered.msh"
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        occ.fragment([(3, occ.addBox(0, 0, 0, 0.02, 0.01, 0.01))], [(3, occ.addBox(0, 0, 0, 0.02, 0.01, 0.002))])
        occ.synchronize()
        vols = [t for _, t in gmsh.model.getEntities(3)]
        low = [v for v in vols if gmsh.model.getBoundingBox(3, v)[5] < 0.003]
        gmsh.model.setPhysicalName(3, gmsh.model.addPhysicalGroup(3, low), "layer")
        base = [t for _, t in gmsh.model.getEntities(2) if gmsh.model.getBoundingBox(2, t)[3] < 1e-6]
        gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, base), "base")
        gmsh.model.addPhysicalGroup(3, vols)  # unnamed: group_3_<tag>
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.002)
        gmsh.model.mesh.generate(3)
        n_tets = sum(len(n) // 4 for n in gmsh.model.mesh.getElements(3, -1)[2])
        gmsh.write(str(path))
    finally:
        gmsh.finalize()
    f = SoftMeshFile(path=str(path))
    mesh = f.mesh(SETTINGS)
    assert mesh.n_tets == n_tets  # kept, not remeshed
    volumes = f.volume_regions(mesh)
    assert _tet_volumes(mesh)[volumes["layer"]].sum() == pytest.approx(0.02 * 0.01 * 0.002, rel=1e-9)
    assert len(volumes["body"]) == n_tets and any(k.startswith("group_3_") for k in volumes)
    base = f.regions(mesh)["base"]
    assert np.all(mesh.points[base, 0] < 1e-12)
    # outward: the base face's normals point to -x
    tri = mesh.points[base]
    assert np.all(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 0] < 0)
    with pytest.raises(ConfigError, match="mesh is used as it is"):
        from softsim.meshing import load_mesh_file
        load_mesh_file(path, 0.004, remesh=True)


def test_placement_and_errors(tmp_path):
    SoftPneuNet(**FINGER).export(tmp_path / "finger.brep")
    f = SoftMeshFile(path=str(tmp_path / "finger.brep"), rotate=(0, -90, 0), center="base",
                     translate=(0, 0, 0.001))
    mesh = f.mesh(MeshSettings(size=0.003, method="gmsh"))
    lo, hi = mesh.points.min(axis=0), mesh.points.max(axis=0)
    # finger along +x rotated -90 deg about y points up +z; base centre at (0, 0, 1 mm)
    np.testing.assert_allclose(hi - lo, [0.016, 0.016, SoftPneuNet(**FINGER).length], atol=1e-12)
    np.testing.assert_allclose([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]], [0, 0, 0.001], atol=1e-12)
    empty = SoftMeshFile(path=str(tmp_path / "finger.brep"), box_regions={"x": [[1, 1, 1], [2, 2, 2]]})
    with pytest.raises(ConfigError, match="no boundary triangle"):
        empty.regions(empty.mesh(SETTINGS))
    (tmp_path / "x.ply").write_text("")
    with pytest.raises(ConfigError, match="unsupported mesh file type"):
        SoftMeshFile(path=str(tmp_path / "x.ply")).mesh(SETTINGS)
    with pytest.raises(ConfigError, match="not found"):
        SoftMeshFile(path=str(tmp_path / "missing.step")).mesh(SETTINGS)


def test_pneunet_from_its_step_file(tmp_path):
    """The shipped STEP scenario (path relative to the config) is the parametric finger: same volume,
    cavity, layer and bonded base."""
    bodies = []
    for name in ("pneunet.yaml", "pneunet_from_step.yaml"):
        sc = InflationScenario(validate(InflationConfig, load_yaml(ACTUATION / name)), ACTUATION)
        bodies.append(sc.build_scene().soft_bodies[0])
    (a, b) = bodies
    assert b.mesh.volume() == pytest.approx(a.mesh.volume(), rel=1e-9)
    for body in bodies:
        assert len(open_edges(body.regions["cavity"])) == 0
    assert enclosed_volume(b.mesh.points, b.regions["cavity"]) == pytest.approx(
        enclosed_volume(a.mesh.points, a.regions["cavity"]), rel=1e-9)
    assert _tet_volumes(b.mesh)[b.zones[0].elements].sum() == pytest.approx(
        _tet_volumes(a.mesh)[a.zones[0].elements].sum(), rel=1e-9)
    assert len(b.pinned) == len(a.pinned)
    # the CLI writes the same CAD
    assert main(["export", str(ACTUATION / "pneunet.yaml"), "--cad", str(tmp_path / "f.step")]) == 0
    f = SoftMeshFile(path=str(tmp_path / "f.step"))
    assert f.mesh(SETTINGS).volume() == pytest.approx(a.mesh.volume(), rel=1e-9)
