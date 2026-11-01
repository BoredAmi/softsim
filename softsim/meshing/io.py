"""Soft bodies from files: CAD (STEP, IGES, BREP), surface meshes (STL, OBJ), volume meshes (gmsh
``.msh``, VTK, Medit ``.mesh``) and gmsh scripts.

Everything goes through gmsh:
    * CAD (``.step``/``.stp``/``.iges``/``.igs``/``.brep``) is meshed at the requested size;
      STEP and IGES carry their units and are converted to metres. Several solids are fragmented
      into one conforming body (bonded where they touch; e.g. a layer of another material).
    * STL or OBJ (a closed surface, possibly with closed inner shells: cavities) is turned into
      a volume and meshed; with ``remesh`` its surface is remeshed at the requested size,
      otherwise its triangles are kept as the boundary. OBJ polygons are split into triangles.
    * ``.msh``, ``.vtk`` and ``.mesh`` are used as they are (linear tetrahedra); ``.geo``
      scripts are meshed if they hold no mesh.

Named physical groups become regions: 2D groups name boundary triangles (``base``, ``cavity``),
3D groups name element sets (volume regions, e.g. a material zone). A file with several volumes
and no 3D groups (solids of a STEP, references of a Medit mesh) names them ``volume_<tag>``. Files
without units (all but STEP and IGES) are scaled by ``scale`` (e.g. 1e-3 for millimetres).
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from softsim.errors import ConfigError
from softsim.meshing.tet import TetMesh, _orient

CAD = {".step", ".stp", ".iges", ".igs", ".brep"}
WITH_UNITS = {".step", ".stp", ".iges", ".igs"}
SURFACES = {".stl", ".obj"}
MESHES = {".msh", ".vtk", ".mesh"}
FORMATS = CAD | SURFACES | MESHES | {".geo"}


@dataclass(frozen=True)
class MeshFile:
    """A meshed file: ``mesh`` (metres), named boundary triangle sets ``surfaces`` (index rows of
    ``mesh.points``, unoriented) and named element sets ``volumes`` (indices into ``mesh.tets``)."""

    mesh: TetMesh
    surfaces: dict[str, np.ndarray]
    volumes: dict[str, np.ndarray]


def load_mesh_file(path: str | Path, size: float, scale: float = 1.0, remesh: bool = False,
                   feature_angle: float = 40.0) -> MeshFile:
    """Tetrahedral mesh of the file at ``path`` (see module docstring); ``size`` is the target edge
    length in metres, ``feature_angle`` [deg] the STL edge angle kept as a sharp feature."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise ConfigError(f"{path}: unsupported mesh file type {suffix!r}; use one of {sorted(FORMATS)}")
    if not path.is_file():
        raise ConfigError(f"mesh file not found: {path}")
    try:
        import gmsh
    except ImportError:
        raise ConfigError("mesh files need the gmsh package (pip install softsim[mesh])") from None
    if suffix in WITH_UNITS:
        scale = 1.0
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        if suffix in WITH_UNITS:
            gmsh.option.setString("Geometry.OCCTargetUnit", "M")
        gmsh.option.setNumber("Mesh.MeshSizeMax", size / scale)
        gmsh.option.setNumber("Mesh.MeshSizeMin", size / scale)
        if suffix in CAD:
            gmsh.model.occ.importShapes(str(path))
            gmsh.model.occ.synchronize()
            solids = gmsh.model.getEntities(3)
            if len(solids) > 1:  # touching solids share their faces: one conforming, bonded body
                gmsh.model.occ.fragment(solids[:1], solids[1:])
                gmsh.model.occ.synchronize()
            gmsh.model.mesh.generate(3)
        elif suffix in SURFACES:
            with tempfile.TemporaryDirectory() as tmp:
                gmsh.merge(str(path) if suffix == ".stl" else _obj_to_stl(path, Path(tmp) / "surface.stl"))
            _stl_volume(gmsh, feature_angle, remesh)
            gmsh.model.mesh.generate(3)
        else:
            gmsh.open(str(path))
            has_tets = any(len(t) for t in gmsh.model.mesh.getElements(3, -1)[2])
            if remesh and suffix in MESHES:
                raise ConfigError(f"{path}: a {suffix} mesh is used as it is (its volumes are discrete); remesh "
                                  "it in gmsh, or give the CAD or an STL")
            if not has_tets:
                gmsh.model.mesh.generate(3)
        return _extract(gmsh, path, scale)
    except Exception as exc:
        if isinstance(exc, ConfigError):
            raise
        raise ConfigError(f"{path}: gmsh could not mesh it: {exc}") from None
    finally:
        gmsh.finalize()


def _obj_to_stl(path: Path, out: Path) -> str:
    """Write the faces of a Wavefront OBJ as an ASCII STL (polygons fanned into triangles)."""
    vertices, triangles = [], []
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(c) for c in parts[1:4]])
        elif parts[0] == "f":
            idx = [int(p.split("/")[0]) for p in parts[1:]]
            idx = [i - 1 if i > 0 else len(vertices) + i for i in idx]  # 1-based, or negative from the end
            triangles += [(idx[0], idx[k], idx[k + 1]) for k in range(1, len(idx) - 1)]
    if not triangles:
        raise ConfigError(f"{path}: no faces")
    v, t = np.array(vertices), np.array(triangles)
    with out.open("w") as f:
        f.write("solid obj\n")
        for a, b, c in v[t]:
            n = np.cross(b - a, c - a)
            n = n / (np.linalg.norm(n) or 1.0)
            f.write(f"facet normal {n[0]:.9g} {n[1]:.9g} {n[2]:.9g}\n outer loop\n")
            for p in (a, b, c):
                f.write(f"  vertex {p[0]:.17g} {p[1]:.17g} {p[2]:.17g}\n")
            f.write(" endloop\nendfacet\n")
        f.write("endsolid obj\n")
    return str(out)


def _stl_volume(gmsh, feature_angle: float, remesh: bool) -> None:
    """Turn a merged STL surface into geometry with one volume: the outer shell around the others
    (holes, i.e. cavities)."""
    gmsh.model.mesh.classifySurfaces(np.radians(feature_angle), True, remesh, np.pi)
    gmsh.model.mesh.createGeometry()
    surfaces = [t for _, t in gmsh.model.getEntities(2)]
    # connected shells: surfaces sharing a boundary curve
    parent = {s: s for s in surfaces}

    def root(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    by_curve: dict[int, int] = {}
    for s in surfaces:
        for _, c in gmsh.model.getBoundary([(2, s)], oriented=False):
            if abs(c) in by_curve:
                parent[root(s)] = root(by_curve[abs(c)])
            else:
                by_curve[abs(c)] = s
    shells: dict[int, list[int]] = {}
    for s in surfaces:
        shells.setdefault(root(s), []).append(s)

    def extent(shell):
        boxes = np.array([gmsh.model.getBoundingBox(2, s) for s in shell])
        return float(np.prod(boxes[:, 3:].max(axis=0) - boxes[:, :3].min(axis=0)))

    ordered = sorted(shells.values(), key=extent, reverse=True)  # the outer shell encloses the rest
    loops = [gmsh.model.geo.addSurfaceLoop(shell) for shell in ordered]
    gmsh.model.geo.addVolume(loops)
    gmsh.model.geo.synchronize()


def _extract(gmsh, path: Path, scale: float) -> MeshFile:
    tags, xyz, _ = gmsh.model.mesh.getNodes()
    index = np.full(int(tags.max()) + 1, -1)
    index[tags.astype(int)] = np.arange(len(tags))
    blocks, owner = [], []
    for _, entity in gmsh.model.getEntities(3):
        types, _, nodes = gmsh.model.mesh.getElements(3, entity)
        for etype, n in zip(types, nodes):
            if etype != 4:  # 4-node tetrahedra only
                raise ConfigError(f"{path}: volume elements of gmsh type {etype}; only linear tetrahedra are read "
                                  "(export first order, or let softsim remesh it)")
            blocks.append(index[n.astype(int)].reshape(-1, 4))
            owner.append(np.full(len(blocks[-1]), entity))
    if not blocks:
        raise ConfigError(f"{path}: no tetrahedra (is the surface closed?)")
    tets, owner = np.concatenate(blocks), np.concatenate(owner)
    used = np.unique(tets)
    remap = np.full(len(tags), -1)
    remap[used] = np.arange(len(used))
    points = xyz.reshape(-1, 3)[used] * scale
    tets = remap[tets]

    surfaces, volumes = {}, {}
    groups = gmsh.model.getPhysicalGroups()
    solids = np.unique(owner)
    if len(solids) > 1 and not any(dim == 3 for dim, _ in groups):
        volumes = {f"volume_{e}": np.nonzero(owner == e)[0] for e in solids}
    for dim, group in groups:
        name = gmsh.model.getPhysicalName(dim, group) or f"group_{dim}_{group}"
        entities = gmsh.model.getEntitiesForPhysicalGroup(dim, group)
        if dim == 3:
            volumes[name] = np.nonzero(np.isin(owner, entities))[0]
        elif dim == 2:
            tris = []
            for e in entities:
                types, _, nodes = gmsh.model.mesh.getElements(2, e)
                tris += [remap[index[n.astype(int)]].reshape(-1, 3) for t, n in zip(types, nodes) if t == 2]
            if tris:
                surfaces[name] = np.concatenate(tris)
    return MeshFile(TetMesh(points, _orient(points, tets)), surfaces, volumes)


def surface_cavities(mesh: TetMesh) -> list[np.ndarray]:
    """Closed inner shells of the boundary (cavities), largest first: the boundary's connected
    components except the one with the largest bounding box (the outer surface)."""
    tris = mesh.surface_triangles()
    parent = np.arange(mesh.n_points)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c in tris:
        ra, rb, rc = root(a), root(b), root(c)
        parent[rb] = ra
        parent[root(rc)] = ra
    labels = np.array([root(t[0]) for t in tris])
    shells = [tris[labels == lab] for lab in np.unique(labels)]
    extent = [np.prod(np.ptp(mesh.points[np.unique(s)], axis=0)) for s in shells]
    shells = [s for _, s in sorted(zip(extent, shells), key=lambda p: -p[0])]
    return shells[1:]
