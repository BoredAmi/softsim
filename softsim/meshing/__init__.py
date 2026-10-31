"""Tetrahedral meshing of soft-body geometry."""

from softsim.meshing.tet import (
    TetMesh,
    gmsh_box,
    gmsh_closed_tube,
    gmsh_cylinder,
    gmsh_hollow_sphere,
    structured_box,
    structured_cylinder,
)

__all__ = ["TetMesh", "gmsh_box", "gmsh_closed_tube", "gmsh_cylinder", "gmsh_hollow_sphere", "structured_box",
           "structured_cylinder"]
