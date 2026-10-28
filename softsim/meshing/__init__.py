"""Tetrahedral meshing of soft-body geometry."""

from softsim.meshing.tet import (
    TetMesh,
    gmsh_box,
    gmsh_cylinder,
    structured_box,
    structured_cylinder,
)

__all__ = ["TetMesh", "gmsh_box", "gmsh_cylinder", "structured_box", "structured_cylinder"]
