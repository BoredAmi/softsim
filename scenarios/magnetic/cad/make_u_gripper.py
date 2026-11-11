"""Writes u_gripper.step: the U of scenarios/magnetic/hard_magnetic_gripper.yaml, in millimetres. A 28 x 4 x 4 mm palm
bar (z 0-4) with two 4 x 4 x 40 mm arms standing on its ends (z 4-44), the inner faces 20 mm apart (x = -10 and 10).
    python scenarios/magnetic/cad/make_u_gripper.py"""
from pathlib import Path

import gmsh

gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
occ = gmsh.model.occ
palm = (3, occ.addBox(-14, -2, 0, 28, 4, 4))
arms = [(3, occ.addBox(x0, -2, 4, 4, 4, 40)) for x0 in (-14, 10)]
occ.fuse([palm], arms)
occ.synchronize()
gmsh.write(str(Path(__file__).with_name("u_gripper.step")))
gmsh.finalize()
