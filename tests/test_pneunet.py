"""PneuNet geometry: exact volumes, one closed cavity, the conforming layer, the scenario."""

from pathlib import Path

import numpy as np
import pytest

from softsim.actuation import enclosed_volume, open_edges
from softsim.config.loader import apply_overrides, load_yaml
from softsim.config.schema import validate
from softsim.errors import ConfigError
from softsim.geometry import MeshSettings, SoftPneuNet
from softsim.scenarios.inflation import InflationConfig, InflationScenario

pytest.importorskip("gmsh")

FINGER = Path(__file__).parents[1] / "scenarios" / "actuation" / "pneunet.yaml"
DIMS = dict(chambers=3, chamber_length=0.006, wall=0.002, end_wall=0.004, width=0.016, side_wall=0.002,
            height=0.016, base=0.005, top_wall=0.002, channel_width=0.004, channel_height=0.002, layer=0.0015)


def test_pneunet_geometry_is_exact():
    g = SoftPneuNet(**DIMS)
    assert g.length == pytest.approx(0.004 * 2 + 3 * 0.006 + 2 * 0.002)
    mesh = g.mesh(MeshSettings(size=0.002, method="gmsh"))
    assert mesh.volume() == pytest.approx(g.volume(), rel=1e-9)  # flat faces: meshed exactly
    regions = g.regions(mesh)
    cavity = regions["cavity"]
    assert len(open_edges(cavity)) == 0  # chambers and channel form one closed cavity
    chambers = 3 * 0.006 * 0.012 * 0.009
    channel = 2 * 0.002 * 0.004 * 0.002  # through the walls between chambers
    assert enclosed_volume(mesh.points, cavity) == pytest.approx(chambers + channel, rel=1e-9)
    assert sum(len(t) for t in regions.values()) == len(mesh.surface_triangles())
    layer = g.volume_regions(mesh)["layer"]
    x = mesh.points[mesh.tets[layer]]
    vol = np.abs(np.linalg.det(x[:, 1:] - x[:, :1])).sum() / 6
    assert vol == pytest.approx(g.length * 0.016 * 0.0015, rel=1e-9)  # the mesh conforms to z = layer
    with pytest.raises(ValueError, match="layer < base"):
        SoftPneuNet(**{**DIMS, "layer": 0.006})


def test_pneunet_scenario():
    raw = load_yaml(FINGER)
    sc = InflationScenario(validate(InflationConfig, raw), FINGER.parent)
    body = sc.build_scene().soft_bodies[0]
    assert body.zones[0].material.elastic.youngs_modulus == 1e6 and len(body.pinned) > 0
    m = sc.metrics(type("S", (), {"time": 0.0, "soft": {"finger": type("V", (), {"vertices": body.mesh.points})},
                                  "controls": {"cavity": 0.0}})())
    assert m["bend_angle"] == pytest.approx(0.0, abs=1e-12) and m["tip_x"] == pytest.approx(0.054)
    bad = apply_overrides(raw, ["body.zones[0].region=lid"])
    with pytest.raises(ConfigError, match="no volume region 'lid'"):
        InflationScenario(validate(InflationConfig, bad), FINGER.parent).build_scene()
