"""Rod (BeamAdapter) cantilever against the Euler-Bernoulli solution; needs SOFA, skipped without it."""

import pytest

from softsim.backends.sofa.rod import RodCantileverConfig, euler_bernoulli_tip, rod_cantilever_tip
from softsim.errors import SoftSimError

BASE = dict(length=0.1, width=0.01, height=0.01, youngs=1e6, tip_force=0.01)


def _tip_z(**over) -> float:
    cfg = RodCantileverConfig(**{**BASE, **over})
    try:
        return float(rod_cantilever_tip(cfg)[2])
    except SoftSimError as exc:
        pytest.skip(str(exc))


def test_cantilever_matches_euler_bernoulli():
    cfg = RodCantileverConfig(**BASE)
    assert _tip_z() == pytest.approx(euler_bernoulli_tip(cfg), rel=0.01)


@pytest.mark.parametrize("over", [dict(length=0.2, tip_force=0.00125), dict(youngs=2e6), dict(height=0.015)])
def test_deflection_follows_length_modulus_and_section(over):
    """L^3, 1/E and 1/h^3 together (force reduced for the long rod to stay in the linear range)."""
    cfg = RodCantileverConfig(**{**BASE, **over})
    assert _tip_z(**over) == pytest.approx(euler_bernoulli_tip(cfg), rel=0.01)


def test_result_is_independent_of_the_number_of_elements():
    # a mutation that makes the stiffness depend on the element count would fail here
    assert _tip_z(segments=4) == pytest.approx(_tip_z(segments=20), rel=0.01)
