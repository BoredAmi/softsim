"""The drop page's solver (tools/drop_solver.js), run under Node as the page runs it: stress in closed form, relaxation,
free flight, and the filmed drops. docs/drop.md."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="needs Node")

CUBE = """
const {DropSim, declaredCard, scaledMesh, cylinderMass, objectMass} = require(%s);
const n = 3, pts = [], tets = [], id = (i, j, k) => (i * (n + 1) + j) * (n + 1) + k;
for (let i = 0; i <= n; i++) for (let j = 0; j <= n; j++) for (let k = 0; k <= n; k++) pts.push(i / n, j / n, k / n);
const K = [[0, 3, 1, 7], [0, 2, 3, 7], [0, 6, 2, 7], [0, 4, 6, 7], [0, 5, 4, 7], [0, 1, 5, 7]];
for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) for (let k = 0; k < n; k++) {
  const c = [0, 1, 2, 3, 4, 5, 6, 7].map(b => id(i + (b >> 2 & 1), j + (b >> 1 & 1), k + (b & 1)));
  for (const t of K) tets.push(...t.map(q => c[q]));
}
function stretched(card, [a, b, c]) {
  const s = new DropSim({points: pts, tets}, {blend: 0, bulkViscosity: 0, density: 1000, ...card}, {shape: "none"});
  for (let i = 0; i < s.n; i++) { s.x[3 * i] *= a; s.x[3 * i + 1] *= b; s.x[3 * i + 2] *= c; }
  return s;
}
function traction(s) {  // first Piola-Kirchhoff xx from the forces on the face x = 1 (the internal force opposes it)
  s.f.fill(0); s.internal();
  let fx = 0;
  for (let i = 0; i < s.n; i++) if (pts[3 * i] === 1) fx += s.f[3 * i];
  return -fx;
}
""" % json.dumps(str(ROOT / "tools" / "drop_solver.js"))


def node(script: str):
    return json.loads(subprocess.run(["node", "-e", CUBE + script], capture_output=True, text=True, check=True).stdout)


@pytest.mark.parametrize("nu, stretch", [(0.47, (0.8, 1.0, 0.8)), (0.4, (1.1, 0.9, 1.0)), (0.45, (1.3, 0.9, 0.85))])
def test_homogeneous_stress_is_the_split_neo_hookean_law(nu, stretch):
    r = node(f"""const s = stretched({{E: 1e5, nu: {nu}}}, {list(stretch)});
    const P = traction(s); let inner = 0, net = [0, 0, 0];
    for (let i = 0; i < s.n; i++) {{
      for (let k = 0; k < 3; k++) net[k] += s.f[3 * i + k];
      if ([0, 1, 2].every(k => pts[3 * i + k] > 0 && pts[3 * i + k] < 1)) inner = Math.max(inner, Math.hypot(s.f[3 * i], s.f[3 * i + 1], s.f[3 * i + 2]));
    }}
    console.log(JSON.stringify({{P, inner, net, mu: s.mu, kappa: s.kappa}}));""")
    a, b, c = stretch
    j, i1 = a * b * c, a * a + b * b + c * c
    expected = r["mu"] * j ** (-2 / 3) * (a - i1 / 3 / a) + r["kappa"] * (j - 1) * b * c
    assert r["P"] == pytest.approx(expected, rel=1e-9)
    assert r["inner"] < 1e-8 and np.abs(r["net"]).max() < 1e-8  # equilibrium inside, no net force


def test_a_held_stretch_relaxes_as_the_prony_term_says():
    """All of the isochoric stress at once, then down to (1 - g) of it. (It once did the reverse and started 3x soft.)"""
    r = node("""const s = stretched({E: 1e5, nu: 0.3, g: 0.7, tau: 1}, [1.2, 1 / Math.sqrt(1.2), 1 / Math.sqrt(1.2)]);
    s.tau = 1; s.decay = Math.exp(-0.01); s.alpha = (1 - s.decay) / 0.01;
    const out = [traction(s)]; for (let i = 0; i < 500; i++) out.push(traction(s)); console.log(JSON.stringify(out));""")
    p = np.array(r)
    t = 0.01 * np.arange(len(p))
    assert p / p[0] == pytest.approx(0.3 + 0.7 * np.exp(-t), abs=2e-3)


def test_object_in_free_flight_follows_the_parabola_and_keeps_its_spin():
    r = node("""const s = new DropSim({points: pts.map(v => v * 0.01), tets}, {E: 1e5, nu: 0.3, density: 1000},
      {shape: "disc", mass: 0.05, radius: 0.01, height: 0.01, position: [0, 0, 1], velocity: [0, 0, 1], spin: 2});
    while (s.t < 0.05) s.step();
    console.log(JSON.stringify({t: s.t, dt: s.dt, z: s.object.p[2], w: s.object.w}));""")
    assert r["z"] == pytest.approx(1 + r["t"] - 0.5 * 9.81 * r["t"] ** 2, abs=9.81 * r["t"] * r["dt"])
    assert r["w"][1] == pytest.approx(2, rel=1e-6)


@pytest.fixture(scope="module")
def cells():
    from check_drop_solver import mesh
    return mesh()


@pytest.mark.skipif(not (ROOT / "data/processed/czopek2026_ecoflex_drop/w020.yaml").exists(), reason="needs the processed recordings")
@pytest.mark.parametrize("case, measured", [("w020", 3.74), ("w050", 4.31)])
def test_filmed_drops_peak_within_15_percent_with_nothing_fitted(cells, case, measured):
    from check_drop_solver import PHYSICAL, drop
    assert drop(PHYSICAL, case, cells).indentation.max() * 1e3 == pytest.approx(measured, rel=0.15)


@pytest.mark.parametrize("obj", [dict(shape="disc", mass=0.5, radius=0.0215, height=0.038), dict(shape="sphere", mass=0.0642, radius=0.0125)])
def test_heaviest_objects_survive_the_filmed_height(cells, obj):
    from check_drop_solver import HEIGHT, PHYSICAL, SPECIMEN_MASS, run_js
    low = obj.get("height", 2 * obj["radius"]) / 2
    job = dict(mesh=cells, material=PHYSICAL | dict(mass=SPECIMEN_MASS), friction=dict(object=5, table=5), duration=0.012, sample=1e-3,
               object=obj | dict(position=[0, 0, HEIGHT + low + 1e-4], velocity=[0, 0, -2.73]))
    r = run_js(job)
    assert not r["failed"] and min(r["z"]) > 0


@pytest.mark.parametrize("by, value", [("E", 143.3e3), ("G", 47.8e3), ("M100", 83.6e3)])
def test_every_way_of_declaring_stiffness_gives_e_g_k_of_one_solid(by, value):
    nu = 0.4995
    r = node(f"console.log(JSON.stringify(declaredCard({{by: '{by}', value: {value}, nu: {nu}}})))")
    shear = {"E": value / (2 * (1 + nu)), "G": value, "M100": value / 1.75}[by]
    assert r["G"] == pytest.approx(shear)
    assert r["E"] == pytest.approx(2 * shear * (1 + nu))
    assert 1 / r["E"] == pytest.approx(1 / (3 * r["G"]) + 1 / (9 * r["K"]))  # the identity for E, G and K


def test_a_declared_100_percent_modulus_is_the_stress_at_twice_the_length():
    """Stretch the solver's solid to twice its length at constant volume: the engineering stress, the pressure taken
    out by the zero lateral stress of a uniaxial test, is the declared modulus."""
    r = node("""const l = 1 / Math.sqrt(2), s = stretched(declaredCard({by: "M100", value: 1e5, nu: 0.4995}), [2, l, l]);
    const px = traction(s); let py = 0;
    for (let i = 0; i < s.n; i++) if (pts[3 * i + 1] === 1) py -= s.f[3 * i + 1];
    console.log(JSON.stringify(px - py * l / 2));""")
    assert r == pytest.approx(1e5, rel=1e-6)


def test_a_stretched_mesh_has_the_declared_size_and_masses_are_closed_form():
    r = node("""const m = scaledMesh({points: [1, 0, 2, 0, 3, 4], tets: []}, 2, 0.5);
    console.log(JSON.stringify({p: m.points, cyl: cylinderMass(1000, 0.02, 0.01), disc: objectMass({shape: "disc", radius: 0.02, height: 0.01}, 1000),
                                ball: objectMass({shape: "sphere", radius: 0.01}, 1000)}));""")
    assert r["p"] == [2, 0, 1, 0, 6, 2]
    assert r["cyl"] == r["disc"] == pytest.approx(1000 * np.pi * 0.02**2 * 0.01)
    assert r["ball"] == pytest.approx(1000 * 4 / 3 * np.pi * 0.01**3)


def _impact(cells, size=1.0, stiffness=1.0):
    """Peak indentation [m] and its time [s] of a 50 g disc, weightless, on the cylinder ``size`` times as large and
    ``stiffness`` times as stiff. Same density; object scaled with the body; speed and relaxation time as similarity says."""
    from check_drop_solver import HEIGHT, PHYSICAL, SPECIMEN_MASS, run_js
    rate = stiffness**0.5
    radius, height, gap = 0.0135 * size, 0.0099 * size, 1e-4 * size
    job = dict(mesh=dict(cells, points=[v * size for v in cells["points"]]), friction=dict(object=5, table=5), options=dict(gravity=0), sample=2e-6 * size / rate,
               material=PHYSICAL | dict(E=PHYSICAL["E"] * stiffness, tau=PHYSICAL["tau"] * size / rate, mass=SPECIMEN_MASS * size**3),
               object=dict(shape="disc", mass=0.05 * size**3, radius=radius, height=height, position=[0, 0, HEIGHT * size + height / 2 + gap],
                           velocity=[0, 0, -2.73 * rate]), duration=0.006 * size / rate)
    r = run_js(job)
    assert not r["failed"]
    ind = HEIGHT * size - (np.array(r["z"]) - height / 2)
    return ind.max(), r["t"][int(ind.argmax())]


def test_a_larger_body_and_object_scale_the_impact_exactly(cells):
    """Everything twice the size, at the same speed and density: a similar solution, twice the indentation and time."""
    peak, when = _impact(cells)
    assert _impact(cells, size=2.0) == pytest.approx((2 * peak, 2 * when), rel=2e-3)


def test_a_stiffer_body_scales_the_impact_exactly(cells):
    """Four times as stiff, twice the speed, half the relaxation time: the same indentation in half the time."""
    peak, when = _impact(cells)
    assert _impact(cells, stiffness=4.0) == pytest.approx((peak, when / 2), rel=2e-3)
