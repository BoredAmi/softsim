"""Break the soft-magnetic model on purpose, one way at a time, and check its tests fail each time.

    python tools/mutate_magnetic.py        # about 25 min; prints CAUGHT or SURVIVED per mutation, exits 1 if any survive

Works on a copy of softsim/, tests/, the magnetic scenarios, the rolling-torque tool and pyproject.toml in a temporary folder,
so the tree is never edited. Every mutation is a plausible slip: in the energy (a material axis that doesn't follow the
deformation, the wrong tensor for the normal, a missing term or factor, the field read in the wrong place), in a magnet's
field (an elliptic integral cut short, a wrong factor or sign, a ring's hole not reversed, the length for the half-length),
in the material (a demagnetising factor's term, the mixing rule, the axes' order), in the scene (the field at the step's
start, a body's turn not given to its axes, a solid magnet's size, turn or path) and in the solver (the term's gradient or
Hessian lost, the field's local expansion wrong, a chunk of elements misread).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAG, SOLID, FIELD = "softsim/fem/magnetic.py", "softsim/fem/solid3d.py", "softsim/magnetism.py"
MODELS, FEM = "softsim/materials/models.py", "softsim/backends/fem/backend.py"
TESTS = ["tests/test_magnetic.py", "tests/test_magnet_field.py", "tests/test_scenario_magnetic.py"]
MUTATIONS = [
    ("the fibre does not follow F", MAG, "a, n = f @ jnp.asarray(material.along),", "a, n = jnp.asarray(material.along),"),
    ("the normal follows F, not F^-T", MAG, "jnp.linalg.inv(f).T @ jnp.asarray(material.normal)", "f @ jnp.asarray(material.normal)"),
    ("across loses the normal term", MAG, "- ha ** 2 - hn ** 2", "- ha ** 2"),
    ("energy without the half", MAG, "return -0.5 * MU0", "return -1.0 * MU0"),
    ("dipole: 2 r instead of 3 r", MAG, "(3.0 * r * jnp.dot(m, r)", "(2.0 * r * jnp.dot(m, r)"),
    ("energy: field read at the rest position", MAG, "density(jnp.eye(3) + ue.T @ d, n @ (x0 + ue), q, material)",
     "density(jnp.eye(3) + ue.T @ d, n @ x0, q, material)"),
    ("expansion: offset from the rest position", MAG, "s = n @ (x0 + ue) - at", "s = n @ x0 - at"),
    ("chi_normal used for chi_across", MAG, "material.chi_across * across", "material.chi_normal * across"),
    ("solver drops the magnetic gradient", SOLID, "grad += np.bincount(self._edofs.ravel(), weights=gm.ravel(), minlength=self.ndof)", "pass"),
    ("solver drops the magnetic Hessian", SOLID, "hess += pat.scatter(self._epos, hm.reshape(-1))", "pass"),
    ("elliptic integral: 3 steps", MAG, "CEL_STEPS = 24", "CEL_STEPS = 3"),
    ("magnet: axial field without a / (a + rho)", MAG, "hz = scale * a / (a + rho) * hz", "hz = scale * hz"),
    ("magnet: both end faces the same sign", MAG, "((1.0, z + b), (-1.0, z - b))", "((1.0, z + b), (1.0, z - b))"),
    ("magnet: gamma for 1 in the axial integral", MAG, "cel(kc, gamma * gamma + 1e-30, 1.0, gamma)", "cel(kc, gamma * gamma + 1e-30, 1.0, 1.0)"),
    ("magnet: B0 = mu0 M / 2 pi", MAG, "scale = m / jnp.pi", "scale = m / (2.0 * jnp.pi)"),
    ("ring: the hole not reversed", FIELD, "(centre, axis, -m, inner, length, edge)", "(centre, axis, m, inner, length, edge)"),
    ("magnet: length for half-length", FIELD, "np.array([m, radius, 0.5 * length,", "np.array([m, radius, length,"),
    ("demagnetisation: a term's sign", MAG, "+ 2 * np.arctan(a * b / (c * r))", "- 2 * np.arctan(a * b / (c * r))"),
    ("mixing rule: dilute only", MODELS, "return 3.0 * volume_fraction / (1.0 - volume_fraction)", "return 3.0 * volume_fraction"),
    ("block factors in the wrong order", FEM, "np.roll(extents, 2 - k)", "np.roll(extents, k)"),
    ("field at the step's start", FEM, "parameters(self.solver.t + dt)", "parameters(self.solver.t)"),
    ("axes not turned with the body", FEM, "along=tuple(rot @ along), normal=tuple(rot @ normal)", "along=tuple(along), normal=tuple(normal)"),
    ("uniform schedule held, not ramped", FIELD, "np.interp(t, self.times, v[:, k])", "v[0, k]"),
    ("field expansion without its curvature", MAG, 'h + j @ s + 0.5 * jnp.einsum("ijk,j,k->i", k, s, s)', "h + j @ s"),
    # (J transposed is no mutation: outside a magnet curl H = 0, so J is symmetric; test_magnet_field checks that)
    ("field expansion with half its slope", MAG, "h + j @ s + 0.5", "h + 0.5 * j @ s + 0.5"),
    ("a chunk read at the wrong rest positions", SOLID, "grad_hess(u[s], x0[s], dndx[s], wv[s], q)",
     "grad_hess(u[s], x0[: s.stop - s.start], dndx[s], wv[s], q)"),
    ("solid magnet as big as its hole", FEM, 'Part("cylinder", (magnet.outer_radius,', 'Part("cylinder", (magnet.inner_radius,'),
    ("z turned to the axis with the wrong sign", FEM, "return np.eye(3) + k + k @ k / (1.0 + c)", "return np.eye(3) - k + k @ k / (1.0 + c)"),
    ("a moving solid magnet left where it was", FEM, 'mount="prescribed" if magnet.path is not None else "fixed"', 'mount="fixed"'),
    ("inside a magnet, B / mu0 taken for H", MAG, " - jnp.where((rho < a) & (jnp.abs(z) < b), m, 0.0)", ""),
    ("edge softening ignored", MAG, " + (0.5 * edge / a) ** 2)", ")"),
    ("a solid magnet rounded like other rigid bodies", FEM, "edge_radius=max(magnet.edge, 1e-5)", "edge_radius=EDGE_RADIUS"),
]


def main() -> None:
    survived = 0
    with tempfile.TemporaryDirectory(prefix="softsim_mutate_") as tmp:
        copy = Path(tmp)
        for name in ("softsim", "tests", "scenarios/magnetic"):
            shutil.copytree(ROOT / name, copy / name, ignore=shutil.ignore_patterns("__pycache__"))
        (copy / "tools").mkdir()
        shutil.copy(ROOT / "tools" / "mre_rolling_torque.py", copy / "tools")
        (copy / "datasets").mkdir()
        shutil.copy(ROOT / "datasets" / "bernat2024_mre_gripper.yaml", copy / "datasets")
        shutil.copy(ROOT / "pyproject.toml", copy)
        env = {**os.environ, "PYTHONPATH": str(copy)}
        where = subprocess.run([sys.executable, "-c", "import softsim; print(softsim.__file__)"], cwd=copy, env=env,
                               capture_output=True, text=True).stdout.strip()
        if not where.startswith(str(copy)):
            raise SystemExit(f"the copy is not what gets imported ({where}): a mutation would test the tree instead")
        base = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *TESTS], cwd=copy,
                              capture_output=True, text=True, env=env)
        if base.returncode != 0:  # a failure here would make every mutation look caught
            raise SystemExit("the unmutated copy fails its tests:\n" + base.stdout[-2000:])
        print(f"baseline  {base.stdout.strip().splitlines()[-1]}", flush=True)
        for name, rel, before, after in MUTATIONS:
            path = copy / rel
            original = path.read_text()
            if original.count(before) != 1:
                raise SystemExit(f"{name}: the line to mutate is not in {rel} exactly once; update this tool")
            path.write_text(original.replace(before, after))
            try:  # the copy comes first on the path: pytest runs from it, imports from it
                r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", *TESTS],
                                   cwd=copy, capture_output=True, text=True, env=env)
            finally:
                path.write_text(original)
            verdict = {0: "SURVIVED", 1: "CAUGHT"}.get(r.returncode, f"ERROR {r.returncode}")
            survived += r.returncode == 0
            print(f"{verdict:9} {name}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]}", flush=True)
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
