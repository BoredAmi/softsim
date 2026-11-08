"""Break the soft-magnetic model on purpose, one way at a time, and check tests/test_magnetic.py fails each time.

    python tools/mutate_magnetic.py        # about a minute; prints CAUGHT or SURVIVED per mutation, exits 1 if any survive

Works on a copy of softsim/, tests/ and pyproject.toml in a temporary folder, so the tree is never edited. Every mutation is
a plausible slip: a material axis that doesn't follow the deformation, the wrong tensor for the normal, a missing term or
factor, the field read in the wrong place, and the solver losing the term's gradient or Hessian.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAG, SOLID = "softsim/fem/magnetic.py", "softsim/fem/solid3d.py"
MUTATIONS = [
    ("the fibre does not follow F", MAG, "a, n = f @ jnp.asarray(material.along),", "a, n = jnp.asarray(material.along),"),
    ("the normal follows F, not F^-T", MAG, "jnp.linalg.inv(f).T @ jnp.asarray(material.normal)", "f @ jnp.asarray(material.normal)"),
    ("across loses the normal term", MAG, "- ha ** 2 - hn ** 2", "- ha ** 2"),
    ("energy without the half", MAG, "return -0.5 * MU0", "return -1.0 * MU0"),
    ("dipole: 2 r instead of 3 r", MAG, "(3.0 * r * jnp.dot(m, r)", "(2.0 * r * jnp.dot(m, r)"),
    ("field read at the rest position", MAG, "n @ (x0 + ue)", "n @ x0"),
    ("chi_normal used for chi_across", MAG, "material.chi_across * across", "material.chi_normal * across"),
    ("solver drops the magnetic gradient", SOLID, "grad += np.bincount(self._edofs.ravel(), weights=gm.ravel(), minlength=self.ndof)", "pass"),
    ("solver drops the magnetic Hessian", SOLID, "hess += pat.scatter(self._epos, hm.reshape(-1))", "pass"),
]


def main() -> None:
    survived = 0
    with tempfile.TemporaryDirectory(prefix="softsim_mutate_") as tmp:
        copy = Path(tmp)
        for name in ("softsim", "tests"):
            shutil.copytree(ROOT / name, copy / name, ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(ROOT / "pyproject.toml", copy)
        for name, rel, before, after in MUTATIONS:
            path = copy / rel
            original = path.read_text()
            if original.count(before) != 1:
                raise SystemExit(f"{name}: the line to mutate is not in {rel} exactly once; update this tool")
            path.write_text(original.replace(before, after))
            try:  # the copy comes first on the path: pytest runs from it
                r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-x", "-p", "no:cacheprovider", "tests/test_magnetic.py"],
                                   cwd=copy, capture_output=True, text=True)
            finally:
                path.write_text(original)
            verdict = {0: "SURVIVED", 1: "CAUGHT"}.get(r.returncode, f"ERROR {r.returncode}")
            survived += r.returncode == 0
            print(f"{verdict:9} {name}: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-200:]}", flush=True)
    sys.exit(1 if survived else 0)


if __name__ == "__main__":
    main()
