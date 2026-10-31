"""Benchmark the 3D reference solver on this machine: where a Newton iteration's time goes.

    python tools/bench_fem.py                 # meshes 8, 5 mm; ~3 min on the laptop
    python tools/bench_fem.py --mesh 0.003    # finer
    JAX_PLATFORMS=cpu python tools/bench_fem.py   # force the CPU on a GPU machine

Per mesh size it times the parts of one Newton iteration of the drop-test
specimen (Ecoflex cylinder, Neo-Hookean + Prony): the element tangents (JAX: CPU
or GPU), the element energy (line search), the sparse assembly and the linear
solve (PARDISO with the ``fast`` extra, else SuperLU). Then it runs a short 50 g
drop through the ``fem`` backend and reports the wall time per simulated step.
Compare the output of two machines to see what a GPU buys.
"""

from __future__ import annotations

import argparse
import platform
import time

import numpy as np

CARD = ("{elastic: {model: neo_hookean, youngs_modulus: 143 kPa, poisson_ratio: 0.4995}, "
        "damping: {model: prony, g: [0.69], tau: [2 ms]}, density: 1070}")


def _timed(fn, repeat: int = 3) -> float:
    fn()  # compile / warm up
    t = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t) / repeat


def _parts(mesh: float) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    from softsim.fem.materials import NeoHookeanIso, Prony
    from softsim.fem.solid3d import Problem3D, QuadraticTetMesh, RigidBody3D, Solid3DSolver
    from softsim.meshing import structured_cylinder

    lin = structured_cylinder(0.020, 0.017, mesh)  # the drop-test specimen
    m = QuadraticTetMesh.from_linear(lin.points, lin.tets)
    law = NeoHookeanIso(143e3 / 3.0)
    s = Solid3DSolver(Problem3D(m, law, 143e3 / (3 * (1 - 2 * 0.4995)), 1070.0, prony=Prony((0.69,), (2e-3,)),
                                rigid=(RigidBody3D.ground(),)), dynamic=True)
    ctx = s._context(1e-4)
    y = s.y + np.random.default_rng(0).normal(0, 1e-5, s.ndof)
    u = jnp.asarray(y[:3 * s.n].reshape(s.n, 3)[m.tets].reshape(-1, 30))
    out = {"tets": len(m.tets), "dofs": s.ndof}
    out["tangents"] = _timed(lambda: jax.block_until_ready(
        s._elem_grad_hess(u, s._dndx, s._wv, ctx["c"], ctx["b"])))
    out["energy"] = _timed(lambda: s.energy(y, ctx))
    out["grad_hess (tangents + assembly)"] = _timed(lambda: s._grad_hess(y, ctx))
    _, hess = s._grad_hess(y, ctx)
    g = np.random.default_rng(1).normal(size=len(s.free))
    out["solve"] = _timed(lambda: s._pattern.solve(hess, g))
    return out


def _drop(mesh: float, duration: float) -> tuple[float, int]:
    from softsim.runner import run_config

    base = ["backend=fem", f"specimen.material={CARD}", f"specimen.mesh.size={mesh}", "output.save_model=false"]
    run_config("scenarios/drop_test/cases/w050.yaml", [*base, "simulation.duration=0.0001"])  # compile
    t = time.perf_counter()
    r = run_config("scenarios/drop_test/cases/w050.yaml", [*base, f"simulation.duration={duration}"])
    return time.perf_counter() - t, len(r.timeseries) - 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mesh", type=float, action="append", help="mesh size [m] (repeatable; default 8 and 5 mm)")
    ap.add_argument("--duration", type=float, default=0.001, help="simulated drop time per mesh [s]")
    ap.add_argument("--no-drop", action="store_true", help="only time the parts")
    args = ap.parse_args()

    import jax

    from softsim.fem import jax_setup

    jax_setup()
    try:
        import pypardiso  # noqa: F401
        solver = "PARDISO"
    except ImportError:
        solver = "SuperLU (install the fast extra for PARDISO)"
    print(f"machine: {platform.processor() or platform.machine()}, JAX {jax.__version__} on "
          f"{', '.join(str(d) for d in jax.devices())}, linear solver: {solver}")
    for mesh in args.mesh or [0.008, 0.005]:
        p = _parts(mesh)
        print(f"\nmesh {1e3 * mesh:g} mm: {p.pop('tets')} tets, {p.pop('dofs')} dofs; per Newton iteration:")
        for name, sec in p.items():
            print(f"  {name:34s} {1e3 * sec:8.1f} ms")
        if not args.no_drop:
            wall, steps = _drop(mesh, args.duration)
            print(f"  50 g drop, {1e3 * args.duration:g} ms simulated: {wall:.1f} s wall, "
                  f"{wall / max(steps, 1):.2f} s per output sample (0.2 ms)")


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
