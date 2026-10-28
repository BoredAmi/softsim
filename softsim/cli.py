"""Command-line interface: ``softsim <command> ...`` (see ``softsim -h``)."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

from softsim.errors import SoftSimError


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="scenario YAML")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value (dotted key, YAML value); repeatable")
    p.add_argument("--backend", help="override the config's backend")


def _cmd_run(args: argparse.Namespace) -> int:
    from softsim.runner import Simulation

    sim = Simulation(args.config, args.overrides, backend=args.backend, strict=args.strict)
    for f in sim.findings:
        print(f, file=sys.stderr)
    last = [-1]

    def progress(frac: float) -> None:
        pct = int(frac * 100)
        if not args.quiet and pct // 10 != last[0]:
            last[0] = pct // 10
            print(f"  {pct:3d}%  t={sim.time * 1e3:8.2f} ms", file=sys.stderr)

    try:
        result = sim.run(duration=args.duration, record_vertices=args.vertices or None, progress=progress)
    finally:
        sim.close()
    name = sim.config.name or Path(args.config).stem
    out = Path(args.out) if args.out else Path(sim.config.output.directory) / (
        f"{name}_{_dt.datetime.now():%Y%m%d_%H%M%S}")
    result.save(out, sim.config.output.format)
    print(json.dumps({k: round(float(v), 6) if math.isfinite(v) else None for k, v in result.summary.items()},
                     indent=2))
    print(f"realtime factor {result.metadata['realtime_factor']:.4f}; results in {out}", file=sys.stderr)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    from softsim.backends import available_backends, get_backend
    from softsim.checks import check_scene
    from softsim.config.loader import apply_overrides, load_yaml
    from softsim.config.schema import validate
    from softsim.scenarios import get_scenario

    path = Path(args.config).resolve()
    raw = apply_overrides(load_yaml(path), args.overrides)
    if args.backend:
        raw["backend"] = args.backend
    scenario_cls = get_scenario(raw.get("scenario", ""))
    config = validate(scenario_cls.config_model, raw, where="config")
    scenario = scenario_cls(config, path.parent)
    scene = scenario.build_scene()
    print(f"scenario: {scenario.name}   requested backend: {config.backend}")
    for body in scene.soft_bodies:
        q = body.mesh.quality()
        print(f"  soft body {body.name}: {q['n_points']} vertices, {q['n_tets']} tets, mean edge "
              f"{q['mean_edge'] * 1e3:.2f} mm, min quality {q['min_quality']:.2f}, mass {body.total_mass * 1e3:.2f} g, "
              f"material {body.material.name} ({body.material.elastic.model})")
    print("estimates:", {k: float(f"{v:.4g}") for k, v in scenario.estimates().items()})
    findings = check_scene(scene, scenario.estimates())
    for f in findings:
        print(" ", f)
    print("backend support:")
    status = 0
    for name in sorted(available_backends()):
        try:
            missing = get_backend(name).unsupported(scene)
        except SoftSimError as exc:
            print(f"  {name}: unavailable ({exc})")
            continue
        print(f"  {name}: {'OK' if not missing else 'cannot simulate: ' + '; '.join(missing)}")
        if name == config.backend and missing:
            status = 1
    return 1 if any(f.level == "error" for f in findings) else status


def _cmd_sweep(args: argparse.Namespace) -> int:
    from softsim.sweep import run_sweep

    out = Path(args.out) if args.out else Path("runs") / f"sweep_{Path(args.sweep).stem}_{_dt.datetime.now():%Y%m%d_%H%M%S}"
    df = run_sweep(args.sweep, out, jobs=args.jobs, extra_overrides=args.overrides)
    failed = (df["error"] != "").sum()
    print(df.drop(columns=[c for c in ("traceback",) if c in df]).to_string(max_cols=12))
    print(f"{len(df)} runs, {failed} failed; results in {out}", file=sys.stderr)
    return 1 if failed else 0


def _cmd_export(args: argparse.Namespace) -> int:
    from softsim.runner import Simulation

    sim = Simulation(args.config, args.overrides, backend=args.backend)
    text = sim.backend.export_model()
    if text is None:
        print(f"backend {sim.backend.name} has no exportable model", file=sys.stderr)
        return 1
    if args.out:
        Path(args.out).write_text(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


def _cmd_materials(args: argparse.Namespace) -> int:
    from softsim.materials import list_presets, load_material

    for name in list_presets(args.path):
        m = load_material(name, extra=args.path)
        e = m.elastic.initial_youngs_modulus()
        print(f"{name:20s} {m.elastic.model:14s} E0={e / 1e3:7.1f} kPa  nu={m.elastic.initial_poisson_ratio():.3f}"
              f"  rho={m.density:6.0f}  {m.description}")
    return 0


def _cmd_scenarios(args: argparse.Namespace) -> int:
    from softsim.scenarios import available_scenarios, get_scenario

    for name, target in sorted(available_scenarios().items()):
        cls = get_scenario(name)
        print(f"{name}: {target}")
        for action, desc in cls.actions.items():
            print(f"    action {action}: {desc}")
    return 0


def _cmd_view(args: argparse.Namespace) -> int:
    from softsim.runner import Simulation

    sim = Simulation(args.config, args.overrides, backend="mujoco")
    import mujoco.viewer

    backend = sim.backend
    dt = sim.config.simulation.timestep
    with mujoco.viewer.launch_passive(backend.model, backend.data) as viewer:
        while viewer.is_running():
            if sim.time >= sim.config.simulation.duration:
                sim.reset()
            start = time.perf_counter()
            sim.step()
            viewer.sync()
            wait = sim.config.simulation.output_every * dt * args.slowdown - (time.perf_counter() - start)
            if wait > 0:
                time.sleep(wait)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="softsim", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a scenario and save results")
    _add_config_args(p)
    p.add_argument("--out", help="output directory (default: <output.directory>/<name>_<timestamp>)")
    p.add_argument("--duration", type=float, help="override duration [s]")
    p.add_argument("--vertices", action="store_true", help="also save soft-body vertex trajectories")
    p.add_argument("--strict", action="store_true", help="treat check warnings as errors")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("check", help="validate a config, print mesh stats, physics findings, backend support")
    _add_config_args(p)
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("sweep", help="run a parameter sweep file")
    p.add_argument("sweep")
    p.add_argument("--out")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("export", help="print/write the backend-native model (e.g. MJCF)")
    _add_config_args(p)
    p.add_argument("--out")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("materials", help="list material presets")
    p.add_argument("--path", action="append", default=[], help="extra preset directory")
    p.set_defaults(func=_cmd_materials)

    p = sub.add_parser("scenarios", help="list registered scenarios and their actions")
    p.set_defaults(func=_cmd_scenarios)

    p = sub.add_parser("view", help="interactive MuJoCo viewer (loops the scenario)")
    _add_config_args(p)
    p.add_argument("--slowdown", type=float, default=200.0, help="wall seconds per simulated second")
    p.set_defaults(func=_cmd_view)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SoftSimError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
