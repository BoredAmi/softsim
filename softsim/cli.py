"""Command-line interface: ``softsim <command> ...`` (see ``softsim -h``)."""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
import time
from pathlib import Path

from softsim.errors import ConfigError, SoftSimError


def _add_config_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("config", help="scenario YAML")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value (dotted key, YAML value); repeatable")
    p.add_argument("--backend", help="override the config's backend")


def _cmd_run(args: argparse.Namespace) -> int:
    from softsim.runner import Simulation

    overrides = [*args.overrides, *([f"output.checkpoint={Path(args.checkpoint).resolve()}"] if args.checkpoint else [])]
    sim = Simulation(args.config, overrides, backend=args.backend, strict=args.strict)
    for f in sim.findings:
        print(f, file=sys.stderr)
    last = [-1]

    def progress(frac: float) -> None:
        pct = int(frac * 100)
        if not args.quiet and pct // 10 != last[0]:
            last[0] = pct // 10
            print(f"  {pct:3d}%  t={sim.time * 1e3:8.2f} ms", file=sys.stderr)

    name = sim.config.name or Path(args.config).stem
    out = Path(args.out) if args.out else Path(sim.config.output.directory) / (
        f"{name}_{_dt.datetime.now():%Y%m%d_%H%M%S}")
    try:
        result = sim.run(duration=args.duration, record_vertices=args.vertices or None, progress=progress)
    except Exception as exc:
        partial = getattr(exc, "partial", None)
        if partial is not None:  # keep what was computed before the failure
            partial.metadata["incomplete"] = f"{type(exc).__name__}: {exc}"
            partial.save(out, sim.config.output.format)
            print(f"the run stopped at t = {float(partial.timeseries['t'].iloc[-1]) * 1e3:.2f} ms; the samples up to "
                  f"there are in {out} (summary.yaml: metadata.incomplete)", file=sys.stderr)
        raise
    finally:
        sim.close()
    result.save(out, sim.config.output.format)
    print(json.dumps({k: float(f"{float(v):.6g}") if math.isfinite(v) else None for k, v in result.summary.items()},
                     indent=2))  # 6 significant digits: volumes [m^3] and forces alike
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
        if body.regions:
            print("    regions: " + ", ".join(f"{k} ({len(v)} triangles)" for k, v in body.regions.items()))
        for zone in body.zones:
            print(f"    zone {zone.name}: {len(zone.elements)} tets, {zone.material.name} "
                  f"({zone.material.elastic.model}, E0 {zone.material.elastic.initial_youngs_modulus() / 1e3:.4g} kPa)")
    print("estimates:", {k: float(f"{v:.4g}") for k, v in scenario.estimates().items()})
    try:
        backend_cls = get_backend(config.backend)
        locking_free, implicit_actuation = backend_cls.locking_free, backend_cls.implicit_actuation
        implicit_prony, single_friction = backend_cls.implicit_prony, backend_cls.single_friction
    except ConfigError:  # unknown or uninstalled backend: reported under "backend support"
        locking_free = implicit_actuation = implicit_prony = single_friction = False
    findings = check_scene(scene, scenario.estimates(), locking_free=locking_free,
                           implicit_actuation=implicit_actuation, implicit_prony=implicit_prony,
                           single_friction=single_friction)
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
    df = run_sweep(args.sweep, out, jobs=args.jobs, extra_overrides=args.overrides, sample_n=args.sample,
                   sample_seed=args.seed)
    failed = (df["error"] != "").sum()
    print(df.drop(columns=[c for c in ("traceback",) if c in df]).to_string(max_cols=12))
    print(f"{len(df)} runs, {failed} failed; results in {out}", file=sys.stderr)
    return 1 if failed else 0


def _cmd_export(args: argparse.Namespace) -> int:
    from softsim.runner import Simulation

    if args.cad:
        return _export_cad(args)
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


def _export_cad(args: argparse.Namespace) -> int:
    """Write the config's soft body geometry as CAD, e.g. to start your own design from a parametric one."""
    from softsim.config.loader import apply_overrides, load_yaml
    from softsim.config.schema import StrictModel, validate
    from softsim.scenarios.registry import get_scenario

    raw = apply_overrides(load_yaml(args.config), args.overrides)
    config = validate(get_scenario(raw["scenario"]).config_model, raw)

    def shapes(model):
        if isinstance(model, StrictModel):
            if getattr(type(model), "export", None) and "type" in type(model).model_fields:
                yield model
            for name in type(model).model_fields:
                yield from shapes(getattr(model, name))
        elif isinstance(model, (list, tuple)):
            for item in model:
                yield from shapes(item)

    found = list(shapes(config))
    if not found:
        print("the config has no soft body shape that can be written as CAD", file=sys.stderr)
        return 1
    found[0].export(args.cad)
    print(f"wrote {args.cad} ({found[0].type}, metres, body frame)", file=sys.stderr)
    return 0


def _cmd_materials(args: argparse.Namespace) -> int:
    from softsim.materials import list_presets, load_material

    for name in list_presets(args.path):
        m = load_material(name, extra=args.path)
        e = m.elastic.initial_youngs_modulus()
        print(f"{name:36s} {m.kind:9s} {m.elastic.model:18s} E0={e / 1e3:7.1f} kPa  "
              f"nu={m.elastic.initial_poisson_ratio():.3f}  rho={m.density:6.0f}  "
              f"{m.damping.model if m.damping else 'no damping':10s}  {m.description[:60]}")
    return 0


def _cmd_params(args: argparse.Namespace) -> int:
    import re

    from softsim.config.params import config_params

    rx = re.compile(args.grep) if args.grep else None
    rows = []
    for p in config_params(args.config, args.overrides, args.backend):
        if rx and not rx.search(p.key):
            continue
        value = p.describe()
        if len(value) > 40:
            value = value[:37] + "..."
        allowed = p.bounds or ("one of " + ", ".join(map(str, p.choices)) if len(p.choices) > 1 else "")
        rows.append((p.key, value, allowed))
    if not rows:
        print("no matching parameters", file=sys.stderr)
        return 1
    w0, w1 = max(len(r[0]) for r in rows), max(len(r[1]) for r in rows)
    for key, value, allowed in rows:
        print(f"{key:{w0}s}  {value:{w1}s}  {allowed}".rstrip())
    print("set any of these with --set KEY=VALUE (units allowed, e.g. --set "
          "specimen.material.elastic.youngs_modulus='40 kPa')", file=sys.stderr)
    return 0


def _fmt_si(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {_fmt_si(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt_si(v) for v in value) + "]"
    return str(value)


def _cmd_material_show(args: argparse.Namespace) -> int:
    import yaml

    from softsim.materials import load_material
    from softsim.materials.summary import material_summary

    ref = yaml.safe_load(args.material) if args.material.lstrip().startswith("{") else args.material
    m = load_material(ref, extra=args.path, base_dir=Path.cwd(), overrides=args.overrides)
    print(f"{m.name} ({m.kind}): {m.description}")
    print(f"  source: {' '.join(m.source.split())}")
    for r in m.references:
        print(f"  reference: {r.doi or r.url or ''} {r.note}".rstrip())
    s = material_summary(m)
    units = {"youngs_modulus_0": ("kPa", 1e-3), "shear_modulus_0": ("kPa", 1e-3), "bulk_modulus_0": ("kPa", 1e-3),
             "density": ("kg/m^3", 1.0), "shear_wave_speed": ("m/s", 1.0), "p_wave_speed": ("m/s", 1.0)}
    for key, value in s.items():
        if key in units:
            unit, f = units[key]
            print(f"  {key:18s} {value * f:.4g} {unit}")
        elif key == "dt_explicit":
            print("  dt_explicit        " + ", ".join(f"{dt * 1e6:.3g} us at h={h * 1e3:g} mm" for h, dt in value.items()))
        elif key not in ("kind",):
            print(f"  {key:18s} {_fmt_si(value)}")
    try:
        from softsim.materials.homogeneous import curve
    except (ImportError, SoftSimError):  # softsim.fem says it needs JAX with a SoftSimError
        print("  (no stress curve: install the fem extra)")
        return 0
    try:
        stretches = (0.7, 1.5, 2.0, 3.0)
        d = curve(m, "uniaxial", stretches)
    except SoftSimError as exc:  # law without a reference energy (linear, stvk)
        print(f"  (no stress curve: {exc})")
        return 0
    print("  uniaxial nominal stress (relaxed): " + ", ".join(
        f"{x:g}: {sv / 1e3:.3g} kPa" for x, sv in zip(stretches, d.stress)))
    return 0


def _material_refs(args: argparse.Namespace):
    import yaml

    from softsim.materials import load_material

    out = []
    for text in args.materials:
        ref = yaml.safe_load(text) if text.lstrip().startswith("{") else text
        out.append(load_material(ref, extra=args.path, base_dir=Path.cwd(), overrides=args.overrides))
    return out


def _cmd_material_plot(args: argparse.Namespace) -> int:
    from softsim.materials.plots import plot_cards

    path = plot_cards(_material_refs(args), args.tests.split(","), args.out)
    print(path)
    return 0


def _cmd_material_project(args: argparse.Namespace) -> int:
    import yaml

    from softsim.materials import save_material
    from softsim.materials.fit import project

    args.materials = [args.material]
    card = _material_refs(args)[0]
    lo, hi = (float(v) for v in args.strain_range.split(","))
    r = project(card, args.to, (lo, hi))
    print(f"{card.name} -> {args.to}: nrmse {100 * r.nrmse:.2f}%, max error {100 * r.max_error:.2f}% of peak stress "
          f"(uniaxial, equibiaxial, planar over strain [{lo:g}, {hi:g}])"
          + ("" if r.stable else f"; UNSTABLE: {r.stability}"), file=sys.stderr)
    if args.out:
        print(save_material(r.material, args.out))
    else:
        print(yaml.safe_dump(r.material.model_dump(mode="json", exclude_none=True), sort_keys=False))
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    from softsim.experiments.material_tests import load_material_tests

    print(load_material_tests(args.dataset).fetch())
    return 0


def _cmd_fit_material(args: argparse.Namespace) -> int:
    from softsim.materials.fit_cli import fit_material

    report = fit_material(args.dataset, args.material, args.laws.split(","), (args.strain_min, args.strain_max),
                          args.poisson_ratio, args.density, args.out)
    print(f"{'law':14s} {'nrmse':>7s} {'max err':>8s} {'AIC':>9s}  stable  parameters")
    for r in report["fits"]:
        params = ", ".join(f"{k.split('.', 1)[1]}={v:.4g}" for k, v in r["values"].items())
        print(f"{r['law']:14s} {100 * r['nrmse']:6.2f}% {100 * r['max_error']:7.2f}% {r['aic']:9.1f}  "
              f"{'yes' if r['stable'] else 'NO ':6s}  {params}")
    print(f"cards, report.yaml and fit.png in {report['output']}", file=sys.stderr)
    return 0


def _cmd_sensitivity(args: argparse.Namespace) -> int:
    from softsim.sensitivity import parse_range, run_sensitivity

    ranges = [parse_range(t, args.config, args.overrides) for t in args.param]
    df = run_sensitivity(args.config, ranges, args.metric, args.trajectories, args.levels, args.seed, args.jobs,
                         args.overrides)
    failed = int((df.attrs["runs"]["error"] != "").sum())
    for metric, g in df.groupby("metric", sort=False):
        print(f"{metric}: influence of each parameter over its full range (mu*, sigma: fraction of the mean)")
        for _, r in g.iterrows():
            print(f"  {r.parameter:50s} mu* {r.mu_star:7.3f}  sigma {r.sigma:7.3f}  [{r.low:.4g}, {r.high:.4g}]"
                  + (" log" if r.log else ""))
    if args.out:
        df.to_csv(args.out, index=False)
        df.attrs["runs"].to_csv(Path(args.out).with_suffix(".runs.csv"), index=False)
    print(f"{len(df.attrs['runs'])} runs, {failed} failed", file=sys.stderr)
    return 1 if failed else 0


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


def _track_one(args: tuple[str, str, str]) -> str:
    from softsim.experiments import load_dataset
    from softsim.experiments.video.track import track_case

    dataset_path, case, out = args
    try:
        track = track_case(load_dataset(dataset_path), case)
    except Exception as exc:  # report and continue with the other cases
        return f"{case}: FAILED {type(exc).__name__}: {exc}"
    track.save(out)
    return track_summary_line(track.meta)


def track_summary_line(m: dict) -> str:
    ff = m["free_fall"]
    return (f"{m['case']}: scale {m['scale_px_per_mm']:.3f} px/mm, elevation {m['camera_elevation_deg']:.1f} deg, "
            f"fit frames {ff['frames']} rms {ff['rms_residual_px']:.2f} px, contact frame {m['contact_frame']}, "
            f"impact {m['impact_speed']:.3f} m/s, offset {m['offset_at_contact_mm']:+.1f} mm, "
            f"rotation rate {m['rotation_rate_at_contact_deg_s']:+.0f} deg/s, tracked {m['tracked_frames']}")


def _cmd_track(args: argparse.Namespace) -> int:
    from concurrent.futures import ProcessPoolExecutor

    from softsim.experiments import load_dataset

    ds = load_dataset(args.dataset)
    cases = args.case or list(ds.cases)
    out = args.out or str(ds.root / "data" / "processed" / ds.name)
    tasks = [(args.dataset, c, out) for c in cases]
    if args.jobs > 1:
        with ProcessPoolExecutor(min(args.jobs, len(tasks))) as pool:
            lines = list(pool.map(_track_one, tasks))
    else:
        lines = [_track_one(t) for t in tasks]
    for line in lines:
        print(line)
    print(f"results in {out}", file=sys.stderr)
    return 1 if any("FAILED" in line for line in lines) else 0


def _cmd_calibrate(args: argparse.Namespace) -> int:
    from softsim.experiments.calibrate import Calibration

    report = Calibration(args.calibration, jobs=args.jobs).run()
    print(json.dumps(report, indent=2, default=float))
    if not args.no_plots:
        from softsim.experiments.plots import calibration_plots

        out = Path(report["material_file"]).parent
        for path in calibration_plots(report, args.calibration, out):
            print(f"plot: {path}", file=sys.stderr)
    return 0


def _cmd_optimize(args: argparse.Namespace) -> int:
    from softsim.design import Optimization

    report = Optimization(args.job, jobs=args.jobs).run()
    print(json.dumps(report, indent=2, default=float))
    return 0 if report["feasible"] else 2


def _cmd_transfer(args: argparse.Namespace) -> int:
    from softsim.transfer import Transfer

    report = Transfer(args.transfer).run()
    print(json.dumps(report, indent=2, default=float))
    return 0


def _cmd_inverse(args: argparse.Namespace) -> int:
    from softsim.control.inverse import run_job

    print(json.dumps(run_job(args.job, args.method, args.surrogate), indent=2, default=float))
    return 0


def _cmd_surrogate(args: argparse.Namespace) -> int:
    from softsim.control.surrogate import run_surrogate_job

    print(json.dumps(run_surrogate_job(args.job), indent=2, default=float))
    return 0


def _cmd_fit(args: argparse.Namespace) -> int:
    from softsim.experiments.fit import Fit

    report = Fit(args.fit).run()
    print(json.dumps(report, indent=2, default=float))
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    from softsim.bench import load_entries, report, run_bench

    entries = load_entries(args.dir)
    cache = Path(args.cache)
    failed = False
    if not args.report_only:
        records = run_bench(entries, cache, tier=args.tier, only=args.only or None, backends=args.backend or None,
                            force=args.force)
        failed = any(r["status"] == "error" or (r["status"] == "fail" and not r.get("known_failure"))
                     for r in records)
    if args.out:
        Path(args.out).write_text(report(entries, cache))
        print(f"wrote {args.out}")
    return 1 if failed else 0


def _cmd_examples(args: argparse.Namespace) -> int:
    from softsim.examples import copy_examples, example_files

    if args.action == "list":
        for rel in example_files():
            print(rel.as_posix())
        return 0
    try:
        written = copy_examples(args.dest, overwrite=args.overwrite)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"copied {len(written)} example configs to {args.dest}; try: softsim run {args.dest}/drop_test/cases/w020.yaml")
    return 0


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("softsim")
    except PackageNotFoundError:
        return "unknown (not installed)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="softsim", description=__doc__)
    parser.add_argument("--version", action="version", version=f"softsim {_version()}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("run", help="run a scenario and save results")
    _add_config_args(p)
    p.add_argument("--out", help="output directory (default: <output.directory>/<name>_<timestamp>)")
    p.add_argument("--duration", type=float, help="override duration [s]")
    p.add_argument("--vertices", action="store_true", help="also save soft-body vertex trajectories")
    p.add_argument("--strict", action="store_true", help="treat check warnings as errors")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--checkpoint", metavar="FILE", help="save the run here as it goes, and go on from FILE if it exists (fem backend)")
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("check", help="validate a config, print mesh stats, physics findings, backend support")
    _add_config_args(p)
    p.set_defaults(func=_cmd_check)

    p = sub.add_parser("sweep", help="run a parameter sweep file")
    p.add_argument("sweep")
    p.add_argument("--out")
    p.add_argument("--jobs", type=int, default=1, help="parallel processes")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--sample", type=int, help="cross the grid with N material variants drawn from the card's "
                                              "uncertainty (specimen.material unless the file's sample: says)")
    p.add_argument("--seed", type=int, help="random seed for --sample")
    p.set_defaults(func=_cmd_sweep)

    p = sub.add_parser("sensitivity", help="which parameters matter: Morris screening over value ranges")
    p.add_argument("config", help="scenario YAML")
    p.add_argument("--param", action="append", required=True, metavar="KEY=LOW:HIGH",
                   help="parameter range, units allowed, e.g. 'specimen.material.elastic.youngs_modulus=30 kPa:150 kPa'")
    p.add_argument("--metric", action="append", required=True, help="summary metric, e.g. peak_indentation")
    p.add_argument("--trajectories", type=int, default=8)
    p.add_argument("--levels", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--out", help="write the indices (and .runs.csv) here")
    p.set_defaults(func=_cmd_sensitivity)

    p = sub.add_parser("export", help="print/write the backend-native model (e.g. MJCF), or the body as CAD")
    _add_config_args(p)
    p.add_argument("--out")
    p.add_argument("--cad", metavar="FILE", help="write the soft body's geometry as CAD (.step, .brep) instead")
    p.set_defaults(func=_cmd_export)

    p = sub.add_parser("materials", help="list material presets")
    p.add_argument("--path", action="append", default=[], help="extra preset directory")
    p.set_defaults(func=_cmd_materials)

    p = sub.add_parser("params", help="list every changeable value of a config (key, value, unit, allowed range)")
    _add_config_args(p)
    p.add_argument("--grep", help="only keys matching this regular expression")
    p.set_defaults(func=_cmd_params)

    p = sub.add_parser("material", help="inspect material cards")
    msub = p.add_subparsers(dest="material_command", required=True)
    q = msub.add_parser("show", help="card summary with derived quantities (moduli, wave speeds, time steps)")
    q.add_argument("material", help="preset name, YAML file or inline mapping")
    q.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a card value, e.g. elastic.youngs_modulus='40 kPa'")
    q.add_argument("--path", action="append", default=[], help="extra preset directory")
    q.set_defaults(func=_cmd_material_show)
    q = msub.add_parser("plot", help="homogeneous test curves of one or more cards (needs the fem extra)")
    q.add_argument("materials", nargs="+", help="preset names, YAML files or inline mappings")
    q.add_argument("--tests", default="uniaxial,simple_shear,relaxation,dma",
                   help="comma list of uniaxial, equibiaxial, planar, simple_shear, volumetric, confined, "
                        "relaxation, dma")
    q.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override applied to every card")
    q.add_argument("--path", action="append", default=[], help="extra preset directory")
    q.add_argument("--out", default="material_tests.png")
    q.set_defaults(func=_cmd_material_plot)
    q = msub.add_parser("project", help="fit another law to a card's homogeneous curves (e.g. for a backend "
                                        "that lacks the card's law); prints or writes the new card")
    q.add_argument("material", help="preset name, YAML file or inline mapping")
    q.add_argument("--to", required=True, help="target law: neo_hookean, mooney_rivlin, yeoh, ogden1, ogden2, "
                                               "gent, arruda_boyce")
    q.add_argument("--strain-range", default="-0.4,1.0", help="engineering strain range, e.g. -0.4,1.0")
    q.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    q.add_argument("--path", action="append", default=[], help="extra preset directory")
    q.add_argument("--out", help="write the card here (default: print it)")
    q.set_defaults(func=_cmd_material_project)

    p = sub.add_parser("fetch", help="download a dataset's archive into its local_dir and verify it")
    p.add_argument("dataset", help="dataset YAML, e.g. datasets/marechal2021_srmd.yaml")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("fit-material", help="fit constitutive laws to a material-test dataset; writes cards")
    p.add_argument("dataset", help="material-test dataset YAML, e.g. datasets/marechal2021_srmd.yaml")
    p.add_argument("material", help="material key in the dataset, e.g. ecoflex_00_30")
    p.add_argument("--laws", default="neo_hookean,mooney_rivlin,yeoh,ogden1,ogden2,gent,arruda_boyce")
    p.add_argument("--strain-min", type=float, default=0.0,
                   help="engineering strain; negative includes compression tests (e.g. -0.25)")
    p.add_argument("--strain-max", type=float, default=2.0, help="engineering strain")
    p.add_argument("--poisson-ratio", type=float, default=0.49, help="fixed during the fit")
    p.add_argument("--density", help="override the dataset's density, e.g. '1.07 g/cm^3'")
    p.add_argument("--out", help="output directory (default runs/material_fits/<dataset>/<material>)")
    p.set_defaults(func=_cmd_fit_material)

    p = sub.add_parser("examples", help="list the example configs shipped with the package, or copy them to a directory")
    p.add_argument("action", choices=["list", "copy"])
    p.add_argument("dest", nargs="?", default="softsim_examples", help="target directory for copy (default: softsim_examples)")
    p.add_argument("--overwrite", action="store_true", help="replace files that already exist")
    p.set_defaults(func=_cmd_examples)

    p = sub.add_parser("scenarios", help="list registered scenarios and their actions")
    p.set_defaults(func=_cmd_scenarios)

    p = sub.add_parser("track", help="track experimental videos of a dataset (weight, contact, bulge, lift-off)")
    p.add_argument("dataset", help="dataset YAML, e.g. datasets/czopek2026_ecoflex_drop.yaml")
    p.add_argument("--case", action="append", help="case name (repeatable; default: all)")
    p.add_argument("--out", help="output directory (default: data/processed/<dataset name>)")
    p.add_argument("--jobs", type=int, default=1)
    p.set_defaults(func=_cmd_track)

    p = sub.add_parser("calibrate", help="fit material parameters to tracked experiments")
    p.add_argument("calibration", help="calibration YAML, e.g. calibrations/ecoflex_00_10_drop.yaml")
    p.add_argument("--jobs", type=int, default=1, help="cases simulated in parallel")
    p.add_argument("--no-plots", action="store_true")
    p.set_defaults(func=_cmd_calibrate)

    p = sub.add_parser("optimize", help="design optimisation over parametric geometry (docs/design.md)")
    p.add_argument("job", help="job YAML, e.g. jobs/cantilever_target_angle.yaml")
    p.add_argument("--jobs", type=int, default=1, help="designs simulated in parallel (the initial simplex)")
    p.set_defaults(func=_cmd_optimize)

    p = sub.add_parser("transfer", help="fit a fast backend's effective card to the reference (docs/actuation.md)")
    p.add_argument("transfer", help="transfer YAML, e.g. calibrations/transfer_inflation_mujoco.yaml")
    p.set_defaults(func=_cmd_transfer)

    p = sub.add_parser("inverse", help="quasi-static inverse control: actuator values that reach target outputs "
                                       "(docs/control.md)")
    p.add_argument("job", help="job YAML, e.g. jobs/inverse_cable_cantilever.yaml")
    p.add_argument("--method", choices=["repeated", "sofa-qp", "surrogate"], default=None,
                   help="repeated equilibrium solves (default, any backend) or SOFA's QP inverse solver "
                        "(SoftRobots.Inverse; cables and tip position/angle targets) or a fitted surrogate "
                        "(`--surrogate file.json`, milliseconds); overrides the job's `method:`")
    p.add_argument("--surrogate", default=None, metavar="FILE",
                   help="surrogate JSON from `softsim surrogate` (implies --method surrogate; or `surrogate:` in the job)")
    p.set_defaults(func=_cmd_inverse)

    p = sub.add_parser("surrogate", help="fit a reduced-order surrogate of the equilibrium map for real-time control "
                                         "(docs/surrogate.md)")
    p.add_argument("job", help="surrogate job YAML, e.g. jobs/surrogate_cable_cantilever.yaml")
    p.set_defaults(func=_cmd_surrogate)

    p = sub.add_parser("fit", help="fit a scenario's parameters to a measured table (softsim.experiments.fit)")
    p.add_argument("fit", help="fit YAML with a `data:` section, e.g. calibrations/pneunet_fit.yaml")
    p.set_defaults(func=_cmd_fit)

    p = sub.add_parser("bench", help="run the benchmark cases (benchmarks/*.yaml) and write the table")
    p.add_argument("--dir", default="benchmarks", help="folder of benchmark entries")
    p.add_argument("--tier", choices=["quick", "full"], default="quick", help="quick: minutes; full: every case")
    p.add_argument("--only", action="append", default=[], metavar="NAME", help="run only this case (repeatable)")
    p.add_argument("--backend", action="append", default=[], help="run only this backend (repeatable)")
    p.add_argument("--force", action="store_true", help="re-run cases whose result is cached")
    p.add_argument("--cache", default="runs/bench", help="result cache folder")
    p.add_argument("--out", default="docs/benchmarks.md", help="report page ('' to skip)")
    p.add_argument("--report-only", action="store_true", help="rebuild the page from the cache, run nothing")
    p.set_defaults(func=_cmd_bench)

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
