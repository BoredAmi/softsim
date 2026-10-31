"""Physical material cards for every material of the registered test datasets, plus docs/material_library.md.

    softsim fetch datasets/roels2025_elastomers.yaml
    softsim fetch datasets/marechal2021_srmd.yaml
    python tools/make_material_library.py            # ~5 min on 8 cores

Per material, every law is fitted and the best stable invariant-based law by
BIC is kept (Yeoh, Gent, Arruda-Boyce, Neo-Hookean, Mooney-Rivlin: they run on
the 3D reference quickly and project onto every backend), unless Ogden's error is
below 70% of theirs.

* ``<material>_roels2025`` (Roels et al. 2025, CC-BY-4.0): the law is fitted
  jointly to all tension specimens (0-200% strain, less where they broke
  earlier) and all compression specimens (0-25%), with each test kind weighted
  equally. The Prony series comes from the compression relaxation hold
  (1 s to 3 h). The scatter between specimens enters the uncertainties.
* Fast relaxation (milliseconds) where it is measured: Croquette et al. 2026
  (arXiv:2604.27722) give a fractional Kelvin-Voigt shear modulus
  mu0 (1 + (i w tau)^n) for Ecoflex 00-30 and 00-50 from guided waves (checked
  against a rheometer, 1-100 Hz). Its shape relative to the test rate becomes
  Prony branches over 1 Hz-2 kHz (``prony_from_fractional``; slower is the measured relaxation), added so that
  the response at the test rate is unchanged (``with_fast_relaxation``). Their
  static modulus is not used (their samples were ~2x softer than Roels').
* ``<material>_marechal2021`` (Marechal et al. 2021, ODbL): one tension curve,
  0-200%. Crosshead strain, so the moduli are probably ~3x too low; prefer the
  Roels card where both exist.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PRESETS = ROOT / "softsim" / "materials" / "presets"
FIGURES = ROOT / "docs" / "media"
LAWS = ["neo_hookean", "mooney_rivlin", "yeoh", "ogden1", "ogden2", "gent", "arruda_boyce"]
STRAIN_MAX = 2.0
# fractional Kelvin-Voigt (n, tau [s], sd n, sd tau) from Croquette et al. 2026, Table 1
RATE = {"ecoflex_00_30": (0.20, 2.9e-3, 0.04, 1.3e-3), "ecoflex_00_50": (0.24, 2.8e-3, 0.02, 0.6e-3)}
# fast branches stop where the measured relaxation (1 s - 3 h, Prony times >= 0.3 s) starts
FAST_BAND = (1.0, 2000.0)
RATE_REF = {"doi": "10.48550/arXiv.2604.27722",
            "note": "fast rate dependence: fractional Kelvin-Voigt n and tau from guided elastic waves (Table 1)"}
DATASETS = {"roels2025": ROOT / "datasets" / "roels2025_elastomers.yaml",
            "marechal2021": ROOT / "datasets" / "marechal2021_srmd.yaml"}


def _choose(results):
    ranked = sorted(results, key=lambda r: (not r.stable, r.bic))
    invariant = [r for r in ranked if r.stable and not r.law.startswith("ogden")]
    ogden = [r for r in ranked if r.stable and r.law.startswith("ogden")]
    best = invariant[0] if invariant else ranked[0]
    if ogden and ogden[0].nrmse < 0.7 * best.nrmse:
        best = ogden[0]
    return best, {r.law: {"nrmse": round(r.nrmse, 5), "max_error": round(r.max_error, 5), "bic": round(r.bic, 1),
                          "stable": r.stable} for r in ranked}


def _one(task: tuple[str, str]) -> dict:
    from softsim.experiments.material_tests import load_material_tests
    from softsim.materials.fit import fit_laws, fit_relaxation, with_fit_uncertainty
    from softsim.materials.fit_cli import base_card

    source, material = task
    ds = load_material_tests(DATASETS[source])
    entry = ds.entry(material)
    prov = {"fitted_by": "tools/make_material_library.py (softsim.materials.fit)", "dataset": ds.name,
            "dataset_doi": ds.doi, "poisson_ratio_fixed": 0.49,
            "uncertainty_note": "from the fit residuals (specimen scatter where there are several specimens)"}
    if entry.member is not None:  # one tension curve
        full = ds.curve(material)
        strain_max = min(STRAIN_MAX, round(0.9 * float(full.stretch.max() - 1.0), 2))
        curves = [full.within(0.0, strain_max)]
        weights = None
        strain_range = (0.0, strain_max)
        base = base_card(ds, material, strain_range, curves[0].rate)
        plot_curves = [full]
        prov["caveat"] = ("crosshead (global) strain: moduli are likely ~3x too low; tension only, compression and "
                          "rate dependence are not measured")
    else:
        tension = ds.curves(material, "tension")
        compression = ds.curves(material, "compression")
        strain_max = min(STRAIN_MAX, round(0.9 * min(float(c.stretch.max() - 1.0) for c in tension), 2))
        curves = [c.within(0.0, strain_max) for c in tension] + [c.within(-0.25, 0.0) for c in compression]
        weights = [1.0 / len(tension)] * len(tension) + [1.0 / len(compression)] * len(compression)
        strain_range = (-0.25, strain_max)
        base = base_card(ds, material, strain_range, None)
        plot_curves = tension + compression
        prov["specimens"] = {"tension": len(tension), "compression": len(compression)}
        prov["caveat"] = ("elastic law = response at the test rates (~0.4/s); the Prony series covers 1 s to 3 h "
                          "(compression relaxation); faster relaxation (impacts, ms) is not measured")
    results = fit_laws(curves, LAWS, base, weights=weights)
    best, table = _choose(results)
    card = with_fit_uncertainty(best)
    update = {"name": f"{material}_{source}",
              "provenance": {**prov, "law": best.law, "strain_range": list(strain_range), "nrmse": best.nrmse,
                             "max_error": best.max_error, "laws_compared": table}}
    if entry.tests and any(t.kind == "relaxation" for t in entry.tests):
        g, tau, rms = fit_relaxation(ds.curves(material, "relaxation"))
        data = card.model_dump(mode="json", exclude_none=True)
        data["damping"] = {"model": "prony", "g": g, "tau": tau}
        data["validity"] = {**data.get("validity", {}), "strain_rate": [0.0, 0.5]}
        update["provenance"]["relaxation"] = {"window": "1 s to 3 h after the stress peak", "rms": rms,
                                               "total_relaxation": sum(g)}
        card = type(card).model_validate(data)
    if source == "roels2025" and material in RATE:
        from softsim.materials.viscoelastic import prony_from_fractional, with_fast_relaxation

        n, tau, sd_n, sd_tau = RATE[material]
        a, taus, rms = prony_from_fractional(n, tau, band=FAST_BAND)
        card = with_fast_relaxation(card, a, taus)
        data = card.model_dump(mode="json", exclude_none=True)
        data["validity"] = {**data.get("validity", {}), "strain_rate": [0.0, 300.0]}
        data["references"] = [*data.get("references", []), RATE_REF]
        card = type(card).model_validate(data)
        update["provenance"]["fast_relaxation"] = {
            "source": "Croquette et al. 2026, arXiv:2604.27722, Table 1 (guided waves, small strain)",
            "fractional_kelvin_voigt": {"n": n, "n_sd": sd_n, "tau": tau, "tau_sd": sd_tau},
            "band_hz": list(FAST_BAND), "reference_hz": 0.06, "fit_rms": rms,
            "instantaneous_over_test_rate": 1.0 + sum(a),
            "assumption": "the small-strain frequency dependence applies to the isochoric part at large strain"}
    card = card.model_copy(update=update)
    nh = next(r for r in results if r.law == "neo_hookean")
    return {"source": source, "material": material, "card": card.model_dump(mode="json", exclude_none=True),
            "best": best.law, "nrmse": best.nrmse, "strain_range": strain_range,
            "mu0": _test_rate_mu(card),
            "nh_E": nh.values["elastic.youngs_modulus"], "fast": _stiffening(card, material, source),
            "nh_nrmse": nh.nrmse, "product": entry.product,
            "relaxation": _slow_relaxation(card),
            "density_source": entry.density_source or "typical (not in dataset)",
            "plot": [(c.stretch.tolist(), c.stress.tolist()) for c in plot_curves],
            "fit": [(c.stretch.tolist(), p.tolist()) for c, p in best.model_curves]}


def _test_rate_mu(card) -> float:
    """Initial shear modulus at the test rate: the instantaneous one relaxed by the branches faster than 1 s."""
    fast = sum(g for g, t in zip(card.damping.g, card.damping.tau) if t < 1.0) if card.damping is not None else 0.0
    return card.elastic.initial_shear_modulus() * (1.0 - fast)


def _slow_relaxation(card) -> float | None:
    """Relaxation from 1 s to 3 h relative to the stress at 1 s (the Roels holds), from the slow branches."""
    if card.damping is None:
        return None
    fast = sum(g for g, t in zip(card.damping.g, card.damping.tau) if t < 1.0)
    return sum(g for g, t in zip(card.damping.g, card.damping.tau) if t >= 1.0) / (1.0 - fast)


def _stiffening(card, material: str, source: str) -> float | None:
    """Small-strain shear storage modulus at 1 kHz over that at the test rate (0.06 Hz)."""
    if source != "roels2025" or material not in RATE:
        return None
    from softsim.materials.homogeneous import dma_small_strain

    d = dma_small_strain(card, [0.06, 1000.0], mode="simple_shear")
    return float(d.storage.iloc[1] / d.storage.iloc[0])


def _figure(rows: list[dict], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = 6
    nr = (len(rows) + cols - 1) // cols
    fig, axes = plt.subplots(nr, cols, figsize=(2.8 * cols, 2.3 * nr), squeeze=False)
    for ax, r in zip(axes.flat, rows):
        for x, s in r["plot"]:
            x, s = np.array(x) - 1, np.array(s) / 1e3
            sel = x <= 3.0
            ax.plot(x[sel], s[sel], "k.", ms=0.8)
        for x, s in r["fit"]:
            ax.plot(np.array(x) - 1, np.array(s) / 1e3, "C3-", lw=1.0)
        ax.set_title(f"{r['material']}\n{r['best']} ({100 * r['nrmse']:.1f}%)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    fig.supxlabel("engineering strain", fontsize=9)
    fig.supylabel("nominal stress [kPa]", fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def _table(rows: list[dict]) -> list[str]:
    lines = ["| Card | Product | Law (fit error) | μ0 [kPa] | NH E [kPa] (error) | Strain range | Relaxation 1 s–3 h | "
             "Impact stiffening | Density |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        lo, hi = r["strain_range"]
        relax = f"{100 * r['relaxation']:.1f}%" if r["relaxation"] is not None else "–"
        fast = f"{r['fast']:.2f}× at 1 kHz" if r.get("fast") else "–"
        lines.append(f"| `{r['material']}_{r['source']}` | {r['product']} | {r['best']} ({100 * r['nrmse']:.1f}%) | "
                     f"{r['mu0'] / 1e3:.1f} | {r['nh_E'] / 1e3:.1f} ({100 * r['nh_nrmse']:.1f}%) | "
                     f"{100 * lo:.0f}–{100 * hi:.0f}% | {relax} | {fast} | {r['density_source']} |")
    return lines


def _page(roels: list[dict], marechal: list[dict]) -> str:
    return "\n".join([
        "# Material library (generated)",
        "",
        "Physical material cards fitted to public test data by `tools/make_material_library.py`; do not edit by",
        "hand. Use a card by name, e.g. `material: ecoflex_00_30_roels2025`. Every card carries its data's",
        "strain range (`validity`), fit uncertainty and references; `softsim material show <card>` prints them.",
        "",
        "## Roels et al. 2025 (preferred)",
        "",
        "Tension and compression, several specimens each, and a 3-hour compression relaxation per material",
        "(doi:10.1002/aisy.202500699, data doi:10.5281/zenodo.14983287, CC-BY-4.0).",
        "",
        "- **Fit:** the law is fitted to tension and compression together, so these cards hold in",
        "  compression too.",
        "- **Prony series:** from the relaxation hold, 1 s to 3 h. These silicones relax only 2–6% there.",
        "- **Fast relaxation (impacts):** measured for Ecoflex 00-30 and 00-50 by Croquette et al. 2026",
        "  (guided elastic waves, checked against a rheometer; arXiv:2604.27722), and added as Prony branches",
        "  over 1 Hz–2 kHz without changing the response at the test rate. \"Impact stiffening\" is the",
        "  small-strain shear storage modulus at 1 kHz over that at the test rate (0.06 Hz).",
        "- **The other cards have no fast-rate data.** Impacts on them will be too soft and too bouncy.",
        "  For comparison, the Ecoflex 00-10 drop card ([experiments.md](experiments.md)) relaxes about 70%",
        "  within milliseconds.",
        "- μ0 is the initial shear modulus at the test rate.",
        "- The first second of each relaxation recording is a strain overshoot seen through a lagging load",
        "  signal, and is left out.",
        "",
        *_table(roels),
        "",
        "![Roels et al. 2025: data and fitted law](media/material_library_roels2025.png)",
        "",
        "## Marechal et al. 2021",
        "",
        "One tension curve per material, 18 silicones (doi:10.1089/soro.2019.0115, data",
        "doi:10.5281/zenodo.3931808, ODbL).",
        "",
        "- **Moduli are probably about 3× too low.** Strain was measured at the crosshead. For Ecoflex 00-30",
        "  at 100% strain these data give 30 kPa, against 93 kPa from Roels et al. and 69 kPa from the datasheet.",
        "- **Tension only:** compression is extrapolated.",
        "- **No rate data.**",
        "- **Prefer the Roels card** where a material has both.",
        "",
        *_table(marechal),
        "",
        "![Marechal et al. 2021: data and fitted law](media/material_library.png)",
        "",
        "Errors are RMS over the fitted range, relative to each curve's largest stress. \"NH E\" is the",
        "Young's modulus of a Neo-Hookean fit over the same range: a rough single-number stiffness, and",
        "the only law MuJoCo has. For other ranges or laws, use `softsim fit-material` or",
        "`softsim material project`.",
        "",
    ])


def main() -> None:
    from softsim.experiments.material_tests import load_material_tests

    tasks = []
    for source, path in DATASETS.items():
        ds = load_material_tests(path)
        ds.fetch()
        tasks += [(source, m) for m in ds.materials if not sys.argv[1:] or m in sys.argv[1:]]
    with ProcessPoolExecutor(8) as pool:
        rows = list(pool.map(_one, tasks))
    for r in rows:
        (PRESETS / f"{r['material']}_{r['source']}.yaml").write_text(yaml.safe_dump(r["card"], sort_keys=False))
    by = {s: sorted((r for r in rows if r["source"] == s), key=lambda r: r["mu0"]) for s in DATASETS}
    _figure(by["roels2025"], FIGURES / "material_library_roels2025.png",
            "Roels et al. 2025: tension and compression specimens (black) and the card's law (red)")
    _figure(by["marechal2021"], FIGURES / "material_library.png",
            "Marechal et al. 2021 tension data (black) and the card's law (red, fitted range)")
    (ROOT / "docs" / "material_library.md").write_text(_page(by["roels2025"], by["marechal2021"]))
    for r in rows:
        print(f"{r['material'] + '_' + r['source']:40s} {r['best']:13s} {100 * r['nrmse']:5.2f}%  "
              f"mu0 {r['mu0'] / 1e3:7.1f} kPa  NH E {r['nh_E'] / 1e3:7.1f} kPa"
              + (f"  relax {100 * r['relaxation']:.1f}%" if r["relaxation"] is not None else ""))


if __name__ == "__main__":
    main()
