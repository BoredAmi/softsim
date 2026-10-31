"""``softsim fit-material``: fit laws to one material of a material-test dataset and write cards."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from softsim.config.units import parse_quantity
from softsim.experiments.material_tests import load_material_tests
from softsim.materials.fit import fit_laws, with_fit_uncertainty
from softsim.materials.library import save_material
from softsim.materials.models import Material

DEFAULT_DENSITY = 1070.0  # typical platinum-cure silicone, used when the dataset gives none


def base_card(ds, material: str, strain_range: tuple[float, float], rate: float | None,
              density: str | float | None = None) -> Material:
    """Everything but the elastic law, from the dataset entry."""
    entry = ds.entry(material)
    if density is not None:
        rho, rho_src = parse_quantity(density, "density"), "given by the user"
    elif entry.density is not None:
        rho, rho_src = entry.density, entry.density_source or "dataset"
    else:
        rho, rho_src = DEFAULT_DENSITY, "not in the dataset: typical platinum silicone (1.07 g/cm^3)"
    refs = [{"doi": ds.doi, "note": f"data: {ds.title} ({ds.license})"}]
    if ds.paper_doi:
        refs.insert(0, {"doi": ds.paper_doi, "note": "test method and material list"})
    return Material.model_validate({
        "name": material,
        "description": f"{entry.product or material}, fitted to {ds.conditions.mode} test data ({ds.title}).",
        "source": (f"Fitted by softsim fit-material to {ds.name} (doi:{ds.doi}), {ds.conditions.mode} "
                   f"{100 * strain_range[0]:g}-{100 * strain_range[1]:g}% strain. Strain measurement: "
                   f"{ds.conditions.strain_measurement or 'unspecified'}. Density: {rho_src}."),
        "kind": "physical",
        "density": rho,
        "elastic": {"model": "neo_hookean", "youngs_modulus": 1e4, "poisson_ratio": 0.49},
        "validity": {"strain": list(strain_range), **({"strain_rate": [rate, rate]} if rate else {})},
        "references": refs,
    })


def fit_material(dataset: str | Path, material: str, laws: list[str], strain_range: tuple[float, float],
                 poisson_ratio: float = 0.49, density: str | None = None, out: str | Path | None = None) -> dict[str, Any]:
    """Fit ``laws`` to ``material``'s curve; write one card per law, report.yaml and fit.png."""
    from softsim.materials.plots import plot_fits

    ds = load_material_tests(dataset)
    entry = ds.entry(material)
    if entry.member is not None:  # one curve
        full = ds.curve(material)
        curves, weights, plot = [full.within(*strain_range)], None, [full]
        rate = curves[0].rate
    else:  # several specimens: tension (strain >= 0) and compression (a negative strain_min) together
        tension = [c.within(max(strain_range[0], 0.0), strain_range[1]) for c in ds.curves(material, "tension")]
        compression = ([c.within(strain_range[0], 0.0) for c in ds.curves(material, "compression")]
                       if strain_range[0] < 0 else [])
        curves = [c for c in tension + compression if len(c.stretch) >= 3]
        weights = [1.0 / len(tension)] * len(tension) + [1.0 / max(len(compression), 1)] * len(compression)
        weights = [w for w, c in zip(weights, tension + compression) if len(c.stretch) >= 3]
        plot, rate = curves, None
    note = ("misfit to one curve only, not specimen-to-specimen scatter" if entry.member is not None
            else f"misfit to {len(curves)} specimen curves, so it includes specimen-to-specimen scatter")
    base = base_card(ds, material, strain_range, rate, density)
    results = fit_laws(curves, laws, base, poisson_ratio=poisson_ratio, weights=weights)
    out = Path(out) if out else Path("runs") / "material_fits" / ds.name / material
    out.mkdir(parents=True, exist_ok=True)
    fits = []
    for r in results:
        card = with_fit_uncertainty(r)
        prov = {"fitted_by": "softsim fit-material", "dataset": ds.name, "dataset_doi": ds.doi,
                "law": r.law, "strain_range": list(strain_range), "poisson_ratio_fixed": poisson_ratio,
                "nrmse": r.nrmse, "max_error": r.max_error, "aic": r.aic, "bic": r.bic, "n_points": r.n_points,
                "stable": r.stable, "unstable_from": r.stability,
                "uncertainty_note": note}
        card = card.model_copy(update={"name": f"{material}_{r.law}", "provenance": prov})
        path = save_material(card, out / f"{card.name}.yaml")
        fits.append({"law": r.law, "file": str(path), "values": r.values, "sigma": r.sigma, "nrmse": r.nrmse,
                     "max_error": r.max_error, "aic": r.aic, "bic": r.bic, "stable": r.stable,
                     "unstable_from": r.stability})
    plot_fits(plot[0], results, strain_range, out / "fit.png",
              title=f"{material}: {ds.name} and fitted laws (dotted: unstable)")
    report = {"dataset": ds.name, "material": material, "strain_range": list(strain_range), "rate": rate,
              "fits": fits, "output": str(out)}
    (out / "report.yaml").write_text(yaml.safe_dump(report, sort_keys=False))
    return report
