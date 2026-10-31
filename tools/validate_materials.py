"""How far do material cards predict? Hold-out and cross-lab checks: runs/material_validation/.

    python tools/validate_materials.py          # ~5 min on 8 cores (needs the Roels and Marechal data fetched)

1. Hold-out within Roels et al. 2025, per material: fit on tension only and
   predict compression (-25-0%), fit on compression only and predict tension
   (0-100%). The law is chosen as for the library (best stable invariant law by
   BIC); compression -> tension also with Neo-Hookean, since higher-order laws
   fitted on 25% compression extrapolate wildly. Errors are RMS / the specimen's largest stress, averaged over specimens,
   next to the in-sample error of the joint fit.
2. Independent labs: published one-term Ogden fits of quasi-static tension with
   optical strain (Shojaeifard and Bacca 2025, arXiv:2506.05371, Table 1) for
   Ecoflex 00-10 and 00-30, against the cards of the library: nominal stress at
   25-200% strain, as a ratio to theirs.
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "material_validation"
ROELS = ROOT / "datasets" / "roels2025_elastomers.yaml"
LAWS = ["neo_hookean", "mooney_rivlin", "yeoh", "gent", "arruda_boyce"]
# one-term Ogden (E, alpha) from uniaxial tension with optical strain, Shojaeifard & Bacca 2025, Table 1
EXTERNAL = {"ecoflex_00_10": (34e3, 2.8), "ecoflex_00_30": (81e3, 2.4)}
STRETCH = np.array([1.25, 1.5, 2.0, 2.5, 3.0])


def _err(card, curves) -> float:
    from softsim.materials.fit import _model_stress

    return float(np.mean([np.sqrt(np.mean((_model_stress(card, c) - c.stress) ** 2)) / np.abs(c.stress).max()
                          for c in curves]))


def _best(curves, base):
    from softsim.materials.fit import fit_laws

    results = [r for r in fit_laws(curves, LAWS, base) if r.stable] or fit_laws(curves, LAWS, base)
    return min(results, key=lambda r: r.bic)


def _holdout(material: str) -> dict:
    from softsim.experiments.material_tests import load_material_tests
    from softsim.materials.fit import _resample
    from softsim.materials.fit_cli import base_card

    ds = load_material_tests(ROELS)
    tension = [_resample(c.within(0.0, 1.0)) for c in ds.curves(material, "tension")]
    compression = [_resample(c.within(-0.25, 0.0)) for c in ds.curves(material, "compression")]
    base = base_card(ds, material, (-0.25, 1.0), None)
    from softsim.materials.fit import fit

    t_fit, c_fit = _best(tension, base), _best(compression, base)
    c_nh = fit(compression, "neo_hookean", base)
    joint = _best(tension + compression, base)
    return {"material": material,
            "tension_to_compression": {"law": t_fit.law, "fit": _err(t_fit.material, tension),
                                       "predicted": _err(t_fit.material, compression)},
            "compression_to_tension": {"law": c_fit.law, "fit": _err(c_fit.material, compression),
                                       "predicted": _err(c_fit.material, tension),
                                       "predicted_neo_hookean": _err(c_nh.material, tension)},
            "joint": {"law": joint.law, "tension": _err(joint.material, tension),
                      "compression": _err(joint.material, compression)}}


def _pct(x: float) -> str:
    return "diverges (>1000%)" if x > 10 else f"{100 * x:.1f}%"


def _ogden_uniaxial(e: float, alpha: float, lam: np.ndarray) -> np.ndarray:
    mu = e / 3.0
    return 2.0 * mu / alpha * (lam ** (alpha - 1.0) - lam ** (-alpha / 2.0 - 1.0))


def _external() -> list[dict]:
    from softsim.materials import list_presets, load_material
    from softsim.materials.homogeneous import curve

    rows = []
    for material, (e, alpha) in EXTERNAL.items():
        ref = _ogden_uniaxial(e, alpha, STRETCH)
        for name in sorted(p for p in list_presets() if p.startswith(material)):
            card = load_material(name)
            s = curve(card, "uniaxial", STRETCH, "relaxed").stress.to_numpy()
            rows.append({"material": material, "card": name, "kind": card.kind,
                         "ratio": [round(float(x), 3) for x in s / ref]})
    return rows


def main() -> None:
    from softsim.experiments.material_tests import load_material_tests

    OUT.mkdir(parents=True, exist_ok=True)
    materials = list(load_material_tests(ROELS).materials)
    with ProcessPoolExecutor(min(7, len(materials))) as pool:
        hold = list(pool.map(_holdout, materials))
    ext = _external()
    (OUT / "report.json").write_text(json.dumps({"holdout": hold, "external": ext,
                                                 "external_stretch": STRETCH.tolist()}, indent=1))
    print("| Material | Tension → compression (law: fit / predicted) | Compression → tension (best law) | "
          "Compression → tension (Neo-Hookean) | Joint fit (T / C) |")
    print("| --- | --- | --- | --- | --- |")
    for h in hold:
        tc, ct, j = h["tension_to_compression"], h["compression_to_tension"], h["joint"]
        print(f"| {h['material']} | {tc['law']}: {100 * tc['fit']:.1f}% / **{100 * tc['predicted']:.1f}%** | "
              f"{ct['law']}: {100 * ct['fit']:.1f}% / **{_pct(ct['predicted'])}** | "
              f"**{_pct(ct['predicted_neo_hookean'])}** | "
              f"{j['law']}: {100 * j['tension']:.1f}% / {100 * j['compression']:.1f}% |")
    print(f"\nStress ratio card / independent tension (Shojaeifard & Bacca 2025) at stretch {STRETCH.tolist()}:")
    for r in ext:
        print(f"  {r['card']:40s} {r['kind']:9s} {r['ratio']}")


if __name__ == "__main__":
    main()
