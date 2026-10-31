"""Plots of material cards in homogeneous tests (``softsim material plot``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from softsim.materials.models import Material, PronySeries

_RANGES = {"uniaxial": np.linspace(0.5, 3.0, 101), "equibiaxial": np.linspace(0.7, 2.5, 91),
           "planar": np.linspace(0.6, 3.0, 97), "simple_shear": np.linspace(0.0, 2.0, 81),
           "volumetric": np.linspace(0.97, 1.03, 61), "confined": np.linspace(0.97, 1.0, 31)}
_XLABEL = {"uniaxial": "stretch", "equibiaxial": "stretch", "planar": "stretch", "simple_shear": "shear strain",
           "volumetric": "volume ratio J", "confined": "stretch"}


def plot_cards(materials: list[Material], tests: list[str], path: str | Path) -> Path:
    """One panel per test, one line per card (solid: relaxed, dashed: instantaneous)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from softsim.materials.homogeneous import curve, dma, relaxation

    fig, axes = plt.subplots(1, len(tests), figsize=(3.8 * len(tests), 3.3), squeeze=False)
    for ax, test in zip(axes[0], tests):
        for i, m in enumerate(materials):
            color, label = f"C{i}", m.name
            visco = isinstance(m.damping, PronySeries)
            if test in _RANGES:
                x = _RANGES[test]
                ax.plot(x, curve(m, test, x).stress / 1e3, color=color, label=label)
                if visco:
                    ax.plot(x, curve(m, test, x, "instantaneous").stress / 1e3, "--", color=color, lw=1)
                ax.set_xlabel(_XLABEL[test])
                ax.set_ylabel("stress [kPa]" if test != "volumetric" else "mean stress [kPa]")
            elif test == "relaxation":
                if not visco:
                    continue
                d = relaxation(m, 0.05)
                d = d[d.t > d.t.iloc[20]]
                ax.semilogx(d.t, d.stress / d.stress.iloc[0], color=color, label=label)
                ax.set_xlabel("time after 5% step [s]")
                ax.set_ylabel("stress / initial stress")
            elif test == "dma":
                if not visco:
                    continue
                f = np.geomspace(0.1, 1e3, 13)
                d = dma(m, f)
                ax.semilogx(d.frequency, d.storage / 1e3, color=color, label=label)
                ax.semilogx(d.frequency, d.loss / 1e3, ":", color=color)
                ax.set_xlabel("frequency [Hz]")
                ax.set_ylabel("E' (solid), E'' (dotted) [kPa]")
            else:
                raise ValueError(f"unknown test {test!r}")
        ax.set_title(test.replace("_", " "), fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_fits(data, results, fit_range: tuple[float, float], path: str | Path, title: str = "") -> Path:
    """Measured curve against fitted laws: stress (fit range shaded) and relative error in the fit range.

    ``data`` is the full measured curve (a TestCurve), ``results`` FitResults.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from softsim.materials.homogeneous import curve

    fig, (ax, ax_err) = plt.subplots(1, 2, figsize=(10, 3.6))
    x_max = min(float(data.stretch.max()), 1.0 + 1.5 * fit_range[1])
    sel = data.stretch <= x_max
    ax.plot(data.stretch[sel] - 1, data.stress[sel] / 1e3, "k.", ms=2, label="data")
    ax.axvspan(*fit_range, color="0.92", zorder=0)
    x = np.linspace(1.0 + max(fit_range[0], 1e-3), x_max, 200)
    for i, r in enumerate(results):
        style = "-" if r.stable else ":"
        try:
            s = curve(r.material, data.mode, x).stress / 1e3
        except Exception:  # noqa: BLE001 - a law that fails outside its range is simply not drawn there
            continue
        ax.plot(x - 1, s, style, color=f"C{i}", lw=1.3, label=f"{r.law} ({100 * r.nrmse:.1f}%)")
        for c, pred in r.model_curves:
            ax_err.plot(c.stretch - 1, 100 * (pred - c.stress) / np.abs(c.stress).max(), style, color=f"C{i}", lw=1.2)
    ax.set_xlabel("engineering strain")
    ax.set_ylabel("nominal stress [kPa]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    ax_err.axhline(0, color="k", lw=0.8)
    ax_err.set_xlabel("engineering strain (fitted range)")
    ax_err.set_ylabel("error [% of peak stress]")
    ax_err.grid(alpha=0.3)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    path = Path(path)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
