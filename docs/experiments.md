# Experiments: tracking and calibration

```bash
softsim track datasets/czopek2026_ecoflex_drop.yaml --jobs 5        # about 3 min; data/processed/<dataset>/
softsim calibrate calibrations/ecoflex_00_10_drop.yaml --jobs 3      # about 4 min; runs/calibration/...
```

## Dataset description

`datasets/czopek2026_ecoflex_drop.yaml` records the DOI, licence, camera,
specimen and, per case, the zip archive, the scenario config and how the
weight looks. Frames are read directly from the zips (`FrameSource`); nothing
is extracted or redistributed.

## Tracking (`softsim/experiments/video/`)

**Segmentation.** The backdrop is saturated green with no red, and shadows on
it stay red-free. A per-pixel reference background (the median of the first
frames) separates three classes:

- the translucent specimen: red present, bluish;
- brass weights: red present, neutral;
- black weights: much darker than the background behind them.

**Scale and camera.** The specimen's resting silhouette width equals its
diameter, giving about 5.5 px/mm. Its silhouette height, H·cos e + D·sin e,
gives a camera elevation of e ≈ 17–20°. Vertical motion therefore scales by
cos e.

**Weight motion.** The weight is followed as a rigid body (x, y and in-plane
rotation) by incremental ECC alignment of its upper part, frame to frame. The
upper part keeps the specimen and the weight's shadow out of the mask. The
weights rotate strongly during impact (the 20 g seats with about 10° of
rocking, the 100 g rolls about 100°), and a translation-only template loses
them. The tracked point is the silhouette centroid, which is close to the
centre of mass.

**Contact.** Contact is the first frame where the centre of mass departs from
a local free-fall parabola by more than 0.5 px (about 0.1 mm). The same rule is
applied to simulations (`compare.simulated_indentation`).

**Compared quantity.** `indentation_rel` is the centre-of-mass displacement
since contact, relative to the specimen base.

### Findings about the recordings

| Case | Impact speed | Offset | Spin | Usable? |
| --- | --- | --- | --- | --- |
| 20 g (brass) | 2.67 m/s | +6 mm | 317 °/s | yes; lands slightly tilted and rocks while seating |
| 50 g (brass) | 2.47 m/s | 0 mm | 48 °/s | yes for the first contact; tumbles off afterwards |
| 100 g (black, no knob) | 2.47 m/s | +14 mm | 27 °/s | **no**: hits the specimen edge and rolls off |
| 200 g (black) | 2.62 m/s | −13 mm | −59 °/s | **no**: tilted edge impact |
| 500 g (black) | 2.65 m/s | −6 mm | ~0 | first contact only; see below |

- **Frame rate is 2999.3 fps** (overlay timestamps). Impact speeds are 2–10%
  below the nominal 2.73 m/s, so simulations use the measured value per case.
- **Perspective.** The camera looks down, so a falling weight also moves away
  from it. Its apparent acceleration in the image is only about 3–4 m/s². A
  free fall with g fixed in the image is therefore wrong over long distances,
  and the tracker fits a free quadratic locally, just before contact.
- **The table is not rigid.** Next to the specimen, the table texture moves by
  several mm (up to about 5 mm up for 20 g) from about 20 ms after impact,
  while the backdrop stays still. Everything after the first contact phase,
  including rebound height, settling and the apparent specimen lift, is
  contaminated by this support motion. Only the first contact phase (0 to
  about 8 ms; 15 ms for 500 g), measured relative to the specimen base, is
  compared with rigid-ground simulations.
- **Out-of-plane tilt can't be measured** from a single side view. Simulations
  get the measured horizontal offset and in-plane spin; tilt stays zero.
- **Rotation is the less precise output.** On synthetic sequences the tracker
  recovers position to about 2% of the motion and rotation to about 20%.
- **Correction to Step 0:** the 4 mm preview estimate for 20 g happens to match
  (3.7 mm). The method was unreliable, though: it used the weight's bottom
  edge, which the specimen rim hides.

## Calibration (`softsim/experiments/calibrate.py`)

`calibrations/ecoflex_00_10_drop.yaml` fits Neo-Hookean E and Rayleigh β (in log
space) to the 20, 50 and 500 g cases at once:

- **Optimizer:** a coarse 4×4 grid, then Nelder-Mead, 50 evaluations.
- **Simulations:** each case uses its measured impact conditions and a 5 mm
  mesh.
- **Cost:** the mean of RMSE divided by the experimental peak over the first
  contact phase.
- **Outputs:** the fitted preset (with provenance), the evaluation history,
  `report.yaml` and plots.

**Result:** E = 54.8 kPa and β = 0.31 ms. This is preset
[`ecoflex_00_10_drop_fit`](../softsim/materials/presets/ecoflex_00_10_drop_fit.yaml).
E is lower than the 91 kPa datasheet-derived starting estimate, and very close
to the datasheet's 100% modulus (55 kPa).

![Simulation vs experiment, before and after calibration](media/calibration_overlay.png)

| Case | Peak exp | Peak, preset (91 kPa) | Peak, fitted | t_peak exp / fitted | NRMSE fitted |
| --- | --- | --- | --- | --- | --- |
| 20 g | 3.74 mm | 1.40 mm | 3.32 mm | 3.3 / 3.2 ms | 15% |
| 50 g | 4.31 mm | 2.16 mm | 4.20 mm | 3.7 / 3.6 ms | 3% |
| 500 g | 4.52 mm | 8.01 mm | 10.97 mm | 2.7 / 6.6 ms | 113% |
| 100 g, 200 g | edge impacts, not fitted | | | | |

![Peak indentation and time to peak vs mass](media/calibration_peaks.png)

### What this does and does not show

- **For light weights the model works.** For 20 g and 50 g (local indentation
  by a weight smaller than the specimen), a Neo-Hookean solid with one Rayleigh
  damping coefficient reproduces the whole first contact (loading, peak and
  unloading) within 3–15%.
- **For the 500 g weight it fails.** That weight is wider than the specimen and
  compresses its whole face. In the recording it stops within 4.5 mm in 2.7 ms,
  while the model needs about 11 mm and 6.6 ms. No single (E, β) fits both
  regimes. The measured peak indentation is almost independent of mass (3.7 →
  4.5 mm for 25× the energy). Candidate explanations, none verified:
  - rate-dependent stiffening of the silicone, which Rayleigh damping can't
    represent;
  - stronger large-strain stiffening than Neo-Hookean;
  - near-incompressibility under full-face confinement (the model uses ν = 0.45);
  - energy absorbed by the compliant table, which is measurable here but not
    modelled.

  These are what the SOFA backend (Yeoh/Ogden, proper viscoelasticity) and a
  support model should address.
- **The fit depends on the mesh.** With the fitted values, the 20 g peak goes
  3.32 → 3.55 → 3.79 mm as the mesh refines from 5 to 3.5 to 2.5 mm. Use the
  preset with `specimen.mesh.size=5mm`, or refit at the target resolution.
- **50 g after first contact isn't modelled.** In reality the weight tumbles
  off after the first contact, so later times shouldn't be compared.
