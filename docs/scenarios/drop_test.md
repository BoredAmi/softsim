# Drop test

A rigid weight falls onto a soft specimen resting on a table
(`softsim/scenarios/drop_test.py`, configs in `scenarios/drop_test/`).

## Reference dataset

Czopek P., Superczynska P., Marcinkowska A., Gajewski P., Bernat J.,
*Records of falling mass on Ecoflex 00-10 cylinder*, Zenodo,
doi:[10.5281/zenodo.23185808](https://doi.org/10.5281/zenodo.23185808), CC-BY-4.0.

- **Specimen:** Ecoflex 00-10 cylinder, D 40 mm, H 17 mm, 19.49 g. That implies
  912–917 kg/m³ against 1040 kg/m³ on the datasheet. The measured mass is used,
  and the mismatch is reported as a finding.
- **Drop:** from 38 cm, so the impact speed is about 2.73 m/s (air drag neglected).
- **Weights:**

| Weight | Diameter | Height | Notes |
| --- | --- | --- | --- |
| 20 g | 22.5 mm | 5.6 mm | brass |
| 50 g | 27 mm | 9.9 mm | brass |
| 100 g | 32 mm | 18 mm | black, no knob; hits the specimen edge |
| 200 g | 40 mm | 24 mm | black; tilted edge impact |
| 500 g | 43 mm | 38 mm | black, mushroom knob, wider than the specimen |

### What we measured from the data (Step 0)

- **Frame rate is 2999.3 fps.** The overlay `frame/017866 … T=<s>` gives
  200 frames in 66.68 ms, and the zip frames are consecutive camera frames. The
  zips contain 800×800 RGB TIFFs (698 / 1106 / 786 / 1156 / 1506 frames for
  20–500 g); the mp4 previews are the same frames, lossy, played at 12 fps.
- **Scale is about 5.3 px/mm** (about 0.19 mm/px). The camera is slightly
  elevated, so the top face is visible and the weight's bottom edge is partly
  hidden. Track the weight's top edge or knob instead (it is rigid, so it
  moves the same distance).
- **20 g preview, rough tracking:** contact to peak takes about 7–9 frames
  (2.3–3 ms), peak indentation is about 4 mm, then the weight rebounds about
  5 mm and settles.
- **500 g contact sheet:** the weight lands tilted, bounces, then topples off
  the cylinder. The specimen leaves the table about 60 ms after contact. (M3
  wrongly attributed this to the table moving; the specimen does hop, pulling
  up the cloth it sticks to. See [experiments.md](../experiments.md#after-the-first-contact-tack-and-the-hopping-specimen).)
- **Full tracking results** (contact, impact speed, offset, spin, support
  motion) are in [experiments.md](../experiments.md).

## Scenario

The simulation starts with the weight's lowest point `drop.gap` (0.5 mm) above
the specimen, moving at the free-fall speed for the remaining height.
Parameters: weight preset or custom body, drop height or initial speed, xy
offset, tilt about x and y, and initial angular velocity. The specimen can be
`base: free` (on the table, with friction) or `base: bonded` (bottom face pinned).

**Support (optional).** With `support.enabled: true` the specimen sits on a
thin carrier plate (default 80 × 80 × 2 mm, 5 g) that can lift off the table
against a spring and damper (`support.stiffness`, `support.damping`). It is a
lumped model of the cloth in the recordings, which the tacky specimen drags up
when it hops. It needs a backend with `mount: slider_z` (MuJoCo). Give the
specimen material an `adhesion` so it sticks to the plate and the weight:

```yaml
support: {enabled: true, stiffness: 32.9 N/m, damping: 0.346 N*s/m, mass: 45.3 g}
specimen:
  material: ecoflex_00_10_drop_fit_tacky   # the drop fit plus fitted tack: strength 14.6 kPa, range 0.29 mm
```

These are the values fitted to the recordings (see
[experiments.md](../experiments.md#after-the-first-contact-tack-and-the-hopping-specimen));
`sweeps/fitted_tack_support.yaml` runs all five weights with them.

| Metric | Meaning |
| --- | --- |
| `indentation` | undeformed contact surface minus the weight's bottom-centre height |
| `compression` | downward displacement of the specimen's top-centre vertex |
| `bulge_diameter` | twice the maximum radial vertex distance |
| `specimen_min_z`, `specimen_max_z` | lift-off and top height |
| `contact_force_z` | specimen on weight (negative while adhesion pulls) |
| `ground_force_z` | ground (or support plate) on specimen |
| `support_z` | support plate height (with `support.enabled`) |
| `weight_z`, `weight_vz`, `weight_tilt` | weight state |

Summary: `peak_indentation`, `t_to_peak`, `first_contact_duration`,
`rebound_speed`, `restitution`, `peak_ground_force`, `max_bulge_diameter`,
`specimen_lift_off`, `final_indentation`, `settling_time`.

Action `drop(preset=, mass=, height=, offset=, tilt=)` rebuilds the scene and
drops again.

## M1 results: uncalibrated (superseded by M3 calibration)

These use the `ecoflex_00_10` preset (stable Neo-Hookean, E = 91 kPa from the
datasheet, ν = 0.45, Rayleigh β = 2 ms placeholder), a 3.5 mm mesh, dt = 0.1 ms,
contact timeconst 0.2 ms, 250 ms per run
(`softsim sweep scenarios/drop_test/sweeps/all_weights.yaml --jobs 5`):

| Weight | Peak indentation [mm] | t to peak [ms] | 1st contact [ms] | Rebound [m/s] | e | Peak ground force [N] | Max bulge Ø [mm] | Lift-off [mm] | Wall [s] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 20 g | 1.97 | 2.0 | 5.0 | 0.49 | 0.18 | 36 | 42.0 | 1.3 | 68 |
| 50 g | 3.45 | 3.0 | 7.0 | 0.69 | 0.25 | 57 | 43.7 | 3.3 | 66 |
| 100 g | 4.72 | 3.6 | 8.0 | 0.83 | 0.30 | 87 | 46.0 | 7.0 | 63 |
| 200 g | 6.53 | 4.4 | 10.2 | 0.97 | 0.36 | 148 | 48.7 | 11.0 | 53 |
| 500 g | 10.29 | 6.4 | 14.6 | 1.17 | 0.43 | 291 | 53.6 | 16.1 | 47 |

(Five runs in parallel on 8 cores; a single run takes about 40–50 s.)

Compared with the 20 g preview, the model is **too stiff**: about 2 mm against
about 4 mm of peak indentation. It is also **too elastic**: it rebounds at
0.49 m/s against roughly 0.3 m/s estimated from the rebound height. The
restitution *rising* with mass is a signature of Rayleigh damping (heavier
weights mean slower impacts, which are damped less). Whether real Ecoflex shows
this is a key question for the M3 comparison. **M3 calibration** ([experiments.md](../experiments.md)) fits E = 54.8 kPa and
β = 0.31 ms (preset `ecoflex_00_10_drop_fit`, 5 mm mesh). That matches the 20 g
and 50 g first impacts within 3–15%, but not the 500 g, which in reality is far
stiffer and faster than Neo-Hookean + Rayleigh allows. The 100 g and 200 g
recordings are off-centre edge impacts and can't be used. **M4** repeated the
fit on SOFA with an Ogden material: 20 and 50 g within 3–11%, 500 g still not
reproduced ([experiments.md](../experiments.md#backends-and-material-models-m4)).
