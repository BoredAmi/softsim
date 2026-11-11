# Tutorial: change a material

A material is a **card**: a constitutive law with its constants, optional rate dependence (Prony series),
friction and tack, plus where the numbers come from. The full format is in [materials](../materials.md).

## Look at a card

```bash
softsim materials                          # every preset that is visible
softsim material show ecoflex_00_30        # moduli, wave speed, stable time step per mesh size, stresses
softsim material plot ecoflex_00_30 dragon_skin_10 --tests uniaxial,simple_shear   # needs the fem extra
```

## Change it in YAML

Use a preset by name, or a preset with some values changed, or write the card inline:

```yaml
body:
  material: ecoflex_00_30
  # or: material: {preset: ecoflex_00_30, elastic: {youngs_modulus: 60 kPa}}
  # or: material: {density: 1070 kg/m^3, elastic: {model: yeoh, c10: 12 kPa, c20: 0.5 kPa, c30: 0, poisson_ratio: 0.49}}
```

## Change it from the command line

`--set` takes dotted keys, list indices and units:

```bash
softsim run scenarios/actuation/cable_cantilever.yaml \
    --set body.material.elastic.youngs_modulus="400 kPa" --set body.mesh.size=2.5mm
softsim params scenarios/actuation/cable_cantilever.yaml | grep material
```

## Change it from Python

```python
from softsim import Simulation
from softsim.materials import load_material

card = load_material("ecoflex_00_30", overrides=["elastic.youngs_modulus=60 kPa"])

sim = Simulation("scenarios/drop_test/cases/w020.yaml")
sim.set_material("specimen", {"preset": "ecoflex_00_10", "elastic": {"youngs_modulus": "40 kPa"}})
sim.set_material("specimen", None, ["elastic.youngs_modulus=45 kPa"])   # change single values
```

From ROS 2, the `set_material` service takes the same overrides ([ROS 2](../ros2.md)).

## A law the backend doesn't have

MuJoCo has (stable) Neo-Hookean and StVK only. A Yeoh card on MuJoCo is refused, with a hint: project the card
onto a law the backend has, over the strain range you need, and the projection error is recorded in the new card:

```bash
softsim material project ecoflex_00_30_roels2025 --to neo_hookean --out eco30_nh.yaml
```

## Fit your own card

From tension or compression data registered as a dataset ([add a dataset](../adding_a_dataset.md)):

```bash
softsim fetch datasets/roels2025_elastomers.yaml                 # downloads and checks the archives
softsim fit-material datasets/roels2025_elastomers.yaml ecoflex_00_30 --strain-min=-0.25 --strain-max 1.5
```

It fits several laws, ranks them by AIC/BIC, and writes cards with uncertainty and validity ranges.
