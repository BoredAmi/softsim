# Adding a dataset

softsim validates against **public** data only, and records where every number came from. A dataset is a
YAML file in `datasets/`. It describes the data; it doesn't contain it.

## Rules

- **Licence first.** Record the licence and the DOI. If the licence allows redistribution and the data is small,
  it may live in `data/processed/`. Otherwise the file lists the download URLs and checksums, and
  `softsim fetch` downloads into `local_dir` (gitignored). Non-commercial data (CC BY-NC) is never committed.
- **Checksums.** Every downloaded file has an `md5`; `softsim fetch` checks it.
- **Say what was measured, not what you hope was measured.** The comment at the top of the file states the
  protocol, the number of specimens and repeats, what isn't known (strain rate, contact point,
  uncertainty), and any reading or digitising error.
- **Digitised figures** are allowed when no table exists. Mark them `digitised` and say from which figure.

## Material tests (tension, compression, relaxation)

Used by `softsim fit-material` and by calibration `material_tests:` terms. Example:
`datasets/roels2025_elastomers.yaml`.

```yaml
kind: material_tests
name: my_lab_2026
title: ...
doi: 10.5281/zenodo.0000000
license: CC-BY-4.0
local_dir: data/raw/my_lab_2026
format: xy_csv
sources:
  tension: {url: "https://...", file: tension.zip, md5: ..., member_prefix: tension/}
materials:
  my_silicone:
    product: Vendor name and grade
    tests:
      - {source: tension, kind: tension, members: [specimen1.csv, specimen2.csv]}
```

Then `softsim fetch datasets/my_lab_2026.yaml` and
`softsim fit-material datasets/my_lab_2026.yaml my_silicone`.

## A robot's response (pressure → angle, force, ...)

A table with the source and what each column is. Example: `datasets/wang2026_finger.yaml` (pressure and
measured chord angle of a cast finger). The data is compared with any scenario output by `softsim fit` (the
`data:` block of a fit file, [experiments](experiments.md)) and by a `dataset` truth in a benchmark entry
(`benchmarks/wang_finger.yaml`).

## Recordings (video)

The drop-test dataset (`datasets/czopek2026_ecoflex_drop.yaml`) describes the camera (frame rate, scale,
overlay layout), the specimen and each case. `softsim track` turns the videos into tracks in
`data/processed/`, which `softsim calibrate` compares with simulations ([experiments](experiments.md)).

## Checklist

- [ ] DOI, licence, URL, checksums
- [ ] protocol, specimens, repeats and unknowns written in the file's comment
- [ ] a loader test, or a test that uses it (skipped when the data isn't downloaded)
- [ ] a benchmark entry or a fit/calibration file that uses it, so it does some work
