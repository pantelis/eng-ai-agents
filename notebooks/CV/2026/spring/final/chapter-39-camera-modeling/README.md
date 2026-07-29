# Chapter 39: Camera Modeling and Calibration

This chapter folder contains a compact executable notebook companion to MIT Vision Book
Chapter 39. It preserves the official section order, keeps all original MIT Figures `39.1`
through `39.17` visible in the notebook, and adds five small generated visualizations to
make the core geometry and calibration ideas more concrete.

## Notebook contents

- All official sections and subsections from Chapter 39.
- All 19 original MIT figure assets stored locally under `assets/mit-book/`.
- Five deterministic generated figures under `images/`.
- Final section-coverage and figure-coverage tables.

## Original vs generated material

- Original MIT Vision Book figures are shown from local files in `assets/mit-book/` and are
  labeled explicitly as original textbook figures.
- Generated figures are saved under `images/` and are labeled either as computational
  reconstructions or as supplemental visualizations.

## Environment

- Registered notebook environment: `torch.dev.cpu`
- Designed to run on CPU from a fresh kernel

## How to build and execute

```bash
python3 notebooks/CV/2026/spring/final/chapter-39-camera-modeling/build_assets.py
jupyter nbconvert --to notebook --execute notebooks/CV/2026/spring/final/chapter-39-camera-modeling/index.ipynb --output index.executed.ipynb --ExecutePreprocessor.timeout=600
```

## Output locations

- Original MIT assets: `assets/mit-book/`
- Generated notebook figures: `images/`
- Executed notebook: `index.executed.ipynb`

## Attribution

Original textbook figures are from the MIT Vision Book, *Foundations of Computer Vision*,
Chapter 39: Camera Modeling and Calibration.

## Known limitations

- The generated visuals are intentionally selective and do not reproduce every textbook
  derivation.
- The DLT example uses synthetic correspondences rather than the full office-scene recovery
  pipeline from the textbook.
