# Chapter 40: Stereo Vision

This chapter folder contains a compact executable notebook for MIT *Foundations of Computer Vision*, Chapter 40. It follows one focused story: disparity as a depth cue, a synthetic rectified stereo pair, local SSD block matching, quantitative evaluation, parameter tradeoffs, and failure cases.

## MIT Book references used

The notebook now uses local offline copies of four official MIT Chapter 40 figures:

- `assets/mit-book/figure-40-01-titanic.png`
- `assets/mit-book/figure-40-03-random-dot-stereogram.png`
- `assets/mit-book/figure-40-05-triangularization-stereo.png`
- `assets/mit-book/figure-40-07-intensity-matching-failure.png`

Git history for this chapter did not contain local MIT reference images, so these assets were restored from the official MIT chapter page to remove remote notebook-image dependencies. Figures 40.5 and 40.7 are the better fit for the notebook's rectified-stereo geometry and local matching story than the earlier 40.12 and 40.13 choices.

## Generated figures

- `images/01-disparity-depth-intuition.png`
- `images/02-synthetic-stereo-pair.png`
- `images/03-patch-matching-diagnostic.png`
- `images/04-dense-disparity-result.png`
- `images/05-parameter-tradeoffs.png`
- `images/06-failure-cases.png`

## How to run

```bash
python notebooks/CV/2026/spring/final/chapter-40-stereo-vision/build_assets.py
jupyter nbconvert --to notebook --execute notebooks/CV/2026/spring/final/chapter-40-stereo-vision/index.ipynb --output index.executed.ipynb --ExecutePreprocessor.timeout=600
```

## Scope

The notebook assumes a rectified stereo setup and uses an educational integer-pixel SSD matcher. It does not implement calibration, full rectification, essential/fundamental matrix estimation, learned stereo, or global optimization.
