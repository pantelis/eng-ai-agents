# Chapter 40: Stereo Vision

This chapter folder contains an educational stereo-vision notebook that recreates the core ideas from MIT *Foundations of Computer Vision* Chapter 40 using original code-generated figures and a deterministic synthetic stereo scene.

## What the notebook demonstrates

- Why a second view reveals depth through disparity.
- How binocular geometry leads to the rectified stereo depth formula `Z = fB / d`.
- Why epipolar geometry constrains correspondence search.
- How simple local block matching estimates a disparity map.
- How to validate stereo output with ground-truth disparity and failure-case analysis.

## Relation to the MIT Vision Book

The notebook is grounded in [Chapter 40: Stereo Vision](https://visionbook.mit.edu/3d_scene_understanding_stereo.html). It recreates the chapter's main figure themes without copying textbook images:

- Stereo camera setup and triangulation intuition
- Epipolar geometry and rectification
- Disparity-depth relationship
- Pixel-level stereo correspondence on a rectified pair
- Quantitative validation and known stereo failure modes

## Main concepts covered

- Stereo vision intuition
- Binocular geometry
- Disparity
- Depth from disparity
- Epipolar geometry
- Rectified stereo correspondence
- Simple SSD block matching
- Disparity-to-depth reconstruction
- Parameter tradeoffs
- Validation metrics
- Failure cases and limitations

## Generated figures

- `images/01-stereo-setup.png`
- `images/02-epipolar-and-rectification.png`
- `images/03-disparity-depth-relationship.png`
- `images/04-synthetic-stereo-pair.png`
- `images/05-block-matching-results.png`
- `images/06-parameter-and-failure-cases.png`

## How to run

From the repository root:

```bash
python notebooks/CV/2026/spring/final/chapter-40-stereo-vision/build_assets.py
make execute-notebook NOTEBOOK=CV/2026/spring/final/chapter-40-stereo-vision/index.ipynb
```

The first command regenerates the PNG assets and notebook structure. The second uses the repository's notebook execution flow inside the configured container environment.

## Expected outputs

- A synthetic left/right rectified stereo pair
- Ground-truth disparity and depth maps
- A block-matching disparity estimate
- Quantitative metrics and a parameter sweep
- Visualized failure cases for textureless, repeated, and occluded regions

## Validation metrics

- Mean absolute disparity error: `1.526` pixels
- Bad-pixel ratio at 1 pixel: `0.421`
- Reference runtime for the default setting: `2.405` seconds

Best parameter-sweep MAE in the generated grid: `1.081` pixels

## Limitations

- The scene is synthetic and intentionally simple so that ground truth is known.
- The correspondence algorithm is local SSD block matching, not a production stereo method.
- The notebook does not implement subpixel refinement, global regularization, or learned stereo networks.
