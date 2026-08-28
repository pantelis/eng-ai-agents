# Chapter 41 Homographies

This notebook is an **educational Chapter 41 homography companion**. It presents a synthetic, runnable PyTorch walkthrough of **homography intuition, homogeneous coordinates, normalized DLT, RANSAC, reprojection error, and inverse warping** with a strong beginner-facing visual teaching emphasis.

## Teaching focus

The notebook stays intentionally narrow and honest in scope:

- it is a **synthetic tutorial example** rather than a real-image feature-matching pipeline,
- it is CPU-friendly and reproducible,
- it uses runnable PyTorch implementations instead of hiding the logic behind a black box,
- and it is meant to clarify planar projective geometry, not to act like a production panorama or image-stitching system.

## Key visualizations

The notebook includes beginner-friendly visual support for the main Chapter 41 ideas:

- homography intuition with a source plane, warped destination plane, and four-point/non-collinear intuition,
- a homogeneous-coordinate diagram showing `(x, y) -> (x, y, 1) -> (u, v, w) -> (u / w, v / w)`,
- a DLT pipeline diagram from correspondences to SVD to the recovered homography,
- a reprojection-error figure showing observed vs predicted destination points and the error arrows,
- a RANSAC workflow diagram and an inlier/outlier classification figure,
- the grid transformation visualization,
- a checkerboard perspective-correction and rectification figure,
- parameter-study plots for noise, outlier ratio, and threshold tradeoffs,
- and failure-case plots for nearly collinear points and extreme outlier contamination.

Key teaching figures are also saved under [`images/`](./images/), including:

- `fig41_01_homography_intuition.png`
- `fig41_02_homogeneous_coordinates.png`
- `fig41_03_dlt_pipeline.png`
- `fig41_04_ransac_workflow.png`
- `fig41_05_checkerboard_rectification.png`
- `fig41_06_grid_transformation.png`
- `fig41_07_reprojection_error.png`
- `fig41_08_ransac_inliers_outliers.png`
- `fig41_09_parameter_study.png`
- `fig41_10_failure_cases.png`

## Figure guide

Each major figure is meant to answer one quick beginner question:

- `fig41_01_homography_intuition.png`: shows one planar view mapped into another; notice that straight lines stay straight while spacing and apparent parallelism change.
- `fig41_01_homography_intuition.png`: is the conceptual plane-to-plane view; notice the left panel is a source plane and the right panel is the same plane after a homography.
- `fig41_02_homogeneous_coordinates.png`: shows the `(x, y) -> (x, y, 1) -> H(x, y, 1) -> (u, v, w) -> (u / w, v / w)` pipeline; notice that dividing by `w` is the step that returns to the image plane.
- `fig41_03_dlt_pipeline.png`: shows the estimation flow from correspondences to SVD to validation; notice that the algorithm is a short linear-algebra pipeline rather than a black box.
- `fig41_04_ransac_workflow.png` and `fig41_08_ransac_inliers_outliers.png`: show how RANSAC searches for a consensus set; notice that robust fitting depends on identifying a trustworthy subset, not explaining every point.
- `fig41_05_checkerboard_rectification.png`: shows original, perspective-distorted, and rectified views side by side; notice how inverse warping restores a fronto-parallel layout.
- `fig41_06_grid_transformation.png`: is the concrete before/after coordinate view; notice that it focuses on the actual point locations under `H`, not just the plane-mapping idea.
- `fig41_07_reprojection_error.png`: shows predicted vs observed destination points; notice that the error arrow is the geometric quantity RANSAC thresholds.
- `fig41_09_parameter_study.png`: shows contamination and threshold tradeoffs; notice where plain DLT breaks down and how threshold choice moves precision versus recall.
- `fig41_10_failure_cases.png`: shows weak geometric coverage and overwhelming outliers; notice why some failures are easier to understand visually than numerically.

## What students can learn

Students can use the notebook to understand:

- why a homography has 8 degrees of freedom instead of 9,
- why four non-collinear point correspondences are the minimal requirement for DLT,
- how normalized DLT builds and solves a homogeneous linear system,
- how reprojection error measures fit quality,
- how RANSAC uses repeated minimal samples plus a threshold to reject outliers,
- how inverse warping rectifies a perspectively distorted planar image,
- and why degeneracy and high outlier ratios remain important limitations.

## Validation metrics

The notebook reports and visualizes:

- scale-normalized homography matrix error against the ground truth,
- mean reprojection error over correspondences,
- RANSAC precision and recall for inlier classification,
- RANSAC robustness trends under increasing outlier ratio,
- threshold precision / recall tradeoffs,
- and checkerboard rectification mean absolute error over valid rectified pixels.

## Limitations

This remains an **educational homography demo**:

- all data are synthetic,
- the examples assume a planar scene,
- there is no real feature matching,
- there is no lens distortion, occlusion, or lighting variation in the checkerboard example,
- and it is not a production panorama or image-stitching system.

## How to run

From the project root, open the notebook with a Python environment that already has `torch` and `matplotlib`:

```bash
jupyter notebook notebooks/CV/mit-foundations/chapter-41-homographies/index.ipynb
```

The notebook is designed to execute end-to-end on CPU.
