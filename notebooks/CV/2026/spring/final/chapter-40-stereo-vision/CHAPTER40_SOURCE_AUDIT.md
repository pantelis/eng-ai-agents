# Chapter 40 Source Audit

This audit compares the official local MIT source at `reference/chapter40/chapter40.html`
against the current notebook at
`notebooks/CV/2026/spring/final/chapter-40-stereo-vision/index.ipynb`.

The notebook is now organized by the chapter’s actual progression, embeds the pedagogically
useful MIT figures directly in markdown, and labels every generated image as either a faithful
computational reconstruction or a supplemental notebook visualization.

## Chapter headings from the MIT source

- `40 Stereo Vision`
- `40.1 Introduction`
- `40.2 Stereo Cues`
- `40.2.1 How Far Away Is a Boat?`
- `40.2.2 Depth from Image Disparities`
- `40.2.3 Building a Stereo Pinhole Camera`
- `40.3 Model-Based Methods`
- `40.3.1 Triangulation`
- `40.3.2 Stereo Matching`
- `40.3.2.1 Finding image features`
- `40.3.2.2 Local image descriptors`
- `40.3.2.3 Interpolation between feature matches`
- `40.3.3 Constraints for Arbitrary Cameras`
- `40.3.4 The Essential and Fundamental Matrices`
- `40.3.4.1 The fundamental matrix`
- `40.3.4.2 Estimation of the essential/fundamental matrix`
- `40.3.4.3 Finding the epipoles`
- `40.3.4.4 Epipolar lines: The game`
- `40.3.5 Image Rectification`
- `40.4 Learning-Based Methods`
- `40.4.1 Output Representation`
- `40.4.2 Two-Stage Networks`
- `40.5 Evaluation`
- `40.6 Concluding Remarks`

## Notebook section order

- `40.1 Introduction`
- `40.2 Stereo Cues`
- `40.2.1 How Far Away Is a Boat?`
- `40.2.2 Depth from Image Disparities`
- `40.2.3 Building a Stereo Pinhole Camera`
- `40.3 Model-Based Methods`
- `40.3.1 Triangulation`
- `40.3.2 Stereo Matching`
- `40.3.2.1 Finding image features`
- `40.3.2.2 Local image descriptors`
- `40.3.2.3 Interpolation between feature matches`
- `40.3.3 Constraints for Arbitrary Cameras`
- `40.3.4 The Essential and Fundamental Matrices`
- `40.3.4.4 Epipolar lines: The game`
- `40.3.5 Image Rectification`
- `40.4 Learning-Based Methods`
- `40.4.1 Output Representation`
- `40.4.2 Two-Stage Networks`
- `40.5 Evaluation`
- `40.6 Concluding Remarks`

## Figure-by-figure audit

| Figure | Official caption summary | Actual source asset | Visibly embedded in notebook | Original or reconstructed | Notebook section | Explanatory markdown present | Equation/context present | Remaining omission | Coverage judgment |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `40.1` | Titanic stereo anaglyph and red/cyan viewing setup. | `assets/mit-book/figure-40-01-titanic.png` | `Yes` | Original MIT figure | `40.1 Introduction` | `Yes` | `Yes` | No computational analysis of anaglyph formation, which is acceptable because the chapter uses it mainly as motivation. | Strong original-figure coverage. |
| `40.2` | Two boat-distance constructions: one-view horizon geometry and two-view triangulation. | `assets/mit-book/figure-40-02-boats.png` | `Yes` | Original MIT figure plus faithful reconstruction in `images/01-stereo-cues.png` | `40.2.1 How Far Away Is a Boat?` | `Yes` | `Yes` | None. | Strong original-plus-reconstruction coverage. |
| `40.3` | Random-dot stereogram showing depth from disparity alone. | `assets/mit-book/figure-40-03-random-dot-stereogram.png` | `Yes` | Original MIT figure plus faithful reconstruction in `images/01-stereo-cues.png` | `40.2.2 Depth from Image Disparities` | `Yes` | `Yes` | None. | Strong original-plus-reconstruction coverage. |
| `40.4` | Anaglyph pinhole camera with pinholes, filters, projection plane, and resulting image. | `assets/mit-book/figure-40-04-anaglyph-camera.png` | `Yes` | Original MIT figure | `40.2.3 Building a Stereo Pinhole Camera` | `Yes` | `Yes` | No physical build experiment in code. The notebook explicitly treats it as chapter context rather than executable reconstruction. | Good original-figure coverage; computation intentionally omitted. |
| `40.5` | Rectified stereo geometry with focal length `f`, baseline `T`, point `P`, image points, and similar triangles leading to Equation `40.1`. | `assets/mit-book/figure-40-05-triangulation-stereo.png` | `Yes` | Original MIT figure plus faithful reconstruction in `images/02-rectified-stereo-geometry.png` | `40.3.1 Triangulation` | `Yes` | `Yes` | None. | Strong original-plus-reconstruction coverage. |
| `40.6` | Real stereo office pair with highlighted corresponding features and displacements. | `assets/mit-book/figure-40-06-office-left.jpg`, `assets/mit-book/figure-40-06-office-right.jpg` | `Yes` | Original MIT figure plus supplemental related visualization in `images/02-rectified-stereo-geometry.png` | `40.3.2 Stereo Matching` | `Yes` | `Yes` | The generated figure is synthetic, so it remains explicitly supplemental. | Good original coverage with honest supplemental support. |
| `40.7` | Intensity-based matching failure in disparity space, smoothing, and ground-truth comparison. | `assets/mit-book/figure-40-07-intensity-matching-failure.jpg` | `Yes` | Original MIT figure plus faithful reconstruction in `images/03-intensity-matching-failure.png` | `40.3.2 Stereo Matching` | `Yes` | `Yes` | None. | Strong original-plus-reconstruction coverage. |
| `40.8` | Harris features in the stereo pair and the resulting depth image. | `assets/mit-book/figure-40-08-points-left.jpg`, `assets/mit-book/figure-40-08-points-right.jpg`, `assets/mit-book/figure-40-08-depth.jpg` | `Yes` | Original MIT figure plus supplemental notebook visualization in `images/04-feature-based-stereo.png` | `40.3.2.1 Finding image features` | `Yes` | `Yes` | The notebook does not reproduce the exact office-feature output, so the generated figure remains supplemental. | Good original coverage with honest supplemental support. |
| `40.9` | Oriented features and SIFT-style descriptors. | `assets/mit-book/figure-40-09-oriented-features.jpg`, `assets/mit-book/figure-40-09-sift-descriptor.jpg` | `Yes` | Original MIT figure plus supplemental notebook visualization in `images/04-feature-based-stereo.png` | `40.3.2.2 Local image descriptors` | `Yes` | `Yes` | The notebook gives a local orientation histogram, not a full SIFT implementation. | Good original coverage with partial computational support. |
| `40.10` | Candidate-match ambiguity for one point under arbitrary camera geometry. | `assets/mit-book/figure-40-10-correspondence-ambiguity.png` | `Yes` | Original MIT figure plus faithful reconstruction in `images/05-epipolar-geometry.png` | `40.3.3 Constraints for Arbitrary Cameras` | `Yes` | `Yes` | None. | Strong original-plus-reconstruction coverage. |
| `40.11` | A viewing ray in camera 1 projects to a line in camera 2. | `assets/mit-book/figure-40-11-epipolar-ray.png` | `Yes` | Original MIT figure plus faithful reconstruction in `images/05-epipolar-geometry.png` | `40.3.3 Constraints for Arbitrary Cameras` | `Yes` | `Yes` | None. | Strong original-plus-reconstruction coverage. |
| `40.12` | Epipolar plane, epipolar lines, and epipoles. | `assets/mit-book/figure-40-12-epipolar-geometry.png` | `Yes` | Original MIT figure plus faithful reconstruction in `images/05-epipolar-geometry.png` | `40.3.3 Constraints for Arbitrary Cameras` | `Yes` | `Yes` | None. | Strong original-plus-reconstruction coverage. |
| `40.13` | Epipolar-line intuition game and camera-pair matching. | `assets/mit-book/figure-40-13-epipolar-game.png` | `Yes` | Original MIT figure | `40.3.4.4 Epipolar lines: The game` | `Yes` | `Yes` | No executable game solver, which is acceptable because the chapter itself presents this mainly as intuition-building. | Good original-figure coverage. |
| `40.14` | CNN stereo pipeline with rectification, feature extraction, cost volume, cost aggregation, and disparity estimate. | `assets/mit-book/figure-40-14-stereo-cnn-block-diagram.jpg` | `Yes` | Original MIT figure | `40.4.2 Two-Stage Networks` | `Yes` | `Yes` | No implemented CNN pipeline; the notebook labels this section as conceptual chapter context and evaluates only the classical baseline. | Good original-figure coverage with explicit implementation limits. |

## Important prose ideas now restored in the notebook

- Stereo is introduced as a two-part problem: geometry plus correspondence.
- Viewpoint change creates image displacement, and that displacement is a depth cue.
- Binocular perception is illustrated both with the Titanic anaglyph and the random-dot stereogram.
- Triangulation is developed from the boat example before being specialized to stereo cameras.
- Rectified stereo assumptions are named explicitly: calibrated cameras, horizontal baseline, and `y_L = y_R`.
- Disparity is defined and interpreted as inversely related to depth in Equation `40.1`.
- Intensity matching failure is explained using textureless regions, repeated patterns, noise, specularities, and occlusion.
- Feature-based correspondence is separated into feature detection, local descriptors, and interpolation/regularization.
- Arbitrary-camera geometry is explained using rays, epipolar lines, epipolar planes, and epipoles.
- The essential and fundamental matrices are presented as the algebraic form of the epipolar constraint.
- Rectification is described as the step that converts epipolar-line search back into rowwise search.
- The learning-based section is restored in the chapter’s own terms: disparity representation, cost volumes, and two-stage networks.
- The evaluation and concluding sections are represented honestly, including the chapter’s caution that stereo remains difficult.

## Current omissions and limits

- The notebook does not implement a physical anaglyph pinhole build for Figure `40.4`.
- The notebook does not implement a full SIFT descriptor or a CNN stereo network; both are described and labeled honestly as chapter context.
- The feature-based generated figure is supplemental rather than a faithful recreation of the exact office-scene outputs in Figure `40.8`.
- The evaluation remains a classical SSD baseline rather than a benchmark-complete stereo study.

## Final audit judgment

The current notebook functions as a self-contained educational adaptation of MIT Chapter 40.
Every official Figure `40.1` through `40.14` is now visibly present in the notebook, the
surrounding markdown explains why each figure matters, and the generated notebook figures are
distinguished clearly from the original MIT source images.
