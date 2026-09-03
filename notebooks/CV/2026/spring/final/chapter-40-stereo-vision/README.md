# Chapter 40: Stereo Vision

        This chapter folder contains a chapter-aligned executable notebook for MIT
        *Foundations of Computer Vision*, Chapter 40.

        The notebook now keeps all MIT Figures `40.1` to `40.14` visible in the markdown and adds
        eighteen code-generated teaching figures covering stereo geometry, matching, failure modes,
        epipolar geometry, and evaluation.

        ## Generated figures

        - `images/01-stereo-cues.png`
- `images/02-rectified-stereo-geometry.png`
- `images/03-depth-vs-disparity.png`
- `images/04-depth-error-sensitivity.png`
- `images/05-baseline-sensitivity.png`
- `images/06-pixel-vs-patch-matching.png`
- `images/07-cost-volume-slices.png`
- `images/08-winner-takes-all-disparity.png`
- `images/09-patch-size-sweep.png`
- `images/10-max-disparity-sweep.png`
- `images/11-textureless-failure.png`
- `images/12-repetitive-pattern-failure.png`
- `images/13-occlusion-and-left-right-check.png`
- `images/14-subpixel-refinement.png`
- `images/15-epipolar-constraint.png`
- `images/16-before-after-rectification.png`
- `images/17-disparity-depth-error-maps.png`
- `images/18-runtime-accuracy-tradeoff.png`

        ## How to run

        ```bash
        python3 notebooks/CV/2026/spring/final/chapter-40-stereo-vision/build_assets.py
        jupyter nbconvert --to notebook --execute notebooks/CV/2026/spring/final/chapter-40-stereo-vision/index.ipynb --output index.executed.ipynb --ExecutePreprocessor.timeout=600
        ```
