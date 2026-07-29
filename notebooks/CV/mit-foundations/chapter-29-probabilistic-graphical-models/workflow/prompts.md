# Prompts — Ch 29 Probabilistic Graphical Models

## 1. Digest
> Extract the computational core: MRF (Ising) smoothness prior and Gibbs sampling;
> sum-product belief propagation on a tree (message rules) and the book's numerical
> example (29.15); loopy BP on the image grid; the stereo scanline BP (29.14). List
> which figures are diagrams vs computable.

## 2. Build
> Reproduce, all offline from the book's images: Gibbs samples of the Ising prior
> across beta (29.3); exact tree BP with the book's exact potentials, checked vs brute
> force (29.15); loopy BP binary denoising and leaf segmentation on the grid (29.3/29.4);
> 1-D BP stereo along a canoe scanline with the six panels of 29.14.

## 3. Fixes
> Stereo: the canoe crops are only approximately rectified — find the global crop
> offset by edge NCC (~77 px) and search disparity in a band around it; use windowed
> NORMALIZED cross-correlation (raw SSD slides on the textureless hull); truncated-
> linear smoothness so the depth edge survives. Track loopy-BP messages in LOG-ODDS
> and update all edges of one orientation at once by array shifts. Checkerboard Gibbs
> sweeps (bipartite grid).

## Lessons
- Verify the claim, not just "it ran": tree BP must equal brute force (assert); loopy
  BP must LOWER pixel error / boundary length / disparity jitter vs the no-prior baseline.
- Real stereo on handheld crops is messy — say so. Reproduce the BEHAVIOUR of 29.14
  (evidence ridge propagates; textureless region filled in), not the exact pixels.
- Keep grids small (~100-150 px) so loopy BP finishes in seconds; the whole notebook is ~30 s.
