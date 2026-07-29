# Workflow — How this chapter was produced

Companion notebook for Chapter 29 (Probabilistic Graphical Models), following the
Ch 17/18/19/27/28/38/42/47 pattern. 5 figures, arranged as a single arc:
prior -> exact inference -> loopy inference -> two applications.

## Source images — the book's own photographs
- **canoe_left.png / canoe_right.png** — the left/right scanline-region crops of the
  book's canoe stereo pair (lcanoe2.jpg / rcanoe2.jpg), bundled under `assets/`;
  used for the scanline stereo demo (29.14).
- **leaf.png** — the book's leaf image (leaf2.jpg, Fig 29.4, "image to be segmented"),
  bundled under `assets/`.
Credit: visionbook.mit.edu.

## Figures reproduced (5)
| Fig | Content |
|---|---|
| 29.3 | MRF (Ising) prior sampled by Gibbs sampling; beta sweep noise -> critical -> smooth |
| 29.15 | exact sum-product BP on the book's tree; marginals == brute force (assert) |
| 29.3 | loopy BP for binary denoising on the grid MRF; 20% -> 0.8% pixel error |
| 29.4 | two-label MRF segmentation of the book's leaf by loopy BP |
| 29.14 | 1-D BP stereo along a canoe scanline: evidence, L->R / R->L messages, posterior |

## Implementation
- **Gibbs sampler** (`ising_gibbs`): checkerboard sweeps — the grid is bipartite, so
  all one-colour pixels are conditionally independent given the other colour and are
  resampled at once from `P(x_i=+1) = sigmoid(2*beta*neighbour_sum)`.
- **Tree BP** (29.15): the book's EXACT potentials (psi12, psi23, phi2, y2=0). Leaf
  messages fold into x2, then back out to x1/x3; marginals are checked against a full
  2^3 brute-force enumeration (`np.allclose` assert passes).
- **Loopy BP for binary labels** (`loopy_bp_binary`): messages tracked as
  **log-odds**; the Ising-edge update is the closed form
  `logaddexp(J+b, -J) - logaddexp(-J+b, J)`, applied to every edge of one orientation
  at once by array shifts. Used for denoising (data log-odds from the flip model) and
  for leaf segmentation (data log-odds from a greenness colour score).
- **Stereo scanline BP** (29.14): each position is a node, label = disparity. Local
  evidence = `1 - normalized cross-correlation` of a windowed patch between the left
  scanline and the right scanline shifted by the disparity; smoothness = truncated
  linear (so depth edges survive). A forward and a backward sum-product sweep combine
  with the evidence into the marginalized posterior; the six panels mirror the book.

## Numerical checks
- 29.15: `assert np.allclose(bp, brute)` — tree BP is exact.
- 29.3 denoising: pixel error 20% -> ~0.8% after loopy BP.
- 29.4 segmentation: boundary length ~3400 -> ~1300 (per-pixel threshold -> loopy BP).
- 29.14 stereo: disparity jitter drops from winner-take-all to BP.

## Validation
```bash
docker compose run --rm torch.dev.cpu \
  bash -c "python scripts/execute_notebook.py \
    CV/mit-foundations/chapter-29-probabilistic-graphical-models/index.ipynb"
```
13 cells, ~30 s, 5 figures, no errors. CPU-only.
> Windows: prefix `MSYS_NO_PATHCONV=1` and pass `-e PATH=/workspaces/eng-ai-agents/.venv/bin:/usr/local/bin:/usr/bin:/bin`.

## Honest caveats
- The stereo matcher is deliberately simple (windowed normalized correlation + a
  truncated-linear smoothness) and the canoe crops are only approximately rectified,
  so 29.14 reproduces the **behaviour** of the book's figure — a sharp evidence ridge
  in the textured grass that BP propagates, ambiguity on the textureless hull that BP
  fills in smoothly — not the book's exact pixels or parameters.
- Loopy BP on the image grid is approximate (the grid has loops); it is exact only on
  the tree of 29.15.
