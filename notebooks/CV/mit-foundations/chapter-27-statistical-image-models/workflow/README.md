# Workflow — How this chapter was produced

Companion notebook for Chapter 27 (Statistical Image Models), following the
Ch 17/18/19/38/42/47 pattern. 10 figures, each with a numerical check.

## Source images — the book's own images
Every photo is one of the book's own images:
- **wheel** and **MIT dome** — loaded from visionbook.mit.edu (standalone files).
- **street scene, autumn leaves, colourful (Burano) houses, doorway building,
  hair texture, Barcelona building** — cropped from the book's own composite
  figures and bundled under `assets/` (the book has no standalone files for
  these). Credit: visionbook.mit.edu.

The maths figures (generalized Laplacian, coring) and synthetic samples (1/f
clouds, star field) are computed. Per-figure the exact book image is used:
27.10 wheel/dome/leaves, 27.6/27.8/27.15 street, 27.16 noise/doorway/hair,
27.14 Barcelona building, 27.23 Burano houses (colour).

## Figures reproduced (10)
| Fig | Content | Check |
|---|---|---|
| 27.10 | 1/f power law: images, log\|FFT\|, radial spectra vs 1/w^a | natural hugs power law; noise flat |
| 27.11 | cloud samples from a 1/(1+w^1.5) spectrum + random phase (gray + RGB) | — |
| 27.6 | independent-pixel model fails: iid sample with matched histogram | structure lost (only star field survives) |
| 27.8 | pixel-pair correlation decays with distance | corr(d) drops |
| 27.15 | intensity histogram (broad) vs derivative histogram (cusp) | derivative kurtosis ~40 (Gaussian = 3) |
| 27.17 | generalized-Laplacian shapes r = 0.1, 1, 2, 10 | — |
| 27.16 | band-pass histograms (log): noise stays Gaussian, images heavy-tailed | Gaussian fit misses the tails |
| 27.14 | Wiener denoising under the 1/f Gaussian prior | RMSE 0.12 -> 0.043 |
| 27.21 | wavelet coring (soft-threshold shrinkage) | — |
| 27.23 | non-local means denoising | RMSE 0.099 -> 0.046 |

## Notes
- The Wiener gain S/(S+sigma^2) for a 1/f^{2a} prior is written via the crossover
  frequency w0 (signal power = noise power) as H = 1/(1 + (w/w0)^{2a}) — a
  parametric low-pass that preserves structure while cutting noise.
- NLM uses `skimage.restoration.denoise_nl_means` with the known sigma
  (`estimate_sigma` needs PyWavelets, which the container lacks).
- Figures are computed from the equations, so no drag editor was needed.

## Validation
```bash
docker compose run --rm torch.dev.cpu \
  bash -c "python scripts/execute_notebook.py \
    CV/mit-foundations/chapter-27-statistical-image-models/index.ipynb"
```
23 cells, ~66 s, 10 figures, no errors. CPU-only.
> Windows: prefix `MSYS_NO_PATHCONV=1` and pass `-e PATH=/workspaces/eng-ai-agents/.venv/bin:/usr/local/bin:/usr/bin:/bin`.
