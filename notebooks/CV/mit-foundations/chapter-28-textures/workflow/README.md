# Workflow — How this chapter was produced

Companion notebook for Chapter 28 (Textures), following the Ch 17/18/19/27/38/42/47
pattern. 6 figures, each with a numerical check where applicable.

## Source images — the book's own textures
- **plums, zebra, pebbles** — cropped from the book's synthesis figures
  (two_examples*.png) and bundled under `assets/`; used as the exact references in
  28.5/9 (plums HB) and 28.12 (plums + pebbles Efros).
- **efros1a.jpg** (circles) — loaded from visionbook.mit.edu for the 28.11 context demo.
Credit: visionbook.mit.edu.

## Figures reproduced (6)
| Fig | Content |
|---|---|
| 28.1 | infinite texture by tiling: crop, naive tiling (seams), mirror tiling (seamless) |
| 28.5/28.9 | Heeger-Bergen synthesis: reference -> larger synthesised texture |
| 28.6 | white noise -> texture over Heeger-Bergen iterations |
| 28.8 | one-subband histogram matching: noise (Gaussian) -> texture (Laplacian) |
| 28.11 | Efros-Leung context size: small (layout lost) vs large (regular) |
| 28.12 | Efros-Leung on the stone wall: structure preserved |

## Implementation
- **Heeger-Bergen** (`heeger_bergen_gray`): iterate {match pixel histogram; match
  each Laplacian-pyramid subband histogram; reconstruct}. We use an isotropic
  Laplacian pyramid in place of the steerable pyramid, and **PCA-decorrelate the
  colour channels** (as the book does) so the plum/pebble colours are preserved
  (synthesising R,G,B independently gives rainbow noise). It matches the
  reference's colour + blob-scale statistics but scrambles the global layout.
- **Efros-Leung** (`efros_leung`, gray or colour): grow pixel-by-pixel; for each
  border pixel, vectorised SSD (via `sliding_window_view`) against every sample
  window over the known-neighbour mask, sample a near-best match, copy its centre
  (RGB for colour). Large window preserves structure; small window fragments it.
- `match_hist` is exact histogram matching via a sorted-value rank map.

## Numerical check
- 28.8: subband kurtosis — texture ~3.5 (heavy-tailed) vs noise ~1.8; matching
  transfers the shape.

## Validation
```bash
docker compose run --rm torch.dev.cpu \
  bash -c "python scripts/execute_notebook.py \
    CV/mit-foundations/chapter-28-textures/index.ipynb"
```
15 cells, ~73 s, 6 figures, no errors (Efros-Leung is the slow part). CPU-only.
> Windows: prefix `MSYS_NO_PATHCONV=1` and pass `-e PATH=/workspaces/eng-ai-agents/.venv/bin:/usr/local/bin:/usr/bin:/bin`.

## Honest caveats
- Simplified Heeger-Bergen (isotropic Laplacian pyramid instead of the steerable
  pyramid) — captures the colour + statistics matching, but softer/blobbier than
  the oriented steerable version. Synthesis is stochastic: it matches the book's
  CHARACTER, not the exact pixels.
- Efros-Leung can "grow garbage" in a corner (a known failure mode).
