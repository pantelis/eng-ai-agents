# Prompts — Ch 18 Image Derivatives

The prompts that drove the work, in order.

## 1. Digest the chapter
> Extract the chapter's math in full detail. For each figure state what it shows.
> Give exact equations for: the [1,-1] and [1,0,-1] kernels and their DFT; Roberts
> and Sobel; gradient magnitude/orientation; Gaussian derivatives (Hermite
> recursion) and derivative-of-Gaussian; derivative-of-binomial; the image
> Laplacian (five-point stencil); unsharp masking; the Retinex steps. List every
> example image and whether it is colour or grayscale.

Output became `research-findings.md`, and the colour check drove which figures
render in colour.

## 2. Draft the notebook
> Build the companion notebook, one section per book section. Load the book's own
> photos live from visionbook.mit.edu (do not commit them). Colour inputs
> (building, stop-noise, boat) in colour; derivative maps grayscale; colour
> sharpening per-channel. Back every figure with a numerical check.

## 3. Make Retinex robust and honest
> The book's Retinex input is a synthetic Mondrian x illumination. Reproduce that
> as a controlled scene with ground truth, threshold the log-gradients, integrate
> the sharp part with an FFT Poisson solve, and REPORT the reflectance-recovery
> correlation rather than hand-waving. If the basic method leaves a low-frequency
> residual, say so.

## 4. Fix the noise figure display
> The raw-vs-Gaussian gradient panels were too dark. Use strided downsampling to
> keep the noise, and a percentile-clip + gamma display (`show_grad`) so the noise
> speckle (raw) and the clean STOP outline (Gaussian derivative) are both visible.

## 5. Validate in the container
> Execute via `docker compose run --rm torch.dev.cpu ...`; recopy the extracted
> figures to images/fig18_01..12; add the notebook-database.yml entry.

## Lessons carried forward
- **Colour where the book is colour** — check channel spread per image, don't
  assume grayscale.
- **A number beside every picture** — the Laplacian identity (1e-7), the noise
  floor drop (10x), the Retinex correlation (0.92) make the figures proofs.
- **Retinex is finicky** — the basic single-threshold method recovers reflectance
  cleanly but illumination is fragile; report what actually works, caveat the rest.
- **Gradient displays need percentile + gamma**, not plain min-max (outliers crush
  the contrast).
