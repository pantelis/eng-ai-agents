# Workflow — How this chapter was produced

This directory documents the AI-assisted workflow behind Chapter 17's companion
notebook (`../index.ipynb`). Same intent as the other MIT-foundations chapters
(Ch 38, 42, 47): make each figure's provenance auditable, and give the next
contributor a repeatable recipe.

## What's in this directory

| File | Purpose |
|---|---|
| `README.md` | This document — approach, decisions, replication recipe |
| `research-findings.md` | The chapter's math digested from visionbook.mit.edu — every definition and figure the notebook reproduces |
| `prompts.md` | The prompts that drove drafting and validation |
| `editor-scripts/` | (empty) — see "Why no figure editor" below |

## The key decision for this chapter: figures are computed, not hand-placed

Ch 47 (3D motion) needed a browser drag-and-drop editor because its figures are
**geometric diagrams** — cameras, pyramids, vanishing points — whose element
positions had to be matched to the book by eye.

Chapter 17's figures are the opposite: they are **direct visualisations of the
filters and their spectra** — filtered images, kernel stems, DFT magnitude
curves, kernel-vs-Gaussian overlays. Every element's position is *determined by
the math*, so there is nothing to place by hand. This mirrors Ch 47's figures
47.9 / 47.10 (pure quiver fields), which were also written straight from the
equations without an editor. Hence `editor-scripts/` is intentionally empty.

## Source images

The four photo-based figures use the **book's own example images**, loaded live
from `visionbook.mit.edu` at run time so the demonstrations match the textbook
exactly:

| Fig | Book image (URL) | How it's used |
|---|---|---|
| 17.1 | `blur_filters/stop_256_noise_3.jpg` | noisy stop sign (**colour**) → 5×5 box average, filtered per-channel |
| 17.4 | `spatial_filters/gausian_zebra_c_2.jpg` | book's σ=2 zebra (grayscale) → further-blurred to σ=4, σ=8 (composition rule) |
| 17.5 | `blur_filters/Jules_Lincoln_1971.jpg` | blocky Lincoln (grayscale) → Gaussian blur reveals the face |
| 17.8 | `blur_filters/boat_d_binomial.jpg` | book's boat (**colour**) + a true 1-pixel checkerboard → box vs binomial, filtered per-channel |

Colour figures (17.1 stop, 17.8 boat) run each filter independently on the R, G,
and B channels (`conv2d_rgb`); the grayscale figures use a single-plane `conv2d`.

The images are **referenced online, not committed** to the repo, so nothing is
redistributed. The remaining figures (box-directional 17.2, and the spectral /
convergence plots) use a `scikit-image` sample or are computed from equations.

Note on 17.8: the book *displays* the checkerboard as 8×8-pixel squares — an 8×
upscaled view of a low-res image whose sampling grid carries the 1-pixel Nyquist
wave `[1,-1,…]`. A literal 3×3 binomial only cancels that 1-pixel pattern, so we
add a true 1-pixel checkerboard to the book's boat and filter at that scale;
the binomial residual is ≈ 2e-7 (exact cancellation).

## End-to-end flow used for this chapter

```text
1. Read the book chapter at visionbook.mit.edu/blurring_2.html
   -> enumerate every figure and the exact equation behind it (-> research-findings.md)
2. Draft index.ipynb: one section per book section
   (Intro -> Box -> Gaussian -> Binomial -> Concluding), math in LaTeX markdown,
   one code cell per figure.
3. Implement the three filter families as torch convolutions:
   - box_kernel(N,M), gaussian_1d/2d(sigma), binomial_1d(n) (repeated [1,1] conv)
   - a single reflect-padded conv2d() helper shared by all figures
4. Back every claim with an in-notebook numerical check, not just a picture:
   - separability error (2D == cascaded 1D)          ~1e-6
   - Gaussian composition sigma3^2 = sigma1^2+sigma2^2 ~1e-8
   - binomial variance = n/4                          exact (0.5 for b2)
   - checkerboard residual: box > 0, binomial ~ 0
5. Validate end-to-end in the maintainer container (see below).
6. Copy the extracted figure PNGs into images/fig17_01..09.png.
7. Register in notebooks/notebook-database.yml; open PR against pantelis/eng-ai-agents.
```

## Validation command

Executed clean in the maintainer environment:

```bash
docker compose run --rm torch.dev.cpu \
  bash -c "python scripts/execute_notebook.py \
    CV/mit-foundations/chapter-17-blur-filters/index.ipynb"
```

22 cells, **106.0 s**, **9 figures extracted**, no errors. CPU-only — the
notebook has no GPU dependency (registered as `torch.dev.gpu` only because that
is the schema's default service; it runs identically on `torch.dev.cpu`).

> Windows note: Git-Bash mangles container paths and Compose interpolates the
> host `$PATH` into the container. If `bash: not found` appears, prefix the run
> with `MSYS_NO_PATHCONV=1` and pass a clean `-e PATH=/workspaces/eng-ai-agents/.venv/bin:/usr/local/bin:/usr/bin:/bin`.

## What the AI did and did not do

**AI was the workhorse for**
- Digesting the chapter's equations and figure list from the book
- Writing the torch/matplotlib implementations of each filter and figure
- Writing the section markdown (concept + LaTeX math)
- Adding a numerical self-check behind every qualitative figure
- Driving the container validation and wiring up the registry entry

**Human judgement was needed for**
- Deciding to load the book's own images (vs. skimage stand-ins) for fidelity,
  and to reference them online rather than commit them
- Diagnosing the 17.8 checkerboard scale (book displays it 8× upscaled) and
  filtering at the true 1-pixel grid
- Deciding the editor step was unnecessary here (computed vs. placed figures)
- Confirming each rendered figure matches the book's phenomenon
