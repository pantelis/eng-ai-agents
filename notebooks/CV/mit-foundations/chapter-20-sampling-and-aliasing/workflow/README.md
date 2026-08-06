# Workflow — How this chapter was produced

This directory documents the AI-assisted workflow behind Chapter 20's companion
notebook, in the same spirit as the [Chapter 38](../../chapter-38-images-and-geometry/workflow/README.md)
and Chapter 47 workflow folders: make the figures auditable and give the next
contributor a recipe.

## What's in this directory

| File | Purpose |
|---|---|
| `README.md` | This document — workflow overview + decisions |
| `notes.md` | Chapter-20-specific gotchas (NumPy-free trig, `torch.fft`, the cameraman substitution, schematic figures) |

## The chapter at a glance

Chapter 20 is almost entirely signal/image processing, so every figure maps to
PyTorch with no training. The figures fall into three kinds:

- **1-D signal plots** (20.1, 20.2, 20.3, 20.6, 20.7, 20.9, 20.10, 20.11, 20.12) —
  cosines, the delta train, sinc reconstruction. Pure `torch`.
- **Schematic spectra / lattices** (20.5, 20.8, 20.13, 20.14) — the book draws
  these as sketches; we regenerate them from the same equations rather than
  trace them.
- **2-D image effects** (20.4, 20.16, 20.17) — the diagonal chirp, and real-image
  aliasing with vs without an anti-aliasing filter (Kornia Gaussian blur).

Figure **20.15** (cone mosaic in the primate fovea) is measured microscopy from a
cited paper, not a computed plot, so it is referenced in a markdown cell and not
reproduced.

## How it was built

```
1. Read the chapter at visionbook.mit.edu/sampling_and_aliasing.html
   -> enumerate every figure, classify as 1-D plot | schematic | 2-D image effect,
      and capture the book's exact parameters (omega = 18*pi, Ts = 1/11, 52x52, ...)
2. Generate index.ipynb from an nbformat builder script so the ~30 cells and their
   hide-input tags are produced programmatically (the builder lives outside the repo;
   the notebook is the committed artifact)
3. Execute end-to-end with papermill (~60 s, no GPU training)
4. Extract each figure to images/fig20_NN.png and eyeball a contact sheet against
   the book figures
5. Fix any figure that reads wrong (see notes.md — Fig 20.4 was the main one)
6. Register in notebooks/notebook-database.yml and hand off for PR
```

## Conventions followed (from the project brief and the Ch 38 reference)

- **PyTorch + Kornia only**, no NumPy in the computational code. See `notes.md`
  for the scalar-trig and `torch.fft` substitutions.
- **Markdown and inline comments explain the code first, the concept second**, and
  never mention the plotting library or toolchain.
- **Pure-plotting cells carry the `hide-input` tag** so the rendered page shows the
  figure but hides the drawing code (16 such cells here).
- **One directory per chapter** with its own `images/` folder; the notebook is
  `index.ipynb`.

## Faithful vs. schematic

Where the book shows a *sketch* (the band-limited spectrum 20.5, the spectral
replication 20.8, the reciprocal lattices 20.14), we reproduce the same idea from
its equation rather than claiming a pixel match — those originals are themselves
illustrative drawings. Where the book shows a *computed* result (the chirp 20.4,
sinc interpolation 20.10, the rate sweep 20.11), the numbers are real. The one
deliberate content substitution is the **cameraman** image in 20.16/20.17 in place
of the book's copyrighted zebra photo; the demonstrated effect is identical.
