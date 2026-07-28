# Workflow — How this chapter was produced

This directory documents the AI-assisted workflow behind Chapter 18's companion
notebook (`../index.ipynb`), following the Ch 17 / 38 / 42 / 47 pattern.

## What's in this directory

| File | Purpose |
|---|---|
| `README.md` | This document — approach, decisions, replication recipe |
| `research-findings.md` | The chapter's math digested from visionbook.mit.edu |
| `prompts.md` | The prompts that drove drafting and validation |
| `editor-scripts/` | (empty) — figures are computed, not hand-placed; see NOTE.md |

## Source images

The photo figures use the **book's own example images**, loaded live from
`visionbook.mit.edu` at run time (referenced, not committed):

| Fig | Book image | Use |
|---|---|---|
| 18.2 | `derivatives/mit_der_a.jpg` (**colour** MIT dome) | x- and y-derivatives (colour input, grayscale maps) |
| 18.6 | `derivatives/stop_noise.jpg` (**colour**) | raw gradient (noise) vs Gaussian-derivative gradient (clean) |
| 18.8 | `spatial_filters/gausian_zebra_c_2.jpg` (gray) | multiscale Gaussian x-derivative, σ=2,4,8 |
| 18.18 | `spatial_filters/wheel256.jpg` (gray) | ∂²x, ∂²y, and the isotropic Laplacian |
| 18.23 | `spatial_filters/boat_sharp0.jpg` (**colour**) | unsharp masking ×1..5, per-channel |

Colour inputs are shown in colour; **derivative maps are grayscale** because a
derivative acts on a single intensity channel (`luminance()`), and colour
sharpening (18.23) runs the kernel on each channel (`conv2d_rgb`).

The **Retinex** figure (18.25/26) uses a **synthetic Mondrian × smooth
illumination**, matching the book's own synthetic Retinex test — this gives
ground truth so the recovery can be *measured* (reflectance correlation ≈ 0.92).

## Figures reproduced (13)

18.2 x/y derivatives · 18.4 |DFT| of d₀,d₁ vs ideal · 18.6 noise vs Gaussian
derivative (signed per-channel, like the book) · 18.9 Gaussian + Hermite
derivative orders · **18.10 the 2D Gaussian-derivative triangle** · 18.8
multiscale on zebra · derivative-of-binomial (Pascal) · **18.14 |DFT| of
d₀/d₁/Roberts/Sobel as 3D surfaces** · 18.15 directional derivatives · 18.16
Laplacian-of-Gaussian (Mexican hat) · 18.18 wheel Laplacian · 18.23 unsharp
masking · 18.25/26 Retinex.

Note: the book has no standalone image for the discrete derivative-of-binomial
family (it is discussed in text), so that figure is an original illustration;
18.14 is drawn as 3D surfaces to match the book, and 18.10 reproduces the book's
iconic 2D Gaussian-derivative triangle.

## Numerical checks (a number beside every figure)

- centred vs two-tap kernels' DFT vs the ideal `|ω|`
- Gaussian/Hermite derivatives and derivative-of-binomial kernels sum to 0
- noise floor: raw gradient std ≈ 0.047 → Gaussian-derivative std ≈ 0.0047 (~10×)
- `∂²x + ∂²y == Laplacian` to ~1e-7 (five-point stencil)
- sharpen kernel DC gain = 1
- Retinex: recovered reflectance correlates ≈ 0.92 with ground truth

## Validation command

```bash
docker compose run --rm torch.dev.cpu \
  bash -c "python scripts/execute_notebook.py \
    CV/mit-foundations/chapter-18-image-derivatives/index.ipynb"
```
28 cells, **~130 s**, **13 figures**, no errors. CPU-only.

> Windows note: prefix the run with `MSYS_NO_PATHCONV=1` and pass an explicit
> `-e PATH=/workspaces/eng-ai-agents/.venv/bin:/usr/local/bin:/usr/bin:/bin`,
> or the container cannot find `bash`/`python` (host `$PATH` leaks in via Compose).

## Notes / honest caveats

- **Retinex** here is the *basic single-threshold* method: it pins the
  reflectance layer well (corr ≈ 0.9) and takes illumination as the smooth
  remainder. A periodic-BC FFT Poisson solve leaves a faint low-frequency
  residual — production Retinex iterates or uses a Neumann (DCT) solve.
- Figures are **computed from the equations**, so (as in Ch 17) no drag-and-drop
  editor was needed.
