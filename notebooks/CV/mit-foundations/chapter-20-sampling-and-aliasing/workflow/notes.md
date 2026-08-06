# Chapter 20 notes — gotchas and design decisions

## NumPy stripped — `math` for scalars, `torch` for tensors

The brief is PyTorch + Kornia exclusively, no NumPy. Two substitutions matter
here:

- **Scalar trig and constants use `math`, not `torch`.** The book's parameters
  are scalars (`omega = 18*pi`, `f_alias = |9 - 11|`). We use `math.pi` /
  `math.cos` for those and reserve `torch.cos` / `torch.sin` for evaluating a
  signal over a *tensor* grid. Wrapping a scalar as `torch.cos(torch.tensor(x))`
  works but yields a 0-d tensor that then complicates plain-Python arithmetic —
  cleaner to keep scalar math in `math.*`.
- **Fourier transforms use `torch.fft`, not SciPy/NumPy.** `torch.fft.rfft` /
  `rfftfreq` (1-D, Fig 20.11) and `torch.fft.fft2` / `fftshift` (2-D, Figs
  20.16–20.17) cover every spectrum in the chapter and run on CPU or GPU
  unchanged. SciPy is installed in the container but deliberately not imported.

## Loading the image without importing NumPy

`skimage.data.camera()` returns a NumPy `uint8` array. We ingest it with
`torch.from_numpy(camera().copy()).float() / 255.0` — `.copy()` because the
returned array is read-only, and no `import numpy` anywhere. This is data
ingestion, not computation; all subsequent work is `torch`.

## Cameraman, not the zebra (copyright)

Figures 20.16/20.17 in the book downsample a photograph of a **zebra** — ideal
because stripes make orientation-aliasing obvious. That specific photo is the
authors' asset, so we do not lift it. We use the public-domain **cameraman**
SIPI test image (`skimage.data.camera()`, loaded at execute-time, never committed
to the repo) — the same substitution Chapter 38 made for its photographic input.
The aliasing it demonstrates (detail breakup + spectral fold-in, removed by a
pre-filter) is pedagogically identical. The book's zebra is never referenced.

## Kornia is the anti-aliasing filter

Per the brief, Kornia is the recommended library on top of PyTorch. The natural
place for it in this chapter is the **anti-aliasing filter** (Fig 20.17):
`kornia.filters.gaussian_blur2d(img[None, None], (k, k), (sigma, sigma))` is the
low-pass we apply *before* point-sampling. The naive (aliased) and pre-filtered
paths share the same `point_sample` helper so the only variable between 20.16 and
20.17 is the blur — exactly the comparison the book makes.

## Fig 20.4 was the figure most likely to be drawn wrong

The book's 20.4 is **diagonal waves of increasing frequency** that change
*orientation* when sampled to 52×52. An early draft used a radial zone-plate
(concentric rings) — same aliasing concept, wrong figure. The faithful version is
`cos(2*pi*a*(x+y)**2)`: a diagonal chirp whose crests run along the anti-diagonal
and tighten toward the top-right, with `origin="lower"` so "slow at the bottom
left" matches the book's wording. Read the figure caption literally before
coding the pattern.

## Which figures are schematic

20.5 (band-limited spectrum), 20.8 (spectral replication), and 20.14 (reciprocal
lattices + Voronoi cells) are **sketches in the book**, so we regenerate them from
their equations rather than claim a pixel match. The replication in 20.8 is an
explicit sum of shifted copies of the band-limited bump; the 20.14 Voronoi cells
(square for rectangular sampling, hexagon for hexagonal) are drawn as polygons —
the point is the *relative* alias-free area, ~14% larger for the hexagon.

## Fig 20.15 is not reproducible

The cone-mosaic micrograph is real microscopy from Curcio et al. There is nothing
to compute, so it gets a markdown citation, not a fabricated plot. Figure 20.14
already shows *why* the hexagonal arrangement is favoured.

## Plotting cells are split out and tagged

Every figure is a compute cell (visible — the math) followed by a plotting cell
tagged `hide-input` (16 in total), so the rendered page shows the figure while
hiding the drawing code. Keeping the math visible and the drawing hidden matches
the rest of the site.

## Execution

The whole notebook runs end-to-end in ~60 s with no GPU training (the slowest
cells are the first `torch`/CUDA import and `fft2` on the 512×512 image). It is
self-contained and re-runnable — no cached state, no out-of-band assets.