# Chapter 38 notes — gotchas and design decisions

## NumPy stripped, math/PIL substitutes

The Aegean brief says PyTorch + Kornia exclusively; NumPy out. Two specific
substitutions worth noting:

- **`torch.cos` / `torch.sin` need a tensor input**, unlike `np.cos` / `np.sin`
  which take a Python float. Inside `T_rotate(theta)` we use `math.cos(theta)`
  / `math.sin(theta)` — Python's stdlib math module on a scalar — to build the
  3×3 rotation matrix. Using `torch.cos(torch.tensor(theta))` would work but
  produces a 0-d tensor, and `torch.tensor([0d_tensor, 0d_tensor])` then errors
  with "only one element tensors can be converted to Python scalars". Avoid
  the issue: scalar trig with `math.*`, tensor trig with `torch.*`.

- **`torch.tensor([list_of_tensors])` fails** where `np.array([list_of_arrays])`
  would have concatenated. Use `torch.stack(...)` for that case. Cell 6's
  `unit_square()` works with a plain list of `[x, y]` numbers, so no stack
  needed there, but the pattern matters for anything building a tensor from
  arithmetic on 0-d tensors.

- **PIL → torch without NumPy** — `list(img.getdata())` returns a Python list
  of pixel values; `torch.tensor(..., dtype=torch.float32).view(H, W) / 255.0`
  produces the same tensor you'd get via `np.asarray(img)`, no NumPy import.

## Kornia is the right place for warping

Per the brief, Kornia is the recommended library on top of PyTorch. Chapter 38
is the natural home for it because the whole chapter is about image warping:

- `kornia.geometry.transform.warp_affine` — backward-mapping affine warp with
  bilinear interpolation. Used as a side-by-side reference next to our
  hand-rolled `backward_warp` (which uses nearest-neighbour, to show the
  forward-mapping holes more clearly).
- `kornia.geometry.transform.warp_perspective` — 3×3 homography. Originally
  used for the "warping" panel in Fig 38.2, but the result looks like a
  perspective-shear, so we replaced it.
- `kornia.geometry.transform.remap` — arbitrary displacement field, used for
  the non-linear "warping" panel that matches the book's Dalí-style example.

Chapter 47 does *not* import Kornia. Its figures are geometric diagrams of
camera/ray relationships, where the math-on-the-page (`f * X / Z`) IS the
lesson. Kornia abstracts those formulae behind helpers — useful in production,
counter-productive when teaching the formula. (See Ch 47's workflow README.)

## SIREN target is the cameraman, not the book's photo

`skimage.data.camera()` returns the SIPI "Cameraman" test image — the same
photo the book shows in Figure 38.12 (photographed at MIT in 1979, freely
distributed for decades as a CV test image, bundled with scikit-image under
BSD-3). The notebook calls the library function at execute-time; the image is
never committed to this repo.

This is fundamentally different from the wall-clock asset:
- Cameraman: a public CV test image both we and the book load via library
- Clock photo: the book's specific photograph (their asset) — we'd be
  committing their IP if we lifted it

For Ch 38 we use the cameraman directly (no concern) and an AI-generated
clock supplied by the contributor (also no concern). The book's clock photo
is never referenced.

## Fig 38.7 matrix rendering

The book shows the four 3×3 transformation matrices below the four shape
panels using LaTeX `\begin{bmatrix} ... \end{bmatrix}`. Matplotlib's
**mathtext** subset does NOT support `\begin{...}` environments — only inline
math like `$\theta$`, `$\cos\theta$`, etc.

Options:
1. `text.usetex = True` — uses the host's LaTeX installation. Heavy dep,
   and not all reviewers will have MiKTeX/TeXLive set up.
2. Render each matrix as monospace plain text with Unicode brackets. Visually
   utilitarian but reads cleanly and ships with the standard Python stack.

We picked (2). If the project later standardises on `usetex`, this cell is a
one-line change.

## Iteration cycle (SIREN training)

SIREN at 2000 steps takes ~6–8 minutes on CPU and dominates the notebook
runtime. During iteration on layout-only figures (38.1, 38.3 – 38.8), we
temporarily dropped training to 500 steps so the full notebook re-executed in
~1 minute. Bumped back to 2000 for the final commit so Fig 38.12 is clean.

A nicer workflow would be to cache the trained SIREN weights to disk and
only re-train when the architecture or target image changes. Not done here
because the chapter is meant to be self-contained and re-run end-to-end —
no out-of-band state — but worth considering for a future revision.
