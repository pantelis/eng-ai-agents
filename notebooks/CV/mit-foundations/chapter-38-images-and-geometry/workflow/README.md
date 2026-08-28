# Workflow — How this chapter was produced

This directory documents the AI-assisted workflow used to produce Chapter 38's
companion notebook. Intent is the same as the `workflow/` directory under
[Chapter 47](../../chapter-47-3d-motion-projection/workflow/README.md):

1. **Provenance.** Make the reasoning, tooling, and prompts behind each figure
   auditable.
2. **Replication.** Give the next contributor a recipe they can re-apply.

## What's in this directory

| File | Purpose |
|---|---|
| `README.md` | This document — workflow overview + decisions |
| `notes.md` | Things specific to this chapter (Kornia, the wall-clock asset, SIREN) |
| `prompts.md` | Specific Claude prompts that produced the best results |

## How Ch 38 differed from Ch 47

Chapter 47 used a **drag-and-drop SVG editor** per figure to tune element
positions, because every figure was a geometric scene (cameras, rays, planes)
where the *layout* was the work. Chapter 38 is different — most figures are
**math-driven** (a unit square transformed by a matrix; a checkerboard or clock
warped by `kornia.warp_affine`; a SIREN reconstruction). The layout is *derived*
from the math; you don't drag anything.

So the editor workflow from Ch 47 was **not used here**. Instead the workflow
was:

```
1. Read the chapter at visionbook.mit.edu/homogeneous_coordinates.html
   -> enumerate every figure, classify each as
      structural (math-driven) | image-warp (apply matrix to a photo) | SIREN
2. Draft a notebook skeleton: markdown per section + code cell per figure
3. Execute end-to-end, render all 12 figures inline
4. Build a side-by-side comparison HTML: book figure | our render
   (download book figures into output/book-figures-ch38/, base64-embed both
   sides, view in browser)
5. ITERATE on whichever figure the user flags as off — typical loop is:
      a. Look at the book figure carefully
      b. Identify the SPECIFIC visual element that's wrong
         (wrong shape, missing diagonal line, wrong type of warp, ...)
      c. Patch the cell, re-execute, refresh the comparison HTML
   Most iterations were < 2 min thanks to the comparison HTML pattern.
6. When all panels look right, register in notebooks/notebook-database.yml
   and open a PR.
```

The single most important workflow decision: **build the side-by-side
comparison HTML early and refresh it after every change.** It catches "ours
looks nothing like the book" the moment it happens, instead of three steps
later when a reviewer points it out.

## The wall-clock asset (assets/wallclock.png)

The book applies the 2D transformations in Fig 38.2 / 38.11 to a photograph of
a wall clock. We deliberately did *not* embed the book's own photo, because:

- The book's asset is the authors' (or licensed) photograph
- Copying it into a public companion repo is a copyright concern
- We documented this trade-off in [Ch 47's workflow research findings](../../chapter-47-3d-motion-projection/workflow/research-findings.md)

We use an AI-generated wall-clock image (`assets/wallclock.png`) supplied by
the contributor instead. Pedagogically identical — same kind of subject, same
kind of transform demos — without the IP entanglement.

## Why `kornia.geometry.transform.remap` for the warping panel

The book's "Warping" panel in Fig 38.2 is a Salvador-Dalí-style non-linear
deformation. Three approaches we considered:

1. **Homography (`kornia.warp_perspective`)** — too tame; the result looks like
   a shear with perspective and reads as another linear transform. We tried
   this first and it confused the reader.
2. **`kornia.elastic_transform2d`** — random elastic warps. Plausible but the
   `alpha` parameter is delicate; with the defaults we tried, the image got
   destroyed (looks like noise).
3. **`kornia.geometry.transform.remap` with a sinusoidal displacement field**
   — deterministic, bounded, easy to tune (max displacement in pixels). Gives
   a clear non-linear "melting" deformation that is visibly different from
   shear and perspective. This is what we shipped.

If a future chapter needs a more complex non-linear warp (TPS, etc.), the same
`remap` pattern generalises.

## What the AI did and did not do

**AI was the workhorse for**
- Drafting matplotlib boilerplate
- Translating book-figure descriptions into code
- Writing the section markdown (concept + math typeset in LaTeX)
- Iterating on cells based on user feedback ("scaling is wrong, zoom in not
  stretch")

**Human held the wheel for**
- Picking which chapter(s) to take
- Sanity-checking visual similarity to the book (the AI is multimodal but a
  human eye catches mismatches faster)
- Supplying the wall-clock asset (avoids the AI making a copyright-risky call)
- Authorizing public actions (opening PRs against the maintainer's repo)

See `notes.md` for the chapter-specific gotchas and `prompts.md` for the
specific Claude prompts that worked.
