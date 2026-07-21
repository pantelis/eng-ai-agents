# Workflow — How this chapter was produced

Same intent as the `workflow/` directories under
[Chapter 38](../../chapter-38-images-and-geometry/workflow/README.md) and
[Chapter 47](../../chapter-47-3d-motion-projection/workflow/README.md):
provenance + a recipe the next contributor can re-apply.

## What's in this directory

| File | Purpose |
|---|---|
| `README.md` | This document — workflow overview + decisions |
| `notes.md` | Things specific to this chapter (synthetic scene rationale, Algorithm 2/3 cross-ratio gotchas, calibration verification) |
| `prompts.md` | Specific Claude prompts that worked |

## The single workflow decision that mattered

**Replace the book's office photograph with a synthetic 3D scene rendered in PyTorch.**

The chapter relies on a recurring office photo throughout — bookshelf (197 cm),
desk (76 cm), bottle on the desk (25 cm), three orthogonal vanishing points.
Three options:

1. **Use the book's photo.** Copyright concern (their photo asset).
2. **Find / supply a real photo with annotated measurements.** Possible, but
   then we can only verify the algorithm subjectively ("the recovered desk
   height looks about right").
3. **Build a synthetic scene with known dimensions and project it.** No
   copyright, and we get **ground-truth verification for free** — Algorithm 2
   should recover 76 cm exactly; Algorithm 3 should recover 25 cm exactly;
   the K-from-VP algorithm should recover `K_true` element-for-element.

This notebook ships with (3). Every algorithm cell ends with a `print`
comparing the recovered value to the synthetic ground truth, and the
verification is part of the published output.

```
recovered desk height: 76.0 cm  (ground truth: 76 cm)
recovered bottle height: 25.0 cm  (ground truth: 25 cm)
max absolute element error: 0.00 px  (0.000% relative)
```

That last line — recovering the entire $\mathbf{K}$ matrix to 0% relative
error from just three vanishing points — is a much stronger demonstration of
the algorithm than the book's "the recovered numbers are close enough" hand
wave.

## Figures actually produced

| Figure(s) in the book | Our equivalent |
|---|---|
| 42.1, 42.10, 42.13, 42.14, 42.17, 42.19 (office photo + overlay variants) | One ground-truth scene rendered + overlay variants per algorithm |
| 42.2 — "streetgeons" line sketch | not rebuilt (illustrative, not algorithmic) |
| 42.3, 42.4 — Ames room construction + photos | not rebuilt (perceptual illusion, not algorithmic) |
| 42.5 — 3D-line projection diagram | Fig 42.5 panel in cell 6 |
| 42.6 — parallel-line convergence | Fig 42.6 panel in cell 6 |
| 42.7 — Pisa towers illusion | not rebuilt (perceptual) |
| 42.8 — building facade with horizon | covered by the synthetic-office horizon line drawn in the algorithm overlays |
| 42.9 — VP cross-product diagram | Algorithm 1 in cell 9 (output: VPs match ground truth) |
| 42.11 — cross-ratio invariance | Fig 42.11 panel in cell 10 |
| 42.12 — ruler photo across 3 views | Fig 42.12 panel in cell 10 (numerical, three synthetic views of one ruler all give CR=1.6) |
| 42.15 — schematic of 42.14 setup | covered by the Algorithm 2 overlay in cell 14 |
| 42.18 — axis-calibration cross-ratio | covered conceptually (would need a separate cell to render explicitly) |
| 42.20 — point localisation context | Fig 42.20 in cell 19 |

## End-to-end flow used for this chapter

```text
1. Read the chapter at visionbook.mit.edu/3d_scene_understanding_single_view.html
   -> enumerate 20 figures, classify by type (photo / synthetic / perceptual)
   -> identify which algorithms have numerical-verifiable outputs (VP, height
      measurement, K calibration)
2. Reuse the previous author's skeleton:
     - synthetic office scene with known dimensions
     - perspective projection through a chosen camera matrix K_true, R_true, eye_true
3. Fill in the TODOs:
     - Algorithm 2 (desk height by cross-ratio) - verify == 76 cm
     - Algorithm 3 (height propagation to bottle) - verify == 25 cm
     - Algorithm 6 (K from three orthogonal VPs) - verify == K_true
4. Add the missing schematic figures (42.5, 42.6, 42.11, 42.12, 42.20) as
   pure-matplotlib geometric diagrams
5. Replace the open-questions cell with a real 42.7 concluding-remarks markdown
6. Aegean compliance: strip NumPy, hide-input tags, no plotting-lib names
   in markdown
7. Add images/ subfolder with rendered PNGs
8. Update notebook-database.yml
9. Open PR
```

The biggest time-saver vs Ch 38: **no iterative visual review needed**. Each
algorithm has a single line of output ("recovered: X cm vs truth: Y cm") that
is unambiguously right or wrong. The schematic diagrams are math-determined
(no positioning judgement calls).

If a future contributor wants to add the photograph-based figures (42.1,
42.10, etc.) as a bonus, they can supply an AI-generated office photo and
manually annotate the four corner pixels per Algorithm 2/3 — same pattern
as Ch 38's wall-clock asset.
