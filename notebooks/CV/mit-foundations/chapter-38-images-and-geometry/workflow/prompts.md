# Claude prompts that produced the best results for Ch 38

A short catalog of the prompts that consistently moved this chapter forward.
Many of these are duplicates of Ch 47's, since the reading/inspecting/comparing
patterns apply to any chapter. The Ch 38-specific ones are at the bottom.

## Reading the chapter

> "I need a COMPLETE detailed breakdown of Chapter 38 'Representing Images
> and Geometry'. Provide: (1) every section heading in order, (2) every figure
> with its full caption — flag which figures are real photographs vs synthetic
> diagrams vs computational overlays (SIREN reconstructions), (3) every numbered
> equation with LHS and a one-line description, (4) any algorithm/pseudocode,
> (5) any specific objects/scenes used (checkerboard, F-shape, clock, etc.).
> Be exhaustive — I'm reproducing every figure I can."

Use with `WebFetch` against `https://visionbook.mit.edu/homogeneous_coordinates.html`.

## Visual comparison HTML

The single most useful pattern for Ch 38: a side-by-side comparison page.

> "Build a self-contained HTML at output/ch38_compare.html with one section
> per figure. Left column: book figure (read from output/book-figures-ch38/
> fig38_NN.png). Right column: our render (read from <chapter>/images/
> fig38_NN.png). Embed both as base64 so the file works offline. Match column
> widths, dark-mode friendly, label rows by figure number + caption."

Refreshes in < 1 second after a notebook re-execution and any figure mismatch
becomes obvious.

## Iterating on a specific figure

> "Look at the book figure 38.<N> (path) and our render (path). List the
> SPECIFIC visual elements where ours differs — be concrete (`missing diagonal
> line`, `wrong type of warp`, `axes drawn under the shape instead of over`,
> ...). Don't propose code yet — I want the diff first."

Doing the diff explicitly before the fix avoids the "Claude makes 10 changes,
half are regressions" failure mode.

## Then for the fix

> "OK fix only `<specific element>` in cell <N>. Leave the rest of the cell
> alone. Re-execute, regenerate images/, refresh the comparison HTML."

Narrow scope per iteration is the key.

## Chapter-38-specific: non-linear warping

> "The 'Warping' panel in Fig 38.2 should be a clearly NON-LINEAR deformation
> (the book's example is Dalí-style). Currently we use a homography and the
> result looks like a perspective-shear. Replace with `kornia.geometry.transform
> .remap` and a smooth deterministic sinusoidal displacement field; cap the
> max displacement around 15-20 pixels so the clock is still recognisable."

This is the kind of "show the reader the principle, not the abstraction" call
that came up repeatedly during Ch 38 review.

## Chapter-38-specific: clock asset

When the user supplied an AI-generated wall-clock image and asked to use it
instead of the procedural one:

> "Save the supplied image as assets/wallclock.png. Make it 256x256 grayscale,
> no cropping (the natural beige background already provides padding for warps).
> Update `render_clock()` to load this asset directly — no padding, no white
> canvas, no procedural drawing. Re-execute and refresh the comparison HTML."

Important: the user's supplied image, NOT a cropped version of the book's own
photo. That distinction matters for copyright.

## Anti-patterns to avoid

These prompts produced bad results in this chapter:

- **"Match the book exactly"** — the book's clocks are real photographs; ours
  is an AI-generated approximation. Pixel-match is not the goal. Match the
  *concept* (translation visibly moved the clock, rotation visibly tilted it,
  warping visibly deformed it non-linearly).
- **"Use the book's photo"** — copyright concern. Stick to public test images
  (cameraman) or AI-generated assets the contributor supplies.
- **Generic "fix scaling"** — say what about scaling. Anisotropic vs uniform?
  Bigger vs smaller? Different image vs different transform parameters? The
  spec matters here because the matrix that ships ends up in the rendered code.

## Patterns that recurred

- **Read the book first, code second.** Spending 60 seconds looking at the
  book figure before touching the cell prevents 30 minutes of wrong-fix.
- **Visual comparison HTML.** The first time we built this it caught five
  problems we'd been blind to. Build it BEFORE iterating, refresh after every
  change.
- **One figure at a time.** When the user says "scaling is wrong AND warping
  is wrong", patch them in separate iterations. Otherwise you can't tell
  whether your scaling fix broke the warping or whether they were independent.
- **Don't open PRs against the upstream repo without explicit per-action
  confirmation.** Same rule as Ch 47 (see Ch 47 `prompts.md`).
