# Claude prompts that produced the best results for Ch 42

Most of the Ch 42 work used the same prompts as Ch 38 and Ch 47 for the
reading / inspection / iteration parts. The chapter-specific ones below are
where Ch 42 differed.

## Reading the chapter

> "I need a COMPLETE detailed breakdown of Chapter 42 'Single View Metrology'.
> Provide: (1) every section heading in order, (2) every figure with caption
> — flag photo / synthetic / overlay / computational result, (3) every
> numbered equation with LHS and a one-line description, (4) any algorithm or
> pseudocode (cross-ratio height-measurement family especially), (5) any
> specific images used, (6) any Kornia / PyTorch references."

Use with `WebFetch` against `https://visionbook.mit.edu/3d_scene_understanding_single_view.html`.

The chapter slug is non-obvious — it's `3d_scene_understanding_single_view`,
not `single_view_metrology`. Worth checking the book's table of contents first.

## Verifying an algorithm against ground truth

> "Implement Algorithm 2 from the chapter's section 42.4.2. The notebook
> already has the synthetic office scene (cell 4) with known dimensions
> (bookshelf 197 cm, desk 76 cm) and the helpers `line_through`, `intersect`,
> `cross_ratio`. Use those. At the end print the recovered desk height and
> compare to ground truth 76 cm. If it doesn't match, the cross-ratio
> argument order is probably wrong."

The "compare to ground truth" line is what catches every cross-ratio
direction bug in 5 seconds instead of "looks reasonable" wishful thinking.

## Replacing photographs with synthetic scenes

> "The book uses a recurring office photo for figures 42.1, 42.10, 42.13,
> 42.14, 42.16, 42.17, 42.19. We're rendering a synthetic 3D scene instead
> (cell 4). For each of those figures, the equivalent in our notebook is the
> projected synthetic scene with the algorithm's construction overlay drawn
> on top. Don't try to match the photograph pixel-by-pixel; match the
> *information* — same vanishing points, same reference object, same labeled
> points."

## Anti-patterns to avoid

- **"Make the overlay match the book's photo exactly."** Ours is synthetic;
  pixel-match is impossible by construction. Match the algorithm's labeled
  points and lines, that's what's pedagogically necessary.
- **"Use the cross_ratio helper for all four-point ratios."** The generic
  helper has its own argument order; you save no code by routing the book's
  exact equation through it. Inline the four distances explicitly.
- **"Recover K from a real photograph for the validation."** That recovers
  an approximate $\mathbf{K}$ at best. Synthetic data gives 0.000% error,
  which is a much stronger demonstration that the algorithm is correctly
  implemented.

## Patterns that recurred

- **Ground-truth synthetic scenes are king for algorithm chapters.** Pages
  of geometric algorithms become a few lines of "this recovers X, here's the
  proof." Beats subjective inspection of overlay-on-photo every time.
- **Inline math equations, don't route through helpers.** When the book gives
  you an explicit equation, type the four distances out. The helpers are
  optimisations of repeated patterns; the algorithm cells are pedagogy.
- **Don't open PRs against the upstream repo without explicit per-action
  confirmation.** Same rule as Ch 38 / Ch 47 (see those `prompts.md` files).
