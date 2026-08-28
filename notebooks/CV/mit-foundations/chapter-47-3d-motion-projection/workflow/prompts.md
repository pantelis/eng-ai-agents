# Claude prompts that produced the best results

A short catalog of the prompts and patterns that consistently moved this
project forward. Use these as templates for other chapters.

## Reading the chapter end-to-end

> "I need a COMPLETE detailed breakdown of Chapter <XY> '<title>'.
> Provide: (1) every section heading in order, (2) every figure with its full
> caption and the math/concept it illustrates — flagging which figures are
> real photographs vs synthetic diagrams vs computational overlays, (3) every
> numbered equation with LHS and a one-line description, (4) any algorithm or
> pseudocode (especially for ...), (5) any specific objects/scenes used.
> Be exhaustive — I'm reproducing every figure I can."

Use with `WebFetch` against `https://visionbook.mit.edu/<chapter-slug>.html`.

## Inspecting one figure carefully

> "I am replicating Chapter <XY> figures in code. For figure <XY>.<N>, describe
> in detail: (1) what visual style is used — diagram with arrows, vector field
> / quiver, 3D scene, photograph etc., (2) which specific scene objects appear,
> (3) the orientation/layout — single panel or multi-panel, arrows pointing
> which way, ..., (4) any color choices or annotations mentioned, (5) what
> exact configuration of camera/world is shown."

Use against the chapter URL or against the locally downloaded `fig<XY>_<N>.png`
via the `Read` tool (multimodal — Claude reads the actual image).

## Building a per-figure drag-and-drop editor

> "Build a drag-to-position editor for Fig <XY>.<N>. The book PNG is at
> output/book-figures-<XY>/fig<XY>_<N>.png. Embed the PNG (base64) as a faded
> background in an HTML page. Define draggable SVG elements for: <list each
> structural element>. The editor should export Python (torch.tensor(...) values
> + slider parameter constants) that the notebook cell can paste directly.
> Model the build script on workflow/editor-scripts/build_fig<closest>.py."

The key insight: each editor is a SINGLE self-contained HTML file with the
drag-handler JS, the book PNG embedded as base64, the SVG rendering logic, and
a panel that emits Python code as the contributor drags. No server required.

## Updating a notebook cell from an editor export

> "The editor exported the following:
> <paste export>
> Update cell-<id> in the notebook to consume these exact values. Use
> FancyArrowPatch for arrows, ax.scatter for point markers, ax.add_patch
> with Polygon/Rectangle/Circle for shapes. Plotting-only cells should be
> tagged hide-input. Render at figsize matching the book's aspect ratio."

## Verifying a figure matches the book

> "Compare my rendered fig47_<N>.png against the book's fig47_<N>.png. List:
> (1) elements that visually MATCH, (2) elements that are MISSING from my
> version, (3) elements that are EXTRA in mine, (4) any orientation / aspect /
> color discrepancies. Be specific. Don't propose fixes — I want a checklist."

Use the `Read` tool on both images. Claude is multimodal and can compare PNGs
directly.

## Authoring the workflow documentation

> "Create workflow/README.md documenting the AI-assisted workflow used for
> this chapter. Cover: (1) the single most important workflow decision and
> why, (2) what the AI did vs what the human did, (3) the validation checks
> we ran before deciding 'ready to push', (4) a replication recipe for
> another chapter. Keep it terse — readers will skim."

## Anti-patterns to avoid

These prompts produced bad results in this project:

- **"Make it look more like the book"** without specifying which element is
  off. Result: Claude makes 10 micro-changes, half are improvements, half are
  regressions.
- **"Iterate on Fig X until it matches"** — open-ended; consumed many turns
  without convergence. Replace with the editor approach.
- **"Should we use Manim / Kornia / TikZ?"** without first checking the actual
  repository conventions. Result: speculation. Always grep the repo's existing
  code first.
- **"Push the PR"** — too imperative for a public-facing action. Always
  separately confirm before `gh pr create` against an upstream public repo.
  (See [`feedback_pr_opening_authorization`](../../../../../memory) memory.)

## Patterns that recurred

- **Read the book first, code second.** The cost of mis-understanding a
  figure's intent and writing matching code is much higher than the cost of
  spending five extra minutes reading the chapter.
- **Use the editor for layouts, code for math.** They are different concerns
  and the editor isolates the visual-layout iteration loop from the
  mathematical correctness loop.
- **Compare side-by-side, frequently.** A single re-rendered `compare.html`
  with all 10 book vs mine pairs catches problems faster than visualizing one
  figure at a time.
- **Don't open PRs against the upstream repo without explicit per-action
  confirmation.** "Go" authorizes local + fork-side work, never `gh pr create`
  against `pantelis/eng-ai-agents`.
