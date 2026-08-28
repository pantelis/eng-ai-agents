# Workflow — How this chapter was produced

This directory documents the AI-assisted workflow used to produce Chapter 47's
companion notebook. The intent is two-fold:

1. **Provenance.** Make the reasoning, tooling, and prompts behind each figure
   auditable. Anyone reading the notebook can answer "where did this come from?"
2. **Replication.** Give the next contributor (human or coding agent) a recipe
   they can re-apply to other chapters in this project (Ch 38, Ch 42, etc.).

## What's in this directory

| File / dir | Purpose |
|---|---|
| `README.md` | This document — workflow overview, decisions, and replication guide |
| `editor-scripts/build_fig47_*.py` | Python generators that produce drag-to-position HTML editors, one per editor-tuned figure (47.1 – 47.8) |
| `research-findings.md` | Verified findings from the deep-research pass on textbook-figure replication practice |
| `prompts.md` | Specific Claude prompts that produced the best results |

## The single most important workflow decision

**Tune figure layouts with a browser-based drag-and-drop editor, not by
guessing matplotlib parameters.**

Iterating on `ax.text(...)` / `Rectangle(...)` / `torch.tensor([...])` numbers in
code and re-running the notebook to check the visual is slow and error-prone.
Each round you fix one element, break another, and the per-iteration cost is
≥30 seconds (re-execute, re-render, eyeball).

The editor approach:

1. Generate a self-contained HTML page that loads the book figure as a
   semi-transparent background and overlays the matplotlib elements as draggable
   SVG handles.
2. The contributor drags handles over the book image until the layout matches.
3. The editor exports a Python snippet (explicit `torch.tensor` positions, sliders
   for sizes) that is pasted directly into the notebook cell.
4. The cell renders the matplotlib version using those explicit positions —
   re-executes in a few seconds, and the output matches the dragged layout
   because both use the same coordinate system.

Per-iteration cost drops to **a single drag** and the visual feedback is
instant. We applied this to figures 47.1, 47.2, 47.3, 47.4, 47.5, 47.6, 47.7,
and 47.8. Figures 47.9 and 47.10 are pure quiver fields (no draggable
structure) — they were correct without an editor.

## End-to-end flow used for this chapter

```text
1. Read the book chapter at visionbook.mit.edu
   → enumerate every figure and its math
2. Draft a notebook skeleton: markdown for each section + a code cell per figure
   → math first, plot stubs second
3. For each visually structural figure (47.1 - 47.8):
   a. Download the book's PNG of that figure
   b. Write editor-scripts/build_fig47_XX.py that
      - embeds the book PNG (base64) as the canvas background
      - defines a set of draggable SVG elements (points, polygons, labels)
      - emits an HTML page with drag handlers + a Python-snippet export panel
   c. Open the HTML in a browser, drag elements over the faded book image
   d. Click "Export Python"; paste the snippet into the notebook cell
   e. Re-execute the cell; verify the matplotlib output matches the book
4. For purely mathematical figures (47.9, 47.10):
   - Skip the editor; write the quiver code directly from the equations
5. Validate end-to-end execution + visual diff against the book
6. Register in notebooks/notebook-database.yml
7. Open PR against pantelis/eng-ai-agents
```

## What the AI did and did not do

**AI was the workhorse for**
- Drafting matplotlib boilerplate
- Writing the editor HTML/JS (SVG drawing, drag handlers, coordinate-system
  conversions)
- Translating editor exports into matplotlib code
- Writing the section markdown (concept + math typeset in LaTeX)
- Investigating repository conventions (CLAUDE.md, AGENTS.md, notebook-database.yml,
  Makefile, recent merged PRs as exemplars)
- Cross-checking renders against the book and proposing fixes

**Human held the wheel for**
- Picking which chapter(s) to take (38, 42, 47)
- Dragging elements in the editor to match the book — the AI cannot do this
  reliably because it cannot use a browser; the human ground truth here is
  essential
- Approving when a figure is "good enough" — the AI tends to either accept too
  early or chase the book's Illustrator quality indefinitely
- Authorizing public actions (opening PRs against the maintainer's repo)

## Validation we ran before deciding "ready to push"

| Check | Status |
|---|---|
| Notebook executes end-to-end without errors | ✓ ~8s, 0 errors |
| All 10 figures render | ✓ |
| Code cells: 13. Markdown cells: 11 | ✓ |
| Plotting-only cells tagged `hide-input` (per the brief) | ✓ 10 tagged, 3 untagged (the 3 are imports + the `project` and `motion_field` helper definitions — kept visible because they ARE the math) |
| Uses PyTorch | ✓ `motion_field`, `project`, and every coordinate/position is `torch.tensor` |
| No NumPy in the notebook | ✓ stripped per the Aegean brief — every `np.*` swapped for the `torch.*` equivalent |
| No plotting-library names in markdown | ✓ |
| `images/` subfolder with rendered figures | ✓ `images/fig47_01.png … fig47_10.png` (10 files) |
| Pure PyTorch ecosystem (no scikit-learn, no proprietary libs) | ✓ |
| One chapter per directory | ✓ `notebooks/CV/mit-foundations/chapter-47-3d-motion-projection/` |
| Branch named `mit-book-chapter-<chapter>-<section>` | ✓ `mit-book-chapter-47-2` |
| Registered in `notebooks/notebook-database.yml` | ✓ |

## Why this chapter does not import Kornia

The Aegean brief lists Kornia as the recommended library on top of PyTorch
"for differentiable computer vision". Chapter 47's whole pedagogical point is
the explicit perspective-projection formula `(x, y) = (fX/Z, fY/Z)` and the
motion field derived by differentiating it. Replacing the two-line `project`
function with `kornia.geometry.linalg.transform_points(...)` would hide
exactly the expression the chapter is teaching the reader to read. Other
chapters in this project (notably Ch 38 image warping) are a better fit for
Kornia and will use it.

## Replicating this for another chapter

For Ch 38 and Ch 42 (or any future chapter):

1. **Read the chapter** at `visionbook.mit.edu/<chapter-slug>.html` end-to-end
2. **Enumerate every figure** and its underlying equation; classify each as
   "structural" (geometric diagram with positioned elements) or "mathematical"
   (vector field, plot driven entirely by math)
3. **Set up the notebook skeleton:**
   - `notebooks/CV/mit-foundations/chapter-<XY>-<slug>/index.ipynb`
   - Title markdown + a section per chapter sub-section
   - One code cell per figure
4. **For structural figures, build an editor:**
   - Copy `editor-scripts/build_fig47_<closest-match>.py` as a starting template
   - Adapt the `DEFAULTS` element list to the new figure's needs
   - Adapt the SVG rendering logic for any custom shapes
   - Open the HTML, drag, export, paste into the cell
5. **For mathematical figures, write the math directly** in PyTorch and
   matplotlib quiver
6. **Validate** by running the notebook in the devcontainer:
   `make execute-notebook NOTEBOOK=CV/mit-foundations/chapter-<XY>-<slug>/index.ipynb`
7. **Register in `notebooks/notebook-database.yml`** following the
   `notebooks(<chapter>): publish ...` PR pattern (see merged PR #31, #38)
8. **Open PR** against `pantelis/eng-ai-agents:main` from your fork, branch
   `mit-book-chapter-<XY>-<section>`

See `prompts.md` for the specific Claude prompts that produced the best
results during this project.
