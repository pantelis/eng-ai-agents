# Workflow — How this chapter was produced

Companion notebook for Chapter 19 (Temporal Filters), following the
Ch 17/18/38/42/47 pattern.

## What's in this directory
| File | Purpose |
|---|---|
| `README.md` | approach, decisions, replication recipe |
| `research-findings.md` | the chapter's math digested from visionbook.mit.edu |
| `prompts.md` | the prompts that drove drafting and validation |
| `editor-scripts/` | (empty) — figures are computed; see NOTE.md |

## Source data — a real pedestrian video (OpenCV `vtest.avi`)

The book's motion figures use a colour pedestrian video that isn't distributable.
We use **OpenCV's `vtest.avi`** instead — a freely redistributable
(BSD/Apache-licensed) static-camera clip of people walking across a plaza, the
same kind of scene. 56 colour frames were pre-extracted and downsampled to a
compact asset (`assets/ped_seq.npz`, ~2.7 MB) so **the notebook needs no video
decoder at run time** — it just `np.load`s the frames. Because the camera is
static, the background is fixed and the people move at a few pixels per frame —
exactly the dense, small-per-frame motion the temporal filters require.

This real footage drives the space-time figures:
- an **x-t slice** shows the static plaza as vertical streaks and each walker as a
  **diagonal streak** (Fig 19.1);
- **velocity-tuned blur** keeps whatever moves at the tuned velocity sharp —
  v=0 keeps the background crisp, a walker's velocity brings that walker into
  focus (Fig 19.4);
- the **velocity-nulling filter** at v=0 removes the static background so only the
  moving people remain (static energy 0.18 → 0.001); velocity-specific nulling
  suppresses the matching walker (Fig 19.7).

The purely mathematical figures (moving-pulse Fourier 19.2, the spatiotemporal
Gaussian kernel 19.3, its derivatives 19.5/19.6) are computed from the equations.

### Re-extracting the asset (reproducible)
`vtest.avi` → `ped_seq.npz` was produced locally with `imageio` (ffmpeg):
frames 250 : 250+2·56 : 2 of the clip, bilinear-resized to 160 px wide, stacked
to a `uint8 (56,120,160,3)` array and `np.savez_compressed`. Only the derived
`ped_seq.npz` is committed (not the raw video).

## Figures reproduced (9 panels across 7 cells)

19.1 space-time volume (frames + x-t slice) · 19.2 moving pulse + space-time
|DFT| on the constraint line · 19.3 spatiotemporal Gaussian (standard vs
velocity-skewed) · 19.4 velocity-tuned blur · 19.5/19.6 spatiotemporal
derivatives (g_t, g_x, g_y + g_t frame response) · 19.7 velocity-nulling filter
(x-t slices + frames).

## Key implementation
- The (P,H,W,3) colour volume is loaded from `assets/ped_seq.npz`.
- 3D kernels on a (t,y,x) grid: `st_gaussian` (skewable by velocity via a shear
  `x - v_x t`), `st_deriv` for g_t/g_x/g_y.
- `conv3d_seq` / `conv3d_rgb` filter the volume with `F.conv3d` (reflect-padded).
- The nulling filter is `h = g_t + v_x g_x + v_y g_y`; nulling v=0 (pure `g_t`)
  drives the static-background energy from 0.18 to ~0.001.
- Derivative / nulling outputs are shown **signed around mid-gray** (`show_signed`),
  as the book renders them.

## Validation command
```bash
docker compose run --rm torch.dev.cpu \
  bash -c "python scripts/execute_notebook.py \
    CV/mit-foundations/chapter-19-temporal-filters/index.ipynb"
```
15 cells, **~41 s**, **9 figures**, no errors. CPU-only.

> Windows note: prefix with `MSYS_NO_PATHCONV=1` and pass an explicit
> `-e PATH=/workspaces/eng-ai-agents/.venv/bin:/usr/local/bin:/usr/bin:/bin`,
> or the container can't find `bash`/`python`.

## Honest caveats
- The clip is OpenCV's `vtest.avi`, not the book's exact Istanbul scene, but it
  is real static-camera pedestrian footage — the same kind of scene.
- Figures are computed from the equations, so (as in Ch 17/18) no drag editor
  was needed.
