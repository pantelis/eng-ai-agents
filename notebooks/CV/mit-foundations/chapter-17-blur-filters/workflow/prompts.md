# Prompts — Ch 17 Blur Filters

The prompts that drove the useful work, in the order they were used.

## 1. Digest the chapter

> Extract the chapter's math in detail. For each figure (17.1 … last), state what
> it shows. Give the exact definitions for: (1) the box/average kernel and its
> normalization; (2) the continuous Gaussian, its discretization, separability,
> and sigma-vs-kernel-size; (3) binomial filters from Pascal's triangle / repeated
> [1,1] convolution and how they approximate a Gaussian; (4) the Fourier-domain
> interpretation of each filter. List every named example image.

Output became `research-findings.md`.

## 2. Draft the notebook

> Write the companion notebook. One section per book section
> (Intro → Box → Gaussian → Binomial → Concluding). Math in LaTeX markdown, one
> code cell per figure. Implement box/Gaussian/binomial as torch convolutions
> sharing a single reflect-padded `conv2d` helper. Use `skimage.data.camera` — do
> not redistribute the book's photos.

## 3. Make every figure defensible

> For each qualitative figure add a numerical check printed in the same cell:
> separability error (2D vs cascaded 1D), Gaussian composition
> `s3^2 = s1^2 + s2^2`, binomial variance `= n/4`, and box-vs-binomial
> checkerboard residual. The reader should see the claim *proved*, not just
> illustrated.

## 4. Validate in the maintainer container

> Execute via `docker compose run --rm torch.dev.cpu bash -c "python
> scripts/execute_notebook.py CV/mit-foundations/chapter-17-blur-filters/index.ipynb"`.
> Fix any errors, confirm all figures render, then copy the extracted PNGs into
> `images/fig17_01..09.png` and add the `notebook-database.yml` entry.

## Lessons carried forward

- **Computed figures need no editor.** Unlike Ch 47's geometric diagrams, every
  element here is fixed by the math — the drag-editor step was correctly skipped.
- **A number beside every picture.** The self-checks (separability ~1e-6,
  composition ~1e-8, exact variance, checkerboard cancellation) are what make the
  notebook a *proof* rather than a gallery.
- **Windows + Compose gotcha.** `MSYS_NO_PATHCONV=1` and an explicit `-e PATH`
  are needed or the container can't find `bash`/`python` (host `$PATH` leaks in).
