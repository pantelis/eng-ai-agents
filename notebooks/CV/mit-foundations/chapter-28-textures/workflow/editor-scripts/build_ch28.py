#!/usr/bin/env python3
"""Assemble the Chapter 28 (Textures) companion notebook.

Reconstructed from the notebook's cell sources; run:
    python build_ch28.py <path/to/index.ipynb>
to regenerate index.ipynb (with cleared outputs)."""
import json, sys
from pathlib import Path


def md(*lines):
    return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}


def code(*lines):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": _src(lines)}


def _src(lines):
    text = "\n".join(lines); parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]


cells = []

cells.append(md(
    '# Chapter 28 — Textures',
    '',
    '*Companion notebook for* **Foundations of Computer Vision** *(Torralba, Isola, Freeman), Ch. 28 — [visionbook.mit.edu](https://visionbook.mit.edu/textures.html).*',
    '',
    "A **texture** is an image made of many similar elements — what matters is the *statistics* of the elements, not their individual identity. This notebook builds the chapter's tools: generating an **infinite texture** by tiling, the **Heeger–Bergen** parametric method (match multi-scale + pixel **histograms** of white noise to a reference), the single-subband histogram-matching step, and the **Efros–Leung** nonparametric method (grow the texture pixel-by-pixel by copying from matching neighbourhoods), including how the **context size** controls the result.",
    '',
    "Textures are the book's own images — plums, zebra, pebbles, and the Efros circles (from visionbook.mit.edu / cropped from the book's figures)."))

cells.append(code(
    'import io, urllib.request',
    'import numpy as np',
    'import matplotlib.pyplot as plt',
    'from PIL import Image',
    'from scipy.ndimage import gaussian_filter',
    'from skimage.transform import resize',
    '',
    'np.random.seed(0)',
    "plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130,",
    "                     'image.cmap': 'gray', 'image.interpolation': 'nearest', 'axes.grid': False})",
    "BOOK = 'https://visionbook.mit.edu/figures'",
    '',
    '',
    'def load_url(url, gray=False):',
    '    with urllib.request.urlopen(url, timeout=30) as r:',
    '        im = Image.open(io.BytesIO(r.read()))',
    "        im = im.convert('L' if gray else 'RGB')",
    '    return np.asarray(im, np.float32) / 255.0',
    '',
    '',
    'def show(panels, titles, figsize=None):',
    '    n = len(panels); fig, ax = plt.subplots(1, n, figsize=figsize or (3.3 * n, 3.4))',
    '    if n == 1: ax = [ax]',
    '    for a, im, t in zip(ax, panels, titles):',
    "        a.imshow(np.clip(im, 0, 1), cmap=None if (np.ndim(im) == 3) else 'gray')",
    '        a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])',
    '    plt.tight_layout(); plt.show()',
    '',
    'import os',
    "_A = next(p for p in ['assets',",
    "          'notebooks/CV/mit-foundations/chapter-28-textures/assets'] if os.path.isdir(p))",
    'def book_tex(name):',
    "    return np.asarray(Image.open(f'{_A}/{name}').convert('RGB'), np.float32) / 255.0",
    '',
    "STONE = load_url(f'{BOOK}/heeger_bergen/stone_wall.jpg')                # colour stone wall",
    "CIRCLES = load_url(f'{BOOK}/statistical_image_models/efros1a.jpg', gray=True)  # binary circles",
    "PLUMS = book_tex('tex_plums.png')      # the book's plums texture (28.9, 28.12)",
    "ZEBRA = book_tex('tex_zebra.png')      # the book's zebra texture (28.9)",
    "PEBBLES = book_tex('tex_pebbles.png')  # the book's pebbles texture (28.12)",
    "print('plums', PLUMS.shape, ' pebbles', PEBBLES.shape, ' circles', CIRCLES.shape)"))

cells.append(md(
    "## 28.1 — An 'infinite' texture by cropping",
    '',
    'Because a texture is *stationary* — its statistics are the same everywhere — you can generate endless new samples of it simply by **cropping different windows** from one large reference. Each crop is a fresh, plausible piece of the same texture. (Tiling a single crop instead leaves visible seams unless you mirror the edges.)'))

cells.append(code(
    "# Figure 28.1 — the book's plums texture: reference + several crops from it.",
    'ref = PLUMS',
    'H, W = ref.shape[:2]; s = 230',
    'boxes = [(20, 20), (H - s - 20, 30), (30, W - s - 20), (H - s - 20, W - s - 20)]',
    'crops = [ref[y:y + s, x:x + s] for (y, x) in boxes]',
    "fig, axd = plt.subplot_mosaic([['ref', 'ref', 'c0', 'c1'],",
    "                               ['ref', 'ref', 'c2', 'c3']], figsize=(12, 5))",
    "axd['ref'].imshow(ref); axd['ref'].set_title('reference texture (plums)', fontsize=10)",
    'for i in range(4):',
    "    axd[f'c{i}'].imshow(crops[i])",
    "    if i == 0: axd[f'c{i}'].set_title('crops of the same texture', fontsize=10)",
    'for a in axd.values(): a.set_xticks([]); a.set_yticks([])',
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '## 28.5 — Texture = statistics: the Heeger–Bergen method',
    '',
    "Heeger & Bergen model a texture by the **histograms** of a multi-scale, multi-orientation decomposition (a steerable pyramid) plus the pixel histogram. To *synthesise*, start from white noise and repeatedly force those histograms to match the reference. We use an isotropic **Laplacian pyramid** in place of the steerable pyramid — the same histogram-matching idea — and **decorrelate colour with PCA** (as the book does), so the synthesis keeps the reference's colours instead of turning to rainbow noise."))

cells.append(code(
    'def match_hist(src, ref):',
    '    """Force src to have ref\'s histogram (exact, via sorted-value mapping)."""',
    '    s = src.ravel(); t = np.sort(ref.ravel())',
    '    order = np.argsort(np.argsort(s))                       # rank of each src pixel',
    '    idx = np.floor(order * (len(t) - 1) / max(len(s) - 1, 1)).astype(int)',
    '    return t[idx].reshape(src.shape)',
    '',
    'def lap_pyr(img, levels=4):',
    '    g = [img]',
    '    for _ in range(levels):',
    '        g.append(resize(gaussian_filter(g[-1], 2), (max(1, g[-1].shape[0] // 2),',
    '                                                    max(1, g[-1].shape[1] // 2)), anti_aliasing=True))',
    '    lap = [g[i] - resize(g[i + 1], g[i].shape, anti_aliasing=True) for i in range(levels)]',
    '    lap.append(g[-1])',
    '    return lap',
    '',
    'def reconstruct(lap):',
    '    img = lap[-1]',
    '    for i in range(len(lap) - 2, -1, -1):',
    '        img = resize(img, lap[i].shape, anti_aliasing=True) + lap[i]',
    '    return img',
    '',
    'def heeger_bergen_gray(ref, out_shape, iters=10, levels=4, clip=True):',
    '    out = np.random.RandomState(0).rand(*out_shape) * (ref.max() - ref.min()) + ref.min()',
    '    ref_lap = lap_pyr(ref, levels)',
    '    for _ in range(iters):',
    '        out = match_hist(out, ref)                          # pixel histogram',
    '        out_lap = lap_pyr(out, levels)',
    '        out_lap = [match_hist(o, r) for o, r in zip(out_lap, ref_lap)]  # subband histograms',
    '        out = reconstruct(out_lap)',
    '    out = match_hist(out, ref)',
    '    return np.clip(out, 0, 1) if clip else out',
    '',
    'def heeger_bergen_color(ref_rgb, out_shape, iters=12, levels=4):',
    '    """Colour Heeger-Bergen: PCA-decorrelate the channels, synthesise each',
    '    (now-independent) component, rotate back. Avoids the rainbow noise you get',
    '    from synthesising R,G,B separately.',
    '    """',
    '    X = ref_rgb.reshape(-1, 3); mu = X.mean(0)',
    '    evals, evecs = np.linalg.eigh(np.cov((X - mu).T))       # evecs columns = colour axes',
    '    ref_pca = ((X - mu) @ evecs).reshape(*ref_rgb.shape[:2], 3)',
    '    comps = [heeger_bergen_gray(ref_pca[..., c], out_shape, iters, levels, clip=False)',
    '             for c in range(3)]',
    '    out = np.stack(comps, -1).reshape(-1, 3) @ evecs.T + mu',
    '    return np.clip(out.reshape(*out_shape, 3), 0, 1)',
    '',
    "# Figure 28.9 — two Heeger-Bergen examples (the book's plums and zebra).",
    '# The zebra photo is a whole animal; crop a clean stripe patch so the reference is a',
    '# stationary texture (grass + head would violate the stationarity synthesis assumes).',
    'zebra_patch = ZEBRA[190:490, 410:710]',
    'fig, ax = plt.subplots(2, 2, figsize=(8, 8))',
    "for i, (nm, tex) in enumerate([('plums', PLUMS), ('zebra', zebra_patch)]):",
    '    ref = resize(tex, (160, 160), anti_aliasing=True)',
    '    synth = heeger_bergen_color(ref, (224, 224), iters=12)',
    "    ax[i, 0].imshow(ref); ax[i, 0].set_title(f'{nm} reference', fontsize=9)",
    "    ax[i, 1].imshow(synth); ax[i, 1].set_title(f'{nm} Heeger-Bergen synthesis', fontsize=9)",
    'for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])',
    'plt.tight_layout(); plt.show()',
    "# The plums (blobby) synthesise well; the zebra's oriented STRIPES need an",
    '# orientation-selective (steerable) pyramid — our isotropic pyramid blurs them.'))

cells.append(md(
    '## 28.6 — Watching the synthesis converge',
    '',
    "Starting from white noise, each round of histogram matching pushes the sample closer to the reference's statistics. After a handful of iterations the noise has become a convincing texture."))

cells.append(code(
    '# Figure 28.6 — white noise -> texture over Heeger-Bergen iterations.',
    'g = resize(STONE[60:60 + 256, 100:100 + 256].mean(-1), (128, 128), anti_aliasing=True)',
    'ref_lap = lap_pyr(g, 4)',
    "out = np.random.rand(128, 128); snaps = [('noise (0)', out.copy())]",
    'for it in range(1, 13):',
    '    out = match_hist(out, g)',
    '    out = reconstruct([match_hist(o, r) for o, r in zip(lap_pyr(out, 4), ref_lap)])',
    "    if it in (1, 3, 6, 12): snaps.append((f'iter {it}', np.clip(match_hist(out, g), 0, 1)))",
    'show([s for _, s in snaps], [n for n, _ in snaps], figsize=(15, 3.1))'))

cells.append(md(
    '## 28.8 — Histogram matching a single subband',
    '',
    "The core operation: a band-pass **subband of a texture** is heavy-tailed (Laplacian-like), while a subband of **white noise** is Gaussian. Histogram matching is a pointwise monotonic map that turns the Gaussian subband into the texture's heavy-tailed one."))

cells.append(code(
    '# Figure 28.8 — a noise subband (Gaussian) matched to a texture subband (Laplacian).',
    'g = resize(STONE[60:60 + 256, 100:100 + 256].mean(-1), (128, 128), anti_aliasing=True)',
    'tex_band = lap_pyr(g, 4)[0]                                 # a band-pass texture subband',
    'noise_band = lap_pyr(np.random.rand(128, 128), 4)[0]',
    'matched = match_hist(noise_band, tex_band)',
    'fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))',
    "for a, (d, name, col) in zip(ax, [(noise_band, 'noise subband (Gaussian)', 'C0'),",
    "                                  (tex_band, 'texture subband (Laplacian)', 'C3'),",
    "                                  (matched, 'noise after matching', 'C2')]):",
    '    a.hist(d.ravel() / d.std(), bins=100, range=(-6, 6), density=True, log=True, color=col)',
    '    a.set_title(name, fontsize=10); a.set_ylim(1e-4, 1)',
    'plt.tight_layout(); plt.show()',
    "print('kurtosis  noise: %.1f  texture: %.1f  matched: %.1f'",
    '      % tuple(float(((x - x.mean())**4).mean() / (x.var()**2)) for x in (noise_band, tex_band, matched)))'))

cells.append(md(
    '## 28.11 — Efros–Leung: the context window controls everything',
    '',
    'Efros & Leung grow a texture **one pixel at a time**: for each new pixel, search the sample for neighbourhoods (an $w\\times w$ window) similar to the already-synthesised context, and copy a matching centre pixel. A **small** window reproduces local detail but loses the layout; a **large** window preserves the regular arrangement.'))

cells.append(code(
    'from numpy.lib.stride_tricks import sliding_window_view',
    '',
    'from scipy.ndimage import binary_dilation, uniform_filter',
    '',
    'def efros_leung(sample, out_size, w=11, seed=0):',
    '    """Grow an out_size texture from a sample; works for gray (H,W) or colour (H,W,3)."""',
    '    S = sample if sample.ndim == 3 else sample[..., None]      # (Hs,Ws,C)',
    '    C = S.shape[2]; h = w // 2; rng = np.random.RandomState(seed)',
    '    P = sliding_window_view(S, (w, w), axis=(0, 1))            # (.,.,C,w,w)',
    '    P = P.transpose(0, 1, 3, 4, 2).reshape(-1, w, w, C)        # (Np,w,w,C)',
    '    centres = P[:, h, h, :]                                    # (Np,C)',
    '    out = np.full((out_size, out_size, C), np.nan)',
    '    fil = np.zeros((out_size, out_size), bool)',
    '    sy, sx = rng.randint(0, S.shape[0] - 3), rng.randint(0, S.shape[1] - 3)',
    '    c = out_size // 2 - 1',
    '    out[c:c + 3, c:c + 3] = S[sy:sy + 3, sx:sx + 3]; fil[c:c + 3, c:c + 3] = True',
    '    pad = np.pad(out, ((h, h), (h, h), (0, 0)), constant_values=0.0)',
    '    mpad = np.pad(fil, h, constant_values=False)',
    '    while not fil.all():',
    '        border = binary_dilation(fil) & ~fil',
    "        counts = uniform_filter(fil.astype(float), w, mode='constant')",
    '        ys, xs = np.where(border); order = np.argsort(-counts[ys, xs])',
    '        for k in order[:250]:',
    '            y, x = ys[k], xs[k]',
    '            win = np.nan_to_num(pad[y:y + w, x:x + w])         # (w,w,C)',
    '            mask = mpad[y:y + w, x:x + w].astype(float)[..., None]',
    '            if mask.sum() == 0: continue',
    '            ssd = (((P - win) ** 2) * mask).sum((1, 2, 3)) / (mask.sum() * C)',
    '            cand = np.where(ssd <= ssd.min() * 1.3 + 1e-6)[0]',
    '            val = centres[cand[rng.randint(len(cand))]]',
    '            out[y, x] = val; fil[y, x] = True; pad[y + h, x + h] = val; mpad[y + h, x + h] = True',
    '    return out[..., 0] if C == 1 else np.clip(out, 0, 1)',
    '',
    "# Figure 28.11 — same sample, small vs large context window (the book's circles).",
    'samp = resize(CIRCLES, (56, 72), anti_aliasing=True)',
    'samp = (samp > 0.5).astype(float)                          # clean binary circles',
    'small = efros_leung(samp, 72, w=5)',
    'large = efros_leung(samp, 72, w=15)',
    'show([samp, small, large],',
    "     ['sample texture', 'small context (w=5): layout lost', 'large context (w=15): regular'],",
    '     figsize=(11, 4.2))'))

cells.append(md(
    '## 28.12 — Efros–Leung on a natural texture',
    '',
    'On a natural stone texture, a large enough context grows a larger image that **preserves the pebble structure** — coherent elements, not just matched statistics (contrast the Heeger–Bergen output above, which scrambles the arrangement).'))

cells.append(code(
    "# Figure 28.12 — Efros-Leung on the book's plums and pebbles (colour).",
    'fig, ax = plt.subplots(2, 2, figsize=(8, 8))',
    "for i, (nm, tex) in enumerate([('plums', PLUMS), ('pebbles', PEBBLES)]):",
    '    src = resize(tex, (48, 48), anti_aliasing=True)',
    '    grown = efros_leung(src, 80, w=11)',
    "    ax[i, 0].imshow(src); ax[i, 0].set_title(f'{nm} sample (48x48)', fontsize=9)",
    "    ax[i, 1].imshow(grown); ax[i, 1].set_title(f'{nm} Efros-Leung (80x80)', fontsize=9)",
    'for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])',
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '## 28.13 — Concluding remarks',
    '',
    '| Method | Idea | Strength / weakness |',
    '|---|---|---|',
    '| tiling | repeat a crop (mirror the seams) | trivial; globally periodic |',
    '| Heeger–Bergen | match multi-scale + pixel **histograms** of noise | fast, parametric; scrambles global layout |',
    "| Efros–Leung | copy pixels from matching **neighbourhoods** | preserves structure; slow, can 'grow garbage' |",
    '',
    'Texture is captured by **statistics of local elements**. Parametric (Heeger–Bergen) and nonparametric (Efros–Leung) synthesis trade speed against structural fidelity — the same tension that later reappears in learned (GAN / diffusion) texture and image models.'))

nb = {"cells": cells, "metadata": {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(sys.argv[1]); out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(cells)} cells)")
