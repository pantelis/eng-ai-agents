#!/usr/bin/env python3
"""Assemble the Chapter 27 (Statistical Image Models) companion notebook.

Reconstructed from the notebook's cell sources; run:
    python build_ch27.py <path/to/index.ipynb>
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
    '# Chapter 27 — Statistical Image Models',
    '',
    '*Companion notebook for* **Foundations of Computer Vision** *(Torralba, Isola, Freeman), Ch. 27 — [visionbook.mit.edu](https://visionbook.mit.edu/stat_image_models_revised.html).*',
    '',
    'Natural images are a **vanishingly small, highly structured** corner of the space of all pixel arrays. This notebook measures that structure and builds generative priors from it: the **1/f power law**, sampling clouds from it, why an independent-pixel model fails, the **decay of pixel correlations**, the **heavy-tailed (generalized-Laplacian) statistics of derivatives**, and how those priors drive **denoising** — Wiener filtering, wavelet **coring**, and **non-local means**.',
    '',
    "Every photo is one of the **book's own images** — its MIT-dome, wheel, street scene, autumn leaves, colourful houses, doorway, hair, and Barcelona building (loaded from visionbook.mit.edu or cropped from the book's composite figures)."))

cells.append(code(
    'import io, urllib.request',
    'import numpy as np',
    'import matplotlib.pyplot as plt',
    'from PIL import Image',
    'from scipy.special import gamma',
    'from skimage.color import rgb2gray',
    'from skimage.transform import resize',
    '',
    'np.random.seed(0)',
    "plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130,",
    "                     'image.cmap': 'gray', 'axes.grid': False})",
    "BOOK = 'https://visionbook.mit.edu/figures'",
    '',
    '',
    'def load_url_rgb(url):',
    '    with urllib.request.urlopen(url, timeout=30) as r:',
    "        return np.asarray(Image.open(io.BytesIO(r.read())).convert('RGB'), np.float32) / 255.0",
    '',
    '',
    'def to_gray_square(img, size=256):',
    '    """Grayscale, centre-crop to square, resize to size x size, in [0,1]."""',
    '    g = rgb2gray(img) if img.ndim == 3 else img',
    '    h, w = g.shape; s = min(h, w)',
    '    g = g[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]',
    '    return resize(g, (size, size), anti_aliasing=True)',
    '',
    '',
    'def radial_profile(mag):',
    '    """Angular-averaged radial profile of a 2D (fft-shifted) magnitude image."""',
    '    h, w = mag.shape; cy, cx = h // 2, w // 2',
    '    y, x = np.indices((h, w))',
    '    r = np.sqrt((x - cx)**2 + (y - cy)**2).astype(int)',
    '    tbin = np.bincount(r.ravel(), mag.ravel())',
    '    nr = np.bincount(r.ravel())',
    '    return tbin / np.maximum(nr, 1)',
    '',
    '',
    'def gen_laplacian(x, r, s=1.0):',
    '    """Generalized Laplacian pdf  exp(-|x/s|^r) / (2 (s/r) Gamma(1/r))."""',
    '    return np.exp(-np.abs(x / s)**r) / (2 * (s / r) * gamma(1.0 / r))',
    '',
    "# The book's own images. Two are on visionbook.mit.edu as standalone files; the",
    "# rest are cropped from the book's composite figures and bundled under assets/.",
    'import os',
    "_A = next(p for p in ['assets',",
    "          'notebooks/CV/mit-foundations/chapter-27-statistical-image-models/assets']",
    '          if os.path.isdir(p))',
    'def book_img(name):',
    "    return np.asarray(Image.open(f'{_A}/{name}').convert('RGB'), np.float32) / 255.0",
    '',
    "DOME = load_url_rgb(f'{BOOK}/derivatives/mit_der_a.jpg')          # MIT dome (book)",
    "WHEEL = load_url_rgb(f'{BOOK}/spatial_filters/wheel256.jpg')       # wheel (book)",
    "STREET = book_img('book_street.png')      # grayscale street scene (book)",
    "LEAVES = book_img('book_leaves.png')      # autumn leaves, colour (book)",
    "BURANO = book_img('book_burano.png')      # colourful houses, colour (book)",
    "BUILDING = book_img('book_building.png')  # doorway / storefront, gray (book)",
    "HAIR = book_img('book_hair.png')          # hair texture, gray (book)",
    "BCN = book_img('book_bcn.png')            # Barcelona building, gray (book)",
    "STARS = book_img('book_stars.png'); CLOUDS = book_img('book_clouds.png')",
    "PLUMS = book_img('book_plums.png'); CUBE = book_img('book_cube.png')   # 27.6 originals (book)",
    "print('loaded book images; e.g. street', STREET.shape, ' leaves', LEAVES.shape)",
    '',
    '',
    'def _kurt(a):',
    '    a = a - a.mean(); return float((a**4).mean() / (a.var()**2 + 1e-12))',
    ''))

cells.append(md(
    '## 27.10 — The 1/f power law',
    '',
    'The single most robust statistic of natural images: the Fourier magnitude falls off as a **power law** in radial frequency,',
    '',
    '$$\\lVert\\mathscr L(u,v)\\rVert \\;\\simeq\\; \\frac{1}{w^{\\alpha}},\\qquad w=\\sqrt{u^2+v^2},\\ \\ \\alpha\\approx 1\\text{–}1.5.$$',
    '',
    'Three real photos concentrate their energy at low frequency and their angular-averaged spectra hug the $1/w^{1.5}$ curve; **white noise is flat** — no power law at all.'))

cells.append(code(
    '# Figure 27.10 — images (top, COLOUR as in the book), |FFT| (mid), radial spectra (bottom).',
    'def sq_col(img, size=256):',
    '    h, w = img.shape[:2]; s = min(h, w)',
    '    c = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]',
    '    return resize(c, (size, size), anti_aliasing=True)',
    '',
    'noise_rgb = np.random.rand(256, 256, 3)                       # colour white noise (book uses RGB)',
    "imgs = [('wheel', sq_col(WHEEL)), ('MIT dome', sq_col(DOME)),",
    "        ('autumn leaves', sq_col(LEAVES)), ('white noise', noise_rgb)]",
    'fig, ax = plt.subplots(3, 4, figsize=(13, 9.5))',
    'w = np.arange(1, 128)',
    'for j, (name, disp) in enumerate(imgs):',
    '    g = disp.mean(-1) if disp.ndim == 3 else disp          # luminance for the FFT',
    '    ax[0, j].imshow(np.clip(disp, 0, 1)); ax[0, j].set_title(name, fontsize=10)',
    '    mag = np.abs(np.fft.fftshift(np.fft.fft2(g - g.mean())))',
    "    ax[1, j].imshow(np.log(mag + 1e-3), cmap='gray'); ax[1, j].set_title('log |FFT|', fontsize=9)",
    '    prof = radial_profile(mag)[1:128]; prof = prof / prof.max()',
    "    ax[2, j].plot(w, prof, 'k', lw=2, label='angular-avg')",
    "    ax[2, j].plot(w, (w[0] / w)**1.0, 'r--', lw=1, label='1/w')",
    "    ax[2, j].plot(w, (w[0] / w)**1.5, 'g--', lw=1, label='1/w^1.5')",
    "    ax[2, j].plot(w, (w[0] / w)**2.0, 'c--', lw=1, label='1/w^2')",
    "    ax[2, j].set_ylim(0, 1.05); ax[2, j].set_xlabel('radial freq w', fontsize=8)",
    "    if j == 0: ax[2, j].legend(fontsize=7); ax[2, j].set_ylabel('avg |FFT|')",
    'for a in ax[:2].ravel(): a.set_xticks([]); a.set_yticks([])',
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '## 27.11 — Sampling from the power spectrum',
    '',
    'A Gaussian image prior with a $1/(1+w^{\\alpha})$ power spectrum is easy to sample: take that magnitude, attach **random phase**, and inverse-transform. The result has the right spectral falloff but no real structure — it always looks like **clouds**, in grayscale or (sampling each channel) in colour.'))

cells.append(code(
    '# Figure 27.11 — cloud samples from a 1/(1+w^1.5) spectrum, gray (top) + RGB (bottom).',
    'def one_over_f_sample(size=256, alpha=1.5, rng=None):',
    '    rng = rng or np.random',
    '    fy = np.fft.fftfreq(size)[:, None]; fx = np.fft.fftfreq(size)[None, :]',
    '    w = np.sqrt(fx**2 + fy**2); mag = 1.0 / (1.0 + (size * w)**alpha)',
    '    phase = np.exp(2j * np.pi * rng.rand(size, size))',
    '    img = np.real(np.fft.ifft2(mag * phase))',
    '    return (img - img.min()) / (np.ptp(img) + 1e-9)',
    '',
    'fig, ax = plt.subplots(2, 4, figsize=(13, 6.5))',
    'for j in range(4):',
    "    ax[0, j].imshow(one_over_f_sample(), cmap='gray')",
    '    rgb = np.stack([one_over_f_sample() for _ in range(3)], axis=-1)',
    '    ax[1, j].imshow(np.clip(rgb, 0, 1))',
    "ax[0, 0].set_ylabel('grayscale'); ax[1, 0].set_ylabel('RGB (per-channel)')",
    'for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])',
    "plt.suptitle('samples from a 1/(1+w^1.5) power spectrum + random phase', y=1.0)",
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '## 27.6 — Why an independent-pixel model fails',
    '',
    "The simplest model treats every pixel as an independent draw from one histogram. Sampling from it keeps the image's **colour/intensity distribution** exactly but destroys all spatial arrangement — the sample is just noise with the right histogram. (It only ever looks right for images that *are* essentially unstructured, like a star field.)"))

cells.append(code(
    "# Figure 27.6 — original (top) vs independent-pixel sample (bottom), the book's",
    '# four images. An iid draw from the exact colour histogram = a random shuffle of',
    '# the pixels: same histogram, all structure gone.',
    'def iid_sample(img):',
    '    flat = img.reshape(-1, img.shape[-1])',
    '    return flat[np.random.permutation(len(flat))].reshape(img.shape)',
    '',
    'def sq(img, size=200):',
    '    h, w = img.shape[:2]; s = min(h, w)',
    '    return resize(img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s],',
    '                  (size, size), anti_aliasing=True)',
    '',
    "pics = [('star field', sq(STARS)), ('clouds', sq(CLOUDS)),",
    "        ('plums', sq(PLUMS)), ('green cube', sq(CUBE))]",
    'fig, ax = plt.subplots(2, 4, figsize=(13, 6.6))',
    'for j, (name, im) in enumerate(pics):',
    '    ax[0, j].imshow(np.clip(im, 0, 1)); ax[0, j].set_title(name, fontsize=10)',
    '    ax[1, j].imshow(np.clip(iid_sample(im), 0, 1))',
    "ax[0, 0].set_ylabel('original', fontsize=11); ax[1, 0].set_ylabel('iid sample\\n(same histogram)', fontsize=11)",
    'for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])',
    "plt.suptitle('only the star field survives an independent-pixel model', y=1.0)",
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '## 27.8 — Pixel correlations decay with distance',
    '',
    'Neighbouring pixels are highly correlated; the correlation falls as they move apart. Scatter plots of $\\ell[n]$ vs $\\ell[n+d]$ tighten around the diagonal for small $d$ and spread out for large $d$.'))

cells.append(code(
    '# Figure 27.8 — pixel-pair scatter at increasing horizontal distance d.',
    'g = to_gray_square(STREET, 256)',
    'fig, ax = plt.subplots(1, 3, figsize=(11, 3.7))',
    'for a, d in zip(ax, [1, 8, 40]):',
    '    p, q = g[:, :-d].ravel(), g[:, d:].ravel()',
    '    idx = np.random.choice(p.size, 4000, replace=False)',
    "    a.scatter(p[idx], q[idx], s=2, alpha=0.2, c='k')",
    '    rho = np.corrcoef(p, q)[0, 1]',
    "    a.set_title(f'd = {d} px,  corr = {rho:.2f}', fontsize=10)",
    "    a.set_xlabel('l[n]'); a.set_ylabel('l[n+d]'); a.set_aspect('equal')",
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '## 27.15 — Derivatives are heavy-tailed, not Gaussian',
    '',
    'The histogram of raw intensities is broad and near-uniform, but the histogram of **image derivatives** is sharply peaked at zero with heavy tails — most of an image is smooth (derivative $\\approx 0$) with rare large jumps at edges. On a log scale the derivative histogram is a **cusp**, nothing like a parabola (Gaussian).'))

cells.append(code(
    '# Figure 27.15 — image, dx, dy, and the intensity vs derivative histograms.',
    'g = to_gray_square(STREET, 256)',
    'dx = g[:, 1:] - g[:, :-1]',
    'dy = g[1:, :] - g[:-1, :]',
    'fig = plt.figure(figsize=(13, 3.3))',
    "for i, (im, t) in enumerate([(g, 'image'), (0.5 + 4 * np.pad(dx, ((0, 0), (0, 1))), 'd/dx'),",
    "                             (0.5 + 4 * np.pad(dy, ((0, 1), (0, 0))), 'd/dy')]):",
    "    a = fig.add_subplot(1, 5, i + 1); a.imshow(np.clip(im, 0, 1), cmap='gray')",
    '    a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])',
    'a = fig.add_subplot(1, 5, 4)',
    "a.hist(g.ravel(), bins=60, density=True, color='0.4'); a.set_title('intensity hist', fontsize=9)",
    'a = fig.add_subplot(1, 5, 5)',
    "a.hist(dx.ravel(), bins=200, density=True, color='C3', log=True)",
    "a.set_title('derivative hist (log y)', fontsize=9); a.set_xlim(-0.4, 0.4)",
    'plt.tight_layout(); plt.show()',
    "print('kurtosis  intensity: %.1f   derivative: %.1f  (Gaussian = 3)'",
    '      % (_kurt(g.ravel()), _kurt(dx.ravel())))'))

cells.append(md(
    '## 27.17 — The generalized Laplacian',
    '',
    'Derivative statistics are fit by the **generalized Laplacian**',
    '',
    '$$p(x)\\propto \\exp\\!\\big(-|x/s|^{\\,r}\\big),$$',
    '',
    'with $r\\!=\\!2$ Gaussian, $r\\!=\\!1$ Laplacian, and **$r\\in[0.4,0.8]$ for natural images** — sharper peak, heavier tails than a Gaussian.'))

cells.append(code(
    '# Figure 27.17 — generalized-Laplacian shapes for r = 0.1, 1, 2, 10.',
    'x = np.linspace(-4, 4, 600)',
    'fig, ax = plt.subplots(1, 4, figsize=(13, 2.9))',
    'for a, r in zip(ax, [0.1, 1.0, 2.0, 10.0]):',
    "    a.plot(x, gen_laplacian(x, r), 'C0'); a.set_title(f'r = {r}', fontsize=10)",
    "    a.axhline(0, color='k', lw=0.5)",
    "plt.suptitle('p(x) ~ exp(-|x/s|^r):  r<1 natural images,  r=1 Laplacian,  r=2 Gaussian', y=1.02)",
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '## 27.16 — [1,-1] statistics: noise stays Gaussian, images do not',
    '',
    "Following the book: three 'visual worlds' — **Gaussian noise**, the book's **doorway building**, and its **hair texture** — each with its image, its intensity histogram, its $[1,-1]$ derivative, and the derivative histogram (red) with the best Gaussian fit (black). Noise's derivative stays Gaussian; both images give the *same* sharply-peaked, heavy-tailed shape the Gaussian fit misses."))

cells.append(code(
    '# Figure 27.16 — image | intensity hist | [1,-1] output | output hist, x 3 worlds.',
    "worlds = [('Gaussian noise', np.clip(0.5 + 0.15 * np.random.randn(256, 256), 0, 1)),",
    "          ('doorway building', to_gray_square(BUILDING)),",
    "          ('hair texture', to_gray_square(HAIR))]",
    'fig, ax = plt.subplots(3, 4, figsize=(13, 9))',
    'for i, (name, g) in enumerate(worlds):',
    '    d = g[:, 1:] - g[:, :-1]                                  # [1,-1] derivative',
    "    ax[i, 0].imshow(g, cmap='gray'); ax[i, 0].set_ylabel(name, fontsize=10)",
    '    ax[i, 0].set_xticks([]); ax[i, 0].set_yticks([])',
    "    ax[i, 1].hist(g.ravel(), bins=80, color='C3', density=True); ax[i, 1].set_yticks([])",
    "    ax[i, 2].imshow(0.5 + 4 * d, cmap='gray', vmin=0, vmax=1)",
    '    ax[i, 2].set_xticks([]); ax[i, 2].set_yticks([])',
    '    v = d / d.std(); h, e = np.histogram(v, bins=120, range=(-6, 6), density=True)',
    '    c = 0.5 * (e[1:] + e[:-1])',
    "    ax[i, 3].semilogy(c, h + 1e-5, 'C3', lw=2, label='true pdf')",
    "    ax[i, 3].semilogy(c, np.exp(-c**2 / 2) / np.sqrt(2 * np.pi) + 1e-5, 'k--', lw=1, label='Gaussian fit')",
    '    ax[i, 3].set_ylim(1e-4, 1)',
    '    if i == 0:',
    "        for j, t in enumerate(['image', 'intensity hist', '[1,-1] output', '[1,-1] hist (log)']):",
    '            ax[i, j].set_title(t, fontsize=10)',
    '        ax[i, 3].legend(fontsize=7)',
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '## 27.14 — Denoising with the Gaussian (1/f) prior: the Wiener filter',
    '',
    'Under a Gaussian prior with power spectrum $S(w)=A/w^{2\\alpha}$ and white noise of variance $\\sigma^2$, the MAP estimate is the **Wiener filter**',
    '',
    '$$\\mathscr L(w)=\\frac{S(w)}{S(w)+\\sigma^2}\\,\\mathscr L_g(w),$$',
    '',
    'which keeps low frequencies (where the image dominates) and suppresses high frequencies (where noise dominates).'))

cells.append(code(
    '# Figure 27.14 — clean, noisy, Wiener-denoised, and the removed noise.',
    'g = to_gray_square(BCN, 256); sigma = 0.12',
    'noisy = g + sigma * np.random.randn(*g.shape)',
    'fy = np.fft.fftfreq(256)[:, None]; fx = np.fft.fftfreq(256)[None, :]',
    'w = np.sqrt(fx**2 + fy**2) * 256 + 1e-3                   # fftfreq order (unshifted)',
    '# Wiener gain  S/(S+sigma^2) for a 1/f^{2a} prior S = A/w^{2a}; writing A via the',
    '# crossover w0 where signal power = noise power gives  H = 1/(1 + (w/w0)^{2a}).',
    'alpha, w0 = 1.5, 45.0',
    'H = 1.0 / (1.0 + (w / w0)**(2 * alpha))',
    'den = np.real(np.fft.ifft2(np.fft.fft2(noisy - noisy.mean()) * H)) + noisy.mean()',
    "show = [(g, 'clean'), (noisy, 'noisy (sigma=0.12)'), (den, 'Wiener denoised'), (0.5 + (noisy - den), 'removed')]",
    'fig, ax = plt.subplots(1, 4, figsize=(14, 3.6))',
    'for a, (im, t) in zip(ax, show):',
    "    a.imshow(np.clip(im, 0, 1), cmap='gray'); a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])",
    'plt.tight_layout(); plt.show()',
    "print('RMSE  noisy: %.4f   denoised: %.4f' % (np.sqrt(((noisy - g)**2).mean()), np.sqrt(((den - g)**2).mean())))"))

cells.append(md(
    '## 27.21 — Wavelet denoising as coring',
    '',
    'With a **Laplacian prior** on a band-pass coefficient and Gaussian noise, the MAP estimate shrinks small coefficients toward zero and leaves large ones almost untouched — a **coring** curve. Small (probably-noise) responses are cored out; strong (probably-signal) responses survive.'))

cells.append(code(
    '# Figure 27.21 — the coring (shrinkage) curve for a Laplacian prior.',
    'def coring(xhat, s=0.3, sigma=0.5):',
    '    # MAP of  p(x) ~ exp(-|x|/s) * exp(-(x-xhat)^2/2sigma^2): soft-threshold by sigma^2/s.',
    '    t = sigma**2 / s',
    '    return np.sign(xhat) * np.maximum(np.abs(xhat) - t, 0.0)',
    '',
    'xhat = np.linspace(-3, 3, 400)',
    'fig, ax = plt.subplots(figsize=(4.6, 4.2))',
    "ax.plot(xhat, xhat, 'k--', lw=1, label='identity (no denoise)')",
    "ax.plot(xhat, coring(xhat), 'C0', lw=2, label='coring (MAP)')",
    "ax.set_xlabel('observed coefficient'); ax.set_ylabel('estimated coefficient')",
    "ax.legend(fontsize=8); ax.set_aspect('equal'); ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)",
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '## 27.23 — Non-local means',
    '',
    'Rather than a parametric prior, **non-local means** denoises each pixel by averaging other pixels whose surrounding *patch* looks similar — a nonparametric image model. It exploits the self-similarity (repeated structure) of natural images.'))

cells.append(code(
    "# Figure 27.23 — clean, noisy, and non-local-means denoised (the book's houses,",
    '# shown FULL-frame as in the book, not centre-cropped).',
    'from skimage.restoration import denoise_nl_means',
    'g = resize(BURANO, (232, 232), anti_aliasing=True)         # full colour scene',
    'sigma = 0.08',
    'noisy = np.clip(g + sigma * np.random.randn(*g.shape), 0, 1)',
    'nlm = denoise_nl_means(noisy, h=0.8 * sigma, sigma=sigma, patch_size=5,',
    '                       patch_distance=6, fast_mode=True, channel_axis=-1)',
    'fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))',
    "for a, (im, t) in zip(ax, [(g, 'clean'), (noisy, f'noisy (sigma={sigma})'), (nlm, 'non-local means')]):",
    '    a.imshow(np.clip(im, 0, 1)); a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])',
    'plt.tight_layout(); plt.show()',
    "print('RMSE  noisy: %.4f   NLM: %.4f' % (np.sqrt(((noisy - g)**2).mean()), np.sqrt(((nlm - g)**2).mean())))"))

cells.append(md(
    '## More representations from Chapter 27',
    '',
    "The figures above cover the chapter's core arc. Below are several more of the book's representations that are worth reproducing: the *space of visual worlds* (27.2), the **dead-leaves** generative model (27.9), the role of **Fourier phase** (27.12), a **Gaussian texture** model (27.13), and how the prior shapes **reconstruction** (27.19) and **wavelet coefficient** estimation (27.20)."))

cells.append(md(
    '### 27.2 — Eight visual worlds',
    '',
    'Different sources of images — noise, an oriented Gabor, a Mondrian, a star field, clouds, lines, rendered CGI, a street scene — occupy very different regions of image space, each with its own statistics. A single model cannot fit them all.'))

cells.append(code(
    "# the book's own eight worlds, cropped in colour from Fig 27.2 (worlds.png)",
    "worlds = [('world_noise.png', 'colour noise'), ('world_gabor.png', 'Gabor'),",
    "          ('world_mondrian.png', 'Mondrian'), ('world_stars.png', 'stars'),",
    "          ('world_clouds.png', 'clouds'), ('world_lines.png', 'line drawing'),",
    "          ('world_cgi.png', 'CGI render'), ('world_street.png', 'street')]",
    'fig, ax = plt.subplots(2, 4, figsize=(12, 6.2))',
    'for a, (fn, lb) in zip(ax.ravel(), worlds):',
    '    a.imshow(np.clip(book_img(fn), 0, 1)); a.set_title(lb, fontsize=10); a.set_xticks([]); a.set_yticks([])',
    'fig.suptitle("27.2 — the book\'s eight visual worlds, each with its own image statistics", y=1.02)',
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '### 27.9 — The dead-leaves model',
    '',
    'A simple *generative* model of natural images: repeatedly drop opaque coloured shapes (disks and squares) of random size, each occluding what is beneath. Like the book (Fig 27.9), we show a disk version and a square version. Occlusion alone reproduces hallmarks of natural-image statistics — scale-invariant structure and a **heavy-tailed** derivative histogram.'))

cells.append(code(
    'def dead_leaves(size=220, n=1600, rmin=3, rmax=40, seed=0, squares=False):',
    '    g = np.random.default_rng(seed); img = np.full((size, size, 3), 0.5)',
    '    yy, xx = np.mgrid[0:size, 0:size]',
    '    for _ in range(n):',
    '        cx, cy = g.integers(0, size, 2); r = g.integers(rmin, rmax); col = g.random(3)',
    '        m = (np.abs(xx-cx) < r) & (np.abs(yy-cy) < r) if squares else ((xx-cx)**2 + (yy-cy)**2 < r*r)',
    '        img[m] = col',
    '    return img',
    'dl_c = dead_leaves(seed=1); dl_s = dead_leaves(seed=2, squares=True)',
    'fig, ax = plt.subplots(1, 3, figsize=(12, 4))',
    "ax[0].imshow(np.clip(dl_c, 0, 1)); ax[0].set_title('dead leaves — disks')",
    "ax[1].imshow(np.clip(dl_s, 0, 1)); ax[1].set_title('dead leaves — squares')",
    "ax[2].hist(np.diff(dl_c.mean(2), axis=1).ravel(), bins=201, range=(-1, 1), density=True, color='crimson', log=True)",
    "ax[2].set_title('[1,-1] derivative histogram\\n(heavy-tailed)'); ax[2].set_xlabel('derivative')",
    'for a in ax[:2]: a.set_xticks([]); a.set_yticks([])',
    "fig.suptitle('27.9 — the dead-leaves model (colour): occlusion reproduces natural-image statistics', y=1.02)",
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '### 27.12 — Matched Fourier magnitude: phase carries the structure',
    '',
    "The 1/f power spectrum (27.10) fixes the Fourier **magnitude**. But the magnitude alone does not make an image look like anything: keep each image's magnitude and replace its **phase** with random phase, and the recognizable content dissolves into 1/f texture. It is the **phase** that encodes edges and objects."))

cells.append(code(
    'def random_phase(img):',
    '    ph = np.angle(np.fft.fft2(np.random.default_rng(0).standard_normal(img.shape[:2])))  # one phase field, shared across colour',
    '    out = np.zeros_like(img)',
    '    for c in range(img.shape[2]):',
    '        mag = np.abs(np.fft.fft2(img[..., c]))',
    '        out[..., c] = np.fft.ifft2(mag * np.exp(1j*ph)).real',
    '    return np.clip((out - out.min()) / (np.ptp(out) + 1e-9), 0, 1)',
    "pairs = [('stars', STARS), ('clouds', CLOUDS), ('plums', PLUMS), ('cube', CUBE)]",
    'fig, ax = plt.subplots(2, 4, figsize=(12, 6))',
    'for k, (nm, im) in enumerate(pairs):',
    '    im2 = resize(im, (160, 160, 3), anti_aliasing=True)',
    '    ax[0, k].imshow(np.clip(im2, 0, 1)); ax[0, k].set_title(nm)',
    '    ax[1, k].imshow(random_phase(im2))',
    "    if k == 0: ax[0, k].set_ylabel('original', fontsize=9); ax[1, k].set_ylabel('random phase\\n(same |FFT|)', fontsize=9)",
    'for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])',
    "fig.suptitle('27.12 — same Fourier magnitude, random phase: structure lives in the phase', y=1.02)",
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '### 27.13 — A Gaussian texture model',
    '',
    'The fully second-order (Gaussian) model of a texture keeps only the mean and the **power spectrum**, and draws a sample with random phase. On hair it captures the dominant orientation and scale, but — having thrown away the phase — it cannot reproduce the individual **strands**; it looks like a phase-scrambled version.'))

cells.append(code(
    'hair = to_gray_square(HAIR, 200)',
    'F = np.fft.fft2(hair - hair.mean()); mag = np.abs(F)',
    'ph = np.angle(np.fft.fft2(np.random.default_rng(3).standard_normal(hair.shape)))',
    'sample = hair.mean() + np.fft.ifft2(mag * np.exp(1j*ph)).real',
    'fig, ax = plt.subplots(1, 2, figsize=(8.5, 4.4))',
    "ax[0].imshow(hair, cmap='gray'); ax[0].set_title('hair (real texture)')",
    "ax[1].imshow(np.clip((sample - sample.min())/(np.ptp(sample)+1e-9), 0, 1), cmap='gray')",
    "ax[1].set_title('Gaussian model sample\\n(same power spectrum, random phase)')",
    'for a in ax: a.set_xticks([]); a.set_yticks([])',
    "fig.suptitle('27.13 — a second-order model captures the spectrum, not the strands', y=1.02)",
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '### 27.19 — The prior decides how a reconstruction looks',
    '',
    'Reconstructing a 1-D signal from noisy samples: a **Gaussian (L2)** smoothness prior penalizes all differences quadratically and rounds off the edges; a **heavy-tailed (total-variation)** prior tolerates a few large jumps, so it keeps edges sharp while flattening the noise — the 1-D analogue of edge-preserving image denoising.'))

cells.append(code(
    'from numpy.fft import rfft, irfft',
    'from skimage.restoration import denoise_tv_chambolle',
    'g = np.random.default_rng(1); n = 200',
    'sig = np.zeros(n); sig[60:120] = 1.0; sig[120:160] = 0.4',
    'obs = sig + 0.12 * g.standard_normal(n)',
    'w = 2*np.pi*np.fft.rfftfreq(n); l2 = irfft(rfft(obs)/(1 + 40.0*w**2), n)   # Gaussian (L2) prior',
    'tv = denoise_tv_chambolle(obs, weight=0.4)                                 # heavy-tailed (TV) prior',
    'fig, ax = plt.subplots(figsize=(8, 4))',
    "ax.plot(obs, color='0.7', lw=1, label='noisy observation'); ax.plot(sig, 'k:', lw=1.2, label='true signal')",
    "ax.plot(l2, color='#1a56db', lw=2, label='Gaussian (L2) prior — rounds edges')",
    "ax.plot(tv, color='crimson', lw=2, label='heavy-tailed (TV) prior — keeps edges')",
    "ax.set_title('27.19 — reconstructing a 1-D signal: the prior decides edge sharpness')",
    'ax.legend(fontsize=8); ax.set_xticks([]); ax.set_yticks([])',
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '### 27.20 — Prior × likelihood = posterior for a wavelet coefficient',
    '',
    'Estimating a single band-pass (wavelet) coefficient from a noisy measurement: the **heavy-tailed prior** (peaked at 0) multiplied by the **Gaussian likelihood** (centred on the noisy observation) gives a **posterior** whose peak is pulled back toward zero. Small coefficients are shrunk to (near) zero and large ones are kept — exactly the **coring** nonlinearity of 27.21.'))

cells.append(code(
    'x = np.linspace(-6, 6, 600); y_obs = 2.2; sigma = 1.0',
    'prior = gen_laplacian(x, r=0.6, s=0.7); prior /= prior.max()',
    'like = np.exp(-(x - y_obs)**2 / (2*sigma**2)); like /= like.max()',
    'post = prior * like; post /= post.max()',
    'fig, ax = plt.subplots(figsize=(7.5, 4.2))',
    "ax.plot(x, prior, color='#1a56db', label='prior (heavy-tailed)')",
    "ax.plot(x, like, color='0.5', label=f'likelihood (obs = {y_obs})')",
    "ax.plot(x, post, color='crimson', lw=2, label='posterior')",
    "ax.axvline(x[post.argmax()], ls=':', color='crimson'); ax.axvline(y_obs, ls=':', color='0.5')",
    "ax.set_xlabel('wavelet coefficient'); ax.set_yticks([]); ax.legend(fontsize=8)",
    "ax.set_title('27.20 — the MAP estimate is shrunk toward 0 (coring)')",
    'plt.tight_layout(); plt.show()'))

cells.append(md(
    '## 27.25 — Concluding remarks',
    '',
    '| Statistic / model | What it captures | Use |',
    '|---|---|---|',
    '| $1/w^{\\alpha}$ power law | second-order (Fourier) structure | Gaussian prior, Wiener denoising |',
    '| pixel-correlation decay | short-range dependence | why iid pixels fail |',
    '| generalized Laplacian of derivatives | heavy tails / sparsity | wavelet coring |',
    '| non-local self-similarity | repeated structure | non-local means |',
    '',
    'Natural images occupy a tiny, structured sliver of image space. Measuring that structure — a power law here, heavy-tailed derivatives there — gives priors that turn ill-posed problems (denoising, inpainting, super-resolution) into tractable Bayesian estimates, and sets the stage for the learned generative models of the next chapters.'))

nb = {"cells": cells, "metadata": {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(sys.argv[1]); out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(cells)} cells)")
