#!/usr/bin/env python3
"""Assemble the Chapter 29 (Probabilistic Graphical Models) companion notebook.

Implements the chapter's core machinery: the Markov-random-field smoothness prior
(sampled with Gibbs sampling), exact sum-product belief propagation on a tree
(reproducing the book's numerical example, 29.15, and checked against brute force),
loopy belief propagation on a 2-D grid for binary denoising and for segmenting the
book's leaf image (29.4), and 1-D belief propagation for stereo along a scanline of
the book's canoe pair (reproducing 29.14). Section numbering mirrors
visionbook.mit.edu Ch 29.
"""
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
"# Chapter 29 — Probabilistic Graphical Models",
"",
"*Companion notebook for* **Foundations of Computer Vision** *(Torralba, Isola,"
" Freeman), Ch. 29 — [visionbook.mit.edu](https://visionbook.mit.edu/graphical_models.html).*",
"",
"A **graphical model** factorizes a joint distribution over many variables into a"
" product of local terms tied to the edges of a graph. For vision the variables are"
" per-pixel labels (a depth, a segment, a denoised value) and the graph is the image"
" grid: neighbouring pixels are coupled by a **smoothness** prior, and each pixel gets"
" **local evidence** from the data. This notebook builds the chapter's tools:",
"",
"1. **The MRF smoothness prior** — sampled with **Gibbs sampling** (Fig 29.3).",
"2. **Exact belief propagation on a tree** — reproducing the book's numerical example"
" (Fig 29.15) and checking it against brute-force marginalization.",
"3. **Loopy belief propagation** on the image grid — binary **denoising**.",
"4. **Segmentation** of the book's leaf image by loopy BP (Fig 29.4).",
"5. **Stereo** along a scanline of the book's canoe pair by BP (Fig 29.14).",
"",
"The photographs are the book's own images (canoe stereo pair, leaf), cropped from the"
" figures on visionbook.mit.edu.",
))

cells.append(code(
"import os",
"import numpy as np",
"import matplotlib.pyplot as plt",
"from PIL import Image",
"from skimage.transform import resize",
"",
"rng = np.random.default_rng(0)",
"plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130,",
"                     'image.cmap': 'gray', 'image.interpolation': 'nearest', 'axes.grid': False})",
"",
"_A = next(p for p in ['assets',",
"          'notebooks/CV/mit-foundations/chapter-29-probabilistic-graphical-models/assets']",
"          if os.path.isdir(p))",
"def book_img(name, gray=False):",
"    im = Image.open(f'{_A}/{name}')",
"    im = im.convert('L' if gray else 'RGB')",
"    return np.asarray(im, np.float32) / 255.0",
"",
"CANOE_L = book_img('canoe_left.png')    # left  view, scanline crop (29.12/29.14)",
"CANOE_R = book_img('canoe_right.png')   # right view, scanline crop",
"LEAF = book_img('leaf.png')             # image to be segmented (29.4)",
"print('canoe L', CANOE_L.shape, ' canoe R', CANOE_R.shape, ' leaf', LEAF.shape)",
))

# =================================================================== 29.3 MRF prior
cells.append(md(
"## 29.3 — The MRF prior: neighbouring pixels agree",
"",
"An **undirected graphical model** (Markov random field) puts a potential on every"
" edge. The workhorse prior for images is the **Ising / Potts** model, which rewards"
" neighbouring pixels for taking the *same* label:",
"",
"$$p(\\mathbf{x}) \\propto \\exp\\Big(\\beta \\textstyle\\sum_{(i,j)\\in\\mathcal{E}} \\mathbb{1}[x_i = x_j]\\Big).$$",
"",
"There is no data here — this is purely the *prior*. To see what it believes, we draw"
" samples with **Gibbs sampling**: repeatedly replace each pixel by a draw from its"
" conditional given its four neighbours. Because the grid is bipartite, all"
" *black-square* pixels are conditionally independent given the *white-square* ones, so"
" we can update a whole colour at once (checkerboard sweeps). As the coupling"
" $\\beta$ grows, samples go from white noise to ever-larger smooth regions — exactly"
" the 'images are piecewise smooth' assumption the rest of the chapter exploits.",
))

cells.append(code(
"def ising_gibbs(size, beta, sweeps=40, seed=0):",
"    g = np.random.default_rng(seed)",
"    x = g.integers(0, 2, size=(size, size)) * 2 - 1          # spins in {-1,+1}",
"    i, j = np.indices((size, size))",
"    for _ in range(sweeps):",
"        for colour in (0, 1):                                # checkerboard: update one colour",
"            nb = np.zeros_like(x, float)                     # sum of 4 neighbours",
"            nb[1:] += x[:-1]; nb[:-1] += x[1:]",
"            nb[:, 1:] += x[:, :-1]; nb[:, :-1] += x[:, 1:]",
"            p_up = 1.0 / (1.0 + np.exp(-2.0 * beta * nb))     # P(x_i = +1 | neighbours)",
"            flip = (g.random((size, size)) < p_up) * 2 - 1",
"            x = np.where((i + j) % 2 == colour, flip, x)",
"    return x",
"",
"betas = [0.0, 0.44, 0.7]",
"titles = [r'$\\beta=0$ (independent: white noise)',",
"          r'$\\beta=0.44$ (critical: structure at all scales)',",
"          r'$\\beta=0.7$ (strong: large smooth regions)']",
"",
"# Leftmost panel: the MRF itself drawn on the pixels — every pixel is a NODE (its state",
"# is black/white), wired by an EDGE to each of its 4 neighbours. The other panels are",
"# samples of that model at increasing coupling.",
"patch = ising_gibbs(8, 0.7, sweeps=60, seed=1); n = patch.shape[0]",
"fig = plt.figure(figsize=(13.5, 4.3)); gs = fig.add_gridspec(1, 4, width_ratios=[1.1, 1, 1, 1])",
"axg = fig.add_subplot(gs[0, 0])",
"for i in range(n):",
"    for j in range(n):",
"        if j + 1 < n: axg.plot([j, j + 1], [i, i], color='#1a56db', lw=1.3, zorder=1)",
"        if i + 1 < n: axg.plot([j, j], [i, i + 1], color='#1a56db', lw=1.3, zorder=1)",
"for i in range(n):",
"    for j in range(n):",
"        axg.scatter(j, i, s=150, c=('white' if patch[i, j] > 0 else 'black'),",
"                    edgecolor='#1a56db', linewidth=1.5, zorder=2)",
"axg.set_xlim(-0.6, n - 0.4); axg.set_ylim(n - 0.4, -0.6); axg.set_aspect('equal')",
"axg.set_title('the MRF: each pixel is a node,\\nwired to its 4 neighbours', fontsize=9)",
"axg.set_xticks([]); axg.set_yticks([])",
"for k, (b, t) in enumerate(zip(betas, titles)):",
"    a = fig.add_subplot(gs[0, k + 1])",
"    a.imshow(ising_gibbs(120, b, sweeps=60), cmap='gray')",
"    a.set_title(t, fontsize=9); a.set_xticks([]); a.set_yticks([])",
"fig.suptitle('The MRF (Ising) prior: pixels are graph nodes (left); samples as the coupling $\\\\beta$ grows (right)', y=1.03)",
"plt.tight_layout(); plt.show()",
))

# =================================================================== 29.15 exact BP tree
cells.append(md(
"## 29.15 — Exact belief propagation on a tree",
"",
"On a graph **without loops**, belief propagation computes the exact marginals. We"
" reproduce the book's worked example (Fig 29.15): a chain $x_1 - x_2 - x_3$ with an"
" observed node $y_2 = 0$ hanging off $x_2$. Every variable is binary and the"
" potentials are exactly those printed in the figure:",
"",
"$$\\psi_{12}=\\begin{pmatrix}1.0&0.9\\\\0.9&1.0\\end{pmatrix},\\quad"
"\\psi_{23}=\\begin{pmatrix}0.1&1.0\\\\1.0&0.1\\end{pmatrix},\\quad"
"\\phi_{2}=\\begin{pmatrix}1.0&0.1\\\\0.1&1.0\\end{pmatrix}.$$",
"",
"**Sum-product** rule: a node collects the incoming messages, and the message it sends"
" across an edge is $m_{a\\to b}(x_b)=\\sum_{x_a}\\psi_{ab}(x_a,x_b)\\,\\phi_a(x_a)\\prod_{c\\ne b}m_{c\\to a}(x_a)$."
" A node's marginal is the (normalized) product of all its incoming messages times its"
" own evidence. Because this graph is a tree, BP must agree with brute-force"
" marginalization over all $2^3$ joint states — we check that it does.",
))

cells.append(code(
"psi12 = np.array([[1.0, 0.9], [0.9, 1.0]])",
"psi23 = np.array([[0.1, 1.0], [1.0, 0.1]])",
"phi2  = np.array([[1.0, 0.1], [0.1, 1.0]])",
"ev2 = phi2[:, 0]                     # evidence on x2 from the observation y2 = 0  -> [1.0, 0.1]",
"norm = lambda v: v / v.sum()",
"",
"# --- Sum-product BP on the tree (leaves -> x2 -> leaves) ---",
"m_1to2 = psi12.T @ np.ones(2)        # leaf x1 sends: sum_{x1} psi12(x1,x2)",
"m_3to2 = psi23   @ np.ones(2)        # leaf x3 sends: sum_{x3} psi23(x2,x3)",
"bel2 = norm(ev2 * m_1to2 * m_3to2)",
"m_2to1 = psi12 @ (ev2 * m_3to2)      # x2 -> x1 (fold in evidence and the x3 branch)",
"m_2to3 = psi23.T @ (ev2 * m_1to2)    # x2 -> x3",
"bel1, bel3 = norm(m_2to1), norm(m_2to3)",
"bp = np.array([bel1, bel2, bel3])",
"",
"# --- Brute force: full joint over (x1,x2,x3) ---",
"brute = np.zeros((3, 2))",
"Z = 0.0",
"for a in (0, 1):",
"    for b in (0, 1):",
"        for c in (0, 1):",
"            w = psi12[a, b] * psi23[b, c] * ev2[b]",
"            Z += w",
"            brute[0, a] += w; brute[1, b] += w; brute[2, c] += w",
"brute /= Z",
"",
"print('node   P(x=0)  P(x=1)      BP == brute force?')",
"for k in range(3):",
"    print(f'  x{k+1}   {bp[k,0]:.4f}  {bp[k,1]:.4f}    '",
"          f'{np.allclose(bp[k], brute[k])}')",
"assert np.allclose(bp, brute)",
"",
"fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))",
"for k, a in enumerate(ax):",
"    x = np.arange(2)",
"    a.bar(x - 0.2, brute[k], 0.4, label='brute force', color='0.6')",
"    a.bar(x + 0.2, bp[k], 0.4, label='belief propagation', color='#1a56db')",
"    a.set_title(f'marginal $p(x_{k+1})$'); a.set_xticks(x); a.set_xticklabels(['0', '1'])",
"    a.set_ylim(0, 1); a.set_xlabel('state')",
"ax[0].set_ylabel('probability'); ax[0].legend(fontsize=8)",
"fig.suptitle('Tree BP is exact: the bars coincide', y=1.02)",
"plt.tight_layout(); plt.show()",
))

# =================================================================== loopy BP core + denoising
cells.append(md(
"## Loopy belief propagation on the image grid",
"",
"The image grid **has loops**, so BP is no longer exact — but *loopy* BP (just keep"
" passing messages) works remarkably well in practice. For binary labels every message"
" is a two-vector, which we track by its **log-odds** $m = \\log\\frac{m(1)}{m(0)}$."
" Passing a log-odds belief $b$ through an Ising edge with coupling $J$ gives the closed"
" form",
"",
"$$m_{\\text{out}} = \\log\\frac{e^{J}e^{b}+e^{-J}}{e^{-J}e^{b}+e^{J}},$$",
"",
"which we apply to all edges of one orientation at once with array shifts. We first"
" test it on **denoising**: a clean binary image is corrupted by flipping 20% of the"
" pixels; the per-pixel likelihood gives each node a data log-odds"
" $\\pm\\log\\frac{1-q}{q}$, and the Ising prior glues neighbours together.",
))

cells.append(code(
"def ising_edge(b_in, J):",
"    '''Outgoing log-odds along one Ising edge given incoming node log-odds b_in.'''",
"    return np.logaddexp(J + b_in, -J) - np.logaddexp(-J + b_in, J)",
"",
"def loopy_bp_binary(data_logodds, J, iters=30):",
"    '''Loopy BP on a 4-connected grid of binary nodes; returns posterior log-odds.'''",
"    mU = np.zeros_like(data_logodds); mD = np.zeros_like(data_logodds)",
"    mL = np.zeros_like(data_logodds); mR = np.zeros_like(data_logodds)",
"    for _ in range(iters):",
"        b = data_logodds + mU + mD + mL + mR              # current node beliefs",
"        oR = ising_edge(b - mR, J); nL = np.zeros_like(mL); nL[:, 1:] = oR[:, :-1]",
"        oL = ising_edge(b - mL, J); nR = np.zeros_like(mR); nR[:, :-1] = oL[:, 1:]",
"        oD = ising_edge(b - mD, J); nU = np.zeros_like(mU); nU[1:, :] = oD[:-1, :]",
"        oU = ising_edge(b - mU, J); nD = np.zeros_like(mD); nD[:-1, :] = oU[1:, :]",
"        mL, mR, mU, mD = nL, nR, nU, nD",
"    return data_logodds + mU + mD + mL + mR",
"",
"# clean binary image: disk, bar, ring",
"yy, xx = np.mgrid[0:100, 0:100]",
"clean = np.zeros((100, 100), int)",
"clean[(yy - 32) ** 2 + (xx - 30) ** 2 < 18 ** 2] = 1",
"clean[62:86, 16:72] = 1",
"ring = (yy - 68) ** 2 + (xx - 74) ** 2",
"clean[(ring < 20 ** 2) & (ring > 12 ** 2)] = 1",
"",
"q = 0.20                                              # pixel flip probability",
"flips = np.random.default_rng(1).random(clean.shape) < q",
"noisy = np.where(flips, 1 - clean, clean)",
"L = np.log((1 - q) / q)",
"data_lo = (2 * noisy - 1) * L                        # data log-odds favouring label 1",
"post = loopy_bp_binary(data_lo, J=1.0, iters=40)",
"denoised = (post > 0).astype(int)",
"",
"err_noisy = (noisy != clean).mean(); err_bp = (denoised != clean).mean()",
"fig, ax = plt.subplots(1, 3, figsize=(11, 3.9))",
"for a, im, t in zip(ax, [clean, noisy, denoised],",
"                    ['clean', f'noisy (20% flipped): {err_noisy:.0%} error',",
"                     f'loopy-BP MAP: {err_bp:.1%} error']):",
"    a.imshow(im, cmap='gray'); a.set_title(t, fontsize=10); a.set_xticks([]); a.set_yticks([])",
"plt.tight_layout(); plt.show()",
"print(f'pixel error  {err_noisy:.1%}  ->  {err_bp:.1%}   after loopy BP')",
))

# =================================================================== 29.4 leaf segmentation
cells.append(md(
"## 29.4 — Segmentation as a two-label MRF",
"",
"The same machinery segments the book's leaf image (Fig 29.4): label each pixel"
" *foliage* or *background*. The **local evidence** is colour — a simple 'greenness'"
" score $g - \\tfrac12(r+b)$ — turned into a data log-odds; the **prior** is again the"
" Ising smoothness term. Thresholding greenness pixel-by-pixel is speckled and noisy;"
" loopy BP over the grid pulls those local votes into spatially coherent regions.",
))

cells.append(code(
"leaf = resize(LEAF, (150, round(150 * LEAF.shape[1] / LEAF.shape[0])), anti_aliasing=True)",
"r, g, b = leaf[..., 0], leaf[..., 1], leaf[..., 2]",
"greenness = g - 0.5 * (r + b)",
"data_lo = 2.0 * (greenness - np.median(greenness)) / (greenness.std() + 1e-9)",
"raw = data_lo > 0                                    # per-pixel threshold (no prior)",
"seg = loopy_bp_binary(data_lo, J=1.0, iters=30) > 0  # with the smoothness prior",
"",
"boundary = lambda m: int(np.abs(np.diff(m.astype(int), axis=0)).sum()",
"                         + np.abs(np.diff(m.astype(int), axis=1)).sum())",
"fig, ax = plt.subplots(1, 3, figsize=(12, 3.4))",
"ax[0].imshow(leaf); ax[0].set_title('book leaf image (29.4)', fontsize=10)",
"ax[1].imshow(raw, cmap='Greens'); ax[1].set_title(f'per-pixel greenness threshold\\n(boundary {boundary(raw)})', fontsize=10)",
"ax[2].imshow(seg, cmap='Greens'); ax[2].set_title(f'MRF + loopy BP segmentation\\n(boundary {boundary(seg)})', fontsize=10)",
"for a in ax: a.set_xticks([]); a.set_yticks([])",
"plt.tight_layout(); plt.show()",
))

# =================================================================== 29.14 stereo scanline BP
cells.append(md(
"## 29.14 — Stereo along a scanline",
"",
"Stereo makes the graphical-model picture concrete (Fig 29.14). Take one **scanline**"
" from the rectified left and right views of the book's canoe pair. Each pixel"
" position is a node whose label is a **disparity** (depth); the **local evidence** is"
" how well the left patch matches the right patch shifted by that disparity, and the"
" chain of nodes is tied by a smoothness prior. We run **sum-product BP** along the"
" 1-D chain: a forward (left-to-right) sweep and a backward (right-to-left) sweep, then"
" multiply them with the evidence to get the marginal posterior at every position.",
"",
"**How to read the panels below** (they mirror the book):",
"",
"- **(a), (b)** the *same* row of pixels seen by the right and left cameras.",
"- **(c)–(f)** are *position × depth* images: the **horizontal axis is position** along"
" the scanline (lined up with a,b) and the **vertical axis is candidate depth** (small"
" disparity = far, large = near). **Brighter = more probable.** So each vertical slice"
" is a probability-over-depth for one pixel.",
"- **(c) local evidence** — how well the left patch matches the right patch at each"
" depth. Textured grass gives a sharp bright spot (confident); the smooth hull is"
" ambiguous (diffuse).",
"- **(d), (e) messages** — belief passed left→right and right→left along the chain,"
" carrying confident estimates into the ambiguous regions.",
"- **(f) posterior** = evidence × both messages. The **bright ridge is the recovered"
" depth profile**; we overlay it as a line.",
"",
"It is a deliberately simple matcher (windowed normalized correlation + a"
" truncated-linear smoothness), so it captures the *behaviour* of Fig 29.14 rather than"
" the book's exact pixels.",
))

cells.append(code(
"# --- extract one matched scanline band from each view (rows near the book's marked line) ---",
"rowL, rowR, band, win = 225, 234, 8, 7",
"SL = CANOE_L[rowL - band:rowL + band + 1]        # (2*band+1, W, 3)",
"SR = CANOE_R[rowR - band:rowR + band + 1]",
"W = CANOE_L.shape[1]",
"G, DELTA = 77, 14                                # global crop offset, then search +/- DELTA",
"disps = np.arange(G - DELTA, G + DELTA + 1); D = len(disps)",
"",
"def zwin(S, x):                                  # zero-mean unit-norm patch (for NCC)",
"    v = S[:, x - win:x + win + 1].ravel(); v = v - v.mean()",
"    return v / (np.sqrt((v * v).sum()) + 1e-9)",
"",
"# local evidence phi[x, d] from 1 - normalized cross-correlation",
"cost = np.ones((W, D))",
"for x in range(win + G, W - win):",
"    a = zwin(SL, x)",
"    for j, d in enumerate(disps):",
"        xr = x - d",
"        if win <= xr < W - win:",
"            cost[x, j] = 1.0 - a @ zwin(SR, xr)",
"phi = np.exp(-cost / 0.25); phi /= phi.sum(1, keepdims=True)",
"",
"# truncated-linear smoothness: cheap to disagree a little, capped so depth edges survive",
"dd = np.abs(disps[:, None] - disps[None, :])",
"psi = np.exp(-np.minimum(dd, 4) / 1.2)",
"",
"def nrm(v): s = v.sum(); return v / s if s > 0 else np.full_like(v, 1.0 / len(v))",
"mfwd = np.ones((W, D)) / D                        # left-to-right messages",
"for x in range(1, W): mfwd[x] = nrm(psi.T @ (phi[x - 1] * mfwd[x - 1]))",
"mbwd = np.ones((W, D)) / D                        # right-to-left messages",
"for x in range(W - 2, -1, -1): mbwd[x] = nrm(psi @ (phi[x + 1] * mbwd[x + 1]))",
"posterior = phi * mfwd * mbwd; posterior /= posterior.sum(1, keepdims=True)",
"",
"vis = slice(G + win, W - win)                     # valid (matchable) columns only",
"post_idx = posterior[vis].argmax(1)               # recovered depth profile (MAP)",
"panels = [(phi, '(c) local evidence — patch-match quality at each depth'),",
"          (mfwd, '(d) left-to-right messages'),",
"          (mbwd, '(e) right-to-left messages'),",
"          (posterior, '(f) posterior — bright ridge = recovered depth')]",
"fig, ax = plt.subplots(6, 1, figsize=(10, 10))",
"ax[0].imshow(SR[:, vis]); ax[0].set_title('(a) right camera scanline', loc='left', fontsize=10)",
"ax[1].imshow(SL[:, vis]); ax[1].set_title('(b) left camera scanline', loc='left', fontsize=10)",
"for a, (m, t) in zip(ax[2:], panels):",
"    a.imshow(np.sqrt(m[vis].T), aspect='auto', origin='lower', cmap='magma')   # sqrt stretch (brightness = probability)",
"    a.set_title(t, loc='left', fontsize=10)",
"    a.set_yticks([0, D - 1]); a.set_yticklabels(['far', 'near'], fontsize=8); a.set_ylabel('depth', fontsize=9)",
"conf = posterior[vis].max(1)                       # how peaked the posterior is at each pixel",
"depth_line = post_idx.astype(float); depth_line[conf < 0.12] = np.nan   # draw only where BP is confident",
"ax[5].plot(np.arange(len(post_idx)), depth_line, color='#00e5ff', lw=1.8)   # recovered depth (confident spans)",
"ax[5].set_xlim(0, len(post_idx) - 1); ax[5].set_ylim(-0.5, D - 0.5)",
"for a in ax[:2]: a.set_yticks([])",
"for a in ax: a.set_xticks([])",
"ax[5].set_xlabel('position along the scanline  ->', fontsize=9)",
"plt.tight_layout(); plt.show()",
"",
"jit = lambda m: np.abs(np.diff(disps[m[vis].argmax(1)])).mean()",
"print(f'disparity jitter along the scanline:  winner-take-all {jit(phi):.2f}  ->  BP {jit(posterior):.2f}')",
))

cells.append(md(
"### Takeaways",
"",
"- A graphical model splits an image problem into **local evidence** (data terms) and a"
" **smoothness prior** (edge potentials); inference combines them.",
"- On a **tree**, sum-product BP is *exact* — it matched brute force to machine"
" precision (29.15). On the **looped** image grid, *loopy* BP is approximate but"
" effective (denoising, segmentation, stereo).",
"- BP is message passing: each node's belief is the product of neighbours' messages and"
" its own evidence. Smoothness makes confident, textured regions **propagate** into"
" ambiguous, textureless ones — visible directly in the stereo messages (29.14).",
"- These MRF/BP energies are the classical ancestors of today's dense-prediction"
" networks; the priors are now learned, but the evidence-plus-smoothness structure"
" remains.",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(sys.argv[1]); out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(cells)} cells, {sum(c['cell_type']=='code' for c in cells)} code cells)")
