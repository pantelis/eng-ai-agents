#!/usr/bin/env python3
"""Assemble the Chapter 30 (Representation Learning) companion notebook.

Implements the chapter's core demos on the book's toy colored-shapes dataset:
an autoencoder embedding (reconstructions + nearest neighbours + a per-layer probe),
K-means clustering, and contrastive learning (InfoNCE) where the choice of data
augmentation controls which factor — colour or shape — the representation encodes.
Section numbering mirrors visionbook.mit.edu Ch 30. Device-agnostic (uses CUDA when
available; the maintainer validates under torch.dev.gpu).
"""
import json, sys
from pathlib import Path

def md(*lines): return {"cell_type": "markdown", "metadata": {}, "source": _src(lines)}
def code(*lines): return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": _src(lines)}
def _src(lines):
    text = "\n".join(lines); parts = text.split("\n")
    return [p + "\n" for p in parts[:-1]] + [parts[-1]]

cells = []

cells.append(md(
"# Chapter 30 — Representation Learning",
"",
"*Companion notebook for* **Foundations of Computer Vision** *(Torralba, Isola,"
" Freeman), Ch. 30 — [visionbook.mit.edu](https://visionbook.mit.edu/representation_learning.html).*",
"",
"**Representation learning** is the forward half of vision: mapping raw pixels to a"
" compact **embedding** that exposes the underlying factors of a scene. This notebook"
" builds the chapter's demos on the book's toy **colored-shapes** dataset:",
"",
"1. **The dataset** (Fig 30.4) — 3 shapes x 8 colours, random size/position/rotation.",
"2. **Autoencoders** (Fig 30.5) — an encoder-bottleneck-decoder; we inspect its"
" embedding with reconstructions, nearest neighbours, and a per-layer probe.",
"3. **K-means** (Fig 30.11) — clustering as a discrete representation.",
"4. **Contrastive learning** (Fig 30.14) — InfoNCE, where the choice of **data"
" augmentation** decides whether the embedding encodes **colour** or **shape**.",
"",
"The colored-shapes data is procedural (the book's is too), so we generate it exactly"
" rather than loading images.",
))

cells.append(code(
"import numpy as np, torch, torch.nn as nn, torch.nn.functional as F",
"import matplotlib.pyplot as plt",
"from PIL import Image, ImageDraw",
"",
"device = 'cuda' if torch.cuda.is_available() else 'cpu'",
"torch.manual_seed(0); np.random.seed(0)",
"torch.set_num_threads(max(1, torch.get_num_threads()))",
"plt.rcParams.update({'figure.dpi': 130, 'savefig.dpi': 130, 'image.interpolation': 'nearest', 'axes.grid': False})",
"print('device:', device)",
"",
"# ---- the book's toy colored-shapes dataset (Fig 30.4) ----",
"COLORS = {'red':(220,50,50),'orange':(240,150,30),'yellow':(235,220,40),'green':(60,180,75),",
"          'cyan':(40,200,200),'blue':(60,90,220),'purple':(150,60,200),'pink':(240,110,180)}",
"CNAMES = list(COLORS); SHAPES = ['circle','square','triangle']",
"",
"def _one(rng, S=32):",
"    ci, si = rng.integers(8), rng.integers(3)",
"    base = np.array(COLORS[CNAMES[ci]], float)",
"    col = tuple(int(np.clip(c + rng.normal(0, 12), 0, 255)) for c in base)   # small colour jitter",
"    img = Image.new('RGB', (S, S), (20, 20, 20)); d = ImageDraw.Draw(img)",
"    r = rng.integers(int(S*0.22), int(S*0.36))                                # random size",
"    cx, cy = rng.integers(r, S-r), rng.integers(r, S-r)                        # random position",
"    ang = rng.uniform(0, 360)                                                  # random rotation",
"    if si == 0:",
"        d.ellipse([cx-r, cy-r, cx+r, cy+r], fill=col)",
"    else:",
"        n = 4 if si == 1 else 3",
"        a0 = np.deg2rad(ang) + (np.pi/4 if si == 1 else -np.pi/2)",
"        pts = [(cx+r*np.cos(a0+2*np.pi*k/n), cy+r*np.sin(a0+2*np.pi*k/n)) for k in range(n)]",
"        d.polygon(pts, fill=col)",
"    return np.asarray(img, np.float32)/255.0, si, ci",
"",
"def make_dataset(n, seed, S=32):",
"    rng = np.random.default_rng(seed)",
"    X = np.zeros((n, S, S, 3), np.float32); ys = np.zeros(n, int); yc = np.zeros(n, int)",
"    for i in range(n): X[i], ys[i], yc[i] = _one(rng, S)",
"    return X, ys, yc",
"",
"Xtr, ys_tr, yc_tr = make_dataset(3000, seed=0)",
"Xte, ys_te, yc_te = make_dataset(600, seed=99)",
"Xtr_t = torch.tensor(Xtr).permute(0, 3, 1, 2); Xte_t = torch.tensor(Xte).permute(0, 3, 1, 2)",
"print('train', Xtr.shape, ' test', Xte.shape)",
))

# ---- Fig 30.4 dataset ----
cells.append(md(
"## 30.4 — The colored-shapes dataset",
"",
"Each image has one of **three shapes** (circle, square, triangle) in one of **eight"
" colours**, with random size, position and rotation. Two independent factors —"
" **shape** and **colour** — which we will later ask a representation to disentangle.",
))
cells.append(code(
"fig, ax = plt.subplots(4, 12, figsize=(12, 4))",
"for i, a in enumerate(ax.ravel()):",
"    a.imshow(Xtr[i]); a.set_xticks([]); a.set_yticks([])",
"fig.suptitle('Toy colored-shapes dataset (Fig 30.4): 3 shapes x 8 colours', y=1.02)",
"plt.tight_layout(); plt.show()",
))

# ---- Autoencoder ----
cells.append(md(
"## 30.5 — Autoencoders",
"",
"An **autoencoder** maps each image through a low-dimensional **bottleneck** and back,"
" trained to reconstruct its input: $\\min_{f,g}\\;\\mathbb{E}_x\\,\\lVert g(f(x))-x\\rVert^2$."
" The bottleneck $f(x)$ is forced to keep only what is needed to redraw the image, so it"
" becomes a compact **representation**. We train a small convolutional autoencoder and"
" then look at what its embedding captures.",
))
cells.append(code(
"def cbr(i, o, s): return nn.Sequential(nn.Conv2d(i, o, 3, s, 1), nn.ReLU())",
"class AE(nn.Module):",
"    '''The book's autoencoder: six conv layers + a 128-d bottleneck. Downsampling stops",
"    at 4x4 so deep features keep the spatial detail that distinguishes the shapes.'''",
"    def __init__(self, M=128):",
"        super().__init__()",
"        self.c = nn.ModuleList([cbr(3, 24, 2), cbr(24, 48, 1), cbr(48, 64, 2),",
"                                cbr(64, 96, 1), cbr(96, 128, 2), cbr(128, 128, 1)])  # 32->16->16->8->8->4->4",
"        self.bott = nn.Linear(128 * 16, M)",
"        self.dec = nn.Sequential(nn.Linear(M, 128 * 16), nn.ReLU(), nn.Unflatten(1, (128, 4, 4)),",
"            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),",
"            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(),",
"            nn.ConvTranspose2d(32, 3, 4, 2, 1), nn.Sigmoid())",
"    def feats(self, x):                       # input + each of the 6 conv layers (for the probe)",
"        outs = [x.flatten(1)]; h = x",
"        for conv in self.c: h = conv(h); outs.append(h.flatten(1))",
"        return outs",
"    def encode(self, x):",
"        h = x",
"        for conv in self.c: h = conv(h)",
"        return self.bott(h.flatten(1))        # the 128-d embedding",
"    def forward(self, x):",
"        z = self.encode(x); return self.dec(z), z",
"",
"ae = AE().to(device); opt = torch.optim.Adam(ae.parameters(), 1e-3); lossf = nn.MSELoss()",
"N = len(Xtr_t)",
"# The Fig 30.5b crossing needs a real training budget (20k steps): a few minutes on GPU.",
"AE_STEPS = 20000",
"for step in range(AE_STEPS):",
"    xb = Xtr_t[torch.randint(0, N, (128,))].to(device)",
"    out, _ = ae(xb); loss = lossf(out, xb)",
"    opt.zero_grad(); loss.backward(); opt.step()",
"print(f'autoencoder ({AE_STEPS} steps) trained, reconstruction MSE {loss.item():.4f}')",
))

cells.append(md(
"### 30.5(a) — Reconstructions and nearest neighbours",
"",
"The reconstructions confirm the bottleneck preserves shape, colour and rough pose."
" More interesting: images whose embeddings are **nearest neighbours** tend to share"
" both shape and colour — the embedding has organized the data by its true factors.",
))
cells.append(code(
"ae.eval()",
"with torch.no_grad():",
"    recon, _ = ae(Xte_t[:8].to(device)); recon = recon.cpu()",
"    Ztr = ae.encode(Xtr_t.to(device)).cpu()",
"Ztr = F.normalize(Ztr, dim=1)",
"",
"fig = plt.figure(figsize=(12, 5.2))",
"gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.4])",
"# left: input vs reconstruction",
"gl = gs[0].subgridspec(2, 8)",
"for j in range(8):",
"    a = fig.add_subplot(gl[0, j]); a.imshow(Xte[j]); a.set_xticks([]); a.set_yticks([])",
"    if j == 0: a.set_ylabel('input', fontsize=8)",
"    b = fig.add_subplot(gl[1, j]); b.imshow(recon[j].permute(1, 2, 0).numpy().clip(0, 1)); b.set_xticks([]); b.set_yticks([])",
"    if j == 0: b.set_ylabel('recon', fontsize=8)",
"# right: nearest neighbours in embedding space (Fig 30.5a)",
"gr = gs[1].subgridspec(4, 6)",
"for r, q in enumerate([3, 17, 40, 55]):",
"    nbr = (Ztr @ Ztr[q]).argsort(descending=True)[:6].numpy()",
"    for cc, jj in enumerate(nbr):",
"        a = fig.add_subplot(gr[r, cc]); a.imshow(Xtr[jj]); a.set_xticks([]); a.set_yticks([])",
"        for sp in a.spines.values(): sp.set_color('crimson' if cc == 0 else '0.8'); sp.set_linewidth(2 if cc == 0 else 0.5)",
"fig.suptitle('Autoencoder embedding:   (a) input (top) vs reconstruction (bottom)      "
"(b) Fig 30.5a — query (red) and its nearest neighbours', y=1.02, fontsize=11)",
"plt.tight_layout(); plt.show()",
))

cells.append(md(
"### 30.5(b) — What does each layer encode?",
"",
"We probe every layer with a **1-nearest-neighbour classifier** (train features ->"
" predict a test image's shape / colour). The two factors are read out very differently"
" across depth: **shape accuracy rises toward ~99%** in the deeper, more semantic"
" features, while **colour accuracy falls** as the network abstracts away low-level"
" appearance — the **crossing** the book reports (Fig 30.5b). Reproducing it needs the"
" book's architecture (six conv layers, keeping spatial detail) and a real training"
" budget; a too-shallow model, or downsampling all the way to a single vector, does not"
" show it.",
))
cells.append(code(
"def nn_acc(tr, te, ytr, yte):",
"    tr = F.normalize(tr, dim=1); te = F.normalize(te, dim=1)",
"    return (ytr[(te @ tr.T).argmax(1).numpy()] == yte).mean()",
"# Stream the probe layer-by-layer (forward train+test together, keep only the current",
"# activations) so we never hold all seven layers' features at once — memory-lean.",
"ae.eval(); layers = ['pixels', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6']",
"acc_s, acc_c = [], []",
"with torch.no_grad():",
"    htr, hte = Xtr_t.to(device), Xte_t.to(device)",
"    for i, conv in enumerate([None] + list(ae.c)):",
"        if conv is not None: htr = conv(htr); hte = conv(hte)",
"        tr, te = htr.flatten(1).cpu(), hte.flatten(1).cpu()",
"        acc_s.append(nn_acc(tr, te, ys_tr, ys_te) * 100)",
"        acc_c.append(nn_acc(tr, te, yc_tr, yc_te) * 100)",
"        del tr, te",
"import plotly.graph_objects as go",
"figp = go.Figure()",
"figp.add_scatter(x=layers, y=acc_s, mode='lines+markers', name='shape',",
"                 line=dict(color='#1a56db', dash='dash'), marker=dict(symbol='circle', size=9))",
"figp.add_scatter(x=layers, y=acc_c, mode='lines+markers', name='colour',",
"                 line=dict(color='#d1495b'), marker=dict(symbol='square', size=9))",
"figp.add_hline(y=100/3, line=dict(color='gray', dash='dot'), annotation_text='shape chance (33%)')",
"figp.add_hline(y=100/8, line=dict(color='gray', dash='dot'), annotation_text='colour chance (12.5%)')",
"figp.update_layout(title='(Fig 30.5b) what each layer encodes: shape rises, colour falls (the crossing)',",
"                   xaxis_title='layer', yaxis_title='1-NN accuracy (%)', yaxis_range=[0, 100],",
"                   width=780, height=440, template='plotly_white')",
"figp.show()",
"print('shape:', [f'{a:.0f}' for a in acc_s]); print('colour:', [f'{a:.0f}' for a in acc_c])",
))

# ---- K-means ----
cells.append(md(
"## 30.11 — K-means clustering",
"",
"Clustering yields a **discrete** representation: each point is summarised by the index"
" of its nearest code vector. **K-means** minimises"
" $\\sum_i \\lVert z_{a_i} - x^{(i)}\\rVert^2$ by block-coordinate descent, alternating:",
"**(assign)** each point to its nearest mean, **(update)** each mean to the average of"
" its assigned points. We show the first iterations on a 2-D dataset with $k=5$"
" (Fig 30.11).",
))
cells.append(code(
"g = np.random.default_rng(2)",
"centres_true = g.uniform(-1, 1, (5, 2)) * 2.2",
"pts = np.concatenate([c + g.normal(0, 0.45, (120, 2)) for c in centres_true])",
"k = 5",
"z = pts[g.choice(len(pts), k, replace=False)].copy()      # init code vectors",
"palette = np.array(['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00'])",
"snaps = []",
"for it in range(4):",
"    a = np.argmin(((pts[:, None, :] - z[None]) ** 2).sum(2), axis=1)   # assign",
"    snaps.append((z.copy(), a.copy()))",
"    for j in range(k):",
"        if (a == j).any(): z[j] = pts[a == j].mean(0)                  # update",
"import plotly.graph_objects as go",
"# a single interactive plot animated over the iterations (so zoom/pan is per-figure)",
"names = ['1: initialize', '2: assign + update', '3: assign + update', '4: converged']",
"def _frame_traces(zc, a):",
"    tr = [go.Scatter(x=pts[a == j, 0], y=pts[a == j, 1], mode='markers',",
"                     marker=dict(color=palette[j], size=5, opacity=0.6), showlegend=False) for j in range(k)]",
"    tr.append(go.Scatter(x=zc[:, 0], y=zc[:, 1], mode='markers', showlegend=False,",
"                         marker=dict(color=list(palette), size=16, symbol='star', line=dict(color='black', width=1))))",
"    return tr",
"frames = [go.Frame(data=_frame_traces(zc, a), name=names[t]) for t, (zc, a) in enumerate(snaps)]",
"figk = go.Figure(data=frames[0].data, frames=frames)",
"figk.update_layout(title='K-means iterations, k=5 (Fig 30.11) — use the slider / Play',",
"                   width=620, height=580, template='plotly_white',",
"                   updatemenus=[dict(type='buttons', x=0.1, y=1.15, buttons=[",
"                       dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=700, redraw=True), fromcurrent=True)])])],",
"                   sliders=[dict(currentvalue=dict(prefix='step '), steps=[",
"                       dict(method='animate', label=f.name, args=[[f.name], dict(mode='immediate', frame=dict(duration=0, redraw=True))]) for f in frames])])",
"figk.update_xaxes(showticklabels=False); figk.update_yaxes(showticklabels=False, scaleanchor='x')",
"figk.show()",
))

# ---- Contrastive ----
cells.append(md(
"## 30.14 — Contrastive learning: the augmentation chooses the invariance",
"",
"**Contrastive learning** pulls together two augmented **views** of the same image"
" (a *positive* pair) while pushing apart different images, via the **InfoNCE** loss:",
"",
"$$\\mathcal{L} = -\\log \\frac{e^{f(x)^\\top f(x^+)/\\tau}}{e^{f(x)^\\top f(x^+)/\\tau} + \\sum_i e^{f(x)^\\top f(x_i^-)/\\tau}}.$$",
"",
"The representation becomes **invariant** to whatever the augmentation changes. So the"
" *choice of augmentation* decides which factor survives (embedding dim $M=2$, so we can"
" plot it directly on the unit circle):",
"",
"- **crop only** ($T_c$) — keeps colour, varies which part of the shape is seen"
" -> the embedding organizes by **colour**;",
"- **colour jitter** ($T_s$) — destroys colour, keeps shape -> the embedding organizes"
" by **shape**.",
))
cells.append(code(
"class Enc(nn.Module):",
"    def __init__(self):",
"        super().__init__()",
"        self.net = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(), nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),",
"            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(), nn.Flatten(), nn.Linear(64*16, 64), nn.ReLU(), nn.Linear(64, 2))",
"    def forward(self, x): return F.normalize(self.net(x), dim=1)      # onto the unit circle",
"",
"def rand_crop(xb, lo=0.5):",
"    N, C, H, W = xb.shape; out = torch.empty_like(xb)",
"    for i in range(N):",
"        s = float(torch.empty(1).uniform_(lo, 1.0)); ch = max(8, int(H*s)); cw = max(8, int(W*s))",
"        top = int(torch.randint(0, H-ch+1, (1,))); left = int(torch.randint(0, W-cw+1, (1,)))",
"        out[i] = F.interpolate(xb[i:i+1, :, top:top+ch, left:left+cw], size=(H, W), mode='bilinear', align_corners=False)[0]",
"    return out",
"def colour_jitter(xb):",
"    N = xb.shape[0]; gain = torch.empty(N, 3, 1, 1, device=xb.device).uniform_(0.4, 1.6)",
"    bright = torch.empty(N, 1, 1, 1, device=xb.device).uniform_(0.6, 1.4)",
"    perm = torch.stack([torch.randperm(3) for _ in range(N)])",
"    xj = torch.stack([xb[i, perm[i]] for i in range(N)])",
"    return (xj * gain * bright).clamp(0, 1)",
"def small_shift(xb):",
"    sh = torch.randint(-3, 4, (2,)); return torch.roll(xb, (int(sh[0]), int(sh[1])), dims=(2, 3))",
"",
"def infonce(z1, z2, tau=0.2):",
"    z = torch.cat([z1, z2]); sim = z @ z.T / tau; n = len(z1)",
"    sim.fill_diagonal_(-1e9)",
"    targets = torch.cat([torch.arange(n) + n, torch.arange(n)]).to(z.device)",
"    return F.cross_entropy(sim, targets)",
"",
"def train_contrastive(aug, steps, bs=128, seed=1):",
"    torch.manual_seed(seed); net = Enc().to(device); opt = torch.optim.Adam(net.parameters(), 1e-3)",
"    for st in range(steps):",
"        xb = Xtr_t[torch.randint(0, N, (bs,))].to(device)",
"        loss = infonce(net(aug(xb)), net(aug(xb))); opt.zero_grad(); loss.backward(); opt.step()",
"    net.eval()",
"    with torch.no_grad(): Z = net(Xtr_t.to(device)).cpu().numpy()",
"    return Z",
"",
"Z_colour = train_contrastive(lambda x: rand_crop(x), steps=1500)               # crop only -> colour",
"Z_shape  = train_contrastive(lambda x: colour_jitter(small_shift(x)), steps=2500)  # jitter -> shape",
"print('trained both contrastive encoders')",
))
cells.append(code(
"shape_cols = np.array(['#e41a1c', '#377eb8', '#4daf4a'])",
"colour_cols = np.array(['#e6194b', '#f58231', '#ffe119', '#3cb44b', '#42d4f4', '#4363d8', '#911eb4', '#f032e6'])",
"def emb_acc(Z, y):",
"    Zt = F.normalize(torch.tensor(Z), dim=1); sim = Zt @ Zt.T; sim.fill_diagonal_(-9)",
"    return (y[sim.argmax(1).numpy()] == y).mean()*100",
"import plotly.graph_objects as go",
"# each point is drawn AS its data point: marker symbol = the shape, colour = the object",
"# colour; hovering shows the shape/colour identity instead of raw coordinates.",
"SYMS = ['circle', 'square', 'triangle-up']",
"sub = np.arange(0, len(ys_tr), 4)                        # subsample for legibility with symbols",
"symbols = [SYMS[s] for s in ys_tr[sub]]",
"pt_colour = [str(c) for c in colour_cols[yc_tr[sub]]]",
"cust = [[SHAPES[ys_tr[i]], CNAMES[yc_tr[i]]] for i in sub]",
"for Z, name in [(Z_colour, 'T_c: crop-only'), (Z_shape, 'T_s: colour-jitter')]:",
"    ss, cc = emb_acc(Z, ys_tr), emb_acc(Z, yc_tr)",
"    figc = go.Figure(go.Scatter(x=Z[sub, 0], y=Z[sub, 1], mode='markers', customdata=cust,",
"        marker=dict(symbol=symbols, color=pt_colour, size=9, line=dict(color='rgba(0,0,0,0.35)', width=0.5)),",
"        hovertemplate='shape: %{customdata[0]}<br>colour: %{customdata[1]}<extra></extra>'))",
"    figc.update_layout(",
"        title=f'Contrastive embedding — {name}  (Fig 30.14)   ·   1-NN: shape {ss:.0f}%, colour {cc:.0f}%',",
"        width=580, height=600, template='plotly_white',",
"        annotations=[dict(text='marker symbol = shape · marker colour = object colour · hover shows the data point',",
"                          showarrow=False, xref='paper', yref='paper', x=0.5, y=-0.07, font=dict(size=10, color='gray'))])",
"    figc.update_xaxes(showticklabels=False); figc.update_yaxes(showticklabels=False, scaleanchor='x')",
"    figc.show()",
))

cells.append(md(
"### Takeaways",
"",
"- A **representation** re-expresses pixels in terms of their underlying factors. Here"
" the factors are **shape** and **colour**, and different methods expose them"
" differently.",
"- An **autoencoder** learns a reusable embedding for free (no labels); its nearest"
" neighbours respect shape and colour, and depth trades low-level appearance (colour)"
" for more semantic structure (shape).",
"- **K-means** is the discrete cousin — a representation by cluster index.",
"- **Contrastive learning** makes the control explicit: the **augmentation you choose**"
" is exactly the invariance you get. Crop-only keeps colour; colour-jitter keeps shape."
" This is the knob behind modern self-supervised encoders (SimCLR and friends).",
))

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
out = Path(sys.argv[1]); out.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print(f"wrote {out}  ({len(cells)} cells, {sum(c['cell_type']=='code' for c in cells)} code cells)")
